"""Achieved-throughput "roofline" for the repo's kernels when Nsight Compute counters are unavailable.

Uses torch.profiler for the true GPU kernel time of each launch (not Python
latency) and the *algorithmic minimum* traffic / FLOP counts, so GB/s here is
an upper bound on DRAM efficiency, not a measured counter. The ridge point is
compute peak / memory peak. Peaks are datasheet values for the RTX 4090.
python -m harness.roofline
"""
import argparse

from harness.measurement import metadata, write_json

PEAK_TFLOPS = {"fp16_tensor": 165.2, "fp32": 82.6}   # dense, datasheet
PEAK_GBPS = 1008.0


def profile_kernel_time_us(torch, fn, iters=20):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.device_type.name == "CUDA" and e.count >= iters]
    events.sort(key=lambda e: e.cuda_time_total if hasattr(e, "cuda_time_total") else e.self_device_time_total, reverse=True)
    top = events[0]
    total = getattr(top, "cuda_time_total", None) or top.self_device_time_total
    return top.key, total / top.count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/roofline/rtx4090.json")
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from kernels.cuda import gemm, wmma
    from kernels.triton import attention as v1
    from kernels.triton import attention_tuned as tuned
    from kernels.triton.paged_decode import paged_decode

    torch.manual_seed(1)
    rows = []

    def add(name, fn, flops, bytes_min, peak_key):
        kernel, us = profile_kernel_time_us(torch, fn)
        tflops = flops / (us * 1e6)
        gbps = bytes_min / (us * 1e3)
        ridge = PEAK_TFLOPS[peak_key] * 1e12 / (PEAK_GBPS * 1e9)
        intensity = flops / bytes_min
        rows.append({"name": name, "kernel": kernel[:120], "kernel_us": us, "tflops": tflops,
                     "pct_compute_peak": 100 * tflops / PEAK_TFLOPS[peak_key], "min_traffic_gbps": gbps,
                     "pct_bandwidth_peak": 100 * gbps / PEAK_GBPS, "flops_per_byte": intensity,
                     "ridge_flops_per_byte": ridge, "side_of_ridge": "compute" if intensity > ridge else "memory"})
        print(f"{name:34} {us:9.1f} us {tflops:7.1f} TFLOPs ({rows[-1]['pct_compute_peak']:.0f}%) "
              f"{gbps:7.0f} GB/s ({rows[-1]['pct_bandwidth_peak']:.0f}%) AI={intensity:.0f} -> {rows[-1]['side_of_ridge']}-side", flush=True)

    with torch.inference_mode():
        for d in (64, 128):
            b, h, n = 4, 16, 4096
            q, k, v = [torch.randn((b, h, n, d), device="cuda", dtype=torch.float16) for _ in range(3)]
            flops = 4 * b * h * n * n * d
            traffic = 4 * b * h * n * d * 2   # Q, K, V read once + O written once
            def flash():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(q, k, v)
            add(f"attention d={d} sdpa_flash", flash, flops, traffic, "fp16_tensor")
            add(f"attention d={d} triton_v1", lambda: v1.attention(q, k, v, False), flops, traffic, "fp16_tensor")
            add(f"attention d={d} triton_tuned", lambda: tuned.attention(q, k, v, False), flops, traffic, "fp16_tensor")
            add(f"attention d={d} cuda_wmma", lambda: wmma.attention(q, k, v, False), flops, traffic, "fp16_tensor")
        n = 4096
        a16, b16 = [torch.randn((n, n), device="cuda", dtype=torch.float16) for _ in range(2)]
        a32, b32 = a16.float(), b16.float()
        torch.backends.cuda.matmul.allow_tf32 = False
        add("gemm4096 fp16 cuBLAS", lambda: a16 @ b16, 2 * n ** 3, 3 * n * n * 2, "fp16_tensor")
        try:
            add("gemm4096 fp16 CUTLASS", lambda: gemm.cutlass_fp16().gemm(a16, b16), 2 * n ** 3, 3 * n * n * 2, "fp16_tensor")
        except Exception as exc:
            print("CUTLASS skipped:", exc)
        add("gemm4096 fp32 cuBLAS", lambda: a32 @ b32, 2 * n ** 3, 3 * n * n * 4, "fp32")
        add("gemm4096 fp32 regblock", lambda: gemm.fp32().regblock(a32, b32), 2 * n ** 3, 3 * n * n * 4, "fp32")
        add("gemm4096 fp32 tiled16", lambda: gemm.fp32().tiled16(a32, b32), 2 * n ** 3, 3 * n * n * 4, "fp32")
        # paged decode: bytes read are the K and V of every live token
        batch, max_len, page, hq, hkv, d = 64, 8192, 16, 32, 8, 128
        lengths = torch.randint(max_len // 2, max_len + 1, (batch,))
        pages_per = (lengths + page - 1) // page
        mp = int(pages_per.max())
        npg = int(pages_per.sum()) + 8
        kc = torch.randn((npg, page, hkv, d), device="cuda", dtype=torch.float16)
        vc = torch.randn_like(kc)
        perm = torch.randperm(npg, dtype=torch.int32)
        table = torch.zeros((batch, mp), dtype=torch.int32)
        off = 0
        for i in range(batch):
            m = int(pages_per[i])
            table[i, :m] = perm[off: off + m]
            off += m
        table = table.cuda()
        sl = lengths.int().cuda()
        q = torch.randn((batch, hq, d), device="cuda", dtype=torch.float16)
        live = int(lengths.sum())
        add("paged decode b64 len<=8192", lambda: paged_decode(q, kc, vc, table, sl, validate=False),
            4 * live * hq * d, 2 * live * hkv * d * 2, "fp16_tensor")
    write_json(args.out, {"metadata": metadata(["torch", "triton"]), "device": torch.cuda.get_device_name(),
                         "peaks_datasheet": {"tflops": PEAK_TFLOPS, "gbps": PEAK_GBPS},
                         "note": "kernel time from torch.profiler; traffic is algorithmic minimum, not a DRAM counter", "results": rows})


if __name__ == "__main__":
    main()
