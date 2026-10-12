"""Attention forward sweep on an NVIDIA GPU: python -m harness.attention_sweep.

Every backend is checked against an fp32 PyTorch reference (the memory-efficient
SDPA kernel, which accepts fp32 inputs, so it is a different implementation
from the fused fp16 kernels under test) before it is timed. Timing is CUDA
events over CUDA Graph replays, as in harness.measurement. Results report
percent of PyTorch SDPA's flash backend and achieved TFLOPs against the
RTX 4090 datasheet peak (165.2 TFLOPs dense fp16 with fp32 accumulate).
"""
import argparse

from harness.measurement import cuda_time, int_list, metadata, positive_int, write_json

PEAK_TFLOPS_FP16_FP32ACC = 165.2  # RTX 4090 datasheet, dense; not measured here
BACKENDS = ["sdpa_flash", "triton_v1", "triton_tuned", "cuda_scalar", "cuda_wmma"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=int_list, default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--head-dims", type=int_list, default=[64, 128])
    parser.add_argument("--batch", type=positive_int, default=4)
    parser.add_argument("--heads", type=positive_int, default=16)
    parser.add_argument("--causal", choices=["both", "yes", "no"], default="both")
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--scalar-max-seq", type=positive_int, default=2048,
                        help="the scalar CUDA kernel is only run up to this length")
    parser.add_argument("--iterations", type=positive_int, default=20)
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--out", default="results/attention/rtx4090.json")
    args = parser.parse_args()
    names = args.backends.split(",")
    if any(name not in BACKENDS for name in names):
        parser.error(f"backends must be a subset of {BACKENDS}")

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("An NVIDIA CUDA GPU is required")
    impls = {}
    if "triton_v1" in names:
        from kernels.triton.attention import attention as triton_v1
        impls["triton_v1"] = triton_v1
    if "triton_tuned" in names:
        from kernels.triton.attention_tuned import attention as triton_tuned
        impls["triton_tuned"] = triton_tuned
    if "cuda_scalar" in names:
        from kernels.cuda import fused
        impls["cuda_scalar"] = fused.attention
    if "cuda_wmma" in names:
        from kernels.cuda import wmma
        impls["cuda_wmma"] = wmma.attention
    modes = [False, True] if args.causal == "both" else [args.causal == "yes"]
    rows = []

    def sdpa_flash(q, k, v, causal):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    def fp32_reference(q, k, v, causal):
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)

    torch.manual_seed(7)
    with torch.inference_mode():
        for d in args.head_dims:
            for n in args.seq_lens:
                shape = (args.batch, args.heads, n, d)
                q, k, v = [torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3)]
                for causal in modes:
                    ref = fp32_reference(q, k, v, causal)
                    flops = (2 * args.batch * args.heads * n * (n + 1) * d if causal
                             else 4 * args.batch * args.heads * n * n * d)
                    repeats = 4 if n <= 2048 else 1
                    sdpa_ms = None
                    for name in ["sdpa_flash"] + [x for x in names if x != "sdpa_flash"]:
                        if name == "cuda_scalar" and n > args.scalar_max_seq:
                            continue
                        fn = (lambda: sdpa_flash(q, k, v, causal)) if name == "sdpa_flash" else \
                             (lambda impl=impls.get(name): impl(q, k, v, causal))
                        row = {"backend": name, "shape": list(shape), "causal": causal,
                               "dtype": "fp16"}
                        try:
                            out = fn()
                            torch.cuda.synchronize()
                            error = (out.float() - ref).abs().max().item()
                            row["max_abs_error"] = error
                            row["correctness"] = "pass" if error <= 2e-3 else "FAIL"
                        except Exception as exc:  # record, keep sweeping
                            row["correctness"] = f"error: {type(exc).__name__}: {str(exc)[:200]}"
                        if row["correctness"] == "pass":
                            row.update(cuda_time(fn, args.warmup, args.iterations,
                                                 graph_repeats=repeats))
                            row.pop("samples_ms", None)
                            row["tflops"] = flops / (row["median_ms"] * 1e9)
                            row["pct_of_peak"] = 100 * row["tflops"] / PEAK_TFLOPS_FP16_FP32ACC
                            if name == "sdpa_flash":
                                sdpa_ms = row["median_ms"]
                            if sdpa_ms:
                                row["pct_of_sdpa_flash"] = 100 * sdpa_ms / row["median_ms"]
                            if name == "triton_tuned":
                                from kernels.triton.attention_tuned import best_config
                                row["autotuned_config"] = best_config(q, k, v, causal)
                        rows.append(row)
                        print(f"{name:13} d={d:<3} n={n:<5} causal={causal!s:5} "
                              f"{row['correctness']:6} "
                              + (f"{row['median_ms']:.3f} ms {row['tflops']:.1f} TFLOPs "
                                 f"{row.get('pct_of_sdpa_flash', 0):.0f}% of sdpa"
                                 if "median_ms" in row else ""), flush=True)
                del q, k, v, ref
                torch.cuda.empty_cache()
    write_json(args.out, {"metadata": metadata(["torch", "triton"]),
                         "device": torch.cuda.get_device_name(), "config": vars(args),
                         "peak_tflops_fp16_fp32acc_datasheet": PEAK_TFLOPS_FP16_FP32ACC,
                         "reference": "PyTorch fp32 memory-efficient SDPA; atol 2e-3",
                         "results": rows})


if __name__ == "__main__":
    main()
