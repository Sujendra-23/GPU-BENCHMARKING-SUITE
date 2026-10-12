"""Paged GQA decode benchmark: python -m harness.paged_decode_bench.

32 query heads, 8 KV heads, head_dim 128, page size 16, fp16. Each sequence in
a batch gets a different length (uniform in [max_len/2, max_len], fixed seed)
and its pages are scattered through a shuffled physical page pool. Backends:
  triton    : kernels/triton/paged_decode.py (this repo)
  flashinfer: BatchDecodeWithPagedKVCacheWrapper (skipped, with the reason, if unavailable)
  sdpa      : torch SDPA on KV gathered into padded dense tensors OUTSIDE the timed
              region (the gather itself is not timed), with a padding mask; the
              4 query heads of a KV group are passed as 4 query rows.
Reported bandwidth counts the K and V bytes actually read (sum of sequence
lengths); the RTX 4090 datasheet peak is 1008 GB/s.

L2 handling: the RTX 4090 has a 72 MB L2. Replaying the same small KV cache in
a CUDA Graph would be served from L2 and report "bandwidth" above DRAM peak.
So the KV cache is allocated as several identical copies and every launch in
the timed graph reads a different copy, until the rotating working set is at
least 4x the L2 (or the copy cap / memory cap is hit; the actual number of
copies and working-set size are recorded per row).
"""
import argparse

from harness.measurement import cuda_time, int_list, metadata, write_json

PEAK_GBPS = 1008.0
L2_BYTES = 72 * 1024 * 1024
MAX_COPIES = 64
MAX_CACHE_BYTES = 8 * 1024 ** 3   # all KV copies (paged + dense) together


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int_list, default=[1, 4, 16, 64])
    parser.add_argument("--max-lens", type=int_list, default=[512, 2048, 4096, 8192])
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--page", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--out", default="results/paged_decode/rtx4090.json")
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F

    from kernels.triton.paged_decode import paged_decode, reference

    hq, hkv, d, page = args.q_heads, args.kv_heads, args.head_dim, args.page
    flashinfer, fi_error = None, None
    try:
        import flashinfer
    except Exception as exc:
        fi_error = f"{type(exc).__name__}: {str(exc)[:300]}"
    rows = []
    torch.manual_seed(9)
    for max_len in args.max_lens:
        for batch in args.batches:
            lengths = torch.randint(max_len // 2, max_len + 1, (batch,))
            lengths[0] = max_len
            pages_per = (lengths + page - 1) // page
            max_pages = int(pages_per.max())
            num_pages = int(pages_per.sum()) + 8
            bytes_read = 2 * int(lengths.sum()) * hkv * d * 2
            k_cache = torch.randn((num_pages, page, hkv, d), device="cuda", dtype=torch.float16)
            v_cache = torch.randn_like(k_cache)
            paged_bytes = k_cache.numel() * 2 * 2
            wanted = max(1, -(-4 * L2_BYTES // bytes_read))
            copies = max(1, min(MAX_COPIES, wanted, MAX_CACHE_BYTES // (2 * paged_bytes)))
            ks = [k_cache] + [k_cache.clone() for _ in range(copies - 1)]
            vs = [v_cache] + [v_cache.clone() for _ in range(copies - 1)]
            perm = torch.randperm(num_pages, dtype=torch.int32)
            table = torch.zeros((batch, max_pages), dtype=torch.int32)
            offset = 0
            for i in range(batch):
                n = int(pages_per[i])
                table[i, :n] = perm[offset: offset + n]
                offset += n
            table = table.cuda()
            seq_lens = lengths.int().cuda()
            q = torch.randn((batch, hq, d), device="cuda", dtype=torch.float16)
            ref = reference(q, k_cache, v_cache, table, seq_lens)
            shape = {"batch": batch, "max_len": max_len, "mean_len": float(lengths.float().mean()),
                     "q_heads": hq, "kv_heads": hkv, "head_dim": d, "page": page,
                     "kv_copies_rotated": copies,
                     "rotating_working_set_mb": copies * bytes_read / 2 ** 20}

            def rotating(call):
                state = {"i": 0}

                def fn():
                    i = state["i"] % copies
                    state["i"] += 1
                    return call(i)
                return fn

            impls = {"triton": rotating(lambda i: paged_decode(q, ks[i], vs[i], table, seq_lens, validate=False))}
            # SDPA on dense padded KV, gathered once outside the timed function
            kds, vds = [], []
            for i in range(copies):
                kd = torch.zeros((batch, hkv, max_pages * page, d), device="cuda", dtype=torch.float16)
                vd = torch.zeros_like(kd)
                for j in range(batch):
                    n = int(pages_per[j])
                    idx = table[j, :n].long()
                    kd[j, :, : n * page] = ks[i][idx].reshape(-1, hkv, d).transpose(0, 1)
                    vd[j, :, : n * page] = vs[i][idx].reshape(-1, hkv, d).transpose(0, 1)
                kds.append(kd)
                vds.append(vd)
            mask = (torch.arange(max_pages * page, device="cuda")[None, :] < seq_lens[:, None])[:, None, None, :]
            qg = q.view(batch, hkv, hq // hkv, d)
            impls["sdpa_dense_padded"] = rotating(lambda i: F.scaled_dot_product_attention(
                qg, kds[i], vds[i], attn_mask=mask).reshape(batch, hq, d))
            if flashinfer is not None:
                try:
                    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
                    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
                    indptr = torch.cat([torch.zeros(1, dtype=torch.int32), pages_per.cumsum(0).int()]).cuda()
                    indices = torch.cat([table[i, : int(pages_per[i])] for i in range(batch)]).int()
                    last = ((lengths - 1) % page + 1).int().cuda()
                    wrapper.plan(indptr, indices, last, hq, hkv, d, page, data_type=torch.float16,
                                 q_data_type=torch.float16)
                    impls["flashinfer"] = rotating(lambda i: wrapper.run(q, (ks[i], vs[i])))
                except Exception as exc:
                    rows.append({**shape, "backend": "flashinfer", "correctness": f"error: {type(exc).__name__}: {str(exc)[:300]}"})
            elif fi_error:
                rows.append({**shape, "backend": "flashinfer", "correctness": f"unavailable: {fi_error}"})
            for name, fn in impls.items():
                row = {**shape, "backend": name}
                try:
                    out = fn()
                    torch.cuda.synchronize()
                    err = (out.float() - ref).abs().max().item()
                    row["max_abs_error"] = err
                    row["correctness"] = "pass" if err <= 2e-3 else "FAIL"
                except Exception as exc:
                    row["correctness"] = f"error: {type(exc).__name__}: {str(exc)[:300]}"
                if row["correctness"] == "pass":
                    row.update(cuda_time(fn, 5, args.iterations, graph_repeats=max(8, copies)))
                    row.pop("samples_ms")
                    row["gbps"] = bytes_read / (row["median_ms"] * 1e6)
                    row["pct_of_peak_bw"] = 100 * row["gbps"] / PEAK_GBPS
                rows.append(row)
                print(f"b={batch:<3} max_len={max_len:<5} {name:18} {row['correctness'][:40]:40}"
                      + (f" {row['median_ms'] * 1000:8.1f} us {row['gbps']:7.1f} GB/s" if "median_ms" in row else ""), flush=True)
            del k_cache, v_cache, ks, vs, kds, vds
            torch.cuda.empty_cache()
    write_json(args.out, {"metadata": metadata(["torch", "triton", "flashinfer-python"]),
                         "device": torch.cuda.get_device_name(), "config": vars(args),
                         "peak_bandwidth_gbps_datasheet": PEAK_GBPS,
                         "reference": "fp32 PyTorch gather + softmax per sequence", "results": rows})


if __name__ == "__main__":
    main()
