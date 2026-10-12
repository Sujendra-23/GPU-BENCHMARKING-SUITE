"""GEMM comparison on an NVIDIA GPU: python -m harness.gemm_bench.

fp32 section (TF32 disabled, so cuBLAS runs plain SGEMM): torch.matmul vs the
repo's 16x16 tiled kernel and a register-blocked kernel. fp16 section:
torch.matmul (cuBLAS tensor cores) vs a CUTLASS tensor-core GEMM. Every
kernel is checked against a float64 reference before it is timed.
"""
import argparse

from harness.measurement import cuda_time, int_list, metadata, write_json

PEAK_FP32_TFLOPS = 82.6      # RTX 4090 datasheet, non-tensor fp32; not measured here
PEAK_FP16_TFLOPS = 165.2     # datasheet dense fp16 tensor with fp32 accumulate; not measured here


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int_list, default=[2048, 4096, 8192])
    parser.add_argument("--out", default="results/gemm/rtx4090.json")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(3)
    from kernels.cuda import gemm
    fp32_ext = gemm.fp32()
    try:
        cutlass_ext = gemm.cutlass_fp16()
    except Exception as exc:  # report, keep going
        cutlass_ext, cutlass_error = None, f"{type(exc).__name__}: {str(exc)[:300]}"
    rows = []

    def record(section, name, n, fn, ref, tol, peak, baseline):
        out = fn()
        error = (out.double() - ref).abs().max().item() / max(1.0, ref.abs().max().item())
        row = {"section": section, "kernel": name, "n": n, "rel_max_error": error,
               "correctness": "pass" if error <= tol else "FAIL"}
        if row["correctness"] == "pass":
            repeats = 4 if n <= 2048 else 1
            row.update(cuda_time(fn, 3, args.iterations, graph_repeats=repeats))
            row.pop("samples_ms")
            row["tflops"] = 2 * n ** 3 / (row["median_ms"] * 1e9)
            row["pct_of_peak"] = 100 * row["tflops"] / peak
            if name.startswith("torch"):
                baseline[n] = row["median_ms"]
            row["pct_of_cublas"] = 100 * baseline[n] / row["median_ms"]
        rows.append(row)
        print(f"{section} {name:18} n={n:<5} {row['correctness']:5} "
              + (f"{row['median_ms']:.3f} ms {row['tflops']:.1f} TFLOPs {row['pct_of_cublas']:.0f}% of cuBLAS"
                 if "median_ms" in row else f"err={error:.3g}"), flush=True)

    with torch.inference_mode():
        base32, base16 = {}, {}
        for n in args.sizes:
            a, b = [torch.randn((n, n), device="cuda") for _ in range(2)]
            ref = a.double() @ b.double()
            record("fp32", "torch_cublas_sgemm", n, lambda: a @ b, ref, 1e-4, PEAK_FP32_TFLOPS, base32)
            record("fp32", "tiled16", n, lambda: fp32_ext.tiled16(a, b), ref, 1e-4, PEAK_FP32_TFLOPS, base32)
            record("fp32", "regblock", n, lambda: fp32_ext.regblock(a, b), ref, 1e-4, PEAK_FP32_TFLOPS, base32)
            a16, b16 = a.half(), b.half()
            ref16 = a16.double() @ b16.double()
            record("fp16", "torch_cublas_hgemm", n, lambda: a16 @ b16, ref16, 1e-2, PEAK_FP16_TFLOPS, base16)
            if cutlass_ext is not None:
                record("fp16", "cutlass_128x128x32", n, lambda: cutlass_ext.gemm(a16, b16), ref16, 1e-2,
                       PEAK_FP16_TFLOPS, base16)
            del a, b, a16, b16, ref, ref16
            torch.cuda.empty_cache()
    result = {"metadata": metadata(["torch"]), "device": torch.cuda.get_device_name(),
              "tf32_disabled": True, "results": rows,
              "peaks_datasheet_tflops": {"fp32": PEAK_FP32_TFLOPS, "fp16_tensor_fp32acc": PEAK_FP16_TFLOPS}}
    if cutlass_ext is None:
        result["cutlass_error"] = cutlass_error
    write_json(args.out, result)


if __name__ == "__main__":
    main()
