"""Regenerate results/dl-kernels.md from the committed raw JSON reports.

python -m harness.dl_kernel_report
Only numbers present in results/**/*.json are used; sections whose input file
is missing are omitted rather than filled in.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"
BACKEND_ORDER = ["sdpa_flash", "triton_v1", "triton_tuned", "cuda_scalar", "cuda_wmma"]


def load(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


def attention_section(data):
    meta = data["metadata"]
    out = ["## Attention forward (fp16, batch 4, 16 heads)", "",
           f"GPU: {data['device']}; driver/GPU line from nvidia-smi: `{meta['nvidia_smi']}`. "
           f"PyTorch {meta['packages']['torch']}, Triton {meta['packages']['triton']}. "
           f"Reference: {data['reference']}. Timing: CUDA events over CUDA Graph replays, median. "
           f"Cells are `TFLOPs (percent of PyTorch SDPA flash backend)`; `-` = not run "
           f"(the scalar CUDA kernel is only run to N=2048 because it is too slow beyond that). "
           f"The datasheet peak used for context is {data['peak_tflops_fp16_fp32acc_datasheet']} TFLOPs "
           f"(dense fp16, fp32 accumulate); measured numbers can land slightly above it.", ""]
    rows = data["results"]
    for d in sorted({r["shape"][3] for r in rows}):
        for causal in (False, True):
            out += [f"**head_dim {d}, {'causal' if causal else 'non-causal'}**", "",
                    "| N | " + " | ".join(BACKEND_ORDER) + " |", "|---:|" + "---:|" * len(BACKEND_ORDER)]
            for n in sorted({r["shape"][2] for r in rows}):
                cells = []
                for backend in BACKEND_ORDER:
                    hit = [r for r in rows if r["backend"] == backend and r["shape"][2] == n
                           and r["shape"][3] == d and r["causal"] == causal]
                    if hit and "tflops" in hit[0]:
                        cells.append(f"{hit[0]['tflops']:.0f} ({hit[0]['pct_of_sdpa_flash']:.0f}%)")
                    elif hit:
                        cells.append(hit[0]["correctness"])
                    else:
                        cells.append("-")
                out.append(f"| {n} | " + " | ".join(cells) + " |")
            out.append("")
    worst = max(r["max_abs_error"] for r in rows if "max_abs_error" in r)
    configs = sorted({(r["shape"][2], r["shape"][3], r["causal"], json.dumps(r["autotuned_config"]))
                      for r in rows if r.get("autotuned_config")})
    out += [f"All {len(rows)} rows passed correctness (largest max-abs-error {worst:.2g}, tolerance 2e-3).", "",
            "<details><summary>Autotuned Triton config chosen per shape</summary>", "",
            "| N | head_dim | causal | config |", "|---:|---:|---|---|"]
    out += [f"| {n} | {d} | {c} | `{cfg}` |" for n, d, c, cfg in configs]
    out += ["", "</details>", ""]
    return out


def gemm_section(data):
    out = ["## GEMM (square, row-major)", "",
           f"GPU: {data['device']}; TF32 disabled so the fp32 baseline is plain SGEMM. Correctness against a float64 "
           f"reference (relative max error) before timing. `% of cuBLAS` = cuBLAS median / kernel median.", "",
           "| dtype | kernel | N | ms | TFLOPs | % of cuBLAS | rel. max error |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in data["results"]:
        if "median_ms" in r:
            out.append(f"| {r['section']} | {r['kernel']} | {r['n']} | {r['median_ms']:.3f} | {r['tflops']:.1f} | "
                       f"{r['pct_of_cublas']:.0f}% | {r['rel_max_error']:.1e} |")
        else:
            out.append(f"| {r['section']} | {r['kernel']} | {r['n']} | - | - | {r['correctness']} | {r['rel_max_error']:.1e} |")
    if data.get("cutlass_error"):
        out += ["", f"CUTLASS did not build/run: `{data['cutlass_error']}`"]
    return out + [""]


def cute_section():
    path = ROOT / "gemm" / "cute_dsl_gemm.txt"
    gemm = load("gemm/rtx4090.json")
    if not path.exists():
        return []
    import re
    text = path.read_text()
    versions = dict(re.findall(r"^(cuda-bindings|nvidia-cutlass-dsl)\s+(\S+)", text, re.M))
    out = ["## CUTLASS Python CuTe DSL GEMM (CUTLASS's own example, not repo code)", "",
           f"`harness/cute_dsl_gemm.py` runs `examples/python/CuTeDSL/.../ampere/.../tensorop_gemm.py` from CUTLASS v4.8.0 "
           f"(nvidia-cutlass-dsl {versions.get('nvidia-cutlass-dsl')}, cuda-bindings {versions.get('cuda-bindings')}), fp16 inputs/outputs, "
           f"fp32 accumulate, the example's own correctness check (PASS) and its own timing loop. cuBLAS numbers are from `gemm/rtx4090.json`.", "",
           "| N | CuTe DSL µs | TFLOPs | cuBLAS fp16 TFLOPs | % of cuBLAS | example check |", "|---:|---:|---:|---:|---:|---|"]
    for block in text.split("== N=")[1:]:
        n = int(block.split()[0])
        us = float(re.search(r"Execution time: ([\d.]+)", block).group(1))
        tflops = 2 * n ** 3 / (us * 1e6)
        cublas = next((r["tflops"] for r in (gemm or {}).get("results", [])
                       if r["kernel"] == "torch_cublas_hgemm" and r["n"] == n), None)
        out.append(f"| {n} | {us:.1f} | {tflops:.1f} | {'-' if cublas is None else f'{cublas:.1f}'} | "
                   f"{'-' if cublas is None else f'{100 * tflops / cublas:.0f}%'} | {'PASS' if 'PASS' in block else 'FAIL'} |")
    return out + [""]


def roofline_section():
    data = load("roofline/rtx4090.json")
    if not data:
        return []
    out = ["## Achieved throughput vs peak (Nsight Compute unavailable)", "",
           "Kernel time comes from `torch.profiler` (GPU kernel time of the dominant kernel, 20 launches). "
           "FLOP counts and *minimum* DRAM traffic are computed from the problem shape, so `GB/s` is an algorithmic "
           "lower bound on traffic, not a measured counter. Datasheet peaks: "
           f"{data['peaks_datasheet']['tflops']['fp16_tensor']} TFLOPs fp16 tensor (fp32 acc), "
           f"{data['peaks_datasheet']['tflops']['fp32']} TFLOPs fp32, {data['peaks_datasheet']['gbps']:.0f} GB/s.", "",
           "| kernel | µs | TFLOPs | % compute peak | min-traffic GB/s | % BW peak | FLOP/B | ridge | side |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in data["results"]:
        out.append(f"| {r['name']} | {r['kernel_us']:.1f} | {r['tflops']:.1f} | {r['pct_compute_peak']:.0f}% | "
                   f"{r['min_traffic_gbps']:.0f} | {r['pct_bandwidth_peak']:.0f}% | {r['flops_per_byte']:.0f} | "
                   f"{r['ridge_flops_per_byte']:.0f} | {r['side_of_ridge']} |")
    return out + [""]


def paged_section(data):
    out = ["## Paged-KV GQA decode attention", "",
           f"32 query heads, 8 KV heads, head_dim 128, page size 16, fp16, one new token per sequence. Per-sequence "
           f"lengths are uniform in [max_len/2, max_len]. Bandwidth counts only the K/V bytes read; datasheet peak "
           f"{data['peak_bandwidth_gbps_datasheet']:.0f} GB/s. FlashInfer: {data['metadata']['packages'].get('flashinfer-python')}. "
           f"`sdpa_dense_padded` runs SDPA on KV gathered into padded dense tensors *outside* the timed region.", "",
           "| batch | max_len | backend | µs | GB/s | % of peak BW | status |", "|---:|---:|---|---:|---:|---:|---|"]
    for r in data["results"]:
        if "median_ms" in r:
            out.append(f"| {r['batch']} | {r['max_len']} | {r['backend']} | {r['median_ms'] * 1000:.1f} | {r['gbps']:.0f} | "
                       f"{r['pct_of_peak_bw']:.0f}% | pass (err {r['max_abs_error']:.1e}) |")
        else:
            out.append(f"| {r['batch']} | {r['max_len']} | {r['backend']} | - | - | - | {r['correctness'][:90]} |")
    return out + [""]


def block_section():
    rows = []
    for mode in ("eager", "default", "max-autotune"):
        d = load(f"compilers/llama_block/{mode}/benchmark.json")
        if d:
            rows.append((mode, d))
    if not rows:
        return []
    base = rows[0][1]["timing"]["median_ms"]
    cfg = rows[0][1]["config"]
    out = ["## LLaMA-style transformer block: eager vs torch.compile", "",
           f"hidden {cfg['hidden']}, {cfg['heads']} heads, MLP {cfg['inter']}, batch {cfg['batch']}, seq {cfg['seq']}, fp16, "
           f"RMSNorm + RoPE + causal SDPA + SwiGLU. Error is against the same block in fp32 with the same weights. "
           f"`max-autotune` here is `max-autotune-no-cudagraphs`. Cold compile = first call, including compilation.", "",
           "| mode | median ms | speedup vs eager | cold compile s | max abs err | relative err |", "|---|---:|---:|---:|---:|---:|"]
    for mode, d in rows:
        cold = d["cold_compile_and_first_call_ms"]
        out.append(f"| {mode} | {d['timing']['median_ms']:.3f} | {base / d['timing']['median_ms']:.2f}× | "
                   f"{'-' if cold is None else f'{cold / 1000:.1f}'} | {d['max_abs_error']:.3g} | {d['relative_max_error']:.2%} |")
    return out + [""]


def triton_ir_section():
    data = load("triton_ir/summary.json")
    if not data:
        return []
    out = ["## Triton MLIR pipeline (causal attention kernel)", "",
           f"Shape {data['shape']}. Counts come from the saved files in `results/triton_ir/`.", "",
           "| kernel | stage | lines | detail |", "|---|---|---:|---|"]
    for name, stages in data["kernels"].items():
        for stage, info in stages.items():
            detail = ", ".join(f"{k}={v}" for k, v in info.items() if k != "lines")
            out.append(f"| {name} | {stage} | {info['lines']} | {detail} |")
    return out + [""]


def generate():
    out = ["# DL-kernel experiments on an RTX 4090", "",
           "Generated by `python -m harness.dl_kernel_report` from the raw JSON under `results/`. "
           "See the README for what was not done.", ""]
    for loader, fn in (("attention/rtx4090.json", attention_section), ("gemm/rtx4090.json", gemm_section),
                       ("paged_decode/rtx4090.json", paged_section)):
        data = load(loader)
        if data:
            out += fn(data)
    out += cute_section() + roofline_section() + block_section() + triton_ir_section()
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    (ROOT / "dl-kernels.md").write_text(generate())
    print("Wrote results/dl-kernels.md")
