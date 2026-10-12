# RunPod experiment results

Measured on October 12, 2026 on one NVIDIA GeForce RTX 4090 (24 GB),
driver 570.169. Runs were sequential on the same pod. All values below
come from the linked raw reports; no synthetic GPU results are included.

## Kernels

70 CUDA/Triton GPU correctness tests passed, including irregular tiles,
causal/noncausal attention, fp16 softmax extremes and a nondefault stream.
Kernel environment: PyTorch 2.4.1+cu124, Triton 3.0.0, CUDA toolkit 12.4.

Representative medians in microseconds, from 30 CUDA-event samples of
32 repeated CUDA Graph calls per sample. Compilation and host allocation
are excluded. This is device time, not end-to-end Python latency.

| Operation / fp16 shape | Backend | Median µs | Max absolute error |
|---|---|---:|---:|
| Softmax 1024×4096 | torch | 15.90 | 1.90735e-06 |
| Softmax 1024×4096 | triton | 5.66 | 1.90735e-06 |
| Softmax 1024×4096 | cuda | 10.24 | 1.90735e-06 |
| Attention 1×4×1024×64, noncausal | sdpa_auto | 13.63 | 0.00012207 |
| Attention 1×4×1024×64, noncausal | triton | 25.70 | 0.00012207 |
| Attention 1×4×1024×64, noncausal | cuda | 352.61 | 0.00012207 |

Attention references use fp32 math SDPA. Triton softmax improves on
PyTorch for this representative width; SDPA is faster for this longer
attention shape. The CUDA attention path demonstrates tiled online
softmax with scalar arithmetic and is substantially slower than the
tensor-core implementations. These kernels are correctness-checked
educational implementations, not a claim of state-of-the-art performance.

[Full shape sweep, including causal attention and fp16 BLAS matmul](kernels/rtx4090.json).

## Compilers

The same fp16 affine/softmax graph at 1024×1024 passes independent
higher-precision correctness checks in both experiments.

| Experiment | Eager median µs | Compiled median µs | Speedup | Cold compilation ms |
|---|---:|---:|---:|---:|
| torch.compile | 29.89 | 3.14 | 9.53× | 3706.47 |
| JAX jit | 609.20 | 75.39 | 8.08× | 372.40 |

PyTorch reports CUDA Graph device time; JAX reports synchronized
wall-clock time including dispatch. Compare each against its own eager
baseline, not directly against the other framework. PyTorch's cold number
includes the first execution; JAX's cold number covers lowering/compilation.
JAX environment: jax/jaxlib 0.6.2 with CUDA 12 plugin.

Inspection of the preserved Inductor source shows one fused persistent
reduction launch for affine transform, maximum, exponentiation, sum and
division, with `XBLOCK=1` and `RBLOCK=1024`. The optimized JAX HLO
contains a single custom fusion with `__triton` backend configuration,
output tile `[2,1024]`, four warps and one stage. The XLA optimizer
therefore selected a Triton-backed fusion for this run, rather than
emitting each eager operation separately. These findings describe the
preserved artifacts and are not guarantees for other versions or shapes.

[Torch report and generated Triton paths](compilers/torch/benchmark.json),
[JAX report](compilers/jax/benchmark.json),
[StableHLO MLIR](compilers/jax/stablehlo.mlir),
[unoptimized HLO](compilers/jax/unoptimized.hlo),
[optimized HLO](compilers/jax/optimized.hlo).

## Streaming serving

Qwen/Qwen2.5-0.5B-Instruct, fp16 weights and KV cache, 256 exact input
tokens and 128 output tokens, 32 measured requests per concurrency level
after two warmups. Both runs used model snapshot
`7ae557604adf67be50417f59c2c2f167def9a775` (vLLM resolved the cached
main revision; SGLang and the current runner pin it explicitly).
CUDA graphs and prefix/radix caching were disabled. This is a controlled
eager-serving experiment, not peak production-engine performance.

| Engine | Concurrency | TTFT p50 ms | TTFT p95 ms | Output tokens/s | Peak KV used MiB (estimate) | Failed |
|---|---:|---:|---:|---:|---:|---:|
| vllm | 1 | 12.86 | 16.79 | 97.33 | 4.50 | 0 |
| vllm | 4 | 19.22 | 30.84 | 355.86 | 18.00 | 0 |
| vllm | 8 | 28.00 | 44.55 | 723.01 | 36.01 | 0 |
| vllm | 16 | 37.96 | 55.18 | 1418.29 | 72.02 | 0 |
| sglang | 1 | 27.91 | 39.17 | 79.15 | 7.68 | 0 |
| sglang | 4 | 40.00 | 196.08 | 304.16 | 15.36 | 0 |
| sglang | 8 | 44.00 | 70.76 | 603.24 | 38.40 | 0 |
| sglang | 16 | 63.23 | 270.78 | 1118.48 | 69.12 | 0 |

Each engine's configured KV pool is 768 MiB (65536 tokens at 12288
fp16 KV bytes/token). Peak occupied bytes are an estimate from sampled
exported cache fraction × configured capacity. Exporter refresh intervals
and 100 ms sampling can miss or lag instantaneous peaks. The figure is
not total allocated GPU memory. Startup logs confirm the pool sizes.

SGLang's exported fractions in this run have 0.01 increments, or 7.68
MiB of this pool, so its occupied-byte estimates have coarse resolution.
These samples cannot establish small differences in memory efficiency.

[vLLM raw requests and cache samples](serving/vllm.json),
[SGLang raw requests and cache samples](serving/sglang.json). Launch
commands, server logs and frozen package lists are saved alongside them.
The engine environments use their respective supported PyTorch versions;
this compares full engine stacks rather than isolating scheduler effects.

## Nsight and local checks

Nsight Compute 2024.1 was invoked on a warmed Triton attention launch.
It failed with `ERR_NVGPUCTRPERM` because the host restricts performance
counters, and explicitly reported that no kernels were profiled.
[Captured attempt](nsight/triton-attention-attempt.txt). Nsight Systems was
not installed in the pod. No occupancy or memory-counter results are claimed.

The Mac also passed nine serving/compiler unit tests and a CPU JAX
correctness/IR smoke run. Existing validation and performance-per-watt
self-tests and README/CSV consistency checks passed. GPU tests run on the
pod; the Mac explicitly skips them. TVM and a direct CUTLASS C++ kernel
are outside these implemented experiments.

[Reproduction commands and detailed methodology](../docs/gpu-experiments.md).

Regenerate this file with `python -m harness.experiment_report`.
