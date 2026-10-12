# Kernels, serving and compiler experiments

Run commands from the repository root. These additions require an NVIDIA GPU
for kernel and TorchInductor runs. JAX offers an explicitly labeled CPU mode.
No command substitutes CPU numbers for GPU results.

## Environment and kernels

Python 3.11, a CUDA-enabled PyTorch installation, CUDA toolkit (`nvcc`), a C++
compiler and Ninja are required. The RunPod verification uses CUDA 12.4,
PyTorch 2.4.1 and Triton 3.0 on an RTX 4090. Install PyTorch using the wheel
appropriate for the driver, then:

```bash
python -m pip install -r requirements-kernels.txt
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9  # RTX 4090; change for other GPUs
python -m pytest -q tests/test_kernels.py
python -m harness.kernel_benchmark --out results/kernels/latest.json
```

Softmax is one fused kernel for max subtraction, exponential, row sum and
normalization, with fp32 reductions and fp16 input/output. It supports
contiguous nonempty matrices up to 8192 columns, including irregular widths.
Inputs must be finite. All-negative-infinity rows have undefined softmax.

Attention accepts contiguous `[batch, heads, sequence, head_dim]` fp16 Q/K/V,
equal shapes, head dimensions 32/64/128, and causal or noncausal self-attention.
Dropout, arbitrary masks, GQA, backward and cross-attention are not implemented.
Wrappers reject tensors requiring gradients. Tiled K/V and an online softmax
carry a running maximum, denominator and output accumulator in fp32, without
materializing the quadratic score matrix in global memory.

The Triton implementation processes 32 query rows and 32 keys per tile;
fp16 `tl.dot` permits tensor-core execution. The CUDA C++ implementation
shares a 32-key K/V tile across four warps, one query per warp, and uses
shuffle reductions and scalar arithmetic. It is an educational implementation
of the FlashAttention memory strategy, not an optimized FlashAttention-2
library or CUTLASS kernel. Tensor cores are used by the Triton path; a direct
CUTLASS/WMMA C++ implementation is still future work.

The harness checks every result before timing. Attention uses PyTorch's
**math SDPA on fp32** as the independent reference, and times fp16 SDPA with
automatic backend selection as the baseline. `sdpa_auto` does not promise
that a particular FlashAttention backend was selected. The matmul baseline is
fp16 `torch.mm`, dispatched through PyTorch's vendor BLAS path, checked against
fp64 matmul. It is separate from the existing fp32 tiled matmul executable.

Timings use CUDA events over graphs containing 32 repeated warmed calls,
divided by 32. This amortizes host dispatch, excludes JIT/build and host
allocation overhead, and reports device time rather than application latency.
Each configuration records raw samples, median/p95, tolerance and error in
JSON. Softmax bandwidth counts one input read and one output write; attention
TFLOPS counts useful QK/PV operations (the causal triangular count), not
hardware-issued instructions. Small shapes may have little useful parallelism.
Repeated inputs can remain in L2, so effective bandwidth is not a measurement
of physical DRAM throughput. The JSON latency samples describe this
warmed repeated-input workload.

## Nsight Compute

The focused entry point compiles and warms up before `cudaProfilerStart`.
For each backend and operation:

```bash
mkdir -p results/nsight
ncu --profile-from-start off --launch-count 1 \
    --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
    --csv --log-file results/nsight/triton-attention.csv \
    python -m harness.profile_kernels --backend triton --op attention --n 1024
```

Change the backend to `cuda` and operation to `softmax` for the other cases.
Use `--head-dim 128` and `--causal` to examine register pressure and masking.
CUDA builds include `-lineinfo`. Keep profiled runs separate from latency
measurements: replay and metric collection perturb timings.

Inspect SM throughput, DRAM throughput, registers, shared-memory use, achieved
occupancy and memory traffic. Softmax should avoid intermediate global-memory
passes; large rows may spill registers. Attention trades quadratic score
storage for repeated K/V loads and online accumulation. CUDA's scalar dot
products provide a useful contrast to Triton's tensor-core path. Verify actual
tensor instructions with instruction statistics/SASS before drawing hardware
utilization conclusions. These are investigation questions, not claimed
profiling findings. If `ERR_NVGPUCTRPERM` appears, the host must permit GPU
performance counters; a container cannot change the host driver policy.

Nsight Systems, when available, can show launch gaps and idle time:

```bash
nsys profile --trace=cuda,nvtx,osrt -o results/nsight/kernel-timeline \
    python -m harness.kernel_benchmark --op attention --seq-lens 1024 --head-dims 64
```

## vLLM and SGLang serving

The workload is `Qwen/Qwen2.5-0.5B-Instruct`, snapshot
`7ae557604adf67be50417f59c2c2f167def9a775`, one GPU, fp16 weights/KV,
256 exact prompt tokens, 128 requested output tokens and concurrency 1/4/8/16.
Token IDs are deterministic synthetic inputs inside this model's vocabulary,
identical across engines. This is a throughput workload, not a text-quality
evaluation. `ignore_eos` requests a fixed decode length; the report records
actual server token counts. Prefix/radix caching and CUDA graphs are disabled
in both engines to make these explicit experimental settings comparable.
The SGLang runner uses Triton attention. Defaults in other configurations can
produce different results.

Install each server in its own environment; their PyTorch requirements differ:
Frozen transitive versions from the recorded runs are also saved under
`results/serving/*-environment.txt`.

```bash
# Minimal Ubuntu images may need this shared library for sgl-kernel:
apt-get update && apt-get install -y libnuma1
python3.11 -m venv .venv-vllm
.venv-vllm/bin/pip install -r serving/requirements-vllm.txt
.venv-vllm/bin/python -m serving.run --backend vllm --requests 32

python3.11 -m venv .venv-sglang
.venv-sglang/bin/pip install -r serving/requirements-sglang.txt
.venv-sglang/bin/python -m serving.run --backend sglang --requests 32
```

Run one engine at a time. `serving.run` records the launch command and logs,
waits for readiness, warms the server, runs the client locally, then terminates
only the process group it launched. It refuses an occupied port. It downloads
the model on first launch. A RunPod container with only 20 GB writable storage
may need sequential environments or a persistent volume.
When using `uv`, caching wheels on the same filesystem with
`--cache-dir /workspace/gpu-suite-cache --link-mode hardlink` avoids a second
copy of large extracted CUDA packages; `--no-cache` can temporarily exhaust
a 20 GB container during installation even when the final environment fits.

The KV pool is capped at 65536 tokens. For this model,
`2 * 24 layers * 2 KV heads * 64 head_dim * 2 fp16 bytes = 12288 bytes/token`,
so the requested pool capacity is 805306368 bytes. vLLM receives that byte
limit; SGLang receives the token limit. Check startup logs for the actual
allocated capacity (block rounding and runtime limits can reduce it). When
benchmarking an existing server, supply its actual KV capacity:

```bash
python -m pip install -r serving/requirements-client.txt
python -m serving.benchmark --backend vllm --base-url http://127.0.0.1:8000 \
    --kv-capacity-bytes 805306368 --out results/serving/vllm.json
```

TTFT runs from HTTP submission to the first nonempty text chunk and includes
server queuing. Empty initial chunks are ignored. Aggregate output tokens/s
uses final server `usage.completion_tokens` divided by the entire load window;
SSE chunk count is never used as token count. Per-request decode throughput
uses `(completion_tokens - 1) / (stream_end - first_text)`. Servers may coalesce
tokens into chunks, so this is a client-observed decode rate. Failures remain
in the denominator, are recorded, and produce a nonzero exit status. There
is no separate unbounded client queue in TTFT: fixed workers replenish their
requests as each stream finishes (closed-loop load).

Cache usage is sampled from `/metrics` every 100 ms while requests are active:
`vllm:kv_cache_usage_perc` (or its older `gpu_cache_usage_perc` name) and
`sglang:token_usage`. The peak observed fraction times configured pool bytes
is an **estimate of occupied KV payload**, not total GPU memory or an exact
allocator measurement. Sampling may miss short peaks; idle baseline samples
are also affected by each engine's metric-export refresh interval. Samples
are retained separately by their negative elapsed time. Worker labels are
preserved and fractions are never summed. This benchmark is single-GPU;
multi-worker capacity accounting needs a per-worker extension. Missing active
samples produce nulls and a failing exit status rather than invented memory
numbers. Raw requests, cache samples, failures, versions and latency
percentiles are saved in JSON.
In the recorded SGLang run, exported fractions have 0.01 increments, equivalent
to 7.68 MiB of this pool. Its occupied-byte estimates therefore have coarse
resolution and should not be used to rank small memory-efficiency differences.

## TorchInductor and JAX/XLA

Both experiments use the same graph: fp16 input → fp32 affine transform →
stable row softmax → fp16 output. Correctness is checked against independent
higher-precision softmax. Cold compilation is recorded separately from warmed
execution. Inputs stay on device during timing.

```bash
python -m compilers.torch_compile --out-dir results/compilers/torch
# Inspect actual emitted sources and names in benchmark.json.

python3.11 -m venv .venv-jax
.venv-jax/bin/pip install 'jax[cuda12]==0.6.2' numpy
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-jax/bin/python -m compilers.jax_jit
# CPU-only correctness and IR inspection, explicitly labeled:
python -m compilers.jax_jit --device cpu --rows 32 --cols 65
```

TorchInductor uses a fresh cache and debug directory per run, enables
`TORCH_COMPILE_DEBUG` before importing PyTorch, parses generated source
(including embedded Triton strings), and requires actual `@triton.jit`
functions. JSON lists source paths, function names and reduction/exp markers.
Open the emitted files to inspect masks, tile sizes, fusion and launch grids;
the script does not equate function count with launch count.

JAX dumps `stablehlo.mlir`, `unoptimized.hlo` and `optimized.hlo` through
`lower().compiler_ir()` and `compile().as_text()`. For compiler pass dumps,
set `XLA_FLAGS="--xla_dump_to=/tmp/xla --xla_dump_hlo_as_text"` before launching.
JAX measures each call with `block_until_ready()` and wall clock, including
dispatch/synchronization. TorchInductor uses CUDA Graph device time, so compare
each compiler to its own eager baseline; their absolute timings are not a
controlled cross-framework comparison. These experiments exercise Inductor,
Triton, StableHLO/MLIR and XLA; TVM is not implemented.

## Primary references

- [Triton fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [PyTorch compiler observability](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.observability.html)
- [JAX ahead-of-time lowering](https://docs.jax.dev/en/latest/aot.html)
- [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html)
- [vLLM metrics](https://docs.vllm.ai/en/v0.10.2/design/metrics.html)
- [SGLang metrics](https://docs.sglang.io/docs/references/production_metrics)
- [Qwen model configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
