# GPU-Accelerated Training, Inference & Benchmarking Suite

A cross-framework GPU benchmarking harness (PyTorch / TensorFlow-style / TensorRT),
matching CUDA and OpenCL implementations of the same kernel, an NCCL multi-GPU
training script, a deploy-gating validation harness, and Docker/Singularity
packaging with GPU power/utilization monitoring.

## Attention kernels, LLM serving and ML compilers

New experiments extend the suite with fp16 fused softmax and tiled online-softmax
attention in **Triton and CUDA C++**, correctness checks against PyTorch SDPA,
a vendor BLAS matmul baseline, focused Nsight profiling, streaming
**vLLM/SGLang** benchmarks for Qwen2.5-0.5B-Instruct, and **torch.compile / JAX
jit** experiments that preserve generated Triton, StableHLO and XLA HLO.

See [setup, methodology and commands](docs/gpu-experiments.md) and
[RunPod verification results](results/experiments.md). The original measured
results below describe the earlier Blackwell run, separately from these additions.

## DL-kernel experiments (second RTX 4090 session)

A second RunPod session on one **NVIDIA GeForce RTX 4090** (24 GB, sm_89, driver
570.211.01, CUDA toolkit 12.4 / nvcc 12.4.131, Python 3.11.10, PyTorch 2.4.1+cu124,
Triton 3.0.0, FlashInfer 0.2.5+cu124torch2.4, CUTLASS C++ v3.5.1, CUTLASS Python
DSL 4.8.0, Nsight Compute 2024.1.1.0) added the pieces below. Every number is
from a raw JSON/text report committed under `results/`; the full tables are in
[`results/dl-kernels.md`](results/dl-kernels.md) (regenerate with
`python -m harness.dl_kernel_report`) and the commands are in
[`docs/dl-kernel-experiments.md`](docs/dl-kernel-experiments.md). Timing is CUDA
events over CUDA Graph replays (median) unless a section says otherwise; the
pod's container restarted once mid-session, so some runs were relaunched.
Datasheet peaks used for context (not measured here): 165.2 TFLOPs dense fp16
tensor with fp32 accumulate, 82.6 TFLOPs fp32, 1008 GB/s.

**Attention forward** (`harness/attention_sweep.py`; fp16, batch 4, 16 heads,
N = 512 to 8192, head_dim 64 and 128, causal and not; 92 rows, all within 2e-3
of an fp32 reference, worst 1.2e-3). Percent is of PyTorch SDPA's flash backend:

| kernel | non-causal | causal |
|---|---|---|
| `kernels/triton/attention.py` (original) | 64 to 75% (88 to 119 TFLOPs) | 71 to 92% |
| `kernels/triton/attention_tuned.py` (new: block pointers, exp2 softmax, split causal loop, autotuned BLOCK_M/N, warps, stages) | 91 to 105% (128 to 158 TFLOPs); 96 to 98% for N >= 2048 | 79 to 123%; 95 to 104% for N >= 4096; below SDPA for head_dim 128 up to N = 2048 (79 to 92%) |
| `kernels/cuda/attention_wmma.cu` (new: tensor-core WMMA, shared-memory K/V tiles) | 22 to 27% (29 to 43 TFLOPs) | 28 to 38% |
| `kernels/cuda/fused_ops.cu` scalar attention (old) | 2 to 3% (3 to 4 TFLOPs, run only to N = 2048) | 3 to 5% |

The WMMA kernel is about 11x faster than the scalar one (N = 1024, head_dim 64)
but still well below SDPA, so it is a working tensor-core kernel, not a
competitive one. What is known about the gap: it uses 56 to 64 registers per
thread with no spills (`cuobjdump`), and its dynamic shared memory is 71.7 KB
(head_dim 64) or 82.9 KB (head_dim 128) per block, which by arithmetic allows
one 4-warp block per SM on a 100 KB SM; it keeps O in shared memory and has no
`cp.async` pipelining. Those are design facts and static numbers; **no
counter-based profile confirms them as the cause** (see Nsight below).

**GEMM** (`harness/gemm_bench.py`, `kernels/cuda/gemm_fp32.cu`,
`kernels/cutlass/gemm_fp16.cu`; N = 2048, 4096, 8192; checked against float64;
TF32 off so the fp32 baseline is plain SGEMM):

| kernel | N = 2048 | N = 4096 | N = 8192 |
|---|---|---|---|
| fp32 tiled 16x16 (the old `matmul.cu` algorithm), % of cuBLAS | 12% | 12% | 9% |
| fp32 register-blocked 128x128x8, 8x8 per thread (new) | 68% (37 TFLOPs) | 86% (48 TFLOPs) | 76% (41 TFLOPs) |
| fp16 CUTLASS `device::Gemm` 128x128x32, 4 stages (new), % of cuBLAS | 100% | 99% | 101% |
| fp16 CUTLASS Python CuTe DSL example (CUTLASS's own code), % of cuBLAS | 94% | 93% | 103% |

cuBLAS itself ran at 54 to 56 TFLOPs fp32 and 161 to 169 TFLOPs fp16, slightly
above the datasheet fp16 figure, so treat "% of peak" as rough. The register-blocked
kernel is sensitive to measurement method: `harness/roofline.py` (torch.profiler,
no CUDA Graph) saw 42.0 TFLOPs at N = 4096 versus 48.4 here.

**Paged-KV GQA decode** (`kernels/triton/paged_decode.py`,
`harness/paged_decode_bench.py`; 32 query heads, 8 KV heads, head_dim 128,
page 16, variable lengths, shuffled page table). Triton, FlashInfer
(`BatchDecodeWithPagedKVCacheWrapper`) and SDPA on dense padded KV (gather not
timed) all match an fp32 reference (worst 3.2e-4). KV copies are rotated so each
timed launch misses the 72 MB L2; an earlier un-rotated run reported more than
1 TB/s on small shapes (L2 hits) and was discarded; its console lines are transcribed in
`results/paged_decode/discarded_l2_hot_partial_run.log`.
At 64 sequences with lengths up to 8192: Triton 1704 us (930 GB/s), FlashInfer
1692 us (937 GB/s), SDPA-dense 2269 us (699 GB/s). At 64 sequences up to 2048:
444, 441 and 581 us. Small batches are launch/latency-bound for all three
(batch 1, length up to 512: Triton 6.5 us, FlashInfer 8.4 us, SDPA 31.8 us).

**Compilers** (`compilers/llama_block.py`; LLaMA-style block with RMSNorm, RoPE,
causal SDPA and SwiGLU; hidden 4096, 32 heads, MLP 11008, batch 1, seq 2048, fp16;
error against the same block in fp32 is at most 0.1% of the largest output):

| mode | median ms | speedup | cold compile |
|---|---:|---:|---:|
| eager | 6.225 | 1.00x | - |
| `torch.compile` default | 5.794 | 1.07x | 3.7 s |
| `torch.compile` `max-autotune-no-cudagraphs` | 5.464 | 1.14x | 35.8 s |

The Inductor wrapper code with its Triton kernels is saved in
`results/compilers/llama_block/*/output_code_*.py`. The matmuls stay on
cuBLAS/Triton templates and attention stays SDPA, so the gain is modest.
`harness/triton_ir_dump.py` saved Triton's TTIR, TTGIR, LLIR and PTX for the
original and tuned attention kernels in `results/triton_ir/` (e.g. the tuned
kernel is 308 / 534 / 3860 / 2931 lines at the four stages and its PTX contains
128 `mma.sync` and 86 `cp.async` instructions); see
[`docs/dl-kernel-experiments.md`](docs/dl-kernel-experiments.md) for what each stage shows.

**Roofline without Nsight** (`harness/roofline.py`, torch.profiler kernel time
plus algorithmic FLOPs and minimum bytes): at N = 4096 the tuned Triton
attention reaches 158 TFLOPs (96% of the datasheet peak) at head_dim 64 and 155
(94%) at 128 versus 154 and 160 for SDPA; the WMMA kernel reaches 24 to 26%; paged decode
(64 sequences) sits on the memory side of the ridge at 930 GB/s (92% of peak).

**Tests:** `tests/test_kernels_extra.py` adds GPU checks (irregular lengths,
head dims 32/64/128, causal, shuffled page tables, GEMMs); the full suite
(`pytest tests`) passed on the pod: 173 passed.

**Not done, and why:**

- **Nsight Compute counters.** `ncu` 2024.1.1.0 fails with `ERR_NVGPUCTRPERM`
  (the driver has `RmProfilingAdminOnly: 1`; the container is root without the
  capability to change it, no `sudo`): `results/nsight/ncu-attempt-2026-10-12-pod2.txt`.
  Nsight Systems is not installed on the pod. No occupancy or memory-throughput
  counters exist.
- **Apache TVM.** Not run. The only CUDA wheel (`apache-tvm` 0.27.0) is built
  against CUDA 13 (`libcudart.so.13`); with that runtime installed
  `cudaGetDeviceCount` returns error 35 (insufficient driver: the pod's driver
  supports CUDA 12.8 only). The mlc-ai nightly wheel failed to load
  (`undefined symbol` in `libtvm_runtime.so`). A source build was not attempted.
- **cuTile.** Not run. Package metadata (`cuda-tile` 1.6.0) says it needs
  driver r580 or later and lists Ada among supported GPUs; this pod has r570
  (`results/notes/cutile-check.txt`).
- The CuTe DSL row is CUTLASS's own example run through `harness/cute_dsl_gemm.py`,
  not a kernel written for this repo. It needed `cuda-bindings` 12.x because 13.x
  also fails with insufficient driver.
- Only one GPU model (RTX 4090) and one process at a time; no multi-GPU or other
  architectures. The JAX/XLA and serving results above are from the earlier session
  and were not re-run.

## Layout

```
kernels/cuda/matmul.cu        tiled matmul CUDA kernel
kernels/cuda/attention_wmma.cu  tensor-core (WMMA) flash-style attention forward
kernels/cuda/gemm_fp32.cu     fp32 tiled and register-blocked GEMMs (PyTorch extension)
kernels/cutlass/gemm_fp16.cu  CUTLASS device-level fp16 tensor-core GEMM (needs CUTLASS_DIR)
kernels/triton/attention_tuned.py  autotuned FlashAttention-2-style Triton kernel
kernels/triton/paged_decode.py     paged-KV GQA decode attention (split-KV) in Triton
kernels/cuda/fused_ops.cu     fp16 fused softmax + tiled online-softmax attention
kernels/triton/               tensor-core attention + fused softmax in Triton
kernels/opencl/matmul.cl      the same algorithm in OpenCL, + C host program
kernels/hip/matmul.hip.cpp    HIP/ROCm port of the CUDA kernel (hipify-perl output, tidied), for AMD GPUs
kernels/reference/matmul_ref.c  the one CPU reference every kernel above is checked against
kernels/stencil/jacobi.c      GCC vs Clang compiler-comparison benchmark
harness/benchmark.py          cross-framework inference benchmarking harness
harness/kernel_benchmark.py   correctness-gated fp16 kernel and SDPA/BLAS sweep
harness/profile_kernels.py    warmed single-launch Nsight Compute target
serving/                      streaming vLLM/SGLang load test + local server runner
compilers/                    torch.compile source inspection + JAX jit/HLO dumps
harness/validate.py           accuracy/latency deploy gate
harness/power_monitor.py      nvidia-smi power/util/clock logger
harness/perf_per_watt.py      images/s per watt from the recorded benchmark rows (results/)
results/                      recorded ResNet-50 rows + the computed performance-per-watt table
training/train_distributed.py multi-GPU DDP training with NCCL backend
training/export_tensorrt.py   ONNX export + TensorRT engine build
docker/Dockerfile             CUDA-base image packaging
docker/gpu-suite.def          Singularity definition (JD explicitly lists this)
docker/Jenkinsfile            CI: hardware-independent checks + GPU-gated stage
dashboard/                    results dashboard + the same CPU reference, compiled to WASM, verifying live in-browser
orchestration/                Airflow DAG + docker-compose orchestrating benchmark -> validate -> export
```

## What was actually run and verified

This suite has been run end-to-end on a real NVIDIA GPU: an **NVIDIA RTX PRO
6000 Blackwell Server Edition** (1g.24gb MIG slice, driver 580.178.04, CUDA
13.0), a RunPod GPU pod reached over SSH. Every number below is measured
output from that run, not an estimate.

**Genuinely compiled and run, with real output:**

- **`kernels/cuda/matmul.cu`** — built with `nvcc -O3` and run directly on
  the GPU. Checked against a naive CPU reference implementation:
  ```
  $ nvcc -O3 -o matmul_cuda kernels/cuda/matmul.cu && ./matmul_cuda
  matmul_tiled (CUDA): M=512 N=512 K=512
  cuda kernel time:  0.3427 ms
  cpu reference time: 225.2400 ms
  max abs diff vs ref: 0.001953  (PASS)

  $ ./matmul_cuda 2048 2048 2048
  matmul_tiled (CUDA): M=2048 N=2048 K=2048
  cuda kernel time:  8.0587 ms
  cpu reference time: 51786.1570 ms
  max abs diff vs ref: 0.003906  (PASS)
  ```
  Correct output, and ~660x-6400x faster than the single-threaded CPU
  reference across the two sizes tested.

- **`kernels/opencl/matmul.cl` + `matmul_host.c`** — compiled with
  `gcc -lOpenCL` and run against the same physical GPU through NVIDIA's
  OpenCL ICD (not PoCL/CPU):
  ```
  running on OpenCL device: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb
  opencl kernel time:     0.5870 ms
  cpu reference time:     229.5449 ms
  max abs diff vs ref:    0.001953  (PASS)
  ```
  Confirms the CUDA and OpenCL implementations agree (same tiling algorithm,
  same correctness tolerance) and gives a real head-to-head: at this size the
  CUDA kernel runs ~1.7x faster than the OpenCL kernel on identical hardware.

- **`kernels/stencil/jacobi.c`** — a 5-point Jacobi stencil (the access
  pattern behind HPC codes like LAMMPS/GROMACS), built and timed under GCC
  and Clang at `-O0`/`-O2`/`-O3`:
  ```
  gcc   -O0   2.5656s      clang -O0   2.5570s
  gcc   -O2   0.6206s      clang -O2   0.2337s
  gcc   -O3   0.2238s      clang -O3   0.2292s
  ```
  All six builds produced an identical checksum (`1000.000031`), confirming
  correctness across compilers and optimization levels. GCC and Clang land
  within ~3% of each other at `-O3`, but Clang's `-O2` already nearly matches
  both compilers' `-O3` — real, measured variance in autovectorization
  behavior between the two, not an estimate.

- **`harness/benchmark.py`** — run for real against PyTorch + CUDA (not the
  synthetic NumPy fallback), timing `torchvision.models.resnet50` inference
  at multiple batch sizes and precisions with live GPU power draw sampled
  from `nvidia-smi`:
  ```
  framework,batch_size,precision,mean_latency_ms,p95_latency_ms,throughput_samples_per_sec,gpu_util_percent,gpu_power_watts
  torch,1,fp32,2.0897,2.109,478.54,,233.65
  torch,1,fp16,2.6747,5.1062,373.87,,236.55
  torch,8,fp32,7.0495,7.064,1134.84,,258.88
  torch,8,fp16,4.02,4.1378,1990.06,,261.69
  torch,32,fp32,34.417,34.463,929.77,,303.15
  torch,32,fp16,16.4107,16.4761,1949.95,,290.58
  torch,64,fp32,72.1238,72.1814,887.36,,298.96
  torch,64,fp16,36.3724,36.3985,1759.58,,306.48
  ```
  fp16 gives ~2x the throughput of fp32 at batch 8+, as expected. Full row
  data is written to `benchmark_report.csv` on each run.
  `gpu_util_percent` is genuinely empty (not a bug) — on this MIG slice
  `nvidia-smi`'s `utilization.gpu` field itself reports `[N/A]`; power draw
  is unaffected and reports correctly.

- **`harness/validate.py`** — the accuracy/latency deploy-gate logic was run
  via `--self-test`, covering pass, accuracy-fail, latency-fail, and
  both-fail cases; all four matched expected outcomes.

- **`harness/power_monitor.py`** — run directly against the live GPU
  alongside a `benchmark.py` run, polling `nvidia-smi` every 0.5s and
  logging power/temperature/clocks to `power_log.csv`; power draw tracked
  the benchmark load (idle ~95W, up to ~300W under the resnet50 sweep above).

- **`training/export_tensorrt.py`** — installed the TensorRT Python bindings
  and ran the full pipeline end-to-end against a real checkpoint: PyTorch
  model → ONNX export → parsed TensorRT engine, at fp32:
  ```
  $ python export_tensorrt.py --checkpoint model.pt --precision fp32 --batch-size 8 --out engines/resnet18_fp32_bs8.trt
  exported ONNX model to engines/resnet18_fp32_bs8.onnx
  built TensorRT engine (fp32, batch=8) -> engines/resnet18_fp32_bs8.trt
  ```
  That `.trt` engine was then loaded and run through `harness/benchmark.py`'s
  real `build_tensorrt_runner()` path (not skipped/mocked):
  ```
  BenchResult(framework='tensorrt', batch_size=8, precision='fp32', mean_latency_ms=1.9565, throughput_samples_per_sec=4088.98, gpu_power_watts=98.24)
  ```
  fp16/int8 engine builds are not currently supported by this script:
  TensorRT 10+ removed `BuilderFlag.FP16`/`INT8` in favor of a
  "strongly-typed network" precision model (precision now comes from the
  ONNX graph's own tensor dtypes, set via `NetworkDefinitionCreationFlag.STRONGLY_TYPED`,
  rather than a global builder flag) — a real design change, not a rename,
  so it's left as a follow-up rather than papered over.

- **`training/train_distributed.py`** — this pod has a single GPU (one MIG
  slice), so true multi-GPU NCCL all-reduce across separate devices can't be
  exercised here — that part of the JD requirement still needs a multi-GPU
  box. Everything else in the script *was* run for real via
  `torchrun --standalone --nproc_per_node=1`: `dist.init_process_group(backend="nccl")`,
  DDP-wrapping a real pretrained resnet50, a `DistributedSampler`-backed
  `DataLoader`, and a real training step with loss computed:
  ```
  epoch 0: 0.63s, ~25.4 samples/sec across 1 GPUs, loss=1.1075
  ```

- **`docker/Dockerfile`** — attempted a real `docker build` against it (not
  just a syntax read). Installed Docker in this pod and started `dockerd`;
  it fails with `Error response from daemon: unshare: operation not
  permitted` when BuildKit tries to create the namespace for the build
  step. This pod's own container doesn't have the `CAP_SYS_ADMIN`/netlink
  privileges Docker-in-Docker needs (bridge network creation also failed
  with `operation not permitted` before that) — a real, structural
  restriction of running Docker nested inside another container without
  `--privileged`, not something fixable from inside the pod. `gpu-suite.def`
  (Singularity) has the same blocker: no daemon, and building one needs
  privileges this environment doesn't grant either.

**HIP / ROCm port (AMD) — compiled, not yet run on an AMD GPU:**

- **`kernels/hip/matmul.hip.cpp`** — `hipify-perl` (ROCm) translated
  `kernels/cuda/matmul.cu`; the kernel body needed no changes (HIP uses the
  same `__shared__` / `__syncthreads()` / `<<<>>>` syntax). Hand edits: HIP
  labels, `HIP_CHECK` on every runtime call, and `extern "C"` around the shared
  C reference header (`hipcc` otherwise looks for a mangled C++ symbol).
  Verified inside the `rocm/dev-ubuntu-24.04` container (HIP 7.x, AMD clang):
  ```
  $ make -C kernels/hip compile-check      # gfx90a, gfx942, gfx1030, gfx1100
  $ hipcc -O3 --offload-arch=gfx90a -o matmul_hip matmul.hip.cpp ../reference/matmul_ref.c
  ```
  All four architectures compile with no warnings and the binary links against
  the same CPU reference as the CUDA and OpenCL builds. **It has not been
  executed on AMD hardware:** the development machine is an Apple M4 and the
  RunPod session above was NVIDIA, so the only run was on a container with no
  `/dev/kfd`, where it exits with `HIP error ... no ROCm-capable device is
  detected` rather than printing a result. No AMD timings or correctness
  results exist yet; on a ROCm machine run `make -C kernels/hip run
  GPU_ARCH=$(rocminfo | grep -o -m1 'gfx[0-9a-f]*')`. The OpenCL kernel was
  likewise only run on NVIDIA's ICD, not on an AMD device.

**Bugs found and fixed while getting all of the above running:**

- `harness/benchmark.py`'s `query_gpu_metrics()` originally parsed
  `nvidia-smi`'s two CSV fields together and discarded *both* on any parse
  failure. On this MIG-enabled GPU, `utilization.gpu` reports the literal
  string `[N/A]`, which threw a `ValueError` that silently zeroed out the
  adjacent, perfectly valid `power.draw` reading too. Fixed by parsing each
  field independently (`_parse_metric()`) so an unavailable field no longer
  discards a valid one alongside it — visible above in `gpu_power_watts`
  being populated on every row despite `gpu_util_percent` staying empty.
- `harness/validate.py` and `training/export_tensorrt.py` both called
  `torch.load(checkpoint_path, map_location="cpu")` to load a full pickled
  model object. Since PyTorch 2.6, `torch.load` defaults to
  `weights_only=True`, which rejects exactly that (it's meant for tensor
  state dicts, not arbitrary pickled classes) and raises instead of loading.
  Fixed by passing `weights_only=False` explicitly, matching what both
  scripts' checkpoint format actually requires.
- `training/export_tensorrt.py`'s TensorRT builder code used the TensorRT
  8.x API (`NetworkDefinitionCreationFlag.EXPLICIT_BATCH`,
  `config.max_workspace_size`, `builder.build_engine(...).serialize()`),
  all removed in TensorRT 10+. Updated to the current API
  (`create_network()` with implicit explicit-batch, `set_memory_pool_limit`,
  `build_serialized_network`). Also added `external_data=False` to the ONNX
  export call — newer `torch.onnx.export` defaults to writing weights to a
  separate `.onnx.data` file, which TensorRT's ONNX parser failed to import
  cleanly here.
- `harness/benchmark.py`'s `build_tensorrt_runner()` called
  `context.execute_async_v2(bindings=[...])`, also removed in TensorRT 10+
  in favor of `execute_async_v3`, which addresses tensors by name via
  `set_tensor_address()` instead of a positional bindings list. Updated
  accordingly.

## Performance per watt

`harness/perf_per_watt.py` turns the recorded ResNet-50 sweep above into
images/s/W and J/image per batch size and precision, plus fp16-vs-fp32 ratios.
Inputs and outputs are committed under `results/`:
`resnet50_gpu_benchmark.csv` (the eight rows from the table above, verified
identical to this README by `--check-readme`), `perf_per_watt.md`, and
`perf_per_watt.csv`. Regenerate with:

```
python3 harness/perf_per_watt.py results/resnet50_gpu_benchmark.csv \
    --markdown results/perf_per_watt.md --csv-out results/perf_per_watt.csv
python3 harness/perf_per_watt.py --self-test     # arithmetic and row-skipping checks
python3 harness/perf_per_watt.py --check-readme  # CSV == README table
```

From the recorded numbers (single run per configuration): at batch 8, fp16
gives 7.60 images/s/W (1990.06 images/s at 261.69 W) versus 4.38 for fp32
(1134.84 images/s at 258.88 W), a 1.73x improvement; at batch 32 it is 2.19x and
at batch 64 1.93x. At batch 1, fp16 is *worse* than fp32 (0.77x), as recorded.
The arithmetic is plain division of recorded values; the script performs no
new GPU measurement. Caveat: the power figure comes from `nvidia-smi` on a MIG
slice and may describe the whole board, so use the fp16/fp32 ratios, not the
absolute images/s/W, as the comparison. The repo has no committed power log,
so power is the per-row `gpu_power_watts` from the benchmark CSV, not a
time-integrated energy measurement.

**Not done:** a CUDA Sobel + non-maximum-suppression stage compared with
OpenCV. It needs `nvcc` and an NVIDIA GPU to compile and check; neither exists on
this Mac, so it was skipped rather than added unverified.

## Results dashboard

`dashboard/` is a small static page showing the recorded benchmark numbers
above, plus a live correctness check that runs in the browser:
`kernels/reference/matmul_ref.c` — the exact naive-matmul CPU reference the
CUDA and OpenCL host programs both check themselves against (they
`#include` it now instead of each keeping their own copy) — is compiled
with Emscripten to `dashboard/wasm/matmul_ref.{js,wasm}`. The page lets you
pick a matrix size and runs that reference client-side, cross-checking its
output against an independent plain-JS reimplementation of the identical
algorithm, and shows the live WASM timing next to the recorded native CUDA/
CPU-reference times at matching sizes (512×512×512 and 2048×2048×2048) as a
rough plausibility check — not a controlled comparison, since it's a
different machine, a different day, and WASM/JS overhead the native CUDA
build doesn't have.

To view it locally (must be served over HTTP, not `file://`, since browsers
block `fetch()` of local files under `file://`):

```
cd dashboard && python3 -m http.server 8000
# open http://localhost:8000
```

To rebuild the WASM module after editing `matmul_ref.c` (requires the
Emscripten SDK, e.g. `brew install emscripten`):

```
./kernels/reference/build_wasm.sh
```

## Orchestration (Airflow)

`orchestration/` turns three separate, manually-run CLI scripts -
`harness/benchmark.py`, `harness/validate.py`, `training/export_tensorrt.py`
- into one dependency-aware Airflow DAG
(`orchestration/dags/gpu_suite_pipeline.py`): `run_benchmark` -> `run_validation`
-> `run_tensorrt_export`, each gated on the previous task's real exit code
(Airflow's default `trigger_rule=all_success` - no extra config needed for
"only run validation if the benchmark succeeded"). It's a self-contained
local dev setup using Airflow's official docker-compose quick-start
(CeleryExecutor, Airflow 3.3.1), kept entirely inside `orchestration/` so it
doesn't touch the rest of the repo; the actual repo is bind-mounted into the
containers at `/gpu_suite` so the DAG runs the real scripts by their real
paths and real CLI arguments, not reimplementations of their logic.
`orchestration/.env` itself isn't committed (it holds a host-specific UID
and a generated Fernet key) - `orchestration/.env.example` is, with
instructions for generating your own.

Manually triggered only (`schedule=None`) - this is a benchmarking suite run
on demand against whatever hardware is attached, not a recurring production
job.

### Running it locally

```
cd orchestration
cp .env.example .env             # fill in AIRFLOW_UID and a generated FERNET_KEY
                                  # (instructions inline in .env.example) - not committed, see below
docker compose up airflow-init   # one-time DB init + admin user
docker compose up -d             # postgres, redis, apiserver, scheduler,
                                  # dag-processor, worker, triggerer
```

Open http://localhost:8080 (airflow/airflow), unpause `gpu_suite_pipeline`,
and trigger it - or from the CLI:

```
docker compose exec airflow-scheduler airflow dags trigger gpu_suite_pipeline
docker compose exec airflow-scheduler airflow tasks states-for-dag-run \
    gpu_suite_pipeline <run_id>   # run_id printed by the trigger command
```

`docker compose down` to stop; add `-v` only if you also want to discard the
Postgres metadata volume.

### What was actually run and verified here

This DAG was triggered for real against the live CeleryExecutor stack above
(not `airflow dags test`, which fakes the executor) - real task queuing,
a real worker picking it up, real subprocess execution of the actual
scripts, on this machine (no NVIDIA GPU). Every run produces the same
result:

```
run_benchmark         success
run_validation        success
run_tensorrt_export   failed
```

- **`run_benchmark`** genuinely succeeds: `harness/benchmark.py` detects no
  `torch`/`tensorrt` GPU runners are available and falls through to its own
  documented numpy CPU self-test path - a real code path in that script, not
  something this DAG fakes - and writes a real
  `orchestration/output/benchmark_report.csv`.
- **`run_validation`** genuinely succeeds: `--self-test` runs the actual
  pass/accuracy-fail/latency-fail/both-fail decision-logic checks in
  `harness/validate.py` and all four match. This DAG always uses
  `--self-test` rather than a real `--checkpoint`, because
  `validate.py`'s real-checkpoint path (`evaluate_real_model()`) raises
  `NotImplementedError` by design - its docstring says the dataset-specific
  `DataLoader` wiring was deliberately left abstract. That's a limitation of
  the script itself, not something this orchestration layer works around.
- **`run_tensorrt_export`** genuinely fails, every time, with
  `ModuleNotFoundError: No module named 'torch'` - this container has no
  GPU frameworks installed. Even with `torch` installed, it would still
  fail: no trained checkpoint (`model.pt`) is committed to this repo, and
  this environment has no NVIDIA GPU or TensorRT bindings either. All three
  gaps are real and independent of each other; fixing one doesn't fix this
  task without the others.

**A real bug was found and fixed by actually running this**, distinct from
the ones already listed under "What was actually run and verified": the
first DAG run failed at `run_benchmark` itself, before ever reaching the
self-test fallback, with `PermissionError: [Errno 13] Permission denied:
'nvidia-smi'` - not the `FileNotFoundError` `query_gpu_metrics()` in
`harness/benchmark.py` already caught. Airflow's official image runs as a
non-root UID with no access to `/root/bin`, which is first on `$PATH`; when
glibc's PATH search for `nvidia-smi` hits a directory it can't even `stat`,
it surfaces `EACCES` (`PermissionError`) instead of `ENOENT`
(`FileNotFoundError`), even though `nvidia-smi` doesn't exist anywhere on
the system. Fixed by widening the caught exception to `OSError` (which
covers both, and other exec failures) rather than enumerating subtypes.
This is exactly the kind of thing a real non-root container environment
surfaces that a bare-metal or root-shell test run doesn't - see the
comment above `query_gpu_metrics()` in `harness/benchmark.py`.

**Not verified, and not claimed:** a real GPU-backed run of this same DAG
(where `run_benchmark` would exercise the `torch`/`tensorrt` runners for
real and `run_tensorrt_export` could actually build an engine) needs the
same NVIDIA GPU + CUDA + TensorRT setup described earlier in this README,
plus a committed or externally-supplied model checkpoint - none of which
this local orchestration environment has. The DAG's structure, dependency
wiring, and non-GPU code paths are verified; the GPU code paths are not,
and this section says so rather than implying otherwise.
