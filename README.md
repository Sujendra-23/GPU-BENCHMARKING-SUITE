# GPU-Accelerated Training, Inference & Benchmarking Suite

A cross-framework GPU benchmarking harness (PyTorch / TensorFlow-style / TensorRT),
matching CUDA and OpenCL implementations of the same kernel, an NCCL multi-GPU
training script, a deploy-gating validation harness, and Docker/Singularity
packaging with GPU power/utilization monitoring.

## Layout

```
kernels/cuda/matmul.cu        tiled matmul CUDA kernel
kernels/opencl/matmul.cl      the same algorithm in OpenCL, + C host program
kernels/stencil/jacobi.c      GCC vs Clang compiler-comparison benchmark
harness/benchmark.py          cross-framework inference benchmarking harness
harness/validate.py           accuracy/latency deploy gate
harness/power_monitor.py      nvidia-smi power/util/clock logger
training/train_distributed.py multi-GPU DDP training with NCCL backend
training/export_tensorrt.py   ONNX export + TensorRT engine build
docker/Dockerfile             CUDA-base image packaging
docker/gpu-suite.def          Singularity definition (JD explicitly lists this)
docker/Jenkinsfile            CI: hardware-independent checks + GPU-gated stage
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
