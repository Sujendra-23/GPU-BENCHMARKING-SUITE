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

## What was actually run and verified in this sandbox

This sandbox has **no NVIDIA GPU, no CUDA toolkit, and no internet access to
install PyTorch/TensorFlow/TensorRT** (those require `download.pytorch.org` /
NVIDIA's package indices, which weren't reachable). Rather than claim numbers
I couldn't produce, here's exactly what was and wasn't executed:

**Genuinely compiled and run, with real output:**

- **`kernels/opencl/matmul.cl` + `matmul_host.c`** — compiled with `gcc -lOpenCL`
  and run against a real OpenCL device (`pocl`, a CPU-backed OpenCL
  implementation installed for this purpose). Output was checked against a
  naive CPU reference implementation:
  ```
  running on OpenCL device: cpu-skylake-avx512-Intel(R) Xeon(R) Processor @ 2.10GHz
  opencl kernel time:     554.1826 ms
  cpu reference time:     156.3003 ms
  max abs diff vs ref:    0.001953  (PASS)
  ```
  This proves the kernel's tiling/indexing logic is correct. It does **not**
  prove anything about GPU speed — PoCL runs OpenCL kernels on the CPU, so a
  real GPU would look very different (and much faster) on this same kernel.

- **`kernels/stencil/jacobi.c`** — a 5-point Jacobi stencil (the access
  pattern behind HPC codes like LAMMPS/GROMACS), actually built and timed
  under GCC and Clang at `-O0`/`-O2`/`-O3`:
  ```
  gcc   -O0   3.5179s      clang -O0   3.1930s
  gcc   -O2   0.3772s      clang -O2   0.2747s
  gcc   -O3   0.2900s      clang -O3   0.2825s
  ```
  All six builds produced an identical checksum (correctness verified), and
  the ~12x speedup from `-O0`→`-O3` is real, measured data — not an estimate.

- **`harness/benchmark.py`** — the full harness (timing, p95 latency,
  throughput calc, graceful GPU-metric fallback, CSV export) was run
  end-to-end via its built-in `self_test()` path, which swaps in a tiny
  NumPy matmul+ReLU "model" when no GPU framework is detected. Verified the
  CSV report was written correctly with real numbers.

- **`harness/validate.py`** — the accuracy/latency deploy-gate logic was run
  via `--self-test`, covering pass, accuracy-fail, latency-fail, and
  both-fail cases; all four matched expected outcomes.

- **`harness/power_monitor.py`** — run directly; confirmed it detects the
  absence of `nvidia-smi` and exits cleanly (code 0) instead of crashing a
  pipeline that depends on it.

**Written but not executable in this sandbox (no GPU, no CUDA toolkit, no
package index access):**

- `kernels/cuda/matmul.cu` — mirrors the verified OpenCL kernel's algorithm
  exactly (same tiling scheme), but `nvcc` isn't installable here
- The `torch`/`torchvision`/`tensorrt` code paths in `benchmark.py` — real
  code, but conditionally imported and never actually reached in this
  environment (confirmed via the `any_real_runner` fallback triggering)
- `training/train_distributed.py` — NCCL requires 2+ real GPUs
- `training/export_tensorrt.py` — requires the TensorRT Python bindings
  and a GPU
- `docker/Dockerfile`, `docker/gpu-suite.def` — standard patterns against
  NVIDIA's official base images, not built here (no Docker/Singularity
  daemon, and `nvcr.io` isn't reachable from this sandbox)

## Honesty note on scope

Two items from the original job posting are **not attempted** anywhere in
this suite because they need physical lab equipment no software project can
fake: BIOS-level memory timing/overclocking, and electrical power
measurement via multimeters/DAQ hardware (Agilent/NI). Everything else in
the JD's "what we need to see" list is represented above.
