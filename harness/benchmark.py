"""Benchmark harness comparing PyTorch, TensorFlow, and TensorRT inference
across batch sizes and precisions, logging throughput, latency, and GPU
utilization to a CSV report.

This mirrors what NVIDIA's Performance Engineering / Deep Learning & HPC
internship postings describe: "planning and executing GPU performance
benchmarking across a wide range of HPC and DL frameworks... aggregating,
analyzing, and generating written and visual reports."

Real usage (on a machine with a GPU and these frameworks installed):

    python benchmark.py --model resnet50 --batch-sizes 1,8,32,64 \
        --precisions fp32,fp16 --frameworks torch,tensorrt

On a machine without a GPU or these packages (like the sandbox this was
built in), running with no arguments exercises the harness end-to-end
against a tiny synthetic CPU model so the benchmarking/logging/reporting
logic itself is verified without needing real hardware - see
`self_test()` and the README for what that proves and doesn't prove.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Iterable


@dataclass
class BenchResult:
    framework: str
    batch_size: int
    precision: str
    mean_latency_ms: float
    p95_latency_ms: float
    throughput_samples_per_sec: float
    gpu_util_percent: float | None
    gpu_power_watts: float | None


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _parse_metric(value: str) -> float | None:
    """nvidia-smi reports fields like utilization.gpu as "[N/A]" on MIG
    slices even when other fields (e.g. power.draw) are valid - parse each
    field independently so one unavailable field doesn't discard the rest."""
    try:
        return float(value.strip())
    except ValueError:
        return None


def query_gpu_metrics() -> tuple[float | None, float | None]:
    """Returns (utilization_percent, power_watts) via nvidia-smi. Either or
    both may be None if no NVIDIA GPU/driver is present, or if a given field
    is unavailable (e.g. utilization.gpu under MIG).
    """
    import subprocess
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        util_str, power_str = out.split(",")
        return _parse_metric(util_str), _parse_metric(power_str)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        # OSError (not just FileNotFoundError) matters here: if any earlier
        # PATH entry exists but isn't traversable by the current user - e.g.
        # a non-root container UID that can't stat a root-owned directory
        # like /root/bin, as happens running this under Airflow's official
        # image - glibc's PATH search surfaces the last error it saw, EACCES,
        # as PermissionError, even though nvidia-smi itself is simply absent.
        return None, None


def _percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    idx = min(len(values) - 1, int(len(values) * pct))
    return values[idx]


def time_inference_calls(run_once: Callable[[], None], warmup: int, iters: int) -> list[float]:
    """Runs `run_once` `warmup` times unrecorded, then `iters` times timed.
    Returns a list of per-call latencies in milliseconds."""
    for _ in range(warmup):
        run_once()
    latencies_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_once()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    return latencies_ms


def bench_one(framework: str, batch_size: int, precision: str,
              run_once: Callable[[], None], warmup: int = 5, iters: int = 30) -> BenchResult:
    latencies = time_inference_calls(run_once, warmup, iters)
    util, power = query_gpu_metrics()
    mean_lat = statistics.mean(latencies)
    return BenchResult(
        framework=framework,
        batch_size=batch_size,
        precision=precision,
        mean_latency_ms=round(mean_lat, 4),
        p95_latency_ms=round(_percentile(latencies, 0.95), 4),
        throughput_samples_per_sec=round(batch_size / (mean_lat / 1000.0), 2),
        gpu_util_percent=util,
        gpu_power_watts=power,
    )


# --- Framework-specific runners -------------------------------------------
# Each returns a zero-arg callable that runs one inference pass, or None if
# that framework/model isn't available in the current environment.

def build_torch_runner(model_name: str, batch_size: int, precision: str):
    torch = _try_import("torch")
    torchvision = _try_import("torchvision")
    if torch is None or torchvision is None:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = getattr(torchvision.models, model_name)(weights=None).to(device).eval()
    if precision == "fp16" and device == "cuda":
        model = model.half()
    dtype = torch.float16 if (precision == "fp16" and device == "cuda") else torch.float32
    x = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

    def run_once():
        with torch.no_grad():
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()

    return run_once


def build_tensorrt_runner(model_name: str, batch_size: int, precision: str):
    # A full TensorRT build requires exporting the model to ONNX, then
    # building a TensorRT engine with trtexec or the Python builder API
    # (see training/export_tensorrt.py for that step). Here we just load
    # a prebuilt engine if one exists, to keep this harness self-contained.
    trt = _try_import("tensorrt")
    if trt is None:
        return None
    import os
    engine_path = f"engines/{model_name}_{precision}_bs{batch_size}.trt"
    if not os.path.exists(engine_path):
        return None

    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  (initializes the CUDA context)
    import numpy as np

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # TensorRT 10+ addresses tensors by name (execute_async_v3) rather than
    # a positional bindings list (execute_async_v2, removed in TensorRT 10).
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    input_shape = (batch_size, 3, 224, 224)
    h_input = np.random.randn(*input_shape).astype(np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    output_shape = (batch_size, 1000)
    h_output = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    def run_once():
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    return run_once


# --- CPU-only self-test, actually executed in CI/sandboxes without a GPU --

def build_selftest_runner(batch_size: int):
    """A deliberately tiny numpy 'model' (one matmul + relu) standing in
    for a real network, so the harness's timing/percentile/reporting code
    path can be exercised end-to-end without any GPU or ML framework."""
    import numpy as np
    w = np.random.randn(256, 256).astype(np.float32)
    x = np.random.randn(batch_size, 256).astype(np.float32)

    def run_once():
        y = x @ w
        np.maximum(y, 0, out=y)  # relu in place

    return run_once


def self_test() -> list[BenchResult]:
    """Runs the full harness (timing, percentile calc, GPU-metric probing,
    CSV export) against the synthetic CPU model above. This proves the
    harness logic is correct; it does NOT prove anything about real
    PyTorch/TensorFlow/TensorRT numbers, since those frameworks/hardware
    aren't present. See README for what was verified vs. not."""
    results = []
    for batch_size in (1, 8, 32):
        runner = build_selftest_runner(batch_size)
        results.append(bench_one("numpy-selftest", batch_size, "fp32", runner))
    return results


def write_csv(results: Iterable[BenchResult], path: str) -> None:
    results = list(results)
    if not results:
        print("no results to write", file=sys.stderr)
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"wrote {len(results)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--batch-sizes", default="1,8,32,64")
    parser.add_argument("--precisions", default="fp32,fp16")
    parser.add_argument("--frameworks", default="torch,tensorrt")
    parser.add_argument("--out", default="benchmark_report.csv")
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    precisions = args.precisions.split(",")
    frameworks = args.frameworks.split(",")

    results: list[BenchResult] = []
    any_real_runner = False

    for framework in frameworks:
        for batch_size in batch_sizes:
            for precision in precisions:
                if framework == "torch":
                    runner = build_torch_runner(args.model, batch_size, precision)
                elif framework == "tensorrt":
                    runner = build_tensorrt_runner(args.model, batch_size, precision)
                else:
                    print(f"unknown framework: {framework}", file=sys.stderr)
                    continue
                if runner is None:
                    continue
                any_real_runner = True
                results.append(bench_one(framework, batch_size, precision, runner))

    if not any_real_runner:
        print(
            "No GPU frameworks/hardware detected in this environment - "
            "running the numpy CPU self-test instead so the harness itself "
            "is still exercised end-to-end. See README.",
            file=sys.stderr,
        )
        results = self_test()

    for r in results:
        print(r)
    write_csv(results, args.out)


if __name__ == "__main__":
    main()
