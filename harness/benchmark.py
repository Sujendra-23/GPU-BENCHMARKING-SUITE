"""Benchmark pretrained ResNet or Whisper-tiny with eager/compiled PyTorch, ONNX and TRT.

GPU runs fail explicitly when unavailable. Use --self-test for the synthetic
NumPy CPU validation. See docs/runpod.md for engines, calibration and ASR data.
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
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
    model: str = "resnet50"


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
        util_str, power_str = out.splitlines()[0].split(",")
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
        model="synthetic" if framework == "numpy-selftest" else "resnet50",
    )


# --- Framework-specific runners -------------------------------------------
# Each returns a synchronized callable or raises if the requested runtime
# or engine is unavailable; GPU workloads never substitute CPU rows.

def build_torch_runner(model_name, batch_size, precision, compiled=False):
    from harness.tensorrt_utils import require_cuda
    from training.export_tensorrt import load_model
    torch = require_cuda()
    if precision not in ("fp32", "fp16"):
        raise ValueError("Torch runners support fp32/fp16; use calibrated TensorRT engines for INT8")
    model = load_model(model_name).cuda().eval()
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model.to(dtype)
    x = torch.randn(batch_size, 3, 224, 224, device="cuda", dtype=dtype)
    if compiled:
        model = torch.compile(model, backend="inductor", fullgraph=False)
    with torch.inference_mode():
        model(x)  # compilation is outside measured calls
        torch.cuda.synchronize()
    def run_once():
        with torch.inference_mode():
            model(x)
        torch.cuda.synchronize()
    return run_once


def build_tensorrt_runner(model_name, batch_size, precision, engine_dir="engines"):
    from pathlib import Path
    from harness.tensorrt_utils import TensorRTRunner, require_cuda
    torch = require_cuda()
    path = Path(engine_dir) / f"{model_name}_{precision}_bs{batch_size}.trt"
    if not path.exists():
        raise FileNotFoundError(f"Build {path} with training.export_tensorrt first")
    engine = TensorRTRunner(path)
    shape, dtype = engine.inputs["input"]
    x = torch.randn(shape, dtype=dtype, device="cuda")
    def run_once():
        engine(input=x)
        torch.cuda.synchronize()
    return run_once


def build_onnx_runner(model_name, batch_size, precision, onnx_dir="engines"):
    from pathlib import Path
    from harness.tensorrt_utils import require_cuda
    from harness.onnx_utils import ONNXRunner
    from training.export_tensorrt import export_model, load_model
    torch = require_cuda()
    if precision not in ("fp32", "fp16"):
        raise ValueError("ONNX harness supports fp32/fp16")
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = load_model(model_name).cuda().to(dtype)
    x = torch.randn(batch_size, 3, 224, 224, dtype=dtype, device="cuda")
    path = Path(onnx_dir) / f"{model_name}_{precision}_bs{batch_size}.onnx"
    export_model(model, x, path)
    engine = ONNXRunner(path, {"output": (batch_size, 1000)})
    def run_once():
        engine(input=x)
        torch.cuda.synchronize()
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
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--engine-dir", default="engines")
    parser.add_argument("--onnx-dir", default="engines")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--self-test", action="store_true", help="Explicit synthetic NumPy CPU test")
    parser.add_argument("--audio-manifest")
    parser.add_argument("--audio-samples", type=int, default=8)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--asr-out-dir", default="results/whisper")
    parser.add_argument("--out", default="benchmark_report.csv")
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    if min(batch_sizes) < 1 or args.iterations < 1 or args.warmup < 0 or args.audio_samples < 1 or args.max_new_tokens < 1:
        parser.error("positive batch/iteration/sample/token counts and nonnegative warmup required")
    if args.self_test:
        write_csv(self_test(), args.out)
        return
    if args.model == "whisper-tiny":
        from harness.whisper import benchmark
        benchmark(args)
        return
    precisions = args.precisions.split(",")
    frameworks = args.frameworks.split(",")

    results: list[BenchResult] = []
    any_real_runner = False

    for framework in frameworks:
        for batch_size in batch_sizes:
            for precision in precisions:
                if framework == "torch":
                    runner = build_torch_runner(args.model, batch_size, precision)
                elif framework == "torch.compile":
                    runner = build_torch_runner(args.model, batch_size, precision, compiled=True)
                elif framework == "tensorrt":
                    runner = build_tensorrt_runner(args.model, batch_size, precision, args.engine_dir)
                elif framework == "onnx":
                    runner = build_onnx_runner(args.model, batch_size, precision, args.onnx_dir)
                else:
                    parser.error(f"unknown framework: {framework}")
                if runner is None:
                    continue
                any_real_runner = True
                row = bench_one(framework, batch_size, precision, runner, args.warmup, args.iterations)
                row.model = args.model
                results.append(row)

    if not any_real_runner:
        raise RuntimeError("No real inference runners; use --self-test explicitly for CPU validation")

    for r in results:
        print(r)
    write_csv(results, args.out)


if __name__ == "__main__":
    main()
