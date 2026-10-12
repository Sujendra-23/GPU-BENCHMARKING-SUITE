"""Common JSON metadata and synchronized timing; no synthetic GPU fallback."""
import json
import hashlib
import math
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def positive_int(value):
    import argparse

    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def int_list(value):
    return [positive_int(item) for item in value.split(",")]


def metadata(packages=()):
    versions = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=5, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                   text=True, timeout=5, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    root = Path(__file__).resolve().parents[1]
    sources = {}
    for directory in ("harness", "kernels", "compilers", "serving"):
        for path in sorted((root / directory).rglob("*")):
            if path.suffix in (".py", ".cu", ".cpp") and "__pycache__" not in path.parts:
                sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "python": sys.version,
            "packages": versions, "nvidia_smi": gpu, "git_commit": commit,
            "git_dirty": dirty, "source_sha256": sources}


def summarize(samples):
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("Cannot summarize empty measurements")
    return {"mean_ms": statistics.mean(ordered), "median_ms": statistics.median(ordered),
            "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
            "min_ms": ordered[0], "samples_ms": samples}


def cuda_time(fn, warmup=10, iterations=50, graph_repeats=32):
    import torch

    if warmup < 0 or iterations < 1:
        raise ValueError("Invalid warmup/iteration counts")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Capture repeated launches to amortize host dispatch. For microsecond
    # kernels, events around individual Python calls mostly measure GPU idle
    # time waiting for Python, rather than device execution.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_repeats):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    # Event creation and JIT compilation occur before the measured launches.
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(iterations)]
    for start, end in pairs:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return {**summarize([start.elapsed_time(end) / graph_repeats for start, end in pairs]),
            "graph_repeats": graph_repeats,
            "method": "CUDA events over repeated CUDA Graph launches; per-call device time"}


def write_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {path}")
