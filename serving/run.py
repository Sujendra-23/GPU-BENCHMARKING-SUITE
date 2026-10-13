"""Launch one local engine, benchmark it, and clean up its process group."""
import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from harness.measurement import int_list, positive_int, write_json
from serving.benchmark import KV_BYTES_PER_TOKEN, MODEL, MODEL_REVISION, run


def run_name(args):
    return f"{args.backend}-{args.quantization}-tp{args.tensor_parallel_size}"


def kv_capacity(args):
    # Qwen has two KV heads. TP=2 shards them across two ranks (no replication).
    return args.kv_capacity_tokens * KV_BYTES_PER_TOKEN // args.tensor_parallel_size


def server_command(args):
    if args.backend == "vllm":
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", MODEL, "--revision", MODEL_REVISION,
                "--tokenizer-revision", MODEL_REVISION,
                "--host", "127.0.0.1", "--port", str(args.port),
                "--dtype", "half", "--kv-cache-dtype", "auto", "--max-model-len", "2048",
                "--max-num-seqs", str(max(args.concurrency)), "--max-num-batched-tokens", "2048",
                "--no-enable-prefix-caching",
                "--kv-cache-memory-bytes", str(kv_capacity(args)),
                "--tensor-parallel-size", str(args.tensor_parallel_size), "--enforce-eager"]
        if args.quantization == "fp8":
            command.extend(["--quantization", "fp8"])
        return command
    if args.backend == "tensorrt-llm":
        return [sys.executable, "-m", "serving.trtllm_server", MODEL,
                "--backend", "pytorch", "--host", "127.0.0.1", "--port", str(args.port),
                "--tp_size", "1", "--max_seq_len", "2048",
                "--max_batch_size", str(max(args.concurrency)),
                "--max_num_tokens", "2048", "--extra_llm_api_options", str(args.trtllm_config)]
    return [sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL,
            "--revision", MODEL_REVISION,
            "--host", "127.0.0.1", "--port", str(args.port), "--dtype", "float16",
            "--kv-cache-dtype", "auto", "--context-length", "2048",
            "--max-running-requests", str(max(args.concurrency)),
            "--max-total-tokens", str(args.kv_capacity_tokens), "--mem-fraction-static", "0.5",
            "--disable-radix-cache", "--enable-metrics", "--disable-cuda-graph",
            "--attention-backend", "triton"]


def trtllm_config_text(args):
    # The YAML revision fields work with both the CUDA 12 and CUDA 13 CLIs.
    return (
        "dtype: float16\n"
        f"revision: {MODEL_REVISION}\n"
        f"tokenizer_revision: {MODEL_REVISION}\n"
        "enable_iter_perf_stats: true\n"
        "cuda_graph_config: null\n"
        "kv_cache_config:\n"
        f"  max_tokens: {args.kv_capacity_tokens}\n"
        "  enable_block_reuse: false\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["vllm", "sglang", "tensorrt-llm"], required=True)
    parser.add_argument("--quantization", choices=["none", "fp8"], default="none")
    parser.add_argument("--tensor-parallel-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--port", type=positive_int, default=8000)
    parser.add_argument("--concurrency", type=int_list, default=[1, 4, 8, 16])
    parser.add_argument("--requests", type=positive_int, default=64)
    parser.add_argument("--input-tokens", type=positive_int, default=256)
    parser.add_argument("--output-tokens", type=positive_int, default=128)
    parser.add_argument("--kv-capacity-tokens", type=positive_int, default=65536)
    parser.add_argument("--startup-timeout", type=positive_int, default=600)
    parser.add_argument("--out-dir", default="results/serving")
    args = parser.parse_args()
    if args.port > 65535:
        parser.error("port must be <= 65535")
    if args.requests < max(args.concurrency) or args.input_tokens + args.output_tokens > 2048:
        parser.error("requests must cover concurrency; prompt + output must fit 2048 tokens")
    if args.backend != "vllm" and (args.quantization != "none" or args.tensor_parallel_size != 1):
        parser.error("FP8 and TP=2 runs are implemented for vLLM only")
    import torch
    if torch.cuda.device_count() < args.tensor_parallel_size:
        parser.error(f"Need {args.tensor_parallel_size} visible CUDA GPUs; use a separate two-GPU pod for TP=2")
    if args.quantization == "fp8" and torch.cuda.get_device_capability()[0] < 8:
        parser.error("FP8 requires supported NVIDIA hardware; use Ada/Hopper or newer")
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    name = run_name(args)
    if args.backend == "tensorrt-llm":
        args.trtllm_config = str((root / f"{name}-config.yaml").resolve())
        Path(args.trtllm_config).write_text(trtllm_config_text(args))
    command = server_command(args)
    args.model, args.base_url = MODEL, f"http://127.0.0.1:{args.port}"
    args.model_revision = MODEL_REVISION
    args.cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.backend == "tensorrt-llm":
        from importlib.metadata import version
        args.trtllm_version = version("tensorrt-llm")
        args.trtllm_metrics_adapter = "bounded real iteration-stats drain (20 ms); empty-deque fix" if args.trtllm_version == "1.0.0" else None
    args.metrics_url, args.metrics_interval, args.timeout = None, .1, 120
    args.kv_capacity_bytes = kv_capacity(args)
    args.kv_capacity_scope = "per-rank; KV heads sharded, no replication for Qwen TP=2"
    args.trtllm_runtime = "pytorch (TensorRT-LLM executor, not a serialized TensorRT engine)" if args.backend == "tensorrt-llm" else None
    args.seed, args.warmup = 7, 2
    args.out = str(root / f"{name}.json")
    # Never attach to an unrelated service already listening on this port.
    import socket
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            parser.error(f"Port {args.port} is occupied; choose a free port")
    write_json(root / f"{name}-launch.json", {"command": command, "config": vars(args)})
    env = os.environ.copy()
    # Direct invocation of a venv Python does not activate its bin directory.
    # FlashInfer/Torch JIT must be able to find that environment's Ninja.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["VLLM_USE_V1"] = "1"
    env["DO_NOT_TRACK"] = "1"
    with (root / f"{name}-server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
        try:
            deadline = time.monotonic() + args.startup_timeout
            with httpx.Client(timeout=3) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited ({process.returncode}); see {log.name}")
                    try:
                        if client.get(args.base_url + "/v1/models").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Server not ready; see {log.name}")
                    time.sleep(1)
            result = asyncio.run(run(args))
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        raise SystemExit(result)


if __name__ == "__main__":
    main()
