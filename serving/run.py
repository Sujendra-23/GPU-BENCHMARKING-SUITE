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


def server_command(args):
    if args.backend == "vllm":
        return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", MODEL, "--revision", MODEL_REVISION,
                "--host", "127.0.0.1", "--port", str(args.port),
                "--dtype", "half", "--kv-cache-dtype", "auto", "--max-model-len", "2048",
                "--max-num-seqs", str(max(args.concurrency)), "--no-enable-prefix-caching",
                "--kv-cache-memory-bytes", str(args.kv_capacity_tokens * KV_BYTES_PER_TOKEN),
                "--enforce-eager"]
    return [sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL,
            "--revision", MODEL_REVISION,
            "--host", "127.0.0.1", "--port", str(args.port), "--dtype", "float16",
            "--kv-cache-dtype", "auto", "--context-length", "2048",
            "--max-running-requests", str(max(args.concurrency)),
            "--max-total-tokens", str(args.kv_capacity_tokens), "--mem-fraction-static", "0.5",
            "--disable-radix-cache", "--enable-metrics", "--disable-cuda-graph",
            "--attention-backend", "triton"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["vllm", "sglang"], required=True)
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
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    command = server_command(args)
    args.model, args.base_url = MODEL, f"http://127.0.0.1:{args.port}"
    args.model_revision = MODEL_REVISION
    args.metrics_url, args.metrics_interval, args.timeout = None, .1, 120
    args.kv_capacity_bytes = args.kv_capacity_tokens * KV_BYTES_PER_TOKEN
    args.seed, args.warmup = 7, 2
    args.out = str(root / f"{args.backend}.json")
    # Never attach to an unrelated service already listening on this port.
    import socket
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            parser.error(f"Port {args.port} is occupied; choose a free port")
    write_json(root / f"{args.backend}-launch.json", {"command": command, "config": vars(args)})
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "1"
    env["DO_NOT_TRACK"] = "1"
    with (root / f"{args.backend}-server.log").open("w") as log:
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
