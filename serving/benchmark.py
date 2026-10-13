"""Closed-loop vLLM/SGLang/TensorRT-LLM load test with client latency and sampled KV usage."""
import argparse
import asyncio
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass

import httpx

from harness.measurement import int_list, metadata, positive_int, write_json

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
KV_BYTES_PER_TOKEN = 2 * 24 * 2 * 64 * 2  # K/V, layers, KV heads, head dim, fp16
GAUGES = {"vllm": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
          "sglang": ("sglang:token_usage",)}


@dataclass
class RequestResult:
    request_id: int
    ttft_ms: float | None = None
    latency_ms: float | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    decode_tokens_per_second: float | None = None
    error: str | None = None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def parse_metrics(text, backend):
    """Retain labels: cache fractions from different workers must not be summed."""
    if backend == "tensorrt-llm":
        # TRT-LLM 1.2 /metrics returns JSON iteration stats, not Prometheus.
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError("Expected TensorRT-LLM iteration-stat list")
        samples = []
        for row in rows:
            cache = row.get("kvCacheStats", {})
            used, capacity = cache.get("usedNumBlocks"), cache.get("maxNumBlocks")
            if isinstance(used, (int, float)) and isinstance(capacity, (int, float)) and capacity > 0 and 0 <= used <= capacity:
                samples.append({"name": "trtllm:kv_blocks_used_fraction", "labels": "", "fraction": used / capacity,
                                "source_active": row.get("numActiveRequests", 0) > 0,
                                "iteration": row.get("iter"), "used_blocks": used,
                                "max_blocks": capacity, "tokens_per_block": cache.get("tokensPerBlock")})
        return samples
    samples = []
    for line in text.splitlines():
        match = re.match(r'^([\w:]+)(\{.*\})?\s+([^\s]+)', line)
        if not match or match[1] not in GAUGES[backend]:
            continue
        value = float(match[3])
        if math.isfinite(value) and 0 <= value <= 1:
            samples.append({"name": match[1], "labels": match[2] or "", "fraction": value})
    return samples


async def sse_events(response):
    """Parse SSE records, including multiline data and an unterminated final record."""
    data = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


def make_prompt(request_id, length, seed):
    # Exact input length without re-tokenizing on the client. IDs are inside
    # Qwen2.5's vocabulary, deterministic, and identical across both engines.
    rng = random.Random(seed + request_id)
    return [rng.randrange(32, 20000) for _ in range(length)]


async def request(client, base_url, model, request_id, input_tokens, output_tokens, seed):
    result = RequestResult(request_id)
    payload = {"model": model, "prompt": make_prompt(request_id, input_tokens, seed),
               "max_tokens": output_tokens, "temperature": 0, "ignore_eos": True,
               "add_special_tokens": False, "stream": True, "stream_options": {"include_usage": True}}
    start = time.perf_counter()
    first = None
    usage = None
    done = False
    try:
        async with client.stream("POST", base_url + "/v1/completions", json=payload) as response:
            response.raise_for_status()
            async for data in sse_events(response):
                if data.strip() == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise ValueError(str(event["error"]))
                if any(choice.get("text") for choice in event.get("choices", [])) and first is None:
                    first = time.perf_counter()
                if event.get("usage"):
                    usage = event["usage"]
        end = time.perf_counter()
        if not done or first is None or not usage:
            raise ValueError("Stream missing DONE, text, or usage; token counts are never inferred from chunks")
        count = usage.get("completion_tokens")
        prompt = usage.get("prompt_tokens")
        if not isinstance(count, int) or count < 1 or not isinstance(prompt, int) or prompt != input_tokens:
            raise ValueError("Invalid completion count or server prompt length differs from requested length")
        if count != output_tokens:
            raise ValueError(f"Requested {output_tokens} output tokens, received {count}; fixed-length workload violated")
        result.ttft_ms = (first - start) * 1000
        result.latency_ms = (end - start) * 1000
        result.completion_tokens = count
        result.prompt_tokens = prompt
        if count > 1 and end > first:
            result.decode_tokens_per_second = (count - 1) / (end - first)
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as error:
        result.error = str(error)
        result.latency_ms = (time.perf_counter() - start) * 1000
    return result


async def benchmark_level(client, args, concurrency):
    queue = asyncio.Queue()
    for i in range(args.requests):
        queue.put_nowait(i)
    results = []
    samples, metric_errors = [], []
    stop = asyncio.Event()
    active, peak_active = 0, 0

    async def sample():
        try:
            response = await client.get(args.metrics_url or args.base_url + "/metrics", timeout=5)
            response.raise_for_status()
            values = parse_metrics(response.text, args.backend)
            samples.append({"time": time.perf_counter(), "gauges": values})
            # A drained TRT-LLM idle queue legitimately returns an empty list.
            absent = not values and (args.backend != "tensorrt-llm" or response.json())
            if absent and "KV gauge absent" not in metric_errors:
                metric_errors.append("KV gauge absent")
        except (httpx.HTTPError, ValueError) as error:
            if str(error) not in metric_errors:
                metric_errors.append(str(error))

    async def monitor():
        while not stop.is_set():
            await sample()
            try:
                await asyncio.wait_for(stop.wait(), args.metrics_interval)
            except asyncio.TimeoutError:
                pass

    async def worker():
        nonlocal active, peak_active
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            active += 1
            peak_active = max(peak_active, active)
            try:
                results.append(await request(client, args.base_url, args.model, index,
                                             args.input_tokens, args.output_tokens, args.seed))
            finally:
                active -= 1

    await sample()  # idle baseline outside workload timing
    task = asyncio.create_task(monitor())
    start = time.perf_counter()
    try:
        await asyncio.gather(*(worker() for _ in range(concurrency)))
    finally:
        end = time.perf_counter()
        stop.set()
        await task
    if args.backend == "tensorrt-llm":
        await sample()  # drain final iteration stats; source_active marks active iterations
    duration = end - start
    successful = [r for r in results if r.error is None]
    total = sum(r.completion_tokens for r in successful)
    fractions = [g["fraction"] for s in samples for g in s["gauges"]
                 if (g.get("source_active", False) and s["time"] >= start
                     if args.backend == "tensorrt-llm" else start <= s["time"] <= end)]
    peak = max(fractions) if fractions else None
    for sample_row in samples:
        sample_row["elapsed_seconds"] = sample_row.pop("time") - start
    return {"concurrency": concurrency, "peak_client_inflight": peak_active,
            "requests": args.requests, "successful": len(successful),
            "failed": len(results) - len(successful), "duration_seconds": duration,
            "output_tokens": total, "output_tokens_per_second": total / duration,
            "requests_per_second": len(successful) / duration,
            "ttft_p50_ms": percentile([r.ttft_ms for r in successful], .5),
            "ttft_p95_ms": percentile([r.ttft_ms for r in successful], .95),
            "latency_p95_ms": percentile([r.latency_ms for r in successful], .95),
            "decode_tps_p50": percentile([r.decode_tokens_per_second for r in successful
                                           if r.decode_tokens_per_second is not None], .5),
            "kv_cache": {"configured_capacity_bytes": args.kv_capacity_bytes,
                         "capacity_scope": getattr(args, "kv_capacity_scope", "per-rank"),
                         "estimated_peak_used_bytes_scope": "maximum observed fraction times per-rank capacity; not aggregate GPU memory",
                         "tensor_parallel_size": getattr(args, "tensor_parallel_size", 1),
                         "peak_sampled_usage_fraction": peak,
                         "estimated_peak_used_bytes": round(peak * args.kv_capacity_bytes)
                         if peak is not None and args.kv_capacity_bytes else None,
                         "metric_errors": metric_errors, "samples": samples},
            "request_results": [asdict(r) for r in sorted(results, key=lambda r: r.request_id)]}


async def run(args):
    levels = []
    async with httpx.AsyncClient(timeout=args.timeout,
                                limits=httpx.Limits(max_connections=max(args.concurrency) + 2)) as client:
        response = await client.get(args.base_url + "/v1/models")
        response.raise_for_status()
        models = response.json()
        if args.model not in [m["id"] for m in models.get("data", [])]:
            raise ValueError(f"Model {args.model} is not advertised by the server")
        for i in range(args.warmup):
            warmup = await request(client, args.base_url, args.model, -i - 1,
                                   args.input_tokens, args.output_tokens, args.seed)
            if warmup.error:
                raise ValueError(f"Warmup failed: {warmup.error}")
        for concurrency in args.concurrency:
            row = await benchmark_level(client, args, concurrency)
            levels.append(row)
            print(f"{args.backend} c={concurrency}: TTFT p50={row['ttft_p50_ms']} ms, "
                  f"{row['output_tokens_per_second']:.2f} output tok/s, failed={row['failed']}")
    write_json(args.out, {"metadata": metadata(["httpx", "vllm", "sglang", "tensorrt_llm", "torch", "triton", "transformers"]),
                         "config": vars(args), "server_models": models,
                         "method": "closed-loop; client TTFT to first nonempty text; server usage token counts",
                         "kv_method": "sampled cache fraction times explicitly configured capacity; estimate",
                         "results": levels})
    # Persist failed requests, but never report a broken run as successful.
    return 1 if any(r["failed"] or r["kv_cache"]["peak_sampled_usage_fraction"] is None
                    for r in levels) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["vllm", "sglang", "tensorrt-llm"], required=True)
    parser.add_argument("--tensor-parallel-size", type=int, choices=[1, 2], default=1)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--metrics-url")
    parser.add_argument("--model", choices=[MODEL], default=MODEL)
    parser.add_argument("--concurrency", type=int_list, default=[1, 4, 8, 16])
    parser.add_argument("--requests", type=positive_int, default=64)
    parser.add_argument("--input-tokens", type=positive_int, default=256)
    parser.add_argument("--output-tokens", type=positive_int, default=128)
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--metrics-interval", type=float, default=.1)
    parser.add_argument("--kv-capacity-bytes", type=positive_int, required=True,
                        help="Configured KV payload bytes per rank (TP=2 shards KV heads); not total VRAM")
    parser.add_argument("--out", default="results/serving/latest.json")
    args = parser.parse_args()
    if args.timeout <= 0 or args.metrics_interval <= 0 or args.requests < max(args.concurrency):
        parser.error("positive timeouts required; requests must be >= maximum concurrency")
    args.base_url = args.base_url.rstrip("/")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
