import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from serving.benchmark import benchmark_level, make_prompt, parse_metrics, request


def stream_response(usage=True, done=True):
    events = [{"choices": [{"text": ""}]},
              {"choices": [{"text": "two tokens"}]}]
    if usage:
        events.append({"choices": [], "usage": {"prompt_tokens": 16, "completion_tokens": 2}})
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    if done:
        body += "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


@pytest.mark.parametrize("usage,done,success", [(True, True, True), (False, True, False), (True, False, False)])
def test_stream_usage_not_chunk_count(usage, done, success):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: stream_response(usage, done))) as client:
            return await request(client, "http://test", "model", 0, 16, 2, 7)
    result = asyncio.run(run())
    assert (result.error is None) == success
    if success:
        assert result.completion_tokens == 2  # one text chunk contains two tokens
        assert result.ttft_ms >= 0


def test_metrics_keep_worker_labels_and_ignore_histograms():
    samples = parse_metrics('vllm:kv_cache_usage_perc{engine="0"} 0.25\n'
                            'vllm:kv_cache_usage_perc{engine="1"} 0.50\n'
                            'vllm:time_to_first_token_seconds_count 20\n', "vllm")
    assert [s["fraction"] for s in samples] == [.25, .5]
    assert samples[0]["labels"] != samples[1]["labels"]
    assert parse_metrics("sglang:token_usage 0.125", "sglang")[0]["fraction"] == .125


def test_prompt_reproducibility():
    assert make_prompt(3, 16, 7) == make_prompt(3, 16, 7)
    assert make_prompt(3, 16, 7) != make_prompt(4, 16, 7)


def test_load_level_records_failures_and_kv():
    def handler(req):
        if req.url.path == "/metrics":
            return httpx.Response(200, text="vllm:kv_cache_usage_perc 0.25\n")
        return stream_response()
    args = SimpleNamespace(requests=4, base_url="http://test", metrics_url=None,
                           backend="vllm", metrics_interval=.001, model="model",
                           input_tokens=16, output_tokens=2, seed=7, kv_capacity_bytes=1024)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await benchmark_level(client, args, 2)
    row = asyncio.run(run())
    assert row["successful"] == 4
    assert row["output_tokens"] == 8
    assert row["failed"] == 0
    assert row["kv_cache"]["samples"]


def test_concurrent_streams_sample_active_cache_and_keep_http_failures():
    class DelayedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(.01)
            yield b'data: {"choices":[{"text":"first"}]}\n\n'
            await asyncio.sleep(.01)
            yield b'data: {"choices":[],"usage":{"prompt_tokens":16,"completion_tokens":2}}\n\n'
            yield b'data: [DONE]\n\n'

    def handler(req):
        if req.url.path == "/metrics":
            return httpx.Response(200, text="sglang:token_usage 0.5\n")
        if json.loads(req.content)["prompt"] == make_prompt(0, 16, 7):
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, stream=DelayedStream())

    args = SimpleNamespace(requests=4, base_url="http://test", metrics_url=None,
                           backend="sglang", metrics_interval=.001, model="model",
                           input_tokens=16, output_tokens=2, seed=7, kv_capacity_bytes=1024)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await benchmark_level(client, args, 2)
    row = asyncio.run(run())
    assert row["peak_client_inflight"] == 2
    assert row["successful"] == 3 and row["failed"] == 1
    assert row["output_tokens"] == 6
    assert row["kv_cache"]["estimated_peak_used_bytes"] == 512
    assert row["request_results"][0]["error"]
