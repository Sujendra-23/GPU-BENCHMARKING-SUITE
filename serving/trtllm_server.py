"""TRT-LLM CLI with a bounded, lossless iteration-stats drain for release 1.0.0.

That release's /metrics can wait indefinitely while stats keep arriving, or
raise IndexError after collecting stats. Preserve actual stats and bound the
HTTP request; model execution and the stats producer are unchanged.
"""
import asyncio
from importlib.metadata import version


async def collect_iteration_stats(llm, timeout=.02):
    stats = []

    async def drain():
        try:
            async for stat in llm.get_stats_async(timeout):
                stats.append(stat)
        except IndexError:
            # TRT-LLM 1.0's AsyncQueue can signal an already-empty deque.
            pass

    try:
        await asyncio.wait_for(drain(), timeout)
    except asyncio.TimeoutError:
        pass
    return stats


def main():
    if version('tensorrt-llm') == '1.0.0':
        from fastapi.responses import JSONResponse
        from tensorrt_llm.serve.openai_server import OpenAIServer

        async def bounded_metrics(self):
            return JSONResponse(content=await collect_iteration_stats(self.llm))

        OpenAIServer.get_iteration_stats = bounded_metrics
    from tensorrt_llm.commands.serve import main as serve
    serve()


if __name__ == '__main__':
    main()
