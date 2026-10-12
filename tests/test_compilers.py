from compilers.torch_compile import inspect_source
from harness.measurement import summarize


def test_inspection_requires_real_decorated_function():
    source = "import triton\n@triton.jit\ndef emitted(X):\n    return tl.exp(tl.sum(X))\n"
    report = inspect_source(source)
    assert report["triton_jit_functions"] == ["emitted"]
    assert report["contains_reduction"] and report["contains_exp"]
    assert inspect_source("# @triton.jit\npass")["triton_jit_functions"] == []
    assert inspect_source("async_compile.triton('emitted', " + repr(source) + ")")["triton_jit_functions"] == ["emitted"]


def test_nearest_rank_p95():
    report = summarize(list(range(1, 21)))
    assert report["p95_ms"] == 19
    assert report["median_ms"] == 10.5
