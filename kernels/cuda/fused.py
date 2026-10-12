"""Lazy build keeps CUDA compilation outside benchmark timings."""
from functools import lru_cache
from pathlib import Path

from kernels.contracts import check_attention, check_softmax


@lru_cache(maxsize=1)
def load_extension():
    from torch.utils.cpp_extension import load

    root = Path(__file__).parent
    return load(name="gpu_suite_fused", sources=[str(root / "fused_ops.cpp"),
                str(root / "fused_ops.cu")], extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


def softmax(x):
    check_softmax(x)
    return load_extension().softmax(x)


def attention(q, k, v, causal=False):
    check_attention(q, k, v)
    return load_extension().attention(q, k, v, causal)
