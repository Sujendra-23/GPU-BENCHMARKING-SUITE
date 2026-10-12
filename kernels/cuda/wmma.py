"""Lazy build of the tensor-core (WMMA) attention extension."""
from functools import lru_cache
from pathlib import Path

from kernels.contracts import check_attention


@lru_cache(maxsize=1)
def load_extension():
    from torch.utils.cpp_extension import load

    return load(name="gpu_suite_attention_wmma",
                sources=[str(Path(__file__).parent / "attention_wmma.cu")],
                extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math"], verbose=False)


def attention(q, k, v, causal=False):
    check_attention(q, k, v)
    return load_extension().attention(q, k, v, causal)
