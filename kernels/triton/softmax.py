"""A single fused max/subtract/exp/sum/divide kernel per row."""
import torch
import triton
import triton.language as tl

from kernels.contracts import check_softmax


@triton.jit
def _softmax(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, mask=col < N, other=-float("inf")).to(tl.float32)
    numerator = tl.exp(x - tl.max(x, axis=0))
    y = numerator / tl.sum(numerator, axis=0)
    tl.store(Y + row * N + col, y, mask=col < N)


def softmax(x):
    check_softmax(x)
    y = torch.empty_like(x)
    block = triton.next_power_of_2(x.shape[1])
    with torch.cuda.device(x.device):
        _softmax[(x.shape[0],)](x, y, x.shape[1], block, num_warps=8 if block > 2048 else 4)
    return y
