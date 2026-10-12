"""Tiled online-softmax attention, without an N x N global-memory buffer.

FP16 QK/PV dot products use tensor cores; running max, sum and output use
FP32. Implements self-attention only, with optional causal masking.
"""
import torch
import triton
import triton.language as tl

from kernels.contracts import check_attention


@triton.jit
def _attention(Q, K, V, O, N: tl.constexpr, D: tl.constexpr,
               SCALE: tl.constexpr, CAUSAL: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.arange(0, BN)
    dim = tl.arange(0, D)
    base = tl.program_id(1).to(tl.int64) * N * D
    q = tl.load(Q + base + row[:, None].to(tl.int64) * D + dim[None, :],
                mask=row[:, None] < N, other=0)
    running_max = tl.full((BM,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BM,), tl.float32)
    output = tl.zeros((BM, D), tl.float32)
    end = N
    if CAUSAL:
        end = tl.minimum(N, (tl.program_id(0) + 1) * BM)
    for start in range(0, end, BN):
        key = start + col
        k = tl.load(K + base + dim[:, None] + key[None, :].to(tl.int64) * D,
                    mask=key[None, :] < N, other=0)
        v = tl.load(V + base + key[:, None].to(tl.int64) * D + dim[None, :],
                    mask=key[:, None] < N, other=0)
        score = tl.dot(q, k) * SCALE
        valid = key[None, :] < N
        if CAUSAL:
            valid = valid & (row[:, None] >= key[None, :])
        score = tl.where(valid, score, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(score, axis=1))
        alpha = tl.exp(running_max - new_max)
        probability = tl.exp(score - new_max[:, None])
        output = output * alpha[:, None] + tl.dot(probability.to(tl.float16), v)
        running_sum = running_sum * alpha + tl.sum(probability, axis=1)
        running_max = new_max
    output = output / running_sum[:, None]
    tl.store(O + base + row[:, None].to(tl.int64) * D + dim[None, :], output,
             mask=row[:, None] < N)


def attention(q, k, v, causal=False):
    check_attention(q, k, v)
    b, h, n, d = q.shape
    out = torch.empty_like(q)
    with torch.cuda.device(q.device):
        _attention[(triton.cdiv(n, 32), b * h)](
            q, k, v, out, n, d, d ** -0.5, causal, 32, 32,
            num_warps=4, num_stages=2)
    return out
