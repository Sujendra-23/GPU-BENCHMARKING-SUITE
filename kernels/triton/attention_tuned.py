"""Autotuned FlashAttention-2-style forward kernel in Triton.

Differences from kernels/triton/attention.py (kept unchanged for comparison):
block pointers, exp2-based online softmax with the softmax scale folded in,
causal tiles split into an unmasked part and a masked diagonal part, and an
autotuned BLOCK_M / BLOCK_N / num_warps / num_stages per (N, D, causal).
"""
import torch
import triton
import triton.language as tl

from kernels.contracts import check_attention

LOG2E = 1.4426950408889634

CONFIGS = [
    triton.Config({"BM": bm, "BN": bn}, num_warps=warps, num_stages=stages)
    for bm in (64, 128)
    for bn in (32, 64, 128)
    if bn <= bm
    for warps in (4, 8)
    for stages in (2, 3, 4)
]


@triton.jit
def _inner(acc, row_sum, row_max, q, k_ptr, v_ptr, lo, hi, offs_m, offs_n, n_ctx,
           qk_scale, BN: tl.constexpr, MASK_CAUSAL: tl.constexpr, MASK_BOUNDS: tl.constexpr):
    k_ptr = tl.advance(k_ptr, (0, lo))
    v_ptr = tl.advance(v_ptr, (lo, 0))
    for start in range(lo, hi, BN):
        if MASK_BOUNDS:
            k = tl.load(k_ptr, boundary_check=(1,), padding_option="zero")
        else:
            k = tl.load(k_ptr)
        score = tl.dot(q, k) * qk_scale
        key = start + offs_n
        if MASK_CAUSAL:
            score = tl.where(offs_m[:, None] >= key[None, :], score, -float("inf"))
        if MASK_BOUNDS:
            score = tl.where(key[None, :] < n_ctx, score, -float("inf"))
        new_max = tl.maximum(row_max, tl.max(score, axis=1))
        alpha = tl.math.exp2(row_max - new_max)
        prob = tl.math.exp2(score - new_max[:, None])
        row_sum = row_sum * alpha + tl.sum(prob, axis=1)
        if MASK_BOUNDS:
            v = tl.load(v_ptr, boundary_check=(0,), padding_option="zero")
        else:
            v = tl.load(v_ptr)
        acc = acc * alpha[:, None] + tl.dot(prob.to(tl.float16), v)
        row_max = new_max
        k_ptr = tl.advance(k_ptr, (0, BN))
        v_ptr = tl.advance(v_ptr, (BN, 0))
    return acc, row_sum, row_max


@triton.autotune(configs=CONFIGS, key=["N", "D", "CAUSAL"])
@triton.jit
def _attn_fwd(Q, K, V, O, N, D: tl.constexpr, qk_scale, CAUSAL: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr):
    start_m = tl.program_id(0)
    base = tl.program_id(1).to(tl.int64) * N * D
    q_ptr = tl.make_block_ptr(Q + base, (N, D), (D, 1), (start_m * BM, 0), (BM, D), (1, 0))
    o_ptr = tl.make_block_ptr(O + base, (N, D), (D, 1), (start_m * BM, 0), (BM, D), (1, 0))
    k_ptr = tl.make_block_ptr(K + base, (D, N), (1, D), (0, 0), (D, BN), (0, 1))
    v_ptr = tl.make_block_ptr(V + base, (N, D), (D, 1), (0, 0), (BN, D), (1, 0))
    offs_m = start_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    row_max = tl.full((BM,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    q = tl.load(q_ptr, boundary_check=(0,), padding_option="zero")
    if CAUSAL:
        # Tiles strictly below the diagonal block need no mask at all (BN <= BM
        # and BM is a multiple of BN, so start_m * BM is a tile boundary).
        acc, row_sum, row_max = _inner(acc, row_sum, row_max, q, k_ptr, v_ptr, 0, start_m * BM,
                                       offs_m, offs_n, N, qk_scale, BN, False, False)
        hi = tl.minimum(N, (start_m + 1) * BM)
        acc, row_sum, row_max = _inner(acc, row_sum, row_max, q, k_ptr, v_ptr, start_m * BM, hi,
                                       offs_m, offs_n, N, qk_scale, BN, True, True)
    else:
        acc, row_sum, row_max = _inner(acc, row_sum, row_max, q, k_ptr, v_ptr, 0, N,
                                       offs_m, offs_n, N, qk_scale, BN, False, True)
    acc = acc / row_sum[:, None]
    tl.store(o_ptr, acc.to(tl.float16), boundary_check=(0,))


def attention(q, k, v, causal=False):
    check_attention(q, k, v)
    b, h, n, d = q.shape
    out = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(n, meta["BM"]), b * h)
    with torch.cuda.device(q.device):
        _attn_fwd[grid](q, k, v, out, n, d, d ** -0.5 * LOG2E, causal)
    return out


def best_config(q, k, v, causal=False):
    """Return the config autotune picked for this shape (after attention() ran)."""
    key = (q.shape[2], q.shape[3], causal)
    for cached_key, config in _attn_fwd.cache.items():
        if tuple(cached_key[:3]) == key:
            return {**config.kwargs, "num_warps": config.num_warps, "num_stages": config.num_stages}
    return None
