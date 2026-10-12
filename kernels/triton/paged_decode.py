"""Paged-KV grouped-query (GQA) decode attention in Triton, split-KV.

One new token per sequence (decode). K/V live in a paged cache laid out like
FlashInfer's NHD layout: cache[num_pages, page_size, num_kv_heads, head_dim].
Each sequence has its own length and a row of the page table. Stage 1: one
program per (sequence, kv head, KV split) computes a partial online-softmax
result for all query heads of that KV group using tensor-core tl.dot (the
group is padded to 16 rows). Stage 2 merges the splits.
"""
import torch
import triton
import triton.language as tl

GROUP_PAD = 16  # minimum M for tl.dot


@triton.jit
def _stage1(Q, KC, VC, PT, SEQ, PO, PM, PL, sm_scale_log2, max_pages,
            HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
            SPLITS: tl.constexpr, BN: tl.constexpr, GP: tl.constexpr):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    group: tl.constexpr = HQ // HKV
    seq_len = tl.load(SEQ + b)
    per_split = tl.cdiv(tl.cdiv(seq_len, SPLITS), BN) * BN
    lo = sp * per_split
    hi = tl.minimum(seq_len, lo + per_split)
    g = tl.arange(0, GP)
    head = hk * group + g
    dim = tl.arange(0, D)
    q = tl.load(Q + (b * HQ + head)[:, None] * D + dim[None, :], mask=(g < group)[:, None], other=0.0)
    m = tl.full((GP,), -float("inf"), tl.float32)
    l = tl.zeros((GP,), tl.float32)
    acc = tl.zeros((GP, D), tl.float32)
    offs = tl.arange(0, BN)
    for start in range(lo, hi, BN):
        tok = start + offs
        valid = tok < hi
        page = tl.load(PT + b * max_pages + tok // PAGE, mask=valid, other=0)
        row = (page.to(tl.int64) * PAGE + tok % PAGE) * HKV + hk
        kt = tl.load(KC + row[None, :] * D + dim[:, None], mask=valid[None, :], other=0.0)   # [D, BN]
        s = tl.dot(q, kt) * sm_scale_log2
        s = tl.where(valid[None, :], s, -float("inf"))
        new_m = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.math.exp2(m - new_m)
        p = tl.math.exp2(s - new_m[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        v = tl.load(VC + row[:, None] * D + dim[None, :], mask=valid[:, None], other=0.0)    # [BN, D]
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m = new_m
    base = ((b * HQ + head) * SPLITS + sp)
    mask_g = g < group
    tl.store(PO + base[:, None] * D + dim[None, :], acc, mask=mask_g[:, None])
    tl.store(PM + base, m, mask=mask_g)
    tl.store(PL + base, l, mask=mask_g)


@triton.jit
def _stage2(PO, PM, PL, O, HQ: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr):
    bh = tl.program_id(0)
    sp = tl.arange(0, SPLITS)
    dim = tl.arange(0, D)
    m = tl.load(PM + bh * SPLITS + sp)
    l = tl.load(PL + bh * SPLITS + sp)
    top = tl.max(m, axis=0)
    w = tl.math.exp2(m - top)
    o = tl.load(PO + (bh * SPLITS + sp)[:, None] * D + dim[None, :])
    out = tl.sum(o * w[:, None], axis=0) / tl.sum(l * w, axis=0)
    tl.store(O + bh * D + dim, out.to(tl.float16))


def default_splits(batch, kv_heads, max_len, block_n, target_programs=512):
    splits = 1
    while batch * kv_heads * splits < target_programs and splits * block_n * 2 <= max_len and splits < 32:
        splits *= 2
    return splits


def paged_decode(q, k_cache, v_cache, page_table, seq_lens, splits=None, block_n=64, validate=True):
    """q [B,HQ,D] fp16; caches [P,PAGE,HKV,D] fp16; page_table [B,max_pages] int32; seq_lens [B] int32.

    validate=False skips the device-to-host check on seq_lens (needed inside CUDA Graph capture)."""
    b, hq, d = q.shape
    _, page, hkv, _ = k_cache.shape
    if not (q.dtype == k_cache.dtype == v_cache.dtype == torch.float16 and q.is_cuda):
        raise ValueError("fp16 CUDA tensors required")
    if not (q.is_contiguous() and k_cache.is_contiguous() and v_cache.is_contiguous() and page_table.is_contiguous()):
        raise ValueError("contiguous tensors required")
    if hq % hkv or hq // hkv > GROUP_PAD or d not in (64, 128) or page & (page - 1):
        raise ValueError("need HQ % HKV == 0, group <= 16, head_dim 64 or 128, power-of-two page size")
    if validate and (seq_lens < 1).any():
        raise ValueError("every sequence needs at least one cached token")
    max_pages = page_table.shape[1]
    if splits is None:
        splits = default_splits(b, hkv, max_pages * page, block_n)
    out = torch.empty_like(q)
    po = torch.empty((b * hq, splits, d), dtype=torch.float32, device=q.device)
    pm = torch.empty((b * hq, splits), dtype=torch.float32, device=q.device)
    pl = torch.empty_like(pm)
    with torch.cuda.device(q.device):
        _stage1[(b, hkv, splits)](q, k_cache, v_cache, page_table, seq_lens, po, pm, pl,
                                  d ** -0.5 * 1.4426950408889634, max_pages,
                                  HQ=hq, HKV=hkv, D=d, PAGE=page, SPLITS=splits, BN=block_n,
                                  GP=GROUP_PAD, num_warps=4, num_stages=3)
        _stage2[(b * hq,)](po, pm, pl, out, HQ=hq, D=d, SPLITS=splits, num_warps=4)
    return out


def reference(q, k_cache, v_cache, page_table, seq_lens):
    """fp32 PyTorch reference: gather each sequence's pages and run softmax(QK^T)V per query head."""
    b, hq, d = q.shape
    _, page, hkv, _ = k_cache.shape
    out = torch.empty((b, hq, d), dtype=torch.float32, device=q.device)
    for i in range(b):
        n = int(seq_lens[i])
        pages = page_table[i, : (n + page - 1) // page].long()
        k = k_cache[pages].reshape(-1, hkv, d)[:n].float()   # [n,HKV,D]
        v = v_cache[pages].reshape(-1, hkv, d)[:n].float()
        k = k.repeat_interleave(hq // hkv, dim=1)            # [n,HQ,D]
        v = v.repeat_interleave(hq // hkv, dim=1)
        s = torch.einsum("hd,nhd->hn", q[i].float(), k) * d ** -0.5
        out[i] = torch.einsum("hn,nhd->hd", torch.softmax(s, -1), v)
    return out
