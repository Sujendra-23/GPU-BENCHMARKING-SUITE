"""GPU correctness checks for the tuned Triton attention, the WMMA CUDA attention,
the paged GQA decode kernel and the GEMMs. Skips explicitly without NVIDIA hardware."""
import importlib.util
import os

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch not installed", allow_module_level=True)
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="NVIDIA CUDA required")]


def reference(q, k, v, causal):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)


@pytest.fixture(params=["triton_tuned", "cuda_wmma"])
def attention(request):
    if request.param == "triton_tuned":
        from kernels.triton.attention_tuned import attention
    else:
        from kernels.cuda.wmma import attention
    return attention


@pytest.mark.parametrize("n", [1, 31, 64, 65, 257, 1000])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_attention(attention, n, d, causal):
    torch.manual_seed(7)
    q, k, v = [torch.randn((2, 3, n, d), device="cuda", dtype=torch.float16) for _ in range(3)]
    torch.testing.assert_close(attention(q, k, v, causal).float(), reference(q, k, v, causal),
                               atol=2e-3, rtol=2e-3)


def test_attention_contract(attention):
    q = torch.randn((1, 1, 16, 64), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        attention(q.float(), q.float(), q.float())


def test_gemm_fp32():
    from kernels.cuda import gemm

    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    a, b = [torch.randn((256, 256), device="cuda") for _ in range(2)]
    expected = a.double() @ b.double()
    for fn in (gemm.fp32().tiled16, gemm.fp32().regblock):
        torch.testing.assert_close(fn(a, b).double(), expected, atol=1e-3, rtol=1e-4)
    with pytest.raises(RuntimeError):
        gemm.fp32().regblock(a[:100], b)  # not a multiple of the 128 tile


@pytest.mark.skipif(not os.environ.get("CUTLASS_DIR"), reason="CUTLASS_DIR not set")
def test_gemm_cutlass_fp16():
    from kernels.cuda import gemm

    torch.manual_seed(1)
    a, b = [torch.randn((384, 512), device="cuda", dtype=torch.float16) for _ in range(1)] * 1 + \
           [torch.randn((512, 256), device="cuda", dtype=torch.float16)]
    expected = a.double() @ b.double()
    torch.testing.assert_close(gemm.cutlass_fp16().gemm(a, b).double(), expected, atol=0.25, rtol=1e-2)


@pytest.mark.parametrize("lengths", [[1], [16, 17, 300], [1000, 5, 64, 2049]])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("splits", [None, 1, 4])
def test_paged_decode(lengths, d, splits):
    from kernels.triton.paged_decode import paged_decode, reference

    torch.manual_seed(5)
    page, hq, hkv = 16, 32, 8
    b = len(lengths)
    max_pages = max((n + page - 1) // page for n in lengths)
    num_pages = b * max_pages + 3
    k_cache = torch.randn((num_pages, page, hkv, d), device="cuda", dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    # shuffled physical pages so the page table is really exercised
    perm = torch.randperm(num_pages, device="cuda", dtype=torch.int32)
    table = perm[: b * max_pages].reshape(b, max_pages).contiguous()
    seq_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    q = torch.randn((b, hq, d), device="cuda", dtype=torch.float16)
    actual = paged_decode(q, k_cache, v_cache, table, seq_lens, splits=splits)
    torch.testing.assert_close(actual.float(), reference(q, k_cache, v_cache, table, seq_lens),
                               atol=2e-3, rtol=2e-3)
