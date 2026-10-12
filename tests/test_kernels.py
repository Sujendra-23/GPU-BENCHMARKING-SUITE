"""Real GPU checks; skips are explicit when NVIDIA hardware is unavailable."""
import importlib.util

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch not installed", allow_module_level=True)
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="NVIDIA CUDA required")]


@pytest.fixture(params=["cuda", "triton"])
def backend(request):
    if request.param == "cuda":
        from kernels.cuda import fused
        return fused
    from kernels.triton import attention, softmax
    from types import SimpleNamespace
    return SimpleNamespace(attention=attention.attention, softmax=softmax.softmax)


@pytest.mark.parametrize("shape", [(1, 1), (3, 31), (7, 781), (17, 1024), (2, 8192)])
@pytest.mark.parametrize("scale", [1., 20.])
def test_softmax(backend, shape, scale):
    torch.manual_seed(7)
    x = torch.randn(shape, device="cuda", dtype=torch.float16) * scale
    actual = backend.softmax(x)
    torch.testing.assert_close(actual.float(), torch.softmax(x.float(), -1), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual.float().sum(-1), torch.ones(shape[0], device="cuda"),
                               atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("n", [1, 31, 65, 257])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_attention(backend, n, d, causal):
    torch.manual_seed(7)
    q, k, v = [torch.randn((2, 2, n, d), device="cuda", dtype=torch.float16) for _ in range(3)]
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)
    actual = backend.attention(q, k, v, causal)
    torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=2e-3)


def test_contracts_and_current_stream(backend):
    x = torch.randn((4, 129), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        backend.softmax(x.t())
    with pytest.raises(ValueError):
        backend.softmax(x.float())
    with pytest.raises(ValueError):
        backend.softmax(x.requires_grad_())
    x.requires_grad_(False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = backend.softmax(x * 2)
    stream.synchronize()
    torch.testing.assert_close(y, torch.softmax(x * 2, -1), atol=1e-3, rtol=1e-3)
