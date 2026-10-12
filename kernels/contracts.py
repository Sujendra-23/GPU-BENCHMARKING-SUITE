"""Shared tensor contracts for the CUDA and Triton inference kernels."""


def check_tensor(x, ndim):
    import torch

    if not x.is_cuda or x.dtype != torch.float16:
        raise ValueError("Expected an NVIDIA CUDA fp16 tensor")
    if torch.version.hip is not None:
        raise ValueError("These benchmarks target NVIDIA CUDA, not ROCm")
    if x.ndim != ndim or not x.is_contiguous() or min(x.shape) < 1:
        raise ValueError(f"Expected nonempty contiguous rank-{ndim} input")
    if x.requires_grad:
        raise ValueError("Forward-only kernels do not support autograd")


def check_softmax(x):
    check_tensor(x, 2)
    if x.shape[1] > 8192:
        raise ValueError("Fused softmax supports at most 8192 columns")


def check_attention(q, k, v):
    for x in (q, k, v):
        check_tensor(x, 4)
        if x.shape != q.shape or x.device != q.device:
            raise ValueError("Q, K, V must have equal [B,H,N,D] shapes and devices")
    if q.shape[-1] not in (32, 64, 128):
        raise ValueError("Supported head dimensions: 32, 64, 128")
