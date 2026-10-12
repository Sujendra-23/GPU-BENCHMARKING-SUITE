"""Lazy builds of the fp32 hand-written GEMMs and the CUTLASS fp16 GEMM."""
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def fp32():
    from torch.utils.cpp_extension import load

    return load(name="gpu_suite_gemm_fp32", sources=[str(ROOT / "cuda" / "gemm_fp32.cu")],
                extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


@lru_cache(maxsize=1)
def cutlass_fp16():
    """CUTLASS is header-only; set CUTLASS_DIR to a checkout (the runs used tag v3.5.1)."""
    from torch.utils.cpp_extension import load

    cutlass_dir = os.environ.get("CUTLASS_DIR")
    if not cutlass_dir:
        raise RuntimeError("Set CUTLASS_DIR to a CUTLASS checkout (git clone https://github.com/NVIDIA/cutlass)")
    include = [str(Path(cutlass_dir) / "include"), str(Path(cutlass_dir) / "tools" / "util" / "include")]
    return load(name="gpu_suite_cutlass_gemm", sources=[str(ROOT / "cutlass" / "gemm_fp16.cu")],
                extra_include_paths=include, extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
                verbose=False)
