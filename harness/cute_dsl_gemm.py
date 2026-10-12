"""Run CUTLASS's Python CuTe DSL Ampere tensor-op GEMM example on an NVIDIA GPU.

This is NOT code written for this repo: it runs examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/
tensorop_gemm.py from a CUTLASS checkout (CUTLASS_DSL_DIR, tag v4.8.0 for the recorded run) with the
installed `nvidia-cutlass-dsl`. Two environment details were needed on the 4090 pod:
  * `pip install "cuda-bindings>=12.9,<13"` -- the default cuda-bindings 13.x needs a CUDA 13 driver and
    fails with cudaErrorInsufficientDriver on driver 570 (CUDA 12.8).
  * a torch-dtype shim below, because the example references torch dtypes missing from torch 2.4
    (the fp16 path used here does not touch them).
Usage: CUTLASS_DSL_DIR=/path/to/cutlass python -m harness.cute_dsl_gemm --mnkl 4096,4096,4096,1 [example args]
"""
import os
import runpy
import sys

import torch

for name in ("float4_e2m1fn_x2", "float8_e8m0fnu", "float8_e4m3fn", "float8_e5m2"):
    if not hasattr(torch, name):
        setattr(torch, name, torch.uint8)

root = os.environ.get("CUTLASS_DSL_DIR")
if not root:
    raise SystemExit("Set CUTLASS_DSL_DIR to a CUTLASS (v4.x) checkout")
script = os.path.join(root, "examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
