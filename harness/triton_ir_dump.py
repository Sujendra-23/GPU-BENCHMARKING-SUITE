"""Dump Triton's own MLIR pipeline for the repo's attention kernels.

For a compiled Triton kernel, `.asm` holds each lowering stage:
  ttir  - Triton dialect (tt.*): hardware-independent tile program
  ttgir - TritonGPU dialect (ttg.*/#triton_gpu.*): layouts (blocked/mma/shared) assigned,
          tensor-core dots and async copies chosen
  llir  - LLVM IR (NVVM): lowered via the LLVM dialect
  ptx   - NVIDIA PTX assembly
  cubin - machine code (not saved as text)
"""
import argparse
import json
import re
from pathlib import Path

from harness.measurement import metadata, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--out-dir", default="results/triton_ir")
    args = parser.parse_args()
    import torch
    import triton

    from kernels.triton import attention as v1
    from kernels.triton import attention_tuned as tuned

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    b, h, n, d = 1, 4, args.n, args.head_dim
    q, k, v = [torch.randn((b, h, n, d), device="cuda", dtype=torch.float16) for _ in range(3)]
    o = torch.empty_like(q)
    summary = {}

    handle = v1._attention[(triton.cdiv(n, 32), b * h)](q, k, v, o, n, d, d ** -0.5, True, 32, 32,
                                                         num_warps=4, num_stages=2)
    handles = {"attention_v1_causal_BM32_BN32": handle}
    tuned.attention(q, k, v, True)  # populate the autotuner
    config = tuned.best_config(q, k, v, True)
    handle = tuned._attn_fwd.fn[(triton.cdiv(n, config["BM"]), b * h)](
        q, k, v, o, n, d, d ** -0.5 * tuned.LOG2E, True, BM=config["BM"], BN=config["BN"],
        num_warps=config["num_warps"], num_stages=config["num_stages"])
    handles[f"attention_tuned_causal_BM{config['BM']}_BN{config['BN']}_w{config['num_warps']}_s{config['num_stages']}"] = handle
    for name, handle in handles.items():
        stages = {}
        for stage in ("ttir", "ttgir", "llir", "ptx"):
            text = handle.asm[stage]
            (out / f"{name}.{stage}").write_text(text)
            stages[stage] = {"lines": len(text.splitlines())}
        ttir, ttgir, ptx = (handle.asm[s] for s in ("ttir", "ttgir", "ptx"))
        stages["ttir"]["tt.dot"] = len(re.findall(r"tt\.dot", ttir))
        stages["ttgir"]["dot_ops"] = len(re.findall(r"tt\.dot", ttgir))
        stages["ttgir"]["layouts"] = sorted(set(re.findall(r"#(?:triton_gpu\.)?(blocked\d*|mma\d*|shared\d*|dot_op\d*)", ttgir)))
        stages["ptx"]["mma_instructions"] = len(re.findall(r"\bmma\.sync", ptx))
        stages["ptx"]["ex2_instructions"] = len(re.findall(r"ex2\.approx", ptx))
        stages["ptx"]["cp_async_instructions"] = len(re.findall(r"cp\.async", ptx))
        summary[name] = stages
    write_json(out / "summary.json", {"metadata": metadata(["torch", "triton"]),
                                      "shape": [b, h, n, d], "causal": True, "kernels": summary})


if __name__ == "__main__":
    main()
