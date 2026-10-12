"""LLaMA-style transformer block (RMSNorm, RoPE attention, SwiGLU MLP) in fp16:
eager vs torch.compile. One process per mode so each gets a fresh Inductor/Triton
cache and a clean cold-compile measurement:

    python -m compilers.llama_block --mode eager
    python -m compilers.llama_block --mode default
    python -m compilers.llama_block --mode max-autotune
"""
import argparse
import hashlib
import os
import shutil
import time
from pathlib import Path

from harness.measurement import cuda_time, metadata, positive_int, write_json


def build(torch, hidden, heads, inter, device):
    g = torch.Generator(device="cpu").manual_seed(11)

    def w(*shape):
        return (torch.randn(shape, generator=g) * 0.02).to(device=device, dtype=torch.float16)

    return {"wq": w(hidden, hidden), "wk": w(hidden, hidden), "wv": w(hidden, hidden), "wo": w(hidden, hidden),
            "w_gate": w(inter, hidden), "w_up": w(inter, hidden), "w_down": w(hidden, inter),
            "norm1": torch.ones(hidden, device=device, dtype=torch.float16),
            "norm2": torch.ones(hidden, device=device, dtype=torch.float16)}


def make_block(torch, heads):
    F = torch.nn.functional

    def rms_norm(x, weight, eps=1e-5):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def block(x, p, cos, sin):
        b, s, hidden = x.shape
        d = hidden // heads
        h = rms_norm(x, p["norm1"])
        q, k, v = [(h @ p[n].t()).view(b, s, heads, d).transpose(1, 2) for n in ("wq", "wk", "wv")]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + a.transpose(1, 2).reshape(b, s, hidden) @ p["wo"].t()
        h = rms_norm(x, p["norm2"])
        return x + (F.silu(h @ p["w_gate"].t()) * (h @ p["w_up"].t())) @ p["w_down"].t()

    return block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["eager", "default", "max-autotune"], required=True)
    parser.add_argument("--hidden", type=positive_int, default=4096)
    parser.add_argument("--heads", type=positive_int, default=32)
    parser.add_argument("--inter", type=positive_int, default=11008)
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--seq", type=positive_int, default=2048)
    parser.add_argument("--iterations", type=positive_int, default=30)
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--out-dir", default="results/compilers/llama_block")
    args = parser.parse_args()
    root = Path(args.out_dir).resolve() / args.mode
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(root / "debug")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root / "cache")
    os.environ["TRITON_CACHE_DIR"] = str(root / "triton-cache")
    import torch

    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("NVIDIA CUDA required")
    torch.manual_seed(0)
    params = build(torch, args.hidden, args.heads, args.inter, "cuda")
    d = args.hidden // args.heads
    x = torch.randn((args.batch, args.seq, args.hidden), device="cuda", dtype=torch.float16)
    pos = torch.arange(args.seq, device="cuda").float()
    freqs = torch.outer(pos, 1.0 / (10000 ** (torch.arange(0, d, 2, device="cuda").float() / d)))
    angles = torch.cat((freqs, freqs), dim=-1)
    cos, sin = angles.cos().half()[None, None], angles.sin().half()[None, None]
    block = make_block(torch, args.heads)

    with torch.inference_mode():
        # fp32 reference: same weights upcast, run through the same eager code.
        ref_params = {k: v.float() for k, v in params.items()}
        reference = block(x.float(), ref_params, cos.float(), sin.float())
        del ref_params
        fn = block
        compile_ms = None
        if args.mode != "eager":
            mode = None if args.mode == "default" else "max-autotune-no-cudagraphs"
            fn = torch.compile(block, backend="inductor", mode=mode, fullgraph=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn(x, params, cos, sin)
            torch.cuda.synchronize()
            compile_ms = (time.perf_counter() - start) * 1000
        actual = fn(x, params, cos, sin)
        error = (actual.float() - reference).abs().max().item()
        rel = error / reference.abs().max().item()
        timing = cuda_time(lambda: fn(x, params, cos, sin), args.warmup, args.iterations, graph_repeats=2)
    timing.pop("samples_ms")
    saved = []
    if args.mode != "eager":
        for path in sorted(root.rglob("output_code.py")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            target = root / f"output_code_{digest}.py"
            shutil.copy(path, target)
            saved.append({"path": str(target.relative_to(root)),
                          "triton_kernels": path.read_text().count("@triton.jit")})
        shutil.rmtree(root / "cache", ignore_errors=True)
        shutil.rmtree(root / "triton-cache", ignore_errors=True)
        shutil.rmtree(root / "debug", ignore_errors=True)
    result = {"metadata": metadata(["torch", "triton"]), "config": vars(args),
              "device": torch.cuda.get_device_name(), "dtype": "fp16",
              "inductor_mode": {"default": "default", "max-autotune": "max-autotune-no-cudagraphs"}.get(args.mode),
              "reference": "same block in fp32 with the same (upcast) weights",
              "max_abs_error": error, "relative_max_error": rel,
              "cold_compile_and_first_call_ms": compile_ms, "timing": timing, "generated": saved}
    write_json(root / "benchmark.json", result)
    print(f"{args.mode}: median {timing['median_ms']:.3f} ms, max abs err {error:.3g} ({rel:.2%} of max |ref|)"
          + (f", cold compile {compile_ms / 1000:.1f} s" if compile_ms else ""))


if __name__ == "__main__":
    main()
