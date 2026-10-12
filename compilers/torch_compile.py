"""Inspect actual Inductor-generated Triton for a fused fp16 softmax graph."""
import argparse
import ast
import hashlib
import os
import time
from pathlib import Path

from harness.measurement import cuda_time, metadata, positive_int, write_json


def inspect_source(source):
    """Inspect the emitted Python AST; do not infer generated code from timing."""
    if "@triton.jit" not in source:
        return {"triton_jit_functions": [], "contains_reduction": False,
                "contains_exp": False, "source_lines": len(source.splitlines())}
    tree = ast.parse(source)
    functions = []
    for node in ast.walk(tree):
        # Inductor embeds Triton modules as string literals passed to
        # async_compile.triton(...), rather than defining them at top level.
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "@triton.jit" in node.value:
            functions.extend(inspect_source(node.value)["triton_jit_functions"])
        if isinstance(node, ast.FunctionDef):
            decorators = [ast.unparse(d) for d in node.decorator_list]
            if any(d.endswith("triton.jit") for d in decorators):
                functions.append(node.name)
    return {"triton_jit_functions": functions,
            "contains_reduction": "tl.sum" in source or "tl.max" in source,
            "contains_exp": "exp" in source,
            "source_lines": len(source.splitlines())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=positive_int, default=1024)
    parser.add_argument("--cols", type=positive_int, default=1024)
    parser.add_argument("--iterations", type=positive_int, default=50)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--out-dir", default="results/compilers/torch")
    args = parser.parse_args()
    root = Path(args.out_dir).resolve()
    # A distinct directory avoids stale cached output being mistaken for
    # newly emitted code. Environment flags must precede importing torch.
    run = root / f"run-{time.time_ns()}"
    run.mkdir(parents=True)
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(run / "debug")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(run / "cache")
    os.environ["TRITON_CACHE_DIR"] = str(run / "triton-cache")
    import torch

    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("NVIDIA CUDA required to inspect GPU-generated Triton")
    torch.manual_seed(7)

    def graph(x):
        values = x.float() * 0.7 + 0.1
        values = values - values.amax(dim=-1, keepdim=True)
        numerator = values.exp()
        return (numerator / numerator.sum(dim=-1, keepdim=True)).half()

    with torch.inference_mode():
        x = torch.randn((args.rows, args.cols), dtype=torch.float16, device="cuda")
        reference = torch.softmax(x.double() * .7 + .1, dim=-1).half()
        eager = graph(x)
        torch.testing.assert_close(eager, reference, atol=1e-3, rtol=1e-3)
        torch.cuda.synchronize()
        start = time.perf_counter()
        compiled = torch.compile(graph, backend="inductor", fullgraph=True)
        actual = compiled(x)
        torch.cuda.synchronize()
        compile_first_call_ms = (time.perf_counter() - start) * 1000
        torch.testing.assert_close(actual, reference, atol=1e-3, rtol=1e-3)
        eager_timing = cuda_time(lambda: graph(x), args.warmup, args.iterations)
        compiled_timing = cuda_time(lambda: compiled(x), args.warmup, args.iterations)
    artifacts = []
    seen = set()
    generated = root / "generated"
    generated.mkdir(exist_ok=True)
    for path in sorted(run.rglob("*.py")):
        source = path.read_text()
        info = inspect_source(source)
        if info["triton_jit_functions"]:
            digest = hashlib.sha256(source.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            saved = generated / f"{digest[:16]}.py"
            saved.write_text(source)
            artifacts.append({"path": str(saved.relative_to(root)),
                              "emitted_from": str(path.relative_to(root)), **info})
    if not artifacts:
        raise RuntimeError(f"No actual Triton functions found in {run}; inspect debug logs")
    result = {"metadata": metadata(["torch", "triton"]), "config": vars(args),
              "device": torch.cuda.get_device_name(), "dtype": "fp16", "correctness": "pass",
              "max_abs_error": (actual.float() - reference.float()).abs().max().item(),
              "compile_and_first_call_ms": compile_first_call_ms,
              "eager": eager_timing, "compiled": compiled_timing,
              "speedup": eager_timing["median_ms"] / compiled_timing["median_ms"],
              "generated_triton": artifacts}
    write_json(root / "benchmark.json", result)
    print(f"Compiled/eager speedup: {result['speedup']:.2f}x; inspect {len(artifacts)} emitted files")


if __name__ == "__main__":
    main()
