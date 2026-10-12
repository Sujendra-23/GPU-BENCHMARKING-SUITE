"""Same softmax graph through JAX jit, with StableHLO and XLA HLO dumps."""
import argparse
import time
from pathlib import Path

from harness.measurement import metadata, positive_int, summarize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=positive_int, default=1024)
    parser.add_argument("--cols", type=positive_int, default=1024)
    parser.add_argument("--iterations", type=positive_int, default=50)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--out-dir", default="results/compilers/jax")
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np

    try:
        device = jax.devices(args.device)[0]
    except RuntimeError:
        parser.error(f"No JAX {args.device} backend installed; select --device cpu for CPU-only validation")
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    def graph(x):
        values = x.astype(jnp.float32) * .7 + .1
        values = values - jnp.max(values, axis=-1, keepdims=True)
        numerator = jnp.exp(values)
        return (numerator / jnp.sum(numerator, axis=-1, keepdims=True)).astype(jnp.float16)

    host = np.random.default_rng(7).normal(size=(args.rows, args.cols)).astype(np.float16)
    x = jax.device_put(host, device).block_until_ready()  # transfers outside timing
    jit = jax.jit(graph)
    start = time.perf_counter()
    lowered = jit.lower(x)
    compiled = lowered.compile()
    compile_ms = (time.perf_counter() - start) * 1000
    (root / "stablehlo.mlir").write_text(str(lowered.compiler_ir(dialect="stablehlo")))
    (root / "unoptimized.hlo").write_text(lowered.compiler_ir(dialect="hlo").as_hlo_text())
    (root / "optimized.hlo").write_text(compiled.as_text())
    values = host.astype(np.float64) * .7 + .1
    values -= values.max(axis=-1, keepdims=True)
    numerator = np.exp(values)
    reference = (numerator / numerator.sum(axis=-1, keepdims=True)).astype(np.float16)
    actual = np.asarray(compiled(x).block_until_ready())
    eager = np.asarray(graph(x).block_until_ready())
    np.testing.assert_allclose(actual, reference, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(eager, reference, atol=1e-3, rtol=1e-3)

    def measure(fn):
        for _ in range(args.warmup):
            fn(x).block_until_ready()
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            fn(x).block_until_ready()
            samples.append((time.perf_counter() - start) * 1000)
        return summarize(samples)

    eager_timing, jit_timing = measure(graph), measure(compiled)
    result = {"metadata": metadata(["jax", "jaxlib", "numpy"]), "config": vars(args),
              "device": str(device), "platform": device.platform, "dtype": "fp16",
              "correctness": "pass", "max_abs_error": float(np.max(np.abs(actual.astype(float) - reference))),
              "compile_ms": compile_ms, "eager": eager_timing, "jit": jit_timing,
              "speedup": eager_timing["median_ms"] / jit_timing["median_ms"],
              "timing": "wall-clock per call + block_until_ready; compilation/transfers excluded",
              "artifacts": ["stablehlo.mlir", "unoptimized.hlo", "optimized.hlo"]}
    write_json(root / "benchmark.json", result)
    print(f"JAX {device.platform} jit/eager speedup: {result['speedup']:.2f}x")


if __name__ == "__main__":
    main()
