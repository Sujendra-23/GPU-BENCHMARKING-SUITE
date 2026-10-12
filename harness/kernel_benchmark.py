"""Correctness-gated fp16 kernel sweep: python -m harness.kernel_benchmark."""
import argparse

from harness.measurement import cuda_time, int_list, metadata, positive_int, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=["all", "softmax", "attention", "matmul"], default="all")
    parser.add_argument("--backends", default="triton,cuda", help="comma-separated: triton,cuda")
    parser.add_argument("--rows", type=positive_int, default=1024)
    parser.add_argument("--widths", type=int_list, default=[128, 781, 1024, 4096])
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--heads", type=positive_int, default=4)
    parser.add_argument("--seq-lens", type=int_list, default=[128, 257, 1024])
    parser.add_argument("--head-dims", type=int_list, default=[32, 64, 128])
    parser.add_argument("--causal", choices=["both", "yes", "no"], default="both")
    parser.add_argument("--iterations", type=positive_int, default=50)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--out", default="results/kernels/latest.json")
    args = parser.parse_args()
    backends = args.backends.split(",")
    if any(b not in ("triton", "cuda") for b in backends):
        parser.error("backends must be triton and/or cuda")
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("An NVIDIA CUDA GPU is required; no CPU timing fallback")
    torch.manual_seed(7)
    operations = {}
    if args.op != "matmul":
        for name in backends:
            if name == "cuda":
                from kernels.cuda import fused
                fused.load_extension()
                operations[name] = (fused.softmax, fused.attention)
            else:
                from kernels.triton.softmax import softmax
                from kernels.triton.attention import attention
                operations[name] = (softmax, attention)
    rows = []

    def record(op, backend, shape, fn, reference, *, causal=None, bytes_moved=None, flops=None):
        actual = fn()  # compilation, allocation and correctness outside timing
        tolerance = {"attention": (2e-3, 2e-3), "softmax": (1e-3, 1e-3),
                     "matmul": (1e-2, 5e-3)}[op]
        torch.testing.assert_close(actual.float(), reference.float(), atol=tolerance[0], rtol=tolerance[1])
        error = (actual.float() - reference.float()).abs().max().item()
        timing = cuda_time(fn, args.warmup, args.iterations)
        result = {"op": op, "backend": backend, "shape": list(shape), "dtype": "fp16",
                  "causal": causal, "correctness": "pass", "max_abs_error": error,
                  "atol": tolerance[0], "rtol": tolerance[1], **timing}
        if bytes_moved:
            result["effective_gbps"] = bytes_moved / (timing["median_ms"] * 1e6)
        if flops:
            result["effective_tflops"] = flops / (timing["median_ms"] * 1e9)
        rows.append(result)
        print(f"{op:10} {backend:14} {str(shape):24} causal={causal} "
              f"{timing['median_ms']:.4f} ms, error={error:.3g}")

    with torch.inference_mode():
        if args.op in ("all", "softmax"):
            for width in args.widths:
                x = torch.randn(args.rows, width, dtype=torch.float16, device="cuda")
                ref = torch.softmax(x.float(), dim=-1).half()
                io = 2 * x.numel() * x.element_size()
                record("softmax", "torch", x.shape, lambda: torch.softmax(x, -1), ref, bytes_moved=io)
                for backend, (softmax, _) in operations.items():
                    record("softmax", backend, x.shape, lambda: softmax(x), ref, bytes_moved=io)
        if args.op in ("all", "attention"):
            modes = [False, True] if args.causal == "both" else [args.causal == "yes"]
            for n in args.seq_lens:
                for d in args.head_dims:
                    shape = (args.batch, args.heads, n, d)
                    q, k, v = [torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3)]
                    for causal in modes:
                        # Math SDPA on fp32 is an independent reference, not the
                        # same fused implementation being benchmarked.
                        with sdpa_kernel(SDPBackend.MATH):
                            ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                                                is_causal=causal).half()
                        fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                        flops = 4 * args.batch * args.heads * n * n * d
                        if causal:
                            flops = 2 * args.batch * args.heads * n * (n + 1) * d
                        record("attention", "sdpa_auto", shape, fn, ref, causal=causal, flops=flops)
                        for backend, (_, attention) in operations.items():
                            record("attention", backend, shape, lambda: attention(q, k, v, causal),
                                   ref, causal=causal, flops=flops)
        if args.op in ("all", "matmul"):
            # torch.mm is a vendor BLAS baseline, not a CUTLASS implementation.
            for n in args.seq_lens:
                a, b = [torch.randn((n, n), device="cuda", dtype=torch.float16) for _ in range(2)]
                ref = (a.double() @ b.double()).half()
                record("matmul", "torch_blas", a.shape, lambda: a @ b, ref, flops=2 * n ** 3)
    write_json(args.out, {"metadata": metadata(["torch", "triton"]),
                         "device": torch.cuda.get_device_name(), "config": vars(args),
                         "timing": "CUDA events over 32-call CUDA Graphs; compilation and host allocation excluded",
                         "attention_reference": "PyTorch fp32 SDPA math backend; dropout=0",
                         "results": rows})


if __name__ == "__main__":
    main()
