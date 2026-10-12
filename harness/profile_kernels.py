"""Profile one warmed launch using Nsight Compute --profile-from-start off."""
import argparse

from harness.measurement import positive_int


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cuda", "triton"], required=True)
    parser.add_argument("--op", choices=["softmax", "attention"], required=True)
    parser.add_argument("--n", type=positive_int, default=1024)
    parser.add_argument("--head-dim", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()
    import torch
    if args.backend == "cuda":
        from kernels.cuda.fused import softmax, attention
    else:
        from kernels.triton.softmax import softmax
        from kernels.triton.attention import attention
    torch.manual_seed(7)
    with torch.inference_mode():
        if args.op == "softmax":
            x = torch.randn((1024, args.n), device="cuda", dtype=torch.float16)
            fn = lambda: softmax(x)
        else:
            q, k, v = [torch.randn((1, 4, args.n, args.head_dim), device="cuda", dtype=torch.float16)
                       for _ in range(3)]
            fn = lambda: attention(q, k, v, args.causal)
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
