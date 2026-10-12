// CUTLASS device-level fp16 tensor-core GEMM (Sm80 kernel, runs on sm_89),
// row-major A[M,K] * B[K,N] -> C[M,N], fp32 accumulation, fp16 output.
// Tile config follows CUTLASS's Ampere tensor-op examples (128x128x32 block,
// 64x64x32 warp, 16x8x16 mma, 4 stages).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

torch::Tensor gemm(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf &&
                a.dim() == 2 && b.dim() == 2 && a.is_contiguous() && b.is_contiguous() && a.size(1) == b.size(0),
                "expected contiguous fp16 CUDA matrices A[M,K], B[K,N]");
    c10::cuda::CUDAGuard guard(a.device());
    int M = a.size(0), K = a.size(1), N = b.size(1);
    TORCH_CHECK(K % 8 == 0 && N % 8 == 0, "K and N must be multiples of 8 (128-bit alignment)");
    auto c = torch::empty({M, N}, a.options());
    Gemm op;
    Gemm::Arguments args({M, N, K},
        {reinterpret_cast<cutlass::half_t*>(a.data_ptr<at::Half>()), K},
        {reinterpret_cast<cutlass::half_t*>(b.data_ptr<at::Half>()), N},
        {reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()), N},
        {reinterpret_cast<cutlass::half_t*>(c.data_ptr<at::Half>()), N},
        {1.0f, 0.0f});
    TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "CUTLASS cannot implement this problem");
    auto status = op(args, nullptr, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM failed: ", cutlassGetStatusString(status));
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "CUTLASS fp16 tensor-core GEMM (128x128x32, 4 stages)");
}
