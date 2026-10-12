// fp32 GEMM kernels exposed to PyTorch so they can be timed against cuBLAS
// (torch.matmul with TF32 disabled) on the same tensors.
//   tiled16   : the 16x16 shared-memory tiling of kernels/cuda/matmul.cu
//   regblock  : 128x128x8 block tile, 8x8 outputs per thread in registers,
//               A staged transposed in shared memory, float4 global loads
// Row-major C[M,N] = A[M,K] * B[K,N]; regblock requires M,N multiples of 128
// and K a multiple of 8 (checked on the host).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 16

__global__ void matmul_tiled16(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x, bRow = t * TILE_SIZE + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

constexpr int RB_BM = 128, RB_BN = 128, RB_BK = 8, RB_TM = 8, RB_TN = 8;

__global__ void __launch_bounds__(256) matmul_regblock(const float* __restrict__ A, const float* __restrict__ B,
                                                        float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[RB_BK][RB_BM];  // A tile stored transposed: As[k][m]
    __shared__ float Bs[RB_BK][RB_BN];
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    const float* Ablk = A + (size_t)blockIdx.y * RB_BM * K;
    const float* Bblk = B + (size_t)blockIdx.x * RB_BN;
    float acc[RB_TM][RB_TN] = {};
    // loader mapping: 256 threads x float4 covers one 128x8 (A) and one 8x128 (B) tile
    const int a_row = threadIdx.x / 2, a_col = (threadIdx.x % 2) * 4;
    const int b_row = threadIdx.x / 32, b_col = (threadIdx.x % 32) * 4;
    for (int k0 = 0; k0 < K; k0 += RB_BK) {
        float4 a4 = *reinterpret_cast<const float4*>(Ablk + (size_t)a_row * K + k0 + a_col);
        As[a_col + 0][a_row] = a4.x; As[a_col + 1][a_row] = a4.y;
        As[a_col + 2][a_row] = a4.z; As[a_col + 3][a_row] = a4.w;
        *reinterpret_cast<float4*>(&Bs[b_row][b_col]) =
            *reinterpret_cast<const float4*>(Bblk + (size_t)(k0 + b_row) * N + b_col);
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < RB_BK; ++k) {
            float ra[RB_TM], rb[RB_TN];
            #pragma unroll
            for (int i = 0; i < RB_TM; i += 4)
                *reinterpret_cast<float4*>(&ra[i]) = *reinterpret_cast<float4*>(&As[k][ty * 4 + (i / 4) * 64]);
            #pragma unroll
            for (int j = 0; j < RB_TN; j += 4)
                *reinterpret_cast<float4*>(&rb[j]) = *reinterpret_cast<float4*>(&Bs[k][tx * 4 + (j / 4) * 64]);
            #pragma unroll
            for (int i = 0; i < RB_TM; ++i)
                #pragma unroll
                for (int j = 0; j < RB_TN; ++j) acc[i][j] = fmaf(ra[i], rb[j], acc[i][j]);
        }
        __syncthreads();
    }
    float* Cblk = C + (size_t)blockIdx.y * RB_BM * N + (size_t)blockIdx.x * RB_BN;
    #pragma unroll
    for (int i = 0; i < RB_TM; ++i) {
        int r = ty * 4 + (i / 4) * 64 + (i % 4);
        #pragma unroll
        for (int j = 0; j < RB_TN; j += 4) {
            int c = tx * 4 + (j / 4) * 64;
            *reinterpret_cast<float4*>(Cblk + (size_t)r * N + c) = *reinterpret_cast<float4*>(&acc[i][j]);
        }
    }
}

static void check_inputs(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat &&
                a.dim() == 2 && b.dim() == 2 && a.is_contiguous() && b.is_contiguous() && a.size(1) == b.size(0),
                "expected contiguous fp32 CUDA matrices A[M,K], B[K,N]");
}

torch::Tensor tiled16(torch::Tensor a, torch::Tensor b) {
    check_inputs(a, b);
    c10::cuda::CUDAGuard guard(a.device());
    int M = a.size(0), K = a.size(1), N = b.size(1);
    auto c = torch::empty({M, N}, a.options());
    dim3 block(TILE_SIZE, TILE_SIZE), grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled16<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

torch::Tensor regblock(torch::Tensor a, torch::Tensor b) {
    check_inputs(a, b);
    int M = a.size(0), K = a.size(1), N = b.size(1);
    TORCH_CHECK(M % RB_BM == 0 && N % RB_BN == 0 && K % RB_BK == 0, "regblock needs M,N % 128 == 0 and K % 8 == 0");
    c10::cuda::CUDAGuard guard(a.device());
    auto c = torch::empty({M, N}, a.options());
    matmul_regblock<<<dim3(N / RB_BN, M / RB_BM), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled16", &tiled16, "16x16 shared-memory tiled fp32 GEMM");
    m.def("regblock", &regblock, "128x128x8 register-blocked fp32 GEMM");
}
