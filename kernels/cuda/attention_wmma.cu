// Tensor-core (WMMA) FlashAttention-style forward kernel, fp16 in / fp16 out,
// fp32 accumulation and fp32 online softmax. One 128-thread block handles 64
// query rows of one (batch, head); each warp owns 16 of those rows. K/V tiles
// are staged in shared memory and shared by the 4 warps. Simple on purpose:
// no cp.async pipelining, no register-resident O accumulator (O lives in
// per-warp shared memory so it can be rescaled row by row).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

constexpr int BM = 64;      // query rows per block (4 warps x 16)
constexpr int THREADS = 128;

template <int D, int BN>
struct Smem {
    static constexpr int LDH = D + 8;   // halves, padded
    static constexpr int LDO = D + 4;   // floats
    static constexpr int LDS = BN + 4;  // floats
    static constexpr int LDP = BN + 8;  // halves
    static constexpr size_t bytes =
        (size_t)(BM * LDH + 2 * BN * LDH + 4 * 16 * LDP) * sizeof(half) +
        (size_t)(4 * 16 * LDS + 4 * 16 * LDO) * sizeof(float);
};

template <int D, int BN, bool CAUSAL>
__global__ void __launch_bounds__(THREADS)
attn_wmma(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
          half* __restrict__ O, int N, float scale_log2) {
    using S = Smem<D, BN>;
    extern __shared__ __align__(128) unsigned char smem[];
    half* Qs = reinterpret_cast<half*>(smem);
    half* Ks = Qs + BM * S::LDH;
    half* Vs = Ks + BN * S::LDH;
    half* Ps = Vs + BN * S::LDH;
    float* Ss = reinterpret_cast<float*>(Ps + 4 * 16 * S::LDP);
    float* Os = Ss + 4 * 16 * S::LDS;

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const size_t base = static_cast<size_t>(blockIdx.y) * N * D;
    const int q0 = blockIdx.x * BM;

    for (int i = tid; i < BM * D / 8; i += THREADS) {
        int r = i / (D / 8), c = (i % (D / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (q0 + r < N) v = *reinterpret_cast<const uint4*>(Q + base + (size_t)(q0 + r) * D + c);
        *reinterpret_cast<uint4*>(Qs + r * S::LDH + c) = v;
    }
    half* Pw = Ps + warp * 16 * S::LDP;
    float* Sw = Ss + warp * 16 * S::LDS;
    float* Ow = Os + warp * 16 * S::LDO;
    for (int i = lane; i < 16 * S::LDO; i += 32) Ow[i] = 0.f;

    const int r = lane >> 1, h = lane & 1;       // row within warp tile, half of the columns
    const int row = q0 + warp * 16 + r;
    float m = -INFINITY, l = 0.f;

    const int kmax = CAUSAL ? min(N, q0 + BM) : N;
    for (int k0 = 0; k0 < kmax; k0 += BN) {
        __syncthreads();  // previous tile fully consumed
        for (int i = tid; i < BN * D / 8; i += THREADS) {
            int rr = i / (D / 8), c = (i % (D / 8)) * 8;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (k0 + rr < N) {
                size_t off = base + (size_t)(k0 + rr) * D + c;
                kv = *reinterpret_cast<const uint4*>(K + off);
                vv = *reinterpret_cast<const uint4*>(V + off);
            }
            *reinterpret_cast<uint4*>(Ks + rr * S::LDH + c) = kv;
            *reinterpret_cast<uint4*>(Vs + rr * S::LDH + c) = vv;
        }
        __syncthreads();

        // S = Q K^T for this warp's 16 rows
        for (int jn = 0; jn < BN / 16; ++jn) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.f);
            for (int kd = 0; kd < D / 16; ++kd) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                wmma::load_matrix_sync(a, Qs + warp * 16 * S::LDH + kd * 16, S::LDH);
                wmma::load_matrix_sync(b, Ks + jn * 16 * S::LDH + kd * 16, S::LDH);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(Sw + jn * 16, acc, S::LDS, wmma::mem_row_major);
        }
        __syncwarp();

        // online softmax: two lanes per row, each owns half of the BN columns
        float* srow = Sw + r * S::LDS + h * (BN / 2);
        float mx = -INFINITY;
        for (int j = 0; j < BN / 2; ++j) {
            int key = k0 + h * (BN / 2) + j;
            bool valid = key < N && (!CAUSAL || key <= row);
            float s = valid ? srow[j] * scale_log2 : -INFINITY;
            srow[j] = s;
            mx = fmaxf(mx, s);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        float m_new = fmaxf(m, mx);
        float alpha = exp2f(m - m_new);  // m_new is finite: key 0 is valid for every row
        l *= alpha;
        for (int j = 0; j < BN / 2; ++j) {
            float p = exp2f(srow[j] - m_new);
            l += p;
            Pw[r * S::LDP + h * (BN / 2) + j] = __float2half(p);
        }
        for (int c = 0; c < D / 2; ++c) Ow[r * S::LDO + h * (D / 2) + c] *= alpha;
        m = m_new;
        __syncwarp();

        // O += P V
        for (int jd = 0; jd < D / 16; ++jd) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> o;
            wmma::load_matrix_sync(o, Ow + jd * 16, S::LDO, wmma::mem_row_major);
            for (int kk = 0; kk < BN / 16; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v;
                wmma::load_matrix_sync(p, Pw + kk * 16, S::LDP);
                wmma::load_matrix_sync(v, Vs + kk * 16 * S::LDH + jd * 16, S::LDH);
                wmma::mma_sync(o, p, v, o);
            }
            wmma::store_matrix_sync(Ow + jd * 16, o, S::LDO, wmma::mem_row_major);
        }
        __syncwarp();
    }

    l += __shfl_xor_sync(0xffffffff, l, 1);
    const float inv = 1.f / l;
    if (row < N) {
        for (int c = 0; c < D / 2; ++c)
            O[base + (size_t)row * D + h * (D / 2) + c] = __float2half(Ow[r * S::LDO + h * (D / 2) + c] * inv);
    }
}

template <int D, int BN>
void launch(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            torch::Tensor& out, bool causal) {
    const int b = q.size(0), heads = q.size(1), n = q.size(2);
    dim3 grid((n + BM - 1) / BM, b * heads);
    constexpr size_t bytes = Smem<D, BN>::bytes;
    auto kernel = causal ? attn_wmma<D, BN, true> : attn_wmma<D, BN, false>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    kernel<<<grid, THREADS, bytes, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), n, 1.4426950408889634f / sqrtf((float)D));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor attention_wmma(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kHalf && q.dim() == 4 && q.is_contiguous(),
                "q must be a contiguous fp16 CUDA [B,H,N,D] tensor");
    TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes() && k.is_contiguous() && v.is_contiguous(),
                "k and v must match q");
    c10::cuda::CUDAGuard guard(q.device());
    auto out = torch::empty_like(q);
    switch (q.size(3)) {
        case 32: launch<32, 64>(q, k, v, out, causal); break;
        case 64: launch<64, 64>(q, k, v, out, causal); break;
        case 128: launch<128, 32>(q, k, v, out, causal); break;
        default: TORCH_CHECK(false, "head dim must be 32, 64 or 128");
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention", &attention_wmma, "Tensor-core WMMA flash-style attention forward");
}
