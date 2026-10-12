#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int THREADS = 256;

__device__ float warp_sum(float x) {
    for (int offset = 16; offset; offset /= 2)
        x += __shfl_down_sync(0xffffffff, x, offset);
    return __shfl_sync(0xffffffff, x, 0);
}

template<bool MAXIMUM>
__device__ float block_reduce(float value, float* scratch) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int offset = 16; offset; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, value, offset);
        value = MAXIMUM ? fmaxf(value, other) : value + other;
    }
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < THREADS / 32 ? scratch[lane] : (MAXIMUM ? -INFINITY : 0.f);
        for (int offset = 16; offset; offset /= 2) {
            float other = __shfl_down_sync(0xffffffff, value, offset);
            value = MAXIMUM ? fmaxf(value, other) : value + other;
        }
        if (lane == 0) scratch[0] = value;
    }
    __syncthreads();
    return scratch[0];
}

__global__ void fused_softmax(const half* x, half* y, int n) {
    __shared__ float scratch[THREADS / 32];
    float values[32]; // n <= 8192: keep the entire row on chip.
    float maximum = -INFINITY;
    const int64_t base = static_cast<int64_t>(blockIdx.x) * n;
    for (int i = 0; i < 32; ++i) {
        int col = threadIdx.x + i * THREADS;
        values[i] = col < n ? __half2float(x[base + col]) : -INFINITY;
        maximum = fmaxf(maximum, values[i]);
    }
    maximum = block_reduce<true>(maximum, scratch);
    // All readers of scratch[0] must finish before the next reduction.
    __syncthreads();
    float denominator = 0.f;
    for (int i = 0; i < 32; ++i) {
        values[i] = expf(values[i] - maximum);
        denominator += values[i];
    }
    denominator = block_reduce<false>(denominator, scratch);
    for (int i = 0; i < 32; ++i) {
        int col = threadIdx.x + i * THREADS;
        if (col < n) y[base + col] = __float2half(values[i] / denominator);
    }
}

// Four warps share a K/V tile; each warp owns one query and its output.
// This is an educational SIMT implementation, not FlashAttention-2 or WMMA.
template<int D, bool CAUSAL>
__global__ void online_attention(const half* q, const half* k, const half* v,
                                 half* out, int n) {
    constexpr int TILE = 32;
    __shared__ half keys[TILE * D];
    __shared__ half vals[TILE * D];
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * 4 + threadIdx.x / 32;
    const int64_t base = static_cast<int64_t>(blockIdx.y) * n * D;
    float query[D / 32], acc[D / 32];
    #pragma unroll
    for (int c = 0; c < D / 32; ++c) {
        query[c] = row < n ? __half2float(q[base + static_cast<int64_t>(row) * D + lane + c * 32]) : 0.f;
        acc[c] = 0.f;
    }
    float maximum = -INFINITY, denominator = 0.f;
    const float scale = rsqrtf(static_cast<float>(D));
    const int end = CAUSAL ? min(n, static_cast<int>((blockIdx.x + 1) * 4)) : n;
    for (int start = 0; start < end; start += TILE) {
        for (int i = threadIdx.x; i < TILE * D; i += blockDim.x) {
            const int key_row = start + i / D;
            keys[i] = key_row < n ? k[base + static_cast<int64_t>(key_row) * D + i % D] : __float2half(0.f);
            vals[i] = key_row < n ? v[base + static_cast<int64_t>(key_row) * D + i % D] : __float2half(0.f);
        }
        __syncthreads();
        for (int j = 0; j < TILE && start + j < end; ++j) {
            if (CAUSAL && start + j > row) continue; // warp-uniform
            float score = 0.f;
            #pragma unroll
            for (int c = 0; c < D / 32; ++c)
                score += query[c] * __half2float(keys[j * D + lane + c * 32]);
            score = warp_sum(score) * scale;
            const float new_max = fmaxf(maximum, score);
            const float alpha = expf(maximum - new_max);
            const float p = expf(score - new_max);
            denominator = denominator * alpha + p;
            #pragma unroll
            for (int c = 0; c < D / 32; ++c)
                acc[c] = acc[c] * alpha + p * __half2float(vals[j * D + lane + c * 32]);
            maximum = new_max;
        }
        __syncthreads(); // readers finish before the tile is overwritten
    }
    if (row < n) {
        #pragma unroll
        for (int c = 0; c < D / 32; ++c)
            out[base + static_cast<int64_t>(row) * D + lane + c * 32] = __float2half(acc[c] / denominator);
    }
}

void check_input(const torch::Tensor& x, int rank) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
                "Expected CUDA fp16 input");
    TORCH_CHECK(x.dim() == rank && x.is_contiguous() && x.numel() > 0,
                "Expected nonempty contiguous input of rank ", rank);
    TORCH_CHECK(!x.requires_grad(), "Forward-only kernel does not support autograd");
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    check_input(x, 2);
    TORCH_CHECK(x.size(1) <= 8192, "Maximum softmax width is 8192");
    TORCH_CHECK(x.size(0) <= 2147483647, "Too many rows");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_softmax<<<x.size(0), THREADS, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), x.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

template<int D>
void launch_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                      torch::Tensor out, bool causal) {
    auto qp = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
    auto kp = reinterpret_cast<const half*>(k.data_ptr<at::Half>());
    auto vp = reinterpret_cast<const half*>(v.data_ptr<at::Half>());
    auto op = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    dim3 grid((q.size(2) + 3) / 4, q.size(0) * q.size(1));
    auto stream = at::cuda::getCurrentCUDAStream();
    if (causal) online_attention<D, true><<<grid, 128, 0, stream>>>(qp, kp, vp, op, q.size(2));
    else online_attention<D, false><<<grid, 128, 0, stream>>>(qp, kp, vp, op, q.size(2));
}

torch::Tensor attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
    for (const auto& x : {q, k, v}) {
        check_input(x, 4);
        TORCH_CHECK(x.sizes() == q.sizes() && x.device() == q.device(),
                    "Q/K/V shapes and devices must match");
    }
    TORCH_CHECK(q.size(2) <= 2147483615 && q.size(0) * q.size(1) <= 65535,
                "Shape exceeds CUDA grid limits");
    c10::cuda::CUDAGuard guard(q.device());
    auto out = torch::empty_like(q);
    switch (q.size(3)) {
        case 32: launch_attention<32>(q, k, v, out, causal); break;
        case 64: launch_attention<64>(q, k, v, out, causal); break;
        case 128: launch_attention<128>(q, k, v, out, causal); break;
        default: TORCH_CHECK(false, "Supported head dimensions: 32, 64, 128");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
