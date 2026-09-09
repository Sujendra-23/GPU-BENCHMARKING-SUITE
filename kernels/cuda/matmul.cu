// Tiled matrix multiplication: C = A * B, A is (M x K), B is (K x N).
// Same algorithm as kernels/opencl/matmul.cl (shared memory tiling), so the
// two can be benchmarked head-to-head against the same CPU reference
// (kernels/reference/matmul_ref.c - also the one compiled to WASM for
// dashboard/, so the browser check and this correctness check run the
// identical reference code).
//
// Requires the CUDA toolkit (nvcc) and an NVIDIA GPU - not available in the
// sandbox this repo was built in, see ../../README.md "Verification notes".
// Build: nvcc -O3 -I../reference -o matmul_cuda matmul.cu ../reference/matmul_ref.c

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "../reference/matmul_ref.h"

#define TILE_SIZE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

int main(int argc, char** argv) {
    int M = 512, N = 512, K = 512;
    if (argc == 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    printf("matmul_tiled (CUDA): M=%d N=%d K=%d\n", M, N, K);

    size_t sizeA = sizeof(float) * M * K;
    size_t sizeB = sizeof(float) * K * N;
    size_t sizeC = sizeof(float) * M * N;

    float *hA = (float*)malloc(sizeA), *hB = (float*)malloc(sizeB);
    float *hC = (float*)malloc(sizeC), *hC_ref = (float*)malloc(sizeC);

    srand(42);
    for (size_t i = 0; i < (size_t)M * K; i++) hA[i] = (float)(rand() % 100) / 10.0f;
    for (size_t i = 0; i < (size_t)K * N; i++) hB[i] = (float)(rand() % 100) / 10.0f;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeA));
    CUDA_CHECK(cudaMalloc(&dB, sizeB));
    CUDA_CHECK(cudaMalloc(&dC, sizeC));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaGetLastError());
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost));

    clock_t c0 = clock();
    matmul_cpu_reference(hA, hB, hC_ref, M, N, K);
    clock_t c1 = clock();

    double max_diff = 0.0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        double diff = fabs(hC[i] - hC_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("cuda kernel time:  %.4f ms\n", ms);
    printf("cpu reference time: %.4f ms\n", (double)(c1 - c0) * 1000.0 / CLOCKS_PER_SEC);
    printf("max abs diff vs ref: %.6f  %s\n", max_diff, max_diff < 1e-2 ? "(PASS)" : "(FAIL)");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hC_ref);
    return max_diff < 1e-2 ? 0 : 1;
}
