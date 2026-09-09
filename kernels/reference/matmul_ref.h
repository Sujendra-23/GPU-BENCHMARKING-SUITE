// Naive CPU reference matmul: C = A * B, A is (M x K), B is (K x N).
// This is the single source of truth used to check the CUDA kernel
// (kernels/cuda/matmul.cu) and the OpenCL kernel (kernels/opencl/matmul.cl)
// for correctness. It is also compiled to WebAssembly (see build_wasm.sh)
// so the same check can run live in a browser against recorded results —
// see dashboard/.
#ifndef MATMUL_REF_H
#define MATMUL_REF_H

void matmul_cpu_reference(const float* A, const float* B, float* C, int M, int N, int K);

#endif
