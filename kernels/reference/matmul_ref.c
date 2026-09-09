// See matmul_ref.h. Kept deliberately naive (no tiling/blocking) since its
// job is to be obviously correct, not fast - it's the ground truth that the
// tiled CUDA and OpenCL kernels are checked against.
#include "matmul_ref.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

EMSCRIPTEN_KEEPALIVE
void matmul_cpu_reference(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}
