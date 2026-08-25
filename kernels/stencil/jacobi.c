// A 5-point 2D Jacobi stencil - the core access pattern behind a lot of HPC
// codes (finite-difference solvers, simple fluid/heat simulations, and
// conceptually similar to force-summation loops in MD codes like LAMMPS/
// GROMACS). Used here purely as a vehicle to compare GCC vs Clang code
// generation at different optimization levels on identical source.
//
// Build with the Makefile in this directory, which builds every
// compiler x optimization-level combination and runs a timed comparison.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 2000       // grid is N x N
#define ITERS 200    // number of Jacobi sweeps

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void jacobi_step(float* restrict out, const float* restrict in, int n) {
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            out[i * n + j] = 0.25f * (
                in[(i - 1) * n + j] +
                in[(i + 1) * n + j] +
                in[i * n + (j - 1)] +
                in[i * n + (j + 1)]
            );
        }
    }
}

int main(void) {
    float* a = malloc(sizeof(float) * N * N);
    float* b = malloc(sizeof(float) * N * N);

    // Simple deterministic initial condition: a hot spot in the center.
    memset(a, 0, sizeof(float) * N * N);
    a[(N / 2) * N + (N / 2)] = 1000.0f;
    memcpy(b, a, sizeof(float) * N * N);

    double t0 = now_seconds();
    for (int it = 0; it < ITERS; it++) {
        jacobi_step(b, a, N);
        float* tmp = a; a = b; b = tmp;
    }
    double t1 = now_seconds();

    // Print a checksum so different builds can be diffed for correctness,
    // and the total time so -O0/-O2/-O3 and gcc/clang can be compared.
    double checksum = 0.0;
    for (int i = 0; i < N * N; i++) checksum += a[i];

    printf("grid=%dx%d iters=%d time=%.4fs checksum=%.6f\n", N, N, ITERS, t1 - t0, checksum);

    free(a); free(b);
    return 0;
}
