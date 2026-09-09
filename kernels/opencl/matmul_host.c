// Host program for kernels/opencl/matmul.cl.
// Verifies correctness against a naive CPU reference and reports timing.
// Build: see Makefile in this directory.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../reference/matmul_ref.h"

static char* read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen"); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = '\0';
    fclose(f);
    if (out_len) *out_len = (size_t)len;
    return buf;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char** argv) {
    int M = 512, N = 512, K = 512;
    if (argc == 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    printf("matmul_tiled: M=%d N=%d K=%d\n", M, N, K);

    // --- host buffers ---
    float* A = malloc(sizeof(float) * M * K);
    float* B = malloc(sizeof(float) * K * N);
    float* C = malloc(sizeof(float) * M * N);
    float* C_ref = malloc(sizeof(float) * M * N);

    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 100) / 10.0f;

    // --- OpenCL boilerplate: platform, device, context, queue ---
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "clGetPlatformIDs failed: %d\n", err); return 1; }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "clGetDeviceIDs failed: %d\n", err); return 1; }

    char device_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("running on OpenCL device: %s\n", device_name);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, NULL, &err);

    // --- build the kernel from source ---
    size_t src_len;
    char* src = read_file("kernels/opencl/matmul.cl", &src_len);
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char**)&src, &src_len, &err);
    err = clBuildProgram(program, 1, &device, "-DTILE_SIZE=16", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "build failed:\n%s\n", log);
        return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "matmul_tiled", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateKernel failed: %d\n", err); return 1; }

    // --- device buffers ---
    cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * M * K, A, &err);
    cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * K * N, B, &err);
    cl_mem bufC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, NULL, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &M);
    clSetKernelArg(kernel, 4, sizeof(int), &N);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    size_t local[2] = {16, 16};
    size_t global[2] = {
        ((N + local[0] - 1) / local[0]) * local[0],
        ((M + local[1] - 1) / local[1]) * local[1]
    };

    double t0 = now_seconds();
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
    clFinish(queue);
    double t1 = now_seconds();
    if (err != CL_SUCCESS) { fprintf(stderr, "clEnqueueNDRangeKernel failed: %d\n", err); return 1; }

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * M * N, C, 0, NULL, NULL);

    // --- correctness check against CPU reference ---
    double t2 = now_seconds();
    matmul_cpu_reference(A, B, C_ref, M, N, K);
    double t3 = now_seconds();

    double max_diff = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = fabs(C[i] - C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("opencl kernel time:     %.4f ms\n", (t1 - t0) * 1000.0);
    printf("cpu reference time:     %.4f ms\n", (t3 - t2) * 1000.0);
    printf("max abs diff vs ref:    %.6f  %s\n", max_diff, max_diff < 1e-2 ? "(PASS)" : "(FAIL)");

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(A); free(B); free(C); free(C_ref); free(src);
    return max_diff < 1e-2 ? 0 : 1;
}
