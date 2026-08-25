// Tiled matrix multiplication: C = A * B, where A is (M x K) and B is (K x N).
// Uses local memory (OpenCL's equivalent of CUDA shared memory) to cut
// global memory traffic by a factor of TILE_SIZE versus the naive version.
#define TILE_SIZE 16

__kernel void matmul_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    const int row = get_group_id(1) * TILE_SIZE + get_local_id(1);
    const int col = get_group_id(0) * TILE_SIZE + get_local_id(0);
    const int lrow = get_local_id(1);
    const int lcol = get_local_id(0);

    float acc = 0.0f;
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        const int aCol = t * TILE_SIZE + lcol;
        const int bRow = t * TILE_SIZE + lrow;

        tileA[lrow][lcol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[lrow][lcol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += tileA[lrow][k] * tileB[k][lcol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
