#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

#define BLOCK_SIZE 256

__global__ void compute_acc_parallel_inner(float3* positionsGPU,
                                           float3* accelerationsGPU,
                                           float* massesGPU,
                                           int n_particles) {
    int i = blockIdx.x;
    if (i >= n_particles) return;

    int tid = threadIdx.x;
    float3 pos_i = positionsGPU[i];
    float3 acc_local = make_float3(0.0f, 0.0f, 0.0f);

    // Coalesced memory access pattern
    for (int j_start = 0; j_start < n_particles; j_start += blockDim.x) {
        int j = j_start + tid;
        if (j < n_particles) {
            float3 pos_j = positionsGPU[j];
            float diffx = pos_j.x - pos_i.x;
            float diffy = pos_j.y - pos_i.y;
            float diffz = pos_j.z - pos_i.z;

            float dij_sq = diffx * diffx + diffy * diffy + diffz * diffz;
            dij_sq = fmaxf(dij_sq, 1.0f); // Clamp to avoid small distances
            float inv_d = rsqrtf(dij_sq); // Fast reciprocal square root
            float factor = 10.0f * inv_d * inv_d * inv_d;

            acc_local.x += diffx * factor * massesGPU[j];
            acc_local.y += diffy * factor * massesGPU[j];
            acc_local.z += diffz * factor * massesGPU[j];
        }
    }

    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_local.x += __shfl_down_sync(0xffffffff, acc_local.x, offset);
        acc_local.y += __shfl_down_sync(0xffffffff, acc_local.y, offset);
        acc_local.z += __shfl_down_sync(0xffffffff, acc_local.z, offset);
    }

    // First thread in warp stores partial sum
    __shared__ float3 shared_acc[BLOCK_SIZE / 32];
    if (tid % 32 == 0) {
        shared_acc[tid / 32] = acc_local;
    }
    __syncthreads();

    // Sum partial warp sums
    if (tid < (BLOCK_SIZE / 32)) {
        float3 sum = shared_acc[tid];
        for (int offset = (BLOCK_SIZE / 64); offset > 0; offset >>= 1) {
            sum.x += __shfl_down_sync(0xffffffff, sum.x, offset);
            sum.y += __shfl_down_sync(0xffffffff, sum.y, offset);
            sum.z += __shfl_down_sync(0xffffffff, sum.z, offset);
        }
        if (tid == 0) {
            accelerationsGPU[i] = sum;
        }
    }
}

__global__ void maj_pos(float3* positionsGPU, float3* velocitiesGPU, 
                        float3* accelerationsGPU, int n_particles) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
    velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
    velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
    
    positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
    positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
    positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;
}

__global__ void zero_accelerations(float3* accelerationsGPU, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        accelerationsGPU[i].x = 0.0f;
        accelerationsGPU[i].y = 0.0f;
        accelerationsGPU[i].z = 0.0f;
    }
}

int divup(int n, int dim) {
    return (n + dim - 1) / dim;
}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, 
                        float3* accelerationsGPU, float* massesGPU, int n_particles) {
    int nthreads = 256;
    int nblocks = divup(n_particles, nthreads);

    // Ensure accelerations are reset before computation
    zero_accelerations<<<nblocks, nthreads>>>(accelerationsGPU, n_particles);
    cudaDeviceSynchronize(); // Ensure zeroing completes before next kernel

    compute_acc_parallel_inner<<<n_particles, BLOCK_SIZE>>>(positionsGPU, accelerationsGPU, 
                                                           massesGPU, n_particles);
    maj_pos<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}

#endif // GALAX_MODEL_GPU