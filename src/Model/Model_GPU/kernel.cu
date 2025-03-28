#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

#define BLOCK_SIZE 256

__global__ void compute_acc_parallel_inner(float3* positionsGPU,
                                             float3* accelerationsGPU,
                                             float* massesGPU,
                                             int n_particles)
{
    int i = blockIdx.x;
    if (i >= n_particles) return;


    int tid = threadIdx.x;
    float3 pos_i = positionsGPU[i];

    float3 acc_local = make_float3(0.0f, 0.0f, 0.0f);


    for (int j = tid; j < n_particles; j += blockDim.x) 
    {

        float diffx = positionsGPU[j].x - pos_i.x;
        float diffy = positionsGPU[j].y - pos_i.y;
        float diffz = positionsGPU[j].z - pos_i.z;

        float dij = diffx * diffx + diffy * diffy + diffz * diffz;
        dij = fmaxf(dij, 1.0f); // Clamp to avoid small distances
        float inv_d = rsqrtf(dij); // Fast reciprocal square root
        float factor = 10.0f * inv_d * inv_d * inv_d;
        
        acc_local.x += diffx * factor * massesGPU[j];
        acc_local.y += diffy * factor * massesGPU[j];
        acc_local.z += diffz * factor * massesGPU[j];
        
    }


    __shared__ float3 shared_acc[BLOCK_SIZE];
    shared_acc[tid] = acc_local;
    __syncthreads();


    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_acc[tid].x += shared_acc[tid + stride].x;
            shared_acc[tid].y += shared_acc[tid + stride].y;
            shared_acc[tid].z += shared_acc[tid + stride].z;
        }
        __syncthreads();
    }


    if (tid == 0) {
        accelerationsGPU[i] = shared_acc[0];
    }
}



__global__ void maj_pos(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, int n_particles)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;

    velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
    velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
    velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
    positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
    positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
    positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;
}

__global__ void zero_accelerations(float3* accelerationsGPU, int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles)
    {
        accelerationsGPU[i].x = 0.0f;
        accelerationsGPU[i].y = 0.0f;
        accelerationsGPU[i].z = 0.0f;
    }
}

int divup(int n, int dim)
{
    return (n + (dim - 1)) / dim;
}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{

    int nthreads = 128*2;
    int nblocks = (n_particles + (nthreads - 1)) / nthreads;
    compute_acc_parallel_inner<<< n_particles, BLOCK_SIZE >>>(positionsGPU, accelerationsGPU, massesGPU, n_particles);

    maj_pos<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}

#endif // GALAX_MODEL_GPU
