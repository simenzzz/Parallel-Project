#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "../common/voxelize.h"

static __device__ float3 fsub(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __device__ float3 fcross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

static __device__ float fdot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __device__ int axis_overlap(const float3 verts[3], float3 axis, float half)
{
    float len2 = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;
    float min_p;
    float max_p;
    float p1;
    float p2;
    float r;

    if (len2 <= 1e-12f) {
        return 1;
    }
    min_p = fdot(verts[0], axis);
    max_p = min_p;
    p1 = fdot(verts[1], axis);
    p2 = fdot(verts[2], axis);
    if (p1 < min_p) {
        min_p = p1;
    }
    if (p1 > max_p) {
        max_p = p1;
    }
    if (p2 < min_p) {
        min_p = p2;
    }
    if (p2 > max_p) {
        max_p = p2;
    }
    r = half * (fabsf(axis.x) + fabsf(axis.y) + fabsf(axis.z));
    return !(min_p > r || max_p < -r);
}

static __device__ int axis_aligned_overlap(const float3 verts[3], float half, int axis)
{
    float min_v;
    float max_v;
    float a;
    float b;

    if (axis == 0) {
        min_v = verts[0].x;
        max_v = verts[0].x;
        a = verts[1].x;
        b = verts[2].x;
    } else if (axis == 1) {
        min_v = verts[0].y;
        max_v = verts[0].y;
        a = verts[1].y;
        b = verts[2].y;
    } else {
        min_v = verts[0].z;
        max_v = verts[0].z;
        a = verts[1].z;
        b = verts[2].z;
    }

    if (a < min_v) {
        min_v = a;
    }
    if (a > max_v) {
        max_v = a;
    }
    if (b < min_v) {
        min_v = b;
    }
    if (b > max_v) {
        max_v = b;
    }

    return !(min_v > half || max_v < -half);
}

static __device__ int sat_device(float3 v0, float3 v1, float3 v2, float3 center, float half)
{
    float3 v[3];
    float3 e[3];
    float3 basis[3] = {
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f)
    };
    float3 normal;

    v[0] = fsub(v0, center);
    v[1] = fsub(v1, center);
    v[2] = fsub(v2, center);

    if (!axis_aligned_overlap(v, half, 0) ||
        !axis_aligned_overlap(v, half, 1) ||
        !axis_aligned_overlap(v, half, 2)) {
        return 0;
    }

    e[0] = fsub(v[1], v[0]);
    e[1] = fsub(v[2], v[1]);
    e[2] = fsub(v[0], v[2]);
    normal = fcross(e[0], e[1]);
    if (!axis_overlap(v, normal, half)) {
        return 0;
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (!axis_overlap(v, fcross(e[i], basis[j]), half)) {
                return 0;
            }
        }
    }

    return 1;
}

#define TILE_DIM 8
#define CHUNK_SIZE 256

__global__ void voxelize_kernel(const float3 *__restrict__ verts,
                                const int3 *__restrict__ faces,
                                const VoxelBounds *__restrict__ face_bounds,
                                int n_faces,
                                unsigned int *__restrict__ grid,
                                int res)
{
    __shared__ float3 s_v0[CHUNK_SIZE];
    __shared__ float3 s_v1[CHUNK_SIZE];
    __shared__ float3 s_v2[CHUNK_SIZE];
    __shared__ VoxelBounds s_bounds[CHUNK_SIZE];
    __shared__ unsigned int s_grid[TILE_DIM * TILE_DIM * TILE_DIM];

    int tid = threadIdx.z * TILE_DIM * TILE_DIM + threadIdx.y * TILE_DIM + threadIdx.x;
    s_grid[tid] = 0;

    int block_vx = blockIdx.x * TILE_DIM;
    int block_vy = blockIdx.y * TILE_DIM;
    int block_vz = blockIdx.z * TILE_DIM;
    
    int vx = block_vx + threadIdx.x;
    int vy = block_vy + threadIdx.y;
    int vz = block_vz + threadIdx.z;
    float half = 0.5f / (float)res;

    for (int i = 0; i < n_faces; i += CHUNK_SIZE) {
        int left = n_faces - i;
        int chunk = left < CHUNK_SIZE ? left : CHUNK_SIZE;
        
        if (tid < chunk) {
            int f_idx = i + tid;
            int3 f = faces[f_idx];
            s_v0[tid] = verts[f.x];
            s_v1[tid] = verts[f.y];
            s_v2[tid] = verts[f.z];
            s_bounds[tid] = face_bounds[f_idx];
        }
        __syncthreads();
        
        for (int t = 0; t < chunk; ++t) {
            VoxelBounds b = s_bounds[t];
            if (b.max_x >= block_vx && b.min_x < block_vx + TILE_DIM &&
                b.max_y >= block_vy && b.min_y < block_vy + TILE_DIM &&
                b.max_z >= block_vz && b.min_z < block_vz + TILE_DIM) 
            {
                if (vx >= b.min_x && vx <= b.max_x &&
                    vy >= b.min_y && vy <= b.max_y &&
                    vz >= b.min_z && vz <= b.max_z) 
                {
                    float3 center = make_float3(((float)vx + 0.5f) / (float)res,
                                                ((float)vy + 0.5f) / (float)res,
                                                ((float)vz + 0.5f) / (float)res);
                    if (sat_device(s_v0[t], s_v1[t], s_v2[t], center, half)) {
                        s_grid[tid] = 1;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (s_grid[tid] && vx < res && vy < res && vz < res) {
        int idx = vx + vy * res + vz * res * res;
        atomicOr(&grid[idx], 1u);
    }
}
