#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/config.h"
#include "../common/off_parser.h"
#include "../common/voxel_grid.h"
#include "../common/voxelize.h"

__global__ void voxelize_kernel(const float3 *__restrict__ verts,
                                const int3 *__restrict__ faces,
                                const VoxelBounds *__restrict__ face_bounds,
                                int n_faces,
                                unsigned int *__restrict__ grid,
                                int res);

typedef struct {
    const char *input;
    const char *output;
    const char *csv;
    int res;
    int block_size;
} Options;

static int parse_args(int argc, char **argv, Options *opt)
{
    opt->input = NULL;
    opt->output = NULL;
    opt->csv = NULL;
    opt->res = DEFAULT_GRID_RES;
    opt->block_size = DEFAULT_BLOCK_SIZE;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            return -1;
        }
        if (strcmp(argv[i], "-i") == 0) {
            opt->input = argv[i + 1];
        } else if (strcmp(argv[i], "-o") == 0) {
            opt->output = argv[i + 1];
        } else if (strcmp(argv[i], "-c") == 0) {
            opt->csv = argv[i + 1];
        } else if (strcmp(argv[i], "-r") == 0) {
            opt->res = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-b") == 0) {
            opt->block_size = atoi(argv[i + 1]);
        } else {
            return -1;
        }
    }

    return opt->input != NULL && opt->output != NULL && opt->csv != NULL &&
                   path_exists(opt->input) &&
                   opt->res >= MIN_GRID_RES && opt->res <= MAX_GRID_RES &&
                   opt->block_size > 0
               ? 0
               : -1;
}

static int check_cuda(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    Mesh mesh;
    VoxelGrid grid;
    Options opt;
    float3 *verts_h = NULL;
    int3 *faces_h = NULL;
    VoxelBounds *bounds_h = NULL;
    float3 *verts_d = NULL;
    int3 *faces_d = NULL;
    VoxelBounds *bounds_d = NULL;
    unsigned int *grid_d = NULL;
    unsigned int *grid_h32 = NULL;
    cudaEvent_t start = NULL;
    cudaEvent_t stop = NULL;
    float elapsed_ms = 0.0f;
    char config[32];
    int total_voxels;
    int blocks;
    dim3 blockDim3(8, 8, 8);
    dim3 gridDim3(1, 1, 1);

    if (parse_args(argc, argv, &opt) != 0) {
        fprintf(stderr, "Usage: %s -i <mesh.off> -r <res> -o <out.voxel> -b <block_size> -c <results.csv>\n",
                argv[0]);
        return 1;
    }
    if (ensure_directory("results") != 0) {
        return 1;
    }
    if (mesh_load(opt.input, &mesh) != 0) {
        return 1;
    }
    mesh_normalize(&mesh);
    if (voxel_grid_alloc(&grid, opt.res) != 0) {
        mesh_free(&mesh);
        return 1;
    }

    verts_h = (float3 *)malloc((size_t)mesh.num_vertices * sizeof(float3));
    faces_h = (int3 *)malloc((size_t)mesh.num_faces * sizeof(int3));
    total_voxels = opt.res * opt.res * opt.res;
    bounds_h = (VoxelBounds *)malloc((size_t)mesh.num_faces * sizeof(VoxelBounds));
    grid_h32 = (unsigned int *)calloc((size_t)total_voxels, sizeof(unsigned int));
    if (verts_h == NULL || faces_h == NULL || bounds_h == NULL || grid_h32 == NULL) {
        fprintf(stderr, "Failed to allocate CUDA host buffers\n");
        goto fail;
    }

    for (int i = 0; i < mesh.num_vertices; ++i) {
        verts_h[i] = make_float3(mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z);
    }
    for (int i = 0; i < mesh.num_faces; ++i) {
        Triangle tri;

        faces_h[i] = make_int3(mesh.faces[i].v[0], mesh.faces[i].v[1], mesh.faces[i].v[2]);
        tri = triangle_from_face(&mesh, i);
        bounds_h[i] = triangle_voxel_bounds(&tri, opt.res);
    }

    if (check_cuda(cudaMalloc((void **)&verts_d, (size_t)mesh.num_vertices * sizeof(float3)), "cudaMalloc verts") != 0 ||
        check_cuda(cudaMalloc((void **)&faces_d, (size_t)mesh.num_faces * sizeof(int3)), "cudaMalloc faces") != 0 ||
        check_cuda(cudaMalloc((void **)&bounds_d, (size_t)mesh.num_faces * sizeof(VoxelBounds)), "cudaMalloc bounds") != 0 ||
        check_cuda(cudaMalloc((void **)&grid_d, (size_t)total_voxels * sizeof(unsigned int)), "cudaMalloc grid") != 0 ||
        check_cuda(cudaMemcpy(verts_d, verts_h, (size_t)mesh.num_vertices * sizeof(float3),
                              cudaMemcpyHostToDevice), "cudaMemcpy verts") != 0 ||
        check_cuda(cudaMemcpy(faces_d, faces_h, (size_t)mesh.num_faces * sizeof(int3),
                              cudaMemcpyHostToDevice), "cudaMemcpy faces") != 0 ||
        check_cuda(cudaMemcpy(bounds_d, bounds_h, (size_t)mesh.num_faces * sizeof(VoxelBounds),
                              cudaMemcpyHostToDevice), "cudaMemcpy bounds") != 0 ||
        check_cuda(cudaMemset(grid_d, 0, (size_t)total_voxels * sizeof(unsigned int)), "cudaMemset grid") != 0 ||
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start") != 0 ||
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop") != 0) {
        goto fail;
    }

    gridDim3 = dim3((opt.res + 7) / 8, (opt.res + 7) / 8, (opt.res + 7) / 8);
    if (check_cuda(cudaEventRecord(start), "cudaEventRecord start") != 0) {
        goto fail;
    }
    voxelize_kernel<<<gridDim3, blockDim3>>>(verts_d, faces_d, bounds_d, mesh.num_faces, grid_d, opt.res);
    if (check_cuda(cudaGetLastError(), "voxelize_kernel launch") != 0) {
        goto fail;
    }
    if (check_cuda(cudaEventRecord(stop), "cudaEventRecord stop") != 0 ||
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop") != 0 ||
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime") != 0) {
        goto fail;
    }

    if (check_cuda(cudaMemcpy(grid_h32, grid_d, (size_t)total_voxels * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost), "cudaMemcpy grid") != 0) {
        goto fail;
    }
    for (int i = 0; i < total_voxels; ++i) {
        grid.data[i] = grid_h32[i] != 0U ? 1U : 0U;
    }

    snprintf(config, sizeof(config), "cuda_%dtpb", opt.block_size);
    if (voxel_grid_save(&grid, opt.output) != 0 ||
        append_csv_row(opt.csv, "cuda", config, opt.input,
                       mesh.num_faces, opt.res, (double)elapsed_ms) != 0) {
        goto fail;
    }

    cudaFree(verts_d);
    cudaFree(faces_d);
    cudaFree(bounds_d);
    cudaFree(grid_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(verts_h);
    free(faces_h);
    free(bounds_h);
    free(grid_h32);
    voxel_grid_free(&grid);
    mesh_free(&mesh);
    return 0;

fail:
    cudaFree(verts_d);
    cudaFree(faces_d);
    cudaFree(bounds_d);
    cudaFree(grid_d);
    if (start != NULL) {
        cudaEventDestroy(start);
    }
    if (stop != NULL) {
        cudaEventDestroy(stop);
    }
    free(verts_h);
    free(faces_h);
    free(bounds_h);
    free(grid_h32);
    voxel_grid_free(&grid);
    mesh_free(&mesh);
    return 1;
}
