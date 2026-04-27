#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/config.h"
#include "../common/off_parser.h"
#include "../common/sat.h"
#include "../common/timer.h"
#include "../common/voxel_grid.h"
#include "../common/voxelize.h"

#define MPI_CHECK(call)                                                         \
    do {                                                                        \
        int mpi_status__ = (call);                                              \
        if (mpi_status__ != MPI_SUCCESS) {                                      \
            fprintf(stderr, "MPI call failed: %s\n", #call);                    \
            MPI_Abort(MPI_COMM_WORLD, mpi_status__);                            \
        }                                                                       \
    } while (0)

_Static_assert(sizeof(Vec3) == 3 * sizeof(float), "Vec3 must be tightly packed for MPI_Bcast");
_Static_assert(sizeof(Face) == 3 * sizeof(int),   "Face must be tightly packed for MPI_Bcast");

typedef struct {
    const char *input;
    const char *output;
    const char *csv;
    int res;
} Options;

static int parse_args(int argc, char **argv, Options *opt)
{
    opt->input = NULL;
    opt->output = NULL;
    opt->csv = NULL;
    opt->res = DEFAULT_GRID_RES;

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
        } else {
            return -1;
        }
    }

    return opt->input != NULL && opt->output != NULL && opt->csv != NULL &&
                   opt->res >= MIN_GRID_RES && opt->res <= MAX_GRID_RES
               ? 0
               : -1;
}

static void slab_for_rank(int res, int size, int rank, int *z_start, int *depth)
{
    int base = res / size;
    int rem = res % size;

    *depth = base + (rank < rem ? 1 : 0);
    *z_start = rank * base + (rank < rem ? rank : rem);
}

int main(int argc, char **argv)
{
    Options opt;
    Mesh mesh = {0};
    VoxelGrid grid = {0};
    Timer timer;
    double elapsed_ms = 0.0;
    int rank;
    int size;
    int counts[3];
    float bbox[6];
    int z_start;
    int slab_depth;
    int z_end;
    uint8_t *local_slab = NULL;
    char config[32];

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    if (parse_args(argc, argv, &opt) != 0) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s -i <mesh.off> -r <res> -o <out.voxel> -c <results.csv>\n",
                    argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    {
        int load_ok = 0;
        if (rank == 0) {
            load_ok = (ensure_directory("results") == 0 && mesh_load(opt.input, &mesh) == 0);
            if (load_ok) {
                mesh_normalize(&mesh);
            }
        }
        MPI_CHECK(MPI_Bcast(&load_ok, 1, MPI_INT, 0, MPI_COMM_WORLD));
        if (!load_ok) {
            mesh_free(&mesh);
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }

    if (rank == 0) {
        counts[0] = mesh.num_vertices;
        counts[1] = mesh.num_faces;
        counts[2] = opt.res;
        bbox[0] = mesh.bbox_min.x;
        bbox[1] = mesh.bbox_min.y;
        bbox[2] = mesh.bbox_min.z;
        bbox[3] = mesh.bbox_max.x;
        bbox[4] = mesh.bbox_max.y;
        bbox[5] = mesh.bbox_max.z;
    }

    MPI_CHECK(MPI_Bcast(counts, 3, MPI_INT, 0, MPI_COMM_WORLD));
    opt.res = counts[2];
    if (rank != 0) {
        mesh.num_vertices = counts[0];
        mesh.num_faces = counts[1];
        mesh.vertices = (Vec3 *)malloc((size_t)mesh.num_vertices * sizeof(Vec3));
        mesh.faces = (Face *)malloc((size_t)mesh.num_faces * sizeof(Face));
        if (mesh.vertices == NULL || mesh.faces == NULL) {
            fprintf(stderr, "Rank %d failed to allocate mesh buffers\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_CHECK(MPI_Bcast((float *)mesh.vertices, mesh.num_vertices * 3, MPI_FLOAT, 0,
                        MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast((int *)mesh.faces, mesh.num_faces * 3, MPI_INT, 0,
                        MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(bbox, 6, MPI_FLOAT, 0, MPI_COMM_WORLD));
    mesh.bbox_min = (Vec3){bbox[0], bbox[1], bbox[2]};
    mesh.bbox_max = (Vec3){bbox[3], bbox[4], bbox[5]};

    slab_for_rank(opt.res, size, rank, &z_start, &slab_depth);
    z_end = z_start + slab_depth;
    local_slab = (uint8_t *)calloc((size_t)opt.res * (size_t)opt.res * (size_t)slab_depth,
                                   sizeof(uint8_t));
    if (local_slab == NULL) {
        fprintf(stderr, "Rank %d failed to allocate slab\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) {
        timer_start(&timer);
    }

    for (int f = 0; f < mesh.num_faces; ++f) {
        Triangle tri = triangle_from_face(&mesh, f);
        VoxelBounds bounds = triangle_voxel_bounds(&tri, opt.res);
        float half = 0.5f / (float)opt.res;
        int min_z = bounds.min_z < z_start ? z_start : bounds.min_z;
        int max_z = bounds.max_z >= z_end ? z_end - 1 : bounds.max_z;

        if (min_z > max_z) {
            continue;
        }
        for (int vx = bounds.min_x; vx <= bounds.max_x; ++vx) {
            for (int vy = bounds.min_y; vy <= bounds.max_y; ++vy) {
                for (int vz = min_z; vz <= max_z; ++vz) {
                    Vec3 center = {
                        ((float)vx + 0.5f) / (float)opt.res,
                        ((float)vy + 0.5f) / (float)opt.res,
                        ((float)vz + 0.5f) / (float)opt.res
                    };
                    if (sat_triangle_aabb(&tri, center, half)) {
                        int idx = vx + vy * opt.res + (vz - z_start) * opt.res * opt.res;
                        local_slab[idx] = 1;
                    }
                }
            }
        }
    }

    if (rank == 0 && voxel_grid_alloc(&grid, opt.res) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        int *recv_counts = (int *)malloc((size_t)size * sizeof(int));
        int *recv_displs = (int *)malloc((size_t)size * sizeof(int));
        int offset = 0;
        if (recv_counts == NULL || recv_displs == NULL) {
            fprintf(stderr, "Rank 0 failed to allocate gather metadata\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int r = 0; r < size; ++r) {
            int start_r;
            int depth_r;
            slab_for_rank(opt.res, size, r, &start_r, &depth_r);
            recv_counts[r] = depth_r * opt.res * opt.res;
            recv_displs[r] = offset;
            offset += recv_counts[r];
        }
        MPI_CHECK(MPI_Gatherv(local_slab, slab_depth * opt.res * opt.res, MPI_UNSIGNED_CHAR,
                              grid.data, recv_counts, recv_displs, MPI_UNSIGNED_CHAR,
                              0, MPI_COMM_WORLD));
        free(recv_counts);
        free(recv_displs);
    } else {
        MPI_CHECK(MPI_Gatherv(local_slab, slab_depth * opt.res * opt.res, MPI_UNSIGNED_CHAR,
                              NULL, NULL, NULL, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD));
    }

    if (rank == 0) {
        elapsed_ms = timer_stop_ms(&timer);
    }

    if (rank == 0) {
        snprintf(config, sizeof(config), "mpi_%dp", size);
        if (voxel_grid_save(&grid, opt.output) != 0 ||
            append_csv_row(opt.csv, "mpi", config, opt.input,
                           mesh.num_faces, opt.res, elapsed_ms) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    free(local_slab);
    voxel_grid_free(&grid);
    mesh_free(&mesh);
    MPI_CHECK(MPI_Finalize());
    return 0;
}
