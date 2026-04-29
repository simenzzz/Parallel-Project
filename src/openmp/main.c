#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/config.h"
#include "../common/off_parser.h"
#include "../common/sat.h"
#include "../common/timer.h"
#include "../common/voxel_grid.h"
#include "../common/voxelize.h"

typedef struct {
    const char *input;
    const char *output;
    const char *csv;
    int res;
    int threads;
} Options;

static int parse_args(int argc, char **argv, Options *opt)
{
    opt->input = NULL;
    opt->output = NULL;
    opt->csv = NULL;
    opt->res = DEFAULT_GRID_RES;
    opt->threads = DEFAULT_THREADS;

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
        } else if (strcmp(argv[i], "-t") == 0) {
            opt->threads = atoi(argv[i + 1]);
        } else {
            return -1;
        }
    }

    return opt->input != NULL && opt->output != NULL && opt->csv != NULL &&
                   path_exists(opt->input) &&
                   opt->res >= MIN_GRID_RES && opt->res <= MAX_GRID_RES &&
                   opt->threads > 0 && opt->threads <= MAX_THREADS
               ? 0
               : -1;
}

int main(int argc, char **argv)
{
    Mesh mesh;
    VoxelGrid grid;
    Options opt;
    Timer timer;
    double elapsed_ms;
    char config[32];
    int res;

    if (parse_args(argc, argv, &opt) != 0) {
        fprintf(stderr, "Usage: %s -i <mesh.off> -r <res> -o <out.voxel> -t <threads> -c <results.csv>\n",
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

    res = grid.res;
    omp_set_num_threads(opt.threads);
    timer_start(&timer);
#pragma omp parallel for schedule(dynamic, 64) default(none) shared(mesh, grid, res)
    for (int f = 0; f < mesh.num_faces; ++f) {
        Triangle tri = triangle_from_face(&mesh, f);
        VoxelBounds bounds = triangle_voxel_bounds(&tri, res);
        float half = 0.5f / (float)res;

        for (int vx = bounds.min_x; vx <= bounds.max_x; ++vx) {
            for (int vy = bounds.min_y; vy <= bounds.max_y; ++vy) {
                for (int vz = bounds.min_z; vz <= bounds.max_z; ++vz) {
                    Vec3 center = {
                        ((float)vx + 0.5f) / (float)res,
                        ((float)vy + 0.5f) / (float)res,
                        ((float)vz + 0.5f) / (float)res
                    };
                    int idx = vg_idx(vx, vy, vz, res);
                    if (sat_triangle_aabb(&tri, center, half)) {
                        #pragma omp atomic write
                        grid.data[idx] = 1;
                    }
                }
            }
        }
    }
    elapsed_ms = timer_stop_ms(&timer);

    snprintf(config, sizeof(config), "omp_%dt", opt.threads);
    if (voxel_grid_save(&grid, opt.output) != 0 ||
        append_csv_row(opt.csv, "openmp", config, opt.input,
                       mesh.num_faces, opt.res, elapsed_ms) != 0) {
        voxel_grid_free(&grid);
        mesh_free(&mesh);
        return 1;
    }

    voxel_grid_free(&grid);
    mesh_free(&mesh);
    return 0;
}
