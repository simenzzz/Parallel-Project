#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/config.h"
#include "../common/off_parser.h"
#include "../common/timer.h"
#include "../common/voxel_grid.h"
#include "../common/voxelize.h"
#include "worker.h"

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
    pthread_t *threads = NULL;
    WorkerArgs *args = NULL;
    char config[32];
    double elapsed_ms;

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

    threads = (pthread_t *)malloc((size_t)opt.threads * sizeof(pthread_t));
    args = (WorkerArgs *)malloc((size_t)opt.threads * sizeof(WorkerArgs));
    if (threads == NULL || args == NULL) {
        fprintf(stderr, "Failed to allocate pthread resources\n");
        free(threads);
        free(args);
        voxel_grid_free(&grid);
        mesh_free(&mesh);
        return 1;
    }

    timer_start(&timer);
    for (int i = 0; i < opt.threads; ++i) {
        int start_face = (mesh.num_faces * i) / opt.threads;
        int end_face = (mesh.num_faces * (i + 1)) / opt.threads;
        args[i] = (WorkerArgs){&mesh, &grid, opt.res, start_face, end_face};
        if (pthread_create(&threads[i], NULL, worker_fn, &args[i]) != 0) {
            fprintf(stderr, "pthread_create failed for thread %d\n", i);
            for (int j = 0; j < i; ++j) {
                pthread_join(threads[j], NULL);
            }
            free(threads);
            free(args);
            voxel_grid_free(&grid);
            mesh_free(&mesh);
            return 1;
        }
    }
    for (int i = 0; i < opt.threads; ++i) {
        if (pthread_join(threads[i], NULL) != 0) {
            fprintf(stderr, "pthread_join failed for thread %d\n", i);
            free(threads);
            free(args);
            voxel_grid_free(&grid);
            mesh_free(&mesh);
            return 1;
        }
    }
    elapsed_ms = timer_stop_ms(&timer);

    snprintf(config, sizeof(config), "pth_%dt", opt.threads);
    if (voxel_grid_save(&grid, opt.output) != 0 ||
        append_csv_row(opt.csv, "pthreads", config, opt.input,
                       mesh.num_faces, opt.res, elapsed_ms) != 0) {
        free(threads);
        free(args);
        voxel_grid_free(&grid);
        mesh_free(&mesh);
        return 1;
    }

    free(threads);
    free(args);
    voxel_grid_free(&grid);
    mesh_free(&mesh);
    return 0;
}
