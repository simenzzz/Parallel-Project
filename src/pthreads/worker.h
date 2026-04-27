#ifndef PTHREADS_WORKER_H
#define PTHREADS_WORKER_H

#include "../common/off_parser.h"
#include "../common/voxel_grid.h"

typedef struct {
    const Mesh *mesh;
    VoxelGrid *grid;
    int res;
    int start_face;
    int end_face;
} WorkerArgs;

void *worker_fn(void *arg);

#endif
