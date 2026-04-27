#ifndef VOXEL_GRID_H
#define VOXEL_GRID_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t *data;
    int res;
} VoxelGrid;

int voxel_grid_alloc(VoxelGrid *g, int res);
void voxel_grid_free(VoxelGrid *g);
void voxel_grid_clear(VoxelGrid *g);
int voxel_grid_save(const VoxelGrid *g, const char *path);
int voxel_grid_load(VoxelGrid *g, const char *path);

static inline int vg_idx(int x, int y, int z, int res)
{
    return x + y * res + z * res * res;
}

#ifdef __cplusplus
}
#endif

#endif
