#ifndef VOXELIZE_H
#define VOXELIZE_H

#include "off_parser.h"
#include "sat.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int min_x;
    int min_y;
    int min_z;
    int max_x;
    int max_y;
    int max_z;
} VoxelBounds;

int path_exists(const char *path);
int ensure_directory(const char *path);
int append_csv_row(const char *csv_path, const char *impl, const char *config,
                   const char *mesh_file, int num_triangles, int grid_res,
                   double elapsed_ms);
Triangle triangle_from_face(const Mesh *mesh, int face_idx);
VoxelBounds triangle_voxel_bounds(const Triangle *tri, int res);

#ifdef __cplusplus
}
#endif

#endif
