#include "voxelize.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "config.h"

static int clampi(int v, int lo, int hi)
{
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

int path_exists(const char *path)
{
    struct stat st;
    return path != NULL && stat(path, &st) == 0;
}

int ensure_directory(const char *path)
{
    if (path == NULL) {
        return -1;
    }
    if (mkdir(path, 0777) == 0 || errno == EEXIST) {
        return 0;
    }
    perror(path);
    return -1;
}

int append_csv_row(const char *csv_path, const char *impl, const char *config,
                   const char *mesh_file, int num_triangles, int grid_res,
                   double elapsed_ms)
{
    FILE *fp;
    int need_header;

    if (csv_path == NULL || impl == NULL || config == NULL || mesh_file == NULL) {
        return -1;
    }

    if (ensure_directory("results") != 0) {
        return -1;
    }

    need_header = !path_exists(csv_path);
    fp = fopen(csv_path, "a");
    if (fp == NULL) {
        perror(csv_path);
        return -1;
    }
    if (need_header && fputs(CSV_COL_HEADER, fp) == EOF) {
        fprintf(stderr, "Failed to write CSV header to %s\n", csv_path);
        fclose(fp);
        return -1;
    }
    if (fprintf(fp, "%s,%s,%s,%d,%d,%.6f\n",
                impl, config, mesh_file, num_triangles, grid_res, elapsed_ms) < 0) {
        fprintf(stderr, "Failed to append CSV row to %s\n", csv_path);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

Triangle triangle_from_face(const Mesh *mesh, int face_idx)
{
    Face f = mesh->faces[face_idx];
    Triangle tri;

    tri.v[0] = mesh->vertices[f.v[0]];
    tri.v[1] = mesh->vertices[f.v[1]];
    tri.v[2] = mesh->vertices[f.v[2]];
    return tri;
}

VoxelBounds triangle_voxel_bounds(const Triangle *tri, int res)
{
    float min_x = tri->v[0].x;
    float min_y = tri->v[0].y;
    float min_z = tri->v[0].z;
    float max_x = tri->v[0].x;
    float max_y = tri->v[0].y;
    float max_z = tri->v[0].z;

    for (int i = 1; i < 3; ++i) {
        if (tri->v[i].x < min_x) {
            min_x = tri->v[i].x;
        }
        if (tri->v[i].y < min_y) {
            min_y = tri->v[i].y;
        }
        if (tri->v[i].z < min_z) {
            min_z = tri->v[i].z;
        }
        if (tri->v[i].x > max_x) {
            max_x = tri->v[i].x;
        }
        if (tri->v[i].y > max_y) {
            max_y = tri->v[i].y;
        }
        if (tri->v[i].z > max_z) {
            max_z = tri->v[i].z;
        }
    }

    return (VoxelBounds){
        clampi((int)floorf(min_x * res), 0, res - 1),
        clampi((int)floorf(min_y * res), 0, res - 1),
        clampi((int)floorf(min_z * res), 0, res - 1),
        clampi((int)floorf(max_x * res), 0, res - 1),
        clampi((int)floorf(max_y * res), 0, res - 1),
        clampi((int)floorf(max_z * res), 0, res - 1)
    };
}
