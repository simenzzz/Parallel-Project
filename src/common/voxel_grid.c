#include "voxel_grid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int voxel_grid_alloc(VoxelGrid *g, int res)
{
    size_t bytes;

    if (g == NULL || res <= 0) {
        return -1;
    }
    bytes = (size_t)res * (size_t)res * (size_t)res;
    g->data = (uint8_t *)calloc(bytes, sizeof(uint8_t));
    if (g->data == NULL) {
        fprintf(stderr, "Failed to allocate voxel grid for res=%d\n", res);
        g->res = 0;
        return -1;
    }
    g->res = res;
    return 0;
}

void voxel_grid_free(VoxelGrid *g)
{
    if (g == NULL) {
        return;
    }
    free(g->data);
    g->data = NULL;
    g->res = 0;
}

void voxel_grid_clear(VoxelGrid *g)
{
    size_t bytes;

    if (g == NULL || g->data == NULL || g->res <= 0) {
        return;
    }
    bytes = (size_t)g->res * (size_t)g->res * (size_t)g->res;
    memset(g->data, 0, bytes);
}

int voxel_grid_save(const VoxelGrid *g, const char *path)
{
    FILE *fp;
    size_t bytes;

    if (g == NULL || g->data == NULL || path == NULL) {
        return -1;
    }
    fp = fopen(path, "wb");
    if (fp == NULL) {
        perror(path);
        return -1;
    }
    bytes = (size_t)g->res * (size_t)g->res * (size_t)g->res;
    if (fwrite(g->data, 1, bytes, fp) != bytes) {
        fprintf(stderr, "Failed to write voxel grid to %s\n", path);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

int voxel_grid_load(VoxelGrid *g, const char *path)
{
    FILE *fp;
    long file_size;
    size_t expected;

    if (g == NULL || g->data == NULL || path == NULL || g->res <= 0) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        perror(path);
        return -1;
    }
    if (fseek(fp, 0L, SEEK_END) != 0) {
        fprintf(stderr, "Failed to seek %s\n", path);
        fclose(fp);
        return -1;
    }
    file_size = ftell(fp);
    if (file_size < 0) {
        fprintf(stderr, "Failed to size %s\n", path);
        fclose(fp);
        return -1;
    }
    if (fseek(fp, 0L, SEEK_SET) != 0) {
        fprintf(stderr, "Failed to rewind %s\n", path);
        fclose(fp);
        return -1;
    }

    expected = (size_t)g->res * (size_t)g->res * (size_t)g->res;
    if ((size_t)file_size != expected) {
        fprintf(stderr, "Voxel grid size mismatch for %s: got %ld expected %zu\n",
                path, file_size, expected);
        fclose(fp);
        return -1;
    }
    if (fread(g->data, 1, expected, fp) != expected) {
        fprintf(stderr, "Failed to read voxel grid from %s\n", path);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}
