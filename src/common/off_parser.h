#ifndef OFF_PARSER_H
#define OFF_PARSER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x;
    float y;
    float z;
} Vec3;

typedef struct {
    int v[3];
} Face;

typedef struct {
    Vec3 *vertices;
    Face *faces;
    int num_vertices;
    int num_faces;
    Vec3 bbox_min;
    Vec3 bbox_max;
} Mesh;

int mesh_load(const char *path, Mesh *out);
void mesh_free(Mesh *m);
void mesh_normalize(Mesh *m);

#ifdef __cplusplus
}
#endif

#endif
