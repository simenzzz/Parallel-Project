#include "off_parser.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void mesh_reset(Mesh *m)
{
    m->vertices = NULL;
    m->faces = NULL;
    m->num_vertices = 0;
    m->num_faces = 0;
    m->bbox_min = (Vec3){0.0f, 0.0f, 0.0f};
    m->bbox_max = (Vec3){0.0f, 0.0f, 0.0f};
}

static int next_data_line(FILE *fp, char *buf, size_t size)
{
    while (fgets(buf, (int)size, fp) != NULL) {
        char *p = buf;
        while (isspace((unsigned char)*p)) {
            ++p;
        }
        if (*p == '\0' || *p == '\n' || *p == '#') {
            continue;
        }
        return 0;
    }
    return -1;
}

int mesh_load(const char *path, Mesh *out)
{
    FILE *fp;
    char line[1024];
    int num_edges;

    if (path == NULL || out == NULL) {
        fprintf(stderr, "mesh_load: invalid argument\n");
        return -1;
    }

    mesh_reset(out);
    fp = fopen(path, "r");
    if (fp == NULL) {
        perror(path);
        return -1;
    }

    if (next_data_line(fp, line, sizeof(line)) != 0) {
        fprintf(stderr, "Failed to read OFF header from %s\n", path);
        fclose(fp);
        return -1;
    }
    if (strncmp(line, "OFF", 3) != 0) {
        fprintf(stderr, "Invalid OFF magic in %s\n", path);
        fclose(fp);
        return -1;
    }

    /* handle both "OFF\nnV nF nE" and compact "OFFnV nF nE" on one line */
    const char *count_src = (line[3] != '\0' && line[3] != '\n' && line[3] != '\r')
        ? line + 3
        : NULL;
    if (count_src == NULL) {
        if (next_data_line(fp, line, sizeof(line)) != 0) {
            fprintf(stderr, "Invalid OFF counts in %s\n", path);
            fclose(fp);
            mesh_reset(out);
            return -1;
        }
        count_src = line;
    }
    if (sscanf(count_src, "%d %d %d", &out->num_vertices, &out->num_faces, &num_edges) != 3 ||
        out->num_vertices <= 0 || out->num_faces <= 0) {
        fprintf(stderr, "Invalid OFF counts in %s\n", path);
        fclose(fp);
        mesh_reset(out);
        return -1;
    }
    (void)num_edges;

    out->vertices = (Vec3 *)malloc((size_t)out->num_vertices * sizeof(Vec3));
    out->faces = (Face *)malloc((size_t)out->num_faces * sizeof(Face));
    if (out->vertices == NULL || out->faces == NULL) {
        fprintf(stderr, "Out of memory while loading %s\n", path);
        fclose(fp);
        mesh_free(out);
        return -1;
    }

    for (int i = 0; i < out->num_vertices; ++i) {
        if (next_data_line(fp, line, sizeof(line)) != 0 ||
            sscanf(line, "%f %f %f",
                   &out->vertices[i].x, &out->vertices[i].y, &out->vertices[i].z) != 3) {
            fprintf(stderr, "Invalid vertex line %d in %s\n", i, path);
            fclose(fp);
            mesh_free(out);
            return -1;
        }
        if (i == 0) {
            out->bbox_min = out->vertices[i];
            out->bbox_max = out->vertices[i];
        } else {
            Vec3 v = out->vertices[i];
            if (v.x < out->bbox_min.x) {
                out->bbox_min.x = v.x;
            }
            if (v.y < out->bbox_min.y) {
                out->bbox_min.y = v.y;
            }
            if (v.z < out->bbox_min.z) {
                out->bbox_min.z = v.z;
            }
            if (v.x > out->bbox_max.x) {
                out->bbox_max.x = v.x;
            }
            if (v.y > out->bbox_max.y) {
                out->bbox_max.y = v.y;
            }
            if (v.z > out->bbox_max.z) {
                out->bbox_max.z = v.z;
            }
        }
    }

    for (int i = 0; i < out->num_faces; ++i) {
        int n;
        if (next_data_line(fp, line, sizeof(line)) != 0 ||
            sscanf(line, "%d %d %d %d", &n,
                   &out->faces[i].v[0], &out->faces[i].v[1], &out->faces[i].v[2]) != 4 ||
            n != 3) {
            fprintf(stderr, "Invalid face line %d in %s\n", i, path);
            fclose(fp);
            mesh_free(out);
            return -1;
        }
        for (int k = 0; k < 3; ++k) {
            if (out->faces[i].v[k] < 0 || out->faces[i].v[k] >= out->num_vertices) {
                fprintf(stderr, "Face index out of range at face %d in %s\n", i, path);
                fclose(fp);
                mesh_free(out);
                return -1;
            }
        }
    }

    fclose(fp);
    return 0;
}

void mesh_free(Mesh *m)
{
    if (m == NULL) {
        return;
    }
    free(m->vertices);
    free(m->faces);
    mesh_reset(m);
}

void mesh_normalize(Mesh *m)
{
    float dx;
    float dy;
    float dz;
    float scale;

    if (m == NULL || m->vertices == NULL || m->num_vertices <= 0) {
        return;
    }

    dx = m->bbox_max.x - m->bbox_min.x;
    dy = m->bbox_max.y - m->bbox_min.y;
    dz = m->bbox_max.z - m->bbox_min.z;
    scale = dx;
    if (dy > scale) {
        scale = dy;
    }
    if (dz > scale) {
        scale = dz;
    }
    if (scale <= 0.0f) {
        scale = 1.0f;
    }

    for (int i = 0; i < m->num_vertices; ++i) {
        m->vertices[i].x = (m->vertices[i].x - m->bbox_min.x) / scale;
        m->vertices[i].y = (m->vertices[i].y - m->bbox_min.y) / scale;
        m->vertices[i].z = (m->vertices[i].z - m->bbox_min.z) / scale;
    }

    m->bbox_min = (Vec3){0.0f, 0.0f, 0.0f};
    m->bbox_max = (Vec3){dx / scale, dy / scale, dz / scale};
}
