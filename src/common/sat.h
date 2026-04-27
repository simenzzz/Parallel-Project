#ifndef SAT_H
#define SAT_H

#include "off_parser.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    Vec3 v[3];
} Triangle;

int sat_triangle_aabb(const Triangle *tri, Vec3 center, float half);

#ifdef __cplusplus
}
#endif

#endif
