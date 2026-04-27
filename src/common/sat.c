#include "sat.h"

#include <math.h>

static Vec3 vsub(Vec3 a, Vec3 b)
{
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static Vec3 vcross(Vec3 a, Vec3 b)
{
    return (Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static float vdot(Vec3 a, Vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static int axis_overlap(const Vec3 *verts, Vec3 axis, float half)
{
    float len2;
    float min_p;
    float max_p;
    float p1;
    float p2;
    float r;

    len2 = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;
    if (len2 <= 1e-12f) {
        return 1;
    }

    min_p = vdot(verts[0], axis);
    max_p = min_p;
    p1 = vdot(verts[1], axis);
    p2 = vdot(verts[2], axis);

    if (p1 < min_p) {
        min_p = p1;
    }
    if (p1 > max_p) {
        max_p = p1;
    }
    if (p2 < min_p) {
        min_p = p2;
    }
    if (p2 > max_p) {
        max_p = p2;
    }

    r = half * (fabsf(axis.x) + fabsf(axis.y) + fabsf(axis.z));
    return !(min_p > r || max_p < -r);
}

static int axis_aligned_overlap(const Vec3 *verts, float half, int axis)
{
    float min_v;
    float max_v;
    float a;
    float b;

    if (axis == 0) {
        min_v = verts[0].x;
        max_v = verts[0].x;
        a = verts[1].x;
        b = verts[2].x;
    } else if (axis == 1) {
        min_v = verts[0].y;
        max_v = verts[0].y;
        a = verts[1].y;
        b = verts[2].y;
    } else {
        min_v = verts[0].z;
        max_v = verts[0].z;
        a = verts[1].z;
        b = verts[2].z;
    }

    if (a < min_v) {
        min_v = a;
    }
    if (a > max_v) {
        max_v = a;
    }
    if (b < min_v) {
        min_v = b;
    }
    if (b > max_v) {
        max_v = b;
    }

    return !(min_v > half || max_v < -half);
}

int sat_triangle_aabb(const Triangle *tri, Vec3 center, float half)
{
    Vec3 v[3];
    Vec3 e[3];
    Vec3 axes[3] = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };
    Vec3 normal;

    v[0] = vsub(tri->v[0], center);
    v[1] = vsub(tri->v[1], center);
    v[2] = vsub(tri->v[2], center);

    if (!axis_aligned_overlap(v, half, 0) ||
        !axis_aligned_overlap(v, half, 1) ||
        !axis_aligned_overlap(v, half, 2)) {
        return 0;
    }

    e[0] = vsub(v[1], v[0]);
    e[1] = vsub(v[2], v[1]);
    e[2] = vsub(v[0], v[2]);

    normal = vcross(e[0], e[1]);
    if (!axis_overlap(v, normal, half)) {
        return 0;
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Vec3 axis = vcross(e[i], axes[j]);
            if (!axis_overlap(v, axis, half)) {
                return 0;
            }
        }
    }

    return 1;
}
