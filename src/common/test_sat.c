#include "sat.h"

#include <stdio.h>

typedef struct {
    const char *name;
    Triangle tri;
    Vec3 center;
    float half;
    int expected;
} SatCase;

int main(void)
{
    SatCase cases[] = {
        {
            "inside",
            {{{0.45f, 0.45f, 0.50f}, {0.55f, 0.45f, 0.50f}, {0.50f, 0.55f, 0.50f}}},
            {0.50f, 0.50f, 0.50f}, 0.10f, 1
        },
        {
            "outside",
            {{{1.20f, 1.20f, 1.20f}, {1.30f, 1.20f, 1.20f}, {1.20f, 1.30f, 1.20f}}},
            {0.50f, 0.50f, 0.50f}, 0.10f, 0
        },
        {
            "edge_cross",
            {{{0.39f, 0.50f, 0.50f}, {0.61f, 0.50f, 0.50f}, {0.50f, 0.61f, 0.50f}}},
            {0.50f, 0.50f, 0.50f}, 0.10f, 1
        },
        {
            "axis_aligned",
            {{{0.40f, 0.40f, 0.50f}, {0.60f, 0.40f, 0.50f}, {0.40f, 0.60f, 0.50f}}},
            {0.50f, 0.50f, 0.50f}, 0.10f, 1
        },
        {
            "near_miss",
            {{{0.61f, 0.61f, 0.50f}, {0.70f, 0.61f, 0.50f}, {0.61f, 0.70f, 0.50f}}},
            {0.50f, 0.50f, 0.50f}, 0.10f, 0
        }
    };
    int failures = 0;

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        int got = sat_triangle_aabb(&cases[i].tri, cases[i].center, cases[i].half);
        if (got != cases[i].expected) {
            fprintf(stderr, "SAT test failed: %s expected=%d got=%d\n",
                    cases[i].name, cases[i].expected, got);
            ++failures;
        }
    }

    if (failures != 0) {
        return 1;
    }
    puts("SAT tests passed");
    return 0;
}
