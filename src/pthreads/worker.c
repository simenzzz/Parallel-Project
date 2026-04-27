#include "worker.h"

#include <stddef.h>

#include "../common/sat.h"
#include "../common/voxelize.h"

void *worker_fn(void *arg)
{
    WorkerArgs *wa = (WorkerArgs *)arg;
    float half = 0.5f / (float)wa->res;

    for (int f = wa->start_face; f < wa->end_face; ++f) {
        Triangle tri = triangle_from_face(wa->mesh, f);
        VoxelBounds bounds = triangle_voxel_bounds(&tri, wa->res);

        for (int vx = bounds.min_x; vx <= bounds.max_x; ++vx) {
            for (int vy = bounds.min_y; vy <= bounds.max_y; ++vy) {
                for (int vz = bounds.min_z; vz <= bounds.max_z; ++vz) {
                    Vec3 center = {
                        ((float)vx + 0.5f) / (float)wa->res,
                        ((float)vy + 0.5f) / (float)wa->res,
                        ((float)vz + 0.5f) / (float)wa->res
                    };
                    if (sat_triangle_aabb(&tri, center, half)) {
                        /* Benign data race: all stores write constant 1; single-byte
                         * stores are atomic on x86/ARM. pthread_join provides the
                         * memory barrier before the caller reads the grid. */
                        wa->grid->data[vg_idx(vx, vy, vz, wa->res)] = 1;
                    }
                }
            }
        }
    }

    return NULL;
}
