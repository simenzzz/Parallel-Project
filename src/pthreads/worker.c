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
                        pthread_mutex_lock(&wa->z_mutexes[vz]);
                        wa->grid->data[vg_idx(vx, vy, vz, wa->res)] = 1;
                        pthread_mutex_unlock(&wa->z_mutexes[vz]);
                    }
                }
            }
        }
    }

    return NULL;
}
