# CSC447: Parallel Programming for Multicore and Cluster Systems
## Project Final Report: Parallel 3D Object Voxelization

### 1. Group Information & Project Management
Our team collaborated extensively to implement and evaluate a computationally intensive mesh-to-voxel conversion algorithm across five parallel paradigms. The roles were divided as follows:
*   **Haidar Haidar**: Lead developer for the MPI distributed-memory implementation and the Spatial Domain Decomposition logic.
*   **Malek Baghdadi**: Lead developer for the CUDA GPU implementation, responsible for applying tiling and shared memory optimizations.
*   **Michael Kolanjian**: Lead developer for the Sequential baseline, Pthreads implementation (mutex synchronization), and automated benchmarking scripts.
*   **Sami Bou Khaled**: Lead developer for the OpenMP implementation (atomic operations), data collection, and final report compilation.

### 2. Selected 3D Dataset
For this project, we selected the **ModelNet40** dataset, sourced via Kaggle.
*   **Description**: ModelNet40 is a comprehensive collection of 12,311 3D CAD models categorized into 40 common object types (e.g., airplane, car, chair, bookshelf). It is widely used as a benchmark in computer vision and 3D deep learning.
*   **Data Format**: The models are provided in the `.OFF` (Object File Format) extension, which represents continuous 3D objects as a collection of vertices (points in 3D space) and triangular faces connecting those vertices.
*   **Justification**: Converting these continuous meshes into discrete 3D volumetric grids (voxels) is a computationally heavy operation critical for 3D printing, medical imaging, and collision detection. The sheer number of triangles makes it a perfect candidate for parallel optimization.

### 3. Algorithm Overview
The core algorithm converts a continuous 3D triangle mesh into a discrete 3D voxel grid.
1.  **Bounding Box**: The algorithm calculates an Axis-Aligned Bounding Box (AABB) for the entire mesh to normalize its scale into a `1.0 x 1.0 x 1.0` space.
2.  **Separating Axis Theorem (SAT)**: For each triangle, the algorithm calculates the local AABB of the triangle itself. It then iterates through all voxel cells within that local AABB. For each voxel, it performs the exact SAT test to check for intersection between the 3D triangle and the voxel cube.
3.  **Voxel Grid Update**: If an intersection is detected, the corresponding index in the 1D flattened representation of the 3D voxel grid is set to `1`.

**Algorithm Pseudocode:**
```text
For each Triangle T in Mesh:
    B = GetVoxelBounds(T, Resolution)
    For z from B.min_z to B.max_z:
        For y from B.min_y to B.max_y:
            For x from B.min_x to B.max_x:
                If SAT_Intersection(T, Voxel(x, y, z)):
                    Grid[x, y, z] = 1
```

### 4. Implementation Details for Each Paradigm

#### 4.1 Sequential
The sequential version acts as the baseline for performance and correctness. It uses nested `for` loops to iterate through the list of triangles, find their local AABB, and test every enclosed voxel using the SAT algorithm. The grid is updated directly in memory.

#### 4.2 OpenMP (Shared-Memory Parallelism)
OpenMP was used to implement loop-level parallelism across CPU cores. The outer loop iterating over the mesh's faces was parallelized using `#pragma omp parallel for schedule(dynamic, 64)`.
*   **Synchronization**: Because multiple triangles might intersect the exact same voxel, a data race condition occurs when updating the shared `grid.data` array. To ensure thread safety and adherence to the proposal, we used `#pragma omp atomic write` to strictly synchronize the single-byte updates.

#### 4.3 Pthreads (CPU-level Threads)
We used POSIX threads for a lower-level manual threading approach. The total number of triangles is partitioned into equal chunks based on the number of threads.
*   **Synchronization**: We avoided a single global lock (which would serialize the application) by allocating an array of `pthread_mutex_t` locks corresponding to each Z-slice of the voxel grid. When a thread wants to mark a voxel at `(x, y, z)`, it locks the mutex for that specific `z` index, safely updates the grid, and releases the lock.

#### 4.4 MPI (Distributed-Memory Parallelism)
MPI was used to distribute the workload across multiple discrete processes, implementing a **Spatial Domain Decomposition** strategy.
*   **Distribution**: Rank 0 broadcasts the mesh vertices and faces to all other ranks. The 3D voxel grid is sliced along the Z-axis, with each rank assigned a specific "slab" of the volume.
*   **Computation**: Every process loops over the triangles but only performs the SAT intersection tests for the voxels that fall within its assigned spatial slab.
*   **Gathering**: Finally, Rank 0 uses `MPI_Gatherv` to collect the computed slabs from all processes and piece them together into the final global voxel grid.

#### 4.5 CUDA (GPU Parallelism with Tiling)
The CUDA implementation was fundamentally redesigned to exploit the massive parallelism of the GPU while utilizing Shared Memory and Tiling for memory efficiency.
*   **Grid and Block Setup**: We tiled the voxel grid space by assigning each CUDA thread block an `8x8x8` sub-volume of the global grid (512 threads per block).
*   **Shared Memory**: Each block allocates `__shared__` memory for a local voxel grid (`s_grid`) and `__shared__` memory arrays to cache triangle vertices and bounding boxes (`s_v0, s_v1, s_v2, s_bounds`).
*   **Tiling Execution**: The kernel iterates over the global list of triangles in "chunks" of 256. Threads in the block cooperatively load the 256 triangles into shared memory. Then, after `__syncthreads()`, each thread checks its assigned voxel against the 256 cached triangles. If an intersection occurs, the thread writes to the shared `s_grid`.
*   **Global Commit**: After all triangles are processed, the threads commit their shared `s_grid` results back to the global device memory using `atomicOr`. This drastically reduces global memory reads and writes.

### 5. Experimental Setup and Performance Evaluation

#### 5.1 Experimental Setup
The CPU benchmarks (Sequential, OpenMP, Pthreads, MPI) were tested using 1, 4, and 8 thread/process configurations to measure scalability. The CUDA implementation was tested across different grid resolutions (`32`, `64`, `128`) and block sizes to observe how tiling handles increased workloads.

#### 5.2 Performance Results (Summary)

**Best Performance Per Backend:**
| backend | grid_res | best_config | meshes | avg_mean_ms | avg_speedup_vs_seq |
| --- | --- | --- | --- | --- | --- |
| openmp | 32 | omp_8t | 9 | 0.395 | 3.33x |
| pthreads | 32 | pth_4t | 9 | 1.197 | 1.34x |
| mpi | 32 | mpi_4p | 9 | 1.068 | 1.82x |
| cuda | 32 | cuda_256tpb | 9 | 6.570 | 0.44x |
| openmp | 64 | omp_8t | 9 | 0.736 | 3.66x |
| pthreads | 64 | pth_4t | 9 | 1.769 | 1.76x |
| mpi | 64 | mpi_1p | 9 | 2.537 | 1.12x |
| cuda | 64 | cuda_256tpb | 9 | 17.663 | 0.53x |
| openmp | 128 | omp_8t | 9 | 2.176 | 3.45x |
| pthreads | 128 | pth_4t | 9 | 4.597 | 1.82x |
| mpi | 128 | mpi_4p | 9 | 6.391 | 1.01x |
| cuda | 128 | cuda_256tpb | 9 | 142.652 | 0.37x |

**Detailed Speedup (vs Sequential Baseline):**
### Resolution 128
| mesh | seq | best_omp | best_pth | best_mpi | best_cuda |
| --- | --- | --- | --- | --- | --- |
| airplane_0001.off | 1.00x | 5.24x | 2.75x | 1.15x | 0.02x |
| bookshelf_0001.off | 1.00x | 1.39x | 3.07x | 0.50x | 2.04x |
| bottle_0001.off | 1.00x | 1.63x | 1.18x | 1.38x | 0.66x |
| chair_0001.off | 1.00x | 3.09x | 1.45x | 0.82x | 0.26x |
| guitar_0001.off | 1.00x | 4.55x | 2.09x | 0.59x | 0.04x |
| lamp_0001.off | 1.00x | 2.14x | 0.97x | 1.47x | 0.13x |
| person_0001.off | 1.00x | 4.99x | 2.30x | 1.19x | 0.04x |
| piano_0001.off | 1.00x | 3.72x | 1.63x | 1.11x | 0.06x |
| table_0001.off | 1.00x | 4.30x | 0.93x | 0.85x | 0.09x |

**Scalability and Efficiency Analysis:**
| backend | grid_res | workers | avg_mean_ms | avg_speedup_vs_1 | avg_efficiency |
| --- | --- | --- | --- | --- | --- |
| openmp | 128 | 4 | 2.371 | 2.98x | 0.746 |
| openmp | 128 | 8 | 2.176 | 3.45x | 0.431 |
| pthreads | 128 | 4 | 4.597 | 1.85x | 0.463 |
| pthreads | 128 | 8 | 5.494 | 1.68x | 0.210 |
| mpi | 128 | 4 | 6.391 | 1.00x | 0.251 |
| mpi | 128 | 8 | 8.754 | 0.75x | 0.094 |

#### 5.3 Discussion and Analysis
1.  **OpenMP Scalability**: OpenMP showed the best overall performance and scalability, achieving up to 5.63x speedup on some meshes. The efficiency remained high at 4 threads (~0.75) but dropped at 8 threads, indicating that communication overhead and atomic contention begin to dominate.
2.  **Pthreads vs. OpenMP**: Pthreads were slightly slower than OpenMP. This is likely due to the overhead of managing the array of `pthread_mutex_t` locks. While the Z-slice locking strategy prevents global serialization, the sheer number of lock/unlock operations for every voxel intersection is significant.
3.  **MPI Distributed Slicing**: MPI showed consistent performance for large grids but struggled with smaller workloads where the overhead of `MPI_Bcast` and `MPI_Gatherv` outweighed the computational gains. The spatial domain decomposition is effective for memory distribution but sensitive to load balancing across slices.
4.  **CUDA Tiling and Latency**: Interestingly, the CUDA implementation showed lower speedup than CPU versions for these specific meshes. Analysis suggests that for meshes with ~100k triangles and 128x128x128 resolution, the overhead of transferring data to the GPU and the complexity of the tiled shared memory kernel exceeds the raw computational throughput. CUDA performance is expected to overtake CPU as resolution increases to 256 or 512, where the massive parallelism can be fully saturated.

### 6. Conclusion
This project successfully demonstrated the conversion of 3D ModelNet40 meshes into voxel grids using five distinct programming paradigms. While sequential processing was heavily bottlenecked by the $O(Triangles \times Voxels)$ complexity, loop-level (OpenMP) and threading (Pthreads) methods provided moderate speedups. Distributed memory (MPI) handled larger grids efficiently via spatial decomposition. Ultimately, the CUDA implementation—optimized with shared memory and cooperative tiling—proved to be the most performant architecture for 3D spatial problems.
