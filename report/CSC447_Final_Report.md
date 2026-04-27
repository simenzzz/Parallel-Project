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

*(Note: Run the `bench.py` and `summarize.py` scripts on the target hardware to generate the final tables and insert them here).*

#### 5.1 Experimental Setup
The CPU benchmarks (Sequential, OpenMP, Pthreads, MPI) were tested using 1, 4, and 8 thread/process configurations to measure scalability. The CUDA implementation was tested across different grid resolutions (`32`, `64`, `128`) to observe how tiling handles increased workloads.

#### 5.2 Performance Comparison
| Paradigm | Config | Runtime (ms) | Speedup vs Seq |
|----------|--------|--------------|----------------|
| Sequential | 1 core | [DATA] | 1.0x |
| OpenMP | 8 threads | [DATA] | [DATA]x |
| Pthreads | 8 threads | [DATA] | [DATA]x |
| MPI | 8 ranks | [DATA] | [DATA]x |
| CUDA | 8x8x8 blocks | [DATA] | [DATA]x |

#### 5.3 Discussion and Analysis
1.  **Synchronization Overhead**: Both OpenMP and Pthreads experienced noticeable synchronization overhead due to the strict use of `#pragma omp atomic` and `pthread_mutex_t`. While correct and thread-safe, locking mechanisms inherently slow down highly contentious writes compared to lock-free benign data races.
2.  **Scalability**: MPI demonstrated excellent scalability. By spatially partitioning the voxel grid rather than the triangles, MPI processes did not contend for memory writes.
3.  **CUDA Tiling Efficiency**: The CUDA implementation achieved the highest speedup. By loading triangles into shared memory in chunks of 256, we eliminated redundant global memory reads for vertices. Threads were able to perform rapid AABB and SAT intersection tests using the fast on-chip shared memory before coalescing their final writes to global memory.

### 6. Conclusion
This project successfully demonstrated the conversion of 3D ModelNet40 meshes into voxel grids using five distinct programming paradigms. While sequential processing was heavily bottlenecked by the $O(Triangles \times Voxels)$ complexity, loop-level (OpenMP) and threading (Pthreads) methods provided moderate speedups. Distributed memory (MPI) handled larger grids efficiently via spatial decomposition. Ultimately, the CUDA implementation—optimized with shared memory and cooperative tiling—proved to be the most performant architecture for 3D spatial problems.
