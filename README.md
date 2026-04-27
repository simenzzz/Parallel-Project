# Parallel 3D Voxelization Project

This project converts normalized triangle meshes in `.off` format into voxel grids and compares five implementations of the same voxelization algorithm:

- sequential
- OpenMP
- POSIX threads
- MPI
- CUDA

The benchmark workflow measures kernel runtime, verifies every backend against the sequential reference, and exports report-ready summary tables.

## Repository Layout

```text
.
├── data/                  Input meshes
├── src/
│   ├── common/            Shared parser, SAT test, voxel grid, timing, CSV helpers
│   ├── sequential/        Sequential baseline
│   ├── openmp/            OpenMP implementation
│   ├── pthreads/          Pthreads implementation
│   ├── mpi/               MPI implementation
│   └── cuda/              CUDA implementation
├── scripts/
│   ├── bench.py           Full benchmark runner
│   ├── bench.sh           Shell wrapper for bench.py
│   └── summarize.py       Aggregation and report export
├── results/               Timestamped benchmark sessions and summaries
├── report/                Report assets
└── Makefile               Top-level build
```

## Requirements

- `gcc`
- `make`
- `mpicc` and `mpirun` for the MPI target
- CUDA toolkit with `nvcc` for the CUDA target
- Python 3

Notes:

- `make all` builds CPU targets unconditionally and builds CUDA only if `nvcc` is available.
- The default CUDA architecture is `sm_89`. Override it if needed:

```bash
make CUDA_ARCH=sm_75 all
```

## Build

Build everything:

```bash
make all
```

Useful individual targets:

```bash
make sequential
make openmp
make pthreads
make mpi
make cuda
make verify
make test
```

Clean generated binaries, object files, benchmark sessions, and summaries:

```bash
make clean
```

## Running Individual Implementations

All binaries are written to `bin/`.

Sequential:

```bash
bin/voxelize_seq -i data/airplane_0001.off -r 64 -o results/seq.voxel -c results/sequential.csv
```

OpenMP:

```bash
bin/voxelize_omp -i data/airplane_0001.off -r 64 -t 4 -o results/omp.voxel -c results/openmp.csv
```

Pthreads:

```bash
bin/voxelize_pth -i data/airplane_0001.off -r 64 -t 4 -o results/pth.voxel -c results/pthreads.csv
```

MPI:

```bash
mpirun -np 4 bin/voxelize_mpi -i data/airplane_0001.off -r 64 -o results/mpi.voxel -c results/mpi.csv
```

CUDA:

```bash
bin/voxelize_cuda -i data/airplane_0001.off -r 64 -b 128 -o results/cuda.voxel -c results/cuda.csv
```

Verify a backend output against the sequential output:

```bash
bin/verify results/seq.voxel results/omp.voxel
```

## Benchmark Matrix

The benchmark runner uses:

- Meshes: 9 input meshes from `data/`
- Resolutions: `32`, `64`, `128`
- Repeats: `3`
- OpenMP threads: `1`, `4`, `8`
- Pthreads threads: `1`, `4`, `8`
- MPI processes: `1`, `4`, `8`
- CUDA block sizes: `64`, `128`, `256`

Each benchmark run:

- creates a fresh timestamped session under `results/run_*`
- writes one CSV per backend into that session
- writes voxel outputs for each run/configuration
- verifies every non-sequential output against the sequential reference

## Reproducing All Results

From the project root:

```bash
make clean
make all
python3 scripts/bench.py
python3 scripts/summarize.py --fresh-only --format csv --per-resolution
```

After `bench.py` finishes, it prints the session directory it wrote to, for example:

```text
results/run_20260427_173810
```

`summarize.py --fresh-only` uses the newest complete benchmark session under `results/run_*`.

## Summary and Report Export

Default summary to stdout plus generated artifacts:

```bash
python3 scripts/summarize.py
```

Summarize only the newest benchmark session and emit report-ready files:

```bash
python3 scripts/summarize.py --fresh-only --format csv --per-resolution
```

Summarize explicit CSVs instead of auto-selecting a session:

```bash
python3 scripts/summarize.py --input results/run_20260427_173810/*.csv
```

Write summary outputs somewhere else:

```bash
python3 scripts/summarize.py --fresh-only --output-dir /tmp/voxel-summary
```

Generated files under `results/summary/` include:

- `overall.csv`
- `best_per_backend.csv`
- `backend_best_configs.csv`
- `runtime_table.csv`
- `speedup_table.csv`
- `runtime_by_resolution.csv`
- `runtime_by_resolution.md`
- `speedup_by_resolution.csv`
- `speedup_by_resolution.md`
- `cuda_blocksize_comparison.csv`
- `cuda_blocksize_comparison.md`
- `scalability.csv`
- `scalability_notes.md`

These files are intended to be pasted directly into the report without manual rebuilding of tables.

## Current Implementation Notes

- The sequential baseline is the source of truth for correctness checks.
- All benchmark CSVs use the schema:

```text
impl,config,mesh_file,num_triangles,grid_res,elapsed_ms
```

- CUDA timing records kernel time only.
- The CUDA path precomputes per-face voxel bounds on the host and uploads them to the kernel to reduce repeated per-face min/max work in device code.

## Troubleshooting

- If CUDA does not build, confirm that `nvcc` exists and that `CUDA_ARCH` matches your GPU.
- If MPI fails to launch, verify that `mpirun` works on your machine outside restricted sandboxes.
- If you want a completely fresh benchmark dataset, run `make clean` before `python3 scripts/bench.py`.

## Remote

The local git repository is configured to use:

```text
https://github.com/simenzzz/Parallel-Project
```

## Google Colab

To run the CUDA optimizations on Google Colab (if you do not have a local NVIDIA GPU):

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File > Upload notebook** and upload the provided `CUDA_Colab.ipynb` file from the project root.
3. Make sure to change your Colab runtime to use a GPU:
   - Click **Runtime > Change runtime type**.
   - Select **T4 GPU** (or any available GPU).
   - Click **Save**.
4. Run the cells in the notebook. It will clone the repository, compile the CUDA backend, and run the benchmark.
