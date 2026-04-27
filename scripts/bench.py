#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path


MESHES = [
    "airplane_0001.off",
    "chair_0001.off",
    "table_0001.off",
    "guitar_0001.off",
    "lamp_0001.off",
    "person_0001.off",
    "bookshelf_0001.off",
    "bottle_0001.off",
    "piano_0001.off",
]
RESOLUTIONS = [32, 64, 128]
THREAD_COUNTS = [1, 4, 8]
BLOCK_SIZES = [64, 128, 256]
REPEATS = 3


@dataclass
class BenchFailure:
    step: str
    command: list[str]
    returncode: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run voxelizer benchmark matrix.")
    parser.add_argument("--bin-dir", default="./bin", help="Directory containing binaries.")
    parser.add_argument("--data-dir", default="./data", help="Directory containing OFF meshes.")
    parser.add_argument(
        "--results-dir",
        default="./results",
        help="Parent directory for timestamped benchmark sessions.",
    )
    parser.add_argument(
        "--mpirun",
        default="mpirun",
        help="MPI launcher binary to use for MPI runs.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS,
        help="Number of repetitions per mesh/resolution combination.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Record failures and continue running the remaining matrix.",
    )
    return parser.parse_args()


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def infer_resolution(byte_count: int) -> int | None:
    if byte_count <= 0:
        return None

    res = round(byte_count ** (1.0 / 3.0))
    if res > 0 and res * res * res == byte_count:
        return res
    return None


def print_verify_mismatch_details(reference_path: Path, candidate_path: Path) -> None:
    try:
        reference = reference_path.read_bytes()
        candidate = candidate_path.read_bytes()
    except OSError as exc:
        print(f"  detail read failed: {exc}", file=sys.stderr)
        return

    if len(reference) != len(candidate):
        print(
            f"  detail: size mismatch ref={len(reference)} candidate={len(candidate)}",
            file=sys.stderr,
        )
        return

    res = infer_resolution(len(reference))
    if res is None:
        print(f"  detail: could not infer cubic grid resolution from {len(reference)} bytes", file=sys.stderr)
        return

    mismatches: list[tuple[int, int, int, int, int, int]] = []
    slice_counts: dict[int, list[int]] = {}

    for idx, (ref_value, cand_value) in enumerate(zip(reference, candidate)):
        if ref_value == cand_value:
            continue

        x = idx % res
        y = (idx // res) % res
        z = idx // (res * res)
        mismatches.append((idx, x, y, z, ref_value, cand_value))
        counts = slice_counts.setdefault(z, [0, 0])
        counts[0] += ref_value
        counts[1] += cand_value

    if not mismatches:
        print("  detail: verify failed but no byte mismatches were found", file=sys.stderr)
        return

    mismatch_slices = sorted(slice_counts)
    print(
        f"  detail: total_mismatches={len(mismatches)} resolution={res} z_slices={mismatch_slices}",
        file=sys.stderr,
    )
    print("  detail: first_mismatches=", file=sys.stderr)
    for idx, x, y, z, ref_value, cand_value in mismatches[:8]:
        print(
            f"    byte={idx} coord=({x},{y},{z}) ref={ref_value} candidate={cand_value}",
            file=sys.stderr,
        )

    for z in mismatch_slices[:8]:
        base = z * res * res
        ref_count = sum(reference[base : base + res * res])
        cand_count = sum(candidate[base : base + res * res])
        print(
            f"  detail: slice z={z} occupied ref={ref_count} candidate={cand_count} delta={ref_count - cand_count}",
            file=sys.stderr,
        )


def run_command(
    command: list[str],
    step: str,
    keep_going: bool,
    *,
    verify_pair: tuple[Path, Path] | None = None,
) -> BenchFailure | None:
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        return None

    failure = BenchFailure(step=step, command=command, returncode=result.returncode)
    print(
        f"FAILED [{step}] exit={result.returncode}: {quote_command(command)}",
        file=sys.stderr,
    )
    if verify_pair is not None:
        print_verify_mismatch_details(verify_pair[0], verify_pair[1])
    if not keep_going:
        raise SystemExit(result.returncode)
    return failure


def verify_inputs(meshes: list[str], data_dir: Path) -> None:
    missing = [str(data_dir / mesh) for mesh in meshes if not (data_dir / mesh).is_file()]
    if missing:
        for path in missing:
            print(f"Missing mesh: {path}", file=sys.stderr)
        raise SystemExit(1)


def build_command(binary: Path, *args: str | int) -> list[str]:
    return [str(binary), *(str(arg) for arg in args)]


def create_session_dir(results_root: Path) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    session_dir = results_root / timestamp
    suffix = 1
    while session_dir.exists():
        session_dir = results_root / f"{timestamp}_{suffix}"
        suffix += 1
    session_dir.mkdir()
    return session_dir


def main() -> int:
    args = parse_args()
    bin_dir = Path(args.bin_dir)
    data_dir = Path(args.data_dir)
    results_root = Path(args.results_dir)
    keep_going = args.keep_going
    failures: list[BenchFailure] = []

    results_dir = create_session_dir(results_root)
    verify_inputs(MESHES, data_dir)
    print(f"Writing benchmark outputs to {results_dir}", flush=True)

    voxelize_seq = bin_dir / "voxelize_seq"
    voxelize_omp = bin_dir / "voxelize_omp"
    voxelize_pth = bin_dir / "voxelize_pth"
    voxelize_mpi = bin_dir / "voxelize_mpi"
    voxelize_cuda = bin_dir / "voxelize_cuda"
    verify = bin_dir / "verify"

    for res in RESOLUTIONS:
        for mesh in MESHES:
            stem = mesh.removesuffix(".off")
            mesh_path = data_dir / mesh
            seq_out = results_dir / f"seq_{res}_{stem}.voxel"

            for run in range(1, args.repeats + 1):
                step_prefix = f"mesh={mesh} res={res} run={run}"

                failure = run_command(
                    build_command(
                        voxelize_seq,
                        "-i",
                        mesh_path,
                        "-r",
                        res,
                        "-o",
                        seq_out,
                        "-c",
                        results_dir / "sequential.csv",
                    ),
                    f"{step_prefix} sequential",
                    keep_going,
                )
                if failure:
                    failures.append(failure)
                    continue

                for threads in THREAD_COUNTS:
                    omp_out = results_dir / f"omp_{threads}t_{res}_{stem}.voxel"
                    pth_out = results_dir / f"pth_{threads}t_{res}_{stem}.voxel"
                    mpi_out = results_dir / f"mpi_{threads}p_{res}_{stem}.voxel"

                    failure = run_command(
                        build_command(
                            voxelize_omp,
                            "-i",
                            mesh_path,
                            "-r",
                            res,
                            "-t",
                            threads,
                            "-o",
                            omp_out,
                            "-c",
                            results_dir / "openmp.csv",
                        ),
                        f"{step_prefix} openmp t={threads}",
                        keep_going,
                    )
                    if failure:
                        failures.append(failure)
                    else:
                        failure = run_command(
                            build_command(verify, seq_out, omp_out),
                            f"{step_prefix} verify openmp t={threads}",
                            keep_going,
                            verify_pair=(seq_out, omp_out),
                        )
                        if failure:
                            failures.append(failure)

                    failure = run_command(
                        build_command(
                            voxelize_pth,
                            "-i",
                            mesh_path,
                            "-r",
                            res,
                            "-t",
                            threads,
                            "-o",
                            pth_out,
                            "-c",
                            results_dir / "pthreads.csv",
                        ),
                        f"{step_prefix} pthreads t={threads}",
                        keep_going,
                    )
                    if failure:
                        failures.append(failure)
                    else:
                        failure = run_command(
                            build_command(verify, seq_out, pth_out),
                            f"{step_prefix} verify pthreads t={threads}",
                            keep_going,
                            verify_pair=(seq_out, pth_out),
                        )
                        if failure:
                            failures.append(failure)

                    failure = run_command(
                        [
                            args.mpirun,
                            "-np",
                            str(threads),
                            str(voxelize_mpi),
                            "-i",
                            str(mesh_path),
                            "-r",
                            str(res),
                            "-o",
                            str(mpi_out),
                            "-c",
                            str(results_dir / "mpi.csv"),
                        ],
                        f"{step_prefix} mpi p={threads}",
                        keep_going,
                    )
                    if failure:
                        failures.append(failure)
                    else:
                        failure = run_command(
                            build_command(verify, seq_out, mpi_out),
                            f"{step_prefix} verify mpi p={threads}",
                            keep_going,
                            verify_pair=(seq_out, mpi_out),
                        )
                        if failure:
                            failures.append(failure)

                if voxelize_cuda.is_file() and voxelize_cuda.stat().st_mode & 0o111:
                    for block_size in BLOCK_SIZES:
                        cuda_out = results_dir / f"cuda_{block_size}tpb_{res}_{stem}.voxel"
                        failure = run_command(
                            build_command(
                                voxelize_cuda,
                                "-i",
                                mesh_path,
                                "-r",
                                res,
                                "-b",
                                block_size,
                                "-o",
                                cuda_out,
                                "-c",
                                results_dir / "cuda.csv",
                            ),
                            f"{step_prefix} cuda b={block_size}",
                            keep_going,
                        )
                        if failure:
                            failures.append(failure)
                        else:
                            failure = run_command(
                                build_command(verify, seq_out, cuda_out),
                                f"{step_prefix} verify cuda b={block_size}",
                                keep_going,
                                verify_pair=(seq_out, cuda_out),
                            )
                            if failure:
                                failures.append(failure)

    if failures:
        print("\nBenchmark completed with failures:", file=sys.stderr)
        for failure in failures:
            print(
                f"- {failure.step}: exit={failure.returncode} command={quote_command(failure.command)}",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
