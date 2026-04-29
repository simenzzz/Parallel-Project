#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import statistics
import sys
import typing
from collections import defaultdict
from pathlib import Path


BACKEND_ORDER = {
    "sequential": 0,
    "openmp": 1,
    "pthreads": 2,
    "mpi": 3,
    "cuda": 4,
}
BEST_BACKENDS = ("openmp", "pthreads", "mpi", "cuda")
SCALING_BACKENDS = ("openmp", "pthreads", "mpi")
CONFIG_PREFIX = {
    "sequential": "seq",
    "openmp": "omp",
    "pthreads": "pth",
    "mpi": "mpi",
    "cuda": "cuda",
}
REQUIRED_COLUMNS = {
    "impl",
    "config",
    "mesh_file",
    "num_triangles",
    "grid_res",
    "elapsed_ms",
}
CORE_SESSION_CSVS = {"sequential.csv", "openmp.csv", "pthreads.csv", "mpi.csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize voxelizer benchmark CSVs.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="Explicit CSV files or glob patterns. Overrides --fresh-only.",
    )
    parser.add_argument(
        "--fresh-only",
        action="store_true",
        help="Use the newest timestamped run under results/run_*/ when available.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Stdout format. CSV mode still writes all artifacts and prints a manifest.",
    )
    parser.add_argument(
        "--per-resolution",
        action="store_true",
        help="Print runtime/speedup tables as one Markdown table per resolution.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/summary",
        help="Directory for generated CSV and Markdown artifacts.",
    )
    return parser.parse_args()


def normalize_mesh_name(mesh_file: str) -> str:
    return Path(mesh_file).name


def config_sort_value(impl: str, config: str) -> tuple[int, str]:
    number = ""
    for char in config:
        if char.isdigit():
            number += char
    if number:
        return (int(number), config)
    return (math.inf, config)


def row_sort_key(row: dict[str, object]) -> tuple[int, str, int, tuple[int, str], str]:
    return (
        int(row["grid_res"]),
        str(row["mesh_display"]),
        BACKEND_ORDER.get(str(row["impl"]), 999),
        config_sort_value(str(row["impl"]), str(row["config"])),
        str(row["config"]),
    )


def expand_input_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        if not matches:
            candidate = Path(pattern)
            if candidate.is_file():
                matches = [candidate]
        for match in sorted(matches):
            resolved = match.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(match)
    return paths


def newest_session_csvs(root: Path) -> list[Path]:
    sessions = [path for path in root.glob("run_*") if path.is_dir()]
    if not sessions:
        return []
    complete_sessions = [
        path
        for path in sessions
        if CORE_SESSION_CSVS.issubset({csv_path.name for csv_path in path.glob("*.csv")})
    ]
    candidates = complete_sessions if complete_sessions else sessions
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return sorted(path for path in newest.glob("*.csv") if path.is_file())


def select_input_paths(args: argparse.Namespace) -> list[Path]:
    if args.input:
        paths = expand_input_patterns(args.input)
        if not paths:
            raise ValueError("No CSV files matched --input patterns.")
        return paths

    results_root = Path("results")
    if args.fresh_only:
        session_paths = newest_session_csvs(results_root)
        if session_paths:
            return session_paths

    root_paths = sorted(path for path in results_root.glob("*.csv") if path.is_file())
    if root_paths:
        return root_paths

    session_paths = newest_session_csvs(results_root)
    if session_paths:
        return session_paths

    raise ValueError("No benchmark CSV files found under results/.")


def read_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = REQUIRED_COLUMNS - fieldnames
            if missing:
                raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
            for row in reader:
                rows.append(
                    {
                        "impl": row["impl"],
                        "config": row["config"],
                        "mesh_file": row["mesh_file"],
                        "mesh_display": normalize_mesh_name(row["mesh_file"]),
                        "num_triangles": int(row["num_triangles"]),
                        "grid_res": int(row["grid_res"]),
                        "elapsed_ms": float(row["elapsed_ms"]),
                        "source_csv": str(path),
                    }
                )
    if not rows:
        raise ValueError("Input CSV files did not contain any benchmark rows.")
    return rows


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["impl"]),
            str(row["config"]),
            str(row["mesh_file"]),
            int(row["grid_res"]),
        )
        groups[key].append(row)

    seq_baseline: dict[tuple[str, int], float] = {}
    for (impl, _config, mesh_file, grid_res), group_rows in groups.items():
        if impl != "sequential":
            continue
        seq_baseline[(mesh_file, grid_res)] = statistics.mean(
            float(row["elapsed_ms"]) for row in group_rows
        )

    overall: list[dict[str, object]] = []
    for (impl, config, mesh_file, grid_res), group_rows in groups.items():
        elapsed_values = [float(row["elapsed_ms"]) for row in group_rows]
        mean_ms = statistics.mean(elapsed_values)
        stdev_ms = statistics.stdev(elapsed_values) if len(elapsed_values) > 1 else 0.0
        base = seq_baseline.get((mesh_file, grid_res))
        overall.append(
            {
                "impl": impl,
                "config": config,
                "mesh_file": mesh_file,
                "mesh_display": normalize_mesh_name(mesh_file),
                "grid_res": grid_res,
                "num_triangles": int(group_rows[0]["num_triangles"]),
                "runs": len(elapsed_values),
                "mean_ms": mean_ms,
                "stdev_ms": stdev_ms,
                "min_ms": min(elapsed_values),
                "max_ms": max(elapsed_values),
                "speedup_vs_seq": (base / mean_ms) if base is not None and mean_ms > 0.0 else None,
            }
        )

    overall.sort(key=row_sort_key)
    return overall


def build_best_per_backend(overall_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in overall_rows:
        impl = str(row["impl"])
        if impl not in BEST_BACKENDS:
            continue
        grouped[(impl, int(row["grid_res"]), str(row["config"]))].append(row)

    candidates: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for (impl, grid_res, config), rows_for_config in grouped.items():
        avg_runtime = statistics.mean(float(row["mean_ms"]) for row in rows_for_config)
        avg_speedup = statistics.mean(
            float(row["speedup_vs_seq"])
            for row in rows_for_config
            if row["speedup_vs_seq"] is not None
        )
        candidates[(impl, grid_res)].append(
            {
                "impl": impl,
                "grid_res": grid_res,
                "config": config,
                "meshes": len(rows_for_config),
                "avg_mean_ms": avg_runtime,
                "avg_speedup_vs_seq": avg_speedup,
            }
        )

    best_rows: list[dict[str, object]] = []
    for backend in BEST_BACKENDS:
        for grid_res in sorted({int(row["grid_res"]) for row in overall_rows}):
            options = candidates.get((backend, grid_res), [])
            if not options:
                continue
            winner = min(
                options,
                key=lambda row: (
                    float(row["avg_mean_ms"]),
                    config_sort_value(str(row["impl"]), str(row["config"])),
                    str(row["config"]),
                ),
            )
            best_rows.append(winner)

    best_rows.sort(
        key=lambda row: (
            int(row["grid_res"]),
            BACKEND_ORDER.get(str(row["impl"]), 999),
            config_sort_value(str(row["impl"]), str(row["config"])),
        )
    )
    return best_rows


def build_best_config_lookup(best_rows: list[dict[str, object]]) -> dict[tuple[str, int], str]:
    return {
        (str(row["impl"]), int(row["grid_res"])): str(row["config"])
        for row in best_rows
    }


def build_runtime_or_speedup_table(
    overall_rows: list[dict[str, object]],
    field_name: str,
) -> tuple[list[str], list[dict[str, object]]]:
    seen_columns: set[tuple[str, str]] = set()
    column_pairs: list[tuple[str, str]] = []
    value_map: dict[tuple[str, str, int], dict[tuple[str, str], object]] = defaultdict(dict)

    for row in overall_rows:
        pair = (str(row["impl"]), str(row["config"]))
        if pair not in seen_columns:
            seen_columns.add(pair)
            column_pairs.append(pair)
        value_map[(str(row["mesh_file"]), str(row["mesh_display"]), int(row["grid_res"]))][pair] = row[field_name]

    column_pairs.sort(
        key=lambda pair: (
            BACKEND_ORDER.get(pair[0], 999),
            config_sort_value(pair[0], pair[1]),
            pair[1],
        )
    )

    headers = ["mesh_file", "mesh", "grid_res", *[f"{impl}:{config}" for impl, config in column_pairs]]
    records: list[dict[str, object]] = []
    for mesh_file, mesh_display, grid_res in sorted(value_map, key=lambda item: (item[2], item[1])):
        record: dict[str, object] = {
            "mesh_file": mesh_file,
            "mesh": mesh_display,
            "grid_res": grid_res,
        }
        for pair in column_pairs:
            record[f"{pair[0]}:{pair[1]}"] = value_map[(mesh_file, mesh_display, grid_res)].get(pair)
        records.append(record)

    return headers, records


def build_report_tables(
    overall_rows: list[dict[str, object]],
    best_lookup: dict[tuple[str, int], str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_key = {
        (str(row["impl"]), str(row["config"]), str(row["mesh_file"]), int(row["grid_res"])): row
        for row in overall_rows
    }
    grid_resolutions = sorted({int(row["grid_res"]) for row in overall_rows})
    mesh_pairs = sorted(
        {(str(row["mesh_file"]), str(row["mesh_display"])) for row in overall_rows},
        key=lambda pair: pair[1],
    )

    runtime_rows: list[dict[str, object]] = []
    speedup_rows: list[dict[str, object]] = []
    for grid_res in grid_resolutions:
        for mesh_file, mesh_display in mesh_pairs:
            seq_row = by_key.get(("sequential", "seq_1t", mesh_file, grid_res))
            if seq_row is None:
                continue

            runtime_record: dict[str, object] = {
                "mesh_file": mesh_file,
                "mesh": mesh_display,
                "grid_res": grid_res,
                "seq": seq_row["mean_ms"],
            }
            speedup_record: dict[str, object] = {
                "mesh_file": mesh_file,
                "mesh": mesh_display,
                "grid_res": grid_res,
                "seq": 1.0,
            }

            for backend, runtime_key, speedup_key in (
                ("openmp", "best_omp", "best_omp"),
                ("pthreads", "best_pth", "best_pth"),
                ("mpi", "best_mpi", "best_mpi"),
                ("cuda", "best_cuda", "best_cuda"),
            ):
                config = best_lookup.get((backend, grid_res))
                chosen = by_key.get((backend, config, mesh_file, grid_res)) if config else None
                runtime_record[runtime_key] = chosen["mean_ms"] if chosen else None
                speedup_record[speedup_key] = chosen["speedup_vs_seq"] if chosen else None

            runtime_rows.append(runtime_record)
            speedup_rows.append(speedup_record)

    runtime_rows.sort(key=lambda row: (int(row["grid_res"]), str(row["mesh"])))
    speedup_rows.sort(key=lambda row: (int(row["grid_res"]), str(row["mesh"])))
    return runtime_rows, speedup_rows


def parse_workers(config: str) -> typing.Optional[int]:
    digits = "".join(char for char in config if char.isdigit())
    return int(digits) if digits else None


def build_scalability_rows(overall_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row["impl"]), str(row["config"]), str(row["mesh_file"]), int(row["grid_res"])): row
        for row in overall_rows
    }
    meshes = sorted({str(row["mesh_file"]) for row in overall_rows})
    grid_resolutions = sorted({int(row["grid_res"]) for row in overall_rows})
    results: list[dict[str, object]] = []

    for backend in SCALING_BACKENDS:
        prefix = CONFIG_PREFIX[backend]
        for grid_res in grid_resolutions:
            for config in (f"{prefix}_1{'t' if backend != 'mpi' else 'p'}", f"{prefix}_4{'t' if backend != 'mpi' else 'p'}", f"{prefix}_8{'t' if backend != 'mpi' else 'p'}"):
                workers = parse_workers(config)
                if workers is None:
                    continue
                mesh_runtimes: list[float] = []
                mesh_speedups: list[float] = []
                mesh_efficiencies: list[float] = []
                for mesh_file in meshes:
                    base_config = f"{prefix}_1{'t' if backend != 'mpi' else 'p'}"
                    base_row = by_key.get((backend, base_config, mesh_file, grid_res))
                    row = by_key.get((backend, config, mesh_file, grid_res))
                    if base_row is None or row is None:
                        continue
                    mean_runtime = float(row["mean_ms"])
                    speedup = float(base_row["mean_ms"]) / mean_runtime if mean_runtime > 0.0 else None
                    if speedup is None:
                        continue
                    mesh_runtimes.append(mean_runtime)
                    mesh_speedups.append(speedup)
                    mesh_efficiencies.append(speedup / workers)
                if not mesh_runtimes:
                    continue
                results.append(
                    {
                        "impl": backend,
                        "grid_res": grid_res,
                        "config": config,
                        "workers": workers,
                        "meshes": len(mesh_runtimes),
                        "avg_mean_ms": statistics.mean(mesh_runtimes),
                        "avg_speedup_vs_1": statistics.mean(mesh_speedups),
                        "avg_efficiency": statistics.mean(mesh_efficiencies),
                    }
                )

    results.sort(
        key=lambda row: (
            BACKEND_ORDER.get(str(row["impl"]), 999),
            int(row["grid_res"]),
            int(row["workers"]),
        )
    )
    return results


def build_cuda_comparison_rows(overall_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in overall_rows:
        if row["impl"] != "cuda":
            continue
        grouped[(int(row["grid_res"]), str(row["config"]))].append(row)

    by_res: dict[int, list[dict[str, object]]] = defaultdict(list)
    for (grid_res, config), rows_for_config in grouped.items():
        avg_runtime = statistics.mean(float(row["mean_ms"]) for row in rows_for_config)
        avg_speedup = statistics.mean(float(row["speedup_vs_seq"]) for row in rows_for_config if row["speedup_vs_seq"] is not None)
        by_res[grid_res].append(
            {
                "grid_res": grid_res,
                "config": config,
                "meshes": len(rows_for_config),
                "avg_mean_ms": avg_runtime,
                "avg_speedup_vs_seq": avg_speedup,
            }
        )

    results: list[dict[str, object]] = []
    for grid_res in sorted(by_res):
        configs = sorted(
            by_res[grid_res],
            key=lambda row: (
                float(row["avg_mean_ms"]),
                config_sort_value("cuda", str(row["config"])),
                str(row["config"]),
            ),
        )
        winner = configs[0]
        runner_up = configs[1] if len(configs) > 1 else None
        margin_ms = (
            float(runner_up["avg_mean_ms"]) - float(winner["avg_mean_ms"])
            if runner_up is not None
            else 0.0
        )
        margin_pct = (
            (margin_ms / float(runner_up["avg_mean_ms"])) * 100.0
            if runner_up is not None and float(runner_up["avg_mean_ms"]) > 0.0
            else 0.0
        )
        results.append(
            {
                "grid_res": grid_res,
                "best_config": winner["config"],
                "best_mean_ms": winner["avg_mean_ms"],
                "best_avg_speedup_vs_seq": winner["avg_speedup_vs_seq"],
                "runner_up_config": runner_up["config"] if runner_up is not None else "",
                "runner_up_mean_ms": runner_up["avg_mean_ms"] if runner_up is not None else None,
                "margin_ms": margin_ms,
                "margin_pct_vs_runner_up": margin_pct,
            }
        )
    return results


def format_number(value: object, decimals: int = 3, suffix: str = "") -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{float(value):.{decimals}f}{suffix}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_overall_markdown(overall_rows: list[dict[str, object]]) -> str:
    headers = ["impl", "config", "mesh", "grid_res", "runs", "mean_ms", "stdev_ms", "min_ms", "max_ms", "speedup_vs_seq"]
    rows: list[list[str]] = []
    for row in overall_rows:
        rows.append(
            [
                str(row["impl"]),
                str(row["config"]),
                str(row["mesh_display"]),
                str(row["grid_res"]),
                str(row["runs"]),
                format_number(row["mean_ms"]),
                format_number(row["stdev_ms"]),
                format_number(row["min_ms"]),
                format_number(row["max_ms"]),
                format_number(row["speedup_vs_seq"], decimals=2, suffix="x"),
            ]
        )
    return markdown_table(headers, rows)


def render_best_configs_markdown(best_rows: list[dict[str, object]]) -> str:
    headers = ["backend", "grid_res", "best_config", "meshes", "avg_mean_ms", "avg_speedup_vs_seq"]
    rows: list[list[str]] = []
    for row in best_rows:
        rows.append(
            [
                str(row["impl"]),
                str(row["grid_res"]),
                str(row["config"]),
                str(row["meshes"]),
                format_number(row["avg_mean_ms"]),
                format_number(row["avg_speedup_vs_seq"], decimals=2, suffix="x"),
            ]
        )
    return markdown_table(headers, rows)


def render_wide_markdown(headers: list[str], records: list[dict[str, object]], *, speedup: bool) -> str:
    rows: list[list[str]] = []
    for record in records:
        rendered = [str(record["mesh"]), str(record["grid_res"])]
        for header in headers[3:]:
            value = record.get(header)
            rendered.append(format_number(value, decimals=2 if speedup else 3, suffix="x" if speedup else ""))
        rows.append(rendered)
    display_headers = ["mesh", "grid_res", *headers[3:]]
    return markdown_table(display_headers, rows)


def render_report_tables_by_resolution(rows: list[dict[str, object]], *, speedup: bool) -> str:
    sections: list[str] = []
    headers = ["mesh", "seq", "best_omp", "best_pth", "best_mpi", "best_cuda"]
    for grid_res in sorted({int(row["grid_res"]) for row in rows}):
        section_rows: list[list[str]] = []
        for row in rows:
            if int(row["grid_res"]) != grid_res:
                continue
            section_rows.append(
                [
                    str(row["mesh"]),
                    format_number(row["seq"], decimals=2 if speedup else 3, suffix="x" if speedup else ""),
                    format_number(row["best_omp"], decimals=2 if speedup else 3, suffix="x" if speedup else ""),
                    format_number(row["best_pth"], decimals=2 if speedup else 3, suffix="x" if speedup else ""),
                    format_number(row["best_mpi"], decimals=2 if speedup else 3, suffix="x" if speedup else ""),
                    format_number(row["best_cuda"], decimals=2 if speedup else 3, suffix="x" if speedup else ""),
                ]
            )
        sections.append(f"### Resolution {grid_res}\n\n{markdown_table(headers, section_rows)}")
    return "\n\n".join(sections)


def render_scalability_markdown(rows: list[dict[str, object]]) -> str:
    headers = ["backend", "grid_res", "config", "workers", "avg_mean_ms", "avg_speedup_vs_1", "avg_efficiency"]
    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                str(row["impl"]),
                str(row["grid_res"]),
                str(row["config"]),
                str(row["workers"]),
                format_number(row["avg_mean_ms"]),
                format_number(row["avg_speedup_vs_1"], decimals=2, suffix="x"),
                format_number(row["avg_efficiency"], decimals=3),
            ]
        )
    return markdown_table(headers, table_rows)


def render_cuda_comparison_markdown(rows: list[dict[str, object]]) -> str:
    headers = [
        "grid_res",
        "best_config",
        "best_mean_ms",
        "best_avg_speedup_vs_seq",
        "runner_up_config",
        "runner_up_mean_ms",
        "margin_ms",
        "margin_pct_vs_runner_up",
    ]
    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                str(row["grid_res"]),
                str(row["best_config"]),
                format_number(row["best_mean_ms"]),
                format_number(row["best_avg_speedup_vs_seq"], decimals=2, suffix="x"),
                str(row["runner_up_config"]),
                format_number(row["runner_up_mean_ms"]),
                format_number(row["margin_ms"]),
                format_number(row["margin_pct_vs_runner_up"], decimals=2, suffix="%"),
            ]
        )
    return markdown_table(headers, table_rows)


def csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: csv_value(row.get(header)) for header in headers})


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_paths = select_input_paths(args)
        raw_rows = read_rows(input_paths)
        overall_rows = aggregate_rows(raw_rows)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    best_rows = build_best_per_backend(overall_rows)
    best_lookup = build_best_config_lookup(best_rows)
    runtime_headers, runtime_records = build_runtime_or_speedup_table(overall_rows, "mean_ms")
    speedup_headers, speedup_records = build_runtime_or_speedup_table(overall_rows, "speedup_vs_seq")
    report_runtime_rows, report_speedup_rows = build_report_tables(overall_rows, best_lookup)
    scalability_rows = build_scalability_rows(overall_rows)
    cuda_rows = build_cuda_comparison_rows(overall_rows)

    write_csv(
        output_dir / "overall.csv",
        [
            "impl",
            "config",
            "mesh_file",
            "mesh_display",
            "grid_res",
            "num_triangles",
            "runs",
            "mean_ms",
            "stdev_ms",
            "min_ms",
            "max_ms",
            "speedup_vs_seq",
        ],
        overall_rows,
    )
    write_csv(
        output_dir / "best_per_backend.csv",
        ["impl", "grid_res", "config", "meshes", "avg_mean_ms", "avg_speedup_vs_seq"],
        best_rows,
    )
    write_csv(
        output_dir / "backend_best_configs.csv",
        ["impl", "grid_res", "config", "meshes", "avg_mean_ms", "avg_speedup_vs_seq"],
        best_rows,
    )
    write_csv(output_dir / "runtime_table.csv", runtime_headers, runtime_records)
    write_csv(output_dir / "speedup_table.csv", speedup_headers, speedup_records)
    write_csv(
        output_dir / "runtime_by_resolution.csv",
        ["mesh_file", "mesh", "grid_res", "seq", "best_omp", "best_pth", "best_mpi", "best_cuda"],
        report_runtime_rows,
    )
    write_csv(
        output_dir / "speedup_by_resolution.csv",
        ["mesh_file", "mesh", "grid_res", "seq", "best_omp", "best_pth", "best_mpi", "best_cuda"],
        report_speedup_rows,
    )
    write_csv(
        output_dir / "scalability.csv",
        ["impl", "grid_res", "config", "workers", "meshes", "avg_mean_ms", "avg_speedup_vs_1", "avg_efficiency"],
        scalability_rows,
    )
    write_csv(
        output_dir / "cuda_blocksize_comparison.csv",
        [
            "grid_res",
            "best_config",
            "best_mean_ms",
            "best_avg_speedup_vs_seq",
            "runner_up_config",
            "runner_up_mean_ms",
            "margin_ms",
            "margin_pct_vs_runner_up",
        ],
        cuda_rows,
    )

    backend_best_md = render_best_configs_markdown(best_rows)
    runtime_by_resolution_md = render_report_tables_by_resolution(report_runtime_rows, speedup=False)
    speedup_by_resolution_md = render_report_tables_by_resolution(report_speedup_rows, speedup=True)
    scalability_md = render_scalability_markdown(scalability_rows)
    cuda_md = render_cuda_comparison_markdown(cuda_rows)

    write_text(output_dir / "backend_best_configs.md", backend_best_md)
    write_text(output_dir / "runtime_by_resolution.md", runtime_by_resolution_md)
    write_text(output_dir / "speedup_by_resolution.md", speedup_by_resolution_md)
    write_text(output_dir / "scalability_notes.md", scalability_md)
    write_text(output_dir / "cuda_blocksize_comparison.md", cuda_md)

    if args.format == "csv":
        print(f"inputs: {', '.join(str(path) for path in input_paths)}")
        print(f"wrote: {output_dir / 'overall.csv'}")
        print(f"wrote: {output_dir / 'best_per_backend.csv'}")
        print(f"wrote: {output_dir / 'backend_best_configs.csv'}")
        print(f"wrote: {output_dir / 'runtime_table.csv'}")
        print(f"wrote: {output_dir / 'speedup_table.csv'}")
        print(f"wrote: {output_dir / 'runtime_by_resolution.md'}")
        print(f"wrote: {output_dir / 'speedup_by_resolution.md'}")
        print(f"wrote: {output_dir / 'backend_best_configs.md'}")
        print(f"wrote: {output_dir / 'cuda_blocksize_comparison.md'}")
        print(f"wrote: {output_dir / 'scalability_notes.md'}")
        return 0

    stdout_sections = [
        "# Overall\n\n" + render_overall_markdown(overall_rows),
        "# Best Per Backend\n\n" + backend_best_md,
    ]
    if args.per_resolution:
        stdout_sections.append("# Runtime By Resolution\n\n" + runtime_by_resolution_md)
        stdout_sections.append("# Speedup By Resolution\n\n" + speedup_by_resolution_md)
    else:
        stdout_sections.append("# Runtime Table\n\n" + render_wide_markdown(runtime_headers, runtime_records, speedup=False))
        stdout_sections.append("# Speedup Table\n\n" + render_wide_markdown(speedup_headers, speedup_records, speedup=True))
    print("\n\n".join(stdout_sections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
