#!/usr/bin/env python3
"""Benchmark MPI sample sort, oversampling, and ping-pong communication."""

import argparse
import csv
import pathlib
import re
import statistics
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
RESULT = re.compile(r"RESULT (.*)")
PING = re.compile(r"PING (\d+) ([0-9.eE+-]+)")
PHASES = (
    "local_sort", "sampling", "partition", "alltoall_counts",
    "alltoallv", "final_sort",
)


def parse_result(output):
    match = RESULT.search(output)
    if not match:
        raise RuntimeError(f"No RESULT line in MPI output: {output}")
    return dict(token.split("=", 1) for token in match.group(1).split())


def execute(command):
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=True,
    )
    return completed.stdout


def parallel_run(ranks, n, distribution, seed, oversampling):
    output = execute([
        "mpirun", "--oversubscribe", "-np", str(ranks), "./samplesort",
        "parallel", str(n), distribution, str(seed), str(oversampling),
    ])
    result = parse_result(output)
    if int(result["valid"]) != 1 or int(result["total_count"]) != n:
        raise RuntimeError(f"Invalid distributed sort result: {result}")
    return result


def serial_run(n, distribution, seed, shards):
    output = execute([
        "mpirun", "--oversubscribe", "-np", "1", "./samplesort", "serial",
        str(n), distribution, str(seed), str(shards),
    ])
    result = parse_result(output)
    if int(result["sorted"]) != 1:
        raise RuntimeError(f"Serial std::sort validation failed: {result}")
    return result


def pingpong_fit():
    output = execute([
        "mpirun", "--oversubscribe", "-np", "2", "./samplesort", "pingpong",
    ])
    points = [(int(size), float(seconds))
              for size, seconds in PING.findall(output)]
    if len(points) < 3:
        raise RuntimeError(f"Insufficient ping-pong points: {output}")
    mean_x = statistics.mean(size for size, _ in points)
    mean_y = statistics.mean(seconds for _, seconds in points)
    denominator = sum((size - mean_x) ** 2 for size, _ in points)
    beta = sum((size - mean_x) * (seconds - mean_y)
               for size, seconds in points) / denominator
    alpha = mean_y - beta * mean_x
    residual = sum((seconds - (alpha + beta * size)) ** 2
                   for size, seconds in points)
    total = sum((seconds - mean_y) ** 2 for _, seconds in points)
    r_squared = 1.0 - residual / total if total else 1.0
    return points, max(0.0, alpha), max(0.0, beta), r_squared


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows to write to {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_parallel(records, baseline_seconds, alpha, beta, category):
    times = [float(row["total"]) for row in records]
    median_time = statistics.median(times)
    n = int(records[0]["N"])
    ranks = int(records[0]["ranks"])
    max_bytes = int(statistics.median(
        int(row["max_bytes_sent"]) for row in records
    ))
    predicted = (ranks - 1) * alpha + beta * max_bytes
    measured_alltoallv = statistics.median(
        float(row["alltoallv"]) for row in records
    )
    result = {
        "category": category,
        "distribution": records[0]["distribution"],
        "N": n,
        "ranks": ranks,
        "oversampling": int(records[0]["oversampling"]),
        "serial_median_seconds": baseline_seconds,
        "median_seconds": median_time,
        "speedup_vs_serial": baseline_seconds / median_time,
        "efficiency": baseline_seconds / median_time / ranks,
    }
    for phase in PHASES:
        result[f"{phase}_seconds"] = statistics.median(
            float(row[phase]) for row in records
        )
    result["measured_alltoallv_seconds"] = measured_alltoallv
    result["predicted_alltoallv_seconds"] = predicted
    result["alltoallv_model_ratio"] = (
        measured_alltoallv / predicted if predicted > 0 else 0.0
    )
    result["max_mean_ratio"] = statistics.median(
        float(row["max_mean_ratio"]) for row in records
    )
    result["rank_counts"] = records[-1]["rank_counts"]
    result["max_bytes_sent"] = max_bytes
    result["total_count"] = n
    return result


def raw_record(category, distribution, n, ranks, oversampling, repeat, result):
    row = {
        "category": category,
        "distribution": distribution,
        "N": n,
        "ranks": ranks,
        "oversampling": oversampling,
        "repeat": repeat,
    }
    for key in (*PHASES, "total", "max_mean_ratio", "max_bytes_sent",
                "total_count", "valid", "rank_counts"):
        row[key] = result.get(key, "")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000_000)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--oversampling", type=int, default=16)
    parser.add_argument("--oversampling-tests", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--output", default="results/samplesort")
    args = parser.parse_args()
    if args.n < 100_000_000:
        parser.error("N must be at least 100,000,000 for the reported benchmark")
    if args.n > 0xFFFFFFFF or args.repeats < 3:
        parser.error("N must fit the unique-id field and repeats must be at least 3")
    if not args.ranks or any(rank < 1 for rank in args.ranks):
        parser.error("provide positive MPI rank counts")
    if any(factor < 1 for factor in args.oversampling_tests):
        parser.error("oversampling factors must be positive")
    if not (ROOT / "samplesort").exists():
        parser.error("Build first with `make`")

    max_ranks = max(args.ranks)
    raw_rows = []
    summary_rows = []
    baseline_times = {}
    baseline_raw = []
    for distribution in ("uniform", "skewed"):
        samples = []
        latest = None
        for repeat in range(1, args.repeats + 1):
            latest = serial_run(args.n, distribution, args.seed, max_ranks)
            samples.append(float(latest["total"]))
            baseline_raw.append({
                "category": "baseline",
                "distribution": distribution,
                "N": args.n,
                "ranks": 1,
                "oversampling": "",
                "repeat": repeat,
                **{phase: "" for phase in PHASES},
                "total": latest["total"],
                "max_mean_ratio": "",
                "max_bytes_sent": "",
                "total_count": args.n,
                "valid": latest["sorted"],
                "rank_counts": args.n,
            })
        median = statistics.median(samples)
        baseline_times[distribution] = median
        summary_rows.append({
            "category": "baseline",
            "distribution": distribution,
            "N": args.n,
            "ranks": 1,
            "oversampling": "",
            "serial_median_seconds": median,
            "median_seconds": median,
            "speedup_vs_serial": 1.0,
            "efficiency": 1.0,
            **{f"{phase}_seconds": "" for phase in PHASES},
            "measured_alltoallv_seconds": "",
            "predicted_alltoallv_seconds": "",
            "alltoallv_model_ratio": "",
            "max_mean_ratio": "",
            "rank_counts": "",
            "max_bytes_sent": "",
            "total_count": args.n,
        })

    ping_points, alpha, beta, r_squared = pingpong_fit()
    ping_rows = [
        {"message_bytes": size, "one_way_seconds": seconds}
        for size, seconds in ping_points
    ]
    communication_model = [{
        "alpha_seconds": alpha,
        "beta_seconds_per_byte": beta,
        "beta_nanoseconds_per_byte": beta * 1e9,
        "r_squared": r_squared,
        "fit_points": len(ping_points),
    }]

    records_by_key = {}
    for distribution in ("uniform", "skewed"):
        for ranks in args.ranks:
            samples = []
            for repeat in range(1, args.repeats + 1):
                result = parallel_run(
                    ranks, args.n, distribution, args.seed, args.oversampling,
                )
                row = raw_record(
                    "scaling", distribution, args.n, ranks,
                    args.oversampling, repeat, result,
                )
                raw_rows.append(row)
                samples.append(row)
            records_by_key[(distribution, ranks, args.oversampling)] = samples

    tested_factors = sorted(set(args.oversampling_tests) |
                            {args.oversampling})
    for distribution in ("uniform", "skewed"):
        for factor in tested_factors:
            key = (distribution, max_ranks, factor)
            if factor == args.oversampling:
                samples = records_by_key[key]
            else:
                samples = []
                for repeat in range(1, args.repeats + 1):
                    result = parallel_run(
                        max_ranks, args.n, distribution, args.seed, factor,
                    )
                    row = raw_record(
                        "oversampling", distribution, args.n, max_ranks,
                        factor, repeat, result,
                    )
                    raw_rows.append(row)
                    samples.append(row)
                records_by_key[key] = samples
            summary_rows.append(summarize_parallel(
                samples, baseline_times[distribution], alpha, beta, "oversampling",
            ))

    for distribution in ("uniform", "skewed"):
        for ranks in args.ranks:
            samples = records_by_key[(distribution, ranks, args.oversampling)]
            summary_rows.append(summarize_parallel(
                samples, baseline_times[distribution], alpha, beta, "scaling",
            ))

    results_dir = ROOT / args.output
    results_dir.parent.mkdir(parents=True, exist_ok=True)
    prefix = results_dir
    write_csv(prefix.with_name(prefix.name + "_raw.csv"), baseline_raw + raw_rows)
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), summary_rows)
    write_csv(prefix.with_name(prefix.name + "_pingpong.csv"), ping_rows)
    write_csv(prefix.with_name(prefix.name + "_comm_model.csv"), communication_model)
    print(f"Serial baseline medians: {baseline_times}")
    print(f"Ping-pong fit: alpha={alpha:.6e} s, beta={beta:.6e} s/byte, R2={r_squared:.4f}")
    print(f"Wrote {prefix}_raw.csv, {prefix}_summary.csv, ping-pong, and model CSVs")


if __name__ == "__main__":
    main()
