#!/usr/bin/env python3
"""Measure fixed-work scaling, Amdahl estimates, RNG correlation, and fixed-time SE."""

import argparse
import csv
import os
import pathlib
import re
import statistics
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent
RESULT = re.compile(r"RESULT (.*)")


def parse_result(output):
    match = RESULT.search(output)
    if not match:
        raise RuntimeError(f"Unable to parse result line: {output}")
    fields = {}
    for token in match.group(1).split():
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def run(command, threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )
    output = completed.stdout.strip()
    print(output, flush=True)
    return parse_result(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=100_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seed", default="20261123")
    parser.add_argument("--fixed-seconds", type=float, default=2.0)
    parser.add_argument("--output", default="results/monte_carlo")
    args = parser.parse_args()
    if args.paths < 100_000_000:
        parser.error("Use at least 100,000,000 paths for the required benchmark")
    if args.repeats < 3 or any(t < 1 for t in args.threads):
        parser.error("Use at least three repeats and positive thread counts")
    if args.fixed_seconds <= 0:
        parser.error("fixed-time duration must be positive")
    if not (ROOT / "monte_carlo").exists():
        parser.error("Build first with `make`")

    raw_rows = []
    summary_rows = []
    serial_samples = []
    for repeat in range(1, args.repeats + 1):
        result = run(["./monte_carlo", "serial", str(args.paths), args.seed], 1)
        elapsed = float(result["time"])
        serial_samples.append(elapsed)
        raw_rows.append({
            "category": "serial", "paths": args.paths, "threads": 1,
            "repeat": repeat, "time": elapsed,
            "estimate": result["estimate"], "standard_error": result["standard_error"],
            "black_scholes": result["black_scholes"], "checksum": result["checksum"],
            "seed": args.seed,
            "OMP_NUM_THREADS": 1, "OMP_PROC_BIND": "close", "OMP_PLACES": "cores",
        })
    serial_median = statistics.median(serial_samples)
    summary_rows.append({
        "category": "baseline", "paths": args.paths, "threads": 1,
        "median_seconds": serial_median, "speedup_vs_serial": 1.0,
        "efficiency": 1.0, "estimate": "", "standard_error": "",
        "throughput_mpaths_s": args.paths / serial_median / 1e6,
        "serial_fraction_amdahl_vs_p1": "", "seed": args.seed,
    })

    fixed_work = {}
    for threads in args.threads:
        samples = []
        latest = None
        for repeat in range(1, args.repeats + 1):
            latest = run([
                "./monte_carlo", "parallel", str(args.paths), args.seed,
            ], threads)
            elapsed = float(latest["time"])
            samples.append(elapsed)
            raw_rows.append({
                "category": "parallel", "paths": args.paths, "threads": threads,
                "repeat": repeat, "time": elapsed, "estimate": latest["estimate"],
                "standard_error": latest["standard_error"],
                "black_scholes": latest["black_scholes"], "checksum": latest["checksum"],
                "seed": args.seed,
                "OMP_NUM_THREADS": threads, "OMP_PROC_BIND": "close",
                "OMP_PLACES": "cores",
            })
        median = statistics.median(samples)
        speedup = serial_median / median
        fixed_work[threads] = median
        if threads > 1 and 1 in fixed_work:
            p1_speedup = fixed_work[1] / median
            serial_fraction = (
                (1 / p1_speedup - 1 / threads) / (1 - 1 / threads)
            )
        else:
            p1_speedup = 1.0
            serial_fraction = 0.0 if threads == 1 else ""
        summary_rows.append({
            "category": "parallel", "paths": args.paths, "threads": threads,
            "median_seconds": median, "speedup_vs_serial": speedup,
            "efficiency": speedup / threads,
            "estimate": latest["estimate"],
            "standard_error": latest["standard_error"],
            "throughput_mpaths_s": args.paths / median / 1e6,
            "serial_fraction_amdahl_vs_p1": serial_fraction,
            "seed": args.seed,
        })

    max_threads = max(args.threads)
    correlated = run([
        "./monte_carlo", "same-seed", str(args.paths), args.seed,
    ], max_threads)
    raw_rows.append({
        "category": "same_seed", "paths": args.paths, "threads": max_threads,
        "repeat": 1, "time": correlated["time"],
        "estimate": correlated["estimate"],
        "standard_error": correlated["standard_error"],
        "black_scholes": correlated["black_scholes"],
        "checksum": correlated["checksum"], "OMP_NUM_THREADS": max_threads,
        "seed": args.seed,
        "OMP_PROC_BIND": "close", "OMP_PLACES": "cores",
    })
    summary_rows.append({
        "category": "same_seed", "paths": args.paths, "threads": max_threads,
        "median_seconds": correlated["time"], "speedup_vs_serial": "",
        "efficiency": "", "estimate": correlated["estimate"],
        "standard_error": correlated["standard_error"],
        "throughput_mpaths_s": correlated["throughput_mpaths_s"],
        "serial_fraction_amdahl_vs_p1": "", "seed": args.seed,
    })

    fixed_time = {}
    for threads in sorted({1, max_threads}):
        samples = []
        sample_paths = []
        sample_times = []
        final = None
        for repeat in range(1, args.repeats + 1):
            final = run([
                "./monte_carlo", "fixed-time", str(args.fixed_seconds), args.seed,
            ], threads)
            samples.append(float(final["standard_error"]))
            sample_paths.append(int(final["samples"]))
            sample_times.append(float(final["time"]))
            raw_rows.append({
                "category": "fixed_time", "paths": final["samples"],
                "threads": threads, "repeat": repeat, "time": final["time"],
                "estimate": final["estimate"],
                "standard_error": final["standard_error"],
                "black_scholes": final["black_scholes"],
                "checksum": final["checksum"], "seed": args.seed,
                "OMP_NUM_THREADS": threads,
                "OMP_PROC_BIND": "close", "OMP_PLACES": "cores",
            })
        fixed_time[threads] = statistics.median(samples)
        median_paths = statistics.median(sample_paths)
        median_time = statistics.median(sample_times)
        summary_rows.append({
            "category": "fixed_time", "paths": int(median_paths),
            "threads": threads, "median_seconds": median_time,
            "speedup_vs_serial": "", "efficiency": "",
            "estimate": final["estimate"],
            "standard_error": fixed_time[threads],
            "throughput_mpaths_s": median_paths / median_time / 1e6,
            "serial_fraction_amdahl_vs_p1": "", "seed": args.seed,
        })

    prefix = ROOT / args.output
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, rows in (("_raw.csv", raw_rows), ("_summary.csv", summary_rows)):
        with prefix.with_name(prefix.name + suffix).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=rows[0].keys(), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {prefix}_raw.csv and {prefix}_summary.csv")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.stderr.write(error.stderr or str(error) + "\n")
        raise
