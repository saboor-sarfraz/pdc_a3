#!/usr/bin/env python3
"""Run repeated MPI scaling/variant experiments and write raw plus median CSVs."""

import argparse
import csv
import os
import pathlib
import re
import statistics
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent
RESULT_TIME = re.compile(r"\btime=([0-9.]+)")
SERIAL_TIME = re.compile(r"\btime=([0-9.]+)")
RESULT_FIELDS = {
    name: re.compile(rf"\b{name}=([0-9.]+)")
    for name in ("comm_post_or_block", "comm_wait", "interior")
}


def run_once(ranks, n, steps, interval, layout, exchange, env):
    command = [
        "mpirun", "--oversubscribe", "-np", str(ranks),
        "./heat_mpi", str(n), str(steps), str(interval), layout, exchange,
    ]
    result = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, check=True
    )
    output = result.stdout.strip()
    match = RESULT_TIME.search(output)
    if not match:
        raise RuntimeError(f"Could not parse MPI timing from: {output}")
    print(output, flush=True)
    values = {"seconds": float(match.group(1))}
    for name, pattern in RESULT_FIELDS.items():
        field = pattern.search(output)
        values[name] = float(field.group(1)) if field else 0.0
    return values


def serial_once(n, steps):
    result = subprocess.run(
        ["./heat_serial", str(n), str(steps)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    match = SERIAL_TIME.search(result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse serial timing from: {result.stdout}")
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", default="results/heat")
    parser.add_argument(
        "--full", action="store_true",
        help="also measure both layouts/exchanges and K=1 vs K=50",
    )
    args = parser.parse_args()
    if args.n < 3 or args.steps < 1 or args.repeats < 3:
        parser.error("Use N >= 3, steps >= 1, and at least 3 repeats")
    if any(p < 1 for p in args.ranks):
        parser.error("rank counts must be positive")
    if not (ROOT / "heat_mpi").exists() or not (ROOT / "heat_serial").exists():
        parser.error("Build first with `make`")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_PROC_BIND"] = "close"
    configurations = []
    for ranks in args.ranks:
        configurations.append(("strong", args.n, ranks, 50, "2d", "nonblocking"))
    if args.full:
        for ranks in args.ranks:
            for layout in ("row", "2d"):
                for exchange in ("blocking", "nonblocking"):
                    configurations.append(
                        ("variant", args.n, ranks, 50, layout, exchange)
                    )
            for interval in (1, 50):
                configurations.append(
                    ("convergence", args.n, ranks, interval, "2d", "nonblocking")
                )
    for ranks in args.ranks:
        weak_n = max(3, round(args.n * (ranks ** 0.5)))
        configurations.append(("weak", weak_n, ranks, 50, "2d", "nonblocking"))

    raw_rows = []
    summary_rows = []
    serial_baselines = {}
    for category, n, ranks, interval, layout, exchange in configurations:
        if n not in serial_baselines:
            baseline_samples = [serial_once(n, args.steps) for _ in range(args.repeats)]
            serial_baselines[n] = statistics.median(baseline_samples)
            for repeat, elapsed in enumerate(baseline_samples, 1):
                raw_rows.append({
                    "category": "serial_baseline", "N": n, "steps": args.steps,
                    "ranks": 1, "K": 0, "layout": "serial", "exchange": "none",
                    "repeat": repeat, "seconds": f"{elapsed:.9f}",
                    "comm_post_or_block": "", "comm_wait": "", "interior": "",
                    "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
                    "OMP_PROC_BIND": env["OMP_PROC_BIND"],
                })
        samples = []
        communication_samples = []
        wait_samples = []
        interior_samples = []
        for repeat in range(1, args.repeats + 1):
            result = run_once(ranks, n, args.steps, interval, layout, exchange, env)
            samples.append(result["seconds"])
            communication_samples.append(result["comm_post_or_block"])
            wait_samples.append(result["comm_wait"])
            interior_samples.append(result["interior"])
            raw_rows.append({
                "category": category, "N": n, "steps": args.steps, "ranks": ranks,
                "K": interval, "layout": layout, "exchange": exchange,
                "repeat": repeat, "seconds": f"{result['seconds']:.9f}",
                "comm_post_or_block": f"{result['comm_post_or_block']:.9f}",
                "comm_wait": f"{result['comm_wait']:.9f}",
                "interior": f"{result['interior']:.9f}",
                "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
                "OMP_PROC_BIND": env["OMP_PROC_BIND"],
            })
        median = statistics.median(samples)
        speedup = serial_baselines[n] / median
        summary_rows.append({
            "category": category, "N": n, "steps": args.steps, "ranks": ranks,
            "K": interval, "layout": layout, "exchange": exchange,
            "median_seconds": f"{median:.9f}",
            "serial_baseline_seconds": f"{serial_baselines[n]:.9f}",
            "speedup_vs_serial": f"{speedup:.6f}",
            "efficiency": f"{speedup / ranks:.6f}",
            "median_comm_post_or_block_seconds":
                f"{statistics.median(communication_samples):.9f}",
            "median_comm_wait_seconds": f"{statistics.median(wait_samples):.9f}",
            "median_interior_seconds": f"{statistics.median(interior_samples):.9f}",
            "samples_seconds": ";".join(f"{sample:.9f}" for sample in samples),
        })

    prefix = ROOT / args.output
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_name(prefix.name + "_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_rows[0].keys())
        writer.writeheader()
        writer.writerows(raw_rows)
    with prefix.with_name(prefix.name + "_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {prefix.name}_raw.csv and {prefix.name}_summary.csv")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.stderr.write(error.stderr or str(error) + "\n")
        raise
