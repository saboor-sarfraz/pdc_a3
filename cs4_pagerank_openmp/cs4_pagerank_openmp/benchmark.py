#!/usr/bin/env python3
"""Measure PageRank strong scaling, schedules, imbalance, and pull/push."""

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
        raise RuntimeError(f"Unable to parse PageRank result: {output}")
    return dict(token.split("=", 1) for token in match.group(1).split())


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


def run_solver(mode, vertices, edges, seed, epsilon, max_iterations,
               threads, schedule="static", chunk=0):
    command = [
        "./pagerank", mode, str(vertices), str(edges), str(seed),
        str(epsilon), str(max_iterations),
    ]
    if mode != "serial":
        command.extend((schedule, str(chunk)))
    return run(command, threads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertices", type=int, default=131072)
    parser.add_argument("--edges", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output", default="results/pagerank")
    args = parser.parse_args()
    if args.vertices < 2 or args.vertices & (args.vertices - 1):
        parser.error("vertices must be a power of two")
    if args.edges < 1_000_000:
        parser.error("use at least one million edges")
    if args.repeats < 3 or any(thread < 1 for thread in args.threads):
        parser.error("use at least three repeats and positive thread counts")
    if not (ROOT / "pagerank").exists():
        parser.error("Build first with `make`")

    results = ROOT / args.output
    results.parent.mkdir(parents=True, exist_ok=True)
    degree_path = ROOT / "results" / "degree_distribution.csv"
    degree_path.parent.mkdir(parents=True, exist_ok=True)
    degree_command = [
        "./pagerank", "degrees", str(args.vertices), str(args.edges),
        str(args.seed), str(degree_path),
    ]
    graph_env = os.environ.copy()
    graph_env["OMP_NUM_THREADS"] = "1"
    graph_env["OMP_PROC_BIND"] = "close"
    graph_env["OMP_PLACES"] = "cores"
    degree_result = subprocess.run(
        degree_command, cwd=ROOT, env=graph_env, text=True, capture_output=True,
        check=True,
    )
    print(degree_result.stdout.strip(), flush=True)

    raw_rows = []
    summary_rows = []
    serial_samples = []
    serial_iterations = []
    for repeat in range(1, args.repeats + 1):
        result = run_solver(
            "serial", args.vertices, args.edges, args.seed, args.epsilon,
            args.max_iterations, 1,
        )
        elapsed = float(result["time"])
        serial_samples.append(elapsed)
        serial_iterations.append(int(result["iterations"]))
        raw_rows.append({
            "category": "serial", "mode": "serial", "vertices": args.vertices,
            "edges": args.edges, "seed": args.seed, "threads": 1,
            "schedule": "none", "chunk": 0, "repeat": repeat,
            "time": elapsed, "iterations": result["iterations"],
            "residual_l1": result["residual_l1"], "rank_sum": result["rank_sum"],
            "imbalance_R": result["imbalance_R"],
            "edges_per_iteration_mps": result["edges_per_iteration_mps"],
            "OMP_NUM_THREADS": 1, "OMP_PROC_BIND": "close", "OMP_PLACES": "cores",
        })
    serial_median = statistics.median(serial_samples)
    median_iterations = statistics.median(serial_iterations)
    summary_rows.append({
        "category": "baseline", "mode": "serial", "vertices": args.vertices,
        "edges": args.edges, "threads": 1, "schedule": "none", "chunk": 0,
        "median_seconds": serial_median, "speedup_vs_serial": 1.0,
        "efficiency": 1.0, "iterations": int(median_iterations),
        "imbalance_R": 1.0, "edges_per_iteration_mps":
            args.edges * median_iterations / serial_median / 1e6,
        "serial_fraction_amdahl_vs_p1": "",
    })

    scaling_times = {}
    for threads in args.threads:
        for mode in ("pull", "push"):
            times = []
            imbalances = []
            iterations = []
            last_result = None
            for repeat in range(1, args.repeats + 1):
                last_result = run_solver(
                    mode, args.vertices, args.edges, args.seed, args.epsilon,
                    args.max_iterations, threads,
                )
                elapsed = float(last_result["time"])
                times.append(elapsed)
                imbalances.append(float(last_result["imbalance_R"]))
                iterations.append(int(last_result["iterations"]))
                raw_rows.append({
                    "category": "scaling", "mode": mode,
                    "vertices": args.vertices, "edges": args.edges,
                    "seed": args.seed, "threads": threads, "schedule": "static",
                    "chunk": 0, "repeat": repeat, "time": elapsed,
                    "iterations": last_result["iterations"],
                    "residual_l1": last_result["residual_l1"],
                    "rank_sum": last_result["rank_sum"],
                    "imbalance_R": last_result["imbalance_R"],
                    "edges_per_iteration_mps": last_result["edges_per_iteration_mps"],
                    "OMP_NUM_THREADS": threads,
                    "OMP_PROC_BIND": "close", "OMP_PLACES": "cores",
                })
            median = statistics.median(times)
            median_iterations = int(statistics.median(iterations))
            speedup = serial_median / median
            scaling_times[(mode, threads)] = median
            summary_rows.append({
                "category": "scaling", "mode": mode,
                "vertices": args.vertices, "edges": args.edges,
                "threads": threads, "schedule": "static", "chunk": 0,
                "median_seconds": median, "speedup_vs_serial": speedup,
                "efficiency": speedup / threads,
                "iterations": median_iterations,
                "imbalance_R": statistics.median(imbalances),
                "edges_per_iteration_mps":
                    args.edges * median_iterations / median / 1e6,
                "serial_fraction_amdahl_vs_p1": "",
            })

    comparison_threads = 4 if 4 in args.threads else max(args.threads)
    schedule_cases = [
        ("static", 0), ("static", 64),
        ("dynamic", 1), ("dynamic", 16), ("dynamic", 64),
        ("guided", 1), ("guided", 16), ("guided", 64),
    ]
    for schedule, chunk in schedule_cases:
        samples = []
        imbalances = []
        iterations = []
        latest = None
        for repeat in range(1, args.repeats + 1):
            latest = run_solver(
                "pull", args.vertices, args.edges, args.seed, args.epsilon,
                args.max_iterations, comparison_threads, schedule, chunk,
            )
            elapsed = float(latest["time"])
            samples.append(elapsed)
            imbalances.append(float(latest["imbalance_R"]))
            iterations.append(int(latest["iterations"]))
            raw_rows.append({
                "category": "schedule", "mode": "pull",
                "vertices": args.vertices, "edges": args.edges,
                "seed": args.seed, "threads": comparison_threads,
                "schedule": schedule, "chunk": chunk, "repeat": repeat,
                "time": elapsed, "iterations": latest["iterations"],
                "residual_l1": latest["residual_l1"], "rank_sum": latest["rank_sum"],
                "imbalance_R": latest["imbalance_R"],
                "edges_per_iteration_mps": latest["edges_per_iteration_mps"],
                "OMP_NUM_THREADS": comparison_threads,
                "OMP_PROC_BIND": "close", "OMP_PLACES": "cores",
            })
        median = statistics.median(samples)
        median_iterations = int(statistics.median(iterations))
        summary_rows.append({
            "category": "schedule", "mode": "pull",
            "vertices": args.vertices, "edges": args.edges,
            "threads": comparison_threads, "schedule": schedule, "chunk": chunk,
            "median_seconds": median,
            "speedup_vs_serial": serial_median / median,
            "efficiency": serial_median / median / comparison_threads,
            "iterations": median_iterations,
            "imbalance_R": statistics.median(imbalances),
            "edges_per_iteration_mps":
                args.edges * median_iterations / median / 1e6,
            "serial_fraction_amdahl_vs_p1": "",
        })

    for mode, threads in (("pull", comparison_threads), ("push", comparison_threads)):
        elapsed = scaling_times[(mode, threads)]
        raw_rows.append({
            "category": "pull_push_reference", "mode": mode,
            "vertices": args.vertices, "edges": args.edges, "seed": args.seed,
            "threads": threads, "schedule": "static", "chunk": 0, "repeat": 0,
            "time": elapsed, "iterations": "", "residual_l1": "", "rank_sum": "",
            "imbalance_R": "", "edges_per_iteration_mps": "",
            "OMP_NUM_THREADS": threads, "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
        })

    prefix = ROOT / args.output
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
