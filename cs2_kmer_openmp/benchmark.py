#!/usr/bin/env python3
"""Collect repeated scaling, race, and false-sharing measurements."""

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
    for part in match.group(1).split():
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def run(command, env):
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )
    output = completed.stdout.strip()
    print(output, flush=True)
    return parse_result(output)


def require_complete_count(fields, context):
    if int(fields["total"]) != int(fields["windows"]):
        raise RuntimeError(
            f"{context} counted {fields['total']} windows; expected {fields['windows']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=100_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", default="results/kmer")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--false-sharing-iterations", type=int, default=20_000_000)
    args = parser.parse_args()
    if args.length < 12 or args.repeats < 3 or any(t < 1 for t in args.threads):
        parser.error("length must be >= 12, repeats >= 3, and thread counts positive")
    if not (ROOT / "kmer").exists():
        parser.error("Build first with `make`")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"
    rows = []
    summary = []
    for k in (8, 12):
        serial_times = []
        for repeat in range(1, args.repeats + 1):
            fields = run(["./kmer", "serial", str(args.length), str(k)], env)
            require_complete_count(fields, f"serial k={k}")
            elapsed = float(fields["time"])
            serial_times.append(elapsed)
            rows.append({
                "category": "serial", "k": k, "threads": 1, "repeat": repeat,
                "N": args.length, "seconds": elapsed, "total": fields["total"],
                "expected": fields["windows"], "private_scratch_bytes": 0,
                "OMP_NUM_THREADS": 1, "OMP_PROC_BIND": env["OMP_PROC_BIND"],
                "OMP_PLACES": env["OMP_PLACES"],
            })
        serial_median = statistics.median(serial_times)
        summary.append({
            "category": "baseline", "N": args.length, "k": k, "threads": 1,
            "median_seconds": serial_median,
            "speedup_vs_serial": 1.0, "efficiency": 1.0,
            "throughput_mkmer_s":
                (args.length - k + 1) / serial_median / 1e6,
            "private_scratch_bytes": 0,
        })

        if args.full:
            race_threads = max(t for t in args.threads if t > 1) if any(
                t > 1 for t in args.threads
            ) else 1
            fields = run([
                "./kmer", "race", str(args.length), str(k)
            ], {**env, "OMP_NUM_THREADS": str(race_threads)})
            rows.append({
                "category": "race", "k": k, "threads": race_threads, "repeat": 1,
                "N": args.length, "seconds": float(fields["time"]), "total": fields["total"],
                "expected": fields["windows"], "private_scratch_bytes": 0,
                "OMP_NUM_THREADS": race_threads, "OMP_PROC_BIND": env["OMP_PROC_BIND"],
                "OMP_PLACES": env["OMP_PLACES"],
            })

        for threads in args.threads:
            for mode in ("atomic", "private"):
                samples = []
                scratch = 0
                for repeat in range(1, args.repeats + 1):
                    fields = run([
                        "./kmer", mode, str(args.length), str(k)
                    ], {**env, "OMP_NUM_THREADS": str(threads)})
                    require_complete_count(fields, f"{mode} k={k}, threads={threads}")
                    elapsed = float(fields["time"])
                    samples.append(elapsed)
                    scratch = int(fields["private_scratch_bytes"])
                    rows.append({
                        "category": mode, "N": args.length, "k": k, "threads": threads,
                        "repeat": repeat, "seconds": elapsed,
                        "total": fields["total"], "expected": fields["windows"],
                        "private_scratch_bytes": scratch,
                        "OMP_NUM_THREADS": threads,
                        "OMP_PROC_BIND": env["OMP_PROC_BIND"],
                        "OMP_PLACES": env["OMP_PLACES"],
                    })
                median = statistics.median(samples)
                speedup = serial_median / median
                summary.append({
                    "category": mode, "N": args.length, "k": k, "threads": threads,
                    "median_seconds": median, "speedup_vs_serial": speedup,
                    "efficiency": speedup / threads,
                    "throughput_mkmer_s":
                        (args.length - k + 1) / median / 1e6,
                    "private_scratch_bytes": scratch,
                })

    for threads in args.threads:
        for padding in ("compact", "padded"):
            times = []
            for repeat in range(1, args.repeats + 1):
                fields = run([
                    "./kmer", "false-sharing",
                    str(args.false_sharing_iterations), padding,
                ], {**env, "OMP_NUM_THREADS": str(threads)})
                elapsed = float(fields["time"])
                times.append(elapsed)
                rows.append({
                    "category": f"false_sharing_{padding}", "k": 0,
                    "N": args.false_sharing_iterations,
                    "threads": threads, "repeat": repeat, "seconds": elapsed,
                    "total": fields["checksum"], "expected": fields["total_updates"],
                    "private_scratch_bytes": fields["counters_bytes"],
                    "OMP_NUM_THREADS": threads,
                    "OMP_PROC_BIND": env["OMP_PROC_BIND"],
                    "OMP_PLACES": env["OMP_PLACES"],
                })
            summary.append({
                "category": f"false_sharing_{padding}",
                "N": args.false_sharing_iterations, "k": 0,
                "threads": threads, "median_seconds": statistics.median(times),
                "speedup_vs_serial": "", "efficiency": "",
                "throughput_mkmer_s": "", "private_scratch_bytes":
                    int(fields["counters_bytes"]),
            })

    prefix = ROOT / args.output
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, data in (("_raw.csv", rows), ("_summary.csv", summary)):
        with prefix.with_name(prefix.name + suffix).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    print(f"Wrote {prefix}_raw.csv and {prefix}_summary.csv")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.stderr.write(error.stderr or str(error) + "\n")
        raise
