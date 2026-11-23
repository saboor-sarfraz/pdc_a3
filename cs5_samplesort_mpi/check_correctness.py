#!/usr/bin/env python3
"""Exercise global ordering and count checks on both input distributions."""

import os
import pathlib
import re
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
RESULT = re.compile(r"RESULT (.*)")


def run(ranks, n, distribution, oversampling):
    env = os.environ.copy()
    command = [
        "mpirun", "--oversubscribe", "-np", str(ranks), "./samplesort",
        "parallel", str(n), distribution, "20261123", str(oversampling),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )
    match = RESULT.search(completed.stdout)
    if not match:
        raise RuntimeError(f"No RESULT line in MPI output: {completed.stdout}")
    result = dict(token.split("=", 1) for token in match.group(1).split())
    counts = [int(value) for value in result["rank_counts"].split(";")]
    if int(result["valid"]) != 1 or int(result["total_count"]) != n:
        raise RuntimeError(f"Global order/count validation failed: {result}")
    if len(counts) != ranks or sum(counts) != n:
        raise RuntimeError(f"Per-rank counts do not sum to N: {counts}")
    print(
        f"{distribution} p={ranks} oversampling={oversampling} "
        f"counts={counts} max/mean={float(result['max_mean_ratio']):.4f} PASS"
    )


def main():
    if not (ROOT / "samplesort").exists():
        raise SystemExit("Build first with `make`")
    n = 100_003
    for distribution in ("uniform", "skewed"):
        for ranks in (1, 2, 4):
            run(ranks, n, distribution, 16)
        run(4, n, distribution, 1)


if __name__ == "__main__":
    main()
