#!/usr/bin/env python3
"""Check Monte Carlo correctness, fixed-seed reproducibility, and convergence."""

import math
import os
import pathlib
import re
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
FIELDS = re.compile(r"(\w+)=([^\s]+)")


def run(mode, paths=None, threads=1, seed="20261123", seconds=None):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"
    if seconds is None:
        command = ["./monte_carlo", mode, str(paths), seed]
    else:
        command = ["./monte_carlo", "fixed-time", str(seconds), seed]
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )
    line = next(line for line in completed.stdout.splitlines()
                if line.startswith("RESULT "))
    return {key: value for key, value in FIELDS.findall(line)}


def main():
    if not (ROOT / "monte_carlo").exists():
        raise SystemExit("Build first with `make`")
    exact = float(subprocess.run(
        ["./monte_carlo", "black-scholes"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip().split("=")[1])

    reference = run("serial", 500_000, 1)
    reference_estimate = float(reference["estimate"])
    reference_error = float(reference["standard_error"])
    z_score = abs(reference_estimate - exact) / reference_error
    print(
        f"serial sample check: estimate={reference_estimate:.10f}, "
        f"SE={reference_error:.3e}, |z|={z_score:.3f}"
    )
    if z_score > 6.0:
        raise SystemExit("FAIL: serial Monte Carlo estimate is implausibly far from Black-Scholes")

    for threads in (1, 2, 4, 8):
        first = run("parallel", 200_003, threads)
        second = run("parallel", 200_003, threads)
        for field in ("estimate", "standard_error", "checksum", "samples"):
            if first[field] != second[field]:
                raise SystemExit(
                    f"FAIL: p={threads} is not reproducible for {field}: "
                    f"{first[field]} != {second[field]}"
                )
        print(
            f"p={threads}: deterministic estimate={first['estimate']}, "
            f"SE={first['standard_error']}"
        )

    effective_serial = run("serial", 100_000, 1)
    correlated = run("same-seed", 400_000, 4)
    effective_error = float(effective_serial["standard_error"])
    correlated_error = float(correlated["standard_error"])
    print(
        f"stream check p=4: 100k independent-path SE={effective_error:.3e}; "
        f"same-seed reported SE for 400k duplicated paths={correlated_error:.3e}"
    )
    if not math.isclose(
        float(effective_serial["estimate"]),
        float(correlated["estimate"]),
        rel_tol=1e-12,
    ):
        raise SystemExit("FAIL: same-seed workers did not repeat the same sample stream")
    if correlated_error >= effective_error * 0.6:
        raise SystemExit("FAIL: same-seed demonstration did not expose SE underestimation")
    print("PASS: fixed-worker reproducibility and statistical sanity checks.")


if __name__ == "__main__":
    main()
