#!/usr/bin/env python3
"""Compare atomic and private histograms against serial, including chunk edges."""

import os
import pathlib
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parent


def run(mode, length, k, distribution, path, threads):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "close"
    result = subprocess.run(
        ["./kmer", mode, str(length), str(k), distribution, str(path)],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )
    print(result.stdout.strip())


def main():
    if not (ROOT / "kmer").exists():
        raise SystemExit("Build first with `make`")
    cases = [
        (97, 8, "uniform"), (4099, 8, "uniform"),
        (4099, 12, "uniform"), (100003, 8, "uniform"),
        (100003, 12, "uniform"), (100003, 8, "skewed"),
        (100003, 12, "skewed"),
    ]
    with tempfile.TemporaryDirectory(prefix="kmer-check-") as directory:
        for length, k, distribution in cases:
            baseline = pathlib.Path(directory) / (
                f"serial-{length}-{k}-{distribution}.bin"
            )
            run("serial", length, k, distribution, baseline, 1)
            reference = baseline.read_bytes()
            for mode in ("atomic", "private"):
                for threads in (1, 2, 4):
                    output = pathlib.Path(directory) / (
                        f"{mode}-{length}-{k}-{distribution}-{threads}.bin"
                    )
                    run(mode, length, k, distribution, output, threads)
                    if output.read_bytes() != reference:
                        raise SystemExit(
                            f"FAIL: {mode} differs from serial for "
                            f"L={length}, k={k}, distribution={distribution}, "
                            f"threads={threads}"
                        )
    print("PASS: serial, atomic, and private histograms match exactly.")


if __name__ == "__main__":
    main()
