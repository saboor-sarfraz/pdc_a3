#!/usr/bin/env python3
"""Compare serial, pull, and atomic push ranks on a graph with dangling nodes."""

import argparse
import os
import pathlib
import re
import struct
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parent
FIELDS = re.compile(r"(\w+)=([^\s]+)")


def run(mode, path, vertices, edges, epsilon, threads=1,
        schedule="static", chunk=0):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"
    common = [
        str(vertices), str(edges), "20261123", str(epsilon), "200",
    ]
    if mode == "serial":
        command = ["./pagerank", mode, *common, str(path)]
    else:
        command = [
            "./pagerank", mode, *common, schedule, str(chunk), str(path),
        ]
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )
    line = next(line for line in completed.stdout.splitlines()
                if line.startswith("RESULT "))
    return {key: value for key, value in FIELDS.findall(line)}


def read_ranks(path, count):
    data = path.read_bytes()
    if len(data) != count * 8:
        raise RuntimeError(f"Bad rank-vector size: {len(data)} bytes")
    return struct.unpack(f"={count}d", data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertices", type=int, default=8192)
    parser.add_argument("--edges", type=int, default=50000)
    parser.add_argument("--epsilon", type=float, default=1e-10)
    args = parser.parse_args()
    if not (ROOT / "pagerank").exists():
        raise SystemExit("Build first with `make`")
    vertices = args.vertices
    with tempfile.TemporaryDirectory(prefix="pagerank-check-") as directory:
        directory = pathlib.Path(directory)
        serial_path = directory / "serial.bin"
        serial = run("serial", serial_path, args.vertices, args.edges, args.epsilon)
        reference = read_ranks(serial_path, vertices)
        if int(serial["dangling_nodes"]) == 0:
            raise SystemExit("Test graph did not contain dangling vertices")

        cases = [
            ("pull", 1, "static", 0),
            ("pull", 2, "static", 0),
            ("pull", 4, "static", 0),
            ("pull", 8, "static", 0),
            ("pull", 4, "dynamic", 16),
            ("pull", 4, "guided", 16),
            ("push", 2, "static", 0),
            ("push", 4, "static", 0),
            ("push", 8, "static", 0),
        ]
        for mode, threads, schedule, chunk in cases:
            output_path = directory / f"{mode}-{threads}-{schedule}-{chunk}.bin"
            result = run(
                mode, output_path, args.vertices, args.edges, args.epsilon,
                threads, schedule, chunk,
            )
            actual = read_ranks(output_path, vertices)
            l1_error = sum(abs(a - b) for a, b in zip(reference, actual))
            print(
                f"{mode} p={threads} {schedule},{chunk}: iterations="
                f"{result['iterations']} rank_sum={result['rank_sum']} "
                f"L1_vs_serial={l1_error:.3e}"
            )
            if l1_error > 2e-8:
                raise SystemExit(
                    f"FAIL: {mode} differs from serial by L1={l1_error:.3e}"
                )
    print("PASS: serial, pull, and atomic push ranks agree within L1 tolerance 2e-8.")


if __name__ == "__main__":
    main()
