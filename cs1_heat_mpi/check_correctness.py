#!/usr/bin/env python3
"""Compare serial and MPI output grids using a stated floating-point tolerance."""

import argparse
import os
import pathlib
import struct
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parent


def run(command, env=None):
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def read_grid(path, n):
    data = path.read_bytes()
    expected = n * n * 8
    if len(data) != expected:
        raise RuntimeError(f"{path}: expected {expected} bytes, got {len(data)}")
    return struct.unpack(f"={n * n}d", data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=129)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    if args.n < 3 or args.steps < 0 or args.tolerance < 0:
        parser.error("N must be >= 3, steps and tolerance must be non-negative")
    if not (ROOT / "heat_serial").exists() or not (ROOT / "heat_mpi").exists():
        parser.error("Build first with `make`")

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMP_PROC_BIND", "close")
    with tempfile.TemporaryDirectory(prefix="heat-check-") as directory:
        serial_path = pathlib.Path(directory) / "serial.bin"
        serial_message = run(
            ["./heat_serial", str(args.n), str(args.steps), str(serial_path)]
        )
        reference = read_grid(serial_path, args.n)
        print(f"serial: {serial_message}")

        for ranks in args.ranks:
            for layout in ("row", "2d"):
                for exchange in ("blocking", "nonblocking"):
                    parallel_path = pathlib.Path(directory) / (
                        f"mpi-{ranks}-{layout}-{exchange}.bin"
                    )
                    command = [
                        "mpirun", "--oversubscribe", "-np", str(ranks),
                        "./heat_mpi", str(args.n), str(args.steps), "10",
                        layout, exchange, str(parallel_path),
                    ]
                    message = run(command, env=env)
                    result = read_grid(parallel_path, args.n)
                    error = max(abs(a - b) for a, b in zip(reference, result))
                    print(f"{message} max_abs_error={error:.3e}")
                    if error > args.tolerance:
                        raise SystemExit(
                            f"FAIL: {ranks} ranks, {layout}/{exchange}, "
                            f"error {error:.3e} exceeds {args.tolerance:.3e}"
                        )
    print(f"PASS: all grids agree within absolute tolerance {args.tolerance:.1e}")


if __name__ == "__main__":
    main()
