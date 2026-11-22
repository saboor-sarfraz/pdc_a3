# Case Study 1: MPI Heat Diffusion

This directory contains a serial baseline and an MPI implementation of a
two-dimensional, fixed-boundary heat-diffusion stencil. The parallel program
supports a 1-D row decomposition and a 2-D Cartesian decomposition, each with
blocking (`MPI_Sendrecv`) and nonblocking (`MPI_Isend`/`MPI_Irecv`) halo
exchange.

## Model

For each interior cell, the program applies

```text
u_next[i,j] = u[i,j] + 0.2 * (u[i-1,j] + u[i+1,j]
                              + u[i,j-1] + u[i,j+1] - 4*u[i,j])
```

The outer boundary is held at zero. Interior initial values are deterministic
and in `[1.00, 1.99]`. Both programs use two grids and swap them after each
step. The convergence metric is the global maximum absolute change, reduced
every `K` steps.

## Build and correctness

```sh
make
make test
```

Both programs compile with `-O3 -std=c11` (same optimization flags). `make
test` compares every grid element from the serial solver with MPI outputs for
1, 2, and 4 ranks, both decompositions, and both exchange modes. The checker
uses an absolute tolerance of `1e-12`; output files are raw row-major doubles.

## Run

```sh
# Serial baseline (N=1024, T=500)
./heat_serial 1024 500

# MPI: 4 ranks, 2-D Cartesian decomposition, nonblocking exchange, K=50
OMP_NUM_THREADS=1 OMP_PROC_BIND=close \
  mpirun -np 4 ./heat_mpi 1024 500 50 2d nonblocking

# Optional final-grid binary output, e.g. for independent checking
mpirun -np 4 ./heat_mpi 129 40 10 2d nonblocking parallel.bin
./heat_serial 129 40 serial.bin
```

The MPI result prints the maximum-rank elapsed time measured between barriers;
the optional gather/write is outside the timed interval. The reported
`comm_post_or_block`, `comm_wait`, and `interior` values support the overlap
analysis. The MPI implementation does not use OpenMP; benchmark runs set
`OMP_NUM_THREADS=1` and `OMP_PROC_BIND=close`. Record the exact `mpirun -np`
value with each run.

## Repeated scaling measurements

Run at least three repeats; medians, raw samples, speedup, and efficiency are
written as CSV files under `results/`.

```sh
python3 benchmark.py --n 1024 --steps 500 --repeats 3 --ranks 1 2 4
python3 benchmark.py --n 1024 --steps 500 --repeats 3 --ranks 1 2 4 --full
```

The second command additionally records row/2-D and blocking/nonblocking
variants, plus `K=1` versus `K=50`. Weak-scaling grids use
`N(p) = round(1024 * sqrt(p))`, keeping approximately the same cells per rank.
The benchmark script accepts smaller sizes for quick development checks. Do
not claim CPU speedup beyond the four physical cores on the reference machine.

To compare outputs after building:

```sh
python3 check_correctness.py --n 129 --steps 40 --ranks 1 2 4
```

## Editable report

From this directory, generate the current report section after running the
full benchmark:

```sh
python3 generate_report.py
```

This creates `../Assignment3_Report.docx` and a standalone graph in
`results/`. The report remains editable and is intentionally not exported to
PDF.
