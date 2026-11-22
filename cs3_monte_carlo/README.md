# Case Study 3: OpenMP Monte Carlo Option Pricing

This case study estimates the value of a European call under geometric
Brownian motion:

```text
S_T = S_0 exp((r - sigma^2/2) T + sigma sqrt(T) Z),  Z ~ N(0,1)
Call = exp(-rT) E[max(S_T - K, 0)]
```

Chosen parameters are `S0=100`, `K=100`, `r=0.05`, `sigma=0.20`, and `T=1`.
The closed-form Black-Scholes call price is calculated by the program for
comparison. The fixed default RNG seed is `20261123`.

## Why OpenMP

Each Monte Carlo path is independent. OpenMP divides paths among workers and
combines per-worker online statistics, avoiding MPI process setup and
communication for this shared-memory host. The machine has four physical
cores and eight logical CPUs; measurements include 1, 2, 4, and 8 threads and
identify the 8-thread point as hyper-threaded rather than an additional four
physical cores.

## Implementations and RNG

- `serial`: single-thread reference.
- `parallel`: per-worker xoshiro256** streams initialized from the specified
  master seed and a distinct worker seed. Partial means and variances are
  combined in worker-index order.
- `same-seed`: intentionally flawed demonstration; every worker starts from
  the identical stream, producing correlated path samples and an
  underestimated standard error if treated as independent.

For a fixed seed and thread count, each worker gets the same stream and fixed
contiguous sample assignment, so the estimate and standard error are exactly
reproducible. Changing the worker count changes stream-to-path assignment and
the order of floating-point combination; small changes in the result are
expected. Online Welford statistics and deterministic worker-order merging
reduce variance cancellation and make same-worker runs repeatable.

## Build and correctness

```sh
make
make test
```

The serial baseline and parallel implementation use the same
`-O3 -std=c11 -fopenmp` flags. The test checks a 500,000-path estimate against
Black-Scholes within six measured standard errors, verifies repeatability for
1/2/4/8 threads, and demonstrates the error reported by correlated streams.

## Run

```sh
./monte_carlo black-scholes
./monte_carlo serial 100000000 20261123

OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./monte_carlo parallel 100000000 20261123

# Intentionally correlated streams; demonstration only
OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./monte_carlo same-seed 100000000 20261123

# Run for a fixed duration to compare error at fixed time
OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./monte_carlo fixed-time 2.0 20261123
OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./monte_carlo fixed-time 2.0 20261123
```

Each result prints the price, standard error, Black-Scholes price, absolute
error, z-score, number of samples, runtime, throughput, and a checksum. The
parallel count and fixed-time commands use per-thread RNG streams; `rand()` is
not used.

## Repeated measurements

The benchmark runs the required fixed workload of 100 million paths with
three repeats for the serial baseline and 1/2/4/8 thread counts. It also
measures the same-seed failure and runs three fixed-time (default 2-second)
trials at one and eight threads:

```sh
python3 benchmark.py --paths 100000000 --repeats 3 --threads 1 2 4 8
```

Raw results and medians are written to `results/monte_carlo_raw.csv` and
`results/monte_carlo_summary.csv`. Speedup uses the serial baseline:
`S(p)=T_serial/T_p`, `E(p)=S(p)/p`. The benchmark also estimates the Amdahl
serial fraction from the one-thread and p-thread OpenMP medians.

The editable assignment report is refreshed from the project root using:

```sh
python3 cs1_heat_mpi/generate_report.py
```

No PDF is generated.
