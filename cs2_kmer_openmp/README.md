# Case Study 2: OpenMP k-mer Counting

This case study counts every length-`k` substring in a generated DNA sequence
for `k=8` and `k=12`. Bases use two-bit encoding (`A=0, C=1, G=2, T=3`), so
each k-mer maps to one of `4^k` histogram bins. The default sequence is
uniformly distributed and deterministic (fixed seed); `skewed` mode is also
available with base frequencies A=70%, C/G/T=10% each.

## Implementations

- `serial`: single-thread reference.
- `race`: intentionally incorrect OpenMP shared-histogram increments without
  synchronization. This version has a data race and undefined behavior; it is
  included only to demonstrate the failure.
- `atomic`: shared histogram with `#pragma omp atomic update`.
- `private`: one histogram per thread followed by a merge.
- `false-sharing`: isolated, race-free per-thread counters in compact adjacent
  storage versus cache-line-padded storage.

Every implementation partitions the *window start positions* into contiguous
disjoint ranges. A thread assigned starts `[a,b)` reads the full sequence window
at each of those positions, including up to `k-1` bases beyond its final start.
Thus, boundary-spanning k-mers are included once, by the thread that owns their
start index; no output windows are omitted or duplicated.

## Build and correctness

```sh
make
make test
```

Serial and parallel code use the same `-O3 -std=c11 -fopenmp` flags. The
correctness test compares all `uint64_t` histogram bins exactly for both
correct parallel implementations, multiple lengths, k values, and 1/2/4
threads. It includes sequence lengths that do not divide evenly among threads.

## Run

```sh
./kmer serial 100000000 8
OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./kmer atomic 100000000 8
OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./kmer private 100000000 12

# Intentional data race demonstration; output is not valid/correctness-safe.
OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./kmer race 100000000 8

# Explicit histogram file output for independent comparisons.
./kmer serial 100003 8 uniform serial.bin
OMP_NUM_THREADS=4 ./kmer private 100003 8 uniform private.bin

# Skewed-base input
OMP_NUM_THREADS=4 ./kmer atomic 100000000 8 skewed

# False-sharing microbenchmark
OMP_NUM_THREADS=4 ./kmer false-sharing 20000000 compact
OMP_NUM_THREADS=4 ./kmer false-sharing 20000000 padded
```

The generated sequence and allocations are prepared before timing. Timed
histogram work includes histogram zeroing, counting, and (for `private`) the
merge; sequence generation and scratch allocation are excluded. The false
sharing timer covers the counter update loop. Record `OMP_NUM_THREADS`,
`OMP_PROC_BIND`, and `OMP_PLACES` for each run.

## Repeated measurements

The full benchmark performs three repeats for serial baselines, atomic and
private implementations at the requested thread counts, plus one racy
demonstration and compact/padded false-sharing trials:

```sh
python3 benchmark.py --length 100000000 --repeats 3 --threads 1 2 4 --full
```

It saves raw runs and medians to `results/kmer_raw.csv` and
`results/kmer_summary.csv`. The strong-scaling problem size is fixed at
100,000,000 bases. Speedup and efficiency use the serial baseline:
`S(p)=T_serial/T_p`, `E(p)=S(p)/p`. This machine has four physical cores; the
report does not claim physical-core speedup beyond four threads.

The report generator in `../cs1_heat_mpi/generate_report.py` updates the shared
editable `../Assignment3_Report.docx` after the full benchmark is available.
No PDF is generated.

```sh
python3 ../cs1_heat_mpi/generate_report.py
```
