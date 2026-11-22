# Case Study 4: OpenMP PageRank

This case study computes PageRank on a deterministic synthetic R-MAT graph in
CSR form. It includes a serial implementation, a parallel pull implementation
with destination ownership, and a parallel push implementation with atomic
rank updates.

## Graph and model

The default graph has V=131,072 vertices and E=1,000,000 directed edges.
R-MAT edges are generated using quadrant probabilities 0.57/0.19/0.19/0.05
and seed 20261123. Self-loops and repeated edges are retained. The graph is
synthetic, so no external dataset citation is needed. CSR stores both outgoing
and incoming adjacency; all graph-generation and CSR-build work is outside the
timed PageRank region.

The iteration uses damping `d=0.85` and redistributes dangling-node rank
uniformly:

```text
r_next[v] = (1-d)/V + d*(dangling_mass/V
                         + sum(r[u]/outdegree[u] for u -> v))
```

Iterations stop when the L1 rank change is below epsilon (default `1e-8`) or
the maximum iteration count is reached.

## Build, correctness, and run

```sh
make
make test

# Compare the full production-size graph against the serial rank vector
python3 check_correctness.py --vertices 131072 --edges 1000000 --epsilon 1e-8

# Serial baseline
./pagerank serial 131072 1000000 20261123 1e-8 100

# Parallel pull, static default chunks
OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./pagerank pull 131072 1000000 20261123 1e-8 100 static 0

# Parallel push with atomic destination updates
OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./pagerank push 131072 1000000 20261123 1e-8 100 static 0

# Schedule experiments
OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./pagerank pull 131072 1000000 20261123 1e-8 100 dynamic 16
OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./pagerank pull 131072 1000000 20261123 1e-8 100 guided 16
```

All variants compile with `-O3 -std=c11 -fopenmp`. Correctness checks compare
the full rank vector to the serial CSR result using L1 tolerance `2e-8`, with
graphs containing dangling vertices and several OpenMP schedules/thread
counts. Optional final rank output is a row-major binary vector of doubles.

## Measurements

The repeated benchmark uses at least one million edges and three runs per
configuration:

```sh
python3 benchmark.py --vertices 131072 --edges 1000000 --repeats 3 --threads 1 2 4 8
```

Raw and median records are stored in `results/pagerank_raw.csv` and
`results/pagerank_summary.csv`. The degree histogram is
`results/degree_distribution.csv`. The benchmark measures serial and parallel
strong scaling, compares pull and push, and tests static, dynamic, and guided
schedules with multiple chunks. It records `OMP_NUM_THREADS`,
`OMP_PROC_BIND=close`, and `OMP_PLACES=cores`. This host has four physical
cores and eight logical CPUs; eight-thread results are not described as
eight-physical-core speedup.

Pull scans incoming neighbors and writes each destination rank once; push
scans outgoing neighbors and performs an atomic update for every edge.
Reported thread imbalance `R` is the maximum accumulated work-loop time
divided by the mean across threads.

The report generator updates the editable shared DOCX from the project root:

```sh
python3 cs1_heat_mpi/generate_report.py
```

No PDF is generated.
