# Case Study 5: MPI Distributed Sample Sort

This case study implements distributed sample sort for 32-bit unsigned integer
keys. Each rank generates and locally sorts its shard, selects regular samples,
and participates in splitter selection, range partitioning, `MPI_Alltoall`
count exchange, and `MPI_Alltoallv` data redistribution. Each rank sorts its
received range, then the program verifies local order, global rank boundaries,
and total element count.

## Input distributions and duplicate handling

The default workload is N=100,000,000 keys with fixed seed `20261123`.
`uniform` draws keys from the full 32-bit range. `skewed` places 90% of keys
uniformly in [0,255] and 10% across the full range, producing many duplicates.
Each MPI rank has a deterministic seed derived from the base seed and rank.

Internally, each key is paired with its unique global position in a 64-bit
token. The upper 32 bits hold the key and the lower 32 bits break ties. This
preserves the sorted order of the original keys while allowing equal keys to
be split across ranks; without a tie-breaker, a popular duplicate value could
all be assigned to one rank. MPI therefore moves 8-byte tokens, while the
logical input remains 4-byte keys.

## Build, test, and run

```sh
make
make test

# Four ranks; oversampling factor 16 (48 regular samples per rank)
mpirun --oversubscribe -np 4 ./samplesort parallel \
  100000000 skewed 20261123 16

# Serial std::sort baseline (one process holds the full data set)
mpirun -np 1 ./samplesort serial \
  100000000 skewed 20261123 4
```

The serial baseline uses `std::sort` with the same `-O3 -std=c++17`
optimization flags. Its `SHARDS` argument generates the same fixed-seed
multiset as a parallel run with that many ranks; generation is outside the
timed sort. The parallel run stores only its local input shard plus temporary
communication buffers on each rank.

`make test` checks both distributions at 1, 2, and 4 ranks on a small fixture.
For each run, the executable checks that the received range is locally sorted,
that nonempty rank ranges are globally ordered, and that exactly N tokens were
preserved.

## Measurements

The final argument is the oversampling factor: each rank selects
`factor * (p-1)` regular samples. The benchmark runs three repetitions of both
distributions at fixed N for p=1, 2, and 4, then tests factors 1, 4, and 16 at
p=4. It records the median of each phase, final per-rank element counts,
max/mean load ratio, and raw repetitions:

```sh
python3 benchmark.py --n 100000000 --repeats 3 --ranks 1 2 4
```

The benchmark also performs an MPI ping-pong test on two ranks for messages
from zero bytes to 1 MiB. A least-squares fit of one-way time to
`alpha + beta * bytes` is applied to the measured per-rank `MPI_Alltoallv`
payload as a simple communication-cost prediction. Since collective exchange
can overlap peer transfers, this pairwise model is an estimate, not an exact
collective algorithm model.

Results are written under `results/`: raw and summary CSV files, ping-pong
measurements, the fitted communication model, and speedup and phase-breakdown
graphs.
The benchmark times sort and communication phases; input generation is
excluded. It does not write or generate a PDF.
