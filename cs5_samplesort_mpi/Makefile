MPICXX = mpicxx
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -Wpedantic -DOMPI_SKIP_MPICXX=1

.PHONY: all clean test benchmark

all: samplesort

samplesort: samplesort.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

test: samplesort
	python3 check_correctness.py

benchmark: samplesort
	python3 benchmark.py --n 100000000 --repeats 3 --ranks 1 2 4

clean:
	rm -f samplesort
