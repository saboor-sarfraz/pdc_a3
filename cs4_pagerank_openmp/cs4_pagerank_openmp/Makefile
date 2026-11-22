CC = gcc
CFLAGS = -O3 -std=c11 -Wall -Wextra -Wpedantic -fopenmp
LDFLAGS = -fopenmp -lm

.PHONY: all clean test benchmark

all: pagerank

pagerank: pagerank.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test: pagerank
	python3 check_correctness.py

benchmark: pagerank
	python3 benchmark.py --vertices 131072 --edges 1000000 --repeats 3 --threads 1 2 4 8

clean:
	rm -f pagerank *.bin
