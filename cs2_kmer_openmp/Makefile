CC = gcc
CFLAGS = -O3 -std=c11 -Wall -Wextra -Wpedantic -fopenmp
LDFLAGS = -fopenmp

.PHONY: all clean test benchmark

all: kmer

kmer: kmer.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test: kmer
	python3 check_correctness.py

benchmark: kmer
	python3 benchmark.py --length 100000000 --repeats 3 --threads 1 2 4 --full

clean:
	rm -f kmer *.bin
