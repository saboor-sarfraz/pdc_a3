CC = gcc
CFLAGS = -O3 -std=c11 -Wall -Wextra -Wpedantic -fopenmp
LDFLAGS = -fopenmp -lm

.PHONY: all clean test benchmark

all: monte_carlo

monte_carlo: monte_carlo.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test: monte_carlo
	python3 check_correctness.py

benchmark: monte_carlo
	python3 benchmark.py --paths 100000000 --repeats 3 --threads 1 2 4 8

clean:
	rm -f monte_carlo
