CC = mpicc
CFLAGS = -O3 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS = -lm

.PHONY: all clean test

all: heat_serial heat_mpi

heat_serial: heat_serial.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

heat_mpi: heat_mpi.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test: all
	python3 check_correctness.py --n 129 --steps 40

clean:
	rm -f heat_serial heat_mpi *.bin
