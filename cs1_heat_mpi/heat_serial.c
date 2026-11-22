#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void usage(const char *program) {
    fprintf(stderr, "Usage: %s N STEPS [OUTPUT.bin]\n", program);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    char *end = NULL;
    errno = 0;
    long parsed_n = strtol(argv[1], &end, 10);
    if (errno || *end != '\0' || parsed_n < 3 || parsed_n > 100000) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    const size_t n = (size_t)parsed_n;
    errno = 0;
    long parsed_steps = strtol(argv[2], &end, 10);
    if (errno || *end != '\0' || parsed_steps < 0 || parsed_steps > 10000000) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    const size_t steps = (size_t)parsed_steps;
    if (n > SIZE_MAX / n || n * n > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "Grid is too large\n");
        return EXIT_FAILURE;
    }

    const size_t cells = n * n;
    double *current = malloc(cells * sizeof(*current));
    double *next = malloc(cells * sizeof(*next));
    if (!current || !next) {
        fprintf(stderr, "Unable to allocate grid (%zu x %zu)\n", n, n);
        free(current);
        free(next);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double value = 0.0;
            if (i > 0 && i + 1 < n && j > 0 && j + 1 < n) {
                value = 1.0 + (double)((i * 17 + j * 13) % 100) / 100.0;
            }
            current[i * n + j] = value;
            next[i * n + j] = value;
        }
    }

    const double alpha = 0.2;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (size_t step = 0; step < steps; ++step) {
        for (size_t i = 1; i + 1 < n; ++i) {
            for (size_t j = 1; j + 1 < n; ++j) {
                const size_t index = i * n + j;
                next[index] = current[index] + alpha *
                    (current[index - n] + current[index + n] +
                     current[index - 1] + current[index + 1] -
                     4.0 * current[index]);
            }
        }
        double *swap = current;
        current = next;
        next = swap;
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    const double elapsed = (double)(finish.tv_sec - start.tv_sec) +
                           (double)(finish.tv_nsec - start.tv_nsec) / 1e9;

    double checksum = 0.0;
    for (size_t i = 0; i < cells; ++i) {
        checksum += current[i];
    }
    printf("RESULT N=%zu steps=%zu time=%.9f checksum=%.17g\n",
           n, steps, elapsed, checksum);

    if (argc == 4) {
        FILE *output = fopen(argv[3], "wb");
        if (!output) {
            perror(argv[3]);
            free(current);
            free(next);
            return EXIT_FAILURE;
        }
        if (fwrite(current, sizeof(*current), cells, output) != cells) {
            perror("writing output");
            fclose(output);
            free(current);
            free(next);
            return EXIT_FAILURE;
        }
        if (fclose(output) != 0) {
            perror("closing output");
            free(current);
            free(next);
            return EXIT_FAILURE;
        }
    }

    free(current);
    free(next);
    return EXIT_SUCCESS;
}
