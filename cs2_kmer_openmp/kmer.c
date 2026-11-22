#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <omp.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum {
    MODE_SERIAL,
    MODE_RACE,
    MODE_ATOMIC,
    MODE_PRIVATE
} mode_t;

typedef struct {
    alignas(64) volatile uint64_t value;
} padded_counter_t;

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s serial|race|atomic|private LENGTH K [uniform|skewed] [HISTOGRAM.bin]\n"
            "       %s false-sharing ITERATIONS compact|padded\n",
            program, program);
}

static int parse_u64(const char *text, uint64_t minimum, uint64_t maximum,
                     uint64_t *value) {
    if (*text == '-') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || text == end || *end != '\0' || parsed < minimum || parsed > maximum) {
        return 0;
    }
    *value = (uint64_t)parsed;
    return 1;
}

static uint64_t next_random(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static unsigned char random_base(uint64_t *state, int skewed) {
    const uint64_t random = next_random(state);
    if (!skewed) return (unsigned char)(random & 3U);
    const unsigned int value = (unsigned int)(random >> 32) % 100;
    if (value < 70) return 0;
    if (value < 80) return 1;
    if (value < 90) return 2;
    return 3;
}

static void make_sequence(unsigned char *sequence, size_t length, int skewed) {
    uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
    for (size_t i = 0; i < length; ++i) {
        sequence[i] = random_base(&state, skewed);
    }
}

static void count_serial(const unsigned char *sequence, size_t length, int k,
                         uint64_t *histogram) {
    const size_t windows = length - (size_t)k + 1;
    const uint64_t mask = (UINT64_C(1) << (2 * k)) - 1;
    uint64_t index = 0;
    for (int j = 0; j < k; ++j) {
        index = (index << 2) | sequence[j];
    }
    for (size_t start = 0; start < windows; ++start) {
        ++histogram[index];
        if (start + 1 < windows) {
            index = ((index << 2) & mask) | sequence[start + (size_t)k];
        }
    }
}

static void count_chunk_plain(const unsigned char *sequence, size_t start,
                              size_t end, int k, uint64_t mask,
                              uint64_t *histogram) {
    if (start == end) return;
    uint64_t index = 0;
    for (int j = 0; j < k; ++j) {
        index = (index << 2) | sequence[start + (size_t)j];
    }
    for (size_t pos = start; pos < end; ++pos) {
        ++histogram[index];
        if (pos + 1 < end) {
            index = ((index << 2) & mask) | sequence[pos + (size_t)k];
        }
    }
}

static void count_chunk_atomic(const unsigned char *sequence, size_t start,
                               size_t end, int k, uint64_t mask,
                               uint64_t *histogram) {
    if (start == end) return;
    uint64_t index = 0;
    for (int j = 0; j < k; ++j) {
        index = (index << 2) | sequence[start + (size_t)j];
    }
    for (size_t pos = start; pos < end; ++pos) {
#pragma omp atomic update
        histogram[index] += 1;
        if (pos + 1 < end) {
            index = ((index << 2) & mask) | sequence[pos + (size_t)k];
        }
    }
}

static void count_parallel(const unsigned char *sequence, size_t length, int k,
                           uint64_t *histogram, mode_t mode,
                           uint64_t *private_histograms, size_t bins) {
    const size_t windows = length - (size_t)k + 1;
    const uint64_t mask = (UINT64_C(1) << (2 * k)) - 1;

    if (mode == MODE_PRIVATE) {
        const size_t row_bytes = bins * sizeof(*histogram);
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int team_size = omp_get_num_threads();
            uint64_t *local = private_histograms + (size_t)tid * bins;
            memset(local, 0, row_bytes);
            const size_t start = windows * (size_t)tid / (size_t)team_size;
            const size_t end = windows * (size_t)(tid + 1) / (size_t)team_size;
            count_chunk_plain(sequence, start, end, k, mask, local);

#pragma omp barrier
#pragma omp for schedule(static)
            for (size_t bin = 0; bin < bins; ++bin) {
                uint64_t total = 0;
                for (int source = 0; source < team_size; ++source) {
                    total += private_histograms[(size_t)source * bins + bin];
                }
                histogram[bin] = total;
            }
        }
        return;
    }

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int team_size = omp_get_num_threads();
        const size_t start = windows * (size_t)tid / (size_t)team_size;
        const size_t end = windows * (size_t)(tid + 1) / (size_t)team_size;
        if (mode == MODE_ATOMIC) {
            count_chunk_atomic(sequence, start, end, k, mask, histogram);
        } else {
            count_chunk_plain(sequence, start, end, k, mask, histogram);
        }
    }
}

static int write_histogram(const char *path, const uint64_t *histogram,
                           size_t bins) {
    FILE *file = fopen(path, "wb");
    if (!file) {
        perror(path);
        return 0;
    }
    const int write_ok = fwrite(histogram, sizeof(*histogram), bins, file) == bins;
    if (!write_ok) perror("writing histogram");
    const int close_ok = fclose(file) == 0;
    if (!close_ok) perror("closing histogram");
    return write_ok && close_ok;
}

static int run_false_sharing(uint64_t iterations, int padded) {
    const int threads = omp_get_max_threads();
    volatile uint64_t *compact = NULL;
    padded_counter_t *separated = NULL;
    if (padded) {
        separated = aligned_alloc(64, (size_t)threads * sizeof(*separated));
        if (!separated) {
            perror("allocating padded counters");
            return EXIT_FAILURE;
        }
        memset(separated, 0, (size_t)threads * sizeof(*separated));
    } else {
        compact = calloc((size_t)threads, sizeof(*compact));
        if (!compact) {
            perror("allocating compact counters");
            return EXIT_FAILURE;
        }
    }

    const double start_time = omp_get_wtime();
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        if (padded) {
            for (uint64_t i = 0; i < iterations; ++i) {
                separated[tid].value++;
            }
        } else {
            for (uint64_t i = 0; i < iterations; ++i) {
                compact[tid]++;
            }
        }
    }
    const double elapsed = omp_get_wtime() - start_time;

    uint64_t checksum = 0;
    for (int tid = 0; tid < threads; ++tid) {
        checksum += padded ? separated[tid].value : compact[tid];
    }
    printf("RESULT mode=false-sharing padding=%s threads=%d iterations_per_thread=%"
           PRIu64 " total_updates=%" PRIu64 " time=%.9f counters_bytes=%zu checksum=%"
           PRIu64 "\n",
           padded ? "padded" : "compact", threads, iterations,
           iterations * (uint64_t)threads, elapsed,
           padded ? (size_t)threads * sizeof(*separated)
                  : (size_t)threads * sizeof(*compact),
           checksum);
    free((void *)compact);
    free(separated);
    return checksum == iterations * (uint64_t)threads ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char **argv) {
    omp_set_dynamic(0);
    if (argc == 4 && strcmp(argv[1], "false-sharing") == 0) {
        uint64_t iterations;
        if (!parse_u64(argv[2], 1, UINT64_MAX, &iterations) ||
            (strcmp(argv[3], "compact") != 0 && strcmp(argv[3], "padded") != 0)) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        return run_false_sharing(iterations, strcmp(argv[3], "padded") == 0);
    }
    if (argc < 4 || argc > 6) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    mode_t mode;
    if (strcmp(argv[1], "serial") == 0) mode = MODE_SERIAL;
    else if (strcmp(argv[1], "race") == 0) mode = MODE_RACE;
    else if (strcmp(argv[1], "atomic") == 0) mode = MODE_ATOMIC;
    else if (strcmp(argv[1], "private") == 0) mode = MODE_PRIVATE;
    else {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    uint64_t parsed_length, parsed_k;
    if (!parse_u64(argv[2], 3, SIZE_MAX, &parsed_length) ||
        !parse_u64(argv[3], 1, 12, &parsed_k) ||
        parsed_length < parsed_k) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    const int k = (int)parsed_k;
    int skewed = 0;
    const char *output_path = NULL;
    if (argc >= 5) {
        if (strcmp(argv[4], "uniform") == 0 || strcmp(argv[4], "skewed") == 0) {
            skewed = strcmp(argv[4], "skewed") == 0;
            if (argc == 6) output_path = argv[5];
        } else {
            if (argc == 5) output_path = argv[4];
            else {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        }
    }

    const size_t length = (size_t)parsed_length;
    const size_t bins = (size_t)1 << (2 * k);
    const size_t windows = length - (size_t)k + 1;
    const int threads = mode == MODE_SERIAL ? 1 : omp_get_max_threads();
    if (bins > SIZE_MAX / sizeof(uint64_t) ||
        (mode == MODE_PRIVATE &&
         (size_t)threads > SIZE_MAX / (bins * sizeof(uint64_t)))) {
        fprintf(stderr, "Histogram allocation would overflow address space\n");
        return EXIT_FAILURE;
    }

    unsigned char *sequence = malloc(length);
    uint64_t *histogram = malloc(bins * sizeof(*histogram));
    const size_t private_cells =
        mode == MODE_PRIVATE ? bins * (size_t)threads : 0;
    uint64_t *private_histograms =
        mode == MODE_PRIVATE ? malloc(private_cells * sizeof(*private_histograms)) : NULL;
    if (!sequence || !histogram || (mode == MODE_PRIVATE && !private_histograms)) {
        fprintf(stderr, "Unable to allocate sequence or histogram buffers\n");
        free(sequence);
        free(histogram);
        free(private_histograms);
        return EXIT_FAILURE;
    }
    make_sequence(sequence, length, skewed);

    const double start_time = omp_get_wtime();
    memset(histogram, 0, bins * sizeof(*histogram));
    if (mode == MODE_SERIAL) {
        count_serial(sequence, length, k, histogram);
    } else {
        count_parallel(sequence, length, k, histogram, mode, private_histograms, bins);
    }
    const double elapsed = omp_get_wtime() - start_time;

    uint64_t total = 0;
    uint64_t checksum = 0;
    for (size_t bin = 0; bin < bins; ++bin) {
        total += histogram[bin];
        checksum += histogram[bin] * (uint64_t)(bin + 1);
    }
    const size_t scratch_bytes = private_cells * sizeof(*private_histograms);
    printf("RESULT mode=%s length=%zu k=%d bins=%zu threads=%d distribution=%s "
           "windows=%zu total=%" PRIu64 " lost=%" PRIu64 " time=%.9f "
           "throughput_mkmer_s=%.6f private_scratch_bytes=%zu "
           "checksum=%" PRIu64 "\n",
           argv[1], length, k, bins, threads, skewed ? "skewed" : "uniform",
           windows, total, (uint64_t)windows - total, elapsed,
           (double)windows / elapsed / 1e6, scratch_bytes, checksum);

    int ok = 1;
    if (output_path) ok = write_histogram(output_path, histogram, bins);
    free(sequence);
    free(histogram);
    free(private_histograms);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
