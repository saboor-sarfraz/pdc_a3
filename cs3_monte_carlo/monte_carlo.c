#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const double S0 = 100.0;
static const double STRIKE = 100.0;
static const double RATE = 0.05;
static const double VOLATILITY = 0.20;
static const double MATURITY = 1.0;
static const uint64_t DEFAULT_SEED = UINT64_C(20261123);
static const uint64_t FIXED_BATCH = UINT64_C(4096);

typedef struct {
    uint64_t state[4];
} rng_t;

typedef struct {
    uint64_t count;
    long double mean;
    long double m2;
    uint64_t checksum;
} statistics_t;

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s serial|parallel|same-seed PATHS [SEED]\n"
            "       %s fixed-time SECONDS [SEED]\n"
            "       %s black-scholes\n",
            program, program, program);
}

static int parse_u64(const char *text, uint64_t *value) {
    if (*text == '-') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || text == end || *end != '\0') return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static void rng_seed(rng_t *rng, uint64_t seed) {
    for (int i = 0; i < 4; ++i) {
        rng->state[i] = splitmix64_next(&seed);
    }
}

static uint64_t rotate_left(uint64_t value, int shift) {
    return (value << shift) | (value >> (64 - shift));
}

static uint64_t rng_next(rng_t *rng) {
    const uint64_t result = rotate_left(rng->state[1] * 5, 7) * 9;
    const uint64_t temporary = rng->state[1] << 17;
    rng->state[2] ^= rng->state[0];
    rng->state[3] ^= rng->state[1];
    rng->state[1] ^= rng->state[2];
    rng->state[0] ^= rng->state[3];
    rng->state[2] ^= temporary;
    rng->state[3] = rotate_left(rng->state[3], 45);
    return result;
}

static void split_paths(uint64_t paths, int tid, int team_size,
                        uint64_t *begin, uint64_t *end) {
    const uint64_t base = paths / (uint64_t)team_size;
    const uint64_t remainder = paths % (uint64_t)team_size;
    *begin = base * (uint64_t)tid +
             ((uint64_t)tid < remainder ? (uint64_t)tid : remainder);
    *end = *begin + base + ((uint64_t)tid < remainder ? 1U : 0U);
}

static double uniform_open01(rng_t *rng) {
    return ((double)(rng_next(rng) >> 11) + 0.5) * 0x1.0p-53;
}

static double standard_normal(rng_t *rng) {
    const double u1 = uniform_open01(rng);
    const double u2 = uniform_open01(rng);
    const double radius = sqrt(-2.0 * log(u1));
    const double angle = 6.283185307179586476925286766559 * u2;
    return radius * cos(angle);
}

static double payoff(rng_t *rng) {
    const double drift = (RATE - 0.5 * VOLATILITY * VOLATILITY) * MATURITY;
    const double diffusion = VOLATILITY * sqrt(MATURITY) * standard_normal(rng);
    const double terminal = S0 * exp(drift + diffusion);
    const double intrinsic = terminal - STRIKE;
    return intrinsic > 0.0 ? intrinsic : 0.0;
}

static void statistics_add(statistics_t *stats, double sample) {
    ++stats->count;
    const long double value = (long double)sample;
    const long double delta = value - stats->mean;
    stats->mean += delta / (long double)stats->count;
    const long double delta_after = value - stats->mean;
    stats->m2 += delta * delta_after;
    stats->checksum += (uint64_t)(sample * 1000000.0);
}

static void statistics_merge(statistics_t *total, const statistics_t *part) {
    if (part->count == 0) return;
    if (total->count == 0) {
        *total = *part;
        return;
    }
    const uint64_t combined_count = total->count + part->count;
    const long double delta = part->mean - total->mean;
    total->m2 += part->m2 +
        delta * delta * (long double)total->count * (long double)part->count /
        (long double)combined_count;
    total->mean += delta * (long double)part->count / (long double)combined_count;
    total->count = combined_count;
    total->checksum += part->checksum;
}

static double black_scholes_call(void) {
    const double root_t = sqrt(MATURITY);
    const double d1 = (log(S0 / STRIKE) +
                       (RATE + 0.5 * VOLATILITY * VOLATILITY) * MATURITY) /
                      (VOLATILITY * root_t);
    const double d2 = d1 - VOLATILITY * root_t;
    const double cdf_d1 = 0.5 * erfc(-d1 / sqrt(2.0));
    const double cdf_d2 = 0.5 * erfc(-d2 / sqrt(2.0));
    return S0 * cdf_d1 - STRIKE * exp(-RATE * MATURITY) * cdf_d2;
}

static void print_result(const char *mode, uint64_t requested, uint64_t seed,
                         int threads, const statistics_t *stats,
                         double elapsed, double target_seconds) {
    if (stats->count < 2) {
        fprintf(stderr, "At least two paths are required to estimate standard error\n");
        return;
    }
    const double discount = exp(-RATE * MATURITY);
    const double estimate = discount * (double)stats->mean;
    const long double sample_variance =
        stats->m2 / (long double)(stats->count - 1);
    const double standard_error =
        discount * sqrt((double)(sample_variance / (long double)stats->count));
    const double exact = black_scholes_call();
    const double z_score = (estimate - exact) / standard_error;
    const double throughput = elapsed > 0.0 ? (double)stats->count / elapsed / 1e6 : 0.0;
    printf("RESULT mode=%s paths=%" PRIu64 " samples=%" PRIu64
           " seed=%" PRIu64 " threads=%d estimate=%.15f standard_error=%.12g "
           "black_scholes=%.15f abs_error=%.12g z_score=%.6f "
           "time=%.9f throughput_mpaths_s=%.6f checksum=%" PRIu64,
           mode, requested, stats->count, seed, threads, estimate, standard_error,
           exact, fabs(estimate - exact), z_score, elapsed, throughput,
           stats->checksum);
    if (target_seconds > 0.0) {
        printf(" target_seconds=%.6f", target_seconds);
    }
    putchar('\n');
}

static int run_serial(uint64_t paths, uint64_t seed) {
    rng_t rng;
    rng_seed(&rng, seed);
    statistics_t stats = {0, 0.0L, 0.0L, 0};
    const double start_time = omp_get_wtime();
    for (uint64_t i = 0; i < paths; ++i) {
        statistics_add(&stats, payoff(&rng));
    }
    const double elapsed = omp_get_wtime() - start_time;
    print_result("serial", paths, seed, 1, &stats, elapsed, 0.0);
    return EXIT_SUCCESS;
}

static int run_parallel(uint64_t paths, uint64_t seed, int correlated) {
    const int threads = omp_get_max_threads();
    if ((uint64_t)threads > paths) {
        fprintf(stderr, "Path count must be at least the number of threads\n");
        return EXIT_FAILURE;
    }
    statistics_t *partials = calloc((size_t)threads, sizeof(*partials));
    if (!partials) {
        perror("allocating thread statistics");
        return EXIT_FAILURE;
    }

    const double start_time = omp_get_wtime();
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int team_size = omp_get_num_threads();
        uint64_t begin, end;
        split_paths(paths, tid, team_size, &begin, &end);
        rng_t rng;
        const uint64_t stream_seed =
            correlated ? seed : seed + (uint64_t)tid;
        rng_seed(&rng, stream_seed);
        for (uint64_t i = begin; i < end; ++i) {
            statistics_add(&partials[tid], payoff(&rng));
        }
    }
    statistics_t total = {0, 0.0L, 0.0L, 0};
    for (int tid = 0; tid < threads; ++tid) {
        statistics_merge(&total, &partials[tid]);
    }
    const double elapsed = omp_get_wtime() - start_time;
    print_result(correlated ? "same-seed" : "parallel", paths, seed,
                 threads, &total, elapsed, 0.0);
    free(partials);
    return EXIT_SUCCESS;
}

static int run_fixed_time(double seconds, uint64_t seed) {
    const int threads = omp_get_max_threads();
    statistics_t *partials = calloc((size_t)threads, sizeof(*partials));
    if (!partials) {
        perror("allocating thread statistics");
        return EXIT_FAILURE;
    }

    const double start_time = omp_get_wtime();
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        rng_t rng;
        rng_seed(&rng, seed + (uint64_t)tid);
        statistics_t *local = &partials[tid];
        while (omp_get_wtime() - start_time < seconds) {
            for (uint64_t i = 0; i < FIXED_BATCH; ++i) {
                statistics_add(local, payoff(&rng));
            }
        }
    }
    statistics_t total = {0, 0.0L, 0.0L, 0};
    for (int tid = 0; tid < threads; ++tid) {
        statistics_merge(&total, &partials[tid]);
    }
    const double elapsed = omp_get_wtime() - start_time;
    print_result("fixed-time", 0, seed, threads, &total, elapsed, seconds);
    free(partials);
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    omp_set_dynamic(0);
    if (argc == 2 && strcmp(argv[1], "black-scholes") == 0) {
        printf("Black-Scholes call price=%.15f\n", black_scholes_call());
        return EXIT_SUCCESS;
    }
    if (argc >= 2 && strcmp(argv[1], "fixed-time") == 0) {
        if (argc < 3 || argc > 4) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        char *end = NULL;
        errno = 0;
        const double seconds = strtod(argv[2], &end);
        if (errno || end == argv[2] || *end != '\0' ||
            !isfinite(seconds) || seconds <= 0.0 || seconds > 3600.0) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        uint64_t seed = DEFAULT_SEED;
        if (argc == 4 && !parse_u64(argv[3], &seed)) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        return run_fixed_time(seconds, seed);
    }
    if (argc < 3 || argc > 4) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const int serial = strcmp(argv[1], "serial") == 0;
    const int parallel = strcmp(argv[1], "parallel") == 0;
    const int same_seed = strcmp(argv[1], "same-seed") == 0;
    if (!serial && !parallel && !same_seed) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    uint64_t paths, seed = DEFAULT_SEED;
    if (!parse_u64(argv[2], &paths) || paths < 2 ||
        (argc == 4 && !parse_u64(argv[3], &seed))) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (serial) return run_serial(paths, seed);
    return run_parallel(paths, seed, same_seed);
}
