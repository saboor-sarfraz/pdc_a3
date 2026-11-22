#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const double DAMPING = 0.85;

typedef struct {
    uint32_t vertices;
    uint64_t edges;
    uint64_t *out_offsets;
    uint64_t *in_offsets;
    uint32_t *out_neighbors;
    uint32_t *in_neighbors;
    uint32_t *out_degree;
    uint64_t dangling_nodes;
    double generation_seconds;
} graph_t;

typedef enum {
    SCHEDULE_STATIC,
    SCHEDULE_DYNAMIC,
    SCHEDULE_GUIDED
} schedule_t;

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: %s serial V E SEED EPSILON MAX_ITERATIONS [RANKS.bin]\n"
            "       %s pull|push V E SEED EPSILON MAX_ITERATIONS "
            "static|dynamic|guided CHUNK [RANKS.bin]\n"
            "       %s degrees V E SEED OUTPUT.csv\n",
            program, program, program);
}

static int parse_u64(const char *text, uint64_t minimum, uint64_t maximum,
                     uint64_t *value) {
    if (*text == '-') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || text == end || *end != '\0' ||
        parsed < minimum || parsed > maximum) {
        return 0;
    }
    *value = (uint64_t)parsed;
    return 1;
}

static int parse_double(const char *text, double minimum, double maximum,
                        double *value) {
    char *end = NULL;
    errno = 0;
    const double parsed = strtod(text, &end);
    if (errno || text == end || *end != '\0' ||
        !isfinite(parsed) || parsed < minimum || parsed > maximum) {
        return 0;
    }
    *value = parsed;
    return 1;
}

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static void generate_edge(uint64_t *state, uint32_t bits,
                          uint32_t *source, uint32_t *destination) {
    uint32_t from = 0;
    uint32_t to = 0;
    for (uint32_t bit = 0; bit < bits; ++bit) {
        const unsigned int quadrant = (unsigned int)(splitmix64_next(state) % 10000);
        unsigned int source_bit, destination_bit;
        if (quadrant < 5700) {
            source_bit = 0;
            destination_bit = 0;
        } else if (quadrant < 7600) {
            source_bit = 0;
            destination_bit = 1;
        } else if (quadrant < 9500) {
            source_bit = 1;
            destination_bit = 0;
        } else {
            source_bit = 1;
            destination_bit = 1;
        }
        from |= source_bit << (bits - bit - 1);
        to |= destination_bit << (bits - bit - 1);
    }
    *source = from;
    *destination = to;
}

static int build_graph(graph_t *graph, uint32_t vertices, uint64_t edges,
                       uint64_t seed) {
    memset(graph, 0, sizeof(*graph));
    const double generation_start = omp_get_wtime();
    uint32_t bits = 0;
    for (uint32_t value = vertices; value > 1; value >>= 1) ++bits;
    if (((uint32_t)1 << bits) != vertices) {
        fprintf(stderr, "R-MAT vertex count must be a power of two\n");
        return 0;
    }

    uint32_t *sources = malloc((size_t)edges * sizeof(*sources));
    uint32_t *destinations = malloc((size_t)edges * sizeof(*destinations));
    graph->out_degree = calloc(vertices, sizeof(*graph->out_degree));
    uint64_t *in_degree = calloc(vertices, sizeof(*in_degree));
    graph->out_offsets = calloc((size_t)vertices + 1, sizeof(*graph->out_offsets));
    graph->in_offsets = calloc((size_t)vertices + 1, sizeof(*graph->in_offsets));
    if (!sources || !destinations || !graph->out_degree || !in_degree ||
        !graph->out_offsets || !graph->in_offsets) {
        fprintf(stderr, "Unable to allocate graph construction buffers\n");
        free(sources);
        free(destinations);
        free(in_degree);
        return 0;
    }

    uint64_t state = seed;
    for (uint64_t edge = 0; edge < edges; ++edge) {
        generate_edge(&state, bits, &sources[edge], &destinations[edge]);
        ++graph->out_degree[sources[edge]];
        ++in_degree[destinations[edge]];
    }
    for (uint32_t vertex = 0; vertex < vertices; ++vertex) {
        graph->out_offsets[vertex + 1] =
            graph->out_offsets[vertex] + graph->out_degree[vertex];
        graph->in_offsets[vertex + 1] =
            graph->in_offsets[vertex] + in_degree[vertex];
        if (graph->out_degree[vertex] == 0) ++graph->dangling_nodes;
    }

    graph->out_neighbors = malloc((size_t)edges * sizeof(*graph->out_neighbors));
    graph->in_neighbors = malloc((size_t)edges * sizeof(*graph->in_neighbors));
    uint64_t *out_cursor = malloc((size_t)vertices * sizeof(*out_cursor));
    uint64_t *in_cursor = malloc((size_t)vertices * sizeof(*in_cursor));
    if (!graph->out_neighbors || !graph->in_neighbors ||
        !out_cursor || !in_cursor) {
        fprintf(stderr, "Unable to allocate CSR adjacency arrays\n");
        free(sources);
        free(destinations);
        free(in_degree);
        free(out_cursor);
        free(in_cursor);
        return 0;
    }
    memcpy(out_cursor, graph->out_offsets, (size_t)vertices * sizeof(*out_cursor));
    memcpy(in_cursor, graph->in_offsets, (size_t)vertices * sizeof(*in_cursor));
    for (uint64_t edge = 0; edge < edges; ++edge) {
        const uint32_t source = sources[edge];
        const uint32_t destination = destinations[edge];
        graph->out_neighbors[out_cursor[source]++] = destination;
        graph->in_neighbors[in_cursor[destination]++] = source;
    }

    free(sources);
    free(destinations);
    free(in_degree);
    free(out_cursor);
    free(in_cursor);
    graph->vertices = vertices;
    graph->edges = edges;
    graph->generation_seconds = omp_get_wtime() - generation_start;
    return 1;
}

static void free_graph(graph_t *graph) {
    free(graph->out_offsets);
    free(graph->in_offsets);
    free(graph->out_neighbors);
    free(graph->in_neighbors);
    free(graph->out_degree);
    memset(graph, 0, sizeof(*graph));
}

static int parse_graph_args(int argc, char **argv, uint32_t *vertices,
                            uint64_t *edges, uint64_t *seed) {
    uint64_t parsed_vertices;
    if (!parse_u64(argv[2], 2, 1048576, &parsed_vertices) ||
        !parse_u64(argv[3], 1, 20000000, edges) ||
        !parse_u64(argv[4], 0, UINT64_MAX, seed)) {
        usage(argv[0]);
        return 0;
    }
    if ((parsed_vertices & (parsed_vertices - 1)) != 0) {
        fprintf(stderr, "V must be a power of two for this R-MAT generator\n");
        return 0;
    }
    *vertices = (uint32_t)parsed_vertices;
    (void)argc;
    return 1;
}

static int write_degree_distribution(const graph_t *graph, const char *path) {
    uint32_t max_degree = 0;
    for (uint32_t vertex = 0; vertex < graph->vertices; ++vertex) {
        const uint64_t in_degree =
            graph->in_offsets[vertex + 1] - graph->in_offsets[vertex];
        if (in_degree > max_degree) max_degree = (uint32_t)in_degree;
        if (graph->out_degree[vertex] > max_degree) {
            max_degree = graph->out_degree[vertex];
        }
    }
    uint64_t *in_counts = calloc((size_t)max_degree + 1, sizeof(*in_counts));
    uint64_t *out_counts = calloc((size_t)max_degree + 1, sizeof(*out_counts));
    if (!in_counts || !out_counts) {
        fprintf(stderr, "Unable to allocate degree histogram\n");
        free(in_counts);
        free(out_counts);
        return 0;
    }
    for (uint32_t vertex = 0; vertex < graph->vertices; ++vertex) {
        const uint32_t indegree = (uint32_t)(
            graph->in_offsets[vertex + 1] - graph->in_offsets[vertex]);
        ++in_counts[indegree];
        ++out_counts[graph->out_degree[vertex]];
    }

    FILE *file = fopen(path, "w");
    if (!file) {
        perror(path);
        free(in_counts);
        free(out_counts);
        return 0;
    }
    int ok = fprintf(file, "degree,in_vertices,out_vertices\n") > 0;
    for (uint32_t degree = 0; degree <= max_degree; ++degree) {
        if (in_counts[degree] || out_counts[degree]) {
            if (fprintf(file, "%u,%" PRIu64 ",%" PRIu64 "\n",
                        degree, in_counts[degree], out_counts[degree]) < 0) {
                ok = 0;
                break;
            }
        }
    }
    if (fclose(file) != 0) {
        perror("closing degree histogram");
        ok = 0;
    }
    free(in_counts);
    free(out_counts);
    return ok;
}

static void set_schedule(schedule_t schedule, int chunk) {
    omp_sched_t kind = omp_sched_static;
    if (schedule == SCHEDULE_DYNAMIC) kind = omp_sched_dynamic;
    if (schedule == SCHEDULE_GUIDED) kind = omp_sched_guided;
    omp_set_schedule(kind, chunk);
}

static const char *schedule_name(schedule_t schedule) {
    if (schedule == SCHEDULE_DYNAMIC) return "dynamic";
    if (schedule == SCHEDULE_GUIDED) return "guided";
    return "static";
}

static int parse_schedule(const char *text, schedule_t *schedule) {
    if (strcmp(text, "static") == 0) *schedule = SCHEDULE_STATIC;
    else if (strcmp(text, "dynamic") == 0) *schedule = SCHEDULE_DYNAMIC;
    else if (strcmp(text, "guided") == 0) *schedule = SCHEDULE_GUIDED;
    else return 0;
    return 1;
}

static int write_ranks(const char *path, const double *ranks, uint32_t vertices) {
    FILE *file = fopen(path, "wb");
    if (!file) {
        perror(path);
        return 0;
    }
    const int ok = fwrite(ranks, sizeof(*ranks), vertices, file) == vertices;
    if (!ok) perror("writing ranks");
    const int close_ok = fclose(file) == 0;
    if (!close_ok) perror("closing rank file");
    return ok && close_ok;
}

static void print_result(const char *mode, const graph_t *graph,
                         uint64_t seed, double epsilon, int iterations, int threads,
                         schedule_t schedule,
                         double residual, double elapsed,
                         const double *ranks, const double *thread_work,
                         int chunk) {
    double rank_sum = 0.0;
    for (uint32_t vertex = 0; vertex < graph->vertices; ++vertex) {
        rank_sum += ranks[vertex];
    }
    double max_work = 0.0;
    double total_work = 0.0;
    for (int tid = 0; tid < threads; ++tid) {
        if (thread_work[tid] > max_work) max_work = thread_work[tid];
        total_work += thread_work[tid];
    }
    const double mean_work = threads ? total_work / threads : 0.0;
    const double imbalance = mean_work > 0.0 ? max_work / mean_work : 1.0;
    const double edge_rate =
        elapsed > 0.0 ? (double)graph->edges * iterations / elapsed / 1e6 : 0.0;
    printf("RESULT mode=%s V=%u E=%" PRIu64 " seed=%" PRIu64
           " epsilon=%.3g iterations=%d threads=%d schedule=%s chunk=%d "
           "time=%.9f generation_time=%.9f dangling_nodes=%" PRIu64
           " residual_l1=%.12g rank_sum=%.15f imbalance_R=%.6f "
           "edges_per_iteration_mps=%.6f\n",
           mode, graph->vertices, graph->edges, seed,
           epsilon, iterations, threads,
           mode[0] == 's' ? "none" : schedule_name(schedule), chunk,
           elapsed, graph->generation_seconds, graph->dangling_nodes,
           residual, rank_sum, imbalance, edge_rate);
}

static int run_serial(const graph_t *graph, uint64_t seed,
                      double epsilon, int max_iterations,
                      double *result, int save_output, const char *output_path) {
    const uint32_t vertices = graph->vertices;
    double *current = malloc((size_t)vertices * sizeof(*current));
    double *next = malloc((size_t)vertices * sizeof(*next));
    double *work = calloc(1, sizeof(*work));
    if (!current || !next || !work) {
        fprintf(stderr, "Unable to allocate PageRank vectors\n");
        free(current);
        free(next);
        free(work);
        return EXIT_FAILURE;
    }
    for (uint32_t vertex = 0; vertex < vertices; ++vertex) {
        current[vertex] = 1.0 / vertices;
    }

    double residual = INFINITY;
    int iterations = 0;
    const double start_time = omp_get_wtime();
    while (iterations < max_iterations && residual >= epsilon) {
        double dangling_mass = 0.0;
        for (uint32_t vertex = 0; vertex < vertices; ++vertex) {
            if (graph->out_degree[vertex] == 0) dangling_mass += current[vertex];
        }
        const double base =
            (1.0 - DAMPING) / vertices + DAMPING * dangling_mass / vertices;
        residual = 0.0;
        for (uint32_t vertex = 0; vertex < vertices; ++vertex) {
            double incoming = 0.0;
            for (uint64_t edge = graph->in_offsets[vertex];
                 edge < graph->in_offsets[vertex + 1]; ++edge) {
                const uint32_t source = graph->in_neighbors[edge];
                incoming += current[source] / graph->out_degree[source];
            }
            next[vertex] = base + DAMPING * incoming;
            residual += fabs(next[vertex] - current[vertex]);
        }
        double *swap = current;
        current = next;
        next = swap;
        ++iterations;
    }
    const double elapsed = omp_get_wtime() - start_time;
    memcpy(result, current, (size_t)vertices * sizeof(*result));
    print_result("serial", graph, seed, epsilon, iterations, 1,
                 SCHEDULE_STATIC,
                 residual, elapsed, result, work, 0);
    int ok = 1;
    if (save_output) ok = write_ranks(output_path, result, vertices);
    free(current);
    free(next);
    free(work);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int run_parallel(const graph_t *graph, uint64_t seed,
                        double epsilon, int max_iterations,
                        int threads, schedule_t schedule, int chunk,
                        int push, double *result, int save_output,
                        const char *output_path) {
    const uint32_t vertices = graph->vertices;
    double *current = malloc((size_t)vertices * sizeof(*current));
    double *next = malloc((size_t)vertices * sizeof(*next));
    double *thread_work = calloc((size_t)threads, sizeof(*thread_work));
    if (!current || !next || !thread_work) {
        fprintf(stderr, "Unable to allocate PageRank vectors or thread timings\n");
        free(current);
        free(next);
        free(thread_work);
        return EXIT_FAILURE;
    }
    for (uint32_t vertex = 0; vertex < vertices; ++vertex) {
        current[vertex] = 1.0 / vertices;
    }
    set_schedule(schedule, chunk);

    double residual = INFINITY;
    double dangling_mass = 0.0;
    double delta_l1 = 0.0;
    int iterations = 0;
    const int chunk_value = chunk;
    const double start_time = omp_get_wtime();
#pragma omp parallel num_threads(threads) shared(current, next, residual, iterations, thread_work, dangling_mass, delta_l1)
    {
        const int tid = omp_get_thread_num();
        while (iterations < max_iterations && residual >= epsilon) {
#pragma omp single
            {
                dangling_mass = 0.0;
                delta_l1 = 0.0;
            }
            double loop_start = omp_get_wtime();
#pragma omp for reduction(+:dangling_mass) schedule(runtime) nowait
            for (uint32_t vertex = 0; vertex < vertices; ++vertex) {
                if (graph->out_degree[vertex] == 0) {
                    dangling_mass += current[vertex];
                }
            }
            thread_work[tid] += omp_get_wtime() - loop_start;
#pragma omp barrier
            const double base =
                (1.0 - DAMPING) / vertices + DAMPING * dangling_mass / vertices;

            if (push) {
#pragma omp for schedule(runtime) nowait
                for (uint32_t vertex = 0; vertex < vertices; ++vertex) {
                    next[vertex] = base;
                }
#pragma omp barrier
                loop_start = omp_get_wtime();
#pragma omp for schedule(runtime) nowait
                for (uint32_t source = 0; source < vertices; ++source) {
                    const uint32_t degree = graph->out_degree[source];
                    if (degree == 0) continue;
                    const double contribution = DAMPING * current[source] / degree;
                    for (uint64_t edge = graph->out_offsets[source];
                         edge < graph->out_offsets[source + 1]; ++edge) {
                        const uint32_t destination = graph->out_neighbors[edge];
#pragma omp atomic update
                        next[destination] += contribution;
                    }
                }
                thread_work[tid] += omp_get_wtime() - loop_start;
#pragma omp barrier
            } else {
                loop_start = omp_get_wtime();
#pragma omp for schedule(runtime) reduction(+:delta_l1) nowait
                for (uint32_t destination = 0; destination < vertices; ++destination) {
                    double incoming = 0.0;
                    for (uint64_t edge = graph->in_offsets[destination];
                         edge < graph->in_offsets[destination + 1]; ++edge) {
                        const uint32_t source = graph->in_neighbors[edge];
                        incoming += current[source] / graph->out_degree[source];
                    }
                    next[destination] = base + DAMPING * incoming;
                    delta_l1 += fabs(next[destination] - current[destination]);
                }
                thread_work[tid] += omp_get_wtime() - loop_start;
#pragma omp barrier
            }

            if (push) {
#pragma omp for reduction(+:delta_l1) schedule(runtime) nowait
                for (uint32_t vertex = 0; vertex < vertices; ++vertex) {
                    delta_l1 += fabs(next[vertex] - current[vertex]);
                }
#pragma omp barrier
            }
#pragma omp single
            {
                double *swap = current;
                current = next;
                next = swap;
                residual = delta_l1;
                ++iterations;
            }
        }
    }
    const double elapsed = omp_get_wtime() - start_time;
    memcpy(result, current, (size_t)vertices * sizeof(*result));
    print_result(push ? "push" : "pull", graph, seed, epsilon,
                 iterations, threads, schedule, residual, elapsed,
                 result, thread_work, chunk_value);
    int ok = 1;
    if (save_output) ok = write_ranks(output_path, result, vertices);
    free(current);
    free(next);
    free(thread_work);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char **argv) {
    omp_set_dynamic(0);
    if (argc < 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc >= 2 && strcmp(argv[1], "degrees") == 0) {
        if (argc != 6) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        uint32_t vertices;
        uint64_t edges, seed;
        if (!parse_graph_args(argc, argv, &vertices, &edges, &seed)) {
            return EXIT_FAILURE;
        }
        graph_t graph;
        if (!build_graph(&graph, vertices, edges, seed)) return EXIT_FAILURE;
        const int ok = write_degree_distribution(&graph, argv[5]);
        printf("GRAPH V=%u E=%" PRIu64 " seed=%" PRIu64
               " dangling_nodes=%" PRIu64 " generation_time=%.9f\n",
               vertices, edges, seed, graph.dangling_nodes,
               graph.generation_seconds);
        free_graph(&graph);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const int serial = strcmp(argv[1], "serial") == 0;
    const int pull = strcmp(argv[1], "pull") == 0;
    const int push = strcmp(argv[1], "push") == 0;
    if (!serial && !pull && !push) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    const int graph_argc = serial ? 7 : 9;
    if (argc != graph_argc && argc != graph_argc + 1) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    uint32_t vertices;
    uint64_t edges, seed;
    if (!parse_graph_args(argc, argv, &vertices, &edges, &seed)) {
        return EXIT_FAILURE;
    }
    double epsilon;
    uint64_t parsed_iterations;
    if (!parse_double(argv[5], 1e-15, 1.0, &epsilon) ||
        !parse_u64(argv[6], 1, 10000, &parsed_iterations)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    int threads = 1;
    schedule_t schedule = SCHEDULE_STATIC;
    int chunk = 0;
    if (!serial) {
        threads = omp_get_max_threads();
        if (!parse_schedule(argv[7], &schedule)) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        uint64_t parsed_chunk;
        if (!parse_u64(argv[8], 0, INT32_MAX, &parsed_chunk)) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        chunk = (int)parsed_chunk;
    }

    graph_t graph;
    if (!build_graph(&graph, vertices, edges, seed)) return EXIT_FAILURE;
    double *ranks = malloc((size_t)vertices * sizeof(*ranks));
    if (!ranks) {
        fprintf(stderr, "Unable to allocate final rank vector\n");
        free_graph(&graph);
        return EXIT_FAILURE;
    }
    const int save_output = argc == graph_argc + 1;
    const char *output_path = save_output ? argv[graph_argc] : NULL;
    const int result = serial
        ? run_serial(&graph, seed, epsilon, (int)parsed_iterations,
                     ranks, save_output, output_path)
        : run_parallel(&graph, seed, epsilon, (int)parsed_iterations,
                       threads, schedule, chunk, push,
                       ranks, save_output, output_path);
    free(ranks);
    free_graph(&graph);
    return result;
}
