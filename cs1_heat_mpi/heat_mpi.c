#include <errno.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *program) {
    fprintf(stderr,
            "Usage: mpirun -np P %s N STEPS K row|2d blocking|nonblocking [OUTPUT.bin]\n",
            program);
}

static int parse_positive(const char *text, int minimum, int *value) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno || *end != '\0' || parsed < minimum || parsed > 10000000) {
        return 0;
    }
    *value = (int)parsed;
    return 1;
}

static void split_range(int total, int parts, int coordinate, int *start, int *length) {
    const int base = total / parts;
    const int remainder = total % parts;
    *length = base + (coordinate < remainder ? 1 : 0);
    *start = coordinate * base + (coordinate < remainder ? coordinate : remainder);
}

static double initial_value(int global_i, int global_j, int n) {
    if (global_i == 0 || global_i == n - 1 || global_j == 0 || global_j == n - 1) {
        return 0.0;
    }
    return 1.0 + (double)((global_i * 17LL + global_j * 13LL) % 100) / 100.0;
}

static void update_region(const double *current, double *next, int stride,
                          int row_begin, int row_end, int col_begin, int col_end,
                          int global_row_start, int global_col_start, int n,
                          int check_change, double *local_max_change) {
    const double alpha = 0.2;
    for (int i = row_begin; i <= row_end; ++i) {
        const int global_i = global_row_start + i - 1;
        for (int j = col_begin; j <= col_end; ++j) {
            const int global_j = global_col_start + j - 1;
            const int index = i * stride + j;
            if (global_i == 0 || global_i == n - 1 ||
                global_j == 0 || global_j == n - 1) {
                next[index] = current[index];
            } else {
                const double value = current[index] + alpha *
                    (current[index - stride] + current[index + stride] +
                     current[index - 1] + current[index + 1] -
                     4.0 * current[index]);
                next[index] = value;
                if (check_change) {
                    const double delta = fabs(value - current[index]);
                    if (delta > *local_max_change) {
                        *local_max_change = delta;
                    }
                }
            }
        }
    }
}

static void exchange_blocking(double *grid, int rows, int cols, MPI_Comm cart,
                              int north, int south, int west, int east,
                              MPI_Datatype column) {
    MPI_Sendrecv(&grid[1 * (cols + 2) + 1], cols, MPI_DOUBLE, north, 10,
                 &grid[(rows + 1) * (cols + 2) + 1], cols, MPI_DOUBLE, south, 10,
                 cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&grid[rows * (cols + 2) + 1], cols, MPI_DOUBLE, south, 11,
                 &grid[0 * (cols + 2) + 1], cols, MPI_DOUBLE, north, 11,
                 cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&grid[1 * (cols + 2) + 1], 1, column, west, 12,
                 &grid[1 * (cols + 2) + cols + 1], 1, column, east, 12,
                 cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&grid[1 * (cols + 2) + cols], 1, column, east, 13,
                 &grid[1 * (cols + 2)], 1, column, west, 13,
                 cart, MPI_STATUS_IGNORE);
}

static void exchange_nonblocking(double *grid, int rows, int cols, MPI_Comm cart,
                                 int north, int south, int west, int east,
                                 MPI_Datatype column, MPI_Request requests[8]) {
    const int stride = cols + 2;
    MPI_Irecv(&grid[0 * stride + 1], cols, MPI_DOUBLE, north, 11, cart, &requests[0]);
    MPI_Irecv(&grid[(rows + 1) * stride + 1], cols, MPI_DOUBLE, south, 10, cart, &requests[1]);
    MPI_Irecv(&grid[1 * stride], 1, column, west, 13, cart, &requests[2]);
    MPI_Irecv(&grid[1 * stride + cols + 1], 1, column, east, 12, cart, &requests[3]);
    MPI_Isend(&grid[1 * stride + 1], cols, MPI_DOUBLE, north, 10, cart, &requests[4]);
    MPI_Isend(&grid[rows * stride + 1], cols, MPI_DOUBLE, south, 11, cart, &requests[5]);
    MPI_Isend(&grid[1 * stride + 1], 1, column, west, 12, cart, &requests[6]);
    MPI_Isend(&grid[1 * stride + cols], 1, column, east, 13, cart, &requests[7]);
}

static int write_gathered_grid(const double *packed, const int *counts,
                               const int *displacements, int n, int dims[2],
                               int rank_count, MPI_Comm cart, const char *path) {
    const size_t cells = (size_t)n * (size_t)n;
    double *full = malloc(cells * sizeof(*full));
    if (!full) {
        fprintf(stderr, "Unable to allocate gathered grid\n");
        return 0;
    }
    for (size_t i = 0; i < cells; ++i) {
        full[i] = 0.0;
    }
    for (int rank = 0; rank < rank_count; ++rank) {
        int coordinates[2];
        int row_start, rows, col_start, cols;
        MPI_Cart_coords(cart, rank, 2, coordinates);
        split_range(n, dims[0], coordinates[0], &row_start, &rows);
        split_range(n, dims[1], coordinates[1], &col_start, &cols);
        int offset = displacements[rank];
        for (int i = 0; i < rows; ++i) {
            memcpy(&full[(size_t)(row_start + i) * (size_t)n + (size_t)col_start],
                   &packed[offset + i * cols], (size_t)cols * sizeof(*full));
        }
        (void)counts;
    }

    FILE *output = fopen(path, "wb");
    if (!output) {
        perror(path);
        free(full);
        return 0;
    }
    const int ok = fwrite(full, sizeof(*full), cells, output) == cells;
    if (!ok) {
        perror("writing output");
    }
    const int close_status = fclose(output);
    if (close_status != 0) {
        perror("closing output");
    }
    free(full);
    return ok && close_status == 0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6 || argc > 7) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int n, steps, check_interval;
    int valid = parse_positive(argv[1], 3, &n) &&
                parse_positive(argv[2], 0, &steps) &&
                parse_positive(argv[3], 1, &check_interval) &&
                (strcmp(argv[4], "row") == 0 || strcmp(argv[4], "2d") == 0) &&
                (strcmp(argv[5], "blocking") == 0 ||
                 strcmp(argv[5], "nonblocking") == 0);
    valid = valid && n <= 10000;
    if (!valid) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    const int is_row = strcmp(argv[4], "row") == 0;
    const int is_blocking = strcmp(argv[5], "blocking") == 0;
    const int write_output = argc == 7;

    int dims[2] = {0, 0};
    if (is_row) {
        dims[0] = size;
        dims[1] = 1;
    } else {
        MPI_Dims_create(size, 2, dims);
    }
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);

    int coordinates[2], north, south, west, east;
    MPI_Cart_coords(cart, rank, 2, coordinates);
    MPI_Cart_shift(cart, 0, 1, &north, &south);
    MPI_Cart_shift(cart, 1, 1, &west, &east);
    int row_start, rows, col_start, cols;
    split_range(n, dims[0], coordinates[0], &row_start, &rows);
    split_range(n, dims[1], coordinates[1], &col_start, &cols);
    if (rows < 1 || cols < 1 || rows > 100000 || cols > 100000) {
        if (rank == 0) fprintf(stderr, "Invalid local grid dimensions\n");
        MPI_Comm_free(&cart);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int stride = cols + 2;
    const size_t local_cells = (size_t)(rows + 2) * (size_t)stride;
    double *current = calloc(local_cells, sizeof(*current));
    double *next = calloc(local_cells, sizeof(*next));
    if (!current || !next) {
        fprintf(stderr, "Rank %d cannot allocate local grid\n", rank);
        free(current);
        free(next);
        MPI_Abort(cart, EXIT_FAILURE);
    }
    for (int i = 1; i <= rows; ++i) {
        for (int j = 1; j <= cols; ++j) {
            const double value = initial_value(row_start + i - 1, col_start + j - 1, n);
            current[i * stride + j] = value;
            next[i * stride + j] = value;
        }
    }

    MPI_Datatype column;
    MPI_Type_vector(rows, 1, stride, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);

    MPI_Barrier(cart);
    const double start_time = MPI_Wtime();
    double last_max_change = 0.0;
    double communication_seconds = 0.0;
    double nonblocking_wait_seconds = 0.0;
    double interior_seconds = 0.0;
    for (int step = 1; step <= steps; ++step) {
        const int check_now = step % check_interval == 0;
        double local_max_change = 0.0;
        if (is_blocking) {
            const double exchange_start = MPI_Wtime();
            exchange_blocking(current, rows, cols, cart, north, south, west, east, column);
            communication_seconds += MPI_Wtime() - exchange_start;
            update_region(current, next, stride, 1, rows, 1, cols,
                          row_start, col_start, n, check_now, &local_max_change);
        } else {
            MPI_Request requests[8];
            const double post_start = MPI_Wtime();
            exchange_nonblocking(current, rows, cols, cart, north, south, west, east,
                                 column, requests);
            communication_seconds += MPI_Wtime() - post_start;
            const double interior_start = MPI_Wtime();
            update_region(current, next, stride, 2, rows - 1, 2, cols - 1,
                          row_start, col_start, n, check_now, &local_max_change);
            interior_seconds += MPI_Wtime() - interior_start;
            const double wait_start = MPI_Wtime();
            MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
            nonblocking_wait_seconds += MPI_Wtime() - wait_start;
            update_region(current, next, stride, 1, 1, 1, cols,
                          row_start, col_start, n, check_now, &local_max_change);
            if (rows > 1) {
                update_region(current, next, stride, rows, rows, 1, cols,
                              row_start, col_start, n, check_now, &local_max_change);
            }
            if (rows > 2 && cols > 0) {
                update_region(current, next, stride, 2, rows - 1, 1, 1,
                              row_start, col_start, n, check_now, &local_max_change);
                if (cols > 1) {
                    update_region(current, next, stride, 2, rows - 1, cols, cols,
                                  row_start, col_start, n, check_now, &local_max_change);
                }
            }
        }
        if (check_now) {
            MPI_Allreduce(&local_max_change, &last_max_change, 1, MPI_DOUBLE, MPI_MAX, cart);
        }
        double *swap = current;
        current = next;
        next = swap;
    }
    MPI_Barrier(cart);
    const double local_elapsed = MPI_Wtime() - start_time;
    double elapsed;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    if (rank == 0) {
        printf("RESULT N=%d steps=%d ranks=%d layout=%s exchange=%s K=%d "
               "time=%.9f max_change=%.17g dims=%dx%d comm_post_or_block=%.9f "
               "comm_wait=%.9f interior=%.9f\n",
               n, steps, size, is_row ? "row" : "2d",
               is_blocking ? "blocking" : "nonblocking", check_interval,
               elapsed, last_max_change, dims[0], dims[1], communication_seconds,
               nonblocking_wait_seconds, interior_seconds);
    }

    if (write_output) {
        int *counts = NULL;
        int *displacements = NULL;
        if (rank == 0) {
            counts = malloc((size_t)size * sizeof(*counts));
            displacements = malloc((size_t)size * sizeof(*displacements));
            if (!counts || !displacements) {
                fprintf(stderr, "Unable to allocate gather metadata\n");
                MPI_Abort(cart, EXIT_FAILURE);
            }
            int offset = 0;
            for (int process = 0; process < size; ++process) {
                int coords[2], rstart, rcount, cstart, ccount;
                MPI_Cart_coords(cart, process, 2, coords);
                split_range(n, dims[0], coords[0], &rstart, &rcount);
                split_range(n, dims[1], coords[1], &cstart, &ccount);
                (void)rstart;
                (void)cstart;
                counts[process] = rcount * ccount;
                displacements[process] = offset;
                offset += counts[process];
            }
        }
        const int send_count = rows * cols;
        double *packed = malloc((size_t)send_count * sizeof(*packed));
        if (!packed) {
            fprintf(stderr, "Rank %d cannot allocate gather buffer\n", rank);
            MPI_Abort(cart, EXIT_FAILURE);
        }
        for (int i = 0; i < rows; ++i) {
            memcpy(&packed[i * cols], &current[(i + 1) * stride + 1],
                   (size_t)cols * sizeof(*packed));
        }
        int total = n * n;
        double *all_packed = rank == 0 ? malloc((size_t)total * sizeof(*all_packed)) : NULL;
        if (rank == 0 && !all_packed) {
            fprintf(stderr, "Unable to allocate gather result\n");
            MPI_Abort(cart, EXIT_FAILURE);
        }
        MPI_Gatherv(packed, send_count, MPI_DOUBLE, all_packed, counts, displacements,
                    MPI_DOUBLE, 0, cart);
        if (rank == 0) {
            if (!write_gathered_grid(all_packed, counts, displacements, n, dims, size,
                                     cart, argv[6])) {
                MPI_Abort(cart, EXIT_FAILURE);
            }
            free(all_packed);
            free(counts);
            free(displacements);
        }
        free(packed);
    }

    MPI_Type_free(&column);
    free(current);
    free(next);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
