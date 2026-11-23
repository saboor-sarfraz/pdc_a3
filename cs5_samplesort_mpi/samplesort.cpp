#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using token_t = std::uint64_t;

struct timings_t {
    double local_sort = 0.0;
    double sampling = 0.0;
    double partition = 0.0;
    double alltoall_counts = 0.0;
    double alltoallv = 0.0;
    double final_sort = 0.0;
};

std::uint64_t parse_u64(const char *text, const char *name) {
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (text == end || *end != '\0') {
        throw std::runtime_error(std::string("Invalid ") + name + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

bool valid_distribution(const std::string &distribution) {
    return distribution == "uniform" || distribution == "skewed";
}

std::uint64_t shard_offset(std::uint64_t n, int shards, int shard) {
    const std::uint64_t base = n / static_cast<std::uint64_t>(shards);
    const std::uint64_t remainder = n % static_cast<std::uint64_t>(shards);
    return static_cast<std::uint64_t>(shard) * base +
           std::min<std::uint64_t>(static_cast<std::uint64_t>(shard), remainder);
}

std::uint64_t shard_size(std::uint64_t n, int shards, int shard) {
    return n / static_cast<std::uint64_t>(shards) +
           (static_cast<std::uint64_t>(shard) < n % static_cast<std::uint64_t>(shards));
}

token_t make_token(std::uint32_t key, std::uint32_t global_id) {
    return (static_cast<token_t>(key) << 32) | global_id;
}

void generate_shard(std::vector<token_t> &values, std::uint64_t count,
                    std::uint64_t offset, std::uint64_t seed, int shard,
                    const std::string &distribution) {
    const std::uint64_t shard_seed =
        seed + static_cast<std::uint64_t>(shard) * UINT64_C(0x9e3779b97f4a7c15);
    std::mt19937_64 generator(shard_seed);
    values.resize(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        std::uint32_t key;
        if (distribution == "skewed" && generator() % 10 < 9) {
            key = static_cast<std::uint32_t>(generator() % 256);
        } else {
            key = static_cast<std::uint32_t>(generator());
        }
        values[static_cast<std::size_t>(i)] =
            make_token(key, static_cast<std::uint32_t>(offset + i));
    }
}

void require_mpi(bool condition, const char *message, int rank) {
    if (!condition) {
        if (rank == 0) std::cerr << message << '\n';
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

void run_serial(std::uint64_t n, const std::string &distribution,
                std::uint64_t seed, int shards, int rank, int size) {
    require_mpi(size == 1, "serial mode must run with one MPI process", rank);
    require_mpi(n <= std::numeric_limits<std::uint32_t>::max(),
                "N must fit in the 32-bit unique-id field", rank);
    require_mpi(shards > 0, "shards must be positive", rank);
    std::vector<token_t> values;
    values.reserve(static_cast<std::size_t>(n));
    for (int shard = 0; shard < shards; ++shard) {
        std::vector<token_t> part;
        generate_shard(part, shard_size(n, shards, shard),
                       shard_offset(n, shards, shard), seed, shard, distribution);
        values.insert(values.end(), part.begin(), part.end());
    }
    const double start = MPI_Wtime();
    std::sort(values.begin(), values.end());
    const double elapsed = MPI_Wtime() - start;
    const bool sorted = std::is_sorted(values.begin(), values.end());
    std::cout << std::setprecision(9)
              << "RESULT mode=serial distribution=" << distribution
              << " N=" << n << " ranks=1 shards=" << shards
              << " total=" << elapsed << " sorted=" << (sorted ? 1 : 0)
              << " data_bytes=" << n * sizeof(std::uint32_t)
              << " sort_bytes=" << values.size() * sizeof(token_t) << '\n';
    require_mpi(sorted, "serial std::sort validation failed", rank);
}

void run_parallel(std::uint64_t n, const std::string &distribution,
                  std::uint64_t seed, int oversampling, int rank, int size) {
    require_mpi(n <= std::numeric_limits<std::uint32_t>::max(),
                "N must fit in the 32-bit unique-id field", rank);
    require_mpi(n >= static_cast<std::uint64_t>(size),
                "N must be at least the MPI process count", rank);
    require_mpi(oversampling > 0, "oversampling must be positive", rank);
    const std::uint64_t local_n = shard_size(n, size, rank);
    const std::uint64_t offset = shard_offset(n, size, rank);

    std::vector<token_t> local;
    generate_shard(local, local_n, offset, seed, rank, distribution);
    timings_t local_times;
    std::vector<token_t> final_values;

    MPI_Barrier(MPI_COMM_WORLD);
    const double total_start = MPI_Wtime();
    double phase_start = MPI_Wtime();
    std::sort(local.begin(), local.end());
    local_times.local_sort = MPI_Wtime() - phase_start;

    std::vector<int> send_counts(size, 0), send_displacements(size, 0);
    std::vector<int> receive_counts(size, 0), receive_displacements(size, 0);
    std::vector<token_t> send_buffer;
    std::vector<token_t> receive_buffer;
    std::vector<token_t> splitters;

    if (size > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
        phase_start = MPI_Wtime();
        const std::uint64_t requested_samples =
            static_cast<std::uint64_t>(oversampling) * (size - 1);
        const int sample_count = static_cast<int>(
            std::min<std::uint64_t>(requested_samples, local_n));
        std::vector<token_t> local_samples;
        local_samples.reserve(static_cast<std::size_t>(sample_count));
        for (int i = 0; i < sample_count; ++i) {
            const std::uint64_t index =
                (static_cast<std::uint64_t>(i + 1) * local_n) /
                static_cast<std::uint64_t>(sample_count + 1);
            local_samples.push_back(local[static_cast<std::size_t>(index)]);
        }
        std::vector<int> sample_counts(size, 0), sample_displacements(size, 0);
        MPI_Gather(&sample_count, 1, MPI_INT, sample_counts.data(), 1, MPI_INT,
                   0, MPI_COMM_WORLD);
        int gathered_sample_count = 0;
        if (rank == 0) {
            for (int peer = 0; peer < size; ++peer) {
                sample_displacements[peer] = gathered_sample_count;
                gathered_sample_count += sample_counts[peer];
            }
        }
        std::vector<token_t> all_samples;
        if (rank == 0) {
            all_samples.resize(static_cast<std::size_t>(gathered_sample_count));
        }
        MPI_Gatherv(local_samples.data(), sample_count, MPI_UINT64_T,
                    all_samples.data(), sample_counts.data(),
                    sample_displacements.data(), MPI_UINT64_T, 0, MPI_COMM_WORLD);
        splitters.resize(static_cast<std::size_t>(size - 1));
        if (rank == 0) {
            std::sort(all_samples.begin(), all_samples.end());
            for (int i = 1; i < size; ++i) {
                const std::uint64_t sample_index =
                    static_cast<std::uint64_t>(i) * all_samples.size() /
                    static_cast<std::uint64_t>(size);
                splitters[static_cast<std::size_t>(i - 1)] =
                    all_samples[std::min<std::uint64_t>(
                        sample_index, all_samples.size() - 1)];
            }
        }
        MPI_Bcast(splitters.data(), size - 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        local_times.sampling = MPI_Wtime() - phase_start;

        MPI_Barrier(MPI_COMM_WORLD);
        phase_start = MPI_Wtime();
        for (token_t value : local) {
            const auto boundary =
                std::upper_bound(splitters.begin(), splitters.end(), value);
            ++send_counts[static_cast<std::size_t>(boundary - splitters.begin())];
        }
        for (int peer = 1; peer < size; ++peer) {
            send_displacements[peer] =
                send_displacements[peer - 1] + send_counts[peer - 1];
        }
        send_buffer.resize(local.size());
        std::vector<int> cursor = send_displacements;
        for (token_t value : local) {
            const auto boundary =
                std::upper_bound(splitters.begin(), splitters.end(), value);
            const auto destination =
                static_cast<std::size_t>(boundary - splitters.begin());
            send_buffer[static_cast<std::size_t>(cursor[destination]++)] = value;
        }
        local_times.partition = MPI_Wtime() - phase_start;

        MPI_Barrier(MPI_COMM_WORLD);
        phase_start = MPI_Wtime();
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, receive_counts.data(), 1,
                     MPI_INT, MPI_COMM_WORLD);
        local_times.alltoall_counts = MPI_Wtime() - phase_start;
        for (int peer = 1; peer < size; ++peer) {
            receive_displacements[peer] =
                receive_displacements[peer - 1] + receive_counts[peer - 1];
        }
        const int receive_total =
            receive_displacements[size - 1] + receive_counts[size - 1];
        receive_buffer.resize(static_cast<std::size_t>(receive_total));

        MPI_Barrier(MPI_COMM_WORLD);
        phase_start = MPI_Wtime();
        MPI_Alltoallv(send_buffer.data(), send_counts.data(),
                      send_displacements.data(), MPI_UINT64_T,
                      receive_buffer.data(), receive_counts.data(),
                      receive_displacements.data(), MPI_UINT64_T, MPI_COMM_WORLD);
        local_times.alltoallv = MPI_Wtime() - phase_start;
        std::vector<token_t>().swap(local);
        std::vector<token_t>().swap(send_buffer);

        MPI_Barrier(MPI_COMM_WORLD);
        phase_start = MPI_Wtime();
        std::sort(receive_buffer.begin(), receive_buffer.end());
        local_times.final_sort = MPI_Wtime() - phase_start;
        final_values = std::move(receive_buffer);
    } else {
        final_values = std::move(local);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double local_total = MPI_Wtime() - total_start;
    double total = 0.0;
    MPI_Reduce(&local_total, &total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const double local_phase_values[] = {
        local_times.local_sort, local_times.sampling, local_times.partition,
        local_times.alltoall_counts, local_times.alltoallv, local_times.final_sort,
    };
    double max_phase_values[6] = {};
    MPI_Reduce(local_phase_values, max_phase_values, 6, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    const bool local_sorted =
        std::is_sorted(final_values.begin(), final_values.end());
    const std::uint64_t local_count = final_values.size();
    const std::uint64_t local_min = final_values.empty()
        ? std::numeric_limits<std::uint64_t>::max() : final_values.front();
    const std::uint64_t local_max = final_values.empty()
        ? 0 : final_values.back();
    std::vector<std::uint64_t> counts(static_cast<std::size_t>(size));
    std::vector<std::uint64_t> minima(static_cast<std::size_t>(size));
    std::vector<std::uint64_t> maxima(static_cast<std::size_t>(size));
    MPI_Allgather(&local_count, 1, MPI_UINT64_T, counts.data(), 1, MPI_UINT64_T,
                  MPI_COMM_WORLD);
    MPI_Allgather(&local_min, 1, MPI_UINT64_T, minima.data(), 1, MPI_UINT64_T,
                  MPI_COMM_WORLD);
    MPI_Allgather(&local_max, 1, MPI_UINT64_T, maxima.data(), 1, MPI_UINT64_T,
                  MPI_COMM_WORLD);

    int local_valid = local_sorted ? 1 : 0;
    std::uint64_t previous_max = 0;
    bool have_previous = false;
    for (int peer = 0; peer < size; ++peer) {
        if (counts[static_cast<std::size_t>(peer)] == 0) continue;
        if (have_previous && previous_max > minima[static_cast<std::size_t>(peer)]) {
            local_valid = 0;
        }
        previous_max = maxima[static_cast<std::size_t>(peer)];
        have_previous = true;
    }
    int globally_valid = 0;
    MPI_Allreduce(&local_valid, &globally_valid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    const std::uint64_t global_count =
        std::accumulate(counts.begin(), counts.end(), UINT64_C(0));
    const std::uint64_t max_count =
        *std::max_element(counts.begin(), counts.end());
    const double mean_count = static_cast<double>(global_count) / size;
    const double imbalance = mean_count == 0.0 ? 0.0 : max_count / mean_count;
    const std::uint64_t local_bytes_sent =
        std::accumulate(send_counts.begin(), send_counts.end(), UINT64_C(0)) *
        sizeof(token_t);
    std::uint64_t max_bytes_sent = 0;
    MPI_Reduce(&local_bytes_sent, &max_bytes_sent, 1, MPI_UINT64_T, MPI_MAX, 0,
               MPI_COMM_WORLD);

    require_mpi(global_count == n && globally_valid,
                "Global order or element-count validation failed", rank);
    if (rank == 0) {
        std::cout << std::setprecision(9)
                  << "RESULT mode=parallel distribution=" << distribution
                  << " N=" << n << " ranks=" << size
                  << " oversampling=" << oversampling
                  << " local_sort=" << max_phase_values[0]
                  << " sampling=" << max_phase_values[1]
                  << " partition=" << max_phase_values[2]
                  << " alltoall_counts=" << max_phase_values[3]
                  << " alltoallv=" << max_phase_values[4]
                  << " final_sort=" << max_phase_values[5]
                  << " total=" << total << " max_mean_ratio=" << imbalance
                  << " max_count=" << max_count
                  << " max_bytes_sent=" << max_bytes_sent
                  << " total_count=" << global_count
                  << " valid=" << globally_valid << " rank_counts=";
        for (int peer = 0; peer < size; ++peer) {
            if (peer) std::cout << ';';
            std::cout << counts[static_cast<std::size_t>(peer)];
        }
        std::cout << '\n';
    }
}

void run_pingpong(int rank, int size) {
    if (size != 2) {
        if (rank == 0) std::cerr << "pingpong requires exactly two MPI processes\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    const std::vector<int> message_sizes = {
        0, 8, 64, 512, 4096, 32768, 262144, 1048576,
    };
    std::vector<char> buffer(1048576);
    if (rank == 0) std::cout << "PING bytes one_way_seconds\n";
    for (int bytes : message_sizes) {
        const int iterations = std::max(
            50, std::min(5000, 1000000 / std::max(bytes, 64)));
        std::vector<double> samples;
        for (int repeat = 0; repeat < 5; ++repeat) {
            MPI_Barrier(MPI_COMM_WORLD);
            const double start = MPI_Wtime();
            if (rank == 0) {
                for (int i = 0; i < iterations; ++i) {
                    MPI_Send(buffer.data(), bytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(buffer.data(), bytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                }
                samples.push_back((MPI_Wtime() - start) / (2.0 * iterations));
            } else {
                for (int i = 0; i < iterations; ++i) {
                    MPI_Recv(buffer.data(), bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    MPI_Send(buffer.data(), bytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
                }
            }
        }
        if (rank == 0) {
            std::sort(samples.begin(), samples.end());
            std::cout << "PING " << bytes << ' ' << std::setprecision(12)
                      << samples[samples.size() / 2] << '\n';
        }
    }
}

void usage(int rank) {
    if (rank == 0) {
        std::cerr
            << "Usage:\n"
            << "  samplesort parallel N uniform|skewed SEED OVERSAMPLING\n"
            << "  samplesort serial N uniform|skewed SEED SHARDS\n"
            << "  samplesort pingpong\n";
    }
}

}  // namespace

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try {
        if (argc < 2) {
            usage(rank);
            MPI_Finalize();
            return 2;
        }
        const std::string mode(argv[1]);
        if (mode == "pingpong") {
            run_pingpong(rank, size);
        } else if (mode == "serial" || mode == "parallel") {
            if (argc != 6) {
                usage(rank);
                MPI_Finalize();
                return 2;
            }
            const std::uint64_t n = parse_u64(argv[2], "N");
            const std::string distribution(argv[3]);
            const std::uint64_t seed = parse_u64(argv[4], "seed");
            const std::uint64_t final_arg = parse_u64(argv[5], "shards/oversampling");
            require_mpi(valid_distribution(distribution),
                        "distribution must be uniform or skewed", rank);
            if (mode == "serial") {
                run_serial(n, distribution, seed, static_cast<int>(final_arg),
                           rank, size);
            } else {
                run_parallel(n, distribution, seed,
                             static_cast<int>(final_arg), rank, size);
            }
        } else {
            usage(rank);
            MPI_Finalize();
            return 2;
        }
    } catch (const std::exception &error) {
        if (rank == 0) std::cerr << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    MPI_Finalize();
    return 0;
}
