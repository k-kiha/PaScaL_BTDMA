#include "pascal_btdma_cuda.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kTimingFields = 8;

int parse_positive_int(const int argc,
                       char** argv,
                       const int index,
                       const int default_value,
                       const char* name) {
    if (argc <= index) {
        return default_value;
    }
    char* end = nullptr;
    const long value = std::strtol(argv[index], &end, 10);
    if (end == argv[index] || *end != '\0' || value <= 0) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return static_cast<int>(value);
}

const char* mpi_mode_name(const pascal_btdma::MpiBufferMode mode) {
    return mode == pascal_btdma::MpiBufferMode::HostStaging ? "host" : "device";
}

std::array<double, kTimingFields> timing_fields(const pascal_btdma::BtdmaSolveTimings& timings) {
    return {timings.total,
            timings.local_compute,
            timings.forward_exchange,
            timings.reduced_compute,
            timings.backward_exchange,
            timings.update_compute,
            timings.computation(),
            timings.communication()};
}

void initialize_coefficients(std::vector<double>& h_a,
                             std::vector<double>& h_b,
                             std::vector<double>& h_c,
                             std::vector<double>& h_d,
                             const int nsys,
                             const int nrow,
                             const int m) {
    std::fill(h_a.begin(), h_a.end(), 0.0);
    std::fill(h_b.begin(), h_b.end(), 0.0);
    std::fill(h_c.begin(), h_c.end(), 0.0);
    std::fill(h_d.begin(), h_d.end(), 1.0);

    for (int row = 0; row < nrow; ++row) {
        for (int sys = 0; sys < nsys; ++sys) {
            for (int p = 0; p < m; ++p) {
                h_a[pascal_btdma::matrix_index(sys, row, p, p, nsys, nrow, m)] = 0.25;
                h_b[pascal_btdma::matrix_index(sys, row, p, p, nsys, nrow, m)] = 2.00;
                h_c[pascal_btdma::matrix_index(sys, row, p, p, nsys, nrow, m)] = 0.25;
            }
        }
    }
}

void write_signature_csv_if_requested(const std::vector<double>& solution,
                                      const int nsys,
                                      const int nrow,
                                      const int z_first,
                                      const int z_last,
                                      const int nprocs,
                                      const int n1,
                                      const int n2,
                                      const int n3,
                                      const int m,
                                      const int nrow_min,
                                      const int nrow_max,
                                      const int rank,
                                      const char* mpi_mode) {
    const char* output_path = std::getenv("PASCAL_BTDMA_SIGNATURE_OUT");

    double local_sum = 0.0;
    double local_sumsq = 0.0;
    double local_linf = 0.0;
    for (const double value : solution) {
        local_sum += value;
        local_sumsq += value * value;
        local_linf = std::max(local_linf, std::abs(value));
    }

    const std::array<int, 3> sample_z = {0, n3 / 2, n3 - 1};
    std::array<double, 3> local_samples = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        if (sample_z[i] >= z_first && sample_z[i] <= z_last) {
            const int local_row = sample_z[i] - z_first;
            local_samples[i] = solution[pascal_btdma::vector_index(0, local_row, 0, nsys, nrow)];
        }
    }

    double global_sum = 0.0;
    double global_sumsq = 0.0;
    double global_linf = 0.0;
    std::array<double, 3> global_samples = {0.0, 0.0, 0.0};
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sumsq, &global_sumsq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_linf, &global_linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_samples.data(), global_samples.data(), 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        return;
    }
    if (output_path == nullptr || output_path[0] == '\0') {
        return;
    }

    std::ifstream existing(output_path);
    const bool write_header = !existing.good() ||
                              existing.peek() == std::ifstream::traits_type::eof();
    existing.close();

    std::ofstream out(output_path, std::ios::app);
    if (!out) {
        throw std::runtime_error(std::string("Failed to open solution signature CSV: ") + output_path);
    }

    if (write_header) {
        out << "solver,variant,implementation,nranks,n1,n2,n3,m,nsys,nrow_min,nrow_max,"
            << "mpi_mode,solution_sum,solution_l2,solution_linf,sample_z0,"
            << "sample_zmid,sample_zlast\n";
    }

    out << std::setprecision(16)
        << "btdma,noncyclic,cuda-cxx"
        << ',' << nprocs
        << ',' << n1
        << ',' << n2
        << ',' << n3
        << ',' << m
        << ',' << nsys
        << ',' << nrow_min
        << ',' << nrow_max
        << ',' << mpi_mode
        << ',' << global_sum
        << ',' << std::sqrt(global_sumsq)
        << ',' << global_linf
        << ',' << global_samples[0]
        << ',' << global_samples[1]
        << ',' << global_samples[2]
        << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    try {
        const int n1 = parse_positive_int(argc, argv, 1, 32, "n1");
        const int n2 = parse_positive_int(argc, argv, 2, 32, "n2");
        const int n3 = parse_positive_int(argc, argv, 3, 128, "n3");
        const int m = parse_positive_int(argc, argv, 4, 5, "m");
        const int iterations = parse_positive_int(argc, argv, 5, 10, "iterations");
        if (argc > 6) {
            throw std::invalid_argument(
                "usage: example_cuda_cxx_btdma_profile [n1] [n2] [n3] [m] [iterations]");
        }
        if (m > pascal_btdma::kMaxBlockSize) {
            throw std::invalid_argument("m must be <= 8 for the current GPU BTDMA kernels");
        }

        int device_count = 0;
        PASCAL_BTDMA_CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            throw std::runtime_error("No CUDA device is visible to this MPI rank.");
        }
        PASCAL_BTDMA_CUDA_CHECK(cudaSetDevice(rank % device_count));
        PASCAL_BTDMA_CUDA_CHECK(cudaDeviceSynchronize());

        int z_first = 0;
        int z_last = -1;
        pascal_btdma::partition_1d(0, n3 - 1, nprocs, rank, z_first, z_last);
        const int nrow = z_last - z_first + 1;
        if (nrow < 2) {
            throw std::invalid_argument("Each rank needs at least two local rows.");
        }
        const int nsys = n1 * n2;
        const std::size_t matrix_size = static_cast<std::size_t>(nsys) * nrow * m * m;
        const std::size_t vector_size = static_cast<std::size_t>(nsys) * nrow * m;

        int nrow_min = 0;
        int nrow_max = 0;
        MPI_Reduce(&nrow, &nrow_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&nrow, &nrow_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        std::vector<double> h_a(matrix_size);
        std::vector<double> h_b(matrix_size);
        std::vector<double> h_c(matrix_size);
        std::vector<double> h_d(vector_size);
        initialize_coefficients(h_a, h_b, h_c, h_d, nsys, nrow, m);

        double* d_a = nullptr;
        double* d_b = nullptr;
        double* d_c = nullptr;
        double* d_d = nullptr;
        PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), matrix_size * sizeof(double)));
        PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), matrix_size * sizeof(double)));
        PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), matrix_size * sizeof(double)));
        PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_d), vector_size * sizeof(double)));

        const pascal_btdma::MpiBufferMode mpi_mode = pascal_btdma::mpi_mode_from_env();
        pascal_btdma::BtdmaGpuPlan plan;
        plan.create(m, nsys, nrow, MPI_COMM_WORLD, mpi_mode);

        if (rank == 0) {
            std::cout << "solver,variant,implementation,nranks,n1,n2,n3,m,nsys,"
                      << "nrow_min,nrow_max,iter,iterations,mpi_mode,total_s_max,total_s_avg,"
                      << "local_compute_s_max,forward_exchange_s_max,reduced_compute_s_max,"
                      << "backward_exchange_s_max,update_compute_s_max,compute_s_max,"
                      << "communication_s_max\n";
        }

        for (int iter = 0; iter < iterations; ++iter) {
            PASCAL_BTDMA_CUDA_CHECK(
                cudaMemcpy(d_a, h_a.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
            PASCAL_BTDMA_CUDA_CHECK(
                cudaMemcpy(d_b, h_b.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
            PASCAL_BTDMA_CUDA_CHECK(
                cudaMemcpy(d_c, h_c.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
            PASCAL_BTDMA_CUDA_CHECK(
                cudaMemcpy(d_d, h_d.data(), vector_size * sizeof(double), cudaMemcpyHostToDevice));

            MPI_Barrier(MPI_COMM_WORLD);

            pascal_btdma::BtdmaSolveTimings timings;
            pascal_btdma::solve_noncyclic_profiled(plan, d_a, d_b, d_c, d_d, m, nsys, nrow, &timings);

            if (iter == 0) {
                PASCAL_BTDMA_CUDA_CHECK(
                    cudaMemcpy(h_d.data(), d_d, vector_size * sizeof(double), cudaMemcpyDeviceToHost));
                write_signature_csv_if_requested(h_d, nsys, nrow, z_first, z_last, nprocs,
                                                 n1, n2, n3, m, nrow_min, nrow_max,
                                                 rank, mpi_mode_name(mpi_mode));
            }

            const std::array<double, kTimingFields> local_fields = timing_fields(timings);
            std::array<double, kTimingFields> max_fields{};
            std::array<double, kTimingFields> sum_fields{};
            MPI_Reduce(local_fields.data(), max_fields.data(), kTimingFields,
                       MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(local_fields.data(), sum_fields.data(), kTimingFields,
                       MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                std::cout << std::setprecision(12)
                          << "btdma,noncyclic,cuda-cxx"
                          << ',' << nprocs
                          << ',' << n1
                          << ',' << n2
                          << ',' << n3
                          << ',' << m
                          << ',' << nsys
                          << ',' << nrow_min
                          << ',' << nrow_max
                          << ',' << iter
                          << ',' << iterations
                          << ',' << mpi_mode_name(mpi_mode)
                          << ',' << max_fields[0]
                          << ',' << (sum_fields[0] / nprocs);
                for (int field = 1; field < kTimingFields; ++field) {
                    std::cout << ',' << max_fields[field];
                }
                std::cout << '\n';
            }
        }

        plan.destroy();
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFree(d_d);
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " error: " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
