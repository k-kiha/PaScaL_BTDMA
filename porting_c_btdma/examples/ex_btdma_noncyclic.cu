#include "pascal_btdma_cuda.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    try {
        int device_count = 0;
        PASCAL_BTDMA_CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            throw std::runtime_error("No CUDA device is visible to this MPI rank.");
        }
        PASCAL_BTDMA_CUDA_CHECK(cudaSetDevice(rank % device_count));
        PASCAL_BTDMA_CUDA_CHECK(cudaDeviceSynchronize());

        const int n1 = 32;
        const int n2 = 32;
        const int n3 = 128;
        const int m = 5;
        const int nsys = n1 * n2;

        int z_first = 0;
        int z_last = -1;
        pascal_btdma::partition_1d(0, n3 - 1, nprocs, rank, z_first, z_last);
        const int nrow = z_last - z_first + 1;

        const std::size_t matrix_size = static_cast<std::size_t>(nsys) * nrow * m * m;
        const std::size_t vector_size = static_cast<std::size_t>(nsys) * nrow * m;

        std::vector<double> h_a(matrix_size, 0.0);
        std::vector<double> h_b(matrix_size, 0.0);
        std::vector<double> h_c(matrix_size, 0.0);
        std::vector<double> h_d(vector_size, 1.0);

        for (int row = 0; row < nrow; ++row) {
            for (int sys = 0; sys < nsys; ++sys) {
                for (int q = 0; q < m; ++q) {
                    for (int p = 0; p < m; ++p) {
                        const bool diag = (p == q);
                        h_a[pascal_btdma::matrix_index(sys, row, p, q, nsys, nrow, m)] = diag ? 0.25 : 0.0;
                        h_b[pascal_btdma::matrix_index(sys, row, p, q, nsys, nrow, m)] = diag ? 2.00 : 0.0;
                        h_c[pascal_btdma::matrix_index(sys, row, p, q, nsys, nrow, m)] = diag ? 0.25 : 0.0;
                    }
                }
            }
        }

        double* d_a = nullptr;
        double* d_b = nullptr;
        double* d_c = nullptr;
        double* d_d = nullptr;
        PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), matrix_size * sizeof(double)));
        PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), matrix_size * sizeof(double)));
        PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), matrix_size * sizeof(double)));
        PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_d), vector_size * sizeof(double)));

        PASCAL_BTDMA_CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
        PASCAL_BTDMA_CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
        PASCAL_BTDMA_CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), matrix_size * sizeof(double), cudaMemcpyHostToDevice));
        PASCAL_BTDMA_CUDA_CHECK(cudaMemcpy(d_d, h_d.data(), vector_size * sizeof(double), cudaMemcpyHostToDevice));

        pascal_btdma::BtdmaGpuPlan plan;
        plan.create(m, nsys, nrow, MPI_COMM_WORLD, pascal_btdma::mpi_mode_from_env());
        pascal_btdma::solve_noncyclic(plan, d_a, d_b, d_c, d_d, m, nsys, nrow);
        PASCAL_BTDMA_CUDA_CHECK(cudaDeviceSynchronize());
        plan.destroy();

        PASCAL_BTDMA_CUDA_CHECK(cudaMemcpy(h_d.data(), d_d, vector_size * sizeof(double), cudaMemcpyDeviceToHost));

        for (int r = 0; r < nprocs; ++r) {
            if (rank == r) {
                std::cout << "Rank " << rank << " z=[" << z_first << ',' << z_last << "] sample D(sys=0):";
                for (int p = 0; p < m; ++p) {
                    std::cout << ' ' << h_d[pascal_btdma::vector_index(0, 0, p, nsys, nrow)];
                }
                std::cout << '\n';
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

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
