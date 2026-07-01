#pragma once

#include <cuda_runtime.h>
#include <mpi.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace pascal_btdma {

constexpr int kMaxBlockSize = 8;

enum class MpiBufferMode {
    DeviceDirect,
    HostStaging
};

struct CudaError : public std::runtime_error {
    explicit CudaError(const std::string& message) : std::runtime_error(message) {}
};

void cuda_check(cudaError_t status, const char* expr, const char* file, int line);

#define PASCAL_BTDMA_CUDA_CHECK(expr) \
    ::pascal_btdma::cuda_check((expr), #expr, __FILE__, __LINE__)

inline std::size_t matrix_index(const int sys,
                                const int row,
                                const int p,
                                const int q,
                                const int nsys,
                                const int nrow,
                                const int m) {
    return static_cast<std::size_t>(sys) +
           static_cast<std::size_t>(nsys) *
               (row + static_cast<std::size_t>(nrow) * (p + static_cast<std::size_t>(m) * q));
}

inline std::size_t vector_index(const int sys,
                                const int row,
                                const int p,
                                const int nsys,
                                const int nrow) {
    return static_cast<std::size_t>(sys) +
           static_cast<std::size_t>(nsys) * (row + static_cast<std::size_t>(nrow) * p);
}

void partition_1d(int start, int end, int nprocs, int rank, int& first, int& last);
MpiBufferMode mpi_mode_from_env();

struct CommPlan {
    int stride = 0;
    std::vector<int> a_counts;
    std::vector<int> a_displs;
    std::vector<int> b_counts;
    std::vector<int> b_displs;
    std::vector<int> a_count0;
    std::vector<int> a_count1;
    std::vector<int> a_displ0;
    std::vector<int> a_displ1;
    std::vector<int> b_count0;
    std::vector<int> b_count1;
    std::vector<int> b_displ0;
    std::vector<int> b_displ1;
    std::size_t a_total = 0;
    std::size_t b_total = 0;
};

struct BtdmaGpuPlan {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int nprocs = 1;
    int m = 0;
    int nsys = 0;
    int nrow = 0;
    int nsys_sub = 0;
    int local_first_sys = 0;
    int local_last_sys = -1;

    CommPlan matrix_comm;
    CommPlan vector_comm;

    double* rd_a = nullptr;
    double* rd_b = nullptr;
    double* rd_c = nullptr;
    double* rd_d = nullptr;
    double* tr_a = nullptr;
    double* tr_b = nullptr;
    double* tr_c = nullptr;
    double* tr_d = nullptr;
    double* buf_rd = nullptr;
    double* buf_tr = nullptr;

    std::size_t buf_rd_size = 0;
    std::size_t buf_tr_size = 0;

    dim3 threads = dim3(64, 1, 1);
    dim3 blocks = dim3(1, 1, 1);
    MpiBufferMode mpi_mode = MpiBufferMode::DeviceDirect;
    bool created = false;

    BtdmaGpuPlan() = default;
    BtdmaGpuPlan(const BtdmaGpuPlan&) = delete;
    BtdmaGpuPlan& operator=(const BtdmaGpuPlan&) = delete;
    BtdmaGpuPlan(BtdmaGpuPlan&& other) noexcept;
    BtdmaGpuPlan& operator=(BtdmaGpuPlan&& other) noexcept;
    ~BtdmaGpuPlan();

    void create(int m_in,
                int nsys_in,
                int nrow_in,
                MPI_Comm comm_in,
                MpiBufferMode mode = mpi_mode_from_env());

    void destroy() noexcept;
};

void solve_noncyclic(BtdmaGpuPlan& plan,
                     double* a_dev,
                     double* b_dev,
                     double* c_dev,
                     double* d_dev,
                     int m,
                     int nsys,
                     int nrow,
                     cudaStream_t stream = nullptr);

void solve_cyclic(BtdmaGpuPlan& plan,
                  double* a_dev,
                  double* b_dev,
                  double* c_dev,
                  double* d_dev,
                  int m,
                  int nsys,
                  int nrow,
                  cudaStream_t stream = nullptr);

}  // namespace pascal_btdma
