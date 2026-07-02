#include "pascal_btdma_cuda.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>
#include <utility>

namespace pascal_btdma {
namespace {

inline int ceil_div(const int a, const int b) {
    return (a + b - 1) / b;
}

inline std::size_t total_count(const std::vector<int>& values) {
    return static_cast<std::size_t>(
        std::accumulate(values.begin(), values.end(), 0));
}

double wall_time() {
    return MPI_Wtime();
}

void add_elapsed(BtdmaSolveTimings* timings,
                 double BtdmaSolveTimings::*field,
                 const double start_time) {
    if (timings != nullptr) {
        timings->*field += wall_time() - start_time;
    }
}

void sync_for_timing(BtdmaSolveTimings* timings, cudaStream_t stream) {
    if (timings != nullptr) {
        PASCAL_BTDMA_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

__device__ __forceinline__ int lm(const int i, const int j) {
    return i + j * kMaxBlockSize;
}

__device__ __forceinline__ std::size_t d_matrix_index(const int sys,
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

__device__ __forceinline__ std::size_t d_vector_index(const int sys,
                                                       const int row,
                                                       const int p,
                                                       const int nsys,
                                                       const int nrow) {
    return static_cast<std::size_t>(sys) +
           static_cast<std::size_t>(nsys) * (row + static_cast<std::size_t>(nrow) * p);
}

__device__ void zero_matrix(double* a) {
    for (int j = 0; j < kMaxBlockSize; ++j) {
        for (int i = 0; i < kMaxBlockSize; ++i) {
            a[lm(i, j)] = 0.0;
        }
    }
}

__device__ void zero_vector(double* x) {
    for (int i = 0; i < kMaxBlockSize; ++i) {
        x[i] = 0.0;
    }
}

__device__ void gemm_local(const double* a, const double* b, double* c, const int m) {
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            for (int l = 0; l < m; ++l) {
                sum += a[lm(i, l)] * b[lm(l, j)];
            }
            c[lm(i, j)] = sum;
        }
    }
}

__device__ void gemv_local(const double* a, const double* x, double* b, const int m) {
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int j = 0; j < m; ++j) {
            sum += a[lm(i, j)] * x[j];
        }
        b[i] = sum;
    }
}

__device__ void gesv_local(const int m, double* a, double* x) {
    double factor[kMaxBlockSize];
    for (int k = 0; k < m - 1; ++k) {
        for (int j = k + 1; j < m; ++j) {
            factor[j] = a[lm(j, k)] / a[lm(k, k)];
            for (int i = k; i < m; ++i) {
                a[lm(j, i)] -= factor[j] * a[lm(k, i)];
            }
        }
        for (int i = k + 1; i < m; ++i) {
            x[i] -= factor[i] * x[k];
        }
    }

    x[m - 1] /= a[lm(m - 1, m - 1)];
    for (int i = m - 2; i >= 0; --i) {
        for (int j = i + 1; j < m; ++j) {
            x[i] -= a[lm(i, j)] * x[j];
        }
        x[i] /= a[lm(i, i)];
    }
}

__device__ void gesv_mrhs2_local(const int m,
                                  double* a,
                                  double* x1,
                                  const int nrhs1,
                                  double* x2) {
    double factor[kMaxBlockSize];
    for (int k = 0; k < m - 1; ++k) {
        for (int j = k + 1; j < m; ++j) {
            factor[j] = a[lm(j, k)] / a[lm(k, k)];
            for (int i = k; i < m; ++i) {
                a[lm(j, i)] -= factor[j] * a[lm(k, i)];
            }
        }
        for (int irhs = 0; irhs < nrhs1; ++irhs) {
            for (int i = k + 1; i < m; ++i) {
                x1[lm(i, irhs)] -= factor[i] * x1[lm(k, irhs)];
            }
        }
        for (int i = k + 1; i < m; ++i) {
            x2[i] -= factor[i] * x2[k];
        }
    }

    for (int irhs = 0; irhs < nrhs1; ++irhs) {
        x1[lm(m - 1, irhs)] /= a[lm(m - 1, m - 1)];
        for (int i = m - 2; i >= 0; --i) {
            for (int j = i + 1; j < m; ++j) {
                x1[lm(i, irhs)] -= a[lm(i, j)] * x1[lm(j, irhs)];
            }
            x1[lm(i, irhs)] /= a[lm(i, i)];
        }
    }
    x2[m - 1] /= a[lm(m - 1, m - 1)];
    for (int i = m - 2; i >= 0; --i) {
        for (int j = i + 1; j < m; ++j) {
            x2[i] -= a[lm(i, j)] * x2[j];
        }
        x2[i] /= a[lm(i, i)];
    }
}

__device__ void gesv_mrhs3_local(const int m,
                                  double* a,
                                  double* x1,
                                  const int nrhs1,
                                  double* x2,
                                  const int nrhs2,
                                  double* x3) {
    double factor[kMaxBlockSize];
    for (int k = 0; k < m - 1; ++k) {
        for (int j = k + 1; j < m; ++j) {
            factor[j] = a[lm(j, k)] / a[lm(k, k)];
            for (int i = k; i < m; ++i) {
                a[lm(j, i)] -= factor[j] * a[lm(k, i)];
            }
        }
        for (int irhs = 0; irhs < nrhs1; ++irhs) {
            for (int i = k + 1; i < m; ++i) {
                x1[lm(i, irhs)] -= factor[i] * x1[lm(k, irhs)];
            }
        }
        for (int irhs = 0; irhs < nrhs2; ++irhs) {
            for (int i = k + 1; i < m; ++i) {
                x2[lm(i, irhs)] -= factor[i] * x2[lm(k, irhs)];
            }
        }
        for (int i = k + 1; i < m; ++i) {
            x3[i] -= factor[i] * x3[k];
        }
    }

    for (int irhs = 0; irhs < nrhs1; ++irhs) {
        x1[lm(m - 1, irhs)] /= a[lm(m - 1, m - 1)];
        for (int i = m - 2; i >= 0; --i) {
            for (int j = i + 1; j < m; ++j) {
                x1[lm(i, irhs)] -= a[lm(i, j)] * x1[lm(j, irhs)];
            }
            x1[lm(i, irhs)] /= a[lm(i, i)];
        }
    }
    for (int irhs = 0; irhs < nrhs2; ++irhs) {
        x2[lm(m - 1, irhs)] /= a[lm(m - 1, m - 1)];
        for (int i = m - 2; i >= 0; --i) {
            for (int j = i + 1; j < m; ++j) {
                x2[lm(i, irhs)] -= a[lm(i, j)] * x2[lm(j, irhs)];
            }
            x2[lm(i, irhs)] /= a[lm(i, i)];
        }
    }
    x3[m - 1] /= a[lm(m - 1, m - 1)];
    for (int i = m - 2; i >= 0; --i) {
        for (int j = i + 1; j < m; ++j) {
            x3[i] -= a[lm(i, j)] * x3[j];
        }
        x3[i] /= a[lm(i, i)];
    }
}

__device__ void load_matrix(const double* src,
                             const int sys,
                             const int row,
                             const int nsys,
                             const int nrow,
                             const int m,
                             double* dst) {
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            dst[lm(i, j)] = src[d_matrix_index(sys, row, i, j, nsys, nrow, m)];
        }
    }
}

__device__ void store_matrix(double* dst,
                              const int sys,
                              const int row,
                              const int nsys,
                              const int nrow,
                              const int m,
                              const double* src) {
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            dst[d_matrix_index(sys, row, i, j, nsys, nrow, m)] = src[lm(i, j)];
        }
    }
}

__device__ void load_vector(const double* src,
                             const int sys,
                             const int row,
                             const int nsys,
                             const int nrow,
                             const int m,
                             double* dst) {
    for (int i = 0; i < m; ++i) {
        dst[i] = src[d_vector_index(sys, row, i, nsys, nrow)];
    }
}

__device__ void store_vector(double* dst,
                              const int sys,
                              const int row,
                              const int nsys,
                              const int nrow,
                              const int m,
                              const double* src) {
    for (int i = 0; i < m; ++i) {
        dst[d_vector_index(sys, row, i, nsys, nrow)] = src[i];
    }
}

__global__ void btdma_many_kernel(const int n,
                                  const int nsys,
                                  const int m,
                                  double* a,
                                  double* b,
                                  double* c,
                                  double* d) {
    const int isys = threadIdx.x + blockIdx.x * blockDim.x;
    if (isys >= nsys) {
        return;
    }

    double rtmp[kMaxBlockSize * kMaxBlockSize];
    double atmp[kMaxBlockSize * kMaxBlockSize];
    double ctmp[kMaxBlockSize * kMaxBlockSize];
    double dtmp[kMaxBlockSize];
    double dtmp0[kMaxBlockSize];
    zero_matrix(rtmp);
    zero_matrix(atmp);
    zero_matrix(ctmp);
    zero_vector(dtmp);
    zero_vector(dtmp0);

    load_matrix(b, isys, 0, nsys, n, m, rtmp);
    load_matrix(c, isys, 0, nsys, n, m, ctmp);
    load_vector(d, isys, 0, nsys, n, m, dtmp);
    gesv_mrhs2_local(m, rtmp, ctmp, m, dtmp);
    store_matrix(c, isys, 0, nsys, n, m, ctmp);
    store_vector(d, isys, 0, nsys, n, m, dtmp);

    for (int q = 1; q < n; ++q) {
        load_matrix(a, isys, q, nsys, n, m, atmp);
        load_matrix(c, isys, q - 1, nsys, n, m, ctmp);
        gemm_local(atmp, ctmp, rtmp, m);
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < m; ++i) {
                rtmp[lm(i, j)] = b[d_matrix_index(isys, q, i, j, nsys, n, m)] - rtmp[lm(i, j)];
            }
        }

        load_matrix(c, isys, q, nsys, n, m, ctmp);
        load_vector(d, isys, q - 1, nsys, n, m, dtmp0);
        gemv_local(atmp, dtmp0, dtmp, m);
        for (int i = 0; i < m; ++i) {
            dtmp[i] = d[d_vector_index(isys, q, i, nsys, n)] - dtmp[i];
        }

        gesv_mrhs2_local(m, rtmp, ctmp, m, dtmp);
        store_matrix(c, isys, q, nsys, n, m, ctmp);
        store_vector(d, isys, q, nsys, n, m, dtmp);
    }

    for (int q = n - 2; q >= 0; --q) {
        load_matrix(c, isys, q, nsys, n, m, ctmp);
        load_vector(d, isys, q + 1, nsys, n, m, dtmp0);
        gemv_local(ctmp, dtmp0, dtmp, m);
        for (int i = 0; i < m; ++i) {
            d[d_vector_index(isys, q, i, nsys, n)] -= dtmp[i];
        }
    }
}

__global__ void btdma_many_modi_kernel(double* a,
                                       double* b,
                                       double* c,
                                       double* d,
                                       double* rd_a,
                                       double* rd_b,
                                       double* rd_c,
                                       double* rd_d,
                                       const int m,
                                       const int nsys,
                                       const int nrow) {
    const int isys = threadIdx.x + blockIdx.x * blockDim.x;
    if (isys >= nsys) {
        return;
    }

    double rtmp[kMaxBlockSize * kMaxBlockSize];
    double atmp[kMaxBlockSize * kMaxBlockSize];
    double ctmp[kMaxBlockSize * kMaxBlockSize];
    double dtmp[kMaxBlockSize];
    double dtmp0[kMaxBlockSize];
    double mtmp[kMaxBlockSize * kMaxBlockSize];
    zero_matrix(rtmp);
    zero_matrix(atmp);
    zero_matrix(ctmp);
    zero_matrix(mtmp);
    zero_vector(dtmp);
    zero_vector(dtmp0);

    load_matrix(b, isys, 0, nsys, nrow, m, rtmp);
    load_matrix(a, isys, 0, nsys, nrow, m, atmp);
    load_matrix(c, isys, 0, nsys, nrow, m, ctmp);
    load_vector(d, isys, 0, nsys, nrow, m, dtmp);
    gesv_mrhs3_local(m, rtmp, atmp, m, ctmp, m, dtmp);
    store_matrix(a, isys, 0, nsys, nrow, m, atmp);
    store_matrix(c, isys, 0, nsys, nrow, m, ctmp);
    store_vector(d, isys, 0, nsys, nrow, m, dtmp);

    load_matrix(b, isys, 1, nsys, nrow, m, rtmp);
    load_matrix(a, isys, 1, nsys, nrow, m, atmp);
    load_matrix(c, isys, 1, nsys, nrow, m, ctmp);
    load_vector(d, isys, 1, nsys, nrow, m, dtmp);
    gesv_mrhs3_local(m, rtmp, atmp, m, ctmp, m, dtmp);
    store_matrix(a, isys, 1, nsys, nrow, m, atmp);
    store_matrix(c, isys, 1, nsys, nrow, m, ctmp);
    store_vector(d, isys, 1, nsys, nrow, m, dtmp);

    for (int q = 2; q < nrow; ++q) {
        load_matrix(a, isys, q, nsys, nrow, m, mtmp);
        load_matrix(c, isys, q - 1, nsys, nrow, m, ctmp);
        gemm_local(mtmp, ctmp, rtmp, m);
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < m; ++i) {
                rtmp[lm(i, j)] = b[d_matrix_index(isys, q, i, j, nsys, nrow, m)] - rtmp[lm(i, j)];
            }
        }

        load_vector(d, isys, q - 1, nsys, nrow, m, dtmp0);
        gemv_local(mtmp, dtmp0, dtmp, m);
        for (int i = 0; i < m; ++i) {
            dtmp[i] = d[d_vector_index(isys, q, i, nsys, nrow)] - dtmp[i];
        }

        load_matrix(a, isys, q - 1, nsys, nrow, m, ctmp);
        gemm_local(mtmp, ctmp, atmp, m);
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < m; ++i) {
                atmp[lm(i, j)] = -atmp[lm(i, j)];
            }
        }

        load_matrix(c, isys, q, nsys, nrow, m, ctmp);
        gesv_mrhs3_local(m, rtmp, atmp, m, ctmp, m, dtmp);
        store_matrix(a, isys, q, nsys, nrow, m, atmp);
        store_matrix(c, isys, q, nsys, nrow, m, ctmp);
        store_vector(d, isys, q, nsys, nrow, m, dtmp);
    }

    for (int q = nrow - 3; q >= 1; --q) {
        load_matrix(c, isys, q, nsys, nrow, m, mtmp);
        load_vector(d, isys, q + 1, nsys, nrow, m, dtmp0);
        gemv_local(mtmp, dtmp0, dtmp, m);
        for (int i = 0; i < m; ++i) {
            d[d_vector_index(isys, q, i, nsys, nrow)] -= dtmp[i];
        }

        load_matrix(a, isys, q + 1, nsys, nrow, m, rtmp);
        gemm_local(mtmp, rtmp, atmp, m);
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < m; ++i) {
                a[d_matrix_index(isys, q, i, j, nsys, nrow, m)] -= atmp[lm(i, j)];
            }
        }

        load_matrix(c, isys, q + 1, nsys, nrow, m, rtmp);
        gemm_local(mtmp, rtmp, ctmp, m);
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < m; ++i) {
                c[d_matrix_index(isys, q, i, j, nsys, nrow, m)] = -ctmp[lm(i, j)];
            }
        }
    }

    load_matrix(a, isys, 1, nsys, nrow, m, atmp);
    load_matrix(c, isys, 0, nsys, nrow, m, mtmp);
    gemm_local(mtmp, atmp, rtmp, m);
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            rtmp[lm(i, j)] = -rtmp[lm(i, j)];
        }
        rtmp[lm(j, j)] += 1.0;
    }

    load_vector(d, isys, 1, nsys, nrow, m, dtmp0);
    gemv_local(mtmp, dtmp0, dtmp, m);
    for (int i = 0; i < m; ++i) {
        dtmp[i] = d[d_vector_index(isys, 0, i, nsys, nrow)] - dtmp[i];
    }

    load_matrix(c, isys, 1, nsys, nrow, m, atmp);
    gemm_local(mtmp, atmp, ctmp, m);
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            ctmp[lm(i, j)] = -ctmp[lm(i, j)];
        }
    }

    load_matrix(a, isys, 0, nsys, nrow, m, atmp);
    gesv_mrhs3_local(m, rtmp, atmp, m, ctmp, m, dtmp);
    store_matrix(a, isys, 0, nsys, nrow, m, atmp);
    store_matrix(c, isys, 0, nsys, nrow, m, ctmp);
    store_vector(d, isys, 0, nsys, nrow, m, dtmp);

    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            rd_a[d_matrix_index(isys, 0, i, j, nsys, 2, m)] =
                a[d_matrix_index(isys, 0, i, j, nsys, nrow, m)];
            rd_a[d_matrix_index(isys, 1, i, j, nsys, 2, m)] =
                a[d_matrix_index(isys, nrow - 1, i, j, nsys, nrow, m)];
            rd_c[d_matrix_index(isys, 0, i, j, nsys, 2, m)] =
                c[d_matrix_index(isys, 0, i, j, nsys, nrow, m)];
            rd_c[d_matrix_index(isys, 1, i, j, nsys, 2, m)] =
                c[d_matrix_index(isys, nrow - 1, i, j, nsys, nrow, m)];
            rd_b[d_matrix_index(isys, 0, i, j, nsys, 2, m)] = 0.0;
            rd_b[d_matrix_index(isys, 1, i, j, nsys, 2, m)] = 0.0;
        }
        rd_b[d_matrix_index(isys, 0, j, j, nsys, 2, m)] = 1.0;
        rd_b[d_matrix_index(isys, 1, j, j, nsys, 2, m)] = 1.0;
        rd_d[d_vector_index(isys, 0, j, nsys, 2)] =
            d[d_vector_index(isys, 0, j, nsys, nrow)];
        rd_d[d_vector_index(isys, 1, j, nsys, 2)] =
            d[d_vector_index(isys, nrow - 1, j, nsys, nrow)];
    }
}

__global__ void btdma_many_update_kernel(double* a,
                                         double* b,
                                         double* c,
                                         double* d,
                                         const double* rd_d,
                                         const int m,
                                         const int nsys,
                                         const int nrow) {
    (void)b;
    const int isys = threadIdx.x + blockIdx.x * blockDim.x;
    if (isys >= nsys) {
        return;
    }

    double atmp[kMaxBlockSize * kMaxBlockSize];
    double ctmp[kMaxBlockSize * kMaxBlockSize];
    double d0[kMaxBlockSize];
    double dn[kMaxBlockSize];
    double ad0[kMaxBlockSize];
    double cdn[kMaxBlockSize];

    for (int i = 0; i < m; ++i) {
        d0[i] = rd_d[d_vector_index(isys, 0, i, nsys, 2)];
        dn[i] = rd_d[d_vector_index(isys, 1, i, nsys, 2)];
        d[d_vector_index(isys, 0, i, nsys, nrow)] = d0[i];
        d[d_vector_index(isys, nrow - 1, i, nsys, nrow)] = dn[i];
    }

    for (int q = 1; q < nrow - 1; ++q) {
        load_matrix(a, isys, q, nsys, nrow, m, atmp);
        load_matrix(c, isys, q, nsys, nrow, m, ctmp);
        gemv_local(atmp, d0, ad0, m);
        gemv_local(ctmp, dn, cdn, m);
        for (int i = 0; i < m; ++i) {
            d[d_vector_index(isys, q, i, nsys, nrow)] -= ad0[i] + cdn[i];
        }
    }
}

__global__ void pack_matrix_kernel(double* out,
                                   const double* in,
                                   const int nsys,
                                   const int nrow,
                                   const int m,
                                   const int count0,
                                   const int count1,
                                   const int displ0,
                                   const int displ1,
                                   const int offset) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= count0 || j >= count1) {
        return;
    }
    const int sys = i + displ0;
    const int row = j + displ1;
    const int pair_count = count0 * count1;
    const int pair = i + j * count0;
    for (int k = 0; k < m * m; ++k) {
        const int p = k % m;
        const int q = k / m;
        out[offset + k * pair_count + pair] = in[d_matrix_index(sys, row, p, q, nsys, nrow, m)];
    }
}

__global__ void unpack_matrix_kernel(double* out,
                                     const double* in,
                                     const int nsys,
                                     const int nrow,
                                     const int m,
                                     const int count0,
                                     const int count1,
                                     const int displ0,
                                     const int displ1,
                                     const int offset) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= count0 || j >= count1) {
        return;
    }
    const int sys = i + displ0;
    const int row = j + displ1;
    const int pair_count = count0 * count1;
    const int pair = i + j * count0;
    for (int k = 0; k < m * m; ++k) {
        const int p = k % m;
        const int q = k / m;
        out[d_matrix_index(sys, row, p, q, nsys, nrow, m)] = in[offset + k * pair_count + pair];
    }
}

__global__ void pack_vector_kernel(double* out,
                                   const double* in,
                                   const int nsys,
                                   const int nrow,
                                   const int m,
                                   const int count0,
                                   const int count1,
                                   const int displ0,
                                   const int displ1,
                                   const int offset) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= count0 || j >= count1) {
        return;
    }
    const int sys = i + displ0;
    const int row = j + displ1;
    const int pair_count = count0 * count1;
    const int pair = i + j * count0;
    for (int k = 0; k < m; ++k) {
        out[offset + k * pair_count + pair] = in[d_vector_index(sys, row, k, nsys, nrow)];
    }
}

__global__ void unpack_vector_kernel(double* out,
                                     const double* in,
                                     const int nsys,
                                     const int nrow,
                                     const int m,
                                     const int count0,
                                     const int count1,
                                     const int displ0,
                                     const int displ1,
                                     const int offset) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= count0 || j >= count1) {
        return;
    }
    const int sys = i + displ0;
    const int row = j + displ1;
    const int pair_count = count0 * count1;
    const int pair = i + j * count0;
    for (int k = 0; k < m; ++k) {
        out[d_vector_index(sys, row, k, nsys, nrow)] = in[offset + k * pair_count + pair];
    }
}

void alltoallv_double(double* send_dev,
                      const std::size_t send_size,
                      const std::vector<int>& send_counts,
                      const std::vector<int>& send_displs,
                      double* recv_dev,
                      const std::size_t recv_size,
                      const std::vector<int>& recv_counts,
                      const std::vector<int>& recv_displs,
                      MPI_Comm comm,
                      const MpiBufferMode mode,
                      cudaStream_t stream) {
    PASCAL_BTDMA_CUDA_CHECK(cudaStreamSynchronize(stream));
    int ierr = MPI_SUCCESS;
    if (mode == MpiBufferMode::DeviceDirect) {
        ierr = MPI_Alltoallv(send_dev, send_counts.data(), send_displs.data(), MPI_DOUBLE,
                             recv_dev, recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                             comm);
    } else {
        std::vector<double> send_host(send_size);
        std::vector<double> recv_host(recv_size);
        PASCAL_BTDMA_CUDA_CHECK(cudaMemcpyAsync(send_host.data(), send_dev,
                                                send_size * sizeof(double),
                                                cudaMemcpyDeviceToHost, stream));
        PASCAL_BTDMA_CUDA_CHECK(cudaStreamSynchronize(stream));
        ierr = MPI_Alltoallv(send_host.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
                             recv_host.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                             comm);
        if (ierr == MPI_SUCCESS) {
            PASCAL_BTDMA_CUDA_CHECK(cudaMemcpyAsync(recv_dev, recv_host.data(),
                                                    recv_size * sizeof(double),
                                                    cudaMemcpyHostToDevice, stream));
        }
    }
    if (ierr != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Alltoallv failed in PaScaL_BTDMA CUDA C++ port");
    }
}

CommPlan make_comm_plan(const int stride, const int nsys, const int nsys_sub, const int nprocs) {
    CommPlan plan;
    plan.stride = stride;
    plan.a_counts.assign(nprocs, 0);
    plan.a_displs.assign(nprocs, 0);
    plan.b_counts.assign(nprocs, 0);
    plan.b_displs.assign(nprocs, 0);
    plan.a_count0.assign(nprocs, 0);
    plan.a_count1.assign(nprocs, 2);
    plan.a_displ0.assign(nprocs, 0);
    plan.a_displ1.assign(nprocs, 0);
    plan.b_count0.assign(nprocs, nsys_sub);
    plan.b_count1.assign(nprocs, 2);
    plan.b_displ0.assign(nprocs, 0);
    plan.b_displ1.assign(nprocs, 0);

    int sys_prefix = 0;
    int a_prefix = 0;
    int b_prefix = 0;
    for (int r = 0; r < nprocs; ++r) {
        int first = 0;
        int last = -1;
        partition_1d(0, nsys - 1, nprocs, r, first, last);
        const int rows = last - first + 1;
        plan.a_count0[r] = rows;
        plan.a_displ0[r] = sys_prefix;
        plan.b_displ1[r] = 2 * r;

        plan.a_counts[r] = stride * rows * 2;
        plan.a_displs[r] = a_prefix;
        a_prefix += plan.a_counts[r];

        plan.b_counts[r] = stride * nsys_sub * 2;
        plan.b_displs[r] = b_prefix;
        b_prefix += plan.b_counts[r];
        sys_prefix += rows;
    }
    plan.a_total = total_count(plan.a_counts);
    plan.b_total = total_count(plan.b_counts);
    return plan;
}

void launch_pack_matrix(const CommPlan& comm, const int r, const double* src,
                        const int nsys, const int nrow, const int m,
                        double* buffer, const dim3 threads, cudaStream_t stream,
                        const bool use_b_side) {
    const int count0 = use_b_side ? comm.b_count0[r] : comm.a_count0[r];
    const int count1 = use_b_side ? comm.b_count1[r] : comm.a_count1[r];
    const int displ0 = use_b_side ? comm.b_displ0[r] : comm.a_displ0[r];
    const int displ1 = use_b_side ? comm.b_displ1[r] : comm.a_displ1[r];
    const int offset = use_b_side ? comm.b_displs[r] : comm.a_displs[r];
    const dim3 blocks(ceil_div(count0, static_cast<int>(threads.x)),
                      ceil_div(count1, static_cast<int>(threads.y)), 1);
    pack_matrix_kernel<<<blocks, threads, 0, stream>>>(
        buffer, src, nsys, nrow, m, count0, count1, displ0, displ1, offset);
    PASCAL_BTDMA_CUDA_CHECK(cudaGetLastError());
}

void launch_unpack_matrix(const CommPlan& comm, const int r, double* dst,
                          const int nsys, const int nrow, const int m,
                          const double* buffer, const dim3 threads, cudaStream_t stream,
                          const bool use_b_side) {
    const int count0 = use_b_side ? comm.b_count0[r] : comm.a_count0[r];
    const int count1 = use_b_side ? comm.b_count1[r] : comm.a_count1[r];
    const int displ0 = use_b_side ? comm.b_displ0[r] : comm.a_displ0[r];
    const int displ1 = use_b_side ? comm.b_displ1[r] : comm.a_displ1[r];
    const int offset = use_b_side ? comm.b_displs[r] : comm.a_displs[r];
    const dim3 blocks(ceil_div(count0, static_cast<int>(threads.x)),
                      ceil_div(count1, static_cast<int>(threads.y)), 1);
    unpack_matrix_kernel<<<blocks, threads, 0, stream>>>(
        dst, buffer, nsys, nrow, m, count0, count1, displ0, displ1, offset);
    PASCAL_BTDMA_CUDA_CHECK(cudaGetLastError());
}

void launch_pack_vector(const CommPlan& comm, const int r, const double* src,
                        const int nsys, const int nrow, const int m,
                        double* buffer, const dim3 threads, cudaStream_t stream,
                        const bool use_b_side) {
    const int count0 = use_b_side ? comm.b_count0[r] : comm.a_count0[r];
    const int count1 = use_b_side ? comm.b_count1[r] : comm.a_count1[r];
    const int displ0 = use_b_side ? comm.b_displ0[r] : comm.a_displ0[r];
    const int displ1 = use_b_side ? comm.b_displ1[r] : comm.a_displ1[r];
    const int offset = use_b_side ? comm.b_displs[r] : comm.a_displs[r];
    const dim3 blocks(ceil_div(count0, static_cast<int>(threads.x)),
                      ceil_div(count1, static_cast<int>(threads.y)), 1);
    pack_vector_kernel<<<blocks, threads, 0, stream>>>(
        buffer, src, nsys, nrow, m, count0, count1, displ0, displ1, offset);
    PASCAL_BTDMA_CUDA_CHECK(cudaGetLastError());
}

void launch_unpack_vector(const CommPlan& comm, const int r, double* dst,
                          const int nsys, const int nrow, const int m,
                          const double* buffer, const dim3 threads, cudaStream_t stream,
                          const bool use_b_side) {
    const int count0 = use_b_side ? comm.b_count0[r] : comm.a_count0[r];
    const int count1 = use_b_side ? comm.b_count1[r] : comm.a_count1[r];
    const int displ0 = use_b_side ? comm.b_displ0[r] : comm.a_displ0[r];
    const int displ1 = use_b_side ? comm.b_displ1[r] : comm.a_displ1[r];
    const int offset = use_b_side ? comm.b_displs[r] : comm.a_displs[r];
    const dim3 blocks(ceil_div(count0, static_cast<int>(threads.x)),
                      ceil_div(count1, static_cast<int>(threads.y)), 1);
    unpack_vector_kernel<<<blocks, threads, 0, stream>>>(
        dst, buffer, nsys, nrow, m, count0, count1, displ0, displ1, offset);
    PASCAL_BTDMA_CUDA_CHECK(cudaGetLastError());
}

void forward_matrix(BtdmaGpuPlan& plan, double* src, double* dst, cudaStream_t stream) {
    for (int r = 0; r < plan.nprocs; ++r) {
        launch_pack_matrix(plan.matrix_comm, r, src, plan.nsys, 2, plan.m,
                           plan.buf_rd, plan.threads, stream, false);
    }
    alltoallv_double(plan.buf_rd, plan.matrix_comm.a_total,
                     plan.matrix_comm.a_counts, plan.matrix_comm.a_displs,
                     plan.buf_tr, plan.matrix_comm.b_total,
                     plan.matrix_comm.b_counts, plan.matrix_comm.b_displs,
                     plan.comm, plan.mpi_mode, stream);
    for (int r = 0; r < plan.nprocs; ++r) {
        launch_unpack_matrix(plan.matrix_comm, r, dst, plan.nsys_sub, 2 * plan.nprocs, plan.m,
                             plan.buf_tr, plan.threads, stream, true);
    }
}

void forward_vector(BtdmaGpuPlan& plan, double* src, double* dst, cudaStream_t stream) {
    for (int r = 0; r < plan.nprocs; ++r) {
        launch_pack_vector(plan.vector_comm, r, src, plan.nsys, 2, plan.m,
                           plan.buf_rd, plan.threads, stream, false);
    }
    alltoallv_double(plan.buf_rd, plan.vector_comm.a_total,
                     plan.vector_comm.a_counts, plan.vector_comm.a_displs,
                     plan.buf_tr, plan.vector_comm.b_total,
                     plan.vector_comm.b_counts, plan.vector_comm.b_displs,
                     plan.comm, plan.mpi_mode, stream);
    for (int r = 0; r < plan.nprocs; ++r) {
        launch_unpack_vector(plan.vector_comm, r, dst, plan.nsys_sub, 2 * plan.nprocs, plan.m,
                             plan.buf_tr, plan.threads, stream, true);
    }
}

void backward_vector(BtdmaGpuPlan& plan, double* src, double* dst, cudaStream_t stream) {
    for (int r = 0; r < plan.nprocs; ++r) {
        launch_pack_vector(plan.vector_comm, r, src, plan.nsys_sub, 2 * plan.nprocs, plan.m,
                           plan.buf_tr, plan.threads, stream, true);
    }
    alltoallv_double(plan.buf_tr, plan.vector_comm.b_total,
                     plan.vector_comm.b_counts, plan.vector_comm.b_displs,
                     plan.buf_rd, plan.vector_comm.a_total,
                     plan.vector_comm.a_counts, plan.vector_comm.a_displs,
                     plan.comm, plan.mpi_mode, stream);
    for (int r = 0; r < plan.nprocs; ++r) {
        launch_unpack_vector(plan.vector_comm, r, dst, plan.nsys, 2, plan.m,
                             plan.buf_rd, plan.threads, stream, false);
    }
}

void swap_plan(BtdmaGpuPlan& lhs, BtdmaGpuPlan& rhs) noexcept {
    using std::swap;
    swap(lhs.comm, rhs.comm);
    swap(lhs.rank, rhs.rank);
    swap(lhs.nprocs, rhs.nprocs);
    swap(lhs.m, rhs.m);
    swap(lhs.nsys, rhs.nsys);
    swap(lhs.nrow, rhs.nrow);
    swap(lhs.nsys_sub, rhs.nsys_sub);
    swap(lhs.local_first_sys, rhs.local_first_sys);
    swap(lhs.local_last_sys, rhs.local_last_sys);
    swap(lhs.matrix_comm, rhs.matrix_comm);
    swap(lhs.vector_comm, rhs.vector_comm);
    swap(lhs.rd_a, rhs.rd_a);
    swap(lhs.rd_b, rhs.rd_b);
    swap(lhs.rd_c, rhs.rd_c);
    swap(lhs.rd_d, rhs.rd_d);
    swap(lhs.tr_a, rhs.tr_a);
    swap(lhs.tr_b, rhs.tr_b);
    swap(lhs.tr_c, rhs.tr_c);
    swap(lhs.tr_d, rhs.tr_d);
    swap(lhs.buf_rd, rhs.buf_rd);
    swap(lhs.buf_tr, rhs.buf_tr);
    swap(lhs.buf_rd_size, rhs.buf_rd_size);
    swap(lhs.buf_tr_size, rhs.buf_tr_size);
    swap(lhs.threads, rhs.threads);
    swap(lhs.blocks, rhs.blocks);
    swap(lhs.mpi_mode, rhs.mpi_mode);
    swap(lhs.created, rhs.created);
}

void solve_noncyclic_impl(BtdmaGpuPlan& plan,
                          double* a_dev,
                          double* b_dev,
                          double* c_dev,
                          double* d_dev,
                          const int m,
                          const int nsys,
                          const int nrow,
                          BtdmaSolveTimings* timings,
                          cudaStream_t stream) {
    const double total_start = wall_time();

    if (!plan.created) {
        throw std::runtime_error("solve_noncyclic called with an uncreated BtdmaGpuPlan");
    }
    if (m != plan.m || nsys != plan.nsys || nrow != plan.nrow) {
        throw std::invalid_argument("solve_noncyclic dimensions do not match the plan");
    }

    const double local_start = wall_time();
    btdma_many_modi_kernel<<<plan.blocks, plan.threads, 0, stream>>>(
        a_dev, b_dev, c_dev, d_dev, plan.rd_a, plan.rd_b, plan.rd_c, plan.rd_d,
        m, nsys, nrow);
    PASCAL_BTDMA_CUDA_CHECK(cudaGetLastError());
    sync_for_timing(timings, stream);
    add_elapsed(timings, &BtdmaSolveTimings::local_compute, local_start);

    const double forward_start = wall_time();
    forward_matrix(plan, plan.rd_a, plan.tr_a, stream);
    forward_matrix(plan, plan.rd_b, plan.tr_b, stream);
    forward_matrix(plan, plan.rd_c, plan.tr_c, stream);
    forward_vector(plan, plan.rd_d, plan.tr_d, stream);
    sync_for_timing(timings, stream);
    add_elapsed(timings, &BtdmaSolveTimings::forward_exchange, forward_start);

    const double reduced_start = wall_time();
    const dim3 reduced_blocks(static_cast<unsigned int>(ceil_div(plan.nsys_sub, 64)), 1, 1);
    btdma_many_kernel<<<reduced_blocks, plan.threads, 0, stream>>>(
        2 * plan.nprocs, plan.nsys_sub, m, plan.tr_a, plan.tr_b, plan.tr_c, plan.tr_d);
    PASCAL_BTDMA_CUDA_CHECK(cudaGetLastError());
    sync_for_timing(timings, stream);
    add_elapsed(timings, &BtdmaSolveTimings::reduced_compute, reduced_start);

    const double backward_start = wall_time();
    backward_vector(plan, plan.tr_d, plan.rd_d, stream);
    sync_for_timing(timings, stream);
    add_elapsed(timings, &BtdmaSolveTimings::backward_exchange, backward_start);

    const double update_start = wall_time();
    btdma_many_update_kernel<<<plan.blocks, plan.threads, 0, stream>>>(
        a_dev, b_dev, c_dev, d_dev, plan.rd_d, m, nsys, nrow);
    PASCAL_BTDMA_CUDA_CHECK(cudaGetLastError());
    sync_for_timing(timings, stream);
    add_elapsed(timings, &BtdmaSolveTimings::update_compute, update_start);
    add_elapsed(timings, &BtdmaSolveTimings::total, total_start);
}

}  // namespace

void cuda_check(const cudaError_t status, const char* expr, const char* file, const int line) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << file << ':' << line << ": CUDA call failed: " << expr
            << " -> " << cudaGetErrorString(status);
        throw CudaError(oss.str());
    }
}

void partition_1d(const int start, const int end, const int nprocs, const int rank, int& first, int& last) {
    const int n = end - start + 1;
    const int base = n / nprocs;
    const int rem = n % nprocs;
    first = rank * base + start + std::min(rank, rem);
    last = first + base - 1;
    if (rem > rank) {
        ++last;
    }
}

MpiBufferMode mpi_mode_from_env() {
    const char* value = std::getenv("PASCAL_BTDMA_MPI_MODE");
    if (value != nullptr && std::strcmp(value, "host") == 0) {
        return MpiBufferMode::HostStaging;
    }
    return MpiBufferMode::DeviceDirect;
}

BtdmaGpuPlan::BtdmaGpuPlan(BtdmaGpuPlan&& other) noexcept {
    swap_plan(*this, other);
}

BtdmaGpuPlan& BtdmaGpuPlan::operator=(BtdmaGpuPlan&& other) noexcept {
    if (this != &other) {
        destroy();
        swap_plan(*this, other);
    }
    return *this;
}

BtdmaGpuPlan::~BtdmaGpuPlan() {
    destroy();
}

void BtdmaGpuPlan::create(const int m_in,
                          const int nsys_in,
                          const int nrow_in,
                          MPI_Comm comm_in,
                          const MpiBufferMode mode) {
    destroy();
    if (m_in <= 0 || m_in > kMaxBlockSize) {
        throw std::invalid_argument("BtdmaGpuPlan::create supports 1 <= m <= 8 in this first port");
    }
    if (nsys_in <= 0 || nrow_in < 2) {
        throw std::invalid_argument("BtdmaGpuPlan::create requires nsys > 0 and nrow >= 2");
    }

    MPI_Comm_dup(comm_in, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    m = m_in;
    nsys = nsys_in;
    nrow = nrow_in;
    mpi_mode = mode;
    partition_1d(0, nsys - 1, nprocs, rank, local_first_sys, local_last_sys);
    nsys_sub = local_last_sys - local_first_sys + 1;

    matrix_comm = make_comm_plan(m * m, nsys, nsys_sub, nprocs);
    vector_comm = make_comm_plan(m, nsys, nsys_sub, nprocs);

    const std::size_t rd_matrix_size = static_cast<std::size_t>(nsys) * 2 * m * m;
    const std::size_t tr_matrix_size = static_cast<std::size_t>(nsys_sub) * 2 * nprocs * m * m;
    const std::size_t rd_vector_size = static_cast<std::size_t>(nsys) * 2 * m;
    const std::size_t tr_vector_size = static_cast<std::size_t>(nsys_sub) * 2 * nprocs * m;

    PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rd_a), rd_matrix_size * sizeof(double)));
    PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rd_b), rd_matrix_size * sizeof(double)));
    PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rd_c), rd_matrix_size * sizeof(double)));
    PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rd_d), rd_vector_size * sizeof(double)));
    PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tr_a), tr_matrix_size * sizeof(double)));
    PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tr_b), tr_matrix_size * sizeof(double)));
    PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tr_c), tr_matrix_size * sizeof(double)));
    PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tr_d), tr_vector_size * sizeof(double)));

    buf_rd_size = matrix_comm.a_total;
    buf_tr_size = matrix_comm.b_total;
    PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf_rd), buf_rd_size * sizeof(double)));
    PASCAL_BTDMA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf_tr), buf_tr_size * sizeof(double)));

    threads = dim3(64, 1, 1);
    blocks = dim3(static_cast<unsigned int>(ceil_div(nsys, 64)), 1, 1);
    created = true;
}

void BtdmaGpuPlan::destroy() noexcept {
    if (rd_a) cudaFree(rd_a);
    if (rd_b) cudaFree(rd_b);
    if (rd_c) cudaFree(rd_c);
    if (rd_d) cudaFree(rd_d);
    if (tr_a) cudaFree(tr_a);
    if (tr_b) cudaFree(tr_b);
    if (tr_c) cudaFree(tr_c);
    if (tr_d) cudaFree(tr_d);
    if (buf_rd) cudaFree(buf_rd);
    if (buf_tr) cudaFree(buf_tr);
    rd_a = rd_b = rd_c = rd_d = nullptr;
    tr_a = tr_b = tr_c = tr_d = nullptr;
    buf_rd = buf_tr = nullptr;

    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Finalized(&finalized);
    }
    if (initialized && !finalized && comm != MPI_COMM_NULL) {
        MPI_Comm_free(&comm);
    }
    comm = MPI_COMM_NULL;
    created = false;
}

void solve_noncyclic(BtdmaGpuPlan& plan,
                     double* a_dev,
                     double* b_dev,
                     double* c_dev,
                     double* d_dev,
                     const int m,
                     const int nsys,
                     const int nrow,
                     cudaStream_t stream) {
    solve_noncyclic_impl(plan, a_dev, b_dev, c_dev, d_dev, m, nsys, nrow, nullptr, stream);
}

void solve_noncyclic_profiled(BtdmaGpuPlan& plan,
                              double* a_dev,
                              double* b_dev,
                              double* c_dev,
                              double* d_dev,
                              const int m,
                              const int nsys,
                              const int nrow,
                              BtdmaSolveTimings* timings,
                              cudaStream_t stream) {
    if (timings != nullptr) {
        *timings = BtdmaSolveTimings{};
    }
    solve_noncyclic_impl(plan, a_dev, b_dev, c_dev, d_dev, m, nsys, nrow, timings, stream);
}

void solve_cyclic(BtdmaGpuPlan&,
                  double*,
                  double*,
                  double*,
                  double*,
                  int,
                  int,
                  int,
                  cudaStream_t) {
    throw std::logic_error(
        "solve_cyclic is intentionally not implemented in this first CUDA C++ BTDMA port. "
        "The non-cyclic sample path is ported first; cyclic reduced coupling is a follow-up item.");
}

}  // namespace pascal_btdma
