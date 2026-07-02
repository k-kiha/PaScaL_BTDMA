# PaScaL_BTDMA Study

This directory contains matched profiling drivers for the original CUDA
Fortran implementation and the CUDA C++ port.

The first Study target is the non-cyclic BTDMA path:

- Fortran original: `btdma_many_mpi_gpu_v2_profiled(...)`
- CUDA C++ port: `solve_noncyclic_profiled(...)`

The file names intentionally omit `noncyclic`; Study CSV metadata records the
solver variant as `noncyclic`.

## Drivers

```text
example_fortran_btdma_profile.f90
example_cuda_cxx_btdma_profile.cu
```

After build, the executables are also written in this flat Study directory:

```text
example_fortran_btdma_profile
example_cuda_cxx_btdma_profile
```

Both drivers use:

```text
[n1] [n2] [n3] [m] [iterations]
```

Defaults:

```text
32 32 128 5 10
```

The matched test coefficients use diagonal block matrices:

```text
A = 0.25 I
B = 2.00 I
C = 0.25 I
D = 1.00
```

This is a simple profiling workload, not a manufactured-solution correctness
proof. The optional solution CSV is a first-solve signature/check only.

## Build

```bash
make
```

Important defaults:

```text
CUDA_ARCH=90
FC=mpifort
NVCC=nvcc
MPICXX=mpicxx
```

## Study Runs

Use a small smoke matrix:

```bash
./run_study_sweep.sh
```

Use custom cases:

```bash
STUDY_PRESET=custom \
NP_LIST="2 4" \
SIZE_LIST="32,32,128 64,64,512" \
M_LIST="3 5" \
./run_study_sweep.sh
```

Before running a server job, inspect the planned commands and manifest:

```bash
DRY_RUN=1 ./run_study_sweep.sh
```

For the CUDA C++ driver, MPI device-buffer communication is the default. Set
`CXX_DEFAULT_MPI_MODES=host` or use `MPI_MODE_LIST="device host"` in comparison
cases when host-staging fallback is needed.

## Report Data Run

The report-data case matrix is stored under `result/`:

```text
result/btdma_report_case_matrix.md
result/btdma_report_case_matrix.csv
```

Use the full wrapper to run the explicit case matrix:

```bash
./run_full_study.sh
```

Preview the complete command list without running MPI jobs:

```bash
DRY_RUN=1 ./run_full_study.sh
```

Build the Study drivers before running:

```bash
BUILD_BEFORE_RUN=1 ./run_full_study.sh
```

## Study Outputs

`run_study_sweep.sh` writes one output set per run:

```text
btdma_total_profile_YYMMDD_HHMMSS.csv
btdma_solution_signature_YYMMDD_HHMMSS.csv
btdma_environment_YYMMDD_HHMMSS.txt
btdma_case_manifest_YYMMDD_HHMMSS.csv
```

The timing CSV schema is:

```text
solver,variant,implementation,nranks,n1,n2,n3,m,nsys,nrow_min,nrow_max,iter,iterations,mpi_mode,total_s_max,total_s_avg,local_compute_s_max,forward_exchange_s_max,reduced_compute_s_max,backward_exchange_s_max,update_compute_s_max,compute_s_max,communication_s_max
```

Aggregated phase columns:

```text
compute_s_max       = local_compute + reduced_compute + update_compute
communication_s_max = forward_exchange + backward_exchange
```

For BTDMA, `forward_exchange` and `backward_exchange` currently include the
whole exchange path: pack kernels, `MPI_Alltoallv`, and unpack kernels. They are
not pure MPI-only timings.

The solution signature CSV schema is:

```text
solver,variant,implementation,nranks,n1,n2,n3,m,nsys,nrow_min,nrow_max,mpi_mode,solution_sum,solution_l2,solution_linf,sample_z0,sample_zmid,sample_zlast
```

The environment file records the GPU, CUDA/MPI toolchain, git revision, and
sweep settings used for the run. Keep it with the CSV outputs when moving
results between server and local workspaces.

`run_full_study.sh` writes report outputs under `result/`:

```text
btdma_total_profile_YYMMDD_HHMMSS.csv
btdma_solution_signature_YYMMDD_HHMMSS.csv
btdma_environment_YYMMDD_HHMMSS.txt
btdma_full_case_list_YYMMDD_HHMMSS.csv
btdma_full_study_YYMMDD_HHMMSS.log
btdma_full_study_YYMMDD_HHMMSS_case_files/
```
