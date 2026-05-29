# PaScaL_BTDMA

PaScaL_BTDMA is a compact MPI + CUDA Fortran package for solving many block
tridiagonal systems on distributed-memory NVIDIA GPU systems.

This cleaned package is intentionally narrow:

- NVIDIA HPC SDK Fortran compiler through `mpifort`
- CUDA Fortran through `cudafor`
- CUDA-aware MPI for GPU examples
- BLAS/LAPACK only for the optional CPU example

No SLURM scripts are included. Build the examples with `make`, then run the
executables with `mpirun` or your site's launcher. The `setup_env.sh` script is
an example build/run environment helper for the programs under `examples/`; edit
its module names for your cluster before sourcing it.

## Program Summary

- **Program title:** PaScaL_BTDMA
- **License:** MIT
- **Programming language:** Fortran 90/95, CUDA Fortran
- **Primary target:** MPI-parallel NVIDIA GPU systems, with an optional MPI CPU
  implementation

### Nature Of Problem

Block tridiagonal matrix systems arise frequently in computational physics,
especially in implicit schemes for partial differential equations. High-order
discretizations such as fifth-order WENO can also be reorganized as block
tridiagonal systems with small block sizes, typically `m = 2` to `5`.
Multi-dimensional applications often require thousands of independent systems to
be solved in a batched manner, while directional sweeps along different
coordinate axes require communication in distributed-memory environments.

### Solution Method

PaScaL_BTDMA uses a modified block Thomas algorithm with a two-stage reduction
strategy. The reduced system exchanges only two block rows per MPI process,
independent of the local matrix size. The CPU implementation uses batched
BLAS/LAPACK-style operations for the block matrix products and block linear
solves. The GPU implementation uses CUDA Fortran kernels, with one CUDA thread
block assigned to one independent block tridiagonal system.

The library supports both MPI-parallel CPU execution and CUDA-accelerated MPI
execution. It is intended for small block sizes, with the current implementation
designed for `m <= 10`, and is most effective when many independent systems are
solved simultaneously. The block solves use partial pivoting for numerical
stability.

## Layout

```text
PaScaL_BTDMA/
├── Makefile
├── Makefile.inc
├── setup_env.sh  # example build/run environment helper
├── src/
│   ├── mpiutil.f90
│   ├── mod_cudatools.f90
│   ├── mod_btdma_cpu.f90
│   └── mod_btdma_gpu_v2.f90
├── examples/
│   ├── btdma_cpu/sample_cpu.f90
│   ├── btdma_gpu/sample_gpu.f90
│   ├── heat3d/heat3d_mpi_gpu.f90
│   ├── euler3d/euler3d_mpi_gpu.f90
│   └── weno2d/weno_mpi_gpu.f90
├── include/   # generated module files
├── build/     # generated object files
├── lib/       # generated static libraries
└── run/       # generated executables
```

## Build

Load the NVIDIA environment first. The provided helper is site-specific and may
need small module-name edits.

```bash
source setup_env.sh
make
```

Default `make` builds the CUDA library and the main GPU examples:

```text
lib/libPaScaL_BTDMA_cuda.a
run/btdma_gpu.out
run/heat3d.out
run/euler3d.out
run/weno2d.out
```

Optional CPU build:

```bash
make cpu
```

This additionally builds:

```text
lib/libPaScaL_BTDMA_cpu.a
run/btdma_cpu.out
```

The CPU solver uses `dgemm` and `dgesv`, so `Makefile.inc` links BLAS/LAPACK.
If your CPU compiler or BLAS/LAPACK setup is different, edit only `FC`,
`FFLAGS_CPU`, and `LDFLAGS_CPU` in `Makefile.inc`.

## Useful Targets

```bash
make lib          # CUDA library only
make examples     # GPU examples only
make cpu          # CPU library and CPU example
make clean        # remove build objects
make veryclean    # remove objects, modules, libraries, and executables
```

Individual examples:

```bash
make ex_btdma_gpu
make ex_heat3d
make ex_euler3d
make ex_weno2d
make ex_btdma_cpu
```

## Run

Use a process count consistent with the decomposition macros used at build time.
The defaults are small enough for quick checks but still exercise the MPI paths.

```bash
mpirun -np 4 ./run/btdma_gpu.out
mpirun -np 8 ./run/heat3d.out
mpirun -np 8 ./run/euler3d.out
mpirun -np 4 ./run/weno2d.out
mpirun -np 4 ./run/btdma_cpu.out
```

Some examples write VTK or text files into the current working directory. To keep
outputs separate, run from a dedicated directory:

```bash
mkdir -p run/output_weno2d
cd run/output_weno2d
mpirun -np 4 ../weno2d.out
```

## Change Problem Sizes

Override the example preprocessor definitions at build time.

```bash
make ex_heat3d HEAT3D_DEFS="-DNX_VAL=256 -DNY_VAL=256 -DNZ_VAL=256 -DNP1=2 -DNP2=2 -DNP3=2 -DNRUN=5"
make ex_weno2d WENO2D_DEFS="-DN1=252 -DN2=252 -DM=3 -DNP1=2 -DNP2=2 -DNRUN=5"
```

The process count should match the product of the `NP*` macros. For example,
`NP1=2, NP2=2, NP3=2` requires `mpirun -np 8`.

## Main GPU API

```fortran
use mod_btdma_gpu_v2

type(BTDMA_PLAN_gpu_v2) :: plan

call btdma_makeplan_gpu_v2(plan, m, nsys, nrow_sub, comm)
call btdma_many_mpi_gpu_v2(A, B, C, D, m, nsys, nrow_sub, plan)
call btdma_many_cycl_mpi_gpu_v2(A, B, C, D, m, nsys, nrow_sub, plan)
call btdma_cleanplan_gpu_v2(plan)
```
