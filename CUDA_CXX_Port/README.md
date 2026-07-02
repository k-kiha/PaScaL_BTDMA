# PaScaL_BTDMA CUDA C++ Port

This directory is a first-pass CUDA C++ port of the non-cyclic GPU path in
`../Fortran_Original/src/mod_btdma_gpu_v2.f90`.

## Scope

Implemented:

- `BtdmaGpuPlan`
- non-cyclic `solve_noncyclic`
- local modified block Thomas reduction
- matrix/vector all-to-all pack/unpack
- transformed reduced block solve
- backward `trD -> rdD` exchange
- local update
- small dense helper routines with the original no-pivot GPU behavior

Not implemented in this first pass:

- cyclic `solve_cyclic`
- pivoting in GPU dense solves
- `m > 8`

## Layout

The C++ layout is explicit:

```cpp
A[sys + nsys * (row + nrow * (p + m * q))]
D[sys + nsys * (row + nrow * p)]
```

This removes the implicit rank reinterpretation used by the Fortran/CUDA code.

## Build

Requires NVIDIA CUDA Toolkit, `nvcc`, and MPI.

```bash
make CUDA_ARCH=80
```

If your MPI wrapper is not `mpicxx`:

```bash
make MPICXX=/path/to/mpicxx CUDA_ARCH=90
```

## Run

```bash
mpirun -np 4 ./run/ex_btdma_noncyclic
```

For non CUDA-aware MPI:

```bash
PASCAL_BTDMA_MPI_MODE=host mpirun -np 4 ./run/ex_btdma_noncyclic
```

## Verification Status

The current workstation does not expose `nvcc` or CUDA hardware, so this port has not been compiled or run here. Build and runtime validation must be done on a CUDA-capable environment.
