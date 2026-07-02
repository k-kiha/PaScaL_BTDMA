# PaScaL_BTDMA Report Case Matrix

This file records the test cases for the non-cyclic PaScaL_BTDMA Study report.
The executable runner reads the machine-readable list in
`btdma_report_case_matrix.csv`.

Scope:

- solver variant: `noncyclic`
- implementations: CUDA Fortran original and CUDA C++ port
- CUDA C++ MPI modes: `device` by default, `host` only for fallback comparison
- iterations per run: `10` by default
- timing analysis rule: use `iter >= 1` for steady-state timing and keep
  `iter = 0` as first-solve/warm-up context

The current case matrix has:

- 28 device-mode base cases
- 3 CUDA C++ host-fallback cases
- 31 custom sweep calls
- 56 Fortran/CUDA C++ device runs plus 3 CUDA C++ host runs
- 590 raw timing rows when `ITERATIONS=10`

## Report Questions

1. Does the CUDA C++ port produce the same first-solve solution signature as
   the original CUDA Fortran implementation?
2. Which phase dominates non-cyclic BTDMA: local block compute, forward
   exchange, reduced solve, backward exchange, or update?
3. How close is the CUDA C++ port to the original Fortran implementation in
   total and phase-level timings?
4. How does block size `m` change total time and compute/exchange balance?
5. How does batch size `nsys = n1*n2` affect GPU throughput?
6. How does local row length `nrow = n3/nranks` affect the measured exchange
   cost?
7. How does the non-cyclic BTDMA port scale from 2 to 8 GPUs?
8. How much slower is CUDA C++ host-staging MPI than CUDA-aware device MPI?
9. What remains outside the current scope, especially cyclic BTDMA and
   all-to-all/CPU baselines?

## Case Groups

| group | cases | purpose |
| --- | ---: | --- |
| single_gpu_reference | 4 | solution signature and small single-GPU reference for `m = 2,3,5,8` |
| strong_scaling | 12 | `np = 2,4,8` at `64x64x2048` for `m = 2,3,5,8` |
| weak_nrow_scaling | 9 logical points | local `nrow = 512` path for `m = 2,5,8`; the `np=4` midpoint reuses strong-scaling rows |
| nsys_sensitivity | 9 logical points | `np = 8`, `n3 = 2048`, `nsys = 32^2,64^2,128^2` for `m = 2,5,8`; the midpoint reuses strong-scaling rows |
| mpi_mode_comparison | 6 logical points | CUDA C++ device reference from strong `m=5` rows plus host fallback at `np = 2,4,8` |

## Exact Execution List

The authoritative exact list is:

```text
PaScaL_BTDMA/Study/result/btdma_report_case_matrix.csv
```

The wrapper copies that CSV to a timestamped full case list before execution:

```text
btdma_full_case_list_YYMMDD_HHMMSS.csv
```

## Output Files

`run_full_study.sh` writes report data under this `result/` directory:

```text
btdma_total_profile_YYMMDD_HHMMSS.csv
btdma_solution_signature_YYMMDD_HHMMSS.csv
btdma_environment_YYMMDD_HHMMSS.txt
btdma_full_case_list_YYMMDD_HHMMSS.csv
btdma_full_study_YYMMDD_HHMMSS.log
btdma_full_study_YYMMDD_HHMMSS_case_files/
```

Each per-case call also writes its own small manifest and environment capture
under `btdma_full_study_YYMMDD_HHMMSS_case_files/`, so failed or partial runs
can be traced without guessing which custom sweep generated a row.

## Server Use

Preview only:

```bash
DRY_RUN=1 ./run_full_study.sh
```

Build first, then run:

```bash
BUILD_BEFORE_RUN=1 ./run_full_study.sh
```

Use fewer iterations for a quick server check:

```bash
ITERATIONS=3 DRY_RUN=1 ./run_full_study.sh
```

Normal report-data collection:

```bash
./run_full_study.sh
```
