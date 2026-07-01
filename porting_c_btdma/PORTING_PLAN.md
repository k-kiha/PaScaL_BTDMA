# PaScaL_BTDMA CUDA C++ Porting Plan

date: 2026-06-30
source: `../src/mod_btdma_gpu_v2.f90`
reference report: `../../brain/4_260630_PaScaL_BTDMA_analysis.md`

## 1. Porting 기준

- NVIDIA가 공식 지원하는 CUDA C++ runtime API와 `nvcc`를 기준으로 한다.
- 원본 GPU v2의 실제 구현처럼 one CUDA thread가 one independent block-tridiagonal system을 담당하는 구조를 1차 포팅에서 보존한다.
- CUDA 하드웨어가 현재 없으므로 실제 실행 검증은 하지 못한다. 이번 단계는 code translation, build skeleton, static review까지 수행한다.

## 2. 1차 범위

이번 포팅은 `examples/btdma_gpu/sample_gpu.f90`가 호출하는 non-cyclic 경로를 우선 구현한다.

```text
btdma_many_modi_gpu_v2
  -> btdma_many_a2av_forward_gpu_v2
  -> btdma_many_gpu_v2
  -> btdma_many_a2av_backward_gpu_v2
  -> btdma_many_update_gpu_v2
```

cyclic 경로 `btdma_many_cycl_mpi_gpu_v2`는 public entry를 남기되 이번 단계에서는 명시적으로 `not implemented`로 둔다. 이유는 cyclic kernel이 `reduceE`와 추가 coupling update를 포함하고, 현재 CUDA 하드웨어 없이 한 번에 포팅하면 검증 불가능한 오류 표면이 너무 커지기 때문이다.

## 3. Layout 계약

Fortran GPU solver dummy argument:

```fortran
A(1:nsys,1:nrow_sub,1:m,1:m)
D(1:nsys,1:nrow_sub,1:m)
```

CUDA C++ flat layout:

```cpp
A[sys + nsys * (row + nrow * (p + m * q))]
D[sys + nsys * (row + nrow * p)]
```

여기서 `sys = i + j*n1`, `row = k`, `p/q`는 block matrix index다.

## 4. MPI 전략

원본은 matrix payload `m*m`와 vector payload `m`를 분리한 all-to-all plan을 만든다. C++ plan도 이를 유지한다.

- matrix plan: `rdA/rdB/rdC <-> trA/trB/trC`, width `m*m`
- vector plan: `rdD <-> trD`, width `m`

기본 통신은 CUDA-aware `MPI_Alltoallv`다. 비 CUDA-aware MPI 환경을 위해 host staging fallback을 둔다.

```bash
export PASCAL_BTDMA_MPI_MODE=host
```

## 5. `m` 상한 정책

원본 `mod_btdma_gpu_v2`의 main GPU helper는 local scratch 배열 `1:8`을 사용한다. 따라서 1차 C++ 포팅도 `MAX_M = 8`을 기본 상한으로 둔다.

- `m <= 8`: 지원
- `m > 8`: plan creation에서 예외 처리

README의 `m <= 10` 설명과 원본 GPU v2 scratch 구현의 불일치는 후속 검증 과제로 둔다.

## 6. Pivoting 정책

원본 GPU helper `gesv`, `gesv_mrhs2`, `gesv_mrhs3`에는 명시적 pivoting이 없다. C++ 1차 포팅도 원본 GPU behavior를 보존하기 위해 pivoting 없는 Gaussian elimination을 사용한다.

CPU `dgesv` reference와 수치 안정성을 맞추는 pivoting 추가는 후속 개선으로 둔다.

## 7. 자체 검토와 수정 사항

### 검토 1

- 문제: Fortran GPU plan은 `rdA(1:nsys,1:2,1:m,1:m)`이지만 pack helper는 이를 stride `m*m`, system, reduced-row 차원으로 재해석한다.
- 판단: C++에서는 암묵적 rank reinterpretation을 피하고, logical layout `(sys,row,p,q)`에서 pack kernel이 block element `p+q*m`를 명시적으로 꺼내도록 한다.
- 수정: matrix pack/unpack kernel을 별도 구현한다.

### 검토 2

- 문제: 원본 GPU backward exchange는 `trA/trB/trC`를 되돌리지 않고 `trD`만 되돌린다.
- 판단: update 단계에는 solved reduced RHS만 필요하다. 원본 GPU v2 최적화를 보존한다.
- 수정: backward all-to-all은 vector plan으로 `trD -> rdD`만 수행한다.

### 검토 3

- 문제: cyclic solver까지 구현하지 않으면 전체 application examples coverage가 부족하다.
- 판단: 현재 하드웨어 검증이 불가능하므로 non-cyclic sample path를 먼저 실제 코드화하고, cyclic은 명시적 후속 항목으로 관리하는 것이 안전하다.
- 수정: `solve_cyclic` API는 제공하되 `std::logic_error`를 던지도록 한다.

### 결론

1차 계획은 non-cyclic BTDMA의 algorithm/MPI/layout/device helper를 보존한다. 남은 큰 범위는 cyclic solver와 pivoting policy 검증이다.
