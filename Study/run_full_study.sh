#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MPIRUN=${MPIRUN:-mpirun}
MAKE=${MAKE:-make}
CUDA_ARCH=${CUDA_ARCH:-90}
ITERATIONS=${ITERATIONS:-10}
BUILD_BEFORE_RUN=${BUILD_BEFORE_RUN:-0}
DRY_RUN=${DRY_RUN:-0}
if [[ "${DRY_RUN_ONLY:-0}" == "1" ]]; then
    DRY_RUN=1
fi

TIMESTAMP=${TIMESTAMP:-$(date +%y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/result"}
CASE_MATRIX=${CASE_MATRIX:-"$OUTPUT_DIR/btdma_report_case_matrix.csv"}
RUN_SWEEP=${RUN_SWEEP:-"$SCRIPT_DIR/run_study_sweep.sh"}
FORTRAN_EXE=${FORTRAN_EXE:-"$SCRIPT_DIR/example_fortran_btdma_profile"}
CXX_EXE=${CXX_EXE:-"$SCRIPT_DIR/example_cuda_cxx_btdma_profile"}
SET_CUDA_VISIBLE_DEVICES=${SET_CUDA_VISIBLE_DEVICES:-1}

TIMING_OUT=${TIMING_OUT:-"$OUTPUT_DIR/btdma_total_profile_${TIMESTAMP}.csv"}
SIGNATURE_OUT=${SIGNATURE_OUT:-"$OUTPUT_DIR/btdma_solution_signature_${TIMESTAMP}.csv"}
ENV_OUT=${ENV_OUT:-"$OUTPUT_DIR/btdma_environment_${TIMESTAMP}.txt"}
CASE_LIST_OUT=${CASE_LIST_OUT:-"$OUTPUT_DIR/btdma_full_case_list_${TIMESTAMP}.csv"}
RUN_LOG=${RUN_LOG:-"$OUTPUT_DIR/btdma_full_study_${TIMESTAMP}.log"}
CASE_FILES_DIR=${CASE_FILES_DIR:-"$OUTPUT_DIR/btdma_full_study_${TIMESTAMP}_case_files"}

usage() {
    cat <<'USAGE'
Usage:
  ./run_full_study.sh
  DRY_RUN=1 ./run_full_study.sh
  BUILD_BEFORE_RUN=1 ./run_full_study.sh

Purpose:
  Run the explicit non-cyclic PaScaL_BTDMA report case matrix and collect
  timing/signature/environment data under Study/result.

Important variables:
  ITERATIONS=10
  CUDA_ARCH=90
  BUILD_BEFORE_RUN=0
  DRY_RUN=1
  OUTPUT_DIR=./result
  CASE_MATRIX=./result/btdma_report_case_matrix.csv
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

mkdir -p "$OUTPUT_DIR" "$CASE_FILES_DIR"

main() {
    echo "# PaScaL_BTDMA full Study"
    echo "timestamp=$TIMESTAMP"
    echo "root_dir=$ROOT_DIR"
    echo "script_dir=$SCRIPT_DIR"
    echo "output_dir=$OUTPUT_DIR"
    echo "case_matrix=$CASE_MATRIX"
    echo "iterations=$ITERATIONS"
    echo "cuda_arch=$CUDA_ARCH"
    echo "build_before_run=$BUILD_BEFORE_RUN"
    echo "dry_run=$DRY_RUN"
    echo

    if [[ ! -f "$CASE_MATRIX" ]]; then
        echo "error: missing case matrix: $CASE_MATRIX" >&2
        exit 2
    fi
    if [[ ! -x "$RUN_SWEEP" ]]; then
        echo "error: missing executable run_study_sweep.sh: $RUN_SWEEP" >&2
        exit 2
    fi

    cp "$CASE_MATRIX" "$CASE_LIST_OUT"

    devices_for_np() {
        case "$1" in
            1) echo "0" ;;
            2) echo "0,1" ;;
            4) echo "0,1,2,3" ;;
            8) echo "0,1,2,3,4,5,6,7" ;;
            *) echo "" ;;
        esac
    }

    capture_environment() {
        {
            echo "# PaScaL_BTDMA Full Study Environment"
            echo "date=$(date '+%Y-%m-%dT%H:%M:%S%z')"
            echo "hostname=$(hostname)"
            echo "pwd=$PWD"
            echo "root_dir=$ROOT_DIR"
            echo "script_dir=$SCRIPT_DIR"
            echo "output_dir=$OUTPUT_DIR"
            echo "case_matrix=$CASE_MATRIX"
            echo "case_list=$CASE_LIST_OUT"
            echo "timing_csv=$TIMING_OUT"
            echo "signature_csv=$SIGNATURE_OUT"
            echo "case_files_dir=$CASE_FILES_DIR"
            echo "iterations=$ITERATIONS"
            echo "cuda_arch=$CUDA_ARCH"
            echo "build_before_run=$BUILD_BEFORE_RUN"
            echo "dry_run=$DRY_RUN"
            echo "set_cuda_visible_devices=$SET_CUDA_VISIBLE_DEVICES"
            echo
            echo "## git"
            git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true
            git -C "$ROOT_DIR" status --short 2>/dev/null || true
            echo
            echo "## nvidia-smi"
            if command -v nvidia-smi >/dev/null 2>&1; then
                nvidia-smi || true
                echo
                echo "## nvidia-smi topo -m"
                nvidia-smi topo -m || true
            else
                echo "nvidia-smi not found"
            fi
            echo
            echo "## nvcc --version"
            if command -v nvcc >/dev/null 2>&1; then
                nvcc --version || true
            else
                echo "nvcc not found"
            fi
            echo
            echo "## mpirun --version"
            "$MPIRUN" --version || true
            echo
            echo "## mpifort --version"
            if command -v mpifort >/dev/null 2>&1; then
                mpifort --version || true
            else
                echo "mpifort not found"
            fi
            echo
            echo "## mpicxx --version"
            if command -v mpicxx >/dev/null 2>&1; then
                mpicxx --version || true
            else
                echo "mpicxx not found"
            fi
        } > "$ENV_OUT"
    }

    capture_environment

    if [[ "$BUILD_BEFORE_RUN" == "1" ]]; then
        echo "building Study drivers"
        "$MAKE" -C "$SCRIPT_DIR" all CUDA_ARCH="$CUDA_ARCH"
    fi

    if [[ "$DRY_RUN" != "1" ]]; then
        if [[ ! -x "$FORTRAN_EXE" ]]; then
            echo "error: missing Fortran Study executable: $FORTRAN_EXE" >&2
            echo "hint: run BUILD_BEFORE_RUN=1 ./run_full_study.sh" >&2
            exit 2
        fi
        if [[ ! -x "$CXX_EXE" ]]; then
            echo "error: missing CUDA C++ Study executable: $CXX_EXE" >&2
            echo "hint: run BUILD_BEFORE_RUN=1 ./run_full_study.sh" >&2
            exit 2
        fi
    fi

    case_count=0
    run_count=0
    {
        read -r header
        while IFS=, read -r case_order run_case_id study_tags variant nranks n1 n2 n3 m \
            cxx_mpi_modes run_fortran run_cxx notes; do
            if [[ -z "${case_order:-}" ]]; then
                continue
            fi

            case_count=$((case_count + 1))
            run_count=$((run_count + run_fortran + run_cxx))
            devices="$(devices_for_np "$nranks")"
            case_prefix="${case_order}_${run_case_id}"
            case_manifest="$CASE_FILES_DIR/${case_prefix}_manifest.csv"
            case_env="$CASE_FILES_DIR/${case_prefix}_environment.txt"

            echo
            echo "## case $case_order: $run_case_id"
            echo "tags=$study_tags"
            echo "variant=$variant np=$nranks n1=$n1 n2=$n2 n3=$n3 m=$m cxx_mpi_modes=$cxx_mpi_modes"
            echo "run_fortran=$run_fortran run_cxx=$run_cxx notes=$notes"
            if [[ "$SET_CUDA_VISIBLE_DEVICES" == "1" ]]; then
                echo "CUDA_VISIBLE_DEVICES=$devices"
            fi

            (
                cd "$SCRIPT_DIR"
                if [[ "$SET_CUDA_VISIBLE_DEVICES" == "1" && -n "$devices" ]]; then
                    export CUDA_VISIBLE_DEVICES="$devices"
                fi
                export MPIRUN="$MPIRUN"
                export FORTRAN_EXE="$FORTRAN_EXE"
                export CXX_EXE="$CXX_EXE"
                export STUDY_PRESET=custom
                export NP_LIST="$nranks"
                export SIZE_LIST="${n1},${n2},${n3}"
                export M_LIST="$m"
                export ITERATIONS="$ITERATIONS"
                export CXX_DEFAULT_MPI_MODES="$cxx_mpi_modes"
                export RUN_FORTRAN="$run_fortran"
                export RUN_CXX="$run_cxx"
                export DRY_RUN="$DRY_RUN"
                export OUT="$TIMING_OUT"
                export SIGNATURE_OUT="$SIGNATURE_OUT"
                export ENV_OUT="$case_env"
                export MANIFEST_OUT="$case_manifest"
                "$RUN_SWEEP"
            )
        done
    } < "$CASE_MATRIX"

    echo
    echo "completed full Study wrapper"
    echo "case rows=$case_count"
    echo "implementation runs=$run_count"
    echo "timing_csv=$TIMING_OUT"
    echo "signature_csv=$SIGNATURE_OUT"
    echo "environment_file=$ENV_OUT"
    echo "case_list=$CASE_LIST_OUT"
    echo "case_files_dir=$CASE_FILES_DIR"
    echo "run_log=$RUN_LOG"
}

set +e
main "$@" 2>&1 | tee -a "$RUN_LOG"
status=${PIPESTATUS[0]}
set -e
exit "$status"
