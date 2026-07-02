#!/usr/bin/env bash
set -euo pipefail

MPIRUN=${MPIRUN:-mpirun}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FORTRAN_EXE=${FORTRAN_EXE:-"$SCRIPT_DIR/example_fortran_btdma_profile"}
CXX_EXE=${CXX_EXE:-"$SCRIPT_DIR/example_cuda_cxx_btdma_profile"}

STUDY_PRESET=${STUDY_PRESET:-quick}
ITERATIONS=${ITERATIONS:-10}
BASELINE_NP=${BASELINE_NP:-2}
SCALING_NP_LIST=${SCALING_NP_LIST:-"2 4 8"}

MPI_MODE=${MPI_MODE:-device}
CXX_DEFAULT_MPI_MODES=${CXX_DEFAULT_MPI_MODES:-"$MPI_MODE"}
MPI_MODE_LIST=${MPI_MODE_LIST:-"device host"}

RUN_FORTRAN=${RUN_FORTRAN:-1}
RUN_CXX=${RUN_CXX:-1}
DRY_RUN=${DRY_RUN:-0}
TIMESTAMP=${TIMESTAMP:-$(date +%y%m%d_%H%M%S)}
OUT=${OUT:-"$SCRIPT_DIR/btdma_total_profile_${TIMESTAMP}.csv"}
SIGNATURE_OUT=${SIGNATURE_OUT:-"$SCRIPT_DIR/btdma_solution_signature_${TIMESTAMP}.csv"}
ENV_OUT=${ENV_OUT:-"$SCRIPT_DIR/btdma_environment_${TIMESTAMP}.txt"}
MANIFEST_OUT=${MANIFEST_OUT:-"$SCRIPT_DIR/btdma_case_manifest_${TIMESTAMP}.csv"}

NP_LIST=${NP_LIST:-"2"}
SIZE_LIST=${SIZE_LIST:-"32,32,128"}
M_LIST=${M_LIST:-"5"}

mkdir -p "$(dirname "$OUT")"
mkdir -p "$(dirname "$SIGNATURE_OUT")"
mkdir -p "$(dirname "$ENV_OUT")"
mkdir -p "$(dirname "$MANIFEST_OUT")"

EXEC_CASES="$(mktemp "${TMPDIR:-/tmp}/btdma_exec_cases.XXXXXX")"
trap 'rm -f "$EXEC_CASES"' EXIT

usage() {
    cat <<'USAGE'
Usage:
  ./run_study_sweep.sh
  STUDY_PRESET=quick ./run_study_sweep.sh
  STUDY_PRESET=custom NP_LIST="2 4" SIZE_LIST="32,32,128" M_LIST="3 5" ./run_study_sweep.sh

Study presets:
  quick      Small smoke matrix for matched non-cyclic BTDMA profiling.
  portfolio  Broader first-pass matrix with rank, nrow, nsys, m, and MPI-mode axes.
  custom     Direct NP_LIST x SIZE_LIST x M_LIST execution.

Important variables:
  ITERATIONS=10
  CXX_DEFAULT_MPI_MODES="device"
  MPI_MODE_LIST="device host"    # used only by mpi_mode_compare cases
  RUN_FORTRAN=1
  RUN_CXX=1
  DRY_RUN=1                      # write manifest/environment, print commands only
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

append_timing_csv() {
    if [[ ! -s "$OUT" ]]; then
        tee -a "$OUT"
    else
        awk 'NR == 1 && /^solver,/ { next } { print }' | tee -a "$OUT"
    fi
}

manifest_header() {
    cat > "$MANIFEST_OUT" <<'EOF'
study_suite,case_id,variant,nranks,n1,n2,n3,m,baseline_nranks,scaling_kind,cxx_mpi_modes,notes
EOF
}

add_case() {
    local suite="$1"
    local np="$2"
    local n1="$3"
    local n2="$4"
    local n3="$5"
    local m="$6"
    local scaling_kind="$7"
    local cxx_modes="$8"
    local notes="$9"
    local case_id="${suite}_np${np}_${n1}x${n2}x${n3}_m${m}"
    local mode

    printf '%s,%s,noncyclic,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$suite" "$case_id" "$np" "$n1" "$n2" "$n3" "$m" \
        "$BASELINE_NP" "$scaling_kind" "$cxx_modes" "$notes" >> "$MANIFEST_OUT"

    if [[ "$RUN_FORTRAN" == "1" ]]; then
        printf 'fortran-original,device,%s,%s,%s,%s,%s\n' "$np" "$n1" "$n2" "$n3" "$m" >> "$EXEC_CASES"
    fi

    if [[ "$RUN_CXX" == "1" ]]; then
        for mode in $cxx_modes; do
            printf 'cuda-cxx,%s,%s,%s,%s,%s,%s\n' "$mode" "$np" "$n1" "$n2" "$n3" "$m" >> "$EXEC_CASES"
        done
    fi
}

add_size_case() {
    local suite="$1"
    local np="$2"
    local size="$3"
    local m="$4"
    local scaling_kind="$5"
    local cxx_modes="$6"
    local notes="$7"
    local n1 n2 n3

    IFS=',' read -r n1 n2 n3 <<< "$size"
    add_case "$suite" "$np" "$n1" "$n2" "$n3" "$m" "$scaling_kind" "$cxx_modes" "$notes"
}

build_quick_cases() {
    add_size_case "single_gpu_reference" 1 "32,32,128" 5 "reference" \
        "$CXX_DEFAULT_MPI_MODES" "local_btdma_no_mpi_small"
    add_size_case "strong_scaling" 2 "32,32,256" 5 "strong_2gpu_baseline" \
        "$CXX_DEFAULT_MPI_MODES" "fixed_global_problem_small"
    add_size_case "m_sensitivity" 2 "32,32,256" 3 "m_sweep" \
        "$CXX_DEFAULT_MPI_MODES" "same_grid_smaller_block"
    add_size_case "mpi_mode_compare" 2 "32,32,256" 5 "mpi_mode" \
        "$MPI_MODE_LIST" "cxx_device_vs_host_for_same_case"
}

build_portfolio_cases() {
    local np

    add_size_case "single_gpu_reference" 1 "32,32,128" 3 "reference" \
        "$CXX_DEFAULT_MPI_MODES" "local_btdma_m3"
    add_size_case "single_gpu_reference" 1 "32,32,128" 5 "reference" \
        "$CXX_DEFAULT_MPI_MODES" "local_btdma_m5"

    for np in $SCALING_NP_LIST; do
        add_size_case "strong_scaling" "$np" "64,64,1024" 5 "strong_2gpu_baseline" \
            "$CXX_DEFAULT_MPI_MODES" "fixed_global_problem"
        add_size_case "nrow_sensitivity" "$np" "64,64,512" 5 "nrow_sweep" \
            "$CXX_DEFAULT_MPI_MODES" "nsys_fixed_vary_n3"
        add_size_case "nrow_sensitivity" "$np" "64,64,2048" 5 "nrow_sweep" \
            "$CXX_DEFAULT_MPI_MODES" "nsys_fixed_vary_n3"
        add_size_case "m_sensitivity" "$np" "64,64,1024" 3 "m_sweep" \
            "$CXX_DEFAULT_MPI_MODES" "same_grid_m3"
    done

    add_size_case "nsys_sensitivity" 2 "32,32,1024" 5 "nsys_sweep" \
        "$CXX_DEFAULT_MPI_MODES" "n3_fixed_vary_nsys"
    add_size_case "nsys_sensitivity" 2 "64,64,1024" 5 "nsys_sweep" \
        "$CXX_DEFAULT_MPI_MODES" "n3_fixed_vary_nsys"

    for np in $SCALING_NP_LIST; do
        add_size_case "mpi_mode_compare" "$np" "64,64,1024" 5 "mpi_mode" \
            "$MPI_MODE_LIST" "cxx_device_vs_host_for_same_case"
    done
}

build_custom_cases() {
    local np size m
    for np in $NP_LIST; do
        for size in $SIZE_LIST; do
            for m in $M_LIST; do
                add_size_case "custom" "$np" "$size" "$m" "custom" \
                    "$CXX_DEFAULT_MPI_MODES" "user_supplied_case"
            done
        done
    done
}

capture_environment() {
    {
        echo "# PaScaL_BTDMA Study Environment"
        echo "date=$(date '+%Y-%m-%dT%H:%M:%S%z')"
        echo "hostname=$(hostname)"
        echo "pwd=$PWD"
        echo "root_dir=$ROOT_DIR"
        echo "script_dir=$SCRIPT_DIR"
        echo "study_preset=$STUDY_PRESET"
        echo "baseline_np=$BASELINE_NP"
        echo "scaling_np_list=$SCALING_NP_LIST"
        echo "custom_np_list=$NP_LIST"
        echo "custom_size_list=$SIZE_LIST"
        echo "custom_m_list=$M_LIST"
        echo "iterations=$ITERATIONS"
        echo "cxx_default_mpi_modes=$CXX_DEFAULT_MPI_MODES"
        echo "mpi_mode_list=$MPI_MODE_LIST"
        echo "run_fortran=$RUN_FORTRAN"
        echo "run_cxx=$RUN_CXX"
        echo "dry_run=$DRY_RUN"
        echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
        echo "timing_csv=$OUT"
        echo "signature_csv=$SIGNATURE_OUT"
        echo "environment_file=$ENV_OUT"
        echo "case_manifest=$MANIFEST_OUT"
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

manifest_header
case "$STUDY_PRESET" in
    quick)
        build_quick_cases
        ;;
    portfolio)
        build_portfolio_cases
        ;;
    custom)
        build_custom_cases
        ;;
    *)
        echo "error: unknown STUDY_PRESET=$STUDY_PRESET" >&2
        usage >&2
        exit 2
        ;;
esac

capture_environment

sort -u "$EXEC_CASES" | while IFS=, read -r implementation mode np n1 n2 n3 m; do
    echo "running implementation=$implementation mpi_mode=$mode np=$np n1=$n1 n2=$n2 n3=$n3 m=$m iterations=$ITERATIONS" >&2

    if [[ "$implementation" == "fortran-original" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[dry-run] $MPIRUN -np $np $FORTRAN_EXE $n1 $n2 $n3 $m $ITERATIONS" >&2
            continue
        fi
        PASCAL_BTDMA_SIGNATURE_OUT="$SIGNATURE_OUT" \
        "$MPIRUN" -np "$np" "$FORTRAN_EXE" "$n1" "$n2" "$n3" "$m" "$ITERATIONS" \
            | append_timing_csv
    elif [[ "$implementation" == "cuda-cxx" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[dry-run] PASCAL_BTDMA_MPI_MODE=$mode $MPIRUN -np $np $CXX_EXE $n1 $n2 $n3 $m $ITERATIONS" >&2
            continue
        fi
        PASCAL_BTDMA_SIGNATURE_OUT="$SIGNATURE_OUT" \
        PASCAL_BTDMA_MPI_MODE="$mode" \
        "$MPIRUN" -np "$np" "$CXX_EXE" "$n1" "$n2" "$n3" "$m" "$ITERATIONS" \
            | append_timing_csv
    else
        echo "error: unknown implementation=$implementation" >&2
        exit 2
    fi
done

echo "wrote $OUT" >&2
echo "wrote $SIGNATURE_OUT" >&2
echo "wrote $ENV_OUT" >&2
echo "wrote $MANIFEST_OUT" >&2
