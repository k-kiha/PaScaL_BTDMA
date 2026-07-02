#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRY_RUN=1

usage() {
    cat <<'USAGE'
Usage:
  scripts/clean_for_sync.sh          # dry-run: show what would be removed
  scripts/clean_for_sync.sh --apply  # actually remove generated files

Purpose:
  Clean generated build/run/profiler/output files before git pull/push.

Preserved:
  - all *.csv files
  - profile_results/ directories and their contents
  - source files, scripts, Makefiles, README files, and git metadata
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
elif [[ "${1:-}" == "--apply" ]]; then
    DRY_RUN=0
elif [[ $# -gt 0 ]]; then
    usage >&2
    exit 2
fi

remove_path() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        return
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '[dry-run] remove %s\n' "${path#$ROOT/}"
    else
        rm -rf "$path"
        printf 'removed %s\n' "${path#$ROOT/}"
    fi
}

clean_generated_files_under() {
    local base="$1"
    if [[ ! -e "$base" ]]; then
        return
    fi

    while IFS= read -r -d '' path; do
        remove_path "$path"
    done < <(
        find "$base" \
            -path "$ROOT/.git" -prune -o \
            -path "$ROOT/porting_c_btdma/profile_results" -prune -o \
            -path "$ROOT/profile_results" -prune -o \
            -type f \( \
                -name '*.o' -o \
                -name '*.mod' -o \
                -name '*.a' -o \
                -name '*.out' -o \
                -name 'a.out' -o \
                -name '*.exe' -o \
                -name '*.so' -o \
                -name '*.dylib' -o \
                -name '*.nsys-rep' -o \
                -name '*.qdrep' -o \
                -name '*.ncu-rep' -o \
                -name '*.sqlite' -o \
                -name '*.log' -o \
                -name '*.err' -o \
                -name '*.vtk' -o \
                -name '*.vtr' -o \
                -name '*.vtu' -o \
                -name '*.pvtr' -o \
                -name '*.pvtu' -o \
                -name '*.pvd' -o \
                -name '*.dat' -o \
                -name '*.bin' -o \
                -name '*.plt' -o \
                -name 'plot1d_*.txt' -o \
                -name 'output_*.txt' -o \
                -name '.DS_Store' \
            \) ! -name '*.csv' -print0
    )
}

remove_tree_or_clean_contents() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        return
    fi

    if [[ -d "$path" ]] && [[ -n "$(find "$path" -type f -name '*.csv' -print -quit)" ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            printf '[dry-run] preserve %s because it contains csv files\n' "${path#$ROOT/}"
        else
            printf 'preserve %s because it contains csv files\n' "${path#$ROOT/}"
        fi
        clean_generated_files_under "$path"
    else
        remove_path "$path"
    fi
}

remove_tree_or_clean_contents "$ROOT/build"
remove_tree_or_clean_contents "$ROOT/include"
remove_tree_or_clean_contents "$ROOT/lib"
remove_tree_or_clean_contents "$ROOT/porting_c_btdma/build"
remove_tree_or_clean_contents "$ROOT/porting_c_btdma/lib"
remove_path "$ROOT/porting_c_btdma/run/ex_btdma_noncyclic"
clean_generated_files_under "$ROOT"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "dry-run only. Re-run with --apply to remove these files."
fi
