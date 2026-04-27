#!/bin/bash
# SoftMig multi-CUDA robustness matrix orchestrator.
#
# Usage: test/run_matrix.sh
# Outputs everything under test_results/matrix_<ts>/
#
# Matrix:
#   CUDA versions : 12.2 12.6 12.9 13.2  (11.8 unavailable under StdEnv/2023)
#   Slices        : l40s.2 l40s.4
#   Suites / ver  : smoke direct sm oom crossjob
#   One-offs      : mixed (12.2+13.2), soak (.4 + 12.2), nvsmi (.4 + 12.2)
#
# Fail-open: a failing suite just logs its result and the matrix continues.

set -u
SOFTMIG_ROOT=/scratch/rahimk/SoftMig
cd "$SOFTMIG_ROOT"

# Optional extra srun flags (e.g. export SRUN_EXTRA='--partition=gpubase_bygpu_b1'
# if GPU jobs would otherwise pend with ReqNodeNotAvail).
export SRUN_EXTRA="${SRUN_EXTRA:-}"

# Preflight: ensure a GPU step can start soon (reservation node is up).
# Set SOFTMIG_PREFLIGHT_SECS=0 to skip (jobs will queue until rack01-12 is up).
PFS="${SOFTMIG_PREFLIGHT_SECS:-120}"
if [ "${PFS}" != "0" ]; then
    echo "Preflight: waiting up to ${PFS}s for softmig GPU (l40s.4)..."
    if ! command -v timeout >/dev/null 2>&1; then
        echo "WARN: no 'timeout' command; skipping preflight"
    elif ! timeout "${PFS}" srun --reservation=softmig ${SRUN_EXTRA:-} \
            --gres=gpu:l40s.4:1 --cpus-per-task=1 --mem=1G --time=00:02:00 \
            bash -lc 'hostname' >/dev/null 2>&1; then
        echo "ERROR: no GPU step started within ${PFS}s."
        echo "  rack01-12 may be down/drained (check: sinfo -N -n rack01-12)."
        echo "  Retry when healthy, or: SOFTMIG_PREFLIGHT_SECS=0 $0   # queue and wait on each srun"
        exit 1
    fi
    echo "Preflight OK."
    echo
fi

TS=$(date +%Y%m%d_%H%M%S)
ROOT="$SOFTMIG_ROOT/test_results/matrix_${TS}"
mkdir -p "$ROOT"

TSV="$ROOT/results.tsv"
printf 'cuda_ver\tslice\tsuite\tjobid\tstatus\tmetric\tdetail\n' > "$TSV"

VERSIONS=(12.2 12.6 12.9 13.2)
SLICES=(l40s.2 l40s.4)
PER_VER_SUITES=(smoke direct sm oom crossjob)

echo "Matrix root: $ROOT"
echo "Versions   : ${VERSIONS[*]}"
echo "Slices     : ${SLICES[*]}"
echo "Suites     : ${PER_VER_SUITES[*]}"
echo "Start      : $(date)"
echo

run_suite() {
    local script="$1" outdir="$2"
    export OUT="$outdir"
    mkdir -p "$OUT"
    local line
    line=$(bash "$script" 2>/dev/null | tail -1)
    if [ -n "$line" ]; then
        echo "$line" | tee -a "$TSV"
    else
        printf '%s\t%s\t%s\tNA\tFAIL\t0\tno output from suite\n' \
            "${CUDA_VER:-NA}" "${SLICE:-NA}" "$(basename "$script" .sh | sed 's/^suite_//')" \
            | tee -a "$TSV"
    fi
}

# --- per-version x per-slice x per-suite ---
for CUDA_VER in "${VERSIONS[@]}"; do
    for SLICE in "${SLICES[@]}"; do
        echo "=== cuda/${CUDA_VER} on ${SLICE} ==="
        export CUDA_VER SLICE
        for s in "${PER_VER_SUITES[@]}"; do
            echo "  [${s}]"
            run_suite "test/suite_${s}.sh" "$ROOT/${CUDA_VER}/${SLICE}/${s}"
        done
    done
done

# --- one-offs ---
echo
echo "=== one-offs ==="
unset CUDA_VER SLICE

echo "  [mixed 12.2+13.2 on l40s.4]"
CUDA_VER=mixed SLICE=l40s.4 \
  run_suite "test/suite_mixed.sh" "$ROOT/oneoff/mixed"

echo "  [soak 5min on l40s.4 cuda/12.2]"
CUDA_VER=12.2 SLICE=l40s.4 \
  run_suite "test/suite_soak.sh" "$ROOT/oneoff/soak"

echo "  [nvsmi filter on l40s.4 cuda/12.2]"
CUDA_VER=12.2 SLICE=l40s.4 \
  run_suite "test/suite_nvsmi.sh" "$ROOT/oneoff/nvsmi"

echo
echo "End: $(date)"
echo
echo "Generating summary..."
awk -f test/summarize.awk "$TSV" > "$ROOT/SUMMARY.md"
cat "$ROOT/SUMMARY.md"
echo
echo "Full artifacts: $ROOT"
