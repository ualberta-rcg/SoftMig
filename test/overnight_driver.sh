#!/bin/bash
# Overnight cycle driver. Runs INSIDE a CPU-only sbatch job on the reservation
# node (submitted by test/run_overnight.sh) so no test logic runs on the login
# node; every GPU test is still its own Slurm job (srun with SLURM_* unset).
#
# Env from the launcher: ROOT HOURS SAMPLER_SHARED SOFTMIG_ROOT
# One cycle (~45 min): legacy per-version checks, share/guarantee scenarios,
# passthrough, mixed pieces, gpu-burn, isolation, a soak, storm A and B.
set -u
cd "${SOFTMIG_ROOT:?}" || exit 1
: "${ROOT:?}"; : "${HOURS:?}"
umask 002

# Steps must become new jobs, not steps of this allocation.
for v in $(env | grep -oE '^SLURM_[A-Z_]+' ); do unset "$v"; done
export SRUN_EXTRA="${SRUN_EXTRA:-}"
export SHARE_SECS="${SHARE_SECS:-60}"

RESULTS="$ROOT/results.tsv"
[ -s "$RESULTS" ] || printf 'cycle\tcuda_ver\tslice\tsuite\tjobid\tstatus\tmetric\tdetail\n' > "$RESULTS"
END_EPOCH=$(( $(date +%s) + HOURS * 3600 ))
CYCLE=0
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$ROOT/phase.log"; }

run_suite() {   # run_suite SUITE OUTDIR [ENV=VAL ...]
    local suite="$1" outdir="$2"; shift 2
    mkdir -p "$outdir"
    local line
    line=$(env "$@" OUT="$outdir" bash "test/suite_${suite}.sh" 2> "$outdir/suite.err" | tail -1)
    [ -n "$line" ] || line=$(printf 'NA\tNA\t%s\tNA\tFAIL\t0\tno output from suite' "$suite")
    printf '%s\t%s\n' "$CYCLE" "$line" >> "$RESULTS"
    log "c$CYCLE $(echo "$line" | cut -f1-5 | tr '\t' ' ') :: $(echo "$line" | cut -f7 | cut -c1-90)"
}

log "driver job ${SLURM_JOB_ID:-?} start; hours=$HOURS root=$ROOT sampler=${SAMPLER_SHARED:-none}"
while [ "$(date +%s)" -lt "$END_EPOCH" ]; do
    CYCLE=$((CYCLE + 1)); C="$ROOT/cycle_$CYCLE"; mkdir -p "$C"
    log "=== cycle $CYCLE start ==="

    # legacy single-job checks, every CUDA version
    for v in 12.2 12.6 12.9 13.2; do
        run_suite direct "$C/direct_$v" CUDA_VER=$v SLICE=l40s.4
        run_suite oom    "$C/oom_$v"    CUDA_VER=$v SLICE=l40s.2
        run_suite sm     "$C/sm_$v"     CUDA_VER=$v SLICE=l40s.2
    done
    # share / guarantees (CUDA versions rotate across the jobs)
    for sc in S1 S2 S3 S7 G2 G3 W1; do
        run_suite share "$C/share_$sc" SCENARIO=$sc CUDA_VER=12.6 SLICE=mixed SHARE_ROTATE=1
    done
    run_suite passthrough  "$C/passthrough_P1" SCENARIO=P1 CUDA_VER=12.6 SLICE=mixed
    run_suite passthrough  "$C/passthrough_P4" SCENARIO=P4 CUDA_VER=13.2 SLICE=mixed
    run_suite mixed_pieces "$C/mixed_M1" SCENARIO=M1 CUDA_VER=12.6 SLICE=mixed
    run_suite mixed_pieces "$C/mixed_M4" SCENARIO=M4 CUDA_VER=12.9 SLICE=mixed
    run_suite gpuburn      "$C/gpuburn_B1" SCENARIO=B1 CUDA_VER=12.6 SLICE=mixed
    run_suite gpuburn      "$C/gpuburn_B3" SCENARIO=B3 CUDA_VER=13.2 SLICE=mixed
    run_suite isolation    "$C/isolation" CUDA_VER=12.6 SLICE=l40s.4
    run_suite soak         "$C/soak" CUDA_VER=12.2 SLICE=l40s.4
    # storms
    run_suite storm "$C/storm_A" ROUND=A STORM_MINUTES=8 STORM_PAR=10 STORM_SEED=$CYCLE CUDA_VER=mixed SLICE=mixed
    run_suite storm "$C/storm_B" ROUND=B STORM_MINUTES=5 CUDA_VER=mixed SLICE=l40s.4

    { echo "cycle=$CYCLE $(date)"
      awk -F'\t' -v c="$CYCLE" 'NR>1 && $1==c {cnt[$6]++} END{for (k in cnt) printf "%s=%d ", k, cnt[k]; print ""}' "$RESULTS"
    } >> "$ROOT/cycle_summary.txt"
    log "=== cycle $CYCLE end ==="
done
python3 test/sets_summary.py "$ROOT" > "$ROOT/SUMMARY.md" 2>/dev/null
log "driver done after $CYCLE cycles"
