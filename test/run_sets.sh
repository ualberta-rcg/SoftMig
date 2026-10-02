#!/bin/bash
# SoftMig 2.06 pressure / share / leak campaign: run every multi-job suite
# (share, guarantees, wrapper, passthrough, mixed pieces, gpuburn, isolation,
# bypass) as reservation jobs and collect one results.tsv + SUMMARY.md.
#
#   bash test/run_sets.sh                 # full campaign (~2 h on one node)
#   SETS=share,bypass bash test/run_sets.sh
#   SETS_VERSIONS="12.6 13.2" bash test/run_sets.sh
#
# Artifacts: test_results/sets_<ts>/<suite>/<scenario or version>/...
# Fail-open: a failing scenario is recorded and the campaign continues.

set -u
SOFTMIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SOFTMIG_ROOT"
umask 002
export SRUN_EXTRA="${SRUN_EXTRA:-}"

TS=$(date +%Y%m%d_%H%M%S)
ROOT="${SETS_ROOT:-$SOFTMIG_ROOT/test_results/sets_${TS}}"
mkdir -p "$ROOT"
echo "$ROOT" > "$SOFTMIG_ROOT/test_results/sets_latest.txt"
TSV="$ROOT/results.tsv"
printf 'cuda_ver\tslice\tsuite\tjobid\tstatus\tmetric\tdetail\n' > "$TSV"

VERSIONS=(${SETS_VERSIONS:-12.2 12.6 12.9 13.2})
SETS="${SETS:-share,guarantee,wrapper,passthrough,mixed,gpuburn,isolation,bypass}"
SECS="${SHARE_SECS:-60}"
has() { case ",$SETS," in *",$1,"*) return 0 ;; esac; return 1; }

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ROOT/run.log"; }

run_suite() {   # run_suite SUITE OUTDIR [ENV=VAL ...]
    local suite="$1" outdir="$2"; shift 2
    mkdir -p "$outdir"
    local line
    line=$(env "$@" OUT="$outdir" bash "test/suite_${suite}.sh" 2> "$outdir/suite.err" | tail -1)
    if [ -z "$line" ]; then
        line=$(printf '%s\t%s\t%s\tNA\tFAIL\t0\tno output from suite' "${CUDA_VER:-NA}" "${SLICE:-NA}" "$suite")
    fi
    echo "$line" >> "$TSV"
    log "$(echo "$line" | cut -f1-5 | tr '\t' ' ') :: $(echo "$line" | cut -f7 | cut -c1-100)"
}

log "campaign root: $ROOT   sets: $SETS   versions: ${VERSIONS[*]}   burn: ${SECS}s"
log "lib on node: $(sudo -n ssh -o BatchMode=yes rack01-11 'sha256sum /usr/local/lib/libsoftmig.so' 2>/dev/null | cut -c1-16)"
export SHARE_SECS="$SECS"

# 1. share scenarios, CUDA versions rotated across the jobs of each scenario
if has share; then
    for sc in S1 S2 S3 S4 S5 S6 S7; do
        run_suite share "$ROOT/share/$sc" SCENARIO=$sc CUDA_VER=12.6 SLICE=mixed SHARE_ROTATE=1
    done
fi
if has guarantee; then
    for sc in G1 G2 G3 G4 G5 G6; do
        run_suite share "$ROOT/share/$sc" SCENARIO=$sc CUDA_VER=12.6 SLICE=mixed SHARE_ROTATE=1
    done
fi
# 2. nvidia-smi wrapper: S1 again with the wrapper on PATH (visibility must be clean)
if has wrapper; then
    run_suite share "$ROOT/share/W1" SCENARIO=W1 CUDA_VER=12.6 SLICE=mixed SHARE_ROTATE=1
fi
# 3. passthrough: whole GPU untouched while slices are enforced
if has passthrough; then
    for v in "${VERSIONS[@]}"; do
        run_suite passthrough "$ROOT/passthrough/P1_$v" SCENARIO=P1 CUDA_VER=$v SLICE=mixed
    done
    for sc in P2 P3 P4; do
        run_suite passthrough "$ROOT/passthrough/${sc}_12.6" SCENARIO=$sc CUDA_VER=12.6 SLICE=mixed
    done
fi
# 4. mixed pieces: shapes users submit
if has mixed; then
    for sc in M1 M2 M3 M4 M5 M6; do
        run_suite mixed_pieces "$ROOT/mixed/${sc}_12.6" SCENARIO=$sc CUDA_VER=12.6 SLICE=mixed
    done
    for sc in M1 M4; do
        run_suite mixed_pieces "$ROOT/mixed/${sc}_13.2" SCENARIO=$sc CUDA_VER=13.2 SLICE=mixed
    done
fi
# 5. real gpu-burn regression
if has gpuburn; then
    for v in "${VERSIONS[@]}"; do
        run_suite gpuburn "$ROOT/gpuburn/B1_$v" SCENARIO=B1 CUDA_VER=$v SLICE=mixed
    done
    for sc in B2 B3 B4 B5; do
        run_suite gpuburn "$ROOT/gpuburn/${sc}_12.6" SCENARIO=$sc CUDA_VER=12.6 SLICE=mixed
    done
    run_suite gpuburn "$ROOT/gpuburn/B3_13.2" SCENARIO=B3 CUDA_VER=13.2 SLICE=mixed
fi
# 6. isolation: accounting leak / kill scope / foreign uid / regions
if has isolation; then
    for v in "${VERSIONS[@]}"; do
        run_suite isolation "$ROOT/isolation/$v" CUDA_VER=$v SLICE=l40s.4
    done
fi
# 7. bypass hunt: every allocation API, enforced and passive
if has bypass; then
    for v in "${VERSIONS[@]}"; do
        run_suite bypass "$ROOT/bypass/q_$v" CUDA_VER=$v SLICE=l40s.4
    done
    run_suite bypass "$ROOT/bypass/whole_12.6" CUDA_VER=12.6 SLICE=l40s
    run_suite bypass "$ROOT/bypass/whole_13.2" CUDA_VER=13.2 SLICE=l40s
fi

python3 test/sets_summary.py "$ROOT" > "$ROOT/SUMMARY.md"
cat "$ROOT/SUMMARY.md"
log "done: $ROOT"
