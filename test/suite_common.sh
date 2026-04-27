#!/bin/bash
# Shared helpers for SoftMig matrix suites.
# Each caller sets SLICE (e.g. l40s.2), CUDA_VER (e.g. 12.2), OUT (dir).
#
# Every suite prints ONE TSV line to stdout:
#   CUDA_VER \t SLICE \t SUITE \t JOBID \t STATUS \t METRIC \t DETAIL
#
# and stores artifacts (per-proc logs, softmig.log, nvsmi.log) in OUT/.

set -u
: "${SLICE:?SLICE is required}"
: "${CUDA_VER:?CUDA_VER is required}"
: "${OUT:?OUT is required}"
: "${SUITE:?SUITE is required}"

SOFTMIG_ROOT=/scratch/rahimk/SoftMig
mkdir -p "$OUT"

_emit() {
    local jobid="$1" status="$2" metric="$3" detail="$4"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$CUDA_VER" "$SLICE" "$SUITE" "$jobid" "$status" "$metric" "$detail"
}

# Copy softmig log off /var/log while still inside the job (only the owning uid
# can read it). Caller passes $SLURM_JOB_ID as $1 and the target path as $2.
_copy_softmig_log() {
    local jid="$1" dest="$2"
    if [ -r "/var/log/softmig/${jid}.log" ]; then
        cp "/var/log/softmig/${jid}.log" "$dest" 2>/dev/null || true
    fi
}
export -f _copy_softmig_log

# Seconds of wall-clock slack we give srun for small tests.
DEFAULT_SRUN_TIME="${DEFAULT_SRUN_TIME:-00:05:00}"
