#!/bin/bash
# Suite 4: memory OOM — 4x gpu_burn_lite requesting ~1.5x the mem limit.
# PASS iff softmig.log shows 'Device 0 OOM' + 'ACTIVE_OOM_KILLER' + at least
# one 'KILLED PID .* successfully'.

SUITE=oom
. "$(dirname "$0")/suite_common.sh"

# Pick MB per proc so aggregate is ~1.5x slice limit. Rough heuristic:
#   .2 (~24GB limit): 4 * 8GB = 32GB
#   .4 (~11.5GB limit): 4 * 4GB = 16GB
case "$SLICE" in
    *\.2*|*l40s.2*) MB_PER=8192 ;;
    *\.4*|*l40s.4*) MB_PER=4096 ;;
    *)              MB_PER=4096 ;;
esac

srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=8 --mem=16G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=5
N=4 MB=${MB_PER} DUR=15 OUT='${OUT}' test/run_burn.sh >/dev/null 2>&1
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/softmig.log'
echo \$SLURM_JOB_ID > '${OUT}/jid.txt'
" >/dev/null 2>&1

jid=$(cat "$OUT/jid.txt" 2>/dev/null || echo NA)
slog="$OUT/softmig.log"

if [ ! -s "$slog" ]; then
    _emit "$jid" FAIL 0 "no softmig log"
    exit 0
fi

oom_detect=$(grep -c "Device 0 OOM " "$slog")
killer_fired=$(grep -c "ACTIVE_OOM_KILLER" "$slog")
kills=$(grep -c "KILLED PID [0-9]\+ successfully" "$slog")
reg_cnt=$(grep -c "set_task_pid: Found current process PID" "$slog")

metric="oom=${oom_detect} killer=${killer_fired} kills=${kills} reg=${reg_cnt}"

if [ "$oom_detect" -ge 1 ] && [ "$killer_fired" -ge 1 ] && [ "$kills" -ge 1 ]; then
    _emit "$jid" PASS "$kills" "$metric"
else
    _emit "$jid" FAIL "$kills" "$metric"
fi
