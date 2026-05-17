#!/bin/bash
# Suite 4: memory OOM — 4x gpu_burn_lite requesting ~1.5x the mem limit.
#
# As of 2026-05-17 the default OOM behavior is native-CUDA-style:
# cuMemAlloc over the limit returns CUDA_ERROR_OUT_OF_MEMORY and the
# process exits non-zero, with no SIGKILL. PASS criteria:
#   * at least one of the 4 procs completed (exit 0)
#   * at least one proc exited with a normal (non-signal) OOM error
#   * zero SIGKILL-style exits (>128)
#
# To exercise the legacy in-library OOM killer instead, run with
# SOFTMIG_ENABLE_OOM_KILLER=1 in the environment (it will be propagated
# into the srun job below). In that mode we also expect 'ACTIVE_OOM_KILLER'
# + 'KILLED PID .* successfully' in the softmig log.

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

LEGACY_KILLER="${SOFTMIG_ENABLE_OOM_KILLER:-}"

srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=8 --mem=16G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=5
${LEGACY_KILLER:+export SOFTMIG_ENABLE_OOM_KILLER=$LEGACY_KILLER}

OUT='${OUT}'
mkdir -p \$OUT
pids=()
for i in 1 2 3 4; do
    build/test/gpu_burn_lite ${MB_PER} 15 >\"\$OUT/proc_\$i.log\" 2>&1 &
    pids+=(\"\$!\")
done
declare -A exits
for p in \"\${pids[@]}\"; do wait \$p; exits[\$p]=\$?; done
{
  for p in \"\${pids[@]}\"; do echo \"exit \$p=\${exits[\$p]}\"; done
} > \$OUT/exits.txt

if [ -r \"/var/log/softmig/\${SLURM_JOB_ID}.log\" ]; then
    cp \"/var/log/softmig/\${SLURM_JOB_ID}.log\" '${OUT}/softmig.log' 2>/dev/null || true
fi
echo \$SLURM_JOB_ID > '${OUT}/jid.txt'
" >/dev/null 2>&1

jid=$(cat "$OUT/jid.txt" 2>/dev/null || echo NA)
slog="$OUT/softmig.log"
exits_file="$OUT/exits.txt"

if [ ! -s "$exits_file" ]; then
    _emit "$jid" FAIL 0 "no exit data"
    exit 0
fi

success=0; oom=0; signaled=0
while read -r line; do
    rc=${line##*=}
    if [ "$rc" = "0" ]; then success=$((success+1));
    elif [ "$rc" -gt 128 ] 2>/dev/null; then signaled=$((signaled+1));
    else oom=$((oom+1));
    fi
done < "$exits_file"

metric="ok=${success} oom=${oom} signaled=${signaled}"

if [ -n "$LEGACY_KILLER" ]; then
    # Legacy path: expect killer firing + at least one SIGKILL exit
    oom_detect=$(grep -c "Device 0 OOM " "$slog" 2>/dev/null || echo 0)
    killer_fired=$(grep -c "ACTIVE_OOM_KILLER" "$slog" 2>/dev/null || echo 0)
    kills=$(grep -c "KILLED PID [0-9]\+ successfully" "$slog" 2>/dev/null || echo 0)
    metric="$metric oom_evt=${oom_detect} killer=${killer_fired} kills=${kills}"
    if [ "$signaled" -ge 1 ] && [ "$kills" -ge 1 ]; then
        _emit "$jid" PASS "$signaled" "$metric (legacy killer mode)"
    else
        _emit "$jid" FAIL "$signaled" "$metric (legacy killer expected)"
    fi
else
    # Default path: native CUDA OOM, no signals
    if [ "$signaled" = "0" ] && [ "$success" -ge 1 ] && [ "$oom" -ge 1 ]; then
        _emit "$jid" PASS "$oom" "$metric"
    else
        _emit "$jid" FAIL "$oom" "$metric"
    fi
fi
