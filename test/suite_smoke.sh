#!/bin/bash
# Suite 1: smoke — runtime_hold via cudart.
# PASS iff softmig.log shows Initializing, config read, and set_task_pid: Found current.

SUITE=smoke
. "$(dirname "$0")/suite_common.sh"

srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=4 --mem=4G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=5
./build/test/runtime_hold 256 6 > '${OUT}/run.out' 2>&1
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/softmig.log'
echo \$SLURM_JOB_ID > '${OUT}/jid.txt'
" >/dev/null 2>&1

jid=$(cat "$OUT/jid.txt" 2>/dev/null || echo NA)
slog="$OUT/softmig.log"

if [ ! -s "$slog" ]; then
    _emit "$jid" FAIL 0 "no softmig log produced"
    exit 0
fi

init_ok=$(grep -c "Initializing\.\.\.\.\." "$slog")
reg_ok=$(grep -c "set_task_pid: Found current process PID" "$slog")
err_cnt=$(grep -c "softmig ERROR" "$slog")
unhooked=$(grep -c "UNHOOKED" "$slog")

if [ "${SLICE}" = "l40s" ]; then
    # Full-GPU passive job: no config file. The library must decide "passive"
    # and then do nothing at all: no init, no PID registration, no shared
    # region (those lines appearing means SoftMig state was touched).
    cfg_ok=$(grep -c "softmig disabled (passive mode)" "$slog")
    metric="passive=${cfg_ok} init=${init_ok} reg=${reg_ok} err=${err_cnt} unhooked=${unhooked}"
    if [ "$cfg_ok" -ge 1 ] && [ "$init_ok" -eq 0 ] && [ "$reg_ok" -eq 0 ] && [ "$unhooked" -eq 0 ]; then
        _emit "$jid" PASS "$cfg_ok" "$metric"
    else
        _emit "$jid" FAIL "$cfg_ok" "$metric"
    fi
    exit 0
fi

cfg_ok=$(grep -c "Read CUDA_DEVICE_MEMORY_LIMIT=" "$slog")
metric="init=${init_ok} cfg=${cfg_ok} reg=${reg_ok} err=${err_cnt} unhooked=${unhooked}"

if [ "$init_ok" -ge 1 ] && [ "$cfg_ok" -ge 1 ] && [ "$reg_ok" -ge 1 ] && [ "$unhooked" -eq 0 ]; then
    _emit "$jid" PASS "$reg_ok" "$metric"
else
    _emit "$jid" FAIL "$reg_ok" "$metric"
fi
