#!/bin/bash
# Suite 3: SM enforcement — 3x gpu_burn_lite.
# PASS iff nvidia-smi-sampled GPU util averages within +/-10pp of configured
# CUDA_DEVICE_SM_LIMIT.

SUITE=sm
. "$(dirname "$0")/suite_common.sh"

srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=8 --mem=8G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=5
N=3 MB=1024 DUR=20 OUT='${OUT}' test/run_burn.sh >/dev/null 2>&1
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/softmig.log'
cat /var/run/softmig/\$SLURM_JOB_ID.conf > '${OUT}/softmig.conf' 2>/dev/null
echo \$SLURM_JOB_ID > '${OUT}/jid.txt'
" >/dev/null 2>&1

jid=$(cat "$OUT/jid.txt" 2>/dev/null || echo NA)
slog="$OUT/softmig.log"
nvl="$OUT/nvsmi.log"
conf="$OUT/softmig.conf"

if [ ! -s "$nvl" ] || [ ! -s "$slog" ]; then
    _emit "$jid" FAIL 0 "no nvsmi/softmig log"
    exit 0
fi

# Configured SM limit
sm_limit=$(grep -oE "CUDA_DEVICE_SM_LIMIT=[0-9]+" "$conf" 2>/dev/null | cut -d= -f2)
[ -z "$sm_limit" ] && sm_limit=0

# Average observed GPU util (skip zero-samples which are just pre-start)
avg=$(grep -E "^[0-9]+ %" "$nvl" | awk -F% 'BEGIN{s=0;n=0} {v=$1+0; if(v>0){s+=v;n++}} END{if(n>0) printf "%.1f", s/n; else print "0"}')

err_cnt=$(grep -c "softmig ERROR" "$slog")
reg_cnt=$(grep -c "set_task_pid: Found current process PID" "$slog")

if [ "$sm_limit" -gt 0 ] && [ "$sm_limit" -le 100 ]; then
    diff=$(awk -v a="$avg" -v b="$sm_limit" 'BEGIN{d=a-b; if(d<0)d=-d; print d}')
    close=$(awk -v d="$diff" 'BEGIN{print (d<10)?"1":"0"}')
    metric="avg_util=${avg}% target=${sm_limit}% reg=${reg_cnt}"
    if [ "$close" = "1" ] && [ "$reg_cnt" -ge 3 ]; then
        _emit "$jid" PASS "$avg" "$metric"
    else
        _emit "$jid" FAIL "$avg" "$metric"
    fi
else
    _emit "$jid" SKIP "$avg" "no sm_limit configured"
fi
