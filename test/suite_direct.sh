#!/bin/bash
# Suite 2: direct-linked — 4x nvml_probe (linked -lcuda -lnvidia-ml).
# PASS iff all 4 launched PIDs show up in either set_task_pid:Found or
# cgroup_check=1 entries in softmig.log (proves ELF interposition worked
# against that driver/runtime combo).

SUITE=direct
# Direct-linked probes can stall on driver init races; give srun headroom.
DEFAULT_SRUN_TIME=00:10:00
. "$(dirname "$0")/suite_common.sh"

srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=8 --mem=8G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=5
N=4 MB=384 HOLD=12 OUT='${OUT}' test/run_multiproc.sh >/dev/null 2>&1
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/softmig.log'
echo \$SLURM_JOB_ID > '${OUT}/jid.txt'
" >/dev/null 2>&1

jid=$(cat "$OUT/jid.txt" 2>/dev/null || echo NA)
slog="$OUT/softmig.log"

if [ ! -s "$slog" ]; then
    _emit "$jid" FAIL 0 "no softmig log"
    exit 0
fi

# Launched PIDs recorded by nvml_probe's own stdout logs (harness writes proc_N.log)
launched=$(grep -hoE "^\[pid=[0-9]+" "$OUT"/proc_*.log 2>/dev/null \
           | grep -oE "[0-9]+" | sort -u)
n_launched=$(printf '%s\n' "$launched" | grep -c '^[0-9]')

# PIDs softmig actually registered
regged=$(grep -oE "Found current process PID [0-9]+|cgroup_check=1 .*PID [0-9]+" "$slog" \
         | grep -oE "[0-9]+" | sort -u)

hits=0
for p in $launched; do
    if printf '%s\n' "$regged" | grep -qx "$p"; then
        hits=$((hits+1))
    fi
done

err_cnt=$(grep -c "softmig ERROR" "$slog")
metric="launched=${n_launched} registered=${hits} err=${err_cnt}"

if [ "$n_launched" -ge 1 ] && [ "$hits" -eq "$n_launched" ]; then
    _emit "$jid" PASS "$hits/$n_launched" "$metric"
elif [ "$hits" -gt 0 ]; then
    _emit "$jid" PARTIAL "$hits/$n_launched" "$metric"
else
    _emit "$jid" FAIL "$hits/$n_launched" "$metric"
fi
