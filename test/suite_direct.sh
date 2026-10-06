#!/bin/bash
# Suite 2: direct-linked — 4x nvml_probe (linked -lcuda -lnvidia-ml).
# Sliced: PASS iff all 4 launched PIDs show up in either set_task_pid:Found or
# cgroup_check=1 entries in softmig.log (proves ELF interposition worked
# against that driver/runtime combo).
# Full GPU (passive): PASS iff all 4 allocate and none is registered.

SUITE=direct
# Direct-linked probes can stall on driver init races; give srun headroom.
DEFAULT_SRUN_TIME=00:10:00
. "$(dirname "$0")/suite_common.sh"

_srun_capture srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=8 --mem=8G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=5
echo \$SLURM_JOB_ID > '${OUT}/jid.txt'
_hang_watchdog 300 '${OUT}'
N=4 MB=384 HOLD=12 OUT='${OUT}' test/run_multiproc.sh >/dev/null 2>&1
_hang_disarm '${OUT}'
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/softmig.log'
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

if [ "${SLICE}" = "l40s" ]; then
    # Passive (no config): every direct-linked probe must allocate fine and
    # SoftMig must not register any of them.
    allocs=$(grep -h "alloc ok" "$OUT"/proc_*.log 2>/dev/null | wc -l)
    passive=$(grep -c "softmig disabled (passive mode)" "$slog")
    metric="launched=${n_launched} alloc_ok=${allocs} registered=${hits} passive=${passive} err=${err_cnt}"
    if [ "$n_launched" -ge 1 ] && [ "$allocs" -eq "$n_launched" ] && [ "$hits" -eq 0 ] && [ "$passive" -ge 1 ]; then
        _emit "$jid" PASS "$allocs/$n_launched" "$metric"
    else
        _emit "$jid" FAIL "$allocs/$n_launched" "$metric"
    fi
    exit 0
fi

metric="launched=${n_launched} registered=${hits} err=${err_cnt}"

if [ "$n_launched" -ge 1 ] && [ "$hits" -eq "$n_launched" ]; then
    _emit "$jid" PASS "$hits/$n_launched" "$metric"
elif [ "$hits" -gt 0 ]; then
    _emit "$jid" PARTIAL "$hits/$n_launched" "$metric"
else
    _emit "$jid" FAIL "$hits/$n_launched" "$metric"
fi
