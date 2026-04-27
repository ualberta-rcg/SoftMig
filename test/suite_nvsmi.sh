#!/bin/bash
# Suite 8: nvidia-smi cgroup filtering — run nvidia-smi from inside a job and
# verify it shows ONLY our own process(es), not whatever else is on the GPU.
# This validates that our hook of nvmlDeviceGetComputeRunningProcesses_v2
# correctly filters by cgroup.

SUITE=nvsmi
SLICE=${SLICE:-l40s.4}
CUDA_VER=${CUDA_VER:-12.2}
. "$(dirname "$0")/suite_common.sh"

srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=4 --mem=4G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=4

# Hold memory in background
./build/test/runtime_hold 512 25 > '${OUT}/hold.log' 2>&1 &
HOLD_PID=\$!
sleep 6

# Sample nvidia-smi --query-compute-apps once; filter out any softmig log
# noise that appears on stderr.
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null > '${OUT}/nvsmi_pids.txt'
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null > '${OUT}/nvsmi_apps.txt'

echo \$HOLD_PID > '${OUT}/hold_pid.txt'
# Also capture our own job's cgroup pid list for comparison
cat /proc/\$HOLD_PID/cgroup > '${OUT}/hold_cgroup.txt' 2>/dev/null
wait \$HOLD_PID 2>/dev/null
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/softmig.log'
echo \$SLURM_JOB_ID > '${OUT}/jid.txt'
" >/dev/null 2>&1

jid=$(cat "$OUT/jid.txt" 2>/dev/null || echo NA)
hold_pid=$(cat "$OUT/hold_pid.txt" 2>/dev/null || echo 0)
pids_file="$OUT/nvsmi_pids.txt"

if [ ! -s "$pids_file" ]; then
    _emit "$jid" FAIL 0 "no nvsmi output"
    exit 0
fi

# Parse nvidia-smi pids (trim whitespace)
seen=$(awk 'NF{gsub(/[[:space:]]+/,""); print}' "$pids_file" | sort -u)
n_seen=$(printf '%s\n' "$seen" | grep -c '^[0-9]')

# Does our hold_pid appear? (It should — this validates we didn't over-filter)
saw_ours=$(printf '%s\n' "$seen" | grep -c "^${hold_pid}$")

# Any other PIDs not ours?
others=$(printf '%s\n' "$seen" | grep -v "^${hold_pid}$" | grep -c '^[0-9]')

metric="seen=${n_seen} ours=${saw_ours} others=${others} hold_pid=${hold_pid}"

if [ "$saw_ours" = "1" ] && [ "$others" = "0" ]; then
    _emit "$jid" PASS "$n_seen" "$metric"
elif [ "$saw_ours" = "1" ]; then
    _emit "$jid" PARTIAL "$n_seen" "leak: $metric"
else
    _emit "$jid" FAIL "$n_seen" "$metric"
fi
