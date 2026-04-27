#!/bin/bash
# Multi-process test using cudart-based runtime_hold (gets full SoftMig hooking).
set -u
N="${N:-4}"
MB="${MB:-512}"
HOLD="${HOLD:-25}"
OUT="${OUT:-/tmp/softmig_rt_${SLURM_JOB_ID:-$$}}"
mkdir -p "$OUT"

echo "[harness] job=$SLURM_JOB_ID node=$(hostname) N=$N mb=$MB hold=$HOLD out=$OUT"
echo "[harness] softmig config:"
ls -la "/var/run/softmig/" 2>/dev/null | grep "$SLURM_JOB_ID" || echo "  no conf for this job"
cat "/var/run/softmig/${SLURM_JOB_ID}.conf" 2>/dev/null | sed 's/^/  /'
echo "[harness] LD_PRELOAD=$(cat /etc/ld.so.preload 2>/dev/null)"

pids=()
for i in $(seq 1 "$N"); do
  ( exec ./build/test/runtime_hold "$MB" "$HOLD" ) >"$OUT/proc_$i.log" 2>&1 &
  pids+=("$!")
  sleep 0.4
done
echo "[harness] started pids: ${pids[*]}"

for t in 2 6 12 18; do
  sleep 4
  echo "==== nvidia-smi t~${t}s ====" >> "$OUT/nvsmi.log"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | grep -v "softmig " >> "$OUT/nvsmi.log"
done

wait
echo "[harness] done"
SLOG="/var/log/softmig/${SLURM_JOB_ID}.log"
[ -r "$SLOG" ] && cp "$SLOG" "$OUT/softmig.log"
wc -l "$OUT"/*.log 2>/dev/null
echo RES=$OUT
