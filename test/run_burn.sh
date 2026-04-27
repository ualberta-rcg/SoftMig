#!/bin/bash
# Multi-process gpu-burn-lite harness.
set -u
N="${N:-3}"
MB="${MB:-1024}"
DUR="${DUR:-30}"
OUT="${OUT:-/tmp/softmig_burn_${SLURM_JOB_ID:-$$}}"
mkdir -p "$OUT"

echo "[harness] N=$N MB=$MB DUR=${DUR}s OUT=$OUT job=$SLURM_JOB_ID"
cat "/var/run/softmig/${SLURM_JOB_ID}.conf" 2>/dev/null | sed 's/^/  /'

pids=()
for i in $(seq 1 "$N"); do
  ( exec ./build/test/gpu_burn_lite "$MB" "$DUR" ) >"$OUT/proc_$i.log" 2>&1 &
  pids+=("$!")
done
echo "[harness] started pids: ${pids[*]}"

# Sample nvidia-smi utilization/memory every 2s
( for t in $(seq 1 $((DUR/2))); do
    sleep 2
    echo "==== t=$((t*2)) ===" >> "$OUT/nvsmi.log"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 | grep -v softmig >> "$OUT/nvsmi.log"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1 | grep -v softmig >> "$OUT/nvsmi.log"
  done ) &
SAMPLER=$!

wait "${pids[@]}"
kill $SAMPLER 2>/dev/null
wait 2>/dev/null

SLOG="/var/log/softmig/${SLURM_JOB_ID}.log"
[ -r "$SLOG" ] && cp "$SLOG" "$OUT/softmig.log"
echo "[harness] done. results in $OUT"
