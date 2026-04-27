#!/bin/bash
# Run inside a SLURM allocation on rack01-12. Starts N concurrent nvml_probe
# instances, polls nvidia-smi, then collects logs.
set -u
N="${N:-4}"
MB="${MB:-512}"
HOLD="${HOLD:-25}"
OUT="${OUT:-/tmp/softmig_mp_${SLURM_JOB_ID:-$$}}"
mkdir -p "$OUT"

# Always copy softmig log on exit (even on timeout / hung child).
_copy_log_on_exit() {
    if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${OUT:-}" ]; then
        if [ -r "/var/log/softmig/${SLURM_JOB_ID}.log" ]; then
            cp "/var/log/softmig/${SLURM_JOB_ID}.log" "$OUT/softmig.log" 2>/dev/null || true
        fi
    fi
}
trap _copy_log_on_exit EXIT INT TERM

echo "[harness] job=$SLURM_JOB_ID node=$(hostname) N=$N mb=$MB hold=$HOLD out=$OUT"
echo "[harness] softmig config:"
ls -la "/var/run/softmig/$SLURM_JOB_ID" 2>/dev/null || echo "  no /var/run/softmig/$SLURM_JOB_ID"
cat "/var/run/softmig/$SLURM_JOB_ID"/* 2>/dev/null | sed 's/^/  /'
echo "[harness] LD_PRELOAD=$(cat /etc/ld.so.preload 2>/dev/null)"
echo "[harness] libsoftmig:"; ls -la /usr/local/lib/libsoftmig* 2>/dev/null

# Per-probe wall clock: HOLD + 90s margin (driver init, SM watcher, etc.)
TLIM=$((HOLD + 90))
if ! command -v timeout >/dev/null 2>&1; then
    TLIM=0
fi

pids=()
for i in $(seq 1 "$N"); do
  if [ "$TLIM" -gt 0 ]; then
    ( exec timeout --preserve-status "$TLIM" ./build/test/nvml_probe "$MB" "$HOLD" ) >"$OUT/proc_$i.log" 2>&1 &
  else
    ( exec ./build/test/nvml_probe "$MB" "$HOLD" ) >"$OUT/proc_$i.log" 2>&1 &
  fi
  pids+=("$!")
  sleep 0.5
done
echo "[harness] started pids: ${pids[*]}"

# Light nvidia-smi polling (~30s total) — do not block longer than probes
for i in 1 2 3 4 5 6; do
  sleep 5
  echo "==== nvidia-smi +$((i*5))s ====" >> "$OUT/nvsmi.log"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> "$OUT/nvsmi.log" 2>&1
done

wait || true
echo "[harness] done. children exited."
echo "[harness] softmig log tail:"
ls -la "/var/log/softmig/" 2>/dev/null | head
SLOG="/var/log/softmig/${SLURM_JOB_ID}.log"
if [ -r "$SLOG" ]; then
  cp "$SLOG" "$OUT/softmig.log" 2>/dev/null || true
  wc -l "$OUT/softmig.log" 2>/dev/null
fi
echo "[harness] results in $OUT"
