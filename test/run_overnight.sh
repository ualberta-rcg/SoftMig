#!/bin/bash
# Overnight SoftMig reliability runbook.
# Does NOT reserve an allocation itself; it launches suites that call srun.
#
# Usage:
#   HOURS=10 bash test/run_overnight.sh
#
# Artifacts:
#   test_results/overnight_<timestamp>/

set -u
SOFTMIG_ROOT=/scratch/rahimk/SoftMig
cd "$SOFTMIG_ROOT" || exit 1

HOURS="${HOURS:-10}"
if ! [[ "$HOURS" =~ ^[0-9]+$ ]] || [ "$HOURS" -le 0 ]; then
  echo "HOURS must be a positive integer"
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
ROOT="$SOFTMIG_ROOT/test_results/overnight_${TS}"
mkdir -p "$ROOT"
echo "$ROOT" > "$SOFTMIG_ROOT/test_results/overnight_latest.txt"

RESULTS="$ROOT/results.tsv"
printf 'cycle\tcuda_ver\tslice\tsuite\tjobid\tstatus\tmetric\tdetail\n' > "$RESULTS"

END_EPOCH=$(( $(date +%s) + HOURS * 3600 ))
CYCLE=0

run_suite() {
  local cycle="$1" ver="$2" slice="$3" suite="$4" outdir="$5"
  mkdir -p "$outdir"
  local line
  line=$(OUT="$outdir" CUDA_VER="$ver" SLICE="$slice" bash "test/suite_${suite}.sh" 2>/dev/null | tail -1)
  if [ -n "$line" ]; then
    printf '%s\t%s\n' "$cycle" "$line" >> "$RESULTS"
  else
    printf '%s\t%s\t%s\t%s\tNA\tFAIL\t0\tno output from suite\n' "$cycle" "$ver" "$slice" "$suite" >> "$RESULTS"
  fi
}

while [ "$(date +%s)" -lt "$END_EPOCH" ]; do
  CYCLE=$((CYCLE + 1))
  CROOT="$ROOT/cycle_${CYCLE}"
  mkdir -p "$CROOT"
  echo "=== cycle $CYCLE start $(date) ===" | tee -a "$ROOT/phase.log"

  # Control checks: correctness and compatibility in relatively isolated runs.
  for ver in 12.2 12.6 12.9 13.2; do
    run_suite "$CYCLE" "$ver" "l40s.4" "direct" "$CROOT/direct_${ver}_4"
    run_suite "$CYCLE" "$ver" "l40s.4" "nvsmi" "$CROOT/nvsmi_${ver}_4"
    run_suite "$CYCLE" "$ver" "l40s.2" "oom" "$CROOT/oom_${ver}_2"
    run_suite "$CYCLE" "$ver" "l40s.2" "sm" "$CROOT/sm_${ver}_2"
  done

  # Soak check (script now fixed and should emit a verdict)
  run_suite "$CYCLE" "12.2" "l40s.4" "soak" "$CROOT/soak_12.2_4"

  # Parallel leak stress: many nvsmi checks + background burn pressure.
  SROOT="$CROOT/stress_nvsmi"
  mkdir -p "$SROOT"
  printf 'cycle\tcuda_ver\tslice\tsuite\tjobid\tstatus\tmetric\tdetail\n' > "$SROOT/results.tsv"

  for i in $(seq 1 8); do
    case $(( (i - 1) % 4 )) in
      0) ver=12.2 ;;
      1) ver=12.6 ;;
      2) ver=12.9 ;;
      3) ver=13.2 ;;
    esac
    (
      line=$(OUT="$SROOT/nvsmi_$i" CUDA_VER="$ver" SLICE="l40s.4" bash test/suite_nvsmi.sh 2>/dev/null | tail -1)
      if [ -n "$line" ]; then
        printf '%s\t%s\n' "$CYCLE" "$line" >> "$SROOT/results.tsv"
      else
        printf '%s\t%s\t%s\tnvsmi\tNA\tFAIL\t0\tno output from suite\n' "$CYCLE" "$ver" "l40s.4" >> "$SROOT/results.tsv"
      fi
    ) &
  done

  for j in $(seq 1 4); do
    case $(( (j - 1) % 4 )) in
      0) ver=12.2 ;;
      1) ver=12.6 ;;
      2) ver=12.9 ;;
      3) ver=13.2 ;;
    esac
    (
      srun --reservation=softmig --gres=gpu:l40s.4:1 --cpus-per-task=8 --mem=10G --time=00:08:00 \
        bash -lc "module load cuda/${ver}; cd ${SOFTMIG_ROOT}; export SOFTMIG_LOG_LEVEL=4; N=4 MB=4096 DUR=30 OUT='${SROOT}/burn_${j}' test/run_burn.sh >/dev/null 2>&1" \
        >> "$SROOT/burn.log" 2>&1
    ) &
  done
  wait

  # Append stress nvsmi lines to main results.
  if [ -s "$SROOT/results.tsv" ]; then
    awk 'NR>1 {print}' "$SROOT/results.tsv" >> "$RESULTS"
  fi

  # Per-cycle quick summary.
  {
    echo "cycle=$CYCLE"
    awk -F'\t' -v c="$CYCLE" 'NR>1 && $1==c {cnt[$6]++} END{for (k in cnt) print k "=" cnt[k]} ' "$RESULTS" | sort
    echo
  } >> "$ROOT/cycle_summary.txt"

  echo "=== cycle $CYCLE end $(date) ===" | tee -a "$ROOT/phase.log"
done

echo "Overnight run complete: $ROOT"
