#!/bin/bash
# Suite: passthrough — a whole-GPU job must be pure pass-through (no config,
# no shared region, no clamp on memory or SM, unfiltered view) WHILE sliced
# jobs on the same node are enforced. Measured from the node side (sampler).
#
#   SCENARIO=P1  W(l40s:1) burns 30 GB at full speed  +  Q(l40s.4:1) burns
#   SCENARIO=P2  W(l40s:2) two whole GPUs, one burner per GPU  +  Q(l40s.4:1)
#   SCENARIO=P3  W holds 40 GB (> any slice limit) while Q over-allocates 1.5x
#                its quarter: Q gets OOM, W is untouched
#   SCENARIO=P4  W(l40s:1) + H(l40s.2:1) + Q(l40s.4:1) all burning: three
#                enforcement classes on one node at once
#
# Checks (share_report.py + expect.txt per job):
#   W: no conf, no region, mem_peak >= 20 GB (P3: 38 GB), sm_mean >= 80,
#      view total = 46068, no Device OOM / kill lines.
#   Q: limit 11517 / SM 25, mem <= limit+256, sm within tolerance, view 11517.
#   H: limit 23034 / SM 50.
# PASS iff every job PASSes. Visibility is not judged for W (passive = unfiltered).

SUITE=passthrough
DEFAULT_SRUN_TIME="${DEFAULT_SRUN_TIME:-00:12:00}"
. "$(dirname "$0")/suite_common.sh"
. "$(dirname "$0")/share_lib.sh"

SCENARIO="${SCENARIO:-P1}"
SECS="${SHARE_SECS:-60}"
CARD_MIB=46068
Q_LIMIT=11517; H_LIMIT=23034

burn() { echo "\$B/gpu_burn_lite $1 $2 ${3:-0}"; }
hold() { echo "\$B/runtime_hold $1 $2"; }

W_EXPECT="limit=0 sm=0 min_mem=20000 min_sm=80 view_total=$CARD_MIB"
Q_EXPECT="limit=$Q_LIMIT sm=25 view_total=$Q_LIMIT"
H_EXPECT="limit=$H_LIMIT sm=50 view_total=$H_LIMIT"

sampler_start
case "$SCENARIO" in
  P1) EXPECT="$W_EXPECT" run_job W_whole l40s:1   "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 30000 $SECS)"
      EXPECT="$Q_EXPECT" run_job Q_slice l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 6000 $SECS)" ;;
  P2) EXPECT="limit=0 sm=0 min_mem=40000 min_sm=160 view_total=$CARD_MIB" \
      run_job W_two l40s:2 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "
nvidia-smi --query-gpu=uuid --format=csv,noheader > \$D/uuids.txt
CUDA_VISIBLE_DEVICES=0 $(burn 30000 $SECS) > \$D/burn0.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 $(burn 30000 $SECS) > \$D/burn1.out 2>&1 &
wait; cat \$D/burn0.out \$D/burn1.out"
      EXPECT="$Q_EXPECT" run_job Q_slice l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 6000 $SECS)" ;;
  P3) EXPECT="limit=0 sm=0 min_mem=38000 view_total=$CARD_MIB oom=0" \
      run_job W_hold l40s:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(hold 40000 $((SECS + 20)))"
      EXPECT="$Q_EXPECT oom=1 rc=1" run_job Q_oom l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" \
          "sleep 10; $(hold $((Q_LIMIT * 3 / 2)) 15); $(hold $((Q_LIMIT - 1500)) 15); exit 1" ;;
  P4) EXPECT="$W_EXPECT" run_job W_whole l40s:1   "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 30000 $SECS)"
      EXPECT="$H_EXPECT" run_job H_half  l40s.2:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 12000 $SECS)"
      EXPECT="$Q_EXPECT" run_job Q_slice l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 6000 $SECS)" ;;
  *) _emit NA FAIL 0 "unknown SCENARIO=$SCENARIO"; exit 0 ;;
esac

wait_jobs
sleep 3
sampler_stop

summary=$(python3 "$SOFTMIG_ROOT/test/share_report.py" "$OUT" --scenario "$SCENARIO" \
            --md "$OUT/report.md" --tsv "$OUT/jobs.tsv" 2> "$OUT/report.err" | tail -1)
status=$(echo "$summary" | cut -f2); det=$(echo "$summary" | cut -f3)
[ -z "$status" ] && status=FAIL

# P3: the Q job's second hold (inside its limit) must have succeeded after the OOM
if [ "$SCENARIO" = P3 ]; then
    q="$OUT/jobs/Q_oom/run.out"
    grep -q "FAILED" "$q" 2>/dev/null || { status=FAIL; det="$det q-no-oom"; }
    [ "$(_grepc 'ok' "$q")" -ge 1 ] || { status=FAIL; det="$det q-second-hold-failed"; }
    grep -q "FAILED" "$OUT/jobs/W_hold/run.out" 2>/dev/null && { status=FAIL; det="$det w-hold-failed"; }
fi

metric=$(awk -F'\t' 'NR>1{printf "%s:%s/%s:%s/%s:view%s ", $2, $9, $8, $13, $14, $16}' "$OUT/jobs.tsv")
jids=$(cat "$OUT"/jobs/*/jid.txt 2>/dev/null | paste -sd,)
_emit "$jids" "$status" "$metric" "scenario=$SCENARIO $det"
