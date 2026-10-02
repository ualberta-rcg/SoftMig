#!/bin/bash
# Suite: mixed_pieces — the shapes real users submit, packed on one GPU with
# neighbours, measured from the node side. Every job carries expect.txt
# (limit / SM% the prolog must have produced for that request).
#
#   M1  l40s.4:2 (two shards -> 23034M / 50%) + two .4 neighbours  (card full)
#   M2  l40s.4:3 (-> 34551M / 75%) + one .4 neighbour
#   M3  one job asking gpu:l40s.2:1,gpu:l40s.4:1 (mixed shard types) + one .4;
#       Slurm may reject the request: that is recorded as INFO, not FAIL
#   M4  sbatch --array=0-3 of .4 tasks on one GPU: per-task conf <jid>_<task>
#   M5  one l40s.4:2 allocation running two srun steps at once (+ one .4):
#       both steps share the job's 23034M / 50% budget
#   M6  whole GPU (passive) + .2 + .4 + .4 burning on the same node
#
# PASS iff every job PASSes in share_report.py (limits as expected, memory
# within limit, SM within tolerance, in-job view total == limit, no UNHOOKED).
# LEAK = only cross-job PID visibility (nvidia-smi 595, see nvidia-smi-hook.sh).

SUITE=mixed_pieces
DEFAULT_SRUN_TIME="${DEFAULT_SRUN_TIME:-00:12:00}"
. "$(dirname "$0")/suite_common.sh"
. "$(dirname "$0")/share_lib.sh"

SCENARIO="${SCENARIO:-M1}"
SECS="${SHARE_SECS:-60}"
CARD_MIB=46068
Q=11517; H=23034; TQ=34551

burn() { echo "\$B/gpu_burn_lite $1 $2 ${3:-0}"; }
QX="limit=$Q sm=25 view_total=$Q"
HX="limit=$H sm=50 view_total=$H"
TX="limit=$TQ sm=75 view_total=$TQ"
WX="limit=0 sm=0 min_mem=20000 min_sm=80 view_total=$CARD_MIB"

info=""
case "$SCENARIO" in
  M1) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      EXPECT="$HX" run_job A_2shards l40s.4:2 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 12000 $SECS)"
      EXPECT="$QX" run_job B_q       l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 6000 $SECS)"
      EXPECT="$QX" run_job C_q       l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 6000 $SECS)" ;;
  M2) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      EXPECT="$TX" run_job A_3shards l40s.4:3 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 20000 $SECS)"
      EXPECT="$QX" run_job B_q       l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 6000 $SECS)" ;;
  M3) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      # 1 x .2 + 1 x .4 = 3 shards if Slurm accepts the request
      EXPECT="$TX" run_job A_mixed "l40s.2:1,gpu:l40s.4:1" "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 20000 $SECS)"
      EXPECT="$QX" run_job B_q     l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 6000 $SECS)" ;;
  M4) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      EXPECT="$QX" run_array T 4 l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 6000 $SECS)" ;;
  M5) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      JOB_CPUS=4 EXPECT="$HX" run_array A_steps 1 l40s.4:2 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "
srun -n1 --exact --cpus-per-task=2 --mem=3G \$B/gpu_burn_lite 9000 $SECS > \$D/step0.out 2>&1 &
srun -n1 --exact --cpus-per-task=2 --mem=3G \$B/gpu_burn_lite 9000 $SECS > \$D/step1.out 2>&1 &
wait; cat \$D/step0.out \$D/step1.out"
      EXPECT="$QX" run_job B_q l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 6000 $SECS)" ;;
  M6) pack_blocker 2 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      EXPECT="$WX" run_job W_whole l40s:1   "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 30000 $SECS)"
      EXPECT="$HX" run_job H_half  l40s.2:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 12000 $SECS)"
      EXPECT="$QX" run_job Q1      l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 6000 $SECS)"
      EXPECT="$QX" run_job Q2      l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(burn 6000 $SECS)" ;;
  *) _emit NA FAIL 0 "unknown SCENARIO=$SCENARIO"; exit 0 ;;
esac

wait_jobs
sleep 3
sampler_stop

# M3: a request Slurm refuses is a documented behaviour, not a SoftMig failure
if [ "$SCENARIO" = M3 ] && [ ! -s "$OUT/jobs/A_mixed/jid.txt" ]; then
    info="mixed-gres-rejected:$(grep -m1 -i 'error' "$OUT/jobs/A_mixed/srun.log" 2>/dev/null | cut -c1-80)"
    rm -rf "$OUT/jobs/A_mixed"
fi

summary=$(python3 "$SOFTMIG_ROOT/test/share_report.py" "$OUT" --scenario "$SCENARIO" \
            --md "$OUT/report.md" --tsv "$OUT/jobs.tsv" 2> "$OUT/report.err" | tail -1)
status=$(echo "$summary" | cut -f2); det=$(echo "$summary" | cut -f3)
[ -z "$status" ] && status=FAIL

# M4/M5: every task/step must have been on the same GPU as its siblings and
# every array task must have had its own conf file
if [ "$SCENARIO" = M4 ]; then
    nconf=$(cat "$OUT"/jobs/T_*/conf.txt 2>/dev/null | grep -c CUDA_DEVICE_MEMORY_LIMIT)
    [ "$nconf" = 4 ] || { status=FAIL; det="$det array-confs=$nconf/4"; }
fi
if [ "$SCENARIO" = M5 ]; then
    steps=$(awk -v j="$(job_meta A_steps_0 jid)" '$3==j {print $4}' "$OUT/sampler/pidmap.txt" | sort -u | wc -l)
    [ "$steps" -ge 2 ] || { status=FAIL; det="$det steps-seen=$steps/2"; }
fi

metric=$(awk -F'\t' 'NR>1{printf "%s:%s/%s:%s/%s ", $2, $9, $8, $13, $14}' "$OUT/jobs.tsv")
jids=$(cat "$OUT"/jobs/*/jid.txt 2>/dev/null | paste -sd,)
uuids=$(cat "$OUT"/jobs/*/uuid.txt 2>/dev/null | sort -u | wc -l)
[ "$status" = PASS ] && [ "$SCENARIO" != M6 ] && [ "$uuids" -gt 1 ] && { status=PARTIAL; det="$det jobs-on-$uuids-GPUs"; }
_emit "$jids" "$status" "$metric" "scenario=$SCENARIO $det $info"
