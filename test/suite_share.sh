#!/bin/bash
# Suite: share — sets of concurrent jobs packed on ONE physical GPU, running
# real GPU work, measured from the node side (root nvidia-smi pmon, unfiltered)
# so each job's achieved SM% and memory can be compared with its limit.
#
#   SCENARIO=S1..S7 | G1..G6   CUDA_VER=12.6   SLICE=mixed   OUT=dir
#   SHARE_ROTATE=1  -> jobs rotate through 12.2 12.6 12.9 13.2 starting at CUDA_VER
#   SHARE_SECS=60   -> burn length
#
#   S1 .4+.4            S2 .2+.4 (the 60/40 question)   S3 .4 x4 (card full)
#   S4 .2+.2            S5 .4 burn + .4 idle holder       S6 .4 slice + whole-GPU
#   S7 .4 x3 burn + one .4 over-allocating (OOM must hit only the offender)
#   G1 .4 x4 allocate their whole limit at the same instant (4 x 11517 = card)
#   G2 .4 victim allocates its whole limit 30 s after 3 neighbours started burning
#   G3 fixed-work victim: alone, then with 1, 2, 3 burning neighbours (throughput)
#   G4 .2 + .4 + .4 all burning (each must get its own share)
#   G5 victim burns while the neighbour runs an allocation storm (stress_alloc)
#   G6 victim burns while the neighbour SIGSTOPs its own lock-holding workers
#
# Output: OUT/report.md (per-job table), OUT/jobs.tsv, OUT/sampler/*, one TSV
# line on stdout. PASS iff every job's verdict is PASS (see share_report.py).

SUITE=share
DEFAULT_SRUN_TIME="${DEFAULT_SRUN_TIME:-00:12:00}"
. "$(dirname "$0")/suite_common.sh"
. "$(dirname "$0")/share_lib.sh"

SCENARIO="${SCENARIO:-S1}"
SECS="${SHARE_SECS:-60}"
VERSIONS=(12.2 12.6 12.9 13.2)
ver_for() {   # ver_for i -> CUDA version for job i
    if [ "${SHARE_ROTATE:-0}" = 1 ]; then
        local base=0 k
        for k in "${!VERSIONS[@]}"; do [ "${VERSIONS[$k]}" = "$CUDA_VER" ] && base=$k; done
        echo "${VERSIONS[$(( (base + $1) % ${#VERSIONS[@]} ))]}"
    else
        echo "$CUDA_VER"
    fi
}

Q_MB=6000      # burn allocation for a quarter slice (limit 11517)
H_MB=12000     # for a half slice (limit 23034)
Q_LIMIT=11517

burn()      { echo "\$B/gpu_burn_lite $1 $2 ${3:-0}"; }
hold()      { echo "\$B/runtime_hold $1 $2"; }
wait_go()   { echo "for i in \$(seq 1 300); do [ -f '$OUT/go' ] && break; sleep 0.2; done"; }

oversub=""
case "$SCENARIO" in
  S1) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      run_job A l40s.4:1 "$(ver_for 0)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB $SECS)"
      run_job B l40s.4:1 "$(ver_for 1)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB $SECS)" ;;
  S2) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      run_job A_half    l40s.2:1 "$(ver_for 0)" "$DEFAULT_SRUN_TIME" "$(burn $H_MB $SECS)"
      run_job B_quarter l40s.4:1 "$(ver_for 1)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB $SECS)" ;;
  S3) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      for i in 0 1 2 3; do run_job "Q$i" l40s.4:1 "$(ver_for $i)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB $SECS)"; done ;;
  S4) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      run_job A_half l40s.2:1 "$(ver_for 0)" "$DEFAULT_SRUN_TIME" "$(burn $H_MB $SECS)"
      run_job B_half l40s.2:1 "$(ver_for 1)" "$DEFAULT_SRUN_TIME" "$(burn $H_MB $SECS)" ;;
  S5) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      run_job A_burn l40s.4:1 "$(ver_for 0)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB $SECS)"
      run_job B_idle l40s.4:1 "$(ver_for 1)" "$DEFAULT_SRUN_TIME" "$(hold 8000 $SECS)" ;;
  S6) pack_blocker 2 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      run_job A_slice l40s.4:1 "$(ver_for 0)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB $SECS)"
      run_job B_whole l40s:1   "$(ver_for 1)" "$DEFAULT_SRUN_TIME" "$(burn 30000 $SECS)" ;;
  S7) oversub="--oversub"
      pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      for i in 0 1 2; do run_job "Q$i" l40s.4:1 "$(ver_for $i)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB $SECS)"; done
      run_job "X_oom" l40s.4:1 "$(ver_for 3)" "$DEFAULT_SRUN_TIME" "sleep 10; $(hold $((Q_LIMIT * 3 / 2)) 20); $(hold $((Q_LIMIT - 1500)) 20)" ;;
  G1) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start; rm -f "$OUT/go"
      for i in 0 1 2 3; do run_job "Q$i" l40s.4:1 "$(ver_for $i)" "$DEFAULT_SRUN_TIME" "$(wait_go); $(hold $((Q_LIMIT - 700)) 25)"; done
      wait_started Q0 Q1 Q2 Q3; sleep 5; touch "$OUT/go" ;;
  G2) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      for i in 0 1 2; do run_job "N$i" l40s.4:1 "$(ver_for $i)" "$DEFAULT_SRUN_TIME" "$(burn 9000 $((SECS + 30)))"; done
      wait_started N0 N1 N2; sleep 30
      run_job V_late l40s.4:1 "$(ver_for 3)" "$DEFAULT_SRUN_TIME" "$(hold $((Q_LIMIT - 700)) 25)" ;;
  G3) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      W=3000   # fixed work: launches
      run_job V_alone l40s.4:1 "$(ver_for 0)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB 300 $W)"; wait_jobs
      for k in 1 2 3; do
          for i in $(seq 1 $k); do run_job "N${k}_$i" l40s.4:1 "$(ver_for $i)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB 150)"; done
          wait_started $(for i in $(seq 1 $k); do echo "N${k}_$i"; done); sleep 15
          run_job "V_with$k" l40s.4:1 "$(ver_for 0)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB 300 $W)"
          wait_jobs
      done ;;
  G4) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      run_job A_half l40s.2:1 "$(ver_for 0)" "$DEFAULT_SRUN_TIME" "$(burn $H_MB $SECS)"
      run_job B_q    l40s.4:1 "$(ver_for 1)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB $SECS)"
      run_job C_q    l40s.4:1 "$(ver_for 2)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB $SECS)" ;;
  G5) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      run_job V_burn  l40s.4:1 "$(ver_for 0)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB $SECS)"
      JOB_CPUS=16 run_job N_storm l40s.4:1 "$(ver_for 1)" "$DEFAULT_SRUN_TIME" "for i in 1 2 3 4; do \$B/stress_alloc 4 $SECS 512 \$((i % 2)) > \$D/storm_\$i.log 2>&1 & done; wait" ;;
  G6) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      run_job V_burn  l40s.4:1 "$(ver_for 0)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB $SECS)"
      JOB_CPUS=16 run_job N_hog l40s.4:1 "$(ver_for 1)" "$DEFAULT_SRUN_TIME" "
pids=(); for i in 1 2 3 4; do \$B/stress_alloc 4 $SECS 256 0 > \$D/hog_\$i.log 2>&1 & pids+=(\$!); done
sleep 10; kill -STOP \${pids[0]} \${pids[1]}; echo stopped \${pids[0]} \${pids[1]}; sleep 25; kill -CONT \${pids[0]} \${pids[1]}; echo continued
wait" ;;
  W1) # S1 with the nvidia-smi wrapper on PATH: the other job must disappear from the in-job view
      export SMI_WRAPPER=1
      pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      snap='bp=$!; sleep 20; command -v nvidia-smi > $D/smi_which.txt; nvidia-smi > $D/smi_table.txt 2>&1; nvidia-smi pmon -c 1 > $D/smi_pmon.txt 2>&1; wait $bp'
      run_job A l40s.4:1 "$(ver_for 0)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB $SECS) & $snap"
      run_job B l40s.4:1 "$(ver_for 1)" "$DEFAULT_SRUN_TIME" "$(burn $Q_MB $SECS) & $snap" ;;
  *) _emit NA FAIL 0 "unknown SCENARIO=$SCENARIO"; exit 0 ;;
esac

wait_jobs
sleep 3
sampler_stop

summary=$(python3 "$SOFTMIG_ROOT/test/share_report.py" "$OUT" --scenario "$SCENARIO" $oversub \
            --md "$OUT/report.md" --tsv "$OUT/jobs.tsv" 2> "$OUT/report.err" | tail -1)
status=$(echo "$summary" | cut -f2); det=$(echo "$summary" | cut -f3)

# G3 guarantee: the fixed-work victim must keep >= 60 % of its stand-alone
# throughput (launches/s) with three burning neighbours on the same GPU.
if [ "$SCENARIO" = G3 ]; then
    lps_alone=$(awk -F'\t' '$2=="V_alone"{print $25}' "$OUT/jobs.tsv")
    lps_w3=$(awk -F'\t' '$2=="V_with3"{print $25}' "$OUT/jobs.tsv")
    ratio=$(awk -v a="${lps_alone:-0}" -v b="${lps_w3:-0}" 'BEGIN{ if (a>0) printf "%.2f", b/a; else print 0 }')
    det="$det lps_alone=${lps_alone:-0} lps_with3=${lps_w3:-0} ratio=$ratio"
    awk -v r="$ratio" 'BEGIN{exit !(r >= 0.6)}' || { status=FAIL; det="$det throughput<60%"; }
fi

# compact metric: job:sm_mean/target:mem_peak/limit
metric=$(awk -F'\t' 'NR>1{printf "%s:%s/%s:%s/%s ", $2, $9, $8, $13, $14}' "$OUT/jobs.tsv")
jids=$(cat "$OUT"/jobs/*/jid.txt 2>/dev/null | paste -sd,)
uuids=$(cat "$OUT"/jobs/*/uuid.txt 2>/dev/null | sort -u | wc -l)
[ -z "$status" ] && status=FAIL
[ "$status" = PASS ] && [ "$SCENARIO" != S6 ] && [ "$uuids" -gt 1 ] && { status=PARTIAL; det="$det jobs-on-$uuids-GPUs"; }
_emit "$jids" "$status" "$metric" "scenario=$SCENARIO rotate=${SHARE_ROTATE:-0} $det"
