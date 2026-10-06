#!/bin/bash
# Suite: storm — many users at once. Keeps up to STORM_PAR concurrent jobs of
# random shapes churning on the node for STORM_MINUTES, every job carrying the
# limits the prolog must have given it (expect.txt), then judges every job
# from the node-side sampler (share_report.py).
#
#   ROUND=A   mixed churn: .4 / .2 lite burners, .4 gpu-burn, l40s.4:2 holder,
#             whole-GPU burner, .4 array of 4, .4 over-allocator (OOM expected),
#             .4 nvidia-smi poller, .4 PyTorch fw_limit, .4 alloc storm
#   ROUND=B   nvidia-smi leak stress: 8 parallel nvsmi suites + 4 lite burners
#             (the historical suite_nvsmi.sh, cross-version)
#   ROUND=C   round A with a different seed (STORM_SEED)
#
#   STORM_MINUTES=10  STORM_PAR=10  STORM_SEED=1  CUDA_VER rotates per job
#
# PASS iff no job FAILs (LEAK = only visibility; see nvidia-smi-hook.sh) and
# every nvsmi suite in round B passes. Runs from a login shell or from the
# sbatch driver (with SAMPLER_SHARED set by the launcher).

SUITE=storm
DEFAULT_SRUN_TIME="${DEFAULT_SRUN_TIME:-00:15:00}"
. "$(dirname "$0")/suite_common.sh"
. "$(dirname "$0")/share_lib.sh"

ROUND="${ROUND:-A}"
MINUTES="${STORM_MINUTES:-10}"
PAR="${STORM_PAR:-10}"
SEED="${STORM_SEED:-1}"
VERSIONS=(12.2 12.6 12.9 13.2)
Q=11517; H=23034; CARD_MIB=46068
VENV="${FW_VENV:-${SCRATCH:-$HOME/scratch}/softmig-fwvenv}"
RANDOM=$SEED

QX="limit=$Q sm=25 view_total=$Q"; HX="limit=$H sm=50 view_total=$H"
WX="limit=0 sm=0 min_mem=20000 view_total=$CARD_MIB"

n_running() { jobs -rp | wc -l; }
ver_pick()  { echo "${VERSIONS[$((RANDOM % 4))]}"; }

# One random job. $1 = sequence number.
spawn_one() {
    local i="$1" v; v=$(ver_pick)
    local secs=$((30 + RANDOM % 60)) kind=$((RANDOM % 10))
    case "$kind" in
      0|1) EXPECT="$QX" run_job "q_lite_$i" l40s.4:1 "$v" "$DEFAULT_SRUN_TIME" "\$B/gpu_burn_lite 6000 $secs" ;;
      2)   EXPECT="$HX" run_job "h_lite_$i" l40s.2:1 "$v" "$DEFAULT_SRUN_TIME" "\$B/gpu_burn_lite 12000 $secs" ;;
      3)   EXPECT="$QX" run_job "q_gpuburn_$i" l40s.4:1 "$v" "$DEFAULT_SRUN_TIME" "( cd \$B/../gpu-burn && ./gpu_burn $secs )" ;;
      4)   EXPECT="$HX" run_job "two_hold_$i" l40s.4:2 "$v" "$DEFAULT_SRUN_TIME" "\$B/runtime_hold 15000 $secs" ;;
      5)   EXPECT="$WX" run_job "whole_$i" l40s:1 "$v" "$DEFAULT_SRUN_TIME" "\$B/gpu_burn_lite 30000 $secs" ;;
      6)   EXPECT="$QX" run_array "arr_$i" 4 l40s.4:1 "$v" "$DEFAULT_SRUN_TIME" "\$B/gpu_burn_lite 4000 $secs" ;;
      7)   EXPECT="$QX oom=1 rc=1" run_job "q_oom_$i" l40s.4:1 "$v" "$DEFAULT_SRUN_TIME" \
               "\$B/runtime_hold $((Q * 3 / 2)) 10; \$B/runtime_hold $((Q - 1500)) 15; exit 1" ;;
      8)   EXPECT="$QX" run_job "q_poll_$i" l40s.4:1 "$v" "$DEFAULT_SRUN_TIME" \
               "\$B/gpu_burn_lite 2000 $secs & for k in \$(seq 1 $secs); do nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >/dev/null; nvidia-smi >/dev/null; sleep 0.5; done; wait" ;;
      9)   if [ -x "$VENV/bin/python" ]; then
               JOB_CPUS=8 JOB_MEM=24G EXPECT="$QX" run_job "q_torch_$i" l40s.4:1 "$v" "$DEFAULT_SRUN_TIME" \
                   "module load StdEnv/2023 python/3.11 cudnn >/dev/null 2>&1; export SOFTMIG_LIMIT_MIB=\$(sed -n 's/^CUDA_DEVICE_MEMORY_LIMIT=\([0-9]*\).*/\1/p' \$D/conf.txt); '$VENV/bin/python' test/python/fw_limit.py torch enabled"
           else
               JOB_CPUS=8 EXPECT="$QX" run_job "q_alloc_$i" l40s.4:1 "$v" "$DEFAULT_SRUN_TIME" "for k in 1 2 3 4; do \$B/stress_alloc 4 $secs 512 \$((k % 2)) > \$D/storm_\$k.log 2>&1 & done; wait"
           fi ;;
    esac
}

sampler_start
t_end=$(( $(date +%s) + MINUTES * 60 ))
case "$ROUND" in
  A|C)
    i=0
    while [ "$(date +%s)" -lt "$t_end" ]; do
        if [ "$(n_running)" -lt "$PAR" ]; then i=$((i + 1)); spawn_one "$i"; sleep 2; else sleep 5; fi
    done
    wait_jobs ;;
  B)
    mkdir -p "$OUT/nvsmi"
    printf 'cuda_ver\tslice\tsuite\tjobid\tstatus\tmetric\tdetail\n' > "$OUT/nvsmi/results.tsv"
    for k in $(seq 1 8); do
        v="${VERSIONS[$(( (k - 1) % 4 ))]}"
        ( line=$(OUT="$OUT/nvsmi/$k" CUDA_VER="$v" SLICE=l40s.4 bash "$SOFTMIG_ROOT/test/suite_nvsmi.sh" 2>/dev/null | tail -1)
          [ -n "$line" ] || line=$(printf '%s\tl40s.4\tnvsmi\tNA\tFAIL\t0\tno output' "$v")
          echo "$line" >> "$OUT/nvsmi/results.tsv" ) &
    done
    for k in 1 2 3 4; do
        EXPECT="$QX" run_job "burn_$k" l40s.4:1 "${VERSIONS[$((k - 1))]}" "$DEFAULT_SRUN_TIME" "\$B/gpu_burn_lite 4000 $((MINUTES * 60 / 2))"
    done
    wait ;;
  *) _emit NA FAIL 0 "unknown ROUND=$ROUND"; exit 0 ;;
esac
sleep 3
sampler_stop

summary=$(python3 "$SOFTMIG_ROOT/test/share_report.py" "$OUT" --scenario "storm-$ROUND" --oversub \
            --md "$OUT/report.md" --tsv "$OUT/jobs.tsv" 2> "$OUT/report.err" | tail -1)
status=$(echo "$summary" | cut -f2); det=$(echo "$summary" | cut -f3)
[ -z "$status" ] && status=FAIL
njobs=$(awk 'NR>1' "$OUT/jobs.tsv" 2>/dev/null | wc -l)
nfail=$(awk -F'\t' 'NR>1 && $(NF-1)=="FAIL"' "$OUT/jobs.tsv" 2>/dev/null | wc -l)
nleak=$(awk -F'\t' 'NR>1 && $(NF-1)=="LEAK"' "$OUT/jobs.tsv" 2>/dev/null | wc -l)
if [ "$ROUND" = B ]; then
    nv_fail=$(awk -F'\t' 'NR>1 && $5!="PASS"' "$OUT/nvsmi/results.tsv" | wc -l)
    [ "$nv_fail" = 0 ] || { status=FAIL; det="$det nvsmi_fail=$nv_fail"; }
fi
jids=$(cat "$OUT"/jobs/*/jid.txt 2>/dev/null | wc -l)
_emit "n=$jids" "$status" "jobs=$njobs fail=$nfail leak=$nleak" "round=$ROUND minutes=$MINUTES par=$PAR seed=$SEED $det"
