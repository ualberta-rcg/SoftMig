#!/bin/bash
# Suite: gpuburn — the real wilicc gpu-burn (parent + forked per-GPU child,
# cuBLAS, asks for 90% of "available" memory) against SoftMig limits, packed
# on one GPU with neighbours. This is the historical "gpuburn && gpuburn"
# regression: every gpu_burn process must be registered (seen with the job
# in the node pidmap), capped at the slice, and the second run must work.
#
#   B1  one .4 job: gpu_burn T && gpu_burn T  (+ a .4 lite-burner neighbour)
#   B2  one .4 job: three gpu_burn at once      (+ a .4 lite-burner neighbour)
#   B3  four .4 jobs, one gpu_burn each (card full)
#   B4  gpu_burn in a .4 job while a PyTorch job (fw_limit.py torch enabled)
#       shares the GPU: torch must get its normal OOM inside its own limit
#   B5  gpu_burn on a whole GPU (passive) + gpu_burn in a .4 job
#
# Binaries: build-cuda<ver>/gpu-burn/{gpu_burn,compare.ptx} (test/build_gpuburn.sh).
# PASS iff share_report.py passes every job AND every gpu_burn run printed
# "GPU 0: OK" AND every gpu_burn PID was registered to its job.

SUITE=gpuburn
DEFAULT_SRUN_TIME="${DEFAULT_SRUN_TIME:-00:15:00}"
. "$(dirname "$0")/suite_common.sh"
. "$(dirname "$0")/share_lib.sh"

SCENARIO="${SCENARIO:-B1}"
SECS="${SHARE_SECS:-45}"
Q=11517; CARD_MIB=46068
QX="limit=$Q sm=25 view_total=$Q"
VENV="${FW_VENV:-${SCRATCH:-$HOME/scratch}/softmig-fwvenv}"

GB="\$(cd \$B/../gpu-burn && pwd)"
# gpu_burn must run from its dir (compare.ptx is looked up relative to cwd)
gb()   { echo "( cd $GB && ./gpu_burn ${1:-} $SECS )"; }
lite() { echo "\$B/gpu_burn_lite $1 $SECS"; }

case "$SCENARIO" in
  B1) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      EXPECT="$QX" run_job A_seq l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(gb) && echo FIRST_OK && $(gb) && echo SECOND_OK"
      EXPECT="$QX" run_job N_lite l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(lite 6000)" ;;
  B2) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      JOB_CPUS=8 EXPECT="$QX" run_job A_x3 l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "
$(gb) > \$D/gb1.out 2>&1 & $(gb) > \$D/gb2.out 2>&1 & $(gb) > \$D/gb3.out 2>&1 & wait
cat \$D/gb1.out \$D/gb2.out \$D/gb3.out"
      EXPECT="$QX" run_job N_lite l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(lite 6000)" ;;
  B3) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      for i in 0 1 2 3; do EXPECT="$QX" run_job "G$i" l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(gb)"; done ;;
  B4) pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      EXPECT="$QX" run_job A_burn l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(gb)"
      JOB_CPUS=8 JOB_MEM=24G EXPECT="$QX" run_job T_torch l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "
module load StdEnv/2023 python/3.11 cudnn >/dev/null 2>&1
export SOFTMIG_LIMIT_MIB=\$(sed -n 's/^CUDA_DEVICE_MEMORY_LIMIT=\([0-9]*\).*/\1/p' \$D/conf.txt)
sleep 10; '$VENV/bin/python' test/python/fw_limit.py torch enabled" ;;
  B5) pack_blocker 2 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
      sampler_start
      EXPECT="limit=0 sm=0 min_mem=30000 min_sm=80 view_total=$CARD_MIB" run_job W_whole l40s:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(gb)"
      EXPECT="$QX" run_job Q_burn l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "$(gb)" ;;
  *) _emit NA FAIL 0 "unknown SCENARIO=$SCENARIO"; exit 0 ;;
esac

wait_jobs
sleep 3
sampler_stop

summary=$(python3 "$SOFTMIG_ROOT/test/share_report.py" "$OUT" --scenario "$SCENARIO" \
            --md "$OUT/report.md" --tsv "$OUT/jobs.tsv" 2> "$OUT/report.err" | tail -1)
status=$(echo "$summary" | cut -f2); det=$(echo "$summary" | cut -f3)
[ -z "$status" ] && status=FAIL

# gpu_burn verdicts: every run must report "GPU 0: OK"; count expected runs
runs_ok=$(cat "$OUT"/jobs/*/run.out 2>/dev/null | grep -c "GPU 0: OK")
runs_bad=$(cat "$OUT"/jobs/*/run.out 2>/dev/null | grep -cE "FAULTY|Couldn't init|No clients are alive|errors: [1-9]")
case "$SCENARIO" in B1) want=2 ;; B2) want=3 ;; B3) want=4 ;; B4) want=1 ;; B5) want=2 ;; esac
[ "$runs_ok" = "$want" ] || { status=FAIL; det="$det gpu_burn_ok=$runs_ok/$want"; }
[ "$runs_bad" = 0 ] || { status=FAIL; det="$det gpu_burn_errors=$runs_bad"; }
[ "$SCENARIO" = B1 ] && { grep -q SECOND_OK "$OUT/jobs/A_seq/run.out" 2>/dev/null || { status=FAIL; det="$det second-run-missing"; }; }
if [ "$SCENARIO" = B4 ]; then
    grep -q "fw_limit: PASS" "$OUT/jobs/T_torch/run.out" 2>/dev/null || { status=FAIL; det="$det torch:$(grep -m1 fw_limit "$OUT/jobs/T_torch/run.out" 2>/dev/null | cut -c1-60)"; }
fi

# registration: every gpu_burn process on the node must map to one of our jobs
# (parent + child per run); none may be 'none' (unregistered / foreign job)
gb_pids=$(awk '$5=="gpu_burn"' "$OUT/sampler/pidmap.txt" 2>/dev/null | wc -l)
gb_orphan=$(awk '$5=="gpu_burn" && $3=="none"' "$OUT/sampler/pidmap.txt" 2>/dev/null | wc -l)
[ "$gb_pids" -ge "$want" ] || { status=FAIL; det="$det gpu_burn_pids_seen=$gb_pids<$want"; }
[ "$gb_orphan" = 0 ] || { status=FAIL; det="$det unregistered_gpu_burn=$gb_orphan"; }

metric=$(awk -F'\t' 'NR>1{printf "%s:%s/%s:%s/%s ", $2, $9, $8, $13, $14}' "$OUT/jobs.tsv")
jids=$(cat "$OUT"/jobs/*/jid.txt 2>/dev/null | paste -sd,)
uuids=$(cat "$OUT"/jobs/*/uuid.txt 2>/dev/null | sort -u | wc -l)
[ "$status" = PASS ] && [ "$SCENARIO" != B5 ] && [ "$uuids" -gt 1 ] && { status=PARTIAL; det="$det jobs-on-$uuids-GPUs"; }
_emit "$jids" "$status" "$metric" "scenario=$SCENARIO gpu_burn_ok=$runs_ok/$want pids=$gb_pids $det"
