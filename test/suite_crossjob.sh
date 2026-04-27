#!/bin/bash
# Suite 5: cross-job cgroup isolation — two concurrent jobs on same node.
# Each job runs runtime_hold for ~30s. PASS iff each job's softmig.log shows
# the OTHER job's PID in 'different cgroup, skipping' AND never in a
# 'Found current process PID' line.

SUITE=crossjob
. "$(dirname "$0")/suite_common.sh"

mkdir -p "$OUT/jobA" "$OUT/jobB"

srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=4 --mem=6G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=5
./build/test/runtime_hold 1024 30 > '${OUT}/jobA/run.out' 2>&1
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/jobA/softmig.log'
echo \$SLURM_JOB_ID > '${OUT}/jobA/jid.txt'
" >/dev/null 2>&1 &
PIDA=$!
sleep 4
srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=4 --mem=6G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=5
./build/test/runtime_hold 1024 24 > '${OUT}/jobB/run.out' 2>&1
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/jobB/softmig.log'
echo \$SLURM_JOB_ID > '${OUT}/jobB/jid.txt'
" >/dev/null 2>&1 &
PIDB=$!

wait $PIDA $PIDB

jidA=$(cat "$OUT/jobA/jid.txt" 2>/dev/null || echo NA)
jidB=$(cat "$OUT/jobB/jid.txt" 2>/dev/null || echo NA)
slogA="$OUT/jobA/softmig.log"
slogB="$OUT/jobB/softmig.log"

if [ ! -s "$slogA" ] || [ ! -s "$slogB" ]; then
    _emit "${jidA},${jidB}" FAIL 0 "missing one or both softmig logs"
    exit 0
fi

pidA=$(grep -oE "Found current process PID [0-9]+" "$slogA" | head -1 | grep -oE "[0-9]+")
pidB=$(grep -oE "Found current process PID [0-9]+" "$slogB" | head -1 | grep -oE "[0-9]+")
[ -z "$pidA" ] && pidA=0
[ -z "$pidB" ] && pidB=0

# Cross-registration (bad)
crossA_regs_B=$(grep -c "Found current process PID ${pidB}" "$slogA" 2>/dev/null)
crossB_regs_A=$(grep -c "Found current process PID ${pidA}" "$slogB" 2>/dev/null)

# Correct filtering (good)
A_skipped_B=$(grep -c "PID ${pidB} - different cgroup" "$slogA" 2>/dev/null)
B_skipped_A=$(grep -c "PID ${pidA} - different cgroup" "$slogB" 2>/dev/null)

metric="A->B=${A_skipped_B} B->A=${B_skipped_A} crossA=${crossA_regs_B} crossB=${crossB_regs_A}"

# PASS: zero cross-regs AND each side saw+skipped the other at least once
if [ "$crossA_regs_B" = "0" ] && [ "$crossB_regs_A" = "0" ] \
   && [ "$A_skipped_B" -ge 1 ] && [ "$B_skipped_A" -ge 1 ]; then
    _emit "${jidA},${jidB}" PASS "${A_skipped_B}+${B_skipped_A}" "$metric"
elif [ "$crossA_regs_B" = "0" ] && [ "$crossB_regs_A" = "0" ]; then
    # No cross-contamination but didn't see each other (maybe didn't overlap)
    _emit "${jidA},${jidB}" PARTIAL "0" "no overlap but no leak: $metric"
else
    _emit "${jidA},${jidB}" FAIL "${crossA_regs_B}+${crossB_regs_A}" "$metric"
fi
