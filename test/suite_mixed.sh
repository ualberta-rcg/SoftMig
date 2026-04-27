#!/bin/bash
# Suite 6: mixed-CUDA same-node — two concurrent jobs with DIFFERENT CUDA
# runtime modules (oldest vs newest). Uses .4 slices so both fit. PASS iff
# both jobs register their procs, neither crashes, neither kills the other.
#
# Environment inputs: OUT dir. Does not consume SLICE/CUDA_VER (has its own).

SUITE=mixed
SLICE=${SLICE:-l40s.4}
CUDA_VER=${CUDA_VER:-mixed}
. "$(dirname "$0")/suite_common.sh"

VER_A=12.2
VER_B=13.2

mkdir -p "$OUT/jobA_${VER_A}" "$OUT/jobB_${VER_B}"

srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:l40s.4:1 --cpus-per-task=4 --mem=6G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${VER_A}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=5
./build/test/runtime_hold 768 30 > '${OUT}/jobA_${VER_A}/run.out' 2>&1
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/jobA_${VER_A}/softmig.log'
echo \$SLURM_JOB_ID > '${OUT}/jobA_${VER_A}/jid.txt'
" >/dev/null 2>&1 &
PIDA=$!
sleep 4
srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:l40s.4:1 --cpus-per-task=4 --mem=6G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${VER_B}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=5
./build/test/runtime_hold 768 24 > '${OUT}/jobB_${VER_B}/run.out' 2>&1
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/jobB_${VER_B}/softmig.log'
echo \$SLURM_JOB_ID > '${OUT}/jobB_${VER_B}/jid.txt'
" >/dev/null 2>&1 &
PIDB=$!

wait $PIDA $PIDB

jidA=$(cat "$OUT/jobA_${VER_A}/jid.txt" 2>/dev/null || echo NA)
jidB=$(cat "$OUT/jobB_${VER_B}/jid.txt" 2>/dev/null || echo NA)
slogA="$OUT/jobA_${VER_A}/softmig.log"
slogB="$OUT/jobB_${VER_B}/softmig.log"

if [ ! -s "$slogA" ] || [ ! -s "$slogB" ]; then
    _emit "${jidA},${jidB}" FAIL 0 "missing softmig log"
    exit 0
fi

regA=$(grep -c "set_task_pid: Found current process PID" "$slogA")
regB=$(grep -c "set_task_pid: Found current process PID" "$slogB")
errA=$(grep -c "softmig ERROR" "$slogA")
errB=$(grep -c "softmig ERROR" "$slogB")
killsA=$(grep -c "KILLED PID" "$slogA")
killsB=$(grep -c "KILLED PID" "$slogB")

metric="cuda${VER_A}: reg=${regA} err=${errA} kills=${killsA} | cuda${VER_B}: reg=${regB} err=${errB} kills=${killsB}"

if [ "$regA" -ge 1 ] && [ "$regB" -ge 1 ] \
   && [ "$killsA" = "0" ] && [ "$killsB" = "0" ]; then
    _emit "${jidA},${jidB}" PASS "${regA}+${regB}" "$metric"
else
    _emit "${jidA},${jidB}" FAIL "${regA}+${regB}" "$metric"
fi
