#!/bin/bash
# Suite 5: cross-job cgroup isolation — two concurrent jobs on the same GPU.
#
# Shard jobs do not consume GPU GRES, and SLURM spreads them across cards
# (each job's CUDA_VISIBLE_DEVICES=0 is a *different* physical GPU remapped
# to index 0). CUDA_VISIBLE_DEVICES cannot override that: NVML only exposes
# the assigned GPU. To force overlap, hold 3/4 L40S with a full-GPU blocker
# so the two 1/4 slices must share the remaining card.
#
# PASS iff: same GPU UUID, neither softmig.log registers the other PID as
# "Found current process", and the NVML cgroup filter skipped the other
# PID (or hooked nvidia-smi hid it). nvidia-smi may still list both PIDs
# if the optional nvidia-smi-hook.sh wrapper is not installed.

SUITE=crossjob
DEFAULT_SRUN_TIME="${DEFAULT_SRUN_TIME:-00:08:00}"
. "$(dirname "$0")/suite_common.sh"

mkdir -p "$OUT/jobA" "$OUT/jobB" "$OUT/blocker"

# Pack onto one physical GPU regardless of the matrix SLICE column.
PACK_GRES=l40s.4

BLOCKER_SRUN_PID=""
BLOCKER_JOBID=""
cleanup_blocker() {
    if [ -n "${BLOCKER_JOBID}" ]; then
        scancel "$BLOCKER_JOBID" >/dev/null 2>&1 || true
    fi
    if [ -n "${BLOCKER_SRUN_PID}" ]; then
        kill "$BLOCKER_SRUN_PID" >/dev/null 2>&1 || true
        wait "$BLOCKER_SRUN_PID" 2>/dev/null || true
    fi
}
trap cleanup_blocker EXIT

# Occupy 3 of 4 GPUs so the two shard jobs land on the last card.
srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:l40s:3 \
     --cpus-per-task=2 --mem=4G --time="$DEFAULT_SRUN_TIME" bash -lc "
echo \$SLURM_JOB_ID > '${OUT}/blocker/jid.txt'
sleep 180
" >/dev/null 2>&1 &
BLOCKER_SRUN_PID=$!

for _i in $(seq 1 90); do
    if [ -s "$OUT/blocker/jid.txt" ]; then
        break
    fi
    sleep 1
done
BLOCKER_JOBID=$(tr -d '[:space:]' < "$OUT/blocker/jid.txt" 2>/dev/null || true)
if [ -z "$BLOCKER_JOBID" ]; then
    _emit "NA,NA" FAIL 0 "blocker did not start (need 3 free full GPUs to pack shards)"
    exit 0
fi

run_hold() {
    local dest="$1" hold="$2"
    srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${PACK_GRES}:1 --cpus-per-task=4 --mem=6G \
         --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=5
nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1 > '${dest}/uuid.txt'
: > '${dest}/smi_pids.txt'
(
    for t in \$(seq 1 ${hold}); do
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | grep -v softmig >> '${dest}/smi_pids.txt'
        sleep 1
    done
) &
./build/test/runtime_hold 1024 ${hold} > '${dest}/run.out' 2>&1
_copy_softmig_log \$SLURM_JOB_ID '${dest}/softmig.log'
echo \$SLURM_JOB_ID > '${dest}/jid.txt'
" >/dev/null 2>&1
}

run_hold "$OUT/jobA" 40 &
PIDA=$!
sleep 2
run_hold "$OUT/jobB" 36 &
PIDB=$!
wait $PIDA $PIDB

jidA=$(cat "$OUT/jobA/jid.txt" 2>/dev/null || echo NA)
jidB=$(cat "$OUT/jobB/jid.txt" 2>/dev/null || echo NA)
slogA="$OUT/jobA/softmig.log"
slogB="$OUT/jobB/softmig.log"
uuidA=$(tr -d '[:space:]' < "$OUT/jobA/uuid.txt" 2>/dev/null || true)
uuidB=$(tr -d '[:space:]' < "$OUT/jobB/uuid.txt" 2>/dev/null || true)

if [ ! -s "$slogA" ] || [ ! -s "$slogB" ]; then
    _emit "${jidA},${jidB}" FAIL 0 "missing one or both softmig logs"
    exit 0
fi

pidA=$(grep -oE "Found current process PID [0-9]+" "$slogA" | head -1 | grep -oE "[0-9]+")
pidB=$(grep -oE "Found current process PID [0-9]+" "$slogB" | head -1 | grep -oE "[0-9]+")
[ -z "$pidA" ] && pidA=0
[ -z "$pidB" ] && pidB=0

crossA_regs_B=$(grep -c "Found current process PID ${pidB}" "$slogA" 2>/dev/null || true)
crossB_regs_A=$(grep -c "Found current process PID ${pidA}" "$slogB" 2>/dev/null || true)
A_skipped_B=$(grep -c "PID ${pidB} - different cgroup" "$slogA" 2>/dev/null || true)
B_skipped_A=$(grep -c "PID ${pidA} - different cgroup" "$slogB" 2>/dev/null || true)
A_saw_B=$(grep -c -w "$pidB" "$OUT/jobA/smi_pids.txt" 2>/dev/null || true)
B_saw_A=$(grep -c -w "$pidA" "$OUT/jobB/smi_pids.txt" 2>/dev/null || true)

metric="uuidA=${uuidA:-none} uuidB=${uuidB:-none} A->B_skip=${A_skipped_B} B->A_skip=${B_skipped_A} smiA_sawB=${A_saw_B} smiB_sawA=${B_saw_A} crossA=${crossA_regs_B} crossB=${crossB_regs_A}"

if [ "$pidA" = "0" ] || [ "$pidB" = "0" ]; then
    _emit "${jidA},${jidB}" FAIL 0 "missing Found PID ($metric)"
    exit 0
fi

if [ -z "$uuidA" ] || [ -z "$uuidB" ] || [ "$uuidA" != "$uuidB" ]; then
    _emit "${jidA},${jidB}" PARTIAL 0 "different GPUs: $metric"
    exit 0
fi

# Same GPU: SoftMig must not register the other PID as current, and must
# have skipped it in the NVML filter (accounting isolation). nvidia-smi 595
# fetches its process list through NVML's private export table, so it can
# still list both PIDs unless nvidia-smi-hook.sh is on PATH: that is reported
# as LEAK (visibility only), never silently as PASS.
if [ "$crossA_regs_B" != "0" ] || [ "$crossB_regs_A" != "0" ]; then
    _emit "${jidA},${jidB}" FAIL "leak" "$metric"
elif [ "$A_saw_B" != "0" ] || [ "$B_saw_A" != "0" ]; then
    _emit "${jidA},${jidB}" LEAK "visible" "accounting isolated, nvidia-smi shows other job: $metric"
elif [ "$A_skipped_B" != "0" ] && [ "$B_skipped_A" != "0" ]; then
    _emit "${jidA},${jidB}" PASS "isolated" "$metric"
else
    _emit "${jidA},${jidB}" PARTIAL 0 "overlap but no skip logs: $metric"
fi
