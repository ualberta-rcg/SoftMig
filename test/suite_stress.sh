#!/bin/bash
# Suite: stress — many processes x threads allocating/freeing on one slice.
# One job, three rounds (N=16, 32, 64 processes; 4 threads each; mixed
# cuMemAlloc and cuMemAllocAsync; up to STRESS_MB MiB per buffer, so the
# slice limit is hit and OOM paths run), with nvidia-smi polled throughout.
# After each round shrreg_check validates the shared region.
#
# Then an enforcement round: 16 processes x 4 threads asking for 1 GiB
# buffers; real usage (sum of per-process NVML used_memory) is polled.
#
# PASS iff: no watchdog HANG, every worker exits 0 (OOM is fine, other
# errors are not), shrreg_check OK after every round, zero lock recovery /
# timeout lines in the job log, and in the enforcement round OOMs occurred
# while peak real usage stayed within limit + 512 MiB.

SUITE=stress
DEFAULT_SRUN_TIME=00:15:00
. "$(dirname "$0")/suite_common.sh"

ROUNDS="${STRESS_ROUNDS:-16 32 64}"
SECS="${STRESS_SECS:-30}"
MB="${STRESS_MB:-96}"

_srun_capture srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=16 --mem=32G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=1
echo \$SLURM_JOB_ID > '${OUT}/jid.txt'
_hang_watchdog 780 '${OUT}'
for n in ${ROUNDS}; do
    pids=()
    for i in \$(seq 1 \$n); do
        ./build/test/stress_alloc 4 ${SECS} ${MB} \$((i % 2)) > '${OUT}'/round\${n}_w\${i}.log 2>&1 &
        pids+=(\$!)
    done
    ( while :; do nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >> '${OUT}'/nvsmi_round\${n}.log 2>&1; sleep 2; done ) &
    poller=\$!
    rc=0
    for p in \"\${pids[@]}\"; do wait \$p || rc=\$((rc+1)); done
    kill \$poller 2>/dev/null; wait \$poller 2>/dev/null
    ./build/test/shrreg_check > '${OUT}'/check_round\${n}.txt 2>&1
    echo \"round=\$n failed_workers=\$rc check_rc=\$?\" >> '${OUT}/rounds.txt'
done
# Enforcement under concurrency: 16 processes asking for 1 GiB buffers (far
# beyond the slice). Real usage = sum of per-process used_memory (NVML raw
# values, not SoftMig's virtualized memory.used).
lim=\$(sed -n 's/^CUDA_DEVICE_MEMORY_LIMIT=\\([0-9]*\\).*/\\1/p' /var/run/softmig/\${SLURM_JOB_ID}*.conf | head -1)
pids=()
for i in \$(seq 1 16); do
    ./build/test/stress_alloc 4 20 1024 \$((i % 2)) > '${OUT}'/enforce_w\${i}.log 2>&1 &
    pids+=(\$!)
done
( while :; do nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=\$1} END{print s+0}' >> '${OUT}/enforce_used.log'; sleep 0.5; done ) &
poller=\$!
rc=0
for p in \"\${pids[@]}\"; do wait \$p || rc=\$((rc+1)); done
kill \$poller 2>/dev/null; wait \$poller 2>/dev/null
peak=\$(sort -n '${OUT}/enforce_used.log' | tail -1)
ooms=\$(cat '${OUT}'/enforce_w*.log | grep -oE ' oom=[0-9]+' | awk -F= '{s+=\$2} END{print s+0}')
echo \"limit=\${lim:-0} peak=\${peak:-0} ooms=\$ooms failed_workers=\$rc\" > '${OUT}/enforce.txt'
_hang_disarm '${OUT}'
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/softmig.log'
" >/dev/null 2>&1

jid=$(cat "$OUT/jid.txt" 2>/dev/null || echo NA)
rounds=$(wc -l < "$OUT/rounds.txt" 2>/dev/null || echo 0)
want=$(echo $ROUNDS | wc -w)
failed=$(awk -F'failed_workers=' '{split($2,a," "); s+=a[1]} END{print s+0}' "$OUT/rounds.txt" 2>/dev/null)
badcheck=$(grep -c "check_rc=[1-9]" "$OUT/rounds.txt" 2>/dev/null)
lockmsg=$(grep -cE "Lock shrreg timeout|forcing recovery|Owner pid equals|Fail to lock shrreg|Failed to take lock" "$OUT/softmig.log" 2>/dev/null)
ok=$(cat "$OUT"/round*_w*.log 2>/dev/null | grep -oE " ok=[0-9]+" | awk -F= '{s+=$2} END{print s+0}')
oom=$(cat "$OUT"/round*_w*.log 2>/dev/null | grep -oE " oom=[0-9]+" | awk -F= '{s+=$2} END{print s+0}')
checks=$(cat "$OUT"/check_round*.txt 2>/dev/null | sed 's/shrreg_check: //' | awk '{print $1}' | tr '\n' ',')
enf=$(cat "$OUT/enforce.txt" 2>/dev/null)
eget() { printf '%s\n' "$enf" | grep -oE "$1=[0-9]+" | cut -d= -f2; }
e_lim=$(eget limit); e_peak=$(eget peak); e_oom=$(eget ooms); e_fail=$(eget failed_workers)
enforced=0
if [ -n "$e_lim" ] && [ "${e_lim:-0}" -gt 0 ] && [ "${e_oom:-0}" -gt 0 ] && [ "${e_fail:-1}" -eq 0 ] && \
   [ "${e_peak:-999999}" -le $(( ${e_lim:-0} + 512 )) ]; then
    enforced=1
fi
metric="rounds=${rounds}/${want} failed_workers=${failed:-NA} bad_checks=${badcheck:-NA} lock_msgs=${lockmsg:-0} allocs=${ok} ooms=${oom} checks=${checks%,} enforce=[${enf:-none}]"

if [ "$rounds" -eq "$want" ] && [ "${failed:-1}" -eq 0 ] && [ "${badcheck:-1}" -eq 0 ] && [ "${lockmsg:-0}" -eq 0 ] && [ "$enforced" -eq 1 ]; then
    _emit "$jid" PASS "$ok" "$metric"
else
    _emit "$jid" FAIL "$ok" "$metric"
fi
