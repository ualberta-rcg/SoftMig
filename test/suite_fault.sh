#!/bin/bash
# Suite: fault — kill and stop workers while they contend for the lock.
# 16 stress_alloc workers run for FAULT_SECS. Meanwhile:
#   - 12 times: SIGKILL a random worker (some die holding the region lock)
#     and start a replacement;
#   - 4 rounds of: 4 random workers SIGSTOPped for 25 s, then SIGCONTed
#     (a stopped lock holder must be waited for, never robbed).
# Afterwards one short "sweeper" process runs (its init clears dead slots)
# and shrreg_check validates the region.
#
# PASS iff: no watchdog HANG, every worker that was not killed exits 0,
# the sweeper exits 0, shrreg_check OK, and no forced-recovery (lock
# stolen) lines in the job log.

SUITE=fault
DEFAULT_SRUN_TIME=00:10:00
. "$(dirname "$0")/suite_common.sh"

SECS="${FAULT_SECS:-150}"
MB="${FAULT_MB:-96}"

_srun_capture srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=16 --mem=16G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=1
echo \$SLURM_JOB_ID > '${OUT}/jid.txt'
_hang_watchdog 480 '${OUT}'
declare -A killed
pids=()
start() { ./build/test/stress_alloc 4 \$1 ${MB} \$((RANDOM % 2)) > '${OUT}'/w_\$2.log 2>&1 & pids+=(\$!); }
for i in \$(seq 1 16); do start ${SECS} \$i; done
sleep 8
for k in \$(seq 1 12); do
    v=\${pids[\$((RANDOM % \${#pids[@]}))]}
    if [ -z \"\${killed[\$v]:-}\" ] && kill -9 \$v 2>/dev/null; then
        killed[\$v]=1
        echo \"killed \$v\" >> '${OUT}/faults.txt'
        start \$((${SECS} / 3)) r\$k
    fi
    sleep 1.5
done
nstop=0
for round in 1 2 3 4; do
    stopped=()
    for p in \$(printf '%s\n' \"\${pids[@]}\" | shuf); do
        [ -n \"\${killed[\$p]:-}\" ] && continue
        kill -0 \$p 2>/dev/null || continue
        kill -STOP \$p && stopped+=(\$p) && echo \"round \$round stopped \$p\" >> '${OUT}/faults.txt'
        [ \${#stopped[@]} -ge 4 ] && break
    done
    nstop=\$((nstop + \${#stopped[@]}))
    sleep 25
    for p in \"\${stopped[@]}\"; do kill -CONT \$p; done
    sleep 2
done
rc=0
for p in \"\${pids[@]}\"; do
    [ -n \"\${killed[\$p]:-}\" ] && { wait \$p 2>/dev/null; continue; }
    wait \$p || { rc=\$((rc+1)); echo \"failed \$p\" >> '${OUT}/faults.txt'; }
done
./build/test/stress_alloc 1 0 1 > '${OUT}/sweeper.log' 2>&1; sw=\$?
./build/test/shrreg_check > '${OUT}/check.txt' 2>&1; ck=\$?
echo \"failed_workers=\$rc sweeper_rc=\$sw check_rc=\$ck killed=\${#killed[@]} stopped=\$nstop\" > '${OUT}/result.txt'
_hang_disarm '${OUT}'
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/softmig.log'
" >/dev/null 2>&1

jid=$(cat "$OUT/jid.txt" 2>/dev/null || echo NA)
res=$(cat "$OUT/result.txt" 2>/dev/null)
get() { printf '%s\n' "$res" | grep -oE "$1=[0-9]+" | cut -d= -f2; }
failed=$(get failed_workers); sw=$(get sweeper_rc); ck=$(get check_rc)
stolen=$(grep -cE "forcing recovery|Owner pid equals" "$OUT/softmig.log" 2>/dev/null)
recovered=$(grep -cE "Kick dead|died holding the lock" "$OUT/softmig.log" 2>/dev/null)
waited=$(grep -c "Waiting [0-9]*s for shrreg lock" "$OUT/softmig.log" 2>/dev/null)
check=$(sed 's/shrreg_check: //' "$OUT/check.txt" 2>/dev/null)
metric="${res:-no result} stolen=${stolen:-0} recovered=${recovered:-0} waited_for_live_holder=${waited:-0} check=[${check}]"

if [ -n "$res" ] && [ "${failed:-1}" -eq 0 ] && [ "${sw:-1}" -eq 0 ] && [ "${ck:-1}" -eq 0 ] && [ "${stolen:-0}" -eq 0 ]; then
    _emit "$jid" PASS "$(get killed)" "$metric"
else
    _emit "$jid" FAIL "$(get killed)" "$metric"
fi
