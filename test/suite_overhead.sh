#!/bin/bash
# Suite: overhead — per-call cost of SoftMig wrappers (informational).
# bench_overhead in a full-GPU (passive) job and in a ${SLICE} slice:
# direct-linked calls (through libsoftmig's exports) vs the driver's own
# functions, for cuLaunchKernel, cuMemAlloc+cuMemFree and cuMemGetInfo.
# Emits INFO with the ns/call deltas.

SUITE=overhead
. "$(dirname "$0")/suite_common.sh"

for gres in l40s "$SLICE"; do
    sub="$OUT/$gres"; mkdir -p "$sub"
    _srun_capture srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${gres}:1 --cpus-per-task=2 --mem=4G \
         --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
echo \$SLURM_JOB_ID > '${sub}/jid.txt'
_hang_watchdog 240 '${sub}'
./build/test/bench_overhead 100000 10000 > '${sub}/bench.log' 2>&1
_hang_disarm '${sub}'
" >/dev/null 2>&1
done

jids=$(cat "$OUT"/*/jid.txt 2>/dev/null | paste -sd,)
fmt() { awk '/bench_overhead/ {split($3,a,"="); split($4,b,"="); split($5,c,"="); printf "%s=%s/%s(+%s) ", $2, a[2], b[2], c[2]}' "$1" 2>/dev/null; }
detail="passive: $(fmt "$OUT/l40s/bench.log")| slice: $(fmt "$OUT/$SLICE/bench.log")(hook/real ns, +delta)"
if grep -q "launch_ns" "$OUT/l40s/bench.log" 2>/dev/null; then
    _emit "$jids" INFO "$(awk '/launch_ns/ {split($5,c,"="); print c[2]}' "$OUT/l40s/bench.log")" "$detail"
else
    _emit "$jids" FAIL 0 "bench did not run: $(tail -1 "$OUT/l40s/bench.log" 2>/dev/null)"
fi
