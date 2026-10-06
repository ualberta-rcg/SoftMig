#!/bin/bash
# Suite: fork — fork() after CUDA init, in a slice and on a full GPU.
# fork_probe: parent CUDA + allocations, 3 CPU-only children, 1 child that
# execs stress_alloc (a CUDA process started from a CUDA process). In the
# slice, shrreg_check must also find the region consistent afterwards.
# (The Python DataLoader-fork / spawn case lives in suite_frameworks.)
#
# PASS iff fork_probe RESULT: PASS in both modes and, in the slice, the
# region check is OK.

SUITE=fork
. "$(dirname "$0")/suite_common.sh"

for gres in "$SLICE" l40s; do
    sub="$OUT/$gres"; mkdir -p "$sub"
    _srun_capture srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${gres}:1 --cpus-per-task=8 --mem=8G \
         --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
echo \$SLURM_JOB_ID > '${sub}/jid.txt'
_hang_watchdog 240 '${sub}'
./build/test/fork_probe ./build/test/stress_alloc > '${sub}/fork_probe.log' 2>&1
if [ -f /var/run/softmig/\$SLURM_JOB_ID.conf ]; then
  ./build/test/shrreg_check > '${sub}/check.txt' 2>&1 || true
fi
_hang_disarm '${sub}'
_copy_softmig_log \$SLURM_JOB_ID '${sub}/softmig.log'
" >/dev/null 2>&1
    [ -f "$sub/HANG" ] && cp "$sub/HANG" "$OUT/HANG"
done

jids=$(cat "$OUT/$SLICE/jid.txt" "$OUT/l40s/jid.txt" 2>/dev/null | paste -sd,)
rs=$(grep -m1 "^RESULT" "$OUT/$SLICE/fork_probe.log" 2>/dev/null)
rf=$(grep -m1 "^RESULT" "$OUT/l40s/fork_probe.log" 2>/dev/null)
ck=$(sed 's/shrreg_check: //' "$OUT/$SLICE/check.txt" 2>/dev/null | awk '{print $1}')
metric="slice=[${rs:-none}] full=[${rf:-none}] region=${ck:-none} fails=$(grep -h '^FAIL' "$OUT"/*/fork_probe.log 2>/dev/null | paste -sd';')"
if [[ "$rs" == "RESULT: PASS"* ]] && [[ "$rf" == "RESULT: PASS"* ]] && [ "$ck" = OK ]; then
    _emit "$jids" PASS 2 "$metric"
else
    _emit "$jids" FAIL 0 "$metric"
fi
