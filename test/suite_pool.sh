#!/bin/bash
# Suite: pool — regression test for the pool-allocation free failure
# (cuMemAllocFromPoolAsync was never tracked, so cuMemFreeAsync returned -1
# for pool allocations and device memory leaked until the card was
# exhausted — the failure that killed the JAX cuda_async condense-sweep
# jobs).
#
# SLICE=l40s   -> full GPU, passive mode: test_pool_free --passive
# SLICE=l40s.N -> sliced, enabled mode: test_pool_free --limit-bytes <limit
#                 (limit read from /var/run/softmig/<jobid>.conf on the node)
#
# PASS iff the binary exits 0 AND the softmig log shows the expected mode:
#   passive: contains "softmig disabled (passive mode)", no tracking lines
#   enabled: contains "Read CUDA_DEVICE_MEMORY_LIMIT=" and a "Device 0 OOM"
#            line from the stage-6 over-limit rejection.

SUITE=pool
. "$(dirname "$0")/suite_common.sh"

srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=4 --mem=8G \
     --time="$DEFAULT_SRUN_TIME" \
     bash -lc "
export SLICE='${SLICE}' CUDA_VER='${CUDA_VER}' SOFTMIG_ROOT='${SOFTMIG_ROOT}' OUT='${OUT}'
bash '${SOFTMIG_ROOT}/test/suite_pool_inner.sh'
" > "$OUT/srun.log" 2>&1

jid=$(cat "$OUT/jid.txt" 2>/dev/null || echo NA)
rc=$(cat "$OUT/rc.txt" 2>/dev/null || echo NA)
mode=$(awk -F= '/^mode=/{print $2}' "$OUT/mode.txt" 2>/dev/null || echo unknown)
tlog="$OUT/test_pool_free.log"
slog="$OUT/softmig.log"
detail="mode=${mode} rc=${rc}"

if [ ! -s "$tlog" ]; then
    _emit "$jid" FAIL 0 "no test output (see srun.log)"
    exit 0
fi

ok=1
if [ "$rc" != "0" ]; then
    ok=0
    fail_line=$(grep -m1 '^FAIL:' "$tlog")
    detail="${detail} ${fail_line:-rc=$rc}"
fi

if [ -s "$slog" ]; then
    if [ "$mode" = "passive" ]; then
        grep -q "softmig disabled (passive mode)" "$slog" \
            || { ok=0; detail="$detail missing-passive-marker"; }
        if grep -qE 'cuMemAllocAsync:|add_chunk|oom_check_nolock|Device 0 OOM' "$slog"; then
            ok=0; detail="$detail passive-mode-has-tracking-lines"
        fi
    elif [ "$mode" = "enabled" ]; then
        grep -q "Read CUDA_DEVICE_MEMORY_LIMIT=" "$slog" \
            || { ok=0; detail="$detail missing-limit-read"; }
        grep -q "Device 0 OOM" "$slog" \
            || { ok=0; detail="$detail missing-oom-line"; }
    else
        ok=0; detail="$detail unknown-mode"
    fi
else
    ok=0; detail="$detail no-softmig-log"
fi

stages=$(grep -c '^PASS: stage' "$tlog")
detail="${detail} stages_pass=${stages}"

if [ "$ok" = "1" ]; then
    _emit "$jid" PASS "$stages" "$detail"
else
    _emit "$jid" FAIL "$stages" "$detail"
fi
