#!/bin/bash
# Suite: passive — proves "off" really is off, and "on" really hooks.
#
# SLICE=l40s   -> full GPU, no SoftMig config: passive_probe --expect passive
#                 (dlsym/cuGetProcAddress hand out raw driver pointers for every
#                 flag, direct-linked calls return driver values, no shared
#                 region, no signal handlers, no init lines in the log).
# SLICE=l40s.N -> sliced job: passive_probe --expect enabled (enforcement entry
#                 points incl. per-thread-stream variants resolve to hooks).
#
# Both modes: FAIL if the softmig log contains an UNHOOKED line.

SUITE=passive
. "$(dirname "$0")/suite_common.sh"

if [ "${SLICE}" = "l40s" ]; then expect=passive; else expect=enabled; fi

srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=2 --mem=4G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=5
./build/test/passive_probe --expect ${expect} > '${OUT}/probe.log' 2>&1
echo \$? > '${OUT}/rc.txt'
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/softmig.log'
echo \$SLURM_JOB_ID > '${OUT}/jid.txt'
" > "$OUT/srun.log" 2>&1

jid=$(cat "$OUT/jid.txt" 2>/dev/null || echo NA)
rc=$(cat "$OUT/rc.txt" 2>/dev/null || echo NA)
plog="$OUT/probe.log"
slog="$OUT/softmig.log"

if [ ! -s "$plog" ]; then
    _emit "$jid" FAIL 0 "no probe output (see srun.log)"
    exit 0
fi

pass_n=$(grep -c '^PASS:' "$plog")
fail_n=$(grep -c '^FAIL:' "$plog")
unhooked=0
[ -s "$slog" ] && unhooked=$(grep -c 'UNHOOKED' "$slog")
detail="expect=${expect} rc=${rc} pass=${pass_n} fail=${fail_n} unhooked=${unhooked}"

ok=1
[ "$rc" = "0" ] || { ok=0; detail="$detail $(grep -m1 '^FAIL:' "$plog")"; }
[ "$unhooked" = "0" ] || ok=0

if [ "$expect" = "passive" ]; then
    if [ -s "$slog" ]; then
        grep -q "softmig disabled (passive mode)" "$slog" || { ok=0; detail="$detail missing-passive-marker"; }
        if grep -qE 'Initializing\.\.\.\.\.|shrreg created|set_task_pid|init_utilization_watcher' "$slog"; then
            ok=0; detail="$detail passive-mode-initialized"
        fi
    else
        ok=0; detail="$detail no-softmig-log"
    fi
else
    [ -s "$slog" ] && grep -q "Read CUDA_DEVICE_MEMORY_LIMIT=" "$slog" \
        || { ok=0; detail="$detail missing-limit-read"; }
fi

if [ "$ok" = "1" ]; then
    _emit "$jid" PASS "$pass_n" "$detail"
else
    _emit "$jid" FAIL "$pass_n" "$detail"
fi
