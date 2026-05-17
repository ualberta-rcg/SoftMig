#!/bin/bash
# OOM validation suite — run inside a softmig slice allocation.
# Exercises the post-2026-05-17 default behavior (killer OFF) and the legacy
# opt-in path (SOFTMIG_ENABLE_OOM_KILLER=1) so we can confirm both still work
# end-to-end. Each scenario writes its own logs under $OUT.
set -u

OUT="${OUT:-/tmp/softmig_oomval_${SLURM_JOB_ID:-$$}}"
SOFTMIG_ROOT="${SOFTMIG_ROOT:-/scratch/rahimk/SoftMig}"
mkdir -p "$OUT"

cd "$SOFTMIG_ROOT"

echo "=== softmig OOM validation ==="
echo "job=$SLURM_JOB_ID node=$(hostname) out=$OUT"
echo "lib hash: $(sha256sum /usr/local/lib/libsoftmig.so 2>/dev/null | awk '{print $1}')"
echo "slice cfg:"
sed 's/^/  /' "/var/run/softmig/${SLURM_JOB_ID}.conf" 2>/dev/null || echo "  (no config file)"
echo

# ----------------------------------------------------------------------------
# Scenario 1: single-process OOM behavior (test_oom_behavior probe).
# Expect: cudaErrorMemoryAllocation, process keeps running, full memory budget
# accessible, no SIGKILL.
# ----------------------------------------------------------------------------
echo "=== [scenario 1] test_oom_behavior (default, killer off) ==="
SCEN1_LOG="$OUT/scenario1_oom_probe.log"
( unset SOFTMIG_ENABLE_OOM_KILLER
  exec build/test/test_oom_behavior
) >"$SCEN1_LOG" 2>&1
rc=$?
grep -q "cudaErrorMemoryAllocation" "$SCEN1_LOG" && \
    echo "  PASS: cudaErrorMemoryAllocation returned" || echo "  FAIL: no OOM error code"
grep -q "OOM at iter=" "$SCEN1_LOG" && \
    echo "  PASS: incremental allocation hit OOM near the limit" || \
    echo "  WARN: did not see iterative OOM line"
util=$(awk -F'[ =]' '/OOM at iter=/{for(i=1;i<=NF;i++) if ($i=="iter") {iter=$(i+1)}; for(i=1;i<=NF;i++) if ($i=="after") print $(i+1)}' "$SCEN1_LOG")
[ -n "$util" ] && echo "  INFO: ${util} MB allocated before OOM"
grep -q "Killed" "$SCEN1_LOG" && echo "  FAIL: process was Killed (SIGKILL leaked)" || \
    echo "  PASS: no SIGKILL"
echo "  exit=$rc"
echo

# ----------------------------------------------------------------------------
# Scenario 2: gpu_burn && gpu_burn — second process must see first process's
# memory via NVML and refuse to over-allocate.
# Use MB sized so that 2x sum exceeds the slice limit.
#   .2 slice ~24 GB -> 16 GB per proc (2x = 32 GB > 24 GB)
#   .4 slice ~11.5 GB -> 8 GB per proc (2x = 16 GB > 11.5 GB)
#   anything else: 1024 (won't actually OOM)
# ----------------------------------------------------------------------------
echo "=== [scenario 2] gpu_burn_lite serial && gpu_burn_lite (mem-aware) ==="
TOTAL_MB=$(awk -F= '/^TotalMemoryMB|TOTAL_MEM_MB/{print $2}' "/var/run/softmig/${SLURM_JOB_ID}.conf" 2>/dev/null | head -1)
if [ -z "$TOTAL_MB" ]; then
    # Probe via nvidia-smi inside the slice
    TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
fi
TOTAL_MB="${TOTAL_MB:-12000}"
MB_PER=$((TOTAL_MB * 70 / 100))   # 70% each -> 140% total -> guaranteed OOM
echo "  slice total ~${TOTAL_MB} MB, each gpu_burn asks for ${MB_PER} MB"

SCEN2_LOG1="$OUT/scenario2_proc1.log"
SCEN2_LOG2="$OUT/scenario2_proc2.log"
unset SOFTMIG_ENABLE_OOM_KILLER

build/test/gpu_burn_lite "$MB_PER" 30 >"$SCEN2_LOG1" 2>&1 &
P1=$!
sleep 5  # let proc1 actually allocate before proc2 starts (mimics &&-like staggering)
build/test/gpu_burn_lite "$MB_PER" 25 >"$SCEN2_LOG2" 2>&1 &
P2=$!

wait $P1; r1=$?
wait $P2; r2=$?

echo "  proc1 exit=$r1 (PASS expected: 0)"
echo "  proc2 exit=$r2 (PASS expected: non-zero, cudaErrorMemoryAllocation)"
grep -E "allocated|out of memory" "$SCEN2_LOG1" | head -3 | sed 's/^/    p1: /'
grep -E "allocated|out of memory" "$SCEN2_LOG2" | head -3 | sed 's/^/    p2: /'
if [ "$r1" = "0" ] && [ "$r2" != "0" ]; then
    echo "  PASS: 2nd proc was rejected with OOM, 1st proc completed"
else
    echo "  CHECK: review logs in $OUT"
fi
echo

# ----------------------------------------------------------------------------
# Scenario 3: 4x gpu_burn_lite concurrent — same slice, total way over limit.
# All four start ~simultaneously, racing on the allocator. With proper locked
# accounting, whoever fits gets allocations; the rest get OOM errors and exit
# with non-zero — no SIGKILL.
# ----------------------------------------------------------------------------
echo "=== [scenario 3] 4x gpu_burn_lite simultaneous (default, killer off) ==="
SCEN3_DIR="$OUT/scenario3"
mkdir -p "$SCEN3_DIR"
MB_PER3=$((TOTAL_MB * 40 / 100))   # 40% each, 4 procs => 160% total
echo "  4 x ${MB_PER3} MB per proc"
unset SOFTMIG_ENABLE_OOM_KILLER

pids=()
for i in 1 2 3 4; do
    build/test/gpu_burn_lite "$MB_PER3" 20 >"$SCEN3_DIR/proc_$i.log" 2>&1 &
    pids+=("$!")
done
echo "  pids: ${pids[*]}"

# Wait + capture exit codes
declare -A exits
for p in "${pids[@]}"; do
    wait "$p"
    exits["$p"]=$?
done

success=0; oom=0; signaled=0
for p in "${pids[@]}"; do
    rc=${exits[$p]}
    if [ "$rc" = "0" ]; then success=$((success+1));
    elif [ "$rc" -gt 128 ]; then signaled=$((signaled+1));
    else oom=$((oom+1));
    fi
    echo "    pid $p exit=$rc"
done
echo "  summary: success=$success oom=$oom signaled(SIGKILL?)=$signaled"
[ "$signaled" = "0" ] && echo "  PASS: no SIGKILLs" || echo "  FAIL: $signaled SIGKILLs"
[ "$success" -ge 1 ] && echo "  PASS: at least one proc completed" || echo "  FAIL: no procs completed"
[ "$oom" -ge 1 ] && echo "  PASS: at least one proc OOM-rejected" || echo "  WARN: no OOM rejection (try a larger MB_PER)"
echo

# ----------------------------------------------------------------------------
# Scenario 4: same 4x test but with SOFTMIG_ENABLE_OOM_KILLER=1, verifying
# the legacy kill path still works for users who want it.
# ----------------------------------------------------------------------------
echo "=== [scenario 4] 4x gpu_burn_lite + SOFTMIG_ENABLE_OOM_KILLER=1 (legacy) ==="
SCEN4_DIR="$OUT/scenario4"
mkdir -p "$SCEN4_DIR"
export SOFTMIG_ENABLE_OOM_KILLER=1

pids=()
for i in 1 2 3 4; do
    build/test/gpu_burn_lite "$MB_PER3" 20 >"$SCEN4_DIR/proc_$i.log" 2>&1 &
    pids+=("$!")
done
declare -A exits4
for p in "${pids[@]}"; do
    wait "$p"
    exits4["$p"]=$?
done

success=0; signaled=0; other=0
for p in "${pids[@]}"; do
    rc=${exits4[$p]}
    if [ "$rc" = "0" ]; then success=$((success+1));
    elif [ "$rc" -gt 128 ]; then signaled=$((signaled+1));
    else other=$((other+1));
    fi
    echo "    pid $p exit=$rc"
done
echo "  summary: success=$success signaled=$signaled other=$other"
[ "$signaled" -ge 1 ] && echo "  PASS: at least one SIGKILL (legacy killer fired)" || \
    echo "  CHECK: no SIGKILLs (maybe allocator-side OOM was enough)"
unset SOFTMIG_ENABLE_OOM_KILLER

# Capture softmig log
SLOG="/var/log/softmig/${SLURM_JOB_ID}.log"
[ -r "$SLOG" ] && cp "$SLOG" "$OUT/softmig.log" && echo "  softmig.log copied"

echo
echo "=== done. results: $OUT ==="
