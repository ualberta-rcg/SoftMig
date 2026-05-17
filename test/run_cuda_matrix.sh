#!/bin/bash
# Runtime matrix: for each CUDA version available in cvmfs, rebuild ONLY the
# test probes against that toolkit, then run them against the installed
# /usr/local/lib/libsoftmig.so (which was built against the canonical CUDA
# version). This validates that the same hot-swapped library serves CUDA
# runtimes 12.2 through 13.2.
#
# Run INSIDE an srun on a softmig slice.
set -u

VERSIONS=(12.2 12.6 12.9 13.2)
SOFTMIG_ROOT="${SOFTMIG_ROOT:-/scratch/rahimk/SoftMig}"
OUT="${OUT:-/tmp/softmig_cudamatrix_${SLURM_JOB_ID:-$$}}"
mkdir -p "$OUT"

cd "$SOFTMIG_ROOT"

echo "=== softmig cross-CUDA runtime matrix ==="
echo "job=${SLURM_JOB_ID:-N/A} node=$(hostname)"
echo "installed libsoftmig.so: $(sha256sum /usr/local/lib/libsoftmig.so 2>/dev/null | awk '{print $1}')"
echo "slice cfg:"
sed 's/^/  /' "/var/run/softmig/${SLURM_JOB_ID}.conf" 2>/dev/null || echo "  (none)"
echo

printf "%-8s  %-12s  %-12s  %-6s  %s\n" CUDA OOM_PROBE BURN_2X RESULT NOTES | tee "$OUT/summary.txt"
printf "%-8s  %-12s  %-12s  %-6s  %s\n" "----" "---------" "------" "------" "-----" | tee -a "$OUT/summary.txt"

for v in "${VERSIONS[@]}"; do
    SUB="$OUT/cuda-$v"
    mkdir -p "$SUB"
    LOG="$SUB/build.log"

    # Rebuild ONLY the test binaries with this CUDA toolkit. Library stays
    # the canonical hot-swapped one.
    (
        module purge 2>/dev/null
        module load StdEnv/2023 2>/dev/null
        module load cuda/$v 2>/dev/null
        nvcc --version 2>&1 | tail -1
        echo "--- rebuilding test binaries ---"
        cd "$SOFTMIG_ROOT"
        # Force test target rebuild against this toolkit
        rm -f build/test/test_oom_behavior build/test/gpu_burn_lite
        cmake --build build --target test_oom_behavior gpu_burn_lite -j 4
    ) > "$LOG" 2>&1
    if [ ! -x build/test/test_oom_behavior ] || [ ! -x build/test/gpu_burn_lite ]; then
        printf "%-8s  %-12s  %-12s  %-6s  %s\n" \
            "cuda/$v" "build-skip" "build-skip" FAIL "test build failed (see $LOG)" \
            | tee -a "$OUT/summary.txt"
        continue
    fi

    # ----- test 1: OOM probe -----
    (
        module purge 2>/dev/null
        module load StdEnv/2023 2>/dev/null
        module load cuda/$v 2>/dev/null
        unset SOFTMIG_ENABLE_OOM_KILLER
        build/test/test_oom_behavior
    ) > "$SUB/oom_probe.log" 2>&1
    rc1=$?
    if grep -q "cudaErrorMemoryAllocation" "$SUB/oom_probe.log" && \
       ! grep -q "Killed" "$SUB/oom_probe.log" && [ $rc1 -eq 0 ]; then
        oom_status=PASS
        oom_note="$(grep -m1 "OOM at iter=" "$SUB/oom_probe.log" | sed -E 's/.*after ([0-9.]+ MB).*/\1/' | head -1)"
    else
        oom_status=FAIL
        oom_note="rc=$rc1 see oom_probe.log"
    fi

    # ----- test 2: gpu_burn serial && gpu_burn (mem-aware) -----
    TOTAL_MB=$(awk -F= '/^CUDA_DEVICE_MEMORY_LIMIT/{gsub(/[^0-9.]/,"",$2); print int($2)}' \
        "/var/run/softmig/${SLURM_JOB_ID}.conf" 2>/dev/null | head -1)
    [ -z "$TOTAL_MB" ] && TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    MB_PER=$((TOTAL_MB * 70 / 100))

    (
        module purge 2>/dev/null
        module load StdEnv/2023 2>/dev/null
        module load cuda/$v 2>/dev/null
        unset SOFTMIG_ENABLE_OOM_KILLER
        build/test/gpu_burn_lite "$MB_PER" 12 > "$SUB/burn_p1.log" 2>&1 &
        P1=$!
        sleep 4
        build/test/gpu_burn_lite "$MB_PER" 8 > "$SUB/burn_p2.log" 2>&1 &
        P2=$!
        wait $P1; r1=$?
        wait $P2; r2=$?
        echo "p1=$r1 p2=$r2"
    ) > "$SUB/burn_run.log" 2>&1
    p1_rc=$(awk -F= '{print $2}' <(awk '{for(i=1;i<=NF;i++) if($i~/^p1=/) print $i}' "$SUB/burn_run.log"))
    p2_rc=$(awk -F= '{print $2}' <(awk '{for(i=1;i<=NF;i++) if($i~/^p2=/) print $i}' "$SUB/burn_run.log"))
    if [ "$p1_rc" = "0" ] && [ -n "$p2_rc" ] && [ "$p2_rc" != "0" ] && [ "$p2_rc" -lt 128 ] 2>/dev/null && \
       grep -q "out of memory" "$SUB/burn_p2.log"; then
        burn_status=PASS
        burn_note="p1=0 p2=$p2_rc (oom)"
    else
        burn_status=FAIL
        burn_note="p1=$p1_rc p2=$p2_rc"
    fi

    overall=PASS
    [ "$oom_status" = "FAIL" -o "$burn_status" = "FAIL" ] && overall=FAIL
    printf "%-8s  %-12s  %-12s  %-6s  %s\n" \
        "cuda/$v" "$oom_status" "$burn_status" "$overall" \
        "${oom_note}; burn ${burn_note}" \
        | tee -a "$OUT/summary.txt"
done

echo
echo "Results: $OUT"
