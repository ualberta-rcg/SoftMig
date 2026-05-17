#!/bin/bash
# Build SoftMig against every cuda/X.Y module in cvmfs and record pass/fail.
# Final state: the build dir is reset to the LAST_KEEP version so the in-place
# libsoftmig.so on disk matches what is currently installed on the reservation.
set -u

VERSIONS=(11.8 12.2 12.6 12.9 13.2)
LAST_KEEP="${LAST_KEEP:-12.6}"
SOFTMIG_ROOT="${SOFTMIG_ROOT:-/scratch/rahimk/SoftMig}"
OUT="${OUT:-/scratch/rahimk/SoftMig/test/results/build_matrix_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

cd "$SOFTMIG_ROOT"

printf "%-8s  %-6s  %s\n" CUDA RESULT FIRST_ERROR | tee "$OUT/summary.txt"
printf "%-8s  %-6s  %s\n" "----" "------" "-----------" | tee -a "$OUT/summary.txt"

for v in "${VERSIONS[@]}"; do
    LOG="$OUT/build_cuda${v}.log"
    rm -rf build
    (
        module purge 2>/dev/null
        module load StdEnv/2023 2>/dev/null
        module load cuda/$v 2>/dev/null
        nvcc --version 2>&1 | tail -1
        echo "--- cmake/build ---"
        ./build.sh
    ) > "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -s build/libsoftmig.so ]; then
        hash=$(sha256sum build/libsoftmig.so | awk '{print substr($1,1,16)}')
        printf "%-8s  %-6s  %s\n" "cuda/$v" PASS "lib=$hash" | tee -a "$OUT/summary.txt"
    else
        first_err=$(grep -m1 "error:" "$LOG" | head -1 | sed 's|.*/include/||' | cut -c1-80)
        [ -z "$first_err" ] && first_err="rc=$rc (see $LOG)"
        printf "%-8s  %-6s  %s\n" "cuda/$v" FAIL "$first_err" | tee -a "$OUT/summary.txt"
    fi
done

# Reset to the canonical version
echo
echo "Resetting build to cuda/$LAST_KEEP..."
rm -rf build
(
    module purge 2>/dev/null
    module load StdEnv/2023 2>/dev/null
    module load cuda/$LAST_KEEP 2>/dev/null
    ./build.sh
) > "$OUT/build_final_cuda${LAST_KEEP}.log" 2>&1
if [ -s build/libsoftmig.so ]; then
    echo "Final build OK: $(sha256sum build/libsoftmig.so | awk '{print $1}')"
else
    echo "Final reset build FAILED — see $OUT/build_final_cuda${LAST_KEEP}.log"
fi

echo
echo "Results: $OUT"
