#!/bin/bash
# One Slurm allocation, many SoftMig exercises (cuda/12.2, l40s.4 slice).
# Run from repo root or via: srun ... bash test/run_whack.sh
#
# Artifacts: test_results/whack_<timestamp>/

set -u
SOFTMIG_ROOT=/scratch/rahimk/SoftMig
cd "$SOFTMIG_ROOT" || exit 1

TS=$(date +%Y%m%d_%H%M%S)
OUT="${SOFTMIG_ROOT}/test_results/whack_${TS}"
mkdir -p "$OUT"
export SOFTMIG_LOG_LEVEL="${SOFTMIG_LOG_LEVEL:-5}"

module load cuda/12.2

summarize() {
    local tag="$1"
    local f="/var/log/softmig/${SLURM_JOB_ID}.log"
    if [ -r "$f" ]; then
        cp "$f" "${OUT}/${tag}_softmig.log" 2>/dev/null || true
        local n err
        n=$(wc -l < "$f")
        err=$(grep -c 'softmig ERROR' "$f" 2>/dev/null || echo 0)
        echo "[$tag] softmig log lines: $n errors: $err"
    else
        echo "[$tag] no softmig log at $f"
    fi
}

echo "=== whack bundle OUT=$OUT job=${SLURM_JOB_ID:-local} host=$(hostname) ===" | tee "$OUT/00_header.txt"

echo "=== A: runtime_hold (cudart) ===" | tee -a "$OUT/00_header.txt"
./build/test/runtime_hold 384 12 >"$OUT/A_runtime_hold.stdout" 2>&1 || true
summarize A

echo "=== B: 4x runtime_hold (multiproc cudart) ===" | tee -a "$OUT/00_header.txt"
N=4 MB=512 HOLD=14 OUT="${OUT}/B_rt_mp" test/run_multiproc_rt.sh >"$OUT/B_harness.stdout" 2>&1 || true
summarize B

echo "=== C: 4x nvml_probe (direct-linked) ===" | tee -a "$OUT/00_header.txt"
N=4 MB=384 HOLD=12 OUT="${OUT}/C_nvml" test/run_multiproc.sh >"$OUT/C_harness.stdout" 2>&1 || true
summarize C

echo "=== D: 3x gpu_burn_lite (SM stress) ===" | tee -a "$OUT/00_header.txt"
N=3 MB=1024 DUR=22 OUT="${OUT}/D_burn" test/run_burn.sh >"$OUT/D_harness.stdout" 2>&1 || true
summarize D

echo "=== E: 4x gpu_burn_lite OOM-ish (4x3GB .4) ===" | tee -a "$OUT/00_header.txt"
N=4 MB=3072 DUR=18 OUT="${OUT}/E_oom" test/run_burn.sh >"$OUT/E_harness.stdout" 2>&1 || true
summarize E

echo "=== F: nvidia-smi snapshot ===" | tee -a "$OUT/00_header.txt"
nvidia-smi >"$OUT/F_nvsmi.txt" 2>&1 || true

echo "=== G: second CUDA toolkit (12.6) quick hold ===" | tee -a "$OUT/00_header.txt"
if module load cuda/12.6 2>/dev/null; then
    ./build/test/runtime_hold 256 8 >"$OUT/G_rt_hold_126.stdout" 2>&1 || true
    summarize G
else
    echo "[G] skip cuda/12.6" | tee -a "$OUT/00_header.txt"
fi

echo "=== whack DONE $(date) ===" | tee -a "$OUT/00_header.txt"
echo "OUT=$OUT"
