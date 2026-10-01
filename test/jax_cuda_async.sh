#!/bin/bash
# JAX end-to-end test for the pool-allocation free failure (FIX_PLAN Phase 3d).
# Run INSIDE a GPU allocation (passive full GPU or a softmig slice):
#   bash test/jax_cuda_async.sh
#
# Phases (each a fresh python process):
#   1. 200 iterations of a 4096x4096 matmul-reduce under
#      XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async (the condense-sweep config)
#   2. same workload under the default allocator (env unset)
#   3. 50x allocate/free of a 2 GiB array under cuda_async
#
# PASS (both modes): every phase exits 0, no 'cudaFreeAsync failed' or
# 'UNKNOWN ERROR (-1)', and nvidia-smi used memory returns to within 512 MiB
# of the start value after all processes exit.
#
# RESOURCE_EXHAUSTED is mode-dependent:
#   passive (no config file): none expected (full GPU has headroom)
#   enabled (slice): CUDA_ERROR_OUT_OF_MEMORY is the limit working; any
#   RESOURCE_EXHAUSTED without that code, or UNKNOWN ERROR (-1), is FAIL.

set -u
OUT="${OUT:-/tmp/softmig_jax_${SLURM_JOB_ID:-$$}}"
VENV="${JAX_VENV:-${SCRATCH:-/tmp}/jaxvenv}"
MATMUL_ITERS="${MATMUL_ITERS:-200}"
CHURN_ITERS="${CHURN_ITERS:-50}"
mkdir -p "$OUT"

module load python/3.11 cuda/12.6

if [ ! -x "$VENV/bin/python" ]; then
    echo "creating one-time venv at $VENV"
    python -m venv "$VENV" || { echo "FAIL: venv creation failed"; exit 1; }
    # Plain 'jax jaxlib' from the wheelhouse is CPU-only; the CUDA-enabled
    # combo for cp311 is jax 0.10.2 + jax-cuda12-plugin/-pjrt 0.10.2
    # (jaxlib 0.11.x is cp312+ only).
    "$VENV/bin/pip" install --no-index jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt \
        || { echo "FAIL: jax install from wheelhouse failed"; exit 1; }
fi
"$VENV/bin/python" -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)" \
    || { echo "FAIL: jax not usable"; exit 1; }
# Refuse to run on CPU (a CPU-only jaxlib silently falls back;
# the CUDA backend reports platform 'gpu')
"$VENV/bin/python" -c "import jax; d=jax.devices(); print('devices:', [x.platform for x in d]); exit(0 if d[0].platform in ('gpu','cuda') else 1)" \
    || { echo "FAIL: no CUDA jax device (CPU-only jaxlib installed?)"; exit 1; }

used_mem() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        nvidia-smi --id="${CUDA_VISIBLE_DEVICES%%,*}" \
            --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
    else
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
    fi
}

cat > "$OUT/workload.py" <<'EOF'
import sys
import jax
import jax.numpy as jnp

mode = sys.argv[1]
iters = int(sys.argv[2])

x = jnp.ones((4096, 4096), dtype=jnp.float32)
if mode == "matmul":
    for i in range(iters):
        y = (x @ x).sum().block_until_ready()
    print(f"matmul-reduce completed {iters} iterations")
elif mode == "oversize":
    # ~16 GiB: larger than an l40s.4 slice (~11.25 GiB). Must fail with
    # RESOURCE_EXHAUSTED / CUDA_ERROR_OUT_OF_MEMORY and not kill the process.
    try:
        big = jnp.ones((4096, 1024, 1024), dtype=jnp.float32)
        big.block_until_ready()
        print("oversized alloc unexpectedly succeeded")
        sys.exit(2)
    except Exception as e:
        print(f"oversized alloc raised {type(e).__name__}: {e}")
else:
    for i in range(iters):
        big = jnp.ones((512, 1024, 1024), dtype=jnp.float32)  # 2 GiB
        del big
        if i % 10 == 9:
            print(f"churn iteration {i} done")
    print(f"alloc/free churn completed {iters} iterations")
EOF

run_phase() {
    local name="$1" alloc="$2" mode="$3" iters="$4"
    echo "=== ${name} (allocator=${alloc:-default}) ==="
    local log="$OUT/${name}.log"
    if [ -n "$alloc" ]; then
        XLA_PYTHON_CLIENT_ALLOCATOR="$alloc" \
            "$VENV/bin/python" "$OUT/workload.py" "$mode" "$iters" \
            > "$log" 2> "$log.stderr"
    else
        env -u XLA_PYTHON_CLIENT_ALLOCATOR \
            "$VENV/bin/python" "$OUT/workload.py" "$mode" "$iters" \
            > "$log" 2> "$log.stderr"
    fi
    local rc=$?
    echo "  exit=$rc  $(tail -1 "$log")"
    if [ -s "$log.stderr" ]; then
        echo "  stderr (first lines):"
        head -3 "$log.stderr" | sed 's/^/    /'
    fi
    return $rc
}

START_USED=$(used_mem)
echo "job=$SLURM_JOB_ID node=$(hostname) out=$OUT"
echo "nvidia-smi used at start: ${START_USED:-unknown} MiB"
echo

overall=0

run_phase "cuda_async_matmul" cuda_async matmul "$MATMUL_ITERS" || overall=1
run_phase "default_matmul" "" matmul "$MATMUL_ITERS" || overall=1
run_phase "cuda_async_churn" cuda_async churn "$CHURN_ITERS" || overall=1

CONF="/var/run/softmig/${SLURM_JOB_ID:-}.conf"
if [ -n "${SLURM_JOB_ID:-}" ] && [ -r "$CONF" ]; then
    # D4 enabled: deliberately oversized array must OOM with the proper code
    run_phase "cuda_async_oversize" cuda_async oversize 1 || overall=1
fi

sleep 3   # let driver accounting settle after process exit
END_USED=$(used_mem)
echo
echo "nvidia-smi used at end:   ${END_USED:-unknown} MiB"

echo
echo "=== verdict ==="
fail=0

# 1. all phases exited 0 (process not SIGKILLed)
if [ "$overall" = "0" ]; then
    echo "  PASS: all phases exited 0"
else
    echo "  FAIL: at least one phase exited non-zero"
    fail=1
fi

# 2. no cudaFreeAsync failures (the original leak signature) — both modes
if grep -l "cudaFreeAsync failed" "$OUT"/*.stderr >/dev/null 2>&1; then
    echo "  FAIL: 'cudaFreeAsync failed' found in: $(grep -l 'cudaFreeAsync failed' "$OUT"/*.stderr | tr '\n' ' ')"
    fail=1
else
    echo "  PASS: zero 'cudaFreeAsync failed' in stderr"
fi

# UNKNOWN ERROR (-1) is always a fail (untracked-free returning -1)
if grep -l "UNKNOWN ERROR (-1)" "$OUT"/*.stderr >/dev/null 2>&1; then
    echo "  FAIL: 'UNKNOWN ERROR (-1)' found in: $(grep -l 'UNKNOWN ERROR (-1)' "$OUT"/*.stderr | tr '\n' ' ')"
    fail=1
else
    echo "  PASS: zero 'UNKNOWN ERROR (-1)' in stderr"
fi

# 3. RESOURCE_EXHAUSTED: passive = none; enabled = must be CUDA_ERROR_OUT_OF_MEMORY
CONF="/var/run/softmig/${SLURM_JOB_ID:-}.conf"
if [ -n "${SLURM_JOB_ID:-}" ] && [ -r "$CONF" ]; then
    mode=enabled
else
    mode=passive
fi
echo "  mode: $mode"

if [ "$mode" = "passive" ]; then
    if grep -l "RESOURCE_EXHAUSTED" "$OUT"/*.stderr >/dev/null 2>&1; then
        echo "  FAIL: 'RESOURCE_EXHAUSTED' found in: $(grep -l 'RESOURCE_EXHAUSTED' "$OUT"/*.stderr | tr '\n' ' ')"
        fail=1
    else
        echo "  PASS: zero 'RESOURCE_EXHAUSTED' in stderr (passive)"
    fi
else
    if grep -h "RESOURCE_EXHAUSTED" "$OUT"/*.stderr >/dev/null 2>&1; then
        if grep -h "RESOURCE_EXHAUSTED" "$OUT"/*.stderr | grep -v "CUDA_ERROR_OUT_OF_MEMORY" | grep -q .; then
            echo "  FAIL: RESOURCE_EXHAUSTED without CUDA_ERROR_OUT_OF_MEMORY"
            fail=1
        else
            echo "  PASS: RESOURCE_EXHAUSTED paired with CUDA_ERROR_OUT_OF_MEMORY (limit enforced)"
        fi
    else
        echo "  FAIL: enabled oversize alloc did not produce RESOURCE_EXHAUSTED"
        fail=1
    fi
fi

# 4. device memory returned to baseline (+/- 512 MiB)
if [ -n "${START_USED:-}" ] && [ -n "${END_USED:-}" ]; then
    delta=$(( END_USED > START_USED ? END_USED - START_USED : START_USED - END_USED ))
    echo "  used memory delta: ${delta} MiB"
    if [ "$delta" -le 512 ]; then
        echo "  PASS: used memory back to baseline within 512 MiB"
    else
        echo "  FAIL: used memory did not return to baseline (${delta} MiB delta)"
        fail=1
    fi
else
    echo "  WARN: could not read nvidia-smi used memory"
fi

# capture softmig log for diagnosis
if [ -n "${SLURM_JOB_ID:-}" ] && [ -r "/var/log/softmig/${SLURM_JOB_ID}.log" ]; then
    cp "/var/log/softmig/${SLURM_JOB_ID}.log" "$OUT/softmig.log" 2>/dev/null || true
    echo
    echo "softmig.log copied to $OUT/softmig.log"
fi
echo "results in $OUT"
exit "$fail"
