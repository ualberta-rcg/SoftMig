#!/bin/bash
# Inner half of the pool suite - runs INSIDE the srun allocation on the node
# (invoked by suite_pool.sh via srun; do not run standalone).
# Env expected (propagated by srun): SLICE, CUDA_VER, SOFTMIG_ROOT, OUT.
set -u

module load "cuda/${CUDA_VER}"
cd "$SOFTMIG_ROOT"
export SOFTMIG_LOG_LEVEL=5

CONF="/var/run/softmig/${SLURM_JOB_ID}.conf"
mode_args=""
if [ "${SLICE}" = "l40s" ]; then
    mode=passive
    mode_args="--passive"
else
    mode=enabled
    limit=$(awk -F= '/^CUDA_DEVICE_MEMORY_LIMIT=/{print $2}' "$CONF" 2>/dev/null)
    bytes=$(awk -v v="$limit" 'BEGIN{
        if (v == "") { print 0; exit }
        s = 1
        u = toupper(substr(v, length(v), 1))
        if (u == "K") s = 1024
        else if (u == "M") s = 1048576
        else if (u == "G") s = 1073741824
        print int(v * s)}')
    mode_args="--limit-bytes ${bytes}"
fi

{
    echo "mode=${mode}"
    echo "slice=${SLICE} cuda=${CUDA_VER} args=${mode_args}"
    echo "conf=$(tr '\n' ';' < "$CONF" 2>/dev/null || echo none)"
} > "$OUT/mode.txt"

build/test/test_pool_free $mode_args > "$OUT/test_pool_free.log" 2>&1
echo $? > "$OUT/rc.txt"
cp "$CONF" "$OUT/job.conf" 2>/dev/null || true
if [ -r "/var/log/softmig/${SLURM_JOB_ID}.log" ]; then
    cp "/var/log/softmig/${SLURM_JOB_ID}.log" "$OUT/softmig.log" 2>/dev/null || true
fi
echo "$SLURM_JOB_ID" > "$OUT/jid.txt"
