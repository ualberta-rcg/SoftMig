#!/bin/bash
# Suite: multigpu — per-device reporting with two GPUs.
#   passive: --gres=gpu:l40s:2, multigpu_probe --expect passive (values must
#            equal the driver's on both devices).
#   slices:  --gres=gpu:${SLICE}:2 if the site's job_submit accepts two
#            slices; multigpu_probe --expect enabled (per-device caps, and an
#            allocation on device 1 must only reduce device 1's free).
#            If the request is rejected that half is reported as "rejected"
#            (site policy, not a failure).
#
# Also requests mixed sizes (l40s.2 + l40s.4) and reports whether the
# prolog's limit matches the shards Slurm allocated (site policy; reported
# in the detail, does not affect PASS).
#
# PASS iff the passive half passes and the slice half passes or is rejected.

SUITE=multigpu
. "$(dirname "$0")/suite_common.sh"

run_half() {   # $1 = gres, $2 = mode
    local sub="$OUT/$2"; mkdir -p "$sub"
    _srun_capture srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:$1:2 --cpus-per-task=4 --mem=8G \
         --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
echo \$SLURM_JOB_ID > '${sub}/jid.txt'
_hang_watchdog 240 '${sub}'
echo \"CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\" > '${sub}/env.txt'
cat /var/run/softmig/\$SLURM_JOB_ID*.conf >> '${sub}/env.txt' 2>/dev/null
./build/test/multigpu_probe --expect $2 1024 > '${sub}/probe.log' 2>&1
_hang_disarm '${sub}'
_copy_softmig_log \$SLURM_JOB_ID '${sub}/softmig.log'
" > "$sub/srun.out" 2>&1
    [ -f "$sub/HANG" ] && cp "$sub/HANG" "$OUT/HANG"
}

run_half l40s passive
run_half "$SLICE" enabled

# Mixed slice sizes in one request: does the memory limit the prolog writes
# match the shards Slurm actually allocated? (site policy check, reported)
mkdir -p "$OUT/mixed"
srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:l40s.2:1,gpu:l40s.4:1 --cpus-per-task=1 --mem=1G \
     --time=00:02:00 bash -c '
scontrol show job $SLURM_JOB_ID | grep -oE "AllocTRES=[^ ]*" 
cat /var/run/softmig/$SLURM_JOB_ID*.conf 2>/dev/null
nvidia-smi --query-gpu=memory.total --format=csv,noheader' > "$OUT/mixed/out.txt" 2>&1
m_alloc=$(grep -oE "gres/shard=[0-9]+" "$OUT/mixed/out.txt" | head -1 | cut -d= -f2)
m_lim=$(sed -n 's/^CUDA_DEVICE_MEMORY_LIMIT=\([0-9]*\).*/\1/p' "$OUT/mixed/out.txt")
m_tot=$(tail -1 "$OUT/mixed/out.txt" | grep -oE "^[0-9]+")
mixed="alloc_shards=${m_alloc:-?} limit_mib=${m_lim:-none}"
if [ -n "$m_alloc" ] && [ -n "$m_lim" ] && [ -n "$m_tot" ]; then
    # 4 shards per GPU; nvidia-smi total in a slice is the limit, so use the
    # unsliced size from the passive half when available.
    full=$(grep -m1 -oE "driver [0-9]+ MiB" "$OUT/passive/probe.log" | grep -oE "[0-9]+")
    if [ -n "$full" ] && [ $(( m_lim * 4 / full )) -ne "$m_alloc" ]; then
        mixed="$mixed MISMATCH(limit=$(( m_lim * 4 / full ))/4 GPU but $m_alloc shard(s) allocated)"
    fi
fi

jids=$(cat "$OUT"/passive/jid.txt "$OUT"/enabled/jid.txt 2>/dev/null | paste -sd,)
rp=$(grep -m1 "^RESULT" "$OUT/passive/probe.log" 2>/dev/null)
if [ -f "$OUT/enabled/jid.txt" ]; then
    re=$(grep -m1 "^RESULT" "$OUT/enabled/probe.log" 2>/dev/null)
else
    re="rejected: $(sed 's/\x1b\[[0-9;]*m//g' "$OUT/enabled/srun.out" | grep -m1 -oE "ERROR: .*" | cut -c8-110)"
fi
fails=$(grep -h "^FAIL" "$OUT"/*/probe.log 2>/dev/null | paste -sd';')
metric="passive=[${rp:-none}] slices=[${re:-none}] mixed=[${mixed}] ${fails}"
if [[ "$rp" == "RESULT: PASS"* ]] && { [[ "$re" == "RESULT: PASS"* ]] || [[ "$re" == rejected* ]]; }; then
    _emit "$jids" PASS 1 "$metric"
else
    _emit "$jids" FAIL 0 "$metric"
fi
