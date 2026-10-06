#!/bin/bash
# Suite: container — does SoftMig apply inside Apptainer?  (informational)
# In a ${SLICE} slice, with `apptainer exec --nv`:
#   default: is libsoftmig mapped into a process inside the container? (the
#            container has its own /etc/ld.so.preload, so normally not), and
#            what does cuMemGetInfo-equivalent nvidia-smi report?
#   preload: same, with the library, /var/run/softmig and /var/log/softmig
#            bound in and APPTAINERENV_LD_PRELOAD set (what a site would do
#            to enforce slices in containers).
# Image: $SCRATCH/softmig-cuda.sif (pulled once from docker://nvidia/cuda).
#
# Emits INFO with what was observed; SKIP if apptainer or the image is
# unavailable. Observed on rack01-11 (2026-10): not loaded by default; when
# preloaded, the config is rejected because the user namespace shows the
# root-owned file as nobody (65534), so the library stays passive.

SUITE=container
DEFAULT_SRUN_TIME=00:20:00
. "$(dirname "$0")/suite_common.sh"

SIF="${CONTAINER_SIF:-${SCRATCH:-$HOME/scratch}/softmig-cuda.sif}"
IMG="${CONTAINER_IMAGE:-docker://nvidia/cuda:12.6.3-base-ubuntu22.04}"

_srun_capture srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=2 --mem=8G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load apptainer 2>/dev/null
cd ${SOFTMIG_ROOT}
echo \$SLURM_JOB_ID > '${OUT}/jid.txt'
_hang_watchdog 1080 '${OUT}'
command -v apptainer > '${OUT}/apptainer.txt' 2>&1 || { echo none > '${OUT}/apptainer.txt'; exit 0; }
apptainer --version >> '${OUT}/apptainer.txt' 2>&1
[ -s '${SIF}' ] || apptainer pull '${SIF}' '${IMG}' > '${OUT}/pull.log' 2>&1
[ -s '${SIF}' ] || exit 0
lib=\$(sed 's/#.*//' /etc/ld.so.preload | grep -v '^\s*\$' | head -1)
echo \"host_mem=\$(nvidia-smi --query-gpu=memory.total --format=csv,noheader)\" > '${OUT}/host.txt'
apptainer exec --nv '${SIF}' sh -c 'echo maps=\$(grep -c softmig /proc/self/maps); echo preload=\$(cat /etc/ld.so.preload 2>/dev/null | head -1); nvidia-smi --query-gpu=memory.total --format=csv,noheader' > '${OUT}/default.txt' 2>&1
APPTAINERENV_SOFTMIG_LOG_LEVEL=3 APPTAINERENV_LD_PRELOAD=\$lib apptainer exec --nv -B \$lib -B /var/run/softmig -B /var/log/softmig '${SIF}' \
    sh -c 'echo maps=\$(grep -c softmig /proc/self/maps); nvidia-smi --query-gpu=memory.total --format=csv,noheader' > '${OUT}/preload.txt' 2>&1
_hang_disarm '${OUT}'
" >/dev/null 2>&1

jid=$(cat "$OUT/jid.txt" 2>/dev/null || echo NA)
if grep -q none "$OUT/apptainer.txt" 2>/dev/null || [ ! -f "$OUT/default.txt" ]; then
    _emit "$jid" SKIP 0 "apptainer=$(head -1 "$OUT/apptainer.txt" 2>/dev/null) image=$([ -s "$SIF" ] && echo ok || echo "unavailable: $(tail -1 "$OUT/pull.log" 2>/dev/null | cut -c1-100)")"
    exit 0
fi
d_maps=$(sed -n 's/^maps=//p' "$OUT/default.txt"); d_mem=$(tail -1 "$OUT/default.txt")
p_maps=$(sed -n 's/^maps=//p' "$OUT/preload.txt"); p_mem=$(tail -1 "$OUT/preload.txt")
p_why=$(grep -m1 -oE "Config file [^:]*" "$OUT/preload.txt")
host=$(sed -n 's/^host_mem=//p' "$OUT/host.txt")
_emit "$jid" INFO "${d_maps:-?}/${p_maps:-?}" \
    "host_nvsmi_total=[${host}] default:softmig_mapped=${d_maps:-?} total=[${d_mem}] | preload_bound:softmig_mapped=${p_maps:-?} total=[${p_mem}] reason=[${p_why:-none}]"
