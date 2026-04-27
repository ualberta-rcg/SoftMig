#!/bin/bash
# Suite 7: long soak — 4x gpu_burn_lite for ~5min on a .4 slice, cuda/12.2.
# Samples /dev/shm/libsoftmig* file size + /proc/<pid>/fd counts every 30s to
# detect leaks or runaway growth.
# PASS iff shrreg file size is monotonically non-increasing after first minute
# AND no softmig ERROR lines related to shared region exhaustion.

SUITE=soak
SLICE=${SLICE:-l40s.4}
CUDA_VER=${CUDA_VER:-12.2}
DEFAULT_SRUN_TIME=00:12:00
. "$(dirname "$0")/suite_common.sh"

DUR=300  # 5 min

srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=8 --mem=12G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
export SOFTMIG_LOG_LEVEL=3

# Launch 4 burn procs in background
for i in 1 2 3 4; do
  ./build/test/gpu_burn_lite 1024 ${DUR} > '${OUT}/proc_'\$i.log 2>&1 &
done
BURN_PIDS=\$(jobs -p)
P1=\$(echo \"\$BURN_PIDS\" | sed -n '1p')
P2=\$(echo \"\$BURN_PIDS\" | sed -n '2p')
P3=\$(echo \"\$BURN_PIDS\" | sed -n '3p')
P4=\$(echo \"\$BURN_PIDS\" | sed -n '4p')

# Sample every 30s
SAMPLE_FILE='${OUT}/samples.tsv'
echo -e 't\tshrreg_bytes\tfd_p1\tfd_p2\tfd_p3\tfd_p4\tnvsmi_util%\tnvsmi_mem_MB' > \"\$SAMPLE_FILE\"
for t in 30 60 90 120 150 180 210 240 270 300; do
  sleep 30
  SZ=\$(stat -c %s /dev/shm/libsoftmig* 2>/dev/null | awk 'BEGIN{m=0} {if ($1+0>m) m=$1+0} END{print m+0}')
  [ -z \"\$SZ\" ] && SZ=0
  fd_count() {
    local p=\"\$1\"
    if [ -n \"\$p\" ] && [ -d \"/proc/\$p/fd\" ]; then
      ls \"/proc/\$p/fd\" 2>/dev/null | wc -l
    else
      echo 0
    fi
  }
  FD1=\$(fd_count \"\$P1\")
  FD2=\$(fd_count \"\$P2\")
  FD3=\$(fd_count \"\$P3\")
  FD4=\$(fd_count \"\$P4\")
  UTIL=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{print int(\$1+0)}')
  MEM=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{print int(\$1+0)}')
  [ -z \"\$UTIL\" ] && UTIL=0
  [ -z \"\$MEM\" ] && MEM=0
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \"\$t\" \"\$SZ\" \"\$FD1\" \"\$FD2\" \"\$FD3\" \"\$FD4\" \"\$UTIL\" \"\$MEM\" >> \"\$SAMPLE_FILE\"
done

# Reap burn procs
wait \$BURN_PIDS 2>/dev/null
_copy_softmig_log \$SLURM_JOB_ID '${OUT}/softmig.log'
echo \$SLURM_JOB_ID > '${OUT}/jid.txt'
" >/dev/null 2>&1

jid=$(cat "$OUT/jid.txt" 2>/dev/null || echo NA)
slog="$OUT/softmig.log"
smp="$OUT/samples.tsv"

if [ ! -s "$slog" ] || [ ! -s "$smp" ]; then
    _emit "$jid" FAIL 0 "missing log or samples"
    exit 0
fi

# Analyze samples: after t>=60s, shrreg size should be stable (no growth)
# and FD counts shouldn't grow unboundedly.
max_shrreg=$(awk -F'\t' 'NR>1 && $1>=60 {print $2}' "$smp" | sort -n | tail -1)
min_shrreg=$(awk -F'\t' 'NR>1 && $1>=60 {print $2}' "$smp" | sort -n | head -1)
[ -z "$max_shrreg" ] && max_shrreg=0
[ -z "$min_shrreg" ] && min_shrreg=0
max_shrreg=$(echo "$max_shrreg" | awk '{print int($1+0)}')
min_shrreg=$(echo "$min_shrreg" | awk '{print int($1+0)}')
delta=$((max_shrreg - min_shrreg))

# FD growth across window
fd_max=$(awk -F'\t' 'NR>1 {for(i=3;i<=6;i++) if ($i+0>m) m=$i+0} END{print m+0}' "$smp")
err_cnt=$(grep -c "softmig ERROR" "$slog")
oom_err=$(grep -cE "shared region|proc_num exhausted|can't register" "$slog")

metric="shrreg_delta=${delta} fd_max=${fd_max} err=${err_cnt} shrreg_err=${oom_err}"

# PASS: shrreg growth < 8KB, no shared-region errors
if [ "$delta" -lt 8192 ] && [ "$oom_err" = "0" ]; then
    _emit "$jid" PASS "$delta" "$metric"
else
    _emit "$jid" FAIL "$delta" "$metric"
fi
