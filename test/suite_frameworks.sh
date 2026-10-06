#!/bin/bash
# Suite: frameworks — real framework allocators against the limit.
# Runs test/python/fw_limit.py for PyTorch (native caching allocator and
# PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync), TensorFlow, and a
# DataLoader-fork + spawn-pool check, first in a slice (enabled: OOM must be
# the framework's normal OOM error, within the limit) and then on a full GPU
# (passive: 16 GiB must just work). SLICE is the slice to use (default
# l40s.4); the full-GPU half always uses l40s.
#
# Venv: $SCRATCH/softmig-fwvenv (torch + tensorflow from the wheelhouse),
# created once under flock so parallel suites do not race.
#
# PASS iff every case prints "fw_limit: PASS".

SUITE=frameworks
DEFAULT_SRUN_TIME=00:30:00
. "$(dirname "$0")/suite_common.sh"

VENV="${FW_VENV:-${SCRATCH:-$HOME/scratch}/softmig-fwvenv}"

run_half() {   # $1 = gres type, $2 = mode, $3 = cases
    local gres="$1" mode="$2" cases="$3" sub="$OUT/$2"
    mkdir -p "$sub"
    _srun_capture srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${gres}:1 --cpus-per-task=8 --mem=32G \
         --time="$DEFAULT_SRUN_TIME" bash -lc "
module load StdEnv/2023 python/3.11 cuda/${CUDA_VER} cudnn 2>/dev/null
cd ${SOFTMIG_ROOT}
echo \$SLURM_JOB_ID > '${sub}/jid.txt'
_hang_watchdog 1500 '${sub}'
(
  flock 9
  if [ ! -x '${VENV}/bin/python' ]; then
    python -m venv '${VENV}' && '${VENV}/bin/pip' install --no-index torch tensorflow > '${sub}/venv_install.log' 2>&1
  fi
) 9> '${VENV}.lock'
'${VENV}/bin/python' -c 'import torch, tensorflow as tf; print(\"torch\", torch.__version__, \"tf\", tf.__version__)' > '${sub}/versions.txt' 2>&1
lim=\$(sed -n 's/^CUDA_DEVICE_MEMORY_LIMIT=\\([0-9]*\\).*/\\1/p' /var/run/softmig/\${SLURM_JOB_ID}*.conf 2>/dev/null | head -1)
export SOFTMIG_LIMIT_MIB=\${lim:-0}
for c in ${cases}; do
  case \$c in
    torch_async) PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync '${VENV}/bin/python' test/python/fw_limit.py torch_async ${mode} ;;
    *) '${VENV}/bin/python' test/python/fw_limit.py \$c ${mode} ;;
  esac > '${sub}'/\$c.log 2>&1
done
_hang_disarm '${sub}'
_copy_softmig_log \$SLURM_JOB_ID '${sub}/softmig.log'
" >/dev/null 2>&1
}

run_half "$SLICE" enabled "torch torch_async tf torch_fork"
run_half l40s passive "torch torch_async tf"

jids=$(cat "$OUT"/enabled/jid.txt "$OUT"/passive/jid.txt 2>/dev/null | paste -sd,)
[ -f "$OUT/enabled/HANG" ] && cp "$OUT/enabled/HANG" "$OUT/HANG"
[ -f "$OUT/passive/HANG" ] && cp "$OUT/passive/HANG" "$OUT/HANG"
pass=0; total=0; detail=""
for f in "$OUT"/enabled/*.log "$OUT"/passive/*.log; do
    case "$(basename "$f")" in softmig.log|venv_install.log) continue ;; esac
    [ -f "$f" ] || continue
    total=$((total + 1))
    mode=$(basename "$(dirname "$f")")
    line=$(grep -m1 "^fw_limit:" "$f")
    if printf '%s' "$line" | grep -q "fw_limit: PASS"; then
        pass=$((pass + 1)); detail="$detail ${mode}/$(basename "$f" .log)=PASS"
    else
        detail="$detail ${mode}/$(basename "$f" .log)=FAIL(${line:-$(tail -1 "$f" | cut -c1-80)})"
    fi
done
detail="$(head -1 "$OUT/enabled/versions.txt" 2>/dev/null) |${detail}"
if [ "$total" -eq 7 ] && [ "$pass" -eq "$total" ]; then
    _emit "$jids" PASS "$pass/$total" "$detail"
else
    _emit "$jids" FAIL "$pass/$total" "$detail"
fi
