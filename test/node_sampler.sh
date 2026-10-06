#!/bin/bash
# Node-side ground-truth sampler for SoftMig share/leak tests.
#
# Runs as root on the reservation node over `sudo ssh` (the only node we may
# touch). Root has no SoftMig config, so its nvidia-smi is unfiltered: this is
# the truth that the in-job (hooked) views are compared against.
#
#   test/node_sampler.sh start OUTDIR [NODE]   # background loops, returns at once
#   test/node_sampler.sh stop  OUTDIR [NODE]
#
# OUTDIR must be on a filesystem the node's root can write (project space is).
# Files written by the node, 1 Hz:
#   pmon.log   nvidia-smi pmon -s um -o DT   per-PID sm% / mem% / fb MiB, all GPUs
#   apps.log   <epoch> <gpu_uuid>,<pid>,<used_MiB>        (query-compute-apps)
#   gpu.log    <epoch> <idx>,<uuid>,<util%>,<used_MiB>     (query-gpu)
#   pidmap.txt <pid> <uid> <jobid|none> <step|none> <comm>  (from /proc, first sight)
#   sampler.pids, sampler.started, sampler.stopped

set -u
cmd="${1:?start|stop}"; out="${2:?OUTDIR}"; node="${3:-rack01-11}"
mkdir -p "$out"; out="$(cd "$out" && pwd -P)"   # the node runs this from /root: must be absolute

remote_start='
out="$1"
mkdir -p "$out"
cd "$out" || exit 1
rm -f sampler.pids sampler.stopped
date +%s > sampler.started
setsid nohup nvidia-smi pmon -s um -d 1 -o DT > pmon.log 2> pmon.err < /dev/null &
echo $! > sampler.pids
setsid nohup bash -c '"'"'
declare -A seen
while :; do
  t=$(date +%s)
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | sed "s/^/$t /" >> apps.log
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | sed "s/^/$t /" >> gpu.log
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    # re-record a PID not seen for 30 s (PID reuse during long shared runs)
    if [ -n "${seen[$pid]:-}" ] && [ $((t - seen[$pid])) -lt 30 ]; then seen[$pid]=$t; continue; fi
    seen[$pid]=$t
    uid=$(awk "/^Uid:/{print \$2}" /proc/$pid/status 2>/dev/null)
    cg=$(cat /proc/$pid/cgroup 2>/dev/null)
    job=$(echo "$cg" | grep -oE "job_[0-9]+" | head -1 | sed "s/job_//")
    step=$(echo "$cg" | grep -oE "step_[^/]+" | head -1 | sed "s/step_//")
    comm=$(cat /proc/$pid/comm 2>/dev/null)
    echo "$pid ${uid:-?} ${job:-none} ${step:-none} ${comm:-?} $t" >> pidmap.txt
  done
  sleep 1
done'"'"' > loop.err 2>&1 < /dev/null &
echo $! >> sampler.pids
echo started
'

remote_stop='
out="$1"
cd "$out" || exit 1
if [ -f sampler.pids ]; then
  for p in $(cat sampler.pids); do
    pkill -TERM -P "$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null
  done
  sleep 1
  for p in $(cat sampler.pids); do kill -KILL "$p" 2>/dev/null; done
fi
date +%s > sampler.stopped
echo stopped lines: pmon=$(wc -l < pmon.log 2>/dev/null) apps=$(wc -l < apps.log 2>/dev/null) pids=$(wc -l < pidmap.txt 2>/dev/null)
'

case "$cmd" in
    start)
        mkdir -p "$out"
        timeout 60 sudo -n ssh -o BatchMode=yes "$node" "bash -s -- '$out'" <<< "$remote_start"
        ;;
    stop)
        timeout 60 sudo -n ssh -o BatchMode=yes "$node" "bash -s -- '$out'" <<< "$remote_stop"
        ;;
    *) echo "usage: $0 start|stop OUTDIR [NODE]" >&2; exit 2 ;;
esac
