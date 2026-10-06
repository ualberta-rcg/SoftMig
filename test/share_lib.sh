#!/bin/bash
# Helpers for multi-job scenarios (share / isolation / mixed_pieces / gpuburn /
# passthrough suites). Sourced after suite_common.sh.
#
#   pack_blocker N          hold N full GPUs so shard jobs must share the rest
#   run_job NAME GRES VER TIME 'body'   srun one job into $OUT/jobs/NAME (bg)
#   wait_jobs               wait for every run_job started so far
#   sampler_start / sampler_stop        root-side truth into $OUT/sampler
#   job_meta NAME KEY       read jid/uuid/conf values back
#
# Every job directory gets: jid.txt node.txt uuid.txt gres.txt cuda.txt conf.txt
# smi_apps.txt (in-job hooked nvidia-smi view, 1 Hz) softmig.log run.out rc.txt
# and whatever the body writes. $D inside the body is the job directory.

set -u
: "${OUT:?}"; : "${SOFTMIG_ROOT:?}"
mkdir -p "$OUT/jobs" "$OUT/sampler"
OUT="$(cd "$OUT" && pwd -P)"   # job bodies and the node-side sampler need an absolute path

NODE="${SOFTMIG_NODE:-rack01-11}"
JOB_PIDS=()
BLOCKER_SRUN_PID=""; BLOCKER_JOBID=""

_share_cleanup() {
    [ -n "$BLOCKER_JOBID" ] && scancel "$BLOCKER_JOBID" >/dev/null 2>&1
    [ "${#ARRAY_JOBS[@]}" -gt 0 ] && scancel "${ARRAY_JOBS[@]}" >/dev/null 2>&1
    [ -n "$BLOCKER_SRUN_PID" ] && { kill "$BLOCKER_SRUN_PID" 2>/dev/null; wait "$BLOCKER_SRUN_PID" 2>/dev/null; }
    [ -f "$OUT/sampler/sampler.started" ] && [ ! -f "$OUT/sampler/sampler.stopped" ] && sampler_stop
    true
}
trap _share_cleanup EXIT

pack_blocker() {
    local n="${1:-3}" secs="${2:-600}"
    mkdir -p "$OUT/blocker"; rm -f "$OUT/blocker/jid.txt"
    srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:l40s:${n} --cpus-per-task=2 --mem=2G \
         --time=00:20:00 bash -lc "echo \$SLURM_JOB_ID > '$OUT/blocker/jid.txt'; sleep $secs" >/dev/null 2>&1 &
    BLOCKER_SRUN_PID=$!
    for _ in $(seq 1 120); do [ -s "$OUT/blocker/jid.txt" ] && break; sleep 1; done
    BLOCKER_JOBID=$(tr -d '[:space:]' < "$OUT/blocker/jid.txt" 2>/dev/null)
    [ -n "$BLOCKER_JOBID" ]
}

# SAMPLER_SHARED=/dir : a long-running node sampler already writes there (started
# by the login-side launcher; sudo is not available inside jobs). The suite then
# just cuts its own time window out of the shared logs instead of starting one.
sampler_start() {
    if [ -n "${SAMPLER_SHARED:-}" ]; then date +%s > "$OUT/sampler/t0"; return 0; fi
    bash "$SOFTMIG_ROOT/test/node_sampler.sh" start "$OUT/sampler" "$NODE" >/dev/null
}
sampler_stop() {
    if [ -n "${SAMPLER_SHARED:-}" ]; then _sampler_extract; return 0; fi
    bash "$SOFTMIG_ROOT/test/node_sampler.sh" stop  "$OUT/sampler" "$NODE" > "$OUT/sampler/stop.txt" 2>&1
}
_sampler_extract() {
    local a b S="$SAMPLER_SHARED" d="$OUT/sampler"
    a=$(( $(cat "$d/t0" 2>/dev/null || date +%s) - 5 )); b=$(( $(date +%s) + 5 ))
    awk -v a="$a" -v b="$b" '$1>=a && $1<=b' "$S/apps.log" > "$d/apps.log" 2>/dev/null
    awk -v a="$a" -v b="$b" '$1>=a && $1<=b' "$S/gpu.log"  > "$d/gpu.log"  2>/dev/null
    awk -v b="$b" '$6<=b' "$S/pidmap.txt" > "$d/pidmap.txt" 2>/dev/null   # earlier sightings too; the report keeps the latest
    TZ=UTC awk -v a="$a" -v b="$b" '/^#/ {print; next}
        { ts = mktime(substr($1,1,4) " " substr($1,5,2) " " substr($1,7,2) " " substr($2,1,2) " " substr($2,4,2) " " substr($2,7,2));
          if (ts >= a && ts <= b) print }' "$S/pmon.log" > "$d/pmon.log" 2>/dev/null
    echo "extracted $a..$b from $S: pmon=$(wc -l < "$d/pmon.log") apps=$(wc -l < "$d/apps.log") pids=$(wc -l < "$d/pidmap.txt")" > "$d/stop.txt"
}

# Body shared by run_job (srun) and run_array (sbatch). $D is the job dir:
# for arrays it is $OUT/jobs/<name>_<task>. Conf path follows the prolog
# (/var/run/softmig/<jobid>[_<arraytask>].conf). The body runs in a subshell
# so a bare `wait` inside it does not wait for the nvidia-smi poll loop.
_job_script() {
    local D="$1" gres="$2" ver="$3" body="$4" loglvl="${SOFTMIG_JOB_LOG_LEVEL:-2}"
    cat <<EOS
module load cuda/${ver} >/dev/null 2>&1
cd '${SOFTMIG_ROOT}'
export SOFTMIG_LOG_LEVEL=${loglvl}
${JOB_ENV:+export ${JOB_ENV}}
D="${D}"; mkdir -p "\$D"
B=build-cuda${ver}/test; [ -x \$B/gpu_burn_lite ] || B=build/test; echo \$B > \$D/bindir.txt
if [ '${SMI_WRAPPER:-0}' = 1 ]; then mkdir -p \$D/wbin && ln -sf '${SOFTMIG_ROOT}/nvidia-smi-hook.sh' \$D/wbin/nvidia-smi && export PATH=\$D/wbin:\$PATH; fi
echo \$SLURM_JOB_ID > \$D/jid.txt; hostname > \$D/node.txt
echo '${gres}' > \$D/gres.txt; echo '${ver}' > \$D/cuda.txt
nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1 > \$D/uuid.txt
cat /var/run/softmig/\${SLURM_JOB_ID}\${SLURM_ARRAY_TASK_ID:+_\$SLURM_ARRAY_TASK_ID}.conf > \$D/conf.txt 2>/dev/null || : > \$D/conf.txt
ls \${SLURM_TMPDIR:-/tmp}/cudevshr.cache* > \$D/region_before.txt 2>/dev/null
( while [ ! -f \$D/done ]; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | sed "s/^/\$(date +%s) /" >> \$D/smi_apps.txt
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | sed "s/^/\$(date +%s) /" >> \$D/smi_gpu.txt
    sleep 1; done ) &
date +%s > \$D/t_start.txt
(
${body}
) > \$D/run.out 2>&1
echo \$? > \$D/rc.txt
date +%s > \$D/t_end.txt
ls -la \${SLURM_TMPDIR:-/tmp}/cudevshr.cache* > \$D/region_after.txt 2>/dev/null
touch \$D/done
sleep 1
_copy_softmig_log \$SLURM_JOB_ID \$D/softmig.log
EOS
}

# run_job NAME GRES CUDA_VER TIME BODY [extra srun args...]
# env knobs: JOB_CPUS JOB_MEM EXPECT SMI_WRAPPER SOFTMIG_JOB_LOG_LEVEL JOB_ENV ("A=1 B=2" exported in the job)
run_job() {
    local name="$1" gres="$2" ver="$3" tlim="$4" body="$5"; shift 5
    local D="$OUT/jobs/$name"; mkdir -p "$D"; rm -f "$D/done" "$D/jid.txt"
    # EXPECT="limit=23034 sm=50 min_mem=... oom=1 ..." -> checked by share_report.py
    [ -n "${EXPECT:-}" ] && printf '%s\n' $EXPECT > "$D/expect.txt"
    srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${gres} --cpus-per-task="${JOB_CPUS:-4}" \
         --mem="${JOB_MEM:-8G}" --time="$tlim" "$@" bash -lc "$(_job_script "$D" "$gres" "$ver" "$body")" \
         > "$D/srun.log" 2>&1 &
    JOB_PIDS+=($!)
}

# run_array NAME COUNT GRES CUDA_VER TIME BODY  -> sbatch --array=0-(COUNT-1);
# job dirs $OUT/jobs/NAME_<task>. Waited for by wait_jobs.
ARRAY_JOBS=()
run_array() {
    local name="$1" count="$2" gres="$3" ver="$4" tlim="$5" body="$6"; shift 6
    local t D script="$OUT/${name}.sbatch"; mkdir -p "$OUT"
    for t in $(seq 0 $((count - 1))); do
        D="$OUT/jobs/${name}_$t"; mkdir -p "$D"; rm -f "$D/done" "$D/jid.txt"
        [ -n "${EXPECT:-}" ] && printf '%s\n' $EXPECT > "$D/expect.txt"
    done
    {
        echo '#!/bin/bash -l'
        declare -f _copy_softmig_log
        _job_script "$OUT/jobs/${name}_\${SLURM_ARRAY_TASK_ID}" "$gres" "$ver" "$body"
    } > "$script"
    local jid
    jid=$(sbatch --parsable --reservation=softmig ${SRUN_EXTRA:-} --array=0-$((count - 1)) --gres=gpu:${gres} \
          --cpus-per-task="${JOB_CPUS:-4}" --mem="${JOB_MEM:-8G}" --time="$tlim" \
          --output="$OUT/jobs/${name}_%a/sbatch.log" "$@" "$script" 2> "$OUT/${name}.sbatch.err")
    jid="${jid%%;*}"
    [ -n "$jid" ] && ARRAY_JOBS+=("$jid")
    echo "$jid" > "$OUT/${name}.arrayjid"
}

wait_jobs() {
    local p a
    for p in "${JOB_PIDS[@]:-}"; do [ -n "$p" ] && wait "$p" 2>/dev/null; done
    JOB_PIDS=()
    for a in "${ARRAY_JOBS[@]:-}"; do
        [ -n "$a" ] || continue
        while [ -n "$(squeue -h -j "$a" 2>/dev/null)" ]; do sleep 5; done
    done
    ARRAY_JOBS=()
}

job_meta() {   # job_meta NAME jid|uuid|mem_limit|sm_limit|rc
    local D="$OUT/jobs/$1"
    case "$2" in
        jid)  tr -d '[:space:]' < "$D/jid.txt" 2>/dev/null ;;
        uuid) tr -d '[:space:]' < "$D/uuid.txt" 2>/dev/null ;;
        rc)   tr -d '[:space:]' < "$D/rc.txt" 2>/dev/null ;;
        mem_limit) sed -n 's/^CUDA_DEVICE_MEMORY_LIMIT=\([0-9.]*\)M.*/\1/p' "$D/conf.txt" 2>/dev/null | cut -d. -f1 ;;
        sm_limit)  sed -n 's/^CUDA_DEVICE_SM_LIMIT=\([0-9]*\).*/\1/p' "$D/conf.txt" 2>/dev/null ;;
    esac
}

# Wait until all named jobs have written jid.txt (they have started).
wait_started() {
    local n
    for n in "$@"; do
        for _ in $(seq 1 180); do [ -s "$OUT/jobs/$n/jid.txt" ] && break; sleep 1; done
    done
}
