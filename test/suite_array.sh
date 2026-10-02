#!/bin/bash
# Suite: array — a 2-task array job on ${SLICE} slices.
# Each task records which config file the prolog wrote, runs
# passive_probe --expect enabled, and copies its job log
# (/var/log/softmig/<jobid>_<task>.log or <jobid>.log).
#
# PASS iff both tasks pass the probe and each task's log shows the limit was
# read from a config file.

SUITE=array
. "$(dirname "$0")/suite_common.sh"

rm -f "$OUT"/task_*
cat > "$OUT/task.sh" <<EOF
#!/bin/bash -l
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
d="${OUT}/task_\${SLURM_ARRAY_TASK_ID}"
mkdir -p "\$d"
echo "\$SLURM_JOB_ID \$SLURM_ARRAY_JOB_ID \$SLURM_ARRAY_TASK_ID" > "\$d/ids.txt"
# Prolog and library both name per-task files <SLURM_JOB_ID>_<task> (the
# task's own job id, which differs from SLURM_ARRAY_JOB_ID except for one task).
ls /var/run/softmig/ | grep -E "^\${SLURM_JOB_ID}(_\${SLURM_ARRAY_TASK_ID})?\.conf\$" > "\$d/conf.txt" 2>&1
SOFTMIG_LOG_LEVEL=2 ./build/test/passive_probe --expect enabled > "\$d/probe.log" 2>&1
for f in /var/log/softmig/\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.log /var/log/softmig/\${SLURM_JOB_ID}.log; do
  [ -r "\$f" ] && cp "\$f" "\$d/softmig.log" && break
done
EOF
chmod +x "$OUT/task.sh"

jid=$(sbatch --parsable --wait --reservation=softmig ${SRUN_EXTRA:-} --array=0-1 --gres=gpu:${SLICE}:1 \
      --cpus-per-task=2 --mem=4G --time=00:05:00 -o "$OUT/slurm-%A_%a.out" "$OUT/task.sh" 2>"$OUT/sbatch.err")
# --wait can return when the array's own task ends while other tasks (their
# own job ids) are still finishing; wait until no task is left.
for _ in $(seq 1 120); do
    [ -z "$(squeue -h -r -j "${jid:-0}" 2>/dev/null)" ] && break
    sleep 2
done
sleep 2

pass=0; detail=""
for t in 0 1; do
    d="$OUT/task_$t"
    r=$(grep -m1 "^RESULT" "$d/probe.log" 2>/dev/null)
    cfg=$(grep -c "from config file" "$d/softmig.log" 2>/dev/null)
    conf=$(head -1 "$d/conf.txt" 2>/dev/null)
    detail="$detail task$t=[${r:-none} conf=${conf:-none} cfg_lines=${cfg:-0}]"
    if [[ "$r" == "RESULT: PASS"* ]] && [ "${cfg:-0}" -ge 1 ]; then
        pass=$((pass + 1))
    fi
done
if [ "$pass" -eq 2 ]; then
    _emit "${jid:-NA}" PASS "$pass/2" "$detail"
else
    _emit "${jid:-NA}" FAIL "$pass/2" "$detail"
fi
