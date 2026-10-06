#!/bin/bash
# Suite: security — limits cannot be forged or switched off by the user.
#   envvar:   full-GPU job with CUDA_DEVICE_MEMORY_LIMIT/SM_LIMIT exported by
#             the user must stay passive (env is ignored inside Slurm).
#   symlink:  slice job whose /var/run/softmig/<jobid>.conf is replaced (as
#             root, on the job's node) by a symlink to a root-owned copy must
#             be passive (O_NOFOLLOW rejects it).
#   userown:  same, with the config chowned to the job's user.
# The two root-side cases need SOFTMIG_TEST_SUDO=1 (sudo ssh to the
# reservation node the job runs on); otherwise they are reported "skipped".
# The job waits for OUT/<case>.ready, which the outside half writes after
# tampering, and the outside half restores the config before the job ends.
#
# PASS iff every case that ran reports RESULT: PASS for --expect passive.

SUITE=security
. "$(dirname "$0")/suite_common.sh"

# envvar
mkdir -p "$OUT/envvar"
_srun_capture srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:l40s:1 --cpus-per-task=2 --mem=4G \
     --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
echo \$SLURM_JOB_ID > '${OUT}/envvar/jid.txt'
CUDA_DEVICE_MEMORY_LIMIT=1G CUDA_DEVICE_SM_LIMIT=10 ./build/test/passive_probe --expect passive > '${OUT}/envvar/probe.log' 2>&1
" >/dev/null 2>&1

tamper_case() {   # $1 = case name, $2 = root-side tamper command (uses $C for the conf path)
    local name="$1" tamper="$2" sub="$OUT/$1"
    mkdir -p "$sub"; rm -f "$sub"/ready "$sub"/started
    _srun_capture srun --reservation=softmig ${SRUN_EXTRA:-} --gres=gpu:${SLICE}:1 --cpus-per-task=2 --mem=4G \
         --time="$DEFAULT_SRUN_TIME" bash -lc "
module load cuda/${CUDA_VER}
cd ${SOFTMIG_ROOT}
echo \$SLURM_JOB_ID > '${sub}/jid.txt'
echo \"\$(hostname) /var/run/softmig/\$SLURM_JOB_ID.conf\" > '${sub}/started'
for i in \$(seq 1 120); do [ -f '${sub}/ready' ] && break; sleep 1; done
ls -la /var/run/softmig/\$SLURM_JOB_ID.conf* > '${sub}/conf_ls.txt' 2>&1
SOFTMIG_LOG_LEVEL=1 ./build/test/passive_probe --expect passive > '${sub}/probe.log' 2>&1
touch '${sub}/done'
for i in \$(seq 1 60); do [ -f '${sub}/restored' ] && break; sleep 1; done
_copy_softmig_log \$SLURM_JOB_ID '${sub}/softmig.log'
" >/dev/null 2>&1 &
    local job=$!
    for _ in $(seq 1 180); do [ -f "$sub/started" ] && break; sleep 1; done
    if [ -f "$sub/started" ]; then
        local node conf
        read -r node conf < "$sub/started"
        timeout 60 sudo -n ssh -o BatchMode=yes "$node" "C=$conf; cp -a \$C \$C.orig && $tamper" \
            > "$sub/tamper.log" 2>&1
        touch "$sub/ready"
        for _ in $(seq 1 120); do [ -f "$sub/done" ] && break; sleep 1; done
        timeout 60 sudo -n ssh -o BatchMode=yes "$node" \
            "C=$conf; rm -f \$C && mv \$C.orig \$C && chown root:root \$C && chmod 644 \$C" >> "$sub/tamper.log" 2>&1
        touch "$sub/restored"
    fi
    wait "$job"
}

if [ "${SOFTMIG_TEST_SUDO:-0}" = 1 ]; then
    tamper_case symlink 'rm -f $C && ln -s $C.orig $C'
    tamper_case userown "chown $(id -un) \$C"
fi

pass=0; ran=0; detail=""
for c in envvar symlink userown; do
    if [ ! -f "$OUT/$c/probe.log" ]; then
        detail="$detail $c=skipped"
        continue
    fi
    ran=$((ran + 1))
    r=$(grep -m1 "^RESULT" "$OUT/$c/probe.log")
    warn=$(grep -cE "symlink|not owned by root|not a regular|ELOOP|refus|ignor" "$OUT/$c/softmig.log" 2>/dev/null)
    detail="$detail $c=[${r:-none} warn=${warn:-0}]"
    [[ "$r" == "RESULT: PASS"* ]] && pass=$((pass + 1))
done
jids=$(cat "$OUT"/*/jid.txt 2>/dev/null | paste -sd,)
if [ "$ran" -ge 1 ] && [ "$pass" -eq "$ran" ]; then
    _emit "$jids" PASS "$pass/$ran" "$detail"
else
    _emit "$jids" FAIL "$pass/$ran" "$detail"
fi
