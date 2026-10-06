#!/bin/bash
# Suite: isolation — "hard to get into others". Two same-user quarter jobs
# (A, B) packed on one GPU plus, when SOFTMIG_TEST_SUDO=1, a foreign-uid
# process started as root on the node outside Slurm (the "other user").
#
#  acct      A holds 10 GiB; B must still allocate 10 GiB (A's usage must not
#            count against B's 11517 MiB limit), also with the foreign process
#            present.
#  killscope A over-allocates with SOFTMIG_ENABLE_OOM_KILLER=1 while B holds
#            memory: every "KILLING PID" in A's log is A's own PID; B and the
#            foreign process run to completion.
#  visible   B's hooked nvidia-smi / NVML process list never shows A's or the
#            foreign PIDs (LEAK if it does — reported, not a limit failure).
#  region    B sees only its own cudevshr.cache* region (private /tmp).
#  foreign   the foreign process is passive (no OOM, unthrottled) and is not
#            killed by A's OOM killer.
#
# Output: OUT/report.md (share_report per-job table), OUT/checks.txt, one TSV line.

SUITE=isolation
DEFAULT_SRUN_TIME="${DEFAULT_SRUN_TIME:-00:12:00}"
. "$(dirname "$0")/suite_common.sh"
. "$(dirname "$0")/share_lib.sh"

Q_LIMIT=11517
rm -f "$OUT/go1" "$OUT/go2" "$OUT/B_phase2"

pack_blocker 3 || { _emit NA FAIL 0 "blocker did not start"; exit 0; }
sampler_start

run_job A l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "
echo \$EBROOTCUDA > \$D/ebroot.txt
\$B/runtime_hold 10240 45
for i in \$(seq 1 600); do [ -f '$OUT/go2' ] && break; sleep 0.5; done
sleep 2
echo '--- over-allocate with killer enabled ---'
SOFTMIG_ENABLE_OOM_KILLER=1 \$B/bypass_probe --api cudaMalloc --limit-mb $Q_LIMIT --hold 3
echo \"probe rc=\$?\"
echo '--- A done ---'
"
run_job B l40s.4:1 "$CUDA_VER" "$DEFAULT_SRUN_TIME" "
for i in \$(seq 1 600); do [ -f '$OUT/go1' ] && break; sleep 0.5; done
echo '--- phase1: 10 GiB while A holds 10 GiB ---'
\$B/runtime_hold 10240 15
echo '--- nvml view ---'
\$B/nvml_probe 512 6
ls -la /tmp/cudevshr.cache* \${SLURM_TMPDIR:-/tmp}/cudevshr.cache* /dev/shm/ 2>&1 | sed 's/^/region: /'
touch '$OUT/B_phase2'
echo '--- phase2: hold 8000 while A OOMs ---'
\$B/runtime_hold 8000 75
echo '--- phase3: 10 GiB with foreign process present ---'
\$B/runtime_hold 10240 10
echo '--- B done ---'
"
wait_started A B
sleep 15; touch "$OUT/go1"
for _ in $(seq 1 200); do [ -f "$OUT/B_phase2" ] && break; sleep 1; done

FOREIGN_PID_FILE="$OUT/foreign.pid"
if [ "${SOFTMIG_TEST_SUDO:-0}" = 1 ]; then
    uuid=$(job_meta B uuid); eb=$(cat "$OUT/jobs/A/ebroot.txt" 2>/dev/null)
    bin="$SOFTMIG_ROOT/build-cuda${CUDA_VER}/test/gpu_burn_lite"; [ -x "$bin" ] || bin="$SOFTMIG_ROOT/build/test/gpu_burn_lite"
    timeout 120 sudo -n ssh -o BatchMode=yes "$NODE" "
mkdir -p /tmp/softmig_foreign && cp -f '$bin' /tmp/softmig_foreign/gpu_burn_lite && chmod 755 /tmp/softmig_foreign/gpu_burn_lite && chmod 755 /tmp/softmig_foreign
cd /tmp/softmig_foreign
runuser -u nobody -- env CUDA_VISIBLE_DEVICES='$uuid' LD_LIBRARY_PATH='$eb/lib64' /tmp/softmig_foreign/gpu_burn_lite 3000 60 > /tmp/softmig_foreign/out.txt 2>&1 &
echo \$! > /tmp/softmig_foreign/pid; sleep 1; cat /tmp/softmig_foreign/pid
" > "$FOREIGN_PID_FILE" 2> "$OUT/foreign.err"
    sleep 10
fi
touch "$OUT/go2"
wait_jobs
if [ "${SOFTMIG_TEST_SUDO:-0}" = 1 ]; then
    for _ in $(seq 1 90); do
        timeout 30 sudo -n ssh -o BatchMode=yes "$NODE" "grep -q DONE /tmp/softmig_foreign/out.txt 2>/dev/null && exit 0; pgrep -u nobody gpu_burn_lite >/dev/null && exit 1; exit 0" 2>/dev/null && break
        sleep 1
    done
    timeout 60 sudo -n ssh -o BatchMode=yes "$NODE" "pkill -u nobody gpu_burn_lite; cat /tmp/softmig_foreign/out.txt; rm -rf /tmp/softmig_foreign" > "$OUT/foreign.out" 2>&1
fi
sleep 2
sampler_stop

python3 "$SOFTMIG_ROOT/test/share_report.py" "$OUT" --scenario isolation --md "$OUT/report.md" --tsv "$OUT/jobs.tsv" > /dev/null 2> "$OUT/report.err"

A="$OUT/jobs/A"; Bd="$OUT/jobs/B"
jidA=$(job_meta A jid); jidB=$(job_meta B jid)
pidsA=$(awk -v j="$jidA" '$3==j{print $1}' "$OUT/sampler/pidmap.txt" | sort -u)
pidsB=$(awk -v j="$jidB" '$3==j{print $1}' "$OUT/sampler/pidmap.txt" | sort -u)
foreign_pids=$(awk '$3=="none"{print $1}' "$OUT/sampler/pidmap.txt" | sort -u)

ok=1; checks=()
chk() { local name="$1" cond="$2" info="$3"; if [ "$cond" = 1 ]; then checks+=("$name=PASS($info)"); else checks+=("$name=FAIL($info)"); ok=0; fi; }

n_ok=$(_grepc "cudaMalloc(10240MB) ok" "$Bd/run.out")
n_fail=$(_grepc "cudaMalloc(10240MB) FAILED" "$Bd/run.out")
want=1; [ "${SOFTMIG_TEST_SUDO:-0}" = 1 ] && want=2
chk acct "$([ "$n_ok" -ge "$want" ] && [ "$n_fail" = 0 ] && echo 1 || echo 0)" "B_10G_ok=$n_ok fail=$n_fail"

a_oom=$(_grepc "status=OOM" "$A/run.out")
kills=$(grep -oE "KILLING PID [0-9]+" "$A/softmig.log" 2>/dev/null | grep -oE "[0-9]+$" | sort -u)
bad_kill=0; for k in $kills; do echo "$pidsA" | grep -qw "$k" || bad_kill=$((bad_kill + 1)); done
b_done=$(_grepc -- "--- B done ---" "$Bd/run.out")
chk killscope "$([ "$bad_kill" = 0 ] && [ "$b_done" = 1 ] && echo 1 || echo 0)" "A_oom=$a_oom kills=$(echo $kills | wc -w) foreign_kills=$bad_kill B_done=$b_done rcB=$(job_meta B rc)"

vis=$(awk -F'\t' -v j=B 'NR>1 && $2==j {print $22}' "$OUT/jobs.tsv" 2>/dev/null)
nv_other=0
for p in $pidsA $foreign_pids; do c=$(grep -cw "$p" "$Bd/run.out" 2>/dev/null || true); nv_other=$((nv_other + ${c:-0})); done
leak=0; [ "${vis:-0}" != 0 ] || [ "$nv_other" != 0 ] && leak=1
checks+=("visible=$([ $leak = 0 ] && echo PASS || echo LEAK)(smi_samples_with_other=${vis:-0} nvml_lines_with_other=$nv_other)")

own_regions=$(_grepc "cudevshr.cache" "$Bd/run.out")
other_regions=$(grep "region:.*cudevshr.cache" "$Bd/run.out" 2>/dev/null | grep -vc "\.${jidB}\b" || true)
chk region "$([ "${other_regions:-0}" = 0 ] && echo 1 || echo 0)" "own=$own_regions other=${other_regions:-0}"

if [ "${SOFTMIG_TEST_SUDO:-0}" = 1 ]; then
    f_done=$(_grepc "DONE" "$OUT/foreign.out")
    f_oom=$(_grepc -i "out of memory" "$OUT/foreign.out")
    f_sm=$(awk -v pids="$foreign_pids" 'BEGIN{n=split(pids,a," "); for(i=1;i<=n;i++) P[a[i]]=1} !/^#/ && ($4 in P) && $6!="-" {s+=$6; c++} END{if(c) printf "%.0f", s/c; else print 0}' "$OUT/sampler/pmon.log")
    chk foreign "$([ "$f_done" -ge 1 ] && [ "$f_oom" = 0 ] && echo 1 || echo 0)" "done=$f_done oom=$f_oom pids=$(echo $foreign_pids | wc -w) sm_mean=$f_sm"
else
    checks+=("foreign=skipped")
fi

printf '%s\n' "${checks[@]}" > "$OUT/checks.txt"
status=PASS; [ "$ok" = 1 ] || status=FAIL; [ "$ok" = 1 ] && [ "$leak" = 1 ] && status=LEAK
_emit "${jidA},${jidB}" "$status" "$(printf '%s ' "${checks[@]}")" "pidsA=$(echo $pidsA | tr ' ' ,) pidsB=$(echo $pidsB | tr ' ' ,) foreign=$(echo $foreign_pids | tr ' ' ,)"
