#!/bin/bash
# Shared helpers for SoftMig matrix suites.
# Each caller sets SLICE (e.g. l40s.2), CUDA_VER (e.g. 12.2), OUT (dir).
#
# Every suite prints ONE TSV line to stdout:
#   CUDA_VER \t SLICE \t SUITE \t JOBID \t STATUS \t METRIC \t DETAIL
#
# and stores artifacts (per-proc logs, softmig.log, nvsmi.log) in OUT/.

set -u
: "${SLICE:?SLICE is required}"
: "${CUDA_VER:?CUDA_VER is required}"
: "${OUT:?OUT is required}"
: "${SUITE:?SUITE is required}"

SOFTMIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SOFTMIG_ROOT
mkdir -p "$OUT"

# grep -c prints "0" AND exits 1 on no match, so `grep -c ... || echo 0`
# yields "0\n0" and breaks numeric tests. Always use this instead.
_grepc() { local c; c=$(grep -c "$@" 2>/dev/null); echo "${c:-0}"; }

_emit() {
    local jobid="$1" status="$2" metric="$3" detail="$4"
    # A watchdog that fired overrides whatever the suite concluded.
    if [ -f "$OUT/HANG" ]; then
        detail="hang: $(head -1 "$OUT/HANG") | $detail"
        status=HANG
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$CUDA_VER" "$SLICE" "$SUITE" "$jobid" "$status" "$metric" "$detail"
}

# --- Hang capture --------------------------------------------------------
# Inside the job: `_hang_watchdog SECS OUTDIR` (backgrounds itself). If the
# job is still running after SECS it records every job process's per-thread
# state from /proc into OUTDIR/hang_<pid>.txt, writes OUTDIR/HANG, waits up
# to 60s for OUTDIR/HANG.done (full gdb stacks taken from outside, see
# _srun_capture), then SIGKILLs the job's non-ancestor processes so the job
# script can still copy its logs. /tmp is wiped at job end, so everything
# goes to OUTDIR (project space). ptrace_scope=1 on the nodes, so a process
# in the job cannot gdb its siblings; root gdb happens outside.
_hang_watchdog() {
    local secs="$1" outdir="$2" jid="${SLURM_JOB_ID:?}"
    (
        sleep "$secs"
        local self=$BASHPID ancestors="" p=$BASHPID
        while [ -n "$p" ] && [ "$p" -gt 1 ]; do
            ancestors="$ancestors $p"
            p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null)
        done
        local victims="" uid; uid=$(id -u)
        for d in /proc/[0-9]*; do
            local pid=${d#/proc/}
            [ "$pid" = "$self" ] && continue
            case " $ancestors " in *" $pid "*) continue ;; esac
            [ "$(stat -c %u "$d" 2>/dev/null)" = "$uid" ] || continue
            grep -q "/job_${jid}/" "$d/cgroup" 2>/dev/null || continue
            local comm ppid
            comm=$(cat "$d/comm" 2>/dev/null) || continue
            [ "$comm" = slurmstepd ] && continue
            ppid=$(awk '{print $4}' "$d/stat" 2>/dev/null)
            [ "$ppid" = "$self" ] && continue
            victims="$victims $pid"
            {
                echo "pid=$pid comm=$comm"
                tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null; echo
                for t in /proc/$pid/task/*; do
                    # fields: 3=state 14=utime 15=stime
                    echo "tid=${t##*/} $(cat "$t/comm" 2>/dev/null) state=$(awk '{print $3}' "$t/stat" 2>/dev/null) utime=$(awk '{print $14}' "$t/stat" 2>/dev/null) wchan=$(cat "$t/wchan" 2>/dev/null)"
                done
            } > "$outdir/hang_$pid.txt" 2>&1
        done
        echo "watchdog fired after ${secs}s on $(hostname): pids${victims}" > "$outdir/HANG"
        echo "node=$(hostname)" >> "$outdir/HANG"
        echo "pids=${victims# }" >> "$outdir/HANG"
        for _ in $(seq 1 60); do
            [ -f "$outdir/HANG.done" ] && break
            sleep 1
        done
        # shellcheck disable=SC2086
        [ -n "$victims" ] && kill -9 $victims 2>/dev/null
    ) &
    echo $! > "$outdir/watchdog.pid"
}
export -f _hang_watchdog

_hang_disarm() {
    local outdir="$1"
    [ -f "$outdir/watchdog.pid" ] && kill "$(cat "$outdir/watchdog.pid")" 2>/dev/null
    rm -f "$outdir/watchdog.pid"
}
export -f _hang_disarm

# Outside the job: run srun in the background and, if the in-job watchdog
# raises OUT/HANG, take all-thread gdb stacks as root on the node (only when
# SOFTMIG_HANG_GDB_SUDO=1 and only on the reservation node the job ran on).
_srun_capture() {
    "$@" &
    local srun_pid=$!
    while kill -0 "$srun_pid" 2>/dev/null; do
        if [ -f "$OUT/HANG" ] && [ ! -f "$OUT/HANG.done" ]; then
            if [ "${SOFTMIG_HANG_GDB_SUDO:-0}" = 1 ]; then
                local node pids
                node=$(sed -n 's/^node=//p' "$OUT/HANG")
                pids=$(sed -n 's/^pids=//p' "$OUT/HANG")
                for pid in $pids; do
                    timeout 60 sudo -n ssh -o BatchMode=yes "$node" \
                        "gdb -p $pid -batch -ex 'thread apply all bt 25'" \
                        > "$OUT/gdb_$pid.txt" 2>&1 || true
                done
            fi
            touch "$OUT/HANG.done"
        fi
        sleep 2
    done
    wait "$srun_pid"
}

# Copy softmig log off /var/log while still inside the job (only the owning uid
# can read it). Caller passes $SLURM_JOB_ID as $1 and the target path as $2.
_copy_softmig_log() {
    local jid="$1" dest="$2"
    if [ -r "/var/log/softmig/${jid}.log" ]; then
        cp "/var/log/softmig/${jid}.log" "$dest" 2>/dev/null || true
    fi
}
export -f _copy_softmig_log

# Seconds of wall-clock slack we give srun for small tests.
DEFAULT_SRUN_TIME="${DEFAULT_SRUN_TIME:-00:05:00}"
