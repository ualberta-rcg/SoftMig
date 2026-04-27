#!/bin/bash

# Wrapper policy:
# - Root sees normal, unfiltered nvidia-smi output.
# - Non-root is filtered by SLURM job cgroup (fail-closed).
# - If cgroup cannot be determined, show no process rows for compute-app queries.

REAL_NVIDIA_SMI="/usr/bin/nvidia-smi"

extract_job_id() {
    sed -n 's/.*\(job_[0-9][0-9]*\).*/\1/p' | head -1
}

get_current_cgroup() {
    cat /proc/self/cgroup 2>/dev/null | extract_job_id
}

get_pid_cgroup() {
    local pid="$1"
    cat "/proc/$pid/cgroup" 2>/dev/null | extract_job_id
}

check_pid_cgroup() {
    local pid="$1"
    local target_cgroup="$2"
    local pid_cgroup
    pid_cgroup="$(get_pid_cgroup "$pid")"
    [ -n "$pid_cgroup" ] && [ "$pid_cgroup" = "$target_cgroup" ]
}

# User requested: root should get normal nvidia-smi output.
if [ "$(id -u)" -eq 0 ]; then
    exec "$REAL_NVIDIA_SMI" "$@"
fi

CURRENT_CGROUP="$(get_current_cgroup)"

# Compute-app queries are used by monitoring scripts; enforce fail-closed.
if [[ "$*" == *"--query-compute-apps"* ]]; then
    noheader=0
    if [[ "$*" == *"--format=csv,noheader"* ]]; then
        noheader=1
    fi

    "$REAL_NVIDIA_SMI" "$@" | {
        if [ "$noheader" -eq 0 ]; then
            IFS= read -r header || true
            [ -n "$header" ] && echo "$header"
        fi

        # No detectable cgroup => fail-closed (emit no process rows).
        if [ -z "$CURRENT_CGROUP" ]; then
            exit 0
        fi

        while IFS=',' read -r pid rest; do
            pid="$(echo "$pid" | tr -d '[:space:]')"
            if [[ "$pid" =~ ^[0-9]+$ ]] && check_pid_cgroup "$pid" "$CURRENT_CGROUP"; then
                if [ -n "$rest" ]; then
                    echo "$pid,$rest"
                else
                    echo "$pid"
                fi
            fi
        done
    }
    exit $?
fi

# Default for non-root non-compute-app queries: pass through.
exec "$REAL_NVIDIA_SMI" "$@"
