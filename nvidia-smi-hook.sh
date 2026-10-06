#!/bin/bash

# Wrapper policy:
# - Root sees normal, unfiltered nvidia-smi output.
# - Non-root is filtered by SLURM job cgroup (fail-closed).
# - If cgroup cannot be determined, show no process rows for compute-app queries.
#
# Why a wrapper at all: nvidia-smi (driver 535+, verified on 595.91) fetches
# its process list through NVML's private nvmlInternalGetExportTable, not
# through nvmlDeviceGet*RunningProcesses*, so libsoftmig's symbol hooks never
# see that call. Memory limits and accounting are unaffected (SoftMig filters
# its own NVML queries); this is about what users *see*. Install as
# /usr/local/bin/nvidia-smi (ahead of /usr/bin in PATH) or alias it.
#
# Filtered forms: --query-compute-apps (CSV), the default table's Processes
# section, and `pmon` rows. Everything else passes through.

REAL_NVIDIA_SMI="${SOFTMIG_REAL_NVIDIA_SMI:-/usr/bin/nvidia-smi}"

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

# pmon: rows are "gpu pid type sm mem ..." (optionally prefixed by date/time
# with -o). Keep header lines and rows whose PID is ours or "-" (idle GPU).
if [[ "$1" == "pmon" ]]; then
    "$REAL_NVIDIA_SMI" "$@" | while IFS= read -r line; do
        case "$line" in
            '#'*) echo "$line"; continue ;;
        esac
        pid="$(echo "$line" | awk '{for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/ && $(i+1) ~ /^(C|G|C\+G|-)$/) {print $i; exit}}')"
        if [ -z "$pid" ]; then
            echo "$line"
        elif [ -n "$CURRENT_CGROUP" ] && check_pid_cgroup "$pid" "$CURRENT_CGROUP"; then
            echo "$line"
        fi
    done
    exit "${PIPESTATUS[0]}"
fi

# Default table (no args, or plain -i/-q style): filter the Processes section.
# Process rows look like "|    0   N/A  N/A   12345      C   ./app   123MiB |".
if [ $# -eq 0 ] || [[ "$1" == -i ]]; then
    "$REAL_NVIDIA_SMI" "$@" | awk -v cg="$CURRENT_CGROUP" '
        function pid_ok(pid,   f, line, ok) {
            if (cg == "") return 0
            ok = 0; f = "/proc/" pid "/cgroup"
            while ((getline line < f) > 0) { if (index(line, cg "/") || index(line, cg ".") || line ~ cg "$") ok = 1 }
            close(f); return ok
        }
        /^\| Processes:/ { inproc = 1 }
        inproc && /^\|[ ]+[0-9]+[ ]+(N\/A|[0-9]+)[ ]+(N\/A|[0-9]+)[ ]+[0-9]+[ ]+[CG+]+/ {
            if (!pid_ok($5)) { hidden++; next }
        }
        { print }
        END { if (hidden > 0) printf("|  %d process(es) of other jobs hidden by SoftMig%*s|\n", hidden, 44, "") }'
    exit "${PIPESTATUS[0]}"
fi

# Default for non-root non-compute-app queries: pass through.
exec "$REAL_NVIDIA_SMI" "$@"
