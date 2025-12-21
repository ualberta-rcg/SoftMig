#!/bin/bash

get_current_cgroup() {
    cat /proc/self/cgroup 2>/dev/null | grep -o 'job_[0-9]*' | head -1
}

check_pid_cgroup() {
    local pid=$1
    local target_cgroup=$2
    local pid_cgroup=$(cat /proc/$pid/cgroup 2>/dev/null | grep -o 'job_[0-9]*' | head -1)
    [ "$pid_cgroup" == "$target_cgroup" ]
}

CURRENT_CGROUP=$(get_current_cgroup)

# If no cgroup, pass through everything
if [ -z "$CURRENT_CGROUP" ]; then
    exec nvidia-smi "$@"
fi

# Check if it's a CSV query
if [[ "$*" =~ "--query-compute-apps" ]]; then
    nvidia-smi "$@" | {
        read header
        echo "$header"
        while IFS=',' read -r pid rest; do
            pid=$(echo "$pid" | tr -d ' ')
            if [[ "$pid" =~ ^[0-9]+$ ]]; then
                if check_pid_cgroup "$pid" "$CURRENT_CGROUP"; then
                    echo "$pid,$rest"
                fi
            fi
        done
    }
elif [ -z "$*" ]; then
    # Standard nvidia-smi output (no arguments)
    nvidia-smi | while IFS= read -r line; do
        # Always print non-process lines
        if [[ ! "$line" =~ ^[[:space:]]*\|[[:space:]]*[0-9] ]]; then
            echo "$line"
            continue
        fi

        # This is a process line - extract PID
        # Format: |   0   N/A  N/A    123456    C   process_name    1234MiB |
        if [[ "$line" =~ [[:space:]]([0-9]{4,})[[:space:]]+C[[:space:]] ]]; then
            pid="${BASH_REMATCH[1]}"
            if check_pid_cgroup "$pid" "$CURRENT_CGROUP"; then
                echo "$line"
            fi
        else
            # Couldn't parse PID, print anyway (might be header or separator)
            echo "$line"
        fi
    done
else
    # Other nvidia-smi arguments, pass through
    exec nvidia-smi "$@"
fi
