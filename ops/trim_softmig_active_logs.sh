#!/bin/bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-/var/log/softmig}"
AGE_MINUTES="${AGE_MINUTES:-360}"       # 6 hours
KEEP_BYTES="${KEEP_BYTES:-104857600}"   # 100 MB
MAINT_LOG="${MAINT_LOG:-/var/log/softmig-maint.log}"

now_epoch=$(date +%s)

shopt -s nullglob
for f in "$LOG_DIR"/*.log; do
    [ -f "$f" ] || continue

    size=$(stat -c %s "$f" 2>/dev/null || echo 0)
    mtime=$(stat -c %Y "$f" 2>/dev/null || echo "$now_epoch")
    age_min=$(( (now_epoch - mtime) / 60 ))

    # Only trim inactive-ish current logs larger than threshold.
    if [ "$age_min" -ge "$AGE_MINUTES" ] && [ "$size" -gt "$KEEP_BYTES" ]; then
        tmp="${f}.trim.$$"
        tail -c "$KEEP_BYTES" "$f" > "$tmp"
        cat "$tmp" > "$f"
        rm -f "$tmp"
        printf '%s trimmed %s to %s bytes (age_min=%s)\n' \
          "$(date '+%F %T')" "$f" "$KEEP_BYTES" "$age_min" >> "$MAINT_LOG"
    fi
done

# Clean up stale empty current logs from completed jobs.
find "$LOG_DIR" -maxdepth 1 -type f -name "*.log" -size 0c -mmin +120 -delete || true
