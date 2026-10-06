#!/bin/bash
# Static hook audit: list driver entry points that SoftMig should intercept
# (memory, launch, meminfo, NVML process/memory queries) but does not export.
# Uses the same watch / acknowledged prefix lists that drive the runtime
# UNHOOKED log in src/cuda/hook.c, so the two never drift apart.
#
# Usage: test/audit_hooks.sh [libsoftmig.so] [libcuda.so.1] [libnvidia-ml.so.1]
# Run it on a GPU node (the driver libraries live there), e.g. inside srun.
# Exit status: 0 = no unacknowledged gaps, 1 = gaps found, 2 = usage error.
set -u
cd "$(dirname "$0")/.."

SOFTMIG=${1:-build/libsoftmig.so}
LIBCUDA=${2:-$(ldconfig -p | awk '/libcuda\.so\.1 /{print $NF; exit}')}
LIBNVML=${3:-$(ldconfig -p | awk '/libnvidia-ml\.so\.1 /{print $NF; exit}')}

for f in "$SOFTMIG" "$LIBCUDA" "$LIBNVML"; do
    if [ -z "$f" ] || [ ! -r "$f" ]; then
        echo "audit_hooks: cannot read '$f'" >&2
        exit 2
    fi
done

prefix_list() {
    awk -v name="$1" '
        $0 ~ "static const char \\*const " name "\\[\\]" { on = 1; next }
        on { s = $0; while (match(s, /"[^"]+"/)) { print substr(s, RSTART + 1, RLENGTH - 2); s = substr(s, RSTART + RLENGTH) } }
        on && /NULL/ { exit }
    ' src/cuda/hook.c
}

WATCH=$(prefix_list unhooked_watch)
ACK=$(prefix_list unhooked_ack)
LEGACY=$(prefix_list unhooked_legacy)
if [ -z "$WATCH" ]; then
    echo "audit_hooks: could not parse unhooked_watch from src/cuda/hook.c" >&2
    exit 2
fi

exports() { nm -D --defined-only "$1" | awk '$2 ~ /^[TtWi]$/ {sub(/@.*/, "", $3); print $3}' | sort -u; }

HOOKED=$(exports "$SOFTMIG")
DRIVER=$( { exports "$LIBCUDA"; exports "$LIBNVML"; } | grep -E '^(cu|nvml)[A-Z]' | sort -u)

echo "softmig : $SOFTMIG ($(sha256sum "$SOFTMIG" | cut -c1-16))"
echo "libcuda : $(readlink -f "$LIBCUDA")"
echo "libnvml : $(readlink -f "$LIBNVML")"

gaps=0; acked=0; covered=0
while read -r sym; do
    printf '%s\n' "$WATCH" | while read -r p; do [[ $sym == "$p"* ]] && { echo y; break; }; done | grep -q y || continue
    if printf '%s\n' "$HOOKED" | grep -qx "$sym"; then
        covered=$((covered + 1))
    elif printf '%s\n' "$LEGACY" | grep -qx "$sym"; then
        echo "  LEGACY   $sym"
        acked=$((acked + 1))
    elif printf '%s\n' "$ACK" | while read -r p; do [[ $sym == "$p"* ]] && { echo y; break; }; done | grep -q y; then
        echo "  ACK      $sym"
        acked=$((acked + 1))
    else
        echo "  UNHOOKED $sym"
        gaps=$((gaps + 1))
    fi
done <<< "$DRIVER"

echo "audit_hooks: covered=$covered acknowledged=$acked unhooked=$gaps"
[ "$gaps" -eq 0 ]
