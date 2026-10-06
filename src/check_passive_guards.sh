#!/bin/bash
# Build-time lint: every exported cu*/nvml* wrapper whose body touches SoftMig
# internals must start with a passive-mode forward (SOFTMIG_PASSIVE_FORWARD*,
# SOFTMIG_MEM_GUARD or an explicit softmig_is_passive() check). Otherwise a
# job without a SoftMig config can still be affected by SoftMig state, which
# is exactly what passive mode must never do.
#
# Usage: check_passive_guards.sh <src_dir> <version_script>
set -eu

SRC_DIR="${1:-.}"
VER="${2:-}"

if [ -n "$VER" ] && [ -f "$VER" ]; then
  EXPORTS=$(sed -nE 's/^[[:space:]]+((cu|nvml)[A-Za-z0-9_]+);$/\1/p' "$VER" | sort -u)
else
  EXPORTS=$(bash "$SRC_DIR/gen_version_script.sh" "$SRC_DIR" /dev/stdout |
            sed -nE 's/^[[:space:]]+((cu|nvml)[A-Za-z0-9_]+);$/\1/p' | sort -u)
fi

# ENSURE_RUNNING / ENSURE_INITIALIZED are omitted: both are no-ops in passive
# mode by construction (see include/memory_limit.h, ensure_initialized()).
INTERNAL='oom_check|allocate_raw|allocate_async_raw|free_raw|free_raw_async|add_chunk|remove_chunk|rate_limiter|pre_launch_kernel|get_current_device_|get_gpu_memory|filter_nvml|sum_process_memory|get_summed_device|nvml_cached_|nvml_to_cuda_map|cuda_to_nvml_map|lock_shrreg|set_task_pid|init_utilization_watcher|allocator_init|map_cuda_visible_devices|nvml_preInit|nvml_postInit|preInit|postInit|_nvmlDeviceGetMemoryInfo|get_current_usage_for_meminfo|softmig_meminfo|find_symbols_in_table|softmig_gpa_'
GUARD='SOFTMIG_PASSIVE_FORWARD|SOFTMIG_MEM_GUARD|softmig_is_passive[(]'

FILES=$(ls "$SRC_DIR"/cuda/*.c "$SRC_DIR"/nvml/*.c "$SRC_DIR"/libsoftmig.c 2>/dev/null)

# Emit "name<TAB>body" for every top-level function definition.
# shellcheck disable=SC2086
BODIES=$(awk '
  function flush() { if (name != "") { gsub(/\t/, " ", body); print name "\t" body } name=""; body="" }
  depth == 0 && match($0, /^[A-Za-z_][A-Za-z0-9_ \*]*[ \*]((cu|nvml)[A-Za-z0-9_]+)[ \t]*\(/) {
    s = substr($0, RSTART, RLENGTH); sub(/[ \t]*\($/, "", s); n = split(s, parts, /[ \*]+/)
    pending = parts[n]
  }
  {
    line = $0
    if (pending != "" && depth == 0 && index(line, "{") > 0) { name = pending; pending = ""; body = "" }
    if (depth == 0 && index(line, ";") > 0 && index(line, "{") == 0) { pending = "" }
    if (name != "") body = body " " line
    opens = gsub(/\{/, "{", line); closes = gsub(/\}/, "}", line)
    depth += opens - closes
    if (depth == 0 && name != "" && closes > 0) flush()
  }
' $FILES)

RESULT=$(awk -F'\t' -v internal="$INTERNAL" -v guard="$GUARD" '
  NR == FNR { exported[$1] = 1; next }
  ($1 in exported) && !($1 in seen) {
    seen[$1] = 1; checked++
    # Any softmig_* helper counts as SoftMig internals, except the
    # passive-safe ones (mode query, table loaders, unhooked logger).
    b = $2
    gsub(/softmig_is_passive|softmig_ensure_[a-z_]*|softmig_note_unhooked/, "", b)
    if ((b ~ internal || b ~ /softmig_[a-z_]+[ \t]*[(]/) && $2 !~ guard) {
      print "check_passive_guards: " $1 " touches SoftMig internals without a passive forward" > "/dev/stderr"
      bad++
    }
  }
  END { print checked + 0, bad + 0 }
' <(printf '%s\n' "$EXPORTS") <(printf '%s\n' "$BODIES"))
checked=${RESULT% *}
fail=${RESULT#* }

if [ "$fail" -ne 0 ]; then
  echo "check_passive_guards: FAILED (add SOFTMIG_PASSIVE_FORWARD at the top of the wrappers above)" >&2
  exit 1
fi
echo "check_passive_guards: OK ($checked exported wrappers checked)"
