#!/bin/bash
# Suite: bypass — every allocation API, one process each, tries to allocate
# past the slice limit; the root-side sampler supplies the truth.
#
#   SLICE=l40s.4 (enabled)  -> each device API must stop with OOM at or below
#                              limit+slack (CAPPED). Over the limit in truth is
#                              LEAK; reaching 1.5x the limit is UNHOOKED.
#                              Host APIs must NOT be counted (HOST_OK).
#   SLICE=l40s   (passive)  -> every API must reach 1.5x the pseudo limit
#                              (PASSTHROUGH): no SoftMig OOM on a whole GPU.
#
# Output: OUT/bypass.tsv (api, status, allocated, limit, truth_peak, verdict),
# OUT/sampler/*, OUT/probe_<api>.out, OUT/softmig.log.
# PASS iff no LEAK / UNHOOKED / HOST_COUNTED / ENFORCED_PASSIVE rows.

SUITE=bypass
DEFAULT_SRUN_TIME="${DEFAULT_SRUN_TIME:-00:20:00}"
. "$(dirname "$0")/suite_common.sh"
. "$(dirname "$0")/share_lib.sh"

SLACK_MIB="${BYPASS_SLACK_MIB:-512}"
APIS="${BYPASS_APIS:-cudaMalloc cuMemAlloc cudaMallocAsync cuMemAllocAsync cuMemAllocFromPoolAsync cudaMallocFromPoolAsync cuMemCreateMap cuMemAllocManaged cudaMallocManaged cudaMallocPitch cuMemAllocPitch cudaMalloc3D cudaMallocArray cudaMalloc3DArray cudaMallocMipmappedArray cuArrayCreate cuArray3DCreate cuGraphAddMemAllocNode cuMemAllocHost cudaHostAlloc cuCtxCreate}"
HOST_APIS=" cuMemAllocHost cudaHostAlloc "

if [ "$SLICE" = l40s ]; then PSEUDO=11517; else PSEUDO=0; fi

sampler_start
# pinned-host probes allocate ~1.1x the device limit in RAM: give the job room
JOB_MEM="${JOB_MEM:-28G}" run_job probe "${SLICE}:1" "$CUDA_VER" "$DEFAULT_SRUN_TIME" "
limit=\$(sed -n 's/^CUDA_DEVICE_MEMORY_LIMIT=\([0-9]*\).*/\1/p' \$D/conf.txt); [ -z \"\$limit\" ] && limit=$PSEUDO
echo \$limit > \$D/limit.txt
for api in $APIS; do
    \$B/bypass_probe --api \$api --limit-mb \$limit --hold 4 > \$D/probe_\$api.out 2>&1
    echo \"rc=\$?\" >> \$D/probe_\$api.out
    sleep 2
done
"
wait_jobs
sleep 2
sampler_stop

D="$OUT/jobs/probe"
jid=$(job_meta probe jid); limit=$(cat "$D/limit.txt" 2>/dev/null || echo 0)
tsv="$OUT/bypass.tsv"
printf 'api\tstatus\tsteps\tallocated_mb\tlimit_mb\ttruth_peak_mb\tover_mb\tview_total_mb\tverdict\terr\n' > "$tsv"

counts_capped=0; counts_leak=0; counts_unhooked=0; counts_host=0; counts_bad=0; counts_pt=0; counts_info=0
for api in $APIS; do
    line=$(grep -m1 '^BYPASS' "$D/probe_$api.out" 2>/dev/null)
    if [ -z "$line" ]; then
        printf '%s\tNORUN\t0\t0\t%s\t0\t0\t0\tERROR\t%s\n' "$api" "$limit" "$(tail -1 "$D/probe_$api.out" 2>/dev/null | tr '\t' ' ')" >> "$tsv"
        counts_bad=$((counts_bad + 1)); continue
    fi
    pid=$(echo "$line" | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
    status=$(echo "$line" | sed -n 's/.*status=\([A-Z]*\).*/\1/p')
    steps=$(echo "$line" | sed -n 's/.*steps=\([0-9]*\).*/\1/p')
    alloc=$(echo "$line" | sed -n 's/.*allocated_mb=\([0-9]*\).*/\1/p')
    vtot=$(echo "$line" | sed -n 's/.*view_total_mb=\([0-9]*\).*/\1/p')
    err=$(echo "$line" | sed -n 's/.*err=\(.*\)$/\1/p')
    truth=$(awk -v p="$pid" '{gsub(",","",$3); gsub(",","",$4); if ($3==p && $4+0>m) m=$4+0} END{print m+0}' "$OUT/sampler/apps.log")
    over=$((truth - limit))
    is_host=0; case "$HOST_APIS" in *" $api "*) is_host=1 ;; esac
    if [ "$SLICE" = l40s ]; then
        if [ "$status" = REACHED ]; then verdict=PASSTHROUGH; counts_pt=$((counts_pt + 1))
        elif [ "$status" = OOM ]; then verdict=ENFORCED_PASSIVE; counts_bad=$((counts_bad + 1))
        else verdict=ERROR; counts_info=$((counts_info + 1)); fi
    elif [ "$is_host" = 1 ]; then
        if [ "$status" = REACHED ] && [ "$over" -le "$SLACK_MIB" ]; then verdict=HOST_OK; counts_host=$((counts_host + 1))
        else verdict=HOST_COUNTED; counts_bad=$((counts_bad + 1)); fi
    elif [ "$api" = cuCtxCreate ]; then
        verdict=INFO; counts_info=$((counts_info + 1))
        [ "$over" -gt "$SLACK_MIB" ] && { verdict=LEAK; counts_leak=$((counts_leak + 1)); }
    else
        case "$status" in
            OOM)     if [ "$over" -le "$SLACK_MIB" ]; then verdict=CAPPED; counts_capped=$((counts_capped + 1)); else verdict=LEAK; counts_leak=$((counts_leak + 1)); fi ;;
            REACHED) verdict=UNHOOKED; counts_unhooked=$((counts_unhooked + 1)) ;;
            *)       if [ "$over" -gt "$SLACK_MIB" ]; then verdict=LEAK; counts_leak=$((counts_leak + 1)); else verdict=ERROR; counts_info=$((counts_info + 1)); fi ;;
        esac
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$api" "$status" "$steps" "$alloc" "$limit" "$truth" "$over" "$vtot" "$verdict" "$err" >> "$tsv"
done

slog="$D/softmig.log"
unh=$(grep -c UNHOOKED "$slog" 2>/dev/null); unh=${unh:-0}
errs=$(grep -c "softmig ERROR" "$slog" 2>/dev/null); errs=${errs:-0}
metric="capped=$counts_capped leak=$counts_leak unhooked=$counts_unhooked host_ok=$counts_host passthrough=$counts_pt info=$counts_info bad=$counts_bad"
detail="limit=${limit}M slack=${SLACK_MIB} log_unhooked=$unh log_errors=$errs $(awk -F'\t' 'NR>1 && $9!~/CAPPED|HOST_OK|PASSTHROUGH/ {printf "%s=%s(%s,+%s) ", $1, $9, $2, $7}' "$tsv")"
if [ "$counts_leak" = 0 ] && [ "$counts_unhooked" = 0 ] && [ "$counts_bad" = 0 ] && [ "$unh" = 0 ]; then
    _emit "$jid" PASS "$metric" "$detail"
else
    _emit "$jid" FAIL "$metric" "$detail"
fi
