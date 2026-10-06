#!/bin/bash
# Overnight SoftMig reliability run (launcher).
#
#   HOURS=8 bash test/run_overnight.sh
#
# The login side only (1) starts the root node sampler over sudo ssh (sudo is
# not available inside jobs), (2) submits test/overnight_driver.sh as a
# CPU-only job on the reservation node, which runs the cycles and launches
# every GPU test as its own job, and (3) when the driver ends, stops the
# sampler and writes SUMMARY.md.
#
# Artifacts: test_results/overnight_<ts>/ {results.tsv, cycle_N/..., sampler/,
# driver.out, SUMMARY.md}. Morning check: cat $(cat test_results/overnight_latest.txt)/SUMMARY.md
set -u
SOFTMIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SOFTMIG_ROOT" || exit 1
umask 002

HOURS="${HOURS:-8}"
[[ "$HOURS" =~ ^[0-9]+$ ]] && [ "$HOURS" -gt 0 ] || { echo "HOURS must be a positive integer"; exit 1; }
NODE="${SOFTMIG_NODE:-rack01-11}"

TS=$(date +%Y%m%d_%H%M%S)
ROOT="$SOFTMIG_ROOT/test_results/overnight_${TS}"
mkdir -p "$ROOT/sampler"
echo "$ROOT" > "$SOFTMIG_ROOT/test_results/overnight_latest.txt"
echo "root: $ROOT"
echo "lib on node: $(sudo -n ssh -o BatchMode=yes "$NODE" 'sha256sum /usr/local/lib/libsoftmig.so' 2>/dev/null | cut -c1-16)" | tee "$ROOT/lib.txt"

bash test/node_sampler.sh start "$ROOT/sampler" "$NODE" || { echo "sampler did not start"; exit 1; }

# the driver gets HOURS plus margin for the running cycle to finish
DRV_TIME=$(( HOURS + 2 ))
jid=$(sbatch --parsable --reservation=softmig ${SRUN_EXTRA:-} --job-name=softmig-overnight \
      --cpus-per-task=2 --mem=4G --time="${DRV_TIME}:00:00" --output="$ROOT/driver.out" \
      --export=ALL,ROOT="$ROOT",HOURS="$HOURS",SAMPLER_SHARED="$ROOT/sampler",SOFTMIG_ROOT="$SOFTMIG_ROOT",SHARE_SECS="${SHARE_SECS:-60}" \
      test/overnight_driver.sh)
jid="${jid%%;*}"
echo "$jid" > "$ROOT/driver.jid"
echo "driver job: $jid (time limit ${DRV_TIME}h)"

# wait for the driver, then stop the sampler
while [ -n "$(squeue -h -j "$jid" 2>/dev/null)" ]; do sleep 60; done
bash test/node_sampler.sh stop "$ROOT/sampler" "$NODE" > "$ROOT/sampler/stop.txt" 2>&1
python3 test/sets_summary.py "$ROOT" > "$ROOT/SUMMARY.md" 2>/dev/null
echo "Overnight run complete: $ROOT"
tail -3 "$ROOT/driver.out" 2>/dev/null
head -5 "$ROOT/SUMMARY.md"
