# SoftMig Change Log

Chronological log of changes, derived from source diffs.
For deployment and usage instructions, see `README.md`.
For current open operational work, see `docs/PROJECT_STATUS.md`.

---

## 2026-04-27

### `83090b5` — Fix array job config file mismatch + accumulated changes

**Fix: array-job config filename mismatch caused full GPU passthrough.**
The prolog writes `/var/run/softmig/{jobid}.conf` but `SLURM_ARRAY_TASK_ID`
isn't reliably set in the prolog environment. Inside the job, `config_file.c`
looked for `{jobid}_{arrayid}.conf` which didn't exist, so the limit came back
as 0 — SoftMig did nothing. `config_file.c` now falls back from the
array-suffixed filename to the base filename. `prolog_softmig.sh` now extracts
the array task ID from `scontrol --json` output instead of relying on the env
var. Both `SLURM_JOB_ID` and `SLURM_ARRAY_TASK_ID` are validated as numeric
before being used in any path.

**Fix: SLURM jobs could override limits via env vars.**
`get_limit_from_config_or_env()` fell back to user environment variables when
the config file was missing. A user could set `CUDA_DEVICE_MEMORY_LIMIT` and
bypass the prolog's root-owned config. When `SLURM_JOB_ID` is set, env var
fallback is now disabled — config files are the sole source of truth.

**Fix: async allocation path had no shared-region lock.**
`add_chunk_async()` called `oom_check()` without holding `lock_shrreg()`.
Multiple processes doing concurrent async allocations could race on the shared
memory usage counters, leading to over-admission. `lock_shrreg()` is now held
around the entire OOM check + alloc + update sequence, with proper release on
all early-return paths.

**Fix: `add_chunk_only()` deadlocked on double lock.**
Called `oom_check()` (which acquires `lock_shrreg()`) while the caller already
held `lock_shrreg()`. Changed to `oom_check_nolock()`.

**Fix: file locking used fragile O_EXCL create/delete.**
`try_lock_unified_lock()` used `open(O_CREAT | O_EXCL)` in a retry loop with
`sleep()` and `remove()`. Crashed processes left stale lock files. Replaced
with `flock(LOCK_EX)` on a persistent fd — no stale files, no sleeps.

**Fix: `nvidia-smi` wrapper issues.**
Root saw filtered output (should see everything). No-cgroup case was fail-open
(showed all processes). CSV header was broken for `--format=csv,noheader`.
Root now gets unfiltered passthrough. No-cgroup is fail-closed. Only
`--query-compute-apps` queries are filtered now.

**New: dlsym resolver consolidated into shared module.**
The "find real dlsym" logic (dlvsym with multiple glibc version fallbacks,
libdl.so.2 direct open, `_dl_sym` weak symbol) was duplicated in
`libsoftmig.c` and `nvml/hook.c`. Extracted into `dlsym_resolve.c` as a
single `resolve_real_dlsym()`, thread-safe via `pthread_once`.

**New: NVML process-list cache.**
The allocator and utilization watcher both call
`nvmlDeviceGetComputeRunningProcesses` on every allocation and watcher tick.
New `nvml_cache.c` provides a per-device TTL cache (1 second), invalidated
explicitly on alloc/free so stale data is never used for enforcement.
External calls (nvidia-smi, monitoring) also invalidate the cache.

**New: linker version script for symbol export.**
Apps that link `-lcuda` directly (gpu-burn, cuda-samples) resolve CUDA symbols
at load time, not via `dlsym`. SoftMig's wrappers were invisible because the
build used `-fvisibility=hidden`. New `gen_version_script.sh` auto-generates
a `.ver` script exporting every `cu*`/`nvml*` wrapper. Removed `-lnvidia-ml`
from build-time linking — NVML symbols are now resolved purely at runtime via
`dlopen("libnvidia-ml.so.1")`.

**Fix: `nvmlProcessInfo_t` struct definition in `nvml-subset.h` didn't match
driver ABI.** This was the root cause behind `extract_pid_safely` and all the
byte-scanning workarounds. The struct definition was replaced to match the
actual NVML v2 ABI layout used by current drivers.

**Removed: NVML struct mismatch workarounds.**
`extract_pid_safely()` and `extract_memory_safely()` (~230 lines) scanned raw
bytes to handle struct layout mismatches between CUDA toolkit headers and the
driver. With the correct `nvmlProcessInfo_t` from `nvml-subset.h`, all callers
now use `infos[i].pid` and `infos[i].usedGpuMemory` directly. Removed all
manual version field initialization and v1/v2 retry logic from `nvml_entry.c`,
`utils.c`, and `nvml/hook.c`.

**Removed: custom `nvmlProcessInfo_t1`/`nvmlProcessInfo_v1_t` typedefs.**
Code used custom typedefs that diverged from the canonical
`nvmlProcessInfo_t` in `nvml-subset.h`, causing type mismatches and requiring
casts. All code now uses the canonical type. `nvml/hook.c`'s
`sum_process_memory_from_nvml()` simplified accordingly — reads
`infos[i].usedGpuMemory` directly instead of calling `extract_memory_safely()`,
and uses the NVML cache via `nvml_cached_get_compute_processes()` instead of
calling the driver on every invocation.

**Changed: cgroup-session checking now cached.**
`proc_belongs_to_current_cgroup_session()` read `/proc/self/cgroup` on every
call — once per NVML process per query, a hot path. Current process's cgroup
and extracted job ID are now computed once and cached in static variables.

**Changed: config file reads now security-hardened.**
`lstat()` rejects symlinks. Regular-file check rejects FIFOs/device nodes.
Root-ownership check rejects non-root files. `is_numeric()` validates
`SLURM_JOB_ID` and `SLURM_ARRAY_TASK_ID` before path construction.

**Changed: unified root/non-root NVML process filtering.**
Root previously saw all processes unconditionally, making it impossible to
verify SoftMig's filtering was working. Root now also goes through cgroup
filtering, falling back to include-all only when cgroup detection fails.

**Changed: semaphore timeout tuning.**
`SEM_WAIT_TIME` 10→3s, `SEM_WAIT_RETRY_TIMES` 30→5. Total lock timeout
reduced from 300s to 15s. `ctx_activate` array size replaced hardcoded `32`
with `CTX_ACTIVATE_SIZE` constant.

**Changed: utilization watcher always enabled.**
Removed `set_env_utilization_switch()` call; `env_utilization_switch` is now
hardcoded to 1.

**Changed: config file path and limit lookups are now cached.**
`get_config_file_path()` uses a static buffer (computed once).
`get_limit_from_config_or_env()` caches results with a memory barrier
(`__sync_synchronize()`) before setting the `cached` flag to ensure
visibility across threads.

**Changed: allocator free paths invalidate NVML cache.**
`free_raw()`, `free_raw_async()`, and `add_chunk_only()` now call
`nvml_cache_invalidate()` after updating shared state, so subsequent OOM
checks see fresh NVML data.

**Changed: removed duplicate `cuMemcpyDtoDAsync_v2` and unused `cuDeviceGet`
from dlsym hook table.

**Cleanup: dead code removed.**
Unused `region_struct` types, `BITSIZE`/`OVERSIZE`/`CHUNK_SIZE` constants,
`CUMALLOC`/`CUCREATE` macros, commented-out NVML stubs in `nvml_entry.c`,
verbose `LOG_DEBUG` from every CUDA graph wrapper in `graph.c`, unused
`sort()`/`initial_virtual_devices()`/`parser()` declarations from `utils.h`.

**New: test suite.**
14 test scripts and 3 probe binaries under `test/`:
`run_burn.sh`, `run_matrix.sh`, `run_multiproc.sh`, `run_multiproc_rt.sh`,
`run_overnight.sh`, `run_whack.sh` (runners);
`suite_smoke.sh`, `suite_soak.sh`, `suite_oom.sh`, `suite_sm.sh`,
`suite_nvsmi.sh`, `suite_direct.sh`, `suite_mixed.sh`,
`suite_crossjob.sh` (test suites); `suite_common.sh` (shared helpers);
`summarize.awk` (result summarizer). Probe binaries: `gpu_burn_lite.cu`
(lightweight GPU memory stressor), `nvml_probe.c` (direct NVML query probe),
`runtime_hold.cu` (CUDA context holder for multi-process tests).

**New: ops tooling.**
`install_softmig_logrotate.sh`, `logrotate-softmig.conf`,
`trim_softmig_active_logs.sh` under `ops/`.

---

### `3fb7c5f` — Consolidate docs and align project status references

Trimmed `CHANGES.md`, `IMPROVEMENT_IDEAS.md`, `MASTER_FIX_CHECKLIST.md`, and
`SESSION_HANDOFF.md` to reflect current state. Added `docs/PROJECT_STATUS.md`.
Renamed old struct mismatch suggestions doc to `NVML_STRUCT_MISMATCH_HISTORY.md`.

---

### `6cb0cbf` — Consolidate docs into install runbook, cleanup stale files

Slimmed README. Created `docs/BUILD_AND_INSTALL.md` as the single install
runbook. Removed stale files: `build.sh.bak`, `DEPLOYMENT_DRAC.md`,
`FIXES_TO_APPLY.md`, `MASTER_FIX_CHECKLIST.md`, `NVML_STRUCT_MISMATCH_HISTORY.md`,
`SESSION_HANDOFF.md`, and HAMi architecture images.

---

### `68064de` — Fix README inaccuracies found during code audit

Corrected factual errors in README. Added missing details to
`BUILD_AND_INSTALL.md` (unshare -m update procedure, CUDA 12 requirement).

---

### `f9c7a23` — Fix epilog array-job cleanup, remove stale stub

Fixed `docs/examples/epilog_softmig.sh` for array job config file cleanup.
Updated `docs/examples/install_softmig.sh`. Removed stale `IMPROVEMENT_IDEAS.md`
content.

---

### `87eb21f` through `2fd64a5` — README revisions (5 commits)

Multiple README updates improving feature descriptions, formatting, and
accuracy. Added detailed SoftMig features section, revised for clarity.

---

### `8d703a6` — Slim README, add core usage/integration guides

Split README content into dedicated docs: `docs/USAGE.md`,
`docs/SLURM_INTEGRATION.md`, `docs/TESTING.md`, `docs/TROUBLESHOOTING.md`,
`docs/UOFA_VULCAN_NOTES.md`. README reduced to description + links.

---

### `a32399f` — Rewrite README with proper description and credits

Final README rewrite with accurate project description, clickable doc links,
and credits for HAMi-core upstream and Tim Weiers.

---

### `2b7a498` — Align all docs with audited source code behavior

Updated `BUILD_AND_INSTALL.md`, `SLURM_INTEGRATION.md`, `TROUBLESHOOTING.md`,
and `USAGE.md` to match the actual code behavior found during source audit.

---

### `5b93294` — Add Vulcan Docs link and support note

Added link to Vulcan cluster documentation site and support contact info.

---

## Earlier Milestones

### Foundational SoftMig Fork (from HAMi-core)

- Renamed project/library to SoftMig (`libsoftmig.so`)
- Switched to file-based logging under `/var/log/softmig`
- Added secure config-file-first model for SLURM jobs
- Added passive mode when no config/environment is present
- Added SLURM prolog/epilog integration examples
- Standardized cache/lock paths to `SLURM_TMPDIR` for job isolation
