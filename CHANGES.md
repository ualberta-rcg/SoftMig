# SoftMig Change Log

Chronological log of changes, derived from source diffs.
For deployment and usage instructions, see `README.md`.

---

## 2026-05-08

### README — Add demo images, badges, hero line, and section icons

Added `softmig_pic1.png` and `softmig_pic2.png` showing SoftMig running on the
University of Alberta Vulcan cluster: `nvidia-smi` output for a 1/2 L40S slice
(~24 GB visible) and a 1/4 L40S slice (~12 GB visible). Images displayed
side-by-side after the Description section with captions.

Added CUDA 12+, CUDA 13, and NVIDIA GPU compatibility badges. Added hero
tagline ("Software MIG for any NVIDIA GPU — no hardware MIG required.") and
"Deployed on the University of Alberta Vulcan cluster" line. Added emoji icons
to all section headers.

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

## 2026-04 (pre-consolidation)

### `e6410d4` — Enhance README with detailed system requirements and docs

**Docs:** Comprehensive README update. Added system requirements, build
instructions, runtime dependencies, logging setup, configuration file usage,
environment variable priorities, update guidance, and nvidia-smi hook
documentation.

---

### `5da8721` — Add nvidia-smi hook script

**New:** Added `nvidia-smi-hook.sh` to filter nvidia-smi output by SLURM job
cgroup, showing only processes belonging to the current job.

---

### `2610f5c` through `7061ee8` — NVML struct mismatch workarounds (4 commits)

**Fix:** The NVML `nvmlProcessInfo_t` struct layout differed between CUDA
toolkit headers and the installed driver (e.g., CUDA 12.2 headers vs driver
570.x). PID and memory fields were at wrong offsets, causing garbage values.
Implemented `extract_pid_safely()` to scan raw bytes for valid PIDs across
multiple offsets with fast-path optimization. Implemented
`extract_memory_safely()` to extract 64-bit memory values while avoiding
overlaps with the PID field. Added throttled logging for mismatches.
`7061ee8` introduced `extract_memory_safely`; `2610f5c` added validation
to prevent memory values being mistaken for PIDs; `b5699fb` improved PID
offset detection and throttled mismatch logging.

---

### `a381423`, `7f7bc14` — Bypass hooks for memory usage calculation

**Changed:** Modified `nvmlDeviceGetComputeRunningProcesses` calls in memory
usage functions to bypass SoftMig's own hooks, hitting the real NVML driver
directly. This prevented filtered process lists from being used for limit
enforcement. Added fallback for when `nvml_library_entry` is unavailable.
Enhanced debug logging with counters for included/skipped/failed processes.

---

### `fd678d1` — Add process start time retrieval

**New:** Added `proc_get_start_time()` to read process start time from
`/proc/PID/stat`. Updated OOM killer to sort processes by PID (newest first)
instead of memory usage, so the most recently started process is killed first.

---

### `21c886a`, `d1fc3ec` — Logging cleanup and throttling

**Changed:** Moved frequent debug logs (dlsym, NVML hooks, graph wrappers)
to file-only output. Implemented fast-path PID validation in
`extract_pid_safely()` with fallback scanning only when the standard field
fails. Added throttled mismatch logging (every 100th occurrence). Adjusted
utilization watcher interval to approximately 5 seconds.

---

### `3b2c1b9`, `a21d0d9` — Thread-safe logging fixes

**Fix:** `basename()` is not thread-safe (may modify its argument). Replaced
with a local buffer copy. Fixed `va_list` usage in console logging to use
`va_copy` so the argument list isn't consumed by file logging before console
logging runs.

---

### `3f24479` — Prevent segfault in NVML library loading

**Fix:** `load_nvml_libraries()` could proceed with NULL function pointers
when `real_dlsym` wasn't found, causing segfaults. Added early returns after
all failure checks.

---

### `fe82b13`, `aeaca14`, `8e88575` — NVML error handling cleanup

**Changed:** Added error logging to `NVML_OVERRIDE_CALL` macros when function
symbols aren't found, then removed the error-level logging (too noisy for
missing optional symbols). Cleaned up commented-out code and unused includes.

---

### `89753ea` — Rename log level env var, add config caching

**Changed:** Renamed `LIBCUDA_LOG_LEVEL` to `SOFTMIG_LOG_LEVEL`. Added config
value caching in `config_file.c` so the config file isn't re-read on every
limit lookup.

---

### `8a1088d` — Safe PID extraction for NVML struct mismatches

**Fix:** Implemented `extract_pid_safely()` to scan raw struct bytes for valid
PIDs when the header field is wrong. Covers the case where CUDA toolkit
headers and the installed driver disagree on `nvmlProcessInfo_t` layout.
Added detailed logging for debugging mismatches across `process_utils.c`,
`multiprocess_memory_limit.c`, and `nvml_entry.c`.

---

### `5076f79`, `ca1517b`, `cacb8da` — NVML version field initialization

**Fix:** Added manual `infos[i].version = nvmlProcessInfo_v2` initialization
before every NVML process query across `nvml_entry.c`, `utils.c`,
`hook.c`, and `multiprocess_utilization_watcher.c`. Attempted v1 fallback
when v2 returned `NVML_ERROR_INVALID_ARGUMENT`. This was a workaround for
struct version mismatches between CUDA toolkit headers and the driver.

---

### `c4c55f0`, `ab7c19d` — Logging refinements

**Changed:** Removed verbose helper function logs. Added warnings for
insufficient NVML buffer sizes. Enhanced debug logging with raw NVML response
data for troubleshooting.

---

### `dd2aafd` — Gradual OOM killer

**New:** Implemented gradual OOM killer that targets processes by GPU memory
usage. Added `log_oom_to_syslog()` for audit trail. Integrated memory
monitoring into the utilization watcher for proactive OOM detection.

---

### `71904f4` through `aade736` — Active OOM killer with cgroup/UID filtering (7 commits)

**New:** Implemented `active_oom_killer` targeting processes by cgroup/UID
membership for job isolation. Disabled OOM handling for root. Iteratively
refined to bypass NVML hooks for unfiltered process retrieval, add UID
verification in addition to cgroup checks, and improve logging with memory
usage details. `f2b7dc6` introduced the killer; `1d1bca9`/`c12c60d` fixed
type compatibility for NVML v1/v2 structs; `d9f2b06` standardized NVML call
pattern; `b466777` added UID verification; `71904f4` bypassed hooks for
unfiltered access; `aade736` added comprehensive logging.

---

### `3ce21c6` — Include all cgroup/UID processes in memory usage

**Changed:** Modified `nvml_get_device_memory_usage` to include all processes
in the cgroup/UID regardless of whether they're registered in the shared
region. Previously, unregistered processes were invisible to OOM checks.

---

### `5535937`, `6f56c61` — PID detection robustness

**Fix:** Added PID validation (range checks, zero/garbage detection) in
`mergepid` and `set_task_pid`. Initialize PID arrays to zero to avoid garbage
values. Replaced complex PID fallback logic with direct `getpid()` call.

---

### `0e6b26c`, `6cafa3b` — NVML header conflict resolution

**Fix:** System `<nvml.h>` and SoftMig's `nvml-subset.h` defined conflicting
types. Reordered includes so `nvml-subset.h` is included first with
`NVML_NO_UNVERSIONED_FUNC_DEFS` to suppress system definitions. Added
forward declaration of `nvmlReturn_t` in `nvml_override.h`. Added
compatibility alias for `nvmlProcessInfo_v1_t`.

---

### `a72aa02`, `d73e45d` — Improved PID detection in cgroup environments

**Changed:** Enhanced `set_task_pid` to prioritize finding the current process
PID in the filtered list before falling back to the differencing method.
Improved reliability in SLURM cgroup setups. Changed verbose log levels from
INFO to DEBUG.

---

### `7f178ea`, `7f74836` — NVML v2 filtering delegation

**Changed:** Updated `nvmlDeviceGetComputeRunningProcesses` and
`nvmlDeviceGetGraphicsRunningProcesses` to delegate to v2 variants with
built-in cgroup/UID filtering. Updated `nvml_get_device_memory_usage` to
check cgroup membership before falling back to UID checks.

---

### `b76f0ea` — Merge PR #1 (cgroup-based VRAM filtering)

Merged cgroup filtering branch from Karim.

---

### `2af4b41`, `9ccfea2` — Cgroup-based process filtering

**New:** Implemented `proc_belongs_to_current_cgroup_session()` to filter
processes by SLURM job cgroup. Parses `/proc/<pid>/cgroup` for both cgroups
v1 and v2, extracts job IDs from paths like `slurm/uid_*/job_*/`. Updated
memory usage functions to use cgroup filtering with UID fallback. `9ccfea2`
fixed build errors (dlvsym NULL→empty string, missing forward declarations).

---

### `df191c2` — Summed memory usage calculation

**New:** Added `get_summed_device_memory_usage_from_nvml()` to calculate total
CUDA device memory usage for the current user with 9MB minimum + 5% overhead
per process. Filters by UID with enhanced logging.

---

### `76a2e70` through `9943b93` — Memory tracking accuracy (8 commits)

**Changed:** Iteratively improved memory tracking accuracy. Reduced per-process
minimum from 64MB to 9MB and added 5% overhead. Added lock-free variants
(`get_gpu_memory_usage_nolock`) for OOM checks. Removed fallbacks to tracked
usage — all memory reporting now uses summed NVML calculations exclusively.
Updated `cuMemGetInfo` and `cuMemGetInfo_v2` to use NVML-summed values.
Fixed processes being skipped when UID couldn't be read (previously blocked
on shared region locks).

---

### `5e73ca9`, `1251f42` — Process UID filtering

**New:** Implemented `proc_get_uid()` to retrieve process UID from
`/proc/<pid>/status`. Updated memory usage functions to filter processes by
current user's UID so only same-user processes are counted. Added
`proc_alive()`. Separated declarations into `process_utils.h` and
implementation into `process_utils.c`.

---

### `23f3e8e` — Shared region locking for allocations

**Fix:** Added `lock_shrreg`/`unlock_shrreg` to serialize shared region access
during `add_chunk` and `add_chunk_only`, preventing race conditions between
concurrent processes updating GPU memory usage counters.

---

### `47fad1e` — Minimum memory allocation per process

**Changed:** Added 64MB minimum per-process memory count in NVML usage
calculations to improve accuracy for processes with low reported usage.
Later reduced to 9MB in `9943b93`.

---

### `281f3dd`, `fcc45c1` — Config cleanup delegation to SLURM epilog

**Changed:** Removed `cleanup_config_file()` function and its exit handler
call. Config file cleanup is now the SLURM epilog script's responsibility,
streamlining exit handling.

---

### `24963c8` through `d108979` — NVML integration attempt and revert

**New → Reverted:** Extended `get_current_device_memory_usage` to use NVML
process summing. Replaced direct NVML calls with `NVML_FIND_ENTRY` dynamic
symbol resolution. Added `nvml_symbols_available` checks with weak stubs.
Then reverted all NVML integration changes due to issues, returning to direct
NVML API calls and tracked usage only.

---

### `ba8d788` — NVML process memory summing

**New:** Implemented `sum_process_memory_from_nvml()` to query NVML for
running process memory instead of relying solely on tracked allocations.
Added 2MB per-process overhead. Updated `nvmlDeviceGetMemoryInfo` to use
NVML-summed values with tracked usage fallback.

---

### `541c95f`, `6682a80` — Logging format and type fixes

**Fix:** Changed `cuMemMap` and `cuMemCreate` logging from `%lld` to `%zu`
for `size_t` parameters. Increased device index name buffer from 8 to 16
bytes. Added forward declaration for `is_softmig_configured`. Standardized
`nvmlProcessInfo_v1_t` to `nvmlProcessInfo_t`.

---

### `6d15690` — Log file path safety and EOF handling

**Fix:** Replaced `strncpy` with safe memcpy and null-termination in
`get_log_file_path`. Added NULL check for `fgets` return value in
`load_env_from_file`. Fixed missing `LOG_WARN` call.

---

### `a9aad77` — Remove redundant NVML declarations

**Cleanup:** Removed duplicate NVML function declarations from
`nvml_override.h` that conflicted with system `nvml.h`. Updated `hook.c` to
include `nvml-subset.h` first with `NVML_NO_UNVERSIONED_FUNC_DEFS`.

---

### `d7b19c8` — Reduce logging verbosity

**Changed:** Removed frequent debug logs from dlsym, NVML library loading,
`nvmlDeviceGetMemoryInfo`, and `nvmlDeviceGetHandleByIndex` to improve
performance. Modified `test_softmig.sh` to conditionally load CUDA module
with CVMFS fallback.

---

### `965504b` — Initial SoftMig codebase

**New:** Forked from HAMi-core. Renamed library from `libvgpu.so` to
`libsoftmig.so`. Updated all logging to file-only output under
`/var/log/softmig`. Implemented SLURM-aware cache and config file handling.
Added CUDA/NVML hooks, multiprocess memory management, allocator, SLURM
prolog/epilog integration scripts, and test suite.

---

### `52a563e` — Initial commit

Empty repository initialization.

---

### `510f9c2`, `8c0b421`, `6840285` — Cleanup commits

Removed unused documentation files (`UNUSED_CODE_REPORT.md`,
`GPU_LIMITER_EXPLANATION.md`), unused functions/macros from allocator and
memory management, commented-out code, and unnecessary variable declarations.
