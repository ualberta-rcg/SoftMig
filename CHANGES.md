# SoftMig Change Log

Chronological log of changes, derived from source diffs.
For deployment and usage instructions, see `README.md`.

---

## 2026-10-01 — branch `2.06`

### Shared-region lock: robust mutex, nobody gets robbed

The region lock was a `sem_t` plus an `owner_pid` used to guess when a
holder was gone; after 18 s a waiter "recovered" the lock by force even from
a live holder and re-initialized the semaphore under other waiters. The new
`fault` suite reproduced it on 2.05: SIGSTOPped workers had the lock stolen
4 times and the semaphore was left in a held state with no owner.

The lock is now a robust, process-shared, error-checking `pthread_mutex_t`
(region layout 2.0). A live holder is waited for (WARN every 15 s); if a
holder dies the kernel hands the next locker `EOWNERDEAD`, which clears dead
slots and marks the mutex consistent. Re-entrant per thread. `fix_lock_shrreg`
and the forced recovery are gone. The layout version is part of the region
file name (`cudevshr.cache.v2.<jobid>`), so a job spanning a library redeploy
never mixes layouts in one file. The exit handler waits up to 10 s (was 3 s).

### Allocations no longer run under the lock

The limit check, the real `cuMemAlloc` and the bookkeeping all ran under
the region lock, serializing every allocation in a job. Now:
`softmig_reserve()` checks the limit and records the bytes as *pending* in
the process's slot (under the lock), the driver allocation runs with no lock
held, and the chunk is committed (or the reservation released). The limit
check counts `max(tracked, NVML) + pending`, so concurrent admissions stay
exact; a process that dies mid-allocation takes its pending bytes with its
slot. The NVML usage read happens before taking the lock, and allocation
commits no longer invalidate the NVML cache (only frees do).
`cuMemAllocManaged`, `cuMemAllocPitch_v2`, `cuMemCreate` and
`cuMemAllocFromPoolAsync` use the same reserve/commit pair; previously they
re-ran the limit check after a successful driver allocation and could
return OOM with the memory live and untracked.

Stress suite (64 processes x 4 threads on a half slice): 199k alloc/free
cycles vs 44k on 2.05 in the same time, no lock timeouts (2.05: 3 exits
could not take the lock); under 1 GiB requests from 16 processes the real
peak stayed at 18.4 of 23.0 GiB with OOMs returned.

### Other fixes

- Tracked usage is stored at the NVML device index but the OOM check, the
  OOM killers and `get_current_device_memory_usage` read it with CUDA
  indices; they now map first (matters when CUDA and NVML numbering differ).
- Region full (1024 processes): the extra process runs untracked with an
  ERROR instead of being killed by `exit(-1)`.
- `remove_chunk_only` (cuMemRelease) now takes the allocator mutex.
- If `/var/log/softmig` is not writable the log falls back to
  `$SLURM_TMPDIR`, which is wiped at job end; ERROR lines are then also
  written to stderr (the job's output file), with a note naming the file.
- `cuGraphAddMemAllocNode` is hooked as a limit check (graph memory nodes
  were an unacknowledged bypass). Nodes created implicitly by stream capture
  are still not seen.

### Tests

- Hang capture: suites arm an in-job watchdog; a hang becomes `HANG` with
  per-thread `/proc` state, and with `SOFTMIG_HANG_GDB_SUDO=1` the runner
  takes root gdb stacks on the reservation node (ptrace_scope=1 blocks it
  inside the job). The stuck processes are then killed so logs are kept.
- New one-off suites (CUDA 12.6): `stress` (16/32/64 procs + enforcement
  round), `fault` (SIGKILL/SIGSTOP injection), `frameworks` (PyTorch native
  and cudaMallocAsync, TensorFlow, DataLoader fork + spawn; slice and full
  GPU), `fork`, `multigpu`, `array`, `security` (env vars ignored, symlinked
  and user-owned configs rejected; root side needs `SOFTMIG_TEST_SUDO=1`),
  `container` and `overhead` (informational). New probes: `stress_alloc`,
  `shrreg_check`, `fork_probe`, `multigpu_probe`, `bench_overhead`,
  `test/python/fw_limit.py`.

### Findings (documented, not changed)

- Apptainer: the library is not loaded inside containers (their own
  `/etc/ld.so.preload`), so slices are not enforced there. Preloading it
  into the container does not help: the user namespace shows the root-owned
  config as uid 65534, so it is rejected and the library stays passive,
  which is the safe outcome. Enforcement in containers needs site policy.
- Slurm: a mixed request `--gres=gpu:l40s.2:1,gpu:l40s.4:1` is accepted,
  allocates one shard (`AllocTRES gres/shard=1`) but the prolog writes a
  half-GPU limit, so the job gets twice the memory it was allocated.
  Identical-size multi-slice requests are rejected by job_submit.
- Overhead (`bench_overhead`): passive adds ~3 ns per launch; a slice adds
  ~0.2 us per launch and ~44 us per alloc+free.

---

## 2026-10-01 — branch `2.05`

### Off means off: one passive decision, enforced at the choke points

The library is preloaded into every process via `/etc/ld.so.preload`, so it
has to decide by itself whether to do anything. 2.05 makes that decision once
per process (`softmig_mode_init()`, `pthread_once`) and, when passive, gets
out of the way structurally instead of per hook:

- **Rule.** In a Slurm job: active only if `/var/run/softmig/<jobid>[_<array>].conf`
  is a root-owned regular file that sets `CUDA_DEVICE_MEMORY_LIMIT` or
  `CUDA_DEVICE_SM_LIMIT` (environment variables are ignored). Outside Slurm:
  active only if those environment variables are set. The config is opened
  with `O_NOFOLLOW` and checked with `fstat` (no symlink/TOCTOU games).
- **`dlsym`** returns the driver's own symbol for every `cu*`/`nvml*` lookup.
- **`cuGetProcAddress` / `_v2`** forward to the driver untouched.
- **`nvmlInit*`** forward without creating the shared region or watcher.
- **Direct-linked callers** hit exported wrappers, each of which starts with
  `SOFTMIG_PASSIVE_FORWARD` (or `SOFTMIG_MEM_GUARD`). A build-time lint,
  `src/check_passive_guards.sh`, fails the build if any exported wrapper that
  reaches SoftMig internals lacks the guard, so a new hook cannot silently
  reintroduce the Bowling `cudaFreeAsync ... UNKNOWN ERROR (-1)` class of bug.
- Passive processes create no `/tmp/cudevshr.cache.*`, start no watcher
  thread, and install no `SIGUSR1`/`SIGUSR2` handlers. Passive mode is full
  pass-through, including NVML (full-GPU jobs are exclusive, so there is
  nothing to filter).

### `cuGetProcAddress` substitutes by pointer identity

Enabled mode used to guess `_v2`/`_v3` names from `cudaVersion` and hand out
our hook for the guess, which could return a hook with the wrong ABI. Now the
real `cuGetProcAddress` runs first and our hook is substituted only if the
returned pointer is exactly a driver function we hook (resolved by exact
name). Per-thread-default-stream lookups (`flags=2`, `*_ptsz`) are covered by
new `_ptsz` wrappers that map stream 0 to `CU_STREAM_PER_THREAD`.

### New hooks (driver 595 / CUDA 13 audit)

- `cuLaunchKernelEx` (CUDA 12+ launch path used by cudart and Triton) and
  `_ptsz` variants of `cuLaunchKernel`, `cuLaunchKernelEx`,
  `cuLaunchCooperativeKernel`, `cuGraphLaunch`, `cuMemAllocAsync`,
  `cuMemFreeAsync`, `cuMemAllocFromPoolAsync`. `cuLaunchCooperativeKernel` is
  now SM-rate-limited like `cuLaunchKernel`.
- `cuMemFreeAsync` was missing from the `dlsym` hook list, so frees through
  cudart were never untracked in enabled mode (tracked usage only grew).
- NVML: `nvmlDeviceGetRunningProcessDetailList`,
  `nvmlDeviceGetProcessesUtilizationInfo` and
  `nvmlDeviceGetMPSComputeRunningProcesses_v2`/`_v3` are filtered to the job's
  processes like the other process lists.
- Any watched entry point (memory, launch, meminfo, NVML process queries) that
  still resolves to the raw driver is logged once as `UNHOOKED` in the job log.
  `test/audit_hooks.sh` does the same check statically against the installed
  driver (driver 595.91.07: 0 unhooked).

### Shared-region lock: no more self-steal and spin

`owner_pid` is per process, so a thread waiting on the region lock while a
sibling thread (e.g. the utilization watcher) held it saw `owner == self`,
declared the lock stale and took it, letting two threads rewrite the process
table at once. That could leave a slot with `pid == 0`, and
`clear_proc_slot_nolock` then looped forever on it with the lock held; every
other process on the GPU stalled behind it (seen as the `direct` suite hang:
`nvidia-smi` spinning in `nvmlInitWithFlags`). Threads of one process now
serialize on a local mutex before the cross-process semaphore (re-entrant per
thread, reset in the fork child), and empty slots are dropped like dead
processes.

### Other fixes

- `cuDeviceTotalMem_v2` reports the real total, capped at the limit (it
  returned the limit, i.e. 0, in passive mode).
- `cuMemGetInfo*` keys limit and usage by the CUDA device index and reports
  `free = 0` when over the limit instead of `CUDA_ERROR_INVALID_VALUE`.
- `cuMemFree` of an async/pool pointer (and vice versa) now untracks it from
  the list it was actually in; tracked usage no longer leaks upward.
  `remove_chunk*` untrack only after a successful driver free.
- Utilization watcher: started only when `0 < CUDA_DEVICE_SM_LIMIT < 100`; a
  failed `nvmlDeviceGetHandleByIndex` no longer returns with the region lock
  held.
- `SIGUSR1`/`SIGUSR2` handlers are installed only with the opt-in OOM killer.
- "Illegal device id" paths return 0 instead of reading past the arrays.
- NVML process-list hooks follow NVML's sizing contract on the filtered list;
  `nvmlDeviceGetProcessUtilization` is filtered in place.
- Removed the `CUDA_REDIRECT` / `vgpulib` dlopen path and
  `multi_func_hook.h`; the version script exports only `cu*`/`nvml*` hooks.
- Branch names with dots (e.g. `2.05`) no longer break the build.

### Tests

- `test/passive_probe.c` + `test/suite_passive.sh`: in a full-GPU job every
  `dlsym`/`cuGetProcAddress` result must equal the driver's, async alloc/free
  through the per-thread stream must work, no signal handlers or shared region
  may appear; in a slice job the same probe must see hooks and a region.
- `suite_smoke` / `suite_direct` gained passive-mode criteria; both fail on any
  `UNHOOKED` line. `passive` is in the per-version and full-GPU matrix.
- `test/audit_hooks.sh`: static driver-vs-hooks audit.

---

## 2026-09-09

### Passive mode is now true pass-through for all memory hooks

Full-GPU jobs (no `gres/shard` -> no config file) previously still ran
SoftMig's interposition logic on every hooked memory call: allocations were
tracked in the `device_overallocated` / `device_allocasync` lists, untracked
frees returned `-1`, `cuMemAllocAsync` issued extra pool-attribute queries,
and `cuMemGetInfo` reported `free = total - cgroup_usage` instead of the
driver's value. This broke JAX/XLA jobs using
`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` (array jobs `828986_*`, 42/42 tasks
failed with `cudaFreeAsync failed ... UNKNOWN ERROR (-1)`, leaking
512 MiB at a time until the card was exhausted).

Every memory hook now checks `softmig_passthrough()` (limit == 0) first and
forwards straight to the real driver call with no tracking, OOM checks, or
usage accounting: `cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemAllocAsync`,
`cuMemFreeAsync`, `cuMemAllocFromPoolAsync`, `cuMemAllocManaged`,
`cuMemAllocPitch_v2`, `cuMemCreate`, `cuMemRelease`, `cuMemAllocHost_v2`,
`cuMemHostAlloc`, `cuMemHostRegister_v2`, `cuArrayCreate_v2`,
`cuArray3DCreate_v2`, `cuMipmappedArrayCreate`, and `cuMemGetInfo` /
`cuMemGetInfo_v2` (which now return the real driver values for passive jobs).
`cuLaunchKernel` throttling and NVML process-list filtering are unchanged —
cgroup isolation is desired for all jobs.

### Untracked frees now fall back to the real driver free

`remove_chunk` and `remove_chunk_async` returned `-1` for any pointer not in
the tracked list without calling the real driver free. This is what broke
pool allocations in enabled mode too (a pool pointer freed with `cuMemFree`,
or any allocation SoftMig failed to track, leaked forever). Both now forward
untracked frees to the real `cuMemFree_v2` / `cuMemFreeAsync` and return the
driver's result, so behavior is native-faithful: double-frees and bad
pointers return real driver errors instead of `-1`. `remove_chunk` also
propagates the real free's result on the tracked path instead of always
returning 0. Fallback paths log at DEBUG level.

### Proper OOM error code; post-alloc attribute failures no longer abandon allocations

- `add_chunk_async` and `add_chunk_only` now return `CUDA_ERROR_OUT_OF_MEMORY`
  instead of `-1` when the limit check fails (XLA/JAX retry logic depends on
  the proper code).
- In `add_chunk_async`, if `cuDeviceGetMemPool` or `cuMemPoolGetAttribute`
  fail after a successful real allocation, the failure is now non-fatal: the
  allocation is still tracked with the requested size instead of being left
  live and untracked. A `RESERVED_MEM_HIGH` of 0 is handled the same way, and
  the `device_allocasync->limit` counter is kept balanced with the tracked
  entry lengths on all paths.

### Pool allocations are now limit-enforced in enabled mode

`cuMemAllocFromPoolAsync` was a pure pass-through that never tracked its
allocations, so `gres/shard` jobs using the `cuda_async` allocator got no
memory enforcement at all. It now runs `oom_check(dev, bytesize)` (returns
`CUDA_ERROR_OUT_OF_MEMORY` when over the limit), calls the real function, and
records successful allocations in `device_allocasync` via the new
`add_chunk_async_only()` helper (requested-size tracking, mirroring
`add_chunk_only`; no `RESERVED_MEM_HIGH` slab-delta accounting —
`oom_check_nolock` already takes `max(tracked, NVML)`).

### Test harness

- New regression binary `test/test_pool_free.c` (exit 0 = PASS, 2 = bug
  reproduced, 1 = setup error) covering the tracked async path, the XLA
  75%-pool-prealloc pattern, a 20x alloc/free loop, sync free of a pool
  pointer, passive `cuMemGetInfo` vs `nvidia-smi`, and over-limit pool
  allocation returning `CUDA_ERROR_OUT_OF_MEMORY`.
- New suites: `test/suite_pool.sh` (+ `suite_pool_inner.sh`) and
  `test/jax_cuda_async.sh` (JAX end-to-end under `cuda_async`).
- `test/run_matrix.sh` runs the new `pool` suite and adds the full-GPU
  passive slice (`l40s`) for the `smoke`, `direct`, and `pool` suites.
- Removed the `SOFTMIG_ROOT=/scratch/rahimk/SoftMig` hardcode from
  `suite_common.sh`, `run_matrix.sh`, `run_oom_validation.sh`, and
  `sbatch_oomval.sbatch` (derived from each script's own location now);
  the lib hash lines in `run_multiproc.sh` / `run_oom_validation.sh` read
  the installed library path from `/etc/ld.so.preload` instead of assuming
  `/usr/local/lib`; fixed stale rack01-12 comments (reservation node is
  rack01-11); `suite_smoke.sh` is passive-mode aware.
- `suite_sm.sh` skips nvidia-smi samples from the first 12s (100% warmup
  spike before the watcher settles) so the l40s.2 50% target is not
  failed by startup.
- `suite_crossjob.sh` occupies 3/4 L40S with a full-GPU blocker so the
  two 1/4 shard jobs share the remaining card (SLURM otherwise spreads
  shard jobs across GPUs; `CUDA_VISIBLE_DEVICES=0` is per-job remapped
  and cannot pin them). PASS requires the same UUID and no cross-PID
  leak in hooked nvidia-smi / `Found current process`.
- Hook `nvmlDeviceGetComputeRunningProcesses_v3` and
  `nvmlDeviceGetGraphicsRunningProcesses_v3`. Driver 595 `nvidia-smi`
  dlsyms `_v3`, so the `_v2` filter never ran and two jobs on one GPU
  could see each other's PIDs.

---

## 2026-05-18

### Fix cuMemGetInfo to use real-time memory tracking

`cuMemGetInfo` and `cuMemGetInfo_v2` were only using NVML-reported usage, which
lags behind actual allocations. This caused the functions to report more free
memory than was actually available during rapid allocation sequences.

Added `get_current_usage_for_meminfo()` helper that returns `max(tracked_usage,
nvml_usage)` — the same pattern used in `oom_check_nolock`. This ensures fast
allocations are immediately reflected in memory queries while retaining
cross-process visibility via NVML.

---

## 2026-05-17

### Fix latent cuMemcpy2D / cuMemcpy2DUnaligned dispatch-table swap

`src/include/libcuda_hook.h` declared the enum order as
`cuMemcpy2DUnaligned_v2` then `cuMemcpy2D_v2`, but `src/cuda/hook.c` listed the
matching `cuda_library_entry` slots in the opposite order. The result was that
calls to `cuMemcpy2D_v2` would resolve through the dispatch table to the
`cuMemcpy2DUnaligned_v2` symbol in libcuda, and vice versa.

Swapped the two entries in `src/cuda/hook.c` to match the enum order. Also
fixed `cuMemcpy2D_v2`'s wrapper in `src/cuda/memory.c` to call
`CUDA_OVERRIDE_CALL(..., cuMemcpy2D_v2, ...)` explicitly instead of relying on
CUDA's `#define cuMemcpy2D cuMemcpy2D_v2` macro, for consistency with the
other `cuMemcpy*_v2` wrappers and so that no CUDA-header version rename
silently re-breaks this dispatch.

### Test cleanup — remove unused `t_size` locals

Removed unused `t_size` variables in `test/test_alloc.c`, `test/test_alloc_hold.c`,
and `test/test_create_array.c`. No behavior change, just silences `-Wunused`.

### CUDA 13 compatibility — drop pass-through hooks with conflicting signatures

CUDA 13 renames several driver entry points to new `_v2/_v4` versions whose
`cuda.h` macros redefine the old names to signatures incompatible with our
existing wrappers (e.g. `cuCtxCreate` → `cuCtxCreate_v4(4 args)`,
`cuGraphGetEdges` → `cuGraphGetEdges_v2(5 args)`,
`cuMemAdvise` → `cuMemAdvise_v2`,
`cuMemPrefetchAsync` → `cuMemPrefetchAsync_v2`,
`cuDeviceGetUuid` → `cuDeviceGetUuid_v2`,
and several `cuGraph*Dependencies*` variants).

All of these were **pure pass-through wrappers** with no SoftMig-specific
logic. Hooking them was never necessary — `dlsym` falls through to libcuda
when we don't override. Removed the wrappers in `src/cuda/context.c`,
`src/cuda/memory.c`, `src/cuda/device.c`, `src/cuda/graph.c` and the matching
entries in `src/cuda/hook.c`, `src/include/libcuda_hook.h`, and
`src/libsoftmig.c`. Only `cuGraphLaunch` remains hooked from `graph.c`
(needed for SM rate-limiting).

Updated `test/test_alloc*.c` and `test/test_create_*.c` to call
`cuCtxCreate_v2` explicitly so the legacy 3-arg signature compiles on
both CUDA 12 and CUDA 13 (where bare `cuCtxCreate` is `cuCtxCreate_v4`).

Verified: clean build against both `cuda/12.6` and `cuda/13.2`. No runtime
behavior change on CUDA 12 — the removed wrappers were already no-ops.

### Restore native CUDA OOM behavior and full memory budget

**Changed: removed the per-process 5% overhead and 9 MB floor in
`sum_process_memory_from_nvml()`** (`src/nvml/hook.c`). Memory usage is now
the raw NVML `usedGpuMemory` summed across processes in the current
cgroup/UID. The previous inflation was a workaround for older NVML
under-reporting; on CUDA 12+ it just shaved ~5% off the configured slice
limit and added a fixed 9 MB tax per process. Comments in `allocator.c`,
`cuda/memory.c`, and `multiprocess_memory_limit.{c,h}` updated to match.

**Changed: in-library OOM killer disabled by default.** Previously
`enable_active_oom_killer` was hardcoded on, which meant `cuMemAlloc()`
calls over the limit triggered `active_oom_killer()` and `SIGKILL`-ed every
process in the calling cgroup/UID — including the caller. The natural CUDA
path already returns `CUDA_ERROR_OUT_OF_MEMORY` from `add_chunk()` when the
per-job limit would be exceeded, so on a SoftMig-sliced GPU `cudaMalloc`
now behaves like a real GPU: returns `cudaErrorMemoryAllocation` and the
process keeps running. The background utilization watcher still logs OOM
events to syslog but no longer invokes `gradual_oom_killer()` by default.

**New: `SOFTMIG_ENABLE_OOM_KILLER` opt-in.** Set the env var to `1`/`true`
(or add `SOFTMIG_ENABLE_OOM_KILLER=1` to the per-job config file
`/var/run/softmig/{jobid}.conf`) to restore the legacy active + gradual
kill defense. Implemented as `get_softmig_oom_killer_enabled()` in
`src/multiprocess/config_file.c`.

---

## 2026-05-08

### README — Add demo images, badges, hero line, and section icons

Added `softmig_pic1.png` and `softmig_pic2.png` showing SoftMig running on the
University of Alberta Vulcan cluster: `nvidia-smi` output for a 1/2 L40S slice
(~24 GB visible) and a 1/4 L40S slice (~12 GB visible). Images displayed
side-by-side after the Description section with captions.

Added CUDA 12+, CUDA 13, and NVIDIA GPU compatibility badges. Added hero
tagline ("Software MIG for any NVIDIA GPU — no hardware MIG required.") and
"Deployed on the University of Alberta Vulcan cluster (operated for AMII)" line.
Updated GPU compatibility badge from specific models to series names (L | A | V | RTX).
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
