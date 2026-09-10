# SoftMig Troubleshooting

This document is aimed at cluster admins/operators diagnosing common SoftMig problems.

## Quick checks

Inside a running SLURM job:

- **Config exists**: `/var/run/softmig/` contains `{jobid}.conf` (or `{jobid}_{arrayid}.conf` for array tasks)
- **Library is loaded**: `/etc/ld.so.preload` includes the installed `libsoftmig.so` path
- **Logs exist**: `/var/log/softmig/{jobid}.log` (or `$SLURM_TMPDIR/softmig_{jobid}.log` fallback)

## Symptom: `cudaFreeAsync failed ... UNKNOWN ERROR (-1)` and jobs leaking VRAM until `RESOURCE_EXHAUSTED`

Signature: JAX/XLA jobs using `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` (or any
app allocating from a memory pool via `cuMemAllocFromPoolAsync`) die in ~40 s
with hundreds of `cudaFreeAsync failed to free ...: UNKNOWN ERROR (-1)` in
stderr, with `nvidia-smi` showing free memory dropping steadily (512 MiB at a
time) until the card is exhausted. Affects **full-GPU jobs too** — the
incident jobs (`828986_*`, `condense-sweep`) were passive-mode jobs.

Root cause (fixed 2026-09-09): `cuMemAllocFromPoolAsync` never recorded its
allocations in the tracked list, and the `cuMemFreeAsync` hook returned `-1`
for any untracked pointer **without calling the real driver free** — so every
pool free failed and nothing was ever released. Related issues fixed at the
same time: passive mode is now true pass-through for all memory hooks
(including `cuMemGetInfo`, which used to report `free = total - cgroup_usage`
instead of the driver value), untracked frees now fall back to the real
driver free, OOM rejections return `CUDA_ERROR_OUT_OF_MEMORY` instead of
`-1`, and pool allocations are limit-enforced in enabled mode.

Checks on an affected node:

- `grep -c 'res=-1' /var/log/softmig/*.log` — free failures log as
  `after free_raw_async ... res=-1` at DEBUG level
- confirm the installed library predates the 2026-09-09 fix:
  `md5sum $(head -1 /etc/ld.so.preload)`
- reproduce with `build/test/test_pool_free` (exit 2 = bug present,
  exit 0 = fixed); see `test/suite_pool.sh`

## Symptom: `nvidia-smi` shows full VRAM in a sliced job

Most common causes:

- prolog did not create the config file
- config file naming mismatch for array jobs
- job is actually a full-GPU request (no limits intended)

Checks:

- `ls -l /var/run/softmig/` on the compute node during the job
- confirm the job requested a slice GRES that your prolog recognizes
- for array jobs, confirm `{jobid}_{arrayid}.conf` exists (or that your deployment supports fallback)

## Symptom: `nvidia-smi` shows processes from other users

SoftMig can enforce limits without changing `nvidia-smi` process visibility. To filter process lists by job cgroup, deploy the optional wrapper:

- script: `nvidia-smi-hook.sh`
- integration notes: `docs/SLURM_INTEGRATION.md`

## Symptom: stale config files under `/var/run/softmig/`

Confirm epilog cleanup:

- epilog should remove both `/var/run/softmig/{jobid}.conf` and `/var/run/softmig/{jobid}_*.conf` (array jobs)
- see example: `docs/examples/epilog_softmig.sh`

## Symptom: limits seem “sticky” after changing policy

SoftMig uses per-job state files under `$SLURM_TMPDIR`. When changing limits during testing, clear the per-job cache before re-testing:

```bash
rm -f ${SLURM_TMPDIR}/cudevshr.cache*
```

## Symptom: no logs / permission errors creating logs

SoftMig writes logs under `/var/log/softmig/` by default. The directory must allow users (job UIDs) to create files. If `/var/log/softmig/` is not writable **and** `SLURM_TMPDIR` is set, logs fall back to `$SLURM_TMPDIR/softmig_{jobid}.log`. If neither is writable, logging silently fails.

Common setups:

```bash
sudo chown root:slurm /var/log/softmig
sudo chmod 775 /var/log/softmig
```

Or:

```bash
sudo chown root:root /var/log/softmig
sudo chmod 1777 /var/log/softmig
```

See also: `docs/BUILD_AND_INSTALL.md`.

