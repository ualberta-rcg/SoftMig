# SoftMig SLURM Integration (Admins)

This document describes how SoftMig is typically integrated into SLURM.

## Components

- **`libsoftmig.so`**: installed on compute nodes and loaded via `/etc/ld.so.preload` (recommended for enforcement).
- **Prolog**: creates a per-job config under `/var/run/softmig/`.
- **Epilog**: removes per-job configs at job end.
- **Optional `job_submit.lua`**: validates slice syntax and translates user-facing slice requests to a scheduler-internal form (e.g. `gres/shard:*`).
- **Optional `nvidia-smi` wrapper**: hides other jobs' GPU processes by cgroup filtering.

Example scripts are provided in `docs/examples/`:

- `docs/examples/prolog_softmig.sh`
- `docs/examples/epilog_softmig.sh`
- `docs/examples/job_submit_softmig.lua`

## File locations (runtime)

| Path | Purpose |
|---|---|
| `/etc/ld.so.preload` | system-wide load of `libsoftmig.so` |
| `/var/run/softmig/{jobid}.conf` | per-job limit config (root-owned) |
| `/var/run/softmig/{jobid}_{arrayid}.conf` | per-array-task config (root-owned) |
| `$SLURM_TMPDIR/cudevshr.cache.{jobid}[.{arrayid}]` | per-job shared memory region for memory tracking |
| `$SLURM_TMPDIR/vgpulock/lock.{jobid}` | per-job serialization lock file |
| `/var/log/softmig/{jobid}.log` or `{jobid}_{arrayid}.log` | per-job logs (admin-visible) |

Notes:

- If `SLURM_TMPDIR` is not set, cache and lock files fall back to `/tmp`.
- Lock files are keyed by job ID only (array tasks within the same job share one lock).
- Log files include the array task ID when `SLURM_ARRAY_TASK_ID` is set. If `/var/log/softmig/` is not writable, logs fall back to `$SLURM_TMPDIR/softmig_{jobid}.log`.

## Config source of truth

- **Inside SLURM jobs**, limits are taken from `/var/run/softmig/*.conf`.
- When `SLURM_JOB_ID` is set, SoftMig does **not** fall back to user environment variables (even if no config file exists).
- Environment variables are intended for **local testing only** (outside SLURM).

## SLURM shard configuration

SoftMig uses SLURM shards as the scheduler-visible representation of GPU slices. A shard is a configurable, even partition of a GPU. For example, on a 48 GB GPU, 2 slices per GPU gives each slice 24 GB, while 4 slices per GPU gives each slice 12 GB.

The number of slices per GPU is site-defined. Set it with `NUM_SHARDS` in `docs/examples/job_submit_softmig.lua`, and keep the matching value in `NUM_SHARDS_PER_GPU` in `docs/examples/prolog_softmig.sh`. The submit plugin uses this value to translate user-facing fractional GPU requests into SLURM shard counts:

```text
shard_count = (requested_count / requested_denominator) * NUM_SHARDS
```

For example, if `NUM_SHARDS=8`, users can request `<gpu_type>.8:1` for 1/8 of a GPU, `<gpu_type>.4:1` for 1/4 of a GPU, or `<gpu_type>.2:1` for 1/2 of a GPU. The example submit plugin currently allows requests up to half of a GPU at a time: `<gpu_type>.1:1` is rejected because it represents a full GPU, and denominators larger than `NUM_SHARDS` are rejected because they are smaller than the configured slice size.

Enable shards for your GPUs before installing the prolog, epilog, and optional submit plugin.

In `slurm.conf`:

- add `shard` to `GresTypes`
- add `gres/shard` to `PriorityWeightTRES` for proper accounting if you use `PriorityType=priority/multifactor`
- add `gres/shard` and `gres/shard:<gpu_type>` to `AccountingStorageTRES`
- set `JobSubmitPlugins=lua` if you use `docs/examples/job_submit_softmig.lua`

In `gres.conf`, expose enough shards per node for the slice sizes you want to support:

```conf
# shard_count = shards per GPU * GPUs per node
NodeName=<nodelist> Name=shard Count=<shard_count>
```

For example, 4 shards per GPU on a 4-GPU node requires `Count=16`.

Keep this per-GPU shard count aligned with `NUM_SHARDS` in `docs/examples/job_submit_softmig.lua` and `NUM_SHARDS_PER_GPU` in `docs/examples/prolog_softmig.sh`.

## Prolog responsibilities

The prolog translates the SLURM shard request into the SoftMig runtime limits for the job. It should:

- detect the requested slice/shard count (site-specific)
- compute `CUDA_DEVICE_MEMORY_LIMIT` and `CUDA_DEVICE_SM_LIMIT`
- write config file(s) as root (must be owned by uid 0; SoftMig rejects symlinks and non-root-owned files):
  - `/var/run/softmig/${SLURM_JOB_ID}.conf`
  - and/or `/var/run/softmig/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.conf` for array jobs
See: `docs/examples/prolog_softmig.sh`.

## Epilog responsibilities

The epilog should remove config files and other per-job state created by the prolog, including array-job variants.

See: `docs/examples/epilog_softmig.sh`.

## Optional: job_submit.lua

If used, add the logic from `docs/examples/job_submit_softmig.lua` to your site `job_submit.lua`. The submit plugin commonly:

- validates that slice requests are sane (e.g., no `denominator=1` “slice”)
- prevents invalid multiple-slice counts
- translates user-facing slice requests such as `gpu:l40s.4:1` or `l40s.4:1` into scheduler-internal shard requests such as `gres/shard:l40s:1`

See: `docs/examples/job_submit_softmig.lua`.

## Optional: `nvidia-smi` wrapper

`nvidia-smi` will normally show processes from all jobs sharing a GPU. A wrapper can filter output by cgroup so users see only their job’s processes.

Repo script: `nvidia-smi-hook.sh` (optional).
