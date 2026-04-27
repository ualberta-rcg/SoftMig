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
| `$SLURM_TMPDIR/cudevshr.cache.*` | per-job shared state for memory tracking |
| `$SLURM_TMPDIR/vgpulock/` | per-job lock files |
| `/var/log/softmig/{jobid}.log` | per-job logs (admin-visible) |

## Config source of truth

- **Inside SLURM jobs**, limits are taken from `/var/run/softmig/*.conf`.
- When `SLURM_JOB_ID` is set, SoftMig does **not** fall back to user environment variables (even if no config file exists).
- Environment variables are intended for **local testing only** (outside SLURM).

## Prolog responsibilities

The prolog should:

- detect the requested slice/shard count (site-specific)
- compute `CUDA_DEVICE_MEMORY_LIMIT` and `CUDA_DEVICE_SM_LIMIT`
- write config file(s) as root:
  - `/var/run/softmig/${SLURM_JOB_ID}.conf`
  - and/or `/var/run/softmig/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.conf` for array jobs
See: `docs/examples/prolog_softmig.sh`.

## Epilog responsibilities

The epilog should remove config files for the job, including array-job variants.

See: `docs/examples/epilog_softmig.sh`.

## Optional: job_submit.lua

If used, the submit plugin commonly:

- validates that slice requests are sane (e.g., no `denominator=1` “slice”)
- prevents invalid multiple-slice counts
- translates `gpu:type.4:1` style syntax into an internal shard format

See: `docs/examples/job_submit_softmig.lua`.

## Optional: `nvidia-smi` wrapper

`nvidia-smi` will normally show processes from all jobs sharing a GPU. A wrapper can filter output by cgroup so users see only their job’s processes.

Repo script: `nvidia-smi-hook.sh` (optional).

