# SoftMig Usage (Users)

This document describes how **users** consume SoftMig once it is deployed on a SLURM cluster.

## What users do (high level)

- Users **request a GPU slice** via SLURM (site-defined GRES syntax).
- SLURM policy (typically a prolog + optional `job_submit.lua`) translates the request into per-job limits.
- SoftMig enforces limits inside the job via a **root-owned config file** under `/var/run/softmig/`.

## Slice requests (examples)

Exact GRES names depend on cluster policy; one common pattern is:

```bash
# Quarter GPU on a 48GB L40S (site policy: 4 shards/GPU)
sbatch --gres=gpu:l40s.4:1 --time=5:00:00 job.sh

# Half GPU
sbatch --gres=gpu:l40s.2:1 --time=2:00:00 job.sh

# Full GPU (no slice limits)
sbatch --gres=gpu:l40s:1 --time=2:00:00 job.sh
```

## Configuration model (important)

SoftMig has three operating modes:

1. **SLURM jobs**: limits come from root-owned config files under `/var/run/softmig/`.
2. **Local testing (outside SLURM)**: limits can be set using environment variables.
3. **Passive mode**: if no config (and no env vars outside SLURM) exists, the library loads but does nothing.

In production SLURM jobs, users generally **do not set limits**. Limits are assigned by the scheduler and written by the SLURM prolog.

When `SLURM_JOB_ID` is set, SoftMig treats the root-owned config file as the source of truth and does **not** fall back to user environment variables.

### Config file format

Config files contain key/value pairs (one per line), e.g.:

```bash
CUDA_DEVICE_MEMORY_LIMIT=12288M
CUDA_DEVICE_SM_LIMIT=25
```

## Default slice layout (example policy)

Many sites start with **4 shards per GPU**:

| User-facing slice | Shards | Memory on 48GB GPU | SM limit | Oversubscription |
|---|---:|---:|---:|---:|
| `l40s` | N/A | 48GB | 100% | 1x |
| `l40s.2` | 2 | 24GB | 50% | 2x |
| `l40s.4` | 1 | 12GB | 25% | 4x |

To offer 1/8 slices, a site typically moves to **8 shards per GPU** (requires updating policy in the prolog and Lua submit plugin).

## Local testing (outside SLURM)

For local/dev testing you can set limits with environment variables and load the library with `LD_PRELOAD`:

```bash
export CUDA_DEVICE_MEMORY_LIMIT=16g
export CUDA_DEVICE_SM_LIMIT=50
export LD_PRELOAD=/path/to/libsoftmig.so

nvidia-smi
```

Notes:

- Size suffixes are **single-character**: `G/g` (GiB), `M/m` (MiB), `K/k` (KiB). No suffix means bytes.
- For SLURM deployment details see: `docs/SLURM_INTEGRATION.md` and `docs/BUILD_AND_INSTALL.md`.

## Environment variable reference

| Variable | Purpose | Notes |
|---|---|---|
| `CUDA_DEVICE_MEMORY_LIMIT` | GPU memory ceiling | Single-char suffix: `G`, `M`, `K`. Only used outside SLURM. |
| `CUDA_DEVICE_SM_LIMIT` | SM utilization % (0-100) | `0` = no limit. Only used outside SLURM. |
| `SOFTMIG_LOG_LEVEL` | Log verbosity | `0`=errors (default), `1`=+warnings, `2`=+debug, `3`=+info+console |
| `SOFTMIG_LOG_FILE` | Override log file path | Overrides the default `/var/log/softmig/{jobid}.log` |
| `SOFTMIG_LOCK_FILE` | Override lock file path | Overrides the default `$SLURM_TMPDIR/vgpulock/lock.{jobid}` |
| `CUDA_DEVICE_MEMORY_SHARED_CACHE` | Override shared cache path | Overrides the default `$SLURM_TMPDIR/cudevshr.cache.{jobid}` |

