# SoftMig

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

**Maintained by:** Rahim Khoja ([khoja1@ualberta.ca](mailto:khoja1@ualberta.ca)) and Kahim Ali ([kali2@ualberta.ca](mailto:kali2@ualberta.ca))

## Description

**SoftMig** is a SLURM-integrated software GPU slicing system for shared NVIDIA GPU clusters. It lets administrators schedule ordinary NVIDIA GPUs in a MIG-like way using software-enforced memory limits, compute time-slicing, and SLURM prolog/epilog automation.

SoftMig is based on [HAMi-core](https://github.com/Project-HAMi/HAMi-core), adapted for SLURM-based HPC environments.

SoftMig does **not** enable, modify, or replace NVIDIA Hardware MIG. Instead, it provides a scheduler-controlled software layer for fractional GPU jobs on GPUs that do not support hardware MIG, or where hardware MIG is too rigid for dynamic SLURM scheduling.

## Features (high level)

- **Memory slicing**: enforce per-job GPU memory ceilings (CUDA allocs return `CUDA_ERROR_OUT_OF_MEMORY` when exceeded)
- **Compute slicing**: kernel launch throttling / SM time-slicing via `CUDA_DEVICE_SM_LIMIT`
- **SLURM lifecycle**: per-job root-owned config files created/removed by prolog/epilog (`/var/run/softmig/`)
- **Per-job isolation**: cache/locks under `$SLURM_TMPDIR` (avoids shared `/tmp` collisions)
- **Silent by default**: per-job logs under `/var/log/softmig/{jobid}.log`
- **Optional `nvidia-smi` filtering**: wrapper to hide other jobs' GPU processes
- **CUDA 12+**: designed for CUDA 12+ environments (CUDA 11 unsupported)

## Quickstart

### SLURM users (example)

Request a slice via GRES (exact names depend on site policy):

```bash
# Quarter GPU example (12GB/25% on a 48GB GPU, with a 4-shard policy)
sbatch --gres=gpu:l40s.4:1 --time=5:00:00 job.sh

# Half GPU example (24GB/50% on a 48GB GPU)
sbatch --gres=gpu:l40s.2:1 --time=2:00:00 job.sh

# Full GPU
sbatch --gres=gpu:l40s:1 --time=2:00:00 job.sh
```

See: `docs/USAGE.md`.

### Cluster admins

- **Build/install/update**: `docs/BUILD_AND_INSTALL.md`
- **SLURM integration** (prolog/epilog/job_submit examples): `docs/SLURM_INTEGRATION.md`

## Repo documents

- `docs/USAGE.md`: user-facing usage and slice model
- `docs/SLURM_INTEGRATION.md`: prolog/epilog/job_submit integration notes
- `docs/TROUBLESHOOTING.md`: common symptoms and checks
- `docs/TESTING.md`: smoke tests and framework tests
- `docs/BUILD_AND_INSTALL.md`: build/install/update runbook (includes safe `unshare -m` updates)
- `CHANGES.md`: release-level architecture/behavior changes
- `docs/PROJECT_STATUS.md`: current operational status
- `docs/FIXES_TO_APPLY.md`: actionable checklist

## Links

- **External docs (ragflow-hosted)**: `https://ualberta-rcg.github.io/ragflow-wiki-data/`
- **Issues**: `https://github.com/ualberta-rcg/SoftMig/issues`

## Support

Open an issue or email:

- `khoja1@ualberta.ca`
- `kali2@ualberta.ca`
