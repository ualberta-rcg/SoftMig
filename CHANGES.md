# SoftMig Change Log

This file tracks major architecture and behavior changes from HAMi-core to
SoftMig, plus recent release-level updates.

For deployment and usage instructions, see `README.md`.
For current open operational work, see `docs/PROJECT_STATUS.md`.

## Current State (2026-04-27)

SoftMig is operating as a SLURM-first software GPU slicing layer with:

- secure per-job config files in `/var/run/softmig`
- per-job cache and lock isolation in `SLURM_TMPDIR`
- NVML/CUDA interposition for memory and process filtering
- optional `nvidia-smi` wrapper for per-cgroup process visibility
- CUDA 12+ support (CUDA 11 unsupported)

## Major Milestones

### Foundational SoftMig Fork (from HAMi-core)

- Renamed project/library to SoftMig (`libsoftmig.so`)
- Switched to file-based logging under `/var/log/softmig`
- Added secure config-file-first model for SLURM jobs
- Added passive mode when no config/environment is present
- Added SLURM prolog/epilog integration examples
- Standardized cache/lock paths to `SLURM_TMPDIR` for job isolation

### Runtime and Hook Reliability Enhancements

- Improved `dlsym` resolution fallback behavior across environments
- Added NVML cache/refresh support to reduce repeated heavy queries
- Hardened process filtering logic for cgroup/UID matching
- Tightened lock recovery and reduced noisy hot-path logging
- Expanded direct-linked and runtime-linked NVML validation coverage

### Test and Ops Tooling Expansion

- Added matrix/overnight/whack test harness scripts under `test/`
- Added lightweight probe binaries (`runtime_hold`, `nvml_probe`, `gpu_burn_lite`)
- Added ops scripts for log rotation and trimming under `ops/`

## Latest Update (2026-04-27)

### Fixed: array-job config filename mismatch in limit loading

Problem:

- In some environments, prolog writes `/var/run/softmig/{jobid}.conf`
- Array task env inside the job causes lookup of `{jobid}_{arrayid}.conf`
- Missing array-suffixed file made SoftMig treat limits as unset

Resolution:

- `src/multiprocess/config_file.c` now falls back from
  `{jobid}_{arrayid}.conf` to `{jobid}.conf` when needed
- `docs/examples/prolog_softmig.sh` updated to derive array task ID from
  `scontrol --json` where prolog env vars may be incomplete

Impact:

- Batch array jobs no longer lose memory/SM limits due to filename mismatch
- Behavior is now robust to prolog environment differences

