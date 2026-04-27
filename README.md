# University of Alberta - SoftMig

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![MkDocs Site](https://img.shields.io/badge/Docs-Live-blue?style=flat-square)](https://ualberta-rcg.github.io/ragflow-wiki-data/)

**Maintained by:** Rahim Khoja ([khoja1@ualberta.ca](mailto:khoja1@ualberta.ca)) and Kahim Ali ([kali2@ualberta.ca](mailto:kali2@ualberta.ca))

---

## Description

**SoftMig** is a SLURM-integrated software GPU slicing system for shared NVIDIA GPU clusters. It lets administrators schedule ordinary NVIDIA GPUs in a MIG-like way using software-enforced memory limits, compute time-slicing, and SLURM prolog/epilog automation.

SoftMig is based on [HAMi-core](https://github.com/Project-HAMi/HAMi-core), adapted for SLURM-based HPC environments. The project was inspired by Tim Weiers from the University of Alberta Unix team, who suggested HAMi-core as a foundation for bringing software GPU slicing into research computing clusters.

SoftMig does **not** enable, modify, or replace NVIDIA Hardware MIG. Instead, it provides a scheduler-controlled software layer for fractional GPU jobs on GPUs that do not support hardware MIG, or where hardware MIG is too rigid for dynamic SLURM scheduling.

A 48GB GPU, for example, can be offered as a full GPU, two half-GPU slices, four quarter-GPU slices, or other site-defined layouts. Each job receives a configured memory limit and proportional access to GPU compute through time-slicing and SM throttling.

Unlike Hardware MIG, SoftMig slice layouts can be changed through SLURM policy without draining nodes, rebooting, or changing GPU MIG mode. This allows full-GPU and fractional GPU jobs to coexist across the same nodes using normal SLURM scheduling.

SoftMig is intended for [Digital Research Alliance of Canada](https://alliancecan.ca/) / Compute Canada-style research clusters where GPU utilization, scheduling flexibility, and broad NVIDIA GPU compatibility matter more than hardware-level isolation.

## Features

### GPU Memory Slicing

SoftMig intercepts CUDA memory allocation calls and enforces a per-job GPU memory ceiling. When a job exceeds its limit, CUDA returns `CUDA_ERROR_OUT_OF_MEMORY`, similar to running on a smaller physical GPU.

Memory usage is tracked across all processes in the same SLURM job using a file-backed shared memory region under `$SLURM_TMPDIR`, so MPI jobs, PyTorch DDP jobs, and other multi-process workloads share the same configured memory pool.

### GPU Compute Slicing

SoftMig limits GPU compute access using kernel launch throttling / SM time-slicing. Slice limits are expressed as percentages, such as `CUDA_DEVICE_SM_LIMIT=25` for a quarter-GPU job or `CUDA_DEVICE_SM_LIMIT=50` for a half-GPU job.

This prevents small fractional jobs from monopolizing the GPU and allows multiple jobs to make meaningful forward progress on the same device.

### Works on Any CUDA 12+ NVIDIA GPU

Hardware MIG is limited to supported datacenter GPUs such as A100, H100, and newer MIG-capable devices. SoftMig works on ordinary NVIDIA GPUs with CUDA 12+ support, including L40S, A40, A30, V100, RTX-class cards, and others.

If the NVIDIA driver loads and CUDA works, SoftMig can provide software slicing.

### SLURM-Native Lifecycle Management

SoftMig is managed through SLURM prolog and epilog scripts. The prolog creates a root-owned config file for each job under `/var/run/softmig/`, and the epilog removes it when the job ends.

Users do not set their own limits. Limits are assigned by SLURM policy based on the requested GRES slice.

### Dynamic Slice Layouts Without Node Downtime

Slice layouts are defined by the SLURM prolog and optional Lua job submit plugin. Changing from one layout to another, such as 4 slices per GPU to 8 slices per GPU, does not require enabling MIG mode, draining the node, rebooting, or reconfiguring the GPU.

Running jobs keep the limits they were assigned. New jobs pick up the new policy.

### Per-Job Isolation

SoftMig stores temporary state, cache files, and lock files under `$SLURM_TMPDIR`, which is job-specific and automatically cleaned up by SLURM.

This avoids shared `/tmp` conflicts, cross-job cache collisions, and manual cleanup issues when multiple users share the same GPU node.

### Scheduler-Enforced Configuration

Config files are created by SLURM as root and stored under `/var/run/softmig/`. Users can inspect their job configuration for debugging, but they cannot modify the limits assigned by the scheduler.

In production, SoftMig can be loaded through `/etc/ld.so.preload`, making enforcement system-wide and preventing users from bypassing it with their own environment.

### Normal SLURM Cleanup Behavior

SoftMig respects the SLURM job lifecycle. When a job exits or is killed, the associated GPU processes are cleaned up with the rest of the job processes, following normal SLURM and cgroup behavior.

### Silent by Default

SoftMig writes logs to `/var/log/softmig/{jobid}.log` and stays quiet during normal user jobs. Admins can inspect logs centrally, while users are not spammed with library output on stdout or stderr.

Debug verbosity can be increased with `SOFTMIG_LOG_LEVEL` when needed.

### Optional `nvidia-smi` Filtering

SoftMig includes an optional `nvidia-smi` wrapper that filters process output by SLURM job cgroup. When several jobs share a GPU, users can see their own GPU processes without seeing every other user's workload on the device.

### Framework Agnostic

SoftMig operates below ML frameworks at the CUDA driver/runtime level. PyTorch, TensorFlow, JAX, MXNet, and other CUDA applications are subject to the same limits without framework-specific integration.

## Project Documents

- `CHANGES.md`: release-level architecture and behavior changes
- `docs/PROJECT_STATUS.md`: current operational status and open follow-ups
- `docs/FIXES_TO_APPLY.md`: actionable checklist of remaining fixes
- `docs/BUILD_AND_INSTALL.md`: build, install, and safe `unshare -m` update guide

## SoftMig vs. NVIDIA Hardware MIG

| Feature | SoftMig | NVIDIA Hardware MIG |
|---------|---------|---------------------|
| **How it works** | Software memory limits + compute time-slicing | Hardware GPU partitioning |
| **Dynamic changes** | Changed through SLURM policy; no reboot or node drain | Requires MIG reconfiguration and usually node drain |
| **GPU support** | Any CUDA 12+ NVIDIA GPU | MIG-capable datacenter GPUs |
| **Isolation** | Software/job-level isolation | Hardware-level isolation |
| **Best fit** | Flexible research scheduling and oversubscription | Strong isolation and predictable hardware partitions |

SoftMig is not a replacement for Hardware MIG. It is a more flexible software scheduling layer for SLURM clusters where dynamic fractional GPU access matters more than hardware-level isolation.

## Differences from Original HAMi-core

SoftMig keeps the core idea of HAMi-core but adapts it for SLURM-based HPC clusters.

| Feature | Original HAMi-core | SoftMig |
|---------|-------------------|---------|
| **Temporary Files** | Uses `/tmp` shared across jobs/users | Uses `$SLURM_TMPDIR` for per-job isolation |
| **Cache Files** | `/tmp/cudevshr.cache` shared globally | `$SLURM_TMPDIR/cudevshr.cache.{jobid}` |
| **Lock Files** | `/tmp/vgpulock/` shared globally | `$SLURM_TMPDIR/vgpulock/` per job |
| **Configuration** | Environment variables | Root-owned config files under `/var/run/softmig/` |
| **Logging** | stderr, visible to users | File-only logs under `/var/log/softmig/` |
| **Library Loading** | `LD_PRELOAD`, user-controlled | `/etc/ld.so.preload`, admin-controlled |
| **SLURM Integration** | Manual setup | Prolog/epilog lifecycle management |
| **CUDA Support** | CUDA 11+ | CUDA 12+ |
| **Library Name** | `libvgpu.so` | `libsoftmig.so` |

## Building and Installation

Build, install, and live-update instructions are maintained in:

- `docs/BUILD_AND_INSTALL.md`

That guide includes initial installation, permissions, `/etc/ld.so.preload` setup, safe live library replacement with `unshare -m`, and post-install verification.

## Usage

### Configuration Model

SoftMig has three operating modes:

1. **SLURM jobs:** limits come from root-owned config files under `/var/run/softmig/`.
2. **Local testing:** limits can be set with environment variables outside SLURM.
3. **Passive mode:** if no config exists, the library loads but does nothing.

In production, users do not set their own limits. SLURM assigns limits through the prolog based on the requested GPU slice, and the epilog removes the config when the job ends.

Example config file:

```bash
CUDA_DEVICE_MEMORY_LIMIT=12288M
CUDA_DEVICE_SM_LIMIT=25
```

### Quick Test

```bash
# Delete cache when changing limits
rm -f ${SLURM_TMPDIR}/cudevshr.cache*

# Testing only; production uses config files and /etc/ld.so.preload
export CUDA_DEVICE_MEMORY_LIMIT=16g
export CUDA_DEVICE_SM_LIMIT=50
export LD_PRELOAD=/var/lib/shared/libsoftmig.so

nvidia-smi
```

### For SLURM Users

Once deployed, users request GPU slices through SLURM:

```bash
# Half GPU slice: 24GB, 50% SM on a 48GB GPU
sbatch --gres=gpu:l40s.2:1 --time=2:00:00 job.sh

# Quarter GPU slice: 12GB, 25% SM on a 48GB GPU
sbatch --gres=gpu:l40s.4:1 --time=5:00:00 job.sh

# Full GPU
sbatch --gres=gpu:l40s:1 --time=2:00:00 job.sh
```

If `job_submit_softmig.lua` is configured, SLURM can validate slice requests and translate user-facing GPU slice syntax into the internal `gres/shard` format.

## Default Slice Layout

Default configuration uses 4 shards per GPU.

| GPU Slice | Shard Count | Memory Limit on 48GB GPU | SM Limit | Oversubscription |
|-----------|-------------|--------------------------|----------|------------------|
| `l40s` | N/A | 48GB | 100% | 1x |
| `l40s.2` | 2 shards | 24GB | 50% | 2x |
| `l40s.4` | 1 shard | 12GB | 25% | 4x |

For 1/8 GPU slices, configure 8 shards per GPU in the prolog and Lua job submit plugin.

## File Locations

| Path | Purpose |
|------|---------|
| `$SLURM_TMPDIR/cudevshr.cache.{jobid}` | Per-job GPU memory tracking |
| `$SLURM_TMPDIR/vgpulock/` | Per-job lock files |
| `/var/run/softmig/{jobid}.conf` | Root-owned job limit configuration |
| `/var/log/softmig/{jobid}.log` | Admin-visible job log |
| `/etc/ld.so.preload` | Optional production preload enforcement |

Array jobs use `{jobid}_{arrayid}` where needed.

## Logging

SoftMig writes job logs to `/var/log/softmig/{jobid}.log` and is silent to users by default.

Recommended setup:

```bash
sudo chown root:slurm /var/log/softmig
sudo chmod 775 /var/log/softmig
```

Less restrictive setup:

```bash
sudo chown root:root /var/log/softmig
sudo chmod 1777 /var/log/softmig
```

Useful commands:

```bash
tail -f /var/log/softmig/*.log
grep -i error /var/log/softmig/*.log
```

Debug verbosity can be controlled with `SOFTMIG_LOG_LEVEL`.

## Deployment for Cluster Administrators

### Installation and Library Updates

Use:

- `docs/BUILD_AND_INSTALL.md`

### SLURM Configuration

Set SLURM prolog and epilog scripts:

```bash
Prolog=/etc/slurm/prolog.sh
Epilog=/etc/slurm/epilog.sh
```

SoftMig expects the prolog to create the job config file and the epilog to remove it. Example scripts are provided under `docs/examples/`.

Optional but recommended:

- `prolog_softmig.sh`: creates `/var/run/softmig/{jobid}.conf`
- `epilog_softmig.sh`: removes config files when jobs end
- `job_submit_softmig.lua`: validates slice syntax and maps fractional GPU requests to shard counts

### How It Works

SoftMig is activated per job by the SLURM prolog and deactivated by the epilog. In production, limits come from root-owned config files under `/var/run/softmig/`; environment variables are only used for local testing outside SLURM. When no config exists, the library remains in passive mode.

### Monitoring and Troubleshooting

```bash
# Confirm job configs are being created
ls -l /var/run/softmig/

# Confirm library preload
cat /etc/ld.so.preload

# View logs
tail -f /var/log/softmig/*.log

# Clear per-job cache when changing limits during testing
rm -f ${SLURM_TMPDIR}/cudevshr.cache*
```

Inside a limited job, `nvidia-smi` should show the configured memory ceiling, such as `0MiB / 12288MiB` for a 12GB slice.

## Testing

Tests are built with the project.

```bash
cd build
make
```

Basic C/CUDA tests:

```bash
export CUDA_DEVICE_MEMORY_LIMIT=4G
export LD_PRELOAD=./libsoftmig.so

cd test
./test_alloc
./test_alloc_host
./test_alloc_managed
./test_runtime_alloc
./test_runtime_launch
```

Python framework tests:

```bash
cd build/test/python

export CUDA_DEVICE_MEMORY_LIMIT=4G
export LD_PRELOAD=../../libsoftmig.so

python limit_pytorch.py
python limit_tensorflow.py
python limit_tensorflow2.py
python limit_mxnet.py
```

Tests use `LD_PRELOAD` for development. Production uses `/etc/ld.so.preload`.

## Optional `nvidia-smi` Hook

SoftMig includes an optional `nvidia-smi` wrapper that filters process output by SLURM job cgroup. This lets users sharing a GPU see only their own job's GPU processes instead of every process on the device.

Install system-wide:

```bash
sudo mv /usr/bin/nvidia-smi /usr/bin/nvidia-smi.real
sudo cp nvidia-smi-hook.sh /usr/bin/nvidia-smi
sudo chmod +x /usr/bin/nvidia-smi
```

The hook is optional. SoftMig works without it, but unfiltered `nvidia-smi` will show all GPU processes on the node.

## How OOM Works

SoftMig checks GPU memory usage before CUDA allocations. If an allocation would exceed the configured job limit, the application receives `CUDA_ERROR_OUT_OF_MEMORY`.

Memory is tracked across all processes in the same SLURM job, so MPI, PyTorch DDP, and other multi-process workloads share one memory pool.

Example:

```bash
# Job with 12GB limit
CUDA_DEVICE_MEMORY_LIMIT=12g

# Process 1 allocates 8GB - succeeds
# Process 2 tries to allocate 6GB - fails: 8GB + 6GB > 12GB
```

Optional active OOM killing can be enabled with:

```bash
ACTIVE_OOM_KILLER=1
```

When enabled, SoftMig kills matching GPU processes from the current SLURM job cgroup or UID. Use with caution.

## Important Notes

- SoftMig requires CUDA 12+.
- SoftMig is software isolation, not hardware isolation.
- In SLURM jobs, limits come from root-owned config files, not user environment variables.
- Temporary state uses `$SLURM_TMPDIR` for per-job isolation and cleanup.
- Compute limiting is implemented through kernel launch throttling / SM time-slicing.
- Logs are written to `/var/log/softmig/{jobid}.log`.
- Delete `$SLURM_TMPDIR/cudevshr.cache*` when changing limits during testing.

## References

* [Digital Research Alliance of Canada](https://alliancecan.ca/)
* [Alliance Documentation (Source)](https://docs.alliancecan.ca/)
* [PAICE (Pan-Canadian AI Compute Environment)](https://alliancecan.ca/en/services/advanced-research-computing/pan-canadian-ai-compute-environment-paice)
* [RAGFlow](https://ragflow.io/)
* [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
* [Research Computing Group](https://www.ualberta.ca/en/information-services-and-technology/research-computing/index.html)
* [AMII](https://www.amii.ca/) — [Amii-Open-Source](https://github.com/Amii-Open-Source) — [amiithinks](https://github.com/amiithinks)
* [U of A RCG GitHub](https://github.com/ualberta-rcg)
* [Vulcan Login / OOD](https://vulcan.alliancecan.ca) — [Vulcan Portal](https://portal.vulcan.alliancecan.ca)

---

## Support

Many Bothans died to bring us this information. This project is provided as-is, but reasonable questions may be answered based on my coffee intake or mood. ;)

Feel free to open an [issue](https://github.com/ualberta-rcg/ragflow-wiki-data/issues) or email **[khoja1@ualberta.ca](mailto:khoja1@ualberta.ca)** or **[kali2@ualberta.ca](mailto:kali2@ualberta.ca)** for U of A related deployments.

---

## License

This project is released under the **MIT License** — see [LICENSE](./LICENSE) for details.

---

## About University of Alberta Research Computing

The [Research Computing Group](https://www.ualberta.ca/en/information-services-and-technology/research-computing/index.html) supports high-performance computing, data-intensive research, and advanced infrastructure for researchers at the University of Alberta and across Canada through the [Digital Research Alliance of Canada](https://alliancecan.ca/).

The [Alberta Machine Intelligence Institute (AMII)](https://amii.ca/) is one of Canada's three national AI institutes and co-operates the Vulcan cluster for machine learning research workloads.
