<img src="./assets/ua_logo_green_rgb.png" alt="University of Alberta Logo" width="50%" />

# SoftMig

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![CUDA 12+](https://img.shields.io/badge/CUDA-12%2B-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![CUDA 13](https://img.shields.io/badge/CUDA-13-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![NVIDIA GPU](https://img.shields.io/badge/GPU-L%20%7C%20A%20%7C%20V%20%7C%20RTX%20Series-76B900.svg)](https://www.nvidia.com/)

> **Software MIG for any NVIDIA GPU — no hardware MIG required.**

*Deployed on the [University of Alberta](https://ualberta-rcg.github.io/ragflow-wiki-data/)/[Alberta Machine Intelligence Institute](https://www.amii.ca/) Vulcan Cluster*  

**Maintained by:** Rahim Khoja ([khoja1@ualberta.ca](mailto:khoja1@ualberta.ca)) and Karim Ali ([kali2@ualberta.ca](mailto:kali2@ualberta.ca))

---

## 📖 Description

**SoftMig** is a SLURM-integrated software GPU slicing system for shared NVIDIA GPU clusters. It lets administrators schedule ordinary NVIDIA GPUs in a MIG-like way using software-enforced memory limits, compute time-slicing, and SLURM prolog/epilog automation.

SoftMig does **not** enable, modify, or replace NVIDIA Hardware MIG. Instead, it provides a scheduler-controlled software layer for running fractional GPU jobs on GPUs that do not support hardware MIG, or on clusters where hardware MIG is too rigid for day-to-day research scheduling.

The goal is to let SLURM treat each GPU as a flexible shared resource. A 48GB GPU, for example, can be offered as a full GPU, two half-GPU slices, four quarter-GPU slices, or other site-defined layouts. Each job receives a configured memory limit and proportional access to GPU compute through time-slicing and SM throttling.

Unlike Hardware MIG, SoftMig slice layouts can be changed through SLURM policy without draining nodes, rebooting, or changing GPU MIG mode. This allows full-GPU jobs, half-GPU jobs, quarter-GPU jobs, and other fractional jobs to coexist across the same GPU nodes using normal SLURM scheduling.

SoftMig is based on [HAMi-core](https://github.com/Project-HAMi/HAMi-core), adapted for SLURM-based HPC environments. The project was inspired by Tim Weiers from the University of Alberta Unix team, who suggested HAMi-core as a foundation for bringing software GPU slicing into research computing clusters.

SoftMig is intended for [Digital Research Alliance of Canada](https://alliancecan.ca/) / Compute Canada-style research clusters where GPU utilization, scheduling flexibility, and broad NVIDIA GPU compatibility matter more than hardware-level isolation.

<p align="center">
<strong>SoftMig in action on the University of Alberta Vulcan cluster</strong>
</p>

<table align="center">
<tr>
<td align="center"><img src="./assets/softmig_pic1.png" alt="SoftMig 1/2 L40S slice on Vulcan" width="400" /></td>
<td align="center"><img src="./assets/softmig_pic2.png" alt="SoftMig 1/4 L40S slice on Vulcan" width="400" /></td>
</tr>
<tr>
<td align="center"><strong>1/2 L40S Slice</strong><br><code>nvidia-smi</code> reports ~24 GB visible</td>
<td align="center"><strong>1/4 L40S Slice</strong><br><code>nvidia-smi</code> reports ~12 GB visible</td>
</tr>
</table>

## ✨ Features

- **GPU Memory Slicing** — enforce per-job GPU memory ceilings; when a job exceeds its limit, CUDA returns `CUDA_ERROR_OUT_OF_MEMORY`, just like running on a smaller physical GPU
- **GPU Compute Slicing** — kernel launch throttling and SM time-slicing to limit GPU compute access per job
- **Works on Any CUDA 12+ GPU** — L40S, A40, V100, RTX-class, and others; no MIG-capable hardware required
- **SLURM-Native Lifecycle** — prolog creates root-owned config files, epilog removes them; users never set their own limits
- **Dynamic Slice Layouts** — change from 4 slices to 8 slices per GPU through policy alone, no node drain or reboot
- **Per-Job Isolation** — cache and lock files under `$SLURM_TMPDIR` (no shared `/tmp` conflicts)
- **Scheduler-Enforced** — loaded via `/etc/ld.so.preload`; users cannot bypass limits
- **Silent by Default** — logs written to `/var/log/softmig/{jobid}.log`, invisible to users
- **Optional `nvidia-smi` Filtering** — wrapper script hides other jobs' GPU processes by cgroup
- **Framework Agnostic** — PyTorch, TensorFlow, JAX, MXNet, and any other CUDA workload

## 🚀 Quickstart

### For SLURM Users

Request a GPU slice through SLURM (exact GRES names depend on site policy):

```bash
# Quarter GPU (12GB / 25% SM on a 48GB GPU)
sbatch --gres=gpu:l40s.4:1 --time=5:00:00 job.sh

# Half GPU (24GB / 50% SM on a 48GB GPU)
sbatch --gres=gpu:l40s.2:1 --time=2:00:00 job.sh

# Full GPU (no limits)
sbatch --gres=gpu:l40s:1 --time=2:00:00 job.sh
```

### For Cluster Admins

- **Build, install, and update**: [docs/BUILD_AND_INSTALL.md](docs/BUILD_AND_INSTALL.md)
- **SLURM integration** (prolog/epilog/job_submit): [docs/SLURM_INTEGRATION.md](docs/SLURM_INTEGRATION.md)

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [docs/USAGE.md](docs/USAGE.md) | User-facing usage, configuration model, and slice layout |
| [docs/SLURM_INTEGRATION.md](docs/SLURM_INTEGRATION.md) | Prolog, epilog, job_submit plugin, and file locations |
| [docs/BUILD_AND_INSTALL.md](docs/BUILD_AND_INSTALL.md) | Build, install, update runbook (includes safe `unshare -m` updates) |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common symptoms, quick checks, and fixes |
| [docs/TESTING.md](docs/TESTING.md) | Smoke tests and framework tests |
| [CHANGES.md](CHANGES.md) | Release-level architecture and behavior changes |

## 🔗 References

- [University of Alberta Research Computing](https://www.ualberta.ca/en/information-services-and-technology/research-computing/index.html)
- [Alberta Machine Intelligence Institute (AMII)](https://www.amii.ca/)
- [Digital Research Alliance of Canada](https://alliancecan.ca/)
- [HAMi-core (upstream project)](https://github.com/Project-HAMi/HAMi-core)
- [Vulcan Docs](https://ualberta-rcg.github.io/ragflow-wiki-data/)

---

## 🤝 Support

Many Bothans died to bring us this information. This project is provided as-is, but reasonable questions may be answered based on my coffee intake or mood. ;)

Feel free to open an issue or email **[khoja1@ualberta.ca](mailto:khoja1@ualberta.ca)** or **[kali2@ualberta.ca](mailto:kali2@ualberta.ca)** for U of A related deployments.

## 📜 License

This project is released under the **MIT License** - one of the most permissive open-source licenses available.

**What this means:**
- ✅ Use it for anything (personal, commercial, whatever)
- ✅ Modify it however you want
- ✅ Distribute it freely
- ✅ Include it in proprietary software

**The only requirement:** Keep the copyright notice somewhere in your project.

That's it! No other strings attached. The MIT License is trusted by major projects worldwide and removes virtually all legal barriers to using this code.

**Full license text:** [MIT License](./LICENSE)

## 🧠 About University of Alberta Research Computing

The [Research Computing Group](https://www.ualberta.ca/en/information-services-and-technology/research-computing/index.html) supports high-performance computing, data-intensive research, and advanced infrastructure for researchers at the University of Alberta and across Canada.

We help design and operate compute environments that power innovation — from AI training clusters to national research infrastructure.
