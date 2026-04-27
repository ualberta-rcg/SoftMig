# SoftMig Troubleshooting

This document is aimed at cluster admins/operators diagnosing common SoftMig problems.

## Quick checks

Inside a running SLURM job:

- **Config exists**: `/var/run/softmig/` contains `{jobid}.conf` (or `{jobid}_{arrayid}.conf` for array tasks)
- **Library is loaded**: `/etc/ld.so.preload` includes the installed `libsoftmig.so` path
- **Logs exist**: `/var/log/softmig/{jobid}.log` (or `$SLURM_TMPDIR/softmig_{jobid}.log` fallback)

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

SoftMig writes logs under `/var/log/softmig/` by default. The directory must allow users (job UIDs) to create files.

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

