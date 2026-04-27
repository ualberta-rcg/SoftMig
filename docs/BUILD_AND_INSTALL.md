# SoftMig Build and Installation Guide

This document is the canonical guide for building, installing, and updating
`libsoftmig.so`, including the safe `unshare -m` update workflow.

## 1) Requirements

### Build system

- CMake (2.8.11+)
- GCC (or compatible C compiler with `-fPIC` and `-shared`)
- CUDA toolkit 12+ (CUDA 11 unsupported)
- Make

### Runtime system (compute nodes)

- NVIDIA driver compatible with CUDA 12+
- SLURM
- `nvidia-smi`

## 2) Build

### DRAC / Compute Canada (CVMFS modules)

```bash
# Confirm available versions first, then load CUDA 12+
# (use module spider on your cluster if needed)
module load cuda/12.2

cd /path/to/SoftMig/
rm -rf build
git pull
./build.sh

# Output artifact
ls -l build/libsoftmig.so
```

### Other environments

```bash
export CUDA_HOME=/path/to/cuda-12.2
./build.sh
ls -l build/libsoftmig.so
```

## 3) Install (initial)

### Option A (recommended): install script

```bash
sudo ./docs/examples/install_softmig.sh /path/to/build/libsoftmig.so
```

### Option B: manual install

```bash
# Create required directories
sudo mkdir -p /var/lib/shared /var/log/softmig /var/run/softmig

# Library directory and file permissions
sudo chown root:root /var/lib/shared
sudo chmod 755 /var/lib/shared
sudo cp build/libsoftmig.so /var/lib/shared/
sudo chown root:root /var/lib/shared/libsoftmig.so
sudo chmod 644 /var/lib/shared/libsoftmig.so

# Log directory must allow users to create files
if getent group slurm >/dev/null 2>&1; then
    sudo chown root:slurm /var/log/softmig
    sudo chmod 775 /var/log/softmig
else
    sudo chown root:root /var/log/softmig
    sudo chmod 1777 /var/log/softmig
fi

# Config directory (root-writable, world-readable)
sudo chown root:root /var/run/softmig
sudo chmod 755 /var/run/softmig

# System-wide preload
echo "/var/lib/shared/libsoftmig.so" | sudo tee -a /etc/ld.so.preload
sudo chown root:root /etc/ld.so.preload
sudo chmod 644 /etc/ld.so.preload
```

## 4) Update an already installed library (safe live update)

When `/etc/ld.so.preload` loads `libsoftmig.so`, overwriting the file directly
can fail or behave unpredictably. Use a temporary mount namespace:

```bash
sudo unshare -m -- sh -c " \
  mount --bind /dev/null /etc/ld.so.preload && \
  cp /path/to/new/libsoftmig.so /usr/local/lib/libsoftmig.so && \
  chown root:root /usr/local/lib/libsoftmig.so && \
  chmod 644 /usr/local/lib/libsoftmig.so \
"
```

**Important**: the destination path must match what is listed in `/etc/ld.so.preload`.
The install script uses `/var/lib/shared/libsoftmig.so`; some deployments use
`/usr/local/lib/libsoftmig.so` instead. Check your `/etc/ld.so.preload` and adjust
the `cp` target accordingly.

### DRAC operational update pattern (from login/admin host)

If you build on a login/admin host and deploy to a compute node over SSH, use:

```bash
# Example: deploy to a target node with preload-safe copy
TARGET_NODE=rack01-12
SOURCE_LIB=/home/$USER/scratch/SoftMig/build/libsoftmig.so
DEST_LIB=/usr/local/lib/libsoftmig.so

sudo ssh "$TARGET_NODE" "unshare -m -- sh -c '\
  mount --bind /dev/null /etc/ld.so.preload && \
  cp \"$SOURCE_LIB\" \"$DEST_LIB\" && \
  chown root:root \"$DEST_LIB\" && \
  chmod 644 \"$DEST_LIB\" \
'"
```

If your production path is `/var/lib/shared/libsoftmig.so`, set `DEST_LIB` accordingly
and make sure `/etc/ld.so.preload` points to that exact path.

### Exact local update command example (no SSH)

Example build artifact location pattern:

`/$LOCATION/SoftMig/build/libsoftmig.so`

Example:

```bash
LOCATION=home/rahimk/scratch
```

Use this local command on the target node:

```bash
sudo unshare -m -- sh -c 'mount --bind /dev/null /etc/ld.so.preload && cp /$LOCATION/SoftMig/build/libsoftmig.so /usr/local/lib/libsoftmig.so && chmod 644 /usr/local/lib/libsoftmig.so && chown root:root /usr/local/lib/libsoftmig.so'
```

### Why this works

- `unshare -m` creates a new mount namespace.
- Binding `/dev/null` over `/etc/ld.so.preload` disables preload only in that namespace.
- The copy runs without preload-lock side effects from the current namespace.

## 5) Verify installation/update

```bash
# Check preload target
cat /etc/ld.so.preload

# Confirm permissions
ls -ld /var/lib/shared /var/log/softmig /var/run/softmig
ls -l /var/lib/shared/libsoftmig.so

# Quick runtime check (inside a job)
nvidia-smi
```

## 6) SLURM integration pointers

- Prolog example: `docs/examples/prolog_softmig.sh`
- Epilog example: `docs/examples/epilog_softmig.sh`
- Job submit plugin example: `docs/examples/job_submit_softmig.lua`
- SLURM config snippet: `docs/slurm.conf.example`

See also:

- SLURM integration overview: `docs/SLURM_INTEGRATION.md`
- User-facing usage: `docs/USAGE.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`
