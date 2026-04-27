# University of Alberta - SoftMig 

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![MkDocs Site](https://img.shields.io/badge/Docs-Live-blue?style=flat-square)](https://ualberta-rcg.github.io/ragflow-wiki-data/)
[![Full Pipeline](https://img.shields.io/badge/Pipeline-Weekly-purple?style=flat-square)](https://github.com/ualberta-rcg/ragflow-wiki-data/actions/workflows/full-pipeline.yml)

**Maintained by:** Rahim Khoja ([khoja1@ualberta.ca](mailto:khoja1@ualberta.ca)) and Kahim Ali ([kali2@ualberta.ca](mailto:kali2@ualberta.ca)) 

---

## Description

**SoftMig** is a fork of [HAMi-core](https://github.com/Project-HAMi/HAMi-core) optimized for **Digital Research Alliance Canada (DRAC) / Compute Canada** SLURM environments. It provides software-based GPU memory and compute cycle limiting for oversubscribed GPU partitions.

Like NVIDIA's hardware MIG, SoftMig enables software-based GPU slicing for any GPU:
- **GPU Memory Slicing**: Divide GPU memory among multiple jobs (e.g., 12GB, 24GB slices on 48GB GPUs)
- **GPU Compute Slicing**: Limit SM utilization per job (e.g., 25%, 50% of GPU cycles)
- **Oversubscription**: Run 2-8 jobs per GPU safely
- **SLURM Integration**: Uses `SLURM_TMPDIR` for per-job isolation (cache files, locks) - no shared `/tmp` conflicts

## Project Documents

- `CHANGES.md`: release-level architecture and behavior changes
- `docs/PROJECT_STATUS.md`: current operational status and open follow-ups
- `docs/FIXES_TO_APPLY.md`: actionable checklist of remaining fixes
- `docs/BUILD_AND_INSTALL.md`: build, install, and safe `unshare -m` update guide

## SoftMig vs. NVIDIA Hardware MIG

| Feature | SoftMig (Software MIG) | NVIDIA Hardware MIG |
|---------|------------------------|---------------------|
| **GPU Resource Usage** | Uses 100% of GPU resources (no overhead) | Loses ~5-10% of GPU resources to MIG overhead |
| **Dynamic Configuration** | ✅ Dynamic, on-the-fly changes via SLURM prolog/epilog | ❌ Requires draining SLURM node and rebooting to change MIG mode |
| **Isolation** | ⚠️ Software-based (process-level) - crashes can affect other jobs | ✅ Hardware-based isolation - crashes isolated to MIG instance |
| **GPU Compatibility** | ✅ Works on any NVIDIA GPU (Pascal, Volta, Ampere, Ada, Hopper) | ❌ Only works on A100, H100, and newer datacenter GPUs |
| **Setup Complexity** | ✅ Simple: install library, configure SLURM | ⚠️ Requires GPU driver support, MIG mode configuration |
| **Performance Overhead** | ⚠️ Minimal (~1-2% from kernel launch throttling) | ✅ No software overhead (hardware-native) |
| **Multi-Instance Support** | ✅ Unlimited slices (memory/compute limited) | ⚠️ Limited by GPU architecture (max 7 instances on A100) |

**When to Use SoftMig:**
- Need dynamic GPU slicing without node downtime
- Want to use 100% of GPU resources
- Using GPUs that don't support hardware MIG (L40S, RTX, etc.)
- Need flexible, on-demand slice sizing

**When to Use Hardware MIG:**
- Need maximum isolation (security, fault tolerance)
- Using A100/H100 datacenter GPUs
- Can tolerate node downtime for configuration changes
- Want zero software overhead

## Differences from Original HAMi-core

SoftMig is optimized for SLURM cluster environments with the following key improvements:

| Feature | Original HAMi-core | SoftMig |
|---------|-------------------|---------|
| **Temporary Files** | Uses `/tmp` (shared across all jobs/users) | ✅ Uses `$SLURM_TMPDIR` (per-job isolation) |
| **Cache Files** | `/tmp/cudevshr.cache` (shared, can conflict) | ✅ `$SLURM_TMPDIR/cudevshr.cache.{jobid}` (job-specific) |
| **Lock Files** | `/tmp/vgpulock/` (shared) | ✅ `$SLURM_TMPDIR/vgpulock/` (job-specific) |
| **Configuration** | Environment variables only | ✅ Secure config files (`/var/run/softmig/{jobid}.conf`) + env vars fallback |
| **Logging** | stderr (visible to users) | ✅ File-only logging (`/var/log/softmig/`) - silent to users |
| **Log File Names** | Process ID based | ✅ Job ID based (`{jobid}.log`) |
| **Library Loading** | `LD_PRELOAD` (users can disable) | ✅ `/etc/ld.so.preload` (users cannot disable) |
| **SLURM Integration** | Manual setup | ✅ Automated via `prolog.sh`/`epilog.sh` |
| **Multi-CUDA Support** | CUDA 11+ | ✅ CUDA 12+ (CUDA 11 does not work) |
| **Library Name** | `libvgpu.so` | ✅ `libsoftmig.so` |

**Key Benefits:**
- ✅ **Job Isolation**: Each SLURM job gets its own cache/lock files in `SLURM_TMPDIR` (no conflicts)
- ✅ **Security**: Config files in `/var/run/softmig/` prevent users from modifying limits
- ✅ **Silent Operation**: No user-visible logs (file-only logging)
- ✅ **Enforcement**: System-wide preload ensures users cannot disable the library
- ✅ **Auto-cleanup**: Cache files automatically cleaned when job ends (SLURM_TMPDIR is job-specific)

## Building and Installation

Build, install, and live-update instructions (including safe `unshare -m`
library replacement under `/etc/ld.so.preload`) are maintained in:

- `docs/BUILD_AND_INSTALL.md`

## Usage

### Configuration

SoftMig uses a priority system for configuration:

1. **Config files** (only source in SLURM jobs) - `/var/run/softmig/{jobid}.conf`
2. **Environment variables** - Used **only outside SLURM** (local testing). Inside a SLURM job, env vars are ignored even if no config file exists.
3. **Passive mode** - No limits enforced (library loads but does nothing)

#### Config Files (SLURM Jobs)

**Location**: `/var/run/softmig/{jobid}.conf` or `/var/run/softmig/{jobid}_{arrayid}.conf` (for array jobs)

**Format**: Key-value pairs, one per line (same format as environment variables):
```
CUDA_DEVICE_MEMORY_LIMIT=12288M
CUDA_DEVICE_SM_LIMIT=25
```

**Created by**: `prolog_softmig.sh` (SLURM prolog script) - automatically creates config files based on requested GPU slice type

**Deleted by**: `epilog_softmig.sh` (SLURM epilog script) - automatically cleans up when job ends

**Security**: 
- Files are owned by `root:root` with permissions `644` (rw-r--r--)
- Users cannot modify these files (directory is writable only by root)
- Prevents users from bypassing limits

**Example config file** (for a quarter GPU slice on a 48GB GPU):
```
CUDA_DEVICE_MEMORY_LIMIT=12288M
CUDA_DEVICE_SM_LIMIT=25
```

#### Environment Variables (Testing)

**Outside SLURM only (local testing)**: When `SLURM_JOB_ID` is not set, SoftMig reads limits from environment variables. Inside a SLURM job, environment variables are never used — only the config file applies.

**Example**:
```bash
export CUDA_DEVICE_MEMORY_LIMIT=16g
export CUDA_DEVICE_SM_LIMIT=50
export LD_PRELOAD=/var/lib/shared/libsoftmig.so
# Now run your application
```

#### Passive Mode

**When**: No config file exists AND no environment variables are set

**Behavior**: 
- Library loads (via `/etc/ld.so.preload`)
- All SoftMig functions check `is_softmig_enabled()` and return early (no-op)
- No limits are enforced
- No performance impact (minimal overhead from function call checks)

**Purpose**: Allows safe system-wide preload - the library is loaded for all processes but only activates when configured. This ensures:
- Users cannot disable the library (it's in `/etc/ld.so.preload`)
- The library is safe for all processes (does nothing until configured)
- No conflicts with applications that don't need GPU limiting

### Environment Variables

SoftMig supports the following environment variables. **In production SLURM jobs, these are set via config files** (see Configuration section). Environment variables are only used for testing or when no config file exists.

#### Memory and Compute Limits

- **`CUDA_DEVICE_MEMORY_LIMIT`**: GPU memory limit per device
  - Format: Integer followed by a single-character unit suffix: `G`/`g`, `M`/`m`, `K`/`k`
  - Examples: `16g`, `24G`, `12288M` (note: `12GB` does NOT work — only single-char suffixes)
  - No suffix = bytes
  - Per-device override: `CUDA_DEVICE_MEMORY_LIMIT_0`, `CUDA_DEVICE_MEMORY_LIMIT_1`, etc.
  - **Used for**: Limiting GPU memory allocations. When exceeded, allocations return `CUDA_ERROR_OUT_OF_MEMORY`.

- **`CUDA_DEVICE_SM_LIMIT`**: SM (Streaming Multiprocessor) utilization percentage
  - Format: Integer 0-100 (percentage)
  - `0` = no limit (disabled)
  - `50` = 50% of GPU compute cycles
  - `100` = 100% (no throttling)
  - Per-device override: `CUDA_DEVICE_SM_LIMIT_0`, `CUDA_DEVICE_SM_LIMIT_1`, etc.
  - **Used for**: Throttling kernel launches to limit GPU compute utilization. Only monitors device 0 by default (intentional - fractional GPU jobs typically only get 1 GPU).

#### Logging

- **`SOFTMIG_LOG_LEVEL`**: Log verbosity level
  - `0` (default): Errors only
  - `1`: Errors + warnings + messages
  - `2`: Errors + warnings + messages + debug
  - `3`: Errors + warnings + messages + debug + info (console output enabled)
  - `4`: All logs (maximum verbosity)
  - **Used for**: Controlling how much detail is written to log files. Level 3+ also enables console output for debugging.

- **`SOFTMIG_LOG_FILE`**: Custom log file path (optional)
  - Format: Full path to log file (e.g., `/tmp/my_softmig.log`)
  - **Used for**: Overriding the default log location (`/var/log/softmig/{jobid}.log`)

#### Testing/Development

- **`LD_PRELOAD`**: Path to `libsoftmig.so` (for testing only)
  - Format: Full path (e.g., `/var/lib/shared/libsoftmig.so` or `./build/libsoftmig.so`)
  - **Used for**: Loading the library for testing. **Production uses `/etc/ld.so.preload`** (users cannot disable it).

#### OOM Killer (Advanced)

- **`ACTIVE_OOM_KILLER`**: Enable active OOM killer (optional, disabled by default)
  - Format: `1` (enabled) or `0` (disabled)
  - **Used for**: When enabled, SoftMig will actively kill processes from the current cgroup/UID when memory limit is exceeded. See "How OOM Works" section for details.
  - **Warning**: This is aggressive and will kill all processes from the current user/cgroup on the GPU when OOM occurs.

#### Environment Variable Priority

1. **Inside SLURM** (`SLURM_JOB_ID` is set): Config file is the **only source**. Environment variables are ignored. If no config file exists, the job runs in passive mode (no limits).
2. **Outside SLURM** (`SLURM_JOB_ID` is not set): Environment variables are used. If none are set, passive mode.
3. **Passive mode** - No limits enforced; library is a no-op.

### Quick Test

```bash
# 1. Delete cache (important when changing limits!)
rm -f ${SLURM_TMPDIR}/cudevshr.cache*

# 2. Set limit (for testing - production uses config files)
export CUDA_DEVICE_MEMORY_LIMIT=16g

# 3. Load library
export LD_PRELOAD=/var/lib/shared/libsoftmig.so

# 4. Test
nvidia-smi  # Should show: 0MiB / 16384MiB (16GB limit)
```

### For SLURM Users

Once deployed, simply request GPU slices:

```bash
# Half GPU slice (24GB, 50% SM) - with job_submit.lua, this translates to gres/shard:l40s:2
sbatch --gres=gpu:l40s.2:1 --time=2:00:00 job.sh

# Quarter GPU slice (12GB, 25% SM) - translates to gres/shard:l40s:1
sbatch --gres=gpu:l40s.4:1 --time=5:00:00 job.sh

# Full GPU (no limits)
sbatch --gres=gpu:l40s:1 --time=2:00:00 job.sh
```

**Note**: The `job_submit_softmig.lua` plugin (if configured) automatically:
- Validates that slice requests have `count=1` (no multiple slices of same size)
- Translates `gpu:type.denominator:count` to `gres/shard:type:shard_count` format
- Default configuration: 4 shards per GPU (so `l40s.2:1` = 2 shards, `l40s.4:1` = 1 shard)

Limits are automatically configured by `prolog_softmig.sh` based on the requested GPU slice type.

## Memory Limits by GPU Slice

**Default Configuration**: 4 shards per GPU (configurable in `prolog_softmig.sh` and `job_submit_softmig.lua`)

| Slice Type | Shard Count | Memory (48GB GPU) | SM Limit | Oversubscription |
|------------|-------------|-------------------|----------|------------------|
| l40s (full) | N/A | 48GB | 100% | 1x |
| l40s.2 (half) | 2 shards | 24GB | 50% | 2x |
| l40s.4 (quarter) | 1 shard | 12GB | 25% | 4x |

**Note**: With 4 shards per GPU, the smallest slice is 1/4 GPU. For 1/8 GPU slices, configure 8 shards per GPU.

## File Locations

### Cache Files
- **Location**: `$SLURM_TMPDIR/cudevshr.cache.{jobid}` or `$SLURM_TMPDIR/cudevshr.cache.{jobid}.{arrayid}`
- **Purpose**: Shared memory region for tracking GPU memory usage across processes in the same job
- **Cleanup**: Automatically cleaned when job ends (SLURM_TMPDIR is job-specific)
- **Important**: Delete cache files when changing limits: `rm -f ${SLURM_TMPDIR}/cudevshr.cache*`

### Config Files
- **Location**: `/var/run/softmig/{jobid}.conf` or `/var/run/softmig/{jobid}_{arrayid}.conf` (for array jobs)
- **Format**: Key-value pairs, one per line (same format as environment variables)
  ```
  CUDA_DEVICE_MEMORY_LIMIT=12288M
  CUDA_DEVICE_SM_LIMIT=25
  ```
- **Created by**: `prolog_softmig.sh` (SLURM prolog script)
- **Deleted by**: `epilog_softmig.sh` (SLURM epilog script)
- **Permissions**: `644` (rw-r--r--), owned by root:root (users cannot modify)
- **Purpose**: Secure configuration for SLURM jobs (prevents users from modifying limits)

### Log Files
- **Location**: `/var/log/softmig/{jobid}.log` or `/var/log/softmig/{jobid}_{arrayid}.log` (for array jobs)
- **Fallback**: If `/var/log/softmig` is not writable, logs use `$SLURM_TMPDIR/softmig_{jobid}.log`
- **Outside SLURM**: Logs use `/var/log/softmig/pid{pid}.log` (process ID based)
- **Permissions**: Directory must be writable by all users (see installation section)
  - Recommended: `775` with `slurm` group, or `1777` with sticky bit
- **Purpose**: Silent logging (users don't see logs by default; admins can view them)
- **Viewing**: `tail -f /var/log/softmig/*.log` (as admin)

## Logging

Logs are written to `/var/log/softmig/{jobid}.log` and are completely silent to users by default (file-only logging).

### Log Directory Setup

**Critical**: The `/var/log/softmig` directory **must allow any user to write** to it. This is required because:
- Each SLURM job runs as a different user
- SoftMig needs to **create new log files** for each job (e.g., `{jobid}.log`)
- **The directory itself must be writable** - even if log files have write permissions, users cannot create new files without write permission on the directory
- The directory is created automatically if it doesn't exist, but proper permissions must be set

**Recommended setup** (choose one):

1. **Group writable (recommended - more secure)**:
   ```bash
   sudo chown root:slurm /var/log/softmig
   sudo chmod 775 /var/log/softmig  # drwxrwxr-x (group writable)
   # Ensure all SLURM users are in the slurm group
   ```

2. **Sticky bit + world writable (works but less secure)**:
   ```bash
   sudo chown root:root /var/log/softmig
   sudo chmod 1777 /var/log/softmig  # drwxrwxrwt (sticky bit prevents deletion of others' files)
   ```

**Verification**:
```bash
# Test if users can CREATE new files in the directory
sudo -u <test_user> touch /var/log/softmig/test.log
# Should succeed without errors

# If it fails with "Permission denied", check directory permissions:
ls -ld /var/log/softmig
# Should show: drwxrwxr-x (775) or drwxrwxrwt (1777)

# The directory MUST have write permission (w) for users/group/others
# 775 = rwxrwxr-x (group writable)
# 1777 = rwxrwxrwt (world writable with sticky bit)
```

### Log Levels

Use `SOFTMIG_LOG_LEVEL` environment variable to control verbosity:
- `0` (default): Errors only
- `1`: Errors + warnings + messages
- `2`: Errors + warnings + messages + debug
- `3`: Errors + warnings + messages + debug + info (console output enabled)
- `4`: All logs (maximum verbosity)

**Example for debugging**:
```bash
export SOFTMIG_LOG_LEVEL=3
# Now logs will also appear on stderr (useful for debugging)
```

### Viewing Logs

**As admin**:
```bash
# View all logs
tail -f /var/log/softmig/*.log

# View specific job log
tail -f /var/log/softmig/12345.log

# Search for errors
grep -i error /var/log/softmig/*.log
```

**Note**: Users cannot view logs by default (they're in `/var/log/softmig` which requires admin access). For debugging, users can set `SOFTMIG_LOG_LEVEL=3` to see logs on stderr.

## Deployment for Cluster Administrators

### Installation and Library Updates

Build/install/update runbook is centralized in:

- `docs/BUILD_AND_INSTALL.md`

That guide includes:

- initial installation (script and manual)
- permissions model (`/var/lib/shared`, `/var/log/softmig`, `/var/run/softmig`)
- `/etc/ld.so.preload` setup
- safe live library replacement with `unshare -m`
- post-install verification steps

### SLURM Configuration

1. **Update `slurm.conf`**:
   ```bash
   Prolog=/etc/slurm/prolog.sh
   Epilog=/etc/slurm/epilog.sh
   ```
   See `docs/slurm.conf.example` for minimal configuration.

2. **Create/update `prolog.sh`**:
   - Creates secure config files in `/var/run/softmig/{jobid}.conf` (or `{jobid}_{arrayid}.conf` for array jobs)
   - See `docs/examples/prolog_softmig.sh` for complete example
   - Configures limits based on requested GPU slice type (e.g., `gres/shard:l40s:2` for half GPU)
   - Calculates memory and SM limits from shard count (default: 4 shards per GPU)

3. **Create/update `epilog.sh`**:
   - Cleans up config files after job ends
   - See `docs/examples/epilog_softmig.sh` for complete example

4. **Create/update `job_submit.lua`** (optional but recommended):
   - Validates GPU slice requests (ensures count=1 for slices, prevents invalid denominators)
   - Translates GPU slice syntax (e.g., `l40s.2:1`) to `gres/shard:l40s:2` format
   - See `docs/examples/job_submit_softmig.lua` for complete example

### How It Works

**System-Wide Preload (`/etc/ld.so.preload`):**
- Library is loaded for ALL processes (users cannot disable it)
- Library is passive (does nothing) until a config file is created
- Prolog creates config file → activates limits for that job
- Epilog deletes config file → deactivates limits when job ends

**Config File Priority:**
1. Config file (`/var/run/softmig/{jobid}.conf` or `/var/run/softmig/{jobid}_{arrayid}.conf`) - **Only source for SLURM jobs** (created by `prolog_softmig.sh`)
2. Environment variables - **Only used outside SLURM** (when `SLURM_JOB_ID` is not set). Inside a SLURM job without a config file, limits are 0 (passive mode).

**Passive Mode:**
- If neither config file nor environment variables are set, SoftMig operates in passive mode
- Library loads but does not enforce any limits (checks `is_softmig_configured()` internally)
- This allows safe system-wide preload via `/etc/ld.so.preload` - the library is loaded for all processes but only activates when configured
- All SoftMig functions check `is_softmig_enabled()` and return early (no-op) when in passive mode

**Security:**
- Users cannot modify config files (admin-only directory)
- Users cannot disable library (system-wide preload)
- Library only activates when config file exists (safe for all processes)

### Memory and SM Limits by GPU Slice

**Default Configuration**: 4 shards per GPU (configurable in `prolog_softmig.sh` via `NUM_SHARDS_PER_GPU`)

| GPU Slice | Shard Count | Memory Limit (48GB GPU) | SM Limit | Oversubscription | Use Case |
|-----------|-------------|------------------------|----------|------------------|----------|
| l40s (full) | N/A | 48GB | 100% | 1x | Large models, full GPU needed |
| l40s.2 (half) | 2 shards | 24GB | 50% | 2x | Medium models, 2x oversubscription |
| l40s.4 (quarter) | 1 shard | 12GB | 25% | 4x | Small models, 4x oversubscription |

**SM Limiting:** GPU compute utilization limiting works via kernel launch throttling. Only monitors device 0 (intentional for fractional jobs that receive one GPU).

### Monitoring and Troubleshooting

**Log Files:**
- Location: `/var/log/softmig/{jobid}.log`
- View: `tail -f /var/log/softmig/*.log` (as admin)
- Silent to users by default (set `SOFTMIG_LOG_LEVEL=2` in job for debugging)

**Cache Files:**
- Location: `$SLURM_TMPDIR/cudevshr.cache.{jobid}` or `$SLURM_TMPDIR/cudevshr.cache.{jobid}.{arrayid}` (auto-cleaned when job ends)
- **Important**: Delete cache files when changing limits: `rm -f ${SLURM_TMPDIR}/cudevshr.cache*`

**Verification:**
- Check that config files are created: `ls -l /var/run/softmig/` (should see `{jobid}.conf` files)
- Check that library is loaded: `cat /etc/ld.so.preload` (should contain `/var/lib/shared/libsoftmig.so`)
- Test in job: `nvidia-smi` should show limited memory (e.g., `0MiB / 12288MiB` for 12GB limit)
- Check logs: `tail -f /var/log/softmig/{jobid}.log` (as admin)

## Testing

Tests are automatically built when you compile the project. To run the tests:

### C/CUDA Tests

```bash
# Build the project (tests are built automatically)
cd build
make

# Set up environment for testing
export CUDA_DEVICE_MEMORY_LIMIT=4G  # Set a memory limit for testing
export LD_PRELOAD=./libsoftmig.so   # Load the library

# Run individual tests (from build/test directory)
cd test
./test_alloc                    # Test basic memory allocation
./test_alloc_host              # Test host memory allocation
./test_alloc_managed           # Test managed memory
./test_runtime_alloc           # Test CUDA runtime API allocation
./test_runtime_launch          # Test kernel launches (CUDA)

# Or run all tests
for test in test_*; do
    echo "Running $test..."
    ./$test
done
```

### Python Framework Tests

```bash
# Python tests are copied to build/test/python/ during build
cd build/test/python

# Test with PyTorch
export CUDA_DEVICE_MEMORY_LIMIT=4G
export LD_PRELOAD=../../libsoftmig.so
python limit_pytorch.py

# Test with TensorFlow
python limit_tensorflow.py

# Test with TensorFlow 2
python limit_tensorflow2.py

# Test with MXNet
python limit_mxnet.py
```

### Testing with Different Limits

```bash
# Test with different memory limits
export CUDA_DEVICE_MEMORY_LIMIT=2G
export LD_PRELOAD=./libsoftmig.so
./test_alloc

# Test with SM utilization limit
export CUDA_DEVICE_SM_LIMIT=50  # 50% utilization
export LD_PRELOAD=./libsoftmig.so
./test_runtime_launch
```

**Note**: Tests use `LD_PRELOAD` for development/testing. In production, the library is loaded via `/etc/ld.so.preload`.

## nvidia-smi Hook

SoftMig includes a `nvidia-smi-hook.sh` script that filters `nvidia-smi` output to show only processes from the current SLURM job's cgroup. This is useful when multiple jobs share a GPU - each job will only see its own processes.

### Setup

**Option 1: Replace nvidia-smi system-wide** (recommended for SLURM clusters):
```bash
# Backup original nvidia-smi
sudo mv /usr/bin/nvidia-smi /usr/bin/nvidia-smi.real

# Install hook
sudo cp nvidia-smi-hook.sh /usr/bin/nvidia-smi
sudo chmod +x /usr/bin/nvidia-smi
```

**Option 2: Use as wrapper** (users call it explicitly):
```bash
# Users can create an alias or wrapper script
alias nvidia-smi='~/softmig-testing/SoftMig/nvidia-smi-hook.sh'
```

### How It Works

The hook:
1. Detects the current SLURM job cgroup from `/proc/self/cgroup`
2. Filters `nvidia-smi` output to show only processes in the same cgroup
3. For `--query-compute-apps` (CSV format), filters by PID and cgroup
4. For standard output, filters process lines by PID and cgroup
5. Passes through all other `nvidia-smi` arguments unchanged

**Example**:
```bash
# Without hook: shows all processes on GPU
nvidia-smi
# |    0   N/A  N/A    12345    C   python    4096MiB |
# |    0   N/A  N/A    67890    C   python    2048MiB |  # From another job

# With hook: shows only processes from current job
nvidia-smi
# |    0   N/A  N/A    12345    C   python    4096MiB |  # Only this job's processes
```

**Note**: The hook is optional. SoftMig works without it, but `nvidia-smi` will show all processes on the GPU (from all jobs).

## How OOM (Out of Memory) Works

When a GPU memory allocation would exceed the configured limit, SoftMig enforces the limit through the following mechanism:

### Memory Limit Enforcement

1. **Pre-allocation Check**: Before each GPU memory allocation, SoftMig checks if the allocation would exceed the limit:
   - Queries NVML for current GPU memory usage (summed across all processes in the job)
   - Adds the requested allocation size
   - Compares against the configured limit

2. **OOM Detection**: If the limit would be exceeded:
   - SoftMig returns `CUDA_ERROR_OUT_OF_MEMORY` to the application
   - The application receives the error and must handle it (e.g., reduce batch size, free memory)
   - **No automatic killing** by default (passive enforcement)

3. **Dead Process Cleanup**: Before returning OOM, SoftMig attempts to clean up dead processes:
   - Checks for processes that have exited but still have GPU memory allocated
   - Removes them from tracking
   - Re-checks the limit (may allow the allocation if dead processes freed enough memory)

### Active OOM Killer (Optional)

If `ACTIVE_OOM_KILLER=1` is set (via environment variable or config file), SoftMig will actively kill processes when OOM occurs:

1. **Process Selection**: When OOM is detected, SoftMig:
   - Queries NVML for all processes using GPU memory
   - Filters to only processes from the current SLURM job cgroup (or current UID if cgroup unavailable)
   - **Kills all matching processes** (sends `SIGKILL`)

2. **Safety**: 
   - Root user (UID 0) is **never killed** (OOM killer disabled for root)
   - Only processes from the current job's cgroup/UID are killed
   - Processes are verified to belong to the current job before killing

3. **Use Case**: Useful for aggressive enforcement, but can be disruptive if applications don't handle OOM gracefully.

**Example**:
```bash
# Enable active OOM killer (in config file or environment)
ACTIVE_OOM_KILLER=1

# When OOM occurs, all processes from the current job on the GPU will be killed
# This frees memory for future allocations
```

**Warning**: Active OOM killer will terminate all processes from the current user/cgroup on the GPU. Use with caution - it's better to let applications handle OOM errors gracefully.

### Multi-Process Jobs

SoftMig tracks memory usage across all processes in a SLURM job:
- Uses a file-backed shared memory region (`$SLURM_TMPDIR/cudevshr.cache.{jobid}`) to coordinate between processes
- All processes in the same job share the same memory limit
- Example: If 2 processes each allocate 6GB on a 12GB limit, the second allocation will fail with OOM

**Example scenario**:
```bash
# Job with 12GB limit
export CUDA_DEVICE_MEMORY_LIMIT=12g

# Process 1 allocates 8GB - succeeds
# Process 2 tries to allocate 6GB - fails with OOM (8GB + 6GB = 14GB > 12GB limit)
```

For example, running `(./gpu_burn -tc 3600 &); (./gpu_burn -tc 3600 &)` with a shared limit means both processes compete for the same memory pool.

## Important Notes

- **SLURM_TMPDIR Integration**: All temporary files (cache, locks) use `$SLURM_TMPDIR` for per-job isolation. This prevents conflicts between concurrent jobs and ensures automatic cleanup when jobs end.
- **Changing limits**: Always delete cache files before setting new limits: `rm -f ${SLURM_TMPDIR}/cudevshr.cache*`
- **Config files**: In SLURM jobs, limits come from secure config files (users cannot modify)
- **Cache files**: Auto-cleaned when job ends (SLURM_TMPDIR is job-specific, unlike original HAMi-core which used shared `/tmp`)
- **CUDA Version**: **CUDA 12+ required** (tested with CUDA 12.2, 12.3, 13.0). CUDA 11 does not work. SoftMig works with all CUDA 12+ versions.
- **SM Limiting**: GPU compute utilization limiting works via kernel launch throttling. Only monitors device 0 (intentional for fractional jobs that receive one GPU).
- **Log directory permissions**: `/var/log/softmig` must be writable by all users (see Logging section)


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

