# SoftMig Testing

This repo includes C/CUDA tests and lightweight probe binaries. Most tests require a GPU node.

## Build

```bash
rm -rf build
./build.sh
```

## Minimal smoke test (GPU node)

```bash
cd build
export CUDA_DEVICE_MEMORY_LIMIT=4G
export LD_PRELOAD=./libsoftmig.so

# Should report a limited total memory for the job
nvidia-smi --query-gpu=memory.total --format=csv,noheader
```

## C/CUDA tests

```bash
cd build/test
export CUDA_DEVICE_MEMORY_LIMIT=4G
export LD_PRELOAD=../libsoftmig.so

./test_alloc
./test_alloc_host
./test_alloc_managed
./test_runtime_alloc
./test_runtime_launch
```

## Python framework tests

```bash
cd build/test/python
export CUDA_DEVICE_MEMORY_LIMIT=4G
export LD_PRELOAD=../../libsoftmig.so

python limit_pytorch.py
python limit_tensorflow.py
python limit_tensorflow2.py
python limit_mxnet.py
```

## SLURM-based smoke test (example)

```bash
srun --partition=gpu --gres=gpu:l40s.4:1 --time=0:02:00 bash -lc 'nvidia-smi --query-gpu=memory.total --format=csv,noheader'
```

Notes:

- In production deployments, limits are expected to come from `/var/run/softmig/*.conf` (created by prolog).
- Inside a SLURM job the `CUDA_DEVICE_*` environment variables are ignored; the
  examples above that export them only enable SoftMig outside SLURM.
- `LD_PRELOAD` is intended for development/testing.

## Cluster test matrix (softmig reservation)

`test/run_matrix.sh` runs every suite for CUDA 12.2/12.6/12.9/13.2 on slice
and full-GPU allocations and writes `test_results/matrix_<ts>/summary.tsv`.
The `passive` suite runs `build/test/passive_probe`: in a full-GPU job every
`dlsym`/`cuGetProcAddress` result must be the driver's own function, async
alloc/free through the per-thread stream must work, and no shared region or
signal handler may appear; in a slice job the same probe must see hooks.
`test/jax_cuda_async.sh` is the JAX `cuda_async` end-to-end check.

One-off suites added in 2.06 (run by `run_matrix.sh` on CUDA 12.6):
`stress` (16/32/64 processes x 4 threads allocating concurrently, plus an
enforcement round checked against real per-process NVML usage), `fault`
(SIGKILL and SIGSTOP of workers while they contend for the region lock),
`frameworks` (PyTorch native and cudaMallocAsync, TensorFlow, DataLoader
fork and spawn; slice and full GPU; venv in `$SCRATCH/softmig-fwvenv`),
`fork`, `multigpu`, `array`, `security`, `container` and `overhead`. The last
two report `INFO`. `build/test/shrreg_check` validates a job's shared region.

Every suite arms an in-job watchdog; a hang is reported as `HANG` with
per-thread state in `OUT/hang_<pid>.txt`. Run the matrix with
`SOFTMIG_HANG_GDB_SUDO=1` to also get root gdb stacks (`OUT/gdb_<pid>.txt`)
from the reservation node, and with `SOFTMIG_TEST_SUDO=1` for the root-side
`security` cases. All of this runs as jobs on the reservation; nothing is
tested on the login node.

`test/audit_hooks.sh [libsoftmig.so]` (on a GPU node) compares the driver's
exported entry points with SoftMig's hooks and exits non-zero if a memory,
launch, meminfo or NVML process-query entry point is unhooked and not on the
acknowledged list. Run it after every driver upgrade.
