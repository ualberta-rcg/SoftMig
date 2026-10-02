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

## Pressure campaign: sets of jobs judged against root truth

The suites in this section run *sets* of jobs on the reservation node and
grade every job against what root sees, not what the job reports about
itself. `test/node_sampler.sh start|stop OUT` runs on the node over
`sudo ssh` (from the login node, outside any job) and records
`nvidia-smi pmon -s um`, `--query-compute-apps` and a PID -> Slurm-job map
from `/proc/<pid>/cgroup` once a second. `test/share_lib.sh` launches jobs
(`run_job`, `run_array`), writes an `expect.txt` per job (`limit`, `sm`,
`min_mem`, `min_sm`, `max_sm`, `oom`, `view_total`, `rc`) and
`test/share_report.py` produces `jobs.tsv` with a verdict per job and per
scenario: `PASS`, `LEAK` (another job's process was visible; nothing else
wrong), `FAIL`, `PARTIAL` (jobs landed on more than one GPU, so the share
numbers mean little), `INFO`. Shard jobs are packed onto one GPU with a
3-GPU blocker job (`pack_blocker 3`).

| Suite | Scenarios | Checks |
|---|---|---|
| `suite_share.sh` | S1-S7 | 2/4 quarters, 2 halves, half+2 quarters share SM and memory; OOM neighbour; late joiner; nvidia-smi view |
| | G1-G6 | 4 quarters each allocate their full limit; late victim gets its limit; fixed work loses < 40 % with 3 neighbours; 60/40; allocation-storm neighbour; SIGSTOP lock-hog neighbour |
| | W1 | `nvidia-smi-hook.sh` on `PATH` hides the other job (table, pmon, query) |
| `suite_passthrough.sh` | P1-P4 | whole-GPU job: no conf, `view_total` = card, > 80 % SM, > 20 GB, while a quarter next to it is enforced; two-GPU job; over-allocating quarter beside a whole-GPU holder; whole + half + quarter |
| `suite_mixed_pieces.sh` | M1-M6 | `.4:2` + 2x`.4`; `.4:3` + `.4`; mixed-size request (INFO); 4-task array; two `srun` steps in one `.4:2` job; whole + half + two quarters |
| `suite_gpuburn.sh` | B1-B5 | real `~/gpu-burn`: `gb && gb` in one job, x3 concurrently, x4 jobs, next to PyTorch, whole vs sliced; every PID registered, `GPU 0: OK`, no `FAULTY` |
| `suite_isolation.sh` | | same uid, two jobs: no cross-job accounting, OOM killer stays in its job, foreign-uid and region checks (root side with `SOFTMIG_TEST_SUDO=1`) |
| `suite_bypass.sh` | | `bypass_probe.cu`: every allocation API vs root usage -> `CAPPED` / `LEAK` / `UNHOOKED`; `PASSTHROUGH` on a whole GPU |
| `suite_storm.sh` | A/B/C | random churn of ten job kinds for N minutes (`STORM_MINUTES`, `STORM_PAR`, `STORM_SEED`); B = 8 parallel `suite_nvsmi.sh` + 4 burners |

Drivers:

```bash
umask 002
bash test/run_sets.sh                     # whole campaign, all CUDA versions, ~2 h
SETS=share,gpuburn bash test/run_sets.sh  # subset
cat $(cat test_results/sets_latest.txt)/SUMMARY.md

HOURS=8 nohup bash test/run_overnight.sh > test_results/overnight_run.log 2>&1 &
cat $(cat test_results/overnight_latest.txt)/SUMMARY.md   # in the morning
```

`run_overnight.sh` starts one shared root sampler and submits
`overnight_driver.sh` as a CPU-only job on the reservation; the driver unsets
`SLURM_*` and keeps launching cycles (direct/oom/sm per version, share,
passthrough, mixed, gpuburn, isolation, soak, storm A and B) until the hours
are up. Rules learned the hard way: never reuse an `OUT` directory (NFS keeps
the deleted inode on the node), always give the sampler an absolute path,
never run two campaigns at once on the node (they would share GPUs and the
share numbers become meaningless), and `sudo` is not available inside jobs,
so the sampler must be started from the login node. Real `gpu_burn` binaries
are built once per CUDA version with `test/build_gpuburn.sh`.

`test/audit_hooks.sh [libsoftmig.so]` (on a GPU node) compares the driver's
exported entry points with SoftMig's hooks and exits non-zero if a memory,
launch, meminfo or NVML process-query entry point is unhooked and not on the
acknowledged list. Run it after every driver upgrade.
