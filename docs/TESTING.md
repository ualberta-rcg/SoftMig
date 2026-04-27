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
- `LD_PRELOAD` is intended for development/testing.
