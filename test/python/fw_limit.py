"""Framework memory-limit check for SoftMig.

Usage: python fw_limit.py <torch|torch_async|tf|torch_fork> <passive|enabled> [target_gib]

Allocates 256 MiB blocks until the framework reports out-of-memory (or the
target is reached in passive mode), frees everything, then allocates again.

  enabled: an OOM must be raised as the framework's normal OOM error before
           the total exceeds the slice limit (CUDA_DEVICE_MEMORY_LIMIT from
           the job's config, passed in SOFTMIG_LIMIT_MIB), and at least half
           the limit must be usable; after freeing, half the limit must be
           allocatable again.
  passive: target_gib (default 16, more than any slice) must be allocatable
           without error.
  torch_fork: CUDA in the parent, a DataLoader with 4 forked workers, and a
           2-process spawn pool that each allocate on the GPU.

Prints "fw_limit: PASS ..." or "fw_limit: FAIL ..." and exits 0/1.
"""
import os
import sys

BLOCK = 256 << 20


def result(ok, msg):
    print(f"fw_limit: {'PASS' if ok else 'FAIL'} {msg}", flush=True)
    sys.exit(0 if ok else 1)


def fill(alloc, is_oom, cap_bytes):
    held, total, oom_msg = [], 0, None
    while total < cap_bytes:
        try:
            held.append(alloc())
            total += BLOCK
        except Exception as e:  # noqa: BLE001 - classify below
            if is_oom(e):
                oom_msg = str(e).splitlines()[0][:160]
                break
            raise
    return held, total, oom_msg


def run_torch(mode, target, async_backend):
    import torch

    dev = torch.device("cuda:0")
    alloc = lambda: torch.empty(BLOCK, dtype=torch.uint8, device=dev)  # noqa: E731
    is_oom = lambda e: isinstance(e, torch.OutOfMemoryError)  # noqa: E731
    return check(mode, target, alloc, is_oom, lambda: torch.cuda.empty_cache(),
                 f"torch {torch.__version__} backend={'cudaMallocAsync' if async_backend else 'native'}")


def run_tf(mode, target):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        result(False, "tensorflow sees no GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

    def alloc():
        with tf.device("/GPU:0"):
            return tf.zeros([BLOCK], dtype=tf.uint8)

    is_oom = lambda e: isinstance(e, tf.errors.ResourceExhaustedError)  # noqa: E731
    return check(mode, target, alloc, is_oom, lambda: None, f"tensorflow {tf.__version__}")


def check(mode, target, alloc, is_oom, release, label):
    limit = int(os.environ.get("SOFTMIG_LIMIT_MIB", "0")) << 20
    cap = (target << 30) if mode == "passive" else (limit * 2 if limit else 64 << 30)
    held, total, oom = fill(alloc, is_oom, cap)
    detail = f"{label} first_fill={total >> 20}MiB oom={'yes' if oom else 'no'}"
    del held
    import gc

    gc.collect()
    release()
    if mode == "passive":
        if oom or total < cap:
            result(False, f"{detail} (passive must reach {target} GiB) {oom or ''}")
        result(True, detail)
    if not limit:
        result(False, f"{detail} SOFTMIG_LIMIT_MIB not set")
    if not oom:
        result(False, f"{detail} no OOM before 2x limit")
    if total > limit or total < limit // 2:
        result(False, f"{detail} limit={limit >> 20}MiB usable out of range [{limit >> 21}, {limit >> 20}]")
    held2, total2, oom2 = fill(alloc, is_oom, limit // 2)
    del held2
    if total2 < limit // 2 - BLOCK:
        result(False, f"{detail} refill only {total2 >> 20}MiB after free ({oom2})")
    result(True, f"{detail} limit={limit >> 20}MiB refill={total2 >> 20}MiB oom_msg='{oom}'")


def _spawn_worker(i):
    import torch

    x = torch.ones(64 << 20, dtype=torch.uint8, device="cuda:0")
    return int(x.sum().item() == (64 << 20))


def run_torch_fork():
    import multiprocessing as mp

    import torch
    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, i):
            return torch.full((1024,), float(i))

    base = torch.ones(32 << 20, device="cuda:0")
    s = 0.0
    for batch in DataLoader(DS(), batch_size=8, num_workers=4, multiprocessing_context="fork"):
        s += float(batch.to("cuda:0").sum().item())
    expect = sum(range(64)) * 1024.0
    with mp.get_context("spawn").Pool(2) as pool:
        spawned = pool.map(_spawn_worker, range(2))
    ok = abs(s - expect) < 1e-3 and spawned == [1, 1] and float(base.sum().item()) == float(32 << 20)
    result(ok, f"torch {torch.__version__} dataloader_fork_sum={s:.0f}/{expect:.0f} spawn={spawned}")


if __name__ == "__main__":
    fw, mode = sys.argv[1], sys.argv[2]
    target = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    if fw == "torch":
        run_torch(mode, target, False)
    elif fw == "torch_async":
        run_torch(mode, target, True)
    elif fw == "tf":
        run_tf(mode, target)
    elif fw == "torch_fork":
        run_torch_fork()
    else:
        result(False, f"unknown framework {fw}")
