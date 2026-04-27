/**
 * @file utils.h
 * @brief Core utility declarations for locking and CUDA device virtualization.
 */
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>


/** Acquire the file-based unified lock (SLURM-aware). Returns 0 on success, -1 on failure. */
int try_lock_unified_lock();

/** Release the file-based unified lock. Returns 0 on success, -1 on failure. */
int try_unlock_unified_lock();

/** Check if CUDA_VISIBLE_DEVICES needs remapping (env count matches real device count). */
int need_cuda_virtualize();
