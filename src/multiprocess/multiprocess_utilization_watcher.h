/**
 * @file multiprocess_utilization_watcher.h
 * @brief SM utilization rate limiter and background utilization monitoring thread.
 *
 * Implements a token-bucket rate limiter that throttles CUDA kernel launches
 * to enforce SM utilization limits. A background watcher thread periodically
 * queries NVML for per-process utilization and adjusts the token pool.
 */
#define CAS(ptr, old, new) __sync_bool_compare_and_swap((ptr), (old), (new))

#define MILLISEC (1000UL * 1000UL)

#define TIME_TICK (10)

static const struct timespec g_cycle = {
    .tv_sec = 0,
    .tv_nsec = TIME_TICK * MILLISEC,
};

static const struct timespec g_wait = {
    .tv_sec = 0,
    .tv_nsec = 120 * MILLISEC,
};


/** Block until enough SM tokens are available, then consume them for this kernel launch. */
void rate_limiter(int grids, int blocks);

/** Spawn the background utilization watcher thread (if SM limit < 100%). */
void init_utilization_watcher();

/** Background thread: polls NVML utilization, adjusts token pool, monitors memory OOM. */
void* utilization_watcher();

/** Query GPU SM count and max threads per SM, initialize the token pool size. */
int setspec();