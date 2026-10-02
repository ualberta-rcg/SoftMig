/**
 * @file multiprocess_memory_limit.h
 * @brief Shared memory region, per-process memory tracking, and OOM killer API.
 *
 * Defines the mmap-backed shared_region_t structure that coordinates GPU memory
 * limits and SM utilization across co-located processes in a SLURM job.
 * Processes register via ensure_initialized() and communicate usage through
 * semaphore-protected shared memory slots.
 */
#ifndef __MULTIPROCESS_MEMORY_LIMIT_H__
#define __MULTIPROCESS_MEMORY_LIMIT_H__

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <pthread.h>

#include "static_config.h"
#include "include/log_utils.h"


#define MULTIPROCESS_SHARED_REGION_MAGIC_FLAG  19920718
#define MULTIPROCESS_SHARED_REGION_CACHE_ENV   "CUDA_DEVICE_MEMORY_SHARED_CACHE"
#define MULTIPROCESS_SHARED_REGION_CACHE_DEFAULT  "/tmp/cudevshr.cache"
#define ENV_OVERRIDE_FILE "/overrideEnv"

#define CUDA_DEVICE_MAX_COUNT 16
#define CTX_ACTIVATE_SIZE 32
#define CUDA_DEVICE_MEMORY_UPDATE_SUCCESS 0
#define CUDA_DEVICE_MEMORY_UPDATE_FAILURE 1

#define SHARED_REGION_SIZE_MAGIC  sizeof(shared_region_t)
#define SHARED_REGION_MAX_PROCESS_NUM 1024

// macros for debugging
#define SEQ_ACQUIRE_SEMLOCK_OK 3
#define SEQ_UPDATE_OWNER_OK 4
#define SEQ_RESET_OWNER_OK 5
#define SEQ_RELEASE_SEMLOCK_OK 6
#define SEQ_BEFORE_UNLOCK_SHRREG 7

#define SEQ_AFTER_INC 8
#define SEQ_AFTER_DEC 9

#ifndef SEQ_POINT_MARK
    #define SEQ_POINT_MARK(s) 
#endif

#define FACTOR 32

// 2.0: robust process-shared mutex replaces sem_t + owner_pid recovery.
// The major version is part of the region file name, so processes from
// different layouts (e.g. a job spanning a library redeploy) never share
// a region.
#define MAJOR_VERSION 2
#define MINOR_VERSION 0

typedef struct {
    uint64_t context_size;
    uint64_t module_size;
    uint64_t data_size;
    uint64_t offset;
    uint64_t total;
    uint64_t unused[3];
} device_memory_t;

typedef struct {
    uint64_t dec_util;
    uint64_t enc_util;
    uint64_t sm_util;
    uint64_t unused[3];
} device_util_t;

typedef struct {
    int32_t pid;
    int32_t hostpid;
    device_memory_t used[CUDA_DEVICE_MAX_COUNT];
    uint64_t monitorused[CUDA_DEVICE_MAX_COUNT];
    device_util_t device_util[CUDA_DEVICE_MAX_COUNT];
    uint64_t pending[CUDA_DEVICE_MAX_COUNT];  // admitted, driver allocation in flight (NVML index)
    int32_t status;
    uint64_t unused[3];
} shrreg_proc_slot_t;

typedef char uuid[96];

typedef struct {
    int32_t initialized_flag;
    uint32_t major_version;
    uint32_t minor_version;
    int32_t sm_init_flag;
    size_t owner_pid;       // diagnostic only (who holds lock); never used for recovery
    pthread_mutex_t lock;   // PTHREAD_PROCESS_SHARED | PTHREAD_MUTEX_ROBUST | ERRORCHECK
    uint64_t device_num;
    uuid uuids[CUDA_DEVICE_MAX_COUNT];
    uint64_t limit[CUDA_DEVICE_MAX_COUNT];
    uint64_t sm_limit[CUDA_DEVICE_MAX_COUNT];
    shrreg_proc_slot_t procs[SHARED_REGION_MAX_PROCESS_NUM];
    int proc_num;
    int utilization_switch;
    int recent_kernel;
    int priority;
    uint64_t last_kernel_time;
    uint64_t unused[4];
} shared_region_t;

typedef struct {
    int32_t pid;
    int fd;
    pthread_once_t init_status;
    shared_region_t* shared_region; 
    uint64_t last_kernel_time; // cache for current process
} shared_region_info_t;


typedef struct {
  size_t tid;
  CUcontext ctx;
} thread_context_map;

/** Initialize the shared region (once per process, thread-safe via pthread_once). */
void ensure_initialized();

/** Get SM utilization limit (percentage) for a CUDA device. Returns 100 if disabled. */
int get_current_device_sm_limit(int dev);

/** Get memory limit in bytes for a CUDA device. Returns 0 if no limit. */
uint64_t get_current_device_memory_limit(const int dev);

/** Set memory limit in bytes for a CUDA device in the shared region. */
int set_current_device_memory_limit(const int dev,size_t newlimit);
int set_current_device_sm_limit_scale(int dev,int scale);

/** Scan shared region to check if current process has a registered host PID. */
int update_host_pid();

/** Register the NVML-visible host PID for the current process. */
int set_host_pid(int hostpid);

uint64_t get_current_device_memory_monitor(const int dev);
uint64_t get_current_device_memory_usage(const int dev);
size_t get_gpu_memory_usage(const int dev);
// Get memory usage without locking (caller must hold lock_shrreg)
size_t get_gpu_memory_usage_nolock(const int dev);

/** Bytes admitted by softmig_reserve() whose driver allocation has not
 *  completed yet, summed over all processes (NVML device index; caller holds
 *  lock_shrreg). */
uint64_t get_pending_memory_nolock(const int nvmldev);

/** Adjust this process's pending bytes on a CUDA device (caller holds lock_shrreg). */
void adjust_pending_memory_nolock(int cudadev, int64_t delta);
// Get summed memory usage from NVML for a CUDA device (cgroup/UID filtered, raw per-process values)
uint64_t get_summed_device_memory_usage_from_nvml(int cuda_dev);

// Priority-related
int get_current_priority();
int set_recent_kernel(int value);
int get_recent_kernel();
int get_utilization_switch();

int set_gpu_device_memory_monitor(int32_t pid,int dev,size_t monitor);
int set_gpu_device_sm_utilization(int32_t pid,int dev, unsigned int smUtil);
int init_gpu_device_utilization();
int add_gpu_device_memory_usage(int32_t pid,int dev,size_t usage,int type);
int rm_gpu_device_memory_usage(int32_t pid,int dev,size_t usage,int type);

/** Look up a process slot by its NVML-visible host PID (lazy-registers if needed). */
shrreg_proc_slot_t *find_proc_by_hostpid(int hostpid);

/**
 * Kill all GPU processes belonging to the current cgroup/UID on all devices.
 * Only effective for non-root users. Returns number of processes killed.
 */
int active_oom_killer();

/**
 * Gradually kill processes on a specific CUDA device, newest-PID first,
 * until memory usage drops below the limit or the cgroup is killed.
 * @return Number of processes killed, or -1 on error.
 */
int gradual_oom_killer(int cuda_dev);

/** Record kernel launch timestamp for utilization tracking (rate-limited). */
void pre_launch_kernel();

int shrreg_major_version();
int shrreg_minor_version();
int init_device_info();

/** Acquire the shared region lock. Re-entrant per thread; waits for a live
 *  holder indefinitely (WARN every 15 s) and recovers via EOWNERDEAD if the
 *  holder died. */
void lock_shrreg();

/** Release the shared region lock. */
void unlock_shrreg();

/** Path of this job's shared region file (also used by test/shrreg_check). */
const char *shrreg_default_path(char *buf, size_t len);

//Setspec of the corresponding device
int setspec();
//Remove quitted process

void suspend_all();
void resume_all();
int wait_status_self(int status);
int wait_status_all(int status);

/** Load environment variables from a key=value file (legacy, skipped in SLURM jobs). */
int load_env_from_file(char *filename);

/** Map an NVML device index to its CUDA device index. Returns -1 if not found. */
int nvml_to_cuda_map(unsigned int nvmldev);

/** Map a CUDA device index to its NVML device index. */
unsigned int cuda_to_nvml_map(unsigned int cudadev);

int clear_proc_slot_nolock(int);
#endif  // __MULTIPROCESS_MEMORY_LIMIT_H__

