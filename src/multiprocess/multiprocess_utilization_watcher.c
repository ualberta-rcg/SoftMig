#include <sys/mman.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stddef.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

#include <cuda.h>
#include "include/nvml_prefix.h"
// Note: We use system <nvml.h> here to avoid conflicts with nvml-subset.h
// The nvml_prefix.h maps nvmlDeviceGetComputeRunningProcesses to _v2 version
#include <nvml.h>
#include <sys/time.h>
#include <sys/wait.h>

#include "multiprocess/multiprocess_memory_limit.h"
#include "multiprocess/multiprocess_utilization_watcher.h"
#include "include/log_utils.h"
#include "include/nvml_override.h"
#include "include/process_utils.h"

// Local versioned struct definition to avoid including nvml-subset.h (which conflicts with system nvml.h)
// This matches the v2 struct layout expected by the driver NVML library
#define NVML_PROCESS_INFO_V1 1
#define NVML_PROCESS_INFO_V2 2
typedef struct {
    unsigned int version;              //!< Structure format version (must be set before API calls)
    unsigned int pid;                  //!< Process ID
    unsigned long long usedGpuMemory;  //!< Amount of used GPU memory in bytes
} nvmlProcessInfo_v2_local_t;


static int g_sm_num;
static int g_max_thread_per_sm;
static volatile long g_cur_cuda_cores = 0;
static volatile long g_total_cuda_cores = 0;
extern int pidfound;
int cuda_to_nvml_map_array[CUDA_DEVICE_MAX_COUNT];

void rate_limiter(int grids, int blocks) {
  long before_cuda_cores = 0;
  long after_cuda_cores = 0;
  long kernel_size = grids;

  while (get_recent_kernel()<0) {
    sleep(1);
  }
  set_recent_kernel(2);
  // Note: Only checks device 0 SM limit - this is intentional since fractional GPU
  // jobs (the ones that need SM limiting) only get a single GPU slice.
  // Full GPU jobs typically have SM_LIMIT=100 (no limit) or 0 (disabled).
  if ((get_current_device_sm_limit(0)>=100) || (get_current_device_sm_limit(0)==0))
    	return;
  if (get_utilization_switch()==0)
      return;
  //if (g_vcuda_config.enable) {
    do {
CHECK:
      before_cuda_cores = g_cur_cuda_cores;
      if (before_cuda_cores < 0) {
        nanosleep(&g_cycle, NULL);
        goto CHECK;
      }
      // Atomically decrement tokens by kernel size (throttles if tokens exhausted)
      after_cuda_cores = before_cuda_cores - kernel_size;
      // Safety: prevent integer underflow (shouldn't happen, but defensive)
      if (after_cuda_cores < -g_total_cuda_cores) {
        after_cuda_cores = -g_total_cuda_cores;
      }
    } while (!CAS(&g_cur_cuda_cores, before_cuda_cores, after_cuda_cores));
  //}
}

static void change_token(long delta) {
  int cuda_cores_before = 0, cuda_cores_after = 0;
  // Atomically adjust token pool by delta (can be positive or negative)
  do {
    cuda_cores_before = g_cur_cuda_cores;
    cuda_cores_after = cuda_cores_before + delta;

    // Cap at maximum pool size
    if (cuda_cores_after > g_total_cuda_cores) {
      cuda_cores_after = g_total_cuda_cores;
    }
    // Note: Negative values are allowed (indicates tokens exhausted, throttling active)
  } while (!CAS(&g_cur_cuda_cores, cuda_cores_before, cuda_cores_after));
}

long delta(int up_limit, int user_current, long share) {
  // Calculate utilization difference (minimum 5% to ensure some adjustment)
  int utilization_diff =
      abs(up_limit - user_current) < 5 ? 5 : abs(up_limit - user_current);
  // Calculate token increment based on SM count, threads per SM, and utilization diff
  // Magic number 2560 is a scaling factor to normalize the increment size
  long increment =
      (long)g_sm_num * (long)g_sm_num * (long)g_max_thread_per_sm * (long)utilization_diff / 2560;

  /* Accelerate cuda cores allocation when utilization vary widely */
  if (utilization_diff > up_limit / 2) {
    // Safety: up_limit is checked to be > 0 before calling this function
    increment = increment * utilization_diff * 2 / (up_limit + 1);
  }

  if (user_current <= up_limit) {
    // Utilization below limit: increase tokens (allow more kernels)
    share = (share + increment) > g_total_cuda_cores ? g_total_cuda_cores
                                                   : (share + increment);
  } else {
    // Utilization above limit: decrease tokens (throttle more)
    share = (share - increment) < 0 ? 0 : (share - increment);
  }

  return share;
}

unsigned int nvml_to_cuda_map(unsigned int nvmldev){
    unsigned int devcount;
    CHECK_NVML_API(nvmlDeviceGetCount_v2(&devcount));
    int i=0;
    for (i=0;i<devcount;i++){
        if (cuda_to_nvml_map(i)==nvmldev)
          return i;
    }
    return -1;
}

unsigned int cuda_to_nvml_map(unsigned int cudadev){
    return cuda_to_nvml_map_array[cudadev];
}

int setspec() {
    CHECK_NVML_API(nvmlInit());
    CHECK_CU_RESULT(cuDeviceGetAttribute(&g_sm_num,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,0));
    CHECK_CU_RESULT(cuDeviceGetAttribute(&g_max_thread_per_sm,CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,0));
    g_total_cuda_cores = g_max_thread_per_sm * g_sm_num * FACTOR;
    return 0;
}

int get_used_gpu_utilization(int *userutil,int *sysprocnum) {
    struct timeval cur;
    size_t microsec;

    int i;
    unsigned int infcount;
    // Use local versioned struct that's compatible with driver NVML v2 API
    // This avoids conflicts between nvml-subset.h and system nvml.h
    nvmlProcessInfo_v2_local_t infos[SHARED_REGION_MAX_PROCESS_NUM];

    unsigned int nvmlCounts;
    CHECK_NVML_API(nvmlDeviceGetCount(&nvmlCounts));
    lock_shrreg();

    int devi,cudadev;
    for (devi=0;devi<nvmlCounts;devi++){
      uint64_t sum=0;
      infcount = SHARED_REGION_MAX_PROCESS_NUM;
      shrreg_proc_slot_t *proc;
      cudadev = nvml_to_cuda_map((unsigned int)(devi));
      if (cudadev<0)
        continue;
      userutil[cudadev] = 0;
      nvmlDevice_t device;
      CHECK_NVML_API(nvmlDeviceGetHandleByIndex(cudadev, &device));

      // CRITICAL: Initialize version field for all structs before calling NVML
      // This ensures compatibility with different driver versions (CUDA 12.2 vs driver 570.195.03)
      for (unsigned int j = 0; j < SHARED_REGION_MAX_PROCESS_NUM; j++) {
          infos[j].version = NVML_PROCESS_INFO_V2;
          infos[j].pid = 0;
          infos[j].usedGpuMemory = 0;
      }

      //Get Memory for container
      // Note: This goes through our hook (nvmlDeviceGetComputeRunningProcesses_v2) which expects versioned structs.
      // Cast to match system header signature - the memory layout is compatible since v2 struct
      // is just v1 struct with a version field prepended: {version, pid, usedGpuMemory}
      nvmlReturn_t res = nvmlDeviceGetComputeRunningProcesses(device,&infcount,(nvmlProcessInfo_v1_t *)infos);
      if (res == NVML_ERROR_INSUFFICIENT_SIZE) {
        LOG_WARN("get_used_gpu_utilization: Device %d - Buffer too small! NVML returned %u processes but buffer size is %u. Some processes may be missing.", 
                 cudadev, infcount, SHARED_REGION_MAX_PROCESS_NUM);
      } else if (res != NVML_SUCCESS) {
        LOG_WARN("get_used_gpu_utilization: Device %d - nvmlDeviceGetComputeRunningProcesses failed with error %d", cudadev, res);
      }
      if (res == NVML_SUCCESS || res == NVML_ERROR_INSUFFICIENT_SIZE) {
        for (i=0; i<infcount; i++){
          // CRITICAL: Use safe PID extraction to handle struct mismatches
          unsigned int actual_pid = extract_pid_safely((void *)&infos[i]);
          if (actual_pid == 0) {
            continue;  // Skip if we can't get a valid PID
          }
          proc = find_proc_by_hostpid(actual_pid);
          if (proc != NULL){
              // Extract memory value safely - handles struct mismatches where PID is at wrong offset
              proc->monitorused[cudadev] = extract_memory_safely((void *)&infos[i], actual_pid, infos[i].pid);
          }
        }
      }
      // Get SM util for container
      gettimeofday(&cur,NULL);
      microsec = (cur.tv_sec - 1) * 1000UL * 1000UL + cur.tv_usec;
      nvmlProcessUtilizationSample_t processes_sample[SHARED_REGION_MAX_PROCESS_NUM];
      unsigned int processes_num = SHARED_REGION_MAX_PROCESS_NUM;
      res = nvmlDeviceGetProcessUtilization(device,processes_sample,&processes_num,microsec);
      if (res == NVML_ERROR_INSUFFICIENT_SIZE) {
        LOG_WARN("get_used_gpu_utilization: Device %d - Process utilization buffer too small! NVML returned %u processes but buffer size is %u. Some processes may be missing.", 
                 cudadev, processes_num, SHARED_REGION_MAX_PROCESS_NUM);
      } else if (res != NVML_SUCCESS) {
        LOG_WARN("get_used_gpu_utilization: Device %d - nvmlDeviceGetProcessUtilization failed with error %d", cudadev, res);
      }
      if (res == NVML_SUCCESS || res == NVML_ERROR_INSUFFICIENT_SIZE) {
        for (i=0; i<processes_num; i++){
          proc = find_proc_by_hostpid(processes_sample[i].pid);
          if (proc != NULL){
              sum += processes_sample[i].smUtil;
              proc->device_util[cudadev].sm_util = processes_sample[i].smUtil;
          }
        }
      }
      if (sum < 0)
        sum = 0;
      userutil[cudadev] = sum;
    }
    unlock_shrreg();
    return 0;
}

void* utilization_watcher() {
    nvmlInit();
    int userutil[CUDA_DEVICE_MAX_COUNT];
    int sysprocnum;
    long share = 0;
    // Note: Only monitors device 0 - this is intentional since fractional GPU jobs
    // (the ones that need SM limiting) only get a single GPU slice. Full GPU jobs
    // typically have SM_LIMIT=100 (no limit) or 0 (disabled), so they don't need monitoring.
    int upper_limit = get_current_device_sm_limit(0);
    ensure_initialized();
    LOG_DEBUG("upper_limit=%d\n",upper_limit);
    
    // Memory monitoring: run every ~5 seconds (42 iterations of 120ms = 5.04 seconds)
    unsigned int memory_check_counter = 0;
    const unsigned int MEMORY_CHECK_INTERVAL = 42;  // ~5 seconds
    
    while (1){
        nanosleep(&g_wait, NULL);
        if (pidfound==0) {
          update_host_pid();
          if (pidfound==0)
            continue;
        }
        init_gpu_device_utilization();
        get_used_gpu_utilization(userutil,&sysprocnum);
        //if (sysprocnum == 1 &&
        //    userutil < upper_limit / 10) {
        //    g_cur_cuda_cores =
        //        delta(upper_limit, userutil, share);
        //    continue;
        //}
        // Safety mechanism: if tokens exhausted but share is at max, double the pool
        // This prevents deadlock if initial pool size was too small
        if ((share==g_total_cuda_cores) && (g_cur_cuda_cores<0)) {
          LOG_WARN("utilization_watcher: Tokens exhausted (g_cur_cuda_cores=%ld) but share at max (%ld), doubling pool size", 
                   g_cur_cuda_cores, share);
          g_total_cuda_cores *= 2;
          share = g_total_cuda_cores;
        }
        // Only adjust tokens if utilization is valid (0-100%)
        // Only monitor device 0 (see note above)
        if ((userutil[0]<=100) && (userutil[0]>=0)){
          share = delta(upper_limit, userutil[0], share);
          change_token(share);
        }
        // Log utilization info every ~5 seconds (42 iterations = 5.04 seconds) - INFO level for console (level >= 3)
        static unsigned int util_log_counter = 0;
        if (++util_log_counter >= MEMORY_CHECK_INTERVAL) {
          util_log_counter = 0;
          LOG_INFO("utilization_watcher[5s]: userutil=%d%% currentcores=%ld total=%ld limit=%d%% share=%ld",userutil[0],g_cur_cuda_cores,g_total_cuda_cores,upper_limit,share);
        }
        
        // Memory monitoring: check every ~5 seconds
        memory_check_counter++;
        if (memory_check_counter >= MEMORY_CHECK_INTERVAL) {
            memory_check_counter = 0;
            LOG_FILE_DEBUG("utilization_watcher[5s]: Starting memory check cycle - userutil[0]=%d%% currentcores=%ld share=%ld", 
                    userutil[0], g_cur_cuda_cores, share);
            
            // Only check memory if softmig is enabled and memory limits are configured
            // Check if CUDA_DEVICE_MEMORY_LIMIT is set (similar to how SM limit is checked)
            extern int is_softmig_configured(void);
            if (is_softmig_configured()) {
                // Get number of devices
                unsigned int nvml_devices_count;
                nvmlReturn_t ret = nvmlDeviceGetCount_v2(&nvml_devices_count);
                if (ret == NVML_SUCCESS) {
                    // Check all devices for memory OOM
                    for (unsigned int dev_idx = 0; dev_idx < nvml_devices_count; dev_idx++) {
                        // Map NVML device index to CUDA device index
                        int cuda_dev = -1;
                        for (int i = 0; i < CUDA_DEVICE_MAX_COUNT; i++) {
                            if (cuda_to_nvml_map(i) == dev_idx) {
                                cuda_dev = i;
                                break;
                            }
                        }
                        
                        if (cuda_dev < 0) {
                            continue;  // Skip if device mapping not found
                        }
                        
                        // Get memory limit for this device
                        uint64_t limit = get_current_device_memory_limit(cuda_dev);
                        if (limit == 0) {
                            continue;  // No limit set for this device, skip
                        }
                        
                        // Get current memory usage (summed from NVML, filtered by cgroup/UID)
                        uint64_t usage = get_summed_device_memory_usage_from_nvml(cuda_dev);
                        if (usage == 0) {
                            // Fallback to tracked usage if NVML query failed
                            lock_shrreg();
                            usage = get_gpu_memory_usage_nolock(cuda_dev);
                            unlock_shrreg();
                        }
                        
                        LOG_DEBUG("memory_watcher: Device %d (CUDA %d) - usage=%llu limit=%llu", 
                                 dev_idx, cuda_dev, (unsigned long long)usage, (unsigned long long)limit);
                        
                        // Check if over limit
                        if (usage > limit) {
                            LOG_ERROR("memory_watcher: OOM detected on device %d (CUDA %d) - usage %llu exceeds limit %llu", 
                                     dev_idx, cuda_dev, (unsigned long long)usage, (unsigned long long)limit);
                            
                            // Get job ID for syslog
                            char* job_id = getenv("SLURM_JOB_ID");
                            
                            // Log to syslog
                            char syslog_msg[512];
                            if (job_id != NULL) {
                                snprintf(syslog_msg, sizeof(syslog_msg), 
                                         "Job %s: GPU Memory OOM on device %d - usage %llu bytes (%.2f GB) exceeds limit %llu bytes (%.2f GB)",
                                         job_id, cuda_dev,
                                         (unsigned long long)usage, usage / (1024.0 * 1024.0 * 1024.0),
                                         (unsigned long long)limit, limit / (1024.0 * 1024.0 * 1024.0));
                            } else {
                                snprintf(syslog_msg, sizeof(syslog_msg), 
                                         "GPU Memory OOM on device %d - usage %llu bytes (%.2f GB) exceeds limit %llu bytes (%.2f GB)",
                                         cuda_dev,
                                         (unsigned long long)usage, usage / (1024.0 * 1024.0 * 1024.0),
                                         (unsigned long long)limit, limit / (1024.0 * 1024.0 * 1024.0));
                            }
                            
                            char cmd[1024];
                            snprintf(cmd, sizeof(cmd), "logger -t softmig '%s'", syslog_msg);
                            (void)system(cmd);  // Ignore return value - logging failure is non-critical
                            LOG_ERROR("OOM syslog: %s", syslog_msg);
                            
                            // Trigger gradual OOM killer
                            gradual_oom_killer(cuda_dev);
                        } else {
                            // Log memory usage as INFO (useful for users) - shows on console at level >= 3
                            LOG_INFO("utilization_watcher[5s]: Device %d (CUDA %d) - memory OK: usage=%.2f GB limit=%.2f GB", 
                                    dev_idx, cuda_dev, 
                                    usage / (1024.0 * 1024.0 * 1024.0),
                                    limit / (1024.0 * 1024.0 * 1024.0));
                        }
                    }
                }
            }
            LOG_FILE_DEBUG("utilization_watcher[5s]: Completed memory check cycle");
        }
    }
}

void init_utilization_watcher() {
    LOG_DEBUG("init_utilization_watcher: core utilization limit set to %d%%",get_current_device_sm_limit(0));
    setspec();
    pthread_t tid;
    if ((get_current_device_sm_limit(0)<=100) && (get_current_device_sm_limit(0)>0)){
        pthread_create(&tid, NULL, utilization_watcher, NULL);
    }
    return;
}

