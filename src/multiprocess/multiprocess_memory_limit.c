/**
 * @file multiprocess_memory_limit.c
 * @brief Shared memory region management, OOM killer, and per-device limit enforcement.
 *
 * Creates and maps an mmap-backed shared region (per SLURM job) that tracks
 * per-process GPU memory usage, SM utilization, and device limits. Provides
 * the semaphore-protected lock_shrreg/unlock_shrreg API, the active and
 * gradual OOM killers (cgroup/UID-aware, disabled by default and gated by the
 * SOFTMIG_ENABLE_OOM_KILLER env / config flag), and NVML-based memory
 * summation using the raw per-process usedGpuMemory values.
 */
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

#include <assert.h>
#include <cuda.h>
// Prevent system <nvml.h> from being included - we use nvml-subset.h instead
// This macro tells nvml.h (if included) to skip some definitions
#define NVML_NO_UNVERSIONED_FUNC_DEFS
// Include nvml-subset.h FIRST - it defines structures we need
#include "include/nvml-subset.h"
#include "include/nvml_prefix.h"
#include "include/libnvml_hook.h"
#include "include/nvml_override.h"

#include "include/process_utils.h"
#include "include/memory_limit.h"
#include "multiprocess/multiprocess_memory_limit.h"
#include "include/softmig_mode.h"

// Note: We need to bypass the hook to get ALL processes, then filter ourselves
// This ensures we don't miss any processes due to hook filtering or buffer limits
// Use weak symbol so it's NULL if nvml_mod not linked, allowing fallback
extern entry_t nvml_library_entry[] __attribute__((weak));

// Shared NVML memory summation (defined in nvml/hook.c, weak for shrreg-tool)
extern uint64_t sum_process_memory_from_nvml(nvmlDevice_t device) __attribute__((weak));

// Forward declarations for NVML functions (provided by hooks)
const char *nvmlErrorString(nvmlReturn_t result);
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount);
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length);
nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);


#ifndef SEM_WAIT_TIME
#define SEM_WAIT_TIME 3
#endif

#ifndef SEM_WAIT_TIME_ON_EXIT
#define SEM_WAIT_TIME_ON_EXIT 10
#endif

#ifndef SEM_WAIT_RETRY_TIMES
#define SEM_WAIT_RETRY_TIMES 5
#endif

int pidfound;

int ctx_activate[CTX_ACTIVATE_SIZE];

static shared_region_info_t region_info = {0, -1, PTHREAD_ONCE_INIT, NULL, 0};
//size_t initial_offset=117440512;
int env_utilization_switch;
int enable_active_oom_killer;
size_t context_size;
size_t initial_offset=0;
// External function from config_file.c - returns 1 if SOFTMIG_ENABLE_OOM_KILLER
// is set (env or per-job config), 0 otherwise. Default is OOM killer disabled.
extern int get_softmig_oom_killer_enabled(void);

static int is_softmig_enabled(void) {
    return !softmig_is_passive();
}
//lock for record kernel time
pthread_mutex_t _kernel_mutex;
int _record_kernel_interval = 1;

// forwards

void do_init_device_memory_limits(uint64_t*, int);
void exit_withlock(int exitcode);

void set_current_gpu_status(int status){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;  // No-op when softmig is disabled
    }
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++)
        if (getpid()==region_info.shared_region->procs[i].pid){
            region_info.shared_region->procs[i].status = status;
            return;
        }
}

void sig_restore_stub(int signo){
    set_current_gpu_status(1);
}

void sig_swap_stub(int signo){
    set_current_gpu_status(2);
}


// External function from config_file.c - reads from config file or env
extern size_t get_limit_from_config_or_env(const char* env_name);

// get device memory from config file (priority) or env (fallback)
// This is now a wrapper that calls the config file reader
size_t get_limit_from_env(const char* env_name) {
    return get_limit_from_config_or_env(env_name);
}

int init_device_info() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    unsigned int i,nvmlDevicesCount;
    CHECK_NVML_API(nvmlDeviceGetCount_v2(&nvmlDevicesCount));
    region_info.shared_region->device_num=nvmlDevicesCount;
    nvmlDevice_t dev;
    for(i=0;i<nvmlDevicesCount;i++){
        CHECK_NVML_API(nvmlDeviceGetHandleByIndex(i, &dev));
        CHECK_NVML_API(nvmlDeviceGetUUID(dev,region_info.shared_region->uuids[i],NVML_DEVICE_UUID_V2_BUFFER_SIZE));
    }
    return 0;
}


int load_env_from_file(char *filename) {
    if (getenv("SLURM_JOB_ID") != NULL) {
        // In SLURM jobs, limits must come from /var/run/softmig/*.conf.
        // Ignore legacy env-file injection paths.
        return 0;
    }
    FILE *f=fopen(filename,"r");
    if (f==NULL)
        return 0;
    char tmp[10000];
    while (fgets(tmp, sizeof(tmp), f) != NULL) {
        if (strstr(tmp,"==")==NULL && strstr(tmp,"=")!=NULL) {
            size_t len = strlen(tmp);
            if (len > 0 && tmp[len-1]=='\n') tmp[len-1]='\0';
            for (int cursor=0; tmp[cursor]!='\0'; cursor++) {
                if (tmp[cursor]=='=') {
                    tmp[cursor]='\0';
                    setenv(tmp, tmp+cursor+1, 1);
                    break;
                }
            }
        }
    }
    fclose(f);
    return 0;
}

void do_init_device_memory_limits(uint64_t* arr, int len) {
    size_t fallback_limit = get_limit_from_env(CUDA_DEVICE_MEMORY_LIMIT);
    int i;
    for (i = 0; i < len; ++i) {
        char env_name[CUDA_DEVICE_MEMORY_LIMIT_KEY_LENGTH] = CUDA_DEVICE_MEMORY_LIMIT;
        char index_name[12];
        snprintf(index_name, 12, "_%d", i);
        strcat(env_name, index_name);
        size_t cur_limit = get_limit_from_env(env_name);
        if (cur_limit > 0) {
            arr[i] = cur_limit;
        } else if (fallback_limit > 0) {
            arr[i] = fallback_limit;
        } else {
            arr[i] = 0;
        }
    }
}

void do_init_device_sm_limits(uint64_t *arr, int len) {
    size_t fallback_limit = get_limit_from_env(CUDA_DEVICE_SM_LIMIT);
    if (fallback_limit == 0) fallback_limit = 100;
    int i;
    for (i = 0; i < len; ++i) {
        char env_name[CUDA_DEVICE_SM_LIMIT_KEY_LENGTH] = CUDA_DEVICE_SM_LIMIT;
        char index_name[12];
        snprintf(index_name, 12, "_%d", i);
        strcat(env_name, index_name);
        size_t cur_limit = get_limit_from_env(env_name);
        if (cur_limit > 0) {
            arr[i] = cur_limit;
        } else if (fallback_limit > 0) {
            arr[i] = fallback_limit;
        } else {
            arr[i] = 0;
        }
    }
}

int active_oom_killer() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    
    // Get current user's UID for fallback filtering
    uid_t current_uid = getuid();
    int is_root = (current_uid == 0);
    
    // Root user is disabled from OOM killing - only non-root users get this treatment
    if (is_root) {
        LOG_DEBUG("active_oom_killer: Root user (UID 0) - OOM killer disabled, no processes killed");
        return 0;
    }
    
    LOG_ERROR("active_oom_killer: OOM detected - killing processes from current cgroup/UID %u", current_uid);
    
    // Query NVML for all processes on all devices (same approach as memory counting)
    unsigned int nvml_devices_count;
    nvmlReturn_t ret = nvmlDeviceGetCount_v2(&nvml_devices_count);
    if (ret != NVML_SUCCESS) {
        LOG_ERROR("active_oom_killer: Failed to get device count: %d (%s)", ret, nvmlErrorString(ret));
        // Fallback: kill processes in shared region, but still verify they belong to current cgroup/UID
        int i;
        int fallback_killed = 0;
        for (i=0;i<region_info.shared_region->proc_num;i++) {
            int32_t pid = region_info.shared_region->procs[i].pid;
            
            // Verify process belongs to current cgroup/UID before killing (same filtering as main path)
            int should_kill = 0;
            int cgroup_check = proc_belongs_to_current_cgroup_session(pid);
            
            if (cgroup_check == 1) {
                // Same cgroup - verify UID for extra safety
                uid_t proc_uid = proc_get_uid(pid);
                if (proc_uid != (uid_t)-1 && proc_uid == current_uid) {
                    should_kill = 1;
                } else {
                    LOG_DEBUG("active_oom_killer: Fallback - skipping PID %d (same cgroup but different UID %u != current UID %u)", 
                             pid, proc_uid, current_uid);
                }
            } else if (cgroup_check == -1) {
                uid_t proc_uid = proc_get_uid(pid);
                if (proc_uid != (uid_t)-1 && proc_uid == current_uid) {
                    should_kill = 1;
                }
            }
            
            if (should_kill && proc_alive(pid) == PROC_STATE_ALIVE) {
                LOG_WARN("active_oom_killer: Fallback - killing PID %d from shared region (NVML query failed, verified cgroup/UID)", pid);
                kill(pid, SIGKILL);
                fallback_killed++;
            } else {
                LOG_DEBUG("active_oom_killer: Fallback - skipping PID %d (not in current cgroup/UID or already dead)", pid);
            }
        }
        LOG_ERROR("active_oom_killer: Fallback killed %d processes from shared region", fallback_killed);
        return fallback_killed;
    }
    
    int total_killed = 0;
    
    // Iterate through all devices
    for (unsigned int dev_idx = 0; dev_idx < nvml_devices_count; dev_idx++) {
        nvmlDevice_t device;
        ret = nvmlDeviceGetHandleByIndex(dev_idx, &device);
        if (ret != NVML_SUCCESS) {
            LOG_WARN("active_oom_killer: Failed to get device handle for device %u: %d (%s)", 
                     dev_idx, ret, nvmlErrorString(ret));
            continue;
        }
        
        // Get all processes on this device
        // Bypass our hook to get ALL processes (unfiltered), then filter ourselves
        // This ensures we don't miss any processes due to hook filtering or buffer limits
        unsigned int process_count = SHARED_REGION_MAX_PROCESS_NUM;
        nvmlProcessInfo_t infos[SHARED_REGION_MAX_PROCESS_NUM];
        
        if (nvml_library_entry != NULL) {
            // Bypass hook to get ALL processes directly from NVML
            ret = NVML_OVERRIDE_CALL_NO_LOG(nvml_library_entry, nvmlDeviceGetComputeRunningProcesses_v2,
                                            device, &process_count, infos);
        } else {
            // nvml_library_entry not available - use regular call (will go through hook)
            ret = nvmlDeviceGetComputeRunningProcesses(device, &process_count, infos);
        }
        
        // Handle buffer size issues - retry with larger buffer if needed
        if (ret == NVML_ERROR_INSUFFICIENT_SIZE) {
            LOG_WARN("active_oom_killer: Buffer too small, retrying with larger buffer");
            process_count = SHARED_REGION_MAX_PROCESS_NUM;
            if (nvml_library_entry != NULL) {
                ret = NVML_OVERRIDE_CALL_NO_LOG(nvml_library_entry, nvmlDeviceGetComputeRunningProcesses_v2,
                                                device, &process_count, infos);
            } else {
                ret = nvmlDeviceGetComputeRunningProcesses(device, &process_count, infos);
            }
        }
        
        if (ret != NVML_SUCCESS && ret != NVML_ERROR_INSUFFICIENT_SIZE) {
            LOG_WARN("active_oom_killer: Failed to get processes for device %u: %d (%s)", 
                     dev_idx, ret, nvmlErrorString(ret));
            continue;
        }
        
        unsigned int bounded_count = process_count > SHARED_REGION_MAX_PROCESS_NUM ?
                                     SHARED_REGION_MAX_PROCESS_NUM : process_count;
        LOG_DEBUG("active_oom_killer: Device %u has %u processes (bounded to %u)", dev_idx, process_count, bounded_count);
        
        // Filter and kill processes belonging to current cgroup/UID
        // Only non-root users get this treatment (root is disabled above)
        // In multi-user SLURM environment: filter by cgroup first (job isolation), then UID (user isolation)
        for (unsigned int i = 0; i < bounded_count; i++) {
            unsigned int actual_pid = infos[i].pid;
            if (actual_pid == 0) {
                LOG_WARN("active_oom_killer: Process[%u] - could not extract valid PID, skipping", i);
                continue;  // Skip if we can't get a valid PID
            }
            
            int should_kill = 0;
            
            // Filter by cgroup session first, fall back to UID (same logic as memory counting)
            int cgroup_check = proc_belongs_to_current_cgroup_session(actual_pid);
            
            if (cgroup_check == 1) {
                // Process belongs to current cgroup session - verify UID for extra safety
                uid_t proc_uid = proc_get_uid(actual_pid);
                if (proc_uid != (uid_t)-1 && proc_uid == current_uid) {
                    should_kill = 1;
                } else {
                    LOG_DEBUG("active_oom_killer: Process[%u] PID %u - SKIPPING (same cgroup but different UID %u != current UID %u)", 
                             i, actual_pid, proc_uid, current_uid);
                }
            } else if (cgroup_check == -1) {
                // Couldn't determine cgroup or not in a cgroup session - fall back to UID check
                uid_t proc_uid = proc_get_uid(actual_pid);
                if (proc_uid != (uid_t)-1 && proc_uid == current_uid) {
                    should_kill = 1;
                } else {
                    LOG_DEBUG("active_oom_killer: Process[%u] PID %u - SKIPPING (UID %u != current UID %u)", 
                             i, actual_pid, proc_uid, current_uid);
                }
            }
            
            if (should_kill) {
                // Verify process is still alive before killing
                int proc_state = proc_alive(actual_pid);
                if (proc_state == PROC_STATE_ALIVE) {
                    uint64_t process_mem = infos[i].usedGpuMemory;
                    LOG_ERROR("active_oom_killer: KILLING PID %u (device %u, memory %llu bytes)", 
                             actual_pid, dev_idx, (unsigned long long)process_mem);
                    int kill_result = kill(actual_pid, SIGKILL);
                    if (kill_result == 0) {
                        total_killed++;
                        LOG_ERROR("active_oom_killer: KILLED PID %u successfully (total_killed=%d)", 
                                 actual_pid, total_killed);
                    } else {
                        LOG_WARN("active_oom_killer: FAILED to kill PID %u: errno=%d (%s)", 
                                actual_pid, errno, strerror(errno));
                    }
                } else {
                    LOG_DEBUG("active_oom_killer: Process PID %u - already dead (state=%d), skipping kill", 
                             actual_pid, proc_state);
                }
            }
        }
    }
    
    LOG_ERROR("active_oom_killer: Killed %d processes from current cgroup/UID", total_killed);
    
    // Give processes a moment to terminate
    if (total_killed > 0) {
        usleep(200000);  // 200ms
    }
    
    return total_killed;
}

// Structure to hold process info for sorting
typedef struct {
    uint32_t pid;
    uint64_t memory;
} process_memory_info_t;

/** Sort OOM victims by memory usage descending (largest consumer first). */
static int compare_process_memory(const void* a, const void* b) {
    const process_memory_info_t* pa = (const process_memory_info_t*)a;
    const process_memory_info_t* pb = (const process_memory_info_t*)b;
    
    if (pa->memory > pb->memory) return -1;
    if (pa->memory < pb->memory) return 1;
    return 0;
}

// Extract cgroup path from a cgroup line (helper for kill_current_cgroup)
static char* extract_cgroup_path_for_kill(const char* line) {
    char* path = NULL;
    
    // Try cgroups v2 format first: "0::<path>"
    if (strncmp(line, "0::", 3) == 0) {
        const char* v2_path = line + 3;
        if (*v2_path == '/') {
            v2_path++;
        }
        size_t len = strlen(v2_path);
        if (len > 0 && v2_path[len - 1] == '\n') {
            len--;
        }
        if (len > 0) {
            path = (char*)malloc(len + 1);
            if (path != NULL) {
                strncpy(path, v2_path, len);
                path[len] = '\0';
            }
        }
        return path;
    }
    
    // Try cgroups v1 format: "<id>:<controller>:<path>"
    const char* last_colon = strrchr(line, ':');
    if (last_colon != NULL && last_colon > line) {
        const char* v1_path = last_colon + 1;
        if (*v1_path == '/') {
            v1_path++;
        }
        size_t len = strlen(v1_path);
        if (len > 0 && v1_path[len - 1] == '\n') {
            len--;
        }
        if (len > 0) {
            path = (char*)malloc(len + 1);
            if (path != NULL) {
                strncpy(path, v1_path, len);
                path[len] = '\0';
            }
        }
    }
    
    return path;
}

// Kill all processes in current cgroup (terminates SLURM job)
static int kill_current_cgroup(void) {
    // Read cgroup path from /proc/self/cgroup
    char filename[8192];
    snprintf(filename, sizeof(filename), "/proc/%d/cgroup", getpid());
    
    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        LOG_WARN("kill_current_cgroup: Could not open /proc/%d/cgroup", getpid());
        return -1;
    }
    
    char line[8192];
    char* cgroup_path = NULL;
    
    // Read each line in the cgroup file
    while (fgets(line, sizeof(line), fp) != NULL) {
        char* path = extract_cgroup_path_for_kill(line);
        if (path != NULL) {
            cgroup_path = path;
            break;  // Found a valid path, stop searching
        }
    }
    
    fclose(fp);
    
    if (cgroup_path == NULL) {
        LOG_WARN("kill_current_cgroup: Could not determine cgroup path");
        return -1;
    }
    
    LOG_ERROR("kill_current_cgroup: Killing all processes in cgroup: %s", cgroup_path);
    
    // Try cgroups v2 first: /sys/fs/cgroup/<path>/cgroup.procs
    char cgroup_procs_file[2048];
    snprintf(cgroup_procs_file, sizeof(cgroup_procs_file), 
             "/sys/fs/cgroup/%s/cgroup.procs", cgroup_path);
    
    fp = fopen(cgroup_procs_file, "r");
    if (fp != NULL) {
        pid_t pid;
        int killed = 0;
        while (fscanf(fp, "%d", &pid) == 1) {
            if (pid > 0 && pid != getpid()) {  // Don't kill ourselves
                LOG_ERROR("kill_current_cgroup: Killing PID %d from cgroup", pid);
                kill(pid, SIGKILL);
                killed++;
            }
        }
        fclose(fp);
        free(cgroup_path);
        LOG_ERROR("kill_current_cgroup: Killed %d processes from cgroup v2", killed);
        return killed;
    }
    
    // Try cgroups v1: /sys/fs/cgroup/memory/<path>/cgroup.procs
    snprintf(cgroup_procs_file, sizeof(cgroup_procs_file), 
             "/sys/fs/cgroup/memory/%s/cgroup.procs", cgroup_path);
    
    fp = fopen(cgroup_procs_file, "r");
    if (fp != NULL) {
        pid_t pid;
        int killed = 0;
        while (fscanf(fp, "%d", &pid) == 1) {
            if (pid > 0 && pid != getpid()) {  // Don't kill ourselves
                LOG_ERROR("kill_current_cgroup: Killing PID %d from cgroup", pid);
                kill(pid, SIGKILL);
                killed++;
            }
        }
        fclose(fp);
        free(cgroup_path);
        LOG_ERROR("kill_current_cgroup: Killed %d processes from cgroup v1", killed);
        return killed;
    }
    
    free(cgroup_path);
    LOG_WARN("kill_current_cgroup: Could not find cgroup.procs file (tried v2 and v1)");
    return -1;
}

// Gradual OOM killer: kills processes one by one, sorted by GPU memory (highest first)
// Returns number of processes killed, or -1 on error
int gradual_oom_killer(int cuda_dev) {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    
    uid_t current_uid = getuid();
    if (current_uid == 0) {
        LOG_INFO("gradual_oom_killer: Root user (UID 0) - OOM killer disabled");
        return 0;
    }
    
    unsigned int nvml_dev_idx = cuda_to_nvml_map(cuda_dev);
    nvmlDevice_t device;
    nvmlReturn_t ret = nvmlDeviceGetHandleByIndex(nvml_dev_idx, &device);
    if (ret != NVML_SUCCESS) {
        LOG_WARN("gradual_oom_killer: Failed to get device handle for CUDA device %d (NVML %u): %d (%s)", 
                 cuda_dev, nvml_dev_idx, ret, nvmlErrorString(ret));
        return -1;
    }
    
    // Get all processes on this device
    unsigned int process_count = SHARED_REGION_MAX_PROCESS_NUM;
    nvmlProcessInfo_t infos[SHARED_REGION_MAX_PROCESS_NUM];
    
    if (nvml_library_entry != NULL) {
        ret = NVML_OVERRIDE_CALL_NO_LOG(nvml_library_entry, nvmlDeviceGetComputeRunningProcesses_v2,
                                        device, &process_count, infos);
    } else {
        ret = nvmlDeviceGetComputeRunningProcesses(device, &process_count, infos);
    }
    
    if (ret != NVML_SUCCESS && ret != NVML_ERROR_INSUFFICIENT_SIZE) {
        LOG_WARN("gradual_oom_killer: Failed to get processes for device %d: %d (%s)", 
                 cuda_dev, ret, nvmlErrorString(ret));
        return -1;
    }
    
    unsigned int bounded_count = process_count > SHARED_REGION_MAX_PROCESS_NUM ?
                                 SHARED_REGION_MAX_PROCESS_NUM : process_count;
    LOG_DEBUG("gradual_oom_killer: Received %u processes from NVML, bounded=%u", 
              process_count, bounded_count);
    
    // Filter processes belonging to current cgroup/UID and collect them
    process_memory_info_t filtered_processes[SHARED_REGION_MAX_PROCESS_NUM];
    unsigned int filtered_count = 0;
    
    for (unsigned int i = 0; i < bounded_count; i++) {
        unsigned int actual_pid = infos[i].pid;
        if (actual_pid == 0) {
            continue;  // Skip if we can't get a valid PID
        }
        
        int should_kill = 0;
        int cgroup_check = proc_belongs_to_current_cgroup_session(actual_pid);
        
        if (cgroup_check == 1) {
            uid_t proc_uid = proc_get_uid(actual_pid);
            if (proc_uid != (uid_t)-1 && proc_uid == current_uid) {
                should_kill = 1;
            }
        } else if (cgroup_check == -1) {
            uid_t proc_uid = proc_get_uid(actual_pid);
            if (proc_uid != (uid_t)-1 && proc_uid == current_uid) {
                should_kill = 1;
            }
        }
        
        if (should_kill && proc_alive(actual_pid) == PROC_STATE_ALIVE) {
            filtered_processes[filtered_count].pid = actual_pid;
            filtered_processes[filtered_count].memory = infos[i].usedGpuMemory;
            filtered_count++;
        }
    }
    
    if (filtered_count == 0) {
        LOG_DEBUG("gradual_oom_killer: No processes found to kill on device %d", cuda_dev);
        return 0;
    }
    
    // Sort processes by PID (highest/newest first) - kill newest processes first
    qsort(filtered_processes, filtered_count, sizeof(process_memory_info_t), compare_process_memory);
    
    LOG_ERROR("gradual_oom_killer: Found %u processes on device %d, sorted by memory usage (largest first)", 
              filtered_count, cuda_dev);
    
    uint64_t limit = get_current_device_memory_limit(cuda_dev);
    int killed = 0;
    time_t start_time = time(NULL);
    int MAX_OOM_DURATION = 30;
    char* oom_timeout_str = getenv("SOFTMIG_OOM_TIMEOUT");
    if (oom_timeout_str != NULL) {
        int val = atoi(oom_timeout_str);
        if (val > 0 && val <= 300) MAX_OOM_DURATION = val;
    }
    
    // Kill processes one by one until under limit or only one remains
    for (unsigned int i = 0; i < filtered_count; i++) {
        // Check if we've been over limit for too long
        time_t current_time = time(NULL);
        if (current_time - start_time > MAX_OOM_DURATION) {
            LOG_ERROR("gradual_oom_killer: Over limit for %ld seconds, killing entire cgroup", 
                     current_time - start_time);
            kill_current_cgroup();
            return killed;
        }
        
        // If only one process left and still over limit, kill cgroup
        if (i == filtered_count - 1) {
            LOG_ERROR("gradual_oom_killer: Only one process remaining (PID %u) and still over limit, killing cgroup", 
                     filtered_processes[i].pid);
            kill_current_cgroup();
            return killed;
        }
        
        // Kill the process with highest PID (newest process first)
        LOG_ERROR("gradual_oom_killer: Killing PID %u (newest, memory %llu bytes, device %d)", 
                 filtered_processes[i].pid, 
                 (unsigned long long)filtered_processes[i].memory, 
                 cuda_dev);
        
        int kill_result = kill(filtered_processes[i].pid, SIGKILL);
        if (kill_result == 0) {
            killed++;
            LOG_ERROR("gradual_oom_killer: Successfully killed PID %u", filtered_processes[i].pid);
        } else {
            LOG_WARN("gradual_oom_killer: Failed to kill PID %u: errno=%d (%s)", 
                    filtered_processes[i].pid, errno, strerror(errno));
        }
        
        // Wait for process to terminate and memory to be freed
        usleep(500000);  // 500ms
        
        // Re-check memory usage
        uint64_t usage = get_summed_device_memory_usage_from_nvml(cuda_dev);
        if (usage == 0) {
            usage = get_gpu_memory_usage_nolock(cuda_to_nvml_map(cuda_dev));
        }
        
        LOG_DEBUG("gradual_oom_killer: After killing PID %u, usage=%llu limit=%llu", 
                 filtered_processes[i].pid, (unsigned long long)usage, (unsigned long long)limit);
        
        if (usage <= limit) {
            LOG_DEBUG("gradual_oom_killer: Memory usage now under limit, stopping");
            break;
        }
    }
    
    return killed;
}

void pre_launch_kernel() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;  // No-op when softmig is disabled
    }
    uint64_t now = time(NULL);
    pthread_mutex_lock(&_kernel_mutex);
    if (now - region_info.last_kernel_time < _record_kernel_interval) {
        pthread_mutex_unlock(&_kernel_mutex);
        return;
    }
    region_info.last_kernel_time = now;
    pthread_mutex_unlock(&_kernel_mutex);
    lock_shrreg();
    if (region_info.shared_region->last_kernel_time < now) {
        region_info.shared_region->last_kernel_time = now;
    }
    unlock_shrreg();
}

int shrreg_major_version() {
    return MAJOR_VERSION;
}

int shrreg_minor_version() {
    return MINOR_VERSION;
}


size_t get_gpu_memory_monitor(const int dev) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;
    }
    int i=0;
    size_t total=0;
    lock_shrreg();
    for (i=0;i<region_info.shared_region->proc_num;i++){
        total+=region_info.shared_region->procs[i].monitorused[dev];
    }
    unlock_shrreg();
    return total;
}

// Get memory usage without locking (caller must hold lock_shrreg)
size_t get_gpu_memory_usage_nolock(const int dev) {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;
    }
    int i=0;
    size_t total=0;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        total+=region_info.shared_region->procs[i].used[dev].total;
    }
    total+=initial_offset;
    return total;
}

uint64_t get_pending_memory_nolock(const int nvmldev) {
    if (!is_softmig_enabled() || region_info.shared_region == NULL ||
            nvmldev < 0 || nvmldev >= CUDA_DEVICE_MAX_COUNT) {
        return 0;
    }
    uint64_t total = 0;
    for (int i = 0; i < region_info.shared_region->proc_num; i++) {
        total += region_info.shared_region->procs[i].pending[nvmldev];
    }
    return total;
}

void adjust_pending_memory_nolock(int cudadev, int64_t delta) {
    if (!is_softmig_enabled() || region_info.shared_region == NULL ||
            cudadev < 0 || cudadev >= CUDA_DEVICE_MAX_COUNT) {
        return;
    }
    int dev = cuda_to_nvml_map(cudadev);
    int32_t self = getpid();
    for (int i = 0; i < region_info.shared_region->proc_num; i++) {
        shrreg_proc_slot_t *p = &region_info.shared_region->procs[i];
        if (p->pid != self) {
            continue;
        }
        if (delta < 0 && (uint64_t)(-delta) > p->pending[dev]) {
            p->pending[dev] = 0;
        } else {
            p->pending[dev] += delta;
        }
        return;
    }
}

size_t get_gpu_memory_usage(const int dev) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;
    }
    lock_shrreg();
    size_t total = get_gpu_memory_usage_nolock(dev);
    unlock_shrreg();
    return total;
}

int set_gpu_device_memory_monitor(int32_t pid,int dev,size_t monitor){
    int i;
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    lock_shrreg();
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].hostpid == pid){
            region_info.shared_region->procs[i].monitorused[dev] = monitor;
            break;
        }
    }
    unlock_shrreg();
    return 1;
}

int set_gpu_device_sm_utilization(int32_t pid,int dev, unsigned int smUtil){  // new function
    int i;
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    lock_shrreg();
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].hostpid == pid){
            region_info.shared_region->procs[i].device_util[dev].sm_util = smUtil;
            break;
        }
    }
    unlock_shrreg();
    return 1;
}

int init_gpu_device_utilization(){
    int i,dev;
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    lock_shrreg();
    for (i=0;i<region_info.shared_region->proc_num;i++){
        for (dev=0;dev<CUDA_DEVICE_MAX_COUNT;dev++){
            region_info.shared_region->procs[i].device_util[dev].sm_util = 0;
            region_info.shared_region->procs[i].monitorused[dev] = 0;
        }
    }
    unlock_shrreg();
    return 1;
}

uint64_t get_summed_device_memory_usage_from_nvml(int cuda_dev) {
    if (sum_process_memory_from_nvml == NULL) return 0;
    unsigned int nvml_dev_idx = cuda_to_nvml_map(cuda_dev);
    nvmlDevice_t ndev;
    nvmlReturn_t ret = nvmlDeviceGetHandleByIndex(nvml_dev_idx, &ndev);
    if (ret != NVML_SUCCESS) {
        LOG_WARN("get_summed_device_memory_usage_from_nvml: NVML get device %d (CUDA %d) error, %s",
                 nvml_dev_idx, cuda_dev, nvmlErrorString(ret));
        return 0;
    }

    return sum_process_memory_from_nvml(ndev);
}


int add_gpu_device_memory_usage(int32_t pid,int cudadev,size_t usage,int type){
    int dev = cuda_to_nvml_map(cudadev);
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    lock_shrreg();
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].pid == pid){
            if (region_info.shared_region->procs[i].hostpid == 0) {
                region_info.shared_region->procs[i].hostpid = pid;
            }
            region_info.shared_region->procs[i].used[dev].total+=usage;
            switch (type) {
                case 0:{
                    region_info.shared_region->procs[i].used[dev].context_size += usage;
                    break;
                }
                case 1:{
                    region_info.shared_region->procs[i].used[dev].module_size += usage;
                    break;
                }
                case 2:{
                    region_info.shared_region->procs[i].used[dev].data_size += usage;
                }
            }
        }
    }
    unlock_shrreg();
    return 0;
}

int rm_gpu_device_memory_usage(int32_t pid,int cudadev,size_t usage,int type){
    int dev = cuda_to_nvml_map(cudadev);
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    lock_shrreg();
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].pid == pid){
            region_info.shared_region->procs[i].used[dev].total-=usage;
            switch (type) {
                case 0:{
                    region_info.shared_region->procs[i].used[dev].context_size -= usage;
                    break;
                }
                case 1:{
                    region_info.shared_region->procs[i].used[dev].module_size -= usage;
                    break;
                }
                case 2:{
                    region_info.shared_region->procs[i].used[dev].data_size -= usage;
                }
            }
        }
    }
    unlock_shrreg();
    return 0;
}

void get_timespec(int seconds, struct timespec* spec) {
    clock_gettime(CLOCK_REALTIME, spec);
    spec->tv_sec += seconds;
}

void exit_withlock(int exitcode) {
    unlock_shrreg();
    exit(exitcode);
}

// The region lock is a robust, process-shared, error-checking mutex. The
// kernel tracks the owning thread, so:
//  - a live holder (even one that is SIGSTOPped) is waited for, never robbed;
//  - if the holder dies, the next locker gets EOWNERDEAD, cleans the process
//    table and marks the mutex consistent;
//  - sibling threads of one process serialize like any other contender.
// Re-entrant per thread via shrreg_lock_depth (callers nest, e.g. the
// allocator calls helpers that lock again).
static __thread int shrreg_lock_depth = 0;
static volatile int shrreg_lock_broken = 0;

static void shrreg_owner_died(shared_region_t* region) {
    LOG_WARN("shrreg lock owner (pid %ld) died holding the lock - recovering",
             (long)region->owner_pid);
    clear_proc_slot_nolock(1);
    if (pthread_mutex_consistent(&region->lock) != 0) {
        LOG_ERROR("pthread_mutex_consistent failed: errno=%d", errno);
    }
}

// Returns 0 with the lock held, or an error (ETIMEDOUT when give_up_after > 0
// and that many seconds passed, ENOTRECOVERABLE, ...).
static int shrreg_acquire(shared_region_t* region, int give_up_after) {
    int waited = 0;
    for (;;) {
        struct timespec ts;
        get_timespec(SEM_WAIT_TIME, &ts);
        int rc = pthread_mutex_timedlock(&region->lock, &ts);
        if (rc == 0) {
            break;
        }
        if (rc == EOWNERDEAD) {
            shrreg_owner_died(region);
            break;
        }
        if (rc == ETIMEDOUT) {
            waited += SEM_WAIT_TIME;
            if (give_up_after > 0 && waited >= give_up_after) {
                return ETIMEDOUT;
            }
            if (waited % (SEM_WAIT_TIME * SEM_WAIT_RETRY_TIMES) == 0) {
                LOG_WARN("Waiting %ds for shrreg lock held by pid %ld", waited, (long)region->owner_pid);
            }
            continue;
        }
        return rc;
    }
    region->owner_pid = getpid();
    __sync_synchronize();
    return 0;
}

void exit_handler() {
    if (region_info.init_status == PTHREAD_ONCE_INIT) {
        return;
    }
    shared_region_t* region = region_info.shared_region;
    
    // Check if shared region was never initialized (e.g., program failed to start)
    // This can happen when bash loads the library but the program doesn't exist
    if (region == NULL || region == MAP_FAILED) {
        return;
    }
    
    LOG_MSG("Calling exit handler %d",getpid());
    
    // exit() from inside a locked section: we already own the lock.
    int held = shrreg_lock_depth > 0;
    if (!held && !shrreg_lock_broken) {
        int rc = shrreg_acquire(region, SEM_WAIT_TIME_ON_EXIT);
        if (rc != 0) {
            LOG_WARN("Failed to take lock on exit: errno=%d", rc);
            return;
        }
    }
    int32_t self = getpid();
    for (int slot = 0; slot < region->proc_num; slot++) {
        if (region->procs[slot].pid == self) {
            memset(region->procs[slot].used,0,sizeof(device_memory_t)*CUDA_DEVICE_MAX_COUNT);
            memset(region->procs[slot].device_util,0,sizeof(device_util_t)*CUDA_DEVICE_MAX_COUNT);
            region->proc_num--;
            region->procs[slot] = region->procs[region->proc_num];
            break;
        }
    }
    __sync_synchronize();
    if (!held && !shrreg_lock_broken) {
        region->owner_pid = 0;
        pthread_mutex_unlock(&region->lock);
    }
}

void lock_shrreg() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;
    }
    if (shrreg_lock_depth++ > 0 || shrreg_lock_broken) {
        return;
    }
    int rc = shrreg_acquire(region_info.shared_region, 0);
    SEQ_POINT_MARK(SEQ_ACQUIRE_SEMLOCK_OK);
    if (rc != 0) {
        // ENOTRECOVERABLE (or worse): the lock can never be taken again in
        // this region. Keep running unlocked rather than hang the job.
        LOG_ERROR("shrreg lock unusable (rc=%d); continuing without cross-process locking", rc);
        shrreg_lock_broken = 1;
    }
}

void unlock_shrreg() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;  // No-op when softmig is disabled
    }
    if (shrreg_lock_depth <= 0) {
        LOG_WARN("unlock_shrreg without matching lock_shrreg");
        return;
    }
    if (--shrreg_lock_depth > 0 || shrreg_lock_broken) {
        return;
    }
    SEQ_POINT_MARK(SEQ_BEFORE_UNLOCK_SHRREG);
    shared_region_t* region = region_info.shared_region;
    region->owner_pid = 0;
    __sync_synchronize();
    SEQ_POINT_MARK(SEQ_RESET_OWNER_OK);
    int rc = pthread_mutex_unlock(&region->lock);
    if (rc != 0) {
        LOG_ERROR("shrreg unlock failed: rc=%d", rc);
    }
    SEQ_POINT_MARK(SEQ_RELEASE_SEMLOCK_OK);
}


int clear_proc_slot_nolock(int do_clear) {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    int slot = 0;
    int res=0;
    shared_region_t* region = region_info.shared_region;
    while (slot < region->proc_num) {
        int32_t pid = region->procs[slot].pid;
        // pid 0 is an empty/corrupt slot; drop it like a dead process
        // (leaving it in place used to spin here forever with the lock held).
        if (pid != 0 && !(do_clear > 0 && proc_alive(pid) == PROC_STATE_NONALIVE)) {
            slot++;
            continue;
        }
        LOG_WARN("Kick %s proc slot %d (pid %d)", pid == 0 ? "empty" : "dead", slot, pid);
        res=1;
        region->proc_num--;
        region->procs[slot] = region->procs[region->proc_num];
        __sync_synchronize();
    }
    return res;
}

void init_proc_slot_withlock() {
    int32_t current_pid = getpid();
    lock_shrreg();
    shared_region_t* region = region_info.shared_region;
    clear_proc_slot_nolock(1);
    if (region->proc_num >= SHARED_REGION_MAX_PROCESS_NUM) {
        // Run untracked rather than kill the user's process: the other
        // processes still see this one's memory through NVML usage.
        LOG_ERROR("shrreg full (%d processes); pid %d runs without a slot (not tracked)",
                  SHARED_REGION_MAX_PROCESS_NUM, current_pid);
        unlock_shrreg();
        return;
    }
    // SIGUSR1/2 are commonly used by jobs (e.g. sbatch --signal=USR1@60 for
    // checkpointing). Only take them over for the legacy suspend/resume path
    // (shrreg_tool), which is bundled with the opt-in OOM killer.
    if (enable_active_oom_killer) {
        signal(SIGUSR2,sig_swap_stub);
        signal(SIGUSR1,sig_restore_stub);
    }
    // If, by any means a pid of itself is found in region->proces, then it is probably caused by crashloop
    // we need to reset it.
    int i,found=0;
    for (i=0; i<region->proc_num; i++) {
        if (region->procs[i].pid == current_pid) {
            region->procs[i].status = 1;
            memset(region->procs[i].used,0,sizeof(device_memory_t)*CUDA_DEVICE_MAX_COUNT);
            memset(region->procs[i].pending,0,sizeof(region->procs[i].pending));
            memset(region->procs[i].device_util,0,sizeof(device_util_t)*CUDA_DEVICE_MAX_COUNT);
            found = 1;
            break;
        }
    }
    if (!found) {
        region->procs[region->proc_num].pid = current_pid;
        region->procs[region->proc_num].status = 1;
        memset(region->procs[region->proc_num].used,0,sizeof(device_memory_t)*CUDA_DEVICE_MAX_COUNT);
        memset(region->procs[region->proc_num].pending,0,sizeof(region->procs[region->proc_num].pending));
        memset(region->procs[region->proc_num].device_util,0,sizeof(device_util_t)*CUDA_DEVICE_MAX_COUNT);
        region->proc_num++;
    }

    clear_proc_slot_nolock(1);
    unlock_shrreg();
}

void child_reinit_flag() {
    LOG_DEBUG("Detect child pid: %d -> %d", region_info.pid, getpid());   
    region_info.init_status = PTHREAD_ONCE_INIT;
    // The child does not own a lock its parent's thread held (robust mutex
    // ownership is per thread), so it starts with no nesting.
    shrreg_lock_depth = 0;
}

const char *shrreg_default_path(char *buf, size_t len) {
    const char *env = getenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV);
    if (env != NULL) {
        snprintf(buf, len, "%s", env);
        return buf;
    }
    // SLURM_TMPDIR (/tmp under job_container/tmpfs) is private to the job and
    // wiped at job end. The layout version is part of the name.
    const char *tmpdir = getenv("SLURM_TMPDIR");
    if (tmpdir == NULL) {
        tmpdir = "/tmp";
    }
    const char *job_id = getenv("SLURM_JOB_ID");
    const char *array_id = getenv("SLURM_ARRAY_TASK_ID");
    if (job_id != NULL && array_id != NULL) {
        snprintf(buf, len, "%s/cudevshr.cache.v%d.%s.%s", tmpdir, MAJOR_VERSION, job_id, array_id);
    } else if (job_id != NULL) {
        snprintf(buf, len, "%s/cudevshr.cache.v%d.%s", tmpdir, MAJOR_VERSION, job_id);
    } else {
        snprintf(buf, len, "%s/cudevshr.cache.v%d.uid%d.pid%d", tmpdir, MAJOR_VERSION, (int)getuid(),
                 (int)getpid());
    }
    return buf;
}

void try_create_shrreg() {
    LOG_DEBUG("Try create shrreg")
    if (region_info.fd == -1) {
        // use .fd to indicate whether a reinit after fork happen
        // no need to register exit handler after fork
        if (0 != atexit(exit_handler)) {
            LOG_ERROR("Register exit handler failed: %d", errno);
        }
    }

    // Default: OOM killer disabled, so allocations over the per-job limit return
    // CUDA_ERROR_OUT_OF_MEMORY (matching real-GPU behavior) instead of SIGKILL.
    // Set SOFTMIG_ENABLE_OOM_KILLER=1 in env or the SLURM config file to restore
    // the legacy active/gradual kill defense.
    enable_active_oom_killer = get_softmig_oom_killer_enabled();
    if (enable_active_oom_killer) {
        LOG_WARN("SOFTMIG_ENABLE_OOM_KILLER is set - legacy in-library OOM killer enabled");
    } else {
        LOG_DEBUG("OOM killer disabled (default) - allocations over limit return CUDA_ERROR_OUT_OF_MEMORY");
    }
    env_utilization_switch = 1;
    pthread_atfork(NULL, NULL, child_reinit_flag);

    region_info.pid = getpid();
    region_info.fd = -1;
    region_info.last_kernel_time = time(NULL);

    umask(0);

    static char cache_path[512];
    const char* shr_reg_file = shrreg_default_path(cache_path, sizeof(cache_path));
    // Initialize NVML BEFORE!! open it
    //nvmlInit();

    /* If you need sm modification, do it here */
    /* ... set_sm_scale */

    int fd = open(shr_reg_file, O_RDWR | O_CREAT, 0600);
    if (fd == -1) {
        LOG_ERROR("Fail to open shrreg %s: errno=%d", shr_reg_file, errno);
    }
    region_info.fd = fd;
    if (ftruncate(fd, SHARED_REGION_SIZE_MAGIC) != 0) {
        LOG_ERROR("Fail to size shrreg %s: errno=%d", shr_reg_file, errno);
    }
    region_info.shared_region = (shared_region_t*) mmap(
        NULL, SHARED_REGION_SIZE_MAGIC, 
        PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
    shared_region_t* region = region_info.shared_region;
    if (region == MAP_FAILED) {
        LOG_ERROR("Fail to map shrreg %s: errno=%d", shr_reg_file, errno);
    }
    if (lockf(fd, F_LOCK, SHARED_REGION_SIZE_MAGIC) != 0) {
        LOG_ERROR("Fail to lock shrreg %s: errno=%d", shr_reg_file, errno);
    }
    if (region->initialized_flag != 
          MULTIPROCESS_SHARED_REGION_MAGIC_FLAG) {
        region->major_version = MAJOR_VERSION;
        region->minor_version = MINOR_VERSION;
        do_init_device_memory_limits(
            region->limit, CUDA_DEVICE_MAX_COUNT);
        do_init_device_sm_limits(
            region->sm_limit,CUDA_DEVICE_MAX_COUNT);
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
        int mrc = pthread_mutex_init(&region->lock, &attr);
        pthread_mutexattr_destroy(&attr);
        if (mrc != 0) {
            LOG_ERROR("Fail to init shrreg lock %s: rc=%d", shr_reg_file, mrc);
        }
        region->owner_pid = 0;
        __sync_synchronize();
        region->sm_init_flag = 0;
        region->utilization_switch = 1;
        region->recent_kernel = 2;
        region->priority = 1;  // Default priority (unused, kept for compatibility)
        region->initialized_flag = MULTIPROCESS_SHARED_REGION_MAGIC_FLAG;
    } else {
        if (region->major_version != MAJOR_VERSION || 
                region->minor_version != MINOR_VERSION) {
            LOG_ERROR("The current version number %d.%d"
                    " is different from the file's version number %d.%d",
                    MAJOR_VERSION, MINOR_VERSION,
                    region->major_version, region->minor_version);
        }
        uint64_t local_limits[CUDA_DEVICE_MAX_COUNT];
        do_init_device_memory_limits(local_limits, CUDA_DEVICE_MAX_COUNT);
        int i;
        for (i = 0; i < CUDA_DEVICE_MAX_COUNT; ++i) {
            if (local_limits[i] != region->limit[i]) {
                // Downgrade to DEBUG - this is expected when cache is from different job/limit
                // Recreate cache with correct limits from environment
                LOG_DEBUG("Limit inconsistency detected for %dth device, %lu expected, get %lu - updating cache", 
                    i, local_limits[i], region->limit[i]);
                // Update cache with environment limits (environment is source of truth)
                region->limit[i] = local_limits[i];
            }
        }
        do_init_device_sm_limits(local_limits,CUDA_DEVICE_MAX_COUNT);
        for (i = 0; i < CUDA_DEVICE_MAX_COUNT; ++i) {
            if (local_limits[i] != region->sm_limit[i]) {
                // Update cache with environment limits (environment is source of truth)
                LOG_DEBUG("SM limit inconsistency detected for %dth device, %lu expected, get %lu - updating cache",
                    i, local_limits[i], region->sm_limit[i]);
                region->sm_limit[i] = local_limits[i];
            }
        }
    }
    region->last_kernel_time = region_info.last_kernel_time;
    if (lockf(fd, F_ULOCK, SHARED_REGION_SIZE_MAGIC) != 0) {
        LOG_ERROR("Fail to unlock shrreg %s: errno=%d", shr_reg_file, errno);
    }
    LOG_DEBUG("shrreg created");
}

void initialized() {
    // Check if softmig should be active (if env vars are set)
    if (!is_softmig_enabled()) {
        // softmig is disabled - don't initialize anything
        return;
    }
    
    pthread_mutex_init(&_kernel_mutex, NULL);
    char* _record_kernel_interval_env = getenv("RECORD_KERNEL_INTERVAL");
    if (_record_kernel_interval_env) {
        _record_kernel_interval = atoi(_record_kernel_interval_env);
    }
    try_create_shrreg();
    init_proc_slot_withlock();
}

void ensure_initialized() {
    // Check if softmig should be active before initializing
    if (!is_softmig_enabled()) {
        // softmig is disabled - don't initialize anything
        return;
    }
    
    (void) pthread_once(&region_info.init_status, initialized);
}

int update_host_pid() {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].pid == getpid()){
            if (region_info.shared_region->procs[i].hostpid!=0)
                pidfound=1; 
        }
    }
    return 0;
}

int set_host_pid(int hostpid) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    int i,j,found=0;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].pid == getpid()){
            LOG_DEBUG("SET PID= %d",hostpid);
            found=1;
            region_info.shared_region->procs[i].hostpid = hostpid;
            for (j=0;j<CUDA_DEVICE_MAX_COUNT;j++)
                region_info.shared_region->procs[i].monitorused[j]=0;
        }
    }
    if (!found) {
        LOG_ERROR("HOST PID NOT FOUND. %d",hostpid);
        return -1;
    }
    setspec();
    return 0;
}

int set_current_device_sm_limit_scale(int dev, int scale) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    if (region_info.shared_region->sm_init_flag==1) return 0;
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
        return 0;
    }
    region_info.shared_region->sm_limit[dev]=region_info.shared_region->sm_limit[dev]*scale;
    region_info.shared_region->sm_init_flag = 1;
    return 0;
}

int get_current_device_sm_limit(int dev) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 100;  // No limit (100%) when softmig is disabled
    }
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
        return 0;
    }
    return region_info.shared_region->sm_limit[dev];
}

int set_current_device_memory_limit(const int dev,size_t newlimit) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
        return 0;
    }
    LOG_DEBUG("dev %d new limit set to %ld",dev,newlimit);
    region_info.shared_region->limit[dev]=newlimit;
    return 0; 
}

uint64_t get_current_device_memory_limit(const int dev) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No limit when softmig is disabled
    }
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
        return 0;
    }
    return region_info.shared_region->limit[dev];       
}

uint64_t get_current_device_memory_monitor(const int dev) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No monitoring when softmig is disabled
    }
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
        return 0;
    }
    uint64_t result = get_gpu_memory_monitor(dev);
    return result;
}

uint64_t get_current_device_memory_usage(const int dev) {
    uint64_t result;
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No usage tracking when softmig is disabled
    }
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
        return 0;
    }
    result = get_gpu_memory_usage(cuda_to_nvml_map(dev));
    return result;
}

int get_current_priority() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 1;  // Default priority when softmig is disabled
    }
    return region_info.shared_region->priority;
}

int get_recent_kernel(){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // Default when softmig is disabled
    }
    return region_info.shared_region->recent_kernel;
}

int set_recent_kernel(int value){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    region_info.shared_region->recent_kernel=value;
    return 0;
}

int get_utilization_switch() {
    // Always enabled when softmig is active
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // Disabled when softmig is disabled
    }
    return region_info.shared_region->utilization_switch; 
}

void suspend_all(){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;  // No-op when softmig is disabled
    }
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        kill(region_info.shared_region->procs[i].pid,SIGUSR2);
    }
}

void resume_all(){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;  // No-op when softmig is disabled
    }
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        kill(region_info.shared_region->procs[i].pid,SIGUSR1);
    }
}

int wait_status_self(int status){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 1;  // Always return "ready" when softmig is disabled
    }
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].pid==getpid()){
            if (region_info.shared_region->procs[i].status==status)
                return 1;
            else
                return 0;
        }
    }
    return -1;
}

int wait_status_all(int status){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 1;  // Always return "ready" when softmig is disabled
    }
    int i;
    int released = 1;
    for (i=0;i<region_info.shared_region->proc_num;i++) {
        if ((region_info.shared_region->procs[i].status!=status) && (region_info.shared_region->procs[i].pid!=getpid()))
            released = 0; 
    }
    return released;
}

shrreg_proc_slot_t *find_proc_by_hostpid(int hostpid) {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return NULL;  // No process found when softmig is disabled
    }
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++) {
        if (region_info.shared_region->procs[i].hostpid == hostpid) 
            return &region_info.shared_region->procs[i];
        if (region_info.shared_region->procs[i].hostpid == 0 &&
            region_info.shared_region->procs[i].pid == hostpid) {
            // Lazy hostpid registration fallback:
            // some processes can miss early set_task_pid timing, but their
            // process pid is already known in the shared region slot.
            region_info.shared_region->procs[i].hostpid = hostpid;
            LOG_DEBUG("find_proc_by_hostpid: lazily set hostpid=%d for slot pid=%d",
                      hostpid, region_info.shared_region->procs[i].pid);
            return &region_info.shared_region->procs[i];
        }
    }
    return NULL;
}


