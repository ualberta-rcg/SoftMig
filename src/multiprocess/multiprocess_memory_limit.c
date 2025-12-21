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

// Note: We need to bypass the hook to get ALL processes, then filter ourselves
// This ensures we don't miss any processes due to hook filtering or buffer limits
// Use weak symbol so it's NULL if nvml_mod not linked, allowing fallback
extern entry_t nvml_library_entry[] __attribute__((weak));

// Forward declarations for NVML functions (provided by hooks)
const char *nvmlErrorString(nvmlReturn_t result);
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount);
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length);
nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);


#ifndef SEM_WAIT_TIME
#define SEM_WAIT_TIME 10
#endif

#ifndef SEM_WAIT_TIME_ON_EXIT
#define SEM_WAIT_TIME_ON_EXIT 3
#endif

#ifndef SEM_WAIT_RETRY_TIMES
#define SEM_WAIT_RETRY_TIMES 30
#endif

int pidfound;

int ctx_activate[32];

static shared_region_info_t region_info = {0, -1, PTHREAD_ONCE_INIT, NULL, 0};
//size_t initial_offset=117440512;
int env_utilization_switch;
int enable_active_oom_killer;
size_t context_size;
size_t initial_offset=0;
// Flag to track if softmig is disabled (when env vars are not set)
static int softmig_disabled = -1;  // -1 = not checked yet, 0 = enabled, 1 = disabled

// External function from config_file.c - reads from config file or env
extern int is_softmig_configured(void);

// Helper function to check if softmig is enabled
static int is_softmig_enabled(void) {
    if (softmig_disabled == -1) {
        // First time check - see if environment variables are configured
        if (!is_softmig_configured()) {
            softmig_disabled = 1;
            LOG_DEBUG("softmig: CUDA_DEVICE_MEMORY_LIMIT and CUDA_DEVICE_SM_LIMIT not set - softmig disabled (passive mode)");
            return 0;
        }
        softmig_disabled = 0;
    }
    return (softmig_disabled == 0);
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
    FILE *f=fopen(filename,"r");
    if (f==NULL)
        return 0;
    char tmp[10000];
    int cursor=0;
    while (!feof(f)){
        if (fgets(tmp,10000,f) == NULL)
            break;
        if (strstr(tmp,"=")==NULL)
            break;
        if (tmp[strlen(tmp)-1]=='\n')
            tmp[strlen(tmp)-1]='\0';
        for (cursor=0;cursor<strlen(tmp);cursor++){
            if (tmp[cursor]=='=') {
                tmp[cursor]='\0';
                setenv(tmp,tmp+cursor+1,1);
                break;
            }
        }
    }
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
        
        // CRITICAL: Initialize version field for all structs before calling NVML
        // This ensures compatibility with different driver versions (CUDA 12.2 vs driver 570.195.03)
        for (unsigned int j = 0; j < SHARED_REGION_MAX_PROCESS_NUM; j++) {
            infos[j].version = nvmlProcessInfo_v2;
        }
        
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
            // Re-initialize version fields before retry
            for (unsigned int j = 0; j < SHARED_REGION_MAX_PROCESS_NUM; j++) {
                infos[j].version = nvmlProcessInfo_v2;
            }
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
        
        LOG_DEBUG("active_oom_killer: Device %u has %u processes (unfiltered)", dev_idx, process_count);
        
        // Filter and kill processes belonging to current cgroup/UID
        // Only non-root users get this treatment (root is disabled above)
        // In multi-user SLURM environment: filter by cgroup first (job isolation), then UID (user isolation)
        for (unsigned int i = 0; i < process_count; i++) {
            // CRITICAL: Use safe PID extraction to handle struct mismatches
            unsigned int actual_pid = extract_pid_safely((void *)&infos[i]);
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
                    LOG_ERROR("active_oom_killer: KILLING PID %u (device %u, memory %llu bytes)", 
                             actual_pid, dev_idx, (unsigned long long)infos[i].usedGpuMemory);
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

// Comparison function for qsort - sort by PID descending (highest/newest first)
// Higher PID typically means newer process, so we kill newest processes first
static int compare_process_memory(const void* a, const void* b) {
    const process_memory_info_t* pa = (const process_memory_info_t*)a;
    const process_memory_info_t* pb = (const process_memory_info_t*)b;
    
    // Sort by PID descending (highest/newest PID first)
    if (pa->pid > pb->pid) return -1;
    if (pa->pid < pb->pid) return 1;
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
    
    // CRITICAL: Initialize version field for all structs before calling NVML
    // This ensures compatibility with different driver versions (CUDA 12.2 vs driver 570.195.03)
    for (unsigned int j = 0; j < SHARED_REGION_MAX_PROCESS_NUM; j++) {
        infos[j].version = nvmlProcessInfo_v2;
    }
    
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
    
    // #region Detailed NVML struct logging - helps debug struct mismatches
    LOG_DEBUG("RAW_NVML_BYPASS: Received %u processes from NVML (gradual_oom_killer), struct_size=%zu bytes", 
              process_count, sizeof(nvmlProcessInfo_t));
    for (unsigned int i = 0; i < process_count && i < 10; i++) {
        unsigned int safe_pid = extract_pid_safely((void *)&infos[i]);
        LOG_DEBUG("RAW_NVML_BYPASS Process[%u]: version=%u header_pid=%u safe_pid=%u memory=%llu struct_size=%zu", 
                  i, infos[i].version, infos[i].pid, safe_pid,
                  (unsigned long long)infos[i].usedGpuMemory,
                  sizeof(nvmlProcessInfo_t));
        if (safe_pid != infos[i].pid && safe_pid != 0) {
            LOG_WARN("RAW_NVML_BYPASS Process[%u]: STRUCT MISMATCH - header_pid=%u, safe_pid=%u", 
                     i, infos[i].pid, safe_pid);
        }
    }
    // #endregion
    
    // Filter processes belonging to current cgroup/UID and collect them
    process_memory_info_t filtered_processes[SHARED_REGION_MAX_PROCESS_NUM];
    unsigned int filtered_count = 0;
    
    for (unsigned int i = 0; i < process_count; i++) {
        // CRITICAL: Use safe PID extraction to handle struct mismatches
        unsigned int actual_pid = extract_pid_safely((void *)&infos[i]);
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
    
    LOG_ERROR("gradual_oom_killer: Found %u processes on device %d, sorted by PID (newest/highest PID first)", 
              filtered_count, cuda_dev);
    
    uint64_t limit = get_current_device_memory_limit(cuda_dev);
    int killed = 0;
    time_t start_time = time(NULL);
    const int MAX_OOM_DURATION = 30;  // Kill cgroup if over limit for 30 seconds
    
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
            usage = get_gpu_memory_usage_nolock(cuda_dev);
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
            break;
        }
    }
    unlock_shrreg();
    return 1;
}

// Get summed memory usage from NVML for a CUDA device index
// This uses the same calculation as nvmlDeviceGetMemoryInfo (9MB min + 5% overhead + UID filtering)
// Always uses the summed calculation - no fallback to tracked usage
uint64_t get_summed_device_memory_usage_from_nvml(int cuda_dev) {
    unsigned int nvml_dev_idx = cuda_to_nvml_map(cuda_dev);
    nvmlDevice_t ndev;
    nvmlReturn_t ret = nvmlDeviceGetHandleByIndex(nvml_dev_idx, &ndev);
    if (ret != NVML_SUCCESS) {
        LOG_WARN("get_summed_device_memory_usage_from_nvml: NVML get device %d (CUDA %d) error, %s", 
                 nvml_dev_idx, cuda_dev, nvmlErrorString(ret));
        return 0;
    }
    
    unsigned int process_count = SHARED_REGION_MAX_PROCESS_NUM;
    nvmlProcessInfo_t infos[SHARED_REGION_MAX_PROCESS_NUM];
    
    // CRITICAL: Initialize version field for all structs before calling NVML
    // This ensures compatibility with different driver versions (CUDA 12.2 vs driver 570.195.03)
    for (unsigned int j = 0; j < SHARED_REGION_MAX_PROCESS_NUM; j++) {
        infos[j].version = nvmlProcessInfo_v2;
    }
    
    // Bypass hook to get ALL processes directly from NVML, then filter ourselves
    // This ensures we see all processes, not just the ones the hook filtered
    // The hook filters processes, but we need to see all processes to calculate accurate memory usage
    if (nvml_library_entry != NULL) {
        ret = NVML_OVERRIDE_CALL_NO_LOG(nvml_library_entry, nvmlDeviceGetComputeRunningProcesses_v2,
                                        ndev, &process_count, infos);
    } else {
        // Fallback if nvml_library_entry not available (shouldn't happen in normal operation)
        ret = nvmlDeviceGetComputeRunningProcesses(ndev, &process_count, infos);
    }
    
    if (ret != NVML_SUCCESS && ret != NVML_ERROR_INSUFFICIENT_SIZE) {
        LOG_WARN("get_summed_device_memory_usage_from_nvml: nvmlDeviceGetComputeRunningProcesses failed: %d (%s)", 
                 ret, nvmlErrorString(ret));
        return 0;
    }
    
    uint64_t total_usage = 0;
    const uint64_t MIN_PROCESS_MEMORY = 9 * 1024 * 1024;  // 9 MB minimum per process
    const double PROCESS_OVERHEAD_PERCENT = 0.05;  // 5% overhead
    const uint64_t NVML_VALUE_NOT_AVAILABLE_ULL = 0xFFFFFFFFFFFFFFFFULL;
    
    // Get current user's UID for fallback filtering
    uid_t current_uid = getuid();
    
    // Sum up memory from all processes belonging to current SLURM job (or current user if not in SLURM)
    unsigned int included_count = 0;
    unsigned int skipped_count = 0;
    unsigned int pid_extract_failed = 0;
    
    LOG_FILE_DEBUG("get_summed_device_memory_usage_from_nvml: Starting - process_count=%u current_uid=%u current_pid=%d", 
                   process_count, current_uid, getpid());
    
    for (unsigned int i = 0; i < process_count; i++) {
        // CRITICAL: Use safe PID extraction to handle struct mismatches
        unsigned int actual_pid = extract_pid_safely((void *)&infos[i]);
        if (actual_pid == 0) {
            pid_extract_failed++;
            LOG_FILE_DEBUG("get_summed_device_memory_usage_from_nvml: Process[%u] - could not extract PID (header pid=%u), skipping", 
                         i, infos[i].pid);
            continue;  // Skip if we can't get a valid PID
        }
        
        // First try to check if process belongs to current cgroup session
        int cgroup_check = proc_belongs_to_current_cgroup_session(actual_pid);
        
        LOG_FILE_DEBUG("get_summed_device_memory_usage_from_nvml: Process[%u] PID %u - cgroup_check=%d memory=%llu", 
                     i, actual_pid, cgroup_check, (unsigned long long)infos[i].usedGpuMemory);
        
        if (cgroup_check == -1) {
            // Couldn't determine cgroup or not in a cgroup session - fall back to UID check
            uid_t proc_uid = proc_get_uid(actual_pid);
            
            if (proc_uid == (uid_t)-1) {
                // Couldn't read UID - skip this process to avoid blocking on shared region lock
                LOG_WARN("get_summed_device_memory_usage_from_nvml: Process[%u] PID %u - could not read UID, skipping", 
                         i, actual_pid);
                skipped_count++;
                continue;
            } else if (proc_uid != current_uid) {
                LOG_FILE_DEBUG("get_summed_device_memory_usage_from_nvml: Process[%u] PID %u - UID %u != current UID %u, skipping", 
                             i, actual_pid, proc_uid, current_uid);
                skipped_count++;
                continue;
            }
        } else if (cgroup_check == 0) {
            // Process is in a different cgroup session - skip it
            LOG_FILE_DEBUG("get_summed_device_memory_usage_from_nvml: Process[%u] PID %u - different cgroup (check=%d), skipping", 
                         i, actual_pid, cgroup_check);
            skipped_count++;
            continue;
        }
        // cgroup_check == 1 means process belongs to current cgroup session - include it
        
        // Skip if memory value is not available (NVML_VALUE_NOT_AVAILABLE) or invalid
        if (infos[i].usedGpuMemory != NVML_VALUE_NOT_AVAILABLE_ULL && infos[i].usedGpuMemory > 0) {
            uint64_t process_mem = infos[i].usedGpuMemory;
            // Add 5% overhead, then ensure minimum
            uint64_t process_mem_with_overhead = (uint64_t)(process_mem * (1.0 + PROCESS_OVERHEAD_PERCENT));
            uint64_t process_mem_counted = (process_mem_with_overhead < MIN_PROCESS_MEMORY) ? MIN_PROCESS_MEMORY : process_mem_with_overhead;
            total_usage += process_mem_counted;
            included_count++;
            LOG_FILE_DEBUG("get_summed_device_memory_usage_from_nvml: Process[%u] PID %u - INCLUDED: memory=%llu bytes (%.2f GB), total_usage now=%llu bytes (%.2f GB)", 
                         i, actual_pid, (unsigned long long)process_mem, 
                         process_mem / (1024.0 * 1024.0 * 1024.0),
                         (unsigned long long)total_usage,
                         total_usage / (1024.0 * 1024.0 * 1024.0));
        } else {
            // Even if NVML reports 0 or unavailable, count minimum for the process
            total_usage += MIN_PROCESS_MEMORY;
            included_count++;
            LOG_FILE_DEBUG("get_summed_device_memory_usage_from_nvml: Process[%u] PID %u - INCLUDED (min memory): memory=%llu (invalid/unavailable), total_usage now=%llu bytes", 
                         i, actual_pid, (unsigned long long)infos[i].usedGpuMemory, (unsigned long long)total_usage);
        }
    }
    
    LOG_FILE_DEBUG("get_summed_device_memory_usage_from_nvml: Final - included=%u skipped=%u pid_extract_failed=%u total_usage=%llu bytes (%.2f GB)", 
                  included_count, skipped_count, pid_extract_failed, (unsigned long long)total_usage, 
                  total_usage / (1024.0 * 1024.0 * 1024.0));
    
    return total_usage;
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
    struct timeval tv;
    gettimeofday(&tv, NULL);  // struggle with clock_gettime version
    spec->tv_sec = tv.tv_sec + seconds;
    spec->tv_nsec = 0;
}

int fix_lock_shrreg() {
    int res = 1;
    if (region_info.fd == -1) {
        // should never happen
        LOG_ERROR("Uninitialized shrreg");
    }
    // upgrade
    if (lockf(region_info.fd, F_LOCK, SHARED_REGION_SIZE_MAGIC) != 0) {
        LOG_ERROR("Fail to upgraded lock: errno=%d", errno);
    }
    SEQ_POINT_MARK(SEQ_FIX_SHRREG_ACQUIRE_FLOCK_OK);

    shared_region_t* region = region_info.shared_region;
    int32_t current_owner = region->owner_pid;
    if (current_owner != 0) {
        int flag = 0;
        if (current_owner == region_info.pid) {
            // Detect owner pid = self pid
            LOG_WARN("Owner pid equals self pid (%d), indicates pid loopback or race condition", current_owner);
            flag = 1;
        } else {
            int proc_status = proc_alive(current_owner);
            if (proc_status == PROC_STATE_NONALIVE) {
                LOG_INFO("Kick dead owner proc (%d)", current_owner);
                flag = 1;
            }
        }
        if (flag == 1) {
            region->owner_pid = region_info.pid;
            SEQ_POINT_MARK(SEQ_FIX_SHRREG_UPDATE_OWNER_OK);
            res = 0;     
        }
    }

    if (lockf(region_info.fd, F_ULOCK, SHARED_REGION_SIZE_MAGIC) != 0) {
        LOG_ERROR("Fail to upgraded unlock: errno=%d", errno);
    }
    SEQ_POINT_MARK(SEQ_FIX_SHRREG_RELEASE_FLOCK_OK);
    return res;
}

void exit_withlock(int exitcode) {
    unlock_shrreg();
    exit(exitcode);
}


// External function from config_file.c - cleanup config file
// cleanup_config_file removed - cleanup is handled by SLURM epilog script

void exit_handler() {
    if (region_info.init_status == PTHREAD_ONCE_INIT) {
        return;
    }
    shared_region_t* region = region_info.shared_region;
    
    // Check if shared region was never initialized (e.g., program failed to start)
    // This can happen when bash loads the library but the program doesn't exist
    if (region == NULL) {
        // Nothing to clean up if shared region wasn't initialized
        // Config file cleanup is handled by SLURM epilog script
        return;
    }
    
    int slot = 0;
    LOG_MSG("Calling exit handler %d",getpid());
    
    // Note: Config file cleanup is handled by the SLURM epilog script, not by individual processes
    
    struct timespec sem_ts;
    get_timespec(SEM_WAIT_TIME_ON_EXIT, &sem_ts);
    int status = sem_timedwait(&region->sem, &sem_ts);
    if (status == 0) {  // just give up on lock failure
        region->owner_pid = region_info.pid;
        while (slot < region->proc_num) {
            if (region->procs[slot].pid == region_info.pid) {
                memset(region->procs[slot].used,0,sizeof(device_memory_t)*CUDA_DEVICE_MAX_COUNT);
                memset(region->procs[slot].device_util,0,sizeof(device_util_t)*CUDA_DEVICE_MAX_COUNT);
                region->proc_num--;
                region->procs[slot] = region->procs[region->proc_num];
                break;
            }
            slot++;
        }
        __sync_synchronize();
        region->owner_pid = 0;
        sem_post(&region->sem);
    } else {
        LOG_WARN("Failed to take lock on exit: errno=%d", errno);
    }
}


void lock_shrreg() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;  // No-op when softmig is disabled
    }
    struct timespec sem_ts;
    get_timespec(SEM_WAIT_TIME, &sem_ts);
    shared_region_t* region = region_info.shared_region;
    int trials = 0;
    while (1) {
        int status = sem_timedwait(&region->sem, &sem_ts);
        SEQ_POINT_MARK(SEQ_ACQUIRE_SEMLOCK_OK);

        if (status == 0) {
            // TODO: irregular exit here will hang pending locks
            region->owner_pid = region_info.pid;
            __sync_synchronize();
            SEQ_POINT_MARK(SEQ_UPDATE_OWNER_OK);
            trials = 0;
            break;
        } else if (errno == ETIMEDOUT) {
            LOG_WARN("Lock shrreg timeout, try fix (%d:%ld)", region_info.pid,region->owner_pid);
            int32_t current_owner = region->owner_pid;
            if (current_owner != 0 && (current_owner == region_info.pid ||
                    proc_alive(current_owner) == PROC_STATE_NONALIVE)) {
                LOG_WARN("Owner proc dead (%d), try fix", current_owner);
                if (0 == fix_lock_shrreg()) {
                    break;
                }
            } else {
                trials++;
                if (trials > SEM_WAIT_RETRY_TIMES) {
                    LOG_WARN("Fail to lock shrreg in %d seconds",
                        SEM_WAIT_RETRY_TIMES * SEM_WAIT_TIME);
                    if (current_owner == 0) {
                        LOG_WARN("fix current_owner 0>%d",region_info.pid);
                        region->owner_pid = region_info.pid;
                        if (0 == fix_lock_shrreg()) {
                            break;
                        } 
                    }
                }
                continue;  // slow wait path
            }
        } else {
            LOG_ERROR("Failed to lock shrreg: %d", errno);
        }
    }
}

void unlock_shrreg() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;  // No-op when softmig is disabled
    }
    SEQ_POINT_MARK(SEQ_BEFORE_UNLOCK_SHRREG);
    shared_region_t* region = region_info.shared_region;

    __sync_synchronize();
    region->owner_pid = 0;
    // TODO: irregular exit here will hang pending locks
    SEQ_POINT_MARK(SEQ_RESET_OWNER_OK);

    sem_post(&region->sem);
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
        if (pid != 0) {
            if (do_clear > 0 && proc_alive(pid) == PROC_STATE_NONALIVE) {
                LOG_WARN("Kick dead proc %d", pid);
            } else {
                slot++;
                continue;
            }
            res=1;
            region->proc_num--;
            region->procs[slot] = region->procs[region->proc_num];
            __sync_synchronize();
        }
    }
    return res;
}

void init_proc_slot_withlock() {
    int32_t current_pid = getpid();
    lock_shrreg();
    shared_region_t* region = region_info.shared_region;
    if (region->proc_num >= SHARED_REGION_MAX_PROCESS_NUM) {
        exit_withlock(-1);
    }
    signal(SIGUSR2,sig_swap_stub);
    signal(SIGUSR1,sig_restore_stub);
    // If, by any means a pid of itself is found in region->proces, then it is probably caused by crashloop
    // we need to reset it.
    int i,found=0;
    for (i=0; i<region->proc_num; i++) {
        if (region->procs[i].pid == current_pid) {
            region->procs[i].status = 1;
            memset(region->procs[i].used,0,sizeof(device_memory_t)*CUDA_DEVICE_MAX_COUNT);
            memset(region->procs[i].device_util,0,sizeof(device_util_t)*CUDA_DEVICE_MAX_COUNT);
            found = 1;
            break;
        }
    }
    if (!found) {
        region->procs[region->proc_num].pid = current_pid;
        region->procs[region->proc_num].status = 1;
        memset(region->procs[region->proc_num].used,0,sizeof(device_memory_t)*CUDA_DEVICE_MAX_COUNT);
        memset(region->procs[region->proc_num].device_util,0,sizeof(device_util_t)*CUDA_DEVICE_MAX_COUNT);
        region->proc_num++;
    }

    clear_proc_slot_nolock(1);
    unlock_shrreg();
}

void print_all() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        LOG_INFO("softmig is disabled - no process information available");
        return;
    }
}

void child_reinit_flag() {
    LOG_DEBUG("Detect child pid: %d -> %d", region_info.pid, getpid());   
    region_info.init_status = PTHREAD_ONCE_INIT;
}

int set_active_oom_killer() {
    // Always enabled when softmig is active (no env var needed)
    return 1;
}

int set_env_utilization_switch() {
    // Always enabled when softmig is active (no env var needed)
    return 1;
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

    enable_active_oom_killer = set_active_oom_killer();
    env_utilization_switch = set_env_utilization_switch();
    pthread_atfork(NULL, NULL, child_reinit_flag);

    region_info.pid = getpid();
    region_info.fd = -1;
    region_info.last_kernel_time = time(NULL);

    umask(0);

    char* shr_reg_file = getenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV);
    if (shr_reg_file == NULL) {
        // Compute Canada optimized: Use SLURM_TMPDIR with job ID for isolation
        // Only use SLURM_TMPDIR (not regular /tmp) for proper job isolation
        static char cache_path[512] = {0};
        char* tmpdir = getenv("SLURM_TMPDIR");
        if (tmpdir == NULL) {
            // No SLURM_TMPDIR - this should only happen outside SLURM jobs
            // For local testing, use /tmp with job ID if available
            char* job_id = getenv("SLURM_JOB_ID");
            if (job_id != NULL) {
                // We're in a SLURM job but SLURM_TMPDIR not set - use /tmp with job ID
                tmpdir = "/tmp";
            } else {
                // Not in SLURM job - use /tmp (for local testing only)
                tmpdir = "/tmp";
            }
        }
        
        // Include job ID for proper isolation (per-job cache)
        // For oversubscription, each job gets its own cache but they coordinate via shared memory
        char* job_id = getenv("SLURM_JOB_ID");
        char* array_id = getenv("SLURM_ARRAY_TASK_ID");
        
        if (job_id != NULL) {
            if (array_id != NULL) {
                snprintf(cache_path, sizeof(cache_path), "%s/cudevshr.cache.%s.%s", tmpdir, job_id, array_id);
            } else {
                snprintf(cache_path, sizeof(cache_path), "%s/cudevshr.cache.%s", tmpdir, job_id);
            }
        } else {
            // Fallback: use user ID and PID
            uid_t uid = getuid();
            pid_t pid = getpid();
            snprintf(cache_path, sizeof(cache_path), "%s/cudevshr.cache.uid%d.pid%d", tmpdir, uid, pid);
        }
        shr_reg_file = cache_path;
    }
    // Initialize NVML BEFORE!! open it
    //nvmlInit();

    /* If you need sm modification, do it here */
    /* ... set_sm_scale */

    int fd = open(shr_reg_file, O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        LOG_ERROR("Fail to open shrreg %s: errno=%d", shr_reg_file, errno);
    }
    region_info.fd = fd;
    size_t offset = lseek(fd, SHARED_REGION_SIZE_MAGIC, SEEK_SET);
    if (offset != SHARED_REGION_SIZE_MAGIC) {
        LOG_ERROR("Fail to init shrreg %s: errno=%d", shr_reg_file, errno);
    }
    size_t check_bytes = write(fd, "\0", 1);
    if (check_bytes != 1) {
        LOG_ERROR("Fail to write shrreg %s: errno=%d", shr_reg_file, errno);
    }
    if (lseek(fd, 0, SEEK_SET) != 0) {
        LOG_ERROR("Fail to reseek shrreg %s: errno=%d", shr_reg_file, errno);
    }
    region_info.shared_region = (shared_region_t*) mmap(
        NULL, SHARED_REGION_SIZE_MAGIC, 
        PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
    shared_region_t* region = region_info.shared_region;
    if (region == NULL) {
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
        if (sem_init(&region->sem, 1, 1) != 0) {
            LOG_ERROR("Fail to init sem %s: errno=%d", shr_reg_file, errno);
        }
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
    }
    result = get_gpu_memory_usage(dev);
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
    }
    return NULL;
}


