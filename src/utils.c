/**
 * @file utils.c
 * @brief PID detection, process list merging, CUDA device mapping, and file-based locking.
 *
 * Implements set_task_pid (discovers the NVML host PID for the current process
 * by creating a temporary CUDA context and diffing the process list),
 * CUDA_VISIBLE_DEVICES parsing, and a SLURM-aware file lock used to
 * serialize initialization across co-located processes.
 */
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>
#include <time.h>
#include <sys/file.h>
#include <errno.h>
#include "include/utils.h"
#include "include/log_utils.h"
#include "include/nvml_prefix.h"
// Don't include <nvml.h> directly - use nvml-subset.h through libnvml_hook.h to avoid conflicts
// Include libnvml_hook.h first to get nvmlReturn_t defined before nvml_override.h uses it
#include "include/libnvml_hook.h"  // This includes nvml-subset.h which has all needed types
#include "include/nvml_override.h"
#include "include/libcuda_hook.h"
#include "include/process_utils.h"
#include "multiprocess/multiprocess_memory_limit.h"

// Note: fp1 is now defined in log_file.c to avoid linking issues with standalone tools
// Access to nvml_library_entry to bypass filtering in set_task_pid
extern entry_t nvml_library_entry[];

// Forward declarations for NVML symbols used here.
const char *nvmlErrorString(nvmlReturn_t result);
nvmlReturn_t nvmlInit(void);
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount);

// Helper to get lock file path (Compute Canada optimized - uses SLURM_TMPDIR)
static char* get_unified_lock_path(void) {
    static char lock_path[512] = {0};
    static int initialized = 0;
    
    if (initialized) {
        return lock_path;
    }
    
    // Check for custom lock path
    char* custom_lock = getenv("SOFTMIG_LOCK_FILE");
    if (custom_lock != NULL && strlen(custom_lock) > 0) {
        strncpy(lock_path, custom_lock, sizeof(lock_path) - 1);
        initialized = 1;
        return lock_path;
    }
    
    char* tmpdir = getenv("SLURM_TMPDIR");
    if (tmpdir == NULL) tmpdir = "/tmp";
    
    // Include job ID for isolation
    char* job_id = getenv("SLURM_JOB_ID");
    if (job_id != NULL) {
        snprintf(lock_path, sizeof(lock_path), "%s/vgpulock/lock.%s", tmpdir, job_id);
    } else {
        // Fallback: use user ID
        uid_t uid = getuid();
        snprintf(lock_path, sizeof(lock_path), "%s/vgpulock/lock.uid%d", tmpdir, uid);
    }
    
    // Create directory if needed
    char dir_path[512];
    strncpy(dir_path, lock_path, sizeof(dir_path) - 1);
    dir_path[sizeof(dir_path) - 1] = '\0';
    char* last_slash = strrchr(dir_path, '/');
    if (last_slash != NULL) {
        *last_slash = '\0';
        mkdir(dir_path, 0755);  // Ignore errors
    }
    
    initialized = 1;
    return lock_path;
}

static int unified_lock_fd = -1;
extern size_t context_size;
extern int cuda_to_nvml_map_array[CUDA_DEVICE_MAX_COUNT];

/** Acquire an exclusive flock on the unified lock file. Returns 0 on success. */
int try_lock_unified_lock() {
    if (unified_lock_fd == -1) {
        const char* path = get_unified_lock_path();
        unified_lock_fd = open(path, O_CREAT | O_RDWR, 0600);
        if (unified_lock_fd == -1) {
            LOG_ERROR("try_lock_unified_lock: open(%s) failed errno=%d", path, errno);
            return -1;
        }
    }
    if (flock(unified_lock_fd, LOCK_EX) != 0) {
        LOG_ERROR("try_lock_unified_lock: flock failed errno=%d", errno);
        return -1;
    }
    return 0;
}

/** Release the flock on the unified lock file. Returns 0 on success. */
int try_unlock_unified_lock() {
    if (unified_lock_fd == -1) return -1;
    if (flock(unified_lock_fd, LOCK_UN) != 0) {
        LOG_ERROR("try_unlock_unified_lock: flock unlock failed errno=%d", errno);
        return -1;
    }
    return 0;
}

/**
 * Merge new process entries from sub into merged, skipping duplicates and invalid PIDs.
 * @return 0 on success.
 */
int mergepid(unsigned int *prev, unsigned int *current, nvmlProcessInfo_t *sub, nvmlProcessInfo_t *merged) {
    int i,j;
    int found=0;
    
    // Validate input - skip invalid PIDs (0 or -1/UINT_MAX)
    for (i=0;i<*prev;i++){
        // Skip invalid PIDs - these indicate corrupted or uninitialized data
        if (sub[i].pid == 0 || sub[i].pid == (unsigned int)-1) {
            LOG_DEBUG("mergepid: Skipping invalid PID %u at index %d", sub[i].pid, i);
            continue;
        }
        
        found=0;
        // Only check against already-merged valid PIDs
        for (j=0;j<*current;j++) {
            // Also skip invalid PIDs in merged list during comparison
            if (merged[j].pid == 0 || merged[j].pid == (unsigned int)-1) {
                continue;
            }
            if (sub[i].pid == merged[j].pid) {
                found = 1;
                break;
            } 
        }
        if (!found) {
            // Check bounds before adding
            if (*current >= SHARED_REGION_MAX_PROCESS_NUM) {
                LOG_WARN("mergepid: Merged list full (%u), cannot add PID %u", *current, sub[i].pid);
                break;
            }
            merged[*current].pid = sub[i].pid;
            merged[*current].usedGpuMemory = sub[i].usedGpuMemory;
            (*current)++;
        }
    }
    return 0;
}

/** Find the PID present in pids_on_device but absent from pre_pids_on_device. */
int getextrapid(unsigned int prev, unsigned int current, nvmlProcessInfo_t *pre_pids_on_device, nvmlProcessInfo_t *pids_on_device) {
    int i,j;
    int found = 0;
    LOG_DEBUG("getextrapid: prev=%u current=%u", prev, current);
    if (current <= prev) {
        LOG_WARN("getextrapid: current (%u) <= prev (%u), cannot find new PID", current, prev);
        return 0;
    }
    if (current > SHARED_REGION_MAX_PROCESS_NUM)
        current = SHARED_REGION_MAX_PROCESS_NUM;
    if (prev > SHARED_REGION_MAX_PROCESS_NUM)
        prev = SHARED_REGION_MAX_PROCESS_NUM;
    for (i=0; i<(int)current; i++) {
        found = 0;
        for (j=0; j<(int)prev; j++) {
            if (pids_on_device[i].pid == pre_pids_on_device[j].pid) {
                found = 1;
                break;
            }
        }
        if (!found) {
            LOG_DEBUG("getextrapid: Found new PID %u (not in previous list)", pids_on_device[i].pid);
            return pids_on_device[i].pid;
        }
    }
    LOG_WARN("getextrapid: All current PIDs found in previous list, no new PID detected");
    return 0;
}

/**
 * Detect the NVML-visible host PID for the current process.
 *
 * Snapshots the NVML process list, creates a temporary CUDA context to make
 * the current process appear, then diffs the lists. Falls back to getpid()
 * if the PID cannot be found via differencing. Registers the host PID in
 * the shared region and records the initial context memory size.
 */
nvmlReturn_t set_task_pid() {
    unsigned int running_processes=0,previous=0,merged_num=0;
    nvmlProcessInfo_t tmp_pids_on_device[SHARED_REGION_MAX_PROCESS_NUM];
    nvmlProcessInfo_t pre_pids_on_device[SHARED_REGION_MAX_PROCESS_NUM];
    nvmlProcessInfo_t pids_on_device[SHARED_REGION_MAX_PROCESS_NUM];
    nvmlDevice_t device;
    nvmlReturn_t res;
    CUcontext pctx;
    int i;
    
    // Initialize arrays to zero to avoid garbage values
    memset(pre_pids_on_device, 0, sizeof(pre_pids_on_device));
    memset(pids_on_device, 0, sizeof(pids_on_device));
    memset(tmp_pids_on_device, 0, sizeof(tmp_pids_on_device));
    
    CHECK_NVML_API(nvmlInit());
    CHECK_NVML_API(nvmlDeviceGetHandleByIndex(0, &device));
    
    unsigned int nvmlCounts;
    CHECK_NVML_API(nvmlDeviceGetCount(&nvmlCounts));
    
    int cudaDev;
    for (i=0;i<nvmlCounts;i++){
        cudaDev=nvml_to_cuda_map(i);
        if (cudaDev<0) {
            continue;
        }
        CHECK_NVML_API(nvmlDeviceGetHandleByIndex(i, &device));
        do{
            // Bypass our filtering hook - call real NVML function directly to get ALL processes
            // This is needed for PID detection to work correctly
            res = NVML_OVERRIDE_CALL_NO_LOG(nvml_library_entry, nvmlDeviceGetComputeRunningProcesses_v2,
                                            device, &previous, tmp_pids_on_device);
            if ((res != NVML_SUCCESS) && (res != NVML_ERROR_INSUFFICIENT_SIZE)) {
                LOG_ERROR("Device2GetComputeRunningProcesses failed %d,%d\n",res,i);
                return res;
            }
        }while(res==NVML_ERROR_INSUFFICIENT_SIZE);
        
        // Log raw NVML data before merging (for debugging PID=0 issues and struct mismatches)
        LOG_DEBUG("set_task_pid: BEFORE context - NVML returned %u processes on device %d", previous, i);
        for (int k=0; k<previous && k<10; k++) {  // Log first 10 to avoid spam
            if (tmp_pids_on_device[k].pid == 0) {
                LOG_WARN("  Raw NVML[%d]: INVALID PID=0", k);
            }
        }
        
        mergepid(&previous,&merged_num,tmp_pids_on_device,pre_pids_on_device);
        break;
    }
    previous = merged_num;
    merged_num = 0;
    memset(tmp_pids_on_device,0,sizeof(nvmlProcessInfo_t)*SHARED_REGION_MAX_PROCESS_NUM);
    CHECK_CU_RESULT(cuDevicePrimaryCtxRetain(&pctx,0));
    for (i=0;i<nvmlCounts;i++) {
        cudaDev=nvml_to_cuda_map(i);
        if (cudaDev<0) {
            continue;
        }
        CHECK_NVML_API(nvmlDeviceGetHandleByIndex (i, &device)); 
        do{
            // Bypass our filtering hook - call real NVML function directly to get ALL processes
            // This is needed for PID detection to work correctly
            res = NVML_OVERRIDE_CALL_NO_LOG(nvml_library_entry, nvmlDeviceGetComputeRunningProcesses_v2,
                                            device, &running_processes, tmp_pids_on_device);
            if ((res != NVML_SUCCESS) && (res != NVML_ERROR_INSUFFICIENT_SIZE)) {
                LOG_ERROR("Device2GetComputeRunningProcesses failed %d\n",res);
                return res;
            }
        }while(res == NVML_ERROR_INSUFFICIENT_SIZE);
        
        // Log raw NVML data after creating context (for debugging PID=0 issues and struct mismatches)
        LOG_DEBUG("set_task_pid: AFTER context - NVML returned %u processes on device %d", running_processes, i);
        for (int k=0; k<running_processes && k<10; k++) {  // Log first 10 to avoid spam
            if (tmp_pids_on_device[k].pid == 0) {
                LOG_WARN("  Raw NVML[%d]: INVALID PID=0", k);
            }
        }
        
        mergepid(&running_processes,&merged_num,tmp_pids_on_device,pids_on_device);
        break;
    }
    running_processes = merged_num;
    LOG_DEBUG("set_task_pid: Merged process counts - previous=%u, running=%u", previous, running_processes);
    
    // With cgroup filtering, try to find our own PID first (most reliable in SLURM/cgroup environments)
    pid_t current_pid = getpid();
    unsigned int hostpid = 0;
    
    // Check if our PID is in the filtered process list (after creating CUDA context)
    LOG_DEBUG("set_task_pid: Looking for current PID %d in %u filtered processes", current_pid, running_processes);
    for (i=0; i<running_processes; i++) {
        if (pids_on_device[i].pid == (unsigned int)current_pid) {
            hostpid = current_pid;
            LOG_DEBUG("set_task_pid: Found current process PID %d in filtered GPU process list", current_pid);
            break;
        }
        if (pids_on_device[i].pid == 0) {
            LOG_WARN("set_task_pid: WARNING - Found invalid PID=0 at index %d in merged list!", i);
        }
    }
    
    // If not found, fall back to the old method (difference detection)
    if (hostpid == 0) {
        LOG_WARN("set_task_pid: Current PID %d NOT found in filtered list (checked %u processes), trying difference detection", 
                 current_pid, running_processes);
        hostpid = getextrapid(previous,running_processes,pre_pids_on_device,pids_on_device);
        if (hostpid != 0) {
            LOG_DEBUG("set_task_pid: Difference detection found PID %u", hostpid);
        }
    }
    
    if (running_processes == 0) {
        LOG_WARN("set_task_pid: No processes in merged list after filtering!");
    }
    
    if (hostpid==0) {
        LOG_ERROR("set_task_pid: Current PID %d NOT found in NVML process list (filtered processes=%u, previous=%u)", 
                 current_pid, running_processes, previous);
        LOG_ERROR("set_task_pid: All PIDs in merged list:");
        for (i=0; i<running_processes && i<20; i++) {
            LOG_ERROR("  [%d]=%u %s", i, pids_on_device[i].pid,
                     (pids_on_device[i].pid == 0 || pids_on_device[i].pid == (unsigned int)-1) ? "(INVALID!)" : "");
        }
        // Use getpid() directly as fallback - this is the actual current process PID
        // The hostpid is used for memory tracking, so we must use the real PID, not a wrong one
        // This can happen if the process hasn't created a CUDA context yet, or NVML returns corrupted data
        LOG_WARN("set_task_pid: Using getpid() directly as fallback: %d (not found in NVML list)", current_pid);
        hostpid = current_pid;
    }
    
    LOG_DEBUG("set_task_pid: hostPid=%d",hostpid);
    if (set_host_pid(hostpid)==0) {
        for (i=0;i<running_processes;i++) {
            if (pids_on_device[i].pid==hostpid) {
                LOG_DEBUG("set_task_pid: Primary Context Size=%lld",(long long)tmp_pids_on_device[i].usedGpuMemory);
                context_size = tmp_pids_on_device[i].usedGpuMemory; 
                break;
            }
        }
    }
    CHECK_CU_RESULT(cuDevicePrimaryCtxRelease(0));
    return NVML_SUCCESS; 
}

int parse_cuda_visible_env() {
    char *s = getenv("CUDA_VISIBLE_DEVICES");
    int count = 0;
    for (int i = 0; i < CUDA_DEVICE_MAX_COUNT; i++) {
        cuda_to_nvml_map_array[i] = i;
    }

    if (need_cuda_virtualize()) {
        for (int i = 0; i < strlen(s); i++) {
            if ((s[i] == ',') || (i == 0)) {
                int tmp = (i==0) ? atoi(s) : atoi(s + i +1);
                cuda_to_nvml_map_array[count] = tmp; 
                count++;
            }
        } 
    }
    return count;
}

/** Parse CUDA_VISIBLE_DEVICES and populate the CUDA-to-NVML device mapping. */
int map_cuda_visible_devices() {
    parse_cuda_visible_env();
    return 0;
}

int getenvcount() {
    char *s = getenv("CUDA_VISIBLE_DEVICES");
    if ((s == NULL) || (strlen(s)==0)){
        return -1;
    }
    LOG_DEBUG("get from env %s",s);
    int i,count=0;
    for (i=0;i<strlen(s);i++){
        if (s[i]==',')
            count++;
    }
    return count+1;
}

int need_cuda_virtualize() {
    int count1 = -1;
    char *s = getenv("CUDA_VISIBLE_DEVICES");
    if ((s == NULL) || (strlen(s)==0)){
        return 0;
    }
    int fromenv = getenvcount();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetCount,&count1);
    if (res != CUDA_SUCCESS) {
        return 1;
    }
    LOG_DEBUG("count1=%d",count1);
    if (fromenv ==count1) {
        return 1;
    }
    return 0;
}
