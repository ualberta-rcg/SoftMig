#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>
#include <time.h>
#include "include/utils.h"
#include "include/log_utils.h"
#include "include/nvml_prefix.h"
// Don't include <nvml.h> directly - use nvml-subset.h through libnvml_hook.h to avoid conflicts
// Include libnvml_hook.h first to get nvmlReturn_t defined before nvml_override.h uses it
#include "include/libnvml_hook.h"  // This includes nvml-subset.h which has all needed types
#include "include/nvml_override.h"
#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

// Note: fp1 is now defined in log_file.c to avoid linking issues with standalone tools
// Access to nvml_library_entry to bypass filtering in set_task_pid
extern entry_t nvml_library_entry[];

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
    
    // Use SLURM_TMPDIR only (per-job, auto-cleaned) - not regular /tmp
    char* tmpdir = getenv("SLURM_TMPDIR");
    if (tmpdir == NULL) {
        // No SLURM_TMPDIR - this should only happen outside SLURM jobs
        // Use a job-specific path if we have job ID
        char* job_id = getenv("SLURM_JOB_ID");
        if (job_id != NULL) {
            // We're in a SLURM job but SLURM_TMPDIR not set - use /tmp with job ID
            tmpdir = "/tmp";
        } else {
            // Not in SLURM job - use /tmp (for local testing only)
            tmpdir = "/tmp";
        }
    }
    
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

const char* unified_lock = NULL;  // Will be set dynamically
const int retry_count=20;
extern size_t context_size;
extern int cuda_to_nvml_map_array[CUDA_DEVICE_MAX_COUNT];

// 0 unified_lock lock success
// -1 unified_lock lock fail
int try_lock_unified_lock() {
    // Get lock path (SLURM-aware)
    if (unified_lock == NULL) {
        unified_lock = get_unified_lock_path();
    }
    
    // initialize the random number seed
    srand(time(NULL));
    int fd = open(unified_lock,O_CREAT | O_EXCL,S_IRWXU);
    int cnt = 0;
    while (fd == -1 && cnt <= retry_count) {
        if (cnt == retry_count) {
            LOG_MSG("unified_lock expired,removing...");
            int res = remove(unified_lock);
            LOG_MSG("remove unified_lock:%d",res);
        }else{
            LOG_MSG("unified_lock locked, waiting 1 second...");
            sleep(rand()%5 + 1);
        }
        cnt++;
        fd = open(unified_lock,O_CREAT | O_EXCL,S_IRWXU); 
    }
    LOG_INFO("try_lock_unified_lock:%d",fd);
    if (fd != -1) {
        close(fd);
        return 0;
    }
    return -1;
}

// 0 unified_lock unlock success
// -1 unified_lock unlock fail
int try_unlock_unified_lock() {
    int res = remove(unified_lock);
    LOG_INFO("try unlock_unified_lock:%d",res);
    return res == 0 ? 0 : -1;
}

int mergepid(unsigned int *prev, unsigned int *current, nvmlProcessInfo_t1 *sub, nvmlProcessInfo_t1 *merged) {
    int i,j;
    int found=0;
    for (i=0;i<*prev;i++){
        found=0;
        for (j=0;j<*current;j++) {
            if (sub[i].pid == merged[j].pid) {
                found = 1;
                break;
            } 
        }
        if (!found) {
            merged[*current].pid = sub[i].pid;
            (*current)++;
        }
    }
    return 0;
}

int getextrapid(unsigned int prev, unsigned int current, nvmlProcessInfo_t1 *pre_pids_on_device, nvmlProcessInfo_t1 *pids_on_device) {
    int i,j;
    int found = 0;
    LOG_INFO("getextrapid: prev=%u current=%u", prev, current);
    if (prev > 0) {
        LOG_INFO("Previous PIDs: ");
        for (i=0; i<prev; i++){
            LOG_INFO("  [%d]=%u", i, pre_pids_on_device[i].pid);
        }
    }
    if (current > 0) {
        LOG_INFO("Current PIDs: ");
        for (i=0; i<current; i++) {
            LOG_INFO("  [%d]=%u", i, pids_on_device[i].pid);
        }
    }
    if (current-prev<=0) {
        LOG_WARN("getextrapid: current (%u) <= prev (%u), cannot find new PID", current, prev);
        return 0;
    }
    for (i=0; i<current; i++) {
        found = 0;
        for (j=0; j<prev; j++) {
            if (pids_on_device[i].pid == pre_pids_on_device[j].pid) {
                found = 1;
                break;
            }
        }
        if (!found) {
            LOG_INFO("getextrapid: Found new PID %u (not in previous list)", pids_on_device[i].pid);
            return pids_on_device[i].pid;
        }
    }
    LOG_WARN("getextrapid: All current PIDs found in previous list, no new PID detected");
    return 0;
}

nvmlReturn_t set_task_pid() {
    unsigned int running_processes=0,previous=0,merged_num=0;
    nvmlProcessInfo_v1_t tmp_pids_on_device[SHARED_REGION_MAX_PROCESS_NUM];
    nvmlProcessInfo_t1 pre_pids_on_device[SHARED_REGION_MAX_PROCESS_NUM];
    nvmlProcessInfo_t1 pids_on_device[SHARED_REGION_MAX_PROCESS_NUM];
    nvmlDevice_t device;
    nvmlReturn_t res;
    CUcontext pctx;
    int i;
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
        
        // Log raw NVML data before merging (for debugging PID=0 issues)
        LOG_INFO("set_task_pid: BEFORE context - NVML returned %u processes on device %d", previous, i);
        for (int k=0; k<previous && k<10; k++) {  // Log first 10 to avoid spam
            LOG_INFO("  Raw NVML[%d]: PID=%u, memory=%llu", k, tmp_pids_on_device[k].pid, 
                    (unsigned long long)tmp_pids_on_device[k].usedGpuMemory);
        }
        
        mergepid(&previous,&merged_num,(nvmlProcessInfo_t1 *)tmp_pids_on_device,pre_pids_on_device);
        break;
    }
    previous = merged_num;
    merged_num = 0;
    memset(tmp_pids_on_device,0,sizeof(nvmlProcessInfo_v1_t)*SHARED_REGION_MAX_PROCESS_NUM);
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
        
        // Log raw NVML data after creating context (for debugging PID=0 issues)
        LOG_INFO("set_task_pid: AFTER context - NVML returned %u processes on device %d", running_processes, i);
        for (int k=0; k<running_processes && k<10; k++) {  // Log first 10 to avoid spam
            LOG_INFO("  Raw NVML[%d]: PID=%u, memory=%llu", k, tmp_pids_on_device[k].pid, 
                    (unsigned long long)tmp_pids_on_device[k].usedGpuMemory);
        }
        
        mergepid(&running_processes,&merged_num,(nvmlProcessInfo_t1 *)tmp_pids_on_device,pids_on_device);
        break;
    }
    running_processes = merged_num;
    LOG_INFO("set_task_pid: Merged process counts - previous=%u, running=%u", previous, running_processes);
    
    // With cgroup filtering, try to find our own PID first (most reliable in SLURM/cgroup environments)
    pid_t current_pid = getpid();
    unsigned int hostpid = 0;
    
    // Log all merged PIDs for debugging
    if (running_processes > 0) {
        LOG_INFO("set_task_pid: Merged process list (after filtering):");
        for (i=0; i<running_processes && i<20; i++) {  // Log up to 20 to avoid spam
            LOG_INFO("  Merged[%d]: PID=%u %s", i, pids_on_device[i].pid,
                    (pids_on_device[i].pid == 0) ? "(INVALID!)" : "");
        }
    } else {
        LOG_WARN("set_task_pid: No processes in merged list after filtering!");
    }
    
    // Check if our PID is in the filtered process list (after creating CUDA context)
    LOG_INFO("set_task_pid: Looking for current PID %d in %u filtered processes", current_pid, running_processes);
    for (i=0; i<running_processes; i++) {
        if (pids_on_device[i].pid == (unsigned int)current_pid) {
            hostpid = current_pid;
            LOG_INFO("set_task_pid: ✓ Found current process PID %d in filtered GPU process list (index %d)", current_pid, i);
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
            LOG_INFO("set_task_pid: Difference detection found PID %u", hostpid);
        }
    }
    
    if (hostpid==0) {
        LOG_ERROR("set_task_pid: FAILED - Current PID=%d, filtered processes=%u (previous=%u)", 
                 current_pid, running_processes, previous);
        LOG_ERROR("set_task_pid: All PIDs in merged list:");
        for (i=0; i<running_processes; i++) {
            LOG_ERROR("  [%d]=%u", i, pids_on_device[i].pid);
        }
        return NVML_ERROR_DRIVER_NOT_LOADED;
    }
    
    LOG_INFO("hostPid=%d",hostpid);
    if (set_host_pid(hostpid)==0) {
        for (i=0;i<running_processes;i++) {
            if (pids_on_device[i].pid==hostpid) {
                LOG_INFO("Primary Context Size==%lld",tmp_pids_on_device[i].usedGpuMemory);
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
    for (int i = 0; i < CUDA_DEVICE_MAX_COUNT; i++) {
        LOG_INFO("device %d -> %d",i,cuda_to_nvml_map(i));
    }
    LOG_INFO("get default cuda from %s", getenv("CUDA_VISIBLE_DEVICES"));
    return count;
}

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
