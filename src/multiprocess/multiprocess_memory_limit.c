#include <sys/mman.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stddef.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

#include <assert.h>
#include <cuda.h>
#include "include/nvml_prefix.h"
// Don't include <nvml.h> - it conflicts with nvml-subset.h
// Use nvml-subset.h via libnvml_hook.h instead
#include "include/nvml-subset.h"  // For NVML types

#include "include/process_utils.h"
#include "include/memory_limit.h"
#include "include/libnvml_hook.h"  // For NVML_FIND_ENTRY and driver_sym_t
#include "multiprocess/multiprocess_memory_limit.h"

// Forward declaration - defined in config_file.c
extern int is_softmig_configured(void);


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
// is_softmig_configured is declared above

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
    // Use NVML override mechanism (if available)
    if (!nvml_symbols_available()) {
        // In standalone tools, we can't query NVML, so use a default
        LOG_WARN("NVML symbols not available, using default device count");
        region_info.shared_region->device_num = 1;  // Default to 1 device
        return 0;
    }
    driver_sym_t entry = NVML_FIND_ENTRY(nvml_library_entry, nvmlDeviceGetCount_v2);
    if (entry == NULL) {
        LOG_ERROR("nvmlDeviceGetCount_v2 not found");
        return 0;
    }
    nvmlReturn_t ret = entry(&nvmlDevicesCount);
    if (ret != NVML_SUCCESS) {
        LOG_ERROR("nvmlDeviceGetCount_v2 failed: %d", ret);
        return 0;
    }
    region_info.shared_region->device_num=nvmlDevicesCount;
    nvmlDevice_t dev;
    entry = NVML_FIND_ENTRY(nvml_library_entry, nvmlDeviceGetHandleByIndex);
    if (entry == NULL) {
        LOG_ERROR("nvmlDeviceGetHandleByIndex not found");
        return 0;
    }
    driver_sym_t uuid_entry = NVML_FIND_ENTRY(nvml_library_entry, nvmlDeviceGetUUID);
    if (uuid_entry == NULL) {
        LOG_ERROR("nvmlDeviceGetUUID not found");
        return 0;
    }
    for(i=0;i<nvmlDevicesCount;i++){
        ret = entry(i, &dev);
        if (ret != NVML_SUCCESS) {
            LOG_ERROR("nvmlDeviceGetHandleByIndex failed for device %u: %d", i, ret);
            return 0;
        }
        ret = uuid_entry(dev, region_info.shared_region->uuids[i], NVML_DEVICE_UUID_V2_BUFFER_SIZE);
        if (ret != NVML_SUCCESS) {
            LOG_ERROR("nvmlDeviceGetUUID failed for device %u: %d", i, ret);
            return 0;
        }
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
        if (fgets(tmp,10000,f) == NULL) {
            break;
        }
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
        char index_name[16];  // Increased from 8 to handle large device indices (e.g., "_2147483646")
        snprintf(index_name, sizeof(index_name), "_%d", i);
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
        char index_name[16];  // Increased from 8 to handle large device indices (e.g., "_2147483646")
        snprintf(index_name, sizeof(index_name), "_%d", i);
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
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++) {
        kill(region_info.shared_region->procs[i].pid,9);
    }
    return 0;
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

size_t get_gpu_memory_usage(const int dev) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;
    }
    int i=0;
    size_t total=0;
    lock_shrreg();
    for (i=0;i<region_info.shared_region->proc_num;i++){
        total+=region_info.shared_region->procs[i].used[dev].total;
    }
    total+=initial_offset;
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

uint64_t nvml_get_device_memory_usage(const int dev) {
    if (!nvml_symbols_available()) {
        // In standalone tools, fall back to tracked usage
        return get_gpu_memory_usage(dev);
    }
    nvmlDevice_t ndev;
    nvmlReturn_t ret;
    // Use NVML override mechanism to get device handle
    driver_sym_t entry = NVML_FIND_ENTRY(nvml_library_entry, nvmlDeviceGetHandleByIndex);
    if (entry == NULL) {
        LOG_ERROR("NVML nvmlDeviceGetHandleByIndex not found");
        return get_gpu_memory_usage(dev);
    }
    ret = entry(dev, &ndev);
    if (ret != NVML_SUCCESS) {
        // Get error string via override mechanism
        driver_sym_t err_entry = NVML_FIND_ENTRY(nvml_library_entry, nvmlErrorString);
        const char* err_str = "unknown";
        if (err_entry != NULL) {
            err_str = ((const char*(*)(nvmlReturn_t))err_entry)(ret);
        }
        LOG_ERROR("NVML get device %d error, %s", dev, err_str);
        return get_gpu_memory_usage(dev);
    }
    unsigned int pcnt = SHARED_REGION_MAX_PROCESS_NUM;
    // Use nvmlProcessInfo_t from nvml-subset.h (same as nvmlProcessInfo_v1_t)
    nvmlProcessInfo_t infos[SHARED_REGION_MAX_PROCESS_NUM];
    entry = NVML_FIND_ENTRY(nvml_library_entry, nvmlDeviceGetComputeRunningProcesses);
    if (entry == NULL) {
        LOG_ERROR("NVML nvmlDeviceGetComputeRunningProcesses not found");
        return get_gpu_memory_usage(dev);
    }
    ret = entry(ndev, &pcnt, infos);
    if (ret != NVML_SUCCESS && ret != NVML_ERROR_INSUFFICIENT_SIZE) {
        // Get error string via override mechanism
        driver_sym_t err_entry = NVML_FIND_ENTRY(nvml_library_entry, nvmlErrorString);
        const char* err_str = "unknown";
        if (err_entry != NULL) {
            err_str = ((const char*(*)(nvmlReturn_t))err_entry)(ret);
        }
        LOG_ERROR("NVML get process error, %s", err_str);
        return get_gpu_memory_usage(dev);
    }
    int i = 0;
    uint64_t usage = 0;
    shared_region_t* region = region_info.shared_region;
    lock_shrreg();
    for (; i < pcnt; i++) {
        int slot = 0;
        for (; slot < region->proc_num; slot++) {
            if (infos[i].pid != region->procs[slot].pid)
                continue;
            usage += infos[i].usedGpuMemory;
        }
    }
    unlock_shrreg();
    return usage;
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
extern void cleanup_config_file(void);

void exit_handler() {
    if (region_info.init_status == PTHREAD_ONCE_INIT) {
        return;
    }
    shared_region_t* region = region_info.shared_region;
    
    // Check if shared region was never initialized (e.g., program failed to start)
    // This can happen when bash loads the library but the program doesn't exist
    if (region == NULL) {
        // Clean up config file even if shared region wasn't initialized
        cleanup_config_file();
        return;
    }
    
    int slot = 0;
    LOG_MSG("Calling exit handler %d",getpid());
    
    // Clean up config file (delete it)
    cleanup_config_file();
    
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
    // Function body intentionally empty - reserved for future debugging output
}

void child_reinit_flag() {
    LOG_DEBUG("Detect child pid: %d -> %d", region_info.pid, getpid());   
    region_info.init_status = PTHREAD_ONCE_INIT;
}

int set_active_oom_killer() {
    char *oom_killer_env;
    oom_killer_env = getenv("ACTIVE_OOM_KILLER");
    if (oom_killer_env!=NULL){
        if (strcmp(oom_killer_env,"false") == 0)
            return 0;
        if (strcmp(oom_killer_env,"true") == 0)
            return 1;
        if (strcmp(oom_killer_env,"0")==0)
            return 0;
        if (strcmp(oom_killer_env,"1")==0)
            return 1;
    }
    return 1;
}

int set_env_utilization_switch() {
    char *utilization_env;
    utilization_env = getenv("GPU_CORE_UTILIZATION_POLICY");
    if (utilization_env!=NULL){
        if ((strcmp(utilization_env,"FORCE") ==0 ) || (strcmp(utilization_env,"force") ==0))
            return 1;
        if ((strcmp(utilization_env,"DISABLE") ==0 ) || (strcmp(utilization_env,"disable") ==0 ))
            return 2;
    }
    return 0;
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
    //put_device_info();
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
        region->priority = 1;
        if (getenv(CUDA_TASK_PRIORITY_ENV)!=NULL)
            region->priority = atoi(getenv(CUDA_TASK_PRIORITY_ENV));
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
            LOG_INFO("SET PID= %d",hostpid);
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
    LOG_INFO("dev %d new limit set to %ld",dev,newlimit);
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
//    result= nvml_get_device_memory_usage(dev);
    return result;
}

// Forward declaration - implemented in src/nvml/hook.c (may not be available in standalone tools)
// For standalone tools like shrreg-tool, provide weak stubs
extern uint64_t sum_process_memory_from_nvml(nvmlDevice_t device) __attribute__((weak));
extern entry_t nvml_library_entry[] __attribute__((weak));

// Weak stub implementations for standalone tools (will be overridden if nvml_mod is linked)
uint64_t __attribute__((weak)) sum_process_memory_from_nvml(nvmlDevice_t device) {
    (void)device;  // Unused
    return 0;  // Indicates NVML not available
}

entry_t __attribute__((weak)) nvml_library_entry[] = { {NULL, NULL} };

// Helper function to check if NVML symbols are available
static int nvml_symbols_available(void) {
    // Check if nvml_library_entry is available (not just a stub)
    // In standalone tools like shrreg-tool, this will be the stub
    return (nvml_library_entry != NULL && nvml_library_entry[0].name != NULL);
}

// Helper function to get NVML device from CUDA device ID
static nvmlDevice_t get_nvml_device_from_cuda(int cudadev) {
    if (!nvml_symbols_available()) {
        return NULL;
    }
    nvmlDevice_t nvml_device;
    unsigned int nvml_index = cuda_to_nvml_map(cudadev);
    if (nvml_index >= CUDA_DEVICE_MAX_COUNT) {
        return NULL;
    }
    // Use the NVML override mechanism to get device handle
    // We need to call the real NVML function, not our hook
    nvmlReturn_t ret;
    driver_sym_t entry = NVML_FIND_ENTRY(nvml_library_entry, nvmlDeviceGetHandleByIndex);
    if (entry == NULL) {
        return NULL;
    }
    ret = entry(nvml_index, &nvml_device);
    if (ret != NVML_SUCCESS) {
        return NULL;
    }
    return nvml_device;
}

uint64_t get_current_device_memory_usage(const int dev) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No usage tracking when softmig is disabled
    }
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
        return 0;
    }
    
    // Try NVML process summing first (more accurate, works even for unhooked allocations)
    // Only if NVML symbols are available (not in standalone tools)
    if (nvml_symbols_available()) {
        nvmlDevice_t nvml_device = get_nvml_device_from_cuda(dev);
        if (nvml_device != NULL) {
            uint64_t nvml_usage = sum_process_memory_from_nvml(nvml_device);
            if (nvml_usage > 0) {
                // Successfully got usage from NVML
                return nvml_usage;
            }
            // If NVML returned 0, it might mean no processes or query failed
            // Fall through to tracked usage as fallback
        }
    }
    
    // Fallback to old tracking system (only for processes we've hooked)
    // This is the only option in standalone tools like shrreg-tool
    uint64_t result = get_gpu_memory_usage(dev);
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
    if (env_utilization_switch==1)
        return 1;
    if (env_utilization_switch==2)
        return 0;
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // Default when softmig is disabled
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


int comparelwr(const char *s1,char *s2){
    if ((s1==NULL) || (s2==NULL))
        return 1;
    if (strlen(s1)!=strlen(s2)) {
        return 1;
    }
    int i;
    for (i=0;i<strlen(s1);i++)
        if (tolower(s1[i])!=tolower(s2[i])){
            return 1;
        }
    return 0;
}
