#include "allocator.h"
#include "include/log_utils.h"
#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"
#include <signal.h>
#include <unistd.h>

// Forward declarations for lock functions
extern void lock_shrreg();
extern void unlock_shrreg();
extern int enable_active_oom_killer;


size_t BITSIZE = 512;
size_t IPCSIZE = 2097152;
size_t OVERSIZE = 134217728;
//int pidfound;

region_list *r_list;
allocated_list *device_overallocated;
allocated_list *device_allocasync;

#define ALIGN       2097152
#define MULTI_PARAM 1

#define CHUNK_SIZE  (OVERSIZE/BITSIZE)
#define __CHUNK_SIZE__  CHUNK_SIZE

extern size_t initial_offset;
extern CUresult
    cuMemoryAllocate(CUdeviceptr* dptr, size_t bytesize, void* data);
extern CUresult cuMemoryFree(CUdeviceptr dptr);

pthread_once_t allocator_allocate_flag = PTHREAD_ONCE_INIT;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

size_t round_up(size_t size, size_t unit) {
    if (size & (unit-1))
        return ((size / unit) + 1 ) * unit;
    return size;
}

// Internal function that doesn't lock (caller must hold lock_shrreg)
// Uses summed NVML usage (with 9MB min + 5% overhead) to check against limit
int oom_check_nolock(const int dev, size_t addon) {
    // Root user is disabled from OOM checking - only non-root users get this treatment
    uid_t current_uid = getuid();
    if (current_uid == 0) {
        LOG_DEBUG("oom_check_nolock: Root user (UID 0) - OOM checking disabled");
        return 0;  // Always allow allocation for root
    }
    
    int count1=0;
    CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetCount,&count1);
    CUdevice d;
    if (dev==-1)
        cuCtxGetDevice(&d);
    else
        d=dev;
    uint64_t limit = get_current_device_memory_limit(d);

    if (limit == 0) {
        return 0;
    }

    // Use summed NVML usage (with 9MB min + 5% overhead) - this is the actual current usage
    // This ensures we check against the real summed values, not tracked usage
    uint64_t _usage = get_summed_device_memory_usage_from_nvml(d);
    
    // If summed usage query failed, fall back to tracked usage
    if (_usage == 0) {
        LOG_DEBUG("oom_check_nolock: get_summed_device_memory_usage_from_nvml returned 0, falling back to tracked usage");
        _usage = get_gpu_memory_usage_nolock(d);
    }

    uint64_t new_allocated = _usage + addon;
    LOG_INFO("oom_check_nolock: _usage=%llu limit=%llu addon=%lu new_allocated=%llu", 
             (unsigned long long)_usage, (unsigned long long)limit, addon, (unsigned long long)new_allocated);
    
    if (new_allocated > limit) {
        LOG_ERROR("Device %d OOM %llu / %llu (trying to allocate %lu bytes)", d, (unsigned long long)new_allocated, (unsigned long long)limit, addon);
        
        // Try to clear dead processes first
        if (clear_proc_slot_nolock(1) > 0) {
            // Recheck after clearing dead processes - use summed usage again
            _usage = get_summed_device_memory_usage_from_nvml(d);
            if (_usage == 0) {
                _usage = get_gpu_memory_usage_nolock(d);
            }
            new_allocated = _usage + addon;
            if (new_allocated <= limit) {
                LOG_INFO("After clearing dead processes, allocation now allowed: %llu / %llu", 
                         (unsigned long long)new_allocated, (unsigned long long)limit);
                return 0;  // Allocation is now possible
            }
        }
        
        // If still OOM and OOM killer is enabled, kill processes from current cgroup/UID
        if (enable_active_oom_killer) {
            LOG_ERROR("OOM detected and ACTIVE_OOM_KILLER enabled - killing processes from current cgroup/UID (tried to allocate %lu bytes, would exceed limit %llu, current usage %llu)", 
                     addon, (unsigned long long)limit, (unsigned long long)_usage);
            // Call active_oom_killer which queries NVML and filters by cgroup/UID
            // This will kill all processes from the current user/cgroup, not just self
            active_oom_killer();
            // After killing, we still return error (allocation failed)
            // The killed processes will free up memory for future allocations
        }
        
        return 1;
    }
    return 0;
}

int oom_check(const int dev, size_t addon) {
    lock_shrreg();
    int result = oom_check_nolock(dev, addon);
    unlock_shrreg();
    return result;
}

CUresult view_vgpu_allocator() {
    allocated_list_entry *al;
    size_t total;
    total=0;
    LOG_INFO("[view1]:overallocated:");
    for (al=device_overallocated->head;al!=NULL;al=al->next){
        LOG_INFO("(%p %lu)\t",(void *)al->entry->address,al->entry->length);
        total+=al->entry->length;
    }
    LOG_INFO("total=%lu",total);
    size_t t = get_current_device_memory_usage(0);
    LOG_INFO("current_device_memory_usage:%lu",t);
    return 0;
}

CUresult get_listsize(allocated_list *al, size_t *size) {
    if (al->length == 0){
        *size = 0;
        return CUDA_SUCCESS;
    }
    size_t count=0;
    allocated_list_entry *val;
    for (val=al->head;val!=NULL;val=val->next){
        count+=val->entry->length;
    }
    *size = count;
    return CUDA_SUCCESS;
}

void allocator_init() {
    LOG_DEBUG("Allocator_init\n");
    
    device_overallocated = malloc(sizeof(allocated_list));
    LIST_INIT(device_overallocated);
    device_allocasync=malloc(sizeof(allocated_list));
    LIST_INIT(device_allocasync);

    pthread_mutex_init(&mutex,NULL);
}

int add_chunk(CUdeviceptr *address, size_t size) {
    // Note: This function should be called while holding the mutex (from allocate_raw)
    // We also hold lock_shrreg() during the entire check+allocate+update to prevent
    // race conditions where multiple processes see the same available memory
    size_t addr=0;
    size_t allocsize;
    CUresult res = CUDA_SUCCESS;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    
    // Lock shared region for atomic check+allocate+update
    lock_shrreg();
    
    // Check OOM while holding lock (use nolock version to avoid deadlock)
    if (oom_check_nolock(dev,size)) {
        unlock_shrreg();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    
    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    if (size <= IPCSIZE)
        res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAlloc_v2,&e->entry->address,size);
    else{
        e->entry->length = size;
        res = cuMemoryAllocate(&e->entry->address, size, e->entry->allocHandle);
    }
    if (res!=CUDA_SUCCESS){
        LOG_ERROR("cuMemoryAllocate failed res=%d",res);
        unlock_shrreg();
        return res;
    }
    LIST_ADD(device_overallocated,e);
    //uint64_t t_size;
    *address = e->entry->address;
    allocsize = size;
    cuCtxGetDevice(&dev);
    // Update usage tracking while still holding both locks (atomic with check+allocate)
    add_gpu_device_memory_usage(getpid(), dev, allocsize, 2);
    
    // Release shared region lock
    unlock_shrreg();
    return 0;
}

int add_chunk_only(CUdeviceptr address, size_t size) {
    pthread_mutex_lock(&mutex);
    // Also lock shared region for atomic check+update (allocation already happened)
    lock_shrreg();
    
    size_t addr=0;
    size_t allocsize;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    if (oom_check(dev,size)){
        unlock_shrreg();
        pthread_mutex_unlock(&mutex);
        return -1;
    }
    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    LIST_ADD(device_overallocated,e);
    //uint64_t t_size;
    e->entry->address=address;
    allocsize = size;
    cuCtxGetDevice(&dev);
    add_gpu_device_memory_usage(getpid(), dev, allocsize, 2);
    
    unlock_shrreg();
    pthread_mutex_unlock(&mutex);
    return 0;
}

int check_memory_type(CUdeviceptr address) {
    allocated_list_entry *cursor;
    cursor = device_overallocated->head;
    for (cursor=device_overallocated->head;cursor!=NULL;cursor=cursor->next){
        if ((cursor->entry->address <= address) && (cursor->entry->address+cursor->entry->length>=address))
            return CU_MEMORYTYPE_DEVICE;
    }
    return CU_MEMORYTYPE_HOST;
}

int remove_chunk(allocated_list *a_list, CUdeviceptr dptr) {
    size_t t_size;
    if (a_list->length==0) {
        return -1;
    }
    allocated_list_entry *val;
    for (val=a_list->head;val!=NULL;val=val->next){
        if (val->entry->address == dptr) {
            t_size=val->entry->length;
            cuMemoryFree(dptr);
            LIST_REMOVE(a_list,val);
            CUdevice dev;
            cuCtxGetDevice(&dev);
            rm_gpu_device_memory_usage(getpid(), dev, t_size, 2);
            return 0;
        }
    }
    return -1;
}

int remove_chunk_only(CUdeviceptr dptr) {
    allocated_list *a_list = device_overallocated;
    size_t t_size;
    if (a_list->length == 0) {
        return -1;
    }
    allocated_list_entry *val;
    for (val = a_list->head; val != NULL; val = val->next) {
        if (val->entry->address == dptr) {
            t_size = val->entry->length;
            LIST_REMOVE(a_list, val);
            CUdevice dev;
            cuCtxGetDevice(&dev);
            rm_gpu_device_memory_usage(getpid(), dev, t_size, 2);
            return 0;
        }
    }
    return -1;
}

int allocate_raw(CUdeviceptr *dptr, size_t size) {
    int tmp;
    pthread_mutex_lock(&mutex);
    tmp = add_chunk(dptr, size);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

int free_raw(CUdeviceptr dptr) {
    pthread_mutex_lock(&mutex);
    unsigned int tmp = remove_chunk(device_overallocated, dptr);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

int remove_chunk_async(
    allocated_list *a_list, CUdeviceptr dptr, CUstream hStream) {
    size_t t_size;
    if (a_list->length == 0) {
        return -1;
    }
    allocated_list_entry *val;
    for (val = a_list->head; val != NULL; val = val->next) {
        if (val->entry->address == dptr) {
            t_size=val->entry->length;
            CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFreeAsync,dptr,hStream);
            LIST_REMOVE(a_list,val);
            a_list->limit-=t_size;
            CUdevice dev;
            cuCtxGetDevice(&dev);
            rm_gpu_device_memory_usage(getpid(),dev,t_size,2);
            return 0;
        }
    }
    return -1;
}

int free_raw_async(CUdeviceptr dptr, CUstream hStream) {
    pthread_mutex_lock(&mutex);
    unsigned int tmp = remove_chunk_async(device_allocasync, dptr, hStream);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

int add_chunk_async(CUdeviceptr *address, size_t size, CUstream hStream) {
    size_t addr=0;
    size_t allocsize;
    CUresult res = CUDA_SUCCESS;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    if (oom_check(dev,size))
        return -1;

    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAllocAsync,&e->entry->address,size,hStream);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuMemoryAllocate failed res=%d",res);
        return res;
    }
    *address = e->entry->address;
    CUmemoryPool pool;
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetMemPool,&pool,dev);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuDeviceGetMemPool failed res=%d",res);
        return res;
    }
    size_t poollimit;
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolGetAttribute,pool,CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,&poollimit);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuMemPoolGetAttribute failed res=%d",res);
        return res;
    }
    if (poollimit != 0) {
        if (poollimit> device_allocasync->limit) {
            allocsize = (poollimit-device_allocasync->limit < size)? poollimit-device_allocasync->limit : size;
            cuCtxGetDevice(&dev);
            add_gpu_device_memory_usage(getpid(), dev, allocsize, 2);
            device_allocasync->limit=device_allocasync->limit+allocsize;
            e->entry->length=allocsize;
        }else{
            e->entry->length=0;
        } 
    }
    LIST_ADD(device_allocasync,e);
    return 0;
}

int allocate_async_raw(CUdeviceptr *dptr, size_t size, CUstream hStream) {
    int tmp;
    pthread_mutex_lock(&mutex);
    tmp = add_chunk_async(dptr,size,hStream);
    pthread_mutex_unlock(&mutex);
    return tmp;
}
