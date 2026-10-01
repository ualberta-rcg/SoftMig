/**
 * @file allocator.c
 * @brief GPU memory chunk tracker with OOM checking before every allocation.
 *
 * Maintains a linked list of allocated GPU memory chunks. Before each
 * allocation (sync or async), checks NVML-summed usage against the per-device
 * limit and triggers the OOM killer if the limit would be exceeded.
 * All public functions are mutex-protected for thread safety.
 */
#include "allocator.h"
#include "include/log_utils.h"
#include "include/libcuda_hook.h"
#include "include/nvml_cache.h"
#include "multiprocess/multiprocess_memory_limit.h"
#include <signal.h>
#include <unistd.h>

// Forward declarations for lock functions
extern void lock_shrreg();
extern void unlock_shrreg();
extern int enable_active_oom_killer;


size_t IPCSIZE = 2097152;

allocated_list *device_overallocated;
allocated_list *device_allocasync;

#define ALIGN       2097152
#define MULTI_PARAM 1

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
// Uses summed NVML usage (raw per-process values, cgroup/UID-filtered) to check against limit
int oom_check_nolock(const int dev, size_t addon) {
    // Root user is disabled from OOM checking - only non-root users get this treatment
    uid_t current_uid = getuid();
    if (current_uid == 0) {
        LOG_DEBUG("oom_check_nolock: Root user (UID 0) - OOM checking disabled");
        return 0;  // Always allow allocation for root
    }
    
    CUdevice d;
    if (dev==-1)
        cuCtxGetDevice(&d);
    else
        d=dev;
    uint64_t limit = get_current_device_memory_limit(d);

    if (limit == 0) {
        return 0;
    }

    // Use the maximum of tracked usage and NVML-summed usage.
    // Tracked usage is updated immediately on every allocation, but may miss
    // allocations made before SoftMig loaded. NVML usage lags but catches
    // everything eventually. Taking the max ensures we don't miss either case.
    LOG_DEBUG("oom_check_nolock: Starting OOM check for device %d - current PID %d, current UID %u, limit=%llu, addon=%lu", 
             d, getpid(), getuid(), (unsigned long long)limit, addon);
    
    uint64_t tracked_usage = get_gpu_memory_usage_nolock(d);
    uint64_t nvml_usage = get_summed_device_memory_usage_from_nvml(d);
    uint64_t _usage = (tracked_usage > nvml_usage) ? tracked_usage : nvml_usage;
    
    LOG_DEBUG("oom_check_nolock: tracked=%llu nvml=%llu using=%llu",
             (unsigned long long)tracked_usage, (unsigned long long)nvml_usage, (unsigned long long)_usage);

    uint64_t new_allocated = _usage + addon;
    LOG_DEBUG("oom_check_nolock: Device %d - _usage=%llu limit=%llu addon=%lu new_allocated=%llu (current PID %d, current UID %u)", 
             d, (unsigned long long)_usage, (unsigned long long)limit, addon, (unsigned long long)new_allocated, getpid(), getuid());
    
    if (new_allocated > limit) {
        LOG_ERROR("Device %d OOM %llu / %llu (trying to allocate %lu bytes)", d, (unsigned long long)new_allocated, (unsigned long long)limit, addon);
        
        // Try to clear dead processes first
        if (clear_proc_slot_nolock(1) > 0) {
            // Recheck after clearing dead processes
            tracked_usage = get_gpu_memory_usage_nolock(d);
            nvml_usage = get_summed_device_memory_usage_from_nvml(d);
            _usage = (tracked_usage > nvml_usage) ? tracked_usage : nvml_usage;
            new_allocated = _usage + addon;
            if (new_allocated <= limit) {
                LOG_DEBUG("After clearing dead processes, allocation now allowed: %llu / %llu", 
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
    lock_shrreg();
    
    size_t addr=0;
    size_t allocsize;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    if (oom_check_nolock(dev,size)){
        unlock_shrreg();
        pthread_mutex_unlock(&mutex);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    LIST_ADD(device_overallocated,e);
    e->entry->address=address;
    allocsize = size;
    cuCtxGetDevice(&dev);
    add_gpu_device_memory_usage(getpid(), dev, allocsize, 2);
    
    unlock_shrreg();
    nvml_cache_invalidate((int)dev);
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
        LOG_DEBUG("remove_chunk: list empty, forwarding free of untracked %llx to real cuMemFree_v2",dptr);
        return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFree_v2,dptr);
    }
    allocated_list_entry *val;
    for (val=a_list->head;val!=NULL;val=val->next){
        if (val->entry->address == dptr) {
            t_size=val->entry->length;
            CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFree_v2,dptr);
            LIST_REMOVE(a_list,val);
            CUdevice dev;
            cuCtxGetDevice(&dev);
            rm_gpu_device_memory_usage(getpid(), dev, t_size, 2);
            return res;
        }
    }
    LOG_DEBUG("remove_chunk: %llx not tracked, forwarding to real cuMemFree_v2",dptr);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFree_v2,dptr);
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
    int tmp = remove_chunk(device_overallocated, dptr);
    if (tmp == CUDA_SUCCESS) {
        CUdevice dev;
        if (cuCtxGetDevice(&dev) == CUDA_SUCCESS)
            nvml_cache_invalidate((int)dev);
    }
    pthread_mutex_unlock(&mutex);
    return tmp;
}

int remove_chunk_async(
    allocated_list *a_list, CUdeviceptr dptr, CUstream hStream) {
    size_t t_size;
    if (a_list->length == 0) {
        LOG_DEBUG("remove_chunk_async: list empty, forwarding free of untracked %llx to real cuMemFreeAsync",dptr);
        return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFreeAsync,dptr,hStream);
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
    LOG_DEBUG("remove_chunk_async: %llx not tracked, forwarding to real cuMemFreeAsync",dptr);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFreeAsync,dptr,hStream);
}

int free_raw_async(CUdeviceptr dptr, CUstream hStream) {
    pthread_mutex_lock(&mutex);
    int tmp = remove_chunk_async(device_allocasync, dptr, hStream);
    if (tmp == CUDA_SUCCESS) {
        CUdevice dev;
        if (cuCtxGetDevice(&dev) == CUDA_SUCCESS)
            nvml_cache_invalidate((int)dev);
    }
    pthread_mutex_unlock(&mutex);
    return tmp;
}

int add_chunk_async(CUdeviceptr *address, size_t size, CUstream hStream) {
    size_t addr=0;
    size_t allocsize;
    CUresult res = CUDA_SUCCESS;
    CUdevice dev;
    cuCtxGetDevice(&dev);

    lock_shrreg();
    if (oom_check_nolock(dev,size)) {
        unlock_shrreg();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAllocAsync,&e->entry->address,size,hStream);
    if (res != CUDA_SUCCESS) {
        unlock_shrreg();
        LOG_ERROR("cuMemoryAllocate failed res=%d",res);
        return res;
    }
    *address = e->entry->address;
    CUmemoryPool pool;
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetMemPool,&pool,dev);
    size_t poollimit = 0;
    if (res == CUDA_SUCCESS) {
        res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolGetAttribute,pool,CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,&poollimit);
    }
    if (res != CUDA_SUCCESS) {
        // The real allocation already succeeded; pool attribute reads are
        // best-effort bookkeeping. Track the requested size instead of
        // abandoning a live allocation.
        LOG_DEBUG("pool attribute read failed res=%d (non-fatal), tracking requested size %lu",res,size);
        e->entry->length = size;
        cuCtxGetDevice(&dev);
        add_gpu_device_memory_usage(getpid(), dev, size, 2);
        device_allocasync->limit += size;
    } else if (poollimit != 0) {
        if (poollimit> device_allocasync->limit) {
            allocsize = (poollimit-device_allocasync->limit < size)? poollimit-device_allocasync->limit : size;
            cuCtxGetDevice(&dev);
            add_gpu_device_memory_usage(getpid(), dev, allocsize, 2);
            device_allocasync->limit=device_allocasync->limit+allocsize;
            e->entry->length=allocsize;
        }else{
            e->entry->length=0;
        }
    } else {
        // RESERVED_MEM_HIGH == 0: no slab accounting possible. Track the
        // requested size so the free path stays balanced.
        e->entry->length = size;
        cuCtxGetDevice(&dev);
        add_gpu_device_memory_usage(getpid(), dev, size, 2);
        device_allocasync->limit += size;
    }
    unlock_shrreg();
    LIST_ADD(device_allocasync,e);
    nvml_cache_invalidate((int)dev);
    return 0;
}

int allocate_async_raw(CUdeviceptr *dptr, size_t size, CUstream hStream) {
    int tmp;
    pthread_mutex_lock(&mutex);
    tmp = add_chunk_async(dptr,size,hStream);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

// Track a chunk allocated outside the cuMemAllocAsync path (e.g.
// cuMemAllocFromPoolAsync) in the async list. The caller has already run
// oom_check and the real allocation; this only records bookkeeping, so the
// free path (remove_chunk_async) stays balanced with the driver state.
int add_chunk_async_only(CUdeviceptr address, size_t size) {
    pthread_mutex_lock(&mutex);
    lock_shrreg();

    size_t addr=0;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    LIST_ADD(device_allocasync,e);
    e->entry->address = address;
    add_gpu_device_memory_usage(getpid(), dev, size, 2);
    device_allocasync->limit += size;

    unlock_shrreg();
    nvml_cache_invalidate((int)dev);
    pthread_mutex_unlock(&mutex);
    return 0;
}
