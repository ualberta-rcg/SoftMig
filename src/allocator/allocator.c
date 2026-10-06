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
    softmig_mem_allocate(CUdeviceptr* dptr, size_t bytesize, void* data);
extern CUresult softmig_mem_free(CUdeviceptr dptr);

pthread_once_t allocator_allocate_flag = PTHREAD_ONCE_INIT;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

size_t round_up(size_t size, size_t unit) {
    if (size & (unit-1))
        return ((size / unit) + 1 ) * unit;
    return size;
}

// Caller holds lock_shrreg. nvml_usage is the job's summed NVML usage on
// dev, fetched by the caller (it is a cached read and must not be done
// under the region lock: the uncached path walks /proc for every PID).
static int oom_check_usage_nolock(const int dev, size_t addon, uint64_t nvml_usage) {
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
    
    // Tracked usage lives at the NVML index; pending is admitted-but-in-flight
    // (see softmig_reserve), visible to neither tracked nor NVML usage yet.
    int nd = (int)cuda_to_nvml_map(d);
    uint64_t tracked_usage = get_gpu_memory_usage_nolock(nd);
    uint64_t pending = get_pending_memory_nolock(nd);
    uint64_t _usage = ((tracked_usage > nvml_usage) ? tracked_usage : nvml_usage) + pending;
    
    LOG_DEBUG("oom_check_nolock: tracked=%llu nvml=%llu pending=%llu using=%llu",
             (unsigned long long)tracked_usage, (unsigned long long)nvml_usage,
             (unsigned long long)pending, (unsigned long long)_usage);

    uint64_t new_allocated = _usage + addon;
    LOG_DEBUG("oom_check_nolock: Device %d - _usage=%llu limit=%llu addon=%lu new_allocated=%llu (current PID %d, current UID %u)", 
             d, (unsigned long long)_usage, (unsigned long long)limit, addon, (unsigned long long)new_allocated, getpid(), getuid());
    
    if (new_allocated > limit) {
        LOG_ERROR("Device %d OOM %llu / %llu (trying to allocate %lu bytes)", d, (unsigned long long)new_allocated, (unsigned long long)limit, addon);
        
        // Try to clear dead processes first
        if (clear_proc_slot_nolock(1) > 0) {
            // Recheck after clearing dead processes
            tracked_usage = get_gpu_memory_usage_nolock(nd);
            nvml_usage = get_summed_device_memory_usage_from_nvml(d);
            pending = get_pending_memory_nolock(nd);
            _usage = ((tracked_usage > nvml_usage) ? tracked_usage : nvml_usage) + pending;
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

// Internal function that doesn't lock (caller must hold lock_shrreg)
int oom_check_nolock(const int dev, size_t addon) {
    CUdevice d = dev;
    if (dev == -1) cuCtxGetDevice(&d);
    return oom_check_usage_nolock(d, addon, get_summed_device_memory_usage_from_nvml(d));
}

int oom_check(const int dev, size_t addon) {
    CUdevice d = dev;
    if (dev == -1) cuCtxGetDevice(&d);
    uint64_t nvml_usage = get_summed_device_memory_usage_from_nvml(d);
    lock_shrreg();
    int result = oom_check_usage_nolock(d, addon, nvml_usage);
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

/*
 * Admission control without holding the region lock across the driver call:
 *   softmig_reserve()   - under the lock: limit check, then record the bytes
 *                         as pending in this process's slot;
 *   <driver allocation> - no lock held, so processes allocate concurrently;
 *   commit              - track the chunk and drop the pending bytes
 *                         (add_chunk_only / add_chunk_async_only), or
 *   softmig_unreserve() - drop the pending bytes if the driver call failed.
 * oom_check_nolock counts pending bytes, so concurrent admissions can never
 * overshoot the limit; pending bytes of a process that dies go with its slot.
 */
int softmig_reserve(int dev, size_t size) {
    uint64_t nvml_usage = get_summed_device_memory_usage_from_nvml(dev);
    lock_shrreg();
    if (oom_check_usage_nolock(dev, size, nvml_usage)) {
        unlock_shrreg();
        return 1;
    }
    adjust_pending_memory_nolock(dev, (int64_t)size);
    unlock_shrreg();
    return 0;
}

void softmig_unreserve(int dev, size_t size) {
    lock_shrreg();
    adjust_pending_memory_nolock(dev, -(int64_t)size);
    unlock_shrreg();
}

int add_chunk(CUdeviceptr *address, size_t size) {
    CUdevice dev;
    cuCtxGetDevice(&dev);
    if (softmig_reserve(dev, size)) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    allocated_list_entry *e;
    size_t addr = 0;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    CUresult res;
    if (size <= IPCSIZE) {
        res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAlloc_v2,&e->entry->address,size);
    } else {
        e->entry->length = size;
        res = softmig_mem_allocate(&e->entry->address, size, e->entry->allocHandle);
    }
    if (res != CUDA_SUCCESS) {
        softmig_unreserve(dev, size);
        free(e->entry->allocHandle);
        free(e->entry);
        free(e);
        return res;
    }
    *address = e->entry->address;
    pthread_mutex_lock(&mutex);
    LIST_ADD(device_overallocated,e);
    pthread_mutex_unlock(&mutex);
    lock_shrreg();
    adjust_pending_memory_nolock(dev, -(int64_t)size);
    add_gpu_device_memory_usage(getpid(), dev, size, 2);
    unlock_shrreg();
    return 0;
}

// Commit a chunk whose driver allocation succeeded after softmig_reserve().
int add_chunk_only(CUdeviceptr address, size_t size) {
    CUdevice dev;
    cuCtxGetDevice(&dev);
    allocated_list_entry *e;
    size_t addr = 0;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    e->entry->address = address;
    pthread_mutex_lock(&mutex);
    LIST_ADD(device_overallocated,e);
    pthread_mutex_unlock(&mutex);
    lock_shrreg();
    adjust_pending_memory_nolock(dev, -(int64_t)size);
    add_gpu_device_memory_usage(getpid(), dev, size, 2);
    unlock_shrreg();
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

static allocated_list_entry *find_chunk(allocated_list *a_list, CUdeviceptr dptr) {
    allocated_list_entry *val;
    for (val = a_list->head; val != NULL; val = val->next) {
        if (val->entry->address == dptr) {
            return val;
        }
    }
    return NULL;
}

// Drop bookkeeping for a chunk the driver has already freed.
static void untrack_chunk(allocated_list *a_list, allocated_list_entry *val) {
    size_t t_size = val->entry->length;
    LIST_REMOVE(a_list, val);
    if (a_list == device_allocasync) {
        a_list->limit -= t_size;
    }
    CUdevice dev;
    cuCtxGetDevice(&dev);
    rm_gpu_device_memory_usage(getpid(), dev, t_size, 2);
}

// CUDA lets cuMemFree release a cuMemAllocAsync pointer and vice versa, so a
// pointer missing from the expected list is looked up in the other one.
static allocated_list_entry *find_chunk_any(allocated_list *first, CUdeviceptr dptr,
                                            allocated_list **owner) {
    allocated_list *other = (first == device_allocasync) ? device_overallocated : device_allocasync;
    allocated_list_entry *val = find_chunk(first, dptr);
    *owner = first;
    if (val == NULL) {
        val = find_chunk(other, dptr);
        *owner = other;
    }
    return val;
}

int remove_chunk(allocated_list *a_list, CUdeviceptr dptr) {
    allocated_list *owner;
    allocated_list_entry *val = find_chunk_any(a_list, dptr, &owner);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFree_v2,dptr);
    if (val == NULL) {
        LOG_DEBUG("remove_chunk: %llx not tracked, forwarded to real cuMemFree_v2",dptr);
    } else if (res == CUDA_SUCCESS) {
        untrack_chunk(owner, val);
    }
    return res;
}

int remove_chunk_only(CUdeviceptr dptr) {
    allocated_list *a_list = device_overallocated;
    pthread_mutex_lock(&mutex);
    allocated_list_entry *val = find_chunk(a_list, dptr);
    if (val == NULL) {
        pthread_mutex_unlock(&mutex);
        return -1;
    }
    size_t t_size = val->entry->length;
    LIST_REMOVE(a_list, val);
    pthread_mutex_unlock(&mutex);
    CUdevice dev;
    cuCtxGetDevice(&dev);
    rm_gpu_device_memory_usage(getpid(), dev, t_size, 2);
    return 0;
}

int allocate_raw(CUdeviceptr *dptr, size_t size) {
    return add_chunk(dptr, size);
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
    allocated_list *owner;
    allocated_list_entry *val = find_chunk_any(a_list, dptr, &owner);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFreeAsync,dptr,hStream);
    if (val == NULL) {
        LOG_DEBUG("remove_chunk_async: %llx not tracked, forwarded to real cuMemFreeAsync",dptr);
    } else if (res == CUDA_SUCCESS) {
        untrack_chunk(owner, val);
    }
    return res;
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
    CUdevice dev;
    cuCtxGetDevice(&dev);
    if (softmig_reserve(dev, size)) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    allocated_list_entry *e;
    size_t addr = 0;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAllocAsync,&e->entry->address,size,hStream);
    if (res != CUDA_SUCCESS) {
        softmig_unreserve(dev, size);
        free(e->entry->allocHandle);
        free(e->entry);
        free(e);
        return res;
    }
    *address = e->entry->address;

    // Pool attribute reads are best-effort bookkeeping and stay outside the
    // region lock. The default pool may already hold the memory (reuse), in
    // which case only the growth of RESERVED_MEM_HIGH is new usage.
    CUmemoryPool pool;
    size_t poollimit = 0;
    CUresult pres = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetMemPool,&pool,dev);
    if (pres == CUDA_SUCCESS) {
        pres = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolGetAttribute,pool,CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,&poollimit);
    }

    pthread_mutex_lock(&mutex);
    size_t allocsize;
    if (pres != CUDA_SUCCESS || poollimit == 0) {
        // No slab accounting possible: track the requested size so the free
        // path stays balanced.
        LOG_DEBUG("pool attribute unavailable (res=%d, high=%lu), tracking requested size %lu", pres, poollimit, size);
        allocsize = size;
    } else if (poollimit > device_allocasync->limit) {
        allocsize = (poollimit - device_allocasync->limit < size) ? poollimit - device_allocasync->limit : size;
    } else {
        allocsize = 0;
    }
    e->entry->length = allocsize;
    device_allocasync->limit += allocsize;
    LIST_ADD(device_allocasync,e);
    pthread_mutex_unlock(&mutex);

    lock_shrreg();
    adjust_pending_memory_nolock(dev, -(int64_t)size);
    if (allocsize) {
        add_gpu_device_memory_usage(getpid(), dev, allocsize, 2);
    }
    unlock_shrreg();
    return 0;
}

int allocate_async_raw(CUdeviceptr *dptr, size_t size, CUstream hStream) {
    return add_chunk_async(dptr, size, hStream);
}

// Commit a chunk allocated outside the cuMemAllocAsync path (e.g.
// cuMemAllocFromPoolAsync) after softmig_reserve() and the real allocation,
// so the free path (remove_chunk_async) stays balanced with the driver state.
int add_chunk_async_only(CUdeviceptr address, size_t size) {
    CUdevice dev;
    cuCtxGetDevice(&dev);
    allocated_list_entry *e;
    size_t addr = 0;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    e->entry->address = address;
    pthread_mutex_lock(&mutex);
    LIST_ADD(device_allocasync,e);
    device_allocasync->limit += size;
    pthread_mutex_unlock(&mutex);
    lock_shrreg();
    adjust_pending_memory_nolock(dev, -(int64_t)size);
    add_gpu_device_memory_usage(getpid(), dev, size, 2);
    unlock_shrreg();
    return 0;
}
