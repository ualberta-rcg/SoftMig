#include <string.h>
#include <time.h>
#include <pthread.h>
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include "include/nvml-subset.h"
#include "include/nvml_prefix.h"
#include "include/libnvml_hook.h"
#include "include/nvml_override.h"
#include "include/nvml_cache.h"
#include "include/log_utils.h"

extern entry_t nvml_library_entry[] __attribute__((weak));

static nvml_process_cache_entry_t cache[CUDA_DEVICE_MAX_COUNT];
static pthread_once_t cache_init_once = PTHREAD_ONCE_INIT;

static void cache_init(void) {
    for (int i = 0; i < CUDA_DEVICE_MAX_COUNT; i++) {
        pthread_mutex_init(&cache[i].lock, NULL);
        cache[i].valid = 0;
        cache[i].count = 0;
        cache[i].last_update.tv_sec = 0;
        cache[i].last_update.tv_nsec = 0;
    }
}

static int cache_is_fresh(const struct timespec *last_update) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    long long elapsed_ms = (now.tv_sec - last_update->tv_sec) * 1000LL
                         + (now.tv_nsec - last_update->tv_nsec) / 1000000LL;
    return elapsed_ms < NVML_CACHE_TTL_MS;
}

static int device_index_from_handle(nvmlDevice_t device) {
    nvmlReturn_t (*get_index)(nvmlDevice_t, unsigned int *) =
        NVML_FIND_ENTRY(nvml_library_entry, nvmlDeviceGetIndex);
    if (get_index == NULL) return -1;
    unsigned int idx;
    if (get_index(device, &idx) != NVML_SUCCESS) return -1;
    if (idx >= CUDA_DEVICE_MAX_COUNT) return -1;
    return (int)idx;
}

unsigned int nvml_cached_get_compute_processes(nvmlDevice_t device,
    unsigned int max_count, nvmlProcessInfo_t *infos) {
    pthread_once(&cache_init_once, cache_init);

    if (nvml_library_entry == NULL) return 0;

    int dev_idx = device_index_from_handle(device);
    if (dev_idx < 0) return 0;

    nvml_process_cache_entry_t *entry = &cache[dev_idx];
    pthread_mutex_lock(&entry->lock);

    if (entry->valid && cache_is_fresh(&entry->last_update)) {
        unsigned int n = entry->count < max_count ? entry->count : max_count;
        if (n > 0) memcpy(infos, entry->infos, n * sizeof(nvmlProcessInfo_t));
        pthread_mutex_unlock(&entry->lock);
        return n;
    }

    unsigned int process_count = SHARED_REGION_MAX_PROCESS_NUM;
    nvmlReturn_t ret = NVML_OVERRIDE_CALL_NO_LOG(nvml_library_entry,
        nvmlDeviceGetComputeRunningProcesses_v2,
        device, &process_count, entry->infos);

    if (ret != NVML_SUCCESS && ret != NVML_ERROR_INSUFFICIENT_SIZE) {
        entry->valid = 0;
        pthread_mutex_unlock(&entry->lock);
        return 0;
    }

    if (process_count > SHARED_REGION_MAX_PROCESS_NUM)
        process_count = SHARED_REGION_MAX_PROCESS_NUM;

    entry->count = process_count;
    clock_gettime(CLOCK_MONOTONIC, &entry->last_update);
    entry->valid = 1;

    unsigned int n = process_count < max_count ? process_count : max_count;
    if (n > 0) memcpy(infos, entry->infos, n * sizeof(nvmlProcessInfo_t));

    pthread_mutex_unlock(&entry->lock);
    return n;
}

void nvml_cache_invalidate(int device_idx) {
    pthread_once(&cache_init_once, cache_init);

    if (device_idx < 0 || device_idx >= CUDA_DEVICE_MAX_COUNT) return;

    nvml_process_cache_entry_t *entry = &cache[device_idx];
    pthread_mutex_lock(&entry->lock);
    entry->valid = 0;
    pthread_mutex_unlock(&entry->lock);
}
