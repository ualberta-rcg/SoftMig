#ifndef __NVML_CACHE_H__
#define __NVML_CACHE_H__

#include <pthread.h>
#include <time.h>
#include "include/nvml-subset.h"
#include "include/libnvml_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

#define NVML_CACHE_TTL_MS 1000

typedef struct {
    pthread_mutex_t lock;
    struct timespec last_update;
    unsigned int count;
    nvmlProcessInfo_t infos[SHARED_REGION_MAX_PROCESS_NUM];
    int valid;
} nvml_process_cache_entry_t;

unsigned int nvml_cached_get_compute_processes(nvmlDevice_t device,
    unsigned int max_count, nvmlProcessInfo_t *infos);

void nvml_cache_invalidate(int device_idx);

#endif
