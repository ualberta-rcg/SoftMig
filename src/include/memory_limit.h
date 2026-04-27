/**
 * @file memory_limit.h
 * @brief GPU memory and SM limit environment variable definitions and enforcement macros.
 *
 * Reads per-device limits from CUDA_DEVICE_MEMORY_LIMIT and CUDA_DEVICE_SM_LIMIT,
 * and provides macros to ensure the shared region is initialized before any memory op.
 */
#ifndef __MEMORY_LIMIT_H__
#define __MEMORY_LIMIT_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

#include "static_config.h"


#define CUDA_DEVICE_MEMORY_LIMIT "CUDA_DEVICE_MEMORY_LIMIT"
#define CUDA_DEVICE_MEMORY_LIMIT_KEY_LENGTH 32
#define CUDA_DEVICE_SM_LIMIT "CUDA_DEVICE_SM_LIMIT"
#define CUDA_DEVICE_SM_LIMIT_KEY_LENGTH 32

#define ENSURE_INITIALIZED() ensure_initialized();        \

extern int wait_status_self(int status);

#define ENSURE_RUNNING() {                                \
   /* LOG_DEBUG("Memory op at %d",__LINE__); */              \
    ensure_initialized();                                 \
    while(!wait_status_self(1)) { LOG_DEBUG("E1"); sleep(1); }             \
}                                                         \

#include "multiprocess/multiprocess_memory_limit.h"

#endif  // __MEMORY_LIMIT_H__
