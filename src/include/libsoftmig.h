/**
 * @file libsoftmig.h
 * @brief Main library header — dlsym hook, initialization, and PID detection.
 *
 * Provides the dlsym interposition entry point, CUDA/NVML hook section
 * dispatchers, and the set_task_pid / map_cuda_visible_devices API used
 * during library initialization.
 */
#ifndef __LIBSOFTMIG_H__
#define __LIBSOFTMIG_H__

#include <dlfcn.h>
#include <cuda.h>
#include "include/nvml_prefix.h"
#include <nvml.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <signal.h>

#include "include/log_utils.h"
#include "static_config.h"

#define ENSURE_INITIALIZED() ensure_initialized();        \

extern void load_cuda_libraries();

#if defined(__GNUC__) && defined(__GLIBC__)

#define FUNC_ATTR_VISIBLE  __attribute__((visibility("default"))) 

// _dl_sym is an internal glibc function, use weak linking if available
#ifdef __GLIBC__
extern void* _dl_sym(void*, const char*, void*) __attribute__((weak));
#endif

#if defined(DLSYM_HOOK_DEBUG)
#define DLSYM_HOOK_FUNC(f)                                       \
    if (0 == strcmp(symbol, #f)) {                               \
        LOG_DEBUG("Detect dlsym for %s\n", #f);                  \
        return (void*) f; }                                      \

#else 

#define DLSYM_HOOK_FUNC(f)                                       \
    if (0 == strcmp(symbol, #f)) {                               \
        return (void*) f; }                                      \

#endif     

void* __dlsym_hook_section(void* handle, const char* symbol);
void* __dlsym_hook_section_nvml(void* handle, const char* symbol);

typedef void* (*fp_dlsym)(void*, const char*);

#else
#error error, neither __GLIBC__ nor __GNUC__ defined

#endif

nvmlReturn_t set_task_pid();
int map_cuda_visible_devices();

#endif  // __LIBSOFTMIG_H__