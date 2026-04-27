/**
 * @file dlsym_resolve.c
 * @brief Resolve the real dlsym via dlvsym / libdl.so.2 / _dl_sym fallbacks.
 *
 * Consolidates the "find real dlsym" logic that was previously duplicated in
 * libsoftmig.c and nvml/hook.c. Thread-safe via pthread_once so the probe
 * runs at most once per process.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <stddef.h>
#include <pthread.h>
#include "include/dlsym_resolve.h"

static fp_dlsym cached_real_dlsym = NULL;
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;

static void do_resolve(void) {
    fp_dlsym result = NULL;

    /* Method 1: dlvsym with RTLD_NEXT (most reliable, avoids our hook) */
    result = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    if (result == NULL) {
        result = dlvsym(RTLD_NEXT, "dlsym", "");
    }
    if (result == NULL) {
        const char *glibc_versions[] = {"GLIBC_2.34", "GLIBC_2.17", "GLIBC_2.4", NULL};
        for (int i = 0; glibc_versions[i] != NULL && result == NULL; i++) {
            result = dlvsym(RTLD_NEXT, "dlsym", glibc_versions[i]);
        }
    }

    /* Method 2: dlvsym with RTLD_DEFAULT (system default namespace) */
    if (result == NULL) {
        result = dlvsym(RTLD_DEFAULT, "dlsym", "");
        if (result == NULL) {
            result = dlvsym(RTLD_DEFAULT, "dlsym", "GLIBC_2.2.5");
        }
    }

    /* Method 3: get dlsym from libdl.so.2 directly via dlvsym */
    if (result == NULL) {
        void *libdl = dlopen("libdl.so.2", RTLD_LAZY | RTLD_LOCAL);
        if (libdl != NULL) {
            result = (fp_dlsym)dlvsym(libdl, "dlsym", "");
            if (result == NULL) {
                result = (fp_dlsym)dlvsym(libdl, "dlsym", "GLIBC_2.2.5");
            }
        }
    }

    /* Last resort: _dl_sym weak symbol (glibc internal) */
    if (result == NULL) {
#ifdef __GLIBC__
        extern void* _dl_sym(void*, const char*, void*) __attribute__((weak));
        if (_dl_sym != NULL) {
            result = (fp_dlsym)_dl_sym(RTLD_NEXT, "dlsym", (void*)dlsym);
        }
#endif
    }

    cached_real_dlsym = result;
}

fp_dlsym resolve_real_dlsym(void) {
    pthread_once(&resolve_once, do_resolve);
    return cached_real_dlsym;
}
