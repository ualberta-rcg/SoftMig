/**
 * @file dlsym_resolve.h
 * @brief Shared helper to resolve the real dlsym via multiple fallback methods.
 *
 * Both libsoftmig.c (dlsym interposer) and nvml/hook.c (NVML loader) need
 * the real dlsym to populate their dispatch tables. This header exposes a
 * single thread-safe resolver so the logic lives in one place.
 */
#ifndef __DLSYM_RESOLVE_H__
#define __DLSYM_RESOLVE_H__

typedef void* (*fp_dlsym)(void*, const char*);

/**
 * Resolve the real dlsym using multiple fallback methods.
 * Thread-safe via pthread_once. Returns NULL on total failure.
 */
fp_dlsym resolve_real_dlsym(void);

#endif
