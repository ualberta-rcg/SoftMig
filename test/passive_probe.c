/*
 * passive_probe — checks which code a process actually runs through when
 * libsoftmig.so is preloaded (via /etc/ld.so.preload).
 *
 *   passive_probe --expect passive
 *       Job without a SoftMig config. Every pointer handed out by dlsym and
 *       cuGetProcAddress (all flag variants) must live in libcuda, not in
 *       libsoftmig; direct-linked calls must return the driver's values; no
 *       shared region file and no SIGUSR1/2 handlers may appear.
 *
 *   passive_probe --expect enabled
 *       Sliced job. Enforcement entry points (allocation, async free, launch,
 *       per-thread-stream variants) must resolve to libsoftmig hooks, totals
 *       must be capped, and SIGUSR1/2 must still be left alone by default.
 *
 * Prints one "PASS:"/"FAIL:" line per check and exits non-zero on any FAIL.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <glob.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *(*dlsym_fn)(void *, const char *);
typedef CUresult (*gpa_v2_fn)(const char *, void **, int, cuuint64_t, CUdriverProcAddressQueryResult *);
typedef CUresult (*meminfo_fn)(size_t *, size_t *);

static int failures = 0;

static void check(int ok, const char *what) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

static int in_softmig(void *p) {
    Dl_info info;
    if (p == NULL || dladdr(p, &info) == 0 || info.dli_fname == NULL) return 0;
    return strstr(info.dli_fname, "libsoftmig") != NULL;
}

static dlsym_fn real_dlsym(void) {
    void *libc = dlopen("libc.so.6", RTLD_NOW | RTLD_NOLOAD);
    dlsym_fn f = libc ? (dlsym_fn)dlvsym(libc, "dlsym", "GLIBC_2.34") : NULL;
    if (f == NULL && libc) f = (dlsym_fn)dlvsym(libc, "dlsym", "GLIBC_2.2.5");
    return f;
}

static int handler_is_default(int sig) {
    struct sigaction sa;
    if (sigaction(sig, NULL, &sa) != 0) return 0;
    return sa.sa_handler == SIG_DFL;
}

static int shared_region_present(void) {
    glob_t g;
    int n = 0;
    const char *jid = getenv("SLURM_JOB_ID");
    char pat[256];
    snprintf(pat, sizeof pat, "/tmp/cudevshr.cache.%s*", jid ? jid : "uid*");
    if (glob(pat, 0, NULL, &g) == 0) {
        n = (int)g.gl_pathc;
        globfree(&g);
    }
    return n > 0;
}

static const char *enforced[] = {
    "cuMemAlloc_v2", "cuMemFree_v2", "cuMemAllocAsync", "cuMemFreeAsync",
    "cuMemAllocFromPoolAsync", "cuMemCreate", "cuLaunchKernel", "cuLaunchKernelEx",
    "cuMemGetInfo_v2", "cuDeviceTotalMem_v2", NULL};
static const char *gpa_names[] = {
    "cuMemAlloc", "cuMemFree", "cuMemAllocAsync", "cuMemFreeAsync",
    "cuMemAllocFromPoolAsync", "cuLaunchKernel", "cuLaunchKernelEx",
    "cuGraphLaunch", "cuMemGetInfo", "cuDeviceTotalMem", "cuMemcpyAsync", NULL};

int main(int argc, char **argv) {
    int passive = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--expect") == 0 && i + 1 < argc) {
            passive = strcmp(argv[++i], "passive") == 0;
        }
    }
    printf("mode=%s\n", passive ? "passive" : "enabled");

    dlsym_fn rdlsym = real_dlsym();
    check(rdlsym != NULL, "resolved glibc dlsym for reference lookups");
    if (rdlsym == NULL) return 1;

    void *cu = dlopen("libcuda.so.1", RTLD_NOW);
    check(cu != NULL, "dlopen libcuda.so.1");
    if (cu == NULL) return 1;

    /* 1. dlsym on the libcuda handle (cudart's pre-12 path, ctypes, JAX). */
    int dl_ok = 1;
    for (int i = 0; enforced[i]; i++) {
        void *hooked = dlsym(cu, enforced[i]);
        void *real = rdlsym(cu, enforced[i]);
        int ok = passive ? (hooked == real && !in_softmig(hooked)) : in_softmig(hooked);
        if (!ok) {
            printf("  dlsym %s -> %p (real %p, softmig=%d)\n", enforced[i], hooked, real, in_softmig(hooked));
            dl_ok = 0;
        }
    }
    check(dl_ok, passive ? "dlsym returns raw driver functions" : "dlsym returns SoftMig hooks for enforcement entry points");

    /* 2. cuGetProcAddress_v2 (cudart 12+ path), all default-stream flags. */
    gpa_v2_fn gpa = (gpa_v2_fn)dlsym(cu, "cuGetProcAddress_v2");
    gpa_v2_fn rgpa = (gpa_v2_fn)rdlsym(cu, "cuGetProcAddress_v2");
    check(gpa != NULL && rgpa != NULL, "resolved cuGetProcAddress_v2");
    CUresult r = ((CUresult(*)(unsigned))rdlsym(cu, "cuInit"))(0);
    check(r == CUDA_SUCCESS, "cuInit (raw)");

    if (gpa && rgpa) {
        int gpa_ok = 1;
        cuuint64_t flags[] = {CU_GET_PROC_ADDRESS_DEFAULT, CU_GET_PROC_ADDRESS_LEGACY_STREAM,
                              CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM};
        for (int i = 0; gpa_names[i]; i++) {
            for (int f = 0; f < 3; f++) {
                void *p = NULL, *rp = NULL;
                CUdriverProcAddressQueryResult st = 0, rst = 0;
                CUresult a = gpa(gpa_names[i], &p, 12000, flags[f], &st);
                CUresult b = rgpa(gpa_names[i], &rp, 12000, flags[f], &rst);
                int ok;
                if (passive) {
                    ok = (a == b && p == rp && st == rst && !in_softmig(p));
                } else if (strcmp(gpa_names[i], "cuMemcpyAsync") == 0) {
                    ok = (a == b && st == rst);
                } else {
                    ok = (a == b && st == rst && (rp == NULL || in_softmig(p)));
                }
                if (!ok) {
                    printf("  gpa %s flags=%llu -> %p st=%d (real %p st=%d) softmig=%d\n", gpa_names[i],
                           (unsigned long long)flags[f], p, (int)st, rp, (int)rst, in_softmig(p));
                    gpa_ok = 0;
                }
            }
        }
        check(gpa_ok, passive ? "cuGetProcAddress returns raw driver functions for every flag"
                              : "cuGetProcAddress maps enforcement entry points (incl. _ptsz) to hooks");
    }

    /* 3. Direct-linked calls (binary links -lcuda; ld.so binds to our export). */
    CUdevice dev;
    CUcontext ctx;
    check(cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&dev, 0) == CUDA_SUCCESS &&
              cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS && cuCtxSetCurrent(ctx) == CUDA_SUCCESS,
          "direct-linked cuInit/context");
    size_t tot_direct = 0, tot_real = 0, fr = 0, frr = 0;
    meminfo_fn rmeminfo = (meminfo_fn)rdlsym(cu, "cuMemGetInfo_v2");
    CUresult mi = cuMemGetInfo(&fr, &tot_direct);
    rmeminfo(&frr, &tot_real);
    size_t dev_total = 0;
    CUresult dt = cuDeviceTotalMem(&dev_total, dev);
    printf("  cuMemGetInfo total=%zu real=%zu cuDeviceTotalMem=%zu\n", tot_direct, tot_real, dev_total);
    if (passive) {
        check(mi == CUDA_SUCCESS && tot_direct == tot_real, "direct cuMemGetInfo reports the real total");
        check(dt == CUDA_SUCCESS && dev_total == tot_real, "direct cuDeviceTotalMem reports the real total");
    } else {
        check(mi == CUDA_SUCCESS && tot_direct < tot_real, "direct cuMemGetInfo is capped to the slice");
        check(dt == CUDA_SUCCESS && dev_total == tot_direct, "cuDeviceTotalMem matches the slice limit");
    }

    /* 4. Async alloc/free through the per-thread-stream pointers. */
    if (gpa) {
        CUresult (*alloc_async)(CUdeviceptr *, size_t, CUstream) = NULL;
        CUresult (*free_async)(CUdeviceptr, CUstream) = NULL;
        gpa("cuMemAllocAsync", (void **)&alloc_async, 12000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, NULL);
        gpa("cuMemFreeAsync", (void **)&free_async, 12000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, NULL);
        int ok = alloc_async && free_async;
        for (int i = 0; ok && i < 64; i++) {
            CUdeviceptr p = 0;
            ok = alloc_async(&p, 64ull << 20, NULL) == CUDA_SUCCESS && free_async(p, NULL) == CUDA_SUCCESS;
        }
        ok = ok && cuStreamSynchronize(CU_STREAM_PER_THREAD) == CUDA_SUCCESS;
        check(ok, "64x cuMemAllocAsync/cuMemFreeAsync on the per-thread stream");
        size_t f2 = 0, t2 = 0;
        cuMemGetInfo(&f2, &t2);
        printf("  free before=%zu after=%zu\n", fr, f2);
        check(f2 + (256ull << 20) >= fr, "async loop does not leak tracked/real memory");
    }

    /* 5. Process-level side effects. */
    check(handler_is_default(SIGUSR1) && handler_is_default(SIGUSR2), "SIGUSR1/SIGUSR2 left at SIG_DFL");
    if (passive) {
        check(!shared_region_present(), "no shared region file in /tmp");
    } else {
        check(shared_region_present(), "shared region file exists in /tmp");
    }

    printf("RESULT: %s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
