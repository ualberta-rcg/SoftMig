/*
 * multigpu_probe - per-device memory reporting with more than one GPU.
 *
 * Usage: multigpu_probe --expect passive|enabled [alloc_mb]
 *
 * For every visible device: compares direct-linked cuMemGetInfo /
 * cuDeviceTotalMem with the driver's own functions (real dlsym on libcuda).
 *   passive: values must match the driver exactly (free within 64 MiB).
 *   enabled: total must be capped at or below the driver total, and an
 *            allocation of alloc_mb on device N must reduce free on device N
 *            only (catches CUDA/NVML index mix-ups).
 * Prints PASS:/FAIL: lines and "RESULT: PASS|FAIL (n failures)".
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *(*dlsym_fn)(void *, const char *);
typedef CUresult (*meminfo_fn)(size_t *, size_t *);
typedef CUresult (*totalmem_fn)(size_t *, CUdevice);

static int failures = 0;

static void check(int ok, const char *fmt, int a, double x, double y) {
    printf("%s: ", ok ? "PASS" : "FAIL");
    printf(fmt, a, x, y);
    printf("\n");
    if (!ok) failures++;
}

static dlsym_fn real_dlsym(void) {
    void *libc = dlopen("libc.so.6", RTLD_NOW | RTLD_NOLOAD);
    dlsym_fn f = libc ? (dlsym_fn)dlvsym(libc, "dlsym", "GLIBC_2.34") : NULL;
    if (f == NULL && libc) f = (dlsym_fn)dlvsym(libc, "dlsym", "GLIBC_2.2.5");
    return f;
}

#define MB(x) ((double)(x) / (1 << 20))

int main(int argc, char **argv) {
    int enabled = argc > 2 && strcmp(argv[2], "enabled") == 0;
    size_t alloc_mb = argc > 3 ? strtoull(argv[3], NULL, 10) : 1024;
    dlsym_fn rd = real_dlsym();
    void *cu = dlopen("libcuda.so.1", RTLD_NOW);
    meminfo_fn r_info = rd ? (meminfo_fn)rd(cu, "cuMemGetInfo_v2") : NULL;
    totalmem_fn r_total = rd ? (totalmem_fn)rd(cu, "cuDeviceTotalMem_v2") : NULL;
    if (!r_info || !r_total || cuInit(0) != CUDA_SUCCESS) {
        printf("FAIL: setup\nRESULT: FAIL (1 failures)\n");
        return 1;
    }
    int n = 0;
    cuDeviceGetCount(&n);
    check(n >= 2, "device count %d >= 2 (%.0f %.0f)", n, 0, 0);
    CUcontext ctx[16];
    for (int d = 0; d < n && d < 16; d++) {
        CUdevice dev;
        cuDeviceGet(&dev, d);
        cuDevicePrimaryCtxRetain(&ctx[d], dev);
    }
    size_t hf[16], ht[16];
    for (int d = 0; d < n && d < 16; d++) {
        CUdevice dev;
        cuDeviceGet(&dev, d);
        cuCtxSetCurrent(ctx[d]);
        size_t f, t, rf, rt, tm, rtm;
        cuMemGetInfo(&f, &t);
        r_info(&rf, &rt);
        cuDeviceTotalMem(&tm, dev);
        r_total(&rtm, dev);
        hf[d] = f;
        ht[d] = t;
        if (enabled) {
            check(t <= rt && t > 0, "dev %d cuMemGetInfo total %.0f MiB capped at driver %.0f MiB", d, MB(t), MB(rt));
            check(tm <= rtm && tm > 0, "dev %d cuDeviceTotalMem %.0f MiB capped at driver %.0f MiB", d, MB(tm), MB(rtm));
        } else {
            check(t == rt, "dev %d cuMemGetInfo total %.0f MiB == driver %.0f MiB", d, MB(t), MB(rt));
            check((f > rf ? f - rf : rf - f) < (64u << 20), "dev %d free %.0f MiB ~ driver %.0f MiB", d, MB(f), MB(rf));
            check(tm == rtm, "dev %d cuDeviceTotalMem %.0f MiB == driver %.0f MiB", d, MB(tm), MB(rtm));
        }
    }
    if (enabled && n >= 2) {
        int target = n - 1;
        cuCtxSetCurrent(ctx[target]);
        CUdeviceptr p;
        CUresult r = cuMemAlloc(&p, alloc_mb << 20);
        check(r == CUDA_SUCCESS, "alloc %d MiB on last device (%.0f, %.0f)", (int)alloc_mb, 0, 0);
        for (int d = 0; d < n && d < 16; d++) {
            size_t f, t;
            cuCtxSetCurrent(ctx[d]);
            cuMemGetInfo(&f, &t);
            double drop = MB(hf[d]) - MB(f);
            if (d == target) {
                check(drop > alloc_mb * 0.9, "dev %d free dropped by %.0f MiB (expected ~%.0f)", d, drop,
                      (double)alloc_mb);
            } else {
                check(drop < alloc_mb * 0.5, "dev %d free unchanged: dropped %.0f MiB (alloc was %.0f on other dev)",
                      d, drop, (double)alloc_mb);
            }
        }
        cuCtxSetCurrent(ctx[target]);
        if (r == CUDA_SUCCESS) cuMemFree(p);
    }
    printf("RESULT: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
