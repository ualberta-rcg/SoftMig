/*
 * bench_overhead - per-call cost of SoftMig's exported wrappers.
 *
 * Usage: bench_overhead [launches] [allocs]
 *
 * Times the same driver calls two ways in one process: through the
 * direct-linked symbols (which resolve to libsoftmig's exported wrappers when
 * it is preloaded) and through the driver's own functions obtained with
 * glibc's real dlsym on libcuda. Prints ns/call for each and the difference.
 * Informational: in a passive job the delta is the cost of the passive
 * forward; in a slice job it includes SoftMig's accounting.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void *(*dlsym_fn)(void *, const char *);
typedef CUresult (*launch_fn)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                              CUstream, void **, void **);
typedef CUresult (*alloc_fn)(CUdeviceptr *, size_t);
typedef CUresult (*free_fn)(CUdeviceptr);
typedef CUresult (*meminfo_fn)(size_t *, size_t *);

static const char ptx[] =
    ".version 7.0\n.target sm_52\n.address_size 64\n"
    ".visible .entry k()\n{\n ret;\n}\n";

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static dlsym_fn real_dlsym(void) {
    void *libc = dlopen("libc.so.6", RTLD_NOW | RTLD_NOLOAD);
    dlsym_fn f = libc ? (dlsym_fn)dlvsym(libc, "dlsym", "GLIBC_2.34") : NULL;
    if (f == NULL && libc) f = (dlsym_fn)dlvsym(libc, "dlsym", "GLIBC_2.2.5");
    return f;
}

#define CK(x)                                                         \
    do {                                                              \
        CUresult _r = (x);                                            \
        if (_r != CUDA_SUCCESS) {                                     \
            fprintf(stderr, "%s failed: %d\n", #x, (int)_r);          \
            exit(1);                                                  \
        }                                                             \
    } while (0)

int main(int argc, char **argv) {
    int nl = argc > 1 ? atoi(argv[1]) : 100000;
    int na = argc > 2 ? atoi(argv[2]) : 10000;
    dlsym_fn rd = real_dlsym();
    void *cu = dlopen("libcuda.so.1", RTLD_NOW);
    if (!rd || !cu) {
        fprintf(stderr, "cannot resolve real dlsym/libcuda\n");
        return 1;
    }
    launch_fn r_launch = (launch_fn)rd(cu, "cuLaunchKernel");
    alloc_fn r_alloc = (alloc_fn)rd(cu, "cuMemAlloc_v2");
    free_fn r_free = (free_fn)rd(cu, "cuMemFree_v2");
    meminfo_fn r_info = (meminfo_fn)rd(cu, "cuMemGetInfo_v2");

    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    CK(cuInit(0));
    CK(cuDeviceGet(&dev, 0));
    CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));
    CK(cuModuleLoadData(&mod, ptx));
    CK(cuModuleGetFunction(&fn, mod, "k"));

    size_t f, t;
    CUdeviceptr p;
    double t0;

    // Each pair is timed in ABBA order (hook, real, real, hook) after a
    // warm-up, so driver warm-up and drift do not land on one side.
#define TIME_LAUNCH(fnp, n) (t0 = now(), ({ for (int i = 0; i < (n); i++) CK(fnp(fn, 1, 1, 1, 1, 1, 1, 0, 0, NULL, NULL)); CK(cuCtxSynchronize()); }), (now() - t0) / (n))
#define TIME_ALLOC(af, ff, n) (t0 = now(), ({ for (int i = 0; i < (n); i++) { CK(af(&p, 1 << 20)); CK(ff(p)); } }), (now() - t0) / (n))
#define TIME_INFO(inf, n) (t0 = now(), ({ for (int i = 0; i < (n); i++) CK(inf(&f, &t)); }), (now() - t0) / (n))

    TIME_LAUNCH(cuLaunchKernel, 2000);
    TIME_LAUNCH(r_launch, 2000);
    TIME_ALLOC(cuMemAlloc, cuMemFree, 500);
    TIME_ALLOC(r_alloc, r_free, 500);

    double h1 = TIME_LAUNCH(cuLaunchKernel, nl / 2), r1 = TIME_LAUNCH(r_launch, nl / 2);
    double r2 = TIME_LAUNCH(r_launch, nl / 2), h2 = TIME_LAUNCH(cuLaunchKernel, nl / 2);
    double hook_l = (h1 + h2) / 2, real_l = (r1 + r2) / 2;
    h1 = TIME_ALLOC(cuMemAlloc, cuMemFree, na / 2); r1 = TIME_ALLOC(r_alloc, r_free, na / 2);
    r2 = TIME_ALLOC(r_alloc, r_free, na / 2); h2 = TIME_ALLOC(cuMemAlloc, cuMemFree, na / 2);
    double hook_a = (h1 + h2) / 2, real_a = (r1 + r2) / 2;
    h1 = TIME_INFO(cuMemGetInfo, na / 2); r1 = TIME_INFO(r_info, na / 2);
    r2 = TIME_INFO(r_info, na / 2); h2 = TIME_INFO(cuMemGetInfo, na / 2);
    double hook_i = (h1 + h2) / 2, real_i = (r1 + r2) / 2;

    printf("bench_overhead launch_ns hook=%.0f real=%.0f delta=%.0f\n", hook_l, real_l, hook_l - real_l);
    printf("bench_overhead allocfree_ns hook=%.0f real=%.0f delta=%.0f\n", hook_a, real_a, hook_a - real_a);
    printf("bench_overhead meminfo_ns hook=%.0f real=%.0f delta=%.0f\n", hook_i, real_i, hook_i - real_i);
    return 0;
}
