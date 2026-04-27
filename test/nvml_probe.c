// nvml_probe.c - allocate N MB of device mem, then periodically dump what NVML
// reports for running processes on device 0. Useful for checking that every
// sibling process is visible through NVML while SoftMig is hooked in.
//
// Build: nvcc -O2 -o nvml_probe nvml_probe.c -lcuda -lnvidia-ml
// Run  : ./nvml_probe <alloc_mb> <hold_seconds>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <cuda.h>
#include <nvml.h>

static void ck(CUresult r, const char *w) {
    if (r != CUDA_SUCCESS) {
        const char *s = NULL; cuGetErrorString(r, &s);
        fprintf(stderr, "[pid=%d] CUDA err in %s: %s\n", (int)getpid(), w, s ? s : "?");
        exit(1);
    }
}
static void cn(nvmlReturn_t r, const char *w) {
    if (r != NVML_SUCCESS) {
        fprintf(stderr, "[pid=%d] NVML err in %s: %s\n", (int)getpid(), w, nvmlErrorString(r));
        exit(1);
    }
}

int main(int argc, char **argv) {
    size_t alloc_mb = (argc > 1) ? strtoull(argv[1], NULL, 10) : 512;
    int hold_s      = (argc > 2) ? atoi(argv[2]) : 30;

    ck(cuInit(0), "cuInit");
    CUdevice dev; ck(cuDeviceGet(&dev, 0), "cuDeviceGet");
    CUcontext ctx; ck(cuDevicePrimaryCtxRetain(&ctx, dev), "ctxRetain");
    ck(cuCtxSetCurrent(ctx), "ctxSet");

    size_t bytes = alloc_mb * 1024ULL * 1024ULL;
    CUdeviceptr p;
    CUresult r = cuMemAlloc(&p, bytes);
    if (r != CUDA_SUCCESS) {
        const char *s=NULL; cuGetErrorString(r, &s);
        fprintf(stderr, "[pid=%d] ALLOC FAILED (%zu MB): %s\n", (int)getpid(), alloc_mb, s?s:"?");
    } else {
        fprintf(stderr, "[pid=%d] alloc ok (%zu MB)\n", (int)getpid(), alloc_mb);
    }
    fflush(stderr);

    cn(nvmlInit_v2(), "nvmlInit");
    nvmlDevice_t nd; cn(nvmlDeviceGetHandleByIndex_v2(0, &nd), "getHandle");

    for (int t = 0; t < hold_s; ++t) {
        unsigned int cnt = 64;
        nvmlProcessInfo_t infos[64];
        memset(infos, 0, sizeof(infos));
        nvmlReturn_t qr = nvmlDeviceGetComputeRunningProcesses_v2(nd, &cnt, infos);
        if (qr == NVML_ERROR_INSUFFICIENT_SIZE) qr = NVML_SUCCESS;
        printf("[pid=%d t=%d] nvml_cnt=%u :", (int)getpid(), t, cnt);
        if (qr == NVML_SUCCESS) {
            for (unsigned i = 0; i < cnt && i < 64; ++i) {
                printf(" %u/%lluMB", infos[i].pid,
                       (unsigned long long)(infos[i].usedGpuMemory >> 20));
            }
        } else {
            printf(" err=%s", nvmlErrorString(qr));
        }
        printf("\n"); fflush(stdout);
        sleep(1);
    }

    if (r == CUDA_SUCCESS) cuMemFree(p);
    cuDevicePrimaryCtxRelease(dev);
    nvmlShutdown();
    return 0;
}
