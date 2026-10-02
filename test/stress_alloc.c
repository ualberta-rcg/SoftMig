// stress_alloc.c - hammer SoftMig's allocation path from many threads.
//
// Usage: stress_alloc <threads> <seconds> <max_mb> [async]
//
// Each thread binds the primary context and, until <seconds> have passed,
// allocates a random 1..max_mb MiB buffer (cuMemAlloc, or alternating
// cuMemAllocAsync/cuMemFreeAsync on a per-thread stream when async=1),
// touches it, and frees it. CUDA_ERROR_OUT_OF_MEMORY is expected under a
// slice limit and counted; any other error is a failure. <seconds> = 0 does
// one tiny alloc/free and exits (the "sweeper" used after fault injection).
// Exit 0 iff no unexpected errors.

#include <cuda.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static CUcontext ctx;
static int seconds, max_mb, use_async;
static double t_end;

typedef struct {
    int id;
    long ok, oom, err;
    CUresult first_err;
} stats_t;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void *worker(void *arg) {
    stats_t *s = arg;
    unsigned int seed = (unsigned int)(getpid() * 131 + s->id);
    cuCtxSetCurrent(ctx);
    CUstream stream = NULL;
    if (use_async && cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
        stream = NULL;
    }
    do {
        size_t bytes = seconds == 0 ? 4096 : (size_t)(1 + rand_r(&seed) % max_mb) << 20;
        int async = use_async && stream != NULL && (rand_r(&seed) & 1);
        CUdeviceptr p = 0;
        CUresult r = async ? cuMemAllocAsync(&p, bytes, stream) : cuMemAlloc(&p, bytes);
        if (r == CUDA_ERROR_OUT_OF_MEMORY) {
            s->oom++;
            continue;
        }
        if (r != CUDA_SUCCESS) {
            if (!s->err++) s->first_err = r;
            continue;
        }
        r = async ? cuMemsetD8Async(p, 0x5a, 4096, stream) : cuMemsetD8(p, 0x5a, 4096);
        if (r == CUDA_SUCCESS && async) r = cuStreamSynchronize(stream);
        CUresult fr = async ? cuMemFreeAsync(p, stream) : cuMemFree(p);
        if (fr == CUDA_SUCCESS && async) fr = cuStreamSynchronize(stream);
        if (r != CUDA_SUCCESS || fr != CUDA_SUCCESS) {
            if (!s->err++) s->first_err = r != CUDA_SUCCESS ? r : fr;
            continue;
        }
        s->ok++;
    } while (seconds > 0 && now() < t_end);
    if (stream) cuStreamDestroy(stream);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <threads> <seconds> <max_mb> [async]\n", argv[0]);
        return 2;
    }
    int threads = atoi(argv[1]);
    seconds = atoi(argv[2]);
    max_mb = atoi(argv[3]);
    use_async = argc > 4 && atoi(argv[4]);
    if (threads < 1 || threads > 64 || max_mb < 1) return 2;

    CUdevice dev;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) {
        fprintf(stderr, "[pid=%d] stress_alloc: CUDA init failed\n", (int)getpid());
        return 1;
    }
    t_end = now() + seconds;

    pthread_t th[64];
    stats_t st[64] = {0};
    for (int i = 0; i < threads; i++) {
        st[i].id = i;
        pthread_create(&th[i], NULL, worker, &st[i]);
    }
    long ok = 0, oom = 0, err = 0;
    CUresult first = CUDA_SUCCESS;
    for (int i = 0; i < threads; i++) {
        pthread_join(th[i], NULL);
        ok += st[i].ok;
        oom += st[i].oom;
        err += st[i].err;
        if (first == CUDA_SUCCESS && st[i].err) first = st[i].first_err;
    }
    cuDevicePrimaryCtxRelease(dev);
    printf("[pid=%d] stress_alloc done threads=%d ok=%ld oom=%ld err=%ld first_err=%d\n", (int)getpid(),
           threads, ok, oom, err, (int)first);
    return err ? 1 : 0;
}
