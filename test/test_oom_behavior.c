/*
 * test_oom_behavior.c - verify SoftMig now mimics real-GPU OOM behavior.
 *
 * Builds against the CUDA runtime. Without SoftMig, on any normal GPU,
 * an over-budget cudaMalloc returns cudaErrorMemoryAllocation and the
 * process keeps running. The legacy SoftMig used to SIGKILL the caller
 * via active_oom_killer(). After the fix, both behaviors should match.
 *
 * Designed to be run inside a SoftMig-sliced job, e.g.:
 *   srun --reservation=softmig --gres=gpu:l40s.4:1 ./test_oom_behavior
 *
 * Exits 0 on success (got cudaErrorMemoryAllocation, process survived).
 * Exits 1 on unexpected error (still alive but wrong error code).
 * Gets killed externally only if the OOM killer fires (regression).
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>

static const char* err_name(cudaError_t e) {
    const char* n = cudaGetErrorName(e);
    return n ? n : "(unknown)";
}

static void print_meminfo(const char* tag) {
    size_t free_b = 0, total_b = 0;
    cudaError_t r = cudaMemGetInfo(&free_b, &total_b);
    printf("[MEM %-8s] cudaMemGetInfo=%d (%s) free=%.2f MB total=%.2f MB used=%.2f MB\n",
           tag, (int)r, err_name(r),
           free_b / (1024.0 * 1024.0),
           total_b / (1024.0 * 1024.0),
           (total_b - free_b) / (1024.0 * 1024.0));
    fflush(stdout);
}

int main(void) {
    printf("[START] PID=%d\n", (int)getpid());
    fflush(stdout);

    cudaError_t r = cudaSetDevice(0);
    printf("[SETUP] cudaSetDevice(0)=%d (%s)\n", (int)r, err_name(r));
    if (r != cudaSuccess) return 1;

    print_meminfo("initial");

    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);

    /* TEST 1: ask for 150% of free memory. Must return cudaErrorMemoryAllocation
       and the process must survive. */
    size_t want = (size_t)(free_b * 1.5);
    void* p = NULL;
    printf("[TEST1 ] requesting 150%% of free: %.2f MB\n", want / (1024.0 * 1024.0));
    fflush(stdout);
    r = cudaMalloc(&p, want);
    printf("[TEST1 ] cudaMalloc returned %d (%s) ptr=%p\n", (int)r, err_name(r), p);
    fflush(stdout);
    if (r != cudaErrorMemoryAllocation) {
        printf("[TEST1 ] FAIL: expected cudaErrorMemoryAllocation, got %s\n", err_name(r));
        if (p) cudaFree(p);
        return 1;
    }
    print_meminfo("post-T1");

    /* Confirm the device is still functional - small allocation should still work. */
    void* small = NULL;
    r = cudaMalloc(&small, 1024);
    printf("[TEST1 ] post-OOM small cudaMalloc(1024)=%d (%s)\n", (int)r, err_name(r));
    if (small) cudaFree(small);
    fflush(stdout);

    /* TEST 2: 50 MB chunks until we hit OOM. */
    const size_t CHUNK = 50ULL * 1024 * 1024;
    enum { MAX_CHUNKS = 4096 };
    void** ptrs = (void**)calloc(MAX_CHUNKS, sizeof(void*));
    int n = 0;
    printf("[TEST2 ] looping 50 MB chunks...\n");
    fflush(stdout);
    for (n = 0; n < MAX_CHUNKS; ++n) {
        r = cudaMalloc(&ptrs[n], CHUNK);
        if (r != cudaSuccess) {
            printf("[TEST2 ] OOM at iter=%d after %.2f MB allocated, ret=%d (%s)\n",
                   n, n * (CHUNK / (1024.0 * 1024.0)), (int)r, err_name(r));
            break;
        }
    }
    if (r != cudaErrorMemoryAllocation) {
        printf("[TEST2 ] FAIL: expected cudaErrorMemoryAllocation, got %s\n", err_name(r));
    }
    print_meminfo("at-OOM");
    for (int i = 0; i < n; ++i) cudaFree(ptrs[i]);
    free(ptrs);
    print_meminfo("cleanup");

    printf("[DONE  ] PID=%d survived OOM cleanly\n", (int)getpid());
    return (r == cudaErrorMemoryAllocation) ? 0 : 1;
}
