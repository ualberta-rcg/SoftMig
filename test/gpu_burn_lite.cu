// gpu_burn_lite.cu - minimalist gpu-burn replacement for SoftMig stress tests.
// Allocates ALLOC_MB of device memory and launches a long-running compute
// kernel in a hot loop, periodically reporting achieved throughput.
// Deliberately uses the cuda runtime (cudart) so the app is representative
// of how gpu-burn, pytorch, etc. interact with the driver through
// cuGetProcAddress/dlsym (which SoftMig hooks).
//
// Build: nvcc -O2 -arch=sm_80 -o gpu_burn_lite gpu_burn_lite.cu
// Run  : ./gpu_burn_lite <alloc_mb> <run_seconds>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <cuda_runtime.h>

#define CK(e) do { cudaError_t _e=(e); if (_e!=cudaSuccess){fprintf(stderr,"[pid=%d] %s: %s\n",(int)getpid(),#e,cudaGetErrorString(_e)); exit(1);} } while(0)

// Single-precision hot loop kernel. Each thread does N iterations of FMAs on
// values loaded from global memory, writing a reduction back. Designed to
// hit high sustained SM utilization.
__global__ void burn_kernel(float *a, float *b, float *c, size_t n, int iters) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = a[idx];
    float y = b[idx];
    #pragma unroll 8
    for (int i = 0; i < iters; ++i) {
        x = x * y + 1.0f;
        y = y * x + 1.0f;
    }
    c[idx] = x + y;
}

int main(int argc, char **argv) {
    size_t alloc_mb = (argc > 1) ? strtoull(argv[1], NULL, 10) : 1024;
    int run_s      = (argc > 2) ? atoi(argv[2]) : 30;

    CK(cudaSetDevice(0));

    size_t bytes = alloc_mb * 1024ULL * 1024ULL;
    size_t n = bytes / (3 * sizeof(float));  // three float buffers of n elements each
    float *a=NULL, *b=NULL, *c=NULL;
    CK(cudaMalloc(&a, n*sizeof(float)));
    CK(cudaMalloc(&b, n*sizeof(float)));
    CK(cudaMalloc(&c, n*sizeof(float)));
    CK(cudaMemset(a, 1, n*sizeof(float)));
    CK(cudaMemset(b, 2, n*sizeof(float)));
    fprintf(stderr, "[pid=%d] allocated 3x %.1f MB, n=%zu floats\n",
            (int)getpid(), (n*sizeof(float))/(1024.0*1024.0), n);
    fflush(stderr);

    int block = 256;
    int grid  = (int)((n + block - 1) / block);
    if (grid > 65535) grid = 65535;

    struct timespec t0, now;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    size_t launches = 0;
    while (1) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        double dt = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) / 1e9;
        if (dt >= run_s) break;
        burn_kernel<<<grid, block>>>(a, b, c, n, 256);
        launches++;
        // Force a sync every 100 launches so we can actually observe utilization
        if ((launches % 100) == 0) {
            CK(cudaDeviceSynchronize());
            fprintf(stdout, "[pid=%d t=%.1f] launches=%zu grid=%d block=%d\n",
                    (int)getpid(), dt, launches, grid, block);
            fflush(stdout);
        }
    }
    CK(cudaDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) / 1e9;
    fprintf(stdout, "[pid=%d] DONE: %zu launches in %.2f s (%.1f launches/s)\n",
            (int)getpid(), launches, elapsed, launches / elapsed);

    cudaFree(a); cudaFree(b); cudaFree(c);
    cudaDeviceReset();
    return 0;
}
