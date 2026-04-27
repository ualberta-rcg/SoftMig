// runtime_hold.cu - cudart-based memory holder. Uses cudaMalloc so that cudart's
// internal cuGetProcAddress/dlsym path is used, which SoftMig intercepts.
//
// Build: nvcc -O2 -o runtime_hold runtime_hold.cu
// Run  : ./runtime_hold <alloc_mb> <hold_seconds>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <cuda_runtime.h>

#define CK(e) do { cudaError_t _e=(e); if (_e!=cudaSuccess){fprintf(stderr,"[pid=%d] %s: %s\n",(int)getpid(),#e,cudaGetErrorString(_e)); exit(1);} } while(0)

int main(int argc, char **argv) {
    size_t alloc_mb = (argc > 1) ? strtoull(argv[1], NULL, 10) : 512;
    int hold_s      = (argc > 2) ? atoi(argv[2]) : 30;
    CK(cudaSetDevice(0));
    size_t bytes = alloc_mb * 1024ULL * 1024ULL;
    void *p = NULL;
    cudaError_t r = cudaMalloc(&p, bytes);
    if (r != cudaSuccess) {
        fprintf(stderr, "[pid=%d] cudaMalloc(%zuMB) FAILED: %s\n",
                (int)getpid(), alloc_mb, cudaGetErrorString(r));
    } else {
        fprintf(stderr, "[pid=%d] cudaMalloc(%zuMB) ok\n", (int)getpid(), alloc_mb);
        CK(cudaMemset(p, 0, bytes));
    }
    for (int t = 0; t < hold_s; ++t) {
        size_t f=0,tot=0; cudaMemGetInfo(&f,&tot);
        fprintf(stdout, "[pid=%d t=%d] free=%zuMB total=%zuMB\n",
                (int)getpid(), t, f>>20, tot>>20);
        fflush(stdout);
        sleep(1);
    }
    if (p) cudaFree(p);
    cudaDeviceReset();
    return 0;
}
