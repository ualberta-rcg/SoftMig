// bypass_probe.cu - try to get past the SoftMig memory limit through one
// allocation API at a time. The node-side sampler (root nvidia-smi) supplies
// the truth; this program only reports what the API returned.
//
// Usage: bypass_probe --api NAME --limit-mb L [--step-mb 512] [--hold 4] [--list]
//
// Allocates STEP MiB through NAME until the API fails or the total reaches
// 1.5 x L, touches every allocation so memory is really committed, holds the
// peak for --hold seconds (so the sampler sees it), then prints one line:
//
//   BYPASS api=<name> pid=<pid> limit_mb=<L> allocated_mb=<X> steps=<N>
//          status=OOM|REACHED|ERROR err=<text>
//
// The suite classifies: OOM with truth <= limit+slack -> CAPPED,
// REACHED (device APIs) -> UNHOOKED, truth > limit+slack -> LEAK.
// Host APIs (cuMemAllocHost, cudaHostAlloc) are expected to REACH: they must
// not be counted against the device limit.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

static const char *cuerr(CUresult r) { const char *s = "?"; cuGetErrorString(r, &s); return s ? s : "?"; }

struct Ctx {
    size_t step;            // bytes per step
    std::vector<void *> keep;
    std::vector<CUmemGenericAllocationHandle> handles;
    std::vector<cudaArray_t> arrays;
    std::vector<cudaMipmappedArray_t> mips;
    std::vector<CUcontext> ctxs;
    std::vector<cudaGraphExec_t> graphs;
    cudaStream_t stream = nullptr;
    CUmemoryPool cupool = nullptr;
    cudaMemPool_t rtpool = nullptr;
    std::string err;
};

__global__ void touch_kernel(unsigned char *p, size_t n) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4096;
    if (i < n) p[i] = 1;
}

static bool touch(void *p, size_t n) {
    // Commit physical pages. Memset on the stream covers async/pool/graph memory too.
    cudaError_t e = cudaMemsetAsync(p, 1, n, 0);
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { cudaGetLastError(); return false; }
    return true;
}

// Each step function returns: 0 ok, 1 OOM, 2 other error (sets c.err)
typedef int (*step_fn)(Ctx &c);

static int classify_rt(cudaError_t e, Ctx &c) {
    if (e == cudaSuccess) return 0;
    c.err = cudaGetErrorString(e); cudaGetLastError();
    return e == cudaErrorMemoryAllocation ? 1 : 2;
}
static int classify_cu(CUresult r, Ctx &c) {
    if (r == CUDA_SUCCESS) return 0;
    c.err = cuerr(r);
    return r == CUDA_ERROR_OUT_OF_MEMORY ? 1 : 2;
}

static int s_cudaMalloc(Ctx &c) {
    void *p; int k = classify_rt(cudaMalloc(&p, c.step), c); if (k) return k;
    c.keep.push_back(p); return touch(p, c.step) ? 0 : 2;
}
static int s_cuMemAlloc(Ctx &c) {
    CUdeviceptr d; int k = classify_cu(cuMemAlloc(&d, c.step), c); if (k) return k;
    c.keep.push_back((void *)d); return touch((void *)d, c.step) ? 0 : 2;
}
static int s_cudaMallocAsync(Ctx &c) {
    void *p; int k = classify_rt(cudaMallocAsync(&p, c.step, c.stream), c); if (k) return k;
    cudaStreamSynchronize(c.stream);
    c.keep.push_back(p); return touch(p, c.step) ? 0 : 2;
}
static int s_cuMemAllocAsync(Ctx &c) {
    CUdeviceptr d; int k = classify_cu(cuMemAllocAsync(&d, c.step, c.stream), c); if (k) return k;
    cudaStreamSynchronize(c.stream);
    c.keep.push_back((void *)d); return touch((void *)d, c.step) ? 0 : 2;
}
static int s_cuMemAllocFromPoolAsync(Ctx &c) {
    if (!c.cupool) {
        CUmemPoolProps props = {}; props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
        props.handleTypes = CU_MEM_HANDLE_TYPE_NONE; props.location.type = CU_MEM_LOCATION_TYPE_DEVICE; props.location.id = 0;
        int k = classify_cu(cuMemPoolCreate(&c.cupool, &props), c); if (k) return k;
        cuuint64_t thr = UINT64_MAX; cuMemPoolSetAttribute(c.cupool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &thr);
    }
    CUdeviceptr d; int k = classify_cu(cuMemAllocFromPoolAsync(&d, c.step, c.cupool, c.stream), c); if (k) return k;
    cudaStreamSynchronize(c.stream);
    c.keep.push_back((void *)d); return touch((void *)d, c.step) ? 0 : 2;
}
static int s_cudaMallocFromPoolAsync(Ctx &c) {
    if (!c.rtpool) {
        cudaMemPoolProps props = {}; props.allocType = cudaMemAllocationTypePinned;
        props.handleTypes = cudaMemHandleTypeNone; props.location.type = cudaMemLocationTypeDevice; props.location.id = 0;
        int k = classify_rt(cudaMemPoolCreate(&c.rtpool, &props), c); if (k) return k;
        cuuint64_t thr = UINT64_MAX; cudaMemPoolSetAttribute(c.rtpool, cudaMemPoolAttrReleaseThreshold, &thr);
        int one = 1;
        cudaMemPoolSetAttribute(c.rtpool, cudaMemPoolReuseFollowEventDependencies, &one);
        cudaMemPoolSetAttribute(c.rtpool, cudaMemPoolReuseAllowOpportunistic, &one);
        cudaMemPoolSetAttribute(c.rtpool, cudaMemPoolReuseAllowInternalDependencies, &one);
    }
    void *p; int k = classify_rt(cudaMallocFromPoolAsync(&p, c.step, c.rtpool, c.stream), c); if (k) return k;
    cudaStreamSynchronize(c.stream);
    c.keep.push_back(p); return touch(p, c.step) ? 0 : 2;
}
static int s_cuMemCreateMap(Ctx &c) {   // VMM: what PyTorch expandable_segments uses
    CUmemAllocationProp prop = {}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
    size_t gran = 0; cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    size_t sz = gran ? ((c.step + gran - 1) / gran) * gran : c.step;
    CUmemGenericAllocationHandle h; int k = classify_cu(cuMemCreate(&h, sz, &prop, 0), c); if (k) return k;
    CUdeviceptr d; k = classify_cu(cuMemAddressReserve(&d, sz, 0, 0, 0), c); if (k) { cuMemRelease(h); return k; }
    k = classify_cu(cuMemMap(d, sz, 0, h, 0), c); if (k) { cuMemRelease(h); return k; }
    CUmemAccessDesc acc = {}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    k = classify_cu(cuMemSetAccess(d, sz, &acc, 1), c); if (k) return k;
    c.handles.push_back(h); c.keep.push_back((void *)d);
    return touch((void *)d, sz) ? 0 : 2;
}
static int s_cuMemAllocManaged(Ctx &c) {
    CUdeviceptr d; int k = classify_cu(cuMemAllocManaged(&d, c.step, CU_MEM_ATTACH_GLOBAL), c); if (k) return k;
    c.keep.push_back((void *)d); return touch((void *)d, c.step) ? 0 : 2;
}
static int s_cudaMallocManaged(Ctx &c) {
    void *p; int k = classify_rt(cudaMallocManaged(&p, c.step), c); if (k) return k;
    c.keep.push_back(p); return touch(p, c.step) ? 0 : 2;
}
static int s_cudaMallocPitch(Ctx &c) {
    void *p; size_t pitch; size_t w = 65536, h = c.step / w;
    int k = classify_rt(cudaMallocPitch(&p, &pitch, w, h), c); if (k) return k;
    c.keep.push_back(p); return touch(p, pitch * h) ? 0 : 2;
}
static int s_cuMemAllocPitch(Ctx &c) {
    CUdeviceptr d; size_t pitch; size_t w = 65536, h = c.step / w;
    int k = classify_cu(cuMemAllocPitch(&d, &pitch, w, h, 4), c); if (k) return k;
    c.keep.push_back((void *)d); return touch((void *)d, pitch * h) ? 0 : 2;
}
static int s_cudaMalloc3D(Ctx &c) {
    cudaPitchedPtr pp; cudaExtent ext = make_cudaExtent(4096, 1024, c.step / (4096 * 1024));
    int k = classify_rt(cudaMalloc3D(&pp, ext), c); if (k) return k;
    c.keep.push_back(pp.ptr); return touch(pp.ptr, pp.pitch * ext.height * ext.depth) ? 0 : 2;
}
static int s_cudaMallocArray(Ctx &c) {
    cudaChannelFormatDesc d = cudaCreateChannelDesc<float>();
    cudaArray_t a; size_t w = 16384, h = c.step / (w * 4);
    int k = classify_rt(cudaMallocArray(&a, &d, w, h, cudaArrayDefault), c); if (k) return k;
    c.arrays.push_back(a);
    // arrays can't be memset; fill via a copy from a device buffer that we free again
    void *tmp; if (cudaMalloc(&tmp, w * 4 * h) != cudaSuccess) { cudaGetLastError(); return 0; }
    cudaMemset(tmp, 1, w * 4 * h);
    cudaMemcpy2DToArray(a, 0, 0, tmp, w * 4, w * 4, h, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize(); cudaFree(tmp); cudaGetLastError();
    return 0;
}
static int s_cudaMalloc3DArray(Ctx &c) {
    cudaChannelFormatDesc d = cudaCreateChannelDesc<float>();
    cudaArray_t a; cudaExtent ext = make_cudaExtent(1024, 1024, c.step / (1024 * 1024 * 4));
    int k = classify_rt(cudaMalloc3DArray(&a, &d, ext, cudaArrayDefault), c); if (k) return k;
    c.arrays.push_back(a); return 0;
}
static int s_cudaMallocMipmappedArray(Ctx &c) {
    cudaChannelFormatDesc d = cudaCreateChannelDesc<float>();
    cudaMipmappedArray_t m; cudaExtent ext = make_cudaExtent(1024, 1024, c.step / (1024 * 1024 * 4));
    int k = classify_rt(cudaMallocMipmappedArray(&m, &d, ext, 1, cudaArrayDefault), c); if (k) return k;
    c.mips.push_back(m); return 0;
}
static int s_cuArrayCreate(Ctx &c) {
    CUDA_ARRAY_DESCRIPTOR d = {}; d.Format = CU_AD_FORMAT_FLOAT; d.NumChannels = 1;
    d.Width = 16384; d.Height = c.step / (16384 * 4);
    CUarray a; int k = classify_cu(cuArrayCreate(&a, &d), c); if (k) return k;
    c.arrays.push_back((cudaArray_t)a); return 0;
}
static int s_cuArray3DCreate(Ctx &c) {
    CUDA_ARRAY3D_DESCRIPTOR d = {}; d.Format = CU_AD_FORMAT_FLOAT; d.NumChannels = 1;
    d.Width = 1024; d.Height = 1024; d.Depth = c.step / (1024 * 1024 * 4);
    CUarray a; int k = classify_cu(cuArray3DCreate(&a, &d), c); if (k) return k;
    c.arrays.push_back((cudaArray_t)a); return 0;
}
static int s_cuGraphAddMemAllocNode(Ctx &c) {
    cudaGraph_t g; int k = classify_rt(cudaGraphCreate(&g, 0), c); if (k) return k;
    cudaMemAllocNodeParams np = {}; np.poolProps.allocType = cudaMemAllocationTypePinned;
    np.poolProps.location.type = cudaMemLocationTypeDevice; np.poolProps.location.id = 0;
    np.bytesize = c.step;
    cudaGraphNode_t an; k = classify_rt(cudaGraphAddMemAllocNode(&an, g, nullptr, 0, &np), c); if (k) return k;
    cudaMemsetParams mp = {}; mp.dst = np.dptr; mp.value = 1; mp.elementSize = 1; mp.width = c.step; mp.height = 1;
    cudaGraphNode_t mn; k = classify_rt(cudaGraphAddMemsetNode(&mn, g, &an, 1, &mp), c); if (k) return k;
    cudaGraphExec_t ge; k = classify_rt(cudaGraphInstantiate(&ge, g, 0), c); if (k) return k;
    k = classify_rt(cudaGraphLaunch(ge, c.stream), c); if (k) return k;
    k = classify_rt(cudaStreamSynchronize(c.stream), c); if (k) return k;
    c.graphs.push_back(ge); c.keep.push_back(np.dptr);   // memory stays owned by the graph (no free node)
    return 0;
}
static int s_cuMemAllocHost(Ctx &c) {
    void *p; int k = classify_cu(cuMemAllocHost(&p, c.step), c); if (k) return k;
    memset(p, 1, c.step); c.keep.push_back(p); return 0;
}
static int s_cudaHostAlloc(Ctx &c) {
    void *p; int k = classify_rt(cudaHostAlloc(&p, c.step, cudaHostAllocDefault), c); if (k) return k;
    memset(p, 1, c.step); c.keep.push_back(p); return 0;
}
static int s_cuCtxCreate(Ctx &c) {   // context overhead: each new context costs real device memory
    CUdevice dev; cuDeviceGet(&dev, 0);
#if CUDA_VERSION >= 13000
    CUcontext cx; int k = classify_cu(cuCtxCreate(&cx, nullptr, 0, dev), c); if (k) return k;
#else
    CUcontext cx; int k = classify_cu(cuCtxCreate(&cx, 0, dev), c); if (k) return k;
#endif
    CUdeviceptr d; k = classify_cu(cuMemAlloc(&d, 64 << 20), c); if (k) return k;
    cuMemsetD8(d, 1, 64 << 20); cuCtxSynchronize();
    c.ctxs.push_back(cx);
    return 0;
}

struct Api { const char *name; step_fn fn; bool host; size_t step_override; };
static Api APIS[] = {
    {"cudaMalloc", s_cudaMalloc, false, 0},
    {"cuMemAlloc", s_cuMemAlloc, false, 0},
    {"cudaMallocAsync", s_cudaMallocAsync, false, 0},
    {"cuMemAllocAsync", s_cuMemAllocAsync, false, 0},
    {"cuMemAllocFromPoolAsync", s_cuMemAllocFromPoolAsync, false, 0},
    {"cudaMallocFromPoolAsync", s_cudaMallocFromPoolAsync, false, 0},
    {"cuMemCreateMap", s_cuMemCreateMap, false, 0},
    {"cuMemAllocManaged", s_cuMemAllocManaged, false, 0},
    {"cudaMallocManaged", s_cudaMallocManaged, false, 0},
    {"cudaMallocPitch", s_cudaMallocPitch, false, 0},
    {"cuMemAllocPitch", s_cuMemAllocPitch, false, 0},
    {"cudaMalloc3D", s_cudaMalloc3D, false, 0},
    {"cudaMallocArray", s_cudaMallocArray, false, 0},
    {"cudaMalloc3DArray", s_cudaMalloc3DArray, false, 0},
    {"cudaMallocMipmappedArray", s_cudaMallocMipmappedArray, false, 0},
    {"cuArrayCreate", s_cuArrayCreate, false, 0},
    {"cuArray3DCreate", s_cuArray3DCreate, false, 0},
    {"cuGraphAddMemAllocNode", s_cuGraphAddMemAllocNode, false, 0},
    {"cuMemAllocHost", s_cuMemAllocHost, true, 0},
    {"cudaHostAlloc", s_cudaHostAlloc, true, 0},
    {"cuCtxCreate", s_cuCtxCreate, false, 64 << 20},
};

int main(int argc, char **argv) {
    const char *api = nullptr; long limit_mb = 0, step_mb = 512; int hold = 4;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--api") && i + 1 < argc) api = argv[++i];
        else if (!strcmp(argv[i], "--limit-mb") && i + 1 < argc) limit_mb = atol(argv[++i]);
        else if (!strcmp(argv[i], "--step-mb") && i + 1 < argc) step_mb = atol(argv[++i]);
        else if (!strcmp(argv[i], "--hold") && i + 1 < argc) hold = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--list")) { for (auto &a : APIS) printf("%s%s\n", a.name, a.host ? " (host)" : ""); return 0; }
    }
    if (!api || limit_mb <= 0) { fprintf(stderr, "usage: %s --api NAME --limit-mb L [--step-mb 512] [--hold 4] | --list\n", argv[0]); return 2; }
    Api *A = nullptr; for (auto &a : APIS) if (!strcmp(a.name, api)) A = &a;
    if (!A) { fprintf(stderr, "unknown api %s\n", api); return 2; }

    if (cudaFree(0) != cudaSuccess) { fprintf(stderr, "cuda init failed\n"); return 2; }
    Ctx c; c.step = A->step_override ? A->step_override : (size_t)step_mb << 20;
    cudaStreamCreate(&c.stream);
    size_t free0 = 0, total0 = 0; cudaMemGetInfo(&free0, &total0);

    // Host (pinned) allocations are bounded by the job's RAM cgroup, not the
    // GPU: going 10% past the device limit is enough to prove they are not
    // counted against it.
    size_t cap = A->host ? ((size_t)(limit_mb * 11 / 10) << 20) : ((size_t)(limit_mb * 3 / 2) << 20);
    size_t allocated = 0; int steps = 0; const char *status = "REACHED";
    while (allocated < cap) {
        int k = A->fn(c);
        if (k == 1) { status = "OOM"; break; }
        if (k == 2) { status = "ERROR"; break; }
        allocated += c.step; steps++;
        if (A->name[0] == 'c' && !strcmp(A->name, "cuCtxCreate") && steps >= 8) { status = "REACHED"; break; }
    }
    size_t free1 = 0, total1 = 0; cudaMemGetInfo(&free1, &total1);
    fprintf(stderr, "[pid=%d] %s: %s after %d steps (%zu MiB); view total=%zu MiB free %zu->%zu MiB; holding %ds\n",
            (int)getpid(), api, status, steps, allocated >> 20, total1 >> 20, free0 >> 20, free1 >> 20, hold);
    fflush(stderr);
    sleep(hold);
    printf("BYPASS api=%s pid=%d limit_mb=%ld allocated_mb=%zu steps=%d status=%s host=%d view_total_mb=%zu view_free_mb=%zu err=%s\n",
           api, (int)getpid(), limit_mb, allocated >> 20, steps, status, A->host ? 1 : 0,
           total1 >> 20, free1 >> 20, c.err.empty() ? "-" : c.err.c_str());
    fflush(stdout);
    // Deliberately no cleanup: process exit releases everything; this also
    // exercises SoftMig's exit path with many live allocations.
    return strcmp(status, "ERROR") == 0 ? 1 : 0;
}
