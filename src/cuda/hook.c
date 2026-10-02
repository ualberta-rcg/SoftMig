/**
 * @file hook.c (cuda)
 * @brief CUDA driver library loading, dispatch table, and cuGetProcAddress hook.
 *
 * Populates the cuda_library_entry[] dispatch table by dlopen-ing libcuda.so.1.
 * Hooks cuGetProcAddress / cuGetProcAddress_v2 so that CUDA runtime calls are
 * redirected through SoftMig's hooked functions (memory allocation, kernel
 * launch rate limiting, etc.).
 */
#include "include/libcuda_hook.h"
#include <string.h>
#include "include/libsoftmig.h"
#include "include/dlsym_resolve.h"


typedef void* (*fp_dlsym)(void*, const char*);
extern fp_dlsym real_dlsym;

cuda_entry_t cuda_library_entry[] = {
    /* Init Part    */ 
    {.name = "cuInit"},
    /* Deivce Part */
    {.name = "cuDeviceGetAttribute"},
    {.name = "cuDeviceGet"},
    {.name = "cuDeviceGetCount"},
    {.name = "cuDeviceGetName"},
    {.name = "cuDeviceCanAccessPeer"},
    {.name = "cuDeviceGetP2PAttribute"},
    {.name = "cuDeviceGetByPCIBusId"},
    {.name = "cuDeviceGetPCIBusId"},
    {.name = "cuDeviceGetDefaultMemPool"},
    {.name = "cuDeviceGetLuid"},
    {.name = "cuDeviceGetMemPool"},
    {.name = "cuDeviceTotalMem_v2"},
    {.name = "cuDriverGetVersion"},
    {.name = "cuDeviceGetTexture1DLinearMaxWidth"},
    {.name = "cuDeviceSetMemPool"},
    {.name = "cuFlushGPUDirectRDMAWrites"},

    /* Context Part */
    {.name = "cuDevicePrimaryCtxGetState"},
    {.name = "cuDevicePrimaryCtxRetain"},
    {.name = "cuDevicePrimaryCtxSetFlags_v2"},
    {.name = "cuDevicePrimaryCtxRelease_v2"},
    {.name = "cuCtxGetDevice"},
    {.name = "cuCtxDestroy_v2"},
    {.name = "cuCtxGetApiVersion"},
    {.name = "cuCtxGetCacheConfig"},
    {.name = "cuCtxGetCurrent"},
    {.name = "cuCtxGetFlags"},
    {.name = "cuCtxGetLimit"},
    {.name = "cuCtxGetSharedMemConfig"},
    {.name = "cuCtxGetStreamPriorityRange"},
    {.name = "cuCtxPopCurrent_v2"},
    {.name = "cuCtxPushCurrent_v2"},
    {.name = "cuCtxSetCacheConfig"},
    {.name = "cuCtxSetCurrent"},
    {.name = "cuCtxSetLimit"},
    {.name = "cuCtxSetSharedMemConfig"},
    {.name = "cuCtxSynchronize"},
    //{.name = "cuCtxEnablePeerAccess"},
    {.name = "cuGetExportTable"},
    /* Stream Part */
    {.name = "cuStreamCreate"},
    {.name = "cuStreamDestroy_v2"},
    {.name = "cuStreamSynchronize"},
    /* Memory Part */
    {.name = "cuArray3DCreate_v2"},
    {.name = "cuArrayCreate_v2"},
    {.name = "cuArrayDestroy"},
    {.name = "cuMemAlloc_v2"},
    {.name = "cuMemAllocHost_v2"},
    {.name = "cuMemAllocManaged"},
    {.name = "cuMemAllocPitch_v2"},
    {.name = "cuMemFree_v2"},
    {.name = "cuMemFreeHost"},
    {.name = "cuMemHostAlloc"},
    {.name = "cuMemHostRegister_v2"},
    {.name = "cuMemHostUnregister"},
    {.name = "cuMemcpyDtoH_v2"},
    {.name = "cuMemcpyHtoD_v2"},
    {.name = "cuMipmappedArrayCreate"},
    {.name = "cuMipmappedArrayDestroy"},
    {.name = "cuMemGetInfo"},
    {.name = "cuMemGetInfo_v2"},
    {.name = "cuMemcpy"},
    {.name = "cuPointerGetAttribute"},
    {.name = "cuPointerGetAttributes"},
    {.name = "cuPointerSetAttribute"},
    {.name = "cuIpcCloseMemHandle"},
    {.name = "cuIpcGetMemHandle"},
    {.name = "cuIpcOpenMemHandle_v2"},
    {.name = "cuMemGetAddressRange_v2"},
    {.name = "cuMemcpyAsync"},
    {.name = "cuMemcpyAtoD_v2"},
    {.name = "cuMemcpyDtoA_v2"},
    {.name = "cuMemcpyDtoD_v2"},
    {.name = "cuMemcpyDtoDAsync_v2"},
    {.name = "cuMemcpyDtoHAsync_v2"},
    {.name = "cuMemcpyHtoDAsync_v2"},
    {.name = "cuMemcpyPeer"},
    {.name = "cuMemcpyPeerAsync"},
    {.name = "cuMemsetD16_v2"},
    {.name = "cuMemsetD16Async"},
    {.name = "cuMemsetD2D16_v2"},
    {.name = "cuMemsetD2D16Async"},
    {.name = "cuMemsetD2D32_v2"},
    {.name = "cuMemsetD2D32Async"},
    {.name = "cuMemsetD2D8_v2"},
    {.name = "cuMemsetD2D8Async"},
    {.name = "cuMemsetD32_v2"},
    {.name = "cuMemsetD32Async"},
    {.name = "cuMemsetD8_v2"},
    {.name = "cuMemsetD8Async"},
    {.name = "cuFuncSetCacheConfig"},
    {.name = "cuFuncSetSharedMemConfig"},
    {.name = "cuFuncGetAttribute"},
    {.name = "cuFuncSetAttribute"},
    {.name = "cuLaunchKernel"},
    {.name = "cuLaunchCooperativeKernel"},
    /* cuEvent Part */
    {.name = "cuEventCreate"},
    {.name = "cuEventDestroy_v2"},
    {.name = "cuModuleLoad"},
    {.name = "cuModuleLoadData"},
    {.name = "cuModuleLoadDataEx"},
    {.name = "cuModuleLoadFatBinary"},
    {.name = "cuModuleGetFunction"},
    {.name = "cuModuleUnload"},
    {.name = "cuModuleGetGlobal_v2"},
    {.name = "cuModuleGetTexRef"},
    {.name = "cuModuleGetSurfRef"},
    {.name = "cuLinkAddData_v2"},
    {.name = "cuLinkCreate_v2"},
    {.name = "cuLinkAddFile_v2"},
    {.name = "cuLinkComplete"},
    {.name = "cuLinkDestroy"},
    /* Virtual Memory Part */
    {.name = "cuMemAddressReserve"},
    {.name = "cuMemCreate"},
    {.name = "cuMemRelease"},
    {.name = "cuMemMap"},
    {.name = "cuMemImportFromShareableHandle"},
    {.name = "cuMemAllocAsync"},
    {.name = "cuMemFreeAsync"},
    /* cuda11.7 new api memory part */
    {.name = "cuMemHostGetDevicePointer_v2"},
    {.name = "cuMemHostGetFlags"},
    {.name = "cuMemPoolTrimTo"},
    {.name = "cuMemPoolSetAttribute"},
    {.name = "cuMemPoolGetAttribute"},
    {.name = "cuMemPoolSetAccess"},
    {.name = "cuMemPoolGetAccess"},
    {.name = "cuMemPoolCreate"},
    {.name = "cuMemPoolDestroy"},
    {.name = "cuMemAllocFromPoolAsync"},
    {.name = "cuMemPoolExportToShareableHandle"},
    {.name = "cuMemPoolImportFromShareableHandle"},
    {.name = "cuMemPoolExportPointer"},
    {.name = "cuMemPoolImportPointer"},
    {.name = "cuMemcpy2DUnaligned_v2"},
    {.name = "cuMemcpy2D_v2"},
    {.name = "cuMemcpy2DAsync_v2"},
    {.name = "cuMemcpy3D_v2"},
    {.name = "cuMemcpy3DAsync_v2"},
    {.name = "cuMemcpy3DPeer"},
    {.name = "cuMemcpy3DPeerAsync"},
    {.name = "cuMemRangeGetAttribute"},
    {.name = "cuMemRangeGetAttributes"},
    /* cuda 11.7 external resource interoperability */
    {.name = "cuImportExternalMemory"},
    {.name = "cuExternalMemoryGetMappedBuffer"},
    {.name = "cuExternalMemoryGetMappedMipmappedArray"},
    {.name = "cuDestroyExternalMemory"},
    {.name = "cuImportExternalSemaphore"},
    {.name = "cuSignalExternalSemaphoresAsync"},
    {.name = "cuWaitExternalSemaphoresAsync"},
    {.name = "cuDestroyExternalSemaphore"},
    /* Graph part - only hook cuGraphLaunch (rate limiter). */
    {.name = "cuGraphLaunch"},

    {.name = "cuGetProcAddress"},
    {.name = "cuGetProcAddress_v2"},

    {.name = "cuLaunchKernelEx"},
    {.name = "cuLaunchKernel_ptsz"},
    {.name = "cuLaunchKernelEx_ptsz"},
    {.name = "cuLaunchCooperativeKernel_ptsz"},
    {.name = "cuGraphLaunch_ptsz"},
    {.name = "cuMemAllocAsync_ptsz"},
    {.name = "cuMemFreeAsync_ptsz"},
    {.name = "cuMemAllocFromPoolAsync_ptsz"},
    {.name = "cuGraphAddMemAllocNode"},
};

_Static_assert(sizeof(cuda_library_entry) / sizeof(cuda_library_entry[0]) == CUDA_ENTRY_END,
               "cuda_library_entry[] must match cuda_override_enum_t entry for entry");

int prior_function(char tmp[500]) {
    char *pos = tmp + strlen(tmp) - 3;
    if (pos[0]=='_' && pos[1]=='v') {
        if (pos[2]=='2')
            pos[0]='\0';
        else
            pos[2]--;
        return 1;
    }
    return 0;
}

/* 1 if cuda_library_entry[i] was resolved under its own name (not via the
 * prior-version fallback), so its real pointer has exactly that entry's ABI. */
static unsigned char cuda_entry_exact[CUDA_ENTRY_END];

/** Resolve all CUDA driver symbols from libcuda.so.1 into cuda_library_entry[]. */
void load_cuda_libraries() {
    void *table = NULL;
    int i = 0;
    char cuda_filename[FILENAME_MAX];
    char tmpfunc[500];

    snprintf(cuda_filename, FILENAME_MAX - 1, "%s","libcuda.so.1");
    cuda_filename[FILENAME_MAX - 1] = '\0';

    table = dlopen(cuda_filename, RTLD_NOW | RTLD_NODELETE);
    if (!table) {
        LOG_WARN("can't find library %s", cuda_filename);
    }

    for (i = 0; i < CUDA_ENTRY_END; i++) {
        // Never look up with a NULL handle: RTLD_DEFAULT would find our own
        // exported wrapper and every call through the table would recurse.
        cuda_library_entry[i].fn_ptr = table ? real_dlsym(table, cuda_library_entry[i].name) : NULL;
        if (!cuda_library_entry[i].fn_ptr) {
            cuda_library_entry[i].fn_ptr=real_dlsym(RTLD_NEXT,cuda_library_entry[i].name);
        }
        if (cuda_library_entry[i].fn_ptr) {
            cuda_entry_exact[i] = 1;
            continue;
        }
        LOG_DEBUG("can't find function %s in %s", cuda_library_entry[i].name,cuda_filename);
        memset(tmpfunc,0,500);
        strcpy(tmpfunc,cuda_library_entry[i].name);
        while (prior_function(tmpfunc)) {
            cuda_library_entry[i].fn_ptr=real_dlsym(RTLD_NEXT,tmpfunc);
            if (cuda_library_entry[i].fn_ptr) {
                LOG_INFO("found prior function %s",tmpfunc);
                break;
            }
        }
    }
    if (cuda_library_entry[0].fn_ptr==NULL){
        LOG_WARN("is NULL");
    }
    if (table) {
        dlclose(table);
    }
}

volatile int softmig_cuda_table_ready = 0;
static pthread_once_t cuda_table_once = PTHREAD_ONCE_INIT;

static void cuda_table_load_once(void) {
    if (real_dlsym == NULL) {
        real_dlsym = resolve_real_dlsym();
    }
    load_cuda_libraries();
    __sync_synchronize();
    softmig_cuda_table_ready = 1;
}

/** Populate cuda_library_entry[] exactly once (safe from any entry point). */
void softmig_ensure_cuda_table(void) {
    pthread_once(&cuda_table_once, cuda_table_load_once);
}

/*
 * Runtime hook audit. Enforcement only works if every allocation, free,
 * launch and memory-report entry point the app actually uses is hooked. When
 * a lookup for one of those resolves to the raw driver function instead, log
 * it (file only) so smoke tests and admins can spot driver/toolkit drift.
 */
static const char *const unhooked_watch[] = {
    "cuMemAlloc", "cuMemCreate", "cuMemFree", "cuArrayCreate", "cuArray3DCreate",
    "cuMipmappedArrayCreate", "cuLaunch", "cuGraphLaunch", "cuGraphAddMemAllocNode",
    "cuMemGetInfo", "cuDeviceTotalMem",
    "nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses",
    "nvmlDeviceGetMPSComputeRunningProcesses", "nvmlDeviceGetMemoryInfo",
    "nvmlDeviceGetProcessUtilization", "nvmlDeviceGetProcessesUtilizationInfo",
    "nvmlDeviceGetRunningProcessDetailList",
    NULL};
/* Known, documented gaps (see docs/TROUBLESHOOTING.md): host-side callbacks,
 * the deprecated multi-device cooperative launch, graph memory nodes, and
 * pinned host allocations (not device memory). */
static const char *const unhooked_ack[] = {
    "cuLaunchHostFunc", "cuLaunchCooperativeKernelMultiDevice",
    "cuMemAllocHost", "cuMemFreeHost", "cuMemAllocManaged_ptsz",
    NULL};
/* Exact names: pre-CUDA-3.2 ABI (cuda.h maps these to _v2 since 3.2, and
 * cuGetProcAddress only returns them for cudaVersion < 3020) and the legacy
 * cuFuncSetBlockShape-era launch calls. */
static const char *const unhooked_legacy[] = {
    "cuMemAlloc", "cuMemFree", "cuMemAllocPitch", "cuArrayCreate", "cuArray3DCreate",
    "cuDeviceTotalMem", "cuMemGetInfo", "cuLaunch", "cuLaunchGrid", "cuLaunchGridAsync",
    NULL};

static int has_prefix_in(const char *symbol, const char *const *list) {
    for (int i = 0; list[i]; i++) {
        if (strncmp(symbol, list[i], strlen(list[i])) == 0) {
            return 1;
        }
    }
    return 0;
}

static int has_exact_in(const char *symbol, const char *const *list) {
    for (int i = 0; list[i]; i++) {
        if (strcmp(symbol, list[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

void softmig_note_unhooked(const char *symbol, const char *via) {
    if (symbol == NULL || !has_prefix_in(symbol, unhooked_watch) || has_prefix_in(symbol, unhooked_ack) ||
        has_exact_in(symbol, unhooked_legacy)) {
        return;
    }
    log_to_file_only("UNHOOKED", "%s resolved to the raw driver via %s", symbol, via);
}

/*
 * cuGetProcAddress: ask the real driver first, so the returned pointer has
 * the ABI the caller asked for (cudaVersion) and the right default-stream
 * semantics (flags, e.g. PER_THREAD_DEFAULT_STREAM -> *_ptsz). Then swap in
 * our hook only if that exact driver function is one we hook. No guessing of
 * _v2/_v3 names, so a hook can never be handed out with the wrong signature.
 */
static void *softmig_gpa_hook_for(void *real) {
    if (real == NULL) {
        return NULL;
    }
    for (int i = 0; i < CUDA_ENTRY_END; i++) {
        if (cuda_entry_exact[i] && cuda_library_entry[i].fn_ptr == real) {
            void *hook = __dlsym_hook_section(NULL, cuda_library_entry[i].name);
            if (hook != NULL) {
                return hook;
            }
        }
    }
    return NULL;
}

static CUresult softmig_gpa_finish(const char *symbol, void **pfn, CUresult res) {
    if (res != CUDA_SUCCESS || pfn == NULL || *pfn == NULL) {
        return res;
    }
    void *hook = softmig_gpa_hook_for(*pfn);
    if (hook != NULL) {
        LOG_DEBUG("cuGetProcAddress: %s -> hook", symbol);
        *pfn = hook;
    } else {
        softmig_note_unhooked(symbol, "cuGetProcAddress");
    }
    return res;
}

CUresult cuGetProcAddress ( const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags ) {
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuGetProcAddress) == NULL) {
        return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, NULL);
    }
    SOFTMIG_PASSIVE_FORWARD(cuGetProcAddress, symbol, pfn, cudaVersion, flags);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGetProcAddress, symbol, pfn, cudaVersion, flags);
    return softmig_gpa_finish(symbol, pfn, res);
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus){
    SOFTMIG_PASSIVE_FORWARD(cuGetProcAddress_v2, symbol, pfn, cudaVersion, flags, symbolStatus);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGetProcAddress_v2, symbol, pfn, cudaVersion, flags, symbolStatus);
    return softmig_gpa_finish(symbol, pfn, res);
}
