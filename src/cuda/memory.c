/**
 * @file memory.c
 * @brief CUDA memory allocation hooks with OOM enforcement and SM rate limiting.
 *
 * Intercepts cuMemAlloc, cuMemAllocManaged, cuMemCreate, cuMemAllocAsync, and
 * cuMemFree (plus variants) to route allocations through the OOM-checked
 * allocator. Also hooks cuLaunchKernel to inject the SM rate limiter.
 * cuMemGetInfo is overridden to report per-job memory limits.
 */
#include <dirent.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#include "allocator/allocator.h"
#include "include/libcuda_hook.h"
#include "include/libsoftmig.h"
#include "include/memory_limit.h"
#include "multiprocess/multiprocess_memory_limit.h"
// Forward declaration for sum_process_memory_from_nvml
extern uint64_t sum_process_memory_from_nvml(void* device);

extern int pidfound;

// Passive mode (no config file -> limit == 0): memory hooks pass through to
// the real driver with no tracking, OOM checks, or usage accounting. The
// prolog writes the config before the job starts, so the mode cannot change
// during the lifetime of a process. get_current_device_memory_limit()
// returns 0 both when SoftMig is disabled and when there is no shared region.
static inline int softmig_passthrough(CUdevice dev) {
    return get_current_device_memory_limit(dev) == 0;
}

// Guard for memory hooks: on cuCtxGetDevice failure (no context) or in
// passive mode, forward to the real driver call untouched.
#define SOFTMIG_MEM_GUARD(dev, real_fn, ...)                                  \
    CUdevice dev;                                                             \
    if (softmig_is_passive() ||                                               \
        CUDA_OVERRIDE_CALL(cuda_library_entry, cuCtxGetDevice, &dev) != CUDA_SUCCESS || \
        softmig_passthrough(dev)) {                                           \
        return CUDA_OVERRIDE_CALL(cuda_library_entry, real_fn, ##__VA_ARGS__); \
    }

static uint64_t get_current_usage_for_meminfo(CUdevice cuda_dev, unsigned int nvml_dev_idx) {
    uint64_t tracked_usage = get_gpu_memory_usage((int)nvml_dev_idx);
    uint64_t nvml_usage = get_summed_device_memory_usage_from_nvml(cuda_dev);
    return tracked_usage > nvml_usage ? tracked_usage : nvml_usage;
}

const size_t cuarray_format_bytes[33] = {
    0,  // 0x00
    1,  // CU_AD_FORMAT_UNSIGNED_INT8 = 0x01
    2,  // CU_AD_FORMAT_UNSIGNED_INT16 = 0x02
    4,  // CU_AD_FORMAT_UNSIGNED_INT32 = 0x03
    0,  // 0x04
    0,  // 0x05
    0,  // 0x06
    0,  // 0x07
    1,  // CU_AD_FORMAT_SIGNED_INT8 = 0x08
    2,  // CU_AD_FORMAT_SIGNED_INT16 = 0x09
    4,  // CU_AD_FORMAT_SIGNED_INT32 = 0x0a
    0,  // 0x0b
    0,  // 0x0c
    0,  // 0x0d
    0,  // 0x0e
    0,  // 0x0f
    2,  // CU_AD_FORMAT_HALF = 0x10
    0,  // 0x11
    0,  // 0x12
    0,  // 0x13
    0,  // 0x14
    0,  // 0x15
    0,  // 0x16
    0,  // 0x17
    0,  // 0x18
    0,  // 0x19
    0,  // 0x1a
    0,  // 0x1b
    0,  // 0x1c
    0,  // 0x1d
    0,  // 0x1e
    0,  // 0x1f       
    4   // CU_AD_FORMAT_FLOAT = 0x20
};

extern size_t round_up(size_t size,size_t align);
extern void rate_limiter(int grids, int blocks);
static void softmig_throttle_devcopy(size_t bytes);

int check_oom() {
    CUdevice dev;
    CHECK_DRV_API(cuCtxGetDevice(&dev));
    return oom_check(dev,0);
}

uint64_t compute_3d_array_alloc_bytes(const CUDA_ARRAY3D_DESCRIPTOR* desc) {
    if (desc==NULL) {
        LOG_WARN("compute_3d_array_alloc_bytes desc is null");
    }else{
        LOG_DEBUG("compute_3d_array_alloc_bytes height=%ld width=%ld",desc->Height,desc->Width);
    }
    uint64_t bytes = desc->Width * desc->NumChannels;
    if (desc->Height != 0) {
        bytes *= desc->Height;
    }
    if (desc->Depth != 0) {
        bytes *= desc->Depth;
    }
    bytes *= cuarray_format_bytes[desc->Format];

    // TODO: take acount of alignment and etc
    // bytes ++ ???
    return bytes;
}


uint64_t compute_array_alloc_bytes(const CUDA_ARRAY_DESCRIPTOR* desc) {
    if (desc==NULL) {
        LOG_WARN("compute_array_alloc_bytes desc is null");
    }else{
        LOG_DEBUG("compute_array_alloc_bytes height=%ld width=%ld",desc->Height,desc->Width);
    }

    uint64_t bytes = desc->Width * desc->NumChannels;
    if (desc->Height != 0) {
        bytes *= desc->Height;
    }
    bytes *= cuarray_format_bytes[desc->Format];

    // TODO: take acount of alignment and etc
    // bytes ++ ???
    return bytes;
}

/*
 * Arrays are tracked like cuMemCreate handles (reserve -> driver call ->
 * commit, keyed by the CUarray handle). A check-only oom_check() here was not
 * enough: arrays never went through add_chunk, so the NVML-usage side of the
 * check saw a cached value that only tracked allocations refreshed, and a
 * tight cuArrayCreate loop ran 1.5x past the limit (bypass suite, 2.06).
 */
CUresult cuArray3DCreate_v2(CUarray* arr, const CUDA_ARRAY3D_DESCRIPTOR* desc) {
    LOG_DEBUG("cuArray3DCreate_v2");
    uint64_t bytes = compute_3d_array_alloc_bytes(desc);
    ENSURE_RUNNING();
    SOFTMIG_MEM_GUARD(dev, cuArray3DCreate_v2, arr, desc);
    if (softmig_reserve(dev, bytes)) {
        LOG_ERROR("cuArray3DCreate_v2: Device %d OOM (array of %llu bytes)", dev, (unsigned long long)bytes);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuArray3DCreate_v2, arr, desc);
    if (res == CUDA_SUCCESS) {
        add_chunk_only((CUdeviceptr)(uintptr_t)*arr, bytes);
    } else {
        softmig_unreserve(dev, bytes);
    }
    return res;
}


CUresult cuArrayCreate_v2(CUarray* arr, const CUDA_ARRAY_DESCRIPTOR* desc) {
    LOG_DEBUG("cuArrayCreate_v2");
    uint64_t bytes = compute_array_alloc_bytes(desc);
    ENSURE_RUNNING();
    SOFTMIG_MEM_GUARD(dev, cuArrayCreate_v2, arr, desc);
    if (softmig_reserve(dev, bytes)) {
        LOG_ERROR("cuArrayCreate_v2: Device %d OOM (array of %llu bytes)", dev, (unsigned long long)bytes);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuArrayCreate_v2, arr, desc);
    if (res == CUDA_SUCCESS) {
        add_chunk_only((CUdeviceptr)(uintptr_t)*arr, bytes);
    } else {
        softmig_unreserve(dev, bytes);
    }
    return res;
}


CUresult cuArrayDestroy(CUarray arr) {
    LOG_DEBUG("cuArrayDestroy");
    SOFTMIG_MEM_GUARD(dev, cuArrayDestroy, arr);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuArrayDestroy, arr);
    if (res == CUDA_SUCCESS) {
        remove_chunk_only((CUdeviceptr)(uintptr_t)arr);   // -1 for arrays created before tracking: harmless
    }
    return res;
}

CUresult softmig_mem_allocate(CUdeviceptr* dptr, size_t bytesize, void* data) {
    CUresult res;
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAlloc_v2,dptr,bytesize);
    return res;
}

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
    ENSURE_RUNNING();
    SOFTMIG_MEM_GUARD(dev, cuMemAlloc_v2, dptr, bytesize);
    return allocate_raw(dptr,bytesize);
}

CUresult cuMemAllocHost_v2(void** hptr, size_t bytesize) {
    LOG_DEBUG("cuMemAllocHost_v2 hptr=%p bytesize=%ld",hptr,bytesize);
    ENSURE_RUNNING();
    SOFTMIG_MEM_GUARD(dev, cuMemAllocHost_v2, hptr, bytesize);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAllocHost_v2, hptr, bytesize);
    if (res != CUDA_SUCCESS) {
        return res;
    }
    if (check_oom()) {
        CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFreeHost, *hptr);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return res;
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
    LOG_DEBUG("cuMemAllocManaged dptr=%p bytesize=%ld",dptr,bytesize);
    ENSURE_RUNNING();
    SOFTMIG_MEM_GUARD(dev, cuMemAllocManaged, dptr, bytesize, flags);
    if (softmig_reserve(dev,bytesize)){
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAllocManaged, dptr, bytesize, flags);
    if (res == CUDA_SUCCESS) {
        add_chunk_only(*dptr,bytesize);
    } else {
        softmig_unreserve(dev,bytesize);
    }
    return res;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, 
                                      size_t Height, unsigned int ElementSizeBytes) {
    LOG_DEBUG("cuMemAllocPitch_v2 dptr=%p (%ld,%ld)",dptr,WidthInBytes,Height);
    size_t guess_pitch = (((WidthInBytes - 1) / ElementSizeBytes) + 1) * ElementSizeBytes;
    size_t bytesize = guess_pitch * Height;
    ENSURE_RUNNING();
    SOFTMIG_MEM_GUARD(dev, cuMemAllocPitch_v2, dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    if (softmig_reserve(dev,bytesize)){
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAllocPitch_v2, dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    if (res == CUDA_SUCCESS) {
        add_chunk_only(*dptr,bytesize);
    } else {
        softmig_unreserve(dev,bytesize);
    }
    return res;
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    LOG_DEBUG("cuMemFree_v2 dptr=%llx",dptr);
    if (dptr == 0) {  // NULL
        return CUDA_SUCCESS;
    }
    SOFTMIG_MEM_GUARD(dev, cuMemFree_v2, dptr);
    return free_raw(dptr);
}


CUresult cuMemFreeHost(void* hptr) {
    LOG_DEBUG("cuMemFreeHost_v2 hptr=%p",hptr);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFreeHost, hptr);
    return res;
}

CUresult cuMemHostAlloc(void** hptr, size_t bytesize, unsigned int flags) {
    LOG_DEBUG("cuMemHostAlloc hptr=%p bytesize=%lu",hptr,bytesize);
    ENSURE_RUNNING();
    SOFTMIG_MEM_GUARD(dev, cuMemHostAlloc, hptr, bytesize, flags);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemHostAlloc, hptr, bytesize, flags);
    if (res != CUDA_SUCCESS) {
        return res;
    }
    if (check_oom()) {
        CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFreeHost, *hptr);
        *hptr = NULL;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return res;
}


CUresult cuMemHostRegister_v2(void* hptr, size_t bytesize, unsigned int flags) {
    LOG_DEBUG("cuMemHostRegister_v2 hptr=%p bytesize=%ld",hptr,bytesize);
    SOFTMIG_MEM_GUARD(dev, cuMemHostRegister_v2, hptr, bytesize, flags);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemHostRegister_v2, hptr, bytesize, flags);
    LOG_DEBUG("cuMemHostRegister_v2 returned :%d(%p:%ld)",res,hptr,bytesize);
    if (res != CUDA_SUCCESS) {
        return res;
    }
    if (check_oom()) {
        CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemHostUnregister, hptr);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return res;
}


CUresult cuMemHostUnregister(void* hptr) {
    LOG_DEBUG("cuMemHostUnregister hptr=%p",hptr);
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemHostUnregister, hptr);
    return res;
}


CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount ){
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy,dst,src,ByteCount);
    return res;
}

CUresult cuPointerGetAttribute ( void* data, CUpointer_attribute attribute, CUdeviceptr ptr ){
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuPointerGetAttribute,data,attribute,ptr);
    return res;
}

CUresult cuPointerGetAttributes ( unsigned int  numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr ) {
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuPointerGetAttributes,numAttributes,attributes,data,ptr);
    int cur=0;
    for (cur=0;cur<numAttributes;cur++){
        if (attributes[cur]==CU_POINTER_ATTRIBUTE_MEMORY_TYPE){
            int j = check_memory_type(ptr);
            (void)j;
        }else{
            if (attributes[cur]==CU_POINTER_ATTRIBUTE_IS_MANAGED){
                *(int *)(data[cur])=0;    
            }
        }
    }
    return res;
}

CUresult cuPointerSetAttribute ( const void* value, CUpointer_attribute attribute, CUdeviceptr ptr ){
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuPointerSetAttribute,value,attribute,ptr);
    return res;
}


CUresult cuIpcCloseMemHandle(CUdeviceptr dptr){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuIpcCloseMemHandle,dptr);
}

CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) {
    LOG_MSG("cuIpcGetMemHandle dptr=%llx", dptr);
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuIpcGetMemHandle,pHandle,dptr);
}

CUresult cuIpcOpenMemHandle_v2 ( CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int  Flags ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuIpcOpenMemHandle_v2,pdptr,handle,Flags);
}


CUresult cuMemGetAddressRange_v2( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr ){
    //TODO: Translate back
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemGetAddressRange_v2,pbase,psize,dptr);
    return res;
}

CUresult cuMemcpyAsync ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream ){
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyAsync,dst,src,ByteCount,hStream);
    return res; 
}

CUresult cuMemcpyAtoD_v2( CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyAtoD_v2,dstDevice,srcArray,srcOffset,ByteCount);
}

CUresult cuMemcpyDtoA_v2 ( CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyDtoA_v2,dstArray,dstOffset,srcDevice,ByteCount);
}

CUresult cuMemcpyDtoD_v2( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemcpyDtoD_v2,dstDevice,srcDevice,ByteCount);
    ENSURE_RUNNING();
    softmig_throttle_devcopy(ByteCount);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyDtoD_v2,dstDevice,srcDevice,ByteCount);
}

CUresult cuMemcpyDtoDAsync_v2( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemcpyDtoDAsync_v2,dstDevice,srcDevice,ByteCount,hStream);
    ENSURE_RUNNING();
    softmig_throttle_devcopy(ByteCount);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyDtoDAsync_v2,dstDevice,srcDevice,ByteCount,hStream);
}

CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyDtoH_v2, dstHost, srcDevice, ByteCount);
    return res;
}

CUresult cuMemcpyDtoHAsync_v2 ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream ){
    ENSURE_RUNNING();
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyDtoHAsync_v2,dstHost,srcDevice,ByteCount,hStream); 
}


CUresult cuMemcpyHtoD_v2(CUdeviceptr srcDevice, const void* dstHost, size_t ByteCount) {
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyHtoD_v2, srcDevice, dstHost, ByteCount);
    return res;
}

CUresult cuMemcpyHtoDAsync_v2( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream ){
    ENSURE_RUNNING();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyHtoDAsync_v2,dstDevice,srcHost,ByteCount,hStream);
    return res;
}


CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
    SOFTMIG_PASSIVE_FORWARD(cuMemcpyPeer,dstDevice,dstContext,srcDevice,srcContext,ByteCount);
    ENSURE_RUNNING();
    softmig_throttle_devcopy(ByteCount);
    CUresult res=CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyPeer,dstDevice,dstContext,srcDevice,srcContext,ByteCount);
    return res;
}

CUresult cuMemcpyPeerAsync( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) {
    SOFTMIG_PASSIVE_FORWARD(cuMemcpyPeerAsync,dstDevice,dstContext,srcDevice,srcContext,ByteCount,hStream);
    ENSURE_RUNNING();
    softmig_throttle_devcopy(ByteCount);
    CUresult res=CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpyPeerAsync,dstDevice,dstContext,srcDevice,srcContext,ByteCount,hStream);
    return res;
}

CUresult cuMemsetD16_v2( CUdeviceptr dstDevice, unsigned short us, size_t N ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemsetD16_v2,dstDevice,us,N);
    ENSURE_RUNNING();
    softmig_throttle_devcopy(N);
    CUresult res=CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD16_v2,dstDevice,us,N);
    return res;
}

CUresult cuMemsetD16Async( CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemsetD16Async,dstDevice,us,N,hStream);
    ENSURE_RUNNING();
    softmig_throttle_devcopy(N);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD16Async,dstDevice,us,N,hStream);
}

CUresult cuMemsetD2D16_v2( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemsetD2D16_v2,dstDevice,dstPitch,us,Width,Height);
    ENSURE_RUNNING();
    softmig_throttle_devcopy((size_t)dstPitch * Height);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD2D16_v2,dstDevice,dstPitch,us,Width,Height);
}

CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemsetD2D16Async,dstDevice,dstPitch,us,Width,Height,hStream);
    ENSURE_RUNNING();
    softmig_throttle_devcopy((size_t)dstPitch * Height);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD2D16Async,dstDevice,dstPitch,us,Width,Height,hStream);
}

CUresult cuMemsetD2D32_v2( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemsetD2D32_v2,dstDevice,dstPitch,ui,Width,Height);
    ENSURE_RUNNING();
    softmig_throttle_devcopy((size_t)dstPitch * Height);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD2D32_v2,dstDevice,dstPitch,ui,Width,Height);
}


CUresult cuMemsetD2D32Async( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height, CUstream hStream ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemsetD2D32Async,dstDevice,dstPitch,ui,Width,Height,hStream);
    ENSURE_RUNNING();
    softmig_throttle_devcopy((size_t)dstPitch * Height);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD2D32Async,dstDevice,dstPitch,ui,Width,Height,hStream);
}

CUresult cuMemsetD2D8_v2( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemsetD2D8_v2,dstDevice,dstPitch,uc,Width,Height);
    ENSURE_RUNNING();
    softmig_throttle_devcopy((size_t)dstPitch * Height);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD2D8_v2,dstDevice,dstPitch,uc,Width,Height);
}

CUresult cuMemsetD2D8Async( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height, CUstream hStream ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemsetD2D8Async,dstDevice,dstPitch,uc,Width,Height,hStream);
    ENSURE_RUNNING();
    softmig_throttle_devcopy((size_t)dstPitch * Height);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD2D8Async,dstDevice,dstPitch,uc,Width,Height,hStream);
}

CUresult cuMemsetD32_v2( CUdeviceptr dstDevice, unsigned int  ui, size_t N ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemsetD32_v2,dstDevice,ui,N);
    ENSURE_RUNNING();
    softmig_throttle_devcopy(N);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD32_v2,dstDevice,ui,N);
    return res;
}

CUresult cuMemsetD32Async( CUdeviceptr dstDevice, unsigned int  ui, size_t N, CUstream hStream ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemsetD32Async,dstDevice,ui,N,hStream);
    ENSURE_RUNNING();
    softmig_throttle_devcopy(N);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD32Async,dstDevice,ui,N,hStream);
}   


CUresult cuMemsetD8_v2( CUdeviceptr dstDevice, unsigned char  uc, size_t N ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemsetD8_v2,dstDevice,uc,N);
    ENSURE_RUNNING();
    softmig_throttle_devcopy(N);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD8_v2,dstDevice,uc,N);
}

CUresult cuMemsetD8Async( CUdeviceptr dstDevice, unsigned char  uc, size_t N, CUstream hStream ) {
    SOFTMIG_PASSIVE_FORWARD(cuMemsetD8Async,dstDevice,uc,N,hStream);
    ENSURE_RUNNING();
    softmig_throttle_devcopy(N);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemsetD8Async,dstDevice,uc,N,hStream);
}

// cuMemAdvise is a pure pass-through with no SoftMig-specific logic, so
// we deliberately do NOT hook it. dlsym will resolve it directly from libcuda
// for both CUDA 12 (cuMemAdvise) and CUDA 13 (cuMemAdvise_v2 with a different
// signature).

#ifdef HOOK_MEMINFO_ENABLE
// Report free/total against the per-job limit. Limits and tracked usage are
// keyed by CUDA device index (same as oom_check / add_chunk). Over-limit
// usage reports free=0 rather than an error, matching a full real GPU.
static CUresult softmig_meminfo(CUresult real_res, CUdevice dev, size_t* free, size_t* total) {
    if (real_res != CUDA_SUCCESS) {
        return real_res;
    }
    size_t limit = get_current_device_memory_limit(dev);
    if (limit == 0) {
        return CUDA_SUCCESS;
    }
    uint64_t usage = get_current_usage_for_meminfo(dev, cuda_to_nvml_map(dev));
    size_t actual_limit = (limit > *total) ? *total : limit;
    if (usage > actual_limit) {
        LOG_WARN("cuMemGetInfo: usage %lu exceeds limit %lu, reporting free=0", usage, actual_limit);
    }
    *free = (actual_limit > usage) ? (actual_limit - usage) : 0;
    *total = actual_limit;
    return CUDA_SUCCESS;
}

#undef cuMemGetInfo
FUNC_ATTR_VISIBLE CUresult cuMemGetInfo(size_t* free, size_t* total) {
    if (CUDA_FIND_ENTRY(cuda_library_entry, cuMemGetInfo) == NULL) {
        return cuMemGetInfo_v2(free, total);
    }
    SOFTMIG_MEM_GUARD(dev, cuMemGetInfo, free, total);
    LOG_DEBUG("cuMemGetInfo");
    return softmig_meminfo(CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemGetInfo, free, total),
                           dev, free, total);
}

#undef cuMemGetInfo_v2
FUNC_ATTR_VISIBLE CUresult cuMemGetInfo_v2(size_t* free, size_t* total) {
    SOFTMIG_MEM_GUARD(dev, cuMemGetInfo_v2, free, total);
    LOG_DEBUG("cuMemGetInfo_v2");
    return softmig_meminfo(CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemGetInfo_v2, free, total),
                           dev, free, total);
}
#endif

CUresult cuMipmappedArrayCreate(CUmipmappedArray* pHandle, 
                                          const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, 
                                          unsigned int numMipmapLevels) {
    LOG_DEBUG("cuMipmappedArrayCreate\n");
    ENSURE_RUNNING();
    SOFTMIG_MEM_GUARD(dev, cuMipmappedArrayCreate, pHandle, pMipmappedArrayDesc, numMipmapLevels);
    // Level 0 plus the geometric tail of the mip chain (< 1/7 of level 0 for 3D,
    // < 1/3 for 2D); use 4/3 as a conservative upper bound.
    uint64_t bytes = compute_3d_array_alloc_bytes(pMipmappedArrayDesc);
    if (numMipmapLevels > 1) bytes += bytes / 3;
    if (softmig_reserve(dev, bytes)) {
        LOG_ERROR("cuMipmappedArrayCreate: Device %d OOM (array of %llu bytes)", dev, (unsigned long long)bytes);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMipmappedArrayCreate, pHandle, pMipmappedArrayDesc, numMipmapLevels);
    if (res == CUDA_SUCCESS) {
        add_chunk_only((CUdeviceptr)(uintptr_t)*pHandle, bytes);
    } else {
        softmig_unreserve(dev, bytes);
    }
    return res;
}

CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
    LOG_DEBUG("cuMipmappedArrayDestroy\n");
    SOFTMIG_MEM_GUARD(dev, cuMipmappedArrayDestroy, hMipmappedArray);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMipmappedArrayDestroy, hMipmappedArray);
    if (res == CUDA_SUCCESS) {
        remove_chunk_only((CUdeviceptr)(uintptr_t)hMipmappedArray);
    }
    return res;
}

static inline void softmig_before_launch(unsigned int gx, unsigned int gy, unsigned int gz,
                                         unsigned int bx, unsigned int by, unsigned int bz) {
    ENSURE_RUNNING();
    pre_launch_kernel();
    if (pidfound==1){
        rate_limiter(gx * gy * gz, bx * by * bz);
    }
}

/* Device-side memsets and device-to-device copies run as driver-internal
 * kernels that never pass through cuLaunchKernel, so a job made of them
 * (allocation storms zeroing buffers, memset loops) escaped the SM limit.
 * Charge them like a launch: tokens by size (1 per MiB) plus the duty delay. */
static void softmig_throttle_devcopy(size_t bytes) {
    if (softmig_is_passive()) return;
    if (pidfound == 1) {
        rate_limiter((int)((bytes >> 20) + 1), 0);
    }
}

CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra ){
    SOFTMIG_PASSIVE_FORWARD(cuLaunchKernel,f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,hStream,kernelParams,extra);
    softmig_before_launch(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuLaunchKernel,f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,hStream,kernelParams,extra);
}

CUresult cuLaunchKernel_ptsz ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra ){
    return cuLaunchKernel(f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,SOFTMIG_PTSZ_STREAM(hStream),kernelParams,extra);
}

CUresult cuLaunchCooperativeKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams ){
    SOFTMIG_PASSIVE_FORWARD(cuLaunchCooperativeKernel,f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,hStream,kernelParams);
    softmig_before_launch(gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuLaunchCooperativeKernel,f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,hStream,kernelParams);
}

CUresult cuLaunchCooperativeKernel_ptsz ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams ){
    return cuLaunchCooperativeKernel(f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,SOFTMIG_PTSZ_STREAM(hStream),kernelParams);
}

CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra) {
    SOFTMIG_PASSIVE_FORWARD(cuLaunchKernelEx, config, f, kernelParams, extra);
    if (config != NULL) {
        softmig_before_launch(config->gridDimX, config->gridDimY, config->gridDimZ,
                              config->blockDimX, config->blockDimY, config->blockDimZ);
    }
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuLaunchKernelEx, config, f, kernelParams, extra);
}

CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra) {
    if (config == NULL || config->hStream != NULL) {
        return cuLaunchKernelEx(config, f, kernelParams, extra);
    }
    CUlaunchConfig c = *config;
    c.hStream = CU_STREAM_PER_THREAD;
    return cuLaunchKernelEx(&c, f, kernelParams, extra);
}

CUresult softmig_mem_free(CUdeviceptr dptr) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFree_v2,dptr);
    return res;
}

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size,
    size_t alignment, CUdeviceptr addr, unsigned long long flags ) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,
        cuMemAddressReserve, ptr, size, alignment, addr, flags);
    return res;
}

CUresult cuMemCreate ( CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags ) {
    ENSURE_RUNNING();
    SOFTMIG_MEM_GUARD(dev, cuMemCreate, handle, size, prop, flags);
    if (softmig_reserve(dev, size)) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,
        cuMemCreate, handle, size, prop, flags);
    if (res == CUDA_SUCCESS) {
        add_chunk_only(*handle, size);
    } else {
        softmig_unreserve(dev, size);
    }
    return res;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    SOFTMIG_MEM_GUARD(dev, cuMemRelease, handle);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemRelease, handle);
    if (res == CUDA_SUCCESS) {
        remove_chunk_only(handle);
    }
    return res;
}

CUresult cuMemMap( CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags ) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemMap,ptr,size,offset,handle,flags);
    return res;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle,
    void* osHandle, CUmemAllocationHandleType shHandleType) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,
        cuMemImportFromShareableHandle, handle, osHandle, shHandleType);
    return res;
}

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
    SOFTMIG_MEM_GUARD(dev, cuMemAllocAsync, dptr, bytesize, hStream);
    LOG_DEBUG("cuMemAllocAsync:%ld",bytesize);
    return allocate_async_raw(dptr,bytesize,hStream);
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    LOG_DEBUG("cuMemFreeAsync dptr=%llx",dptr);
    if (dptr == 0) {  // NULL
        return CUDA_SUCCESS;
    }
    SOFTMIG_MEM_GUARD(dev, cuMemFreeAsync, dptr, hStream);
    CUresult res = free_raw_async(dptr,hStream);
    LOG_DEBUG("after free_raw_async dptr=%p res=%d",(void *)dptr,res);
    return res;
}

CUresult cuMemAllocAsync_ptsz(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
    return cuMemAllocAsync(dptr, bytesize, SOFTMIG_PTSZ_STREAM(hStream));
}

CUresult cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream hStream) {
    return cuMemFreeAsync(dptr, SOFTMIG_PTSZ_STREAM(hStream));
}

CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags){
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemHostGetDevicePointer_v2,pdptr,p,Flags);
}

CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p){
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemHostGetFlags,pFlags,p);
}

CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep){
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolTrimTo,pool,minBytesToKeep);
}

CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolSetAttribute,pool,attr,value);
}

CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolGetAttribute,pool,attr,value);
    return res;
}

CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolSetAccess,pool,map,count);
}

CUresult cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool, CUmemLocation *location) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolGetAccess,flags,memPool,location);
}

CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolCreate,pool,poolProps);
}

CUresult cuMemPoolDestroy(CUmemoryPool pool) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolDestroy,pool);
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) {
    SOFTMIG_MEM_GUARD(dev, cuMemAllocFromPoolAsync, dptr, bytesize, pool, hStream);
    if (softmig_reserve(dev, bytesize)) {
        LOG_ERROR("cuMemAllocFromPoolAsync: Device %d OOM (requested %lu bytes)", dev, bytesize);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAllocFromPoolAsync,dptr,bytesize,pool,hStream);
    if (res == CUDA_SUCCESS) {
        add_chunk_async_only(*dptr, bytesize);
    } else {
        softmig_unreserve(dev, bytesize);
    }
    return res;
}

/*
 * Graph memory nodes allocate when the graph is launched, from a graph pool
 * the driver manages. The node is reserved and then tracked by the virtual
 * address the driver assigns (nodeParams->dptr), so the limit check counts it
 * before the graph ever runs; a plain check-only version let a loop of alloc
 * nodes run ~1.2 GiB past the limit (bypass suite, 2.06) because nothing
 * refreshed the usage between nodes. The address is released from tracking by
 * cuMemFree(dptr) (tracked path) or a cuGraphAddMemFreeNode for it. Memory
 * freed only by destroying the executable graph stays counted until the
 * process exits: conservative, never under-counts. Allocation nodes created
 * implicitly by stream capture (cudaMallocAsync while capturing) do not pass
 * through here.
 */
CUresult cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
                                size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams) {
    ENSURE_RUNNING();
    SOFTMIG_MEM_GUARD(dev, cuGraphAddMemAllocNode, phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    size_t bytes = nodeParams ? nodeParams->bytesize : 0;
    if (bytes && softmig_reserve(dev, bytes)) {
        LOG_ERROR("cuGraphAddMemAllocNode: Device %d OOM (node of %zu bytes)", dev, bytes);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphAddMemAllocNode, phGraphNode, hGraph, dependencies,
                                      numDependencies, nodeParams);
    if (bytes) {
        if (res == CUDA_SUCCESS && nodeParams->dptr) {
            add_chunk_only(nodeParams->dptr, bytes);
        } else {
            softmig_unreserve(dev, bytes);
        }
    }
    return res;
}

CUresult cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
                               size_t numDependencies, CUdeviceptr dptr) {
    SOFTMIG_MEM_GUARD(dev, cuGraphAddMemFreeNode, phGraphNode, hGraph, dependencies, numDependencies, dptr);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGraphAddMemFreeNode, phGraphNode, hGraph, dependencies,
                                      numDependencies, dptr);
    if (res == CUDA_SUCCESS) {
        remove_chunk_only(dptr);
    }
    return res;
}

CUresult cuMemAllocFromPoolAsync_ptsz(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream) {
    return cuMemAllocFromPoolAsync(dptr, bytesize, pool, SOFTMIG_PTSZ_STREAM(hStream));
}

CUresult cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolExportToShareableHandle,handle_out,pool,handleType,flags);
}

CUresult cuMemPoolImportFromShareableHandle(
        CUmemoryPool *pool_out,
        void *handle,
        CUmemAllocationHandleType handleType,
        unsigned long long flags) {
            return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolImportFromShareableHandle,pool_out,handle,handleType,flags);
        }

CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out, CUdeviceptr ptr) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolExportPointer,shareData_out,ptr);
}

CUresult cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData *shareData) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolImportPointer,ptr_out,pool,shareData);
}
CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuMemcpy2D_v2, pCopy);
}
CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy2DUnaligned_v2,pCopy);
}
CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy2DAsync,pCopy,hStream);
}

CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy3D_v2,pCopy);
}
CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy3DAsync_v2,pCopy,hStream);
}

CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy3DPeer,pCopy);
}

CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemcpy3DPeerAsync,pCopy,hStream);
}

// cuMemPrefetchAsync is a pure pass-through with no SoftMig-specific logic, so
// we deliberately do NOT hook it. dlsym will resolve it directly from libcuda
// for both CUDA 12 (cuMemPrefetchAsync) and CUDA 13 (cuMemPrefetchAsync_v2 with
// a different signature).

CUresult cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemRangeGetAttribute,data,dataSize,attribute,devPtr,count);
}

CUresult cuMemRangeGetAttributes(void **data, size_t *dataSizes, CUmem_range_attribute *attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemRangeGetAttributes,data,dataSizes,attributes,numAttributes,devPtr,count);
}

/* External Resource Management */
CUresult cuImportExternalMemory(CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuImportExternalMemory,extMem_out,memHandleDesc);
}

CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuExternalMemoryGetMappedBuffer,devPtr,extMem,bufferDesc);
}

CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuExternalMemoryGetMappedMipmappedArray,mipmap,extMem,mipmapDesc);
}

CUresult cuDestroyExternalMemory(CUexternalMemory extMem) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDestroyExternalMemory,extMem);
}

CUresult cuImportExternalSemaphore(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuImportExternalSemaphore,extSem_out,semHandleDesc);
}

CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuSignalExternalSemaphoresAsync,extSemArray,paramsArray,numExtSems,stream);
}

CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuWaitExternalSemaphoresAsync,extSemArray,paramsArray,numExtSems,stream);
}

CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDestroyExternalSemaphore,extSem);
}
