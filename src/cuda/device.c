/**
 * @file device.c
 * @brief CUDA device hooks with memory limit enforcement on cuDeviceTotalMem.
 *
 * Intercepts cuDeviceTotalMem_v2 to report the per-device memory limit
 * instead of the physical GPU memory. All other device query functions
 * are thin pass-through wrappers.
 */
#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"
#include "include/nvml_prefix.h"
#include "include/libnvml_hook.h"

#include "allocator/allocator.h"
#include "include/memory_limit.h"

CUresult CUDAAPI cuDeviceGetAttribute ( int* pi, CUdevice_attribute attrib, CUdevice dev ) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetAttribute,pi,attrib,dev);
    //LOG_DEBUG("[%d]cuDeviceGetAttribute dev=%d attrib=%d %d",res,dev,(int)attrib,*pi);
    return res;
}

CUresult cuDeviceGet(CUdevice *device,int ordinal){
    LOG_DEBUG("into cuDeviceGet ordinal=%d\n",ordinal);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGet,device,ordinal);
    return res;
}

CUresult cuDeviceGetCount( int* count ) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetCount,count);
    return res;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetName, name, len, dev);
    return res;
}

CUresult cuDeviceCanAccessPeer( int* canAccessPeer, CUdevice dev, CUdevice peerDev ) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceCanAccessPeer,canAccessPeer,dev,peerDev);
}

CUresult cuDeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib,
                                 CUdevice srcDevice, CUdevice dstDevice) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetP2PAttribute, value,
                         attrib, srcDevice, dstDevice);
}

CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetByPCIBusId, dev,
                         pciBusId);
}

CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetPCIBusId, pciBusId, len,
                        dev);
    return res;
}

// cuDeviceGetUuid is a pure pass-through with no SoftMig-specific logic, so
// we deliberately do NOT hook it. dlsym will resolve it directly from libcuda
// for both CUDA 12 (cuDeviceGetUuid) and CUDA 13 (cuDeviceGetUuid_v2).

CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetDefaultMemPool,
                         pool_out, dev);
}

CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev){
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetMemPool, pool, dev);
}

CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask,
                         CUdevice dev) {
  return CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetLuid, luid,
                         deviceNodeMask, dev);
}

CUresult cuDeviceTotalMem_v2 ( size_t* bytes, CUdevice dev ) {
    SOFTMIG_PASSIVE_FORWARD(cuDeviceTotalMem_v2, bytes, dev);
    LOG_DEBUG("into cuDeviceTotalMem");
    ENSURE_INITIALIZED();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceTotalMem_v2, bytes, dev);
    if (res != CUDA_SUCCESS) {
        return res;
    }
    size_t limit = get_current_device_memory_limit(dev);
    if (limit != 0 && limit < *bytes) {
        *bytes = limit;
    }
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *driverVersion) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDriverGetVersion,driverVersion);
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev){
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetTexture1DLinearMaxWidth,maxWidthInElements,format,numChannels,dev);
}

CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceSetMemPool,dev,pool);
}

CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) {
   return CUDA_OVERRIDE_CALL(cuda_library_entry,cuFlushGPUDirectRDMAWrites,target,scope);
}
