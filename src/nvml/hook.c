/**
 * @file hook.c (nvml)
 * @brief NVML library loading, dispatch table, and memory info hook.
 *
 * Populates the nvml_library_entry[] dispatch table by dlopen-ing
 * libnvidia-ml.so.1 and resolving every hooked symbol. Provides the
 * nvmlDeviceGetMemoryInfo hook that replaces total/used/free with
 * SoftMig's per-job limits and NVML-summed usage (cgroup/UID filtered).
 */
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
// Prevent system <nvml.h> from being included - we use nvml-subset.h instead
// This macro tells nvml.h (if included) to skip some definitions
#define NVML_NO_UNVERSIONED_FUNC_DEFS
// Include nvml-subset.h FIRST - it defines structures we need
#include "include/nvml-subset.h"
#include "include/nvml_prefix.h"
#include "include/libnvml_hook.h"
#include "include/nvml_override.h"
#include "include/utils.h"
#include "include/process_utils.h"
#include "include/dlsym_resolve.h"
#include "include/nvml_cache.h"
// Note: multiprocess_memory_limit.h includes <nvml.h> in its .c file, not the .h file
// So including the header here should be safe
#include "multiprocess/multiprocess_memory_limit.h"

entry_t nvml_library_entry[] = {
    {.name = "nvmlInit"},
    {.name = "nvmlShutdown"},
    {.name = "nvmlErrorString"},
    {.name = "nvmlDeviceGetHandleByIndex"},
    {.name = "nvmlDeviceGetComputeRunningProcesses"},
    {.name = "nvmlDeviceGetPciInfo"},
    {.name = "nvmlDeviceGetProcessUtilization"},
    {.name = "nvmlDeviceGetCount"},
    {.name = "nvmlDeviceClearAccountingPids"},
    {.name = "nvmlDeviceClearCpuAffinity"},
    {.name = "nvmlDeviceClearEccErrorCounts"},
    {.name = "nvmlDeviceDiscoverGpus"},
    {.name = "nvmlDeviceFreezeNvLinkUtilizationCounter"},
    {.name = "nvmlDeviceGetAccountingBufferSize"},
    {.name = "nvmlDeviceGetAccountingMode"},
    {.name = "nvmlDeviceGetAccountingPids"},
    {.name = "nvmlDeviceGetAccountingStats"},
    {.name = "nvmlDeviceGetActiveVgpus"},
    {.name = "nvmlDeviceGetAPIRestriction"},
    {.name = "nvmlDeviceGetApplicationsClock"},
    {.name = "nvmlDeviceGetAutoBoostedClocksEnabled"},
    {.name = "nvmlDeviceGetBAR1MemoryInfo"},
    {.name = "nvmlDeviceGetBoardId"},
    {.name = "nvmlDeviceGetBoardPartNumber"},
    {.name = "nvmlDeviceGetBrand"},
    {.name = "nvmlDeviceGetBridgeChipInfo"},
    {.name = "nvmlDeviceGetClock"},
    {.name = "nvmlDeviceGetClockInfo"},
    {.name = "nvmlDeviceGetComputeMode"},
    {.name = "nvmlDeviceGetCount_v2"},
    {.name = "nvmlDeviceGetCpuAffinity"},
    {.name = "nvmlDeviceGetCreatableVgpus"},
    {.name = "nvmlDeviceGetCudaComputeCapability"},
    {.name = "nvmlDeviceGetCurrentClocksThrottleReasons"},
    {.name = "nvmlDeviceGetCurrPcieLinkGeneration"},
    {.name = "nvmlDeviceGetCurrPcieLinkWidth"},
    {.name = "nvmlDeviceGetDecoderUtilization"},
    {.name = "nvmlDeviceGetDefaultApplicationsClock"},
    {.name = "nvmlDeviceGetDetailedEccErrors"},
    {.name = "nvmlDeviceGetDisplayActive"},
    {.name = "nvmlDeviceGetDisplayMode"},
    {.name = "nvmlDeviceGetDriverModel"},
    {.name = "nvmlDeviceGetEccMode"},
    {.name = "nvmlDeviceGetEncoderCapacity"},
    {.name = "nvmlDeviceGetEncoderSessions"},
    {.name = "nvmlDeviceGetEncoderStats"},
    {.name = "nvmlDeviceGetEncoderUtilization"},
    {.name = "nvmlDeviceGetEnforcedPowerLimit"},
    {.name = "nvmlDeviceGetFanSpeed"},
    {.name = "nvmlDeviceGetFanSpeed_v2"},
    {.name = "nvmlDeviceGetFieldValues"},
    {.name = "nvmlDeviceGetGpuOperationMode"},
    {.name = "nvmlDeviceGetGraphicsRunningProcesses"},
    {.name = "nvmlDeviceGetGridLicensableFeatures"},
    {.name = "nvmlDeviceGetHandleByIndex_v2"},
    {.name = "nvmlDeviceGetHandleByPciBusId"},
    {.name = "nvmlDeviceGetHandleByPciBusId_v2"},
    {.name = "nvmlDeviceGetHandleBySerial"},
    {.name = "nvmlDeviceGetHandleByUUID"},
    {.name = "nvmlDeviceGetIndex"},
    {.name = "nvmlDeviceGetInforomConfigurationChecksum"},
    {.name = "nvmlDeviceGetInforomImageVersion"},
    {.name = "nvmlDeviceGetInforomVersion"},
    {.name = "nvmlDeviceGetMaxClockInfo"},
    {.name = "nvmlDeviceGetMaxCustomerBoostClock"},
    {.name = "nvmlDeviceGetMaxPcieLinkGeneration"},
    {.name = "nvmlDeviceGetMaxPcieLinkWidth"},
    {.name = "nvmlDeviceGetMemoryErrorCounter"},
    {.name = "nvmlDeviceGetMemoryInfo"},
    {.name = "nvmlDeviceGetMemoryInfo_v2"},
    {.name = "nvmlDeviceGetMinorNumber"},
    {.name = "nvmlDeviceGetMPSComputeRunningProcesses"},
    {.name = "nvmlDeviceGetMultiGpuBoard"},
    {.name = "nvmlDeviceGetName"},
    {.name = "nvmlDeviceGetNvLinkCapability"},
    {.name = "nvmlDeviceGetNvLinkErrorCounter"},
    {.name = "nvmlDeviceGetNvLinkRemotePciInfo"},
    {.name = "nvmlDeviceGetNvLinkRemotePciInfo_v2"},
    {.name = "nvmlDeviceGetNvLinkState"},
    {.name = "nvmlDeviceGetNvLinkUtilizationControl"},
    {.name = "nvmlDeviceGetNvLinkUtilizationCounter"},
    {.name = "nvmlDeviceGetNvLinkVersion"},
    {.name = "nvmlDeviceGetP2PStatus"},
    {.name = "nvmlDeviceGetPcieReplayCounter"},
    {.name = "nvmlDeviceGetPcieThroughput"},
    {.name = "nvmlDeviceGetPciInfo_v2"},
    {.name = "nvmlDeviceGetPciInfo_v3"},
    {.name = "nvmlDeviceGetPerformanceState"},
    {.name = "nvmlDeviceGetPersistenceMode"},
    {.name = "nvmlDeviceGetPowerManagementDefaultLimit"},
    {.name = "nvmlDeviceGetPowerManagementLimit"},
    {.name = "nvmlDeviceGetPowerManagementLimitConstraints"},
    {.name = "nvmlDeviceGetPowerManagementMode"},
    {.name = "nvmlDeviceGetPowerState"},
    {.name = "nvmlDeviceGetPowerUsage"},
    {.name = "nvmlDeviceGetRetiredPages"},
    {.name = "nvmlDeviceGetRetiredPagesPendingStatus"},
    {.name = "nvmlDeviceGetSamples"},
    {.name = "nvmlDeviceGetSerial"},
    {.name = "nvmlDeviceGetSupportedClocksThrottleReasons"},
    {.name = "nvmlDeviceGetSupportedEventTypes"},
    {.name = "nvmlDeviceGetSupportedGraphicsClocks"},
    {.name = "nvmlDeviceGetSupportedMemoryClocks"},
    {.name = "nvmlDeviceGetSupportedVgpus"},
    {.name = "nvmlDeviceGetTemperature"},
    {.name = "nvmlDeviceGetTemperatureThreshold"},
    {.name = "nvmlDeviceGetTopologyCommonAncestor"},
    {.name = "nvmlDeviceGetTopologyNearestGpus"},
    {.name = "nvmlDeviceGetTotalEccErrors"},
    {.name = "nvmlDeviceGetTotalEnergyConsumption"},
    {.name = "nvmlDeviceGetUtilizationRates"},
    {.name = "nvmlDeviceGetUUID"},
    {.name = "nvmlDeviceGetVbiosVersion"},
    {.name = "nvmlDeviceGetVgpuMetadata"},
    {.name = "nvmlDeviceGetVgpuProcessUtilization"},
    {.name = "nvmlDeviceGetVgpuUtilization"},
    {.name = "nvmlDeviceGetViolationStatus"},
    {.name = "nvmlDeviceGetVirtualizationMode"},
    {.name = "nvmlDeviceModifyDrainState"},
    {.name = "nvmlDeviceOnSameBoard"},
    {.name = "nvmlDeviceQueryDrainState"},
    {.name = "nvmlDeviceRegisterEvents"},
    {.name = "nvmlDeviceRemoveGpu"},
    {.name = "nvmlDeviceRemoveGpu_v2"},
    {.name = "nvmlDeviceResetApplicationsClocks"},
    {.name = "nvmlDeviceResetNvLinkErrorCounters"},
    {.name = "nvmlDeviceResetNvLinkUtilizationCounter"},
    {.name = "nvmlDeviceSetAccountingMode"},
    {.name = "nvmlDeviceSetAPIRestriction"},
    {.name = "nvmlDeviceSetApplicationsClocks"},
    {.name = "nvmlDeviceSetAutoBoostedClocksEnabled"},
    /** We hijack this call*/
    {.name = "nvmlDeviceSetComputeMode"},
    {.name = "nvmlDeviceSetCpuAffinity"},
    {.name = "nvmlDeviceSetDefaultAutoBoostedClocksEnabled"},
    {.name = "nvmlDeviceSetDriverModel"},
    {.name = "nvmlDeviceSetEccMode"},
    {.name = "nvmlDeviceSetGpuOperationMode"},
    {.name = "nvmlDeviceSetNvLinkUtilizationControl"},
    {.name = "nvmlDeviceSetPersistenceMode"},
    {.name = "nvmlDeviceSetPowerManagementLimit"},
    {.name = "nvmlDeviceSetVirtualizationMode"},
    {.name = "nvmlDeviceValidateInforom"},
    {.name = "nvmlEventSetCreate"},
    {.name = "nvmlEventSetFree"},
    {.name = "nvmlEventSetWait"},
    {.name = "nvmlGetVgpuCompatibility"},
    {.name = "nvmlInit_v2"},
    {.name = "nvmlInitWithFlags"},
    {.name = "nvmlInternalGetExportTable"},
    {.name = "nvmlSystemGetCudaDriverVersion"},
    {.name = "nvmlSystemGetCudaDriverVersion_v2"},
    {.name = "nvmlSystemGetDriverVersion"},
    {.name = "nvmlSystemGetHicVersion"},
    {.name = "nvmlSystemGetNVMLVersion"},
    {.name = "nvmlSystemGetProcessName"},
    {.name = "nvmlSystemGetTopologyGpuSet"},
    {.name = "nvmlUnitGetCount"},
    {.name = "nvmlUnitGetDevices"},
    {.name = "nvmlUnitGetFanSpeedInfo"},
    {.name = "nvmlUnitGetHandleByIndex"},
    {.name = "nvmlUnitGetLedState"},
    {.name = "nvmlUnitGetPsuInfo"},
    {.name = "nvmlUnitGetTemperature"},
    {.name = "nvmlUnitGetUnitInfo"},
    {.name = "nvmlUnitSetLedState"},
    {.name = "nvmlVgpuInstanceGetEncoderCapacity"},
    {.name = "nvmlVgpuInstanceGetEncoderSessions"},
    {.name = "nvmlVgpuInstanceGetEncoderStats"},
    {.name = "nvmlVgpuInstanceGetFbUsage"},
    {.name = "nvmlVgpuInstanceGetFrameRateLimit"},
    {.name = "nvmlVgpuInstanceGetLicenseStatus"},
    {.name = "nvmlVgpuInstanceGetMetadata"},
    {.name = "nvmlVgpuInstanceGetType"},
    {.name = "nvmlVgpuInstanceGetUUID"},
    {.name = "nvmlVgpuInstanceGetVmDriverVersion"},
    {.name = "nvmlVgpuInstanceGetVmID"},
    {.name = "nvmlVgpuInstanceSetEncoderCapacity"},
    {.name = "nvmlVgpuTypeGetClass"},
    {.name = "nvmlVgpuTypeGetDeviceID"},
    {.name = "nvmlVgpuTypeGetFramebufferSize"},
    {.name = "nvmlVgpuTypeGetFrameRateLimit"},
    {.name = "nvmlVgpuTypeGetLicense"},
    {.name = "nvmlVgpuTypeGetMaxInstances"},
    {.name = "nvmlVgpuTypeGetName"},
    {.name = "nvmlVgpuTypeGetNumDisplayHeads"},
    {.name = "nvmlVgpuTypeGetResolution"},
    {.name = "nvmlDeviceGetFBCSessions"},
    {.name = "nvmlDeviceGetFBCStats"},
    {.name = "nvmlDeviceGetGridLicensableFeatures_v2"},
    {.name = "nvmlDeviceGetRetiredPages_v2"},
    {.name = "nvmlDeviceResetGpuLockedClocks"},
    {.name = "nvmlDeviceSetGpuLockedClocks"},
    {.name = "nvmlVgpuInstanceGetAccountingMode"},
    {.name = "nvmlVgpuInstanceGetAccountingPids"},
    {.name = "nvmlVgpuInstanceGetAccountingStats"},
    {.name = "nvmlVgpuInstanceGetFBCSessions"},
    {.name = "nvmlVgpuInstanceGetFBCStats"},
    {.name = "nvmlVgpuTypeGetMaxInstancesPerVm"},
    {.name = "nvmlGetVgpuVersion"},
    {.name = "nvmlSetVgpuVersion"},
    {.name = "nvmlDeviceGetGridLicensableFeatures_v3"},
    {.name = "nvmlDeviceGetHostVgpuMode"},
    {.name = "nvmlDeviceGetPgpuMetadataString"},
    {.name = "nvmlVgpuInstanceGetEccMode"},
    {.name = "nvmlComputeInstanceDestroy"},
    {.name = "nvmlComputeInstanceGetInfo"},
    {.name = "nvmlDeviceCreateGpuInstance"},
    {.name = "nvmlDeviceGetArchitecture"},
    {.name = "nvmlDeviceGetAttributes"},
    {.name = "nvmlDeviceGetAttributes_v2"},
    {.name = "nvmlDeviceGetComputeInstanceId"},
    {.name = "nvmlDeviceGetCpuAffinityWithinScope"},
    {.name = "nvmlDeviceGetDeviceHandleFromMigDeviceHandle"},
    {.name = "nvmlDeviceGetGpuInstanceById"},
    {.name = "nvmlDeviceGetGpuInstanceId"},
    {.name = "nvmlDeviceGetGpuInstancePossiblePlacements"},
    {.name = "nvmlDeviceGetGpuInstanceProfileInfo"},
    {.name = "nvmlDeviceGetGpuInstanceRemainingCapacity"},
    {.name = "nvmlDeviceGetGpuInstances"},
    {.name = "nvmlDeviceGetMaxMigDeviceCount"},
    {.name = "nvmlDeviceGetMemoryAffinity"},
    {.name = "nvmlDeviceGetMigDeviceHandleByIndex"},
    {.name = "nvmlDeviceGetMigMode"},
    {.name = "nvmlDeviceGetRemappedRows"},
    {.name = "nvmlDeviceGetRowRemapperHistogram"},
    {.name = "nvmlDeviceIsMigDeviceHandle"},
    {.name = "nvmlDeviceSetMigMode"},
    {.name = "nvmlEventSetWait_v2"},
    {.name = "nvmlGpuInstanceCreateComputeInstance"},
    {.name = "nvmlGpuInstanceDestroy"},
    {.name = "nvmlGpuInstanceGetComputeInstanceById"},
    {.name = "nvmlGpuInstanceGetComputeInstanceProfileInfo"},
    {.name = "nvmlGpuInstanceGetComputeInstanceRemainingCapacity"},
    {.name = "nvmlGpuInstanceGetComputeInstances"},
    {.name = "nvmlGpuInstanceGetInfo"},
    {.name = "nvmlVgpuInstanceClearAccountingPids"},
    {.name = "nvmlVgpuInstanceGetMdevUUID"},
    {.name = "nvmlComputeInstanceGetInfo_v2"},
    {.name = "nvmlDeviceGetComputeRunningProcesses_v2"},
    {.name = "nvmlDeviceGetGraphicsRunningProcesses_v2"},
    {.name = "nvmlDeviceSetTemperatureThreshold"},
    //{.name = "nvmlRetry_NvRmControl"},
    {.name = "nvmlVgpuInstanceGetGpuInstanceId"},
    {.name = "nvmlVgpuTypeGetGpuInstanceProfileId"},
};

pthread_once_t init_virtual_map_pre_flag = PTHREAD_ONCE_INIT;
pthread_once_t init_virtual_map_post_flag = PTHREAD_ONCE_INIT;

typedef void* (*fp_dlsym)(void*, const char*);
extern fp_dlsym real_dlsym;
extern int cuda_to_nvml_map_array[CUDA_DEVICE_MAX_COUNT];

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
    return NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetIndex, device, index);
}


/** Resolve all NVML symbols from libnvidia-ml.so.1 into nvml_library_entry[]. */
void load_nvml_libraries() {
    void *table = NULL;
    char driver_filename[FILENAME_MAX];

    if (real_dlsym == NULL) {
        real_dlsym = resolve_real_dlsym();
        if (real_dlsym == NULL) {
            LOG_ERROR("real dlsym not found - all methods failed");
            return;
        }
    }
    snprintf(driver_filename, FILENAME_MAX - 1, "%s", "libnvidia-ml.so.1");
    driver_filename[FILENAME_MAX - 1] = '\0';

    table = dlopen(driver_filename, RTLD_NOW | RTLD_NODELETE);
    if (!table) {
        LOG_WARN("can't find library %s", driver_filename);
        return;  // CRITICAL: Return early if we can't open the library to avoid segfault
    }
    int i;
    for (i = 0; i < NVML_ENTRY_END; i++) {
        // Reduced logging - only log missing functions, not every loaded function
        nvml_library_entry[i].fn_ptr = real_dlsym(table, nvml_library_entry[i].name);
        if (!nvml_library_entry[i].fn_ptr) {
            LOG_DEBUG("can't find function %s in %s", nvml_library_entry[i].name,
                driver_filename);
        }
    }
    LOG_DEBUG("loaded nvml libraries");
    dlclose(table);
}

void nvml_preInit() {
    ensure_initialized();
    load_env_from_file(ENV_OVERRIDE_FILE);
    load_nvml_libraries();
    for (int i = 0; i < CUDA_DEVICE_MAX_COUNT; i++) {
        cuda_to_nvml_map_array[i] = i;
    }   
}

void nvml_postInit() {
    init_device_info();
}

/**
 * Sum GPU memory used by processes belonging to the current cgroup/UID on a device.
 *
 * Bypasses the hook to get ALL processes, then filters by cgroup session or UID.
 * Applies a 9 MB minimum and 5% overhead per process. Returns 0 if the NVML
 * query fails (caller should fall back to tracked usage).
 */
uint64_t sum_process_memory_from_nvml(nvmlDevice_t device) {
    nvmlProcessInfo_t infos[SHARED_REGION_MAX_PROCESS_NUM];

    unsigned int process_count = nvml_cached_get_compute_processes(
        device, SHARED_REGION_MAX_PROCESS_NUM, infos);
    
    uint64_t total_usage = 0;
    const uint64_t MIN_PROCESS_MEMORY = 9 * 1024 * 1024;
    const double PROCESS_OVERHEAD_PERCENT = 0.05;
    const uint64_t NVML_VALUE_NOT_AVAILABLE_ULL = 0xFFFFFFFFFFFFFFFFULL;
    
    uid_t current_uid = getuid();
    
    unsigned int bounded_count = process_count;
    unsigned int included_count = 0;
    unsigned int skipped_count = 0;
    
    // Sum up memory from all processes belonging to current cgroup session (or current user if not in cgroup)
    for (unsigned int i = 0; i < bounded_count; i++) {
        unsigned int actual_pid = infos[i].pid;
        if (actual_pid == 0) {
            skipped_count++;
            continue;  // Skip if we can't get a valid PID
        }
        
        // First try to check if process belongs to current cgroup session
        int cgroup_check = proc_belongs_to_current_cgroup_session(actual_pid);
        
        if (cgroup_check == -1) {
            // Couldn't determine cgroup or not in a cgroup session - fall back to UID check
            uid_t proc_uid = proc_get_uid(actual_pid);
            
            if (proc_uid == (uid_t)-1) {
                // Couldn't read UID - skip this process to avoid blocking on shared region lock
                skipped_count++;
                continue;
            } else if (proc_uid != current_uid) {
                skipped_count++;
                continue;
            }
        } else if (cgroup_check == 0) {
            // Process is in a different cgroup session - skip it
            skipped_count++;
            continue;
        }
        // cgroup_check == 1 means process belongs to current cgroup session - include it
        
        included_count++;
        
        uint64_t process_mem = infos[i].usedGpuMemory;
        
        if (process_mem != NVML_VALUE_NOT_AVAILABLE_ULL && process_mem > 0) {
            // Add 5% overhead, then ensure minimum
            uint64_t process_mem_with_overhead = (uint64_t)(process_mem * (1.0 + PROCESS_OVERHEAD_PERCENT));
            uint64_t process_mem_counted = (process_mem_with_overhead < MIN_PROCESS_MEMORY) ? MIN_PROCESS_MEMORY : process_mem_with_overhead;
            total_usage += process_mem_counted;
        } else {
            // Even if NVML reports 0 or unavailable, count minimum for the process
            total_usage += MIN_PROCESS_MEMORY;
        }
    }
    
    return total_usage;
}

/**
 * Hooked nvmlDeviceGetMemoryInfo — reports per-job limits and cgroup-filtered usage.
 *
 * Calls the real NVML to get hardware values, then replaces total/used/free
 * with the SoftMig memory limit and summed per-job usage.
 */
nvmlReturn_t _nvmlDeviceGetMemoryInfo(nvmlDevice_t device,void* memory,int version) {
    // Reduced logging - nvmlDeviceGetMemoryInfo is called very frequently (e.g., by nvidia-smi)
    if (memory == NULL) {
        return NVML_SUCCESS;
    }
    unsigned int dev_id;

    switch (version) {
        case 1:
            CHECK_NVML_API(NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetMemoryInfo, device, memory));
            // Removed frequent debug log
            break;
        case 2:
            CHECK_NVML_API(NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetMemoryInfo_v2, device, (nvmlMemory_v2_t *)memory));
            // Removed frequent debug log
            break;
        default:
            return NVML_ERROR_INVALID_ARGUMENT;
    }
    CHECK_NVML_API(nvmlDeviceGetIndex(device, &dev_id));
    int cudadev = nvml_to_cuda_map(dev_id);
    if (cudadev < 0) {
        return NVML_SUCCESS;
    }
    
    size_t limit = get_current_device_memory_limit(cudadev);
    // Reduced logging - this function is called very frequently
    if (limit == 0) {
        // No limit (e.g., root user) - pass through original NVML values unchanged
        // The original NVML call already set free, total, and used correctly
        return NVML_SUCCESS;
    }
    
    // Always use calculated sum from NVML (with 9MB minimum + 5% overhead and UID filtering)
    // This gives us the actual current usage as seen by NVML, properly filtered and adjusted
    // No fallback - always use the summed calculation to ensure consistency
    uint64_t usage = sum_process_memory_from_nvml(device);
    
    // If NVML query failed, usage will be 0, which is fine - it means no processes are using memory
    // We don't fall back to tracked usage to ensure all processes see the same value
    
    // Ensure usage doesn't exceed limit
    if (usage > limit) {
        LOG_WARN("Calculated usage (%llu bytes, %.2f MiB) exceeds limit (%llu bytes, %.2f MiB), capping to limit",
                 (unsigned long long)usage, usage / (1024.0 * 1024.0),
                 (unsigned long long)limit, limit / (1024.0 * 1024.0));
        usage = limit;
    }
    
    
    switch (version) {
    case 1:
         ((nvmlMemory_t*)memory)->free = (limit > usage) ? (limit - usage) : 0;
         ((nvmlMemory_t*)memory)->total = limit;
         ((nvmlMemory_t*)memory)->used = usage;
        return NVML_SUCCESS;
    case 2:
        ((nvmlMemory_v2_t *)memory)->free = (limit > usage) ? (limit - usage) : 0;
        ((nvmlMemory_v2_t *)memory)->total = limit;
        ((nvmlMemory_v2_t *)memory)->used = usage;
        return NVML_SUCCESS;
    } 
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) {
    return _nvmlDeviceGetMemoryInfo(device,memory,1); 
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) {
    return _nvmlDeviceGetMemoryInfo(device,memory,2);
}


nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2 ( nvmlDevice_t device, unsigned int  link, nvmlPciInfo_t* pci ) {
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetNvLinkRemotePciInfo_v2,device,link,pci);
    return res;
}

nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo ( nvmlDevice_t device, unsigned int  link, nvmlPciInfo_t* pci ) {
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetNvLinkRemotePciInfo,device,link,pci);
    return res;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex ( unsigned int  index, nvmlDevice_t* device ){
    nvmlReturn_t res;
    // Reduced logging - called frequently
    res = NVML_OVERRIDE_CALL_NO_LOG(nvml_library_entry,nvmlDeviceGetHandleByIndex,index,device);
    return res;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2 ( unsigned int  index, nvmlDevice_t* device ){
    nvmlReturn_t res;
    // Reduced logging - called frequently
    res = NVML_OVERRIDE_CALL_NO_LOG(nvml_library_entry,nvmlDeviceGetHandleByIndex_v2,index,device);
    return res;
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2 ( const char* pciBusId, nvmlDevice_t* device ) {
    return NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetHandleByPciBusId_v2,pciBusId,device);
}


nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId,
                                           nvmlDevice_t *device) {
    LOG_DEBUG("NVML DeviceGetHandleByPciBusId %s",pciBusId);
    return NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetHandleByPciBusId,
                         pciBusId, device);
}

nvmlReturn_t nvmlDeviceGetHandleBySerial ( const char* serial, nvmlDevice_t* device ) {
    return NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetHandleBySerial,serial,device);
}

nvmlReturn_t nvmlDeviceGetHandleByUUID ( const char* uuid, nvmlDevice_t* device ) {
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetHandleByUUID,uuid,device);
    return res;
}

nvmlReturn_t nvmlDeviceGetCount ( unsigned int* deviceCount ) {
    return NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetCount_v2,deviceCount);
}

nvmlReturn_t nvmlDeviceGetCount_v2 ( unsigned int* deviceCount ) {
    return NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetCount_v2,deviceCount);
}

nvmlReturn_t nvmlInitWithFlags( unsigned int  flags ) {
    LOG_DEBUG("nvmlInitWithFlags");
    pthread_once(&init_virtual_map_pre_flag, (void(*) (void))nvml_preInit);
    nvmlReturn_t res =  NVML_OVERRIDE_CALL(nvml_library_entry, nvmlInitWithFlags,flags);
    pthread_once(&init_virtual_map_post_flag,(void (*)(void))nvml_postInit);
    return res;
}

nvmlReturn_t nvmlInit(void) {
    LOG_DEBUG("nvmlInit");
    pthread_once(&init_virtual_map_pre_flag,(void (*)(void))nvml_preInit);
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry, nvmlInit_v2);
    pthread_once(&init_virtual_map_post_flag,(void (*)(void))nvml_postInit);
    return res;
}

nvmlReturn_t nvmlInit_v2(void) {
    LOG_DEBUG("nvmlInit_v2");
    pthread_once(&init_virtual_map_pre_flag,(void (*)(void))nvml_preInit);
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry, nvmlInit_v2);
    pthread_once(&init_virtual_map_post_flag,(void (*)(void))nvml_postInit);
    return res;
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci) {
  nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetPciInfo_v3, device,
                         pci);
  return res;
}

nvmlReturn_t nvmlDeviceGetPciInfo_v2(nvmlDevice_t device, nvmlPciInfo_t *pci) {
  nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetPciInfo_v2, device,
                         pci);
  return res;
}

nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t *pci) {
  nvmlReturn_t res =  NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetPciInfo, device, pci);
  return res;
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid,
                               unsigned int length) {
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetUUID, device, uuid,
                         length);
    return res;
}
