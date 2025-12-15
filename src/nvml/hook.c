#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <dlfcn.h>
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
extern int virtual_nvml_devices;
extern int cuda_to_nvml_map_array[CUDA_DEVICE_MAX_COUNT];

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
    return NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetIndex, device, index);
}


// _dl_sym is an internal glibc function, use weak linking if available
#ifdef __GLIBC__
extern void* _dl_sym(void*, const char*, void*) __attribute__((weak));
#endif

void load_nvml_libraries() {
    void *table = NULL;
    char driver_filename[FILENAME_MAX];

    if (real_dlsym == NULL) {
        // Try multiple methods to get the real dlsym, more robust for CVMFS/Compute Canada
        real_dlsym = dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");
        if (real_dlsym == NULL) {
            real_dlsym = dlvsym(RTLD_NEXT,"dlsym","");
        }
        if (real_dlsym == NULL) {
            const char *glibc_versions[] = {"GLIBC_2.34", "GLIBC_2.17", "GLIBC_2.4", NULL};
            for (int i = 0; glibc_versions[i] != NULL && real_dlsym == NULL; i++) {
                real_dlsym = dlvsym(RTLD_NEXT, "dlsym", glibc_versions[i]);
            }
        }
        if (real_dlsym == NULL) {
            // Try getting dlsym from libdl.so.2 directly
            void *libdl = dlopen("libdl.so.2", RTLD_LAZY | RTLD_LOCAL);
            if (libdl != NULL) {
                typedef void* (*dlsym_fn)(void*, const char*);
                dlsym_fn libdl_dlsym = (dlsym_fn)dlvsym(libdl, "dlsym", "");
                if (libdl_dlsym == NULL) {
                    libdl_dlsym = (dlsym_fn)dlvsym(libdl, "dlsym", "GLIBC_2.2.5");
                }
                if (libdl_dlsym != NULL) {
                    real_dlsym = (fp_dlsym)libdl_dlsym(RTLD_DEFAULT, "dlsym");
                }
            }
        }
        if (real_dlsym == NULL) {
            #ifdef __GLIBC__
            extern void* _dl_sym(void*, const char*, void*) __attribute__((weak));
            if (_dl_sym != NULL) {
                real_dlsym = (fp_dlsym)_dl_sym(RTLD_NEXT, "dlsym", (void*)dlsym);
            }
            #endif
            if (real_dlsym == NULL)
                LOG_ERROR("real dlsym not found - all methods failed");
        }
    }
    snprintf(driver_filename, FILENAME_MAX - 1, "%s", "libnvidia-ml.so.1");
    driver_filename[FILENAME_MAX - 1] = '\0';

    table = dlopen(driver_filename, RTLD_NOW | RTLD_NODELETE);
    if (!table) {
        LOG_WARN("can't find library %s", driver_filename);  
    }
    int i;
    for (i = 0; i < NVML_ENTRY_END; i++) {
        // Reduced logging - only log missing functions, not every loaded function
        nvml_library_entry[i].fn_ptr = real_dlsym(table, nvml_library_entry[i].name);
        if (!nvml_library_entry[i].fn_ptr) {
            LOG_INFO("can't find function %s in %s", nvml_library_entry[i].name,
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

// Sum memory usage directly from NVML by querying running processes
// This is more accurate than tracking allocations ourselves
// Made non-static so it can be used for enforcement checks too
uint64_t sum_process_memory_from_nvml(nvmlDevice_t device) {
    unsigned int process_count = SHARED_REGION_MAX_PROCESS_NUM;
    // Use nvmlProcessInfo_t from nvml-subset.h (standard type)
    nvmlProcessInfo_t infos[SHARED_REGION_MAX_PROCESS_NUM];
    
    // Get the real NVML function (bypass our hook to avoid recursion)
    // Note: nvmlDeviceGetComputeRunningProcesses is mapped to _v2 via prefix header
    nvmlReturn_t ret = NVML_OVERRIDE_CALL_NO_LOG(nvml_library_entry, 
                                                   nvmlDeviceGetComputeRunningProcesses, 
                                                   device, &process_count, infos);
    
    LOG_INFO("sum_process_memory_from_nvml: nvmlDeviceGetComputeRunningProcesses returned %d, process_count=%u", ret, process_count);
    
    if (ret != NVML_SUCCESS && ret != NVML_ERROR_INSUFFICIENT_SIZE) {
        // If we can't get process info, fall back to tracked usage
        LOG_WARN("nvmlDeviceGetComputeRunningProcesses failed: %d (%s), falling back to tracked usage", 
                 ret, (ret == NVML_ERROR_UNINITIALIZED ? "UNINITIALIZED" : 
                       ret == NVML_ERROR_INVALID_ARGUMENT ? "INVALID_ARGUMENT" : "OTHER"));
        return 0;  // Return 0 to trigger fallback
    }
    
    uint64_t total_usage = 0;
    const uint64_t MIN_PROCESS_MEMORY = 9 * 1024 * 1024;  // 9 MB minimum per process
    const double PROCESS_OVERHEAD_PERCENT = 0.05;  // 5% overhead
    const uint64_t NVML_VALUE_NOT_AVAILABLE_ULL = 0xFFFFFFFFFFFFFFFFULL;  // NVML constant for unavailable values
    
    // Get current user's UID for fallback filtering
    uid_t current_uid = getuid();
    
    LOG_DEBUG("sum_process_memory_from_nvml: Found %u processes on device, filtering by cgroup session or UID %u", process_count, current_uid);
    
    unsigned int included_count = 0;
    unsigned int skipped_count = 0;
    
    // Sum up memory from all processes belonging to current cgroup session (or current user if not in cgroup)
    for (unsigned int i = 0; i < process_count; i++) {
        // First try to check if process belongs to current cgroup session
        int cgroup_check = proc_belongs_to_current_cgroup_session(infos[i].pid);
        
        if (cgroup_check == -1) {
            // Couldn't determine cgroup or not in a cgroup session - fall back to UID check
            uid_t proc_uid = proc_get_uid(infos[i].pid);
            
            if (proc_uid == (uid_t)-1) {
                // Couldn't read UID - skip this process to avoid blocking on shared region lock
                // We don't want to block nvidia-smi if another process is holding the lock
                // If the process belongs to us, it will be counted when we can read its UID
                LOG_DEBUG("  Process[%u] PID=%u: skipping (could not read UID from /proc/%d/status - avoiding lock contention)", 
                         i, infos[i].pid, infos[i].pid);
                skipped_count++;
                continue;
            } else if (proc_uid != current_uid) {
                LOG_DEBUG("  Process[%u] PID=%u: skipping (UID %u != current UID %u)", 
                         i, infos[i].pid, proc_uid, current_uid);
                skipped_count++;
                continue;
            }
        } else if (cgroup_check == 0) {
            // Process is in a different cgroup session - skip it
            LOG_DEBUG("  Process[%u] PID=%u: skipping (different cgroup session)", 
                     i, infos[i].pid);
            skipped_count++;
            continue;
        }
        // cgroup_check == 1 means process belongs to current cgroup session - include it
        
        included_count++;
        
        // Skip if memory value is not available (NVML_VALUE_NOT_AVAILABLE) or invalid
        if (infos[i].usedGpuMemory != NVML_VALUE_NOT_AVAILABLE_ULL && infos[i].usedGpuMemory > 0) {
            uint64_t process_mem = infos[i].usedGpuMemory;
            // Add 5% overhead, then ensure minimum
            uint64_t process_mem_with_overhead = (uint64_t)(process_mem * (1.0 + PROCESS_OVERHEAD_PERCENT));
            uint64_t process_mem_counted = (process_mem_with_overhead < MIN_PROCESS_MEMORY) ? MIN_PROCESS_MEMORY : process_mem_with_overhead;
            total_usage += process_mem_counted;
            LOG_DEBUG("  Process[%u] PID=%u: %llu bytes (%.2f MiB) + 5%% = %llu bytes (%.2f MiB)", 
                     i, infos[i].pid,
                     (unsigned long long)process_mem, process_mem / (1024.0 * 1024.0),
                     (unsigned long long)process_mem_counted, process_mem_counted / (1024.0 * 1024.0));
        } else {
            // Even if NVML reports 0 or unavailable, count minimum for the process
            total_usage += MIN_PROCESS_MEMORY;
            LOG_DEBUG("  Process[%u] PID=%u: usedGpuMemory=%llu (unavailable/zero), counting minimum %llu bytes (%.2f MiB)", 
                     i, infos[i].pid, (unsigned long long)infos[i].usedGpuMemory,
                     (unsigned long long)MIN_PROCESS_MEMORY, MIN_PROCESS_MEMORY / (1024.0 * 1024.0));
        }
    }
    
    LOG_INFO("sum_process_memory_from_nvml: Included %u processes, skipped %u processes (different cgroup session/UID or couldn't read)", 
             included_count, skipped_count);
    
    LOG_INFO("sum_process_memory_from_nvml: Total usage (current cgroup session/user) = %llu bytes (%.2f MiB)", 
             (unsigned long long)total_usage, total_usage / (1024.0 * 1024.0));
    
    return total_usage;
}

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
    
    LOG_INFO("_nvmlDeviceGetMemoryInfo: limit=%llu bytes (%.2f MiB), usage=%llu bytes (%.2f MiB), free=%llu bytes (%.2f MiB)",
             (unsigned long long)limit, limit / (1024.0 * 1024.0),
             (unsigned long long)usage, usage / (1024.0 * 1024.0),
             (unsigned long long)(limit > usage ? (limit - usage) : 0), 
             (limit > usage ? (limit - usage) : 0) / (1024.0 * 1024.0));
    
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
    LOG_INFO("NVML DeviceGetHandleByPciBusID_v2 %s",pciBusId);
    return NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetHandleByPciBusId_v2,pciBusId,device);
}


nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId,
                                           nvmlDevice_t *device) {
    LOG_DEBUG("NVML DeviceGetHandleByPciBusId %s",pciBusId);
    return NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetHandleByPciBusId,
                         pciBusId, device);
}

nvmlReturn_t nvmlDeviceGetHandleBySerial ( const char* serial, nvmlDevice_t* device ) {
    LOG_INFO("NVML DeviceGetHandleBySerial Not supported %s",serial);
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
    LOG_DEBUG("nvmlInitWithFlags")
    pthread_once(&init_virtual_map_pre_flag, (void(*) (void))nvml_preInit);
    nvmlReturn_t res =  NVML_OVERRIDE_CALL(nvml_library_entry, nvmlInitWithFlags,flags);
    pthread_once(&init_virtual_map_post_flag,(void (*)(void))nvml_postInit);
    return res;
}

nvmlReturn_t nvmlInit(void) {
    LOG_DEBUG("nvmlInit")
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
