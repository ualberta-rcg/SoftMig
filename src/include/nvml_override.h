/**
 * @file nvml_override.h
 * @brief Forward declarations for NVML functions that SoftMig overrides.
 */
#ifndef __NVML_OVERRIDE_H__
#define __NVML_OVERRIDE_H__

// Forward declare nvmlReturn_t - it's defined in nvml-subset.h
// This allows nvml_override.h to be included before nvml-subset.h
// We need to forward declare the enum first
#ifndef __NVML_RETURN_T_DEFINED__
#define __NVML_RETURN_T_DEFINED__
enum nvmlReturn_enum;
typedef enum nvmlReturn_enum nvmlReturn_t;
#endif

// Function declaration - implementation is in hook.c
// Note: This requires nvmlReturn_t to be defined, so files including this
// should also include nvml-subset.h (via libnvml_hook.h) before using this function
nvmlReturn_t nvmlInternalGetExportTable(const void **ppExportTable,
                                        void *pExportTableId);


//nvmlReturn_t nvmlDeviceSetTemperatureThreshold(
//    nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int *temp);


//nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance,
//                                              unsigned int *gpuInstanceId);

//nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory);

// Function declarations removed - these are implemented in hook.c and nvml_entry.c
// Declaring them here causes conflicts with system <nvml.h> when it's included
// The implementations in hook.c serve as the declarations

#endif