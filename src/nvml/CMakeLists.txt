# Try to find system NVML library first (driver's version, not CUDA toolkit's)
# This ensures we link against the driver's library which matches the runtime library
find_library(NVML_LIB nvidia-ml
    PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/lib
    NO_DEFAULT_PATH
)

# Fallback to CUDA_HOME if system library not found (for compatibility)
if(NOT NVML_LIB)
    message(WARNING "System NVML library not found, falling back to CUDA_HOME")
    set(NVML_LIB "-L${CUDA_HOME}/lib64" "-lnvidia-ml")
else()
    message(STATUS "Found system NVML library: ${NVML_LIB}")
    # Use the full path to ensure we link against system driver's library
    # This prevents linking against CUDA toolkit's copy when CUDA module is loaded
    set(NVML_LIB "${NVML_LIB}")
endif()

add_library(nvml_mod OBJECT nvml_entry.c hook.c)
target_compile_options(nvml_mod PUBLIC ${LIBRARY_COMPILE_FLAGS})
# Link against system driver's NVML, not CUDA toolkit's
target_link_libraries(nvml_mod PUBLIC ${NVML_LIB} -lcuda)
