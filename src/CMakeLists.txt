cmake_minimum_required (VERSION 2.8.11)
project (libsoftmig)


# set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
#set(CUDA_FILES libcuda_hook.c libcuda_context.c libcuda_device.c libcuda_stream.c libcuda_memory.c)

add_subdirectory(multiprocess)
add_subdirectory(allocator)
add_subdirectory(cuda)
add_subdirectory(nvml)

set(LIBSOFTMIG softmig)

# Auto-generate a linker version script that exports every cu*/nvml* wrapper
# defined in our sources. This is required for LD_PRELOAD interposition to
# catch apps that link -lcuda / -lnvidia-ml directly (gpu-burn, cuda-samples,
# etc.) instead of going through cudart's dlsym() path.
set(SOFTMIG_VERSION_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/softmig.ver)
add_custom_command(
    OUTPUT ${SOFTMIG_VERSION_SCRIPT}
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/gen_version_script.sh
            ${CMAKE_CURRENT_SOURCE_DIR} ${SOFTMIG_VERSION_SCRIPT}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_version_script.sh
    COMMENT "Generating SoftMig linker version script"
    VERBATIM
)
add_custom_target(softmig_version_script DEPENDS ${SOFTMIG_VERSION_SCRIPT})

add_library(${LIBSOFTMIG} SHARED libsoftmig.c dlsym_resolve.c utils.c log_file.c nvml_cache.c $<TARGET_OBJECTS:nvml_mod> $<TARGET_OBJECTS:cuda_mod> $<TARGET_OBJECTS:allocator_mod> $<TARGET_OBJECTS:multiprocess_mod>)
add_dependencies(${LIBSOFTMIG} softmig_version_script)
# Main TU (libsoftmig.c) also defines cuInit wrapper; compile without hidden
# visibility so the version-script can actually export it. Internal helpers
# are still trimmed by the version-script.
set(MAIN_TU_FLAGS ${LIBRARY_COMPILE_FLAGS})
list(REMOVE_ITEM MAIN_TU_FLAGS "-fvisibility=hidden")
target_compile_options(${LIBSOFTMIG} PUBLIC ${MAIN_TU_FLAGS})
set_property(TARGET ${LIBSOFTMIG} APPEND_STRING PROPERTY LINK_FLAGS
    " -Wl,--version-script=${SOFTMIG_VERSION_SCRIPT}")
# Note: We link against system driver's libnvidia-ml.so.1 (via find_library in nvml/CMakeLists.txt),
# NOT the CUDA toolkit's copy. At runtime, the code loads the driver's NVML library
# (libnvidia-ml.so.1) directly via dlopen in hook.c to ensure compatibility with the installed driver.
# Process info struct handling follows the real NVML v1/v2 layouts from nvml-subset.h
# to stay aligned with the driver NVML library ABI.
# Don't link -lnvidia-ml here. Our NVML wrappers call through a function
# pointer table that is populated at runtime via dlopen("libnvidia-ml.so.1")
# in load_nvml_libraries(). Linking against libnvidia-ml would pull NVML's
# symbol versions into our output and mask our wrappers in the final dynsym
# table, which breaks LD_PRELOAD interposition for direct-linked apps.
target_link_libraries(${LIBSOFTMIG} PUBLIC -lcuda)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
add_custom_target(strip_symbol ALL
    COMMAND strip -x ${CMAKE_BINARY_DIR}/lib${LIBSOFTMIG}.so
    DEPENDS ${LIBSOFTMIG})
endif()
