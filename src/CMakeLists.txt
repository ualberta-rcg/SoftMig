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
add_library(${LIBSOFTMIG} SHARED libsoftmig.c utils.c log_file.c $<TARGET_OBJECTS:nvml_mod> $<TARGET_OBJECTS:cuda_mod> $<TARGET_OBJECTS:allocator_mod> $<TARGET_OBJECTS:multiprocess_mod>)
target_compile_options(${LIBSOFTMIG} PUBLIC ${LIBRARY_COMPILE_FLAGS})
# Note: We link against system driver's libnvidia-ml.so.1 (via find_library in nvml/CMakeLists.txt),
# NOT the CUDA toolkit's copy. At runtime, the code loads the driver's NVML library
# (libnvidia-ml.so.1) directly via dlopen in hook.c to ensure compatibility with the installed driver.
# The struct version fields are initialized before all NVML calls, and extract_pid_safely() handles
# version mismatches between CUDA toolkit NVML headers and driver NVML library (e.g., CUDA 12.2 headers vs driver 570.195.03).
target_link_libraries(${LIBSOFTMIG} PUBLIC -lcuda -lnvidia-ml)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
add_custom_target(strip_symbol ALL
    COMMAND strip -x ${CMAKE_BINARY_DIR}/lib${LIBSOFTMIG}.so
    DEPENDS ${LIBSOFTMIG})
endif()
