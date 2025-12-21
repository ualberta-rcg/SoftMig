# Unused Functions and Code Blobs Report

This report identifies unused functions and code blocks found in the SoftMig codebase.

## Unused Functions

### 1. `view_vgpu_allocator()` 
- **Location**: `src/allocator/allocator.c:125`
- **Declaration**: `src/allocator/allocator.h:152`
- **Status**: Defined but never called anywhere in the codebase
- **Action**: Can be removed or kept for debugging purposes

### 2. `get_listsize()`
- **Location**: `src/allocator/allocator.c:140`
- **Declaration**: `src/allocator/allocator.h` (not explicitly declared in header, but function exists)
- **Status**: Defined but never called
- **Action**: Can be removed

### 3. `put_device_info()`
- **Location**: `src/multiprocess/multiprocess_memory_limit.h:183`
- **Status**: Declared in header but never implemented (only commented out at line 1447: `//put_device_info();`)
- **Action**: Remove declaration from header

### 4. `initial_virtual_map()`
- **Location**: `src/libsoftmig.c:18`
- **Status**: Declared as `extern void initial_virtual_map(void);` but never implemented
- **Action**: Remove the extern declaration

### 5. `comparelwr()`
- **Location**: `src/multiprocess/multiprocess_memory_limit.c:1740`
- **Declaration**: `src/multiprocess/multiprocess_memory_limit.h:182`
- **Status**: Defined but never called
- **Action**: Can be removed

### 6. `nvml_get_device_memory_usage()`
- **Location**: `src/multiprocess/multiprocess_memory_limit.c:924`
- **Status**: Defined but only referenced in commented-out code (lines 1627, 1641)
- **Action**: Can be removed if the commented code is not needed

### 7. `getallochandle()`
- **Location**: `src/allocator/allocator.h:149`
- **Status**: Declared but never implemented
- **Action**: Remove declaration

### 8. `region_fill()`
- **Location**: Referenced in `src/allocator/allocator.h:130` (macro `INIT_REGION_LIST_ENTRY`)
- **Status**: Called in macro but function is never defined
- **Action**: Either implement the function or remove the macro usage

## Unused Variables

### 1. `real_realpath`
- **Location**: `src/libsoftmig.c:22`
- **Status**: Declared as `char *(*real_realpath)(const char *path, char *resolved_path);` and set to NULL at line 939, but never used
- **Action**: Can be removed

## Commented-Out Code Blocks

### 1. Memory limit macros (commented out)
- **Location**: `src/include/memory_limit.h:64-69`
- **Code**: 
  ```c
  /*
  #define OOM_CHECK()                                       \
      CUdevice dev;                                         \
      CHECK_DRV_API(cuCtxGetDevice(&dev));                  \
      oom_check(dev);
  */
  ```
- **Action**: Remove if not needed

### 2. Commented code in `cuMemFreeHost()`
- **Location**: `src/cuda/memory.c:209-213`
- **Code**: Commented-out variable declarations and checks
- **Action**: Clean up if not needed

### 3. Commented code in `cuMemHostUnregister()`
- **Location**: `src/cuda/memory.c:261-277`
- **Code**: Commented-out memory tracking code
- **Action**: Clean up if not needed

### 4. Commented-out `nvml_get_device_memory_usage()` calls
- **Location**: `src/multiprocess/multiprocess_memory_limit.c:1627, 1641`
- **Code**: `//    result= nvml_get_device_memory_usage(dev);`
- **Action**: Remove if function is not being used

### 5. Commented-out `put_device_info()` call
- **Location**: `src/multiprocess/multiprocess_memory_limit.c:1447`
- **Code**: `//put_device_info();`
- **Action**: Remove if function is not implemented

### 6. Commented-out code in `cuMemcpy2D` functions
- **Location**: `src/cuda/memory.c:777-822`
- **Code**: Multiple commented-out function implementations for non-_v2 versions
- **Action**: Remove if not needed for compatibility

### 7. Commented-out `cuCtxEnablePeerAccess` hook
- **Location**: `src/libsoftmig.c:253`
- **Code**: `//DLSYM_HOOK_FUNC(cuCtxEnablePeerAccess);`
- **Action**: Remove if not needed

### 8. Commented-out `cuGetExportTable` hook
- **Location**: `src/libsoftmig.c:254`
- **Code**: `//DLSYM_HOOK_FUNC(cuGetExportTable);`
- **Action**: Remove if not needed (though `cuGetExportTable` is in the entry list)

### 9. Commented-out `nvmlRetry_NvRmControl` hook
- **Location**: `src/libsoftmig.c:925`
- **Code**: `//DLSYM_HOOK_FUNC(nvmlRetry_NvRmControl);`
- **Action**: Remove if not needed

### 10. Commented-out `nvmlInit()` call
- **Location**: `src/libsoftmig.c:941`
- **Code**: `//nvmlInit();`
- **Action**: Remove if not needed

### 11. Commented-out `add_gpu_device_memory_usage` call
- **Location**: `src/libsoftmig.c:959`
- **Code**: `//add_gpu_device_memory_usage(getpid(),0,context_size,0);`
- **Action**: Remove if not needed

### 12. Commented-out `cuCtxGetDevice` hook
- **Location**: `src/libsoftmig.c:221`
- **Code**: `//DLSYM_HOOK_FUNC(cuCtxGetDevice);`
- **Action**: Remove if not needed (though `cuCtxGetDevice` is hooked elsewhere)

### 13. Commented-out code in `check_oom()`
- **Location**: `src/cuda/memory.c:55-56`
- **Code**: `//    return 0;`
- **Action**: Remove if not needed

### 14. Commented-out `memory_limit.h` include
- **Location**: `src/libsoftmig.c:1`
- **Code**: `//#include "memory_limit.h"`
- **Action**: Remove if not needed

## Duplicate Code

### 1. Duplicate `--print` argument handling
- **Location**: `src/multiprocess/shrreg_tool.c:71, 77`
- **Issue**: The `--print` argument is handled twice in the same loop
- **Action**: Remove duplicate

## Potentially Unused Functions (Need Verification)

These functions are defined and may be used externally or in ways not easily detected:

1. `get_current_device_memory_monitor()` - May be used by external tools
2. `set_gpu_device_memory_monitor()` - May be used by external tools
3. `get_gpu_memory_monitor()` - Used internally by `get_current_device_memory_monitor()`

## Recommendations

1. **Remove unused functions** to reduce code complexity and maintenance burden
2. **Clean up commented-out code** - either implement it, document why it's commented, or remove it
3. **Remove duplicate code** in `shrreg_tool.c`
4. **Verify external API usage** before removing functions that might be part of a public API
5. **Consider keeping debugging functions** like `view_vgpu_allocator()` if they're useful for troubleshooting

## Summary Statistics

- **Unused Functions**: 8
- **Unused Variables**: 1
- **Commented-Out Code Blocks**: 14+
- **Duplicate Code**: 1 instance
- **Potentially Unused (Need Verification)**: 3

