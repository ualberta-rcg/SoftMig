#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <time.h>
#include <errno.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>

#include "test_utils.h"

// Maximum number of processes to query
#define MAX_PROCESSES 64

// Retry timeout in seconds
#define RETRY_TIMEOUT_SEC 2

// Number of child processes to spawn for multi-process test
#define NUM_CHILD_PROCESSES 3

// Child mode argv[1] marker
#define CHILD_MODE_ARG "--child"

// Function pointer type for raw NVML calls (for dlsym)
typedef nvmlReturn_t (*nvmlDeviceGetComputeRunningProcesses_fn)(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);

// Get current process PID
static pid_t get_current_pid(void) {
    return getpid();
}

// Check if a PID is in the process info array
static int pid_in_list(unsigned int pid, nvmlProcessInfo_t *infos, unsigned int count) {
    for (unsigned int i = 0; i < count; i++) {
        if (infos[i].pid == pid) {
            return 1;
        }
    }
    return 0;
}

// Wait for current process to appear in NVML process list (with timeout)
static int wait_for_process_in_list(nvmlDevice_t device, pid_t current_pid, 
                                     nvmlProcessInfo_t *infos, unsigned int *count) {
    time_t start_time = time(NULL);
    time_t current_time;
    int attempt = 0;
    
    printf("  DEBUG: Looking for PID %d in NVML process list...\n", current_pid);
    
    do {
        *count = MAX_PROCESSES;
        
        nvmlReturn_t ret = nvmlDeviceGetComputeRunningProcesses(device, count, infos);
        attempt++;
        
        printf("  DEBUG: Attempt %d: nvmlDeviceGetComputeRunningProcesses returned %d, found %u processes\n",
               attempt, ret, *count);
        
        if (ret == NVML_SUCCESS || ret == NVML_ERROR_INSUFFICIENT_SIZE) {
            // Print all PIDs found
            printf("  DEBUG: PIDs in NVML list: ");
            for (unsigned int i = 0; i < *count; i++) {
                printf("%u", infos[i].pid);
                if (infos[i].pid == current_pid) {
                    printf("(TARGET!)");
                }
                printf(" [mem: %llu bytes]", (unsigned long long)infos[i].usedGpuMemory);
                if (i < *count - 1) printf(", ");
            }
            printf("\n");
            
            if (pid_in_list(current_pid, infos, *count)) {
                printf("  DEBUG: ✓ Found target PID %d in NVML list!\n", current_pid);
                return 1; // Found!
            } else {
                printf("  DEBUG: Target PID %d NOT in list, retrying...\n", current_pid);
            }
        } else {
            printf("  DEBUG: NVML query failed with error %d, retrying...\n", ret);
        }
        
        // Sleep briefly before retry
        usleep(100000); // 100ms
        
        current_time = time(NULL);
    } while ((current_time - start_time) < RETRY_TIMEOUT_SEC);
    
    printf("  DEBUG: Timeout after %d attempts - target PID %d not found\n", 
           attempt, current_pid);
    return 0; // Timeout
}

// Query raw NVML driver (bypassing SoftMig interposition) via dlopen/dlsym
static int query_raw_nvml_processes(nvmlDevice_t device, 
                                     nvmlProcessInfo_t *raw_infos, unsigned int *raw_count) {
    printf("  DEBUG: Attempting to dlopen libnvidia-ml.so.1...\n");
    void *nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!nvml_handle) {
        fprintf(stderr, "  DEBUG: Could not dlopen libnvidia-ml.so.1: %s\n", dlerror());
        return 0;
    }
    printf("  DEBUG: ✓ Successfully opened libnvidia-ml.so.1\n");
    
    // Get the function pointer for nvmlDeviceGetComputeRunningProcesses
    printf("  DEBUG: Looking for nvmlDeviceGetComputeRunningProcesses symbol...\n");
    nvmlDeviceGetComputeRunningProcesses_fn raw_get_processes = 
        (nvmlDeviceGetComputeRunningProcesses_fn)dlsym(nvml_handle, 
                                                       "nvmlDeviceGetComputeRunningProcesses");
    
    if (!raw_get_processes) {
        fprintf(stderr, "  DEBUG: Could not dlsym nvmlDeviceGetComputeRunningProcesses: %s\n", dlerror());
        dlclose(nvml_handle);
        return 0;
    }
    printf("  DEBUG: ✓ Found nvmlDeviceGetComputeRunningProcesses function\n");
    
    *raw_count = MAX_PROCESSES;
    printf("  DEBUG: Calling raw nvmlDeviceGetComputeRunningProcesses (bypassing SoftMig)...\n");
    nvmlReturn_t ret = raw_get_processes(device, raw_count, raw_infos);
    
    printf("  DEBUG: Raw NVML call returned: %d, process count: %u\n", ret, *raw_count);
    
    dlclose(nvml_handle);
    
    if (ret == NVML_SUCCESS || ret == NVML_ERROR_INSUFFICIENT_SIZE) {
        printf("  DEBUG: ✓ Raw NVML query successful\n");
        return 1;
    }
    
    printf("  DEBUG: Raw NVML query failed with error %d\n", ret);
    return 0;
}

// Wait for multiple PIDs to appear in NVML process list (with timeout)
static int wait_for_multiple_processes_in_list(nvmlDevice_t device, pid_t *pids, int num_pids,
                                                 nvmlProcessInfo_t *infos, unsigned int *count) {
    time_t start_time = time(NULL);
    time_t current_time;
    int found_count = 0;
    int *found = calloc(num_pids, sizeof(int));
    if (!found) {
        fprintf(stderr, "ERROR: Failed to allocate memory for found array\n");
        return 0;
    }
    
    printf("  DEBUG: Looking for %d PIDs: ", num_pids);
    for (int i = 0; i < num_pids; i++) {
        printf("%d%s", pids[i], (i < num_pids - 1) ? ", " : "\n");
    }
    
    int attempt = 0;
    do {
        *count = MAX_PROCESSES;
        
        nvmlReturn_t ret = nvmlDeviceGetComputeRunningProcesses(device, count, infos);
        attempt++;
        
        printf("  DEBUG: Attempt %d: nvmlDeviceGetComputeRunningProcesses returned %d, found %u processes\n",
               attempt, ret, *count);
        
        if (ret == NVML_SUCCESS || ret == NVML_ERROR_INSUFFICIENT_SIZE) {
            // Print all PIDs found
            printf("  DEBUG: PIDs in NVML list: ");
            for (unsigned int i = 0; i < *count; i++) {
                printf("%u [mem: %llu bytes]", 
                       infos[i].pid,
                       (unsigned long long)infos[i].usedGpuMemory);
                
                // Check if this is one of our target PIDs
                int is_target = 0;
                for (int j = 0; j < num_pids; j++) {
                    if (infos[i].pid == pids[j]) {
                        printf("(TARGET-%d!)", j);
                        is_target = 1;
                        break;
                    }
                }
                
                if (i < *count - 1) printf(", ");
            }
            printf("\n");
            
            found_count = 0;
            // Check which PIDs are found
            for (int i = 0; i < num_pids; i++) {
                if (!found[i] && pid_in_list(pids[i], infos, *count)) {
                    found[i] = 1;
                    found_count++;
                    printf("  DEBUG: ✓ Found target PID %d (target %d/%d)\n", 
                           pids[i], found_count, num_pids);
                }
            }
            
            printf("  DEBUG: Found %d/%d target processes so far\n", found_count, num_pids);
            
            if (found_count == num_pids) {
                printf("  DEBUG: ✓ All %d target processes found!\n", num_pids);
                free(found);
                return 1; // All found!
            } else {
                printf("  DEBUG: Still missing %d processes, retrying...\n", num_pids - found_count);
            }
        } else {
            printf("  DEBUG: NVML query failed with error %d, retrying...\n", ret);
        }
        
        // Sleep briefly before retry
        usleep(100000); // 100ms
        
        current_time = time(NULL);
    } while ((current_time - start_time) < RETRY_TIMEOUT_SEC);
    
    // Print which PIDs were not found
    printf("  DEBUG: Timeout after %d attempts. Final status:\n", attempt);
    printf("  Found %d/%d processes:\n", found_count, num_pids);
    for (int i = 0; i < num_pids; i++) {
        printf("    PID %d: %s\n", pids[i], found[i] ? "FOUND" : "NOT FOUND");
    }
    
    free(found);
    return 0; // Timeout
}

// Child process function: allocate GPU memory and keep it allocated
static void child_process_gpu_worker(int device_id, size_t alloc_size) {
    void *gpu_mem = NULL;
    cudaError_t ret;
    
    printf("  DEBUG: Child PID %d starting, will allocate %zu bytes...\n", 
           getpid(), alloc_size);
    
    // Initialize CUDA
    printf("  DEBUG: Child PID %d initializing CUDA...\n", getpid());
    ret = cudaSetDevice(device_id);
    if (ret != cudaSuccess) {
        fprintf(stderr, "  DEBUG: Child PID %d: cudaSetDevice(%d) failed: %d\n",
                getpid(), device_id, ret);
        exit(1);
    }
    ret = cudaFree(0);
    if (ret != cudaSuccess) {
        fprintf(stderr, "  DEBUG: Child PID %d: cudaFree(0) failed: %d\n", getpid(), ret);
        exit(1);
    }
    printf("  DEBUG: Child PID %d: CUDA initialized\n", getpid());
    
    // Allocate GPU memory
    printf("  DEBUG: Child PID %d allocating %zu bytes...\n", getpid(), alloc_size);
    ret = cudaMalloc(&gpu_mem, alloc_size);
    if (ret != cudaSuccess) {
        fprintf(stderr, "  DEBUG: Child PID %d: cudaMalloc(%zu) failed: %d\n", 
                getpid(), alloc_size, ret);
        exit(1);
    }
    // Touch the allocation to make GPU usage unambiguous
    ret = cudaMemset(gpu_mem, 0, alloc_size);
    if (ret != cudaSuccess) {
        fprintf(stderr, "  DEBUG: Child PID %d: cudaMemset(%p, 0, %zu) failed: %d\n",
                getpid(), gpu_mem, alloc_size, ret);
        exit(1);
    }
    ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess) {
        fprintf(stderr, "  DEBUG: Child PID %d: cudaDeviceSynchronize failed: %d\n", getpid(), ret);
        exit(1);
    }
    
    printf("  DEBUG: ✓ Child PID %d: Successfully allocated %zu bytes at %p\n", 
           getpid(), alloc_size, gpu_mem);
    printf("Child PID %d: Allocated %zu bytes on GPU, keeping it allocated for 10 seconds...\n", 
           getpid(), alloc_size);
    
    // Keep the memory allocated and sleep (so process stays visible to NVML)
    sleep(10); // Sleep for 10 seconds
    
    printf("  DEBUG: Child PID %d: Freeing GPU memory and exiting...\n", getpid());
    // Clean up
    cudaFree(gpu_mem);
    exit(0);
}

static int is_child_mode(int argc, char **argv) {
    return (argc >= 4 && argv[1] && strcmp(argv[1], CHILD_MODE_ARG) == 0);
}

static int parse_int_arg(const char *s, int *out) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || !end || *end != '\0') return 0;
    if (v < INT_MIN || v > INT_MAX) return 0;
    *out = (int)v;
    return 1;
}

static int parse_size_arg(const char *s, size_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || !end || *end != '\0') return 0;
    *out = (size_t)v;
    return 1;
}

static const char *get_self_exe_path(char *buf, size_t buf_len, const char *argv0) {
    if (buf && buf_len > 0) {
        ssize_t n = readlink("/proc/self/exe", buf, buf_len - 1);
        if (n > 0) {
            buf[n] = '\0';
            return buf;
        }
    }
    // Fallback: argv[0] (may be relative, but usually works)
    return argv0 ? argv0 : "./test_nvml_process_filter";
}

// Spawn a child process that uses the GPU
static pid_t spawn_gpu_child_process(const char *self_exe, int device_id, size_t alloc_size) {
    pid_t pid = fork();
    
    if (pid < 0) {
        fprintf(stderr, "ERROR: fork() failed\n");
        return -1;
    }
    
    if (pid == 0) {
        // Child process: exec a fresh instance to avoid CUDA runtime fork issues
        char dev_str[32];
        char size_str[32];
        (void)snprintf(dev_str, sizeof(dev_str), "%d", device_id);
        (void)snprintf(size_str, sizeof(size_str), "%zu", alloc_size);

        execl(self_exe, self_exe, CHILD_MODE_ARG, dev_str, size_str, (char *)NULL);
        // If exec fails, report and exit
        fprintf(stderr, "ERROR: execl(%s) failed in child: %s\n", self_exe, strerror(errno));
        _exit(127);
    }
    
    // Parent process
    return pid;
}

// Kill all child processes
static void kill_child_processes(pid_t *pids, int num_pids) {
    for (int i = 0; i < num_pids; i++) {
        if (pids[i] > 0) {
            printf("Killing child process PID %d...\n", pids[i]);
            kill(pids[i], SIGTERM);
            // Wait for it to exit
            int status;
            waitpid(pids[i], &status, 0);
        }
    }
}

// Check if filtered PIDs are a subset of raw PIDs
static void compare_filtered_vs_raw(nvmlProcessInfo_t *filtered_infos, unsigned int filtered_count,
                                     nvmlProcessInfo_t *raw_infos, unsigned int raw_count) {
    printf("Comparing filtered vs raw NVML process lists:\n");
    printf("  Filtered processes: %u\n", filtered_count);
    printf("  Raw processes: %u\n", raw_count);
    
    printf("\n  DEBUG: Filtered PIDs: ");
    for (unsigned int i = 0; i < filtered_count; i++) {
        printf("%u [mem: %llu]%s", 
               filtered_infos[i].pid,
               (unsigned long long)filtered_infos[i].usedGpuMemory,
               (i < filtered_count - 1) ? ", " : "\n");
    }
    
    printf("  DEBUG: Raw PIDs: ");
    for (unsigned int i = 0; i < raw_count; i++) {
        printf("%u [mem: %llu]%s", 
               raw_infos[i].pid,
               (unsigned long long)raw_infos[i].usedGpuMemory,
               (i < raw_count - 1) ? ", " : "\n");
    }
    
    if (raw_count > filtered_count) {
        printf("  ✓ Filtering detected: %u processes filtered out\n", raw_count - filtered_count);
        
        // Show which PIDs were filtered (with details)
        printf("  Filtered out PIDs:\n");
        for (unsigned int i = 0; i < raw_count; i++) {
            if (!pid_in_list(raw_infos[i].pid, filtered_infos, filtered_count)) {
                printf("    - PID %u (mem: %llu bytes)\n", 
                       raw_infos[i].pid,
                       (unsigned long long)raw_infos[i].usedGpuMemory);
            }
        }
    } else if (raw_count == filtered_count) {
        printf("  Note: No filtering detected (same count)\n");
    } else {
        printf("  WARNING: Filtered count (%u) > raw count (%u) - unexpected!\n", 
               filtered_count, raw_count);
    }
    
    // Verify all filtered PIDs are in raw list
    int validation_failed = 0;
    for (unsigned int i = 0; i < filtered_count; i++) {
        if (!pid_in_list(filtered_infos[i].pid, raw_infos, raw_count)) {
            fprintf(stderr, "  ERROR: Filtered PID %u not found in raw list!\n", filtered_infos[i].pid);
            validation_failed = 1;
        }
    }
    
    if (!validation_failed) {
        printf("  ✓ Validation passed: All filtered PIDs are present in raw list\n");
    }
}

int main(int argc, char **argv) {
    // Child mode: just do GPU work and exit. Must happen before NVML/CUDA init in parent path.
    if (is_child_mode(argc, argv)) {
        // Make stdout/stderr unbuffered so the parent sees logs quickly.
        setvbuf(stdout, NULL, _IONBF, 0);
        setvbuf(stderr, NULL, _IONBF, 0);

        int device_id = 0;
        size_t alloc_size = 0;
        if (!parse_int_arg(argv[2], &device_id) || !parse_size_arg(argv[3], &alloc_size)) {
            fprintf(stderr, "ERROR: Invalid child args. Usage: %s %s <device_id> <alloc_bytes>\n",
                    argv[0], CHILD_MODE_ARG);
            return 2;
        }
        child_process_gpu_worker(device_id, alloc_size);
        return 0; // Not reached
    }

    pid_t current_pid = get_current_pid();
    nvmlReturn_t ret;
    unsigned int device_count = 0;
    nvmlDevice_t device;
    nvmlProcessInfo_t filtered_infos[MAX_PROCESSES];
    unsigned int filtered_count = 0;
    nvmlProcessInfo_t raw_infos[MAX_PROCESSES];
    unsigned int raw_count = 0;
    int found_current_pid = 0;
    
    printf("NVML Process Filter Test\n");
    printf("Current PID: %d\n", current_pid);
    printf("Test device ID: %d\n", TEST_DEVICE_ID);
    
    // Initialize NVML
    ret = nvmlInit();
    if (ret == NVML_ERROR_DRIVER_NOT_LOADED || 
        ret == NVML_ERROR_LIBRARY_NOT_FOUND ||
        ret == NVML_ERROR_NO_PERMISSION) {
        printf("SKIP: NVML not available (error %d: %s)\n", ret, nvmlErrorString(ret));
        return 0;
    }
    
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "ERROR: nvmlInit failed: %d (%s)\n", ret, nvmlErrorString(ret));
        return 1;
    }
    
    // Get device count
    ret = nvmlDeviceGetCount(&device_count);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "ERROR: nvmlDeviceGetCount failed: %d (%s)\n", ret, nvmlErrorString(ret));
        nvmlShutdown();
        return 1;
    }
    
    if (device_count == 0) {
        printf("SKIP: No GPU devices found\n");
        nvmlShutdown();
        return 0;
    }
    
    if (TEST_DEVICE_ID >= device_count) {
        fprintf(stderr, "ERROR: Test device ID %d >= device count %u\n", TEST_DEVICE_ID, device_count);
        nvmlShutdown();
        return 1;
    }
    
    // Get device handle
    ret = nvmlDeviceGetHandleByIndex(TEST_DEVICE_ID, &device);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "ERROR: nvmlDeviceGetHandleByIndex failed: %d (%s)\n", ret, nvmlErrorString(ret));
        nvmlShutdown();
        return 1;
    }
    
    printf("Device count: %u\n", device_count);
    printf("Using device %d\n", TEST_DEVICE_ID);
    
    // Initialize CUDA to make this process visible to NVML
    printf("Initializing CUDA context...\n");
    cudaError_t cuda_ret = cudaFree(0); // Initialize CUDA runtime
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "WARNING: cudaFree(0) failed: %d\n", cuda_ret);
    }
    
    // Allocate a small buffer to ensure we're using the GPU
    void *dummy_ptr = NULL;
    cuda_ret = cudaMalloc(&dummy_ptr, 1024);
    if (cuda_ret != cudaSuccess) {
        fprintf(stderr, "WARNING: cudaMalloc failed: %d\n", cuda_ret);
    } else {
        cudaFree(dummy_ptr);
    }
    
    printf("Waiting for process to appear in NVML process list (max %d seconds)...\n", RETRY_TIMEOUT_SEC);
    
    // Query filtered process list (via SoftMig if active)
    if (!wait_for_process_in_list(device, current_pid, filtered_infos, &filtered_count)) {
        fprintf(stderr, "ERROR: Current process (PID %d) not found in filtered NVML process list after %d seconds\n",
                current_pid, RETRY_TIMEOUT_SEC);
        fprintf(stderr, "Found %u processes:\n", filtered_count);
        for (unsigned int i = 0; i < filtered_count; i++) {
            fprintf(stderr, "  PID %u, memory %llu bytes\n", 
                    filtered_infos[i].pid, 
                    (unsigned long long)filtered_infos[i].usedGpuMemory);
        }
        nvmlShutdown();
        return 1;
    }
    
    printf("✓ Current process (PID %d) found in filtered NVML process list\n", current_pid);
    printf("  Total filtered processes: %u\n", filtered_count);
    
    // Print all filtered processes
    printf("Filtered process list:\n");
    for (unsigned int i = 0; i < filtered_count; i++) {
        int is_current = (filtered_infos[i].pid == current_pid);
        printf("  [%u] PID %u, memory %llu bytes%s\n",
               i, filtered_infos[i].pid,
               (unsigned long long)filtered_infos[i].usedGpuMemory,
               is_current ? " (THIS PROCESS)" : "");
    }
    
    // Try to query raw NVML driver (bypassing SoftMig) for comparison
    printf("\nQuerying raw NVML driver (bypassing SoftMig interposition)...\n");
    if (query_raw_nvml_processes(device, raw_infos, &raw_count)) {
        printf("✓ Raw NVML query successful\n");
        compare_filtered_vs_raw(filtered_infos, filtered_count, raw_infos, raw_count);
    } else {
        printf("  (Could not query raw NVML - this is OK if dlopen/dlsym fails)\n");
    }
    
    // ========== Multi-Process Test ==========
    printf("\n========== Multi-Process Test ==========\n");
    printf("Spawning %d child processes that will use the GPU...\n", NUM_CHILD_PROCESSES);
    
    pid_t child_pids[NUM_CHILD_PROCESSES];
    int all_children_spawned = 1;

    char self_exe_buf[PATH_MAX];
    const char *self_exe = get_self_exe_path(self_exe_buf, sizeof(self_exe_buf), argv[0]);
    
    // Spawn child processes
    for (int i = 0; i < NUM_CHILD_PROCESSES; i++) {
        size_t alloc_size = (1 + i) * 1024 * 1024; // 1MB, 2MB, 3MB, etc.
        child_pids[i] = spawn_gpu_child_process(self_exe, TEST_DEVICE_ID, alloc_size);
        if (child_pids[i] < 0) {
            fprintf(stderr, "ERROR: Failed to spawn child process %d\n", i);
            all_children_spawned = 0;
            // Kill any already-spawned children
            for (int j = 0; j < i; j++) {
                if (child_pids[j] > 0) {
                    kill(child_pids[j], SIGTERM);
                    waitpid(child_pids[j], NULL, 0);
                }
            }
            break;
        }
        printf("  Spawned child process %d: PID %d\n", i, child_pids[i]);
    }
    
    if (all_children_spawned) {
        // Wait briefly for children to initialize CUDA
        printf("Waiting for children to initialize CUDA...\n");
        sleep(1);
        
        // Create array of all PIDs to check (parent + children)
        pid_t all_pids[NUM_CHILD_PROCESSES + 1];
        all_pids[0] = current_pid;
        for (int i = 0; i < NUM_CHILD_PROCESSES; i++) {
            all_pids[i + 1] = child_pids[i];
        }
        
        // Wait for all processes to appear in NVML
        printf("Waiting for all %d processes to appear in NVML process list (max %d seconds)...\n",
               NUM_CHILD_PROCESSES + 1, RETRY_TIMEOUT_SEC);
        
        if (wait_for_multiple_processes_in_list(device, all_pids, NUM_CHILD_PROCESSES + 1, 
                                                  filtered_infos, &filtered_count)) {
            printf("✓ All %d processes found in NVML process list!\n", NUM_CHILD_PROCESSES + 1);
            
            // Print the filtered process list
            printf("Multi-process filtered list:\n");
            for (unsigned int i = 0; i < filtered_count; i++) {
                int is_parent = (filtered_infos[i].pid == current_pid);
                int is_child = 0;
                for (int j = 0; j < NUM_CHILD_PROCESSES; j++) {
                    if (filtered_infos[i].pid == child_pids[j]) {
                        is_child = 1;
                        break;
                    }
                }
                
                printf("  [%u] PID %u, memory %llu bytes",
                       i, filtered_infos[i].pid,
                       (unsigned long long)filtered_infos[i].usedGpuMemory);
                
                if (is_parent) {
                    printf(" (PARENT)");
                } else if (is_child) {
                    printf(" (CHILD)");
                }
                printf("\n");
            }
            
            // Try raw NVML query again for comparison
            printf("\nQuerying raw NVML with child processes active...\n");
            if (query_raw_nvml_processes(device, raw_infos, &raw_count)) {
                printf("✓ Raw NVML query successful\n");
                compare_filtered_vs_raw(filtered_infos, filtered_count, raw_infos, raw_count);
            }
            
        } else {
            fprintf(stderr, "ERROR: Not all processes found in NVML process list!\n");
            fprintf(stderr, "Expected %d processes (1 parent + %d children)\n",
                    NUM_CHILD_PROCESSES + 1, NUM_CHILD_PROCESSES);
            all_children_spawned = 0; // Mark as failed
        }
        
        // Kill all child processes
        printf("Cleaning up child processes...\n");
        kill_child_processes(child_pids, NUM_CHILD_PROCESSES);
        printf("✓ All child processes terminated\n");
    }
    
    // Final validation: ensure current PID is in filtered list
    found_current_pid = pid_in_list(current_pid, filtered_infos, filtered_count);
    if (!found_current_pid) {
        fprintf(stderr, "ERROR: Current process (PID %d) not in final filtered list!\n", current_pid);
        nvmlShutdown();
        return 1;
    }
    
    if (!all_children_spawned) {
        fprintf(stderr, "\nWARNING: Multi-process test failed (could not spawn/find all children)\n");
        fprintf(stderr, "Single-process test passed, but multi-process validation incomplete.\n");
        nvmlShutdown();
        return 1;
    }
    
    printf("\n✓ All Tests PASSED!\n");
    printf("  Single-process test: Current process (PID %d) visible in NVML\n", current_pid);
    printf("  Multi-process test: All %d processes (1 parent + %d children) visible in NVML\n",
           NUM_CHILD_PROCESSES + 1, NUM_CHILD_PROCESSES);
    
    nvmlShutdown();
    return 0;
}

