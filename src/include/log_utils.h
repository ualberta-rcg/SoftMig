#ifndef __LOG_UTILS_H__
#define __LOG_UTILS_H__

#include <libgen.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

extern FILE *fp1;

// Helper function to get log file path (Compute Canada optimized)
static inline char* get_log_file_path(void) {
    static char log_path[2048] = {0};
    static int initialized = 0;
    
    if (initialized) {
        return log_path;
    }
    
    // Check for custom log path
    char* custom_log = getenv("SOFTMIG_LOG_FILE");
    if (custom_log != NULL && strlen(custom_log) > 0) {
        strncpy(log_path, custom_log, sizeof(log_path) - 1);
        log_path[sizeof(log_path) - 1] = '\0';
        initialized = 1;
        return log_path;
    }
    
    // Use /var/log/softmig/ with job ID (matches config file format: {jobid}_{arrayid}.log or {jobid}.log)
    char* job_id = getenv("SLURM_JOB_ID");
    char* array_id = getenv("SLURM_ARRAY_TASK_ID");
    
    // Build log path: /var/log/softmig/{jobid}_{arrayid}.log or /var/log/softmig/{jobid}.log
    // This matches the config file naming convention
    if (job_id != NULL) {
        if (array_id != NULL && strlen(array_id) > 0) {
            // Array job: {jobid}_{arrayid}.log (matches config file format)
            int written = snprintf(log_path, sizeof(log_path), "/var/log/softmig/%s_%s.log", 
                                   job_id, array_id);
            if (written >= sizeof(log_path)) {
                // Truncation occurred, use fallback to SLURM_TMPDIR
                char* tmpdir = getenv("SLURM_TMPDIR");
                if (tmpdir != NULL) {
                    snprintf(log_path, sizeof(log_path), "%s/softmig_%s_%s.log", tmpdir,
                             job_id, array_id);
                } else {
                    snprintf(log_path, sizeof(log_path), "/var/log/softmig/job_%s_%s.log",
                             job_id, array_id);
                }
            }
        } else {
            // Regular job: {jobid}.log (matches config file format)
            int written = snprintf(log_path, sizeof(log_path), "/var/log/softmig/%s.log", job_id);
            if (written >= sizeof(log_path)) {
                // Truncation occurred, use fallback to SLURM_TMPDIR
                char* tmpdir = getenv("SLURM_TMPDIR");
                if (tmpdir != NULL) {
                    snprintf(log_path, sizeof(log_path), "%s/softmig_%s.log", tmpdir, job_id);
                } else {
                    snprintf(log_path, sizeof(log_path), "/var/log/softmig/job_%s.log", job_id);
                }
            }
        }
    } else {
        // Not in SLURM job - use process ID only (PIDs are unique per process)
        // If PID is reused, old log is from a dead process, so overwriting is fine
        pid_t pid = getpid();
        snprintf(log_path, sizeof(log_path), "/var/log/softmig/pid%d.log", pid);
    }
    
    log_path[sizeof(log_path) - 1] = '\0';
    
    // Create directory if it doesn't exist (try, but don't fail if no permission)
    char dir_path[512];
    size_t log_path_len = strlen(log_path);
    size_t copy_len = (log_path_len < sizeof(dir_path) - 1) ? log_path_len : sizeof(dir_path) - 1;
    memcpy(dir_path, log_path, copy_len);
    dir_path[copy_len] = '\0';
    char* last_slash = strrchr(dir_path, '/');
    if (last_slash != NULL) {
        *last_slash = '\0';
        mkdir(dir_path, 0755);  // Ignore errors - may not have permission
    }
    
    // Fallback to SLURM_TMPDIR only (not regular /tmp) if /var/log not writable
    char* tmpdir = getenv("SLURM_TMPDIR");
    if (tmpdir != NULL) {
        // Test if we can write to /var/log, if not use SLURM_TMPDIR
        FILE* test = fopen(log_path, "a");
        if (test == NULL) {
            // Can't write to /var/log, use SLURM_TMPDIR
            snprintf(log_path, sizeof(log_path), "%s/softmig_%s.log", tmpdir, 
                     job_id ? job_id : "unknown");
            log_path[sizeof(log_path) - 1] = '\0';
        } else {
            fclose(test);
        }
    }
    
    initialized = 1;
    return log_path;
}

// Log level system:
// 0 = errors only (default)
// 1 = errors + warns
// 2 = errors + warns + debug
// 3 = errors + warns + debug + info
// Console logging enabled for level >= 2 (debug mode) for important loops

static inline int get_log_level(void) {
    static int cached_level = -1;
    if (cached_level == -1) {
        char* log_level_str = getenv("SOFTMIG_LOG_LEVEL");
        cached_level = log_level_str ? atoi(log_level_str) : 0;
    }
    return cached_level;
}

static inline void log_to_file_and_console(const char* prefix, const char* msg, ...) {
    va_list args;
    va_start(args, msg);
    
    // Thread-safe basename: copy __FILE__ first since basename() may modify its argument
    // Use local buffer (thread-safe since each thread has its own stack)
    char file_buf[256];
    const char* file_name = __FILE__;
    if (strlen(file_name) < sizeof(file_buf)) {
        strncpy(file_buf, file_name, sizeof(file_buf) - 1);
        file_buf[sizeof(file_buf) - 1] = '\0';
        file_name = basename(file_buf);
    } else {
        // Fallback if path is too long - just use last component manually
        const char* last_slash = strrchr(file_name, '/');
        file_name = (last_slash != NULL) ? (last_slash + 1) : file_name;
    }
    
    // Always log to file
    if (fp1 == NULL) {
        char* log_path = get_log_file_path();
        fp1 = fopen(log_path, "a");
    }
    if (fp1 != NULL) {
        fprintf(fp1, "[softmig %s(%d:%ld:%s:%d)]: ", prefix, getpid(), (long)pthread_self(), file_name, __LINE__);
        vfprintf(fp1, msg, args);
        fprintf(fp1, "\n");
        fflush(fp1);
    }
    
    // Console logging for debug mode (level >= 2)
    int log_level = get_log_level();
    if (log_level >= 2) {
        // CRITICAL: va_list cannot be reused after vfprintf - must use va_copy
        va_list args2;
        va_copy(args2, args);
        fprintf(stderr, "[softmig %s]: ", prefix);
        vfprintf(stderr, msg, args2);
        fprintf(stderr, "\n");
        va_end(args2);
    }
    
    va_end(args);
}

#define LOG_DEBUG(msg, ...) { \
    int log_level = get_log_level(); \
    if (log_level >= 2) { \
        log_to_file_and_console("Debug", msg, ##__VA_ARGS__); \
    } \
}

#define LOG_INFO(msg, ...) { \
    int log_level = get_log_level(); \
    if (log_level >= 3) { \
        log_to_file_and_console("Info", msg, ##__VA_ARGS__); \
    } \
}

#define LOG_WARN(msg, ...) { \
    int log_level = get_log_level(); \
    if (log_level >= 1) { \
        log_to_file_and_console("Warn", msg, ##__VA_ARGS__); \
    } \
}

#define LOG_MSG(msg, ...) { \
    int log_level = get_log_level(); \
    if (log_level >= 1) { \
        log_to_file_and_console("Msg", msg, ##__VA_ARGS__); \
    } \
}

#define LOG_ERROR(msg, ...) { \
    log_to_file_and_console("ERROR", msg, ##__VA_ARGS__); \
}

#define CHECK_DRV_API(f)  {                   \
    CUresult status = (f);                    \
    if (status != CUDA_SUCCESS) {             \
        LOG_WARN("Driver error at %d: %d",   \
            __LINE__, status);                \
        return status;                        \
    } }                                       \

#define CHECK_NVML_API(f)  {                  \
    nvmlReturn_t status = (f);                \
    if (status != NVML_SUCCESS) {             \
        LOG_WARN("NVML error at line %d: %d",    \
            __LINE__, status);                \
        return status;                        \
    } }                                       \

#define CHECK_CU_RESULT(res)  {               \
    if (res != CUDA_SUCCESS) {                \
        LOG_WARN("Driver error at %d: %d",   \
            __LINE__, res);                   \
        return res;                           \
    } }                                       \

#define CHECK_SUCCESS(res) {                  \
    if (res != CUDA_SUCCESS)                  \
        return res;                           \
}

#define IF_CHECK_OOM(res) {                   \
    if (res < 0)                              \
        return CUDA_ERROR_OUT_OF_MEMORY;      \
}     


#endif
