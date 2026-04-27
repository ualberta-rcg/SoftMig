/**
 * @file config_file.c
 * @brief Secure config file reader for per-job GPU limits.
 *
 * Reads memory and SM limits from /var/run/softmig/{jobid}.conf (written by
 * the SLURM prolog). Falls back to environment variables for non-SLURM runs.
 * Results are cached to avoid repeated file I/O.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include "include/log_utils.h"

static int is_numeric(const char *s) {
    if (s == NULL || *s == '\0') return 0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9') return 0;
    }
    return 1;
}

/** Build /var/run/softmig/{jobid}.conf path, validating SLURM_JOB_ID is numeric. */
static char* get_config_file_path(void) {
    static char config_path[512] = {0};
    static int initialized = 0;
    
    if (initialized) {
        return config_path;
    }
    
    char* job_id = getenv("SLURM_JOB_ID");
    char* array_id = getenv("SLURM_ARRAY_TASK_ID");
    
    if (job_id == NULL) {
        config_path[0] = '\0';
        initialized = 1;
        return config_path;
    }

    if (!is_numeric(job_id)) {
        LOG_WARN("SLURM_JOB_ID contains non-numeric characters: '%s', ignoring config file", job_id);
        config_path[0] = '\0';
        initialized = 1;
        return config_path;
    }
    if (array_id != NULL && strlen(array_id) > 0 && !is_numeric(array_id)) {
        LOG_WARN("SLURM_ARRAY_TASK_ID contains non-numeric characters: '%s', ignoring", array_id);
        array_id = NULL;
    }
    
    if (array_id != NULL && strlen(array_id) > 0) {
        snprintf(config_path, sizeof(config_path), "/var/run/softmig/%s_%s.conf", job_id, array_id);
        struct stat st;
        if (lstat(config_path, &st) != 0) {
            snprintf(config_path, sizeof(config_path), "/var/run/softmig/%s.conf", job_id);
        }
    } else {
        snprintf(config_path, sizeof(config_path), "/var/run/softmig/%s.conf", job_id);
    }
    
    initialized = 1;
    return config_path;
}

/** Read a key=value pair from config, with symlink and ownership checks. */
static int read_config_value(const char* key, char* value, size_t value_size) {
    char* config_path = get_config_file_path();
    if (config_path[0] == '\0') {
        return 0;
    }

    struct stat st;
    if (lstat(config_path, &st) != 0) {
        return 0;
    }
    if (S_ISLNK(st.st_mode)) {
        LOG_WARN("Config file is a symlink, refusing to read: %s", config_path);
        return 0;
    }
    if (!S_ISREG(st.st_mode)) {
        LOG_WARN("Config file is not a regular file: %s", config_path);
        return 0;
    }
    if (st.st_uid != 0) {
        LOG_WARN("Config file not owned by root (uid=%u): %s", st.st_uid, config_path);
        return 0;
    }
    
    FILE* f = fopen(config_path, "r");
    if (f == NULL) {
        return 0;
    }
    
    char line[1024];
    size_t key_len = strlen(key);
    
    while (fgets(line, sizeof(line), f) != NULL) {
        // Remove newline
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }
        
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\0') {
            continue;
        }
        
        // Check if this line starts with our key
        if (strncmp(line, key, key_len) == 0 && line[key_len] == '=') {
            // Found it - copy value
            strncpy(value, line + key_len + 1, value_size - 1);
            value[value_size - 1] = '\0';
            fclose(f);
            return 1;
        }
    }
    
    fclose(f);
    return 0;  // Key not found in config file
}

// Cache for config values to avoid reading file multiple times
#define MAX_CACHED_KEYS 4
static struct {
    char key[64];
    size_t value;
    int cached;
} config_cache[MAX_CACHED_KEYS] = {0};

/**
 * Read a GPU limit from the SLURM prolog config file, falling back to env vars.
 *
 * Priority: /var/run/softmig/{jobid}.conf > env var (non-SLURM only).
 * Parses human-readable sizes (e.g. "16g", "24G", "512M"). Results are cached.
 */
size_t get_limit_from_config_or_env(const char* env_name) {
    // Check cache first
    for (int i = 0; i < MAX_CACHED_KEYS; i++) {
        if (config_cache[i].cached && strcmp(config_cache[i].key, env_name) == 0) {
            return config_cache[i].value;
        }
    }
    
    char config_value[256] = {0};
    size_t result = 0;
    
    // Try to read from config file first (if in SLURM job)
    if (read_config_value(env_name, config_value, sizeof(config_value))) {
        // Parse the config value (same format as env var: "16g", "24G", etc.)
        size_t len = strlen(config_value);
        if (len == 0) {
            return 0;
        }
        
        size_t scalar = 1;
        char* digit_end = config_value + len;
        if (config_value[len - 1] == 'G' || config_value[len - 1] == 'g') {
            digit_end -= 1;
            scalar = 1024 * 1024 * 1024;
        } else if (config_value[len - 1] == 'M' || config_value[len - 1] == 'm') {
            digit_end -= 1;
            scalar = 1024 * 1024;
        } else if (config_value[len - 1] == 'K' || config_value[len - 1] == 'k') {
            digit_end -= 1;
            scalar = 1024;
        }
        
        size_t res = strtoul(config_value, &digit_end, 0);
        size_t scaled_res = res * scalar;
        
        if (scaled_res == 0) {
            if (strstr(env_name, "SM_LIMIT") != NULL) {
                LOG_INFO("device core util limit set to 0 from config, which means no limit: %s=%s",
                    env_name, config_value);
            } else {
                LOG_WARN("invalid device memory limit from config %s=%s", env_name, config_value);
            }
            return 0;
        }
        
        if (scaled_res != 0 && scaled_res / scalar != res) {
            LOG_ERROR("Limit overflow from config: %s=%s", env_name, config_value);
            return 0;
        }
        
        LOG_DEBUG("Read %s=%s from config file", env_name, config_value);
        result = scaled_res;
    } else {
        // In SLURM jobs, prolog-generated config files are the source of truth.
        // Do not fall back to user environment variables when a job ID exists.
        if (getenv("SLURM_JOB_ID") != NULL) {
            result = 0;
        } else {
        // Fallback to environment variable for non-SLURM runs (local testing)
        char* env_limit = getenv(env_name);
        if (env_limit == NULL) {
            result = 0;
        } else {
            size_t len = strlen(env_limit);
            if (len == 0) {
                result = 0;
            } else {
                size_t scalar = 1;
                char* digit_end = env_limit + len;
                if (env_limit[len - 1] == 'G' || env_limit[len - 1] == 'g') {
                    digit_end -= 1;
                    scalar = 1024 * 1024 * 1024;
                } else if (env_limit[len - 1] == 'M' || env_limit[len - 1] == 'm') {
                    digit_end -= 1;
                    scalar = 1024 * 1024;
                } else if (env_limit[len - 1] == 'K' || env_limit[len - 1] == 'k') {
                    digit_end -= 1;
                    scalar = 1024;
                }
                
                size_t res = strtoul(env_limit, &digit_end, 0);
                size_t scaled_res = res * scalar;
                
                if (scaled_res == 0) {
                    if (strstr(env_name, "SM_LIMIT") != NULL) {
                        LOG_INFO("device core util limit set to 0, which means no limit: %s=%s",
                            env_name, env_limit);
                    } else if (strstr(env_name, "MEMORY_LIMIT") != NULL) {
                        LOG_WARN("invalid device memory limit %s=%s", env_name, env_limit);
                    } else {
                        LOG_WARN("invalid env name:%s", env_name);
                    }
                    result = 0;
                } else if (scaled_res != 0 && scaled_res / scalar != res) {
                    LOG_ERROR("Limit overflow: %s=%s", env_name, env_limit);
                    result = 0;
                } else {
                    result = scaled_res;
                }
            }
        }
        }
    }
    
    // Cache the result (write key+value before setting the flag visible to readers)
    for (int i = 0; i < MAX_CACHED_KEYS; i++) {
        if (!config_cache[i].cached) {
            strncpy(config_cache[i].key, env_name, sizeof(config_cache[i].key) - 1);
            config_cache[i].key[sizeof(config_cache[i].key) - 1] = '\0';
            config_cache[i].value = result;
            __sync_synchronize();
            config_cache[i].cached = 1;
            break;
        }
    }
    
    return result;
}

/** Return 1 if either CUDA_DEVICE_MEMORY_LIMIT or CUDA_DEVICE_SM_LIMIT is configured. */
int is_softmig_configured(void) {
    // Check if CUDA_DEVICE_MEMORY_LIMIT is set (from config file or environment)
    size_t memory_limit = get_limit_from_config_or_env("CUDA_DEVICE_MEMORY_LIMIT");
    if (memory_limit > 0) {
        return 1;
    }
    
    // Check if CUDA_DEVICE_SM_LIMIT is set (from config file or environment)
    size_t sm_limit = get_limit_from_config_or_env("CUDA_DEVICE_SM_LIMIT");
    if (sm_limit > 0) {
        return 1;
    }
    
    // Neither is set - softmig should be passive
    return 0;
}

