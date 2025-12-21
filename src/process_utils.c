#include "include/process_utils.h"
#include "include/log_utils.h"
#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h>
#include <string.h>
#include <unistd.h>
// Include nvml-subset.h for nvmlProcessInfo_t definition
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include "include/nvml-subset.h"

// Get the UID of a process by PID (returns -1 on error)
uid_t proc_get_uid(int32_t pid) {
    char filename[FILENAME_LENGTH] = {0};
    snprintf(filename, sizeof(filename), "/proc/%d/status", pid);
    
    FILE* fp;
    if ((fp = fopen(filename, "r")) == NULL) {
        LOG_WARN("proc_get_uid: PID %d - failed to open /proc/%d/status, errno=%d", pid, pid, errno);
        return (uid_t)-1;
    }
    
    char line[BUFFER_LENGTH];
    uid_t uid = (uid_t)-1;
    unsigned int real_uid = 0, effective_uid = 0, saved_uid = 0, filesystem_uid = 0;
    
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strncmp(line, "Uid:", 4) == 0) {
            // Format: "Uid:    1000    1000    1000    1000"
            // First value is real UID
            if (sscanf(line, "Uid: %u %u %u %u", &real_uid, &effective_uid, &saved_uid, &filesystem_uid) >= 1) {
                uid = (uid_t)real_uid;
                LOG_DEBUG("proc_get_uid: PID %d - UID read: real=%u effective=%u saved=%u fs=%u", 
                         pid, real_uid, effective_uid, saved_uid, filesystem_uid);
                break;
            } else {
                LOG_WARN("proc_get_uid: PID %d - failed to parse Uid line: %s", pid, line);
            }
        }
    }
    
    fclose(fp);
    if (uid == (uid_t)-1) {
        LOG_WARN("proc_get_uid: PID %d - UID not found in /proc/%d/status", pid, pid);
    }
    return uid;
}

int proc_alive(int32_t pid) {
    char filename[FILENAME_LENGTH] = {0};
    sprintf(filename, "/proc/%d/stat", pid);

    FILE* fp;
    if ((fp = fopen(filename, "r")) == NULL) {   
        return PROC_STATE_NONALIVE;
    }

    int __pid;
    char state;
    char name_buf[BUFFER_LENGTH] = {0};
    int num = fscanf(fp, "%d %s %c", &__pid, name_buf, &state);
    int res;
    if (num != 3 || num == EOF) {
        res = PROC_STATE_UNKNOWN;
    } else if (state == 'Z' || state == 'X' || state == 'x') {
        res = PROC_STATE_NONALIVE;
    } else {
        res = PROC_STATE_ALIVE;
    }
    fclose(fp);
    return res;
}

// Extract cgroup path from a cgroup line
// Supports both cgroups v1 and v2 formats:
// v1: <id>:<controller>:<path>
// v2: 0::<path>
// Returns the path portion, or NULL if not found
// Caller must free the returned string if not NULL
static char* extract_cgroup_path(const char* line) {
    char* path = NULL;
    
    // Try cgroups v2 format first: "0::<path>"
    if (strncmp(line, "0::", 3) == 0) {
        const char* v2_path = line + 3;
        // Skip leading slash if present
        if (*v2_path == '/') {
            v2_path++;
        }
        size_t len = strlen(v2_path);
        // Remove trailing newline
        if (len > 0 && v2_path[len - 1] == '\n') {
            len--;
        }
        if (len > 0) {
            path = (char*)malloc(len + 1);
            if (path != NULL) {
                strncpy(path, v2_path, len);
                path[len] = '\0';
            }
        }
        return path;
    }
    
    // Try cgroups v1 format: "<id>:<controller>:<path>"
    // Find the last colon (separates controller from path)
    const char* last_colon = strrchr(line, ':');
    if (last_colon != NULL && last_colon > line) {
        const char* v1_path = last_colon + 1;
        // Skip leading slash if present
        if (*v1_path == '/') {
            v1_path++;
        }
        size_t len = strlen(v1_path);
        // Remove trailing newline
        if (len > 0 && v1_path[len - 1] == '\n') {
            len--;
        }
        if (len > 0) {
            path = (char*)malloc(len + 1);
            if (path != NULL) {
                strncpy(path, v1_path, len);
                path[len] = '\0';
            }
        }
    }
    
    return path;
}

// Get the cgroup path for a process
// Returns NULL on error, caller must free the returned string if not NULL
static char* proc_get_cgroup_path(int32_t pid) {
    char filename[FILENAME_LENGTH] = {0};
    snprintf(filename, sizeof(filename), "/proc/%d/cgroup", pid);
    
    FILE* fp;
    if ((fp = fopen(filename, "r")) == NULL) {
        return NULL;
    }
    
    char line[BUFFER_LENGTH];
    char* cgroup_path = NULL;
    
    // Read each line in the cgroup file
    // We'll use the first valid cgroup path we find
    while (fgets(line, sizeof(line), fp) != NULL) {
        char* path = extract_cgroup_path(line);
        if (path != NULL) {
            cgroup_path = path;
            break;  // Found a valid path, stop searching
        }
    }
    
    fclose(fp);
    return cgroup_path;
}

// Extract job/session identifier from cgroup path
// Looks for patterns like "job_<ID>" in the path
// Returns NULL if no job identifier found, caller must free if not NULL
static char* extract_job_id_from_path(const char* cgroup_path) {
    if (cgroup_path == NULL) {
        return NULL;
    }
    
    // Look for "job_" pattern in the path
    char* job_pos = strstr(cgroup_path, "job_");
    if (job_pos == NULL) {
        return NULL;
    }
    
    // Extract job ID (skip "job_")
    job_pos += 4;  // Skip "job_"
    
    // Find the end of the job ID (either '/' or end of string)
    char* job_end = job_pos;
    while (*job_end != '\0' && *job_end != '/') {
        job_end++;
    }
    
    if (job_end > job_pos) {
        // Allocate memory for job ID
        size_t job_id_len = job_end - job_pos;
        char* job_id = (char*)malloc(job_id_len + 1);
        if (job_id != NULL) {
            strncpy(job_id, job_pos, job_id_len);
            job_id[job_id_len] = '\0';
        }
        return job_id;
    }
    
    return NULL;
}

// Check if a process belongs to the same cgroup session as the current process
// Returns 1 if process belongs to same session, 0 if not, -1 on error
// Falls back to -1 if cgroups cannot be determined (should use UID filtering)
int proc_belongs_to_current_cgroup_session(int32_t pid) {
    pid_t current_pid = getpid();
    uid_t current_uid = getuid();
    
    // Get current process's cgroup path
    char* current_cgroup = proc_get_cgroup_path(current_pid);
    if (current_cgroup == NULL) {
        // Cannot determine current cgroup - fall back to UID check
        LOG_DEBUG("proc_belongs_to_current_cgroup_session: Current PID %d (UID %u) - cannot determine cgroup, returning -1 (fallback to UID)", 
                 current_pid, current_uid);
        return -1;
    }
    
    // Get target process's cgroup path
    char* proc_cgroup = proc_get_cgroup_path(pid);
    if (proc_cgroup == NULL) {
        free(current_cgroup);
        // Cannot determine process cgroup - fall back to UID check
        LOG_DEBUG("proc_belongs_to_current_cgroup_session: Target PID %d - cannot determine cgroup, returning -1 (fallback to UID)", pid);
        return -1;
    }
    
    // Try to extract job IDs from both paths
    char* current_job_id = extract_job_id_from_path(current_cgroup);
    char* proc_job_id = extract_job_id_from_path(proc_cgroup);
    
    int result = -1;
    
    if (current_job_id != NULL && proc_job_id != NULL) {
        // Both have job IDs - compare them
        int job_match = (strcmp(current_job_id, proc_job_id) == 0);
        result = job_match ? 1 : 0;
    } else if (current_job_id == NULL && proc_job_id == NULL) {
        // Neither has a job ID - compare full cgroup paths
        // Check if they share the same parent path (up to the job level)
        // For now, do a simple string comparison of the paths
        // This will match if they're in the same cgroup hierarchy
        int path_match = (strcmp(current_cgroup, proc_cgroup) == 0);
        result = path_match ? 1 : 0;
    }
    // If one has a job ID and the other doesn't, they're different (result stays -1)
    
    free(current_cgroup);
    free(proc_cgroup);
    if (current_job_id != NULL) {
        free(current_job_id);
    }
    if (proc_job_id != NULL) {
        free(proc_job_id);
    }
    
    return result;
}

/**
 * Safely extract PID from nvmlProcessInfo_t, handling struct mismatches
 * between CUDA toolkit headers and driver library.
 * 
 * This function scans the struct for a valid PID value, handling cases where:
 * - CUDA 12.2 headers define struct one way
 * - Driver 570.x (CUDA 12.8) has different struct layout
 * 
 * @param proc Pointer to nvmlProcessInfo_t struct from NVML (void* for header compatibility)
 * @return Valid PID if found, 0 if not found
 */
unsigned int extract_pid_safely(void *proc) {
    nvmlProcessInfo_t *info = (nvmlProcessInfo_t *)proc;
    
    // FIRST: Try the standard PID field - this is the fast path (most common case)
    // Most of the time, the PID field is correct, so we can avoid expensive scanning
    unsigned int pid = info->pid;
    if (pid >= 100 && pid < 4000000 && pid != 0xffffffff) {
        // Quick validation: check if process exists (this is the only /proc access in fast path)
        char path[64];
        snprintf(path, sizeof(path), "/proc/%u/cmdline", pid);
        if (access(path, F_OK) == 0) {
            return pid;  // Fast path: standard field is valid - no expensive scanning needed
        }
    }
    
    // SLOW PATH: Standard field is invalid (0, 0xffffffff, or process doesn't exist)
    // Only do expensive scanning if the standard field failed
    // This handles struct mismatches between CUDA headers and driver
    unsigned char *raw = (unsigned char *)proc;
    
    // Scan for valid PID in the struct
    // Scan first 16 bytes (covers version + pid + start of usedGpuMemory)
    // PID is typically at offset 4 (after version) or 0 (if version is missing)
    for (int offset = 0; offset <= 12; offset += 4) {
        unsigned int candidate = *(unsigned int*)(raw + offset);
        
        // Skip invalid PID ranges
        if (candidate < 100 || candidate > 4000000) continue;
        if (candidate == 0xffffffff) continue;
        // Skip if we already checked this value (it's the standard field)
        if (candidate == pid) continue;
        
        // Verify it's a real process by checking /proc/%u/cmdline
        // Using cmdline is more reliable than just checking the directory
        char path[64];
        snprintf(path, sizeof(path), "/proc/%u/cmdline", candidate);
        if (access(path, F_OK) == 0) {
            // Found PID at different offset - struct mismatch detected
            // Only log mismatch as warning (not debug) since this is called very frequently
            if (pid != 0 && pid != 0xffffffff) {
                // Only warn on actual mismatches (not invalid header values)
                static unsigned int mismatch_log_counter = 0;
                if (++mismatch_log_counter % 100 == 0) {  // Log every 100th mismatch to avoid spam
                    LOG_WARN("extract_pid_safely: PID mismatch detected - found PID %u at offset %d (header pid field was %u)", 
                             candidate, offset, pid);
                }
            }
            return candidate;
        }
    }
    
    return 0;  // No valid PID found
}

/**
 * Safely extract memory value from nvmlProcessInfo_t when PID is at wrong offset
 * This handles struct mismatches where PID and memory fields are shifted
 * 
 * @param proc Pointer to nvmlProcessInfo_t struct from NVML
 * @param actual_pid The PID found by extract_pid_safely (may be at wrong offset)
 * @param header_pid The PID from the header field (infos[i].pid)
 * @return Memory value in bytes, or 0 if not found
 */
uint64_t extract_memory_safely(void *proc, unsigned int actual_pid, unsigned int header_pid) {
    nvmlProcessInfo_t *info = (nvmlProcessInfo_t *)proc;
    unsigned char *raw = (unsigned char *)proc;
    
    // If PID matches header and is valid, struct layout is correct - use standard field
    if (actual_pid == header_pid && header_pid != 0 && header_pid != 0xffffffff) {
        return info->usedGpuMemory;
    }
    
    // Struct mismatch detected - PID is at wrong offset, so memory is probably at wrong offset too
    // First, find where the PID actually is in the struct
    int pid_offset = -1;
    for (int offset = 0; offset <= 12; offset += 4) {
        unsigned int candidate = *(unsigned int*)(raw + offset);
        if (candidate == actual_pid) {
            pid_offset = offset;
            break;
        }
    }
    
    // Add detailed logging to help debug struct layout issues (throttled to avoid spam)
    static unsigned int memory_log_counter = 0;
    if (++memory_log_counter % 100 == 0) {  // Log every 100th call
        LOG_FILE_DEBUG("extract_memory_safely: PID %u (header %u) at offset %d, raw_bytes[0-23]: "
                      "%02x %02x %02x %02x %02x %02x %02x %02x "
                      "%02x %02x %02x %02x %02x %02x %02x %02x "
                      "%02x %02x %02x %02x %02x %02x %02x %02x",
                      actual_pid, header_pid, pid_offset,
                      raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                      raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15],
                      raw[16], raw[17], raw[18], raw[19], raw[20], raw[21], raw[22], raw[23]);
    }
    
    // If we couldn't find the PID offset, fall back to scanning without overlap detection
    if (pid_offset == -1) {
        // PID not found at expected offsets - try scanning for memory anyway
        for (int offset = 8; offset <= 20; offset += 4) {
            uint64_t candidate = *(uint64_t*)(raw + offset);
            if (candidate > 1048576 && candidate < 107374182400ULL) {
                if (candidate != (uint64_t)actual_pid && candidate != (uint64_t)header_pid) {
                    // Also check if either half of the 64-bit value matches PID
                    uint32_t low_bits = (uint32_t)(candidate & 0xFFFFFFFFULL);
                    uint32_t high_bits = (uint32_t)((candidate >> 32) & 0xFFFFFFFFULL);
                    if (low_bits != actual_pid && high_bits != actual_pid && 
                        low_bits != header_pid && high_bits != header_pid) {
                        return candidate;
                    }
                }
            }
        }
        return info->usedGpuMemory;  // Fallback
    }
    
    // Now we know where the PID is - read memory from offsets that don't overlap with it
    // Memory is 8 bytes, PID is 4 bytes
    // We need to avoid reading 8-byte values that include the PID
    
    // Try 8-byte aligned offsets that don't overlap with PID
    // Common safe offsets: 0, 8, 16 (8-byte aligned for 64-bit reads)
    int safe_offsets[] = {0, 8, 16};
    int num_safe_offsets = 3;
    
    for (int i = 0; i < num_safe_offsets; i++) {
        int mem_offset = safe_offsets[i];
        
        // Check if this 8-byte read would overlap with PID
        // Overlap if: mem_offset <= pid_offset < mem_offset+8 OR mem_offset < pid_offset+4 <= mem_offset+8
        int pid_end = pid_offset + 4;  // PID occupies 4 bytes
        if ((mem_offset <= pid_offset && pid_offset < mem_offset + 8) ||
            (mem_offset < pid_end && pid_end <= mem_offset + 8)) {
            // This offset overlaps with PID - skip it
            continue;
        }
        
        // This offset is safe - try reading memory from here
        uint64_t candidate = *(uint64_t*)(raw + mem_offset);
        
        // Validate memory value (typically > 1MB and < 100GB)
        if (candidate > 1048576 && candidate < 107374182400ULL) {
            // Additional validation: make sure it's not the PID
            if (candidate != (uint64_t)actual_pid && candidate != (uint64_t)header_pid) {
                // Also check if either half of the 64-bit value matches PID
                uint32_t low_bits = (uint32_t)(candidate & 0xFFFFFFFFULL);
                uint32_t high_bits = (uint32_t)((candidate >> 32) & 0xFFFFFFFFULL);
                if (low_bits != actual_pid && high_bits != actual_pid && 
                    low_bits != header_pid && high_bits != header_pid) {
                    return candidate;
                }
            }
        }
    }
    
    // Also try offset 12, 20 (4-byte aligned but might contain memory)
    // Only if they don't overlap with PID
    for (int offset = 12; offset <= 20; offset += 8) {
        // Check if this 8-byte read would overlap with PID
        int pid_end = pid_offset + 4;
        if ((offset <= pid_offset && pid_offset < offset + 8) ||
            (offset < pid_end && pid_end <= offset + 8)) {
            continue;  // Skip overlapping offsets
        }
        
        uint64_t candidate = *(uint64_t*)(raw + offset);
        if (candidate > 1048576 && candidate < 107374182400ULL) {
            if (candidate != (uint64_t)actual_pid && candidate != (uint64_t)header_pid) {
                uint32_t low_bits = (uint32_t)(candidate & 0xFFFFFFFFULL);
                uint32_t high_bits = (uint32_t)((candidate >> 32) & 0xFFFFFFFFULL);
                if (low_bits != actual_pid && high_bits != actual_pid && 
                    low_bits != header_pid && high_bits != header_pid) {
                    return candidate;
                }
            }
        }
    }
    
    // Fallback: try standard field anyway (might work in some cases)
    return info->usedGpuMemory;
}

