#include "include/process_utils.h"
#include "include/log_utils.h"
#include <stdlib.h>
#include <errno.h>

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

