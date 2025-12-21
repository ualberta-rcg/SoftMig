#ifndef __UTILS_PROCESS_UTILS_H__
#define __UTILS_PROCESS_UTILS_H__ 

#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define BUFFER_LENGTH 8192  // ensure larger than linux max filename length
#define FILENAME_LENGTH 8192

#define PROC_STATE_ALIVE 0
#define PROC_STATE_NONALIVE 1
#define PROC_STATE_UNKNOWN 2

// Get the UID of a process by PID (returns -1 on error)
uid_t proc_get_uid(int32_t pid);

// Check if a process is alive
int proc_alive(int32_t pid);

// Check if a process belongs to the same cgroup session as the current process
// Returns 1 if process belongs to same session, 0 if not, -1 on error (fallback to UID)
// Supports both cgroups v1 and v2
int proc_belongs_to_current_cgroup_session(int32_t pid);

// Safely extract PID from nvmlProcessInfo_t, handling struct mismatches
// between CUDA toolkit headers and driver library
// Returns valid PID if found, 0 if not found
unsigned int extract_pid_safely(void *proc);

// Get process start time (clock ticks since boot) from /proc/PID/stat
// Returns 0 on error, or starttime (field 22) on success
unsigned long long proc_get_starttime(int32_t pid);


#endif  // __UTILS_PROCESS_UTILS_H__
