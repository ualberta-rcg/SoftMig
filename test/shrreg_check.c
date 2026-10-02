// shrreg_check.c - consistency check of a job's SoftMig shared region.
//
// Usage: shrreg_check [path]   (default: this job's region, same naming as the library)
//
// Prints one key=value line and exits 0 iff the region is consistent:
// version matches this build, 0 <= proc_num <= max, no empty (pid 0) slot
// below proc_num, no slot of a dead process, lock not held, and no live
// usage left in slots of processes that are gone. Run it after all workers
// have exited (and after one short-lived "sweeper" process, which clears
// slots of killed processes on init).

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "multiprocess/multiprocess_memory_limit.h"

static int alive(int pid) {
    return pid > 0 && (kill(pid, 0) == 0 || errno == EPERM);
}

int main(int argc, char **argv) {
    char path[512];
    if (argc > 1) {
        snprintf(path, sizeof(path), "%s", argv[1]);
    } else {
        // Same naming rule as the library (shrreg_default_path), which this
        // binary cannot call: it is not linked against libsoftmig.
        const char *tmpdir = getenv("SLURM_TMPDIR");
        const char *jid = getenv("SLURM_JOB_ID");
        const char *aid = getenv("SLURM_ARRAY_TASK_ID");
        if (!tmpdir) tmpdir = "/tmp";
#if MAJOR_VERSION >= 2
        if (aid)
            snprintf(path, sizeof(path), "%s/cudevshr.cache.v%d.%s.%s", tmpdir, MAJOR_VERSION, jid ? jid : "none", aid);
        else
            snprintf(path, sizeof(path), "%s/cudevshr.cache.v%d.%s", tmpdir, MAJOR_VERSION, jid ? jid : "none");
#else
        if (aid)
            snprintf(path, sizeof(path), "%s/cudevshr.cache.%s.%s", tmpdir, jid ? jid : "none", aid);
        else
            snprintf(path, sizeof(path), "%s/cudevshr.cache.%s", tmpdir, jid ? jid : "none");
#endif
    }
    int fd = open(path, O_RDWR);
    if (fd < 0) {
        printf("shrreg_check: region=absent path=%s errno=%d\n", path, errno);
        return 2;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < sizeof(shared_region_t)) {
        printf("shrreg_check: region=short size=%lld want=%zu\n", (long long)st.st_size,
               sizeof(shared_region_t));
        return 1;
    }
    shared_region_t *r = mmap(NULL, sizeof(shared_region_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (r == MAP_FAILED) {
        printf("shrreg_check: mmap failed errno=%d\n", errno);
        return 1;
    }

    int bad = 0;
    int version_ok = r->major_version == MAJOR_VERSION && r->minor_version == MINOR_VERSION;
    bad |= !version_ok;
    int n = r->proc_num;
    int n_ok = n >= 0 && n <= SHARED_REGION_MAX_PROCESS_NUM;
    bad |= !n_ok;
    int empty = 0, dead = 0, live = 0;
    unsigned long long stale_bytes = 0;
    if (n_ok) {
        for (int i = 0; i < n; i++) {
            int pid = r->procs[i].pid;
            if (pid == 0) {
                empty++;
            } else if (!alive(pid)) {
                dead++;
                for (int d = 0; d < CUDA_DEVICE_MAX_COUNT; d++) {
                    stale_bytes += r->procs[i].used[d].total;
                }
            } else {
                live++;
            }
        }
    }
    bad |= empty > 0 || dead > 0;

    const char *lock_state;
#if MAJOR_VERSION >= 2
    int lr = pthread_mutex_trylock(&r->lock);
    if (lr == 0) {
        pthread_mutex_unlock(&r->lock);
        lock_state = "free";
    } else if (lr == EOWNERDEAD) {
        pthread_mutex_consistent(&r->lock);
        pthread_mutex_unlock(&r->lock);
        lock_state = "ownerdead";
        bad = 1;
    } else {
        lock_state = "held";
        bad = 1;
    }
#else
    int sv = -1;
    sem_getvalue(&r->sem, &sv);
    lock_state = (sv == 1 && r->owner_pid == 0) ? "free" : "held";
    bad |= !(sv == 1 && r->owner_pid == 0);
#endif

    printf("shrreg_check: %s version=%u.%u(%s) proc_num=%d live=%d empty=%d dead=%d "
           "stale_bytes=%llu lock=%s\n",
           bad ? "BAD" : "OK", r->major_version, r->minor_version, version_ok ? "ok" : "mismatch", n,
           live, empty, dead, stale_bytes, lock_state);
    return bad ? 1 : 0;
}
