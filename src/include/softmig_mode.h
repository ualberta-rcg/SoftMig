/**
 * @file softmig_mode.h
 * @brief Process-wide passive/enabled decision and pass-through helpers.
 *
 * libsoftmig.so is loaded into every process via /etc/ld.so.preload, so the
 * library must decide on its own whether to act. The decision is made once
 * per process (first CUDA/NVML lookup or first wrapper call):
 *
 *   - Inside a Slurm job: enabled only if /var/run/softmig/<jobid>.conf is a
 *     root-owned regular file that sets CUDA_DEVICE_MEMORY_LIMIT or
 *     CUDA_DEVICE_SM_LIMIT. User env vars are ignored.
 *   - Outside Slurm: enabled only if one of those env vars is set.
 *
 * In passive mode every exported CUDA/NVML entry point must forward straight
 * to the real driver without touching SoftMig state (shared region, watcher,
 * allocator lists, signal handlers). check_passive_guards.sh enforces this at
 * build time for wrappers that reference SoftMig internals.
 */
#ifndef __SOFTMIG_MODE_H__
#define __SOFTMIG_MODE_H__

extern volatile int softmig_mode_cached;  /* -1 unknown, 0 enabled, 1 passive */
int softmig_mode_init(void);

static inline int softmig_is_passive(void) {
    int m = softmig_mode_cached;
    if (__builtin_expect(m < 0, 0)) {
        m = softmig_mode_init();
    }
    return m;
}

extern volatile int softmig_cuda_table_ready;
void softmig_ensure_cuda_table(void);

/* Log (file only) an enforcement-relevant CUDA/NVML lookup that resolved to
 * the raw driver instead of a SoftMig hook. */
void softmig_note_unhooked(const char *symbol, const char *via);

#define SOFTMIG_PASSIVE_FORWARD(sym, ...)                                      \
    do {                                                                       \
        if (softmig_is_passive()) {                                            \
            return CUDA_OVERRIDE_CALL(cuda_library_entry, sym, ##__VA_ARGS__); \
        }                                                                      \
    } while (0)

#define SOFTMIG_PASSIVE_FORWARD_NVML(sym, ...)                                 \
    do {                                                                       \
        if (softmig_is_passive()) {                                            \
            return NVML_OVERRIDE_CALL(nvml_library_entry, sym, ##__VA_ARGS__); \
        }                                                                      \
    } while (0)

#endif  // __SOFTMIG_MODE_H__
