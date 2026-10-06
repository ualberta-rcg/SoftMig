/**
 * @file test_pool_free.c
 * @brief Regression test for the pool-allocation free failure bug.
 *
 * Reproduces the JAX/XLA cuda_async failure mode: allocations from a memory
 * pool (cuMemAllocFromPoolAsync) were never tracked, so cuMemFreeAsync
 * returned -1 for them without ever reaching the driver, leaking device
 * memory until the card was exhausted.
 *
 * Exit codes:
 *   0 = PASS (all requested stages passed)
 *   2 = bug reproduced / stage failure
 *   1 = setup error
 *
 * Modes:
 *   (no flags)        stages 1-4
 *   --passive         stages 1-5 (full-GPU job, no config file); stage 5
 *                     verifies cuMemGetInfo reports real driver values
 *   --limit-bytes N   stages 1-4 + 6 (sliced job); stage 6 verifies a pool
 *                     alloc beyond the limit returns CUDA_ERROR_OUT_OF_MEMORY
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GIB (1073741824.0)
#define MIB (1048576.0)

static const char *errstr(CUresult r) {
    const char *s = NULL;
    cuGetErrorString(r, &s);
    return s ? s : "?";
}

#define CHECK(e) do { CUresult r_ = (e); if (r_ != CUDA_SUCCESS) { \
    printf("FATAL: %s res=%d (%s)\n", #e, (int)r_, errstr(r_)); exit(1); } } while (0)

#define FAIL(...) do { \
    printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); fflush(stdout); exit(2); } while (0)

#define PASS_MSG(...) do { \
    printf("PASS: "); printf(__VA_ARGS__); printf("\n"); fflush(stdout); } while (0)

static size_t show_free(const char *tag) {
    size_t fr = 0, tot = 0;
    CUresult r = cuMemGetInfo(&fr, &tot);
    if (r != CUDA_SUCCESS) {
        printf("FATAL: cuMemGetInfo res=%d (%s)\n", (int)r, errstr(r));
        exit(1);
    }
    printf("    %-34s free=%.2f GiB / total=%.2f GiB\n",
           tag, fr / GIB, tot / GIB);
    fflush(stdout);
    return fr;
}

static size_t get_free(void) {
    size_t fr = 0, tot = 0;
    if (cuMemGetInfo(&fr, &tot) != CUDA_SUCCESS) {
        printf("FATAL: cuMemGetInfo failed\n");
        exit(1);
    }
    return fr;
}

static void sync_stream(CUstream s) {
    CUresult r = cuStreamSynchronize(s);
    if (r != CUDA_SUCCESS) {
        printf("FATAL: cuStreamSynchronize res=%d (%s)\n", (int)r, errstr(r));
        exit(1);
    }
}

/* --- self-describing header -------------------------------------------- */

static int softmig_mapped(void) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return -1;
    char line[1024];
    while (fgets(line, sizeof line, f))
        if (strstr(line, "libsoftmig.so")) { fclose(f); return 1; }
    fclose(f);
    return 0;
}

static void print_lib_info(void) {
    const char *lp = getenv("LD_PRELOAD");
    const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
    printf("libsoftmig in process maps: %d\n", softmig_mapped());
    printf("LD_PRELOAD=%s\n", lp ? lp : "(unset)");
    printf("CUDA_VISIBLE_DEVICES=%s\n", cvd ? cvd : "(unset)");
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return;
    char line[1024];
    char seen[8][512];
    int nseen = 0;
    while (fgets(line, sizeof line, f)) {
        char *p = strstr(line, "libsoftmig.so");
        if (!p) continue;
        char path[512] = {0};
        char *sp = strchr(line, '/');
        if (!sp) continue;
        char *nl = strchr(sp, '\n');
        if (nl) *nl = '\0';
        int dup = 0;
        for (int i = 0; i < nseen; i++)
            if (strcmp(seen[i], sp) == 0) { dup = 1; break; }
        if (dup) continue;
        snprintf(path, sizeof path, "%s", sp);
        printf("mapped lib: %s\n", path);
        char cmd[1024];
        snprintf(cmd, sizeof cmd, "md5sum '%s' 2>/dev/null", path);
        FILE *md = popen(cmd, "r");
        if (md) {
            char out[256];
            if (fgets(out, sizeof out, md)) {
                char *c = strchr(out, ' ');
                if (c) *c = '\0';
                printf("mapped lib md5: %s\n", out);
            }
            pclose(md);
        }
        if (nseen < 8) { snprintf(seen[nseen], 512, "%s", path); nseen++; }
    }
    fclose(f);
    fflush(stdout);
}

/* --- nvidia-smi comparison (stage 5) ------------------------------------ */

/* Query nvidia-smi for the GPU this process is bound to (CUDA_VISIBLE_DEVICES
 * first entry; nvidia-smi --id accepts index, UUID, or PCI bus id).
 * Returns free/total in bytes via out params, 0 on success. */
static int nvidia_smi_free_total(size_t *out_free, size_t *out_total) {
    const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
    char cmd[512];
    if (cvd && cvd[0] != '\0') {
        char first[128] = {0};
        snprintf(first, sizeof first, "%s", cvd);
        char *comma = strchr(first, ',');
        if (comma) *comma = '\0';
        snprintf(cmd, sizeof cmd,
                 "nvidia-smi --id=%s --query-gpu=memory.free,memory.total "
                 "--format=csv,noheader,nounits 2>/dev/null",
                 first);
    } else {
        snprintf(cmd, sizeof cmd,
                 "nvidia-smi --query-gpu=memory.free,memory.total "
                 "--format=csv,noheader,nounits 2>/dev/null");
    }
    FILE *p = popen(cmd, "r");
    if (!p) return -1;
    char buf[256];
    long f_mib = -1, t_mib = -1;
    if (fgets(buf, sizeof buf, p))
        if (sscanf(buf, "%ld, %ld", &f_mib, &t_mib) != 2) { f_mib = t_mib = -1; }
    pclose(p);
    if (f_mib < 0 || t_mib < 0) return -1;
    *out_free = (size_t)f_mib * (size_t)1048576;
    *out_total = (size_t)t_mib * (size_t)1048576;
    printf("    nvidia-smi                        free=%.2f GiB / total=%.2f GiB\n",
           *out_free / GIB, *out_total / GIB);
    return 0;
}

int main(int argc, char **argv) {
    int passive = 0;
    unsigned long long limit_bytes = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--passive") == 0) {
            passive = 1;
        } else if (strcmp(argv[i], "--limit-bytes") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            char *end = NULL;
            double val = strtod(v, &end);
            if (end && (*end == 'K' || *end == 'k')) val *= 1024.0;
            else if (end && (*end == 'M' || *end == 'm')) val *= 1048576.0;
            else if (end && (*end == 'G' || *end == 'g')) val *= 1073741824.0;
            limit_bytes = (unsigned long long)val;
        } else {
            printf("FATAL: unknown argument %s\n", argv[i]);
            return 1;
        }
    }
    printf("mode: %s\n",
           passive ? "passive (expect real driver behavior)" :
           limit_bytes ? "enabled (limit enforced)" : "default (stages 1-4)");
    print_lib_info();

    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));
    CUstream s; CHECK(cuStreamCreate(&s, 0));

    /* ---------- stage 1: tracked async path ---------- */
    printf("\n[stage 1] tracked path: cuMemAllocAsync + cuMemFreeAsync (256 MiB)\n");
    {
        CUdeviceptr t = 0;
        CUresult r = cuMemAllocAsync(&t, (size_t)256 << 20, s);
        printf("  cuMemAllocAsync  res=%d (%s)\n", (int)r, errstr(r));
        if (r != CUDA_SUCCESS) FAIL("stage1 alloc res=%d (%s)", (int)r, errstr(r));
        sync_stream(s);
        r = cuMemFreeAsync(t, s);
        printf("  cuMemFreeAsync   res=%d (%s)\n", (int)r, errstr(r));
        sync_stream(s);
        if (r != CUDA_SUCCESS)
            FAIL("stage1 free res=%d (%s) - cuMemFreeAsync failed on tracked async alloc",
                 (int)r, errstr(r));
        PASS_MSG("stage1: tracked async alloc/free OK");
    }

    /* ---------- stage 2: XLA pattern - private pool, 75% prealloc ---------- */
    printf("\n[stage 2] XLA cuda_async pattern: private pool, prealloc 75%% of free\n");
    CUmemoryPool pool;
    size_t free0 = get_free();
    {
        CUmemPoolProps props;
        memset(&props, 0, sizeof props);
        props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
        props.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
        props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.id = 0;
        CUresult rpc = cuMemPoolCreate(&pool, &props);
        if (rpc != CUDA_SUCCESS) {
            printf("  cuMemPoolCreate res=%d (%s), falling back to device default pool\n",
                   (int)rpc, errstr(rpc));
            CHECK(cuDeviceGetDefaultMemPool(&pool, dev));
        } else {
            printf("  private pool created\n");
        }

        size_t prealloc = free0 - free0 / 4;  /* 75% of reported free */
        printf("  prealloc size: %.2f GiB (75%% of reported free)\n", prealloc / GIB);
        CUdeviceptr p = 0;
        CUresult r = cuMemAllocFromPoolAsync(&p, prealloc, pool, s);
        printf("  cuMemAllocFromPoolAsync res=%d (%s)\n", (int)r, errstr(r));
        sync_stream(s);
        if (r != CUDA_SUCCESS || p == 0)
            FAIL("stage2 prealloc res=%d (%s)", (int)r, errstr(r));
        size_t free_mid = get_free();
        if (free_mid > free0)
            FAIL("stage2: free grew after prealloc (%.2f -> %.2f GiB)?",
                 free0 / GIB, free_mid / GIB);
        r = cuMemFreeAsync(p, s);
        printf("  cuMemFreeAsync prealloc    res=%d (%s)%s\n",
               (int)r, errstr(r), r != CUDA_SUCCESS ? "   << BUG?" : "");
        sync_stream(s);
        if (r != CUDA_SUCCESS)
            FAIL("stage2 prealloc free res=%d (%s) - pool allocation free failed; "
                 "same signature as 'cudaFreeAsync failed ... UNKNOWN ERROR (-1)'",
                 (int)r, errstr(r));
        CHECK(cuMemPoolTrimTo(pool, 0));
        sync_stream(s);
        size_t free_after = get_free();
        if (free0 > free_after + ((size_t)256 << 20))
            FAIL("stage2 leak: free %.2f GiB at start, %.2f GiB after prealloc+free+trim "
                 "(dropped %.2f GiB)", free0 / GIB, free_after / GIB,
                 (free0 - free_after) / GIB);
        PASS_MSG("stage2: 75%% pool prealloc freed OK, free back to baseline (+/- 256 MiB)");
    }

    /* ---------- stage 3: alloc/free loop from pool ---------- */
    printf("\n[stage 3] 20x (cuMemAllocFromPoolAsync 512 MiB + cuMemFreeAsync)\n");
    {
        for (int i = 0; i < 20; i++) {
            CUdeviceptr q = 0;
            CUresult ra = cuMemAllocFromPoolAsync(&q, (size_t)512 << 20, pool, s);
            sync_stream(s);
            if (ra != CUDA_SUCCESS)
                FAIL("stage3 iter %d alloc res=%d (%s)", i, (int)ra, errstr(ra));
            CUresult rf = cuMemFreeAsync(q, s);
            sync_stream(s);
            if (rf != CUDA_SUCCESS)
                FAIL("stage3 iter %d free res=%d (%s) - pool free failed; "
                     "memory leaks until the card is exhausted",
                     i, (int)rf, errstr(rf));
            if (i % 5 == 4)
                printf("  iter %2d ok, device free=%.2f GiB\n", i, get_free() / GIB);
        }
        CHECK(cuMemPoolTrimTo(pool, 0));
        sync_stream(s);
        size_t free_end = get_free();
        if (free0 > free_end + ((size_t)256 << 20))
            FAIL("stage3 leak: free %.2f GiB at start, %.2f GiB after 20 alloc/free+trim",
                 free0 / GIB, free_end / GIB);
        PASS_MSG("stage3: 20 alloc/free iterations, all frees OK, free flat");
    }

    /* ---------- stage 4: pool pointer freed via sync path ---------- */
    printf("\n[stage 4] pool pointer freed with cuMemFree (sync fallback path)\n");
    {
        CUdeviceptr q = 0;
        CUresult r = cuMemAllocFromPoolAsync(&q, (size_t)256 << 20, pool, s);
        sync_stream(s);
        if (r != CUDA_SUCCESS)
            FAIL("stage4 alloc res=%d (%s)", (int)r, errstr(r));
        r = cuMemFree(q);
        printf("  cuMemFree(pool ptr) res=%d (%s)%s\n",
               (int)r, errstr(r), r != CUDA_SUCCESS ? "   << BUG?" : "");
        if (r != CUDA_SUCCESS)
            FAIL("stage4 sync free of pool ptr res=%d (%s) - untracked free did not "
                 "reach the driver", (int)r, errstr(r));
        CHECK(cuMemPoolTrimTo(pool, 0));
        sync_stream(s);
        size_t free_end = get_free();
        if (free0 > free_end + ((size_t)256 << 20))
            FAIL("stage4 leak: free %.2f GiB at start, %.2f GiB after sync free+trim",
                 free0 / GIB, free_end / GIB);
        PASS_MSG("stage4: sync free of pool pointer OK");
    }

    /* ---------- stage 5 (passive only): cuMemGetInfo = real values ---------- */
    if (passive) {
        printf("\n[stage 5] passive: cuMemGetInfo must match nvidia-smi (+/- 64 MiB)\n");
        size_t fr = 0, tot = 0;
        CHECK(cuMemGetInfo(&fr, &tot));
        printf("  cuMemGetInfo                     free=%.2f GiB / total=%.2f GiB\n",
               fr / GIB, tot / GIB);
        size_t smi_free = 0, smi_total = 0;
        if (nvidia_smi_free_total(&smi_free, &smi_total) != 0)
            FAIL("stage5: could not query nvidia-smi");
        /* free must match: the old bug reported free = total - cgroup_usage
         * (ignoring other users' memory), so a mismatch here is the
         * regression signal. total may legitimately differ: the CUDA API
         * excludes driver-reserved memory (~600 MiB on L40S) that
         * nvidia-smi includes. */
        size_t tol = (size_t)64 << 20;
        if (fr > smi_free + tol || smi_free > fr + tol)
            FAIL("stage5: cuMemGetInfo free=%.2f GiB != nvidia-smi free=%.2f GiB "
                 "(passive mode must report real driver values)",
                 fr / GIB, smi_free / GIB);
        size_t ttol = (size_t)1024 << 20;
        if (tot > smi_total || smi_total > tot + ttol)
            FAIL("stage5: cuMemGetInfo total=%.2f GiB vs nvidia-smi total=%.2f GiB "
                 "(expected <= physical, within 1 GiB)",
                 tot / GIB, smi_total / GIB);
        PASS_MSG("stage5: cuMemGetInfo free matches nvidia-smi within 64 MiB");
    }

    /* ---------- stage 6 (enabled only): over-limit pool alloc must OOM ---------- */
    if (limit_bytes > 0) {
        printf("\n[stage 6] enabled: pool alloc of limit+1GiB must OOM with code 2\n");
        printf("  limit: %llu bytes (%.2f GiB)\n", limit_bytes,
               limit_bytes / GIB);
        CUdeviceptr q = 0;
        size_t over = (size_t)limit_bytes + ((size_t)1 << 30);
        CUresult r = cuMemAllocFromPoolAsync(&q, over, pool, s);
        printf("  cuMemAllocFromPoolAsync(limit+1GiB) res=%d (%s)\n",
               (int)r, errstr(r));
        if (r != CUDA_ERROR_OUT_OF_MEMORY)
            FAIL("stage6: expected CUDA_ERROR_OUT_OF_MEMORY(2), got %d (%s) "
                 "- XLA/JAX retry logic depends on the proper OOM code",
                 (int)r, errstr(r));
        /* process must still be healthy */
        CUdeviceptr q2 = 0;
        r = cuMemAllocFromPoolAsync(&q2, (size_t)256 << 20, pool, s);
        sync_stream(s);
        printf("  post-OOM 256 MiB pool alloc       res=%d (%s)\n",
               (int)r, errstr(r));
        if (r != CUDA_SUCCESS)
            FAIL("stage6: post-OOM 256 MiB alloc res=%d (%s) - process unhealthy",
                 (int)r, errstr(r));
        r = cuMemFreeAsync(q2, s);
        sync_stream(s);
        if (r != CUDA_SUCCESS)
            FAIL("stage6: post-OOM free res=%d (%s)", (int)r, errstr(r));
        PASS_MSG("stage6: over-limit pool alloc returned CUDA_ERROR_OUT_OF_MEMORY, "
                 "process healthy afterwards");
    }

    printf("\n[verdict]\n");
    printf("  all requested stages passed\n");
    cuStreamDestroy(s);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
