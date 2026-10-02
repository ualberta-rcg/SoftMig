/*
 * fork_probe - SoftMig state across fork() after CUDA is initialized.
 *
 * Usage: fork_probe <exec_path>
 *
 * Parent initializes CUDA and allocates, then forks 4 children:
 *   - 3 children do CPU work without CUDA and _exit(0) (DataLoader-style);
 *   - 1 child execs <exec_path> (e.g. stress_alloc), i.e. a fresh CUDA
 *     process started from a CUDA process.
 * The parent keeps allocating and freeing while the children run, then
 * waits for them. Prints PASS:/FAIL: lines and a RESULT line.
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures = 0;
static void check(int ok, const char *what) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    fflush(stdout);
    if (!ok) failures++;
}

int main(int argc, char **argv) {
    const char *exec_path = argc > 1 ? argv[1] : NULL;
    CUdevice dev;
    CUcontext ctx;
    CUdeviceptr p;
    check(cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&dev, 0) == CUDA_SUCCESS &&
              cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS && cuCtxSetCurrent(ctx) == CUDA_SUCCESS,
          "parent CUDA init");
    check(cuMemAlloc(&p, 256u << 20) == CUDA_SUCCESS, "parent alloc 256 MiB before fork");

    pid_t kids[4];
    for (int i = 0; i < 4; i++) {
        kids[i] = fork();
        if (kids[i] == 0) {
            if (i == 3 && exec_path) {
                execl(exec_path, exec_path, "2", "5", "32", "1", (char *)NULL);
                _exit(127);
            }
            volatile double x = 0;
            for (long k = 0; k < 50000000L; k++) x += k * 1e-9;
            _exit(0);
        }
    }
    int ok = 1;
    for (int i = 0; i < 200; i++) {
        CUdeviceptr q;
        if (cuMemAlloc(&q, 8u << 20) != CUDA_SUCCESS || cuMemFree(q) != CUDA_SUCCESS) ok = 0;
    }
    check(ok, "parent alloc/free while children run");
    for (int i = 0; i < 4; i++) {
        int st = 0;
        waitpid(kids[i], &st, 0);
        char what[96];
        snprintf(what, sizeof what, "child %d (%s) exit status %d", i, i == 3 ? "exec CUDA" : "cpu-only",
                 WIFEXITED(st) ? WEXITSTATUS(st) : -1);
        check(WIFEXITED(st) && WEXITSTATUS(st) == 0, what);
    }
    check(cuMemFree(p) == CUDA_SUCCESS, "parent free after children exit");
    size_t f, t;
    check(cuMemGetInfo(&f, &t) == CUDA_SUCCESS, "parent cuMemGetInfo after children exit");
    printf("RESULT: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
