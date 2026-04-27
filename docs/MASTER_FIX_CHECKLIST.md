# SoftMig Master Fix Checklist (Current)
## Updated: 2026-04-27

This file now tracks only **remaining open work**.  
Previously listed items that are now implemented (NVML struct fix, caching, dlsym dedupe, lock hardening, config hardening, OOM enforcement fixes, wrapper leak fix, etc.) were removed from this checklist.

---

## P1 - Production readiness follow-ups

- [ ] **Finalize Warewulf persistence for `nvidia-smi` wrapper**
  - Current wrapper is active on `rack01-12` at runtime (`/usr/local/bin/nvidia-smi`), but rootfs is Warewulf-managed.
  - Persist via the correct Warewulf overlay/chroot workflow so it survives reprovision/reboot.

- [ ] **Stabilize `.2` SM acceptance gates**
  - Keep SM limiter as time-sliced average target (not hard instant cap), but gate on statistical windows.
  - Define production thresholds (example: median and p90 over repeated runs) and update suite criteria accordingly.

- [ ] **Harden direct suite against transient `no softmig log` outcomes**
  - Keep the core pass criteria (`launched=4 registered=4`) but improve log capture robustness/retries.
  - Avoid classifying missing artifact as a core-hook failure when retests pass.

---

## P2 - Test harness reliability

- [ ] **Run and review overnight soak/stress cycles**
  - Use `test/run_overnight.sh` (added and syntax-checked) for repeated distribution checks:
    - `.2` SM variance
    - direct no-log flakiness
    - nvsmi leak regressions
  - Produce go/no-go gate table from overnight artifacts.

- [ ] **Tune `suite_soak.sh` thresholds using overnight data**
  - Script parsing/arithmetic is fixed, but pass/fail limits (`shrreg_delta`, FD behavior) should be calibrated with real long-window data.

---

## P3 - Optional cleanup (non-blocking)

- [ ] **Additional dead code/minor cleanup**
  - Continue removing stale declarations/comments and simplifying wrappers where low risk.
  - Keep this non-blocking unless it touches correctness paths.

---

## Go/No-Go checks before production cut

- [ ] `nvsmi` leak stress remains `others=0` under concurrent pressure.
- [ ] `.2` SM metrics are within agreed statistical tolerance bands.
- [ ] OOM enforcement remains PASS across CUDA versions and slices.
- [ ] Direct-linked hooks remain stable with no sustained registration regressions.
- [ ] Wrapper persistence verified after node reboot/reprovision.

