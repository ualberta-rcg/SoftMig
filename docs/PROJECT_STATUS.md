# SoftMig Project Status
## Updated: 2026-04-27

This is the canonical status document for ongoing work. It consolidates
historical handoff/checklist/backlog files into one place.

## What Is Complete

- Core source tree audit completed.
- NVML process-info struct mismatch fixed and validated.
- Cgroup filtering behavior tightened for SLURM isolation.
- dlsym resolution and lock-recovery hardening implemented.
- NVML caching and process-query reliability improvements merged.
- `nvidia-smi` wrapper behavior updated and stress-validated.

## Current Open Follow-Ups

### P1 - Production Readiness

- [ ] Finalize Warewulf persistence for `nvidia-smi` wrapper
  - Runtime copy is active on test node, but persistence across
    reprovision/reboot still needs finalization.

- [ ] Stabilize `.2` SM acceptance gates
  - Keep time-sliced average behavior and gate with statistical windows
    (e.g., median/p90 bands), not single-sample checks.

- [ ] Harden direct-suite artifact capture
  - Improve retries/capture for transient `no softmig log` outcomes so
    artifact misses are not misclassified as core hook failures.

### P2 - Harness Reliability

- [ ] Run and evaluate overnight soak/stress cycles
  - Continue validating `.2` variance, direct-suite stability, and wrapper
    leak regression behavior over long windows.

- [ ] Calibrate `suite_soak.sh` thresholds with real data
  - Tune `shrreg_delta` and FD stability thresholds based on overnight runs.

### P3 - Optional Cleanup

- [ ] Continue low-risk dead-code/header cleanup
  - Keep as non-blocking unless correctness paths are touched.

## Go / No-Go Gate Before Production Cut

- [ ] `nvsmi` leak stress remains `others=0` under pressure.
- [ ] `.2` SM metrics stay within agreed statistical tolerance bands.
- [ ] OOM enforcement remains PASS across CUDA versions and slices.
- [ ] Direct-linked hook registration remains stable.
- [ ] Wrapper persistence verified after node reboot/reprovision.

## Historical References

- High-level release history: `CHANGES.md`
