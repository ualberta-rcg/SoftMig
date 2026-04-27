# SoftMig Improvement Ideas (Open Only)
## Updated: 2026-04-27
## Status: Active backlog (completed items removed)

---

## High priority

### 1. Persist `nvidia-smi` wrapper in Warewulf overlays
**Why:** Current runtime deployment on compute node can be lost on reprovision/reboot.  
**Needed:** Install wrapper via Warewulf host/runtime overlay workflow and verify persistence after reboot.

### 2. Define statistical gates for `.2` SM validation
**Why:** Time-sliced SM limiting is noisy in short windows; strict single-run pass/fail is unstable.  
**Needed:** Overnight sampling and formal gate thresholds (e.g., median/p90 bands).

### 3. Harden direct-suite artifact capture
**Why:** Rare `direct` suite outcomes still show `no softmig log` even when retest passes.  
**Needed:** Retry/capture improvements to prevent harness artifact misses from being treated as core failures.

---

## Medium priority

### 4. Additional long soak validation
**Why:** `suite_soak.sh` parser/arithmetic issues were fixed, but thresholds should be calibrated with long runs.  
**Needed:** Run overnight cycles and tune `shrreg_delta` / FD stability pass criteria.

### 5. Keep reducing low-value hot-path logs
**Why:** High log volume can obscure signal and increase I/O overhead during stress testing.  
**Needed:** Continue pruning repetitive debug logs while preserving actionable errors.

---

## Low priority cleanup

### 6. Optional dead code/header drift cleanup
**Why:** Maintainability and future audit speed.  
**Needed:** Continue removing stale declarations/comments where risk is low.

---

## Notes

- Historical completed findings were removed from this file intentionally.
- For current production gating, see `docs/MASTER_FIX_CHECKLIST.md`.
- For historical NVML struct mismatch context, see `docs/BUGFIX_struct_mismatch_and_suggestions.md`.
