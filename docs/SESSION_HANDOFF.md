# SoftMig Session Handoff (Current)
## Updated: 2026-04-27

This file now tracks current handoff context only.
The previous "what was unread" audit handoff is obsolete.

---

## Completed since the original handoff

- Core source tree audit completed.
- NVML struct mismatch bug fixed and validated.
- Cgroup filtering tightened for SLURM behavior.
- Lock recovery/log spam hardening implemented.
- `nvidia-smi` wrapper policy updated and stress-validated.

---

## Current open follow-ups

1. Persist wrapper changes via Warewulf overlay workflow (runtime copy alone is not persistent).
2. Continue overnight statistical validation for `.2` SM tolerance bands.
3. Keep monitoring for transient `direct` suite `no softmig log` artifacts under extreme contention.

---

## Canonical planning docs

- `docs/MASTER_FIX_CHECKLIST.md` — open production checklist
- `docs/IMPROVEMENT_IDEAS.md` — open backlog (completed items removed)
- `docs/BUGFIX_struct_mismatch_and_suggestions.md` — historical bug record (fixed)
