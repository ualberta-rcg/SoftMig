# SoftMig Bug Fix Record: NVML Process Info Struct Mismatch

## Date: 2026-04-17
## Status: Fixed

---

## Issue (historical)

SoftMig previously had mismatched NVML process-info struct handling, which could corrupt
PID/memory interpretation and force expensive workaround logic.

---

## Resolution summary

The mismatch remediation was implemented and validated:

- NVML process struct handling aligned with real driver behavior.
- PID/memory parsing paths were corrected/simplified.
- Related caching and filtering improvements were added.
- Stress tests no longer show known garbage PID signatures.

---

## Validation outcome (high level)

- Direct-linked and runtime-linked registration tests pass in current validation runs.
- No recurrence of known bad PID/memory-signature failures in recent stress suites.
- Remaining production follow-ups are now operational/test-harness oriented, not this bug.

---

## Current references

- Open work: `docs/MASTER_FIX_CHECKLIST.md`
- Open backlog: `docs/IMPROVEMENT_IDEAS.md`
