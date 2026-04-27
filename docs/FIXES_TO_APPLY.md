# SoftMig Fixes To Apply
## Updated: 2026-04-27

This is the actionable checklist for remaining fixes and validation work.
It is derived from `docs/PROJECT_STATUS.md`.

## P1 - Production Readiness

- [ ] Persist `nvidia-smi` wrapper via Warewulf overlay/chroot workflow.
- [ ] Define and enforce statistical acceptance gates for `.2` SM behavior.
- [ ] Harden direct-suite artifact capture for transient `no softmig log` cases.

## P2 - Harness Reliability

- [ ] Run and review overnight soak/stress cycles for stability metrics.
- [ ] Calibrate `suite_soak.sh` thresholds (`shrreg_delta`, FD stability) from overnight data.

## P3 - Non-Blocking Cleanup

- [ ] Continue low-risk dead code/header cleanup where safe.

## Production Cut Gates (Must Pass)

- [ ] `nvsmi` leak stress remains `others=0` under concurrent pressure.
- [ ] `.2` SM metrics stay within agreed tolerance bands.
- [ ] OOM enforcement remains PASS across CUDA versions and slice types.
- [ ] Direct-linked hook registration remains stable.
- [ ] Wrapper persistence is verified after reboot/reprovision.
