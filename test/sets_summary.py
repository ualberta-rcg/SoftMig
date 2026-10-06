#!/usr/bin/env python3
"""Summarize a run_sets.sh (or run_storm.sh / overnight) results.tsv as Markdown.

Usage: sets_summary.py ROOT [ROOT...]   (reads ROOT/results.tsv, concatenates
ROOT/*/*/jobs.tsv per-job tables where present)
"""
import glob
import os
import sys
from collections import Counter, defaultdict

STATUS_ORDER = ["PASS", "LEAK", "PARTIAL", "INFO", "FAIL", "HANG"]


def main():
    roots = sys.argv[1:] or ["."]
    rows = []
    for root in roots:
        p = os.path.join(root, "results.tsv")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            off = 0
            for i, line in enumerate(f):
                if i == 0 and (line.startswith("cuda_ver") or line.startswith("cycle")):
                    off = 1 if line.startswith("cycle") else 0   # overnight/storm add a cycle column
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 7 + off:
                    rows.append(parts[off:off + 7])
    total = Counter(r[4] for r in rows)
    print(f"# SoftMig 2.06 campaign summary ({', '.join(roots)})")
    print()
    print("Total: %d   " % len(rows) + "   ".join(f"{s}: {total.get(s, 0)}" for s in STATUS_ORDER))
    print()
    print("LEAK = limits held, but another job's PIDs were visible from inside the job "
          "(nvidia-smi 595 bypasses the NVML process-list hooks; see nvidia-smi-hook.sh).")
    print()

    by_suite = defaultdict(list)
    for r in rows:
        by_suite[r[2]].append(r)
    for suite in sorted(by_suite):
        rs = by_suite[suite]
        c = Counter(r[4] for r in rs)
        print(f"## {suite}  (" + ", ".join(f"{s}={c[s]}" for s in STATUS_ORDER if c[s]) + ")")
        print()
        print("| cuda | slice | jobs | status | metric | detail |")
        print("|---|---|---|---|---|---|")
        for r in rs:
            print(f"| {r[0]} | {r[1]} | {r[3]} | {r[4]} | {r[5][:120]} | {r[6][:160]} |")
        print()

    # per-job tables from the share-style suites
    jobs = []
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "*", "*", "jobs.tsv"))):
            with open(p) as f:
                hdr = None
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if hdr is None:
                        hdr = parts
                        continue
                    jobs.append((os.path.relpath(os.path.dirname(p), root), dict(zip(hdr, parts))))
    if jobs:
        print("## Per-job truth (node-side sampler)")
        print()
        cols = ["job", "jid", "gres", "cuda", "sm_target", "sm_mean", "sm_p95", "mem_peak", "mem_limit", "view", "oom", "kills", "unhooked", "foreign", "verdict", "why"]
        print("| run | " + " | ".join(cols) + " |")
        print("|---|" + "---|" * len(cols))
        for run, j in jobs:
            print(f"| {run} | " + " | ".join(str(j.get(c, "")) for c in cols) + " |")
        print()
        v = Counter(j["verdict"] for _, j in jobs)
        print("Jobs: %d   " % len(jobs) + "   ".join(f"{s}: {v.get(s, 0)}" for s in ["PASS", "LEAK", "FAIL"]))
        print()

    bad = [r for r in rows if r[4] in ("FAIL", "HANG", "PARTIAL")]
    if bad:
        print("## Failures / partials / hangs")
        print()
        for r in bad:
            print(f"- **{r[2]}** {r[0]} {r[1]} jobs={r[3]} {r[4]}: {r[6]}")
        print()


if __name__ == "__main__":
    main()
