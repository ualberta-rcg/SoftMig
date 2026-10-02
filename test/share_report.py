#!/usr/bin/env python3
"""Per-job share/leak report for one multi-job scenario directory.

Joins the root-side sampler (OUT/sampler: pmon.log, apps.log, pidmap.txt)
with each job's own artifacts (OUT/jobs/<name>: jid, conf, run.out,
softmig.log, smi_apps.txt) and prints one row per job:

  target SM%, achieved SM% (mean/p50/p95 after warm-up, summed over the
  job's PIDs), memory peak vs limit (truth), over-limit MiB, OOM returns,
  OOM-killer kills, softmig ERROR / UNHOOKED / lock-wait lines, other jobs'
  PIDs visible from inside the job, and a verdict.

Usage: share_report.py OUTDIR [--scenario NAME] [--warmup 12] [--tol 10]
                       [--tsv FILE] [--md FILE]
Exit code 0 even on FAIL rows (the caller decides); the summary line is last.
"""
import argparse
import calendar
import os
import re
import statistics
import sys
import time
from collections import defaultdict

CTX_SLACK_MIB = 256   # context + allocator rounding allowed above the limit


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def read_int(path, default=None):
    s = read(path).strip()
    try:
        return int(float(s))
    except ValueError:
        return default


def load_pidmap(sdir):
    m = {}
    for line in read(os.path.join(sdir, "pidmap.txt")).splitlines():
        f = line.split()
        if len(f) >= 5:
            m[int(f[0])] = {"uid": f[1], "job": f[2], "step": f[3], "comm": f[4]}
    return m


def load_pmon(sdir):
    """-> list of (epoch, gpu, pid, sm, fb)"""
    rows = []
    for line in read(os.path.join(sdir, "pmon.log")).splitlines():
        if line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 12 or f[3] == "-":
            continue
        try:
            # pmon prints the node's local time; the nodes run UTC
            t = calendar.timegm(time.strptime(f[0] + " " + f[1], "%Y%m%d %H:%M:%S"))
            sm = 0.0 if f[5] == "-" else float(f[5])
            fb = 0.0 if f[11] == "-" else float(f[11])
            rows.append((int(t), int(f[2]), int(f[3]), sm, fb))
        except ValueError:
            continue
    return rows


def load_apps(sdir):
    """-> list of (epoch, uuid, pid, used_mib)"""
    rows = []
    for line in read(os.path.join(sdir, "apps.log")).splitlines():
        f = [x.strip() for x in re.split(r"[ ,]+", line.strip()) if x.strip()]
        if len(f) >= 4:
            try:
                rows.append((int(f[0]), f[1], int(f[2]), int(f[3])))
            except ValueError:
                continue
    return rows


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--scenario", default=os.path.basename(os.path.normpath(sys.argv[1])))
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--tol", type=float, default=10.0, help="SM tolerance pp (isolated)")
    ap.add_argument("--oversub", action="store_true", help="widen SM tolerance to 20pp")
    ap.add_argument("--tsv")
    ap.add_argument("--md")
    a = ap.parse_args()

    sdir = os.path.join(a.out, "sampler")
    jdir = os.path.join(a.out, "jobs")
    pidmap = load_pidmap(sdir)
    pmon = load_pmon(sdir)
    apps = load_apps(sdir)
    tol = 20.0 if a.oversub else a.tol

    jobs = sorted(d for d in os.listdir(jdir) if os.path.isdir(os.path.join(jdir, d))) if os.path.isdir(jdir) else []
    rows, verdicts = [], []
    for name in jobs:
        D = os.path.join(jdir, name)
        jid = read(os.path.join(D, "jid.txt")).strip() or "NA"
        gres = read(os.path.join(D, "gres.txt")).strip()
        cuda = read(os.path.join(D, "cuda.txt")).strip()
        uuid = read(os.path.join(D, "uuid.txt")).strip()
        conf = read(os.path.join(D, "conf.txt"))
        m = re.search(r"CUDA_DEVICE_MEMORY_LIMIT=([0-9.]+)M", conf)
        mem_limit = int(float(m.group(1))) if m else 0
        m = re.search(r"CUDA_DEVICE_SM_LIMIT=([0-9]+)", conf)
        sm_target = int(m.group(1)) if m else 100
        passive = mem_limit == 0
        t0 = read_int(os.path.join(D, "t_start.txt"), 0)
        t1 = read_int(os.path.join(D, "t_end.txt"), 1 << 40)
        rc = read(os.path.join(D, "rc.txt")).strip() or "NA"
        run_out = read(os.path.join(D, "run.out"))
        slog = read(os.path.join(D, "softmig.log"))

        my_pids = {p for p, info in pidmap.items() if info["job"] == jid}
        # in-job view: PIDs it listed that belong to other jobs / no job
        foreign = defaultdict(int)
        samples = 0
        for line in read(os.path.join(D, "smi_apps.txt")).splitlines():
            f = [x for x in re.split(r"[ ,]+", line.strip()) if x]
            if len(f) < 2:
                continue
            samples += 1
            try:
                p = int(f[1])
            except ValueError:
                continue
            if p in my_pids:
                continue
            owner = pidmap.get(p, {}).get("job", "unknown")
            if owner != jid:
                foreign[owner] += 1
        n_foreign_samples = sum(foreign.values())

        # truth: SM% summed over my pids per second, window = [t0+warmup, t1]
        per_t_sm = defaultdict(float)
        per_t_fb = defaultdict(float)
        for (t, gpu, pid, sm, fb) in pmon:
            if pid in my_pids and t0 + a.warmup <= t <= t1:
                per_t_sm[t] += sm
                per_t_fb[t] += fb
        sm_vals = list(per_t_sm.values())
        sm_mean = statistics.mean(sm_vals) if sm_vals else 0.0
        sm_p50 = pct(sm_vals, 50)
        sm_p95 = pct(sm_vals, 95)
        per_t_mem = defaultdict(int)
        for (t, u, pid, used) in apps:
            if pid in my_pids and t0 <= t <= t1 + 2:
                per_t_mem[t] += used
        mem_peak = max(per_t_mem.values()) if per_t_mem else int(max(per_t_fb.values()) if per_t_fb else 0)
        over = mem_peak - mem_limit if mem_limit else 0

        oom_ret = len(re.findall(r"out of memory|OUT_OF_MEMORY|CUDA_ERROR_OUT_OF_MEMORY", run_out, re.I))
        oom_dev = len(re.findall(r"Device \d+ OOM", slog))
        kills = len(re.findall(r"KILLING PID", slog))
        errors = len(re.findall(r"softmig ERROR|\[ERROR\]", slog))
        unhooked = slog.count("UNHOOKED")
        lockwait = len(re.findall(r"Waiting \d+s for shrreg lock", slog))
        owner_died = len(re.findall(r"owner died|EOWNERDEAD|recovered", slog, re.I))
        garbage = len(re.findall(r"set_task_pid.*fail|invalid pid|garbage", slog, re.I))
        done = re.findall(r"DONE: (\d+) launches in ([\d.]+) s", run_out)
        lps = sum(int(n) / float(t) for n, t in done if float(t) > 0)
        region = read(os.path.join(D, "region_after.txt")).strip()
        view_total = 0
        for line in read(os.path.join(D, "smi_gpu.txt")).splitlines():
            f = [x for x in re.split(r"[ ,]+", line.strip()) if x]
            if len(f) >= 4:
                try:
                    view_total = max(view_total, int(f[3]))
                except ValueError:
                    pass
        # optional per-job expectations written by the suite (expect.txt: key=value lines)
        expect = {}
        for line in read(os.path.join(D, "expect.txt")).splitlines():
            if "=" in line:
                k, v = line.strip().split("=", 1)
                expect[k] = v

        why = []
        if "limit" in expect and int(expect["limit"]) != mem_limit:
            why.append(f"limit {mem_limit}!={expect['limit']}")
        if "sm" in expect and int(expect["sm"]) != (0 if passive else sm_target):
            why.append(f"sm_limit {sm_target}!={expect['sm']}")
        if "min_mem" in expect and mem_peak < int(expect["min_mem"]):
            why.append(f"mem {mem_peak}<{expect['min_mem']}")
        if "min_sm" in expect and sm_vals and sm_mean < float(expect["min_sm"]):
            why.append(f"sm {sm_mean:.0f}<{expect['min_sm']}")
        if "max_sm" in expect and sm_vals and sm_mean > float(expect["max_sm"]):
            why.append(f"sm {sm_mean:.0f}>{expect['max_sm']}")
        if "oom" in expect and (oom_ret > 0) != (expect["oom"] == "1"):
            why.append("oom-expected" if expect["oom"] == "1" else f"unexpected-oom:{oom_ret}")
        if "view_total" in expect and view_total and abs(view_total - int(expect["view_total"])) > 64:
            why.append(f"view_total {view_total}!={expect['view_total']}")
        if "rc" in expect and rc != expect["rc"]:
            why.append(f"rc {rc}!={expect['rc']}")
        if rc not in ("0",) and "rc" not in expect:
            # OOM-probe jobs are expected to exit non-zero; caller tags them
            if not name.endswith("_oom"):
                why.append(f"rc={rc}")
        if passive:
            if region:
                why.append("passive-has-region")
            if oom_dev or kills:
                why.append("passive-enforced")
        else:
            if sm_vals and abs(sm_mean - sm_target) > tol and sm_mean > sm_target:
                why.append(f"sm {sm_mean:.0f}>{sm_target}+{tol:.0f}")
            if mem_limit and over > CTX_SLACK_MIB:
                why.append(f"mem +{over}MiB over limit")
            if mem_limit and view_total and abs(view_total - mem_limit) > 64:
                why.append(f"view_total {view_total}!=limit")
        if unhooked:
            why.append(f"unhooked={unhooked}")
        if garbage:
            why.append(f"garbage={garbage}")
        if n_foreign_samples:
            why.append("sees-other-jobs:" + ",".join(f"{k}x{v}" for k, v in foreign.items()))
        verdict = "PASS" if not why else "FAIL"
        # visibility alone is a LEAK (not a limit failure) — call it out separately
        if why and all(w.startswith("sees-other-jobs") for w in why):
            verdict = "LEAK"
        verdicts.append(verdict)

        rows.append(dict(
            scenario=a.scenario, job=name, jid=jid, gres=gres, cuda=cuda, uuid=uuid[-8:],
            pids=len(my_pids), sm_target=("-" if passive else sm_target),
            sm_mean=f"{sm_mean:.1f}", sm_p50=f"{sm_p50:.0f}", sm_p95=f"{sm_p95:.0f}", n=len(sm_vals),
            mem_peak=mem_peak, mem_limit=(mem_limit or "-"), over=(over if mem_limit else "-"), view=view_total,
            oom=f"{oom_ret}/{oom_dev}", kills=kills, err=errors, unhooked=unhooked,
            lock=lockwait, died=owner_died, foreign=n_foreign_samples, samples=samples, lps=f"{lps:.0f}",
            rc=rc, verdict=verdict, why=" ".join(why)))

    hdr = ["job", "jid", "gres", "cuda", "uuid", "pids", "sm_target", "sm_mean", "sm_p50", "sm_p95", "n",
           "mem_peak", "mem_limit", "over", "view", "oom", "kills", "err", "unhooked", "lock", "died",
           "foreign", "samples", "lps", "rc", "verdict", "why"]
    md = [f"### {a.scenario}", "",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        md.append("| " + " | ".join(str(r[h]) for h in hdr) + " |")
    md.append("")
    md.append("oom = returns-in-app/Device-OOM-in-log; foreign = in-job nvidia-smi samples listing another job's PID; "
              f"SM window skips first {a.warmup}s; tolerance ±{tol:.0f}pp; mem slack {CTX_SLACK_MIB} MiB")
    text = "\n".join(md)
    print(text)
    if a.md:
        with open(a.md, "a") as f:
            f.write(text + "\n\n")
    if a.tsv:
        new = not os.path.exists(a.tsv)
        with open(a.tsv, "a") as f:
            if new:
                f.write("scenario\t" + "\t".join(hdr) + "\n")
            for r in rows:
                f.write(r["scenario"] + "\t" + "\t".join(str(r[h]) for h in hdr) + "\n")
    n_fail = verdicts.count("FAIL"); n_leak = verdicts.count("LEAK")
    status = "PASS" if not n_fail and not n_leak else ("LEAK" if not n_fail else "FAIL")
    print(f"SUMMARY\t{status}\tjobs={len(rows)} fail={n_fail} leak={n_leak}")


if __name__ == "__main__":
    main()
