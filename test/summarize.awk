#!/usr/bin/awk -f
# Read results.tsv and emit a Markdown summary.
# Columns: cuda_ver, slice, suite, jobid, status, metric, detail

BEGIN {
    FS = "\t"
    OFS = "\t"
    n_pass = 0; n_fail = 0; n_partial = 0; n_skip = 0
}

NR == 1 { next }  # header

{
    cuda = $1; slice = $2; suite = $3; jobid = $4
    status = $5; metric = $6; detail = $7
    rows[NR] = $0

    # Distinct axes
    versions[cuda] = 1
    slices[slice] = 1
    suites[suite] = 1

    cell = cuda "|" slice "|" suite
    cell_status[cell] = status
    cell_metric[cell] = metric
    cell_detail[cell] = detail

    if (status == "PASS")    n_pass++
    else if (status == "FAIL") n_fail++
    else if (status == "PARTIAL") n_partial++
    else if (status == "SKIP") n_skip++
    else if (status == "HANG") n_hang++
    else if (status == "INFO") n_info++
    else if (status == "LEAK") n_leak++
}

END {
    total = n_pass + n_fail + n_partial + n_skip + n_hang + n_info + n_leak
    print "# SoftMig multi-CUDA test matrix summary"
    print ""
    print "Generated: " strftime("%Y-%m-%d %H:%M:%S")
    print ""
    print "Total: " total "   PASS: " n_pass "   PARTIAL: " n_partial "   FAIL: " n_fail "   SKIP: " n_skip "   HANG: " n_hang+0 "   INFO: " n_info+0 "   LEAK: " n_leak+0
    print ""

    # --- per-version x slice x suite table ---
    # Collect deterministic orders
    ver_order = "12.2 12.6 12.9 13.2 mixed"
    slice_order = "l40s.2 l40s.4"
    suite_order = "smoke direct sm oom crossjob pool passive mixed soak nvsmi"
    pv_set = " smoke direct sm oom crossjob pool passive "
    n_v = split(ver_order, VS, " ")
    n_sl = split(slice_order, SL, " ")
    n_su = split(suite_order, SU, " ")

    # --- per-version breakdown ---
    for (vi = 1; vi <= n_v; vi++) {
        v = VS[vi]
        if (!(v in versions)) continue
        print "## cuda/" v
        print ""
        # header row
        printf("| suite |")
        for (si = 1; si <= n_sl; si++) { s = SL[si]; if (s in slices) printf(" %s |", s) }
        print ""
        printf("|---|")
        for (si = 1; si <= n_sl; si++) { s = SL[si]; if (s in slices) printf("---|") }
        print ""
        for (ui = 1; ui <= n_su; ui++) {
            su = SU[ui]
            if (!(su in suites)) continue
            # Skip non-per-version suites here (they're printed in one-offs section)
            if (index(pv_set, " " su " ") == 0) continue
            printf("| %s |", su)
            for (si = 1; si <= n_sl; si++) {
                s = SL[si]; if (!(s in slices)) continue
                cell = v "|" s "|" su
                st = (cell in cell_status) ? cell_status[cell] : "-"
                mt = (cell in cell_metric) ? cell_metric[cell] : ""
                printf(" %s (%s) |", st, mt)
            }
            print ""
        }
        print ""
    }

    # --- one-offs: every suite that is not part of the per-version grid ---
    any_oneoff = 0
    for (r = 2; r <= NR; r++) {
        if (!(r in rows)) continue
        split(rows[r], f, "\t")
        su = f[3]
        if (index(pv_set, " " su " ") > 0) continue
        if (!any_oneoff) { print "## One-offs"; print ""; print "| suite | cuda | slice | status | metric | detail |"; print "|---|---|---|---|---|---|"; any_oneoff = 1 }
        printf("| %s | %s | %s | %s | %s | %s |\n", su, f[1], f[2], f[5], substr(f[6], 1, 100), substr(f[7], 1, 140))
    }
    if (any_oneoff) print ""

    # --- failures detail ---
    any_fail = 0
    for (r in rows) {
        split(rows[r], f, "\t")
        if (f[5] == "FAIL" || f[5] == "PARTIAL" || f[5] == "HANG" || f[5] == "LEAK") {
            if (!any_fail) { print "## Failures / partials"; print ""; print "| cuda | slice | suite | jobid | status | detail |"; print "|---|---|---|---|---|---|"; any_fail = 1 }
            printf("| %s | %s | %s | %s | %s | %s |\n", f[1], f[2], f[3], f[4], f[5], f[7])
        }
    }
    if (any_fail) print ""
}
