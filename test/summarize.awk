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
}

END {
    total = n_pass + n_fail + n_partial + n_skip
    print "# SoftMig multi-CUDA test matrix summary"
    print ""
    print "Generated: " strftime("%Y-%m-%d %H:%M:%S")
    print ""
    print "Total: " total "   PASS: " n_pass "   PARTIAL: " n_partial "   FAIL: " n_fail "   SKIP: " n_skip
    print ""

    # --- per-version x slice x suite table ---
    # Collect deterministic orders
    ver_order = "12.2 12.6 12.9 13.2 mixed"
    slice_order = "l40s.2 l40s.4"
    suite_order = "smoke direct sm oom crossjob mixed soak nvsmi"
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
            if (su == "mixed" || su == "soak" || su == "nvsmi") continue
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

    # --- one-offs ---
    any_oneoff = 0
    for (ui = 1; ui <= n_su; ui++) {
        su = SU[ui]
        if (su != "mixed" && su != "soak" && su != "nvsmi") continue
        if (!(su in suites)) continue
        if (!any_oneoff) { print "## One-offs"; print ""; print "| suite | status | metric | detail |"; print "|---|---|---|---|"; any_oneoff = 1 }
        # one-offs are keyed under a distinct cuda_ver/slice (mixed/l40s.4, 12.2/l40s.4, ...)
        # just iterate all cells that match the suite
        # (We stashed every row; re-scan the file via ARGV? Simpler: store each suite's row once.)
    }

    # Re-scan all rows to print one-offs (we had to do a second pass because we
    # only kept per-version cells above).
    if (any_oneoff) {
        # Nothing more to do here — print from the remembered lines:
        for (r in rows) {
            split(rows[r], f, "\t")
            su = f[3]
            if (su == "mixed" || su == "soak" || su == "nvsmi") {
                printf("| %s | %s | %s | %s |\n", su, f[5], f[6], f[7])
            }
        }
        print ""
    }

    # --- failures detail ---
    any_fail = 0
    for (r in rows) {
        split(rows[r], f, "\t")
        if (f[5] == "FAIL" || f[5] == "PARTIAL") {
            if (!any_fail) { print "## Failures / partials"; print ""; print "| cuda | slice | suite | jobid | status | detail |"; print "|---|---|---|---|---|---|"; any_fail = 1 }
            printf("| %s | %s | %s | %s | %s | %s |\n", f[1], f[2], f[3], f[4], f[5], f[7])
        }
    }
    if (any_fail) print ""
}
