#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Launch-gap analysis from an nsys sqlite export.

Quantifies GPU idle time between consecutive device operations inside the
profiled steady-state slice — the upper bound of what CUDA-graph capture /
launch-latency work could recover.

Usage: launch_gaps.py <timeline.sqlite> <out.json>

Method:
  * GPU busy = interval union of all kernels + memcpys + memsets (all streams).
  * Window   = [first op start, last op end] of the capture slice.
  * Idle     = window - busy, split into a gap histogram.
  * Per-iteration attribution via NVTX "train_step:<iter>" CPU ranges:
    kernels are mapped to iterations through the CUDA runtime launch rows
    (correlationId) whose CPU timestamp falls inside the NVTX range on the
    same thread; per-iteration span/busy/gap and launch counts are medianed.
"""

import json
import sqlite3
import statistics
import sys


def table_exists(cur, name):
    return cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def load_intervals(cur, table, extra_cols=""):
    if not table_exists(cur, table):
        return []
    cols = "start, end" + (", " + extra_cols if extra_cols else "")
    return cur.execute(f"SELECT {cols} FROM {table}").fetchall()


def union_busy(intervals):
    """Sum of the interval union; also returns sorted merged intervals."""
    if not intervals:
        return 0, []
    ivs = sorted((s, e) for s, e in intervals)
    merged = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged), merged


def main():
    db_path, out_path = sys.argv[1], sys.argv[2]
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    kernels = load_intervals(
        cur, "CUPTI_ACTIVITY_KIND_KERNEL", "streamId, correlationId, shortName"
    )
    memcpys = load_intervals(cur, "CUPTI_ACTIVITY_KIND_MEMCPY", "bytes")
    memsets = load_intervals(cur, "CUPTI_ACTIVITY_KIND_MEMSET", "bytes")
    if not kernels:
        print("ERROR: no kernel rows in export", file=sys.stderr)
        sys.exit(1)

    all_ops = [(r[0], r[1]) for r in kernels]
    all_ops += [(r[0], r[1]) for r in memcpys]
    all_ops += [(r[0], r[1]) for r in memsets]
    busy_ns, merged = union_busy(all_ops)
    window_start = merged[0][0]
    window_end = merged[-1][1]
    window_ns = window_end - window_start
    idle_ns = window_ns - busy_ns

    # Gap histogram between consecutive merged busy intervals.
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    hist = {"lt_2us": [0, 0], "2_10us": [0, 0], "10_100us": [0, 0], "gt_100us": [0, 0]}
    for g in gaps:
        key = ("lt_2us" if g < 2_000 else
               "2_10us" if g < 10_000 else
               "10_100us" if g < 100_000 else "gt_100us")
        hist[key][0] += 1
        hist[key][1] += g

    # ---- per-iteration attribution via NVTX train_step ranges ----
    per_iter = {}
    if table_exists(cur, "NVTX_EVENTS"):
        nvtx = cur.execute(
            "SELECT start, end, globalTid, text FROM NVTX_EVENTS "
            "WHERE text LIKE 'train_step:%' AND end IS NOT NULL ORDER BY start"
        ).fetchall()
        runtime = {}
        if table_exists(cur, "CUPTI_ACTIVITY_KIND_RUNTIME"):
            for s, e, tid, corr in cur.execute(
                "SELECT start, end, globalTid, correlationId "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME"
            ):
                runtime[corr] = (s, tid)
        # kernel correlation -> gpu interval
        kern_by_corr = {}
        for s, e, stream, corr, _name in kernels:
            kern_by_corr.setdefault(corr, []).append((s, e))
        iter_rows = []
        for ns_, ne_, ntid, text in nvtx:
            ops = []
            n_launch = 0
            for corr, (rs, rtid) in runtime.items():
                if rtid == ntid and ns_ <= rs <= ne_ and corr in kern_by_corr:
                    ops.extend(kern_by_corr[corr])
                    n_launch += len(kern_by_corr[corr])
            if len(ops) < 2:
                continue
            b, m = union_busy(ops)
            span = m[-1][1] - m[0][0]
            iter_rows.append(
                {"iter": text.split(":")[1], "launches": n_launch,
                 "span_us": span / 1e3, "busy_us": b / 1e3,
                 "gap_us": (span - b) / 1e3}
            )
        if iter_rows:
            med = lambda k: statistics.median(r[k] for r in iter_rows)
            per_iter = {
                "n_iterations_attributed": len(iter_rows),
                "median_kernel_launches_per_iter": med("launches"),
                "median_span_us": round(med("span_us"), 1),
                "median_busy_us": round(med("busy_us"), 1),
                "median_gap_us": round(med("gap_us"), 1),
                "median_gap_pct_of_span": round(
                    100 * med("gap_us") / med("span_us"), 1),
            }

    result = {
        "window_ms": round(window_ns / 1e6, 3),
        "gpu_busy_ms": round(busy_ns / 1e6, 3),
        "gpu_idle_ms": round(idle_ns / 1e6, 3),
        "gpu_idle_pct": round(100 * idle_ns / window_ns, 2),
        "n_kernels": len(kernels),
        "n_memcpy": len(memcpys),
        "n_memset": len(memsets),
        "n_busy_islands": len(merged),
        "gap_histogram": {
            k: {"count": v[0], "total_ms": round(v[1] / 1e6, 3)}
            for k, v in hist.items()
        },
        "per_iteration": per_iter,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
