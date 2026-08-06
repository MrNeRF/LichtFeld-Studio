#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render perf_campaign/profiles/<label>/summary.md from nsys stats CSVs +
gaps.json. Usage: profile_summary.py <label-dir>  (writes markdown to stdout)."""

import csv
import json
import os
import sys


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        # nsys CSVs may have comment/blank preamble lines before the header.
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    return list(csv.DictReader(lines))


def shorten(name, n=70):
    name = name.strip('"')
    return name if len(name) <= n else name[: n - 1] + "…"


def main():
    d = sys.argv[1]
    meta = {}
    mp = os.path.join(d, "meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    gaps = {}
    gp = os.path.join(d, "gaps.json")
    if os.path.exists(gp):
        gaps = json.load(open(gp))

    print(f"# Profile summary — `{meta.get('label', os.path.basename(d))}`\n")
    print(f"- commit: `{meta.get('commit','?')}`  dataset: `{meta.get('dataset','?')}`")
    print(f"- slice: iters {meta.get('profiled_slice','?')} of {meta.get('iters_total','?')}"
          f"  strategy: {meta.get('strategy','?')}  images: {meta.get('images','?')}")
    print(f"- GPU: {meta.get('gpu','?')}  date: {meta.get('date_utc','?')}\n")

    kern = read_csv(os.path.join(d, "stats_cuda_gpu_kern_sum.csv"))
    if kern:
        print("## Top kernels by total GPU time\n")
        print("| # | time % | total ms | count | avg µs | kernel |")
        print("|--:|-------:|---------:|------:|-------:|:-------|")
        for i, r in enumerate(kern[:20], 1):
            tot_ms = float(r["Total Time (ns)"].replace(",", "")) / 1e6
            avg_us = float(r["Avg (ns)"].replace(",", "")) / 1e3
            cnt = r["Instances"].replace(",", "")
            print(f"| {i} | {r['Time (%)']} | {tot_ms:.2f} | {cnt} | "
                  f"{avg_us:.1f} | `{shorten(r['Name'])}` |")
        print()

    memt = read_csv(os.path.join(d, "stats_cuda_gpu_mem_time_sum.csv"))
    mems = read_csv(os.path.join(d, "stats_cuda_gpu_mem_size_sum.csv"))
    if memt:
        size_by_op = {r["Operation"]: r for r in mems}
        print("## Memory operations\n")
        print("| op | total ms | count | avg µs | total MB |")
        print("|:---|--------:|------:|-------:|---------:|")
        for r in memt:
            op = r["Operation"]
            tot_ms = float(r["Total Time (ns)"].replace(",", "")) / 1e6
            avg_us = float(r["Avg (ns)"].replace(",", "")) / 1e3
            mb = size_by_op.get(op, {}).get("Total (MB)", "?")
            print(f"| {shorten(op,40)} | {tot_ms:.2f} | {r['Count'].replace(',','')} | "
                  f"{avg_us:.1f} | {mb} |")
        print()

    if gaps:
        print("## Launch-gap analysis (CUDA-graphs opportunity)\n")
        print(f"- capture window: **{gaps['window_ms']:.1f} ms**, "
              f"GPU busy {gaps['gpu_busy_ms']:.1f} ms, "
              f"**idle {gaps['gpu_idle_ms']:.1f} ms ({gaps['gpu_idle_pct']}%)**")
        print(f"- ops: {gaps['n_kernels']} kernels, {gaps['n_memcpy']} memcpy, "
              f"{gaps['n_memset']} memset; {gaps['n_busy_islands']} busy islands\n")
        print("| gap bucket | count | total ms |")
        print("|:---|--:|--:|")
        for k, v in gaps["gap_histogram"].items():
            print(f"| {k} | {v['count']} | {v['total_ms']:.2f} |")
        pi = gaps.get("per_iteration") or {}
        if pi:
            print("\nPer-iteration medians (NVTX `train_step` attribution):\n")
            print(f"- kernel launches/iter: **{pi['median_kernel_launches_per_iter']:.0f}**")
            print(f"- span {pi['median_span_us']:.0f} µs = busy {pi['median_busy_us']:.0f} µs "
                  f"+ gap **{pi['median_gap_us']:.0f} µs ({pi['median_gap_pct_of_span']}% of span)**")
            print(f"- iterations attributed: {pi['n_iterations_attributed']}")
        print()


if __name__ == "__main__":
    main()
