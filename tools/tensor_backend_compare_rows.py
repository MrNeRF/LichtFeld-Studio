#!/usr/bin/env python3
"""Numeric comparison of selected corpus rows between two --dump directories.

Usage: tensor_backend_compare_rows.py REF_DIR NEW_DIR [--rows SUBSTRING ...]
Both directories hold manifest.txt and <index>.bin from tensor_backend_corpus --dump.
Rows are matched by their manifest key (launcher, call, profile, dtype); tolerance
rows are compared element-wise as float32 with rtol 1e-5 scaled by log2(n) and atol
1e-6, and the maximum relative error is printed. Digest rows must match textually.
"""
import argparse
import math
import struct
import sys
from pathlib import Path


def load(directory: Path) -> dict[str, tuple[int, str]]:
    rows = {}
    for index, line in enumerate(Path(directory, "manifest.txt").read_text().splitlines()):
        parts = line.split()
        key = " ".join(parts[:4])
        rows[key] = (index, line)
    return rows


def floats(path: Path) -> list[float]:
    data = path.read_bytes()
    return list(struct.unpack("<%df" % (len(data) // 4), data[: len(data) // 4 * 4]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rows", nargs="*", default=[])
    args = parser.parse_args()
    reference = load(args.reference)
    candidate = load(args.candidate)
    failures = 0
    for key, (ref_index, ref_line) in sorted(reference.items()):
        if args.rows and not any(sub in key for sub in args.rows):
            continue
        if key not in candidate:
            print(f"missing in candidate: {key}")
            failures += 1
            continue
        new_index, new_line = candidate[key]
        rule = ref_line.split()[4]
        scan = rule.startswith("tolerance-scan:")
        if not scan and not rule.startswith("tolerance:"):
            if ref_line != new_line:
                print(f"DIFF {key}")
                failures += 1
            continue
        a = floats(args.reference / f"{ref_index}.bin")
        b = floats(args.candidate / f"{new_index}.bin")
        if len(a) != len(b):
            print(f"FAIL {key}: element count {len(a)} vs {len(b)}")
            failures += 1
            continue
        rtol = 1e-5 * max(1.0, math.log2(max(2, len(a))))
        # Prefix-sum rows scale their bound with the row's largest magnitude.
        floor = max((abs(x) for x in a if math.isfinite(x)), default=0.0) if scan else 0.0
        worst = 0.0
        bad = 0
        for x, y in zip(a, b):
            if math.isnan(x) or math.isnan(y):
                bad += math.isnan(x) != math.isnan(y)
                continue
            error = abs(x - y)
            worst = max(worst, error / abs(x) if x else error)
            if error > 1e-6 + rtol * max(abs(x), floor):
                bad += 1
        status = "ok" if bad == 0 else "FAIL"
        print(f"{status} {key}: n={len(a)} max_rel_error={worst:.3e} violations={bad}")
        failures += bad != 0
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
