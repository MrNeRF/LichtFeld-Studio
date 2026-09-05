#!/usr/bin/env python3
"""Paired A/B evaluation of tensor_backend_corpus --time captures for the G5 gate.

Layout: DIR/ref_<i>/timing.txt and DIR/new_<i>/timing.txt for i in 1..pairs,
recorded interleaved (reference build, candidate build, reference, ...). A row
fails when the candidate median exceeds the reference median by more than
max(threshold percent, floor microseconds) in at least `fail-when-worse-in`
of the pairs. Prints every failing row with all its deltas and exits nonzero.
"""
import argparse
import sys
from pathlib import Path


def read_timing(path: Path) -> dict[str, float]:
    rows = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] == "aggregate_median_ms":
            continue
        rows[" ".join(parts[:3])] = float(parts[3])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--fail-when-worse-in", type=int, default=4)
    parser.add_argument("--threshold-pct", type=float, default=2.0)
    parser.add_argument("--floor-us", type=float, default=0.6)
    args = parser.parse_args()

    references = []
    candidates = []
    for index in range(1, args.pairs + 1):
        ref = args.directory / f"ref_{index}" / "timing.txt"
        new = args.directory / f"new_{index}" / "timing.txt"
        if not ref.is_file() or not new.is_file():
            print(f"missing pair {index}: {ref} or {new}", file=sys.stderr)
            return 2
        references.append(read_timing(ref))
        candidates.append(read_timing(new))

    rows = sorted(set().union(*[set(r) for r in references]))
    failing = []
    aggregate_ref = [sum(r.values()) / 1000.0 for r in references]
    aggregate_new = [sum(c.values()) / 1000.0 for c in candidates]
    for row in rows:
        deltas = []
        worse = 0
        for ref, new in zip(references, candidates):
            if row not in ref or row not in new:
                continue
            delta = new[row] - ref[row]
            deltas.append(delta)
            bound = max(ref[row] * args.threshold_pct / 100.0, args.floor_us)
            if delta > bound:
                worse += 1
        if worse >= args.fail_when_worse_in:
            failing.append((row, deltas))

    print("aggregate median ms per pair (reference, candidate):")
    for index, (a, b) in enumerate(zip(aggregate_ref, aggregate_new), start=1):
        print(f"  pair {index}: {a:.3f} {b:.3f}")
    if failing:
        print(f"{len(failing)} rows worse in at least {args.fail_when_worse_in} of {args.pairs} pairs:")
        for row, deltas in failing:
            print("  " + row + "  deltas us: " + " ".join(f"{d:+.3f}" for d in deltas))
        return 1
    print(f"no row worse in {args.fail_when_worse_in} of {args.pairs} pairs beyond "
          f"max({args.threshold_pct} percent, {args.floor_us} us)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
