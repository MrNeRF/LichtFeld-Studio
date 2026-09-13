#!/usr/bin/env python3
"""Paired A/B evaluation of tensor_backend_corpus --time captures for the G5 gate.

Layout: DIR/ref_<i>/timing.txt and DIR/new_<i>/timing.txt for i in 1..pairs,
recorded interleaved (reference build, candidate build, reference, ...). A row
fails when the candidate median exceeds the reference median by more than
max(threshold percent, floor microseconds) in at least `fail-when-worse-in`
of the pairs. Prints every failing row with all its deltas and exits nonzero.
"""
import argparse
import re
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


def check_clocks(path: Path, minimum_sm_mhz: float) -> bool:
    """Every sample after the first (taken before the GPU ramps up) must stay at or above the floor."""
    samples = []
    for line in path.read_text().splitlines()[1:]:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        match = re.match(r"(\d+)", parts[1])
        if match is None:
            continue
        samples.append((parts[0], float(match.group(1)), parts[3]))
    if len(samples) < 2:
        print(f"clock log {path} has {len(samples)} samples; cannot judge the GPU state",
              file=sys.stderr)
        return False
    body = samples[1:]
    low = [sample for sample in body if sample[1] < minimum_sm_mhz]
    clocks = sorted(sample[1] for sample in body)
    states = sorted({sample[0] for sample in body})
    print(f"GPU clock log: {len(samples)} samples, SM clock min {clocks[0]:.0f} median "
          f"{clocks[len(clocks) // 2]:.0f} max {clocks[-1]:.0f} MHz, pstates {' '.join(states)}, "
          f"temperature {min(int(s[2]) for s in body)} to {max(int(s[2]) for s in body)} C")
    if low:
        print(f"{len(low)} samples below {minimum_sm_mhz:.0f} MHz after the first: the box was "
              "not in its measurement state; rerun", file=sys.stderr)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--fail-when-worse-in", type=int, default=4)
    parser.add_argument("--threshold-pct", type=float, default=2.0)
    parser.add_argument("--floor-us", type=float, default=0.6)
    parser.add_argument("--exclude", default="",
                        help="regex over the row key; matching rows are reported but do not "
                             "count toward the verdict (rows whose timed work differs "
                             "between the two harness sources)")
    parser.add_argument("--clocks", type=Path, default=None,
                        help="nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem,temperature.gpu "
                             "CSV sampled during the run; the run is rejected when any sample "
                             "after the first reports an SM clock below --min-sm-mhz")
    parser.add_argument("--min-sm-mhz", type=float, default=2400.0)
    args = parser.parse_args()

    if args.clocks is not None and not check_clocks(args.clocks, args.min_sm_mhz):
        return 2

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
    excluded = []
    exclude = re.compile(args.exclude) if args.exclude else None
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
            if exclude and exclude.search(row):
                excluded.append((row, deltas))
            else:
                failing.append((row, deltas))
    if excluded:
        print(f"{len(excluded)} rows excluded from the verdict (harness differs):")
        for row, deltas in excluded:
            print("  " + row + "  deltas us: " + " ".join(f"{d:+.3f}" for d in deltas))

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
