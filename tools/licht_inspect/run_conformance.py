#!/usr/bin/env python3
"""Single entry point for the spec-authoritative .licht conformance battery."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Callable, Sequence

try:
    from .conformance_adversaries import run_semantic_adversaries
    from .conformance_chapters import run_chapter_structural_hostility
    from .conformance_common import (
        DEFAULT_SEED,
        CategoryResult,
        config_for_mode,
        timed_result,
    )
    from .conformance_fuzz import run_mutation_fuzzer
    from .conformance_geometry import run_geometry_payloads
    from .conformance_lifecycle import run_append_lifecycle
    from .conformance_scale import run_scale_edges
    from .conformance_truncation import run_truncation_sweep
    from .conformance_verifier import run_independent_verifier
    from .oracle_corpus import PREVIEW_CORRUPTION_CASE_COUNT
    from .make_fixtures import check_fixtures
    from .run_selftest import main as selftest_main
except ImportError:  # Direct script execution.
    from conformance_adversaries import run_semantic_adversaries
    from conformance_chapters import run_chapter_structural_hostility
    from conformance_common import (
        DEFAULT_SEED,
        CategoryResult,
        config_for_mode,
        timed_result,
    )
    from conformance_fuzz import run_mutation_fuzzer
    from conformance_geometry import run_geometry_payloads
    from conformance_lifecycle import run_append_lifecycle
    from conformance_scale import run_scale_edges
    from conformance_truncation import run_truncation_sweep
    from conformance_verifier import run_independent_verifier
    from oracle_corpus import PREVIEW_CORRUPTION_CASE_COUNT
    from make_fixtures import check_fixtures
    from run_selftest import main as selftest_main


def _seed_value(text: str) -> int:
    try:
        return int(text, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer (decimal or 0x...)") from error


def _baseline_selftest(fixture_dir: Path, oracle_cases: int) -> CategoryResult:
    started = time.monotonic()
    check_fixtures(fixture_dir)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        result = selftest_main(
            [
                "--fixture-dir",
                str(fixture_dir),
                "--oracle-sample",
                str(oracle_cases),
            ]
        )
    if result != 0:
        raise AssertionError("baseline selftest failed:\n" + captured.getvalue())
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    fixture_count = len(manifest["fixtures"])
    return timed_result(
        "golden-oracle-baseline",
        started,
        oracle_cases + fixture_count + PREVIEW_CORRUPTION_CASE_COUNT,
        f"fixtures={fixture_count}, randomized_oracle={oracle_cases}, "
        f"preview_corruptions={PREVIEW_CORRUPTION_CASE_COUNT}",
    )


def _print_result(result: CategoryResult) -> None:
    print(
        f"PASS {result.name}: cases={result.cases} time={result.seconds:.3f}s; "
        f"{result.detail}",
        flush=True,
    )


def _run_category(
    label: str,
    action: Callable[[], CategoryResult],
) -> CategoryResult:
    print(f"RUN  {label}", flush=True)
    try:
        result = action()
    except Exception:
        print(f"FAIL {label}", flush=True)
        traceback.print_exc()
        raise
    _print_result(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--quick", action="store_true", help="CI battery, target under 60 seconds")
    modes.add_argument("--full", action="store_true", help="exhaustive battery, expected to take minutes")
    modes.add_argument(
        "--fuzz-minutes",
        type=float,
        metavar="N",
        help="run only the seeded mutation fuzzer for N minutes",
    )
    parser.add_argument("--seed", type=_seed_value, default=DEFAULT_SEED)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tests/fixtures/licht",
    )
    args = parser.parse_args(argv)
    if args.fuzz_minutes is not None and args.fuzz_minutes <= 0:
        parser.error("--fuzz-minutes must be greater than zero")
    if not (args.fixture_dir / "manifest.json").is_file():
        parser.error(f"fixture manifest not found under {args.fixture_dir}")

    mode = "quick" if args.quick else "full" if args.full else "fuzz"
    print("LichtFeld .licht conformance battery")
    print(f"mode: {mode}")
    print(f"seed: 0x{args.seed:x}")
    print(f"fixtures: {args.fixture_dir}")
    started = time.monotonic()
    results: list[CategoryResult] = []

    try:
        if mode == "fuzz":
            results.append(
                _run_category(
                    "mutation-fuzz",
                    lambda: run_mutation_fuzzer(
                        args.fixture_dir,
                        seed=args.seed,
                        per_case_timeout_seconds=1.0,
                        fuzz_minutes=args.fuzz_minutes,
                    ),
                )
            )
        else:
            config = config_for_mode(mode, args.seed)
            oracle_cases = 200 if mode == "quick" else 10_000
            results.append(
                _run_category(
                    "golden-oracle-baseline",
                    lambda: _baseline_selftest(args.fixture_dir, oracle_cases),
                )
            )
            results.append(
                _run_category(
                    "truncation",
                    lambda: run_truncation_sweep(args.fixture_dir, config),
                )
            )
            results.append(
                _run_category(
                    "semantic-adversaries",
                    lambda: run_semantic_adversaries(config),
                )
            )
            results.append(
                _run_category(
                    "chapter-structural-hostility",
                    run_chapter_structural_hostility,
                )
            )
            results.append(
                _run_category(
                    "geometry-payloads",
                    lambda: run_geometry_payloads(config),
                )
            )
            results.append(
                _run_category(
                    "append-lifecycle",
                    lambda: run_append_lifecycle(config),
                )
            )
            results.append(
                _run_category("scale-edges", lambda: run_scale_edges(config))
            )
            if config.independent_verifier:
                results.append(
                    _run_category(
                        "independent-verifier",
                        lambda: run_independent_verifier(args.fixture_dir, config),
                    )
                )
            results.append(
                _run_category(
                    "mutation-fuzz",
                    lambda: run_mutation_fuzzer(
                        args.fixture_dir,
                        seed=args.seed ^ 0x46555A5A,
                        per_case_timeout_seconds=config.per_case_timeout_seconds,
                        case_limit=config.fuzz_cases,
                    ),
                )
            )
    except Exception:
        elapsed = time.monotonic() - started
        print(f"CONFORMANCE FAIL: mode={mode} time={elapsed:.3f}s", flush=True)
        return 1

    elapsed = time.monotonic() - started
    total_cases = sum(result.cases for result in results)
    print(
        f"CONFORMANCE PASS: mode={mode} categories={len(results)} "
        f"cases={total_cases} time={elapsed:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
