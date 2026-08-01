"""Spec-oracle truncation and torn-head publication sweeps."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time

try:
    from .conformance_common import (
        CategoryResult,
        ConformanceConfig,
        assert_outcome,
        timed_result,
    )
except ImportError:  # Direct script execution.
    from conformance_common import (
        CategoryResult,
        ConformanceConfig,
        assert_outcome,
        timed_result,
    )


APPEND_OFFSET = 65_536
HEAD_BYTES = 4_096
HEAD_OFFSETS = (4_096, 8_192)


def _fallback_outcome(info: dict[str, object]) -> str:
    generations = info["generations"]
    assert isinstance(generations, list)
    if len(generations) >= 2:
        # The committed fixtures retain a valid slot-A generation 1. Removing
        # any byte of the final generation invalidates its slot by the
        # committed-end bound, leaving that exact prior authority.
        return "open_gen_1"
    if info["fixture_class"] == "duplicate_slot_write":
        # Both slots reference the same sole commit; append truncation kills
        # both, unlike a torn write of only the duplicate slot.
        return "repair_only"
    return "repair_only"


def _head_fallback_outcome(info: dict[str, object]) -> str:
    generations = info["generations"]
    assert isinstance(generations, list)
    if len(generations) >= 2:
        return "open_gen_1"
    if info["fixture_class"] == "duplicate_slot_write":
        return "open_gen_1"
    return "repair_only"


def _quick_append_boundaries(
    info: dict[str, object], start: int, end: int
) -> list[int]:
    points = {start, min(start + 1, end), max(start, end - 1), end}
    generations = info["generations"]
    assert isinstance(generations, list)
    final = generations[-1]
    assert isinstance(final, dict)
    for field in ("index_offset", "commit_offset", "committed_file_end"):
        value = int(final[field])
        for delta in (-1, 0, 1, 63, 64, 251, 252, 255):
            points.add(value + delta)
    rows = final["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        for field in ("header_offset", "payload_offset"):
            value = int(row[field])
            points.update((value - 1, value, value + 1, value + 63, value + 64))
        payload_end = int(row["payload_offset"]) + int(row["stored_bytes"])
        points.update((payload_end - 1, payload_end))
    points.update(range(start, end + 1, 128))
    return sorted(point for point in points if start <= point <= end)


def _head_boundaries(exhaustive: bool) -> list[int]:
    if exhaustive:
        return list(range(HEAD_BYTES + 1))
    points = {
        0,
        1,
        7,
        8,
        15,
        16,
        23,
        24,
        31,
        32,
        63,
        64,
        79,
        80,
        87,
        88,
        95,
        96,
        103,
        104,
        107,
        108,
        111,
        112,
        4_091,
        4_092,
        4_095,
        4_096,
    }
    points.update(range(0, HEAD_BYTES + 1, 256))
    return sorted(points)


def run_truncation_sweep(
    fixture_dir: Path,
    config: ConformanceConfig,
) -> CategoryResult:
    """Exercise EOF cuts and every full-mode head publication boundary."""

    started = time.monotonic()
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    fixtures = manifest["fixtures"]
    assert isinstance(fixtures, dict)
    append_cases = 0
    payload_stride_cases = 0
    head_cases = 0

    with tempfile.TemporaryDirectory(prefix="licht-conformance-trunc-") as temp:
        case_path = Path(temp) / "case.licht"
        for fixture_name, raw_info in sorted(fixtures.items()):
            assert isinstance(raw_info, dict)
            source = (fixture_dir / fixture_name).read_bytes()
            generations = raw_info["generations"]
            assert isinstance(generations, list) and generations
            final = generations[-1]
            assert isinstance(final, dict)
            final_end = int(final["committed_file_end"])
            final_start = (
                int(generations[-2]["committed_file_end"])
                if len(generations) >= 2
                else APPEND_OFFSET
            )
            original_outcome = str(raw_info["expected_outcome"])
            fallback = _fallback_outcome(raw_info)

            if config.exhaustive_truncation:
                append_boundaries = range(final_end, final_start - 1, -1)
                case_path.write_bytes(source[:final_end])
                for cut in append_boundaries:
                    with case_path.open("r+b") as stream:
                        stream.truncate(cut)
                    expected = original_outcome if cut == final_end else fallback
                    warning = (
                        raw_info["fixture_class"] == "torn_head"
                        if cut == final_end and expected.startswith("open_gen_")
                        else cut < final_end and expected.startswith("open_gen_")
                    )
                    assert_outcome(
                        case_path,
                        expected,
                        timeout_seconds=config.per_case_timeout_seconds,
                        recovery_warning=warning if expected.startswith("open_gen_") else None,
                    )
                    append_cases += 1
            else:
                for cut in _quick_append_boundaries(raw_info, final_start, final_end):
                    case_path.write_bytes(source)
                    with case_path.open("r+b") as stream:
                        stream.truncate(cut)
                    expected = original_outcome if cut == final_end else fallback
                    warning = (
                        raw_info["fixture_class"] == "torn_head"
                        if cut == final_end and expected.startswith("open_gen_")
                        else cut < final_end and expected.startswith("open_gen_")
                    )
                    assert_outcome(
                        case_path,
                        expected,
                        timeout_seconds=config.per_case_timeout_seconds,
                        recovery_warning=warning if expected.startswith("open_gen_") else None,
                    )
                    append_cases += 1

            # Earlier payload cuts destroy every published commit. These are
            # deliberately coarser because the final-generation sweep above is
            # byte-exhaustive in full mode.
            stride = 512 if config.exhaustive_truncation else 4_096
            payload_cuts = set(range(APPEND_OFFSET, final_start, stride))
            if final_start > APPEND_OFFSET:
                payload_cuts.add(final_start - 1)
            for cut in sorted(payload_cuts):
                case_path.write_bytes(source)
                with case_path.open("r+b") as stream:
                    stream.truncate(cut)
                assert_outcome(
                    case_path,
                    "repair_only",
                    timeout_seconds=config.per_case_timeout_seconds,
                )
                payload_stride_cases += 1

            # Model a torn inactive-slot publication without truncating the
            # append region: bytes before the boundary reached media, the rest
            # remained zero. Boundary 4096 is the original complete slot.
            target_slot = (
                1
                if raw_info["fixture_class"] == "duplicate_slot_write"
                else int(final["head_slot"])
            )
            head_offset = HEAD_OFFSETS[target_slot]
            original_head = source[head_offset : head_offset + HEAD_BYTES]
            for boundary in _head_boundaries(config.exhaustive_truncation):
                case_path.write_bytes(source)
                torn = original_head[:boundary] + b"\x00" * (HEAD_BYTES - boundary)
                with case_path.open("r+b") as stream:
                    stream.seek(head_offset)
                    stream.write(torn)
                    stream.flush()
                if boundary == HEAD_BYTES:
                    expected = original_outcome
                    warning = (
                        raw_info["fixture_class"] == "torn_head"
                        if expected.startswith("open_gen_")
                        else None
                    )
                else:
                    expected = _head_fallback_outcome(raw_info)
                    warning = (
                        boundary > 0 if expected.startswith("open_gen_") else None
                    )
                assert_outcome(
                    case_path,
                    expected,
                    timeout_seconds=config.per_case_timeout_seconds,
                    recovery_warning=warning,
                )
                head_cases += 1

    total = append_cases + payload_stride_cases + head_cases
    exact = "byte-exhaustive" if config.exhaustive_truncation else "boundary-sampled"
    return timed_result(
        "truncation",
        started,
        total,
        f"append={append_cases}, payload_stride={payload_stride_cases}, "
        f"head_publish={head_cases} ({exact})",
    )
