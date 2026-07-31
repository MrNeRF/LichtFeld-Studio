"""Time-bounded mutation fuzzer with classified-outcome and CRC invariants."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import random
import struct
import tempfile
import time

try:
    from .conformance_common import (
        CategoryResult,
        classified_outcome,
        case_deadline,
        timed_result,
    )
    from .licht_inspect import FormatError, open_container, verify_container
except ImportError:  # Direct script execution.
    from conformance_common import (
        CategoryResult,
        classified_outcome,
        case_deadline,
        timed_result,
    )
    from licht_inspect import FormatError, open_container, verify_container


@dataclasses.dataclass(frozen=True)
class Mutation:
    data: bytes
    description: str
    fixed_coordinates: bool
    single_byte: bool


def _load_chapter_dictionary() -> tuple[bytes, ...]:
    path = Path(__file__).resolve().parent / "fuzz_corpus/dictionary.json"
    entries = json.loads(path.read_text(encoding="utf-8"))["tokens"]
    tokens: list[bytes] = []
    for entry in entries:
        if "ascii" in entry:
            tokens.append(entry["ascii"].encode("utf-8"))
        else:
            tokens.append(bytes.fromhex(entry["hex"]))
    if not tokens or any(not token for token in tokens):
        raise AssertionError("chapter fuzz dictionary contains an empty token")
    return tuple(tokens)


def _load_regression_cases(fixture_dir: Path) -> tuple[tuple[str, Mutation], ...]:
    path = Path(__file__).resolve().parent / "fuzz_corpus/cases.json"
    recipes = json.loads(path.read_text(encoding="utf-8"))["cases"]
    result: list[tuple[str, Mutation]] = []
    for recipe in recipes:
        source_name = str(recipe["source"])
        source = (fixture_dir / source_name).read_bytes()
        operation = recipe["operation"]
        offset = int(recipe.get("offset", 0))
        if operation == "replace":
            replacement = bytes.fromhex(recipe["hex"])
            if offset < 0 or offset + len(replacement) > len(source):
                raise AssertionError(f"invalid fuzz regression replacement: {recipe}")
            data = source[:offset] + replacement + source[offset + len(replacement) :]
            fixed = True
        elif operation == "truncate":
            if offset < 0 or offset >= len(source):
                raise AssertionError(f"invalid fuzz regression truncation: {recipe}")
            data = source[:offset]
            fixed = False
        elif operation == "insert_ascii":
            if offset < 0 or offset > len(source):
                raise AssertionError(f"invalid fuzz regression insertion: {recipe}")
            token = str(recipe["ascii"]).encode("utf-8")
            data = source[:offset] + token + source[offset:]
            fixed = False
        else:
            raise AssertionError(f"unknown fuzz regression operation: {operation}")
        result.append(
            (
                source_name,
                Mutation(
                    data=data,
                    description=str(recipe["id"]),
                    fixed_coordinates=fixed,
                    single_byte=False,
                ),
            )
        )
    return tuple(result)


def _fixture_boundaries(info: dict[str, object], size: int) -> tuple[int, ...]:
    points = {
        0,
        8,
        12,
        20,
        24,
        40,
        56,
        96,
        112,
        128,
        252,
        256,
        4_096,
        8_192,
        12_288,
        65_536,
        max(0, size - 1),
    }
    for head_offset in (4_096, 8_192):
        for relative in (0, 8, 16, 24, 64, 80, 96, 104, 108, 112, 4_092, 4_096):
            points.add(head_offset + relative)
    generations = info["generations"]
    assert isinstance(generations, list)
    for generation in generations:
        assert isinstance(generation, dict)
        index_offset = int(generation["index_offset"])
        commit_offset = int(generation["commit_offset"])
        for relative in (0, 8, 16, 24, 32, 48, 52, int(generation["index_bytes"])):
            points.add(index_offset + relative)
        for relative in (
            0,
            8,
            12,
            48,
            64,
            72,
            88,
            96,
            112,
            136,
            144,
            160,
            168,
            176,
            184,
            188,
            192,
            208,
            224,
            252,
            256,
        ):
            points.add(commit_offset + relative)
        rows = generation["rows"]
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, dict)
            header = int(row["header_offset"])
            payload = int(row["payload_offset"])
            stored = int(row["stored_bytes"])
            points.update((header, header + 4, header + 24, header + 60, header + 64))
            points.update((payload, payload + max(0, stored - 1), payload + stored))
    return tuple(sorted(point for point in points if 0 <= point <= size))


def _weighted_offset(rng: random.Random, boundaries: tuple[int, ...], size: int) -> int:
    if size <= 0:
        return 0
    if boundaries and rng.random() < 0.78:
        base = rng.choice(boundaries)
        return min(size - 1, max(0, base + rng.randint(-3, 3)))
    return rng.randrange(size)


def _mutate(
    rng: random.Random,
    source: bytes,
    boundaries: tuple[int, ...],
    dictionary: tuple[bytes, ...],
) -> Mutation:
    if not source:
        return Mutation(b"\x00", "insert into empty source", False, False)
    operation = rng.randrange(8)
    offset = _weighted_offset(rng, boundaries, len(source))
    if operation == 0:
        data = bytearray(source)
        bit = rng.randrange(8)
        data[offset] ^= 1 << bit
        return Mutation(bytes(data), f"bit flip at 0x{offset:x} bit {bit}", True, True)
    if operation == 1:
        data = bytearray(source)
        original = data[offset]
        replacement = rng.randrange(256)
        if replacement == original:
            replacement ^= 0xFF
        data[offset] = replacement
        return Mutation(
            bytes(data),
            f"byte replacement at 0x{offset:x}: 0x{original:02x}->0x{replacement:02x}",
            True,
            True,
        )
    if operation == 2:
        width = min(len(source) - offset, rng.randint(2, 32))
        data = bytearray(source)
        for index in range(offset, offset + width):
            data[index] ^= rng.randrange(1, 256)
        return Mutation(
            bytes(data),
            f"multi-byte xor at 0x{offset:x} width {width}",
            True,
            False,
        )
    if operation == 3:
        cut = min(len(source) - 1, max(0, offset))
        return Mutation(source[:cut], f"truncate at 0x{cut:x}", False, False)
    if operation == 4:
        width = min(len(source) - offset, rng.randint(1, 64))
        return Mutation(
            source[:offset] + source[offset + width :],
            f"delete splice at 0x{offset:x} width {width}",
            False,
            False,
        )
    if operation == 5:
        inserted = bytes(rng.randrange(256) for _ in range(rng.randint(1, 64)))
        return Mutation(
            source[:offset] + inserted + source[offset:],
            f"insert {len(inserted)} bytes at 0x{offset:x}",
            False,
            False,
        )
    if operation == 6:
        width = min(len(source) - offset, rng.randint(1, 128))
        destination = _weighted_offset(rng, boundaries, len(source))
        duplicate = source[offset : offset + width]
        return Mutation(
            source[:destination] + duplicate + source[destination:],
            f"duplicate [0x{offset:x},0x{offset + width:x}) at 0x{destination:x}",
            False,
            False,
        )
    token = rng.choice(dictionary)
    destination = _weighted_offset(rng, boundaries, len(source))
    return Mutation(
        source[:destination] + token + source[destination:],
        f"dictionary token {token[:32]!r} at 0x{destination:x}",
        False,
        False,
    )


def _selected_protected_ranges(container, data: bytes) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = [
        (0, 256),
        (container.selected_head.offset, container.selected_head.offset + 4_096),
        (
            container.commit.index_offset,
            container.commit.index_offset + container.commit.index_stored_bytes,
        ),
    ]
    commit_offset = container.commit.offset
    seen: set[int] = set()
    while commit_offset and commit_offset not in seen and commit_offset + 256 <= len(data):
        seen.add(commit_offset)
        ranges.append((commit_offset, commit_offset + 256))
        generation = struct.unpack_from("<Q", data, commit_offset + 64)[0]
        if generation <= 1:
            break
        commit_offset = struct.unpack_from("<Q", data, commit_offset + 88)[0]
    for row in container.index.rows:
        if row.row_kind != 0:
            continue
        ranges.append((row.header_offset, row.header_offset + 64))
        if row.block_table is not None:
            ranges.append((row.block_table.offset, row.block_table.end))
        ranges.append((row.payload_offset, row.payload_offset + row.stored_bytes))
    return tuple(ranges)


def _selected_bytes_changed(
    source: bytes,
    mutated: bytes,
    ranges: tuple[tuple[int, int], ...],
) -> bool:
    if len(source) != len(mutated):
        return False
    for start, end in ranges:
        if end <= len(source) and source[start:end] != mutated[start:end]:
            return True
    return False


def run_mutation_fuzzer(
    fixture_dir: Path,
    *,
    seed: int,
    per_case_timeout_seconds: float,
    case_limit: int | None = None,
    fuzz_minutes: float | None = None,
) -> CategoryResult:
    if (case_limit is None) == (fuzz_minutes is None):
        raise ValueError("set exactly one of case_limit or fuzz_minutes")
    started = time.monotonic()
    stop_at = (
        started + fuzz_minutes * 60.0 if fuzz_minutes is not None else None
    )
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    fixture_meta = manifest["fixtures"]
    assert isinstance(fixture_meta, dict)
    sources = [
        (name, (fixture_dir / name).read_bytes(), info)
        for name, info in sorted(fixture_meta.items())
    ]
    rng = random.Random(seed)
    rng.shuffle(sources)
    dictionary = _load_chapter_dictionary()
    regression_cases = _load_regression_cases(fixture_dir)
    boundaries = {
        name: _fixture_boundaries(info, len(data))
        for name, data, info in sources
    }
    outcomes: dict[str, int] = {}
    cases = 0
    successful_opens = 0
    strong_crc_detections = 0
    collision_caveat_candidates = 0
    next_progress = started + 30.0

    with tempfile.TemporaryDirectory(prefix="licht-conformance-fuzz-") as temp:
        case_path = Path(temp) / "fuzz.licht"
        while True:
            now = time.monotonic()
            if case_limit is not None and cases >= case_limit:
                break
            if stop_at is not None and now >= stop_at:
                break
            if cases < len(regression_cases):
                name, mutation = regression_cases[cases]
                source = (fixture_dir / name).read_bytes()
                source_info = fixture_meta[name]
            else:
                name, source, source_info = sources[
                    (cases - len(regression_cases)) % len(sources)
                ]
                mutation = _mutate(
                    rng, source, boundaries[name], dictionary
                )
            case_path.write_bytes(mutation.data)
            try:
                outcome, detail = classified_outcome(
                    case_path, per_case_timeout_seconds
                )
            except Exception as error:
                raise AssertionError(
                    f"fuzz case {cases} source={name} mutation={mutation.description}: "
                    f"unclassified exception {type(error).__name__}: {error}"
                ) from error
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if outcome.startswith("open_gen_"):
                successful_opens += 1
                with case_deadline(per_case_timeout_seconds):
                    container = open_container(case_path)
                source_expected = str(source_info["expected_outcome"])
                if mutation.fixed_coordinates and outcome == source_expected:
                    protected = _selected_protected_ranges(container, mutation.data)
                    if _selected_bytes_changed(source, mutation.data, protected):
                        try:
                            with case_deadline(per_case_timeout_seconds):
                                verify_container(container)
                        except FormatError:
                            strong_crc_detections += 1
                        else:
                            if mutation.single_byte:
                                raise AssertionError(
                                    f"fuzz case {cases} source={name}: selected committed "
                                    f"bytes changed by {mutation.description}, yet open+verify "
                                    "both succeeded"
                                )
                            # CRC32c is not collision-proof for arbitrary
                            # multi-bit edits. Record rather than assert the
                            # mathematically possible 32-bit collision.
                            collision_caveat_candidates += 1
            cases += 1
            if stop_at is not None and time.monotonic() >= next_progress:
                elapsed = time.monotonic() - started
                print(
                    f"FUZZ progress: {cases} cases in {elapsed:.1f}s; "
                    f"outcomes={dict(sorted(outcomes.items()))}",
                    flush=True,
                )
                next_progress += 30.0

    mode_detail = (
        f"timebox={fuzz_minutes:g}m" if fuzz_minutes is not None else f"limit={case_limit}"
    )
    elapsed = max(time.monotonic() - started, 1e-9)
    return timed_result(
        "mutation-fuzz",
        started,
        cases,
        f"{mode_detail}, sources={len(sources)}, opens={successful_opens}, "
        f"dictionary_tokens={len(dictionary)}, regression_cases={len(regression_cases)}, "
        f"cases_per_second={cases / elapsed:.1f}, "
        f"unique_terminal_classes={len(outcomes)}, unclassified=0, "
        f"selected_crc_detections={strong_crc_detections}, "
        f"multi_bit_collision_candidates={collision_caveat_candidates}, "
        f"outcomes={dict(sorted(outcomes.items()))}",
    )
