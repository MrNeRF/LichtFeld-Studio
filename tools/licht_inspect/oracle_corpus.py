#!/usr/bin/env python3
"""Spec-authoritative corruption corpus generator for .licht containers.

This module never imports or calls the reference parser when assigning an
expected outcome. Each family below derives its result from the published head
authority, committed-end, metadata-CRC, and lazy-payload rules. A parser may be
scored against these records; parser agreement cannot change the oracle.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import random
import struct
from typing import Iterator, Sequence

try:
    from .crc32c import crc32c
except ImportError:  # Direct script execution.
    from crc32c import crc32c


SUPERBLOCK_BYTES = 256
HEAD_OFFSETS = (4096, 8192)
HEAD_BYTES = 4096
APPEND_OFFSET = 65536
INDEX_HEADER_BYTES = 64
BLOCK_HEADER_BYTES = 64

EXPECTED_OUTCOMES = frozenset(
    {
        "open_gen_1",
        "open_gen_2",
        "hard_fail",
        "repair_only",
        "unsupported_newer",
    }
)

# Multiple fixed streams make expansion reproducible without depending on the
# process-global RNG or Python hash randomization.
ORACLE_SEEDS = (
    0x4C49434854,
    0x535045435F4F5241,
    0x434C455F5030,
    0xC32C4D4942,
)


@dataclasses.dataclass(frozen=True)
class OracleCase:
    case_id: str
    mutated_file: bytes
    expected_outcome: str
    mutation_description: str
    source_fixture: str
    verify_expected: str | None = None
    recovery_expected: str | None = None
    write_expected: str | None = None

    def __post_init__(self) -> None:
        if self.expected_outcome not in EXPECTED_OUTCOMES:
            raise ValueError(f"invalid oracle outcome {self.expected_outcome}")

    def as_mapping(self) -> dict[str, object]:
        """Return the requested in-memory record, including mutated file bytes."""

        result: dict[str, object] = {
            "mutated_file": self.mutated_file,
            "expected_outcome": self.expected_outcome,
            "mutation_description": self.mutation_description,
        }
        if self.verify_expected is not None:
            result["verify_expected"] = self.verify_expected
        if self.recovery_expected is not None:
            result["recovery_expected"] = self.recovery_expected
        if self.write_expected is not None:
            result["write_expected"] = self.write_expected
        return result


CATALOG = (
    "minimal-valid-single-generation.licht",
    "multi-generation-append.licht",
    "tombstone.licht",
    "orphan-tail.licht",
    "torn-head.licht",
    "duplicate-slot-write.licht",
    "split-brain.licht",
    "out-of-bounds-index-row.licht",
    "overlapping-rows.licht",
    "unsupported-newer-single-head.licht",
    "unsupported-newer-higher-head.licht",
    "write-unsafe.licht",
    "autosave-master.licht",
    "autosave-sidecar-valid.licht.autosave",
    "autosave-sidecar-stale-base.licht.autosave",
)


def _load_manifest(fixture_dir: Path) -> dict[str, object]:
    path = fixture_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run make_fixtures.py before generating the oracle corpus"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog_cases(fixture_dir: Path, manifest: dict[str, object]) -> Iterator[OracleCase]:
    fixture_meta = manifest["fixtures"]
    assert isinstance(fixture_meta, dict)
    for index, name in enumerate(CATALOG):
        info = fixture_meta[name]
        assert isinstance(info, dict)
        recovery = info.get("recovery")
        recovery_expected = recovery.get("expected") if isinstance(recovery, dict) else None
        yield OracleCase(
            case_id=f"catalog-{index:03d}-{info['fixture_class']}",
            mutated_file=(fixture_dir / name).read_bytes(),
            expected_outcome=str(info["expected_outcome"]),
            mutation_description=(
                f"golden catalogue case {info['fixture_class']}; expected result is fixed by "
                "the published fixture class, not inferred from parser output"
            ),
            source_fixture=name,
            verify_expected="pass" if info.get("verify") else None,
            recovery_expected=str(recovery_expected) if recovery_expected is not None else None,
            write_expected=(
                str(info["write_expected"])
                if info.get("write_expected") is not None
                else None
            ),
        )


def _generation(info: dict[str, object], number: int) -> dict[str, object]:
    generations = info["generations"]
    assert isinstance(generations, list)
    for generation in generations:
        assert isinstance(generation, dict)
        if generation["generation"] == number:
            return generation
    raise KeyError(f"generation {number} not found")


def _row(generation: dict[str, object], fourcc: str) -> dict[str, object]:
    rows = generation["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        if row["fourcc"] == fourcc and row["row_kind"] == 0:
            return row
    raise KeyError(f"live {fourcc} row not found")


def _flip(data: bytes, offset: int, bit: int) -> bytes:
    mutated = bytearray(data)
    mutated[offset] ^= 1 << bit
    return bytes(mutated)


def _replace(data: bytes, offset: int, value: int) -> bytes:
    mutated = bytearray(data)
    if mutated[offset] == value:
        value ^= 0xFF
    mutated[offset] = value
    return bytes(mutated)


def _refresh_commit_and_heads(
    data: bytearray,
    commit_offset: int,
    head_slots: Sequence[int],
) -> None:
    """Restore the spec CRC envelope after an intentional commit-field edit."""

    commit_crc = crc32c(data[commit_offset : commit_offset + 252])
    struct.pack_into("<I", data, commit_offset + 252, commit_crc)
    for slot in head_slots:
        head_offset = HEAD_OFFSETS[slot]
        struct.pack_into("<I", data, head_offset + 104, commit_crc)
        head_crc = crc32c(data[head_offset : head_offset + 4092])
        struct.pack_into("<I", data, head_offset + 4092, head_crc)


def _random_cases(
    fixture_dir: Path,
    manifest: dict[str, object],
    count: int,
) -> Iterator[OracleCase]:
    fixture_meta = manifest["fixtures"]
    assert isinstance(fixture_meta, dict)
    multi_name = "multi-generation-append.licht"
    minimal_name = "minimal-valid-single-generation.licht"
    orphan_name = "orphan-tail.licht"
    multi = (fixture_dir / multi_name).read_bytes()
    minimal = (fixture_dir / minimal_name).read_bytes()
    orphan = (fixture_dir / orphan_name).read_bytes()
    multi_info = fixture_meta[multi_name]
    minimal_info = fixture_meta[minimal_name]
    assert isinstance(multi_info, dict) and isinstance(minimal_info, dict)
    gen1 = _generation(multi_info, 1)
    gen2 = _generation(multi_info, 2)
    minimal_gen = _generation(minimal_info, 1)
    gen1_commit = int(gen1["commit_offset"])
    gen2_commit = int(gen2["commit_offset"])
    gen1_index = int(gen1["index_offset"])
    gen2_index = int(gen2["index_offset"])
    gen1_index_bytes = int(gen1["index_bytes"])
    gen2_index_bytes = int(gen2["index_bytes"])
    gen1_end = int(gen1["committed_file_end"])
    gen2_end = int(gen2["committed_file_end"])
    splt = _row(gen2, "SPLT")
    proj = _row(gen1, "PROJ")
    splt_header = int(splt["header_offset"])
    splt_payload = int(splt["payload_offset"])
    splt_size = int(splt["stored_bytes"])
    proj_header = int(proj["header_offset"])
    proj_payload = int(proj["payload_offset"])
    proj_size = int(proj["stored_bytes"])
    minimal_end = int(minimal_gen["committed_file_end"])

    family_count = 23
    for case_number in range(count):
        family = case_number % family_count
        seed = ORACLE_SEEDS[case_number % len(ORACLE_SEEDS)] ^ (
            case_number * 0x9E3779B97F4A7C15
        )
        rng = random.Random(seed)
        case_id = f"random-{case_number:06d}-f{family:02d}"

        if family == 0:
            offset = rng.randrange(0, 252)
            yield OracleCase(
                case_id,
                _flip(multi, offset, rng.randrange(8)),
                "hard_fail",
                f"bit flip at superblock-covered byte 0x{offset:x}; immutable bootstrap CRC/identity failure is a hard failure",
                multi_name,
            )
        elif family == 1:
            relative = rng.randrange(0, 4092)
            offset = HEAD_OFFSETS[1] + relative
            yield OracleCase(
                case_id,
                _flip(multi, offset, rng.randrange(8)),
                "open_gen_1",
                f"bit flip at newer slot-B covered byte 0x{offset:x}; slot B fails its head CRC and valid slot A remains generation 1",
                multi_name,
            )
        elif family == 2:
            relative = rng.randrange(0, 4092)
            offset = HEAD_OFFSETS[0] + relative
            yield OracleCase(
                case_id,
                _flip(multi, offset, rng.randrange(8)),
                "open_gen_2",
                f"bit flip at older slot-A covered byte 0x{offset:x}; slot B and its same-file commit lineage remain valid",
                multi_name,
            )
        elif family == 3:
            mutated = bytearray(multi)
            mutated[HEAD_OFFSETS[0] + 4092] ^= 1 << rng.randrange(8)
            mutated[HEAD_OFFSETS[1] + 4092] ^= 1 << rng.randrange(8)
            yield OracleCase(
                case_id,
                bytes(mutated),
                "repair_only",
                "both head CRC fields are corrupted; no automatic authority remains and EOF scanning is repair-only",
                multi_name,
            )
        elif family == 4:
            relative = rng.randrange(0, 252)
            offset = gen2_commit + relative
            yield OracleCase(
                case_id,
                _replace(multi, offset, rng.randrange(256)),
                "open_gen_1",
                f"byte replacement in generation-2 commit CRC coverage at 0x{offset:x}; newer head is invalid and older head survives",
                multi_name,
            )
        elif family == 5:
            relative = rng.randrange(0, 252)
            offset = gen1_commit + relative
            yield OracleCase(
                case_id,
                _flip(multi, offset, rng.randrange(8)),
                "repair_only",
                f"bit flip in root commit CRC coverage at 0x{offset:x}; slot A fails directly and slot B fails recursive parent linkage",
                multi_name,
            )
        elif family == 6:
            offset = gen2_index + rng.randrange(gen2_index_bytes)
            yield OracleCase(
                case_id,
                _replace(multi, offset, rng.randrange(256)),
                "open_gen_1",
                f"byte replacement in selected generation-2 stored index at 0x{offset:x}; commit index CRC rejects B and A survives",
                multi_name,
            )
        elif family == 7:
            offset = gen1_index + rng.randrange(gen1_index_bytes)
            yield OracleCase(
                case_id,
                _flip(multi, offset, rng.randrange(8)),
                "open_gen_2",
                f"bit flip in obsolete generation-1 index at 0x{offset:x}; complete generation-2 index opens without ancestor-index replay",
                multi_name,
            )
        elif family == 8:
            offset = splt_header + rng.randrange(64)
            yield OracleCase(
                case_id,
                _flip(multi, offset, rng.randrange(8)),
                "open_gen_1",
                f"bit flip in generation-2-only SPLT header at 0x{offset:x}; every-row header validation rejects B while A does not reference it",
                multi_name,
            )
        elif family == 9:
            offset = proj_header + rng.randrange(64)
            yield OracleCase(
                case_id,
                _replace(multi, offset, rng.randrange(256)),
                "repair_only",
                f"byte replacement in carried PROJ header at 0x{offset:x}; both complete indexes reference the corrupt physical span",
                multi_name,
            )
        elif family == 10:
            offset = splt_payload + rng.randrange(splt_size)
            yield OracleCase(
                case_id,
                _flip(multi, offset, rng.randrange(8)),
                "open_gen_2",
                f"bit flip in generation-2 payload at 0x{offset:x}; metadata open remains generation 2 because whole-payload CRC is lazy",
                multi_name,
                verify_expected="hard_fail",
            )
        elif family == 11:
            offset = proj_payload + rng.randrange(proj_size)
            yield OracleCase(
                case_id,
                _replace(multi, offset, rng.randrange(256)),
                "open_gen_2",
                f"byte replacement in carried generation-1 payload at 0x{offset:x}; metadata authority remains generation 2 but full verify must reject the chunk",
                multi_name,
                verify_expected="hard_fail",
            )
        elif family == 12:
            cut = rng.randrange(gen1_end, gen2_end)
            yield OracleCase(
                case_id,
                multi[:cut],
                "open_gen_1",
                f"truncate at 0x{cut:x}, after generation-1 end 0x{gen1_end:x} but before generation-2 end 0x{gen2_end:x}; A is complete and B is out of physical bounds",
                multi_name,
            )
        elif family == 13:
            cut = rng.randrange(APPEND_OFFSET, gen1_end)
            yield OracleCase(
                case_id,
                multi[:cut],
                "repair_only",
                f"truncate at 0x{cut:x} inside the first committed generation; both advertised committed ends exceed physical size",
                multi_name,
            )
        elif family == 14:
            cut = rng.randrange(0, SUPERBLOCK_BYTES)
            yield OracleCase(
                case_id,
                multi[:cut],
                "hard_fail",
                f"truncate immutable superblock at 0x{cut:x}; bootstrap grammar cannot establish project/file identity",
                multi_name,
            )
        elif family == 15:
            offset = HEAD_OFFSETS[1] + rng.randrange(HEAD_BYTES)
            yield OracleCase(
                case_id,
                _flip(minimal, offset, rng.randrange(8)),
                "open_gen_1",
                f"bit flip makes the previously blank inactive slot nonblank and CRC-invalid at 0x{offset:x}; valid slot A remains authoritative with a warning",
                minimal_name,
            )
        elif family == 16:
            suffix_size = 1 + rng.randrange(1, 257)
            suffix = bytes(rng.randrange(256) for _ in range(suffix_size))
            yield OracleCase(
                case_id,
                multi + suffix,
                "open_gen_2",
                f"append {suffix_size} randomized orphan-tail bytes after committed_file_end 0x{gen2_end:x}; physical EOF is not authority",
                multi_name,
            )
        elif family == 17:
            tail_start = minimal_end
            offset = tail_start + rng.randrange(len(orphan) - tail_start)
            yield OracleCase(
                case_id,
                _replace(orphan, offset, rng.randrange(256)),
                "open_gen_1",
                f"byte replacement at orphan-tail offset 0x{offset:x}; readers are pinned to committed_file_end 0x{minimal_end:x}",
                orphan_name,
            )
        elif family == 18:
            # The fixture's one-entry table starts at header+64; entries start
            # after its 64-byte header. Entry-array CRC is open-time metadata.
            entry_offset = splt_header + 64 + BLOCK_HEADER_BYTES
            mutation_offset = entry_offset + rng.randrange(4)
            yield OracleCase(
                case_id,
                _flip(multi, mutation_offset, rng.randrange(8)),
                "open_gen_1",
                f"bit flip in per-block entry bytes at 0x{mutation_offset:x}; entries_crc32c rejects generation 2 before payload use",
                multi_name,
            )
        elif family == 19:
            table_header_offset = splt_header + 64
            offset = table_header_offset + rng.randrange(BLOCK_HEADER_BYTES)
            yield OracleCase(
                case_id,
                _replace(multi, offset, rng.randrange(256)),
                "open_gen_1",
                f"byte replacement in per-block table header at 0x{offset:x}; table metadata CRC/binding rejects generation 2",
                multi_name,
            )
        elif family == 20:
            mutated = bytearray(minimal)
            commit_offset = int(minimal_gen["commit_offset"])
            # Alternate the two reader gates on a sole authority. Commit +186
            # is min_reader_minor; reader bitmap bit 8 is byte +193 bit 0.
            if (case_number // family_count) % 2 == 0:
                struct.pack_into("<H", mutated, commit_offset + 186, 1)
                gate_description = "raise min_reader_version from 1.0 to 1.1"
            else:
                mutated[commit_offset + 193] |= 0x01
                gate_description = "set unknown required-reader capability bit 8"
            _refresh_commit_and_heads(
                mutated,
                commit_offset,
                (0,),
            )
            yield OracleCase(
                case_id,
                bytes(mutated),
                "unsupported_newer",
                f"{gate_description} on the sole structurally valid commit and restore its commit/head CRC envelope; the authority is future-readable, not corrupt or repairable",
                minimal_name,
            )
        elif family == 21:
            mutated = bytearray(multi)
            # Commit +192 is the LE reader bitmap; byte +193 carries bit 8.
            mutated[gen2_commit + 193] |= 0x01
            _refresh_commit_and_heads(mutated, gen2_commit, (1,))
            yield OracleCase(
                case_id,
                bytes(mutated),
                "hard_fail",
                "set unknown required-reader capability bit 8 on the greater-sequence generation-2 authority and restore its commit/head CRC envelope; the reader must hard-fail rather than roll back to supported generation 1",
                multi_name,
            )
        else:
            mutated = bytearray(minimal)
            commit_offset = int(minimal_gen["commit_offset"])
            # Alternate the two independent writer gates. Commit +190 is
            # min_safe_writer_minor; writer bitmap bit 8 is the low bit of
            # byte +209. Neither field affects read compatibility.
            if (case_number // family_count) % 2 == 0:
                struct.pack_into("<H", mutated, commit_offset + 190, 1)
                gate_description = "raise min_safe_writer_version to 1.1"
            else:
                mutated[commit_offset + 209] |= 0x01
                gate_description = "set unknown required-writer capability bit 8"
            _refresh_commit_and_heads(mutated, commit_offset, (0,))
            yield OracleCase(
                case_id,
                bytes(mutated),
                "open_gen_1",
                f"{gate_description} while restoring the commit/head CRC envelope; inspection opens generation 1 but the declared 1.0/bits-0-through-7 writer is unsafe and must refuse",
                minimal_name,
                write_expected="unsafe",
            )


def iter_oracle_cases(
    fixture_dir: Path | None = None,
    *,
    random_cases: int = 10_000,
    include_catalog: bool = True,
) -> Iterator[OracleCase]:
    """Yield authoritative records; 10,000+ cases are supported without storage."""

    if random_cases < 0:
        raise ValueError("random_cases must be non-negative")
    root = fixture_dir or (Path(__file__).resolve().parent / "fixtures")
    manifest = _load_manifest(root)
    if include_catalog:
        yield from _catalog_cases(root, manifest)
    yield from _random_cases(root, manifest, random_cases)


def emit_corpus(
    output_dir: Path,
    *,
    fixture_dir: Path | None = None,
    random_cases: int = 10_000,
    include_catalog: bool = True,
) -> dict[str, object]:
    """Materialize cases and a JSON manifest without consulting a parser."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for case in iter_oracle_cases(
        fixture_dir,
        random_cases=random_cases,
        include_catalog=include_catalog,
    ):
        filename = case.case_id + ".licht"
        (output_dir / filename).write_bytes(case.mutated_file)
        record: dict[str, object] = {
            "case_id": case.case_id,
            "mutated_file": filename,
            "expected_outcome": case.expected_outcome,
            "mutation_description": case.mutation_description,
            "source_fixture": case.source_fixture,
            "bytes": len(case.mutated_file),
            "sha256": hashlib.sha256(case.mutated_file).hexdigest(),
        }
        if case.verify_expected is not None:
            record["verify_expected"] = case.verify_expected
        if case.recovery_expected is not None:
            record["recovery_expected"] = case.recovery_expected
        if case.write_expected is not None:
            record["write_expected"] = case.write_expected
        records.append(record)
    manifest = {
        "schema": 1,
        "authority": "docs/licht_format_spec.md; outcomes assigned without parser execution",
        "fixed_seeds": [f"0x{seed:x}" for seed in ORACLE_SEEDS],
        "case_count": len(records),
        "cases": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--random-cases", type=int, default=10_000)
    parser.add_argument("--no-catalog", action="store_true")
    args = parser.parse_args(argv)
    cases = iter_oracle_cases(
        args.fixture_dir,
        random_cases=args.random_cases,
        include_catalog=not args.no_catalog,
    )
    if args.output_dir is not None:
        manifest = emit_corpus(
            args.output_dir,
            fixture_dir=args.fixture_dir,
            random_cases=args.random_cases,
            include_catalog=not args.no_catalog,
        )
        print(
            f"emitted {manifest['case_count']} parser-independent oracle cases "
            f"to {args.output_dir}"
        )
        return 0
    count = 0
    outcomes: dict[str, int] = {}
    for case in cases:
        count += 1
        outcomes[case.expected_outcome] = outcomes.get(case.expected_outcome, 0) + 1
        print(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "mutated_file": {
                        "bytes": len(case.mutated_file),
                        "sha256": hashlib.sha256(case.mutated_file).hexdigest(),
                    },
                    "expected_outcome": case.expected_outcome,
                    "mutation_description": case.mutation_description,
                    **(
                        {"write_expected": case.write_expected}
                        if case.write_expected is not None
                        else {}
                    ),
                },
                sort_keys=True,
            )
        )
    print(f"oracle cases: {count}; outcomes={dict(sorted(outcomes.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
