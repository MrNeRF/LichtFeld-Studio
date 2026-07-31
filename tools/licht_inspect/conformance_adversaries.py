"""CRC-valid semantic adversaries whose outcomes come directly from the spec."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import struct
import tempfile
import time
import uuid

try:
    from .conformance_common import (
        CategoryResult,
        ConformanceConfig,
        assert_outcome,
        deterministic_uuid,
        timed_result,
    )
    from .crc32c import crc32c
    from .licht_inspect import evaluate_recovery
    from .make_fixtures import (
        APPEND_OFFSET,
        HEAD_OFFSETS,
        HEAD_BYTES,
        INDEX_HEADER_BYTES,
        INDEX_ROW_BYTES,
        ChunkSpec,
        FixtureWriter,
        ROLE_SIDECAR,
    )
except ImportError:  # Direct script execution.
    from conformance_common import (
        CategoryResult,
        ConformanceConfig,
        assert_outcome,
        deterministic_uuid,
        timed_result,
    )
    from crc32c import crc32c
    from licht_inspect import evaluate_recovery
    from make_fixtures import (
        APPEND_OFFSET,
        HEAD_OFFSETS,
        HEAD_BYTES,
        INDEX_HEADER_BYTES,
        INDEX_ROW_BYTES,
        ChunkSpec,
        FixtureWriter,
        ROLE_SIDECAR,
    )


@dataclasses.dataclass(frozen=True)
class SemanticCase:
    name: str
    data: bytes
    expected_outcome: str
    description: str


def _refresh_generation(
    data: bytearray,
    layout,
    head_slots: tuple[int, ...],
) -> None:
    index = bytes(data[layout.index_offset : layout.index_offset + layout.index_bytes])
    index_crc = crc32c(index)
    struct.pack_into("<II", data, layout.commit_offset + 160, index_crc, index_crc)
    commit_crc = crc32c(data[layout.commit_offset : layout.commit_offset + 252])
    struct.pack_into("<I", data, layout.commit_offset + 252, commit_crc)
    for slot in head_slots:
        head_offset = HEAD_OFFSETS[slot]
        struct.pack_into("<I", data, head_offset + 104, commit_crc)
        struct.pack_into(
            "<I",
            data,
            head_offset + HEAD_BYTES - 4,
            crc32c(data[head_offset : head_offset + HEAD_BYTES - 4]),
        )


def _refresh_head_crc(data: bytearray, slot: int) -> None:
    offset = HEAD_OFFSETS[slot]
    struct.pack_into(
        "<I", data, offset + HEAD_BYTES - 4, crc32c(data[offset : offset + HEAD_BYTES - 4])
    )


def _base_writer(namespace: int, *, two_rows: bool = False) -> tuple[FixtureWriter, object]:
    writer = FixtureWriter(
        project_uuid=deterministic_uuid(namespace, 1),
        file_uuid=deterministic_uuid(namespace, 2),
    )
    changes = [
        ChunkSpec(
            b"PROJ",
            deterministic_uuid(namespace, 100),
            b'{"semantic":"first"}',
        )
    ]
    if two_rows:
        changes.append(
            ChunkSpec(
                b"VIEW",
                deterministic_uuid(namespace, 101),
                b'{"semantic":"second"}',
            )
        )
    layout = writer.append_generation(
        commit_uuid=deterministic_uuid(namespace, 200),
        snapshot_uuid=deterministic_uuid(namespace, 300),
        changes=tuple(changes),
        head_slot=0,
        head_sequence=1,
    )
    return writer, layout


def build_semantic_cases() -> tuple[SemanticCase, ...]:
    """Construct errors while retaining all enclosing CRC envelopes."""

    cases: list[SemanticCase] = []

    writer, layout = _base_writer(0xD001, two_rows=True)
    duplicate = bytearray(writer.bytes())
    first_row = layout.index_offset + INDEX_HEADER_BYTES
    second_row = first_row + INDEX_ROW_BYTES
    duplicate[second_row : second_row + 4] = duplicate[first_row : first_row + 4]
    duplicate[second_row + 16 : second_row + 32] = duplicate[first_row + 16 : first_row + 32]
    _refresh_generation(duplicate, layout, (0,))
    cases.append(
        SemanticCase(
            "duplicate-live-key-valid-crcs",
            bytes(duplicate),
            "repair_only",
            "strict key order forbids two live rows with identical fourcc+UUID",
        )
    )

    writer, layout = _base_writer(0xD011, two_rows=True)
    tombstone_live = bytearray(writer.bytes())
    first_row = layout.index_offset + INDEX_HEADER_BYTES
    second_row = first_row + INDEX_ROW_BYTES
    tombstone_live[second_row : second_row + 4] = tombstone_live[
        first_row : first_row + 4
    ]
    tombstone_live[second_row + 16 : second_row + 32] = tombstone_live[
        first_row + 16 : first_row + 32
    ]
    struct.pack_into("<HBBI", tombstone_live, second_row + 4, 0, 1, 0, 0)
    struct.pack_into("<QQQQ", tombstone_live, second_row + 32, 0, 0, 0, 0)
    struct.pack_into("<QII", tombstone_live, second_row + 64, 1, 0, 0)
    _refresh_generation(tombstone_live, layout, (0,))
    cases.append(
        SemanticCase(
            "tombstone-and-live-share-key-valid-crcs",
            bytes(tombstone_live),
            "repair_only",
            "strict key order forbids a tombstone and live row sharing one key",
        )
    )

    writer, layout = _base_writer(0xD002, two_rows=True)
    overlap = bytearray(writer.bytes())
    rows = sorted(layout.rows, key=lambda row: row.key)
    first, second = rows
    expanded_end = second.payload_offset + second.stored_bytes
    expanded_size = expanded_end - first.payload_offset
    expanded_crc = crc32c(overlap[first.payload_offset:expanded_end])
    struct.pack_into("<QQ", overlap, first.header_offset + 32, expanded_size, expanded_size)
    struct.pack_into("<I", overlap, first.header_offset + 56, expanded_crc)
    first_header_crc = crc32c(overlap[first.header_offset : first.header_offset + 60])
    struct.pack_into("<I", overlap, first.header_offset + 60, first_header_crc)
    first_index_row = layout.index_offset + INDEX_HEADER_BYTES
    struct.pack_into("<QQ", overlap, first_index_row + 48, expanded_size, expanded_size)
    struct.pack_into(
        "<II", overlap, first_index_row + 72, expanded_crc, first_header_crc
    )
    _refresh_generation(overlap, layout, (0,))
    cases.append(
        SemanticCase(
            "overlapping-spans-valid-crcs",
            bytes(overlap),
            "repair_only",
            "two individually valid live spans overlap inside generation 1",
        )
    )

    writer, layout = _base_writer(0xD003)
    orphan = bytearray(writer.bytes())
    row = layout.rows[0]
    tail_header = (layout.committed_file_end + 63) & ~63
    tail_payload = tail_header + 64
    tail_bytes = b"orphan-tail-payload"
    if len(orphan) < tail_payload + len(tail_bytes):
        orphan.extend(b"\x00" * (tail_payload + len(tail_bytes) - len(orphan)))
    header = bytearray(orphan[row.header_offset : row.header_offset + 64])
    struct.pack_into("<QQQ", header, 32, len(tail_bytes), len(tail_bytes), 0)
    struct.pack_into("<I", header, 56, crc32c(tail_bytes))
    struct.pack_into("<I", header, 60, crc32c(header[:60]))
    orphan[tail_header : tail_header + 64] = header
    orphan[tail_payload : tail_payload + len(tail_bytes)] = tail_bytes
    index_row = layout.index_offset + INDEX_HEADER_BYTES
    struct.pack_into(
        "<QQQQ",
        orphan,
        index_row + 32,
        tail_header,
        tail_payload,
        len(tail_bytes),
        len(tail_bytes),
    )
    struct.pack_into(
        "<II", orphan, index_row + 72, crc32c(tail_bytes), struct.unpack_from("<I", header, 60)[0]
    )
    _refresh_generation(orphan, layout, (0,))
    cases.append(
        SemanticCase(
            "row-points-into-orphan-tail-valid-crcs",
            bytes(orphan),
            "repair_only",
            "physical EOF contains a valid chunk, but its row lies beyond committed_file_end/index authority",
        )
    )

    writer, layout = _base_writer(0xD004)
    bad_end = bytearray(writer.bytes())
    struct.pack_into("<Q", bad_end, layout.commit_offset + 176, layout.committed_file_end + 64)
    _refresh_generation(bad_end, layout, (0,))
    cases.append(
        SemanticCase(
            "commit-end-equation-valid-crcs",
            bytes(bad_end),
            "repair_only",
            "commit committed_file_end is not commit_offset+256",
        )
    )

    writer, layout = _base_writer(0xD005)
    echo = bytearray(writer.bytes())
    head_offset = HEAD_OFFSETS[0]
    commit_crc = struct.unpack_from("<I", echo, layout.commit_offset + 252)[0]
    struct.pack_into("<I", echo, head_offset + 104, commit_crc ^ 1)
    _refresh_head_crc(echo, 0)
    cases.append(
        SemanticCase(
            "head-commit-crc-echo-mismatch",
            bytes(echo),
            "repair_only",
            "head CRC is valid but its commit CRC echo is false",
        )
    )

    writer, layout = _base_writer(0xD006)
    autosave_master = bytearray(writer.bytes())
    struct.pack_into("<I", autosave_master, layout.commit_offset + 12, 2)
    _refresh_generation(autosave_master, layout, (0,))
    cases.append(
        SemanticCase(
            "autosave-kind-in-master-valid-crcs",
            bytes(autosave_master),
            "repair_only",
            "AUTOSAVE commit kind is forbidden in a master container",
        )
    )

    writer = FixtureWriter(
        project_uuid=deterministic_uuid(0xD007, 1),
        file_uuid=deterministic_uuid(0xD007, 2),
    )
    first_layout = writer.append_generation(
        commit_uuid=deterministic_uuid(0xD007, 200),
        snapshot_uuid=deterministic_uuid(0xD007, 300),
        changes=(
            ChunkSpec(b"PROJ", deterministic_uuid(0xD007, 100), b"generation-one"),
        ),
        head_slot=0,
        head_sequence=1,
    )
    second_layout = writer.append_generation(
        commit_uuid=deterministic_uuid(0xD007, 201),
        snapshot_uuid=deterministic_uuid(0xD007, 301),
        changes=(
            ChunkSpec(b"PROJ", deterministic_uuid(0xD007, 100), b"generation-two"),
        ),
        head_slot=1,
        head_sequence=2,
    )
    regression = bytearray(writer.bytes())
    struct.pack_into("<Q", regression, HEAD_OFFSETS[0] + 16, 2)
    struct.pack_into("<Q", regression, HEAD_OFFSETS[1] + 16, 1)
    _refresh_head_crc(regression, 0)
    _refresh_head_crc(regression, 1)
    cases.append(
        SemanticCase(
            "head-sequence-regression-valid-crcs",
            bytes(regression),
            "hard_fail",
            "the child generation has a lower head_sequence than its direct parent publication",
        )
    )

    writer, layout = _base_writer(0xD008)
    bad_source = bytearray(writer.bytes())
    index_row = layout.index_offset + INDEX_HEADER_BYTES
    struct.pack_into("<Q", bad_source, index_row + 64, layout.generation + 1)
    _refresh_generation(bad_source, layout, (0,))
    cases.append(
        SemanticCase(
            "source-generation-exceeds-commit-valid-crcs",
            bytes(bad_source),
            "repair_only",
            "a live row source_generation exceeds its commit generation",
        )
    )

    # Keep variables live for type checkers: the regression construction
    # intentionally relies on direct parent linkage between these layouts.
    assert second_layout.generation == first_layout.generation + 1
    return tuple(cases)


def _run_ancestor_sidecar_case(temp_dir: Path, config: ConformanceConfig) -> None:
    project_uuid = deterministic_uuid(0xD009, 1)
    master = FixtureWriter(
        project_uuid=project_uuid,
        file_uuid=deterministic_uuid(0xD009, 2),
    )
    first = master.append_generation(
        commit_uuid=deterministic_uuid(0xD009, 200),
        snapshot_uuid=deterministic_uuid(0xD009, 300),
        changes=(
            ChunkSpec(b"PROJ", deterministic_uuid(0xD009, 100), b"ancestor"),
        ),
        head_slot=0,
        head_sequence=1,
    )
    master.append_generation(
        commit_uuid=deterministic_uuid(0xD009, 201),
        snapshot_uuid=deterministic_uuid(0xD009, 301),
        changes=(
            ChunkSpec(b"VIEW", deterministic_uuid(0xD009, 101), b"current"),
        ),
        head_slot=1,
        head_sequence=2,
    )
    sidecar_snapshot = deterministic_uuid(0xD009, 400)
    sidecar = FixtureWriter(
        project_uuid=project_uuid,
        file_uuid=deterministic_uuid(0xD009, 3),
        role=ROLE_SIDECAR,
        base_explicit_commit_uuid=first.commit_uuid,
        autosave_sequence=9,
        sidecar_snapshot_uuid=sidecar_snapshot,
    )
    sidecar.seed_base_refs(first.rows)
    sidecar.append_generation(
        commit_uuid=deterministic_uuid(0xD009, 202),
        snapshot_uuid=sidecar_snapshot,
        head_slot=0,
        head_sequence=1,
    )
    master_path = temp_dir / "ancestor-master.licht"
    sidecar_path = temp_dir / "ancestor-master.licht.autosave"
    master_path.write_bytes(master.bytes())
    sidecar_path.write_bytes(sidecar.bytes())
    decision = evaluate_recovery(master_path, [sidecar_path])
    if decision.outcome != "none":
        raise AssertionError(
            f"ancestor sidecar: spec requires no recovery offer, got {decision}"
        )
    statuses = [candidate.status for candidate in decision.candidates]
    if statuses != ["stale_delete"]:
        raise AssertionError(
            f"ancestor sidecar: expected stale_delete, got {statuses}"
        )


def run_semantic_adversaries(config: ConformanceConfig) -> CategoryResult:
    started = time.monotonic()
    cases = build_semantic_cases()
    with tempfile.TemporaryDirectory(prefix="licht-conformance-semantic-") as temp:
        temp_dir = Path(temp)
        for case in cases:
            path = temp_dir / f"{case.name}.licht"
            path.write_bytes(case.data)
            assert_outcome(
                path,
                case.expected_outcome,
                timeout_seconds=config.per_case_timeout_seconds,
            )
        _run_ancestor_sidecar_case(temp_dir, config)
    return timed_result(
        "semantic-adversaries",
        started,
        len(cases) + 1,
        "9 CRC-valid open adversaries + ancestor-bound sidecar stale-delete",
    )
