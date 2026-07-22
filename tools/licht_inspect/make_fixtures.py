#!/usr/bin/env python3
"""Generate deterministic golden .licht fixtures from the published grammar.

WARNING: this is a deliberately small, non-production writer. It exists only
to create test fixtures and adversarial metadata. It does not implement locks,
flush ordering, atomic replacement, disk preflight, clean proofs, or production
zstd policy. Application code MUST NOT use it to save projects.

The packing declarations here are intentionally separate from the read-only
parser. Both are derived from docs/licht_format_spec.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import struct
import uuid
from typing import Iterable, Sequence

try:
    from .crc32c import crc32c
except ImportError:  # Direct script execution.
    from crc32c import crc32c


SUPERBLOCK_BYTES = 256
HEAD_BYTES = 4096
HEAD_OFFSETS = (4096, 8192)
APPEND_OFFSET = 65536
COMMIT_BYTES = 256
CHUNK_HEADER_BYTES = 64
INDEX_HEADER_BYTES = 64
INDEX_ROW_BYTES = 96
BLOCK_HEADER_BYTES = 64
BLOCK_SIZE = 4 * 1024 * 1024

SUPER_MAGIC = b"\x89LFS\r\n\x1a\n"
HEAD_MAGIC = b"LFSHEAD\x00"
COMMIT_MAGIC = b"LFSCOMIT"
INDEX_MAGIC = b"LFSINDEX"
BLOCK_MAGIC = b"LFSBCRC\x00"

ROLE_MASTER = 0
ROLE_SIDECAR = 1
KIND_EXPLICIT = 1
KIND_AUTOSAVE = 2

ROW_LIVE = 0
ROW_TOMBSTONE = 1
ROW_BASE_REF = 2

CHUNK_TENSOR = 1 << 0
CHUNK_BLOCK_CRCS = 1 << 1

CAP_INDEX_ZSTD = 1 << 0
CAP_CHUNK_ZSTD = 1 << 1
CAP_BLOCK_CRC = 1 << 2
CAP_TOMBSTONES = 1 << 3
CAP_SIDECAR = 1 << 4
CAP_OPAQUE = 1 << 5
CAP_RETAINED_JSON = 1 << 6
CAP_CLEAN_PROOF = 1 << 7
CAP_FUTURE_BIT_8 = 1 << 8

FIXED_CREATION_TIME_NS = 1_735_689_600_000_000_000  # 2025-01-01T00:00:00Z
FIXED_COMMIT_TIME_NS = 1_735_689_601_000_000_000


def _fixed_uuid(tag: int) -> uuid.UUID:
    # RFC-4122-shaped constants: version nibble 4, variant bits 10.
    return uuid.UUID(f"{tag:08x}-0000-4000-8000-{tag:012x}")


# Every fixture identity and timestamp comes from this table; no wallclock or
# random UUID source is consulted.
FIXED_UUIDS = {
    "project": _fixed_uuid(1),
    "project_other": _fixed_uuid(2),
    "file_minimal": _fixed_uuid(10),
    "file_multi": _fixed_uuid(11),
    "file_tombstone": _fixed_uuid(12),
    "file_duplicate": _fixed_uuid(13),
    "file_split": _fixed_uuid(14),
    "file_oob": _fixed_uuid(15),
    "file_overlap": _fixed_uuid(16),
    "file_autosave_master": _fixed_uuid(17),
    "file_autosave_valid": _fixed_uuid(18),
    "file_autosave_stale": _fixed_uuid(19),
    "file_unsupported_single": _fixed_uuid(20),
    "file_unsupported_higher": _fixed_uuid(21),
    "file_write_unsafe": _fixed_uuid(22),
    "commit_minimal": _fixed_uuid(100),
    "commit_multi_1": _fixed_uuid(101),
    "commit_multi_2": _fixed_uuid(102),
    "commit_tombstone_1": _fixed_uuid(103),
    "commit_tombstone_2": _fixed_uuid(104),
    "commit_duplicate": _fixed_uuid(105),
    "commit_split_a": _fixed_uuid(106),
    "commit_split_b": _fixed_uuid(107),
    "commit_oob": _fixed_uuid(108),
    "commit_overlap": _fixed_uuid(109),
    "commit_autosave_master": _fixed_uuid(110),
    "commit_autosave_valid": _fixed_uuid(111),
    "commit_autosave_stale": _fixed_uuid(112),
    "commit_stale_base": _fixed_uuid(113),
    "commit_unsupported_single": _fixed_uuid(114),
    "commit_unsupported_higher_1": _fixed_uuid(115),
    "commit_unsupported_higher_2": _fixed_uuid(116),
    "commit_write_unsafe": _fixed_uuid(117),
    "snapshot_minimal": _fixed_uuid(200),
    "snapshot_multi_1": _fixed_uuid(201),
    "snapshot_multi_2": _fixed_uuid(202),
    "snapshot_tombstone_1": _fixed_uuid(203),
    "snapshot_tombstone_2": _fixed_uuid(204),
    "snapshot_duplicate": _fixed_uuid(205),
    "snapshot_split_a": _fixed_uuid(206),
    "snapshot_split_b": _fixed_uuid(207),
    "snapshot_oob": _fixed_uuid(208),
    "snapshot_overlap": _fixed_uuid(209),
    "snapshot_autosave_master": _fixed_uuid(210),
    "snapshot_autosave_valid": _fixed_uuid(211),
    "snapshot_autosave_stale": _fixed_uuid(212),
    "snapshot_unsupported_single": _fixed_uuid(213),
    "snapshot_unsupported_higher_1": _fixed_uuid(214),
    "snapshot_unsupported_higher_2": _fixed_uuid(215),
    "snapshot_write_unsafe": _fixed_uuid(216),
    "instance_proj": _fixed_uuid(300),
    "instance_view": _fixed_uuid(301),
    "instance_splt": _fixed_uuid(302),
    "instance_ckpt": _fixed_uuid(303),
}


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _ensure_size(data: bytearray, size: int) -> None:
    if len(data) < size:
        data.extend(b"\x00" * (size - len(data)))


@dataclasses.dataclass(frozen=True)
class ChunkSpec:
    fourcc: bytes
    instance_uuid: uuid.UUID
    payload: bytes
    chunk_version: int = 1
    tensor: bool = False
    block_crcs: bool = False


@dataclasses.dataclass(frozen=True)
class RowRecord:
    fourcc: bytes
    instance_uuid: uuid.UUID
    chunk_version: int
    row_kind: int
    compression: int
    flags: int
    header_offset: int
    payload_offset: int
    stored_bytes: int
    uncompressed_bytes: int
    source_generation: int
    payload_crc32c: int
    header_crc32c: int

    @property
    def key(self) -> bytes:
        return self.fourcc + self.instance_uuid.bytes

    @property
    def payload_end(self) -> int:
        return self.payload_offset + self.stored_bytes


@dataclasses.dataclass(frozen=True)
class GenerationLayout:
    generation: int
    head_sequence: int
    head_slot: int
    commit_uuid: uuid.UUID
    snapshot_uuid: uuid.UUID
    index_offset: int
    index_bytes: int
    commit_offset: int
    committed_file_end: int
    min_reader_version: tuple[int, int]
    min_safe_writer_version: tuple[int, int]
    reader_capabilities: int
    writer_capabilities: int
    rows: tuple[RowRecord, ...]


class FixtureWriter:
    """Grammar packer for fixtures only; never a production persistence API."""

    def __init__(
        self,
        *,
        project_uuid: uuid.UUID,
        file_uuid: uuid.UUID,
        role: int = ROLE_MASTER,
        base_explicit_commit_uuid: uuid.UUID = uuid.UUID(int=0),
        autosave_sequence: int = 0,
        sidecar_snapshot_uuid: uuid.UUID = uuid.UUID(int=0),
    ) -> None:
        self.project_uuid = project_uuid
        self.file_uuid = file_uuid
        self.role = role
        self.base_explicit_commit_uuid = base_explicit_commit_uuid
        self.autosave_sequence = autosave_sequence
        self.sidecar_snapshot_uuid = sidecar_snapshot_uuid
        self.data = bytearray(APPEND_OFFSET)
        self.current_rows: dict[bytes, RowRecord] = {}
        self.generations: list[GenerationLayout] = []
        self._write_superblock()

    def _write_superblock(self) -> None:
        sb = bytearray(SUPERBLOCK_BYTES)
        sb[0:8] = SUPER_MAGIC
        struct.pack_into("<HHIII", sb, 8, 1, 0, 0x01020304, SUPERBLOCK_BYTES, self.role)
        sb[24:40] = self.project_uuid.bytes
        sb[40:56] = self.file_uuid.bytes
        struct.pack_into("<QQIIQQ", sb, 56, 4096, 8192, HEAD_BYTES, 2, APPEND_OFFSET, FIXED_CREATION_TIME_NS)
        sb[96:112] = self.base_explicit_commit_uuid.bytes
        struct.pack_into("<Q", sb, 112, self.autosave_sequence)
        sb[128:144] = self.sidecar_snapshot_uuid.bytes
        struct.pack_into("<I", sb, 252, crc32c(sb[:252]))
        self.data[0:SUPERBLOCK_BYTES] = sb

    def seed_base_refs(self, base_rows: Iterable[RowRecord]) -> None:
        if self.role != ROLE_SIDECAR or self.generations:
            raise ValueError("base refs may seed only an empty sidecar writer")
        for base in base_rows:
            if base.row_kind != ROW_LIVE:
                continue
            row = RowRecord(
                fourcc=base.fourcc,
                instance_uuid=base.instance_uuid,
                chunk_version=base.chunk_version,
                row_kind=ROW_BASE_REF,
                compression=base.compression,
                flags=base.flags,
                header_offset=0,
                payload_offset=0,
                stored_bytes=base.stored_bytes,
                uncompressed_bytes=base.uncompressed_bytes,
                source_generation=base.source_generation,
                payload_crc32c=base.payload_crc32c,
                header_crc32c=base.header_crc32c,
            )
            self.current_rows[row.key] = row

    def reset_logical_rows(self) -> None:
        self.current_rows = {}

    def _append_chunk(self, spec: ChunkSpec, generation: int) -> RowRecord:
        if len(spec.fourcc) != 4:
            raise ValueError("fourcc must be four bytes")
        header_offset = _align(len(self.data), 64)
        _ensure_size(self.data, header_offset + CHUNK_HEADER_BYTES)
        flags = (CHUNK_TENSOR if spec.tensor else 0) | (CHUNK_BLOCK_CRCS if spec.block_crcs else 0)
        payload_crc = crc32c(spec.payload)
        table_offset = 0
        if spec.block_crcs:
            table_offset = header_offset + CHUNK_HEADER_BYTES
            block_entries = [
                crc32c(spec.payload[start : start + BLOCK_SIZE])
                for start in range(0, len(spec.payload), BLOCK_SIZE)
            ]
            if not block_entries:
                raise ValueError("block CRC table requires a nonempty payload")
            table_end = table_offset + BLOCK_HEADER_BYTES + 4 * len(block_entries)
        else:
            block_entries = []
            table_end = header_offset + CHUNK_HEADER_BYTES
        alignment = 4096 if spec.tensor else 64
        payload_offset = _align(table_end, alignment)
        _ensure_size(self.data, payload_offset + len(spec.payload))
        self.data[payload_offset : payload_offset + len(spec.payload)] = spec.payload

        if spec.block_crcs:
            entries = struct.pack(f"<{len(block_entries)}I", *block_entries)
            table = bytearray(BLOCK_HEADER_BYTES)
            table[0:8] = BLOCK_MAGIC
            struct.pack_into("<HHHHIIQQQ", table, 8, 1, BLOCK_HEADER_BYTES, 4, 1, BLOCK_SIZE, 0, payload_offset, len(spec.payload), len(block_entries))
            struct.pack_into("<I", table, 48, crc32c(entries))
            struct.pack_into("<I", table, 60, crc32c(table[:60]))
            self.data[table_offset : table_offset + BLOCK_HEADER_BYTES] = table
            self.data[table_offset + BLOCK_HEADER_BYTES : table_end] = entries

        header = bytearray(CHUNK_HEADER_BYTES)
        header[0:4] = spec.fourcc
        struct.pack_into("<HH", header, 4, spec.chunk_version, CHUNK_HEADER_BYTES)
        header[8:24] = spec.instance_uuid.bytes
        struct.pack_into(
            "<IHHQQQI",
            header,
            24,
            flags,
            0,
            1 if spec.block_crcs else 0,
            len(spec.payload),
            len(spec.payload),
            table_offset,
            payload_crc,
        )
        header_crc = crc32c(header[:60])
        struct.pack_into("<I", header, 60, header_crc)
        self.data[header_offset : header_offset + CHUNK_HEADER_BYTES] = header
        return RowRecord(
            fourcc=spec.fourcc,
            instance_uuid=spec.instance_uuid,
            chunk_version=spec.chunk_version,
            row_kind=ROW_LIVE,
            compression=0,
            flags=flags,
            header_offset=header_offset,
            payload_offset=payload_offset,
            stored_bytes=len(spec.payload),
            uncompressed_bytes=len(spec.payload),
            source_generation=generation,
            payload_crc32c=payload_crc,
            header_crc32c=header_crc,
        )

    @staticmethod
    def _pack_row(row: RowRecord) -> bytes:
        raw = bytearray(INDEX_ROW_BYTES)
        raw[0:4] = row.fourcc
        struct.pack_into("<HBBI", raw, 4, row.chunk_version, row.row_kind, row.compression, row.flags)
        raw[16:32] = row.instance_uuid.bytes
        struct.pack_into(
            "<QQQQQII",
            raw,
            32,
            row.header_offset,
            row.payload_offset,
            row.stored_bytes,
            row.uncompressed_bytes,
            row.source_generation,
            row.payload_crc32c,
            row.header_crc32c,
        )
        return bytes(raw)

    def _pack_index(self, generation: int, commit_uuid: uuid.UUID, rows: Sequence[RowRecord]) -> bytes:
        flags = (1 if any(row.row_kind == ROW_TOMBSTONE for row in rows) else 0) | (
            2 if any(row.row_kind == ROW_BASE_REF for row in rows) else 0
        )
        header = bytearray(INDEX_HEADER_BYTES)
        header[0:8] = INDEX_MAGIC
        struct.pack_into("<HHHHQQ", header, 8, 1, INDEX_HEADER_BYTES, INDEX_ROW_BYTES, 1, len(rows), generation)
        header[32:48] = commit_uuid.bytes
        struct.pack_into("<I", header, 48, flags)
        return bytes(header) + b"".join(self._pack_row(row) for row in rows)

    def _derive_reader_caps(self, rows: Sequence[RowRecord]) -> int:
        caps = 0
        if any(row.compression == 1 for row in rows):
            caps |= CAP_CHUNK_ZSTD
        if any(row.flags & CHUNK_BLOCK_CRCS for row in rows if row.row_kind == ROW_LIVE):
            caps |= CAP_BLOCK_CRC
        if any(row.row_kind == ROW_TOMBSTONE for row in rows):
            caps |= CAP_TOMBSTONES
        if self.role == ROLE_SIDECAR or any(row.row_kind == ROW_BASE_REF for row in rows):
            caps |= CAP_SIDECAR
        return caps

    def append_generation(
        self,
        *,
        commit_uuid: uuid.UUID,
        snapshot_uuid: uuid.UUID,
        changes: Sequence[ChunkSpec] = (),
        delete_keys: Sequence[tuple[bytes, uuid.UUID]] = (),
        head_slot: int,
        head_sequence: int,
        generation: int | None = None,
        root: bool = False,
        min_reader_version: tuple[int, int] = (1, 0),
        min_safe_writer_version: tuple[int, int] = (1, 0),
        extra_reader_capabilities: int = 0,
        extra_writer_capabilities: int = 0,
    ) -> GenerationLayout:
        if generation is None:
            generation = 1 if not self.generations else self.generations[-1].generation + 1
        for spec in changes:
            row = self._append_chunk(spec, generation)
            self.current_rows[row.key] = row
        for fourcc, instance_uuid in delete_keys:
            key = fourcc + instance_uuid.bytes
            self.current_rows[key] = RowRecord(
                fourcc=fourcc,
                instance_uuid=instance_uuid,
                chunk_version=0,
                row_kind=ROW_TOMBSTONE,
                compression=0,
                flags=0,
                header_offset=0,
                payload_offset=0,
                stored_bytes=0,
                uncompressed_bytes=0,
                source_generation=generation,
                payload_crc32c=0,
                header_crc32c=0,
            )

        rows = tuple(sorted(self.current_rows.values(), key=lambda row: row.key))
        index = self._pack_index(generation, commit_uuid, rows)
        index_offset = _align(len(self.data), 64)
        _ensure_size(self.data, index_offset + len(index))
        self.data[index_offset : index_offset + len(index)] = index
        commit_offset = _align(index_offset + len(index), 64)
        _ensure_size(self.data, commit_offset + COMMIT_BYTES)
        committed_end = commit_offset + COMMIT_BYTES

        if root or not self.generations or self.role == ROLE_SIDECAR:
            parent_uuid = uuid.UUID(int=0)
            parent_offset = 0
        else:
            parent_uuid = self.generations[-1].commit_uuid
            parent_offset = self.generations[-1].commit_offset
        kind = KIND_AUTOSAVE if self.role == ROLE_SIDECAR else KIND_EXPLICIT
        explicit_ancestor = self.base_explicit_commit_uuid if self.role == ROLE_SIDECAR else commit_uuid

        commit = bytearray(COMMIT_BYTES)
        commit[0:8] = COMMIT_MAGIC
        struct.pack_into("<HHI", commit, 8, COMMIT_BYTES, 1, kind)
        commit[16:32] = self.project_uuid.bytes
        commit[32:48] = self.file_uuid.bytes
        commit[48:64] = commit_uuid.bytes
        struct.pack_into("<Q", commit, 64, generation)
        commit[72:88] = parent_uuid.bytes
        struct.pack_into("<Q", commit, 88, parent_offset)
        commit[96:112] = explicit_ancestor.bytes
        commit[112:128] = snapshot_uuid.bytes
        struct.pack_into(
            "<QQQQII",
            commit,
            128,
            FIXED_COMMIT_TIME_NS + generation,
            index_offset,
            len(index),
            len(index),
            crc32c(index),
            crc32c(index),
        )
        struct.pack_into(
            "<IIQHHHH",
            commit,
            168,
            0,
            0,
            committed_end,
            min_reader_version[0],
            min_reader_version[1],
            min_safe_writer_version[0],
            min_safe_writer_version[1],
        )
        reader_caps = self._derive_reader_caps(rows) | extra_reader_capabilities
        writer_caps = (
            CAP_OPAQUE
            | CAP_CLEAN_PROOF
            | (CAP_SIDECAR if self.role == ROLE_SIDECAR else 0)
            | extra_writer_capabilities
        )
        commit[192:208] = reader_caps.to_bytes(16, "little")
        commit[208:224] = writer_caps.to_bytes(16, "little")
        commit_crc = crc32c(commit[:252])
        struct.pack_into("<I", commit, 252, commit_crc)
        self.data[commit_offset : committed_end] = commit

        layout = GenerationLayout(
            generation=generation,
            head_sequence=head_sequence,
            head_slot=head_slot,
            commit_uuid=commit_uuid,
            snapshot_uuid=snapshot_uuid,
            index_offset=index_offset,
            index_bytes=len(index),
            commit_offset=commit_offset,
            committed_file_end=committed_end,
            min_reader_version=min_reader_version,
            min_safe_writer_version=min_safe_writer_version,
            reader_capabilities=reader_caps,
            writer_capabilities=writer_caps,
            rows=rows,
        )
        self.generations.append(layout)
        self.write_head(layout, head_slot=head_slot, head_sequence=head_sequence)
        return layout

    def write_head(
        self,
        layout: GenerationLayout,
        *,
        head_slot: int,
        head_sequence: int | None = None,
    ) -> None:
        sequence = layout.head_sequence if head_sequence is None else head_sequence
        commit_crc = struct.unpack_from("<I", self.data, layout.commit_offset + 252)[0]
        head = bytearray(HEAD_BYTES)
        head[0:8] = HEAD_MAGIC
        struct.pack_into("<IIQQ", head, 8, head_slot, HEAD_BYTES, sequence, layout.generation)
        head[32:48] = self.project_uuid.bytes
        head[48:64] = self.file_uuid.bytes
        head[64:80] = layout.commit_uuid.bytes
        struct.pack_into("<QQQI", head, 80, layout.commit_offset, COMMIT_BYTES, layout.committed_file_end, commit_crc)
        struct.pack_into("<I", head, 4092, crc32c(head[:4092]))
        offset = HEAD_OFFSETS[head_slot]
        self.data[offset : offset + HEAD_BYTES] = head

    def bytes(self) -> bytes:
        return bytes(self.data)


def _json_payload(label: str, generation: int) -> bytes:
    return json.dumps(
        {"fixture": label, "generation": generation},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fixture_meta(
    name: str,
    fixture_class: str,
    data: bytes,
    expected_outcome: str,
    generations: Sequence[GenerationLayout],
    *,
    verify: bool,
    recovery: dict[str, object] | None = None,
    write_expected: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "file": name,
        "fixture_class": fixture_class,
        "expected_outcome": expected_outcome,
        "verify": verify,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "generations": [
            {
                "generation": layout.generation,
                "head_sequence": layout.head_sequence,
                "head_slot": layout.head_slot,
                "commit_uuid": str(layout.commit_uuid),
                "index_offset": layout.index_offset,
                "index_bytes": layout.index_bytes,
                "commit_offset": layout.commit_offset,
                "committed_file_end": layout.committed_file_end,
                "min_reader_version": list(layout.min_reader_version),
                "min_safe_writer_version": list(layout.min_safe_writer_version),
                "required_reader_capabilities": (
                    f"0x{layout.reader_capabilities:032x}"
                ),
                "required_writer_capabilities": (
                    f"0x{layout.writer_capabilities:032x}"
                ),
                "rows": [
                    {
                        "key": row.key.hex(),
                        "fourcc": row.fourcc.decode("ascii"),
                        "instance_uuid": str(row.instance_uuid),
                        "row_kind": row.row_kind,
                        "header_offset": row.header_offset,
                        "payload_offset": row.payload_offset,
                        "stored_bytes": row.stored_bytes,
                    }
                    for row in layout.rows
                ],
            }
            for layout in generations
        ],
    }
    if recovery is not None:
        result["recovery"] = recovery
    if write_expected is not None:
        result["write_expected"] = write_expected
    return result


def _refresh_generation_envelope(data: bytearray, layout: GenerationLayout, head_slots: Sequence[int]) -> None:
    index = bytes(data[layout.index_offset : layout.index_offset + layout.index_bytes])
    index_crc = crc32c(index)
    struct.pack_into("<II", data, layout.commit_offset + 160, index_crc, index_crc)
    commit_crc = crc32c(data[layout.commit_offset : layout.commit_offset + 252])
    struct.pack_into("<I", data, layout.commit_offset + 252, commit_crc)
    for slot in head_slots:
        head_offset = HEAD_OFFSETS[slot]
        struct.pack_into("<I", data, head_offset + 104, commit_crc)
        struct.pack_into("<I", data, head_offset + 4092, crc32c(data[head_offset : head_offset + 4092]))


def build_fixture_bytes() -> tuple[dict[str, bytes], dict[str, object]]:
    files: dict[str, bytes] = {}
    metadata: dict[str, object] = {}
    project = FIXED_UUIDS["project"]

    minimal = FixtureWriter(project_uuid=project, file_uuid=FIXED_UUIDS["file_minimal"])
    minimal_gen = minimal.append_generation(
        commit_uuid=FIXED_UUIDS["commit_minimal"],
        snapshot_uuid=FIXED_UUIDS["snapshot_minimal"],
        changes=(ChunkSpec(b"PROJ", FIXED_UUIDS["instance_proj"], _json_payload("minimal", 1)),),
        head_slot=0,
        head_sequence=1,
    )
    minimal_bytes = minimal.bytes()
    name = "minimal-valid-single-generation.licht"
    files[name] = minimal_bytes
    metadata[name] = _fixture_meta(name, "minimal_valid_single_generation", minimal_bytes, "open_gen_1", minimal.generations, verify=True)

    multi = FixtureWriter(project_uuid=project, file_uuid=FIXED_UUIDS["file_multi"])
    multi_gen1 = multi.append_generation(
        commit_uuid=FIXED_UUIDS["commit_multi_1"],
        snapshot_uuid=FIXED_UUIDS["snapshot_multi_1"],
        changes=(ChunkSpec(b"PROJ", FIXED_UUIDS["instance_proj"], _json_payload("multi", 1)),),
        head_slot=0,
        head_sequence=1,
    )
    multi_gen2 = multi.append_generation(
        commit_uuid=FIXED_UUIDS["commit_multi_2"],
        snapshot_uuid=FIXED_UUIDS["snapshot_multi_2"],
        changes=(ChunkSpec(b"SPLT", FIXED_UUIDS["instance_splt"], b"fixture tensor payload\x00\x01\x02", tensor=True, block_crcs=True),),
        head_slot=1,
        head_sequence=2,
    )
    multi_bytes = multi.bytes()
    name = "multi-generation-append.licht"
    files[name] = multi_bytes
    metadata[name] = _fixture_meta(name, "multi_generation_append", multi_bytes, "open_gen_2", multi.generations, verify=True)

    tombstone = FixtureWriter(project_uuid=project, file_uuid=FIXED_UUIDS["file_tombstone"])
    tombstone.append_generation(
        commit_uuid=FIXED_UUIDS["commit_tombstone_1"],
        snapshot_uuid=FIXED_UUIDS["snapshot_tombstone_1"],
        changes=(ChunkSpec(b"VIEW", FIXED_UUIDS["instance_view"], _json_payload("tombstone", 1)),),
        head_slot=0,
        head_sequence=1,
    )
    tombstone.append_generation(
        commit_uuid=FIXED_UUIDS["commit_tombstone_2"],
        snapshot_uuid=FIXED_UUIDS["snapshot_tombstone_2"],
        delete_keys=((b"VIEW", FIXED_UUIDS["instance_view"]),),
        head_slot=1,
        head_sequence=2,
    )
    tombstone_bytes = tombstone.bytes()
    name = "tombstone.licht"
    files[name] = tombstone_bytes
    metadata[name] = _fixture_meta(name, "tombstone", tombstone_bytes, "open_gen_2", tombstone.generations, verify=True)

    orphan = bytearray(minimal_bytes)
    orphan.extend(b"ORPHAN-TAIL-NOT-AUTHORITY\x00" * 3)
    orphan_bytes = bytes(orphan)
    name = "orphan-tail.licht"
    files[name] = orphan_bytes
    metadata[name] = _fixture_meta(name, "orphan_tail", orphan_bytes, "open_gen_1", minimal.generations, verify=True)

    torn = bytearray(multi_bytes)
    torn[HEAD_OFFSETS[1] + 4092] ^= 0x01
    torn_bytes = bytes(torn)
    name = "torn-head.licht"
    files[name] = torn_bytes
    metadata[name] = _fixture_meta(name, "torn_head", torn_bytes, "open_gen_1", multi.generations, verify=True)

    duplicate = FixtureWriter(project_uuid=project, file_uuid=FIXED_UUIDS["file_duplicate"])
    duplicate_gen = duplicate.append_generation(
        commit_uuid=FIXED_UUIDS["commit_duplicate"],
        snapshot_uuid=FIXED_UUIDS["snapshot_duplicate"],
        changes=(ChunkSpec(b"PROJ", FIXED_UUIDS["instance_proj"], _json_payload("duplicate", 1)),),
        head_slot=0,
        head_sequence=1,
    )
    duplicate.write_head(duplicate_gen, head_slot=1, head_sequence=1)
    duplicate_bytes = duplicate.bytes()
    name = "duplicate-slot-write.licht"
    files[name] = duplicate_bytes
    metadata[name] = _fixture_meta(name, "duplicate_slot_write", duplicate_bytes, "open_gen_1", duplicate.generations, verify=True)

    split = FixtureWriter(project_uuid=project, file_uuid=FIXED_UUIDS["file_split"])
    split.append_generation(
        commit_uuid=FIXED_UUIDS["commit_split_a"],
        snapshot_uuid=FIXED_UUIDS["snapshot_split_a"],
        changes=(ChunkSpec(b"PROJ", FIXED_UUIDS["instance_proj"], _json_payload("split-a", 1)),),
        head_slot=0,
        head_sequence=1,
        generation=1,
        root=True,
    )
    split.reset_logical_rows()
    split.append_generation(
        commit_uuid=FIXED_UUIDS["commit_split_b"],
        snapshot_uuid=FIXED_UUIDS["snapshot_split_b"],
        changes=(ChunkSpec(b"VIEW", FIXED_UUIDS["instance_view"], _json_payload("split-b", 1)),),
        head_slot=1,
        head_sequence=1,
        generation=1,
        root=True,
    )
    split_bytes = split.bytes()
    name = "split-brain.licht"
    files[name] = split_bytes
    metadata[name] = _fixture_meta(name, "split_brain", split_bytes, "hard_fail", split.generations, verify=False)

    oob = FixtureWriter(project_uuid=project, file_uuid=FIXED_UUIDS["file_oob"])
    oob_gen = oob.append_generation(
        commit_uuid=FIXED_UUIDS["commit_oob"],
        snapshot_uuid=FIXED_UUIDS["snapshot_oob"],
        changes=(ChunkSpec(b"PROJ", FIXED_UUIDS["instance_proj"], _json_payload("oob", 1)),),
        head_slot=0,
        head_sequence=1,
    )
    oob_data = bytearray(oob.bytes())
    first_row = oob_gen.index_offset + INDEX_HEADER_BYTES
    struct.pack_into("<Q", oob_data, first_row + 40, oob_gen.committed_file_end + 64)
    _refresh_generation_envelope(oob_data, oob_gen, (0,))
    oob_bytes = bytes(oob_data)
    name = "out-of-bounds-index-row.licht"
    files[name] = oob_bytes
    metadata[name] = _fixture_meta(name, "out_of_bounds_index_row", oob_bytes, "repair_only", oob.generations, verify=False)

    overlap = FixtureWriter(project_uuid=project, file_uuid=FIXED_UUIDS["file_overlap"])
    overlap_gen = overlap.append_generation(
        commit_uuid=FIXED_UUIDS["commit_overlap"],
        snapshot_uuid=FIXED_UUIDS["snapshot_overlap"],
        changes=(
            ChunkSpec(b"PROJ", FIXED_UUIDS["instance_proj"], _json_payload("overlap-a", 1)),
            ChunkSpec(b"VIEW", FIXED_UUIDS["instance_view"], _json_payload("overlap-b", 1)),
        ),
        head_slot=0,
        head_sequence=1,
    )
    overlap_data = bytearray(overlap.bytes())
    rows_by_key = sorted(overlap_gen.rows, key=lambda row: row.key)
    first = rows_by_key[0]
    second = rows_by_key[1]
    expanded_size = second.payload_end - first.payload_offset
    expanded_payload = bytes(overlap_data[first.payload_offset : second.payload_end])
    expanded_crc = crc32c(expanded_payload)
    struct.pack_into("<QQ", overlap_data, first.header_offset + 32, expanded_size, expanded_size)
    struct.pack_into("<I", overlap_data, first.header_offset + 56, expanded_crc)
    first_header_crc = crc32c(overlap_data[first.header_offset : first.header_offset + 60])
    struct.pack_into("<I", overlap_data, first.header_offset + 60, first_header_crc)
    first_index_position = rows_by_key.index(first)
    first_row_offset = overlap_gen.index_offset + INDEX_HEADER_BYTES + first_index_position * INDEX_ROW_BYTES
    struct.pack_into("<QQ", overlap_data, first_row_offset + 48, expanded_size, expanded_size)
    struct.pack_into("<II", overlap_data, first_row_offset + 72, expanded_crc, first_header_crc)
    _refresh_generation_envelope(overlap_data, overlap_gen, (0,))
    overlap_bytes = bytes(overlap_data)
    name = "overlapping-rows.licht"
    files[name] = overlap_bytes
    metadata[name] = _fixture_meta(name, "overlapping_rows", overlap_bytes, "repair_only", overlap.generations, verify=False)

    unsupported_single = FixtureWriter(
        project_uuid=project,
        file_uuid=FIXED_UUIDS["file_unsupported_single"],
    )
    unsupported_single.append_generation(
        commit_uuid=FIXED_UUIDS["commit_unsupported_single"],
        snapshot_uuid=FIXED_UUIDS["snapshot_unsupported_single"],
        changes=(
            ChunkSpec(
                b"PROJ",
                FIXED_UUIDS["instance_proj"],
                _json_payload("unsupported-newer-single", 1),
            ),
        ),
        head_slot=0,
        head_sequence=1,
        min_reader_version=(1, 1),
    )
    unsupported_single_bytes = unsupported_single.bytes()
    name = "unsupported-newer-single-head.licht"
    files[name] = unsupported_single_bytes
    metadata[name] = _fixture_meta(
        name,
        "unsupported_newer_single_head",
        unsupported_single_bytes,
        "unsupported_newer",
        unsupported_single.generations,
        verify=False,
    )

    unsupported_higher = FixtureWriter(
        project_uuid=project,
        file_uuid=FIXED_UUIDS["file_unsupported_higher"],
    )
    unsupported_higher.append_generation(
        commit_uuid=FIXED_UUIDS["commit_unsupported_higher_1"],
        snapshot_uuid=FIXED_UUIDS["snapshot_unsupported_higher_1"],
        changes=(
            ChunkSpec(
                b"PROJ",
                FIXED_UUIDS["instance_proj"],
                _json_payload("unsupported-higher", 1),
            ),
        ),
        head_slot=0,
        head_sequence=1,
    )
    unsupported_higher.append_generation(
        commit_uuid=FIXED_UUIDS["commit_unsupported_higher_2"],
        snapshot_uuid=FIXED_UUIDS["snapshot_unsupported_higher_2"],
        changes=(
            ChunkSpec(
                b"VIEW",
                FIXED_UUIDS["instance_view"],
                _json_payload("unsupported-higher", 2),
            ),
        ),
        head_slot=1,
        head_sequence=2,
        extra_reader_capabilities=CAP_FUTURE_BIT_8,
    )
    unsupported_higher_bytes = unsupported_higher.bytes()
    name = "unsupported-newer-higher-head.licht"
    files[name] = unsupported_higher_bytes
    metadata[name] = _fixture_meta(
        name,
        "unsupported_newer_higher_head",
        unsupported_higher_bytes,
        "hard_fail",
        unsupported_higher.generations,
        verify=False,
    )

    write_unsafe = FixtureWriter(
        project_uuid=project,
        file_uuid=FIXED_UUIDS["file_write_unsafe"],
    )
    write_unsafe.append_generation(
        commit_uuid=FIXED_UUIDS["commit_write_unsafe"],
        snapshot_uuid=FIXED_UUIDS["snapshot_write_unsafe"],
        changes=(
            ChunkSpec(
                b"PROJ",
                FIXED_UUIDS["instance_proj"],
                _json_payload("write-unsafe", 1),
            ),
        ),
        head_slot=0,
        head_sequence=1,
        min_safe_writer_version=(1, 1),
        extra_writer_capabilities=CAP_FUTURE_BIT_8,
    )
    write_unsafe_bytes = write_unsafe.bytes()
    name = "write-unsafe.licht"
    files[name] = write_unsafe_bytes
    metadata[name] = _fixture_meta(
        name,
        "write_unsafe",
        write_unsafe_bytes,
        "open_gen_1",
        write_unsafe.generations,
        verify=True,
        write_expected="unsafe",
    )

    autosave_master = FixtureWriter(project_uuid=project, file_uuid=FIXED_UUIDS["file_autosave_master"])
    autosave_master_gen = autosave_master.append_generation(
        commit_uuid=FIXED_UUIDS["commit_autosave_master"],
        snapshot_uuid=FIXED_UUIDS["snapshot_autosave_master"],
        changes=(
            ChunkSpec(b"CKPT", FIXED_UUIDS["instance_ckpt"], b"base checkpoint bytes", tensor=True),
            ChunkSpec(b"PROJ", FIXED_UUIDS["instance_proj"], _json_payload("autosave-master", 1)),
        ),
        head_slot=0,
        head_sequence=1,
    )
    autosave_master_bytes = autosave_master.bytes()
    master_name = "autosave-master.licht"
    files[master_name] = autosave_master_bytes
    metadata[master_name] = _fixture_meta(master_name, "autosave_master", autosave_master_bytes, "open_gen_1", autosave_master.generations, verify=True)

    autosave_valid = FixtureWriter(
        project_uuid=project,
        file_uuid=FIXED_UUIDS["file_autosave_valid"],
        role=ROLE_SIDECAR,
        base_explicit_commit_uuid=autosave_master_gen.commit_uuid,
        autosave_sequence=7,
        sidecar_snapshot_uuid=FIXED_UUIDS["snapshot_autosave_valid"],
    )
    autosave_valid.seed_base_refs(autosave_master_gen.rows)
    autosave_valid.append_generation(
        commit_uuid=FIXED_UUIDS["commit_autosave_valid"],
        snapshot_uuid=FIXED_UUIDS["snapshot_autosave_valid"],
        changes=(ChunkSpec(b"CKPT", FIXED_UUIDS["instance_ckpt"], b"autosaved checkpoint bytes", tensor=True),),
        head_slot=0,
        head_sequence=1,
        generation=1,
        root=True,
    )
    autosave_valid_bytes = autosave_valid.bytes()
    valid_name = "autosave-sidecar-valid.licht.autosave"
    files[valid_name] = autosave_valid_bytes
    metadata[valid_name] = _fixture_meta(
        valid_name,
        "autosave_sidecar_valid",
        autosave_valid_bytes,
        "open_gen_1",
        autosave_valid.generations,
        verify=True,
        recovery={"master": master_name, "expected": "offer"},
    )

    autosave_stale = FixtureWriter(
        project_uuid=project,
        file_uuid=FIXED_UUIDS["file_autosave_stale"],
        role=ROLE_SIDECAR,
        base_explicit_commit_uuid=FIXED_UUIDS["commit_stale_base"],
        autosave_sequence=8,
        sidecar_snapshot_uuid=FIXED_UUIDS["snapshot_autosave_stale"],
    )
    autosave_stale.seed_base_refs(autosave_master_gen.rows)
    autosave_stale.append_generation(
        commit_uuid=FIXED_UUIDS["commit_autosave_stale"],
        snapshot_uuid=FIXED_UUIDS["snapshot_autosave_stale"],
        changes=(ChunkSpec(b"CKPT", FIXED_UUIDS["instance_ckpt"], b"stale autosave checkpoint", tensor=True),),
        head_slot=0,
        head_sequence=1,
        generation=1,
        root=True,
    )
    autosave_stale_bytes = autosave_stale.bytes()
    stale_name = "autosave-sidecar-stale-base.licht.autosave"
    files[stale_name] = autosave_stale_bytes
    metadata[stale_name] = _fixture_meta(
        stale_name,
        "autosave_sidecar_stale_base",
        autosave_stale_bytes,
        "open_gen_1",
        autosave_stale.generations,
        verify=True,
        recovery={"master": master_name, "expected": "stale_delete"},
    )

    manifest = {
        "schema": 1,
        "generator": "make_fixtures.py non-production grammar writer",
        "index_compression": "stored",
        "fixed_creation_time_unix_ns": FIXED_CREATION_TIME_NS,
        "fixed_commit_time_base_unix_ns": FIXED_COMMIT_TIME_NS,
        "fixed_uuids": {name: str(value) for name, value in sorted(FIXED_UUIDS.items())},
        "fixtures": metadata,
    }
    return files, manifest


def generate_fixtures(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files, manifest = build_fixture_bytes()
    for name, data in sorted(files.items()):
        (output_dir / name).write_bytes(data)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output_dir / "manifest.json").write_bytes(manifest_bytes)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args(argv)
    manifest = generate_fixtures(args.output_dir)
    print("NON-PRODUCTION fixture writer: generated deterministic stored-index corpus")
    for name, info in sorted(manifest["fixtures"].items()):
        print(f"{name}: {info['bytes']} bytes sha256={info['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
