#!/usr/bin/env python3
"""Independent, read-only parser and CLI for LichtFeld .licht containers.

The parser is intentionally implemented from docs/licht_format_spec.md and has
no dependency on the application or its C++ container implementation.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
from pathlib import Path
import re
import struct
import sys
import uuid
from typing import BinaryIO, Iterable, Iterator, Sequence

try:
    from .crc32c import crc32c
except ImportError:  # Direct script execution.
    from crc32c import crc32c

try:
    import zstandard as _zstd
except ImportError:  # Optional by specification.
    _zstd = None


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
BLOCK_REQUIRED_AT = 1024 * 1024 * 1024

SUPER_MAGIC = b"\x89LFS\r\n\x1a\n"
HEAD_MAGIC = b"LFSHEAD\x00"
COMMIT_MAGIC = b"LFSCOMIT"
INDEX_MAGIC = b"LFSINDEX"
BLOCK_MAGIC = b"LFSBCRC\x00"
BYTE_ORDER_TAG = 0x01020304

ROLE_MASTER = 0
ROLE_SIDECAR = 1

COMPRESSION_STORED = 0
COMPRESSION_ZSTD = 1
COMPRESSION_NAMES = {0: "stored", 1: "zstd"}

KIND_EXPLICIT = 1
KIND_AUTOSAVE = 2
KIND_RECOVERED = 3
KIND_COMPACTION = 4
KIND_NAMES = {
    KIND_EXPLICIT: "EXPLICIT",
    KIND_AUTOSAVE: "AUTOSAVE",
    KIND_RECOVERED: "RECOVERED",
    KIND_COMPACTION: "COMPACTION",
}

ROW_LIVE = 0
ROW_TOMBSTONE = 1
ROW_BASE_REF = 2
ROW_KIND_NAMES = {0: "live", 1: "tombstone", 2: "base_ref"}

CHUNK_TENSOR = 1 << 0
CHUNK_BLOCK_CRCS = 1 << 1
KNOWN_CHUNK_FLAGS = CHUNK_TENSOR | CHUNK_BLOCK_CRCS

CAP_INDEX_ZSTD = 1 << 0
CAP_CHUNK_ZSTD = 1 << 1
CAP_BLOCK_CRC = 1 << 2
CAP_TOMBSTONES = 1 << 3
CAP_SIDECAR = 1 << 4
CAP_OPAQUE = 1 << 5
CAP_RETAINED_JSON = 1 << 6
CAP_CLEAN_PROOF = 1 << 7

CURRENT_VERSION = (1, 0)
_BASE_READER_CAPS = CAP_BLOCK_CRC | CAP_TOMBSTONES | CAP_SIDECAR
SUPPORTED_READER_CAPS = _BASE_READER_CAPS | (
    (CAP_INDEX_ZSTD | CAP_CHUNK_ZSTD) if _zstd is not None else 0
)
DECLARED_V1_WRITER_CAPS = (1 << 8) - 1

_FOURCC_RE = re.compile(br"^[A-Z0-9]{4}$")
NULL_UUID = uuid.UUID(int=0)


class FormatError(Exception):
    """A byte-addressed format validation error."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        offset: int,
        field: str,
        expected: object,
        got: object,
        *,
        code: str = "corruption",
    ) -> None:
        self.path = str(path)
        self.offset = offset
        self.field = field
        self.expected = expected
        self.got = got
        self.code = code
        super().__init__(str(self))

    def __str__(self) -> str:
        return (
            f"{self.path}: 0x{self.offset:016x} {self.field}: "
            f"expected {self.expected}, got {self.got}"
        )


class HardFailure(FormatError):
    """Normal open must stop rather than attempt automatic repair."""


class RepairRequired(FormatError):
    """No authoritative head survived; explicit scan/repair is required."""


class UnsupportedFeature(FormatError):
    """The bytes may be valid, but this parser cannot interpret the feature."""


class WriteRefused(FormatError):
    """A declared writer is not safe for the selected generation."""


def _raise_like(error: FormatError, cls: type[FormatError]) -> FormatError:
    return cls(
        error.path,
        error.offset,
        error.field,
        error.expected,
        error.got,
        code=error.code,
    )


def _u8(data: bytes, offset: int) -> int:
    return data[offset]


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def _uuid(data: bytes, offset: int) -> uuid.UUID:
    return uuid.UUID(bytes=data[offset : offset + 16])


def _fmt_bytes(value: bytes, limit: int = 32) -> str:
    suffix = "..." if len(value) > limit else ""
    return "0x" + value[:limit].hex() + suffix


def _fmt_u32(value: int) -> str:
    return f"0x{value:08x}"


def _checked_end(
    path: str | os.PathLike[str],
    offset: int,
    size: int,
    field: str,
) -> int:
    if offset < 0 or size < 0:
        raise FormatError(path, offset, field, "non-negative offset and size", (offset, size))
    end = offset + size
    if end < offset or end > 0xFFFFFFFFFFFFFFFF:
        raise FormatError(path, offset, field, "u64 range without overflow", (offset, size))
    return end


def _require(
    condition: bool,
    path: str | os.PathLike[str],
    absolute_offset: int,
    field: str,
    expected: object,
    got: object,
    *,
    error_type: type[FormatError] = FormatError,
) -> None:
    if not condition:
        raise error_type(path, absolute_offset, field, expected, got)


def _require_zero(
    data: bytes,
    start: int,
    end: int,
    path: str | os.PathLike[str],
    base: int,
    field: str,
) -> None:
    bad = next((i for i, value in enumerate(data[start:end], start) if value), None)
    if bad is not None:
        raise FormatError(path, base + bad, field, "zero", f"0x{data[bad]:02x}")


class FileReader:
    """Bounded positional reader; it never owns a shared seek cursor."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = str(path)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size

    def close(self) -> None:
        os.close(self.fd)

    def __enter__(self) -> "FileReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read_exact(
        self,
        offset: int,
        size: int,
        field: str,
        *,
        authority_end: int | None = None,
    ) -> bytes:
        end = _checked_end(self.path, offset, size, field)
        limit = self.size if authority_end is None else min(self.size, authority_end)
        if end > limit:
            raise FormatError(
                self.path,
                offset,
                field,
                f"range [0x{offset:x},0x{end:x}) within [0x0,0x{limit:x})",
                f"physical_size=0x{self.size:x}, authority_end="
                + ("none" if authority_end is None else f"0x{authority_end:x}"),
            )
        chunks: list[bytes] = []
        cursor = offset
        remaining = size
        while remaining:
            chunk = os.pread(self.fd, remaining, cursor)
            if not chunk:
                raise FormatError(
                    self.path,
                    cursor,
                    field,
                    f"{remaining} more bytes",
                    "unexpected EOF",
                )
            chunks.append(chunk)
            cursor += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


@dataclasses.dataclass(frozen=True)
class Superblock:
    format_major: int
    format_minor: int
    role: int
    project_uuid: uuid.UUID
    file_uuid: uuid.UUID
    creation_time_ns: int
    base_explicit_commit_uuid: uuid.UUID
    autosave_sequence: int
    sidecar_snapshot_uuid: uuid.UUID
    crc32c: int

    @property
    def role_name(self) -> str:
        return "master" if self.role == ROLE_MASTER else "autosave_sidecar"


@dataclasses.dataclass(frozen=True)
class Commit:
    offset: int
    kind: int
    project_uuid: uuid.UUID
    file_uuid: uuid.UUID
    commit_uuid: uuid.UUID
    generation: int
    parent_commit_uuid: uuid.UUID
    parent_commit_offset: int
    explicit_ancestor_commit_uuid: uuid.UUID
    snapshot_uuid: uuid.UUID
    wallclock_ns: int
    index_offset: int
    index_stored_bytes: int
    index_uncompressed_bytes: int
    index_stored_crc32c: int
    index_uncompressed_crc32c: int
    index_compression: int
    committed_file_end: int
    min_reader_version: tuple[int, int]
    min_safe_writer_version: tuple[int, int]
    reader_capabilities: int
    writer_capabilities: int
    crc32c: int

    @property
    def kind_name(self) -> str:
        return KIND_NAMES.get(self.kind, f"UNKNOWN({self.kind})")


@dataclasses.dataclass(frozen=True)
class BlockTable:
    offset: int
    payload_offset: int
    stored_bytes: int
    block_size: int
    entries: tuple[int, ...]
    entries_crc32c: int
    header_crc32c: int

    @property
    def end(self) -> int:
        return self.offset + BLOCK_HEADER_BYTES + 4 * len(self.entries)


@dataclasses.dataclass(frozen=True)
class IndexRow:
    row_offset: int
    fourcc: bytes
    chunk_version: int
    row_kind: int
    compression: int
    chunk_flags: int
    instance_uuid: uuid.UUID
    header_offset: int
    payload_offset: int
    stored_bytes: int
    uncompressed_bytes: int
    source_generation: int
    payload_crc32c: int
    header_crc32c: int
    block_table: BlockTable | None = None

    @property
    def key(self) -> bytes:
        return self.fourcc + self.instance_uuid.bytes

    @property
    def key_text(self) -> str:
        return f"{self.fourcc.decode('ascii')}:{self.instance_uuid}"

    @property
    def kind_name(self) -> str:
        return ROW_KIND_NAMES.get(self.row_kind, f"unknown({self.row_kind})")

    @property
    def occupied_span(self) -> tuple[int, int] | None:
        if self.row_kind != ROW_LIVE:
            return None
        return (self.header_offset, self.payload_offset + self.stored_bytes)


@dataclasses.dataclass(frozen=True)
class Index:
    stored_bytes: bytes
    uncompressed_bytes: bytes
    generation: int
    commit_uuid: uuid.UUID
    flags: int
    rows: tuple[IndexRow, ...]


@dataclasses.dataclass(frozen=True)
class Head:
    slot_id: int
    offset: int
    head_sequence: int
    generation: int
    project_uuid: uuid.UUID
    file_uuid: uuid.UUID
    commit_uuid: uuid.UUID
    commit_offset: int
    commit_bytes: int
    committed_file_end: int
    commit_crc32c_echo: int
    head_crc32c: int
    commit: Commit
    index: Index

    @property
    def authority_tuple(self) -> tuple[object, ...]:
        return (
            self.generation,
            self.commit_offset,
            self.commit_bytes,
            self.committed_file_end,
            self.project_uuid,
            self.file_uuid,
        )


@dataclasses.dataclass(frozen=True)
class HeadAttempt:
    slot_id: int
    offset: int
    blank: bool
    sequence_hint: int | None
    head: Head | None
    error: FormatError | None
    compatibility_error: UnsupportedFeature | None = None


@dataclasses.dataclass(frozen=True)
class WriteCompatibility:
    writer_version: tuple[int, int]
    writer_capabilities: int
    safe: bool
    reasons: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Container:
    path: str
    physical_size: int
    superblock: Superblock
    head_attempts: tuple[HeadAttempt, HeadAttempt]
    selected_head: Head
    warnings: tuple[str, ...]
    open_state: str = "open"

    @property
    def commit(self) -> Commit:
        return self.selected_head.commit

    @property
    def index(self) -> Index:
        return self.selected_head.index

    @property
    def write_compatibility(self) -> WriteCompatibility:
        return assess_write_compatibility(self)

    def find_row(self, fourcc: str, instance_uuid: uuid.UUID) -> IndexRow:
        key = fourcc.encode("ascii") + instance_uuid.bytes
        matches = [row for row in self.index.rows if row.key == key]
        if not matches:
            raise KeyError(f"no index row for {fourcc}:{instance_uuid}")
        row = matches[0]
        if row.row_kind != ROW_LIVE:
            raise KeyError(f"{row.key_text} is {row.kind_name}, not a local live payload")
        return row


class UnsupportedNewer(FormatError):
    """Structurally valid authority requires a newer reader/capability set."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        offset: int,
        field: str,
        expected: object,
        got: object,
        *,
        inspection: Container,
    ) -> None:
        self.inspection = inspection
        super().__init__(
            path,
            offset,
            field,
            expected,
            got,
            code="unsupported_newer",
        )


def assess_write_compatibility(
    container: Container,
    *,
    writer_version: tuple[int, int] = CURRENT_VERSION,
    writer_capabilities: int = DECLARED_V1_WRITER_CAPS,
) -> WriteCompatibility:
    """Classify whether a declared writer may safely modify the selected head."""

    commit = container.commit
    reasons: list[str] = []
    if commit.min_safe_writer_version > writer_version:
        reasons.append(
            "min_safe_writer_version "
            f"{commit.min_safe_writer_version[0]}.{commit.min_safe_writer_version[1]} "
            f"exceeds declared writer {writer_version[0]}.{writer_version[1]}"
        )
    missing_capabilities = commit.writer_capabilities & ~writer_capabilities
    if missing_capabilities:
        reasons.append(
            "required_writer_capabilities contains unsupported bits "
            f"0x{missing_capabilities:032x}"
        )
    return WriteCompatibility(
        writer_version=writer_version,
        writer_capabilities=writer_capabilities,
        safe=not reasons,
        reasons=tuple(reasons),
    )


def require_safe_write(
    container: Container,
    *,
    writer_version: tuple[int, int] = CURRENT_VERSION,
    writer_capabilities: int = DECLARED_V1_WRITER_CAPS,
) -> None:
    """Refuse before mutation unless the declared writer satisfies both gates."""

    compatibility = assess_write_compatibility(
        container,
        writer_version=writer_version,
        writer_capabilities=writer_capabilities,
    )
    if compatibility.safe:
        return
    commit = container.commit
    offset = (
        commit.offset + 188
        if commit.min_safe_writer_version > writer_version
        else commit.offset + 208
    )
    raise WriteRefused(
        container.path,
        offset,
        "commit.write_compatibility",
        "declared writer satisfies min_safe_writer_version and all required writer capabilities",
        "; ".join(compatibility.reasons),
        code="write_unsafe",
    )


def _parse_superblock(reader: FileReader) -> Superblock:
    data = reader.read_exact(0, SUPERBLOCK_BYTES, "superblock")
    _require(data[0:8] == SUPER_MAGIC, reader.path, 0, "superblock.magic", _fmt_bytes(SUPER_MAGIC), _fmt_bytes(data[0:8]))
    got_crc = _u32(data, 252)
    expected_crc = crc32c(data[:252])
    _require(got_crc == expected_crc, reader.path, 252, "superblock.crc32c", _fmt_u32(expected_crc), _fmt_u32(got_crc))

    major = _u16(data, 8)
    minor = _u16(data, 10)
    _require(major == 1, reader.path, 8, "superblock.format_major", 1, major, error_type=UnsupportedFeature)
    _require(_u32(data, 12) == BYTE_ORDER_TAG, reader.path, 12, "superblock.byte_order_tag", _fmt_u32(BYTE_ORDER_TAG), _fmt_u32(_u32(data, 12)))
    _require(_u32(data, 16) == SUPERBLOCK_BYTES, reader.path, 16, "superblock.superblock_bytes", SUPERBLOCK_BYTES, _u32(data, 16))
    role = _u32(data, 20)
    _require(role in (ROLE_MASTER, ROLE_SIDECAR), reader.path, 20, "superblock.container_role", "0 or 1", role)
    project_uuid = _uuid(data, 24)
    file_uuid = _uuid(data, 40)
    _require(project_uuid != NULL_UUID, reader.path, 24, "superblock.project_uuid", "non-null UUID", str(project_uuid))
    _require(file_uuid != NULL_UUID, reader.path, 40, "superblock.file_uuid", "non-null UUID", str(file_uuid))
    geometry = (
        _u64(data, 56),
        _u64(data, 64),
        _u32(data, 72),
        _u32(data, 76),
        _u64(data, 80),
    )
    expected_geometry = (4096, 8192, HEAD_BYTES, 2, APPEND_OFFSET)
    _require(geometry == expected_geometry, reader.path, 56, "superblock.head_geometry", expected_geometry, geometry)

    base_uuid = _uuid(data, 96)
    autosave_sequence = _u64(data, 112)
    _require_zero(data, 120, 128, reader.path, 0, "superblock.reserved_0")
    snapshot_uuid = _uuid(data, 128)
    _require_zero(data, 144, 252, reader.path, 0, "superblock.reserved_1")
    if role == ROLE_MASTER:
        _require(data[96:144] == b"\x00" * 48, reader.path, 96, "superblock.master_sidecar_binding", "48 zero bytes", _fmt_bytes(data[96:144]))
    else:
        _require(base_uuid != NULL_UUID, reader.path, 96, "superblock.base_explicit_commit_uuid", "non-null UUID", str(base_uuid))
        _require(autosave_sequence >= 1, reader.path, 112, "superblock.autosave_sequence", ">= 1", autosave_sequence)
        _require(snapshot_uuid != NULL_UUID, reader.path, 128, "superblock.sidecar_snapshot_uuid", "non-null UUID", str(snapshot_uuid))

    return Superblock(
        format_major=major,
        format_minor=minor,
        role=role,
        project_uuid=project_uuid,
        file_uuid=file_uuid,
        creation_time_ns=_u64(data, 88),
        base_explicit_commit_uuid=base_uuid,
        autosave_sequence=autosave_sequence,
        sidecar_snapshot_uuid=snapshot_uuid,
        crc32c=got_crc,
    )


def _parse_commit_record(
    reader: FileReader,
    sb: Superblock,
    offset: int,
    authority_end: int,
) -> Commit:
    data = reader.read_exact(offset, COMMIT_BYTES, "commit_record", authority_end=authority_end)
    _require(data[:8] == COMMIT_MAGIC, reader.path, offset, "commit.magic", _fmt_bytes(COMMIT_MAGIC), _fmt_bytes(data[:8]))
    got_crc = _u32(data, 252)
    expected_crc = crc32c(data[:252])
    _require(got_crc == expected_crc, reader.path, offset + 252, "commit.crc32c", _fmt_u32(expected_crc), _fmt_u32(got_crc))
    _require(_u16(data, 8) == COMMIT_BYTES, reader.path, offset + 8, "commit.record_bytes", COMMIT_BYTES, _u16(data, 8))
    _require(_u16(data, 10) == 1, reader.path, offset + 10, "commit.record_version", 1, _u16(data, 10), error_type=UnsupportedFeature)

    kind = _u32(data, 12)
    _require(kind in KIND_NAMES, reader.path, offset + 12, "commit.kind", sorted(KIND_NAMES), kind)
    project_uuid = _uuid(data, 16)
    file_uuid = _uuid(data, 32)
    commit_uuid = _uuid(data, 48)
    generation = _u64(data, 64)
    parent_uuid = _uuid(data, 72)
    parent_offset = _u64(data, 88)
    explicit_uuid = _uuid(data, 96)
    snapshot_uuid = _uuid(data, 112)
    index_offset = _u64(data, 136)
    index_stored_bytes = _u64(data, 144)
    index_uncompressed_bytes = _u64(data, 152)
    index_compression = _u32(data, 168)
    committed_end = _u64(data, 176)

    _require(project_uuid == sb.project_uuid, reader.path, offset + 16, "commit.project_uuid", str(sb.project_uuid), str(project_uuid))
    _require(file_uuid == sb.file_uuid, reader.path, offset + 32, "commit.file_uuid", str(sb.file_uuid), str(file_uuid))
    _require(commit_uuid != NULL_UUID, reader.path, offset + 48, "commit.commit_uuid", "non-null UUID", str(commit_uuid))
    _require(generation >= 1, reader.path, offset + 64, "commit.generation", ">= 1", generation)
    _require(snapshot_uuid != NULL_UUID, reader.path, offset + 112, "commit.snapshot_uuid", "non-null UUID", str(snapshot_uuid))
    _require(index_offset >= APPEND_OFFSET and index_offset % 64 == 0, reader.path, offset + 136, "commit.index_offset", ">= 0x10000 and 64-byte aligned", f"0x{index_offset:x}")
    _require(index_stored_bytes > 0, reader.path, offset + 144, "commit.index_stored_bytes", "> 0", index_stored_bytes)
    _require(index_uncompressed_bytes >= INDEX_HEADER_BYTES, reader.path, offset + 152, "commit.index_uncompressed_bytes", f">= {INDEX_HEADER_BYTES}", index_uncompressed_bytes)
    _require(index_compression in COMPRESSION_NAMES, reader.path, offset + 168, "commit.index_compression", "0 or 1", index_compression)
    _require(_u32(data, 172) == 0, reader.path, offset + 172, "commit.commit_flags", 0, _u32(data, 172))
    _require(committed_end == offset + COMMIT_BYTES, reader.path, offset + 176, "commit.committed_file_end", f"0x{offset + COMMIT_BYTES:x}", f"0x{committed_end:x}")
    index_end = _checked_end(reader.path, index_offset, index_stored_bytes, "commit.index_range")
    _require(index_end <= offset, reader.path, offset + 136, "commit.index_range", f"end <= commit_offset 0x{offset:x}", f"[0x{index_offset:x},0x{index_end:x})")
    if index_end < offset:
        padding = reader.read_exact(index_end, offset - index_end, "commit.index_to_commit_padding", authority_end=authority_end)
        _require_zero(padding, 0, len(padding), reader.path, index_end, "commit.index_to_commit_padding")
    _require_zero(data, 224, 252, reader.path, offset, "commit.reserved")

    if sb.role == ROLE_MASTER:
        _require(kind != KIND_AUTOSAVE, reader.path, offset + 12, "commit.kind", "EXPLICIT, RECOVERED, or COMPACTION in master", KIND_NAMES[kind])
        _require(explicit_uuid == commit_uuid, reader.path, offset + 96, "commit.explicit_ancestor_commit_uuid", str(commit_uuid), str(explicit_uuid))
        if kind == KIND_COMPACTION:
            _require(generation == 1, reader.path, offset + 64, "commit.compaction_generation", 1, generation)
    else:
        _require(kind == KIND_AUTOSAVE, reader.path, offset + 12, "commit.kind", "AUTOSAVE in sidecar", KIND_NAMES[kind])
        _require(generation == 1, reader.path, offset + 64, "commit.sidecar_generation", 1, generation)
        _require(explicit_uuid == sb.base_explicit_commit_uuid, reader.path, offset + 96, "commit.explicit_ancestor_commit_uuid", str(sb.base_explicit_commit_uuid), str(explicit_uuid))
        _require(snapshot_uuid == sb.sidecar_snapshot_uuid, reader.path, offset + 112, "commit.snapshot_uuid", str(sb.sidecar_snapshot_uuid), str(snapshot_uuid))

    if generation == 1:
        _require(parent_uuid == NULL_UUID, reader.path, offset + 72, "commit.parent_commit_uuid", str(NULL_UUID), str(parent_uuid))
        _require(parent_offset == 0, reader.path, offset + 88, "commit.parent_commit_offset", 0, parent_offset)
    else:
        _require(parent_uuid != NULL_UUID, reader.path, offset + 72, "commit.parent_commit_uuid", "non-null UUID", str(parent_uuid))
        _require(parent_offset >= APPEND_OFFSET and parent_offset % 64 == 0 and parent_offset < offset, reader.path, offset + 88, "commit.parent_commit_offset", "earlier 64-byte-aligned append offset", f"0x{parent_offset:x}")

    min_reader = (_u16(data, 184), _u16(data, 186))
    min_writer = (_u16(data, 188), _u16(data, 190))
    reader_caps = int.from_bytes(data[192:208], "little")
    writer_caps = int.from_bytes(data[208:224], "little")

    return Commit(
        offset=offset,
        kind=kind,
        project_uuid=project_uuid,
        file_uuid=file_uuid,
        commit_uuid=commit_uuid,
        generation=generation,
        parent_commit_uuid=parent_uuid,
        parent_commit_offset=parent_offset,
        explicit_ancestor_commit_uuid=explicit_uuid,
        snapshot_uuid=snapshot_uuid,
        wallclock_ns=_u64(data, 128),
        index_offset=index_offset,
        index_stored_bytes=index_stored_bytes,
        index_uncompressed_bytes=index_uncompressed_bytes,
        index_stored_crc32c=_u32(data, 160),
        index_uncompressed_crc32c=_u32(data, 164),
        index_compression=index_compression,
        committed_file_end=committed_end,
        min_reader_version=min_reader,
        min_safe_writer_version=min_writer,
        reader_capabilities=reader_caps,
        writer_capabilities=writer_caps,
        crc32c=got_crc,
    )


def _validate_lineage(
    reader: FileReader,
    sb: Superblock,
    commit: Commit,
) -> tuple[Commit, ...]:
    seen = {commit.offset}
    seen_commit_uuids = {commit.commit_uuid}
    child = commit
    newest_to_oldest = [commit]
    while child.generation > 1:
        parent_offset = child.parent_commit_offset
        _require(parent_offset not in seen, reader.path, child.offset + 88, "commit.parent_commit_offset", "acyclic lineage", f"cycle to 0x{parent_offset:x}")
        seen.add(parent_offset)
        parent = _parse_commit_record(reader, sb, parent_offset, commit.committed_file_end)
        _require(parent.commit_uuid not in seen_commit_uuids, reader.path, parent.offset + 48, "commit.commit_uuid", "unique within same-file lineage", str(parent.commit_uuid))
        seen_commit_uuids.add(parent.commit_uuid)
        _require(parent.commit_uuid == child.parent_commit_uuid, reader.path, child.offset + 72, "commit.parent_commit_uuid", str(parent.commit_uuid), str(child.parent_commit_uuid))
        _require(parent.generation + 1 == child.generation, reader.path, child.offset + 64, "commit.generation_link", parent.generation + 1, child.generation)
        _require(child.index_offset >= parent.committed_file_end, reader.path, child.offset + 136, "commit.generation_append_start", f"index_offset >= parent committed_file_end 0x{parent.committed_file_end:x}", f"0x{child.index_offset:x}")
        _require(child.offset > parent.committed_file_end, reader.path, child.offset, "commit.generation_commit_order", f"> parent committed_file_end 0x{parent.committed_file_end:x}", f"0x{child.offset:x}")
        newest_to_oldest.append(parent)
        child = parent
    _require(child.parent_commit_uuid == NULL_UUID and child.parent_commit_offset == 0, reader.path, child.offset + 72, "commit.root_parent", "null UUID and zero offset", (str(child.parent_commit_uuid), child.parent_commit_offset))
    _require(child.index_offset >= APPEND_OFFSET, reader.path, child.offset + 136, "commit.root_append_start", f">= 0x{APPEND_OFFSET:x}", f"0x{child.index_offset:x}")
    return tuple(reversed(newest_to_oldest))


def _decode_zstd_exact(path: str, offset: int, field: str, stored: bytes, expected_bytes: int) -> bytes:
    if _zstd is None:
        raise UnsupportedFeature(path, offset, field, "Python package 'zstandard' installed", "module unavailable")
    try:
        frame_content_size = _zstd.frame_content_size(stored)
    except Exception as exc:
        raise FormatError(path, offset, field + ".frame_header", "one standard zstd frame", f"{type(exc).__name__}: {exc}") from exc
    _require(
        frame_content_size == expected_bytes,
        path,
        offset,
        field + ".declared_content_size",
        expected_bytes,
        frame_content_size,
    )
    try:
        decoded = _zstd.ZstdDecompressor().decompress(
            stored,
            max_output_size=expected_bytes,
            allow_extra_data=False,
        )
    except Exception as exc:
        raise FormatError(path, offset, field, f"one zstd frame decoding to {expected_bytes} bytes", f"{type(exc).__name__}: {exc}") from exc
    _require(len(decoded) == expected_bytes, path, offset, field + ".decoded_size", expected_bytes, len(decoded))
    return decoded


def _parse_block_table(
    reader: FileReader,
    row: IndexRow,
    table_offset: int,
    authority_end: int,
) -> BlockTable:
    header = reader.read_exact(table_offset, BLOCK_HEADER_BYTES, f"index.row[{row.key_text}].block_table", authority_end=authority_end)
    _require(header[:8] == BLOCK_MAGIC, reader.path, table_offset, "block_table.magic", _fmt_bytes(BLOCK_MAGIC), _fmt_bytes(header[:8]))
    got_crc = _u32(header, 60)
    expected_crc = crc32c(header[:60])
    _require(got_crc == expected_crc, reader.path, table_offset + 60, "block_table.header_crc32c", _fmt_u32(expected_crc), _fmt_u32(got_crc))
    _require(_u16(header, 8) == 1, reader.path, table_offset + 8, "block_table.table_version", 1, _u16(header, 8), error_type=UnsupportedFeature)
    _require(_u16(header, 10) == BLOCK_HEADER_BYTES, reader.path, table_offset + 10, "block_table.header_bytes", BLOCK_HEADER_BYTES, _u16(header, 10))
    _require(_u16(header, 12) == 4, reader.path, table_offset + 12, "block_table.entry_bytes", 4, _u16(header, 12))
    _require(_u16(header, 14) == 1, reader.path, table_offset + 14, "block_table.crc_algorithm", 1, _u16(header, 14), error_type=UnsupportedFeature)
    _require(_u32(header, 16) == BLOCK_SIZE, reader.path, table_offset + 16, "block_table.block_size", BLOCK_SIZE, _u32(header, 16))
    _require(_u32(header, 20) == 0, reader.path, table_offset + 20, "block_table.reserved_0", 0, _u32(header, 20))
    _require(_u64(header, 24) == row.payload_offset, reader.path, table_offset + 24, "block_table.payload_offset", f"0x{row.payload_offset:x}", f"0x{_u64(header, 24):x}")
    _require(_u64(header, 32) == row.stored_bytes, reader.path, table_offset + 32, "block_table.stored_bytes", row.stored_bytes, _u64(header, 32))
    expected_count = (row.stored_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE
    count = _u64(header, 40)
    _require(count == expected_count and count >= 1, reader.path, table_offset + 40, "block_table.block_count", expected_count, count)
    _require_zero(header, 52, 60, reader.path, table_offset, "block_table.reserved_1")
    entry_bytes = _checked_end(reader.path, 0, count * 4, "block_table.entries_size")
    entries_data = reader.read_exact(table_offset + BLOCK_HEADER_BYTES, entry_bytes, "block_table.entries", authority_end=authority_end)
    entries_crc = _u32(header, 48)
    expected_entries_crc = crc32c(entries_data)
    _require(entries_crc == expected_entries_crc, reader.path, table_offset + 48, "block_table.entries_crc32c", _fmt_u32(expected_entries_crc), _fmt_u32(entries_crc))
    entries = struct.unpack(f"<{count}I", entries_data)
    return BlockTable(
        offset=table_offset,
        payload_offset=row.payload_offset,
        stored_bytes=row.stored_bytes,
        block_size=BLOCK_SIZE,
        entries=tuple(entries),
        entries_crc32c=entries_crc,
        header_crc32c=got_crc,
    )


def _validate_live_row(
    reader: FileReader,
    commit: Commit,
    row: IndexRow,
) -> IndexRow:
    prefix = f"index.row[{row.key_text}]"
    _require(row.chunk_version >= 1, reader.path, row.row_offset + 4, prefix + ".chunk_version", ">= 1", row.chunk_version)
    _require(row.compression in COMPRESSION_NAMES, reader.path, row.row_offset + 7, prefix + ".compression", "0 or 1", row.compression)
    _require(row.chunk_flags & ~KNOWN_CHUNK_FLAGS == 0, reader.path, row.row_offset + 8, prefix + ".chunk_flags", f"no bits outside 0x{KNOWN_CHUNK_FLAGS:x}", f"0x{row.chunk_flags:x}")
    _require(1 <= row.source_generation <= commit.generation, reader.path, row.row_offset + 64, prefix + ".source_generation", f"[1,{commit.generation}]", row.source_generation)
    _require(row.header_offset >= APPEND_OFFSET and row.header_offset % 64 == 0, reader.path, row.row_offset + 32, prefix + ".header_offset", ">= 0x10000 and 64-byte aligned", f"0x{row.header_offset:x}")
    _require(row.payload_offset >= row.header_offset + CHUNK_HEADER_BYTES, reader.path, row.row_offset + 40, prefix + ".payload_offset", f">= 0x{row.header_offset + CHUNK_HEADER_BYTES:x}", f"0x{row.payload_offset:x}")
    alignment = 4096 if row.chunk_flags & CHUNK_TENSOR else 64
    _require(row.payload_offset % alignment == 0, reader.path, row.row_offset + 40, prefix + ".payload_offset", f"{alignment}-byte aligned", f"0x{row.payload_offset:x}")
    payload_end = _checked_end(reader.path, row.payload_offset, row.stored_bytes, prefix + ".payload_range")
    _require(payload_end <= commit.index_offset, reader.path, row.row_offset + 40, prefix + ".payload_range", f"subset of [0x{APPEND_OFFSET:x},0x{commit.index_offset:x})", f"[0x{row.payload_offset:x},0x{payload_end:x})")
    _require(row.header_offset < commit.index_offset, reader.path, row.row_offset + 32, prefix + ".header_offset", f"< index_offset 0x{commit.index_offset:x}", f"0x{row.header_offset:x}")
    if row.compression == COMPRESSION_STORED:
        _require(row.stored_bytes == row.uncompressed_bytes, reader.path, row.row_offset + 48, prefix + ".stored_uncompressed_sizes", "equal in stored mode", (row.stored_bytes, row.uncompressed_bytes))

    header = reader.read_exact(row.header_offset, CHUNK_HEADER_BYTES, prefix + ".chunk_header", authority_end=commit.committed_file_end)
    got_crc = _u32(header, 60)
    expected_crc = crc32c(header[:60])
    _require(got_crc == expected_crc, reader.path, row.header_offset + 60, prefix + ".chunk_header.crc32c", _fmt_u32(expected_crc), _fmt_u32(got_crc))
    _require(got_crc == row.header_crc32c, reader.path, row.row_offset + 76, prefix + ".header_crc32c_echo", _fmt_u32(got_crc), _fmt_u32(row.header_crc32c))

    echoes: tuple[tuple[int, str, object, object], ...] = (
        (0, "fourcc", row.fourcc, header[0:4]),
        (4, "chunk_version", row.chunk_version, _u16(header, 4)),
        (6, "header_bytes", CHUNK_HEADER_BYTES, _u16(header, 6)),
        (8, "instance_uuid", row.instance_uuid, _uuid(header, 8)),
        (24, "chunk_flags", row.chunk_flags, _u32(header, 24)),
        (28, "compression", row.compression, _u16(header, 28)),
        (32, "stored_bytes", row.stored_bytes, _u64(header, 32)),
        (40, "uncompressed_bytes", row.uncompressed_bytes, _u64(header, 40)),
        (56, "payload_crc32c", row.payload_crc32c, _u32(header, 56)),
    )
    for relative, name, expected, got in echoes:
        if isinstance(expected, bytes):
            expected_display: object = _fmt_bytes(expected)
            got_display: object = _fmt_bytes(got)  # type: ignore[arg-type]
        else:
            expected_display = str(expected) if isinstance(expected, uuid.UUID) else expected
            got_display = str(got) if isinstance(got, uuid.UUID) else got
        _require(got == expected, reader.path, row.header_offset + relative, prefix + ".chunk_header." + name, expected_display, got_display)

    block_kind = _u16(header, 30)
    table_offset = _u64(header, 48)
    has_blocks = bool(row.chunk_flags & CHUNK_BLOCK_CRCS)
    if has_blocks:
        _require(block_kind == 1, reader.path, row.header_offset + 30, prefix + ".chunk_header.block_crc_kind", 1, block_kind)
        expected_table_offset = row.header_offset + CHUNK_HEADER_BYTES
        _require(table_offset == expected_table_offset, reader.path, row.header_offset + 48, prefix + ".chunk_header.block_crc_table_offset", f"0x{expected_table_offset:x}", f"0x{table_offset:x}")
        table = _parse_block_table(reader, row, table_offset, commit.committed_file_end)
        _require(table.end <= row.payload_offset, reader.path, table.offset, prefix + ".block_table.range", f"end <= payload_offset 0x{row.payload_offset:x}", f"[0x{table.offset:x},0x{table.end:x})")
        pad_start = table.end
    else:
        _require(block_kind == 0, reader.path, row.header_offset + 30, prefix + ".chunk_header.block_crc_kind", 0, block_kind)
        _require(table_offset == 0, reader.path, row.header_offset + 48, prefix + ".chunk_header.block_crc_table_offset", 0, f"0x{table_offset:x}")
        _require(row.stored_bytes < BLOCK_REQUIRED_AT, reader.path, row.row_offset + 48, prefix + ".block_crc_requirement", f"stored_bytes < {BLOCK_REQUIRED_AT} without table", row.stored_bytes)
        table = None
        pad_start = row.header_offset + CHUNK_HEADER_BYTES

    if pad_start < row.payload_offset:
        padding = reader.read_exact(pad_start, row.payload_offset - pad_start, prefix + ".payload_alignment_padding", authority_end=commit.committed_file_end)
        _require_zero(padding, 0, len(padding), reader.path, pad_start, prefix + ".payload_alignment_padding")

    return dataclasses.replace(row, block_table=table)


def _parse_index(
    reader: FileReader,
    sb: Superblock,
    commit: Commit,
    lineage: tuple[Commit, ...],
) -> Index:
    stored = reader.read_exact(commit.index_offset, commit.index_stored_bytes, "index.stored", authority_end=commit.committed_file_end)
    got_stored_crc = crc32c(stored)
    _require(got_stored_crc == commit.index_stored_crc32c, reader.path, commit.offset + 160, "commit.index_stored_crc32c", _fmt_u32(got_stored_crc), _fmt_u32(commit.index_stored_crc32c))
    if commit.index_compression == COMPRESSION_STORED:
        _require(commit.index_stored_bytes == commit.index_uncompressed_bytes, reader.path, commit.offset + 144, "commit.index_sizes", "equal in stored mode", (commit.index_stored_bytes, commit.index_uncompressed_bytes))
        decoded = stored
    else:
        decoded = _decode_zstd_exact(reader.path, commit.index_offset, "index.zstd", stored, commit.index_uncompressed_bytes)
    _require(len(decoded) == commit.index_uncompressed_bytes, reader.path, commit.offset + 152, "commit.index_uncompressed_bytes", commit.index_uncompressed_bytes, len(decoded))
    got_decoded_crc = crc32c(decoded)
    _require(got_decoded_crc == commit.index_uncompressed_crc32c, reader.path, commit.offset + 164, "commit.index_uncompressed_crc32c", _fmt_u32(got_decoded_crc), _fmt_u32(commit.index_uncompressed_crc32c))
    _require(decoded[:8] == INDEX_MAGIC, reader.path, commit.index_offset, "index.magic", _fmt_bytes(INDEX_MAGIC), _fmt_bytes(decoded[:8]))
    _require(_u16(decoded, 8) == 1, reader.path, commit.index_offset + 8, "index.index_version", 1, _u16(decoded, 8), error_type=UnsupportedFeature)
    _require(_u16(decoded, 10) == INDEX_HEADER_BYTES, reader.path, commit.index_offset + 10, "index.header_bytes", INDEX_HEADER_BYTES, _u16(decoded, 10))
    _require(_u16(decoded, 12) == INDEX_ROW_BYTES, reader.path, commit.index_offset + 12, "index.row_bytes", INDEX_ROW_BYTES, _u16(decoded, 12))
    _require(_u16(decoded, 14) == 1, reader.path, commit.index_offset + 14, "index.sort_order", 1, _u16(decoded, 14), error_type=UnsupportedFeature)
    row_count = _u64(decoded, 16)
    expected_size = INDEX_HEADER_BYTES + row_count * INDEX_ROW_BYTES
    _require(expected_size <= 0xFFFFFFFFFFFFFFFF, reader.path, commit.index_offset + 16, "index.row_count", "size arithmetic within u64", row_count)
    _require(len(decoded) == expected_size, reader.path, commit.index_offset + 16, "index.decoded_size_from_row_count", expected_size, len(decoded))
    _require(_u64(decoded, 24) == commit.generation, reader.path, commit.index_offset + 24, "index.generation", commit.generation, _u64(decoded, 24))
    index_commit_uuid = _uuid(decoded, 32)
    _require(index_commit_uuid == commit.commit_uuid, reader.path, commit.index_offset + 32, "index.commit_uuid", str(commit.commit_uuid), str(index_commit_uuid))
    index_flags = _u32(decoded, 48)
    _require(index_flags & ~0x3 == 0, reader.path, commit.index_offset + 48, "index.index_flags", "only bits 0 and 1", f"0x{index_flags:x}")
    _require_zero(decoded, 52, 64, reader.path, commit.index_offset, "index.reserved")

    rows: list[IndexRow] = []
    previous_key: bytes | None = None
    has_tombstone = False
    has_base_ref = False
    for row_index in range(row_count):
        rel = INDEX_HEADER_BYTES + row_index * INDEX_ROW_BYTES
        absolute = commit.index_offset + rel
        raw = decoded[rel : rel + INDEX_ROW_BYTES]
        fourcc = raw[0:4]
        _require(_FOURCC_RE.fullmatch(fourcc) is not None, reader.path, absolute, f"index.rows[{row_index}].fourcc", "[A-Z0-9]{4}", repr(fourcc))
        instance_uuid = _uuid(raw, 16)
        _require(instance_uuid != NULL_UUID, reader.path, absolute + 16, f"index.rows[{row_index}].instance_uuid", "non-null UUID", str(instance_uuid))
        _require_zero(raw, 12, 16, reader.path, absolute, f"index.rows[{row_index}].reserved_0")
        _require_zero(raw, 80, 96, reader.path, absolute, f"index.rows[{row_index}].reserved_1")
        row_kind = _u8(raw, 6)
        _require(row_kind in ROW_KIND_NAMES, reader.path, absolute + 6, f"index.rows[{row_index}].row_kind", "0, 1, or 2", row_kind)
        row = IndexRow(
            row_offset=absolute,
            fourcc=fourcc,
            chunk_version=_u16(raw, 4),
            row_kind=row_kind,
            compression=_u8(raw, 7),
            chunk_flags=_u32(raw, 8),
            instance_uuid=instance_uuid,
            header_offset=_u64(raw, 32),
            payload_offset=_u64(raw, 40),
            stored_bytes=_u64(raw, 48),
            uncompressed_bytes=_u64(raw, 56),
            source_generation=_u64(raw, 64),
            payload_crc32c=_u32(raw, 72),
            header_crc32c=_u32(raw, 76),
        )
        if previous_key is not None:
            _require(row.key > previous_key, reader.path, absolute, f"index.rows[{row_index}].canonical_key_order", f"> {_fmt_bytes(previous_key)}", _fmt_bytes(row.key))
        previous_key = row.key

        prefix = f"index.row[{row.key_text}]"
        if row_kind == ROW_LIVE:
            row = _validate_live_row(reader, commit, row)
        elif row_kind == ROW_TOMBSTONE:
            has_tombstone = True
            sentinel = (
                row.chunk_version,
                row.compression,
                row.chunk_flags,
                row.header_offset,
                row.payload_offset,
                row.stored_bytes,
                row.uncompressed_bytes,
                row.payload_crc32c,
                row.header_crc32c,
            )
            _require(sentinel == (0, 0, 0, 0, 0, 0, 0, 0, 0), reader.path, absolute + 4, prefix + ".tombstone_encoding", "all sentinel fields zero", sentinel)
            _require(1 <= row.source_generation <= commit.generation, reader.path, absolute + 64, prefix + ".source_generation", f"[1,{commit.generation}]", row.source_generation)
        else:
            has_base_ref = True
            _require(sb.role == ROLE_SIDECAR, reader.path, absolute + 6, prefix + ".row_kind", "base_ref only in sidecar", "base_ref in master")
            _require(row.chunk_version >= 1, reader.path, absolute + 4, prefix + ".chunk_version", ">= 1", row.chunk_version)
            _require(row.compression in COMPRESSION_NAMES, reader.path, absolute + 7, prefix + ".compression", "0 or 1", row.compression)
            _require(row.chunk_flags & ~KNOWN_CHUNK_FLAGS == 0, reader.path, absolute + 8, prefix + ".chunk_flags", f"no bits outside 0x{KNOWN_CHUNK_FLAGS:x}", f"0x{row.chunk_flags:x}")
            _require(row.header_offset == 0 and row.payload_offset == 0, reader.path, absolute + 32, prefix + ".base_ref_offsets", "both zero", (row.header_offset, row.payload_offset))
            _require(row.source_generation >= 1, reader.path, absolute + 64, prefix + ".source_generation", ">= 1", row.source_generation)
            if row.compression == COMPRESSION_STORED:
                _require(row.stored_bytes == row.uncompressed_bytes, reader.path, absolute + 48, prefix + ".stored_uncompressed_sizes", "equal in stored mode", (row.stored_bytes, row.uncompressed_bytes))
        rows.append(row)

    derived_flags = (1 if has_tombstone else 0) | (2 if has_base_ref else 0)
    _require(index_flags == derived_flags, reader.path, commit.index_offset + 48, "index.index_flags", f"derived value 0x{derived_flags:x}", f"0x{index_flags:x}")

    spans = [
        (row.occupied_span[0], row.occupied_span[1], row)
        for row in rows
        if row.occupied_span is not None
    ]
    spans.sort(key=lambda span: (span[0], span[1], span[2].key))
    for (_, previous_end, previous), (start, end, current) in zip(spans, spans[1:]):
        _require(start >= previous_end, reader.path, current.row_offset + 32, f"index.row[{current.key_text}].occupied_span", f"start >= previous row end 0x{previous_end:x} ({previous.key_text})", f"[0x{start:x},0x{end:x})")

    lineage_by_generation = {ancestor.generation: ancestor for ancestor in lineage}
    for row in rows:
        if row.row_kind != ROW_LIVE:
            continue
        source = lineage_by_generation.get(row.source_generation)
        _require(source is not None, reader.path, row.row_offset + 64, f"index.row[{row.key_text}].source_generation", f"generation present in selected lineage {sorted(lineage_by_generation)}", row.source_generation)
        assert source is not None
        source_start = (
            APPEND_OFFSET
            if row.source_generation == 1
            else lineage_by_generation[row.source_generation - 1].committed_file_end
        )
        span = row.occupied_span
        assert span is not None
        _require(
            span[0] >= source_start and span[1] <= source.index_offset,
            reader.path,
            row.row_offset + 32,
            f"index.row[{row.key_text}].source_generation_span",
            f"subset of generation {row.source_generation} payload region [0x{source_start:x},0x{source.index_offset:x})",
            f"[0x{span[0]:x},0x{span[1]:x})",
        )

    required_caps = 0
    if commit.index_compression == COMPRESSION_ZSTD:
        required_caps |= CAP_INDEX_ZSTD
    if any(row.compression == COMPRESSION_ZSTD for row in rows if row.row_kind != ROW_TOMBSTONE):
        required_caps |= CAP_CHUNK_ZSTD
    if any(row.block_table is not None for row in rows):
        required_caps |= CAP_BLOCK_CRC
    if has_tombstone:
        required_caps |= CAP_TOMBSTONES
    if sb.role == ROLE_SIDECAR or has_base_ref:
        required_caps |= CAP_SIDECAR
    missing_declared = required_caps & ~commit.reader_capabilities
    _require(missing_declared == 0, reader.path, commit.offset + 192, "commit.required_reader_capabilities", f"include feature bits 0x{required_caps:x}", f"0x{commit.reader_capabilities:x}")

    return Index(
        stored_bytes=stored,
        uncompressed_bytes=decoded,
        generation=commit.generation,
        commit_uuid=commit.commit_uuid,
        flags=index_flags,
        rows=tuple(rows),
    )


def _parse_head(reader: FileReader, sb: Superblock, slot_id: int, raw: bytes) -> Head:
    offset = HEAD_OFFSETS[slot_id]
    _require(raw[:8] == HEAD_MAGIC, reader.path, offset, f"head[{slot_id}].magic", _fmt_bytes(HEAD_MAGIC), _fmt_bytes(raw[:8]))
    got_head_crc = _u32(raw, 4092)
    expected_head_crc = crc32c(raw[:4092])
    _require(got_head_crc == expected_head_crc, reader.path, offset + 4092, f"head[{slot_id}].crc32c", _fmt_u32(expected_head_crc), _fmt_u32(got_head_crc))
    _require(_u32(raw, 8) == slot_id, reader.path, offset + 8, f"head[{slot_id}].slot_id", slot_id, _u32(raw, 8))
    _require(_u32(raw, 12) == HEAD_BYTES, reader.path, offset + 12, f"head[{slot_id}].head_bytes", HEAD_BYTES, _u32(raw, 12))
    sequence = _u64(raw, 16)
    generation = _u64(raw, 24)
    project_uuid = _uuid(raw, 32)
    file_uuid = _uuid(raw, 48)
    commit_uuid = _uuid(raw, 64)
    commit_offset = _u64(raw, 80)
    commit_bytes = _u64(raw, 88)
    committed_end = _u64(raw, 96)
    commit_crc_echo = _u32(raw, 104)
    _require(sequence >= 1, reader.path, offset + 16, f"head[{slot_id}].head_sequence", ">= 1", sequence)
    _require(generation >= 1, reader.path, offset + 24, f"head[{slot_id}].generation", ">= 1", generation)
    _require(project_uuid == sb.project_uuid, reader.path, offset + 32, f"head[{slot_id}].project_uuid", str(sb.project_uuid), str(project_uuid))
    _require(file_uuid == sb.file_uuid, reader.path, offset + 48, f"head[{slot_id}].file_uuid", str(sb.file_uuid), str(file_uuid))
    _require(commit_uuid != NULL_UUID, reader.path, offset + 64, f"head[{slot_id}].commit_uuid", "non-null UUID", str(commit_uuid))
    _require(commit_offset >= APPEND_OFFSET and commit_offset % 64 == 0, reader.path, offset + 80, f"head[{slot_id}].commit_offset", ">= 0x10000 and 64-byte aligned", f"0x{commit_offset:x}")
    _require(commit_bytes == COMMIT_BYTES, reader.path, offset + 88, f"head[{slot_id}].commit_bytes", COMMIT_BYTES, commit_bytes)
    _require(committed_end == commit_offset + COMMIT_BYTES, reader.path, offset + 96, f"head[{slot_id}].committed_file_end", f"0x{commit_offset + COMMIT_BYTES:x}", f"0x{committed_end:x}")
    _require(committed_end <= reader.size, reader.path, offset + 96, f"head[{slot_id}].committed_file_end", f"<= physical_size 0x{reader.size:x}", f"0x{committed_end:x}")
    _require(_u32(raw, 108) == 0, reader.path, offset + 108, f"head[{slot_id}].head_flags", 0, _u32(raw, 108))
    _require_zero(raw, 112, 4092, reader.path, offset, f"head[{slot_id}].reserved")

    commit = _parse_commit_record(reader, sb, commit_offset, committed_end)
    _require(commit.commit_uuid == commit_uuid, reader.path, offset + 64, f"head[{slot_id}].commit_uuid", str(commit.commit_uuid), str(commit_uuid))
    _require(commit.generation == generation, reader.path, offset + 24, f"head[{slot_id}].generation", commit.generation, generation)
    _require(commit.committed_file_end == committed_end, reader.path, offset + 96, f"head[{slot_id}].committed_file_end_echo", f"0x{commit.committed_file_end:x}", f"0x{committed_end:x}")
    _require(commit.crc32c == commit_crc_echo, reader.path, offset + 104, f"head[{slot_id}].commit_crc32c_echo", _fmt_u32(commit.crc32c), _fmt_u32(commit_crc_echo))
    lineage = _validate_lineage(reader, sb, commit)
    index = _parse_index(reader, sb, commit, lineage)
    return Head(
        slot_id=slot_id,
        offset=offset,
        head_sequence=sequence,
        generation=generation,
        project_uuid=project_uuid,
        file_uuid=file_uuid,
        commit_uuid=commit_uuid,
        commit_offset=commit_offset,
        commit_bytes=commit_bytes,
        committed_file_end=committed_end,
        commit_crc32c_echo=commit_crc_echo,
        head_crc32c=got_head_crc,
        commit=commit,
        index=index,
    )


def _reader_compatibility_error(path: str, head: Head) -> UnsupportedFeature | None:
    commit = head.commit
    if commit.min_reader_version > CURRENT_VERSION:
        return UnsupportedFeature(
            path,
            commit.offset + 184,
            "commit.min_reader_version",
            f"<= {CURRENT_VERSION[0]}.{CURRENT_VERSION[1]}",
            f"{commit.min_reader_version[0]}.{commit.min_reader_version[1]}",
            code="unsupported_newer",
        )
    unsupported_reader_caps = commit.reader_capabilities & ~SUPPORTED_READER_CAPS
    if unsupported_reader_caps:
        return UnsupportedFeature(
            path,
            commit.offset + 192,
            "commit.required_reader_capabilities",
            f"subset of 0x{SUPPORTED_READER_CAPS:032x}",
            f"unsupported bits 0x{unsupported_reader_caps:032x}",
            code="unsupported_newer",
        )
    return None


def open_container(path: str | os.PathLike[str]) -> Container:
    """Open and fully validate all metadata required by the reader state machine."""

    with FileReader(path) as reader:
        try:
            sb = _parse_superblock(reader)
        except FormatError as error:
            raise _raise_like(error, HardFailure) from error

        attempts: list[HeadAttempt] = []
        for slot_id, offset in enumerate(HEAD_OFFSETS):
            try:
                raw = reader.read_exact(offset, HEAD_BYTES, f"head[{slot_id}]")
            except FormatError as error:
                attempts.append(HeadAttempt(slot_id, offset, False, None, None, error))
                continue
            if raw == b"\x00" * HEAD_BYTES:
                attempts.append(HeadAttempt(slot_id, offset, True, None, None, None))
                continue
            sequence_hint = _u64(raw, 16)
            try:
                head = _parse_head(reader, sb, slot_id, raw)
                attempts.append(
                    HeadAttempt(
                        slot_id,
                        offset,
                        False,
                        sequence_hint,
                        head,
                        None,
                        _reader_compatibility_error(reader.path, head),
                    )
                )
            except FormatError as error:
                attempts.append(HeadAttempt(slot_id, offset, False, sequence_hint, None, error))

        structural_heads = [
            attempt.head for attempt in attempts if attempt.head is not None
        ]
        if not structural_heads:
            failures = "; ".join(
                "blank" if attempt.blank else str(attempt.error)
                for attempt in attempts
            )
            raise RepairRequired(
                reader.path,
                HEAD_OFFSETS[0],
                "heads.authority",
                "at least one fully valid head",
                failures,
                code="repair_only",
            )

        warnings: list[str] = []
        if len(structural_heads) == 2:
            first, second = structural_heads
            assert first is not None and second is not None
            older, newer = sorted(
                (first, second), key=lambda head: head.generation
            )
            if (
                newer.generation == older.generation + 1
                and newer.commit.parent_commit_uuid == older.commit_uuid
                and newer.commit.parent_commit_offset == older.commit_offset
                and newer.head_sequence <= older.head_sequence
            ):
                raise HardFailure(
                    reader.path,
                    newer.offset + 16,
                    "heads.head_sequence_regression",
                    f"> parent head_sequence {older.head_sequence} for direct child generation",
                    newer.head_sequence,
                    code="head_sequence_regression",
                )
            if first.head_sequence == second.head_sequence:
                if first.commit_uuid != second.commit_uuid:
                    raise HardFailure(reader.path, HEAD_OFFSETS[0] + 16, "heads.equal_sequence", "same commit_uuid for duplicate-slot write", f"slot A {first.commit_uuid}, slot B {second.commit_uuid}", code="split_brain")
                if first.commit_crc32c_echo != second.commit_crc32c_echo:
                    raise HardFailure(reader.path, HEAD_OFFSETS[0] + 104, "heads.equal_sequence_commit_crc", _fmt_u32(first.commit_crc32c_echo), _fmt_u32(second.commit_crc32c_echo))
                if first.authority_tuple != second.authority_tuple:
                    raise HardFailure(reader.path, HEAD_OFFSETS[0] + 24, "heads.equal_sequence_authority_tuple", first.authority_tuple, second.authority_tuple)
                selected = first
                warnings.append(
                    f"duplicate-slot write accepted: slots A/B both publish sequence {first.head_sequence}, commit {first.commit_uuid}"
                )
            else:
                selected = max(
                    structural_heads, key=lambda head: head.head_sequence
                )
        else:
            selected = structural_heads[0]

        selected_attempt = next(
            attempt for attempt in attempts if attempt.head is selected
        )

        for attempt in attempts:
            if attempt.head is None and not attempt.blank:
                if isinstance(attempt.error, UnsupportedFeature) and (
                    attempt.sequence_hint is None or attempt.sequence_hint >= selected.head_sequence
                ):
                    raise HardFailure(
                        reader.path,
                        attempt.error.offset,
                        "heads.newer_unsupported_head",
                        "support newer authoritative head rather than roll back",
                        str(attempt.error),
                        code="unsupported",
                    ) from attempt.error

        supported_attempts = [
            attempt
            for attempt in attempts
            if attempt.head is not None and attempt.compatibility_error is None
        ]
        if selected_attempt.compatibility_error is not None and supported_attempts:
            compatibility_error = selected_attempt.compatibility_error
            raise HardFailure(
                reader.path,
                compatibility_error.offset,
                "heads.newer_unsupported_head",
                "support the greater-sequence authority rather than roll back to an older supported generation",
                str(compatibility_error),
                code="unsupported",
            ) from compatibility_error

        for attempt in attempts:
            if attempt.head is None and not attempt.blank:
                warnings.append(
                    f"recovery warning: rejected nonblank head slot {attempt.slot_id} at 0x{attempt.offset:x}: {attempt.error}"
                )
            elif (
                attempt.head is not None
                and attempt.compatibility_error is not None
                and attempt is not selected_attempt
            ):
                warnings.append(
                    f"compatibility warning: non-selected head slot {attempt.slot_id} "
                    f"requires a newer reader: {attempt.compatibility_error}"
                )

        if selected_attempt.compatibility_error is not None:
            inspection = Container(
                path=reader.path,
                physical_size=reader.size,
                superblock=sb,
                head_attempts=(attempts[0], attempts[1]),
                selected_head=selected,
                warnings=tuple(warnings),
                open_state="unsupported_newer",
            )
            compatibility_error = selected_attempt.compatibility_error
            raise UnsupportedNewer(
                reader.path,
                compatibility_error.offset,
                compatibility_error.field,
                compatibility_error.expected,
                compatibility_error.got,
                inspection=inspection,
            ) from compatibility_error

        return Container(
            path=reader.path,
            physical_size=reader.size,
            superblock=sb,
            head_attempts=(attempts[0], attempts[1]),
            selected_head=selected,
            warnings=tuple(warnings),
        )


def classify_open(path: str | os.PathLike[str]) -> tuple[str, str]:
    """Return the spec-oracle outcome vocabulary for one file."""

    try:
        container = open_container(path)
    except UnsupportedNewer as error:
        return ("unsupported_newer", str(error))
    except RepairRequired as error:
        return ("repair_only", str(error))
    except FormatError as error:
        return ("hard_fail", str(error))
    details = list(container.warnings)
    write_compatibility = container.write_compatibility
    if not write_compatibility.safe:
        details.append(
            "write_unsafe: " + "; ".join(write_compatibility.reasons)
        )
    return (f"open_gen_{container.commit.generation}", "; ".join(details))


def _iter_payload_blocks(reader: FileReader, row: IndexRow, authority_end: int) -> Iterator[tuple[int, int, bytes]]:
    remaining = row.stored_bytes
    offset = row.payload_offset
    block_index = 0
    while remaining:
        size = min(BLOCK_SIZE, remaining)
        data = reader.read_exact(offset, size, f"payload[{row.key_text}].block[{block_index}]", authority_end=authority_end)
        yield block_index, offset, data
        offset += size
        remaining -= size
        block_index += 1
    if row.stored_bytes == 0:
        return


def verify_container(container: Container) -> list[str]:
    """Perform the full selected-generation payload and per-block CRC sweep."""

    reports: list[str] = []
    with FileReader(container.path) as reader:
        for row in container.index.rows:
            if row.row_kind != ROW_LIVE:
                continue
            running_crc = 0
            block_count = 0
            for block_index, absolute, data in _iter_payload_blocks(reader, row, container.commit.committed_file_end):
                block_crc = crc32c(data)
                if row.block_table is not None:
                    expected = row.block_table.entries[block_index]
                    _require(block_crc == expected, reader.path, absolute, f"payload[{row.key_text}].block[{block_index}].crc32c", _fmt_u32(expected), _fmt_u32(block_crc))
                running_crc = crc32c(data, running_crc)
                block_count += 1
            _require(running_crc == row.payload_crc32c, reader.path, row.payload_offset, f"payload[{row.key_text}].crc32c", _fmt_u32(row.payload_crc32c), _fmt_u32(running_crc))
            if row.block_table is not None:
                _require(block_count == len(row.block_table.entries), reader.path, row.block_table.offset + 40, f"payload[{row.key_text}].block_count", len(row.block_table.entries), block_count)
            if row.compression == COMPRESSION_ZSTD:
                decoded_size = _measure_zstd_payload(reader, row, container.commit.committed_file_end)
                _require(decoded_size == row.uncompressed_bytes, reader.path, row.payload_offset, f"payload[{row.key_text}].decoded_size", row.uncompressed_bytes, decoded_size)
            reports.append(f"{row.key_text}: stored_bytes={row.stored_bytes} crc32c={_fmt_u32(running_crc)}")
    return reports


class _BoundedPread(io.RawIOBase):
    def __init__(self, reader: FileReader, offset: int, size: int, authority_end: int, field: str) -> None:
        super().__init__()
        self._reader = reader
        self._start = offset
        self._size = size
        self._position = 0
        self._authority_end = authority_end
        self._field = field

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError("invalid whence")
        if position < 0 or position > self._size:
            raise ValueError("seek outside bounded payload")
        self._position = position
        return position

    def readinto(self, target: bytearray | memoryview) -> int:
        if self._position >= self._size:
            return 0
        count = min(len(target), self._size - self._position)
        data = self._reader.read_exact(
            self._start + self._position,
            count,
            self._field,
            authority_end=self._authority_end,
        )
        target[:count] = data
        self._position += count
        return count


def _measure_zstd_payload(reader: FileReader, row: IndexRow, authority_end: int) -> int:
    if _zstd is None:
        raise UnsupportedFeature(reader.path, row.payload_offset, f"payload[{row.key_text}].zstd", "Python package 'zstandard' installed", "module unavailable")
    source = io.BufferedReader(_BoundedPread(reader, row.payload_offset, row.stored_bytes, authority_end, f"payload[{row.key_text}].stored"))
    total = 0
    try:
        with _zstd.ZstdDecompressor().stream_reader(source, read_across_frames=False) as stream:
            while True:
                data = stream.read(1024 * 1024)
                if not data:
                    break
                total += len(data)
                if total > row.uncompressed_bytes:
                    break
    except Exception as exc:
        raise FormatError(reader.path, row.payload_offset, f"payload[{row.key_text}].zstd", f"one frame decoding to {row.uncompressed_bytes} bytes", f"{type(exc).__name__}: {exc}") from exc
    return total


def extract_payload(
    container: Container,
    row: IndexRow,
    output: BinaryIO,
    *,
    stored: bool = False,
) -> int:
    """Verify, then stream one payload to output. Returns bytes written."""

    # Fail before producing a partial extraction.
    verify_row_container = dataclasses.replace(
        container,
        selected_head=dataclasses.replace(
            container.selected_head,
            index=dataclasses.replace(container.index, rows=(row,)),
        ),
    )
    verify_container(verify_row_container)

    with FileReader(container.path) as reader:
        if stored or row.compression == COMPRESSION_STORED:
            written = 0
            for _, _, data in _iter_payload_blocks(reader, row, container.commit.committed_file_end):
                output.write(data)
                written += len(data)
            return written
        if _zstd is None:
            raise UnsupportedFeature(reader.path, row.payload_offset, f"payload[{row.key_text}].zstd", "Python package 'zstandard' installed", "module unavailable")
        source = io.BufferedReader(_BoundedPread(reader, row.payload_offset, row.stored_bytes, container.commit.committed_file_end, f"payload[{row.key_text}].stored"))
        written = 0
        with _zstd.ZstdDecompressor().stream_reader(source, read_across_frames=False) as stream:
            while True:
                data = stream.read(1024 * 1024)
                if not data:
                    break
                output.write(data)
                written += len(data)
        _require(written == row.uncompressed_bytes, reader.path, row.payload_offset, f"payload[{row.key_text}].decoded_size", row.uncompressed_bytes, written)
        return written


@dataclasses.dataclass(frozen=True)
class RecoveryCandidate:
    path: str
    status: str
    detail: str
    autosave_sequence: int | None = None
    file_uuid: uuid.UUID | None = None
    commit_uuid: uuid.UUID | None = None


@dataclasses.dataclass(frozen=True)
class RecoveryDecision:
    outcome: str
    selected_path: str | None
    candidates: tuple[RecoveryCandidate, ...]
    detail: str


def _base_ref_echo(row: IndexRow) -> tuple[object, ...]:
    return (
        row.chunk_version,
        row.compression,
        row.chunk_flags,
        row.stored_bytes,
        row.uncompressed_bytes,
        row.source_generation,
        row.payload_crc32c,
        row.header_crc32c,
    )


def _validate_sidecar_complete(master: Container, sidecar: Container) -> None:
    base_live = {row.key: row for row in master.index.rows if row.row_kind == ROW_LIVE}
    side_rows = {row.key: row for row in sidecar.index.rows}
    missing = sorted(set(base_live) - set(side_rows))
    if missing:
        key = missing[0]
        raise FormatError(sidecar.path, sidecar.commit.index_offset, "sidecar.complete_overlay", f"row for every base live key ({len(base_live)})", f"missing {_fmt_bytes(key)}")
    for key, row in side_rows.items():
        base = base_live.get(key)
        if row.row_kind == ROW_BASE_REF:
            _require(base is not None, sidecar.path, row.row_offset, f"sidecar.base_ref[{row.key_text}]", "existing live key in base", "absent")
            assert base is not None
            _require(_base_ref_echo(row) == _base_ref_echo(base), sidecar.path, row.row_offset + 4, f"sidecar.base_ref[{row.key_text}].metadata_echo", _base_ref_echo(base), _base_ref_echo(row))
        elif row.row_kind == ROW_TOMBSTONE:
            _require(base is not None, sidecar.path, row.row_offset, f"sidecar.tombstone[{row.key_text}]", "key present in base", "absent")
        elif base is None:
            _require(row.row_kind == ROW_LIVE, sidecar.path, row.row_offset + 6, f"sidecar.new_key[{row.key_text}]", "live row", row.kind_name)


def evaluate_recovery(
    master_path: str | os.PathLike[str],
    candidate_paths: Sequence[str | os.PathLike[str]],
) -> RecoveryDecision:
    """Apply the exact sidecar recovery predicate and candidate ordering."""

    master = open_container(master_path)
    _require(master.superblock.role == ROLE_MASTER, master.path, 20, "recovery.master.container_role", ROLE_MASTER, master.superblock.role, error_type=HardFailure)
    reports: list[RecoveryCandidate] = []
    valid: list[tuple[Container, str]] = []
    for candidate_path in candidate_paths:
        path = str(candidate_path)
        try:
            sidecar = open_container(path)
            _require(sidecar.superblock.role == ROLE_SIDECAR, path, 20, "recovery.sidecar.container_role", ROLE_SIDECAR, sidecar.superblock.role)
            if sidecar.superblock.project_uuid != master.superblock.project_uuid:
                reports.append(RecoveryCandidate(path, "wrong_project", f"project UUID {sidecar.superblock.project_uuid} != {master.superblock.project_uuid}", sidecar.superblock.autosave_sequence, sidecar.superblock.file_uuid, sidecar.commit.commit_uuid))
                continue
            if sidecar.superblock.base_explicit_commit_uuid != master.commit.commit_uuid:
                reports.append(RecoveryCandidate(path, "stale_delete", f"base {sidecar.superblock.base_explicit_commit_uuid} != current master head {master.commit.commit_uuid}", sidecar.superblock.autosave_sequence, sidecar.superblock.file_uuid, sidecar.commit.commit_uuid))
                continue
            _validate_sidecar_complete(master, sidecar)
            reports.append(RecoveryCandidate(path, "valid", "predicate satisfied", sidecar.superblock.autosave_sequence, sidecar.superblock.file_uuid, sidecar.commit.commit_uuid))
            valid.append((sidecar, path))
        except FormatError as error:
            reports.append(RecoveryCandidate(path, "invalid", str(error)))

    if not valid:
        return RecoveryDecision("none", None, tuple(reports), "no candidate satisfies the recovery predicate")
    highest = max(container.superblock.autosave_sequence for container, _ in valid)
    top = [(container, path) for container, path in valid if container.superblock.autosave_sequence == highest]
    identities = {(container.superblock.file_uuid, container.commit.commit_uuid) for container, _ in top}
    if len(identities) != 1:
        return RecoveryDecision("ambiguous", None, tuple(reports), f"multiple distinct valid candidates tie at autosave_sequence {highest}")
    selected_path = sorted(path for _, path in top)[0]
    return RecoveryDecision("offer", selected_path, tuple(reports), f"unique highest autosave_sequence {highest}")


def _uuid_text(value: uuid.UUID) -> str:
    return str(value)


def _row_to_dict(row: IndexRow) -> dict[str, object]:
    result: dict[str, object] = {
        "fourcc": row.fourcc.decode("ascii"),
        "instance_uuid": _uuid_text(row.instance_uuid),
        "row_kind": row.kind_name,
        "chunk_version": row.chunk_version,
        "flags": f"0x{row.chunk_flags:08x}",
        "compression": COMPRESSION_NAMES.get(row.compression, row.compression),
        "header_offset": row.header_offset,
        "payload_offset": row.payload_offset,
        "stored_bytes": row.stored_bytes,
        "uncompressed_bytes": row.uncompressed_bytes,
        "source_generation": row.source_generation,
        "payload_crc32c": _fmt_u32(row.payload_crc32c),
        "header_crc32c": _fmt_u32(row.header_crc32c),
    }
    if row.block_table is not None:
        result["block_table"] = {
            "offset": row.block_table.offset,
            "block_size": row.block_table.block_size,
            "block_count": len(row.block_table.entries),
            "entries_crc32c": _fmt_u32(row.block_table.entries_crc32c),
            "header_crc32c": _fmt_u32(row.block_table.header_crc32c),
        }
    return result


def container_to_dict(container: Container) -> dict[str, object]:
    sb = container.superblock
    commit = container.commit
    write_compatibility = container.write_compatibility
    attempts: list[dict[str, object]] = []
    for attempt in container.head_attempts:
        if attempt.blank:
            attempts.append({"slot_id": attempt.slot_id, "status": "blank", "offset": attempt.offset})
        elif attempt.head is None:
            assert attempt.error is not None
            attempts.append({"slot_id": attempt.slot_id, "status": "invalid", "offset": attempt.offset, "error": str(attempt.error), "sequence_hint": attempt.sequence_hint})
        else:
            head_state: dict[str, object] = {
                "slot_id": attempt.slot_id,
                "status": (
                    "unsupported_newer"
                    if attempt.compatibility_error is not None
                    else "valid"
                ),
                "offset": attempt.offset,
                "head_sequence": attempt.head.head_sequence,
                "generation": attempt.head.generation,
                "commit_uuid": str(attempt.head.commit_uuid),
                "commit_offset": attempt.head.commit_offset,
                "committed_file_end": attempt.head.committed_file_end,
                "head_crc32c": _fmt_u32(attempt.head.head_crc32c),
            }
            if attempt.compatibility_error is not None:
                head_state["compatibility_error"] = str(
                    attempt.compatibility_error
                )
            attempts.append(head_state)
    return {
        "path": container.path,
        "physical_size": container.physical_size,
        "open_state": container.open_state,
        "superblock": {
            "format": f"{sb.format_major}.{sb.format_minor}",
            "role": sb.role_name,
            "project_uuid": str(sb.project_uuid),
            "file_uuid": str(sb.file_uuid),
            "creation_time_unix_ns": sb.creation_time_ns,
            "base_explicit_commit_uuid": str(sb.base_explicit_commit_uuid),
            "autosave_sequence": sb.autosave_sequence,
            "sidecar_snapshot_uuid": str(sb.sidecar_snapshot_uuid),
            "crc32c": _fmt_u32(sb.crc32c),
        },
        "heads": attempts,
        "selected": {
            "slot_id": container.selected_head.slot_id,
            "head_sequence": container.selected_head.head_sequence,
            "generation": commit.generation,
            "commit_kind": commit.kind_name,
            "commit_uuid": str(commit.commit_uuid),
            "parent_commit_uuid": str(commit.parent_commit_uuid),
            "explicit_ancestor_commit_uuid": str(commit.explicit_ancestor_commit_uuid),
            "snapshot_uuid": str(commit.snapshot_uuid),
            "commit_offset": commit.offset,
            "committed_file_end": commit.committed_file_end,
            "index_offset": commit.index_offset,
            "index_stored_bytes": commit.index_stored_bytes,
            "index_uncompressed_bytes": commit.index_uncompressed_bytes,
            "index_compression": COMPRESSION_NAMES[commit.index_compression],
            "min_reader_version": f"{commit.min_reader_version[0]}.{commit.min_reader_version[1]}",
            "min_safe_writer_version": f"{commit.min_safe_writer_version[0]}.{commit.min_safe_writer_version[1]}",
            "required_reader_capabilities": f"0x{commit.reader_capabilities:032x}",
            "required_writer_capabilities": f"0x{commit.writer_capabilities:032x}",
            "commit_crc32c": _fmt_u32(commit.crc32c),
        },
        "index": {
            "flags": f"0x{container.index.flags:08x}",
            "row_count": len(container.index.rows),
            "rows": [_row_to_dict(row) for row in container.index.rows],
        },
        "write_compatibility": {
            "declared_writer_version": (
                f"{write_compatibility.writer_version[0]}."
                f"{write_compatibility.writer_version[1]}"
            ),
            "declared_writer_capabilities": (
                f"0x{write_compatibility.writer_capabilities:032x}"
            ),
            "safe": write_compatibility.safe,
            "reasons": list(write_compatibility.reasons),
        },
        "warnings": list(container.warnings),
    }


def _print_human(container: Container) -> None:
    data = container_to_dict(container)
    sb = data["superblock"]
    selected = data["selected"]
    assert isinstance(sb, dict) and isinstance(selected, dict)
    print(f"file: {container.path}")
    print(f"physical_size: {container.physical_size}")
    print(f"open_state: {container.open_state}")
    print(f"superblock: format={sb['format']} role={sb['role']} project={sb['project_uuid']} file={sb['file_uuid']}")
    for attempt in data["heads"]:  # type: ignore[index]
        assert isinstance(attempt, dict)
        if attempt["status"] in ("valid", "unsupported_newer"):
            print(f"head {attempt['slot_id']}: {attempt['status']} sequence={attempt['head_sequence']} generation={attempt['generation']} commit={attempt['commit_uuid']} end=0x{attempt['committed_file_end']:x}")
            if "compatibility_error" in attempt:
                print(f"  compatibility: {attempt['compatibility_error']}")
        elif attempt["status"] == "blank":
            print(f"head {attempt['slot_id']}: blank")
        else:
            print(f"head {attempt['slot_id']}: invalid: {attempt['error']}")
    print(f"selected: slot={selected['slot_id']} sequence={selected['head_sequence']} generation={selected['generation']} kind={selected['commit_kind']} commit={selected['commit_uuid']}")
    print(f"index: offset=0x{selected['index_offset']:x} rows={len(container.index.rows)} compression={selected['index_compression']}")
    for row in container.index.rows:
        print(f"  {row.key_text} kind={row.kind_name} version={row.chunk_version} header=0x{row.header_offset:x} payload=0x{row.payload_offset:x} stored={row.stored_bytes} crc={_fmt_u32(row.payload_crc32c)}")
    compatibility = data["write_compatibility"]
    assert isinstance(compatibility, dict)
    print(
        "write_compatibility: "
        + ("safe" if compatibility["safe"] else "refused")
    )
    for reason in compatibility["reasons"]:  # type: ignore[index]
        print(f"  reason: {reason}")
    for warning in container.warnings:
        print(f"warning: {warning}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="dump validated superblock, heads, commit, and index")
    inspect_parser.add_argument("file")
    inspect_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    verify_parser = subparsers.add_parser("verify", help="perform a full payload and per-block CRC sweep")
    verify_parser.add_argument("file")
    verify_parser.add_argument("--json", action="store_true")

    extract_parser = subparsers.add_parser("extract", help="extract one live chunk by fourcc and UUID")
    extract_parser.add_argument("file")
    extract_parser.add_argument("--fourcc", required=True)
    extract_parser.add_argument("--uuid", required=True, dest="instance_uuid")
    extract_parser.add_argument("--output", "-o", required=True)
    extract_parser.add_argument("--stored", action="store_true", help="write stored bytes instead of decoded payload")
    extract_parser.add_argument("--force", action="store_true", help="replace an existing output file")

    recovery_parser = subparsers.add_parser("recovery", help="evaluate sidecar candidates against the current master head")
    recovery_parser.add_argument("master")
    recovery_parser.add_argument("sidecars", nargs="+")
    recovery_parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            try:
                container = open_container(args.file)
                terminal_error: UnsupportedNewer | None = None
            except UnsupportedNewer as error:
                container = error.inspection
                terminal_error = error
            if args.json:
                payload = container_to_dict(container)
                if terminal_error is not None:
                    payload["terminal_error"] = str(terminal_error)
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_human(container)
                if terminal_error is not None:
                    print(f"terminal: UNSUPPORTED-NEWER: {terminal_error}")
            return 4 if terminal_error is not None else 0
        if args.command == "verify":
            container = open_container(args.file)
            reports = verify_container(container)
            if args.json:
                print(json.dumps({"outcome": "pass", "generation": container.commit.generation, "payloads": reports, "warnings": list(container.warnings)}, indent=2, sort_keys=True))
            else:
                print(f"PASS generation {container.commit.generation}: {len(reports)} live payload(s) verified")
                for report in reports:
                    print(f"  {report}")
                for warning in container.warnings:
                    print(f"warning: {warning}")
            return 0
        if args.command == "extract":
            fourcc_bytes = args.fourcc.encode("ascii", "strict")
            if _FOURCC_RE.fullmatch(fourcc_bytes) is None:
                raise ValueError("--fourcc must match [A-Z0-9]{4}")
            instance_uuid = uuid.UUID(args.instance_uuid)
            container = open_container(args.file)
            row = container.find_row(args.fourcc, instance_uuid)
            output_path = Path(args.output)
            mode = "wb" if args.force else "xb"
            with output_path.open(mode) as output:
                written = extract_payload(container, row, output, stored=args.stored)
            print(f"extracted {written} byte(s) from {row.key_text} to {output_path}")
            return 0
        if args.command == "recovery":
            decision = evaluate_recovery(args.master, args.sidecars)
            payload = dataclasses.asdict(decision)
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True, default=str))
            else:
                print(f"recovery: {decision.outcome}: {decision.detail}")
                for candidate in decision.candidates:
                    print(f"  {candidate.path}: {candidate.status}: {candidate.detail}")
            return 0 if decision.outcome in ("offer", "none") else 2
        raise AssertionError(f"unhandled command {args.command}")
    except UnsupportedNewer as error:
        print(f"UNSUPPORTED-NEWER: {error}", file=sys.stderr)
        return 4
    except RepairRequired as error:
        print(f"REPAIR-ONLY: {error}", file=sys.stderr)
        return 3
    except (FormatError, KeyError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
