"""Minimal byte-table verifier derived from docs/licht_format_spec.md.

This module deliberately does not import ``licht_inspect.py``.  It is not a
second semantic implementation; it independently re-derives fixed offsets,
CRC envelopes, identities, and stored-index rows for conformance cross-checks.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import struct
import uuid

try:
    from .crc32c import crc32c
except ImportError:  # Direct script execution.
    from crc32c import crc32c


SUPER_MAGIC = b"\x89LFS\r\n\x1a\n"
HEAD_MAGIC = b"LFSHEAD\x00"
COMMIT_MAGIC = b"LFSCOMIT"
INDEX_MAGIC = b"LFSINDEX"
HEAD_OFFSETS = (4_096, 8_192)


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def _uuid(data: bytes, offset: int) -> uuid.UUID:
    return uuid.UUID(bytes=data[offset : offset + 16])


@dataclasses.dataclass(frozen=True)
class RawRow:
    fourcc: bytes
    instance_uuid: uuid.UUID
    row_kind: int
    header_offset: int
    payload_offset: int
    stored_bytes: int
    source_generation: int
    payload_crc32c: int
    header_crc32c: int

    @property
    def key(self) -> bytes:
        return self.fourcc + self.instance_uuid.bytes


@dataclasses.dataclass(frozen=True)
class RawCommit:
    offset: int
    crc_valid: bool
    kind: int
    commit_uuid: uuid.UUID
    generation: int
    parent_commit_uuid: uuid.UUID
    parent_commit_offset: int
    index_offset: int
    index_stored_bytes: int
    committed_file_end: int
    min_reader_version: tuple[int, int]
    min_safe_writer_version: tuple[int, int]
    reader_capabilities: int
    writer_capabilities: int
    index_crc_valid: bool
    index_generation: int | None
    index_commit_uuid: uuid.UUID | None
    rows: tuple[RawRow, ...]
    crc32c: int


@dataclasses.dataclass(frozen=True)
class RawHead:
    slot_id: int
    blank: bool
    head_crc_valid: bool
    encoded_slot_id: int | None
    head_sequence: int | None
    generation: int | None
    commit_uuid: uuid.UUID | None
    commit_offset: int | None
    committed_file_end: int | None
    commit_crc32c_echo: int | None
    commit: RawCommit | None


@dataclasses.dataclass(frozen=True)
class ByteTableView:
    path: Path
    physical_size: int
    format_version: tuple[int, int]
    role: int
    project_uuid: uuid.UUID
    file_uuid: uuid.UUID
    superblock_crc_valid: bool
    heads: tuple[RawHead, RawHead]


class ByteTableError(AssertionError):
    pass


def _pread_exact(fd: int, offset: int, size: int, field: str) -> bytes:
    data = os.pread(fd, size, offset)
    if len(data) != size:
        raise ByteTableError(
            f"0x{offset:016x} {field}: expected {size} bytes, got {len(data)}"
        )
    return data


def _read_commit(fd: int, physical_size: int, offset: int) -> RawCommit | None:
    if offset < 65_536 or offset + 256 > physical_size:
        return None
    raw = _pread_exact(fd, offset, 256, "commit")
    if raw[:8] != COMMIT_MAGIC:
        return None
    got_crc = _u32(raw, 252)
    crc_valid = got_crc == crc32c(raw[:252])
    index_offset = _u64(raw, 136)
    index_bytes = _u64(raw, 144)
    index_crc_valid = False
    index_generation: int | None = None
    index_commit_uuid: uuid.UUID | None = None
    rows: tuple[RawRow, ...] = ()
    if (
        _u32(raw, 168) == 0
        and index_bytes >= 64
        and index_offset + index_bytes <= physical_size
    ):
        index = _pread_exact(fd, index_offset, index_bytes, "stored index")
        index_crc_valid = (
            crc32c(index) == _u32(raw, 160) == _u32(raw, 164)
        )
        if index[:8] == INDEX_MAGIC and len(index) >= 64:
            index_generation = _u64(index, 24)
            index_commit_uuid = _uuid(index, 32)
            row_count = _u64(index, 16)
            if 64 + row_count * 96 == len(index):
                parsed: list[RawRow] = []
                for row_index in range(row_count):
                    row = index[64 + row_index * 96 : 160 + row_index * 96]
                    parsed.append(
                        RawRow(
                            fourcc=row[:4],
                            instance_uuid=_uuid(row, 16),
                            row_kind=row[6],
                            header_offset=_u64(row, 32),
                            payload_offset=_u64(row, 40),
                            stored_bytes=_u64(row, 48),
                            source_generation=_u64(row, 64),
                            payload_crc32c=_u32(row, 72),
                            header_crc32c=_u32(row, 76),
                        )
                    )
                rows = tuple(parsed)
    return RawCommit(
        offset=offset,
        crc_valid=crc_valid,
        kind=_u32(raw, 12),
        commit_uuid=_uuid(raw, 48),
        generation=_u64(raw, 64),
        parent_commit_uuid=_uuid(raw, 72),
        parent_commit_offset=_u64(raw, 88),
        index_offset=index_offset,
        index_stored_bytes=index_bytes,
        committed_file_end=_u64(raw, 176),
        min_reader_version=(_u16(raw, 184), _u16(raw, 186)),
        min_safe_writer_version=(_u16(raw, 188), _u16(raw, 190)),
        reader_capabilities=int.from_bytes(raw[192:208], "little"),
        writer_capabilities=int.from_bytes(raw[208:224], "little"),
        index_crc_valid=index_crc_valid,
        index_generation=index_generation,
        index_commit_uuid=index_commit_uuid,
        rows=rows,
        crc32c=got_crc,
    )


def derive_byte_table(path: Path) -> ByteTableView:
    physical_size = path.stat().st_size
    fd = os.open(path, os.O_RDONLY)
    try:
        superblock = _pread_exact(fd, 0, 256, "superblock")
        if superblock[:8] != SUPER_MAGIC:
            raise ByteTableError("0x0000000000000000 superblock.magic mismatch")
        heads: list[RawHead] = []
        for slot_id, offset in enumerate(HEAD_OFFSETS):
            raw = _pread_exact(fd, offset, 4_096, f"head[{slot_id}]")
            if raw == b"\x00" * 4_096:
                heads.append(
                    RawHead(slot_id, True, False, None, None, None, None, None, None, None, None)
                )
                continue
            magic_valid = raw[:8] == HEAD_MAGIC
            head_crc_valid = magic_valid and _u32(raw, 4_092) == crc32c(raw[:4_092])
            commit_offset = _u64(raw, 80)
            heads.append(
                RawHead(
                    slot_id=slot_id,
                    blank=False,
                    head_crc_valid=head_crc_valid,
                    encoded_slot_id=_u32(raw, 8),
                    head_sequence=_u64(raw, 16),
                    generation=_u64(raw, 24),
                    commit_uuid=_uuid(raw, 64),
                    commit_offset=commit_offset,
                    committed_file_end=_u64(raw, 96),
                    commit_crc32c_echo=_u32(raw, 104),
                    commit=_read_commit(fd, physical_size, commit_offset),
                )
            )
        return ByteTableView(
            path=path,
            physical_size=physical_size,
            format_version=(_u16(superblock, 8), _u16(superblock, 10)),
            role=_u32(superblock, 20),
            project_uuid=_uuid(superblock, 24),
            file_uuid=_uuid(superblock, 40),
            superblock_crc_valid=_u32(superblock, 252) == crc32c(superblock[:252]),
            heads=(heads[0], heads[1]),
        )
    finally:
        os.close(fd)
