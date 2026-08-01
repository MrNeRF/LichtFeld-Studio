"""Index-free preview extraction showing the fixed metadata path stays tiny."""
from __future__ import annotations
import os
from pathlib import Path
import struct
from typing import NamedTuple
import uuid
try:
    from .crc32c import crc32c
except ImportError:  # Direct script execution.
    from crc32c import crc32c

SUPER_MAGIC = b"\x89LFS\r\n\x1a\n"
HEAD_MAGIC = b"LFSHEAD\x00"
HEAD_OFFSETS = (4_096, 8_192)
HEAD_BYTES, APPEND_OFFSET = 4_096, 65_536
MAX_PREVIEW_BYTES = 16 * 1024 * 1024

class ForeignPreviewError(ValueError):
    pass

class ForeignPreview(NamedTuple):
    slot_id: int; head_sequence: int; generation: int
    offset: int; bytes: int; format: int

class _Head(NamedTuple):
    slot: int; sequence: int; generation: int
    commit_uuid: uuid.UUID; commit_crc: int; commit_offset: int; committed_end: int
    project_uuid: uuid.UUID; file_uuid: uuid.UUID; preview: ForeignPreview | None

def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]

def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]

def _uuid(data: bytes, offset: int) -> uuid.UUID:
    return uuid.UUID(bytes=data[offset : offset + 16])

def _pread(fd: int, offset: int, size: int, label: str) -> bytes:
    data = os.pread(fd, size, offset)
    if len(data) != size:
        raise ForeignPreviewError(
            f"{label} at 0x{offset:x}: expected {size} bytes, got {len(data)}"
        )
    return data

def _authority(head: _Head) -> tuple[object, ...]:
    return (
        head.generation, head.commit_offset, 256, head.committed_end,
        head.project_uuid, head.file_uuid,
    )

def _parse_head(
    raw: bytes, slot: int, physical_size: int,
    project_uuid: uuid.UUID, file_uuid: uuid.UUID,
) -> _Head:
    label = f"head[{slot}]"
    if raw[:8] != HEAD_MAGIC or _u32(raw, 4_092) != crc32c(raw[:4_092]):
        raise ForeignPreviewError(f"{label}: invalid magic or CRC32c")
    if _u32(raw, 8) != slot or _u32(raw, 12) != HEAD_BYTES:
        raise ForeignPreviewError(f"{label}: invalid fixed geometry")
    sequence, generation = _u64(raw, 16), _u64(raw, 24)
    commit_uuid = _uuid(raw, 64)
    commit_offset, commit_bytes = _u64(raw, 80), _u64(raw, 88)
    committed_end = _u64(raw, 96)
    if sequence < 1 or generation < 1 or commit_uuid.int == 0:
        raise ForeignPreviewError(f"{label}: invalid authority identity")
    if _uuid(raw, 32) != project_uuid or _uuid(raw, 48) != file_uuid:
        raise ForeignPreviewError(f"{label}: superblock identity mismatch")
    fixed_invalid = (
        commit_offset < APPEND_OFFSET or commit_offset % 64 or commit_bytes != 256
        or committed_end != commit_offset + 256 or committed_end > physical_size
        or _u32(raw, 108) != 0 or any(raw[128:4_092])
    )
    if fixed_invalid:
        raise ForeignPreviewError(f"{label}: invalid bounds or reserved bytes")
    preview_offset = _u64(raw, 112)
    preview_bytes, preview_format = _u32(raw, 120), _u32(raw, 124)
    if preview_offset == 0:
        if preview_bytes or preview_format:
            raise ForeignPreviewError(f"{label}: inconsistent absent preview")
        preview = None
    else:
        preview_end = preview_offset + preview_bytes
        locator_invalid = (
            preview_offset % 64 or not 1 <= preview_bytes <= MAX_PREVIEW_BYTES
            or preview_format != 1 or preview_end > committed_end
            or preview_end > 0xFFFFFFFFFFFFFFFF
        )
        if locator_invalid:
            raise ForeignPreviewError(f"{label}: invalid preview locator")
        preview = ForeignPreview(
            slot, sequence, generation, preview_offset, preview_bytes, preview_format
        )
    return _Head(
        slot, sequence, generation, commit_uuid, _u32(raw, 104), commit_offset,
        committed_end, project_uuid, file_uuid, preview,
    )

def read_foreign_preview(path: str | os.PathLike[str]) -> tuple[ForeignPreview, bytes]:
    """Read the superblock, head slots, locator, and payload—never an index."""
    fd = os.open(path, os.O_RDONLY)
    try:
        physical_size = os.fstat(fd).st_size
        superblock = _pread(fd, 0, 256, "superblock")
        geometry = (
            _u64(superblock, 56), _u64(superblock, 64), _u32(superblock, 72),
            _u32(superblock, 76), _u64(superblock, 80),
        )
        invalid_superblock = (
            superblock[:8] != SUPER_MAGIC
            or _u32(superblock, 252) != crc32c(superblock[:252])
            or struct.unpack_from("<HHII", superblock, 8) != (1, 0, 0x01020304, 256)
            or _u32(superblock, 20) not in (0, 1)
            or geometry != (4_096, 8_192, HEAD_BYTES, 2, APPEND_OFFSET)
        )
        if invalid_superblock:
            raise ForeignPreviewError("invalid .licht superblock")
        project_uuid, file_uuid = _uuid(superblock, 24), _uuid(superblock, 40)
        if project_uuid.int == 0 or file_uuid.int == 0:
            raise ForeignPreviewError("null superblock identity")
        heads: list[_Head] = []
        for slot, offset in enumerate(HEAD_OFFSETS):
            raw = _pread(fd, offset, HEAD_BYTES, f"head[{slot}]")
            if raw == b"\x00" * HEAD_BYTES:
                continue
            try:
                heads.append(_parse_head(raw, slot, physical_size, project_uuid, file_uuid))
            except ForeignPreviewError:
                pass
        if not heads:
            raise ForeignPreviewError("no valid head slot")
        if len(heads) == 2 and heads[0].sequence == heads[1].sequence:
            duplicate = (
                heads[0].commit_uuid == heads[1].commit_uuid
                and heads[0].commit_crc == heads[1].commit_crc
                and _authority(heads[0]) == _authority(heads[1])
            )
            if not duplicate:
                raise ForeignPreviewError("equal-sequence head conflict")
            selected = heads[0]
        else:
            selected = max(heads, key=lambda head: head.sequence)
        if selected.preview is None:
            raise ForeignPreviewError("selected generation has no preview")
        preview = selected.preview
        return preview, _pread(fd, preview.offset, preview.bytes, "preview")
    finally:
        os.close(fd)

def extract_foreign_preview(
    path: str | os.PathLike[str], output_path: str | os.PathLike[str], *,
    force: bool = False,
) -> ForeignPreview:
    preview, payload = read_foreign_preview(path)
    with Path(output_path).open("wb" if force else "xb") as output:
        output.write(payload)
    return preview
