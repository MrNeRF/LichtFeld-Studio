"""Sparse 64-bit geometry, block-table, row-count, and lineage scale cases."""

from __future__ import annotations

import os
from pathlib import Path
import struct
import tempfile
import time
import uuid

try:
    from .conformance_common import (
        CategoryResult,
        ConformanceConfig,
        deterministic_uuid,
        timed_result,
    )
    from .crc32c import crc32c
    from .licht_inspect import open_container, verify_container
    from .make_fixtures import ChunkSpec, FixtureWriter
except ImportError:  # Direct script execution.
    from conformance_common import (
        CategoryResult,
        ConformanceConfig,
        deterministic_uuid,
        timed_result,
    )
    from crc32c import crc32c
    from licht_inspect import open_container, verify_container
    from make_fixtures import ChunkSpec, FixtureWriter


SUPER_MAGIC = b"\x89LFS\r\n\x1a\n"
HEAD_MAGIC = b"LFSHEAD\x00"
COMMIT_MAGIC = b"LFSCOMIT"
INDEX_MAGIC = b"LFSINDEX"
BLOCK_MAGIC = b"LFSBCRC\x00"
APPEND_OFFSET = 65_536
BLOCK_SIZE = 4 * 1_024 * 1_024
BLOCK_REQUIRED_AT = 1 << 30
CAP_BLOCK_CRC = 1 << 2
CAP_OPAQUE = 1 << 5
CAP_CLEAN_PROOF = 1 << 7


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _gf2_matrix_times(matrix: list[int], vector: int) -> int:
    result = 0
    index = 0
    while vector:
        if vector & 1:
            result ^= matrix[index]
        vector >>= 1
        index += 1
    return result & 0xFFFFFFFF


def _gf2_matrix_square(matrix: list[int]) -> list[int]:
    return [_gf2_matrix_times(matrix, matrix[index]) for index in range(32)]


def _crc32c_combine(crc1: int, crc2: int, length2: int) -> int:
    """CRC32c(A+B) from CRC32c(A), CRC32c(B), and len(B)."""

    if length2 <= 0:
        return crc1
    odd = [0] * 32
    odd[0] = 0x82F63B78
    row = 1
    for index in range(1, 32):
        odd[index] = row
        row <<= 1
    even = _gf2_matrix_square(odd)
    odd = _gf2_matrix_square(even)
    while True:
        even = _gf2_matrix_square(odd)
        if length2 & 1:
            crc1 = _gf2_matrix_times(even, crc1)
        length2 >>= 1
        if length2 == 0:
            break
        odd = _gf2_matrix_square(even)
        if length2 & 1:
            crc1 = _gf2_matrix_times(odd, crc1)
        length2 >>= 1
        if length2 == 0:
            break
    return (crc1 ^ crc2) & 0xFFFFFFFF


def _zero_crc_and_entries(length: int) -> tuple[int, tuple[int, ...]]:
    zero_block = b"\x00" * BLOCK_SIZE
    full_crc = crc32c(zero_block)
    full_blocks, remainder = divmod(length, BLOCK_SIZE)
    entries = [full_crc] * full_blocks
    if remainder:
        entries.append(crc32c(memoryview(zero_block)[:remainder]))
    combined = 0
    for index, entry in enumerate(entries):
        block_bytes = BLOCK_SIZE if index < full_blocks else remainder
        combined = _crc32c_combine(combined, entry, block_bytes)
    return combined, tuple(entries)


def _pack_superblock(project_uuid: uuid.UUID, file_uuid: uuid.UUID) -> bytearray:
    data = bytearray(APPEND_OFFSET)
    superblock = bytearray(256)
    superblock[:8] = SUPER_MAGIC
    struct.pack_into("<HHIII", superblock, 8, 1, 0, 0x01020304, 256, 0)
    superblock[24:40] = project_uuid.bytes
    superblock[40:56] = file_uuid.bytes
    struct.pack_into(
        "<QQIIQQ",
        superblock,
        56,
        4_096,
        8_192,
        4_096,
        2,
        APPEND_OFFSET,
        1_735_689_600_000_000_000,
    )
    struct.pack_into("<I", superblock, 252, crc32c(superblock[:252]))
    data[:256] = superblock
    return data


def _write_sparse_container(
    path: Path,
    *,
    namespace: int,
    header_offset: int,
    payload_offset: int,
    payload_size: int,
    block_table: bool,
    payload_byte: bytes | None = None,
) -> tuple[int, int, int]:
    project_uuid = deterministic_uuid(namespace, 1)
    file_uuid = deterministic_uuid(namespace, 2)
    commit_uuid = deterministic_uuid(namespace, 3)
    snapshot_uuid = deterministic_uuid(namespace, 4)
    instance_uuid = deterministic_uuid(namespace, 5)
    bootstrap = _pack_superblock(project_uuid, file_uuid)

    if block_table:
        payload_crc, block_entries = _zero_crc_and_entries(payload_size)
        entries = struct.pack(f"<{len(block_entries)}I", *block_entries)
        table_offset = header_offset + 64
        table = bytearray(64)
        table[:8] = BLOCK_MAGIC
        struct.pack_into(
            "<HHHHIIQQQ",
            table,
            8,
            1,
            64,
            4,
            1,
            BLOCK_SIZE,
            0,
            payload_offset,
            payload_size,
            len(block_entries),
        )
        struct.pack_into("<I", table, 48, crc32c(entries))
        struct.pack_into("<I", table, 60, crc32c(table[:60]))
        flags = 1 << 1
        block_kind = 1
    else:
        if payload_byte is None or len(payload_byte) != payload_size:
            raise ValueError("non-sparse payload bytes must exactly match payload_size")
        payload_crc = crc32c(payload_byte)
        entries = b""
        table = b""
        table_offset = 0
        flags = 0
        block_kind = 0

    header = bytearray(64)
    header[:4] = b"SPLT"
    struct.pack_into("<HH", header, 4, 1, 64)
    header[8:24] = instance_uuid.bytes
    struct.pack_into(
        "<IHHQQQI",
        header,
        24,
        flags,
        0,
        block_kind,
        payload_size,
        payload_size,
        table_offset,
        payload_crc,
    )
    struct.pack_into("<I", header, 60, crc32c(header[:60]))

    index_offset = _align(payload_offset + payload_size, 64)
    index = bytearray(160)
    index[:8] = INDEX_MAGIC
    struct.pack_into("<HHHHQQ", index, 8, 1, 64, 96, 1, 1, 1)
    index[32:48] = commit_uuid.bytes
    index[64:68] = b"SPLT"
    struct.pack_into("<HBBI", index, 68, 1, 0, 0, flags)
    index[80:96] = instance_uuid.bytes
    struct.pack_into(
        "<QQQQQII",
        index,
        96,
        header_offset,
        payload_offset,
        payload_size,
        payload_size,
        1,
        payload_crc,
        struct.unpack_from("<I", header, 60)[0],
    )
    index_crc = crc32c(index)
    commit_offset = _align(index_offset + len(index), 64)
    committed_end = commit_offset + 256
    commit = bytearray(256)
    commit[:8] = COMMIT_MAGIC
    struct.pack_into("<HHI", commit, 8, 256, 1, 1)
    commit[16:32] = project_uuid.bytes
    commit[32:48] = file_uuid.bytes
    commit[48:64] = commit_uuid.bytes
    struct.pack_into("<Q", commit, 64, 1)
    commit[96:112] = commit_uuid.bytes
    commit[112:128] = snapshot_uuid.bytes
    struct.pack_into(
        "<QQQQII",
        commit,
        128,
        1_735_689_601_000_000_001,
        index_offset,
        len(index),
        len(index),
        index_crc,
        index_crc,
    )
    struct.pack_into("<IIQHHHH", commit, 168, 0, 0, committed_end, 1, 0, 1, 0)
    commit[192:208] = (CAP_BLOCK_CRC if block_table else 0).to_bytes(16, "little")
    commit[208:224] = (CAP_OPAQUE | CAP_CLEAN_PROOF).to_bytes(16, "little")
    commit_crc = crc32c(commit[:252])
    struct.pack_into("<I", commit, 252, commit_crc)

    head = bytearray(4_096)
    head[:8] = HEAD_MAGIC
    struct.pack_into("<IIQQ", head, 8, 0, 4_096, 1, 1)
    head[32:48] = project_uuid.bytes
    head[48:64] = file_uuid.bytes
    head[64:80] = commit_uuid.bytes
    struct.pack_into("<QQQI", head, 80, commit_offset, 256, committed_end, commit_crc)
    struct.pack_into("<I", head, 4_092, crc32c(head[:4_092]))
    bootstrap[4_096:8_192] = head

    with path.open("wb") as stream:
        stream.write(bootstrap)
        stream.seek(header_offset)
        stream.write(header)
        if block_table:
            stream.write(table)
            stream.write(entries)
        elif payload_byte is not None:
            stream.seek(payload_offset)
            stream.write(payload_byte)
        stream.seek(index_offset)
        stream.write(index)
        stream.seek(commit_offset)
        stream.write(commit)
    return committed_end, index_offset, len(entries) // 4


def _assert_sparse(path: Path, maximum_allocated: int = 16 * 1_024 * 1_024) -> None:
    stat = path.stat()
    allocated = stat.st_blocks * 512
    if allocated > maximum_allocated:
        raise AssertionError(
            f"{path.name}: sparse case allocated {allocated} bytes for logical size {stat.st_size}"
        )


def _run_many_rows(path: Path, row_count: int) -> None:
    writer = FixtureWriter(
        project_uuid=deterministic_uuid(0xE100, 1),
        file_uuid=deterministic_uuid(0xE100, 2),
    )
    changes: list[ChunkSpec] = []
    for index in range(row_count):
        if index == 0:
            fourcc, instance_uuid = b"0000", uuid.UUID(int=1)
        elif index == 1:
            fourcc, instance_uuid = b"0000", uuid.UUID(int=(1 << 128) - 1)
        elif index == row_count - 1:
            fourcc, instance_uuid = b"ZZZZ", uuid.UUID(int=(1 << 128) - 2)
        else:
            fourcc, instance_uuid = b"A000", uuid.UUID(int=index + 1)
        changes.append(ChunkSpec(fourcc, instance_uuid, bytes([(index % 251) + 1])))
    writer.append_generation(
        commit_uuid=deterministic_uuid(0xE100, 3),
        snapshot_uuid=deterministic_uuid(0xE100, 4),
        changes=tuple(changes),
        head_slot=0,
        head_sequence=1,
    )
    path.write_bytes(writer.bytes())
    container = open_container(path)
    if len(container.index.rows) != row_count:
        raise AssertionError(
            f"large index expected {row_count} rows, got {len(container.index.rows)}"
        )
    keys = [row.key for row in container.index.rows]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise AssertionError("large index lost strict raw-byte key ordering")
    if keys[0][:4] != b"0000" or keys[-1][:4] != b"ZZZZ":
        raise AssertionError("fourcc/UUID edge keys did not bracket the canonical sort")
    verify_container(container)


def _run_many_generations(path: Path, generation_count: int) -> None:
    writer = FixtureWriter(
        project_uuid=deterministic_uuid(0xE200, 1),
        file_uuid=deterministic_uuid(0xE200, 2),
    )
    instance_uuid = deterministic_uuid(0xE200, 5)
    for generation in range(1, generation_count + 1):
        writer.append_generation(
            commit_uuid=deterministic_uuid(0xE200, 10_000 + generation),
            snapshot_uuid=deterministic_uuid(0xE200, 20_000 + generation),
            changes=(
                ChunkSpec(
                    b"PROJ",
                    instance_uuid,
                    f"generation-{generation}".encode(),
                ),
            ),
            head_slot=(generation - 1) & 1,
            head_sequence=generation,
        )
    path.write_bytes(writer.bytes())
    container = open_container(path)
    if container.commit.generation != generation_count:
        raise AssertionError(
            f"long lineage expected generation {generation_count}, got {container.commit.generation}"
        )
    if container.selected_head.head_sequence != generation_count:
        raise AssertionError("long lineage selected wrong head sequence")
    verify_container(container)


def run_scale_edges(config: ConformanceConfig) -> CategoryResult:
    started = time.monotonic()
    # Check the combine helper against direct bytes before using it to describe
    # a GiB-scale sparse zero payload.
    left, right = b"combine-left", b"combine-right"
    if _crc32c_combine(crc32c(left), crc32c(right), len(right)) != crc32c(left + right):
        raise AssertionError("CRC32c combine self-check failed")

    with tempfile.TemporaryDirectory(prefix="licht-conformance-scale-") as temp:
        temp_dir = Path(temp)

        offset_path = temp_dir / "offset-over-4gib.licht"
        header_offset = (1 << 32) + APPEND_OFFSET
        payload_offset = header_offset + 64
        end, index_offset, _ = _write_sparse_container(
            offset_path,
            namespace=0xE001,
            header_offset=header_offset,
            payload_offset=payload_offset,
            payload_size=1,
            block_table=False,
            payload_byte=b"X",
        )
        _assert_sparse(offset_path)
        offset_container = open_container(offset_path)
        if index_offset <= 0xFFFFFFFF or end <= 0xFFFFFFFF:
            raise AssertionError("4-GiB case did not cross the 32-bit boundary")
        verify_container(offset_container)

        block_path = temp_dir / "block-table-over-1gib.licht"
        payload_size = BLOCK_REQUIRED_AT + 1
        block_header_offset = APPEND_OFFSET
        expected_blocks = (payload_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        table_bytes = 64 + expected_blocks * 4
        block_payload_offset = _align(block_header_offset + 64 + table_bytes, 4_096)
        _, _, block_count = _write_sparse_container(
            block_path,
            namespace=0xE002,
            header_offset=block_header_offset,
            payload_offset=block_payload_offset,
            payload_size=payload_size,
            block_table=True,
        )
        _assert_sparse(block_path)
        block_container = open_container(block_path)
        row = block_container.index.rows[0]
        if row.block_table is None or block_count != expected_blocks:
            raise AssertionError(
                f">1-GiB block table expected {expected_blocks} entries, got {block_count}"
            )
        fd = os.open(block_path, os.O_RDONLY)
        try:
            if os.pread(fd, 1, block_payload_offset) != b"\x00":
                raise AssertionError("sparse payload first byte is not zero")
            if os.pread(fd, 1, block_payload_offset + payload_size - 1) != b"\x00":
                raise AssertionError("sparse payload last byte is not zero")
        finally:
            os.close(fd)

        _run_many_rows(temp_dir / "many-rows.licht", config.scale_rows)
        _run_many_generations(
            temp_dir / "many-generations.licht", config.scale_generations
        )

    cases = config.scale_rows + config.scale_generations + 2
    return timed_result(
        "scale-edges",
        started,
        cases,
        f">4GiB sparse offset, >1GiB/257-entry block table, "
        f"rows={config.scale_rows}, generations={config.scale_generations}",
    )
