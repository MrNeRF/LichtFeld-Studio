#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal .licht container helpers for inspect/restore tools."""

from __future__ import annotations

import datetime
import struct
import uuid

import zstandard

HEAD_SLOTS = (4096, 8192)
HEAD_BYTES = 4096
COMMIT_BYTES = 256
INDEX_HEADER = 64
INDEX_ROW = 96
KIND = {1: "Explicit", 2: "Autosave", 3: "Recovered", 4: "Compaction"}
ROWKIND = {0: "Live", 1: "Tombstone", 2: "BaseRef"}
COMP = {0: "Stored", 1: "ZstdFramed", 2: "ByteShuffleZstdFramed"}

_CRC32C_POLY = 0x82F63B78
_CRC32C_TABLE = []
for i in range(256):
    crc = i
    for _ in range(8):
        crc = (crc >> 1) ^ _CRC32C_POLY if crc & 1 else crc >> 1
    _CRC32C_TABLE.append(crc)


def crc32c(data: bytes, crc: int = 0) -> int:
    crc ^= 0xFFFFFFFF
    table = _CRC32C_TABLE
    for byte in data:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def i32(b, o):
    return struct.unpack_from("<i", b, o)[0]


def guid(b, o):
    return str(uuid.UUID(bytes=bytes(b[o : o + 16])))


def ns_to_str(ns):
    if not ns:
        return "-"
    return (
        datetime.datetime.fromtimestamp(ns / 1e9, datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_head(raw: bytes, slot: int | None = None):
    if raw[:8] != b"LFSHEAD\x00":
        return None
    head = {
        "slot_id": u32(raw, 8),
        "head_sequence": u64(raw, 16),
        "generation": u64(raw, 24),
        "project_uuid": guid(raw, 32),
        "file_uuid": guid(raw, 48),
        "commit_uuid": guid(raw, 64),
        "commit_offset": u64(raw, 80),
        "committed_file_end": u64(raw, 96),
        "commit_crc32c_echo": u32(raw, 104),
        "preview_offset": u64(raw, 112),
        "preview_bytes": u32(raw, 120),
        "preview_format": u32(raw, 124),
        "head_crc32c": u32(raw, 4092),
    }
    if slot is not None:
        head["slot"] = slot
    return head


def parse_commit(raw: bytes, offset: int):
    if raw[:8] != b"LFSCOMIT":
        return None
    return {
        "offset": offset,
        "kind": KIND.get(u32(raw, 12), str(u32(raw, 12))),
        "commit_uuid": guid(raw, 48),
        "generation": u64(raw, 64),
        "parent_commit_uuid": guid(raw, 72),
        "parent_commit_offset": u64(raw, 88),
        "snapshot_uuid": guid(raw, 112),
        "wallclock_unix_ns": u64(raw, 128),
        "index_offset": u64(raw, 136),
        "index_stored_bytes": u64(raw, 144),
        "index_uncompressed_bytes": u64(raw, 152),
        "index_compression": COMP.get(u32(raw, 168), str(u32(raw, 168))),
        "committed_file_end": u64(raw, 176),
        "crc32c": u32(raw, 252),
        "raw": raw,
    }


def scan_commits(data: bytes):
    commits = []
    start = 0
    while True:
        pos = data.find(b"LFSCOMIT", start)
        if pos < 0:
            break
        commit = parse_commit(data[pos : pos + COMMIT_BYTES], pos)
        if commit:
            commits.append(commit)
        start = pos + 1
    return commits


def decode_index(data: bytes, commit):
    stored = bytes(
        data[commit["index_offset"] : commit["index_offset"] + commit["index_stored_bytes"]]
    )
    if commit["index_compression"] == "Stored":
        decoded = stored
    else:
        decoded = zstandard.ZstdDecompressor().decompress(
            stored, max_output_size=commit["index_uncompressed_bytes"]
        )
    if decoded[:8] != b"LFSINDEX":
        raise RuntimeError("bad index magic")
    rows = []
    for i in range(u64(decoded, 16)):
        off = INDEX_HEADER + i * INDEX_ROW
        row = decoded[off : off + INDEX_ROW]
        rows.append(
            {
                "fourcc": row[0:4].decode("ascii", "replace"),
                "chunk_version": u16(row, 4),
                "row_kind": ROWKIND.get(row[6], str(row[6])),
                "compression": COMP.get(row[7], str(row[7])),
                "flags": u32(row, 8),
                "uuid": guid(row, 16),
                "header_offset": u64(row, 32),
                "payload_offset": u64(row, 40),
                "stored_bytes": u64(row, 48),
                "uncompressed_bytes": u64(row, 56),
                "source_generation": u64(row, 64),
            }
        )
    return u64(decoded, 24), rows


def read_first_uncompressed(data: bytes, row, limit=4096):
    take = min(row["stored_bytes"], 8 * 1024 * 1024)
    payload = bytes(data[row["payload_offset"] : row["payload_offset"] + take])
    if row["compression"] == "Stored":
        return payload[:limit]
    if payload[:8] != b"LFSZFRM\x00":
        try:
            return zstandard.ZstdDecompressor().decompress(
                payload, max_output_size=min(row["uncompressed_bytes"], 1_000_000)
            )[:limit]
        except Exception:
            return payload[:limit]
    stored_bytes = u64(payload, 16)
    table = 16 + u32(payload, 12) * 16
    frame = payload[table : table + stored_bytes]
    return zstandard.ZstdDecompressor().stream_reader(frame).read(limit)


def parse_ckpt_header(data: bytes):
    if len(data) < 40:
        return None
    magic = u32(data, 0)
    if magic != 0x4C464B50:
        return {"magic": hex(magic), "note": "not LFKP", "prefix": data[:16]}
    return {
        "version": u32(data, 4),
        "iteration": i32(data, 8),
        "num_gaussians": u32(data, 12),
        "sh_degree": i32(data, 16),
        "flags": u32(data, 20),
    }


def parse_metr(data: bytes):
    if data[:6] != b"LFMETR":
        return {"note": "not METR", "prefix": data[:16]}
    loss_count = u64(data, 16)
    psnr_count = u64(data, 24)
    off = 8 + 4 + 4 + 8 + 8 + 8 + 4 + 4 + 4
    last_loss_iter = u32(data, off + (loss_count - 1) * 8) if loss_count else None
    last_psnr_iter = (
        u32(data, off + loss_count * 8 + (psnr_count - 1) * 8) if psnr_count else None
    )
    return {
        "loss_samples": loss_count,
        "psnr_samples": psnr_count,
        "last_eval_iteration": u32(data, 36),
        "last_loss_iteration": last_loss_iter,
        "last_psnr_iteration": last_psnr_iter,
    }


def unshuffle_f32_planes(shuffled: bytes) -> bytes:
    n_words = len(shuffled) // 4
    logical = bytearray(len(shuffled))
    for w in range(n_words):
        logical[w * 4 + 0] = shuffled[0 * n_words + w]
        logical[w * 4 + 1] = shuffled[1 * n_words + w]
        logical[w * 4 + 2] = shuffled[2 * n_words + w]
        logical[w * 4 + 3] = shuffled[3 * n_words + w]
    return bytes(logical)


def decode_framed_payload(data: bytes, row) -> bytes:
    payload = data[row["payload_offset"] : row["payload_offset"] + row["stored_bytes"]]
    if row["compression"] == "Stored":
        return payload
    if payload[:8] != b"LFSZFRM\x00":
        return zstandard.ZstdDecompressor().decompress(
            payload, max_output_size=row["uncompressed_bytes"]
        )
    count = u32(payload, 12)
    table = 16 + count * 16
    uncompressed = bytearray()
    cursor = table
    dctx = zstandard.ZstdDecompressor()
    for i in range(count):
        stored_n = u64(payload, 16 + i * 16)
        uncomp_n = u64(payload, 16 + i * 16 + 8)
        frame = payload[cursor : cursor + stored_n]
        uncompressed += dctx.decompress(frame, max_output_size=uncomp_n)
        cursor += stored_n
    return bytes(uncompressed)


def decode_ckpt_header(data: bytes, row):
    if row["compression"] == "ByteShuffleZstdFramed":
        decoded = decode_framed_payload(data, row)
        if len(decoded) != row["uncompressed_bytes"]:
            return {
                "error": (
                    f"size mismatch {len(decoded)} != {row['uncompressed_bytes']}"
                )
            }
        return parse_ckpt_header(unshuffle_f32_planes(decoded))
    return parse_ckpt_header(read_first_uncompressed(data, row, 64))


def encode_head(superblock: bytes, slot_id: int, sequence: int, commit, preview) -> bytes:
    raw = bytearray(HEAD_BYTES)
    raw[0:8] = b"LFSHEAD\x00"
    struct.pack_into("<I", raw, 8, slot_id)
    struct.pack_into("<I", raw, 12, HEAD_BYTES)
    struct.pack_into("<Q", raw, 16, sequence)
    struct.pack_into("<Q", raw, 24, commit["generation"])
    raw[32:48] = superblock[24:40]
    raw[48:64] = superblock[40:56]
    raw[64:80] = uuid.UUID(commit["commit_uuid"]).bytes
    struct.pack_into("<Q", raw, 80, commit["offset"])
    struct.pack_into("<Q", raw, 88, COMMIT_BYTES)
    struct.pack_into("<Q", raw, 96, commit["committed_file_end"])
    struct.pack_into("<I", raw, 104, commit["crc32c"])
    if preview:
        struct.pack_into("<Q", raw, 112, preview["offset"])
        struct.pack_into("<I", raw, 120, preview["bytes"])
        struct.pack_into("<I", raw, 124, preview["format"])
    struct.pack_into("<I", raw, 4092, crc32c(bytes(raw[:4092])))
    return bytes(raw)
