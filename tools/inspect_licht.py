#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Inspect a .licht file, or restore an older generation to a new file."""

from __future__ import annotations

import argparse
import collections
import datetime
import mmap
import struct
import sys
import uuid
from pathlib import Path

import zstandard

HEAD_SLOTS = (4096, 8192)
HEAD_BYTES = 4096
COMMIT_BYTES = 256
INDEX_HEADER = 64
INDEX_ROW = 96
KIND = {1: "Explicit", 2: "Autosave", 3: "Recovered", 4: "Compaction"}
ROWKIND = {0: "Live", 1: "Tombstone", 2: "BaseRef"}
COMP = {0: "Stored", 1: "ZstdFramed", 2: "ByteShuffleZstdFramed"}
ROLE = {0: "master", 1: "autosave sidecar"}

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


def _iter_uncompressed_chunks(payload: bytes, stop_after: int):
    dctx = zstandard.ZstdDecompressor()
    produced = 0

    def emit(reader, remaining: int):
        nonlocal produced
        while remaining > 0 and produced < stop_after:
            chunk = reader.read(min(1 << 20, remaining, stop_after - produced))
            if not chunk:
                break
            remaining -= len(chunk)
            produced += len(chunk)
            yield chunk

    if payload[:8] != b"LFSZFRM\x00":
        yield from emit(dctx.stream_reader(payload), stop_after)
        return
    count = u32(payload, 12)
    cursor = 16 + count * 16
    for i in range(count):
        stored_n = u64(payload, 16 + i * 16)
        uncomp_n = u64(payload, 16 + i * 16 + 8)
        frame = payload[cursor : cursor + stored_n]
        cursor += stored_n
        yield from emit(dctx.stream_reader(frame), uncomp_n)
        if produced >= stop_after:
            return


def read_first_uncompressed(data: bytes, row, limit=4096):
    take = min(row["stored_bytes"], 8 * 1024 * 1024)
    payload = bytes(data[row["payload_offset"] : row["payload_offset"] + take])
    if row["compression"] == "Stored":
        return payload[:limit]
    out = bytearray()
    for chunk in _iter_uncompressed_chunks(payload, limit):
        out += chunk
        if len(out) >= limit:
            break
    return bytes(out[:limit])


def read_logical_prefix(data: bytes, row, limit=64) -> bytes:
    if row["compression"] == "Stored":
        return bytes(
            data[row["payload_offset"] : row["payload_offset"] + min(limit, row["stored_bytes"])]
        )
    payload = bytes(
        data[row["payload_offset"] : row["payload_offset"] + row["stored_bytes"]]
    )
    if row["compression"] != "ByteShuffleZstdFramed":
        return read_first_uncompressed(data, row, limit)

    uncompressed_bytes = row["uncompressed_bytes"]
    if uncompressed_bytes < 4 or uncompressed_bytes % 4 != 0:
        return read_first_uncompressed(data, row, limit)

    n_words = uncompressed_bytes // 4
    words = min((limit + 3) // 4, n_words)
    last_needed = 3 * n_words + words
    collected = {}
    pos = 0
    for chunk in _iter_uncompressed_chunks(payload, last_needed):
        start = pos
        pos += len(chunk)
        for plane in range(4):
            for word in range(words):
                offset = plane * n_words + word
                if start <= offset < pos:
                    collected[offset] = chunk[offset - start]
        if len(collected) == 4 * words:
            break

    if len(collected) != 4 * words:
        raise RuntimeError(
            f"ByteShuffle prefix incomplete: got {len(collected)} of {4 * words} bytes"
        )
    logical = bytearray(words * 4)
    for word in range(words):
        for plane in range(4):
            logical[word * 4 + plane] = collected[plane * n_words + word]
    return bytes(logical[:limit])


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


def decode_ckpt_header(data: bytes, row):
    return parse_ckpt_header(read_logical_prefix(data, row, 64))


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


def format_bytes(size: int) -> str:
    if size >= 1024 * 1024 * 1024:
        return f"{size / (1024 ** 3):.1f} GiB"
    if size >= 1024 * 1024:
        return f"{size / (1024 ** 2):.1f} MiB"
    if size >= 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size} B"


def format_time(ns: int) -> str:
    if not ns:
        return "-"
    moment = datetime.datetime.fromtimestamp(ns / 1e9, datetime.timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_count(value: int | None) -> str:
    if value is None:
        return "?"
    return f"{value:,}"


def print_table(headers: list[str], rows: list[list[object]], *, right: set[int] | None = None) -> None:
    right = right or set()
    cells = [["" if value is None else str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in cells:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        parts = []
        for index, value in enumerate(values):
            if index in right:
                parts.append(value.rjust(widths[index]))
            else:
                parts.append(value.ljust(widths[index]))
        return "| " + " | ".join(parts) + " |"

    print(format_row(headers))
    print("|-" + "-|-".join("-" * width for width in widths) + "-|")
    for row in cells:
        print(format_row(row))


def checkpoint_cells(ckpt: dict | None, metr: dict | None) -> tuple[str, str, str]:
    if ckpt and "iteration" in ckpt:
        gaussians = (
            format_count(ckpt["num_gaussians"]) if "num_gaussians" in ckpt else "-"
        )
        sh_degree = str(ckpt["sh_degree"]) if "sh_degree" in ckpt else "-"
        return format_count(ckpt["iteration"]), gaussians, sh_degree
    if metr and metr.get("last_loss_iteration") is not None:
        return f"{format_count(metr['last_loss_iteration'])} (metrics)", "-", "-"
    if ckpt is None:
        return "-", "-", "-"
    if ckpt.get("error"):
        return "unreadable", "-", "-"
    return "present", "-", "-"


def generation_status(item, current_gen, current_uuid) -> str:
    ckpt = item["ckpt"]
    notes = []
    if item["generation"] == current_gen:
        notes.append("current")
    elif ckpt and ckpt.get("source_generation") not in (None, item["generation"]):
        notes.append(f"reused from gen {ckpt['source_generation']}")
    elif (
        ckpt
        and current_uuid
        and ckpt.get("uuid") == current_uuid
        and item["generation"] != current_gen
    ):
        notes.append("same checkpoint")
    elif ckpt and item["generation"] != current_gen:
        notes.append("superseded")
    return ", ".join(notes) or "-"


def format_tombstones(items: list[dict]) -> str:
    if not items:
        return "-"
    parts = []
    for item in items:
        origin = item["origin_generation"]
        if origin is None:
            parts.append(item["fourcc"])
        else:
            parts.append(f"{item['fourcc']} gen {origin}")
    return ", ".join(parts)


def checkpoint_uuid(ckpt: dict | None) -> str:
    if ckpt and ckpt.get("uuid"):
        return ckpt["uuid"]
    return "-"


def live_row(rows, fourcc: str):
    return next(
        (row for row in rows if row["fourcc"] == fourcc and row["row_kind"] == "Live"),
        None,
    )


def ckpt_info(mm, row) -> dict | None:
    if row is None:
        return None
    try:
        header = decode_ckpt_header(mm, row)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    info = {
        "uuid": row["uuid"],
        "source_generation": row["source_generation"],
        "stored_bytes": row["stored_bytes"],
        "compression": row["compression"],
    }
    if header and "iteration" in header:
        info.update(header)
    elif header:
        info["unreadable"] = header
    return info


def metr_info(mm, row) -> dict | None:
    if row is None:
        return None
    try:
        return parse_metr(read_first_uncompressed(mm, row, 10_000_000))
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def walk_lineage(mm, selected):
    lineage = []
    if not selected:
        return lineage
    offset = selected["commit_offset"]
    seen = set()
    while offset and offset not in seen:
        seen.add(offset)
        commit = parse_commit(mm[offset : offset + COMMIT_BYTES], offset)
        if not commit:
            break
        lineage.append(commit)
        if commit["generation"] == 1 or commit["parent_commit_offset"] == 0:
            break
        offset = commit["parent_commit_offset"]
    return lineage


def inspect_full(path: Path, mm, superblock, heads, selected, commits, lineage) -> None:
    print("=" * 80)
    print(path.name, f"{path.stat().st_size:,} bytes")
    print(
        "superblock magic",
        superblock[:8],
        "role",
        u32(superblock, 20),
        "project",
        guid(superblock, 24),
        "file",
        guid(superblock, 40),
    )
    for slot, head in enumerate(heads):
        print(" head", slot, head)
    print(" selected head", selected)
    print(f" scanned LFSCOMIT count={len(commits)}")
    for commit in commits:
        print(
            f"  gen={commit['generation']} kind={commit['kind']} "
            f"offset=0x{commit['offset']:x} end={commit['committed_file_end']:,} "
            f"parent_off=0x{commit['parent_commit_offset']:x} "
            f"time={ns_to_str(commit['wallclock_unix_ns'])}"
        )
    print(
        " lineage generations newest->oldest: "
        f"{[commit['generation'] for commit in lineage]}"
    )
    if selected and not lineage:
        print(" broken lineage at", selected["commit_offset"])

    seen_gen = set()
    for commit in lineage:
        if commit["generation"] in seen_gen:
            continue
        seen_gen.add(commit["generation"])
        generation, rows = decode_index(mm, commit)
        print(
            f"\n --- generation {generation} index rows={len(rows)} "
            f"kind={commit['kind']}"
        )
        src_counts = collections.Counter(
            (row["fourcc"], row["row_kind"], row["source_generation"])
            for row in rows
        )
        for key, count in sorted(src_counts.items()):
            print(f"    {key[0]:4} {key[1]:9} source_gen={key[2]} count={count}")
        for row in rows:
            if row["row_kind"] != "Live":
                continue
            extra = (
                f" stored={row['stored_bytes']:,} "
                f"uncompressed={row['uncompressed_bytes']:,}"
            )
            try:
                if row["fourcc"] == "CKPT":
                    extra = f"{extra} CKPT={decode_ckpt_header(mm, row)}"
                elif row["fourcc"] == "METR":
                    extra = (
                        f"{extra} METR="
                        f"{parse_metr(read_first_uncompressed(mm, row, 10_000_000))}"
                    )
                elif row["fourcc"] in ("PROJ", "PRMS"):
                    snippet = (
                        read_first_uncompressed(mm, row, 4000)
                        .decode("utf-8", "replace")[:300]
                        .replace("\n", " ")
                    )
                    extra = f" prefix={snippet!r}"
                print(
                    f"    row {row['fourcc']} src_gen={row['source_generation']} "
                    f"{row['compression']} uuid={row['uuid'][:8]}...{extra}"
                )
            except Exception as error:
                print(
                    f"    row {row['fourcc']} src_gen={row['source_generation']} "
                    f"ERROR {type(error).__name__}: {error}"
                )


def inspect_summary(path: Path, mm, superblock, selected, lineage) -> None:
    role = ROLE.get(u32(superblock, 20), f"role {u32(superblock, 20)}")

    current_ckpt = None
    current_metr = None
    current_uuid = None
    current_gen = selected["generation"] if selected else None
    superseded = []
    snapshots = []
    previous_end = 65536
    live_origin = {}
    tombstoned_by = {}
    for commit in reversed(lineage):
        generation, rows = decode_index(mm, commit)
        ckpt = ckpt_info(mm, live_row(rows, "CKPT"))
        metr = metr_info(mm, live_row(rows, "METR"))
        new_tombstones = []
        for row in rows:
            key = (row["fourcc"], row["uuid"])
            if row["row_kind"] == "Tombstone" and row["source_generation"] == generation:
                origin = live_origin.get(key)
                new_tombstones.append(
                    {
                        "fourcc": row["fourcc"],
                        "uuid": row["uuid"],
                        "origin_generation": origin,
                    }
                )
                if origin is not None:
                    tombstoned_by.setdefault(origin, generation)
            elif row["row_kind"] == "Live":
                live_origin[key] = row["source_generation"]
        snapshots.append(
            {
                "generation": generation,
                "kind": commit["kind"],
                "time": format_time(commit["wallclock_unix_ns"]),
                "end": commit["committed_file_end"],
                "added": commit["committed_file_end"] - previous_end,
                "ckpt": ckpt,
                "metr": metr,
                "tombstones": new_tombstones,
            }
        )
        previous_end = commit["committed_file_end"]
        if selected and generation == selected["generation"]:
            current_ckpt = ckpt
            current_metr = metr
            current_uuid = ckpt.get("uuid") if ckpt else None

    for item in snapshots:
        ckpt = item["ckpt"]
        if not ckpt or item["generation"] == current_gen:
            continue
        if ckpt.get("source_generation") != item["generation"]:
            continue
        if current_uuid and ckpt.get("uuid") == current_uuid:
            continue
        superseded.append(
            [
                item["generation"],
                checkpoint_uuid(ckpt),
                *checkpoint_cells(ckpt, item["metr"]),
            ]
        )

    print_table(
        ["File", "Size", "Role", "Generations", "Project"],
        [
            [
                path.name,
                format_bytes(path.stat().st_size),
                role,
                len(lineage),
                guid(superblock, 24),
            ]
        ],
        right={3},
    )
    print()

    if selected:
        current = next(
            item for item in snapshots if item["generation"] == selected["generation"]
        )
        written = "-"
        if current_ckpt and current_ckpt.get("source_generation") not in (
            None,
            selected["generation"],
        ):
            written = f"generation {current_ckpt['source_generation']}"
        elif current_ckpt:
            written = f"generation {selected['generation']}"
        iteration, gaussians, sh_degree = checkpoint_cells(current_ckpt, current_metr)
        current_rows = [
            ["Generation", selected["generation"]],
            ["Kind", current["kind"]],
            ["Saved", current["time"]],
            ["Checkpoint", iteration],
            ["Checkpoint UUID", checkpoint_uuid(current_ckpt)],
            ["Gaussians", gaussians],
            ["SH degree", sh_degree],
            ["Written in", written],
        ]
        if current_metr and current_metr.get("last_loss_iteration") is not None:
            current_rows.append(
                ["Loss through", format_count(current_metr["last_loss_iteration"])]
            )
            current_rows.append(
                ["Loss samples", format_count(current_metr.get("loss_samples"))]
            )
            if current_metr.get("last_psnr_iteration") is not None:
                current_rows.append(
                    [
                        "Last PSNR sample",
                        format_count(current_metr["last_psnr_iteration"]),
                    ]
                )
        print("Current snapshot")
        print_table(["Field", "Value"], current_rows)
        print()

    print("Generations")
    gen_rows = []
    for item in snapshots:
        iteration, gaussians, sh_degree = checkpoint_cells(item["ckpt"], item["metr"])
        gen_rows.append(
            [
                item["generation"],
                item["kind"],
                item["time"],
                format_bytes(item["end"]),
                format_bytes(item["added"]),
                checkpoint_uuid(item["ckpt"]),
                iteration,
                gaussians,
                sh_degree,
                format_tombstones(item["tombstones"]),
                generation_status(item, current_gen, current_uuid),
            ]
        )
    print_table(
        [
            "Gen",
            "Kind",
            "Saved",
            "Size",
            "Added",
            "CKPT UUID",
            "Iteration",
            "Gaussians",
            "SH",
            "Tombstones",
            "Status",
        ],
        gen_rows,
        right={0, 3, 4, 6, 7, 8},
    )

    print()
    print("Older checkpoints still on disk")
    if superseded:
        print_table(
            ["Gen", "CKPT UUID", "Iteration", "Gaussians", "SH", "Tombstoned by"],
            [
                [*row, tombstoned_by.get(row[0], "-")]
                for row in superseded
            ],
            right={0, 2, 3, 4, 5},
        )
    else:
        print("none")


def inspect(path: Path, full: bool) -> None:
    with path.open("rb") as handle, mmap.mmap(
        handle.fileno(), 0, access=mmap.ACCESS_READ
    ) as mm:
        superblock = mm[:256]
        heads = [
            parse_head(mm[offset : offset + HEAD_BYTES], slot)
            for slot, offset in enumerate(HEAD_SLOTS)
        ]
        valid_heads = [head for head in heads if head]
        selected = (
            max(valid_heads, key=lambda item: item["head_sequence"])
            if valid_heads
            else None
        )
        commits = scan_commits(mm)
        lineage = walk_lineage(mm, selected)
        if full:
            inspect_full(path, mm, superblock, heads, selected, commits, lineage)
        else:
            inspect_summary(path, mm, superblock, selected, lineage)


def generation_iteration(data: bytes, commit) -> int | None:
    _, rows = decode_index(data, commit)
    ckpt = live_row(rows, "CKPT")
    metr = live_row(rows, "METR")
    if ckpt and ckpt["compression"] != "ByteShuffleZstdFramed":
        header = decode_ckpt_header(data, ckpt)
        if header and "iteration" in header:
            return header["iteration"]
    if metr:
        info = parse_metr(read_first_uncompressed(data, metr, 10_000_000))
        if info.get("last_loss_iteration") is not None:
            return info["last_loss_iteration"]
    if ckpt:
        header = decode_ckpt_header(data, ckpt)
        if header and "iteration" in header:
            return header["iteration"]
    return None


def restore(source: Path, destination: Path, generation: int | None, iteration: int | None) -> None:
    data = source.read_bytes()
    existing = parse_head(data[HEAD_SLOTS[0] : HEAD_SLOTS[0] + HEAD_BYTES])
    if existing is None:
        raise SystemExit("source is missing a valid head slot 0")
    computed = crc32c(data[HEAD_SLOTS[0] : HEAD_SLOTS[0] + 4092])
    if existing["head_crc32c"] != computed:
        raise SystemExit(
            f"CRC32C self-test failed: file=0x{existing['head_crc32c']:08x} "
            f"computed=0x{computed:08x}"
        )

    by_gen = {commit["generation"]: commit for commit in scan_commits(data)}
    if generation is None:
        matches = [
            gen
            for gen, commit in sorted(by_gen.items())
            if generation_iteration(data, commit) == iteration
        ]
        if not matches:
            raise SystemExit(f"no generation has training iteration {iteration}")
        generation = matches[-1]
        print(f"iteration {iteration} found at generation(s) {matches}; using {generation}")
    if generation not in by_gen:
        raise SystemExit(f"generation {generation} commit not found")

    commit = by_gen[generation]
    commit_crc = crc32c(commit["raw"][:252])
    if commit_crc != commit["crc32c"]:
        raise SystemExit(
            f"commit CRC mismatch file=0x{commit['crc32c']:08x} computed=0x{commit_crc:08x}"
        )
    print(
        f"found gen {generation} {commit['kind']} at 0x{commit['offset']:x} "
        f"end={commit['committed_file_end']:,} uuid={commit['commit_uuid']}"
    )

    _, rows = decode_index(data, commit)
    ckpt = live_row(rows, "CKPT")
    metr = live_row(rows, "METR")
    thmb = live_row(rows, "THMB")
    if metr:
        print("METR:", parse_metr(read_first_uncompressed(data, metr, 10_000_000)))
    if ckpt:
        print(
            f"live CKPT uuid={ckpt['uuid']} src_gen={ckpt['source_generation']} "
            f"stored={ckpt['stored_bytes']:,} uncompressed={ckpt['uncompressed_bytes']:,} "
            f"{ckpt['compression']}"
        )
        print("CKPT header:", decode_ckpt_header(data, ckpt))

    preview = None
    if thmb:
        preview = {
            "offset": thmb["payload_offset"],
            "bytes": thmb["stored_bytes"],
            "format": 1,
        }
    out = bytearray(data[: commit["committed_file_end"]])
    out[HEAD_SLOTS[0] : HEAD_SLOTS[0] + HEAD_BYTES] = encode_head(
        data[:256], 0, 1, commit, preview
    )
    out[HEAD_SLOTS[1] : HEAD_SLOTS[1] + HEAD_BYTES] = encode_head(
        data[:256], 1, 2, commit, preview
    )
    destination.write_bytes(out)
    print(f"wrote {destination} ({len(out):,} bytes)")

    restored = destination.read_bytes()
    selected = parse_head(restored[HEAD_SLOTS[1] : HEAD_SLOTS[1] + HEAD_BYTES])
    print(
        f"selected head: gen={selected['generation']} seq={selected['head_sequence']} "
        f"end={selected['committed_file_end']:,}"
    )
    restored_commit = parse_commit(
        restored[selected["commit_offset"] : selected["commit_offset"] + COMMIT_BYTES],
        selected["commit_offset"],
    )
    _, restored_rows = decode_index(restored, restored_commit)
    live = [row for row in restored_rows if row["row_kind"] == "Live"]
    print("live chapters:", dict(collections.Counter(row["fourcc"] for row in live)))
    if selected["generation"] != generation:
        raise SystemExit(f"restored file selected generation {selected['generation']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a .licht file. With --output and --generation/--iteration, "
            "write that older save to a new file."
        )
    )
    parser.add_argument("source", type=Path, help="Source .licht file")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Dump raw heads, index rows, UUIDs, and chapter payloads",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write a restored .licht file (requires --generation or --iteration)",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--generation", type=int, help="Container generation to restore")
    target.add_argument(
        "--iteration",
        type=int,
        help="Training iteration of the live CKPT/METR to restore",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    args = parser.parse_args(argv)

    restoring = args.output is not None or args.generation is not None or args.iteration is not None
    if restoring:
        if args.output is None:
            parser.error("--generation/--iteration requires --output")
        if args.generation is None and args.iteration is None:
            parser.error("--output requires --generation or --iteration")
        if args.full:
            parser.error("--full cannot be used when restoring")
    elif args.force:
        parser.error("--force is only valid with --output")

    if not args.source.is_file():
        print(f"missing file: {args.source}", file=sys.stderr)
        return 1

    if restoring:
        if args.output.exists() and not args.force:
            print(f"output exists: {args.output} (pass --force)", file=sys.stderr)
            return 1
        restore(args.source, args.output, args.generation, args.iteration)
        return 0

    inspect(args.source, args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
