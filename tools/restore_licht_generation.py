#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Publish an older .licht generation as a new file.

Rewrites both head slots to the chosen commit and truncates later generations.
Use this to recover a tombstoned training checkpoint that is still on disk.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from licht_container import (
    COMMIT_BYTES,
    HEAD_BYTES,
    HEAD_SLOTS,
    crc32c,
    decode_ckpt_header,
    decode_index,
    encode_head,
    parse_commit,
    parse_head,
    parse_metr,
    read_first_uncompressed,
    scan_commits,
)


def live_row(rows, fourcc: str):
    return next(
        (row for row in rows if row["fourcc"] == fourcc and row["row_kind"] == "Live"),
        None,
    )


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
        description="Restore an older .licht generation (or training iteration) to a new file."
    )
    parser.add_argument("source", type=Path, help="Source .licht file")
    parser.add_argument("destination", type=Path, help="New .licht file to write")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--generation", type=int, help="Container generation to publish")
    target.add_argument("--iteration", type=int, help="Training iteration of the live CKPT/METR")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination if it already exists",
    )
    args = parser.parse_args(argv)
    if not args.source.is_file():
        print(f"missing file: {args.source}", file=sys.stderr)
        return 1
    if args.destination.exists() and not args.force:
        print(f"destination exists: {args.destination} (pass --force)", file=sys.stderr)
        return 1
    restore(args.source, args.destination, args.generation, args.iteration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
