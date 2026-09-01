#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Print .licht generation lineage, live chapters, and checkpoint iterations."""

from __future__ import annotations

import argparse
import collections
import mmap
import sys
from pathlib import Path

from licht_container import (
    COMMIT_BYTES,
    HEAD_BYTES,
    HEAD_SLOTS,
    decode_index,
    guid,
    ns_to_str,
    parse_ckpt_header,
    parse_commit,
    parse_head,
    parse_metr,
    read_first_uncompressed,
    u32,
)


def inspect(path: Path) -> None:
    print("=" * 80)
    print(path.name, f"{path.stat().st_size:,} bytes")
    with path.open("rb") as handle, mmap.mmap(
        handle.fileno(), 0, access=mmap.ACCESS_READ
    ) as mm:
        superblock = mm[:256]
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
        heads = []
        for slot, offset in enumerate(HEAD_SLOTS):
            head = parse_head(mm[offset : offset + HEAD_BYTES], slot)
            print(" head", slot, head)
            if head:
                heads.append(head)
        selected = max(heads, key=lambda item: item["head_sequence"]) if heads else None
        print(" selected head", selected)

        commits_found = []
        start = 0
        while True:
            pos = mm.find(b"LFSCOMIT", start)
            if pos < 0:
                break
            commit = parse_commit(mm[pos : pos + COMMIT_BYTES], pos)
            if commit:
                commits_found.append(commit)
            start = pos + 1
        print(f" scanned LFSCOMIT count={len(commits_found)}")
        for commit in commits_found:
            print(
                f"  gen={commit['generation']} kind={commit['kind']} "
                f"offset=0x{commit['offset']:x} end={commit['committed_file_end']:,} "
                f"parent_off=0x{commit['parent_commit_offset']:x} "
                f"time={ns_to_str(commit['wallclock_unix_ns'])}"
            )

        lineage = []
        if selected:
            offset = selected["commit_offset"]
            seen = set()
            while offset and offset not in seen:
                seen.add(offset)
                commit = parse_commit(mm[offset : offset + COMMIT_BYTES], offset)
                if not commit:
                    print(" broken lineage at", offset)
                    break
                lineage.append(commit)
                if commit["generation"] == 1 or commit["parent_commit_offset"] == 0:
                    break
                offset = commit["parent_commit_offset"]
        print(
            f" lineage generations newest->oldest: "
            f"{[commit['generation'] for commit in lineage]}"
        )

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
                print(
                    f"    {key[0]:4} {key[1]:9} source_gen={key[2]} count={count}"
                )
            for row in rows:
                if row["row_kind"] != "Live":
                    continue
                extra = (
                    f" stored={row['stored_bytes']:,} "
                    f"uncompressed={row['uncompressed_bytes']:,}"
                )
                try:
                    if row["fourcc"] == "CKPT":
                        header = parse_ckpt_header(
                            read_first_uncompressed(mm, row, 64)
                        )
                        extra = f"{extra} CKPT={header}"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect .licht generations, chapters, and checkpoint iterations."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more .licht files",
    )
    args = parser.parse_args(argv)
    for path in args.paths:
        if not path.is_file():
            print(f"missing file: {path}", file=sys.stderr)
            return 1
        inspect(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
