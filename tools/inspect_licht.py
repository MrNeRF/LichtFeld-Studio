#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Print .licht generation lineage, live chapters, and checkpoint iterations."""

from __future__ import annotations

import argparse
import collections
import datetime
import mmap
import sys
from pathlib import Path

from licht_container import (
    COMMIT_BYTES,
    HEAD_BYTES,
    HEAD_SLOTS,
    decode_ckpt_header,
    decode_index,
    guid,
    ns_to_str,
    parse_commit,
    parse_head,
    parse_metr,
    read_first_uncompressed,
    u32,
)

ROLE = {0: "master", 1: "autosave sidecar"}


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


def format_checkpoint(info: dict | None, metr: dict | None) -> str:
    if info is None:
        return "no checkpoint"
    if "iteration" in info:
        parts = [f"iteration {format_count(info['iteration'])}"]
        if "num_gaussians" in info:
            parts.append(f"{format_count(info['num_gaussians'])} gaussians")
        if "sh_degree" in info:
            parts.append(f"SH {info['sh_degree']}")
        return ", ".join(parts)
    if metr and metr.get("last_loss_iteration") is not None:
        return f"iteration {format_count(metr['last_loss_iteration'])} (from metrics)"
    if info.get("error"):
        return f"unreadable ({info['error']})"
    return "checkpoint present (header not decoded)"


def collect_commits(mm):
    commits = []
    start = 0
    while True:
        pos = mm.find(b"LFSCOMIT", start)
        if pos < 0:
            break
        commit = parse_commit(mm[pos : pos + COMMIT_BYTES], pos)
        if commit:
            commits.append(commit)
        start = pos + 1
    return commits


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
        commits = collect_commits(mm)
        lineage = walk_lineage(mm, selected)
        if full:
            inspect_full(path, mm, superblock, heads, selected, commits, lineage)
        else:
            inspect_summary(path, mm, superblock, selected, lineage)


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
    parser.add_argument(
        "--full",
        action="store_true",
        help="Dump raw heads, index rows, UUIDs, and chapter payloads",
    )
    args = parser.parse_args(argv)
    for index, path in enumerate(args.paths):
        if not path.is_file():
            print(f"missing file: {path}", file=sys.stderr)
            return 1
        if index:
            print()
        inspect(path, args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
