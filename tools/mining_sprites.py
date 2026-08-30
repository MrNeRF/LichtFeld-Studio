#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate the 16 mining-progress-bar PNG sprites."""

from __future__ import annotations

import argparse
import random
import struct
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "src" / "visualizer" / "gui" / "assets" / "icon" / "mining"

MINER_PAL = {
    "H": (58, 38, 24, 255),
    "S": (198, 148, 106, 255),
    "E": (46, 38, 100, 255),
    "T": (0, 158, 158, 255),
    "t": (0, 128, 128, 255),
    "P": (70, 84, 168, 255),
    "p": (52, 62, 128, 255),
    "F": (90, 90, 96, 255),
    "W": (126, 86, 48, 255),
    "g": (150, 152, 160, 255),
    "K": (255, 200, 48, 255),
    "k": (255, 236, 140, 255),
    ".": (0, 0, 0, 0),
}

MINER_RAISED = [
    "........ggg...",
    ".HHHHH.g.W.g..",
    ".HSSES..W.....",
    ".HSSSS.S......",
    "..TTTTT.......",
    "..tTTT........",
    "..tTTT........",
    "..tTTT........",
    "..tTTT........",
    "..pPPP........",
    "..pPPP........",
    "..pPPP........",
    "..pPPP........",
    "..FFFF........",
]

MINER_STRIKE = [
    "..............",
    ".HHHHH........",
    ".HSSES........",
    ".HSSSS...g....",
    "..TTTT....g.k.",
    "..tTTTTSWWg.K.",
    "..tTTT....g.k.",
    "..tTTT...g....",
    "..tTTT........",
    "..pPPP........",
    "..pPPP........",
    "..pPPP........",
    "..pPPP........",
    "..FFFF........",
]

STONE = [(125, 125, 125), (110, 110, 110), (96, 96, 96)]
DIRT = [(134, 96, 67), (115, 81, 53), (97, 67, 42)]

ORES = {
    "coal": {"blob": [(38, 38, 38), (58, 58, 58)]},
    "iron": {"blob": [(216, 175, 147), (178, 140, 116)]},
    "gold": {"blob": [(250, 220, 96), (202, 172, 60)]},
    "diamond": {"blob": [(102, 219, 214), (58, 178, 190)]},
}
ORE_SPOTS = [(3, 3), (10, 2), (6, 7), (12, 10), (2, 12), (9, 13)]

CRACK_STAGES = [
    [(7, 7), (8, 8), (8, 6), (6, 9), (9, 9)],
    [
        (7, 7), (8, 8), (8, 6), (6, 9), (9, 9), (5, 6), (4, 5), (9, 4), (10, 3),
        (10, 10), (11, 11), (6, 11), (5, 13), (12, 8), (3, 7),
    ],
    [
        (7, 7), (8, 8), (8, 6), (6, 9), (9, 9), (5, 6), (4, 5), (9, 4), (10, 3),
        (10, 10), (11, 11), (6, 11), (5, 13), (12, 8), (3, 7), (3, 4), (2, 3),
        (11, 2), (12, 1), (13, 9), (14, 11), (4, 14), (2, 8), (1, 9), (7, 2),
        (7, 1), (13, 7), (8, 13), (9, 14), (2, 12), (14, 4),
    ],
]

DEBRIS_GRAY = (105, 105, 105)
DEBRIS_DARK = (78, 78, 78)
DEBRIS_BROWN = (115, 81, 53)
SPARK = (255, 236, 140)

BREAK_FRAMES = [
    {
        "chunks": [
            (2, 2, DEBRIS_GRAY, 2), (9, 1, DEBRIS_DARK, 2), (4, 7, DEBRIS_BROWN, 2),
            (11, 6, DEBRIS_GRAY, 2), (2, 11, DEBRIS_DARK, 2), (8, 10, DEBRIS_GRAY, 2),
            (13, 11, DEBRIS_BROWN, 1), (6, 4, DEBRIS_DARK, 1), (13, 3, SPARK, 1),
        ],
    },
    {
        "chunks": [
            (1, 6, DEBRIS_GRAY, 2), (10, 5, DEBRIS_DARK, 1), (5, 10, DEBRIS_BROWN, 2),
            (12, 10, DEBRIS_GRAY, 1), (3, 13, DEBRIS_DARK, 1), (8, 13, DEBRIS_GRAY, 2),
            (14, 13, DEBRIS_BROWN, 1), (6, 7, SPARK, 1),
        ],
    },
    {
        "chunks": [
            (2, 14, DEBRIS_DARK, 1), (6, 13, DEBRIS_GRAY, 1), (9, 14, DEBRIS_BROWN, 1),
            (13, 14, DEBRIS_GRAY, 1), (11, 12, SPARK, 1),
        ],
    },
]

GEM_W, GEM_L, GEM_C = (245, 255, 255), (150, 240, 235), (70, 200, 210)
GEM_D = (40, 150, 170)
GEM = [
    "...LLLLLL...",
    "..LWWLLLLC..",
    ".LWWLLLLLCC.",
    "LWLLLLLLCCCC",
    "LLLLLLLCCCCC",
    ".LLLLLCCCCC.",
    ".LLLLCCCCCC.",
    "..LLLCCCCC..",
    "..LLCCCCCC..",
    "...LCCCCC...",
    "....LCCC....",
    ".....CC.....",
]
GEM_PAST = {
    GEM_W: (205, 210, 210),
    GEM_L: (150, 160, 162),
    GEM_C: (95, 110, 115),
    GEM_D: (60, 75, 80),
}

BLOCK_SEEDS = {
    "stone": 7,
    "dirt": 11,
    "coal": 13,
    "iron": 17,
    "gold": 19,
    "diamond": 23,
}


def write_png(path: Path, pixels: list[list[tuple[int, int, int, int]]]) -> None:
    height, width = len(pixels), len(pixels[0])
    raw = b"".join(
        b"\x00" + b"".join(struct.pack("4B", *pixel) for pixel in row)
        for row in pixels
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def render_miner(grid: list[str], scale: int) -> list[list[tuple[int, int, int, int]]]:
    height = len(grid) * scale
    width = len(grid[0]) * scale
    return [
        [MINER_PAL[grid[y // scale][x // scale]] for x in range(width)]
        for y in range(height)
    ]


def upscale(grid: list[list[tuple[int, int, int] | None]], scale: int) -> list[list[tuple[int, int, int, int]]]:
    height, width = len(grid) * scale, len(grid[0]) * scale
    out: list[list[tuple[int, int, int, int]]] = []
    for y in range(height):
        row: list[tuple[int, int, int, int]] = []
        for x in range(width):
            color = grid[y // scale][x // scale]
            row.append((0, 0, 0, 0) if color is None else (*color, 255))
        out.append(row)
    return out


def mottle(base: list[tuple[int, int, int]], seed: int) -> list[list[tuple[int, int, int]]]:
    rng = random.Random(seed)
    pixels = [[base[0] for _ in range(16)] for _ in range(16)]
    for _ in range(26):
        x, y = rng.randrange(15), rng.randrange(15)
        shade = base[rng.choice([1, 1, 2])]
        for dy in range(rng.choice([1, 2])):
            for dx in range(rng.choice([1, 2, 2])):
                pixels[min(y + dy, 15)][min(x + dx, 15)] = shade
    for i in range(16):
        pixels[15][i] = base[2]
        pixels[i][15] = base[2]
        pixels[0][i] = base[1] if pixels[0][i] == base[0] else pixels[0][i]
    return pixels


def block(kind: str, seed: int = 7) -> list[list[tuple[int, int, int]]]:
    if kind == "dirt":
        return mottle(DIRT, seed)
    pixels = mottle(STONE, seed)
    if kind in ORES:
        blob = ORES[kind]["blob"]
        for x, y in ORE_SPOTS:
            pixels[y][x] = blob[0]
            pixels[y][min(x + 1, 15)] = blob[0]
            pixels[min(y + 1, 15)][x] = blob[1]
            pixels[min(y + 1, 15)][min(x + 1, 15)] = blob[0]
    return pixels


def crack_overlay(stage: int) -> list[list[tuple[int, int, int] | None]]:
    pixels: list[list[tuple[int, int, int] | None]] = [[None] * 16 for _ in range(16)]
    for x, y in CRACK_STAGES[stage]:
        pixels[y][x] = (22, 22, 22)
    return pixels


def break_frame(index: int) -> list[list[tuple[int, int, int] | None]]:
    pixels: list[list[tuple[int, int, int] | None]] = [[None] * 16 for _ in range(16)]
    for x, y, color, size in BREAK_FRAMES[index]["chunks"]:
        for dy in range(size):
            for dx in range(size):
                pixels[min(y + dy, 15)][min(x + dx, 15)] = color
    return pixels


def gem(past: bool = False) -> list[list[tuple[int, int, int] | None]]:
    pal = {"W": GEM_W, "L": GEM_L, "C": GEM_C, ".": None}
    pixels: list[list[tuple[int, int, int] | None]] = [
        [pal[ch] for ch in row] for row in GEM
    ]
    for y in range(12):
        for x in range(12):
            if pixels[y][x] == GEM_C:
                right = x == 11 or pixels[y][x + 1] is None
                below = y == 11 or pixels[y + 1][x] is None
                if right or below:
                    pixels[y][x] = GEM_D
    if past:
        pixels = [[GEM_PAST[color] if color else None for color in row] for row in pixels]
    return pixels


def write_sprites(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_png(output_dir / "miner-raised.png", render_miner(MINER_RAISED, 8))
    write_png(output_dir / "miner-strike.png", render_miner(MINER_STRIKE, 8))
    for kind, name in (
        ("stone", "stone.png"),
        ("dirt", "dirt.png"),
        ("coal", "coal.png"),
        ("iron", "iron.png"),
        ("gold", "gold.png"),
        ("diamond", "diamond-ore.png"),
    ):
        write_png(output_dir / name, upscale(block(kind, seed=BLOCK_SEEDS[kind]), 8))
    for index in range(3):
        write_png(output_dir / f"crack-{index + 1}.png", upscale(crack_overlay(index), 8))
        write_png(output_dir / f"break-{index + 1}.png", upscale(break_frame(index), 8))
    write_png(output_dir / "gem.png", upscale(gem(False), 8))
    write_png(output_dir / "gem-collected.png", upscale(gem(True), 8))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate mining progress-bar sprites.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(DEFAULT_OUT),
        help="Directory to write the 16 PNG files",
    )
    args = parser.parse_args()
    write_sprites(Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
