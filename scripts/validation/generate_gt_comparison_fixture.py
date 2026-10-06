#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate deterministic COLMAP fixtures for PR 2623; no third-party modules."""
import argparse
import math
import struct
import zlib
from pathlib import Path


def png(path, width, height, camera, large=False):
    # Yellow fiducials correspond to the COLMAP points, so point-cloud crop
    # alignment is observable as well as source identity and orientation.
    markers = {}
    for row in range(-5, 6):
        for col in range(-7, 8):
            x, y = (col * .1 + camera * .1) / 2.0, row * .1 / 2.0
            if large:
                r2 = x*x + y*y
                radial = 1 - .08*r2 + .01*r2*r2
                x, y = (x*radial + 2*.002*x*y - .003*(r2 + 2*x*x),
                        y*radial + .002*(r2 + 2*y*y) - 2*.003*x*y)
            px = math.floor(width * (.8*x + .47))
            py = math.floor(height * (1.2*y + .53))
            radius = 3 if not large else 7
            for yy in range(max(0, py-radius), min(height, py+radius+1)):
                markers.setdefault(yy, []).append((max(0, px-radius), min(width, px+radius+1)))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    compressor = zlib.compressobj(6)
    encoded = []
    for y in range(height):
        row = bytearray(1 + width * 3)
        for x in range(width):
            # Camera identity is the red channel; green and blue identify position.
            # A white asymmetric top-left mark makes orientation observable.
            color = (224 if camera == 0 else 24, 24 + x * 207 // width, 24 + y * 207 // height)
            if x < width // 11 and y < height // 17:
                color = (255, 255, 255)
            if (x % 127 < 2 or y % 113 < 2) and x > width // 11:
                color = (color[0], 8, 8)
            offset = 1 + x * 3
            row[offset:offset + 3] = bytes(color)
        for begin, end in markers.get(y, ()):
            row[1+3*begin:1+3*end] = bytes((220, 180, 70)) * (end-begin)
        encoded.append(compressor.compress(row))
    encoded.append(compressor.flush())
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", b"".join(encoded)) + chunk(b"IEND", b""))


def generate(destination, large=False):
    width, height = (8193, 4321) if large else (1537, 1025)
    images = destination / "images"
    sparse = destination / "sparse" / "0"
    images.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    fx, fy = width * .8, height * 1.2
    cx, cy = width * .47, height * .53
    model = "OPENCV" if large else "PINHOLE"
    distortion = " -0.08 0.01 0.002 -0.003" if large else ""
    (sparse / "cameras.txt").write_text(f"1 {model} {width} {height} {fx} {fy} {cx} {cy}{distortion}\n", encoding="utf-8")
    records = []
    for camera in range(2):
        filename = f"camera_{camera}.png"
        png(images / filename, width, height, camera, large)
        records.append(f"{camera + 1} 1 0 0 0 {camera * .1} 0 0 1 {filename}\n\n")
    (sparse / "images.txt").write_text("".join(records), encoding="utf-8")
    points = []
    for row in range(-5, 6):
        for col in range(-7, 8):
            index = len(points) + 1
            points.append(f"{index} {col * .1} {row * .1} 2.0 220 180 70 0.1\n")
    (sparse / "points3D.txt").write_text("".join(points), encoding="utf-8")
    print(f"Created {destination.resolve()}: two {width}x{height} {model} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--large", action="store_true", help="Generate odd-sized 8K distorted RGB8 sources")
    args = parser.parse_args()
    generate(args.destination, args.large)
