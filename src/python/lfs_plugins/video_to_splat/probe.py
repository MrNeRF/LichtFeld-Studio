# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decisions derived from video metadata: 360 mode, quality presets, frame scale."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

Mode = Literal["equirectangular", "perspective"]
ModeOverride = Literal["auto", "on", "off"]

# A stitched equirectangular frame is exactly 2:1; allow a little slack for odd encoders.
ASPECT_TOLERANCE = 0.02
# Photo sets: what the dataset loader and COLMAP both read.
PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png")
_JPEG_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


@dataclass(frozen=True)
class Preset:
    fps: float
    max_width: int  # 0 = keep source width
    overlap: int    # sequential matcher window


PRESETS: dict[str, Preset] = {
    "fast": Preset(fps=1.0, max_width=2880, overlap=10),
    "balanced": Preset(fps=2.0, max_width=4096, overlap=15),
    "high": Preset(fps=3.0, max_width=0, overlap=25),
}
DEFAULT_PRESET = "balanced"


def looks_equirectangular(width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    return abs(width / height - 2.0) <= 2.0 * ASPECT_TOLERANCE


def decide_mode(info: Mapping[str, object], override: ModeOverride = "auto") -> Mode:
    """Pick the COLMAP camera family from probe facts, unless the user forced one."""
    if override == "on":
        return "equirectangular"
    if override == "off":
        return "perspective"
    if bool(info.get("spherical", False)):
        return "equirectangular"
    if looks_equirectangular(int(info.get("width", 0) or 0), int(info.get("height", 0) or 0)):
        return "equirectangular"
    return "perspective"


def plan_frame_scale(width: int, max_width: int) -> float:
    """Uniform downscale so extracted frames are at most max_width wide (0 = no limit)."""
    if max_width <= 0 or width <= 0 or width <= max_width:
        return 1.0
    return max_width / width


def list_photos(folder: Path) -> list[Path]:
    """Image files directly inside ``folder``, in name order; [] when it is not a folder."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in PHOTO_EXTENSIONS)


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) from a PNG IHDR or the first JPEG SOF marker without decoding; (0, 0) when unreadable."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(24)
            if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
                width, height = struct.unpack(">II", head[16:24])
                return int(width), int(height)
            if head[:2] != b"\xff\xd8":
                return 0, 0
            handle.seek(2)
            while True:
                marker = handle.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return 0, 0
                code = marker[1]
                if code == 0xD9 or code == 0xDA:  # end of image / start of scan without a frame header
                    return 0, 0
                if code in (0xD8, 0x01) or 0xD0 <= code <= 0xD7:  # standalone markers
                    continue
                length_bytes = handle.read(2)
                if len(length_bytes) < 2:
                    return 0, 0
                (length,) = struct.unpack(">H", length_bytes)
                if code in _JPEG_SOF_MARKERS:
                    payload = handle.read(5)
                    if len(payload) < 5:
                        return 0, 0
                    height, width = struct.unpack(">HH", payload[1:5])
                    return int(width), int(height)
                handle.seek(length - 2, 1)
    except (OSError, struct.error):
        return 0, 0


def probe_photo_folder(folder: Path) -> dict:
    """Probe facts for a folder of photos, shaped like lf.video.probe: size of the first image and the count."""
    photos = list_photos(folder)
    width, height = image_size(photos[0]) if photos else (0, 0)
    return {"width": width, "height": height, "spherical": False, "count": len(photos)}
