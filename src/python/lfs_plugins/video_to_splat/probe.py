# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decisions derived from video metadata: 360 mode, quality presets, frame scale."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

Mode = Literal["equirectangular", "perspective"]
ModeOverride = Literal["auto", "on", "off"]

# A stitched equirectangular frame is exactly 2:1; allow a little slack for odd encoders.
ASPECT_TOLERANCE = 0.02


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
