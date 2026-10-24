#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Draw the original collision voxel icon and update its reserved atlas slot (Pillow)."""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1] / "src/visualizer/gui/assets/icon/scene"
SCALE = 8
image = Image.new("RGBA", (24 * SCALE, 24 * SCALE))
draw = ImageDraw.Draw(image)


def line(points, width=1.65):
    draw.line([(round(x * SCALE), round(y * SCALE)) for x, y in points],
              fill="#ffffff", width=round(width * SCALE), joint="curve")


line([(12, 2.5), (21, 7.5), (21, 16.5), (12, 21.5), (3, 16.5), (3, 7.5), (12, 2.5)])
line([(3, 7.5), (12, 12.5), (21, 7.5)])
line([(12, 12.5), (12, 21.5)])
line([(7.5, 5), (16.5, 10), (16.5, 19)], 1.1)
line([(7.5, 19), (7.5, 10), (16.5, 5)], 1.1)
line([(3, 12), (12, 17), (21, 12)], 1.1)
image = image.resize((24, 24), Image.Resampling.LANCZOS)
image.save(ROOT / "collision.png")
atlas = Image.open(ROOT / "scene-sprites.png").convert("RGBA")
atlas.paste(image, (120, 48))
atlas.save(ROOT / "scene-sprites.png")
