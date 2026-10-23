#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate Tessle's sprites from the selected CSS concept (requires Pillow)."""
from pathlib import Path
from PIL import Image, ImageDraw

OUTPUT = Path(__file__).resolve().parents[1] / "src/visualizer/gui/assets/icon/tessle"
SCALE = 8
SIZE = 18


def canvas():
    return Image.new("RGBA", (SIZE * SCALE, SIZE * SCALE))


def box(values):
    return tuple(round(v * SCALE) for v in values)


def save(image, name):
    image.resize((56, 56), Image.Resampling.LANCZOS).save(OUTPUT / f"{name}.png")


def robot(powered, openness):
    image = canvas()
    draw = ImageDraw.Draw(image)
    skin, shade, visor = ("#b3ffe1", "#76b6a0", "#713f77") if powered else ("#62d7af", "#2f9d83", "#153c3b")
    draw.rounded_rectangle(box((0, 5, 2, 12)), radius=SCALE, fill=shade)
    draw.rectangle(box((5, 14, 8, 16)), fill=shade)
    draw.rectangle(box((11, 14, 14, 16)), fill=shade)
    draw.rounded_rectangle(box((3, 8, 16, 13)), radius=SCALE, fill="#13272a")

    jaw = canvas()
    jaw_draw = ImageDraw.Draw(jaw)
    jaw_draw.rounded_rectangle(box((3, 10, 16, 14)), radius=SCALE, fill=shade)
    jaw_draw.rectangle(box((9, 9, 11, 11)), fill="#fff3d7")
    jaw = jaw.rotate(-10 * openness, resample=Image.Resampling.BICUBIC,
                     center=box((5, 10)), translate=box((0, 2 * openness - 1)))
    image.alpha_composite(jaw)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(box((2, 2, 16, 10)), radius=2 * SCALE, fill=skin)
    draw.rectangle(box((3, 9, 16, 10)), fill=shade)
    draw.rectangle(box((13, 9, 15, 12)), fill="#fff3d7")
    draw.rounded_rectangle(box((7, 4, 14, 7)), radius=SCALE // 2, fill=visor)
    for x in (8, 12):
        draw.rectangle(box((x, 5, x + 1.2, 6.3)), fill="#fff8e2" if powered else "#eefdf6")
    return image


def drone(color, alternate=False, powered=False):
    image = canvas()
    draw = ImageDraw.Draw(image)
    y = 0.6 if alternate else 0
    shade = "#397293" if powered else "#946240"
    draw.rectangle(box((1, 7 + y, 3, 10 + y)), fill=shade)
    draw.rectangle(box((15, 7 + y, 17, 10 + y)), fill=shade)
    draw.rounded_rectangle(box((3, 5 + y, 15, 13 + y)), radius=SCALE, fill=shade)
    draw.rounded_rectangle(box((3, 5 + y, 14, 12 + y)), radius=SCALE, fill=color)
    if powered:
        for x in (5, 10):
            draw.rectangle(box((x, 8 + y, x + 1.7, 9.7 + y)), fill="#f6f5e5")
    else:
        draw.rectangle(box((10, 8 + y, 12, 10 + y)), fill="#25313a")
    return image


def generate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for powered in (False, True):
        for name, openness in (("closed", 0), ("half", 0.5), ("open", 1)):
            save(robot(powered, openness), ("power-" if powered else "") + name)
    for name, color in (("amber", "#ddab78"), ("copper", "#c99b8e")):
        for alternate in (False, True):
            save(drone(color, alternate), f"drone-{name}-{'b' if alternate else 'a'}")
    save(drone("#79adc8", powered=True), "powered-drone")

    image = canvas()
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(box((6, 6, 12, 11)), radius=SCALE, fill="#713f77")
    draw.rectangle(box((8, 7, 10, 9)), fill="#b3ffe1")
    draw.rectangle(box((3, 8, 5, 9)), fill="#76b6a0")
    draw.rectangle(box((13, 8, 15, 9)), fill="#76b6a0")
    save(image, "drone-core")

    # Save-step cherries remain the same size as the existing marker hit targets.
    image = Image.new("RGBA", (14 * SCALE, 14 * SCALE))
    draw = ImageDraw.Draw(image)
    draw.line([box((3.5, 8)), box((8, 1)), box((10.5, 7.5))], fill="#6bd05a", width=SCALE)
    draw.ellipse(box((8, 0.3, 12.5, 2.4)), fill="#8fe471")
    for x, y in ((0.8, 7.0), (7.2, 6.0)):
        draw.ellipse(box((x, y, x + 5.5, y + 5.8)), fill="#a81739")
        draw.ellipse(box((x, y, x + 5.0, y + 4.8)), fill="#ff3655")
        draw.ellipse(box((x + 1, y + 0.7, x + 2.3, y + 1.9)), fill="#ffb2b9")
    save(image, "cherries")


if __name__ == "__main__":
    generate()
