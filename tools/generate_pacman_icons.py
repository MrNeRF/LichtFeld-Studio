#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate the status-bar arcade sprites (requires Pillow)."""
from pathlib import Path
from PIL import Image, ImageDraw

OUTPUT = Path(__file__).resolve().parents[1] / "src/visualizer/gui/assets/icon/pacman"
SCALE = 8


def canvas():
    return Image.new("RGBA", (14 * SCALE, 14 * SCALE))


def save(image, name):
    image.resize((56, 56), Image.Resampling.LANCZOS).save(OUTPUT / f"{name}.png")


def box(values):
    return tuple(int(v * SCALE) for v in values)


def generate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for powered in (False, True):
        for name, angle in (("closed", 0), ("half", 18), ("open", 40)):
            image = canvas()
            draw = ImageDraw.Draw(image)
            if powered:
                draw.ellipse(box((0, 0, 13.8, 13.8)), fill="#fff39a65")
            bounds = box((0.6, 0.6, 13.4, 13.4))
            color = "#ffff78" if powered else "#ffdf00"
            if angle:
                draw.pieslice(bounds, angle, 360 - angle, fill=color)
            else:
                draw.ellipse(bounds, fill=color)
            draw.ellipse(box((7.0, 2.9, 8.3, 4.2)), fill="#080c20")
            save(image, ("power-" if powered else "") + name)

    for name, color in (("red", "#ff4545"), ("pink", "#ff9fd5"),
                        ("frightened", "#364cff")):
        frightened = name == "frightened"
        for foot in ((0,) if frightened else (0, 1)):
            image = canvas()
            draw = ImageDraw.Draw(image)
            draw.ellipse(box((1, 0.8, 13, 12)), fill=color)
            draw.rectangle(box((1, 6.5, 13, 10.8)), fill=color)
            points = [(1, 10), (1, 13), (3, 11.4 + foot * 0.8), (5, 13),
                      (7, 11.4 + foot * 0.8), (9, 13), (11, 11.4 + foot * 0.8), (13, 13), (13, 10)]
            draw.polygon([(int(x * SCALE), int(y * SCALE)) for x, y in points], fill=color)
            if frightened:
                ink = "#ffdfaa"
                for x in (4, 8.5):
                    draw.rectangle(box((x, 4, x + 1.4, 5.7)), fill=ink)
                draw.line([(int(x * SCALE), int(y * SCALE)) for x, y in
                           ((3, 9), (4.3, 7.8), (5.6, 9), (7, 7.8), (8.4, 9), (9.7, 7.8), (11, 9))],
                          fill=ink, width=SCALE)
            else:
                for x in (3, 8):
                    draw.ellipse(box((x, 3.3, x + 3.6, 7.8)), fill="#ffffff")
                    draw.ellipse(box((x, 4.6, x + 1.8, 7)), fill="#163477")
            save(image, name if frightened else name + ("-a" if foot == 0 else "-b"))

    image = canvas()
    draw = ImageDraw.Draw(image)
    for x in (2, 8):
        draw.ellipse(box((x, 4, x + 4, 8.5)), fill="#ffffff")
        draw.ellipse(box((x, 5, x + 1.8, 7.4)), fill="#163477")
    save(image, "eyes")

    image = canvas()
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
