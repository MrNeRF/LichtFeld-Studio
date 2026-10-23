#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate Tessle Leafjaw's sprites from the selected CSS concept (requires Pillow)."""
from pathlib import Path
from PIL import Image, ImageDraw

OUTPUT = Path(__file__).resolve().parents[1] / "src/visualizer/gui/assets/icon/tessle"
SCALE = 8
SIZE = 28
SKIN, SHADE, INK, TOOTH = "#70d6ad", "#459c7d", "#193b34", "#fff2d2"


def canvas():
    return Image.new("RGBA", (SIZE * SCALE, SIZE * SCALE))


def box(values):
    return tuple(round(v * SCALE) for v in values)


def save(image, name, size=(56, 56)):
    image.resize(size, Image.Resampling.LANCZOS).save(OUTPUT / f"{name}.png")


def leaf_shape(draw, bounds, color):
    # Opposite rounded corners form the same leaf silhouette as the CSS concept.
    x, y, right, bottom = bounds
    points = []
    for start, control, end in (((right, y), (x, y), (x, bottom)),
                                ((x, bottom), (right, bottom), (right, y))):
        for step in range(21):
            t = step / 20
            points.append(box(tuple((1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c
                                    for a, b, c in zip(start, control, end))))
    draw.polygon(points, fill=color)


def creature(openness):
    image = canvas()
    draw = ImageDraw.Draw(image)
    leaf_shape(draw, (0, 13, 10, 21), SHADE)
    for x in (9, 19):
        draw.rounded_rectangle(box((x, 23, x + 5, 27)), radius=2 * SCALE, fill=SHADE)
    for x, y, color, angle in ((8, 0, SHADE, 20), (13, 1, SKIN, -12)):
        ear = canvas()
        leaf_shape(ImageDraw.Draw(ear), (x, y, x + 6, y + 10), color)
        image.alpha_composite(ear.rotate(angle, Image.Resampling.BICUBIC, center=box((x + 3, y + 5))))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(box((7, 16, 27, 23)), radius=2 * SCALE, fill=INK)

    jaw = canvas()
    draw = ImageDraw.Draw(jaw)
    top = 17 if openness == 0 else 18
    draw.rounded_rectangle(box((6, top, 26, top + 5)), radius=3 * SCALE, fill=SHADE)
    draw.rectangle(box((6, top, 26, top + 2)), fill=SHADE)
    draw.rectangle(box((18, top - 2, 21, top + 1)), fill=TOOTH)
    image.alpha_composite(jaw.rotate(-21 * openness, Image.Resampling.BICUBIC, center=box((8, top))))

    upper = canvas()
    draw = ImageDraw.Draw(upper)
    draw.rounded_rectangle(box((5, 5, 27, 19)), radius=8 * SCALE, fill=SKIN)
    draw.rounded_rectangle(box((5, 12, 27, 19)), radius=2 * SCALE, fill=SKIN)
    draw.ellipse(box((15, 8, 19, 12)), fill=INK)
    draw.rounded_rectangle(box((22, 17, 25, 20)), radius=SCALE // 2, fill=TOOTH)
    image.alpha_composite(upper.rotate(4 * openness, Image.Resampling.BICUBIC, center=box((5, 19))))
    return image


def generate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, openness in (("closed", 0), ("half", 0.5), ("open", 1)):
        save(creature(openness), name)

    image = Image.new("RGBA", (6 * SCALE, 4 * SCALE))
    leaf_shape(ImageDraw.Draw(image), (0, 0, 6, 4), "#b1d2ad")
    save(image, "leaf", (48, 32))

    # Preserve the checkpoint fruit and its existing editing hit target.
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
