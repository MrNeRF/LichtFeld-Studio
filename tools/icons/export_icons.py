#!/usr/bin/env python3
"""Export the checked-in icon SVG sources and the scene sprite sheet as deterministic PNGs.

README
======

To add an icon, put the upstream 24-grid SVG in
``src/visualizer/gui/assets/icon/src/`` (or its ``scene/`` subdirectory) and
keep ``currentColor`` as the glyph color.  Run this script from the repository
root with the vcpkg Python interpreter.  A source at ``src/foo.svg`` becomes
``foo.png`` beside the source tree; a source at ``src/scene/foo.svg`` becomes
``scene/foo.png``.  The script replaces ``currentColor`` with opaque white,
rasterizes at 40x40 unless ``SIZES`` lists another size, and strips PNG
metadata.

``scene/scene-sprites.png`` is assembled from the 24px scene icons using the
cell layout of the ``scene-icons`` sprite sheet in ``rml_theme.cpp``.  A cell
whose icon has an SVG source is rendered from it; otherwise the scene PNG is
used, downscaled when it is larger than the cell.

Outputs whose pixels already match are left untouched, since PNG bytes vary
between ImageMagick and zlib versions.  ``--check`` writes nothing and exits
with status 1 when any output is missing or differs.
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ICON_ROOT = REPOSITORY_ROOT / "src/visualizer/gui/assets/icon"
SOURCE_ROOT = ICON_ROOT / "src"
SPRITE_LAYOUT_SOURCE = REPOSITORY_ROOT / "src/visualizer/gui/rmlui/rml_theme.cpp"
SPRITE_SHEET = ICON_ROOT / "scene/scene-sprites.png"
SPRITE_CELL_RE = re.compile(r'"\s*icon-([a-z-]+):\s+(\d+)px\s+(\d+)px\s+(\d+)px\s+(\d+)px;')
SIZE = 40
SIZES = {
    "broom.png": 64,
    "cursor/select-add.png": 32,
    "cursor/select-intersect.png": 32,
    "cursor/select-remove.png": 32,
    "gallery-error.png": 24,
    "gallery-pause.png": 24,
    "gallery-progress-ring.png": 24,
    "layout-grid.png": 64,
    "layout-list.png": 64,
    "lichtfeld.png": 96,
}


def imagemagick() -> str:
    executable = shutil.which("magick") or shutil.which("convert")
    if executable is None:
        raise RuntimeError("ImageMagick executable not found (tried 'magick' and 'convert')")
    return executable


def normalize(png: bytes) -> bytes:
    return subprocess.run(
        [imagemagick(), "png:-", "-strip", "-depth", "8", "PNG32:-"],
        input=png,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout


def pixels(png: bytes) -> tuple[tuple[int, int], bytes]:
    size = struct.unpack(">II", png[16:24])
    rgba = subprocess.run(
        [imagemagick(), "png:-", "-depth", "8", "rgba:-"],
        input=png,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    return size, rgba


def render_svg(source: Path, size: int) -> bytes:
    svg = source.read_text(encoding="utf-8").replace("currentColor", "#FFFFFF")
    with tempfile.TemporaryDirectory() as temporary_directory:
        transformed_path = Path(temporary_directory) / "transformed.svg"
        rendered_path = Path(temporary_directory) / "rendered.png"
        transformed_path.write_text(svg, encoding="utf-8")
        subprocess.run(
            ["rsvg-convert", "-w", str(size), "-h", str(size), "-o", str(rendered_path), str(transformed_path)],
            check=True,
        )
        rendered = rendered_path.read_bytes()
    return normalize(rendered)


def sprite_cells() -> list[tuple[str, int, int, int]]:
    text = SPRITE_LAYOUT_SOURCE.read_text(encoding="utf-8")
    start = text.index("@spritesheet scene-icons")
    block = text[start : text.index("}}", start)]
    cells = []
    for name, x, y, width, height in SPRITE_CELL_RE.findall(block):
        assert width == height, f"sprite {name} is not square"
        cells.append((name, int(x), int(y), int(width)))
    if not cells:
        raise RuntimeError(f"no scene-icons cells found in {SPRITE_LAYOUT_SOURCE}")
    return cells


def render_sprite_sheet() -> bytes:
    cells = sprite_cells()
    width = max(x + size for _, x, _, size in cells)
    height = max(y + size for _, _, y, size in cells)
    with tempfile.TemporaryDirectory() as temporary_directory:
        command = [imagemagick(), "-size", f"{width}x{height}", "xc:none"]
        for name, x, y, size in cells:
            source = SOURCE_ROOT / "scene" / f"{name}.svg"
            cell_path = Path(temporary_directory) / f"{name}.png"
            if source.exists():
                cell_path.write_bytes(render_svg(source, size))
                cell = [str(cell_path)]
            else:
                icon = ICON_ROOT / "scene" / f"{name}.png"
                cell = ["(", str(icon), "-resize", f"{size}x{size}>", ")"]
            command += [*cell, "-geometry", f"+{x}+{y}", "-composite"]
        command += ["-strip", "-depth", "8", "PNG32:-"]
        return subprocess.run(command, stdout=subprocess.PIPE, check=True).stdout


def outputs():
    for source in sorted(SOURCE_ROOT.rglob("*.svg")):
        relative = source.relative_to(SOURCE_ROOT).with_suffix(".png")
        yield ICON_ROOT / relative, lambda source=source, relative=relative: render_svg(
            source, SIZES.get(relative.as_posix(), SIZE)
        )
    yield SPRITE_SHEET, render_sprite_sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report differences without writing")
    arguments = parser.parse_args()

    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"missing icon source directory: {SOURCE_ROOT}")

    stale = []
    for destination, render in outputs():
        content = render()
        label = destination.relative_to(REPOSITORY_ROOT)
        if destination.exists() and pixels(destination.read_bytes()) == pixels(content):
            print(f"Unchanged {label}")
            continue
        stale.append(label)
        if arguments.check:
            print(f"Differs {label}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            print(f"Wrote {label}")
    return 1 if arguments.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
