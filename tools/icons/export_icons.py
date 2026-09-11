#!/usr/bin/env python3
"""Export the checked-in toolbar SVG sources as deterministic 40px PNGs.

README
======

To add an icon, put the upstream 24-grid SVG in
``src/visualizer/gui/assets/icon/src/`` (or its ``scene/`` subdirectory) and
keep ``currentColor`` as the glyph color.  Run this script from the repository
root with the vcpkg Python interpreter.  A source at ``src/foo.svg`` becomes
``foo.png`` beside the source tree; a source at ``src/scene/foo.svg`` becomes
``scene/foo.png``.  The script replaces ``currentColor`` with opaque white,
rasterizes at the fixed 40x40 target size, and strips PNG metadata.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ICON_ROOT = REPOSITORY_ROOT / "src/visualizer/gui/assets/icon"
SOURCE_ROOT = ICON_ROOT / "src"
SIZE = 40


def render_svg(source: Path) -> bytes:
    svg = source.read_text(encoding="utf-8").replace("currentColor", "#FFFFFF")
    magick = shutil.which("magick") or shutil.which("convert")
    if magick is None:
        raise RuntimeError("ImageMagick executable not found (tried 'magick' and 'convert')")

    with tempfile.TemporaryDirectory() as temporary_directory:
        transformed_path = Path(temporary_directory) / "transformed.svg"
        rendered_path = Path(temporary_directory) / "rendered.png"
        transformed_path.write_text(svg, encoding="utf-8")
        subprocess.run(
            ["rsvg-convert", "-w", str(SIZE), "-h", str(SIZE), "-o", str(rendered_path), str(transformed_path)],
            check=True,
        )
        rendered = rendered_path.read_bytes()
    normalized = subprocess.run(
        [magick, "png:-", "-strip", "-depth", "8", "PNG32:-"],
        input=rendered,
        stdout=subprocess.PIPE,
        check=True,
    )
    return normalized.stdout


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"missing icon source directory: {SOURCE_ROOT}")

    for source in sorted(SOURCE_ROOT.rglob("*.svg")):
        relative = source.relative_to(SOURCE_ROOT).with_suffix(".png")
        destination = ICON_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = render_svg(source)
        if destination.exists() and destination.read_bytes() == content:
            print(f"Unchanged {destination.relative_to(REPOSITORY_ROOT)}")
        else:
            destination.write_bytes(content)
            print(f"Wrote {destination.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
