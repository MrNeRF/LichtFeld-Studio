# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Committed icon PNGs and the scene sprite sheet stay reproducible from their sources."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORTER = PROJECT_ROOT / "tools" / "icons" / "export_icons.py"


def _exporter():
    spec = importlib.util.spec_from_file_location("export_icons", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_scene_sprite_cell_has_an_icon_to_build_from():
    exporter = _exporter()
    cells = exporter.sprite_cells()
    assert len(cells) == 17
    for name, _x, _y, size in cells:
        assert size == 24
        assert (exporter.ICON_ROOT / "scene" / f"{name}.png").exists() or (
            exporter.SOURCE_ROOT / "scene" / f"{name}.svg"
        ).exists(), name


@pytest.mark.skipif(
    shutil.which("rsvg-convert") is None
    or (shutil.which("magick") is None and shutil.which("convert") is None),
    reason="icon export needs rsvg-convert and ImageMagick",
)
def test_committed_icons_match_their_sources():
    result = subprocess.run(
        [sys.executable, str(EXPORTER), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
