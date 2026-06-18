# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the retained video extractor panel."""

import importlib.util
import sys
import types
from pathlib import Path


def _load_video_extractor_panel(monkeypatch):
    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "src" / "python" / "lfs_plugins" / "video_extractor_panel.py"

    lf = types.ModuleType("lichtfeld")
    lf.ui = types.SimpleNamespace(
        PanelSpace=types.SimpleNamespace(FLOATING=1),
        PanelHeightMode=types.SimpleNamespace(CONTENT=1),
        tr=lambda key: "Example: %s%s" if key == "video_extractor.example" else key,
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf)

    package = types.ModuleType("lfs_plugins")
    package.__path__ = [str(module_path.parent)]
    monkeypatch.setitem(sys.modules, "lfs_plugins", package)

    types_module = types.ModuleType("lfs_plugins.types")
    types_module.Panel = object
    monkeypatch.setitem(sys.modules, "lfs_plugins.types", types_module)

    spec = importlib.util.spec_from_file_location(
        "lfs_plugins.video_extractor_panel_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_filename_stem_handles_supported_and_invalid_patterns(monkeypatch):
    panel = _load_video_extractor_panel(monkeypatch)
    make_stem = panel._make_frame_filename_stem

    assert make_stem("frame_%d", 7) == "frame_7"
    assert make_stem("frame_%04d", 7) == "frame_0007"
    assert make_stem("frame_%%d", 7) == "frame_%d_7"
    assert make_stem("frame_%s", 7) == "frame_%s_7"
    assert make_stem("", 7) == "frame_7"


def test_pattern_example_matches_safe_filename_generation(monkeypatch):
    panel = _load_video_extractor_panel(monkeypatch)

    instance = panel.VideoExtractorPanel()
    instance._format_index = 1
    instance._filename_pattern = "clip_%04d"

    assert instance._get_pattern_example() == "Example: clip_0001.jpg"
