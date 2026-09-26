# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the dense-matching wrapper that the densification pipeline uses.

Everything here runs without a GPU and without model weights. The point of this
module is that it depends on nothing but the native tensor library, so the tests
guard that as much as the behaviour.
"""

import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_PYTHON = PROJECT_ROOT / "src" / "python"
MODULE = SRC_PYTHON / "lfs_plugins" / "dense_matching.py"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))


def _imported_modules() -> set:
    tree = ast.parse(MODULE.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_module_depends_on_nothing_we_do_not_ship():
    # LichtFeld Studio ships no third-party array library, so the seam has to
    # go through the native tensor library alone.
    allowed = {"lichtfeld", "typing", "__future__"}
    assert _imported_modules() <= allowed, _imported_modules() - allowed


def test_module_never_reaches_for_numpy_or_torch():
    text = MODULE.read_text()
    for banned in ("numpy", "torch", "PIL"):
        assert f"import {banned}" not in text, banned


# The rest needs the compiled module on the path.
pytest.importorskip("lichtfeld")

from lfs_plugins.dense_matching import DenseMatcher, has_weights, weights_path  # noqa: E402


def test_unknown_setting_is_rejected():
    with pytest.raises(ValueError, match="unknown setting"):
        DenseMatcher(setting="turbo")


def test_every_setting_maps_to_a_usable_resolution():
    assert DenseMatcher.SETTINGS
    for name, resolution in DenseMatcher.SETTINGS.items():
        assert isinstance(resolution, int) and resolution > 0, name
        # The baked position embedding needs a multiple of patch 14 by stride 8.
        assert resolution % 112 == 0, name


def test_weights_helpers_agree():
    path = weights_path()
    assert has_weights() == (path is not None)
    if path is not None:
        assert path.endswith("romav1.lfw")


def test_matcher_surface_covers_the_densification_pipeline():
    # The densification pipeline needs exactly these entry points.
    for name in ("prepare", "match_grids", "match_grids_batch", "match_pairs", "close",
                 "__enter__", "__exit__"):
        assert callable(getattr(DenseMatcher, name)), name
    for name in ("resolution", "is_loaded"):
        assert isinstance(getattr(DenseMatcher, name), property), name


def test_native_matcher_exposes_what_the_wrapper_calls():
    import lichtfeld as lf
    for name in ("prepare", "match_gpu", "match_grid", "close"):
        assert hasattr(lf.nn.RomaV1, name), name
    assert callable(lf.nn.romav1_weights_path)
    # No numpy-returning entry points: results stay in native tensors.
    for gone in ("match", "match_images"):
        assert not hasattr(lf.nn.RomaV1, gone), gone
