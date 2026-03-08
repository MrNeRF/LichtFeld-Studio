# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for scene panel virtualization and mutation handling."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


@pytest.fixture
def scene_panel_module():
    project_root = Path(__file__).parent.parent.parent
    build_python = project_root / "build" / "src" / "python"
    if str(build_python) not in sys.path:
        sys.path.insert(0, str(build_python))
    return import_module("lfs_plugins.scene_panel")


class _HandleStub:
    def __init__(self):
        self.records = {}
        self.dirty_fields = []

    def update_record_list(self, name, rows):
        self.records[name] = rows

    def dirty(self, name):
        self.dirty_fields.append(name)


class _ContainerStub:
    def __init__(self, height=120.0):
        self.client_height = height
        self.offset_height = height
        self.scroll_top = 0.0


def _make_row(index):
    return {
        "name": f"node-{index}",
        "id": index,
        "node_type": "SPLAT",
        "has_children": False,
        "collapsed": False,
        "visible": True,
        "label": f"Node {index}",
        "depth": 0,
        "draggable": True,
        "training_enabled": True,
    }


def test_scene_panel_prefers_consuming_mutation_flags(scene_panel_module, monkeypatch):
    panel = scene_panel_module.ScenePanel()
    lf_stub = SimpleNamespace(consume_scene_mutation_flags=lambda: 17)

    monkeypatch.setattr(scene_panel_module, "lf", lf_stub)

    assert panel._scene_mutation_flags() == 17


def test_collapsed_models_clear_virtual_rows_and_spacers(scene_panel_module, monkeypatch):
    panel = scene_panel_module.ScenePanel()
    handle = _HandleStub()
    container = _ContainerStub()

    monkeypatch.setattr(
        scene_panel_module,
        "lf",
        SimpleNamespace(get_ui_scale=lambda: 1.0),
    )

    panel._handle = handle
    panel.container = container
    panel._scene_has_nodes = True
    panel._flat_rows = [_make_row(i) for i in range(6)]

    assert panel._render_tree_window(force=True)
    assert len(handle.records["visible_rows"]) == 6
    assert panel._visible_row_capacity == 6

    panel._models_collapsed = True

    assert panel._render_tree_window(force=True)
    assert handle.records["visible_rows"] == []
    assert panel._top_spacer_height == "0dp"
    assert panel._bottom_spacer_height == "0dp"
    assert panel._visible_row_capacity == 0
