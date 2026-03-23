# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the retained export panel data model."""

from enum import IntEnum
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _make_node(node_type, name, gaussian_count):
    return SimpleNamespace(type=node_type, name=name, gaussian_count=gaussian_count)


def _install_lf_stub(monkeypatch):
    panel_space = SimpleNamespace(
        SIDE_PANEL="SIDE_PANEL",
        FLOATING="FLOATING",
        VIEWPORT_OVERLAY="VIEWPORT_OVERLAY",
        MAIN_PANEL_TAB="MAIN_PANEL_TAB",
        SCENE_HEADER="SCENE_HEADER",
        STATUS_BAR="STATUS_BAR",
    )
    panel_height_mode = SimpleNamespace(FILL="fill", CONTENT="content")
    panel_option = SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED", HIDE_HEADER="HIDE_HEADER")
    node_type = IntEnum("NodeType", {"SPLAT": 1, "MESH": 2})
    state = SimpleNamespace(
        language=["en"],
        nodes=[],
        export_state={"active": False},
        set_panel_enabled_calls=[],
        export_calls=[],
        cancel_calls=0,
    )

    lf_stub = ModuleType("lichtfeld")
    lf_stub.scene = SimpleNamespace(NodeType=node_type)
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
        get_current_language=lambda: state.language[0],
        get_export_state=lambda: dict(state.export_state),
        set_panel_enabled=lambda panel_id, enabled: state.set_panel_enabled_calls.append((panel_id, enabled)),
        cancel_export=lambda: setattr(state, "cancel_calls", state.cancel_calls + 1),
        save_ply_file_dialog=lambda default_name: f"/tmp/{default_name}.ply",
        save_sog_file_dialog=lambda default_name: f"/tmp/{default_name}.sog",
        save_spz_file_dialog=lambda default_name: f"/tmp/{default_name}.spz",
        save_html_file_dialog=lambda default_name: f"/tmp/{default_name}.html",
    )
    lf_stub.get_scene = lambda: SimpleNamespace(get_nodes=lambda: list(state.nodes))
    lf_stub.export_scene = (
        lambda fmt, path, nodes, sh_degree:
        state.export_calls.append((fmt, path, tuple(nodes), sh_degree))
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return state


@pytest.fixture
def export_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.export_panel", None)
    sys.modules.pop("lfs_plugins", None)
    state = _install_lf_stub(monkeypatch)
    module = import_module("lfs_plugins.export_panel")
    return module, state


class _HandleStub:
    def __init__(self):
        self.records = {}
        self.dirty_fields = []

    def update_record_list(self, name, rows):
        self.records[name] = rows

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_fields.append("__all__")


def test_export_panel_builds_format_and_model_records(export_panel_module):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    panel._format = module.ExportFormat.SPZ
    panel._selected_nodes = {"Tree"}
    state.nodes = [
        _make_node(module.lf.scene.NodeType.SPLAT, "Tree", 128),
        _make_node(module.lf.scene.NodeType.SPLAT, "House", 64),
    ]

    panel._rebuild_format_records()
    panel._rebuild_model_records(state.nodes)

    assert panel._handle.records["formats"] == [
        {"index": "0", "label": "export.format.ply_standard", "selected": False},
        {"index": "1", "label": "export.format.sog_supersplat", "selected": False},
        {"index": "2", "label": "export.format.spz_niantic", "selected": True},
        {"index": "3", "label": "export.format.html_viewer", "selected": False},
    ]
    assert panel._handle.records["models"] == [
        {"name": "Tree", "selected": True, "count_text": "(128)"},
        {"name": "House", "selected": False, "count_text": "(64)"},
    ]
    assert panel._has_models is True


def test_export_panel_seeds_selection_from_scene_nodes(export_panel_module):
    module, _state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    panel._export_sh_degree = 1
    nodes = [
        _make_node(module.lf.scene.NodeType.SPLAT, "Tree", 128),
        _make_node(module.lf.scene.NodeType.SPLAT, "House", 64),
    ]

    assert panel._sync_selection(nodes) is True
    assert panel._selected_nodes == {"Tree", "House"}
    assert panel._selection_seeded is True
    assert panel._export_sh_degree == 3
    assert panel._handle.dirty_fields == ["sh_degree"]


def test_export_panel_progress_updates_bound_value(export_panel_module):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._handle = _HandleStub()
    state.export_state = {
        "active": True,
        "progress": 0.5,
        "stage": "writing",
        "format": "ply",
    }

    assert panel._update_export_progress() is True
    assert panel._progress_value == "0.5"
    assert panel._cached_export_state["stage"] == "writing"
    assert panel._handle.dirty_fields == [
        "progress_value",
        "progress_pct",
        "progress_stage",
    ]


def test_export_panel_closes_when_export_finishes(export_panel_module):
    module, state = export_panel_module
    panel = module.ExportPanel()
    panel._exporting = True
    panel._selection_seeded = True
    state.export_state = {"active": False}

    assert panel._update_export_progress() is True
    assert panel._exporting is False
    assert panel._selection_seeded is False
    assert state.set_panel_enabled_calls == [("lfs.export", False)]
