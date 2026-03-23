# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the retained mesh-to-splat panel."""

from enum import IntEnum
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _make_mesh_node(node_type, name, vertex_count, face_count):
    return SimpleNamespace(
        type=node_type,
        name=name,
        mesh=lambda: SimpleNamespace(vertex_count=vertex_count, face_count=face_count),
    )


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
    node_type = IntEnum("NodeType", {"MESH": 10})
    state = SimpleNamespace(
        language=["en"],
        nodes=[],
        active=False,
        progress=0.0,
        stage="",
        error="",
        mesh_to_splat_calls=[],
    )

    lf_stub = ModuleType("lichtfeld")
    lf_stub.scene = SimpleNamespace(NodeType=node_type)
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
        get_current_language=lambda: state.language[0],
    )
    lf_stub.get_scene = lambda: SimpleNamespace(get_nodes=lambda *args, **kwargs: list(state.nodes))
    lf_stub.mesh_to_splat = lambda mesh_name, **kwargs: state.mesh_to_splat_calls.append((mesh_name, kwargs))
    lf_stub.is_mesh2splat_active = lambda: state.active
    lf_stub.get_mesh2splat_progress = lambda: state.progress
    lf_stub.get_mesh2splat_stage = lambda: state.stage
    lf_stub.get_mesh2splat_error = lambda: state.error

    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return state


@pytest.fixture
def mesh2splat_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.mesh2splat_panel", None)
    sys.modules.pop("lfs_plugins", None)
    state = _install_lf_stub(monkeypatch)
    module = import_module("lfs_plugins.mesh2splat_panel")
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


class _ElementStub:
    def __init__(self, attrs=None, parent=None):
        self.attrs = dict(attrs or {})
        self._parent = parent

    def has_attribute(self, name):
        return name in self.attrs

    def get_attribute(self, name, default_val=""):
        return self.attrs.get(name, default_val)

    def parent(self):
        return self._parent


class _EventStub:
    def __init__(self, target, current_target=None):
        self._target = target
        self._current_target = current_target or target

    def target(self):
        return self._target

    def current_target(self):
        return self._current_target


def test_mesh2splat_panel_builds_mesh_and_resolution_records(mesh2splat_module):
    module, state = mesh2splat_module
    panel = module.Mesh2SplatPanel()
    panel._handle = _HandleStub()
    state.nodes[:] = [
        _make_mesh_node(module.lf.scene.NodeType.MESH, "Hull", 1024, 2048),
        _make_mesh_node(module.lf.scene.NodeType.MESH, "Wheel", 256, 512),
    ]

    assert panel._refresh_scene_state(force=True) is True
    assert panel._selected_mesh_name == "Hull"
    assert panel._handle.records["meshes"] == [
        {"name": "Hull", "selected": True, "stats_text": "1024v / 2048f"},
        {"name": "Wheel", "selected": False, "stats_text": "256v / 512f"},
    ]
    assert panel._handle.records["resolutions"][3]["selected"] is True


def test_mesh2splat_panel_convert_uses_python_api(mesh2splat_module):
    module, state = mesh2splat_module
    panel = module.Mesh2SplatPanel()
    panel._selected_mesh_name = "Hull"
    panel._has_meshes = True
    panel._quality = 0.25
    panel._resolution_index = 4
    panel._gaussian_scale = 0.8

    panel._on_convert()

    assert state.mesh_to_splat_calls == [
        (
            "Hull",
            {
                "sigma": 0.8,
                "quality": 0.25,
                "max_resolution": 2048,
            },
        )
    ]
    assert panel._has_initial_conversion is True


def test_mesh2splat_panel_resolution_click_reconverts_after_first_run(mesh2splat_module):
    module, state = mesh2splat_module
    panel = module.Mesh2SplatPanel()
    panel._handle = _HandleStub()
    panel._selected_mesh_name = "Hull"
    panel._has_meshes = True
    panel._has_initial_conversion = True

    target = _ElementStub({"data-resolution-index": "5"})
    container = _ElementStub()

    panel._on_resolution_click(_EventStub(target, container))

    assert panel._resolution_index == 5
    assert state.mesh_to_splat_calls == [
        (
            "Hull",
            {
                "sigma": 0.65,
                "quality": 0.5,
                "max_resolution": 4096,
            },
        )
    ]


def test_mesh2splat_panel_progress_updates_retained_fields(mesh2splat_module):
    module, state = mesh2splat_module
    panel = module.Mesh2SplatPanel()
    panel._handle = _HandleStub()
    state.active = True
    state.progress = 0.625
    state.stage = "Applying..."
    state.error = "conversion failed"

    assert panel._sync_conversion_state(force=False) is True
    assert panel._last_active is True
    assert panel._last_progress_value == "0.625"
    assert panel._last_progress_stage == "Applying..."
    assert panel._error_text == "conversion failed"
    assert panel._handle.dirty_fields == [
        "can_convert",
        "show_progress",
        "progress_value",
        "progress_pct",
        "progress_stage",
        "show_error",
        "error_text",
    ]
