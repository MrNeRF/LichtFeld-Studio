# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the retained splat simplify controls in the rendering panel."""

from enum import IntEnum
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


class _HandleStub:
    def __init__(self):
        self.dirty_fields = []

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_fields.append("__all__")


def _make_splat_node(node_type, name, visible_count):
    splat = SimpleNamespace(visible_count=lambda: visible_count)
    return SimpleNamespace(type=node_type, name=name, splat_data=lambda: splat)


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
    node_type = IntEnum("NodeType", {"SPLAT": 20})

    state = SimpleNamespace(
        selected_name="",
        nodes={},
        simplify_calls=[],
        cancel_calls=0,
        active=False,
        progress=0.0,
        stage="",
        error="",
    )

    class _SceneStub:
        def get_node(self, name):
            return state.nodes.get(name)

    lf_stub = ModuleType("lichtfeld")
    lf_stub.scene = SimpleNamespace(NodeType=node_type)
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
        set_panel_label=lambda _panel_id, _label: True,
        is_windows_platform=lambda: False,
        toggle_system_console=lambda: None,
    )
    lf_stub.get_render_settings = lambda: SimpleNamespace(
        focal_length_mm=50.0,
        prop_info=lambda prop_id: {"name": prop_id},
    )
    lf_stub.get_current_view = lambda: SimpleNamespace(width=1920, height=1080)
    lf_stub.get_scene = lambda: _SceneStub()
    lf_stub.get_selected_node_name = lambda: state.selected_name
    lf_stub.simplify_splats = lambda name, **kwargs: state.simplify_calls.append((name, kwargs))
    lf_stub.cancel_splat_simplify = lambda: setattr(state, "cancel_calls", state.cancel_calls + 1)
    lf_stub.is_splat_simplify_active = lambda: state.active
    lf_stub.get_splat_simplify_progress = lambda: state.progress
    lf_stub.get_splat_simplify_stage = lambda: state.stage
    lf_stub.get_splat_simplify_error = lambda: state.error

    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return state


@pytest.fixture
def rendering_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.rendering_panel", None)
    sys.modules.pop("lfs_plugins.scrub_fields", None)
    sys.modules.pop("lfs_plugins", None)
    state = _install_lf_stub(monkeypatch)
    module = import_module("lfs_plugins.rendering_panel")
    return module, state


def test_rendering_panel_simplify_tracks_selected_splat_and_applies(rendering_panel_module):
    module, state = rendering_panel_module
    panel = module.RenderingPanel()
    panel._handle = _HandleStub()
    state.selected_name = "Patio"
    state.nodes["Patio"] = _make_splat_node(module.lf.scene.NodeType.SPLAT, "Patio", 608_640)

    assert panel._refresh_simplify_source(force=True) is True
    assert panel._simplify_source_name == "Patio"
    assert panel._simplify_original_count == 608_640
    assert panel._scrub_fields._specs["simplify_target"].min_value == 1.0
    assert panel._scrub_fields._specs["simplify_target"].max_value == 608_640.0
    assert panel._compute_simplify_target_count() == 304_320
    assert panel._simplify_output_name() == "Patio_304320"
    assert panel._can_run_simplify() is True

    panel._set_scrub_value("simplify_target", 100_000)

    assert panel._compute_simplify_target_count() == 100_000
    assert panel._simplify_output_name() == "Patio_100000"

    panel._start_simplify()

    assert state.simplify_calls == [
        (
            "Patio",
            {
                "ratio": pytest.approx(100_000 / 608_640),
            },
        )
    ]


def test_rendering_panel_simplify_target_input_clamps_when_source_changes(rendering_panel_module):
    module, state = rendering_panel_module
    panel = module.RenderingPanel()
    panel._handle = _HandleStub()

    state.selected_name = "Large"
    state.nodes["Large"] = _make_splat_node(module.lf.scene.NodeType.SPLAT, "Large", 608_640)
    assert panel._refresh_simplify_source(force=True) is True

    panel._set_scrub_value("simplify_target", 100_000)
    assert panel._compute_simplify_target_count() == 100_000

    state.selected_name = "Small"
    state.nodes["Small"] = _make_splat_node(module.lf.scene.NodeType.SPLAT, "Small", 50_000)
    assert panel._refresh_simplify_source(force=False) is True
    assert panel._scrub_fields._specs["simplify_target"].max_value == 50_000.0
    assert panel._compute_simplify_target_count() == 50_000
    assert panel._simplify_output_name() == "Small_50000"


def test_rendering_panel_simplify_progress_and_cancel_update_retained_state(rendering_panel_module):
    module, state = rendering_panel_module
    panel = module.RenderingPanel()
    panel._handle = _HandleStub()

    state.active = True
    state.progress = 0.625
    state.stage = "Pass 2: merging 1024 pairs"
    state.error = "simplify failed"

    assert panel._sync_simplify_task_state(force=False) is True
    assert panel._simplify_task_active is True
    assert panel._simplify_progress_value == "0.625"
    assert panel._simplify_progress_pct() == "62%"
    assert panel._simplify_progress_stage == "Pass 2: merging 1024 pairs"
    assert panel._simplify_error_text == "simplify failed"
    assert panel._handle.dirty_fields == [
        "simplify_can_apply",
        "simplify_show_progress",
        "simplify_progress_value",
        "simplify_progress_pct",
        "simplify_progress_stage",
        "simplify_show_error",
        "simplify_error_text",
    ]

    panel._on_simplify_cancel()
    assert state.cancel_calls == 1
