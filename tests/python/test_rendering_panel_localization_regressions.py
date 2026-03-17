# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for retained rendering panel localization bindings."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


class _BindingModelStub:
    def __init__(self):
        self.func_bindings = {}

    def bind(self, _name, _getter, _setter):
        pass

    def bind_func(self, name, getter):
        self.func_bindings[name] = getter

    def bind_event(self, _name, _handler):
        pass

    def get_handle(self):
        return object()


class _BindingContextStub:
    def __init__(self, model):
        self._model = model

    def create_data_model(self, _name):
        return self._model


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

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
        set_panel_label=lambda _panel_id, _label: True,
        is_windows_platform=lambda: False,
        toggle_system_console=lambda: None,
    )
    lf_stub.get_render_settings = lambda: SimpleNamespace()
    lf_stub.get_current_view = lambda: SimpleNamespace(width=1920, height=1080)
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)


@pytest.fixture
def rendering_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.rendering_panel", None)
    sys.modules.pop("lfs_plugins.scrub_fields", None)
    sys.modules.pop("lfs_plugins", None)
    _install_lf_stub(monkeypatch)
    return import_module("lfs_plugins.rendering_panel")


def test_rendering_panel_section_headers_use_literals_without_missing_keys(rendering_panel_module):
    requested_keys = []
    rendering_panel_module.lf.ui.tr = lambda key: requested_keys.append(key) or key
    model = _BindingModelStub()
    panel = rendering_panel_module.RenderingPanel()

    panel.on_bind_model(_BindingContextStub(model))

    assert model.func_bindings["label_hdr_viewport"]() == "Viewport"
    assert model.func_bindings["label_hdr_camera"]() == "Camera & Projection"
    assert model.func_bindings["label_hdr_selection"]() == "Selection & Overlays"
    assert model.func_bindings["label_hdr_post_process"]() == "Post Processing"
    assert "rendering_panel.section_viewport" not in requested_keys
    assert "rendering_panel.section_camera" not in requested_keys
    assert "rendering_panel.section_selection" not in requested_keys
    assert "rendering_panel.section_post_process" not in requested_keys
