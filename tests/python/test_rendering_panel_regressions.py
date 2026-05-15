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
        self.bindings = {}
        self.func_bindings = {}

    def bind(self, name, getter, setter):
        self.bindings[name] = (getter, setter)

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


class _HandleStub:
    def __init__(self):
        self.dirty_fields = []

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_fields.append("__all__")


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
        theme=lambda: None,
        set_theme_vignette_enabled=lambda _value: None,
        set_theme_vignette_intensity=lambda _value: None,
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
    module = import_module("lfs_plugins.rendering_panel")
    return module


def test_rendering_panel_section_headers_use_literals_without_missing_keys(rendering_panel_module):
    module = rendering_panel_module
    requested_keys = []
    translations = {
        "rendering_panel.section_simplify": "Simplify Localized",
        "rendering_panel.simplify_source": "Input Source",
        "rendering_panel.simplify_select_source": "Pick a splat",
        "rendering_panel.simplify_target": "Goal",
        "rendering_panel.simplify_knn_k": "Neighbors",
        "rendering_panel.simplify_merge_cap": "Merge Limit",
        "rendering_panel.simplify_opacity_prune": "Opacity Cutoff",
        "rendering_panel.simplify_original": "Before",
        "rendering_panel.simplify_output": "Result",
        "common.apply": "Run",
        "common.cancel": "Abort",
    }
    module.lf.ui.tr = lambda key: requested_keys.append(key) or translations.get(key, key)
    model = _BindingModelStub()
    panel = module.RenderingPanel()

    panel.on_bind_model(_BindingContextStub(model))

    assert model.func_bindings["label_hdr_viewport"]() == "Viewport"
    assert model.func_bindings["label_hdr_camera"]() == "Camera & Projection"
    assert model.func_bindings["label_hdr_simplify"]() == "Simplify Localized"
    assert model.func_bindings["label_hdr_selection"]() == "Selection & Overlays"
    assert model.func_bindings["label_hdr_post_process"]() == "Post Processing"
    assert model.func_bindings["label_simplify_source"]() == "Input Source:"
    assert model.func_bindings["label_simplify_select_source"]() == "Pick a splat"
    assert model.func_bindings["label_simplify_target"]() == "Goal:"
    assert model.func_bindings["label_simplify_target_stat"]() == "Goal"
    assert model.func_bindings["label_simplify_knn_k"]() == "Neighbors:"
    assert model.func_bindings["label_simplify_merge_cap"]() == "Merge Limit:"
    assert model.func_bindings["label_simplify_opacity_prune"]() == "Opacity Cutoff:"
    assert model.func_bindings["label_simplify_original"]() == "Before"
    assert model.func_bindings["label_simplify_output"]() == "Result:"
    assert model.func_bindings["label_simplify_apply"]() == "Run"
    assert model.func_bindings["label_simplify_cancel"]() == "Abort"
    assert "rendering_panel.section_viewport" not in requested_keys
    assert "rendering_panel.section_camera" not in requested_keys
    assert "rendering_panel.section_simplify" in requested_keys
    assert "rendering_panel.section_selection" not in requested_keys
    assert "rendering_panel.section_post_process" not in requested_keys


def test_rendering_panel_binds_theme_vignette_controls(rendering_panel_module):
    module = rendering_panel_module
    theme_state = SimpleNamespace(vignette=SimpleNamespace(enabled=True, intensity=0.25))
    set_calls = {"enabled": [], "intensity": []}
    module.lf.ui.theme = lambda: theme_state
    module.lf.ui.set_theme_vignette_enabled = lambda value: set_calls["enabled"].append(value)
    module.lf.ui.set_theme_vignette_intensity = lambda value: set_calls["intensity"].append(value)

    model = _BindingModelStub()
    panel = module.RenderingPanel()

    panel.on_bind_model(_BindingContextStub(model))

    enabled_getter, enabled_setter = model.bindings["theme_vignette_enabled"]
    intensity_getter, intensity_setter = model.bindings["theme_vignette_intensity"]

    assert enabled_getter() is True
    assert intensity_getter() == pytest.approx(0.25)

    enabled_setter(False)
    intensity_setter(0.4)

    assert set_calls["enabled"] == [False]
    assert set_calls["intensity"] == [0.4]


def test_rendering_panel_raster_backend_updates_gut_mirror(rendering_panel_module):
    module = rendering_panel_module
    settings = SimpleNamespace(raster_backend="3dgut", gut=True)
    module.lf.get_render_settings = lambda: settings

    model = _BindingModelStub()
    panel = module.RenderingPanel()

    panel.on_bind_model(_BindingContextStub(model))

    backend_getter, backend_setter = model.bindings["raster_backend"]

    assert backend_getter() == "3dgut"

    backend_setter("fast_gs")
    assert settings.raster_backend == "fast_gs"
    assert settings.gut is False

    backend_setter("3dgut")
    assert settings.raster_backend == "3dgut"
    assert settings.gut is True

    backend_setter("vksplat")
    assert settings.raster_backend == "vksplat"
    assert settings.gut is False

    backend_setter("vksplat_3dgut")
    assert settings.raster_backend == "vksplat_3dgut"
    assert settings.gut is True


def test_rendering_panel_custom_environment_map_appears_in_dropdown(rendering_panel_module):
    module = rendering_panel_module
    settings = SimpleNamespace(
        environment_mode="EQUIRECTANGULAR",
        environment_map_path="/tmp/custom_hdri_name.hdr",
        prop_info=lambda prop_id: {"name": prop_id},
    )
    module.lf.get_render_settings = lambda: settings

    panel = module.RenderingPanel()
    panel._handle = _HandleStub()

    assert panel._sync_environment_state() is True
    assert panel._handle.dirty_fields == ["__all__"]
    assert panel._get_environment_map_preset() == module.CUSTOM_ENVIRONMENT_PRESET_VALUE
    assert panel._environment_map_is_custom() is True
    assert panel._environment_map_display_name() == "custom_hdri_name.hdr"
    assert panel._environment_map_has_custom_option() is True
    assert panel._environment_map_last_custom_display_name() == "custom_hdri_name.hdr"
    panel._set_environment_map_preset(module.CUSTOM_ENVIRONMENT_PRESET_VALUE)
    assert settings.environment_map_path == "/tmp/custom_hdri_name.hdr"
    settings.environment_map_path = module.ENVIRONMENT_PRESET_PATHS[1]
    assert panel._sync_environment_state() is True
    assert panel._get_environment_map_preset() == "1"
    assert panel._environment_map_is_custom() is False
    assert panel._environment_map_has_custom_option() is True
    assert panel._environment_map_last_custom_display_name() == "custom_hdri_name.hdr"
    panel._set_environment_map_preset(module.CUSTOM_ENVIRONMENT_PRESET_VALUE)
    assert settings.environment_map_path == "/tmp/custom_hdri_name.hdr"


def test_rendering_rml_exposes_simplify_tooltips_and_locale_labels():
    project_root = Path(__file__).parent.parent.parent
    rendering_rml = project_root / "src" / "visualizer" / "gui" / "rmlui" / "resources" / "rendering.rml"
    content = rendering_rml.read_text()

    assert 'data-tooltip="tooltip.simplify_source"' in content
    assert 'data-tooltip="tooltip.simplify_target"' in content
    assert 'data-tooltip="tooltip.simplify_knn_k"' in content
    assert 'data-tooltip="tooltip.simplify_merge_cap"' in content
    assert 'data-tooltip="tooltip.simplify_opacity_prune"' in content
    assert 'data-tooltip="tooltip.simplify_output"' in content
    assert 'data-tooltip="tooltip.simplify_apply"' in content
    assert 'data-tooltip="tooltip.simplify_cancel"' in content
    custom_option_start = content.find('<option value="__custom__"')
    assert custom_option_start >= 0
    custom_option_end = content.find("</option>", custom_option_start)
    custom_option = content[custom_option_start:custom_option_end]
    assert 'data-if="environment_map_has_custom_option"' in custom_option
    assert "{{environment_map_last_custom_display_name}}" in content
    assert "{{label_simplify_source}}" in content
    assert "{{label_simplify_select_source}}" in content
    assert "{{label_simplify_target}}" in content
    assert "{{label_simplify_knn_k}}" in content
    assert "{{label_simplify_merge_cap}}" in content
    assert "{{label_simplify_opacity_prune}}" in content
    assert "{{label_simplify_original}}" in content
    assert "{{label_simplify_target_stat}}" in content
    assert "{{label_simplify_output}}" in content
    assert "{{label_simplify_apply}}" in content
    assert "{{label_simplify_cancel}}" in content
