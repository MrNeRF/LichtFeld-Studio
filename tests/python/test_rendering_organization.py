# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contracts for retained controls, default scope and old-project chrome."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import pytest
from test_preferences_panel import preferences_panel_module
from test_rendering_panel_regressions import rendering_panel_module, _BindingModelStub, _BindingContextStub

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "src/visualizer/gui/rmlui/resources"
# Visible control bindings in master before the reorganization.
ORIGINAL_CONTROLS = {'lod_page_pool_splats', 'ppisp_mode', 'ppisp_vignette_enabled', 'camera_frustum_scale', 'simplify_target', 'hide_outside_depth_box', 'environment_exposure', 'lod_cone_outer_degrees', 'lod_debug_mode', 'color_exposure', 'desaturate_unselected', 'mesh_light_intensity', 'environment_mode', 'lod_cone_foveation', 'grid_plane', 'background_color_r', 'mesh_ambient', 'ppisp_exposure', 'show_camera_frustums', 'show_pivot', 'voxel_size', 'mesh_wireframe_width', 'simplify_lod_base', 'mesh_shadow_resolution', 'grid_opacity', 'ppisp_gamma_blue', 'theme_vignette_intensity', 'camera_metrics_mode', 'color_tonemapping', 'lod_max_splats', 'selection_color_center_marker_hex', 'raster_backend', 'selection_color_preview_hex', 'focal_length_mm', 'lod_pool_vram_fraction', 'ppisp_crf_shoulder', 'mesh_wireframe', 'lod_enabled', 'theme_vignette_softness', 'navigation_up_idx', 'ppisp_gamma_green', 'desaturate_cropping', 'background_color_hex', 'environment_map_preset', 'ppisp_vignette_strength', 'environment_rotation_degrees', 'render_scale', 'sh_degree', 'show_coord_axes', 'theme_vignette_radius', 'apply_appearance_correction', 'axes_size', 'ppisp_crf_toe', 'lod_render_scale', 'mip_filter', 'background_color_g', 'equirectangular', 'ppisp_gamma_multiplier', 'theme_vignette_enabled', 'mesh_wireframe_color_hex', 'mesh_shadow_enabled', 'mesh_backface_culling', 'lod_cone_inner_degrees', 'ppisp_gamma_red', 'show_grid', 'lod_fade_frames', 'simplify_opacity_prune_threshold', 'background_color_b'}

def test_every_original_control_remains_reachable():
    controls = set()
    for filename in ("rendering.rml", "splat_simplify.rml", "preferences.rml"):
        root = ET.parse(RESOURCES / filename).getroot()
        controls.update(v for e in root.iter() for k, v in e.attrib.items()
                        if k in ("data-value", "data-checked"))
    assert ORIGINAL_CONTROLS <= controls


def test_rendering_groups_image_controls_and_keeps_overlays_in_side_panel():
    root = ET.parse(RESOURCES / "rendering.rml").getroot()
    assert root.find('.//*[@id="sec-simplify"]') is None
    color = root.find('.//*[@id="sec-post-process"]')
    assert color.find('.//*[@data-value="color_tonemapping"]') is not None
    assert color.find('.//*[@data-value="color_exposure"]') is not None
    assert color.find('.//*[@data-checked="apply_appearance_correction"]') is not None
    assert color.find('.//*[@data-checked="theme_vignette_enabled"]') is None
    quality = root.find('.//*[@id="sec-quality"]')
    assert quality.find('.//*[@data-value="render_scale"]') is not None
    assert quality.find('.//*[@id="sec-lod-advanced"]') is not None
    viewport = root.find('.//*[@id="sec-viewport"]')
    assert viewport.find('.//*[@data-checked="show_grid"]') is not None
    assert viewport.find('.//*[@data-event-click="apply_defaults"]') is not None


def test_old_project_chrome_adds_collapsed_advanced_groups(rendering_panel_module):
    panel = rendering_panel_module.RenderingPanel()
    panel.apply_chrome({"collapsed": ["lod", "selection", "simplify"]})
    assert "viewport" not in panel._collapsed
    assert {"lod", "selection", "quality_advanced", "lod_advanced", "environment"} <= panel._collapsed
    captured = panel.capture_chrome()
    panel.apply_chrome(captured)
    assert panel.capture_chrome() == captured


def test_default_change_preserves_current_project_and_other_defaults(preferences_panel_module):
    module, state = preferences_panel_module
    state.rendering_defaults = {"axes_size": 2.0, "grid_opacity": 0.75, "future_field": 42}
    state.render_settings.grid_opacity = 0.9
    panel = module.PreferencesPanel()
    panel._rendering_defaults._set("axes_size", "4.5")
    assert state.rendering_defaults["axes_size"] == 4.5
    assert state.rendering_defaults["grid_opacity"] == 0.75
    assert state.rendering_defaults["future_field"] == 42
    assert state.render_settings.grid_opacity == 0.9


@pytest.mark.parametrize("prop,value", [("axes_size", "nan"), ("grid_opacity", "2"),
                                        ("lod_page_pool_splats", "1.5"),
                                        ("selection_color_preview", "not-a-color")])
def test_invalid_defaults_leave_saved_values_unchanged(preferences_panel_module, prop, value):
    module, state = preferences_panel_module
    state.rendering_defaults = {"axes_size": 3.0, "future_field": 42}
    panel = module.PreferencesPanel()
    before = dict(state.rendering_defaults)
    panel._rendering_defaults._set(prop, value)
    assert state.rendering_defaults == before
    assert panel._rendering_defaults._error


def test_default_text_saves_on_commit_not_per_keystroke(preferences_panel_module):
    module, state = preferences_panel_module
    state.rendering_defaults = {"axes_size": 9.0}
    panel = module.PreferencesPanel()
    section = panel._rendering_defaults
    model = _BindingModelStub()
    section.bind(model)
    getter, setter = model.bindings["default_axes_size"]
    for text in ("1", "15"):
        setter(text)
    assert state.rendering_defaults["axes_size"] == 9.0
    assert getter() == "15"
    section._commit("axes_size")
    assert state.rendering_defaults["axes_size"] == 9.0
    assert section._error and getter() == 9.0
    setter("4")
    section._commit("axes_size")
    assert state.rendering_defaults["axes_size"] == 4.0
    assert not section._error
    setter("7")
    section._discard("axes_size")
    section._commit("axes_size")
    assert state.rendering_defaults["axes_size"] == 4.0


def test_default_colors_are_picked_from_a_swatch_and_saved_on_release(preferences_panel_module):
    from types import SimpleNamespace
    root = ET.parse(RESOURCES / "preferences.rml").getroot()
    for prop in ("selection_color_preview", "selection_color_center_marker"):
        assert root.find(f".//*[@data-event-click=\"default_color_click('{prop}')\"]") is not None
    module, state = preferences_panel_module
    state.rendering_defaults = {"selection_color_preview": [0.0, 0.871, 0.298]}
    section = module.PreferencesPanel()._rendering_defaults
    model = _BindingModelStub()
    section.bind(model)
    model.events["default_color_click"](None, None, ["selection_color_preview"])
    assert model.func_bindings["default_color_editing"]() == "selection_color_preview"
    assert model.func_bindings["default_picker_g"]() == pytest.approx(0.871)
    picked = {"red": "1", "green": "0.5", "blue": "0"}
    model.events["default_picker_change"](None, SimpleNamespace(get_parameter=lambda name, _d: picked[name]), [])
    # Dragging previews the colour without saving it.
    assert state.rendering_defaults["selection_color_preview"] == [0.0, 0.871, 0.298]
    assert model.func_bindings["default_selection_color_preview_swatch"]() == "rgb(255,128,0)"
    model.events["default_color_commit"](None, None, [])
    assert state.rendering_defaults["selection_color_preview"] == pytest.approx([1.0, 128 / 255, 0.0])


def test_apply_defaults_requires_explicit_confirmation(rendering_panel_module):
    module = rendering_panel_module
    panel = module.RenderingPanel()
    applied, dialogs = [], []
    module.lf.ui.apply_rendering_defaults = lambda: applied.append(True)
    module.lf.ui.confirm_dialog = lambda *args: dialogs.append(args)
    panel._confirm_apply_defaults()
    assert applied == []
    title, text, buttons, callback = dialogs.pop()
    callback(buttons[1])
    assert applied == []
    callback(buttons[0])
    assert applied == [True]


def test_vignette_scrub_previews_without_repeated_disk_writes(preferences_panel_module):
    from types import SimpleNamespace
    module, state = preferences_panel_module
    vignette = SimpleNamespace(enabled=True, intensity=0.3, radius=0.7, softness=0.5)
    calls = []
    module.lf.ui.theme = lambda: SimpleNamespace(vignette=vignette)
    def set_style(intensity, radius, softness, persist=True):
        vignette.intensity, vignette.radius, vignette.softness = intensity, radius, softness
        calls.append(persist)
    module.lf.ui.set_theme_vignette_style = set_style
    panel = module.PreferencesPanel()
    for value in (0.4, 0.5, 0.6):
        panel._set_scrub_value("theme_vignette_intensity", value)
    assert calls == [False, False, False]
    assert vignette.intensity == 0.6
    panel._commit_scrub_value("theme_vignette_intensity", 0.6)
    assert calls == [False, False, False, True]


@pytest.mark.parametrize("prop,value", [("axes_size", "4.5"), ("camera_frustum_scale", "0.125")])
def test_preferences_apply_defaults_commits_focused_field_then_requires_confirmation(
        preferences_panel_module, prop, value):
    module, state = preferences_panel_module
    state.rendering_defaults = {"axes_size": 2.0, "camera_frustum_scale": 0.25}
    state.render_settings.camera_frustum_scale = 0.75
    state.render_settings.axes_size = 8.0
    section = module.PreferencesPanel()._rendering_defaults
    events, dialogs, applied = {}, [], []

    class EventModel(_BindingModelStub):
        def bind_event(self, name, callback):
            events[name] = callback

    model = EventModel()
    section.bind(model)
    module.lf.ui.confirm_dialog = lambda *args: dialogs.append(args)
    module.lf.ui.apply_rendering_defaults = lambda: applied.append(dict(state.rendering_defaults))
    model.bindings["default_" + prop][1](value)
    events["apply_viewport_defaults"]()
    assert state.rendering_defaults[prop] == float(value)
    assert state.render_settings.camera_frustum_scale == 0.75
    assert state.render_settings.axes_size == 8.0
    assert applied == []
    _, _, buttons, callback = dialogs.pop()
    callback(buttons[1])
    assert applied == []
    events["apply_viewport_defaults"]()
    _, _, buttons, callback = dialogs.pop()
    callback(buttons[0])
    assert len(applied) == 1
    assert applied[0][prop] == float(value)


def test_preferences_invalid_pending_default_cannot_apply_old_values(preferences_panel_module):
    module, state = preferences_panel_module
    section = module.PreferencesPanel()._rendering_defaults
    model = _BindingModelStub()
    section.bind(model)
    dialogs, applied = [], []
    module.lf.ui.confirm_dialog = lambda *args: dialogs.append(args)
    module.lf.ui.apply_rendering_defaults = lambda: applied.append(True)
    model.bindings["default_axes_size"][1]("nan")
    section._confirm_apply()
    assert section._error
    assert dialogs == [] and applied == []


def test_preferences_exposes_explicit_project_apply_and_translates_it_everywhere():
    root = ET.parse(RESOURCES / "preferences.rml").getroot()
    viewport = root.find('.//*[@data-if="show_viewport"]')
    button = viewport.find('.//*[@data-event-click="apply_viewport_defaults"]')
    assert button is not None
    assert button.text == "@tr:preferences.apply_viewport_defaults"
    for path in (ROOT / "src/visualizer/gui/resources/locales").glob("*.json"):
        locale = json.loads(path.read_text(encoding="utf-8"))
        assert locale["preferences"]["apply_viewport_defaults"]
        assert locale["main_panel"]["axes_size"]
        assert locale["tooltip"]["axes_size"]
