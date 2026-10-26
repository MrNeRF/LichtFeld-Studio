# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for retained rendering panel localization bindings."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from unittest.mock import MagicMock
from test_depth_view_scrub import attach_fields, Event

import pytest

from locale_utils import missing_locale_keys


class _BindingModelStub:
    def __init__(self):
        self.bindings = {}
        self.func_bindings = {}
        self.events = {}
        self.handle = _HandleStub()

    def bind(self, name, getter, setter):
        self.bindings[name] = (getter, setter)

    def bind_func(self, name, getter):
        self.func_bindings[name] = getter

    def bind_event(self, name, handler):
        self.events[name] = handler

    def bind_record_list(self, _name):
        pass

    def get_handle(self):
        return self.handle


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


class _RenderSettingsStub(SimpleNamespace):
    def _set_without_session_dirty(self, prop, value):
        setattr(self, prop, value)


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
        mark_project_session_chapter_committed=lambda _chapter: None,
        set_panel_label=lambda _panel_id, _label: True,
        is_windows_platform=lambda: False,
        toggle_system_console=lambda: None,
    )
    lf_stub.get_navigation_up_axis = lambda: "legacy_y"
    lf_stub.set_navigation_up_axis = lambda *_args, **_kwargs: None
    lf_stub.get_render_settings = lambda: SimpleNamespace()
    lf_stub.get_collision_state = lambda: {"walk_settings": (1.7, .25, .3, 2.)}
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
        "rendering_panel.simplify_lod_base": "LOD Base",
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

    assert model.func_bindings["label_hdr_viewport"]() == "Viewport & Guides"
    assert model.func_bindings["label_hdr_camera"]() == "Camera & Projection"
    assert model.func_bindings["label_hdr_selection"]() == "Selection & Overlays"
    assert model.func_bindings["label_hdr_post_process"]() == "Color & Appearance"
    assert "simplify_target" not in model.bindings
    sys.modules.pop("lfs_plugins.splat_simplify_panel", None)
    tool_module = import_module("lfs_plugins.splat_simplify_panel")
    tool = tool_module.SplatSimplifyPanel()
    model = _BindingModelStub()
    tool.on_bind_model(_BindingContextStub(model))
    assert model.func_bindings["panel_label"]() == "Simplify Localized"
    assert model.func_bindings["label_simplify_source"]() == "Input Source:"
    assert model.func_bindings["label_simplify_select_source"]() == "Pick a splat"
    assert model.func_bindings["label_simplify_target"]() == "Goal:"
    assert model.func_bindings["label_simplify_target_stat"]() == "Goal"
    assert model.func_bindings["label_simplify_lod_base"]() == "LOD Base:"
    assert model.func_bindings["label_simplify_opacity_prune"]() == "Opacity Cutoff:"
    assert model.func_bindings["label_simplify_original"]() == "Before"
    assert model.func_bindings["label_simplify_output"]() == "Result:"
    assert model.func_bindings["label_simplify_apply"]() == "Run"
    assert model.func_bindings["label_simplify_cancel"]() == "Abort"
    # The panel resolves section headers through tr() with a literal fallback.
    # Assert every requested key exists so a missing one cannot silently render
    # the raw key in the UI.
    missing = missing_locale_keys(requested_keys)
    assert not missing, f"rendering panel requests locale keys absent from en.json: {missing}"


def test_rendering_panel_binds_theme_vignette_controls(rendering_panel_module):
    module = rendering_panel_module
    theme_state = SimpleNamespace(vignette=SimpleNamespace(enabled=True, intensity=0.25))
    set_calls = {"enabled": [], "intensity": []}
    module.lf.ui.theme = lambda: theme_state
    module.lf.ui.set_theme_vignette_enabled = lambda value: set_calls["enabled"].append(value)
    module.lf.ui.set_theme_vignette_style = lambda intensity, radius, softness: set_calls["intensity"].append(intensity)

    model = _BindingModelStub()
    sys.modules.pop("lfs_plugins.rendering_defaults", None)
    defaults = import_module("lfs_plugins.rendering_defaults")
    defaults.RenderingDefaultsSection().bind(model)

    enabled_getter, enabled_setter = model.bindings["theme_vignette_enabled"]
    intensity_getter, intensity_setter = model.bindings["theme_vignette_intensity"]

    assert enabled_getter() is True
    assert intensity_getter() == pytest.approx(0.25)

    enabled_setter(False)
    intensity_setter(0.4)

    assert set_calls["enabled"] == [False]
    assert set_calls["intensity"] == [0.4]


def test_rendering_panel_raster_backend_uses_3dgs_ids(rendering_panel_module):
    module = rendering_panel_module
    settings = _RenderSettingsStub(raster_backend="3dgut")
    module.lf.get_render_settings = lambda: settings

    model = _BindingModelStub()
    panel = module.RenderingPanel()

    panel.on_bind_model(_BindingContextStub(model))

    backend_getter, backend_setter = model.bindings["raster_backend"]

    assert backend_getter() == "3dgut"

    backend_setter("3dgs")
    assert settings.raster_backend == "3dgs"

    backend_setter("3dgut")
    assert settings.raster_backend == "3dgut"


def test_rendering_panel_slider_marks_view_dirty_on_release(rendering_panel_module):
    module = rendering_panel_module
    marks = []
    settings = SimpleNamespace(
        render_scale=1.0,
        _set_without_session_dirty=lambda prop, value: setattr(settings, prop, value),
    )
    module.lf.get_render_settings = lambda: settings
    module.lf.ui.mark_project_session_chapter_committed = marks.append
    model = _BindingModelStub()
    panel = module.RenderingPanel()
    panel.on_bind_model(_BindingContextStub(model))

    _, set_render_scale = model.bindings["render_scale"]
    set_render_scale(0.7)
    set_render_scale(0.6)
    assert settings.render_scale == pytest.approx(0.6)
    assert marks == []

    panel._on_body_mouseup(None)
    assert marks == ["VIEW"]
    assert panel._render_slider_pending is False


def test_rendering_panel_noop_property_binding_does_not_mark_view_dirty(rendering_panel_module):
    module = rendering_panel_module
    marks = []
    settings = SimpleNamespace(
        show_grid=False,
        _set_without_session_dirty=lambda prop, value: setattr(settings, prop, value),
    )
    module.lf.get_render_settings = lambda: settings
    module.lf.ui.mark_project_session_chapter_committed = marks.append
    panel = module.RenderingPanel()

    panel._set_render_property("show_grid", False)
    assert marks == []
    panel._set_render_property("show_grid", True)
    assert settings.show_grid is True
    assert marks == ["VIEW"]


def test_selection_outline_checkbox_commits_view_only_on_change(rendering_panel_module):
    module = rendering_panel_module
    marks = []
    settings = _RenderSettingsStub(selection_outline=False)
    module.lf.get_render_settings = lambda: settings
    module.lf.ui.mark_project_session_chapter_committed = marks.append
    model = _BindingModelStub()
    panel = module.RenderingPanel()
    panel.on_bind_model(_BindingContextStub(model))
    getter, setter = model.bindings["selection_outline"]
    assert getter() is False
    setter(True)
    assert settings.selection_outline is True
    assert marks == ["VIEW"]
    setter(True)
    assert marks == ["VIEW"]
    setter(False)
    assert settings.selection_outline is False
    assert marks == ["VIEW", "VIEW"]


def test_rendering_panel_color_text_marks_view_dirty_on_commit(rendering_panel_module):
    module = rendering_panel_module
    marks = []
    settings = _RenderSettingsStub(mesh_wireframe_color=(0.0, 0.0, 0.0))
    module.lf.get_render_settings = lambda: settings
    module.lf.ui.mark_project_session_chapter_committed = marks.append
    model = _BindingModelStub()
    panel = module.RenderingPanel()
    panel.on_bind_model(_BindingContextStub(model))

    model.bindings["mesh_wireframe_color_hex"][1]("#ff0000")
    assert settings.mesh_wireframe_color == (0.0, 0.0, 0.0)
    assert marks == []

    panel._commit_color_text_field("mesh_wireframe_color_hex")
    assert settings.mesh_wireframe_color == (1.0, 0.0, 0.0)
    assert marks == ["VIEW"]


def test_rendering_panel_equirectangular_delegates_backend_to_native_settings(rendering_panel_module):
    module = rendering_panel_module
    settings = _RenderSettingsStub(raster_backend="3dgs", equirectangular=False)
    module.lf.get_render_settings = lambda: settings

    model = _BindingModelStub()
    panel = module.RenderingPanel()

    panel.on_bind_model(_BindingContextStub(model))

    _, equirectangular_setter = model.bindings["equirectangular"]

    equirectangular_setter(True)
    assert settings.equirectangular is True
    assert settings.raster_backend == "3dgs"
    assert "raster_backend" in model.handle.dirty_fields

    equirectangular_setter(False)
    assert settings.equirectangular is False
    assert settings.raster_backend == "3dgs"


def test_rendering_panel_hides_perspective_fov_for_equirectangular(rendering_panel_module):
    module = rendering_panel_module
    settings = _RenderSettingsStub(raster_backend="3dgs", equirectangular=False, focal_length_mm=35.0)
    module.lf.get_render_settings = lambda: settings
    model = _BindingModelStub()
    panel = module.RenderingPanel()

    panel.on_bind_model(_BindingContextStub(model))
    show_fov_controls = model.func_bindings["show_fov_controls"]
    equirectangular, set_equirectangular = model.bindings["equirectangular"]
    assert equirectangular() is False
    assert show_fov_controls() is True

    set_equirectangular(True)
    assert equirectangular() is True
    assert show_fov_controls() is False
    assert settings.focal_length_mm == pytest.approx(35.0)
    assert "show_fov_controls" in model.handle.dirty_fields

    set_equirectangular(False)
    assert equirectangular() is False
    assert show_fov_controls() is True
    assert settings.focal_length_mm == pytest.approx(35.0)

    project_root = Path(__file__).parent.parent.parent
    rendering_rml = project_root / "src" / "visualizer" / "gui" / "rmlui" / "resources" / "rendering.rml"
    content = rendering_rml.read_text()
    controls = content.split('<div data-if="show_fov_controls">', 1)[1].split(
        '<div class="setting-row setting-row--aligned" data-tooltip="tooltip.sh_degree">', 1
    )[0]
    assert 'data-value="focal_length_mm"' in controls and "@tr:main_panel.fov" in controls


def test_rendering_panel_projection_sync_updates_backend_dropdown(rendering_panel_module):
    module = rendering_panel_module
    settings = SimpleNamespace(raster_backend="3dgs", equirectangular=False)
    module.lf.get_render_settings = lambda: settings

    model = _BindingModelStub()
    panel = module.RenderingPanel()

    panel.on_bind_model(_BindingContextStub(model))
    assert panel._sync_projection_state() is True
    model.handle.dirty_fields.clear()

    settings.raster_backend = "3dgut"
    settings.equirectangular = True

    assert panel._sync_projection_state() is True
    assert model.handle.dirty_fields == ["raster_backend", "equirectangular", "show_fov_controls"]


@pytest.mark.parametrize("field", ("scene_generation", "viewport_toolbar_generation"))
def test_rendering_panel_reacts_to_native_scene_generation(rendering_panel_module, field):
    module = rendering_panel_module
    model = _BindingModelStub()
    panel = module.RenderingPanel()

    assert module.RenderingPanel.update_policy == "dirty"
    assert "update_interval_ms" not in module.RenderingPanel.__dict__

    panel.on_bind_model(_BindingContextStub(model))
    panel._subscribe_reactive_state()
    model.handle.dirty_fields.clear()

    signal = getattr(module.RuntimeState, field)
    value = signal.value + 1
    signal.value = value
    assert model.handle.dirty_fields == ["__all__"]
    model.handle.dirty_fields.clear()
    signal.value = value
    assert not model.handle.dirty_fields
    panel._unsubscribe_reactive_state()


def test_collision_popup_is_dirty_driven_and_reopens_current_model(rendering_panel_module, monkeypatch):
    from lfs_plugins import collision_panel
    from lfs_plugins.panels import PANEL_SPECS, _register_lazy_panel
    lf = collision_panel.lf
    monkeypatch.setattr(lf.ui, "Panel", collision_panel.Panel, raising=False)
    monkeypatch.setattr(lf.ui, "set_panel_enabled", lambda *_: None, raising=False)
    monkeypatch.setattr(lf, "register_class", lambda *_: None, raising=False)
    lazy = _register_lazy_panel(lf, "collision")
    panel, other = lazy(), lazy()
    monkeypatch.setattr(lf.ui, "get_panel_object", lambda _: panel, raising=False)
    state = {"has_collision": True, "target_uuid": "first", "target_name": "Floor"}
    monkeypatch.setattr(lf, "get_collision_state", lambda: dict(state))
    lf.ui.get_panel_object("lfs.collision").show(export=True)
    model = _BindingModelStub()
    panel.on_bind_model(_BindingContextStub(model))
    controls = panel._load()
    bindings, events = model.func_bindings, model.events
    assert PANEL_SPECS["collision"].update_policy == "dirty"
    assert set(model.bindings) == {"collision_automatic", "collision_cleanup"}
    assert not bindings["collision_source"]() and bindings["collision_target"]() == "Floor"
    state.update(target_uuid="second", target_name="Wall")
    for stage in ("Preparing", "Processing", "Exporting", "Ready"):
        state.update(source_name="Selected", active=stage == "Processing", stage=stage, progress=.4)
        assert panel.on_update(None)
        assert bindings["collision_source"]() == "Selected"
        assert bindings["collision_active"]() == state.get("active", False)
        assert bindings["collision_status"]().startswith(stage)
    other.show()
    assert panel.export_target == ("first", "Floor") and other.export_target is None
    model.handle.dirty_fields.clear()
    assert panel.on_update(None) is False
    assert not model.handle.dirty_fields
    for export in (True, False):
        model.handle.dirty_fields.clear()
        lf.ui.get_panel_object("lfs.collision").show(export=export)
        assert model.handle.dirty_fields == ["__all__"] and model.func_bindings["export_mode"]() is export
    controls.values.update(size=.005, opacity=.2, area=3., automatic=True)
    for event, name, args in (("generate", "generate_collision", (.005, .2, True, True, 3.)),
                              ("cancel", "cancel_collision", ())):
        action = MagicMock()
        monkeypatch.setattr(lf, name, action, raising=False)
        events[f"collision_{event}"]()
        action.assert_called_once_with(*args)
    state.update(resolved_voxel_size=.01234, generation_settings=(.00500000007, .200000001, True, True, 3.))
    controls.update()
    assert controls.resolved_size() == "0.01234"
    for changes in ({"size": .01}, {"size": .005, "opacity": .3}, {"opacity": .2, "automatic": False},
                    {"automatic": True, "cleanup": False}, {"cleanup": True, "area": 1}, {"area": 3}):
        controls.values.update(changes)
        if changes == {"area": 3}:
            controls.state["resolved_voxel_size"] = 0
        assert controls.resolved_size() == "projects.gallery.review.estimate_pending"
    monkeypatch.setattr(controls, "mount", lambda doc: None)
    panel.on_mount(SimpleNamespace(get_element_by_id=lambda _: None))
    model.handle.dirty_fields.clear()
    state["active"] = True
    collision_panel.RuntimeState.viewport_toolbar_generation.value += 1
    assert model.handle.dirty_fields == ["__all__"]
    panel.on_unmount(SimpleNamespace(remove_data_model=lambda *_: None))
    assert panel.handle is None and not controls.binding.active and not controls.scrubs._fields


def test_rendering_panel_reacts_to_native_tool_state(rendering_panel_module):
    module = rendering_panel_module
    model = _BindingModelStub()
    panel = module.RenderingPanel()

    panel.on_bind_model(_BindingContextStub(model))
    panel._subscribe_reactive_state()
    model.handle.dirty_fields.clear()

    next_tool = (
        "builtin.move"
        if module.RuntimeState.active_tool.value == "builtin.select"
        else "builtin.select"
    )
    module.RuntimeState.active_tool.value = next_tool

    assert model.handle.dirty_fields == ["__all__"]
    panel._unsubscribe_reactive_state()


def test_rendering_panel_reacts_to_native_language_generation(rendering_panel_module):
    module = rendering_panel_module
    model = _BindingModelStub()
    panel = module.RenderingPanel()

    panel.on_bind_model(_BindingContextStub(model))
    panel._subscribe_reactive_state()
    model.handle.dirty_fields.clear()

    module.RuntimeState.language_generation.value = (
        module.RuntimeState.language_generation.value + 1
    )

    assert model.handle.dirty_fields == ["__all__"]
    panel._unsubscribe_reactive_state()


def test_rendering_panel_custom_environment_map_appears_in_dropdown(rendering_panel_module):
    module = rendering_panel_module
    settings = _RenderSettingsStub(
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
    rendering_rml = project_root / "src" / "visualizer" / "gui" / "rmlui" / "resources" / "splat_simplify.rml"
    content = rendering_rml.read_text()

    assert 'data-tooltip="tooltip.simplify_source"' in content
    assert 'data-tooltip="tooltip.simplify_target"' in content
    assert 'data-tooltip="tooltip.simplify_lod_base"' in content
    assert 'data-tooltip="tooltip.simplify_opacity_prune"' in content
    assert 'data-tooltip="tooltip.simplify_output"' in content
    assert 'data-tooltip="tooltip.simplify_apply"' in content
    assert 'data-attr-data-tooltip="simplify_cancel_tooltip"' in content
    rendering_content = rendering_rml.with_name("rendering.rml").read_text(encoding="utf-8")
    custom_option_start = rendering_content.find('<option value="__custom__"')
    assert custom_option_start >= 0
    custom_option_end = rendering_content.find("</option>", custom_option_start)
    custom_option = rendering_content[custom_option_start:custom_option_end]
    assert 'data-if="environment_map_has_custom_option"' in custom_option
    assert "{{environment_map_last_custom_display_name}}" in rendering_content
    assert "{{label_simplify_source}}" in content
    assert "{{label_simplify_select_source}}" in content
    assert "{{label_simplify_target}}" in content
    assert "{{label_simplify_lod_base}}" in content
    assert "{{label_simplify_opacity_prune}}" in content
    assert "{{label_simplify_original}}" in content
    assert "{{label_simplify_target_stat}}" in content
    assert "{{label_simplify_output}}" in content
    assert "{{label_simplify_apply}}" in content
    assert "{{label_simplify_cancel}}" in content


def test_rendering_rml_only_exposes_3dgs_backends():
    project_root = Path(__file__).parent.parent.parent
    rendering_rml = project_root / "src" / "visualizer" / "gui" / "rmlui" / "resources" / "rendering.rml"
    content = rendering_rml.read_text()

    assert '<option value="3dgs">3DGS</option>' in content
    assert '<option value="3dgut">3DGUT</option>' in content


def test_custom_environment_changes_refresh_closed_select(rendering_panel_module):
    module = rendering_panel_module
    settings = SimpleNamespace(environment_mode="EQUIRECTANGULAR", environment_map_path="/tmp/first.hdr")
    module.lf.get_render_settings = lambda: settings
    writes = []
    select = SimpleNamespace(
        get_attribute=lambda _name, _default: module.CUSTOM_ENVIRONMENT_PRESET_VALUE,
        set_attribute=lambda name, value: writes.append((name, value)),
    )
    panel = module.RenderingPanel()
    panel._doc = SimpleNamespace(get_element_by_id=lambda name: select if name == "environment-map-preset" else None)
    panel._handle = _HandleStub()

    for path in ("/tmp/first.hdr", "/tmp/second.hdr", "/tmp/third.exr"):
        settings.environment_map_path = path
        writes.clear()
        assert panel._sync_environment_state() is True
        # RmlUi's closed label is a cached copy of the selected option.
        # Dirtying its text binding alone does not invalidate that copy.
        assert writes == [("value", module.CUSTOM_ENVIRONMENT_PRESET_VALUE)]
        assert panel._environment_map_last_custom_display_name() == Path(path).name
        writes.clear()
        panel._handle.dirty_fields.clear()
        assert panel._sync_environment_state() is False
        assert writes == []
        assert panel._handle.dirty_fields == []

    for index, path in enumerate(module.ENVIRONMENT_PRESET_PATHS):
        settings.environment_map_path = path
        writes.clear()
        assert panel._sync_environment_state() is True
        assert writes == []
        assert panel._environment_map_last_custom_display_name() == "third.exr"


def test_environment_sync_without_mounted_select(rendering_panel_module):
    module = rendering_panel_module
    settings = SimpleNamespace(environment_mode="EQUIRECTANGULAR", environment_map_path="/tmp/custom.hdr")
    module.lf.get_render_settings = lambda: settings
    panel = module.RenderingPanel()
    assert panel._sync_environment_state() is True
    panel._doc = SimpleNamespace(get_element_by_id=lambda _name: None)
    settings.environment_map_path = "/tmp/another.hdr"
    assert panel._sync_environment_state() is True

def test_project_navigation_vertical_does_not_change_global_defaults(rendering_panel_module):
    module = rendering_panel_module
    state = {"axis": "legacy_y"}
    calls = []
    module.lf.get_navigation_up_axis = lambda: state["axis"]
    def set_axis(axis, align_grid=False):
        state["axis"] = axis
        calls.append((axis, align_grid))
    module.lf.set_navigation_up_axis = set_axis
    panel = module.RenderingPanel()
    assert panel._navigation_up_index() == "0"
    panel._set_navigation_up_index("1")
    assert panel._navigation_up_index() == "1"
    assert calls == [("data_z", False)]
    panel._set_navigation_up_index("invalid")
    panel._set_navigation_up_index("2")
    assert len(calls) == 1
    panel._align_navigation_grid()
    assert calls[-1] == ("data_z", True)


@pytest.mark.parametrize("prop,value", [
    ("render_scale", 0.7),
    ("environment_exposure", 1.5),
    ("environment_rotation_degrees", 45.0),
    ("color_exposure", 1.2),
    ("grid_opacity", 0.3),
    ("axes_size", 3.0),
    ("camera_frustum_scale", 0.5),
    ("lod_render_scale", 0.8),
    ("lod_pool_vram_fraction", 0.3),
    ("ppisp_vignette_strength", 0.4),
])
def test_moved_rendering_controls_mark_session_only_when_scrub_commits(rendering_panel_module, prop, value):
    module = rendering_panel_module
    commits = []
    writes = []
    module.lf.ui.mark_project_session_chapter_committed = commits.append
    module.lf.get_render_settings = lambda: SimpleNamespace(
        _set_without_session_dirty=lambda name, value: writes.append((name, value))
    )
    panel = module.RenderingPanel()

    panel._set_scrub_value(prop, value)
    assert writes == [(prop, value)]
    assert commits == []

    panel._scrub_fields._commit_value(prop, value)
    assert commits == ["VIEW"]


class _PickerEvent:
    def __init__(self, x, y):
        self.values = {"mouse_x": str(x), "mouse_y": str(y)}

    def get_parameter(self, name, default="0"):
        return self.values.get(name, default)


class _PickerPopup:
    def __init__(self):
        self.props = {}
        self.classes = {}

    def set_property(self, name, value):
        self.props[name] = value

    def set_class(self, name, value):
        self.classes[name] = value


def test_selection_group_color_picker_stays_inside_panel(rendering_panel_module):
    """A swatch at the bottom of the panel opens the picker above the cursor."""
    module = rendering_panel_module
    module.lf.ui.get_ui_scale = lambda: 1.0
    panel = module.RenderingPanel()
    popup = _PickerPopup()
    body = SimpleNamespace(client_height=468.0, client_width=344.0)
    panel._popup_el = popup
    panel._doc = SimpleNamespace(get_element_by_id=lambda name: body if name == "body" else None)
    panel._handle = _HandleStub()

    panel._open_picker(_PickerEvent(44, 341))
    top = int(str(popup.props["top"]).removesuffix("px"))
    assert top + 170 <= 468
    assert top <= 341 - 170
    assert popup.props["left"] == "0px"
    assert popup.classes["visible"] is True

    panel._open_picker(_PickerEvent(300, 40))
    assert popup.props["top"] == "42px"
    assert popup.props["left"] == "90px"


@pytest.mark.parametrize("prop,value", [("axes_size", 2.0), ("show_grid", True),
    ("mesh_shadow_resolution", "2048"), ("lod_max_splats", 0),
    ("selection_outline", False), ("selection_outline_color", [0.0, .604, .733]),
    ("selection_outline_opacity", 1.0), ("selection_outline_width", 2.0),
    ("selection_outline_threshold", .5), ("selection_outline_holes", True)])
def test_label_reset_uses_registry_defaults_and_commits(rendering_panel_module, monkeypatch, prop, value):
    module = rendering_panel_module
    settings = _RenderSettingsStub(**{prop: 99})
    settings.prop_info = lambda key: {"builtin_default": value, "readonly": False}
    monkeypatch.setattr(module.lf, "get_render_settings", lambda: settings)
    panel = module.RenderingPanel()
    assert panel._reset_default(prop) == value
    assert panel._apply_reset(prop, value)
    assert getattr(settings, prop) == value


def test_render_labels_never_reset_paths_or_unregistered_properties(rendering_panel_module, monkeypatch):
    module = rendering_panel_module
    settings = _RenderSettingsStub(prop_info=lambda key: {"default": ""})
    monkeypatch.setattr(module.lf, "get_render_settings", lambda: settings)
    panel = module.RenderingPanel()
    assert panel._reset_default("environment_map_path") is None
    assert panel._reset_default("delete") is None


@pytest.mark.parametrize("info", [
    {"default": 2, "type": "enum"},
    {"default": 2, "builtin_default": None, "type": "enum"},
    {"builtin_default": "3dgs", "readonly": True},
])
def test_label_reset_requires_a_resolved_editable_default(rendering_panel_module, monkeypatch, info):
    module = rendering_panel_module
    settings = _RenderSettingsStub(prop_info=lambda key: info)
    monkeypatch.setattr(module.lf, "get_render_settings", lambda: settings)
    assert module.RenderingPanel()._reset_default("raster_backend") is None


def test_label_reset_uses_resolved_enum_identifier(rendering_panel_module, monkeypatch):
    module = rendering_panel_module
    settings = _RenderSettingsStub(prop_info=lambda key: {
        "type": "enum", "default": 2, "builtin_default": "3dgs", "default_identifier": "3dgs"})
    monkeypatch.setattr(module.lf, "get_render_settings", lambda: settings)
    assert module.RenderingPanel()._reset_default("raster_backend") == "3dgs"


@pytest.mark.parametrize("prop,value", [
    ("selection_outline_opacity", .4),
    ("selection_outline_width", 1.5),
    ("selection_outline_threshold", .7),
])
def test_outline_appearance_commits_view_settings(rendering_panel_module, prop, value):
    module = rendering_panel_module
    writes, commits = [], []
    module.lf.ui.mark_project_session_chapter_committed = commits.append
    module.lf.get_render_settings = lambda: SimpleNamespace(
        _set_without_session_dirty=lambda name, value: writes.append((name, value)))
    panel = module.RenderingPanel()
    panel._set_scrub_value(prop, value)
    assert writes == [(prop, value)]
    assert commits == []
    panel._scrub_fields._commit_value(prop, value)
    assert commits == ["VIEW"]


def test_outline_controls_reuse_color_picker_and_conditional_container(rendering_panel_module):
    from xml.etree import ElementTree as ET
    root = Path(__file__).parents[2]
    props = {"selection_outline_color", "selection_outline_opacity", "selection_outline_width",
             "selection_outline_threshold", "selection_outline_holes"}
    module = rendering_panel_module
    assert "selection_outline_color" in module.COLOR_PROPS
    assert "selection_outline_holes" in module.BOOL_PROPS
    for filename, condition, prefix in (("rendering.rml", "selection_outline", ""),
                                         ("preferences.rml", "default_selection_outline", "default_")):
        doc = ET.parse(root / "src/visualizer/gui/rmlui/resources" / filename)
        container = next(node for node in doc.iter() if node.get("data-if") == condition)
        bindings = {node.get("data-value", node.get("data-checked", "")) for node in container.iter()}
        for prop in props:
            assert prefix + prop + ("_hex" if prop.endswith("_color") and not prefix else "") in bindings


def test_outline_defaults_validate_and_save_only_committed_values(rendering_panel_module, monkeypatch):
    import json
    module = import_module("lfs_plugins.rendering_defaults")
    monkeypatch.setattr(module, "lf", rendering_panel_module.lf)
    saved, patches = {}, []
    module.lf.ui.get_rendering_defaults_json = lambda: json.dumps(saved)
    def persist(text):
        patch = json.loads(text)
        patches.append(patch)
        saved.update(patch)
    module.lf.ui.set_rendering_defaults_json = persist
    section = module.RenderingDefaultsSection()
    model = _BindingModelStub()
    section.bind(model)
    get, set_value = model.bindings["default_selection_outline_opacity"]
    set_value("0.25")
    assert patches == []
    section._commit("selection_outline_opacity")
    assert patches == [{"selection_outline_opacity": .25}]
    section._set("selection_outline_opacity", "NaN")
    section._set("selection_outline_width", "13")
    assert len(patches) == 1
    section._set("selection_outline_width", "12")
    assert patches[-1] == {"selection_outline_width": 12}
    section._set("selection_outline_color", "#336699")
    assert saved["selection_outline_color"] == pytest.approx([.2, .4, .6])


def test_outline_patterns_are_not_exposed(rendering_panel_module):
    module = rendering_panel_module
    assert "selection_outline_style" not in module.SELECT_PROPS
    for name in ("rendering.rml", "preferences.rml"):
        text = (Path(__file__).resolve().parents[2] / "src/visualizer/gui/rmlui/resources" / name).read_text()
        assert "selection_outline_style" not in text
        assert "selection_outline_dashed" not in text
        assert "selection_outline_dotted" not in text


@pytest.mark.parametrize("prop,edited,default", [
    ("selection_outline", True, False),
    ("selection_outline_color", [.1, .2, .3], [0., .604, .733]),
    ("selection_outline_opacity", .3, 1.),
    ("selection_outline_width", 9., 2.),
    ("selection_outline_threshold", .8, .5),
    ("selection_outline_holes", False, True),
])
def test_outline_new_project_defaults_reset_without_applying_to_scene(
        rendering_panel_module, monkeypatch, prop, edited, default):
    import json
    module = import_module("lfs_plugins.rendering_defaults")
    monkeypatch.setattr(module, "lf", rendering_panel_module.lf)
    saved = {prop: edited}
    monkeypatch.setattr(module.lf.ui, "get_rendering_defaults_json", lambda: json.dumps(saved), raising=False)
    monkeypatch.setattr(module.lf.ui, "set_rendering_defaults_json",
                        lambda text: saved.update(json.loads(text)), raising=False)
    def unexpected_scene_access():
        raise AssertionError("Resetting new-project defaults must not access the active scene")
    monkeypatch.setattr(module.lf, "get_render_settings", unexpected_scene_access)
    section = module.RenderingDefaultsSection()
    section._text[prop] = "stale draft"
    value = section._default_reset._default_value(prop)
    assert value == default
    assert section._default_reset._apply_value(prop, value)
    assert prop not in section._text
    assert saved[prop] == pytest.approx(default)


@pytest.fixture
def controls(rendering_panel_module, monkeypatch):
    from lfs_plugins import collision_panel as module
    lf = MagicMock()
    lf.ui.tr.side_effect = lambda key: key
    lf.get_collision_state.return_value = {"walk_settings": (.5, .1, .2, 4.)}
    lf.get_collision_state.side_effect = lambda: dict(lf.get_collision_state.return_value)
    lf.set_walk_settings.side_effect = lambda *settings: lf.get_collision_state.return_value.update(walk_settings=settings)
    monkeypatch.setattr(module, "lf", lf)
    monkeypatch.setitem(sys.modules, "lichtfeld", lf)
    return module.CollisionPanel()


def test_completed_scrub_edits_recover_and_walk_merges_live_settings(controls):
    from lfs_plugins import collision_panel as module
    controls.update()
    scrub, prop = controls.scrubs, "step"
    keys = attach_fields(scrub)
    field = scrub._fields[prop]
    field.input_el.blur = lambda: scrub._on_input_blur(prop, None)
    enter = Event(key_identifier=keys.KI_RETURN)
    scrub._on_field_keydown(prop, enter)
    for draft in ("0", "0.", "0.35"):
        field.input_el.set_attribute("value", draft)
        scrub._on_input_change(prop, Event())
        assert controls.values["step"] == .2
    module.lf.get_collision_state.return_value = {"walk_settings": (.6, .15, .25, 5.)}
    controls.update()
    assert field.input_el.get_attribute("value") == "0.35"
    scrub._on_input_change(prop, Event(linebreak=True))
    assert not field.editing
    module.lf.set_walk_settings.assert_called_once_with(.6, .15, .35, 5.)
    blocked = RuntimeError("The edited Walk capsule does not fit")
    blocked.message_key = "collision.walk_blocked"
    blocked.message_details = "/tmp/collision.walk_no_support/64000001.bin"
    module.lf.set_walk_settings.side_effect = blocked
    for draft, error in ((".9", f"{blocked.message_key}\n{blocked.message_details}"),
                         ("nan", "preferences.viewport_defaults_invalid")):
        scrub._on_field_keydown(prop, enter)
        field.input_el.set_attribute("value", draft)
        scrub._on_input_change(prop, Event(linebreak=True))
        assert controls.values["step"] == .35
        assert controls.values["error"] == error and not field.editing
    scrub._on_field_keydown(prop, enter)
    assert field.editing
    assert controls.localized_error(RuntimeError(blocked.message_details)) == blocked.message_details


@pytest.mark.parametrize("approved,choice,error", [
    (False, "common.cancel", ""), (False, "export.export", ""),
    (True, None, ""), (False, None, ""), (True, None, "destination denied")])
def test_export_picker_and_pair_confirmation_latch_target(controls, tmp_path, approved, choice, error):
    from lfs_plugins import collision_panel as module
    path = tmp_path / "collision.voxel.json"
    path.with_suffix(".bin").write_bytes(b"old")
    controls.export_target = ("first", "Floor")
    def pick(*args, **kwargs):
        assert args == ("collision.voxel.json",) and not kwargs
        controls.export_target = ("second", "Wall")
        return (str(path) if approved or choice else "", approved)
    module.lf.ui.save_json_file_dialog_with_confirmation.side_effect = pick
    if error:
        module.lf.export_collision.side_effect = RuntimeError(error)
    controls.run(controls.export)
    if choice:
        module.lf.ui.confirm_dialog.assert_called_once()
        prompt = module.lf.ui.confirm_dialog.call_args.args
        assert prompt[1] == f"collision.replace_pair\n{path}\n{path.with_suffix('.bin')}"
        controls.export_target = ("third", "Ceiling")
        prompt[-1](choice)
    else:
        module.lf.ui.confirm_dialog.assert_not_called()
    expected = [("first", str(path))] if approved or choice == "export.export" else []
    assert [call.args for call in module.lf.export_collision.call_args_list] == expected
    assert controls.values["error"] == error
    assert path.with_suffix(".bin").read_bytes() == b"old" and not path.exists()
