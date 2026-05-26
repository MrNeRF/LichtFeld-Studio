# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for hook-driven viewport toolbar updates."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import json
import sys

import pytest


def _install_stub_modules(monkeypatch):
    hook_calls = []
    remove_calls = []

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        add_hook=lambda panel, section, callback, position="append": hook_calls.append(
            (panel, section, callback, position)
        ),
        remove_hook=lambda panel, section, callback: remove_calls.append(
            (panel, section, callback)
        ),
        rml=SimpleNamespace(get_document=lambda _name: None),
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)

    tools_mod = ModuleType("lfs_plugins.tools")

    class _ToolRegistryStub:
        @staticmethod
        def get_all():
            return []

        @staticmethod
        def get(_tool_id):
            return None

        @staticmethod
        def clear_active():
            return None

        @staticmethod
        def set_active(_tool_id):
            return None

    tools_mod.ToolRegistry = _ToolRegistryStub
    monkeypatch.setitem(sys.modules, "lfs_plugins.tools", tools_mod)

    op_context_mod = ModuleType("lfs_plugins.op_context")
    op_context_mod.get_context = lambda: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "lfs_plugins.op_context", op_context_mod)

    ui_pkg = ModuleType("lfs_plugins.ui")
    ui_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "lfs_plugins.ui", ui_pkg)

    state_mod = ModuleType("lfs_plugins.ui.state")
    state_mod.AppState = SimpleNamespace(trainer_state=SimpleNamespace(value="idle"))
    monkeypatch.setitem(sys.modules, "lfs_plugins.ui.state", state_mod)

    return hook_calls, remove_calls


class _DataModelHandleStub:
    def __init__(self):
        self.dirty_all_calls = 0
        self.dirty_calls = []
        self.record_updates = {}

    def dirty_all(self):
        self.dirty_all_calls += 1

    def dirty(self, name):
        self.dirty_calls.append(name)

    def update_record_list(self, name, records):
        self.record_updates[name] = records


class _DataModelStub:
    def __init__(self):
        self.bound_binds = {}
        self.bound_funcs = {}
        self.bound_events = {}
        self.bound_record_lists = []
        self.handle = _DataModelHandleStub()

    def bind(self, name, getter, setter):
        self.bound_binds[name] = (getter, setter)

    def bind_func(self, name, getter):
        self.bound_funcs[name] = getter

    def bind_event(self, name, callback):
        self.bound_events[name] = callback

    def bind_record_list(self, name):
        self.bound_record_lists.append(name)

    def get_handle(self):
        return self.handle


@pytest.fixture
def toolbar_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) in sys.path:
        sys.path.remove(str(source_python))
    sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins", None)
    sys.modules.pop("lfs_plugins.toolbar", None)
    sys.modules.pop("lfs_plugins.selection_controls", None)
    sys.modules.pop("lfs_plugins.transform_controls", None)
    hook_calls, remove_calls = _install_stub_modules(monkeypatch)
    module = import_module("lfs_plugins.toolbar")
    return module, hook_calls, remove_calls


def test_toolbar_binds_overlay_model_fields(toolbar_module):
    module, _hook_calls, _remove_calls = toolbar_module
    model = _DataModelStub()

    module.reset_overlay_state()
    module.bind_overlay_model(model)

    assert "show_render_controls" in model.bound_funcs
    assert "camera_flyout_open" not in model.bound_funcs
    assert "render_flyout_open" not in model.bound_funcs
    assert "selection_flyout_open" not in model.bound_funcs
    assert "transform_flyout_open" not in model.bound_funcs
    assert "selection_group_buttons" in model.bound_record_lists
    assert "selection_mode_buttons" in model.bound_record_lists
    assert "transform_group_buttons" in model.bound_record_lists
    assert "transform_tool_buttons" in model.bound_record_lists
    assert "camera_group_buttons" in model.bound_record_lists
    assert "render_group_buttons" in model.bound_record_lists
    assert "camera_mode_buttons" in model.bound_record_lists
    assert "render_mode_buttons" in model.bound_record_lists
    assert "toolbar_action" in model.bound_events
    assert "selection_tool_label" in model.bound_funcs
    assert "selection_mode_label" in model.bound_funcs
    assert "selection_depth_mode_active" in model.bound_funcs
    assert "selection_can_delete" in model.bound_funcs
    assert "selection_can_undo" in model.bound_funcs
    assert "selection_can_redo" in model.bound_funcs
    assert "selection_depth_near_str" in model.bound_binds
    assert "selection_depth_far_str" in model.bound_binds
    assert "selection_depth_near_slider_min" in model.bound_funcs
    assert "selection_depth_near_slider_max" in model.bound_funcs
    assert "selection_depth_far_slider_min" in model.bound_funcs
    assert "selection_depth_far_slider_max" in model.bound_funcs
    assert "selection_action" in model.bound_events
    assert "transform_tool_label" in model.bound_funcs
    assert "transform_bake_label" in model.bound_funcs
    assert "transform_show_actions" in model.bound_funcs
    assert "transform_submode_label" in model.bound_funcs
    assert "transform_submode_tooltip_key" in model.bound_funcs
    assert "transform_pos_x_str" in model.bound_binds
    assert "transform_num_step" in model.bound_events


def test_toolbar_attach_handle_marks_model_dirty(toolbar_module):
    module, _hook_calls, _remove_calls = toolbar_module
    handle = _DataModelHandleStub()

    module.reset_overlay_state()
    module.attach_overlay_model_handle(handle)

    assert handle.dirty_all_calls == 1


def test_utility_group_button_preserves_active_state(toolbar_module):
    module, _hook_calls, _remove_calls = toolbar_module

    inactive = module._button_record(
        "util-camera-orbit",
        "set_camera_navigation_mode",
        "orbit",
        "../icon/camera-orbit.png",
        tooltip_text="Orbit Camera",
        selected=False,
    )
    active = module._button_record(
        "util-camera-trackball",
        "set_camera_navigation_mode",
        "trackball",
        "../icon/world.png",
        tooltip_text="Free Orbit Camera",
        selected=True,
    )

    group_button = module._UtilityToolbarController._group_button(
        "camera",
        [inactive, active],
        "Camera Mode",
    )[0]

    assert group_button["button_id"] == "group-camera"
    assert group_button["action"] == "noop"
    assert group_button["value"] == "camera"
    assert group_button["icon_src"] == "../icon/world.png"
    assert group_button["selected"] is True


def test_selection_tool_uses_flyout_modes(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    lf_stub = sys.modules["lichtfeld"]
    state = SimpleNamespace(active_tool="", active_submode="")
    select_tool = SimpleNamespace(
        id="builtin.select",
        icon="selection",
        label="Select",
        shortcut="1",
        submodes=(
            SimpleNamespace(id="centers", label="Centers", icon="circle-dot", shortcut=""),
            SimpleNamespace(id="rectangle", label="Rectangle", icon="rectangle", shortcut=""),
            SimpleNamespace(id="lasso", label="Lasso", icon="lasso", shortcut=""),
        ),
        pivot_modes=(),
        selected=None,
        can_activate=lambda _context: True,
    )

    monkeypatch.setattr(lf_stub.ui, "get_active_tool", lambda: state.active_tool, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_active_submode", lambda: state.active_submode, raising=False)
    monkeypatch.setattr(lf_stub.ui, "set_selection_mode", lambda mode: setattr(state, "active_submode", mode), raising=False)
    monkeypatch.setattr(module.ToolRegistry, "get_all", staticmethod(lambda: [select_tool]), raising=False)
    monkeypatch.setattr(
        module.ToolRegistry,
        "get",
        staticmethod(lambda tool_id: select_tool if tool_id == "builtin.select" else None),
        raising=False,
    )
    monkeypatch.setattr(
        module.ToolRegistry,
        "set_active",
        staticmethod(lambda tool_id: setattr(state, "active_tool", tool_id)),
        raising=False,
    )
    monkeypatch.setattr(
        module.ToolRegistry,
        "clear_active",
        staticmethod(lambda: setattr(state, "active_tool", "")),
        raising=False,
    )

    controller = module._GizmoToolbarController()
    snapshot = controller.snapshot()

    assert snapshot["show_selection_controls"] is True
    assert snapshot["show_gizmo_toolbar"] is False
    assert snapshot["show_submode_toolbar"] is False
    assert snapshot["selection_group_buttons"][0]["button_id"] == "group-selection"
    assert snapshot["selection_group_buttons"][0]["action"] == "tool"
    assert snapshot["selection_group_buttons"][0]["value"] == "builtin.select"
    assert snapshot["selection_group_buttons"][0]["icon_src"] == "../icon/selection.png"
    assert [button["action"] for button in snapshot["selection_mode_buttons"]] == [
        "selection_mode",
        "selection_mode",
        "selection_mode",
    ]

    controller.dispatch("selection_mode", "lasso")

    snapshot = controller.snapshot()
    assert state.active_tool == "builtin.select"
    assert state.active_submode == "lasso"
    assert snapshot["selection_group_buttons"][0]["selected"] is True
    assert snapshot["selection_group_buttons"][0]["icon_src"] == "../icon/lasso.png"
    assert next(button for button in snapshot["selection_mode_buttons"] if button["value"] == "lasso")["selected"] is True

    controller.dispatch("tool", "builtin.select")
    snapshot = controller.snapshot()

    assert state.active_tool == ""
    assert snapshot["selection_group_buttons"][0]["selected"] is False


def test_transform_tools_use_flyout_group(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    lf_stub = sys.modules["lichtfeld"]
    state = SimpleNamespace(active_tool="", transform_space=1, pivot_mode=0, mirror_calls=[])

    transform_submodes = (
        SimpleNamespace(id="local", label="Local", icon="local", shortcut=""),
        SimpleNamespace(id="world", label="World", icon="world", shortcut=""),
    )
    transform_pivots = (
        SimpleNamespace(id="origin", label="Origin", icon="circle-dot"),
        SimpleNamespace(id="bounds", label="Bounds", icon="box"),
    )
    mirror_submodes = (
        SimpleNamespace(id="x", label="X Axis", icon="mirror-x", shortcut=""),
        SimpleNamespace(id="y", label="Y Axis", icon="mirror-y", shortcut=""),
        SimpleNamespace(id="z", label="Z Axis", icon="mirror-z", shortcut=""),
    )

    def _tool(tool_id, label, icon, shortcut, submodes=(), pivot_modes=()):
        return SimpleNamespace(
            id=tool_id,
            icon=icon,
            label=label,
            shortcut=shortcut,
            group="transform",
            submodes=submodes,
            pivot_modes=pivot_modes,
            selected=None,
            can_activate=lambda _context: True,
        )

    translate_tool = _tool(
        "builtin.translate",
        "Move",
        "translation",
        "2",
        submodes=transform_submodes,
        pivot_modes=transform_pivots,
    )
    rotate_tool = _tool("builtin.rotate", "Rotate", "rotation", "3")
    scale_tool = _tool("builtin.scale", "Scale", "scaling", "4")
    mirror_tool = _tool("builtin.mirror", "Mirror", "mirror", "5", submodes=mirror_submodes)
    brush_tool = SimpleNamespace(
        id="builtin.brush",
        icon="painting",
        label="Brush",
        shortcut="6",
        group="paint",
        submodes=(),
        pivot_modes=(),
        selected=None,
        can_activate=lambda _context: True,
    )
    tools = [translate_tool, rotate_tool, scale_tool, mirror_tool, brush_tool]
    by_id = {tool.id: tool for tool in tools}

    monkeypatch.setattr(lf_stub.ui, "get_active_tool", lambda: state.active_tool, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_active_submode", lambda: "", raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_transform_space", lambda: state.transform_space, raising=False)
    monkeypatch.setattr(lf_stub.ui, "set_transform_space", lambda value: setattr(state, "transform_space", value), raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_pivot_mode", lambda: state.pivot_mode, raising=False)
    monkeypatch.setattr(lf_stub.ui, "set_pivot_mode", lambda value: setattr(state, "pivot_mode", value), raising=False)
    monkeypatch.setattr(lf_stub.ui, "execute_mirror", lambda axis: state.mirror_calls.append(axis), raising=False)
    monkeypatch.setattr(module.ToolRegistry, "get_all", staticmethod(lambda: tools), raising=False)
    monkeypatch.setattr(module.ToolRegistry, "get", staticmethod(lambda tool_id: by_id.get(tool_id)), raising=False)
    monkeypatch.setattr(
        module.ToolRegistry,
        "set_active",
        staticmethod(lambda tool_id: setattr(state, "active_tool", tool_id)),
        raising=False,
    )
    monkeypatch.setattr(
        module.ToolRegistry,
        "clear_active",
        staticmethod(lambda: setattr(state, "active_tool", "")),
        raising=False,
    )

    controller = module._GizmoToolbarController()
    snapshot = controller.snapshot()

    assert snapshot["show_transform_controls"] is True
    assert snapshot["transform_group_buttons"][0]["button_id"] == "group-transform"
    assert snapshot["transform_group_buttons"][0]["action"] == "tool"
    assert snapshot["transform_group_buttons"][0]["value"] == "builtin.translate"
    assert snapshot["transform_group_buttons"][0]["icon_src"] == "../icon/translation.png"
    assert [button["value"] for button in snapshot["transform_tool_buttons"]] == [
        "builtin.translate",
        "builtin.rotate",
        "builtin.scale",
        "builtin.mirror",
    ]
    assert [button["value"] for button in snapshot["gizmo_buttons"]] == ["builtin.brush"]

    controller.dispatch("tool", "builtin.translate")
    snapshot = controller.snapshot()

    assert state.active_tool == "builtin.translate"
    assert snapshot["transform_group_buttons"][0]["selected"] is True
    assert snapshot["transform_group_buttons"][0]["icon_src"] == "../icon/translation.png"
    assert next(button for button in snapshot["transform_tool_buttons"] if button["value"] == "builtin.translate")["selected"] is True
    assert snapshot["show_submode_toolbar"] is True
    assert snapshot["show_pivot_toolbar"] is True
    assert next(button for button in snapshot["submode_buttons"] if button["value"] == "world")["selected"] is True
    assert next(button for button in snapshot["pivot_buttons"] if button["value"] == "origin")["selected"] is True

    controller.dispatch("submode", "local")
    controller.dispatch("pivot", "bounds")
    snapshot = controller.snapshot()

    assert state.transform_space == 0
    assert state.pivot_mode == 1
    assert next(button for button in snapshot["submode_buttons"] if button["value"] == "local")["selected"] is True
    assert next(button for button in snapshot["pivot_buttons"] if button["value"] == "bounds")["selected"] is True

    controller.dispatch("tool", "builtin.translate")
    snapshot = controller.snapshot()

    assert state.active_tool == ""
    assert snapshot["transform_group_buttons"][0]["selected"] is False

    controller.dispatch("tool", "builtin.mirror")
    snapshot = controller.snapshot()

    assert state.active_tool == "builtin.mirror"
    assert snapshot["transform_group_buttons"][0]["selected"] is True
    assert snapshot["transform_group_buttons"][0]["icon_src"] == "../icon/mirror.png"
    assert next(button for button in snapshot["transform_tool_buttons"] if button["value"] == "builtin.mirror")["selected"] is True
    assert snapshot["show_submode_toolbar"] is True
    assert snapshot["show_pivot_toolbar"] is False
    assert [button["value"] for button in snapshot["submode_buttons"]] == ["x", "y", "z"]

    controller.dispatch("submode", "y")

    assert state.mirror_calls == ["y"]

    controller.dispatch("tool", "builtin.translate")
    controller.dispatch("tool", "builtin.translate")
    snapshot = controller.snapshot()

    assert state.active_tool == ""
    assert snapshot["transform_group_buttons"][0]["selected"] is False


def test_viewport_overlay_template_moves_tools_left_and_transform_numbers_centered():
    project_root = Path(__file__).parent.parent.parent
    rml_path = (
        project_root
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "viewport_overlay.rml"
    )
    rcss_path = rml_path.with_suffix(".rcss")
    rml = rml_path.read_text(encoding="utf-8")
    rcss = rcss_path.read_text(encoding="utf-8")
    rendering_rml = (
        project_root
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "rendering.rml"
    ).read_text(encoding="utf-8")
    locale_dir = project_root / "src" / "visualizer" / "gui" / "resources" / "locales"
    transform_tooltip_keys = (
        "transform_panel",
        "transform_tool",
        "transform_target",
        "transform_pivot",
        "transform_bake",
        "transform_reset",
        "transform_position",
        "transform_position_axis",
        "transform_rotation",
        "transform_rotation_axis",
        "transform_scale",
        "transform_scale_axis",
        "transform_scale_uniform",
    )
    transform_dynamic_tooltip_keys = (
        "transform_space",
        "transform_axis",
    )
    transform_toolbar_tooltip_keys = (
        "local_space",
        "world_space",
        "origin_pivot",
        "bounds_center_pivot",
    )
    selection_tooltip_keys = (
        "selection_panel",
        "selection_tool",
        "selection_mode",
        "selection_depth_range",
        "selection_depth_near",
        "selection_depth_far",
        "selection_depth_mode",
        "selection_delete",
        "selection_undo",
        "selection_redo",
        "selection_invert",
        "selection_select_all",
        "selection_unselect",
    )

    assert "primary-gizmo-toolbar" not in rml
    assert "secondary-gizmo-toolbar" not in rml
    assert "primary-submode-toolbar" not in rml
    assert "secondary-submode-toolbar" not in rml
    assert "primary-pivot-toolbar" not in rml
    assert "secondary-pivot-toolbar" not in rml
    assert "toolbar-context-stack" not in rml
    assert rml.count('data-for="button : gizmo_buttons"') == 2
    assert rml.count('data-for="button : submode_buttons"') == 1
    assert rml.count('data-for="button : pivot_buttons"') == 1
    assert 'class="toolbar-flyout-divider hidden"' not in rml
    assert rml.count('class="toolbar-flyout toolbar-flyout-selection hidden"') == 2
    assert rml.count('class="toolbar-flyout toolbar-flyout-transform hidden"') == 2
    assert rml.count('data-for="button : selection_group_buttons"') == 2
    assert rml.count('data-for="button : selection_mode_buttons"') == 2
    assert rml.count('data-for="button : transform_group_buttons"') == 2
    assert rml.count('data-for="button : transform_tool_buttons"') == 2
    assert 'id="selection-block"' in rml
    assert 'class="viewport-selection-overlay hidden"' in rml
    assert 'data-event-click="selection_action(\'toggle_depth\')"' in rml
    assert 'data-event-click="selection_action(\'delete\')"' in rml
    assert 'data-event-click="selection_action(\'undo\')"' in rml
    assert 'data-event-click="selection_action(\'redo\')"' in rml
    assert 'data-event-click="selection_action(\'invert\')"' in rml
    assert 'data-event-click="selection_action(\'select_all\')"' in rml
    assert 'data-event-click="selection_action(\'unselect\')"' in rml
    assert 'data-value="selection_depth_near_value"' in rml
    assert 'data-value="selection_depth_far_value"' in rml
    assert 'data-attr-min="selection_depth_near_slider_min"' in rml
    assert 'data-attr-max="selection_depth_far_slider_max"' in rml
    assert "../icon/depth-map.png" in rml
    assert "../icon/contrast.png" in rml
    assert "../icon/scene/trash.png" in rml
    assert "../icon/scene/x.png" in rml
    assert 'id="transform-block"' in rml
    assert 'class="viewport-transform-overlay hidden"' in rml
    assert 'class="viewport-transform-header"' in rml
    assert 'class="viewport-transform-context"' in rml
    assert 'class="viewport-transform-actions"' in rml
    assert 'data-if="transform_show_actions"' in rml
    assert 'data-attr-data-tooltip="transform_submode_tooltip_key"' in rml
    assert "{{transform_submode_label}}" in rml
    assert 'data-event-click="transform_action(\'bake\')"' in rml
    assert 'data-event-click="transform_action(\'reset\')"' in rml
    assert "../icon/check.png" in rml
    assert "../icon/reset.png" in rml
    for key in selection_tooltip_keys:
        assert f'data-tooltip="tooltip.{key}"' in rml
    for key in transform_tooltip_keys:
        assert f'data-tooltip="tooltip.{key}"' in rml
    for locale_path in sorted(locale_dir.glob("*.json")):
        data = json.loads(locale_path.read_text(encoding="utf-8"))
        for key in selection_tooltip_keys:
            assert data.get("tooltip", {}).get(key), f"{locale_path.name} missing tooltip.{key}"
        for key in (*transform_tooltip_keys, *transform_dynamic_tooltip_keys):
            assert data.get("tooltip", {}).get(key), f"{locale_path.name} missing tooltip.{key}"
        for key in transform_toolbar_tooltip_keys:
            assert data.get("toolbar", {}).get(key), f"{locale_path.name} missing toolbar.{key}"
    assert "Space: {{transform_space_label}}" not in rml
    assert "Pivot: {{transform_pivot_label}}" not in rml
    assert 'id="transform-block"' not in rendering_rml
    assert "toolbar-flyout-anchor" not in rml
    assert rml.count('class="toolbar-flyout toolbar-flyout-camera hidden"') == 2
    assert rml.count('class="toolbar-flyout toolbar-flyout-render hidden"') == 2
    assert rml.count('<span class="flyout-corner-marker"></span>') == 8
    assert "dropdown-arrow.png" not in rml
    assert "flyout_open" not in rml
    assert 'data-for="button : camera_mode_buttons"' in rml
    assert 'data-for="button : render_mode_buttons"' in rml
    assert 'data-class-selected="button.selected"' in rml
    assert "toolbar-flyout-camera" in rcss
    assert ".toolbar-group-container:hover > .toolbar-flyout.hidden" in rcss
    assert "width: 96dp;" in rcss
    assert "toolbar-flyout-selection" in rcss
    assert "width: 192dp;" in rcss
    assert "toolbar-flyout-transform" in rcss
    assert "min-width: 128dp;" in rcss
    assert "toolbar-context-stack" not in rcss
    assert "toolbar-flyout-divider" not in rcss
    assert "viewport-transform-context" in rcss
    assert "viewport-transform-option" in rcss
    assert "viewport-transform-actions" in rcss
    assert "viewport-transform-action" in rcss
    assert "max-width: 100%;" in rcss
    assert "flex-direction: column;" in rcss
    assert "height: 22dp;" in rcss
    assert "line-height: 16dp;" in rcss
    assert "line-height: 20dp;" in rcss
    assert "width: 64dp;" in rcss
    assert "toolbar-flyout-render" in rcss
    assert "width: 128dp;" in rcss
    assert "left: 100%;" in rcss
    assert ".toolbar-flyout-trigger.hidden" not in rcss
    assert "right: 0;" in rcss
    assert "bottom: 0;" in rcss
    assert "width: 0;" in rcss
    assert "height: 0;" in rcss
    assert "border-left-color: rgba(0, 0, 0, 0);" in rcss
    assert "border-bottom-width: 9dp;" in rcss
    assert ".viewport-transform-overlay" in rcss
    assert ".viewport-selection-overlay" in rcss
    assert ".viewport-selection-panel" in rcss
    assert ".viewport-selection-depth-fields" in rcss
    assert ".viewport-selection-slider" in rcss
    assert ".viewport-transform-panel" in rcss


def test_viewport_toolbar_update_syncs_camera_group_records(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    model = _DataModelStub()
    lf_stub = sys.modules["lichtfeld"]

    lf_stub.RenderMode = SimpleNamespace(
        SPLATS="splats",
        POINTS="points",
        RINGS="rings",
        CENTERS="centers",
    )
    lf_stub.get_camera_navigation_mode = lambda: "orbit"
    lf_stub.get_camera_view_snap_enabled = lambda: False
    lf_stub.get_render_mode = lambda: lf_stub.RenderMode.SPLATS
    lf_stub.is_fullscreen = lambda: False
    lf_stub.is_orthographic = lambda: False
    lf_stub.get_depth_view = lambda: False
    lf_stub.get_selected_node_names = lambda: []
    monkeypatch.setattr(lf_stub.ui, "context", lambda: SimpleNamespace(), raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_active_tool", lambda: "", raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_transform_space", lambda: 1, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_pivot_mode", lambda: 0, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_split_view_mode", lambda: "single", raising=False)
    monkeypatch.setattr(lf_stub.ui, "is_sequencer_visible", lambda: False, raising=False)
    monkeypatch.setattr(lf_stub.ui, "is_panel_enabled", lambda _panel_id: False, raising=False)
    monkeypatch.setattr(module, "histogram_mode_available", lambda _context: False)

    module.reset_overlay_state()
    module.bind_overlay_model(model)
    module.attach_overlay_model_handle(model.handle)
    model.handle.record_updates.clear()

    module.update_overlay(SimpleNamespace())

    camera_buttons = model.handle.record_updates["camera_mode_buttons"]
    camera_group = model.handle.record_updates["camera_group_buttons"][0]
    assert len(camera_buttons) == 3
    assert camera_group["action"] == "noop"
    assert camera_group["icon_src"] == "../icon/camera-orbit.png"
    assert camera_group["selected"] is True


def test_toolbar_tool_action_refreshes_button_records_immediately(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    model = _DataModelStub()
    lf_stub = sys.modules["lichtfeld"]
    state = SimpleNamespace(active_tool="")

    def _tool(tool_id, label, icon, shortcut):
        return SimpleNamespace(
            id=tool_id,
            icon=icon,
            label=label,
            shortcut=shortcut,
            group="transform",
            submodes=(),
            pivot_modes=(),
            selected=None,
            can_activate=lambda _context: True,
        )

    tools = [
        _tool("builtin.translate", "Move", "translation", "2"),
        _tool("builtin.rotate", "Rotate", "rotation", "3"),
    ]
    by_id = {tool.id: tool for tool in tools}

    lf_stub.RenderMode = SimpleNamespace(
        SPLATS="splats",
        POINTS="points",
        RINGS="rings",
        CENTERS="centers",
    )
    lf_stub.get_camera_navigation_mode = lambda: "orbit"
    lf_stub.get_camera_view_snap_enabled = lambda: False
    lf_stub.get_render_mode = lambda: lf_stub.RenderMode.SPLATS
    lf_stub.is_fullscreen = lambda: False
    lf_stub.is_orthographic = lambda: False
    lf_stub.get_depth_view = lambda: False
    monkeypatch.setattr(lf_stub.ui, "context", lambda: SimpleNamespace(), raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_active_tool", lambda: state.active_tool, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_active_submode", lambda: "", raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_transform_space", lambda: 1, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_pivot_mode", lambda: 0, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_split_view_mode", lambda: "single", raising=False)
    monkeypatch.setattr(lf_stub.ui, "is_sequencer_visible", lambda: False, raising=False)
    monkeypatch.setattr(lf_stub.ui, "is_panel_enabled", lambda _panel_id: False, raising=False)
    monkeypatch.setattr(module, "histogram_mode_available", lambda _context: False)
    monkeypatch.setattr(module.ToolRegistry, "get_all", staticmethod(lambda: tools), raising=False)
    monkeypatch.setattr(module.ToolRegistry, "get", staticmethod(lambda tool_id: by_id.get(tool_id)), raising=False)
    monkeypatch.setattr(
        module.ToolRegistry,
        "set_active",
        staticmethod(lambda tool_id: setattr(state, "active_tool", tool_id)),
        raising=False,
    )
    monkeypatch.setattr(
        module.ToolRegistry,
        "clear_active",
        staticmethod(lambda: setattr(state, "active_tool", "")),
        raising=False,
    )

    module.reset_overlay_state()
    module.bind_overlay_model(model)
    module.attach_overlay_model_handle(model.handle)
    module.update_overlay(SimpleNamespace())
    model.handle.record_updates.clear()

    model.bound_events["toolbar_action"](None, None, ["tool", "builtin.translate"])

    assert state.active_tool == "builtin.translate"
    transform_group = model.handle.record_updates["transform_group_buttons"][0]
    assert transform_group["selected"] is True
    assert transform_group["value"] == "builtin.translate"
    assert transform_group["icon_src"] == "../icon/translation.png"
    translate_button = next(
        button
        for button in model.handle.record_updates["transform_tool_buttons"]
        if button["value"] == "builtin.translate"
    )
    assert translate_button["selected"] is True

    model.handle.record_updates.clear()
    model.bound_events["toolbar_action"](None, None, ["tool", "builtin.rotate"])

    assert state.active_tool == "builtin.rotate"
    transform_group = model.handle.record_updates["transform_group_buttons"][0]
    assert transform_group["selected"] is True
    assert transform_group["value"] == "builtin.rotate"
    assert transform_group["icon_src"] == "../icon/rotation.png"
    rotate_button = next(
        button
        for button in model.handle.record_updates["transform_tool_buttons"]
        if button["value"] == "builtin.rotate"
    )
    assert rotate_button["selected"] is True
