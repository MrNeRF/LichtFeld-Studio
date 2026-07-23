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
        get_active_tool=lambda: "",
        get_active_submode=lambda: "",
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
    ui_pkg.RuntimeState = SimpleNamespace(trainer_state=SimpleNamespace(value="idle"))
    ui_pkg.native_value = lambda _field, fallback: fallback
    monkeypatch.setitem(sys.modules, "lfs_plugins.ui", ui_pkg)

    return hook_calls, remove_calls


class _DataModelHandleStub:
    def __init__(self):
        self.dirty_all_calls = 0
        self.dirty_calls = []
        self.request_update_calls = 0
        self.record_updates = {}

    def dirty_all(self):
        self.dirty_all_calls += 1

    def dirty(self, name):
        self.dirty_calls.append(name)

    def request_update(self):
        self.request_update_calls += 1

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


class _ElementStub:
    def __init__(self, element_id):
        self.element_id = element_id
        self.classes = {"hidden"}
        self.attributes = {}
        self.properties = {}
        self.text = ""
        self.animations = []
        self.listeners = {}

    def set_class(self, name, enabled):
        if enabled:
            self.classes.add(name)
        else:
            self.classes.discard(name)

    def set_text(self, text):
        self.text = text

    def get_attribute(self, name, fallback=""):
        return self.attributes.get(name, fallback)

    def set_attribute(self, name, value):
        self.attributes[name] = value

    def set_property(self, name, value):
        self.properties[name] = value
        return True

    def remove_property(self, name):
        self.properties.pop(name, None)

    def animate(
        self,
        property_name,
        target_value,
        duration,
        tween="quadratic-out",
        start_value=None,
        remove_on_complete=False,
    ):
        self.animations.append(
            (
                property_name,
                target_value,
                duration,
                tween,
                start_value,
                remove_on_complete,
            )
        )
        return True

    def add_event_listener(self, event_name, callback):
        self.listeners.setdefault(event_name, []).append(callback)


class _DocumentStub:
    def __init__(self):
        self.elements = {
            element_id: _ElementStub(element_id)
            for element_id in (
                "overlay-body",
                "dm-root",
                "depth-view-block",
                "viewport-export-block",
                "viewport-export-status",
                "selection-block",
                "transform-block",
            )
        }

    def get_element_by_id(self, element_id):
        return self.elements.get(element_id)

    def add_event_listener(self, event_name, callback):
        pass

    def query_selector_all(self, selector):
        return []


def _install_timer_stub(module, monkeypatch):
    timers = []

    class _TimerStub:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.daemon = False
            self.cancelled = False
            self.started = False

        def start(self):
            self.started = True
            timers.append(self)

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(module.threading, "Timer", _TimerStub)
    return timers


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
    sys.modules.pop("lfs_plugins.viewport_export_controls", None)
    hook_calls, remove_calls = _install_stub_modules(monkeypatch)
    module = import_module("lfs_plugins.toolbar")
    return module, hook_calls, remove_calls


def test_toolbar_binds_overlay_model_fields(toolbar_module):
    module, _hook_calls, _remove_calls = toolbar_module
    model = _DataModelStub()

    module.reset_overlay_state()
    module.bind_overlay_model(model)

    assert "camera_flyout_open" not in model.bound_funcs
    assert "render_flyout_open" not in model.bound_funcs
    assert "selection_flyout_open" not in model.bound_funcs
    assert "transform_flyout_open" not in model.bound_funcs
    assert "selection_group_buttons" in model.bound_record_lists
    assert "selection_mode_buttons" in model.bound_record_lists
    assert "selection_volume_gizmo_buttons" in model.bound_record_lists
    assert "transform_group_buttons" in model.bound_record_lists
    assert "transform_tool_buttons" in model.bound_record_lists
    assert "mirror_group_buttons" in model.bound_record_lists
    assert "crop_group_buttons" in model.bound_record_lists
    assert "crop_object_buttons" in model.bound_record_lists
    assert "crop_transform_buttons" in model.bound_record_lists
    assert "crop_action_buttons" in model.bound_record_lists
    assert "utility_primary_buttons" in model.bound_record_lists
    assert "camera_mode_buttons" in model.bound_record_lists
    assert "show_transform_space_controls" in model.bound_funcs
    assert "show_transform_pivot_controls" in model.bound_funcs
    assert "show_crop_toolbar" in model.bound_funcs
    assert "show_selection_volume_gizmos" in model.bound_funcs
    assert "toolbar_action" in model.bound_events
    assert "selection_tool_label" not in model.bound_funcs
    assert "selection_mode_label" not in model.bound_funcs
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
    assert "transform_pos_x_str" in model.bound_binds
    assert "transform_num_step" in model.bound_events
    assert "viewport_export_tool_label" in model.bound_funcs
    assert "viewport_export_format_value" in model.bound_binds
    assert "viewport_export_resolution_value" in model.bound_binds
    assert "viewport_export_transparency" in model.bound_binds
    assert "viewport_export_custom_width_str" in model.bound_binds
    assert "viewport_export_custom_height_str" in model.bound_binds
    assert "viewport_export_can_export" in model.bound_funcs
    assert "viewport_export_action" in model.bound_events


def test_toolbar_attach_handle_marks_model_dirty(toolbar_module):
    module, _hook_calls, _remove_calls = toolbar_module
    handle = _DataModelHandleStub()

    module.reset_overlay_state()
    module.attach_overlay_model_handle(handle)

    assert handle.dirty_all_calls == 1


def test_button_record_resolves_toolbar_tooltip(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    lf_stub = sys.modules["lichtfeld"]

    monkeypatch.setattr(lf_stub.ui, "tr", lambda key: {"toolbar.home": "Home"}.get(key, key), raising=False)

    localized = module._button_record(
        "util-home",
        "home",
        "",
        "../icon/home.png",
        tooltip_key="toolbar.home",
        tooltip_text="Fallback Home",
    )
    fallback = module._button_record(
        "util-custom",
        "noop",
        "",
        "../icon/custom.png",
        tooltip_key="toolbar.missing",
        tooltip_text="Custom Tool",
    )

    assert localized["tooltip_text"] == "Home"
    assert fallback["tooltip_text"] == "Custom Tool"


def test_selection_tool_uses_centered_modes(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    lf_stub = sys.modules["lichtfeld"]
    state = SimpleNamespace(active_tool="", active_submode="")
    lf_stub.keymap = SimpleNamespace(
        Action=SimpleNamespace(TOOL_SELECT="tool-select", SELECT_MODE_CENTERS="select-mode-centers"),
        ToolMode=SimpleNamespace(GLOBAL="global"),
        is_bound=lambda action, mode: True,
        get_trigger_description=lambda action, mode: {
            ("tool-select", "global"): "Alt+8",
            ("select-mode-centers", "global"): "Ctrl+9",
        }.get((action, mode), ""),
    )
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

    assert snapshot["show_transform_toolbar"] is False
    assert snapshot["show_mirror_toolbar"] is False
    assert snapshot["show_transform_space_controls"] is False
    assert snapshot["show_transform_pivot_controls"] is False
    assert snapshot["selection_group_buttons"][0]["button_id"] == "group-selection"
    assert snapshot["selection_group_buttons"][0]["action"] == "tool"
    assert snapshot["selection_group_buttons"][0]["value"] == "builtin.select"
    assert snapshot["selection_group_buttons"][0]["icon_src"] == "../icon/selection.png"
    assert snapshot["selection_group_buttons"][0]["tooltip_text"] == "Select"
    assert snapshot["selection_group_buttons"][0]["shortcut_text"] == "Alt+8"
    assert snapshot["selection_mode_buttons"][0]["shortcut_text"] == "Ctrl+9"
    assert snapshot["selection_mode_buttons"][1]["shortcut_text"] == ""
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
    assert snapshot["selection_group_buttons"][0]["icon_src"] == "../icon/selection.png"
    assert next(button for button in snapshot["selection_mode_buttons"] if button["value"] == "lasso")["selected"] is True

    controller.dispatch("tool", "builtin.select")
    snapshot = controller.snapshot()

    assert state.active_tool == ""
    assert snapshot["selection_group_buttons"][0]["selected"] is False


def test_transform_and_mirror_tools_use_centered_subtool_rows(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    lf_stub = sys.modules["lichtfeld"]
    state = SimpleNamespace(active_tool="", transform_space=1, multi_transform_mode=0, pivot_mode=0, mirror_calls=[])

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
    align_tool = SimpleNamespace(
        id="builtin.align",
        icon="align",
        label="Align",
        shortcut="6",
        group="utility",
        submodes=(),
        pivot_modes=(),
        selected=None,
        can_activate=lambda _context: True,
    )
    tools = [translate_tool, rotate_tool, scale_tool, mirror_tool, align_tool]
    by_id = {tool.id: tool for tool in tools}

    monkeypatch.setattr(lf_stub.ui, "get_active_tool", lambda: state.active_tool, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_active_submode", lambda: "", raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_transform_space", lambda: state.transform_space, raising=False)
    monkeypatch.setattr(lf_stub.ui, "set_transform_space", lambda value: setattr(state, "transform_space", value), raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_multi_transform_mode", lambda: state.multi_transform_mode, raising=False)
    monkeypatch.setattr(lf_stub.ui, "set_multi_transform_mode", lambda value: setattr(state, "multi_transform_mode", value), raising=False)
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

    assert snapshot["show_transform_toolbar"] is False
    assert snapshot["show_mirror_toolbar"] is False
    assert snapshot["show_transform_space_controls"] is False
    assert snapshot["show_transform_pivot_controls"] is False
    assert snapshot["transform_group_buttons"][0]["button_id"] == "group-transform"
    assert snapshot["transform_group_buttons"][0]["action"] == "tool"
    assert snapshot["transform_group_buttons"][0]["value"] == "builtin.translate"
    assert snapshot["transform_group_buttons"][0]["icon_src"] == "../icon/translation.png"
    assert [button["value"] for button in snapshot["transform_tool_buttons"]] == [
        "builtin.translate",
        "builtin.rotate",
        "builtin.scale",
    ]
    assert snapshot["mirror_group_buttons"][0]["value"] == "builtin.mirror"
    assert [button["value"] for button in snapshot["gizmo_buttons"]] == ["builtin.align"]

    controller.dispatch("tool", "builtin.translate")
    snapshot = controller.snapshot()

    assert state.active_tool == "builtin.translate"
    assert snapshot["transform_group_buttons"][0]["selected"] is True
    assert snapshot["transform_group_buttons"][0]["icon_src"] == "../icon/translation.png"
    assert next(button for button in snapshot["transform_tool_buttons"] if button["value"] == "builtin.translate")["selected"] is True
    assert snapshot["show_transform_toolbar"] is True
    assert snapshot["show_transform_space_controls"] is True
    assert snapshot["show_transform_pivot_controls"] is True
    assert next(button for button in snapshot["submode_buttons"] if button["value"] == "world")["selected"] is True
    assert next(button for button in snapshot["pivot_buttons"] if button["value"] == "origin")["selected"] is True

    monkeypatch.setattr(lf_stub, "get_selected_node_names", lambda: ["target"], raising=False)
    snapshot = controller.snapshot()
    assert snapshot["show_transform_toolbar"] is False
    assert next(button for button in snapshot["transform_tool_buttons"] if button["value"] == "builtin.translate")["selected"] is True
    assert [button["value"] for button in snapshot["submode_buttons"]] == ["local", "world"]

    controller.dispatch("submode", "local")
    controller.dispatch("pivot", "bounds")
    snapshot = controller.snapshot()

    assert state.transform_space == 0
    assert state.pivot_mode == 1
    assert next(button for button in snapshot["submode_buttons"] if button["value"] == "local")["selected"] is True
    assert next(button for button in snapshot["pivot_buttons"] if button["value"] == "bounds")["selected"] is True

    monkeypatch.setattr(lf_stub, "get_selected_node_names", lambda: ["left", "right"], raising=False)
    snapshot = controller.snapshot()

    assert snapshot["show_transform_space_controls"] is True
    assert [button["value"] for button in snapshot["submode_buttons"]] == ["selection", "individual"]
    assert next(button for button in snapshot["submode_buttons"] if button["value"] == "selection")["selected"] is True
    assert next(button for button in snapshot["submode_buttons"] if button["value"] == "selection")["tooltip_key"] == "toolbar.selection_transform"
    assert next(button for button in snapshot["submode_buttons"] if button["value"] == "individual")["tooltip_key"] == "toolbar.individual_transform"

    controller.dispatch("submode", "individual")
    snapshot = controller.snapshot()

    assert state.multi_transform_mode == 1
    assert next(button for button in snapshot["submode_buttons"] if button["value"] == "individual")["selected"] is True

    monkeypatch.setattr(lf_stub, "get_selected_node_names", lambda: ["target"], raising=False)
    snapshot = controller.snapshot()
    assert [button["value"] for button in snapshot["submode_buttons"]] == ["local", "world"]

    controller.dispatch("tool", "builtin.translate")
    snapshot = controller.snapshot()

    assert state.active_tool == ""
    assert snapshot["transform_group_buttons"][0]["selected"] is False

    controller.dispatch("tool", "builtin.mirror")
    snapshot = controller.snapshot()

    assert state.active_tool == "builtin.mirror"
    assert snapshot["transform_group_buttons"][0]["selected"] is False
    assert snapshot["mirror_group_buttons"][0]["selected"] is True
    assert snapshot["show_mirror_toolbar"] is True
    assert snapshot["show_transform_space_controls"] is False
    assert snapshot["show_transform_pivot_controls"] is False
    assert [button["value"] for button in snapshot["submode_buttons"]] == ["x", "y", "z"]

    controller.dispatch("submode", "y")

    assert state.mirror_calls == ["y"]

    controller.dispatch("tool", "builtin.translate")
    controller.dispatch("tool", "builtin.translate")
    snapshot = controller.snapshot()

    assert state.active_tool == ""
    assert snapshot["transform_group_buttons"][0]["selected"] is False


def test_crop_tool_uses_centered_object_and_transform_rows(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    lf_stub = sys.modules["lichtfeld"]
    state = SimpleNamespace(
        active_tool="builtin.cropbox",
        gizmo_type="rotate",
        calls=[],
        crop_shape="box",
    )
    crop_tool = SimpleNamespace(
        id="builtin.cropbox",
        icon="cropbox",
        label="Crop",
        shortcut="",
        group="utility",
        submodes=(),
        pivot_modes=(),
        selected=None,
        can_activate=lambda _context: True,
    )

    def set_active_operator(tool_id, gizmo_type=""):
        state.calls.append(("set_active_operator", tool_id, gizmo_type))
        state.active_tool = tool_id
        state.gizmo_type = gizmo_type

    monkeypatch.setattr(lf_stub.ui, "get_active_tool", lambda: state.active_tool, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_gizmo_type", lambda: state.gizmo_type, raising=False)
    monkeypatch.setattr(lf_stub.ui, "set_active_operator", set_active_operator, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_crop_tool_shape", lambda: state.crop_shape, raising=False)
    monkeypatch.setattr(
        lf_stub.ui,
        "set_crop_tool_shape",
        lambda shape: (state.calls.append(("set_crop_tool_shape", shape)), setattr(state, "crop_shape", shape)),
        raising=False,
    )
    monkeypatch.setattr(
        lf_stub.ui,
        "fit_crop_tool",
        lambda use_percentile=False: state.calls.append(("fit_crop_tool", use_percentile)),
        raising=False,
    )
    monkeypatch.setattr(lf_stub.ui, "apply_crop_tool", lambda: state.calls.append(("apply_crop_tool",)), raising=False)
    monkeypatch.setattr(lf_stub.ui, "reset_crop_tool", lambda: state.calls.append(("reset_crop_tool",)), raising=False)
    monkeypatch.setattr(lf_stub.ui, "delete_crop_tool_volume", lambda: state.calls.append(("delete_crop_tool_volume",)), raising=False)
    monkeypatch.setattr(module.ToolRegistry, "get_all", staticmethod(lambda: [crop_tool]), raising=False)
    monkeypatch.setattr(
        module.ToolRegistry,
        "get",
        staticmethod(lambda tool_id: crop_tool if tool_id == "builtin.cropbox" else None),
        raising=False,
    )

    controller = module._GizmoToolbarController()
    snapshot = controller.snapshot()

    assert snapshot["show_crop_toolbar"] is True
    assert snapshot["crop_group_buttons"][0]["selected"] is True
    assert [button["value"] for button in snapshot["crop_object_buttons"]] == ["box", "ellipsoid"]
    assert [button["value"] for button in snapshot["crop_transform_buttons"]] == ["translate", "rotate", "scale"]
    assert [button["action"] for button in snapshot["crop_action_buttons"]] == [
        "crop_fit",
        "crop_trim",
        "crop_reset",
        "crop_apply",
        "crop_delete",
    ]
    assert next(button for button in snapshot["crop_object_buttons"] if button["value"] == "ellipsoid")["icon_src"] == "../icon/sphere.png"
    assert [button["icon_src"] for button in snapshot["crop_action_buttons"]] == [
        "../icon/arrows-maximize.png",
        "../icon/arrows-minimize.png",
        "../icon/reset.png",
        "../icon/check.png",
        "../icon/scene/trash.png",
    ]
    assert [button["tooltip_key"] for button in snapshot["crop_action_buttons"]] == [
        "scene.fit_to_scene",
        "scene.fit_to_scene_trimmed",
        "scene.reset_crop",
        "common.apply",
        "scene.delete",
    ]
    assert snapshot["crop_action_buttons"][-1]["separator_before"] is True
    assert next(button for button in snapshot["crop_object_buttons"] if button["value"] == "box")["selected"] is True
    assert next(button for button in snapshot["crop_transform_buttons"] if button["value"] == "rotate")["selected"] is True

    controller.dispatch("crop_object", "ellipsoid")

    assert ("set_crop_tool_shape", "ellipsoid") in state.calls
    assert ("set_active_operator", "builtin.cropbox", "rotate") in state.calls
    snapshot = controller.snapshot()
    assert next(button for button in snapshot["crop_object_buttons"] if button["value"] == "ellipsoid")["selected"] is True

    controller.dispatch("crop_transform", "scale")

    assert state.gizmo_type == "scale"
    assert state.calls[-1] == ("set_active_operator", "builtin.cropbox", "scale")

    controller.dispatch("crop_fit", "")

    assert state.calls[-1] == ("fit_crop_tool", False)

    controller.dispatch("crop_trim", "")

    assert state.calls[-1] == ("fit_crop_tool", True)

    controller.dispatch("crop_reset", "")

    assert state.calls[-1] == ("reset_crop_tool",)

    controller.dispatch("crop_apply", "")

    assert state.calls[-1] == ("apply_crop_tool",)

    controller.dispatch("crop_delete", "")

    assert state.calls[-1] == ("delete_crop_tool_volume",)


def test_crop_tool_activation_creates_explicitly_but_snapshot_is_passive(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    lf_stub = sys.modules["lichtfeld"]
    state = SimpleNamespace(
        active_tool="",
        gizmo_type="",
        crop_shape="box",
        calls=[],
    )
    crop_tool = SimpleNamespace(
        id="builtin.cropbox",
        icon="cropbox",
        label="Crop",
        shortcut="",
        group="utility",
        submodes=(),
        pivot_modes=(),
        selected=None,
        can_activate=lambda _context: True,
    )

    def set_active_operator(tool_id, gizmo_type=""):
        state.calls.append(("set_active_operator", tool_id, gizmo_type))
        state.active_tool = tool_id
        state.gizmo_type = gizmo_type

    monkeypatch.setattr(lf_stub, "get_selected_node_names", lambda: ["target"], raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_active_tool", lambda: state.active_tool, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_gizmo_type", lambda: state.gizmo_type, raising=False)
    monkeypatch.setattr(lf_stub.ui, "set_active_operator", set_active_operator, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_crop_tool_shape", lambda: state.crop_shape, raising=False)
    monkeypatch.setattr(
        lf_stub.ui,
        "set_crop_tool_shape",
        lambda shape: (state.calls.append(("set_crop_tool_shape", shape)), setattr(state, "crop_shape", shape)),
        raising=False,
    )
    monkeypatch.setattr(lf_stub.ui, "add_cropbox", lambda name: state.calls.append(("add_cropbox", name)), raising=False)
    monkeypatch.setattr(lf_stub.ui, "add_ellipsoid", lambda name: state.calls.append(("add_ellipsoid", name)), raising=False)
    monkeypatch.setattr(module.ToolRegistry, "get_all", staticmethod(lambda: [crop_tool]), raising=False)
    monkeypatch.setattr(
        module.ToolRegistry,
        "get",
        staticmethod(lambda tool_id: crop_tool if tool_id == "builtin.cropbox" else None),
        raising=False,
    )

    controller = module._GizmoToolbarController()

    controller.snapshot()
    assert state.calls == []

    controller.dispatch("tool", "builtin.cropbox")
    assert ("add_cropbox", "target") in state.calls
    assert ("set_active_operator", "builtin.cropbox", "translate") in state.calls
    add_call_count = len([call for call in state.calls if call[0].startswith("add_")])

    controller.snapshot()
    assert len([call for call in state.calls if call[0].startswith("add_")]) == add_call_count

    controller.dispatch("crop_object", "ellipsoid")
    assert ("set_crop_tool_shape", "ellipsoid") in state.calls
    assert ("add_ellipsoid", "target") in state.calls


def test_crop_tool_activation_infers_selected_ellipsoid(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    lf_stub = sys.modules["lichtfeld"]
    state = SimpleNamespace(
        active_tool="",
        gizmo_type="",
        crop_shape="box",
        calls=[],
    )
    crop_tool = SimpleNamespace(
        id="builtin.cropbox",
        icon="cropbox",
        label="Crop",
        shortcut="",
        group="utility",
        submodes=(),
        pivot_modes=(),
        selected=None,
        can_activate=lambda _context: True,
    )
    ellipsoid_node = SimpleNamespace(type=SimpleNamespace(name="ELLIPSOID"), children=[])
    scene = SimpleNamespace(
        get_node=lambda name: ellipsoid_node if name == "Model_ellipsoid" else None,
        get_node_by_id=lambda _node_id: None,
    )

    def set_active_operator(tool_id, gizmo_type=""):
        state.calls.append(("set_active_operator", tool_id, gizmo_type))
        state.active_tool = tool_id
        state.gizmo_type = gizmo_type

    monkeypatch.setattr(lf_stub, "get_scene", lambda: scene, raising=False)
    monkeypatch.setattr(lf_stub, "get_selected_node_names", lambda: ["Model_ellipsoid"], raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_active_tool", lambda: state.active_tool, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_gizmo_type", lambda: state.gizmo_type, raising=False)
    monkeypatch.setattr(lf_stub.ui, "set_active_operator", set_active_operator, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_crop_tool_shape", lambda: state.crop_shape, raising=False)
    monkeypatch.setattr(
        lf_stub.ui,
        "set_crop_tool_shape",
        lambda shape: (state.calls.append(("set_crop_tool_shape", shape)), setattr(state, "crop_shape", shape)),
        raising=False,
    )
    monkeypatch.setattr(lf_stub.ui, "add_cropbox", lambda name: state.calls.append(("add_cropbox", name)), raising=False)
    monkeypatch.setattr(lf_stub.ui, "add_ellipsoid", lambda name: state.calls.append(("add_ellipsoid", name)), raising=False)
    monkeypatch.setattr(module.ToolRegistry, "get_all", staticmethod(lambda: [crop_tool]), raising=False)
    monkeypatch.setattr(
        module.ToolRegistry,
        "get",
        staticmethod(lambda tool_id: crop_tool if tool_id == "builtin.cropbox" else None),
        raising=False,
    )

    controller = module._GizmoToolbarController()
    controller.dispatch("tool", "builtin.cropbox")

    assert ("set_crop_tool_shape", "ellipsoid") in state.calls
    assert ("add_ellipsoid", "Model_ellipsoid") in state.calls
    assert ("add_cropbox", "Model_ellipsoid") not in state.calls
    assert state.crop_shape == "ellipsoid"


def test_crop_tool_activation_infers_existing_target_ellipsoid(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    lf_stub = sys.modules["lichtfeld"]
    state = SimpleNamespace(
        active_tool="",
        gizmo_type="",
        crop_shape="box",
        calls=[],
    )
    crop_tool = SimpleNamespace(
        id="builtin.cropbox",
        icon="cropbox",
        label="Crop",
        shortcut="",
        group="utility",
        submodes=(),
        pivot_modes=(),
        selected=None,
        can_activate=lambda _context: True,
    )
    model_node = SimpleNamespace(type=SimpleNamespace(name="SPLAT"), children=[7])
    ellipsoid_node = SimpleNamespace(type=SimpleNamespace(name="ELLIPSOID"), children=[])
    scene = SimpleNamespace(
        get_node=lambda name: model_node if name == "Model" else None,
        get_node_by_id=lambda node_id: ellipsoid_node if node_id == 7 else None,
    )

    def set_active_operator(tool_id, gizmo_type=""):
        state.calls.append(("set_active_operator", tool_id, gizmo_type))
        state.active_tool = tool_id
        state.gizmo_type = gizmo_type

    monkeypatch.setattr(lf_stub, "get_scene", lambda: scene, raising=False)
    monkeypatch.setattr(lf_stub, "get_selected_node_names", lambda: ["Model"], raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_active_tool", lambda: state.active_tool, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_gizmo_type", lambda: state.gizmo_type, raising=False)
    monkeypatch.setattr(lf_stub.ui, "set_active_operator", set_active_operator, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_crop_tool_shape", lambda: state.crop_shape, raising=False)
    monkeypatch.setattr(
        lf_stub.ui,
        "set_crop_tool_shape",
        lambda shape: (state.calls.append(("set_crop_tool_shape", shape)), setattr(state, "crop_shape", shape)),
        raising=False,
    )
    monkeypatch.setattr(lf_stub.ui, "add_cropbox", lambda name: state.calls.append(("add_cropbox", name)), raising=False)
    monkeypatch.setattr(lf_stub.ui, "add_ellipsoid", lambda name: state.calls.append(("add_ellipsoid", name)), raising=False)
    monkeypatch.setattr(module.ToolRegistry, "get_all", staticmethod(lambda: [crop_tool]), raising=False)
    monkeypatch.setattr(
        module.ToolRegistry,
        "get",
        staticmethod(lambda tool_id: crop_tool if tool_id == "builtin.cropbox" else None),
        raising=False,
    )

    controller = module._GizmoToolbarController()
    controller.dispatch("tool", "builtin.cropbox")

    assert ("set_crop_tool_shape", "ellipsoid") in state.calls
    assert ("add_ellipsoid", "Model") in state.calls
    assert ("add_cropbox", "Model") not in state.calls
    assert state.crop_shape == "ellipsoid"
def test_selection_volume_modes_show_inline_gizmo_controls(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    lf_stub = sys.modules["lichtfeld"]
    state = SimpleNamespace(
        active_tool="builtin.select",
        active_submode="box",
        crop_operation="rotate",
        calls=[],
    )
    select_tool = SimpleNamespace(
        id="builtin.select",
        icon="selection",
        label="Select",
        shortcut="1",
        submodes=(
            SimpleNamespace(id="centers", label="Centers", icon="circle-dot", shortcut=""),
            SimpleNamespace(id="color", label="Color", icon="color-picker", shortcut=""),
            SimpleNamespace(id="box", label="Box", icon="box", shortcut=""),
            SimpleNamespace(id="sphere", label="Sphere", icon="sphere", shortcut=""),
        ),
        pivot_modes=(),
        selected=None,
        can_activate=lambda _context: True,
    )

    monkeypatch.setattr(lf_stub.ui, "get_active_tool", lambda: state.active_tool, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_active_submode", lambda: state.active_submode, raising=False)
    monkeypatch.setattr(lf_stub.ui, "set_selection_mode", lambda mode: setattr(state, "active_submode", mode), raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_crop_tool_operation", lambda: state.crop_operation, raising=False)
    monkeypatch.setattr(
        lf_stub.ui,
        "set_crop_tool_operation",
        lambda operation: (
            state.calls.append(("set_crop_tool_operation", operation)),
            setattr(state, "crop_operation", operation),
        ),
        raising=False,
    )
    monkeypatch.setattr(module.ToolRegistry, "get_all", staticmethod(lambda: [select_tool]), raising=False)
    monkeypatch.setattr(
        module.ToolRegistry,
        "get",
        staticmethod(lambda tool_id: select_tool if tool_id == "builtin.select" else None),
        raising=False,
    )

    controller = module._GizmoToolbarController()
    snapshot = controller.snapshot()

    assert snapshot["show_crop_toolbar"] is False
    assert snapshot["show_selection_volume_gizmos"] is True
    assert snapshot["crop_object_buttons"] == []
    assert snapshot["crop_transform_buttons"] == []
    assert snapshot["crop_action_buttons"] == []
    assert [button["value"] for button in snapshot["selection_volume_gizmo_buttons"]] == [
        "translate",
        "rotate",
        "scale",
    ]
    assert next(button for button in snapshot["selection_volume_gizmo_buttons"] if button["value"] == "rotate")["selected"] is True

    controller.dispatch("crop_transform", "translate")

    assert state.calls[-1] == ("set_crop_tool_operation", "translate")
    snapshot = controller.snapshot()
    assert next(button for button in snapshot["selection_volume_gizmo_buttons"] if button["value"] == "translate")["selected"] is True

    state.active_submode = "color"
    snapshot = controller.snapshot()
    assert snapshot["show_selection_volume_gizmos"] is False
    assert snapshot["selection_volume_gizmo_buttons"] == []


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
        "selection_transform",
        "individual_transform",
        "origin_pivot",
        "bounds_center_pivot",
    )
    utility_toolbar_tooltip_keys = ()
    selection_tooltip_keys = (
        "selection_panel",
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
    assert rml.count('data-for="button : camera_mode_buttons"') == 0
    assert rml.count('data-for="button : utility_primary_buttons"') == 2
    assert rml.count('data-for="button : submode_buttons"') == 3
    assert rml.count('data-for="button : pivot_buttons"') == 1
    assert rml.count('data-for="button : mirror_group_buttons"') == 2
    assert rml.count('data-for="button : crop_group_buttons"') == 2
    assert rml.count('data-for="button : crop_object_buttons"') == 2
    assert rml.count('data-for="button : crop_transform_buttons"') == 2
    assert rml.count('data-for="button : crop_action_buttons"') == 2
    assert rml.count('data-for="button : selection_volume_gizmo_buttons"') == 1
    assert 'class="toolbar-flyout-divider hidden"' not in rml
    assert "toolbar-flyout" not in rml
    assert rml.count('data-for="button : selection_group_buttons"') == 2
    assert rml.count('class="toolbar-separator"') == 6
    assert 'class="toolbar-separator hidden"' in rml
    assert rml.count('class="viewport-gizmo-controls"') == 2
    assert rml.count('class="viewport-gizmo-control-row"') == 2
    assert 'id="primary-viewport-gizmo-controls" class="viewport-gizmo-controls"' in rml
    assert 'id="secondary-viewport-gizmo-controls" class="viewport-gizmo-controls"' in rml
    assert "viewport-nav-toolbar" not in rml
    assert "viewport-nav-separator" not in rml
    primary_left = rml[
        rml.index('id="primary-utility-toolbar"') : rml.index('id="primary-viewport-gizmo-controls"')
    ]
    secondary_left = rml[
        rml.index('id="secondary-utility-toolbar"') : rml.index('id="secondary-viewport-gizmo-controls"')
    ]
    for toolbar_markup in (primary_left, secondary_left):
        assert 'data-for="button : camera_mode_buttons"' not in toolbar_markup
        assert 'data-for="button : utility_primary_buttons"' not in toolbar_markup
    assert rml.count('data-attr-data-shortcut="button.shortcut_text"') == 25
    assert "data-attr-data-tooltip" not in rml
    assert 'data-attr-title="button.tooltip_text"' in rml
    assert rml.count('data-for="button : selection_mode_buttons"') == 1
    assert 'data-class-hidden="!show_selection_volume_gizmos"' in rml
    assert rml.count('data-for="button : transform_group_buttons"') == 2
    assert rml.count('data-for="button : transform_tool_buttons"') == 3
    assert 'id="selection-block"' in rml
    assert 'class="viewport-selection-overlay hidden"' in rml
    assert 'class="viewport-selection-row"' in rml
    assert 'class="viewport-selection-tool-group"' in rml
    assert "{{selection_tool_label}}" not in rml
    assert "{{selection_mode_label}}" not in rml
    assert 'data-if="selection_depth_mode_active"' in rml
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
    assert 'class="viewport-transform-row"' in rml
    assert 'class="viewport-transform-context"' in rml
    assert 'data-class-hidden="!show_transform_space_controls"' in rml
    assert 'data-class-hidden="!show_transform_pivot_controls"' in rml
    assert 'class="viewport-transform-actions"' in rml
    assert 'data-if="transform_show_actions"' in rml
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
        toolbar_labels_without_hardcoded_shortcuts = (
            "selection",
            "translate",
            "rotate",
            "scale",
            "mirror",
            "align_3point",
            "crop_box",
            "ellipsoid",
            "brush_selection",
            "rect_selection",
            "polygon_selection",
            "lasso_selection",
            "ring_selection",
            "color_selection",
            "home",
            "fullscreen",
            "toggle_ui",
        )
        forbidden_shortcut_fragments = (
            "(1)",
            "(2)",
            "(3)",
            "(4)",
            "(5)",
            "(6)",
            "(7)",
            "(H)",
            "(F11)",
            "(F12)",
            "(Ctrl+1)",
            "(Ctrl+2)",
            "(Ctrl+3)",
            "(Ctrl+4)",
            "(Ctrl+5)",
            "(Ctrl+6)",
            "(Strg+1)",
            "(Strg+2)",
            "(Strg+3)",
            "(Strg+4)",
            "(Strg+5)",
            "(Strg+6)",
            "（H）",
            "（F11）",
            "（F12）",
            "（Ctrl+1）",
            "（Ctrl+2）",
            "（Ctrl+3）",
            "（Ctrl+4）",
            "（Ctrl+5）",
            "（Ctrl+6）",
        )
        for key in toolbar_labels_without_hardcoded_shortcuts:
            value = data.get("toolbar", {}).get(key, "")
            if value:
                assert not any(fragment in value for fragment in forbidden_shortcut_fragments), (
                    f"{locale_path.name} toolbar.{key} still contains a hardcoded shortcut: {value}"
                )
        for key in selection_tooltip_keys:
            assert data.get("tooltip", {}).get(key), f"{locale_path.name} missing tooltip.{key}"
        for key in (*transform_tooltip_keys, *transform_dynamic_tooltip_keys):
            assert data.get("tooltip", {}).get(key), f"{locale_path.name} missing tooltip.{key}"
        for key in transform_toolbar_tooltip_keys:
            assert data.get("toolbar", {}).get(key), f"{locale_path.name} missing toolbar.{key}"
        for key in utility_toolbar_tooltip_keys:
            assert data.get("toolbar", {}).get(key), f"{locale_path.name} missing toolbar.{key}"
    assert "Space: {{transform_space_label}}" not in rml
    assert "Pivot: {{transform_pivot_label}}" not in rml
    assert 'id="transform-block"' not in rendering_rml
    assert "toolbar-flyout-anchor" not in rml
    assert '<span class="flyout-corner-marker"></span>' not in rml
    assert "dropdown-arrow.png" not in rml
    assert "flyout_open" not in rml
    assert 'data-for="button : render_mode_buttons"' not in rml
    assert 'data-class-selected="button.selected"' in rml
    assert "toolbar-flyout" not in rcss
    assert "toolbar-group-container" not in rcss
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
    assert "width: 24dp;" in rcss
    assert ".toolbar-vertical .icon-btn {\n    position: relative;\n    display: flex;" in rcss
    assert ".viewport-gizmo-controls {\n    position: absolute;\n    top: 108dp;\n    right: 10dp;" in rcss
    assert "flex-direction: column;\n    align-items: center;\n    width: 95dp;" in rcss
    assert ".viewport-gizmo-control-row {\n    display: flex;\n    flex-direction: row;" in rcss
    assert "justify-content: center;\n    width: 95dp;\n    gap: 3dp;" in rcss
    assert ".viewport-gizmo-controls .icon-btn {\n    position: relative;\n    display: flex;" in rcss
    gizmo_button_start = rcss.index(".viewport-gizmo-controls .icon-btn {")
    gizmo_button_end = rcss.index(".viewport-gizmo-controls .icon-btn:hover")
    gizmo_button_rcss = rcss[gizmo_button_start:gizmo_button_end]
    assert "width: 30dp;\n    height: 30dp;\n    min-width: 30dp;\n    min-height: 30dp;" in gizmo_button_rcss
    assert ".viewport-gizmo-controls .icon-btn img {\n    width: 20dp;\n    height: 20dp;" in rcss
    assert "viewport-nav-toolbar" not in rcss
    assert "viewport-nav-row" not in rcss
    assert "viewport-nav-separator" not in rcss
    assert "width: 30dp;\n    height: 30dp;\n    min-width: 30dp;\n    min-height: 30dp;" in rcss
    assert ".toolbar-hcenter .toolbar-container {\n    align-items: center;\n    padding: 5dp 6dp;" in rcss
    assert ".toolbar-hcenter .icon-btn {\n    display: flex;" in rcss
    assert "width: 30dp;\n    height: 30dp;\n    min-width: 30dp;\n    min-height: 30dp;" in rcss
    assert ".toolbar-hcenter .icon-btn img {\n    width: 20dp;\n    height: 20dp;" in rcss
    assert "margin: 8dp 0 7dp;" in rcss
    assert ".viewport-transform-overlay {\n    position: absolute;\n    top: 5dp;" in rcss
    assert ".viewport-selection-overlay {\n    position: absolute;\n    top: 5dp;" in rcss
    assert "#depth-view-block .viewport-depth-panel {\n    padding: 5dp 6dp;" in rcss
    assert "#depth-view-block .viewport-depth-mode-select {\n    height: 22dp;" in rcss
    assert "#depth-view-block .viewport-depth-axis {\n    height: 22dp;" in rcss
    assert "#depth-view-block .viewport-depth-axis > .scrub-field {\n    width: 112dp;\n    min-width: 112dp;\n    height: 22dp;" in rcss
    assert ".toolbar-flyout-trigger.hidden" not in rcss
    assert ".viewport-transform-overlay" in rcss
    assert ".viewport-selection-overlay" in rcss
    assert ".viewport-selection-panel" in rcss
    assert ".viewport-selection-depth-fields" in rcss
    assert ".viewport-selection-slider" in rcss
    assert ".viewport-transform-panel" in rcss
    assert 'class="viewport-depth-overlay viewport-export-overlay hidden"' in rml
    assert 'id="viewport-export-status"' in rml
    panel_start = rml.index('class="viewport-transform-panel viewport-export-panel"')
    panel_end = rml.index('<span id="viewport-export-status"')
    assert "viewport-export-status" not in rml[panel_start:panel_end]


def test_viewport_toolbar_update_syncs_utility_records(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    model = _DataModelStub()
    lf_stub = sys.modules["lichtfeld"]
    panel_enabled = {
        "lfs.input_settings": True,
        "lfs.plugin_marketplace": True,
    }

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
    monkeypatch.setattr(lf_stub.ui, "get_multi_transform_mode", lambda: 0, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_pivot_mode", lambda: 0, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_split_view_mode", lambda: "single", raising=False)
    monkeypatch.setattr(lf_stub.ui, "is_sequencer_visible", lambda: False, raising=False)
    monkeypatch.setattr(
        lf_stub.ui,
        "tr",
        lambda key: {
            "toolbar.focus_selection": "Focus Selection",
            "menu.tools.plugin_marketplace": "Plugins",
            "window.input_settings": "Input",
            "toolbar.viewport_export": "Export",
        }.get(key, key),
        raising=False,
    )
    monkeypatch.setattr(
        lf_stub.ui,
        "is_panel_enabled",
        lambda panel_id: panel_enabled.get(panel_id, False),
        raising=False,
    )
    monkeypatch.setattr(
        lf_stub.ui,
        "set_panel_enabled",
        lambda panel_id, enabled: panel_enabled.__setitem__(panel_id, enabled),
        raising=False,
    )
    monkeypatch.setattr(module, "histogram_mode_available", lambda _context: False)

    module.reset_overlay_state()
    module.bind_overlay_model(model)
    module.attach_overlay_model_handle(model.handle)
    model.handle.record_updates.clear()

    module.update_overlay(SimpleNamespace())

    camera_buttons = model.handle.record_updates["camera_mode_buttons"]
    primary_buttons = model.handle.record_updates["utility_primary_buttons"]
    extra_buttons = model.handle.record_updates["utility_extra_buttons"]
    assert len(camera_buttons) == 4
    assert [button["value"] for button in camera_buttons] == [
        "orbit",
        "trackball",
        "fpv",
        "drone",
    ]
    assert camera_buttons[3]["icon_src"] == "../icon/drone.png"
    assert [button["action"] for button in primary_buttons] == [
        "home",
        "focus_selection",
    ]
    assert primary_buttons[1]["icon_src"] == "../icon/focus-selection.png"
    assert primary_buttons[1]["tooltip_text"] == "Focus Selection"
    assert [button["button_id"] for button in extra_buttons] == [
        "util-input-settings",
        "util-viewport-export",
        "util-plugin-marketplace",
        "util-sequencer",
    ]
    extra_by_id = {button["button_id"]: button for button in extra_buttons}
    input_settings = extra_by_id["util-input-settings"]
    assert input_settings["action"] == "toggle_panel"
    assert input_settings["value"] == "lfs.input_settings"
    assert input_settings["icon_src"] == "../icon/settings.png"
    assert input_settings["tooltip_text"] == "Input"
    assert input_settings["selected"] is True
    assert extra_by_id["util-viewport-export"]["action"] == "toggle_viewport_export"
    assert extra_by_id["util-viewport-export"]["icon_src"] == "../icon/sequencer/export.png"
    assert extra_by_id["util-viewport-export"]["tooltip_text"] == "Export"
    assert extra_by_id["util-viewport-export"]["selected"] is False
    assert extra_by_id["util-plugin-marketplace"]["action"] == "toggle_panel"
    assert extra_by_id["util-plugin-marketplace"]["value"] == "lfs.plugin_marketplace"
    assert extra_by_id["util-plugin-marketplace"]["icon_src"] == "../icon/puzzle.png"
    assert extra_by_id["util-plugin-marketplace"]["tooltip_text"] == "Plugins"
    assert extra_by_id["util-plugin-marketplace"]["selected"] is True

    model.handle.record_updates.clear()
    model.bound_events["toolbar_action"](None, None, ["toggle_panel", "lfs.plugin_marketplace"])

    assert panel_enabled["lfs.plugin_marketplace"] is False
    extra_buttons = model.handle.record_updates["utility_extra_buttons"]
    extra_by_id = {button["button_id"]: button for button in extra_buttons}
    assert extra_by_id["util-plugin-marketplace"]["selected"] is False

    model.handle.record_updates.clear()
    model.bound_events["toolbar_action"](None, None, ["toggle_viewport_export", ""])

    extra_buttons = model.handle.record_updates["utility_extra_buttons"]
    extra_by_id = {button["button_id"]: button for button in extra_buttons}
    assert extra_by_id["util-viewport-export"]["selected"] is True


def test_viewport_export_toolbar_action_shows_overlay_immediately(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    model = _DataModelStub()
    doc = _DocumentStub()
    lf_stub = sys.modules["lichtfeld"]
    redraw_calls = []

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
    monkeypatch.setattr(lf_stub.ui, "context", lambda: SimpleNamespace(has_scene=True), raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_active_tool", lambda: "", raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_transform_space", lambda: 1, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_pivot_mode", lambda: 0, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_split_view_mode", lambda: "single", raising=False)
    monkeypatch.setattr(lf_stub.ui, "is_sequencer_visible", lambda: False, raising=False)
    monkeypatch.setattr(lf_stub.ui, "is_panel_enabled", lambda _panel_id: False, raising=False)
    monkeypatch.setattr(lf_stub.ui, "request_redraw", lambda: redraw_calls.append(True), raising=False)
    monkeypatch.setattr(module, "histogram_mode_available", lambda _context: False)

    module.reset_overlay_state()
    module.bind_overlay_model(model)
    module.attach_overlay_model_handle(model.handle)
    module._toolbar_controller._current_doc = doc

    assert "hidden" in doc.elements["viewport-export-block"].classes

    model.bound_events["toolbar_action"](None, None, ["toggle_viewport_export", ""])

    assert "hidden" not in doc.elements["viewport-export-block"].classes
    assert "hidden" in doc.elements["depth-view-block"].classes
    assert "hidden" in doc.elements["selection-block"].classes
    assert "hidden" in doc.elements["transform-block"].classes
    assert redraw_calls


def test_viewport_export_close_action_hides_overlay_immediately(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    model = _DataModelStub()
    doc = _DocumentStub()
    lf_stub = sys.modules["lichtfeld"]
    redraw_calls = []

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
    monkeypatch.setattr(lf_stub.ui, "context", lambda: SimpleNamespace(has_scene=True), raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_active_tool", lambda: "", raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_transform_space", lambda: 1, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_pivot_mode", lambda: 0, raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_split_view_mode", lambda: "single", raising=False)
    monkeypatch.setattr(lf_stub.ui, "is_sequencer_visible", lambda: False, raising=False)
    monkeypatch.setattr(lf_stub.ui, "is_panel_enabled", lambda _panel_id: False, raising=False)
    monkeypatch.setattr(lf_stub.ui, "request_redraw", lambda: redraw_calls.append(True), raising=False)
    monkeypatch.setattr(module, "histogram_mode_available", lambda _context: False)

    module.reset_overlay_state()
    module.bind_overlay_model(model)
    module.attach_overlay_model_handle(model.handle)
    module._toolbar_controller._current_doc = doc

    model.bound_events["toolbar_action"](None, None, ["toggle_viewport_export", ""])
    assert "hidden" not in doc.elements["viewport-export-block"].classes

    redraw_calls.clear()
    model.handle.record_updates.clear()
    model.bound_events["viewport_export_action"](None, None, ["close"])

    assert "hidden" in doc.elements["viewport-export-block"].classes
    extra_buttons = model.handle.record_updates["utility_extra_buttons"]
    extra_by_id = {button["button_id"]: button for button in extra_buttons}
    assert extra_by_id["util-viewport-export"]["selected"] is False
    assert redraw_calls


def test_viewport_export_status_animates_then_hides(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    export_module = sys.modules["lfs_plugins.viewport_export_controls"]
    timers = _install_timer_stub(export_module, monkeypatch)
    model = _DataModelStub()
    doc = _DocumentStub()

    module.reset_overlay_state()
    module.bind_overlay_model(model)
    module.attach_overlay_model_handle(model.handle)
    controller = module._toolbar_controller._viewport_export_controls
    controller.mount(doc)

    controller._set_status("Saved 9215 x 6480")

    toast = doc.elements["viewport-export-status"]
    assert "hidden" not in toast.classes
    assert toast.text == "Saved 9215 x 6480"
    assert toast.properties["opacity"] == "1"
    assert toast.animations[-1] == ("opacity", "0", 1.95, "quadratic-out", "1", False)
    assert len(timers) == 1
    assert timers[-1].started
    assert timers[-1].daemon is True

    toast.listeners["animationend"][-1](SimpleNamespace())

    assert "hidden" in toast.classes
    assert "opacity" not in toast.properties
    assert timers[-1].cancelled
    assert "viewport_export_status_text" in model.handle.dirty_calls


def test_viewport_export_status_expires_without_animation_event(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    export_module = sys.modules["lfs_plugins.viewport_export_controls"]
    timers = _install_timer_stub(export_module, monkeypatch)
    now = [100.0]
    monkeypatch.setattr(export_module.time, "monotonic", lambda: now[0])
    model = _DataModelStub()
    doc = _DocumentStub()

    module.reset_overlay_state()
    module.bind_overlay_model(model)
    module.attach_overlay_model_handle(model.handle)
    controller = module._toolbar_controller._viewport_export_controls
    controller.mount(doc)

    controller._set_status("Saved 9215 x 6480")

    toast = doc.elements["viewport-export-status"]
    assert "hidden" not in toast.classes
    assert timers[-1].delay == pytest.approx(2.0)

    now[0] += 2.0
    dirty = controller.update(doc)

    assert "status-expired" in dirty
    assert "hidden" in toast.classes
    assert "opacity" not in toast.properties
    assert timers[-1].cancelled


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
