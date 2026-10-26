# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the Preferences key-binding editor."""

from enum import IntEnum
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import json
import re
import sys

import pytest


TOOL_MODE_NAMES = (
    "GLOBAL",
    "SELECTION",
    "TRANSLATE",
    "ROTATE",
    "SCALE",
    "ALIGN",
    "CROP_BOX",
    "WALK",
)

ACTION_NAMES = (
    "EXIT_WALK",
    "NONE",
    "CAMERA_ORBIT",
    "CAMERA_PAN",
    "CAMERA_ZOOM",
    "CAMERA_ROLL",
    "CAMERA_MOVE_FORWARD",
    "CAMERA_MOVE_BACKWARD",
    "CAMERA_MOVE_LEFT",
    "CAMERA_MOVE_RIGHT",
    "CAMERA_MOVE_UP",
    "CAMERA_MOVE_DOWN",
    "CAMERA_RESET_HOME",
    "CAMERA_SET_HOME",
    "CAMERA_FOCUS_SELECTION",
    "CAMERA_SET_PIVOT",
    "CAMERA_NEXT_VIEW",
    "CAMERA_PREV_VIEW",
    "CAMERA_SPEED_UP",
    "CAMERA_SPEED_DOWN",
    "ZOOM_SPEED_UP",
    "ZOOM_SPEED_DOWN",
    "TOGGLE_SPLIT_VIEW",
    "TOGGLE_INDEPENDENT_SPLIT_VIEW",
    "TOGGLE_GT_COMPARISON",
    "TOGGLE_DEPTH_MODE",
    "CYCLE_PLY",
    "DELETE_SELECTED",
    "DELETE_NODE",
    "UNDO",
    "REDO",
    "INVERT_SELECTION",
    "DESELECT_ALL",
    "SELECT_ALL",
    "COPY_SELECTION",
    "PASTE_SELECTION",
    "DEPTH_ADJUST_FAR",
    "DEPTH_ADJUST_SIDE",
    "TOGGLE_SELECTION_DEPTH_FILTER",
    "TOGGLE_SELECTION_CROP_FILTER",
    "BRUSH_RESIZE",
    "CONFIRM_POLYGON",
    "CANCEL_POLYGON",
    "UNDO_POLYGON_VERTEX",
    "CYCLE_SELECTION_VIS",
    "SELECTION_REPLACE",
    "SELECTION_ADD",
    "SELECTION_REMOVE",
    "SELECT_MODE_CENTERS",
    "SELECT_MODE_RECTANGLE",
    "SELECT_MODE_POLYGON",
    "SELECT_MODE_LASSO",
    "SELECT_MODE_RINGS",
    "SELECT_MODE_COLOR",
    "APPLY_CROP_BOX",
    "NODE_PICK",
    "NODE_RECT_SELECT",
    "TOGGLE_UI",
    "TOGGLE_FULLSCREEN",
    "SEQUENCER_ADD_KEYFRAME",
    "SEQUENCER_UPDATE_KEYFRAME",
    "SEQUENCER_PLAY_PAUSE",
    "TOOL_SELECT",
    "TOOL_TRANSLATE",
    "TOOL_ROTATE",
    "TOOL_SCALE",
    "TOOL_MIRROR",
    "TOOL_ALIGN",
    "PIE_MENU",
    "DEPTH_ADJUST_NEAR",
    "HISTOGRAM_ZOOM_MARKED",
    "TOGGLE_CAMERA_FRUSTUMS",
    "SELECTION_INTERSECT",
    "SELECT_MODE_BOX",
    "SELECT_MODE_SPHERE",
    "CUT_SELECTION",
    "TOGGLE_PERFORMANCE_HUD",
    "OPEN_PREFERENCES",
    "TOGGLE_MCP_SERVER",
    "TOGGLE_MCP_BINDING",
    "TOGGLE_GRID",
    "SELECT_ALL_SCENE_NODES",
    "TOGGLE_SCENE_SELECTION_VISIBILITY",
    "TOGGLE_SCENE_SELECTION_TRAINING",
    "GROUP_SELECTED_SCENE_NODES",
    "UNGROUP_SELECTED_SCENE_NODE",
    "ASSET_GALLERY_PRIMARY",
    "ASSET_GALLERY_COPY_LINK",
    "ASSET_REFRESH",
    "TOGGLE_PYTHON_CONSOLE",
    "CAMERA_RESTORE_HOME",
    "NEXT_SIDEBAR_CATEGORY",
    "PREVIOUS_SIDEBAR_CATEGORY",
    "REMOVE_LAST_POINT",
    "APPLY_ALIGNMENT",
    "INSERT_CROP_POLYGON_VERTEX",
    "REMOVE_CROP_POLYGON_VERTEX",
)


def _install_lf_stub(monkeypatch):
    panel_space = SimpleNamespace(
        SIDE_PANEL="SIDE_PANEL",
        FLOATING="FLOATING",
        VIEWPORT_OVERLAY="VIEWPORT_OVERLAY",
        MAIN_PANEL_TAB="MAIN_PANEL_TAB",
        SCENE_HEADER="SCENE_HEADER",
        BOTTOM_DOCK="BOTTOM_DOCK",
        STATUS_BAR="STATUS_BAR",
    )
    panel_height_mode = SimpleNamespace(FILL="fill", CONTENT="content")
    panel_option = SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED", HIDE_HEADER="HIDE_HEADER")
    tool_mode = IntEnum("ToolMode", {name: index for index, name in enumerate(TOOL_MODE_NAMES)})
    action = IntEnum("Action", {name: index for index, name in enumerate(ACTION_NAMES)})

    state = SimpleNamespace(
        language=["en"],
        profiles=["Default", "Studio"],
        current_profile=["Default"],
        bindings_revision=[0],
        capturing=[False],
        waiting_double=[False],
        conflict=[None],
        captured=[],
        altgr_rejected=[False],
        triggers={},
        cleared=[],
        set_triggers=[],
        started=[],
        cancelled=0,
        section_request="",
        splat_drag_preferences={"format": "ply", "sh_degree": 3, "up_axis": -2},
        mcp_preferences={
            "enabled": True,
            "expose_network": False,
            "port": 45677,
            "request_logging": False,
            "safe_mode": False,
        },
        mcp_status={
            "enabled": True,
            "running": True,
            "phase": "running",
            "expose_network": False,
            "port": 45677,
            "request_count": 0,
            "success_count": 0,
            "error_count": 0,
            "request_logging": False,
            "endpoints": [],
            "log_file": "",
            "error": "",
            "error_kind": "none",
            "error_address": "",
            "error_port": 0,
            "safe_mode": False,
        },
        render_settings=SimpleNamespace(
            scene_upscaler="native",
            scene_upscaler_preset="native",
        ),
        scene_reconstruction_presets={"native": "native", "spatial": "quality"},
    )

    def get_captured_trigger():
        if not state.captured:
            return None
        state.capturing[0] = False
        return state.captured.pop(0)

    def set_trigger_binding(mode, action_value, trigger):
        state.set_triggers.append((mode, action_value, trigger))
        state.triggers[(mode, action_value)] = trigger
        return True

    def start_capture(mode, action_value):
        state.started.append((mode, action_value))
        state.capturing[0] = True

    def cancel_capture():
        state.cancelled += 1
        state.capturing[0] = False

    def take_altgr_capture_rejection():
        rejected, state.altgr_rejected[0] = state.altgr_rejected[0], False
        return rejected

    def get_allowed_trigger_kinds(action_value):
        if action_value in (action.CAMERA_ORBIT, action.CAMERA_PAN):
            return ["mouse_button", "mouse_drag"]
        if action_value in (action.CAMERA_ZOOM, action.CAMERA_ROLL, action.BRUSH_RESIZE):
            return ["mouse_scroll"]
        if action_value in (
            action.SELECTION_REPLACE,
            action.SELECTION_ADD,
            action.SELECTION_REMOVE,
            action.SELECTION_INTERSECT,
        ):
            return ["mouse_button", "mouse_drag"]
        if action_value == action.NODE_PICK:
            return ["mouse_button"]
        if action_value == action.NODE_RECT_SELECT:
            return ["mouse_drag"]
        return ["key"]

    translations = {
        "input_settings.conflict_message":
            "{trigger} conflicts with {action} in {mode}",
        "input_settings.conflict_inline":
            "{binding} :: also {action}",
    }

    def tr(key):
        return translations.get(key, key)

    def set_scene_reconstruction(backend, preset):
        state.render_settings.scene_upscaler = str(backend)
        state.render_settings.scene_upscaler_preset = str(preset)
        state.scene_reconstruction_presets[str(backend)] = str(preset)
        return True

    def reset_scene_reconstruction_preferences():
        state.scene_reconstruction_presets = {"native": "native", "spatial": "quality"}

    keymap = SimpleNamespace(
        ToolMode=tool_mode,
        Action=action,
        get_available_profiles=lambda: list(state.profiles),
        get_current_profile=lambda: state.current_profile[0],
        bindings_revision=lambda: state.bindings_revision[0],
        get_tool_mode_name=lambda mode: f"Mode {mode.name}",
        get_action_name=lambda value: f"Action {value.name}",
        get_trigger_description=lambda value, mode: f"{mode.name}:{value.name}",
        get_trigger=lambda action_value, mode: state.triggers.get((mode, action_value)),
        set_trigger_binding=set_trigger_binding,
        find_conflict_for_action=lambda _mode, _action: state.conflict[0],
        get_allowed_trigger_kinds=get_allowed_trigger_kinds,
        is_capturing=lambda: state.capturing[0],
        is_waiting_for_double_click=lambda: state.waiting_double[0],
        take_altgr_capture_rejection=take_altgr_capture_rejection,
        load_profile=lambda name: state.current_profile.__setitem__(0, name),
        save_profile=lambda _name: None,
        reset_to_default=lambda: None,
        export_profile=lambda _path: None,
        import_profile=lambda _path: None,
        start_capture=start_capture,
        cancel_capture=cancel_capture,
        clear_binding=lambda mode, action_value: state.cleared.append((mode, action_value)),
        get_captured_trigger=get_captured_trigger,
    )

    lf_stub = ModuleType("lichtfeld")
    lf_stub.project_auto_save_on_close_enabled = lambda: False
    lf_stub.project_set_auto_save_on_close = lambda _value: None
    lf_stub.keymap = keymap
    lf_stub.get_camera_navigation_mode = lambda: "orbit"
    lf_stub.get_new_project_navigation_up_axis = lambda: "legacy_y"
    lf_stub.get_new_project_align_grid = lambda: True
    lf_stub.get_camera_view_snap_enabled = lambda: False
    lf_stub.ui = SimpleNamespace(
        get_camera_browse_repeat=lambda: True,
        set_camera_browse_repeat=lambda _enabled: None,
        get_camera_browse_repeat_rate=lambda: 5.0,
        set_camera_browse_repeat_rate=lambda _rate, persist=True: None,
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        get_panel=lambda _panel_id: SimpleNamespace(space=panel_space.BOTTOM_DOCK),
        get_bottom_dock_active_tab=lambda: "",
        set_bottom_dock_active_tab=lambda _panel_id: None,
        set_sequencer_visible=lambda _visible: None,
        is_sequencer_visible=lambda: False,
        set_panel_enabled=lambda _panel_id, _enabled: None,
        tr=tr,
        get_current_language=lambda: state.language[0],
        request_redraw=lambda: None,
        get_mcp_preferences=lambda: dict(state.mcp_preferences),
        get_mcp_port_override=lambda: None,
        get_project_location=lambda: "/home/tester/.lichtfeld/projects",
        get_project_location_preference=lambda: "",
        get_default_project_location=lambda: "/home/tester/.lichtfeld/projects",
        set_project_location=lambda path: "",
        clear_project_location=lambda: None,
        open_folder_dialog=lambda title, start: "",
        get_mcp_status=lambda: dict(state.mcp_status),
        take_preferences_section_request=lambda: "",
        themes=lambda: [],
        get_languages=lambda: [("en", "English")],
        get_theme=lambda: "dark",
        get_theme_family=lambda: "lichtfeld",
        get_theme_mode=lambda: "dark",
        get_ui_scale_preference=lambda: 0.0,
        get_native_floating_windows=lambda: getattr(state, "native_floating_windows", False),
        set_native_floating_windows=lambda enabled: setattr(state, "native_floating_windows", enabled),
        get_zoom_speed_preference=lambda: 0.0,
        set_zoom_speed_preference=lambda _speed: None,
        get_navigation_speed_preference=lambda: 0.0,
        set_navigation_speed_preference=lambda _speed: None,
        remember_camera_navigation=lambda: False,
        remember_camera_view_snap=lambda: False,
        scene_graph_selection_markers=lambda: False,
        set_scene_graph_selection_markers=lambda _enabled: None,
        get_splat_drag_preferences=lambda: dict(state.splat_drag_preferences),
        set_splat_drag_preferences=lambda values: state.splat_drag_preferences.update(values),
        get_progress_bar_style=lambda: "classic",
        set_progress_bar_style=lambda _style: None,
        get_viewport_chrome_style=lambda: "translucent",
        set_viewport_chrome_style=lambda _style: None,
        get_viewport_toolbar_position=lambda: "centered",
        set_viewport_toolbar_position=lambda _position: None,
        get_scene_reconstruction_options=lambda: [
            {
                "id": "native",
                "label_key": "preferences.scene_reconstruction_off",
                "presets": [
                    {
                        "id": "native",
                        "label_key": "preferences.scene_reconstruction_off",
                        "input_scale": 1.0,
                    }
                ],
            },
            {
                "id": "spatial",
                "label_key": "preferences.scene_reconstruction_spatial",
                "presets": [
                    {
                        "id": "quality",
                        "label_key": "preferences.scene_reconstruction_quality",
                        "input_scale": 0.75,
                    },
                    {
                        "id": "balanced",
                        "label_key": "preferences.scene_reconstruction_balanced",
                        "input_scale": 0.67,
                    },
                    {
                        "id": "performance",
                        "label_key": "preferences.scene_reconstruction_performance",
                        "input_scale": 0.5,
                    },
                ],
            },
        ],
        get_scene_reconstruction_preset_preference=lambda backend: (
            state.scene_reconstruction_presets[backend]
        ),
        set_scene_reconstruction=set_scene_reconstruction,
        reset_scene_reconstruction_preferences=reset_scene_reconstruction_preferences,
    )
    lf_stub.get_render_settings = lambda: state.render_settings
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return state


@pytest.fixture
def keymap_bindings_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    for name in (
        "lfs_plugins.keymap_bindings",
        "lfs_plugins.preferences_panel",
        "lfs_plugins",
    ):
        sys.modules.pop(name, None)
    state = _install_lf_stub(monkeypatch)
    prefs = import_module("lfs_plugins.preferences_panel")
    return prefs, state


class _HandleStub:
    def __init__(self):
        self.records = {}
        self.dirty_fields = []
        self.dirty_all_calls = 0
        self.request_update_count = 0

    def update_record_list(self, name, rows):
        self.records[name] = rows

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_all_calls += 1

    def request_update(self):
        self.request_update_count += 1


class _ModelStub:
    def __init__(self):
        self.funcs = {}
        self.binds = {}
        self.events = {}
        self.record_lists = []
        self.handle = _HandleStub()

    def bind_func(self, name, getter):
        self.funcs[name] = getter

    def bind(self, name, getter, setter):
        self.binds[name] = (getter, setter)

    def bind_event(self, name, callback):
        self.events[name] = callback

    def bind_record_list(self, name):
        self.record_lists.append(name)

    def get_handle(self):
        return self.handle


class _ElementStub:
    def __init__(self):
        self.classes = {}
        self.text = ""
        self.listeners = []
        self.attributes = {}

    def set_class(self, name, value):
        self.classes[name] = value

    def set_text(self, value):
        self.text = value

    def add_event_listener(self, event_name, callback):
        self.listeners.append((event_name, callback))

    def get_attribute(self, name):
        return self.attributes.get(name)

    def parent(self):
        return None


class _DocStub:
    def __init__(self, with_conflict_overlay=False, with_bindings_table=False):
        self.elements = {}
        self.attributes = {}
        self.listeners = []
        if with_conflict_overlay:
            self.elements["binding-conflict-overlay"] = _ElementStub()
            self.elements["binding-conflict-message"] = _ElementStub()
        if with_bindings_table:
            self.elements["bindings-table"] = _ElementStub()

    def has_attribute(self, name):
        return name in self.attributes

    def set_attribute(self, name, value):
        self.attributes[name] = value

    def add_event_listener(self, name, callback):
        self.listeners.append((name, callback))

    def get_element_by_id(self, element_id):
        return self.elements.get(element_id)

    def query_selector_all(self, _selector):
        return []


class _EventStub:
    def __init__(self, target):
        self._target = target

    def target(self):
        return self._target


def _bind_panel(prefs):
    panel = prefs.PreferencesPanel()
    model = _ModelStub()
    ctx = SimpleNamespace(create_data_model=lambda name: model if name == "preferences" else None)
    panel.on_bind_model(ctx)
    return panel, model


def _cpp_block(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:index]
    raise AssertionError(f"Unclosed C++ block after {marker}")


def test_default_actions_with_ui_sections_are_listed_in_editor(keymap_bindings_module):
    prefs, _state = keymap_bindings_module
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/visualizer/input/input_bindings.cpp").read_text()
    defaults = _cpp_block(source, "Profile InputBindings::createDefaultProfile()")
    description = _cpp_block(source, "const ActionDescriptor& describe(const Action action)")
    default_actions = set(re.findall(r"Action::([A-Z][A-Z0-9_]*)", defaults))
    descriptor_sections = {
        name: bool(re.search(r"\.ui_section\s*=\s*ActionSection::(?!None\b)", body))
        for name, body in re.findall(
            r"static constexpr ActionDescriptor (d_\w+)\{(.*?)\};", description, re.S
        )
    }
    switch = _cpp_block(description, "switch (action)")
    described = set()
    pending = []
    for match in re.finditer(r"case Action::(\w+):|return (d_\w+);", switch):
        if match.group(1):
            pending.append(match.group(1))
        else:
            if descriptor_sections.get(match.group(2), False):
                described.update(pending)
            pending.clear()

    editor_actions = {
        action.name
        for actions in prefs.KeymapBindingsSection.BINDING_SECTIONS.values()
        for action in actions
    }
    # No intentionally hidden action currently has both a default and a UI section.
    intentionally_hidden = set()
    missing = default_actions & described - editor_actions - intentionally_hidden
    assert not missing, f"Default UI actions missing from the key map editor: {sorted(missing)}"


def test_keymap_section_is_wired_through_preferences(keymap_bindings_module):
    prefs, _state = keymap_bindings_module
    assert "key_bindings" in prefs.PreferencesPanel.EXPANDABLE_SECTIONS

    panel, model = _bind_panel(prefs)
    section = panel._keymap

    assert section._handle is model.handle
    assert panel._handle is model.handle
    assert "panel_label" in model.funcs
    assert model.funcs["panel_label"]() == "preferences.title"
    assert "profile_idx" in model.binds
    assert "mode_idx" in model.binds
    assert "is_capturing" in model.funcs
    assert "bindings_hint" in model.funcs
    for event_name in (
        "save_profile",
        "reset_default",
        "export_profile",
        "import_profile",
        "replace_conflict",
        "cancel_conflict",
    ):
        assert event_name in model.events
    for list_name in ("profiles", "tool_modes", "binding_rows"):
        assert list_name in model.record_lists

    doc = _DocStub(with_conflict_overlay=True, with_bindings_table=True)
    panel.on_mount(doc)
    assert doc.elements["bindings-table"].listeners == [("click", section._on_table_click)]
    assert doc.elements["binding-conflict-overlay"].classes["hidden"] is True
    reset = panel._rendering_defaults._default_reset
    reset_listeners = [("dblclick", reset._on_double_click), ("mouseover", reset._on_hover)]
    assert all(doc.listeners.count(listener) == 1 for listener in reset_listeners)
    # The reset controller must coexist with scrubbing and keymap listeners.
    before_remount = list(doc.listeners)
    panel._rendering_defaults.mount(doc)
    assert doc.listeners == before_remount
    assert doc.elements["bindings-table"].listeners == [("click", section._on_table_click)]


def test_keymap_altgr_rejection_keeps_capture(keymap_bindings_module):
    prefs, state = keymap_bindings_module
    panel, model = _bind_panel(prefs)
    state.capturing[0] = state.altgr_rejected[0] = True
    panel.on_update(_DocStub())
    assert state.capturing[0] and model.funcs["bindings_hint"]() == "input_settings.altgr_capture_unsupported"


def test_keymap_builds_profile_and_mode_records(keymap_bindings_module):
    prefs, _state = keymap_bindings_module
    panel, _model = _bind_panel(prefs)
    section = panel._keymap

    section._rebuild_profile_records()
    section._rebuild_mode_records()

    assert section._handle.records["profiles"] == [
        {"index": "0", "label": "Default"},
        {"index": "1", "label": "Studio"},
    ]
    assert section._handle.records["tool_modes"] == [
        {"index": "0", "label": "Mode GLOBAL"},
        {"index": "1", "label": "Mode SELECTION"},
        {"index": "2", "label": "Mode TRANSLATE"},
        {"index": "3", "label": "Mode ROTATE"},
        {"index": "4", "label": "Mode SCALE"},
        {"index": "5", "label": "Mode ALIGN"},
        {"index": "6", "label": "Mode CROP_BOX"},
        {"index": "7", "label": "Mode WALK"},
    ]


def test_keymap_selecting_the_active_profile_does_not_reload_it(keymap_bindings_module):
    # Catches the profile select reloading the active profile from disk when it binds.
    prefs, state = keymap_bindings_module
    panel, _model = _bind_panel(prefs)
    section = panel._keymap
    loads = []
    sys.modules["lichtfeld"].keymap.load_profile = lambda name: (loads.append(name), state.current_profile.__setitem__(0, name))

    section._set_profile_idx("0")
    assert loads == []

    section._set_profile_idx("1")
    assert loads == ["Studio"]


def test_python_save_profile_rejects_path_names_without_touching_outside_file(tmp_path):
    root = Path(__file__).parents[2]
    source = (root / "src/python/lfs/py_keymap.cpp").read_text()
    start = source.index('"save_profile",')
    end = source.index('"export_profile",', start)
    binding = source[start:end]
    validate = binding.index("InputBindings::isValidProfileName(name)")
    error = binding.index("throw nb::value_error", validate)
    save = binding.index("get_keymap_bindings()->saveProfile(name)")
    assert validate < error < save

    module_dir = root / "build/src/python"
    if not any(module_dir.glob("lichtfeld*.so")):
        return
    outside = tmp_path / "config/x.json"
    outside.parent.mkdir(parents=True)
    outside.write_text("sentinel")
    script = """
import sys
from pathlib import Path
import lichtfeld
target = Path(sys.argv[1])
try:
    lichtfeld.keymap.save_profile('../x')
except ValueError:
    pass
else:
    raise AssertionError('unsafe profile name was accepted')
assert target.read_text() == 'sentinel'
"""
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(module_dir)
    result = __import__("subprocess").run(
        [sys.executable, "-c", script, str(outside)],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert outside.read_text() == "sentinel"


def test_keymap_builds_binding_rows_with_capture_state(keymap_bindings_module):
    prefs, state = keymap_bindings_module
    panel, _model = _bind_panel(prefs)
    section = panel._keymap
    state.capturing[0] = True
    section._rebinding_action = prefs.lf.keymap.Action.CAMERA_ORBIT
    section._rebinding_mode = prefs.lf.keymap.ToolMode.GLOBAL

    section._rebuild_binding_rows(prefs.lf.keymap.ToolMode.GLOBAL)

    rows = section._handle.records["binding_rows"]
    assert rows[0] == {
        "is_section": True,
        "section_title": "input_settings.section.navigation",
    }

    orbit_row = next(
        row for row in rows
        if not row["is_section"]
        and row["action_id"] == str(prefs.lf.keymap.Action.CAMERA_ORBIT.value)
    )
    assert orbit_row["desc_text"] == "input_settings.click_or_drag_mouse"
    assert orbit_row["desc_class"] == "preferences-binding-desc preferences-capturing"
    assert orbit_row["button_action"] == "cancel"
    assert orbit_row["button_label"] == "input_settings.cancel"
    assert orbit_row["button_class"] == "btn--error"

    pan_row = next(
        row for row in rows
        if not row["is_section"]
        and row["action_id"] == str(prefs.lf.keymap.Action.CAMERA_PAN.value)
    )
    assert pan_row["desc_text"] == "GLOBAL:CAMERA_PAN"
    assert pan_row["button_action"] == "rebind"
    assert pan_row["button_label"] == "input_settings.rebind"
    assert pan_row["button_class"] == "btn--primary"


def test_keymap_ensure_rows_does_not_rebuild_on_first_update(keymap_bindings_module):
    prefs, _state = keymap_bindings_module
    _panel, model = _bind_panel(prefs)
    section = _panel._keymap
    binding_rows_calls = []
    original_update_record_list = model.handle.update_record_list

    def update_record_list(name, rows):
        if name == "binding_rows":
            binding_rows_calls.append(rows)
        original_update_record_list(name, rows)

    model.handle.update_record_list = update_record_list

    section.ensure_binding_rows()
    section.on_update(_DocStub())

    assert len(binding_rows_calls) == 1


def test_keymap_editor_import_export_use_native_json_dialogs(keymap_bindings_module, tmp_path):
    prefs, state = keymap_bindings_module; section = _bind_panel(prefs)[0]._keymap
    profile_path, dialog_calls = tmp_path / "roundtrip.json", []
    prefs.lf.ui.save_json_file_dialog = lambda name: (dialog_calls.append(("save", name)) or str(profile_path))
    prefs.lf.ui.open_json_file_dialog = lambda: (dialog_calls.append(("open",)) or str(profile_path))
    key = (prefs.lf.keymap.ToolMode.GLOBAL, prefs.lf.keymap.Action.TOOL_SCALE)
    state.triggers[key] = {"key": 74}; prefs.lf.keymap.export_profile = lambda path: Path(path).write_text(json.dumps({"scale": state.triggers[key]})) or True

    def import_profile(path):
        state.triggers[key] = json.loads(Path(path).read_text())["scale"]
        state.bindings_revision[0] += 1
        return True
    prefs.lf.keymap.import_profile = import_profile
    section._on_export_profile(None, None, None); assert profile_path.exists()
    state.triggers.clear(); section._on_import_profile(None, None, None)
    assert state.triggers[key] == {"key": 74} and dialog_calls == [("save", "Default.json"), ("open",)]


def test_keymap_editor_clear_one_binding_without_reset(keymap_bindings_module):
    prefs, state = keymap_bindings_module; section = _bind_panel(prefs)[0]._keymap
    mode, scale = prefs.lf.keymap.ToolMode.GLOBAL, prefs.lf.keymap.Action.TOOL_SCALE
    state.triggers[(mode, scale)] = {"type": "key", "key": 52, "modifiers": 0}
    section._rebuild_binding_rows(mode)
    row = next(r for r in section._handle.records["binding_rows"] if r.get("action_id") == str(scale.value))
    assert row["show_clear"] is True and row["clear_label"] == "input_settings.clear_binding"

    button = _ElementStub(); button.attributes = {"data-btn-action": "clear", "data-action-id": str(scale.value), "data-mode-id": str(mode.value)}
    section._on_table_click(_EventStub(button))
    assert state.cleared == [(mode, scale)]


def test_keymap_rows_follow_api_binding_revision_without_idle_rebuild(keymap_bindings_module):
    prefs, state = keymap_bindings_module
    panel, model = _bind_panel(prefs)
    section = panel._keymap
    mode = prefs.lf.keymap.ToolMode.GLOBAL
    action = prefs.lf.keymap.Action.CAMERA_ORBIT
    prefs.lf.keymap.get_trigger_description = lambda value, selected_mode: str(state.triggers.get((selected_mode, value), "unbound"))
    rebuilds = []
    original = model.handle.update_record_list

    def record_rebuild(name, rows):
        if name == "binding_rows": rebuilds.append(rows)
        original(name, rows)

    model.handle.update_record_list = record_rebuild
    section.ensure_binding_rows(); doc = _DocStub()
    panel.on_update(doc); assert len(rebuilds) == 1

    def changed(desc, count):
        panel.on_update(doc); assert len(rebuilds) == count
        row = next(r for r in rebuilds[-1] if r.get("action_id") == str(action.value))
        assert row["desc_text"] == desc
        panel.on_update(doc); assert len(rebuilds) == count
    state.triggers[(mode, action)] = "J"; state.bindings_revision[0] += 1; changed("J", 2)
    del state.triggers[(mode, action)]; state.bindings_revision[0] += 1; changed("unbound", 3)
    # An import and a profile switch each update once; idle frames do not rebuild.
    state.triggers[(mode, action)] = "K"; state.bindings_revision[0] += 1; changed("K", 4)
    state.current_profile[0] = "Studio"; state.bindings_revision[0] += 1
    panel.on_update(doc); assert len(rebuilds) == 5
    panel.on_update(doc); assert len(rebuilds) == 5


def test_keymap_marks_conflicting_binding_rows(keymap_bindings_module):
    prefs, state = keymap_bindings_module
    panel, _model = _bind_panel(prefs)
    section = panel._keymap
    state.conflict[0] = {
        "other_action": prefs.lf.keymap.Action.CAMERA_ZOOM,
        "other_mode": prefs.lf.keymap.ToolMode.GLOBAL,
    }

    section._rebuild_binding_rows(prefs.lf.keymap.ToolMode.GLOBAL)

    rows = section._handle.records["binding_rows"]
    orbit_row = next(
        row for row in rows
        if not row["is_section"]
        and row["action_id"] == str(prefs.lf.keymap.Action.CAMERA_ORBIT.value)
    )
    assert orbit_row["desc_text"] == "GLOBAL:CAMERA_ORBIT :: also Action CAMERA_ZOOM"
    assert orbit_row["desc_class"] == "preferences-binding-desc preferences-conflict"


def test_keymap_pending_capture_polls_until_the_key_arrives(keymap_bindings_module):
    # Catches a dirty-policy panel that never notices the native capture finishing.
    prefs, state = keymap_bindings_module
    panel, model = _bind_panel(prefs)
    section = panel._keymap
    doc = _DocStub(with_conflict_overlay=True)
    section.ensure_binding_rows()
    section._rebinding_action = prefs.lf.keymap.Action.CAMERA_ORBIT
    section._rebinding_mode = prefs.lf.keymap.ToolMode.GLOBAL
    section._previous_trigger = None
    state.capturing[0] = True

    panel.on_update(doc)
    assert model.handle.request_update_count == 1

    state.capturing[0] = False
    state.captured.append({"type": "key", "key": 85, "modifiers": 0})
    panel.on_update(doc)
    requests_after_capture = model.handle.request_update_count
    panel.on_update(doc)

    assert section._rebinding_action is None
    assert model.handle.request_update_count == requests_after_capture


def test_keymap_capture_conflict_prompts_to_replace(keymap_bindings_module):
    prefs, state = keymap_bindings_module
    panel, _model = _bind_panel(prefs)
    section = panel._keymap
    doc = _DocStub(with_conflict_overlay=True)
    section.ensure_binding_rows()

    old_trigger = {"type": "drag", "button": 2, "modifiers": 0}
    state.triggers[(prefs.lf.keymap.ToolMode.GLOBAL, prefs.lf.keymap.Action.CAMERA_ORBIT)] = old_trigger
    state.capturing[0] = False
    state.captured.append({"type": "drag", "button": 1, "modifiers": 0})
    state.conflict[0] = {
        "other_action": prefs.lf.keymap.Action.CAMERA_PAN,
        "other_mode": prefs.lf.keymap.ToolMode.GLOBAL,
    }
    section._rebinding_action = prefs.lf.keymap.Action.CAMERA_ORBIT
    section._rebinding_mode = prefs.lf.keymap.ToolMode.GLOBAL
    section._previous_trigger = old_trigger

    panel.on_update(doc)

    assert section._pending_conflict["action"] == prefs.lf.keymap.Action.CAMERA_ORBIT
    assert section._pending_conflict["other_action"] == prefs.lf.keymap.Action.CAMERA_PAN
    assert doc.elements["binding-conflict-overlay"].classes["hidden"] is False
    assert doc.elements["binding-conflict-message"].text == (
        "GLOBAL:CAMERA_ORBIT conflicts with Action CAMERA_PAN in Mode GLOBAL"
    )

    section._on_replace_conflict(None, None, None)

    assert state.cleared == [
        (prefs.lf.keymap.ToolMode.GLOBAL, prefs.lf.keymap.Action.CAMERA_PAN)
    ]
    assert section._pending_conflict is None
    assert doc.elements["binding-conflict-overlay"].classes["hidden"] is True


def test_keymap_capture_conflict_cancel_restores_previous_trigger(keymap_bindings_module):
    prefs, state = keymap_bindings_module
    panel, _model = _bind_panel(prefs)
    section = panel._keymap
    doc = _DocStub(with_conflict_overlay=True)

    old_trigger = {"type": "drag", "button": 2, "modifiers": 0}
    section._doc = doc
    section._pending_conflict = {
        "mode": prefs.lf.keymap.ToolMode.GLOBAL,
        "action": prefs.lf.keymap.Action.CAMERA_ORBIT,
        "other_mode": prefs.lf.keymap.ToolMode.GLOBAL,
        "other_action": prefs.lf.keymap.Action.CAMERA_PAN,
        "previous_trigger": old_trigger,
    }

    section._on_cancel_conflict(None, None, None)

    assert state.cleared == [
        (prefs.lf.keymap.ToolMode.GLOBAL, prefs.lf.keymap.Action.CAMERA_ORBIT)
    ]
    assert state.set_triggers == [
        (prefs.lf.keymap.ToolMode.GLOBAL, prefs.lf.keymap.Action.CAMERA_ORBIT, old_trigger)
    ]
    assert section._pending_conflict is None
    assert doc.elements["binding-conflict-overlay"].classes["hidden"] is True


def test_keymap_unmount_restores_binding_with_pending_conflict(keymap_bindings_module):
    prefs, state = keymap_bindings_module
    section = _bind_panel(prefs)[0]._keymap
    section._doc = _DocStub(with_conflict_overlay=True)
    mode, action = prefs.lf.keymap.ToolMode.GLOBAL, prefs.lf.keymap.Action.CAMERA_MOVE_BACKWARD
    previous_trigger = {"type": "key", "key": 83, "modifiers": 0}
    section._pending_conflict = {"mode": mode, "action": action, "other_mode": mode,
                                 "other_action": prefs.lf.keymap.Action.CAMERA_MOVE_FORWARD,
                                 "previous_trigger": previous_trigger}
    section.on_unmount(); assert state.cleared == [(mode, action)]
    assert state.set_triggers == [(mode, action, previous_trigger)]
    assert section._pending_conflict is None


def test_keymap_language_change_rebuilds_and_dirties_all(keymap_bindings_module):
    prefs, state = keymap_bindings_module
    panel, _model = _bind_panel(prefs)
    section = panel._keymap
    section.ensure_binding_rows()
    section._last_profiles = list(state.profiles)
    section._last_lang = "en"
    section._last_current_profile = "Default"
    section._last_capturing = False
    section._last_state_key = (
        section._selected_mode_idx,
        None,
        False,
        "Default",
        "en",
    )

    state.language[0] = "de"

    panel.on_update(_DocStub())

    assert section._last_lang == "de"
    assert section._handle.dirty_all_calls == 1
    assert section._handle.records["profiles"][0]["label"] == "Default"
    assert section._handle.records["tool_modes"][0]["label"] == "Mode GLOBAL"
    assert section._handle.records["binding_rows"][0]["is_section"] is True


def test_keymap_selection_mode_shows_only_streamlined_depth_actions(keymap_bindings_module):
    prefs, _state = keymap_bindings_module
    panel, _model = _bind_panel(prefs)
    section = panel._keymap

    section._rebuild_binding_rows(prefs.lf.keymap.ToolMode.SELECTION)

    action_ids = {
        row["action_id"]
        for row in section._handle.records["binding_rows"]
        if not row["is_section"]
    }
    section_titles = {
        row["section_title"]
        for row in section._handle.records["binding_rows"]
        if row["is_section"]
    }

    assert str(prefs.lf.keymap.Action.TOGGLE_SELECTION_DEPTH_FILTER.value) in action_ids
    assert str(prefs.lf.keymap.Action.DEPTH_ADJUST_FAR.value) in action_ids
    assert str(prefs.lf.keymap.Action.CONFIRM_POLYGON.value) in action_ids
    assert str(prefs.lf.keymap.Action.CANCEL_POLYGON.value) in action_ids
    assert str(prefs.lf.keymap.Action.UNDO_POLYGON_VERTEX.value) in action_ids
    assert str(prefs.lf.keymap.Action.DELETE_SELECTED.value) in action_ids
    assert "input_settings.section.depth" in section_titles
    assert str(prefs.lf.keymap.Action.TOGGLE_DEPTH_MODE.value) not in action_ids
    assert str(prefs.lf.keymap.Action.DEPTH_ADJUST_NEAR.value) not in action_ids
    assert str(prefs.lf.keymap.Action.DEPTH_ADJUST_SIDE.value) not in action_ids


def test_selection_intersect_is_editable_and_accepts_alt_drag_capture(keymap_bindings_module):
    prefs, state = keymap_bindings_module
    panel, _model = _bind_panel(prefs)
    section = panel._keymap
    action = prefs.lf.keymap.Action.SELECTION_INTERSECT
    mode = prefs.lf.keymap.ToolMode.SELECTION

    section._rebuild_binding_rows(mode)
    rows = section._handle.records["binding_rows"]
    assert str(action.value) in {row["action_id"] for row in rows if not row["is_section"]}
    assert "mouse_drag" in prefs.lf.keymap.get_allowed_trigger_kinds(action)
    assert section._capture_prompt(action) == "input_settings.click_or_drag_mouse"

    button = _ElementStub()
    button.attributes = {
        "data-btn-action": "rebind",
        "data-action-id": str(action.value),
        "data-mode-id": str(mode.value),
    }
    section._on_table_click(_EventStub(button))
    assert state.started == [(mode, action)]


def test_keymap_transform_mode_exposes_node_picking(keymap_bindings_module):
    prefs, _state = keymap_bindings_module
    panel, _model = _bind_panel(prefs)
    section = panel._keymap

    section._rebuild_binding_rows(prefs.lf.keymap.ToolMode.TRANSLATE)

    action_ids = {
        row["action_id"]
        for row in section._handle.records["binding_rows"]
        if not row["is_section"]
    }
    section_titles = {
        row["section_title"]
        for row in section._handle.records["binding_rows"]
        if row["is_section"]
    }

    assert "input_settings.section.node_picking" in section_titles
    assert str(prefs.lf.keymap.Action.NODE_PICK.value) in action_ids
    assert str(prefs.lf.keymap.Action.NODE_RECT_SELECT.value) in action_ids
    assert str(prefs.lf.keymap.Action.DELETE_NODE.value) in action_ids
    assert str(prefs.lf.keymap.Action.CAMERA_ORBIT.value) not in action_ids
    assert str(prefs.lf.keymap.Action.CAMERA_PAN.value) not in action_ids
    assert str(prefs.lf.keymap.Action.UNDO.value) not in action_ids
    assert "input_settings.section.editing" not in section_titles


def test_keymap_global_mode_exposes_system_sections(keymap_bindings_module):
    prefs, _state = keymap_bindings_module
    panel, _model = _bind_panel(prefs)
    section = panel._keymap

    section._rebuild_binding_rows(prefs.lf.keymap.ToolMode.GLOBAL)

    action_ids = {
        row["action_id"]
        for row in section._handle.records["binding_rows"]
        if not row["is_section"]
    }

    assert str(prefs.lf.keymap.Action.TOOL_TRANSLATE.value) in action_ids
    assert str(prefs.lf.keymap.Action.TOGGLE_UI.value) in action_ids
    assert str(prefs.lf.keymap.Action.TOGGLE_PYTHON_CONSOLE.value) in action_ids
    assert str(prefs.lf.keymap.Action.OPEN_PREFERENCES.value) in action_ids
    assert str(prefs.lf.keymap.Action.HISTOGRAM_ZOOM_MARKED.value) in action_ids
    assert str(prefs.lf.keymap.Action.TOGGLE_CAMERA_FRUSTUMS.value) in action_ids
    assert str(prefs.lf.keymap.Action.TOGGLE_GRID.value) in action_ids
    assert str(prefs.lf.keymap.Action.SEQUENCER_PLAY_PAUSE.value) in action_ids
    assert str(prefs.lf.keymap.Action.TOGGLE_MCP_SERVER.value) in action_ids
    assert str(prefs.lf.keymap.Action.TOGGLE_MCP_BINDING.value) in action_ids


def test_keymap_section_delegates_rebind_and_cancel_clicks(keymap_bindings_module):
    prefs, state = keymap_bindings_module
    panel, _model = _bind_panel(prefs)
    section = panel._keymap

    action = prefs.lf.keymap.Action.CAMERA_ORBIT
    mode = prefs.lf.keymap.ToolMode.GLOBAL
    old_trigger = {"type": "drag", "button": 2, "modifiers": 0}
    state.triggers[(mode, action)] = old_trigger

    button = _ElementStub()
    button.attributes = {
        "data-btn-action": "rebind",
        "data-action-id": str(action.value),
        "data-mode-id": str(mode.value),
    }
    section._on_table_click(_EventStub(button))

    assert section._rebinding_action == action
    assert section._rebinding_mode == mode
    assert section._previous_trigger == old_trigger
    assert state.started == [(mode, action)]
    assert state.capturing[0] is True
    assert "is_capturing" in section._handle.dirty_fields

    button.attributes["data-btn-action"] = "cancel"
    section._on_table_click(_EventStub(button))

    assert state.cancelled == 1
    assert section._rebinding_action is None
    assert section._rebinding_mode is None
    assert section._previous_trigger is None
    assert state.capturing[0] is False
