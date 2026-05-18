# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the retained input settings panel data model."""

from enum import IntEnum
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


TOOL_MODE_NAMES = (
    "GLOBAL",
    "SELECTION",
    "BRUSH",
    "TRANSLATE",
    "ROTATE",
    "SCALE",
    "ALIGN",
    "CROP_BOX",
)

ACTION_NAMES = (
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
    "CYCLE_BRUSH_MODE",
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
    "TOOL_BRUSH",
    "TOOL_ALIGN",
    "PIE_MENU",
    "DEPTH_ADJUST_NEAR",
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
    tool_mode = IntEnum("ToolMode", {name: index for index, name in enumerate(TOOL_MODE_NAMES)})
    action = IntEnum("Action", {name: index for index, name in enumerate(ACTION_NAMES)})

    state = SimpleNamespace(
        language=["en"],
        profiles=["Default", "Studio"],
        current_profile=["Default"],
        capturing=[False],
        waiting_double=[False],
        conflict=[None],
        captured=[],
        triggers={},
        cleared=[],
        set_triggers=[],
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

    def get_allowed_trigger_kinds(action_value):
        if action_value in (action.CAMERA_ORBIT, action.CAMERA_PAN):
            return ["mouse_button", "mouse_drag"]
        if action_value in (action.CAMERA_ZOOM, action.CAMERA_ROLL, action.BRUSH_RESIZE):
            return ["mouse_scroll"]
        if action_value in (action.SELECTION_REPLACE, action.SELECTION_ADD, action.SELECTION_REMOVE):
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

    keymap = SimpleNamespace(
        ToolMode=tool_mode,
        Action=action,
        get_available_profiles=lambda: list(state.profiles),
        get_current_profile=lambda: state.current_profile[0],
        get_tool_mode_name=lambda mode: f"Mode {mode.name}",
        get_action_name=lambda value: f"Action {value.name}",
        get_trigger_description=lambda value, mode: f"{mode.name}:{value.name}",
        get_trigger=lambda action_value, mode: state.triggers.get((mode, action_value)),
        set_trigger_binding=set_trigger_binding,
        find_conflict_for_action=lambda _mode, _action: state.conflict[0],
        get_allowed_trigger_kinds=get_allowed_trigger_kinds,
        is_capturing=lambda: state.capturing[0],
        is_waiting_for_double_click=lambda: state.waiting_double[0],
        load_profile=lambda name: state.current_profile.__setitem__(0, name),
        save_profile=lambda _name: None,
        reset_to_default=lambda: None,
        export_profile=lambda _path: None,
        import_profile=lambda _path: None,
        start_capture=lambda _mode, _action: None,
        cancel_capture=lambda: None,
        clear_binding=lambda mode, action_value: state.cleared.append((mode, action_value)),
        get_captured_trigger=get_captured_trigger,
    )

    lf_stub = ModuleType("lichtfeld")
    lf_stub.keymap = keymap
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=tr,
        get_current_language=lambda: state.language[0],
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return state


@pytest.fixture
def input_settings_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.input_settings_panel", None)
    sys.modules.pop("lfs_plugins", None)
    state = _install_lf_stub(monkeypatch)
    module = import_module("lfs_plugins.input_settings_panel")
    return module, state


class _HandleStub:
    def __init__(self):
        self.records = {}
        self.dirty_fields = []
        self.dirty_all_calls = 0

    def update_record_list(self, name, rows):
        self.records[name] = rows

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_all_calls += 1


class _ElementStub:
    def __init__(self):
        self.classes = {}
        self.text = ""

    def set_class(self, name, value):
        self.classes[name] = value

    def set_text(self, value):
        self.text = value


class _DocStub:
    def __init__(self, with_conflict_overlay=False):
        self.elements = {}
        if with_conflict_overlay:
            self.elements["binding-conflict-overlay"] = _ElementStub()
            self.elements["binding-conflict-message"] = _ElementStub()

    def get_element_by_id(self, element_id):
        return self.elements.get(element_id)


def test_input_settings_builds_profile_and_mode_records(input_settings_module):
    module, _state = input_settings_module
    panel = module.InputSettingsPanel()
    panel._handle = _HandleStub()

    panel._rebuild_profile_records()
    panel._rebuild_mode_records()

    assert panel._handle.records["profiles"] == [
        {"index": "0", "label": "Default"},
        {"index": "1", "label": "Studio"},
    ]
    assert panel._handle.records["tool_modes"] == [
        {"index": "0", "label": "Mode GLOBAL"},
        {"index": "1", "label": "Mode SELECTION"},
        {"index": "2", "label": "Mode BRUSH"},
        {"index": "3", "label": "Mode TRANSLATE"},
        {"index": "4", "label": "Mode ROTATE"},
        {"index": "5", "label": "Mode SCALE"},
        {"index": "6", "label": "Mode ALIGN"},
        {"index": "7", "label": "Mode CROP_BOX"},
    ]


def test_input_settings_builds_binding_rows_with_capture_state(input_settings_module):
    module, state = input_settings_module
    panel = module.InputSettingsPanel()
    panel._handle = _HandleStub()
    state.capturing[0] = True
    panel._rebinding_action = module.lf.keymap.Action.CAMERA_ORBIT
    panel._rebinding_mode = module.lf.keymap.ToolMode.GLOBAL

    panel._rebuild_binding_rows(module.lf.keymap.ToolMode.GLOBAL)

    rows = panel._handle.records["binding_rows"]
    assert rows[0] == {
        "is_section": True,
        "section_title": "input_settings.section.navigation",
    }

    orbit_row = next(
        row for row in rows
        if not row["is_section"]
        and row["action_id"] == str(module.lf.keymap.Action.CAMERA_ORBIT.value)
    )
    assert orbit_row["desc_text"] == "input_settings.click_or_drag_mouse"
    assert orbit_row["desc_class"] == "is-binding-desc is-capturing"
    assert orbit_row["button_action"] == "cancel"
    assert orbit_row["button_label"] == "input_settings.cancel"
    assert orbit_row["button_class"] == "btn--error"

    pan_row = next(
        row for row in rows
        if not row["is_section"]
        and row["action_id"] == str(module.lf.keymap.Action.CAMERA_PAN.value)
    )
    assert pan_row["desc_text"] == "GLOBAL:CAMERA_PAN"
    assert pan_row["button_action"] == "rebind"
    assert pan_row["button_label"] == "input_settings.rebind"
    assert pan_row["button_class"] == "btn--primary"


def test_input_settings_marks_conflicting_binding_rows(input_settings_module):
    module, state = input_settings_module
    panel = module.InputSettingsPanel()
    panel._handle = _HandleStub()
    state.conflict[0] = {
        "other_action": module.lf.keymap.Action.CAMERA_ZOOM,
        "other_mode": module.lf.keymap.ToolMode.GLOBAL,
    }

    panel._rebuild_binding_rows(module.lf.keymap.ToolMode.GLOBAL)

    rows = panel._handle.records["binding_rows"]
    orbit_row = next(
        row for row in rows
        if not row["is_section"]
        and row["action_id"] == str(module.lf.keymap.Action.CAMERA_ORBIT.value)
    )
    assert orbit_row["desc_text"] == "GLOBAL:CAMERA_ORBIT :: also Action CAMERA_ZOOM"
    assert orbit_row["desc_class"] == "is-binding-desc is-conflict"


def test_input_settings_capture_conflict_prompts_to_replace(input_settings_module):
    module, state = input_settings_module
    panel = module.InputSettingsPanel()
    panel._handle = _HandleStub()
    doc = _DocStub(with_conflict_overlay=True)

    old_trigger = {"type": "drag", "button": 2, "modifiers": 0}
    state.triggers[(module.lf.keymap.ToolMode.GLOBAL, module.lf.keymap.Action.CAMERA_ORBIT)] = old_trigger
    state.capturing[0] = False
    state.captured.append({"type": "drag", "button": 1, "modifiers": 0})
    state.conflict[0] = {
        "other_action": module.lf.keymap.Action.CAMERA_PAN,
        "other_mode": module.lf.keymap.ToolMode.GLOBAL,
    }
    panel._rebinding_action = module.lf.keymap.Action.CAMERA_ORBIT
    panel._rebinding_mode = module.lf.keymap.ToolMode.GLOBAL
    panel._previous_trigger = old_trigger

    panel.on_update(doc)

    assert panel._pending_conflict["action"] == module.lf.keymap.Action.CAMERA_ORBIT
    assert panel._pending_conflict["other_action"] == module.lf.keymap.Action.CAMERA_PAN
    assert doc.elements["binding-conflict-overlay"].classes["hidden"] is False
    assert doc.elements["binding-conflict-message"].text == (
        "GLOBAL:CAMERA_ORBIT conflicts with Action CAMERA_PAN in Mode GLOBAL"
    )

    panel._on_replace_conflict(None, None, None)

    assert state.cleared == [
        (module.lf.keymap.ToolMode.GLOBAL, module.lf.keymap.Action.CAMERA_PAN)
    ]
    assert panel._pending_conflict is None
    assert doc.elements["binding-conflict-overlay"].classes["hidden"] is True


def test_input_settings_capture_conflict_cancel_restores_previous_trigger(input_settings_module):
    module, state = input_settings_module
    panel = module.InputSettingsPanel()
    panel._handle = _HandleStub()
    doc = _DocStub(with_conflict_overlay=True)

    old_trigger = {"type": "drag", "button": 2, "modifiers": 0}
    panel._doc = doc
    panel._pending_conflict = {
        "mode": module.lf.keymap.ToolMode.GLOBAL,
        "action": module.lf.keymap.Action.CAMERA_ORBIT,
        "other_mode": module.lf.keymap.ToolMode.GLOBAL,
        "other_action": module.lf.keymap.Action.CAMERA_PAN,
        "previous_trigger": old_trigger,
    }

    panel._on_cancel_conflict(None, None, None)

    assert state.cleared == [
        (module.lf.keymap.ToolMode.GLOBAL, module.lf.keymap.Action.CAMERA_ORBIT)
    ]
    assert state.set_triggers == [
        (module.lf.keymap.ToolMode.GLOBAL, module.lf.keymap.Action.CAMERA_ORBIT, old_trigger)
    ]
    assert panel._pending_conflict is None
    assert doc.elements["binding-conflict-overlay"].classes["hidden"] is True


def test_input_settings_language_change_rebuilds_and_dirties_all(input_settings_module):
    module, state = input_settings_module
    panel = module.InputSettingsPanel()
    panel._handle = _HandleStub()
    panel._last_profiles = list(state.profiles)
    panel._last_lang = "en"
    panel._last_current_profile = "Default"
    panel._last_capturing = False
    panel._last_state_key = (
        panel._selected_mode_idx,
        None,
        False,
        "Default",
        "en",
    )

    state.language[0] = "de"

    panel.on_update(_DocStub())

    assert panel._last_lang == "de"
    assert panel._handle.dirty_all_calls == 1
    assert panel._handle.records["profiles"][0]["label"] == "Default"
    assert panel._handle.records["tool_modes"][0]["label"] == "Mode GLOBAL"
    assert panel._handle.records["binding_rows"][0]["is_section"] is True


def test_input_settings_selection_mode_shows_only_streamlined_depth_actions(input_settings_module):
    module, _state = input_settings_module
    panel = module.InputSettingsPanel()
    panel._handle = _HandleStub()

    panel._rebuild_binding_rows(module.lf.keymap.ToolMode.SELECTION)

    action_ids = {
        row["action_id"]
        for row in panel._handle.records["binding_rows"]
        if not row["is_section"]
    }
    section_titles = {
        row["section_title"]
        for row in panel._handle.records["binding_rows"]
        if row["is_section"]
    }

    assert str(module.lf.keymap.Action.TOGGLE_SELECTION_DEPTH_FILTER.value) in action_ids
    assert str(module.lf.keymap.Action.DEPTH_ADJUST_FAR.value) in action_ids
    assert str(module.lf.keymap.Action.CONFIRM_POLYGON.value) in action_ids
    assert str(module.lf.keymap.Action.CANCEL_POLYGON.value) in action_ids
    assert str(module.lf.keymap.Action.UNDO_POLYGON_VERTEX.value) in action_ids
    assert str(module.lf.keymap.Action.DELETE_SELECTED.value) in action_ids
    assert "input_settings.section.depth" in section_titles
    assert str(module.lf.keymap.Action.TOGGLE_DEPTH_MODE.value) not in action_ids
    assert str(module.lf.keymap.Action.DEPTH_ADJUST_NEAR.value) not in action_ids
    assert str(module.lf.keymap.Action.DEPTH_ADJUST_SIDE.value) not in action_ids


def test_input_settings_transform_mode_exposes_node_picking(input_settings_module):
    module, _state = input_settings_module
    panel = module.InputSettingsPanel()
    panel._handle = _HandleStub()

    panel._rebuild_binding_rows(module.lf.keymap.ToolMode.TRANSLATE)

    action_ids = {
        row["action_id"]
        for row in panel._handle.records["binding_rows"]
        if not row["is_section"]
    }
    section_titles = {
        row["section_title"]
        for row in panel._handle.records["binding_rows"]
        if row["is_section"]
    }

    assert "input_settings.section.node_picking" in section_titles
    assert str(module.lf.keymap.Action.NODE_PICK.value) in action_ids
    assert str(module.lf.keymap.Action.NODE_RECT_SELECT.value) in action_ids
    assert str(module.lf.keymap.Action.DELETE_NODE.value) in action_ids
    assert str(module.lf.keymap.Action.CAMERA_ORBIT.value) not in action_ids
    assert str(module.lf.keymap.Action.CAMERA_PAN.value) not in action_ids
    assert str(module.lf.keymap.Action.UNDO.value) not in action_ids
    assert "input_settings.section.editing" not in section_titles


def test_input_settings_global_mode_exposes_system_sections(input_settings_module):
    module, _state = input_settings_module
    panel = module.InputSettingsPanel()
    panel._handle = _HandleStub()

    panel._rebuild_binding_rows(module.lf.keymap.ToolMode.GLOBAL)

    action_ids = {
        row["action_id"]
        for row in panel._handle.records["binding_rows"]
        if not row["is_section"]
    }

    assert str(module.lf.keymap.Action.TOOL_TRANSLATE.value) in action_ids
    assert str(module.lf.keymap.Action.TOGGLE_UI.value) in action_ids
    assert str(module.lf.keymap.Action.SEQUENCER_PLAY_PAUSE.value) in action_ids
