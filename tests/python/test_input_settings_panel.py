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
    "ALIGN",
    "CROP_BOX",
    "TRANSLATE",
    "ROTATE",
    "SCALE",
)

ACTION_NAMES = (
    "CAMERA_ORBIT",
    "CAMERA_PAN",
    "CAMERA_ZOOM",
    "CAMERA_SET_PIVOT",
    "CAMERA_MOVE_FORWARD",
    "CAMERA_MOVE_BACKWARD",
    "CAMERA_MOVE_LEFT",
    "CAMERA_MOVE_RIGHT",
    "CAMERA_MOVE_UP",
    "CAMERA_MOVE_DOWN",
    "CAMERA_SPEED_UP",
    "CAMERA_SPEED_DOWN",
    "ZOOM_SPEED_UP",
    "ZOOM_SPEED_DOWN",
    "CAMERA_RESET_HOME",
    "CAMERA_NEXT_VIEW",
    "CAMERA_PREV_VIEW",
    "SELECTION_REPLACE",
    "SELECTION_ADD",
    "SELECTION_REMOVE",
    "SELECT_MODE_CENTERS",
    "SELECT_MODE_RECTANGLE",
    "SELECT_MODE_POLYGON",
    "SELECT_MODE_LASSO",
    "SELECT_MODE_RINGS",
    "TOGGLE_DEPTH_MODE",
    "TOGGLE_SELECTION_DEPTH_FILTER",
    "TOGGLE_SELECTION_CROP_FILTER",
    "DEPTH_ADJUST_NEAR",
    "DEPTH_ADJUST_FAR",
    "DEPTH_ADJUST_SIDE",
    "CYCLE_BRUSH_MODE",
    "BRUSH_RESIZE",
    "APPLY_CROP_BOX",
    "UNDO",
    "REDO",
    "COPY_SELECTION",
    "PASTE_SELECTION",
    "INVERT_SELECTION",
    "DESELECT_ALL",
    "TOGGLE_SPLIT_VIEW",
    "TOGGLE_INDEPENDENT_SPLIT_VIEW",
    "TOGGLE_GT_COMPARISON",
    "CYCLE_PLY",
    "CYCLE_SELECTION_VIS",
    "DELETE_NODE",
    "DELETE_SELECTED",
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
    )

    keymap = SimpleNamespace(
        ToolMode=tool_mode,
        Action=action,
        get_available_profiles=lambda: list(state.profiles),
        get_current_profile=lambda: state.current_profile[0],
        get_tool_mode_name=lambda mode: f"Mode {mode.name}",
        get_action_name=lambda value: f"Action {value.name}",
        get_trigger_description=lambda value, mode: f"{mode.name}:{value.name}",
        is_capturing=lambda: state.capturing[0],
        is_waiting_for_double_click=lambda: state.waiting_double[0],
        load_profile=lambda name: state.current_profile.__setitem__(0, name),
        save_profile=lambda _name: None,
        reset_to_default=lambda: None,
        export_profile=lambda _path: None,
        import_profile=lambda _path: None,
        start_capture=lambda _mode, _action: None,
        cancel_capture=lambda: None,
        get_captured_trigger=lambda: None,
    )

    lf_stub = ModuleType("lichtfeld")
    lf_stub.keymap = keymap
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
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


class _DocStub:
    def get_element_by_id(self, _element_id):
        return None


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
    assert panel._handle.records["tool_modes"][:5] == [
        {"index": "0", "label": "Mode GLOBAL"},
        {"index": "1", "label": "Mode SELECTION"},
        {"index": "2", "label": "Mode BRUSH"},
        {"index": "3", "label": "Mode ALIGN"},
        {"index": "4", "label": "Mode CROP_BOX"},
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
    assert orbit_row["desc_text"] == "input_settings.press_key_or_click"
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

    assert str(module.lf.keymap.Action.TOGGLE_SELECTION_DEPTH_FILTER.value) in action_ids
    assert str(module.lf.keymap.Action.DEPTH_ADJUST_FAR.value) in action_ids
    assert str(module.lf.keymap.Action.TOGGLE_DEPTH_MODE.value) not in action_ids
    assert str(module.lf.keymap.Action.DEPTH_ADJUST_NEAR.value) not in action_ids
    assert str(module.lf.keymap.Action.DEPTH_ADJUST_SIDE.value) not in action_ids
