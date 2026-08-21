# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for application preferences behavior."""

from enum import IntEnum
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


@pytest.fixture
def preferences_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    state = SimpleNamespace(
        language="it",
        set_language_calls=[],
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
            "endpoints": [
                "http://127.0.0.1:45677/mcp",
                "http://localhost:45677/mcp",
            ],
            "log_file": "",
            "error": "",
            "error_kind": "none",
            "error_address": "",
            "error_port": 0,
            "safe_mode": False,
        },
        set_mcp_calls=[],
        file_associations=[],
        file_association_set_calls=[],
        panel_enabled_calls=[],
        section_request="",
        render_settings=SimpleNamespace(
            scene_upscaler="native",
            scene_upscaler_preset="native",
        ),
        scene_reconstruction_presets={"native": "native", "spatial": "quality"},
        working_directory="",
    )

    def set_working_directory(path):
        state.working_directory = str(path)
        return ""

    def clear_working_directory():
        state.working_directory = ""

    def set_mcp_preferences(enabled, expose_network, port, request_logging):
        config = {
            "enabled": bool(enabled),
            "expose_network": bool(expose_network),
            "port": int(port),
            "request_logging": bool(request_logging),
        }
        state.set_mcp_calls.append(config)
        state.mcp_preferences = dict(config)
        state.mcp_status.update(config)
        return True

    def take_preferences_section_request():
        section = state.section_request
        state.section_request = ""
        return section

    def set_scene_reconstruction(backend, preset):
        state.render_settings.scene_upscaler = str(backend)
        state.render_settings.scene_upscaler_preset = str(preset)
        state.scene_reconstruction_presets[str(backend)] = str(preset)
        return True

    def reset_scene_reconstruction_preferences():
        state.scene_reconstruction_presets = {"native": "native", "spatial": "quality"}

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=SimpleNamespace(FLOATING="FLOATING"),
        PanelHeightMode=SimpleNamespace(FILL="fill"),
        PanelOption=SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED"),
        get_current_language=lambda: state.language,
        set_language=lambda language: state.set_language_calls.append(language),
        get_mcp_preferences=lambda: dict(state.mcp_preferences),
        set_mcp_preferences=set_mcp_preferences,
        get_mcp_status=lambda: dict(state.mcp_status),
        get_working_directory=lambda: state.working_directory or "/home/tester/.lichtfeld",
        get_working_directory_preference=lambda: state.working_directory,
        get_default_working_directory=lambda: "/home/tester/.lichtfeld",
        get_temp_project_directory=lambda: (state.working_directory or "/home/tester/.lichtfeld") + "/tmp",
        set_working_directory=set_working_directory,
        clear_working_directory=clear_working_directory,
        open_folder_dialog=lambda title, start: "",
        set_panel_enabled=lambda panel_id, enabled: state.panel_enabled_calls.append(
            (panel_id, bool(enabled))
        ),
        take_preferences_section_request=take_preferences_section_request,
        tr=lambda key: key,
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
    lf_stub.keymap = SimpleNamespace(
        ToolMode=IntEnum(
            "ToolMode",
            {
                name: index
                for index, name in enumerate(
                    ("GLOBAL", "SELECTION", "TRANSLATE", "ROTATE", "SCALE", "ALIGN", "CROP_BOX")
                )
            },
        ),
        Action=SimpleNamespace(
            CAMERA_ORBIT=SimpleNamespace(name="CAMERA_ORBIT", value=1),
            CAMERA_PAN=SimpleNamespace(name="CAMERA_PAN", value=2),
            CAMERA_ZOOM=SimpleNamespace(name="CAMERA_ZOOM", value=3),
            CAMERA_ROLL=SimpleNamespace(name="CAMERA_ROLL", value=4),
            CAMERA_SET_PIVOT=SimpleNamespace(name="CAMERA_SET_PIVOT", value=5),
            CAMERA_MOVE_FORWARD=SimpleNamespace(name="CAMERA_MOVE_FORWARD", value=6),
            CAMERA_MOVE_BACKWARD=SimpleNamespace(name="CAMERA_MOVE_BACKWARD", value=7),
            CAMERA_MOVE_LEFT=SimpleNamespace(name="CAMERA_MOVE_LEFT", value=8),
            CAMERA_MOVE_RIGHT=SimpleNamespace(name="CAMERA_MOVE_RIGHT", value=9),
            CAMERA_MOVE_UP=SimpleNamespace(name="CAMERA_MOVE_UP", value=10),
            CAMERA_MOVE_DOWN=SimpleNamespace(name="CAMERA_MOVE_DOWN", value=11),
            CAMERA_SPEED_UP=SimpleNamespace(name="CAMERA_SPEED_UP", value=12),
            CAMERA_SPEED_DOWN=SimpleNamespace(name="CAMERA_SPEED_DOWN", value=13),
            ZOOM_SPEED_UP=SimpleNamespace(name="ZOOM_SPEED_UP", value=14),
            ZOOM_SPEED_DOWN=SimpleNamespace(name="ZOOM_SPEED_DOWN", value=15),
            CAMERA_RESET_HOME=SimpleNamespace(name="CAMERA_RESET_HOME", value=16),
            CAMERA_SET_HOME=SimpleNamespace(name="CAMERA_SET_HOME", value=17),
            CAMERA_NEXT_VIEW=SimpleNamespace(name="CAMERA_NEXT_VIEW", value=18),
            CAMERA_PREV_VIEW=SimpleNamespace(name="CAMERA_PREV_VIEW", value=19),
            SELECTION_REPLACE=SimpleNamespace(name="SELECTION_REPLACE", value=20),
            SELECTION_ADD=SimpleNamespace(name="SELECTION_ADD", value=21),
            SELECTION_REMOVE=SimpleNamespace(name="SELECTION_REMOVE", value=22),
            SELECT_MODE_CENTERS=SimpleNamespace(name="SELECT_MODE_CENTERS", value=23),
            SELECT_MODE_RECTANGLE=SimpleNamespace(name="SELECT_MODE_RECTANGLE", value=24),
            SELECT_MODE_POLYGON=SimpleNamespace(name="SELECT_MODE_POLYGON", value=25),
            SELECT_MODE_LASSO=SimpleNamespace(name="SELECT_MODE_LASSO", value=26),
            SELECT_MODE_RINGS=SimpleNamespace(name="SELECT_MODE_RINGS", value=27),
            SELECT_MODE_COLOR=SimpleNamespace(name="SELECT_MODE_COLOR", value=28),
            SELECT_MODE_BOX=SimpleNamespace(name="SELECT_MODE_BOX", value=29),
            SELECT_MODE_SPHERE=SimpleNamespace(name="SELECT_MODE_SPHERE", value=30),
            CONFIRM_POLYGON=SimpleNamespace(name="CONFIRM_POLYGON", value=31),
            CANCEL_POLYGON=SimpleNamespace(name="CANCEL_POLYGON", value=32),
            UNDO_POLYGON_VERTEX=SimpleNamespace(name="UNDO_POLYGON_VERTEX", value=33),
            TOGGLE_SELECTION_DEPTH_FILTER=SimpleNamespace(
                name="TOGGLE_SELECTION_DEPTH_FILTER", value=34
            ),
            TOGGLE_SELECTION_CROP_FILTER=SimpleNamespace(
                name="TOGGLE_SELECTION_CROP_FILTER", value=35
            ),
            DEPTH_ADJUST_FAR=SimpleNamespace(name="DEPTH_ADJUST_FAR", value=36),
            APPLY_CROP_BOX=SimpleNamespace(name="APPLY_CROP_BOX", value=37),
            NODE_PICK=SimpleNamespace(name="NODE_PICK", value=38),
            NODE_RECT_SELECT=SimpleNamespace(name="NODE_RECT_SELECT", value=39),
            UNDO=SimpleNamespace(name="UNDO", value=40),
            REDO=SimpleNamespace(name="REDO", value=41),
            SELECT_ALL=SimpleNamespace(name="SELECT_ALL", value=42),
            COPY_SELECTION=SimpleNamespace(name="COPY_SELECTION", value=43),
            CUT_SELECTION=SimpleNamespace(name="CUT_SELECTION", value=44),
            PASTE_SELECTION=SimpleNamespace(name="PASTE_SELECTION", value=45),
            INVERT_SELECTION=SimpleNamespace(name="INVERT_SELECTION", value=46),
            DESELECT_ALL=SimpleNamespace(name="DESELECT_ALL", value=47),
            TOGGLE_SELECTION_DEPTH=SimpleNamespace(name="TOGGLE_SELECTION_DEPTH", value=48),
            TOGGLE_GRID=SimpleNamespace(name="TOGGLE_GRID", value=49),
            TOGGLE_SPLIT_VIEW=SimpleNamespace(name="TOGGLE_SPLIT_VIEW", value=50),
            TOGGLE_INDEPENDENT_SPLIT_VIEW=SimpleNamespace(
                name="TOGGLE_INDEPENDENT_SPLIT_VIEW", value=51
            ),
            TOGGLE_GT_COMPARISON=SimpleNamespace(name="TOGGLE_GT_COMPARISON", value=52),
            TOGGLE_CAMERA_FRUSTUMS=SimpleNamespace(name="TOGGLE_CAMERA_FRUSTUMS", value=53),
            CYCLE_PLY=SimpleNamespace(name="CYCLE_PLY", value=54),
            CYCLE_SELECTION_VIS=SimpleNamespace(name="CYCLE_SELECTION_VIS", value=55),
            TOOL_SELECT=SimpleNamespace(name="TOOL_SELECT", value=56),
            TOOL_TRANSLATE=SimpleNamespace(name="TOOL_TRANSLATE", value=57),
            TOOL_ROTATE=SimpleNamespace(name="TOOL_ROTATE", value=58),
            TOOL_SCALE=SimpleNamespace(name="TOOL_SCALE", value=59),
            TOOL_MIRROR=SimpleNamespace(name="TOOL_MIRROR", value=60),
            TOOL_ALIGN=SimpleNamespace(name="TOOL_ALIGN", value=61),
            PIE_MENU=SimpleNamespace(name="PIE_MENU", value=62),
            TOGGLE_UI=SimpleNamespace(name="TOGGLE_UI", value=63),
            TOGGLE_FULLSCREEN=SimpleNamespace(name="TOGGLE_FULLSCREEN", value=64),
            OPEN_PREFERENCES=SimpleNamespace(name="OPEN_PREFERENCES", value=65),
            TOGGLE_MCP_SERVER=SimpleNamespace(name="TOGGLE_MCP_SERVER", value=66),
            TOGGLE_MCP_BINDING=SimpleNamespace(name="TOGGLE_MCP_BINDING", value=67),
            HISTOGRAM_ZOOM_MARKED=SimpleNamespace(name="HISTOGRAM_ZOOM_MARKED", value=68),
            TOGGLE_PERFORMANCE_HUD=SimpleNamespace(name="TOGGLE_PERFORMANCE_HUD", value=69),
            SEQUENCER_ADD_KEYFRAME=SimpleNamespace(name="SEQUENCER_ADD_KEYFRAME", value=70),
            SEQUENCER_UPDATE_KEYFRAME=SimpleNamespace(name="SEQUENCER_UPDATE_KEYFRAME", value=71),
            SEQUENCER_PLAY_PAUSE=SimpleNamespace(name="SEQUENCER_PLAY_PAUSE", value=72),
            DELETE_SELECTED=SimpleNamespace(name="DELETE_SELECTED", value=73),
            BRUSH_RESIZE=SimpleNamespace(name="BRUSH_RESIZE", value=74),
            DELETE_NODE=SimpleNamespace(name="DELETE_NODE", value=75),
        ),
        get_available_profiles=lambda: ["Default"],
        get_current_profile=lambda: "Default",
        get_tool_mode_name=lambda mode: str(mode),
        get_action_name=lambda value: str(value),
        get_trigger=lambda *_a, **_k: None,
        set_trigger_binding=lambda *_a, **_k: True,
        find_conflict_for_action=lambda *_a, **_k: None,
        get_allowed_trigger_kinds=lambda *_a, **_k: ["key"],
        is_capturing=lambda: False,
        is_waiting_for_double_click=lambda: False,
        load_profile=lambda _name: None,
        save_profile=lambda _name: None,
        reset_to_default=lambda: None,
        export_profile=lambda _path: None,
        import_profile=lambda _path: None,
        start_capture=lambda *_a, **_k: None,
        cancel_capture=lambda: None,
        clear_binding=lambda *_a, **_k: None,
        get_captured_trigger=lambda: None,
        get_trigger_description=lambda *_a, **_k: "",
    )

    def file_associations_status():
        return [dict(row) for row in state.file_associations]

    def file_association_set(extension, enabled):
        state.file_association_set_calls.append((str(extension), bool(enabled)))
        for row in state.file_associations:
            if row["extension"] == extension:
                row["registered"] = bool(enabled)
                return True
        return False

    lf_stub.get_render_settings = lambda: state.render_settings
    lf_stub.get_camera_navigation_mode = lambda: "orbit"
    lf_stub.get_camera_view_snap_enabled = lambda: False
    lf_stub.file_associations_status = file_associations_status
    lf_stub.file_association_set = file_association_set

    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    sys.modules.pop("lfs_plugins.preferences_panel", None)
    sys.modules.pop("lfs_plugins.keymap_bindings", None)
    sys.modules.pop("lfs_plugins", None)
    module = import_module("lfs_plugins.preferences_panel")
    return module, state


def test_language_selection_does_not_reload_active_language(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._language_catalog = [("en", "English"), ("it", "Italiano")]

    panel._set_language_index("1")

    assert state.set_language_calls == []


def test_scene_reconstruction_uses_backend_specific_presets(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._sync_scene_reconstruction_catalog()
    panel._refresh_selection = lambda: None

    panel._set_scene_upscaler_index("1")
    assert state.render_settings.scene_upscaler == "spatial"
    assert state.render_settings.scene_upscaler_preset == "quality"

    panel._set_scene_upscaler_preset_index("2")
    assert state.render_settings.scene_upscaler_preset == "performance"

    panel._set_scene_upscaler_index("0")
    assert state.render_settings.scene_upscaler == "native"
    assert state.render_settings.scene_upscaler_preset == "native"

    panel._set_scene_upscaler_index("1")
    assert state.render_settings.scene_upscaler == "spatial"
    assert state.render_settings.scene_upscaler_preset == "performance"


def test_scene_reconstruction_restores_the_backend_specific_preset(preferences_panel_module):
    module, state = preferences_panel_module
    state.scene_reconstruction_presets["spatial"] = "performance"
    panel = module.PreferencesPanel()
    panel._sync_scene_reconstruction_catalog()
    panel._refresh_selection = lambda: None

    panel._set_scene_upscaler_index("1")

    assert state.render_settings.scene_upscaler == "spatial"
    assert state.render_settings.scene_upscaler_preset == "performance"


def test_scene_reconstruction_preset_records_mark_the_active_preset(
    preferences_panel_module,
):
    module, state = preferences_panel_module
    state.render_settings.scene_upscaler = "spatial"
    state.render_settings.scene_upscaler_preset = "performance"
    panel = module.PreferencesPanel()
    panel._sync_scene_reconstruction_catalog()

    records = []
    panel._handle = SimpleNamespace(
        update_record_list=lambda name, items: records.extend(
            items if name == "scene_upscaler_presets" else []
        )
    )
    panel._sync_scene_upscaler_preset_records()

    assert [record["selected"] for record in records] == [False, False, True]


def test_language_selection_applies_a_different_language(
    preferences_panel_module, monkeypatch
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._language_catalog = [("en", "English"), ("it", "Italiano")]
    monkeypatch.setattr(panel, "_refresh_selection", lambda: None)

    panel._set_language_index("0")

    assert state.set_language_calls == ["en"]


def test_mcp_port_is_drafted_until_explicit_confirmation(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()

    panel._set_mcp_port("50123")
    assert state.set_mcp_calls == []

    assert panel._commit_mcp_port()
    assert state.set_mcp_calls == [
        {
            "enabled": True,
            "expose_network": False,
            "port": 50123,
            "request_logging": False,
        }
    ]


def test_invalid_mcp_port_blocks_application(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()

    panel._set_mcp_port("70000")

    assert not panel._commit_mcp_port()
    assert state.set_mcp_calls == []
    assert panel._mcp_error_text() == "preferences.mcp_invalid_port"


def test_failed_bind_can_retry_the_same_confirmed_port(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()
    state.mcp_status.update(
        {
            "running": False,
            "phase": "failed",
            "error": "bind failed",
            "error_kind": "bind_failed",
        }
    )

    assert panel._commit_mcp_port()

    assert state.set_mcp_calls == [
        {
            "enabled": True,
            "expose_network": False,
            "port": 45677,
            "request_logging": False,
        }
    ]


def test_starting_listener_does_not_retry_an_already_confirmed_port(
    preferences_panel_module,
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()
    state.mcp_status.update({"running": False, "phase": "starting"})

    assert panel._commit_mcp_port()
    assert state.set_mcp_calls == []


def test_mcp_section_request_is_consumed_once(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    state.section_request = "mcp"

    panel._consume_section_request()

    assert panel._section == "mcp"
    assert state.section_request == ""


def test_mcp_endpoint_fallback_never_exposes_listener_wildcard(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    state.mcp_status.update({"expose_network": True, "endpoints": []})

    endpoints = panel._mcp_endpoint_text()

    assert "0.0.0.0" not in endpoints
    assert "http://127.0.0.1:45677/mcp" in endpoints


def test_mcp_runtime_error_explains_the_failed_endpoint(
    preferences_panel_module, monkeypatch
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    state.mcp_status.update(
        {
            "phase": "failed",
            "error": "Unable to bind 0.0.0.0:45677",
            "error_kind": "bind_failed",
            "error_address": "0.0.0.0",
            "error_port": 45677,
        }
    )
    monkeypatch.setattr(
        module.lf.ui,
        "tr",
        lambda key: "Cannot listen on {endpoint}" if key == "preferences.mcp_bind_failed" else key,
    )

    assert panel._mcp_error_text() == "Cannot listen on 0.0.0.0:45677"


def test_mcp_starting_phase_is_not_reported_as_an_error(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    state.mcp_status.update({"running": False, "phase": "starting"})

    assert panel._mcp_status_text() == "preferences.mcp_status_starting"


def test_title_bar_close_listener_discards_unconfirmed_mcp_port(
    preferences_panel_module, monkeypatch
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()

    class CloseButton:
        def __init__(self):
            self.listeners = []

        def add_event_listener(self, event_name, callback):
            assert event_name == "click"
            self.listeners.append(callback)

    close_button = CloseButton()
    document = SimpleNamespace(
        get_element_by_id=lambda element_id: (
            close_button if element_id == "close-btn" else None
        )
    )
    monkeypatch.setattr(panel, "_dirty_expanded_sections", lambda: None)
    monkeypatch.setattr(panel, "_rebuild_records", lambda: None)
    monkeypatch.setattr(panel, "_consume_section_request", lambda: None)
    monkeypatch.setattr(panel, "_state", lambda: ())
    monkeypatch.setattr(panel, "_refresh_selection", lambda: None)

    panel.on_mount(document)
    panel._set_mcp_port("47000")

    assert len(close_button.listeners) == 1
    close_button.listeners[0](None)

    assert panel._mcp_port == "45677"
    assert state.set_mcp_calls == []
    assert state.panel_enabled_calls[-1] == ("lfs.preferences", False)


def test_external_mcp_toggle_preserves_an_unconfirmed_port_draft(
    preferences_panel_module,
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()
    panel._set_mcp_port("47000")

    state.mcp_preferences["expose_network"] = True
    state.mcp_status["expose_network"] = True
    panel._sync_mcp_runtime()

    assert panel._mcp_port == "47000"
    assert panel._mcp_applied_port == 45677
    assert panel._mcp_expose_network is True


def test_safe_mode_disables_live_mcp_mutations(preferences_panel_module):
    module, state = preferences_panel_module
    state.mcp_preferences.update({"enabled": False, "safe_mode": True})
    state.mcp_status.update({"enabled": False, "phase": "disabled", "safe_mode": True})
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()

    panel._set_mcp_enabled(True)
    panel._set_mcp_expose_network(True)
    panel._set_mcp_port("47000")
    panel._set_mcp_request_logging(True)

    assert panel._mcp_safe_mode is True
    assert panel._mcp_enabled is False
    assert panel._mcp_port == "45677"
    assert state.set_mcp_calls == []


def test_safe_mode_reset_does_not_stage_mcp_changes(preferences_panel_module):
    module, state = preferences_panel_module
    state.mcp_preferences.update({"enabled": False, "safe_mode": True})
    state.mcp_status.update({"enabled": False, "phase": "disabled", "safe_mode": True})
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()
    panel._refresh_selection = lambda: None

    snapshot = (
        panel._mcp_enabled,
        panel._mcp_expose_network,
        panel._mcp_port,
        panel._mcp_applied_port,
        panel._mcp_request_logging,
        panel._mcp_safe_mode,
    )

    panel._section = "mcp"
    panel._reset_section()

    assert state.set_mcp_calls == []
    assert (
        panel._mcp_enabled,
        panel._mcp_expose_network,
        panel._mcp_port,
        panel._mcp_applied_port,
        panel._mcp_request_logging,
        panel._mcp_safe_mode,
    ) == snapshot

    def confirm_dialog(_title, _message, _buttons, callback):
        callback("preferences.reset_all_settings")

    module.lf.ui.confirm_dialog = confirm_dialog
    module.lf.ui.message_dialog = lambda *_a, **_k: None
    module.lf.ui.set_theme = lambda *_a, **_k: None
    module.lf.ui.set_ui_scale = lambda *_a, **_k: None
    module.lf.ui.set_remember_camera_navigation = lambda *_a, **_k: None
    module.lf.ui.set_remember_camera_view_snap = lambda *_a, **_k: None
    module.lf.ui.reset_layout = lambda: None
    module.lf.ui.reset_window_state = lambda: None
    module.lf.set_camera_navigation_mode = lambda *_a, **_k: None
    module.lf.set_camera_view_snap_enabled = lambda *_a, **_k: None

    panel._on_reset_all_settings(None, None, None)

    assert state.set_mcp_calls == []
    assert state.render_settings.scene_upscaler == "native"
    assert state.render_settings.scene_upscaler_preset == "native"
    assert state.scene_reconstruction_presets == {"native": "native", "spatial": "quality"}
    assert (
        panel._mcp_enabled,
        panel._mcp_expose_network,
        panel._mcp_port,
        panel._mcp_applied_port,
        panel._mcp_request_logging,
        panel._mcp_safe_mode,
    ) == snapshot


def test_safe_mode_disables_preferences_and_status_bar_mcp_controls():
    project_root = Path(__file__).parent.parent.parent
    resources = project_root / "src" / "visualizer" / "gui" / "rmlui" / "resources"
    preferences = (resources / "preferences.rml").read_text(encoding="utf-8")
    status_bar = (resources / "statusbar.rml").read_text(encoding="utf-8")

    assert preferences.count('data-attrif-disabled="mcp_safe_mode"') == 7
    assert preferences.count('data-class-disabled="mcp_safe_mode"') == 2
    assert (
        '<button id="mcp-toggle" data-class-is-on="mcp_server_enabled" '
        'data-attr-title="mcp_toggle_label" data-attrif-disabled="safe_mode" '
        'data-class-disabled="safe_mode">'
        in status_bar
    )


def test_preferences_content_scrolls_without_overlapping_the_fixed_footer():
    project_root = Path(__file__).parent.parent.parent
    stylesheet = (
        project_root
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "preferences.rcss"
    ).read_text(encoding="utf-8")
    content_rule = stylesheet.split(".preferences-content {", 1)[1].split("}", 1)[0]

    assert "min-height: 0;" in content_rule
    assert "overflow-y: auto;" in content_rule
    assert "padding-right: 6dp;" in content_rule


def test_mcp_python_bindings_release_the_gil_around_native_work():
    project_root = Path(__file__).parent.parent.parent
    source = (project_root / "src" / "python" / "lfs" / "py_ui.cpp").read_text(
        encoding="utf-8"
    )

    for name in (
        '"get_mcp_preferences"',
        '"set_mcp_preferences"',
        '"get_mcp_status"',
        '"get_mcp_log_directory"',
    ):
        start = source.index(name)
        end = source.find("m.def(", start + len(name))
        block = source[start : end if end != -1 else len(source)]
        assert "nb::gil_scoped_release" in block


def test_footer_ok_commits_mcp_port_before_closing(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._read_mcp_preferences()
    panel._set_mcp_port("47000")

    panel._on_accept_and_close(None, None, None)

    assert state.set_mcp_calls == [
        {
            "enabled": True,
            "expose_network": False,
            "port": 47000,
            "request_logging": False,
        }
    ]


def test_mcp_failed_listener_does_not_advertise_inactive_endpoints(
    preferences_panel_module,
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    state.mcp_status.update(
        {
            "running": False,
            "error": "Unable to bind 127.0.0.1:45677",
        }
    )

    assert panel._mcp_endpoint_text() == "preferences.mcp_no_active_endpoint"


def test_file_associations_section_hidden_when_status_empty(preferences_panel_module):
    module, _state = preferences_panel_module
    panel = module.PreferencesPanel()
    records = {}
    panel._handle = SimpleNamespace(
        update_record_list=lambda name, items: records.__setitem__(name, list(items)),
        dirty=lambda *_a, **_k: None,
    )

    panel._reload_file_associations()

    assert panel._has_file_associations() is False
    panel._section = "file_associations"
    assert panel._show_file_associations() is False
    assert records.get("file_associations") == []

    rml = (
        Path(__file__).parent.parent.parent
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "preferences.rml"
    ).read_text(encoding="utf-8")
    assert 'data-if="has_file_associations"' in rml
    assert 'data-if="show_file_associations"' in rml
    assert 'data-for="row : file_associations"' in rml


def test_file_associations_rows_toggle_calls_set(preferences_panel_module):
    module, state = preferences_panel_module
    state.file_associations = [
        {"extension": ".ply", "registered": False},
        {"extension": ".licht", "registered": True},
    ]
    panel = module.PreferencesPanel()
    records = {}
    panel._handle = SimpleNamespace(
        update_record_list=lambda name, items: records.__setitem__(name, list(items)),
        dirty=lambda *_a, **_k: None,
    )

    panel._reload_file_associations()

    assert panel._has_file_associations() is True
    panel._section = "file_associations"
    assert panel._show_file_associations() is True
    assert [row["extension"] for row in records["file_associations"]] == [".ply", ".licht"]
    assert [row["registered"] for row in records["file_associations"]] == [False, True]

    panel._set_file_association(".ply", True)

    assert state.file_association_set_calls == [(".ply", True)]
    assert [row["registered"] for row in records["file_associations"]] == [True, True]
