# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for application preferences behavior."""

from enum import IntEnum
from importlib import import_module
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def test_project_folder_remove_label_uses_sentence_case_in_all_locales():
    expected = {
        "de": "Ordner entfernen",
        "en": "Remove folder",
        "es": "Quitar carpeta",
        "fr": "Retirer le dossier",
        "it": "Rimuovi cartella",
        "ja": "フォルダーを外す",
        "ko": "폴더 제거",
        "nl": "Map verwijderen",
        "pl": "Usuń folder",
        "zh": "移除文件夹",
    }
    locales = Path(__file__).resolve().parents[2] / "src/visualizer/gui/resources/locales"
    assert {path.stem: json.loads(path.read_text(encoding="utf-8"))["projects"]["action.remove_folder"]
            for path in locales.glob("*.json")} == expected


@pytest.fixture
def preferences_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) in sys.path:
        sys.path.remove(str(source_python))
    sys.path.insert(0, str(source_python))

    state = SimpleNamespace(
        language="it",
        splat_drag_preferences={"format": "ply", "sh_degree": 3, "up_axis": -2},
        sidebar_clear_calls=[],
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
        clipboard_text="",
        file_associations=[],
        file_association_set_calls=[],
        panel_enabled_calls=[],
        section_request="",
        render_settings=SimpleNamespace(
            scene_upscaler="native",
            scene_upscaler_preset="native",
        ),
        scene_reconstruction_presets={"native": "native", "spatial": "quality"},
        viewport_chrome_style="translucent",
        set_viewport_chrome_style_calls=[],
        viewport_toolbar_position="centered",
        set_viewport_toolbar_position_calls=[],
        camera_browse_repeat=True,
        camera_browse_repeat_rate=5.0,
        zoom_speed=11.0,
        navigation_speed=10.0,
        project_location="",
        embed_dataset_by_default=False,
        project_manager_preferences={
            "defaultView": "remember",
            "openAtStartup": True,
            "rememberState": True,
        },
    )

    def set_project_location(path):
        state.project_location = str(path)
        return ""

    def clear_project_location():
        state.project_location = ""

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

    def set_viewport_chrome_style(style):
        state.viewport_chrome_style = str(style)
        state.set_viewport_chrome_style_calls.append(str(style))

    def set_viewport_toolbar_position(position):
        state.viewport_toolbar_position = str(position)
        state.set_viewport_toolbar_position_calls.append(str(position))

    def set_speed(name, value):
        setattr(state, name, max(1.0, min(100.0, float(value))))

    lf_stub = ModuleType("lichtfeld")
    state.marketplace_preferences = {"view": "list", "git_checkout": False}
    state.marketplace_notifications = []
    lf_stub.ui = SimpleNamespace(
        get_splat_drag_preferences=lambda: dict(state.splat_drag_preferences),
        set_splat_drag_preferences=lambda values: state.splat_drag_preferences.update(values),
        get_marketplace_preferences=lambda: dict(state.marketplace_preferences),
        set_marketplace_view=lambda value: state.marketplace_preferences.update(view=value),
        set_marketplace_git_checkout=lambda value: state.marketplace_preferences.update(git_checkout=value),
        reset_marketplace_preferences=lambda: state.marketplace_preferences.update(view="list", git_checkout=False),
        get_panel_object=lambda _panel_id: SimpleNamespace(
            reload_marketplace_preferences=lambda: state.marketplace_notifications.append(True)),
        clear_startup_sidebar=lambda: state.sidebar_clear_calls.append(True),
        PanelSpace=SimpleNamespace(FLOATING="FLOATING"),
        PanelHeightMode=SimpleNamespace(FILL="fill"),
        PanelOption=SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED"),
        get_current_language=lambda: state.language,
        set_language=lambda language: state.set_language_calls.append(language),
        get_theme=lambda: "dark",
        get_theme_family=lambda: "lichtfeld",
        get_theme_mode=lambda: "dark",
        get_ui_scale_preference=lambda: 0.0,
        get_native_floating_windows=lambda: getattr(state, "native_floating_windows", True),
        set_native_floating_windows=lambda enabled: setattr(state, "native_floating_windows", enabled),
        set_ui_scale=lambda *_a, **_k: None,
        get_mcp_preferences=lambda: dict(state.mcp_preferences),
        get_mcp_port_override=lambda: None,
        get_mcp_access_token=lambda: "test-token",
        set_clipboard_text=lambda text: setattr(state, "clipboard_text", text),
        set_mcp_preferences=set_mcp_preferences,
        get_mcp_status=lambda: dict(state.mcp_status),
        get_project_location=lambda: state.project_location or "/home/tester/.lichtfeld/projects",
        get_project_location_preference=lambda: state.project_location,
        get_default_project_location=lambda: "/home/tester/.lichtfeld/projects",
        get_embed_dataset_by_default=lambda: state.embed_dataset_by_default,
        set_project_location=set_project_location,
        clear_project_location=clear_project_location,
        set_embed_dataset_by_default=lambda enabled: setattr(
            state, "embed_dataset_by_default", bool(enabled)
        ),
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
        get_progress_bar_style=lambda: "classic",
        set_progress_bar_style=lambda *_a, **_k: None,
        get_viewport_chrome_style=lambda: state.viewport_chrome_style,
        set_viewport_chrome_style=set_viewport_chrome_style,
        get_viewport_toolbar_position=lambda: state.viewport_toolbar_position,
        set_viewport_toolbar_position=set_viewport_toolbar_position,
        get_camera_browse_repeat=lambda: state.camera_browse_repeat,
        set_camera_browse_repeat=lambda value: setattr(state, "camera_browse_repeat", bool(value)),
        get_camera_browse_repeat_rate=lambda: state.camera_browse_repeat_rate,
        set_camera_browse_repeat_rate=lambda value, persist=True: setattr(state, "camera_browse_repeat_rate", float(value)),
        get_zoom_speed_preference=lambda: state.zoom_speed,
        set_zoom_speed_preference=lambda value, persist=True: set_speed("zoom_speed", value),
        get_navigation_speed_preference=lambda: state.navigation_speed,
        set_navigation_speed_preference=lambda value, persist=True: set_speed("navigation_speed", value),
        remember_camera_navigation=lambda: False,
        set_remember_camera_navigation=lambda _enabled: None,
        remember_camera_view_snap=lambda: False,
        set_remember_camera_view_snap=lambda _enabled: None,
    )
    state.rendering_defaults = {}
    lf_stub.ui.get_rendering_defaults_json = lambda: json.dumps(state.rendering_defaults)
    lf_stub.ui.set_rendering_defaults_json = lambda value: state.rendering_defaults.update(json.loads(value))
    state.auto_save_on_close = False
    lf_stub.project_auto_save_on_close_enabled = lambda: state.auto_save_on_close
    lf_stub.project_set_auto_save_on_close = lambda enabled: setattr(state, "auto_save_on_close", bool(enabled))
    lf_stub.keymap = SimpleNamespace(
        ToolMode=IntEnum(
            "ToolMode",
            {
                name: index
                for index, name in enumerate(
                    ("GLOBAL", "SELECTION", "TRANSLATE", "ROTATE", "SCALE", "ALIGN", "CROP_BOX", "WALK")
                )
            },
        ),
        Action=SimpleNamespace(
            EXIT_WALK=SimpleNamespace(name="EXIT_WALK", value=999),
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
            CAMERA_RESTORE_HOME=SimpleNamespace(name="CAMERA_RESTORE_HOME", value=91),
            CAMERA_NEXT_VIEW=SimpleNamespace(name="CAMERA_NEXT_VIEW", value=18),
            CAMERA_PREV_VIEW=SimpleNamespace(name="CAMERA_PREV_VIEW", value=19),
            CAMERA_FOCUS_SELECTION=SimpleNamespace(name="CAMERA_FOCUS_SELECTION", value=76),
            SELECTION_REPLACE=SimpleNamespace(name="SELECTION_REPLACE", value=20),
            SELECTION_ADD=SimpleNamespace(name="SELECTION_ADD", value=21),
            SELECTION_REMOVE=SimpleNamespace(name="SELECTION_REMOVE", value=22),
            SELECTION_INTERSECT=SimpleNamespace(name="SELECTION_INTERSECT", value=77),
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
            REMOVE_LAST_POINT=SimpleNamespace(name="REMOVE_LAST_POINT", value=94),
            APPLY_ALIGNMENT=SimpleNamespace(name="APPLY_ALIGNMENT", value=95),
            INSERT_CROP_POLYGON_VERTEX=SimpleNamespace(name="INSERT_CROP_POLYGON_VERTEX", value=96),
            REMOVE_CROP_POLYGON_VERTEX=SimpleNamespace(name="REMOVE_CROP_POLYGON_VERTEX", value=97),
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
            SELECT_ALL_SCENE_NODES=SimpleNamespace(
                name="SELECT_ALL_SCENE_NODES", value=82
            ),
            TOGGLE_SCENE_SELECTION_VISIBILITY=SimpleNamespace(
                name="TOGGLE_SCENE_SELECTION_VISIBILITY", value=83
            ),
            TOGGLE_SCENE_SELECTION_TRAINING=SimpleNamespace(
                name="TOGGLE_SCENE_SELECTION_TRAINING", value=84
            ),
            GROUP_SELECTED_SCENE_NODES=SimpleNamespace(
                name="GROUP_SELECTED_SCENE_NODES", value=78
            ),
            UNGROUP_SELECTED_SCENE_NODE=SimpleNamespace(
                name="UNGROUP_SELECTED_SCENE_NODE", value=79
            ),
            ASSET_GALLERY_PRIMARY=SimpleNamespace(
                name="ASSET_GALLERY_PRIMARY", value=80
            ),
            NEXT_SIDEBAR_CATEGORY=SimpleNamespace(name="NEXT_SIDEBAR_CATEGORY", value=92),
            PREVIOUS_SIDEBAR_CATEGORY=SimpleNamespace(name="PREVIOUS_SIDEBAR_CATEGORY", value=93),
            ASSET_GALLERY_COPY_LINK=SimpleNamespace(
                name="ASSET_GALLERY_COPY_LINK", value=81
            ),
            ASSET_REFRESH=SimpleNamespace(name="ASSET_REFRESH", value=85),
            TOGGLE_PYTHON_CONSOLE=SimpleNamespace(name="TOGGLE_PYTHON_CONSOLE", value=90),
        ),
        get_available_profiles=lambda: ["Default"],
        get_current_profile=lambda: "Default",
        bindings_revision=lambda: 0,
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
    state.navigation_up_axis = "legacy_y"
    state.navigation_up_calls = []
    state.new_project_axis = "legacy_y"
    state.new_project_align_grid = True
    state.new_project_navigation_calls = []
    lf_stub.get_new_project_navigation_up_axis = lambda: state.new_project_axis
    lf_stub.get_new_project_align_grid = lambda: state.new_project_align_grid
    def set_new_project_navigation_defaults(axis, align_grid):
        state.new_project_axis = axis
        state.new_project_align_grid = align_grid
        state.new_project_navigation_calls.append((axis, align_grid))
    lf_stub.set_new_project_navigation_defaults = set_new_project_navigation_defaults
    lf_stub.get_navigation_up_axis = lambda: state.navigation_up_axis
    def set_navigation_up_axis(axis, align_grid=False):
        state.navigation_up_axis = axis
        state.navigation_up_calls.append((axis, align_grid))
    lf_stub.set_navigation_up_axis = set_navigation_up_axis
    lf_stub.get_camera_navigation_mode = lambda: "orbit"
    lf_stub.set_camera_navigation_mode = lambda _mode: None
    lf_stub.get_camera_view_snap_enabled = lambda: False
    lf_stub.set_camera_view_snap_enabled = lambda _enabled: None
    lf_stub.file_associations_status = file_associations_status
    lf_stub.file_association_set = file_association_set

    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    sys.modules.pop("lfs_plugins.marketplace_preferences", None)
    sys.modules.pop("lfs_plugins.preferences_panel", None)
    sys.modules.pop("lfs_plugins.rendering_defaults", None)
    sys.modules.pop("lfs_plugins.keymap_bindings", None)
    sys.modules.pop("lfs_plugins", None)
    module = import_module("lfs_plugins.preferences_panel")
    monkeypatch.setattr(
        module,
        "read_project_manager_preferences",
        lambda: dict(state.project_manager_preferences),
    )
    monkeypatch.setattr(
        module,
        "set_project_manager_preference",
        lambda key, value: state.project_manager_preferences.__setitem__(key, value),
    )
    monkeypatch.setattr(
        module,
        "reset_project_manager_preferences",
        lambda: setattr(
            state,
            "project_manager_preferences",
            {"defaultView": "remember", "openAtStartup": True, "rememberState": True},
        ),
    )
    return module, state


def test_language_selection_does_not_reload_active_language(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._language_catalog = [("en", "English"), ("it", "Italiano")]

    panel._set_language_index("1")

    assert state.set_language_calls == []


def test_project_manager_preferences_round_trip_and_reset(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()

    panel._set_project_manager_default_view("gallery")
    panel._set_project_manager_open_at_startup(False)
    panel._set_project_manager_remember_state(False)

    assert state.project_manager_preferences == {
        "defaultView": "gallery",
        "openAtStartup": False,
        "rememberState": False,
    }

    panel._reset_section("general")

    assert state.project_manager_preferences == {
        "defaultView": "remember",
        "openAtStartup": True,
        "rememberState": True,
    }


def test_project_folders_preferences_add_remove_and_protect_default(preferences_panel_module, monkeypatch):
    module, _state = preferences_panel_module
    panel = module.PreferencesPanel()
    records = {}
    panel._handle = SimpleNamespace(
        update_record_list=lambda name, rows: records.__setitem__(name, rows),
        dirty=lambda _name: None,
    )
    folders = {"default": {"name": "Projects", "path": "/tmp/projects"}}
    calls = []
    backend = SimpleNamespace(
        _asset_index=object(),
        _asset_index_folders=lambda: folders,
        _add_folder_from_path=lambda path, recursive: calls.append((path, recursive)) or folders.__setitem__("extra", {"name": "Extra", "path": path}),
        on_delete_folder=lambda _h, _e, args: calls.append(("remove", args[0])),
        refresh_catalog=lambda **kwargs: calls.append(("rescan", kwargs)),
    )
    monkeypatch.setattr(module.lf.ui, "get_panel_object", lambda _id: backend, raising=False)
    monkeypatch.setattr(module.lf.ui, "open_folder_dialog", lambda *_: "/tmp/extra")
    monkeypatch.setattr(module.lf.ui, "confirm_dialog", lambda _title, _message, buttons, callback: callback(buttons[1]), raising=False)

    panel._refresh_project_folders()
    assert records["project_folders"][0]["id"] == "default"
    assert records["project_folders"][0]["can_remove"] is False
    panel._on_add_project_folder()
    assert calls == [("/tmp/extra", True)]
    panel._on_remove_project_folder(args=["default"])
    assert len(calls) == 1
    panel._on_remove_project_folder(args=["extra"])
    panel._on_rescan_project_folders()
    assert calls[-2:] == [("remove", "extra"), ("rescan", {"scan_folders": True})]


def test_portal_section_does_not_offer_settings_reset(preferences_panel_module, monkeypatch):
    module, _state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._section = "portal"
    dialogs = []
    monkeypatch.setattr(module.lf.ui, "confirm_dialog", lambda *args: dialogs.append(args), raising=False)
    panel._on_reset_current_section(None, None, None)

    assert dialogs == []
    assert panel._show_section_reset() is False


def test_preferences_portal_status_uses_shared_connection_state(preferences_panel_module, monkeypatch):
    module, _state = preferences_panel_module
    panel = module.PreferencesPanel()
    translations = {
        "portal.status.connect": "Connect to Portal",
        "portal.status.turn_on": "Turn on Portal",
        "portal.status.busy": "Portal: Working…",
        "portal.status.switched_off": "Portal connected, switched off",
        "portal.status.disconnected": "Portal: Not connected",
        "portal.status.connected_as": "Portal connected as {name}",
    }
    monkeypatch.setattr(module.lf.ui, "tr", lambda key: translations.get(key, key), raising=False)
    snapshots = (
        (SimpleNamespace(signed_in=False, authorized=False, linking=False, disconnecting=False,
                         display_name=""), "Portal: Not connected"),
        (SimpleNamespace(signed_in=False, authorized=True, linking=False, disconnecting=False,
                         display_name=""), "Portal connected, switched off"),
        (SimpleNamespace(signed_in=False, authorized=False, linking=True, disconnecting=False,
                         display_name=""), "Portal: Working…"),
        (SimpleNamespace(signed_in=True, authorized=True, linking=False, disconnecting=False,
                         display_name="A Long Display Name"), "Portal connected as A Long Display Name"),
    )
    for snapshot, expected in snapshots:
        monkeypatch.setattr(panel, "_portal_connection_snapshot", lambda snapshot=snapshot: snapshot)
        assert panel._portal_connection_status() == expected


def test_viewport_chrome_selection_uses_global_style_preference(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._refresh_selection = lambda: None

    assert panel._viewport_chrome_index() == "1"
    panel._set_viewport_chrome_index("2")

    assert state.viewport_chrome_style == "frosted"
    assert state.set_viewport_chrome_style_calls == ["frosted"]
    assert panel._viewport_chrome_index() == "2"


def test_viewport_toolbar_position_uses_global_preference(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._refresh_selection = lambda: None

    assert panel._viewport_toolbar_position_index() == "1"
    panel._set_viewport_toolbar_position_index("2")

    assert state.viewport_toolbar_position == "free"
    assert state.set_viewport_toolbar_position_calls == ["free"]
    assert panel._viewport_toolbar_position_index() == "2"


def test_navigation_speed_preferences_round_trip_and_clamp(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()

    panel._set_zoom_speed(42)
    panel._set_navigation_speed(73)
    assert panel._get_scrub_value("zoom_speed") == 42
    assert panel._get_scrub_value("navigation_speed") == 73

    panel._set_zoom_speed(0)
    panel._set_navigation_speed(101)
    assert state.zoom_speed == 1
    assert state.navigation_speed == 100


@pytest.mark.parametrize("field", ["zoom_speed", "navigation_speed"])
def test_speed_scrub_moves_preview_and_write_once_on_commit(preferences_panel_module,
                                                              monkeypatch, tmp_path, field):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._refresh_selection = lambda: None
    writes = []
    reads = []
    path = tmp_path / "preferences.json"
    path.write_text(json.dumps({field: getattr(state, field)}))

    def set_speed(value, persist=True):
        setattr(state, field, float(value))
        if persist:
            reads.append(path.read_text())
            path.write_text(json.dumps({field: float(value)}))
            writes.append(float(value))

    monkeypatch.setattr(module.lf.ui, f"set_{field}_preference", set_speed)
    for move in range(20):
        panel._set_scrub_value(field, 20 + move)
        assert getattr(state, field) == 20 + move
    assert writes == []
    assert reads == []
    panel._commit_scrub_value(field, getattr(state, field))
    assert writes == [39.0]
    assert len(reads) == 1
    assert json.loads(path.read_text())[field] == 39.0


def test_navigation_speed_preferences_reset_with_input_section(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._section = "input"
    panel._set_zoom_speed(42)
    panel._set_navigation_speed(73)
    panel._refresh_selection = lambda: None

    panel._reset_section()

    assert state.zoom_speed == 11
    assert state.navigation_speed == 10


def test_project_location_is_saved_and_can_return_to_default(
    preferences_panel_module,
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._read_project_location()

    assert panel._project_location == "/home/tester/.lichtfeld/projects"

    panel._set_project_location_draft("/mnt/projects")
    assert panel._commit_project_location() is True
    assert state.project_location == "/mnt/projects"
    assert panel._applied_project_location == "/mnt/projects"

    panel._on_use_default_project_location(None, None, None)
    assert state.project_location == ""
    assert panel._project_location == "/home/tester/.lichtfeld/projects"


def test_general_preferences_expose_project_location_controls():
    project_root = Path(__file__).parent.parent.parent
    rml = (
        project_root
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "preferences.rml"
    ).read_text(encoding="utf-8")

    assert 'data-event-click="toggle_section(\'project_location\')"' in rml
    assert 'data-value="project_location"' in rml
    assert 'data-event-click="browse_project_location"' in rml
    assert 'data-event-click="use_default_project_location"' in rml


def test_project_manager_uses_a_standard_section_and_uniform_control_width():
    project_root = Path(__file__).parent.parent.parent
    resources = project_root / "src" / "visualizer" / "gui" / "rmlui" / "resources"
    rml = (resources / "preferences.rml").read_text(encoding="utf-8")
    rcss = (resources / "preferences.rcss").read_text(encoding="utf-8")

    assert 'data-event-click="toggle_section(\'project_manager\')"' in rml
    assert 'data-if="project_manager_expanded"' in rml
    assert "preferences-subheading" not in rml
    select_rule = rcss.split(".preferences-select {", 1)[1].split("}", 1)[0]
    assert "box-sizing: border-box;" in select_rule
    assert "width: 200dp;" in select_rule


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


def test_mcp_port_note_names_the_command_line_port(preferences_panel_module, monkeypatch):
    module, _state = preferences_panel_module
    panel = module.PreferencesPanel()
    assert panel._mcp_port_override_text() == ""

    monkeypatch.setattr(module.lf.ui, "get_mcp_port_override", lambda: 45696)
    monkeypatch.setattr(module.lf.ui, "tr", lambda key: "uses {port}" if key == "preferences.mcp_port_override" else key)

    assert panel._mcp_port_override_text() == "uses 45696"


def test_mcp_access_token_copy_uses_displayed_value(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()

    assert panel._mcp_token_text() == "test-token"
    panel._on_copy_mcp_token(None, None, None)

    assert state.clipboard_text == "test-token"


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
        ),
        query_selector_all=lambda _selector: [],
        add_event_listener=lambda *_: None,
        set_attribute=lambda *_: None,
        has_attribute=lambda *_: False,
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

    state.marketplace_preferences.update(view="card", git_checkout=True)
    panel._on_reset_all_settings(None, None, None)
    assert state.marketplace_preferences == {"view": "list", "git_checkout": False}
    assert state.sidebar_clear_calls == [True]

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

    import re
    drag_controls = re.findall(r'<(?:input|select)\b[^>]*splat_drag_[^>]*>', preferences)
    assert drag_controls
    assert all('data-attrif-disabled="mcp_safe_mode"' in control for control in drag_controls)
    assert preferences.count('data-attrif-disabled="mcp_safe_mode"') - len(drag_controls) == 8
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


def test_embed_dataset_default_is_exposed_and_settable(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()

    assert module.lf.ui.get_embed_dataset_by_default() is False
    panel._set_embed_dataset_by_default(True)
    assert state.embed_dataset_by_default is True
    assert module.lf.ui.get_embed_dataset_by_default() is True


def test_preferences_close_retains_the_mounted_document(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()

    class Document:
        def has_attribute(self, _name):
            return False

        def set_attribute(self, _name, _value):
            pass

        def add_event_listener(self, _name, _callback):
            pass

        def get_element_by_id(self, _element_id):
            return None

        def query_selector_all(self, _selector):
            return []

    document = Document()
    panel._rebuild_records = lambda: None
    panel._load_mcp_preferences = lambda: None
    panel._consume_section_request = lambda: None
    panel._refresh_selection = lambda: None
    panel._state = lambda: None
    panel._dirty_expanded_sections = lambda: None
    panel._keymap.on_mount = lambda _doc: None
    panel.on_mount(document)
    panel._on_close(None, None, None)

    assert panel.mount_count == 1
    assert panel._document is document
    assert state.panel_enabled_calls[-1] == ("lfs.preferences", False)


def test_preferences_keymap_rows_are_created_when_expanded(preferences_panel_module):
    module, _state = preferences_panel_module
    panel = module.PreferencesPanel()
    records = {}
    panel._keymap._handle = SimpleNamespace(
        update_record_list=lambda name, items: records.__setitem__(name, list(items)),
    )
    panel._section = "input"
    panel._input_page = "shortcuts"

    assert "key_bindings" in panel._expanded_sections
    assert panel._keymap._rows_built is False

    panel._expanded_sections.discard("key_bindings")
    panel._on_toggle_section(None, None, ["key_bindings"])

    assert panel._keymap._rows_built is True
    assert records["binding_rows"]


def test_input_pages_cancel_capture_and_release_diagnostic_device(preferences_panel_module):
    module, _state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._section = "input"
    panel._input_page = "gamepad"
    panel._gamepad_page = "test"
    calls = []
    module.lf.keymap.cancel_capture = lambda: calls.append("cancel")
    panel._gamepad_test.deactivate = lambda: calls.append("release")
    panel._keymap.ensure_binding_rows = lambda: calls.append("rows")
    assert not panel._show_section_reset()
    panel._set_input_page("invalid")
    assert not calls
    panel._set_input_page("shortcuts")
    assert calls == ["cancel", "release", "rows"]
    assert panel._show_section_reset()
    panel._set_input_page("navigation")
    assert calls[-2:] == ["cancel", "release"]
    panel._set_section("general")
    assert calls[-2:] == ["cancel", "release"]


@pytest.mark.parametrize("page", ["navigation", "shortcuts", "gamepad"])
def test_input_page_reset_does_not_reset_other_pages(preferences_panel_module, page):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._section = "input"
    panel._input_page = page
    panel._refresh_selection = lambda: None
    panel._set_zoom_speed(42)
    panel._set_navigation_speed(73)
    resets = []
    panel._keymap.reset_to_default = lambda: resets.append("shortcuts")
    panel._gamepad_test.configure = lambda **_kwargs: resets.append("gamepad")
    module.lf.ui.confirm_dialog = lambda _title, _body, buttons, callback: callback(buttons[1])
    module.lf.ui.message_dialog = lambda *_args: None
    panel._on_reset_current_section(None, None, None)
    assert resets == ([] if page == "navigation" else [page])
    assert state.zoom_speed == (11 if page == "navigation" else 42)
    assert state.navigation_speed == (10 if page == "navigation" else 73)


def test_shortcut_search_keeps_matching_sections_and_typed_bindings(preferences_panel_module):
    module, _state = preferences_panel_module
    keymap = module.PreferencesPanel()._keymap
    heading = dict(is_section=True, section_title="Navigazione")
    orbit = dict(is_section=False, action_name="Orbita camera", desc_text="MMB Trascina", action_id="5", mode_id="0")
    other = dict(is_section=True, section_title="Sequencer")
    play = dict(is_section=False, action_name="Play/Pause", desc_text="Space", action_id="9", mode_id="0")
    rows = [heading, orbit, other, play]
    assert keymap._filter_rows(rows, "  ") == rows
    assert keymap._filter_rows(rows, "NAVIGAZIONE mmb") == [heading, orbit]
    assert keymap._filter_rows(rows, "space") == [other, play]
    assert keymap._filter_rows(rows, "missing") == []
    assert keymap._filter_rows(rows, "camera")[1] is orbit


def test_input_page_labels_and_search_exist_in_all_locales():
    root = Path(__file__).resolve().parents[2] / "src/visualizer/gui/resources/locales"
    keys = ["preferences." + key for key in ("input_navigation", "input_shortcuts", "input_gamepad", "gamepad_controls", "gamepad_diagnostics")]
    keys += ["input_settings." + key for key in ("search_commands", "clear_search", "no_matches")]
    for path in root.glob("*.json"):
        values = json.loads(path.read_text(encoding="utf-8"))
        assert all(values.get(key) for key in keys), path.name


def test_input_group_collapses_without_changing_page_and_children_reopen_it(preferences_panel_module):
    module, _state = preferences_panel_module
    panel = module.PreferencesPanel()
    assert not panel._input_expanded
    panel._toggle_input_group()
    assert panel._section == "input" and panel._input_expanded
    panel._set_input_page("gamepad")
    panel._toggle_input_group()
    assert not panel._input_expanded and panel._input_page == "gamepad"
    assert panel._section == "input"
    panel._set_section("general")
    panel._set_input_page("gamepad")
    assert panel._section == "input" and panel._input_expanded
    assert panel._input_page == "gamepad"


@pytest.mark.parametrize("section,safe,visible", [
    ("portal", False, False), ("file_associations", False, False),
    ("mcp", True, False), ("mcp", False, True), ("input", True, True),
])
def test_reset_visibility_respects_scope_and_safe_mode(preferences_panel_module, section, safe, visible):
    module, _state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._section, panel._mcp_safe_mode = section, safe
    confirmations = []
    module.lf.ui.confirm_dialog = lambda *args: confirmations.append(args)
    assert panel._show_section_reset() == visible
    panel._on_reset_current_section(None, None, None)
    assert bool(confirmations) == visible


def test_page_reset_confirmation_captures_scope_and_explains_hidden_bindings(preferences_panel_module):
    module, _state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._section, panel._input_page, panel._mcp_safe_mode = "input", "shortcuts", True
    confirmations, resets = [], []
    module.lf.ui.confirm_dialog = lambda *args: confirmations.append(args)
    module.lf.ui.message_dialog = lambda *_args: None
    panel._reset_section = lambda scope: resets.append(scope)
    panel._on_reset_current_section(None, None, None)
    _title, message, buttons, callback = confirmations[0]
    assert "input_settings.reset_scope" in message
    assert "preferences.reset_safe_mode_note" in message
    callback(buttons[0])
    assert not resets
    panel._input_page = "gamepad"
    callback(buttons[1])
    assert resets == ["input_shortcuts"]


@pytest.mark.parametrize("confirmed", [False, True])
def test_shortcut_table_reset_also_requires_confirmation(preferences_panel_module, confirmed):
    module, _state = preferences_panel_module
    keymap = module.PreferencesPanel()._keymap
    resets, confirmations = [], []
    keymap.reset_to_default = lambda: resets.append(True)
    module.lf.ui.confirm_dialog = lambda *args: confirmations.append(args)
    keymap._on_reset_default(None, None, None)
    assert not resets
    _title, message, buttons, callback = confirmations[0]
    assert message == "input_settings.reset_scope"
    callback(buttons[1 if confirmed else 0])
    assert resets == ([True] if confirmed else [])


def test_global_reset_covers_input_macro_and_documents_exclusions(preferences_panel_module):
    module, _state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._mcp_safe_mode = True
    panel._refresh_selection = lambda: None
    confirmations, scopes = [], []
    panel._reset_section = lambda scope: scopes.append(scope)
    module.lf.ui.reset_window_state = lambda: None
    module.lf.ui.message_dialog = lambda *_args: None
    module.lf.ui.confirm_dialog = lambda *args: confirmations.append(args)
    panel._on_reset_all_settings(None, None, None)
    _title, message, buttons, callback = confirmations[0]
    assert "preferences.reset_all_scope" in message and "preferences.reset_safe_mode_note" in message
    callback(buttons[0])
    assert not scopes
    callback(buttons[1])
    assert scopes == ["general", "appearance", "viewport", "input", "interface", "marketplace", "mcp"]


@pytest.mark.parametrize("reset_scope", ["section", "all", "bindings"])
@pytest.mark.parametrize("confirmed", [False, True])
def test_input_reset_restores_bindings_and_refreshes_rows(
    preferences_panel_module, reset_scope, confirmed,
):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._section = "input"
    panel._input_page = "shortcuts"
    panel._refresh_selection = lambda: None
    panel._set_zoom_speed(42)
    panel._set_navigation_speed(73)
    panel._set_navigation_up_index("1")
    panel._set_new_project_align_grid(False)
    state.navigation_up_axis = "data_z"
    defaults = {"forward": "W", "backward": "S", "left": "A"}
    bindings = {"forward": "S", "backward": "Unbound", "left": "A"}
    reset_calls = []

    def reset_bindings():
        reset_calls.append(True)
        bindings.update(defaults)

    module.lf.keymap.reset_to_default = reset_bindings
    module.lf.ui.confirm_dialog = lambda _title, _body, buttons, callback: callback(
        buttons[1 if confirmed else 0]
    )
    module.lf.ui.message_dialog = lambda *_args: None
    module.lf.ui.set_theme = lambda _theme: None
    module.lf.ui.reset_layout = lambda: None
    module.lf.ui.reset_window_state = lambda: None
    module.lf.ui.set_scene_graph_selection_markers = lambda _enabled: None
    dirty = []
    panel._keymap._handle = SimpleNamespace(dirty=dirty.append)
    panel._keymap._last_state_key = ("old bindings",)
    panel._keymap._pending_conflict = {"old": "conflict"}

    if reset_scope == "section":
        panel._on_reset_current_section(None, None, None)
    elif reset_scope == "all":
        panel._on_reset_all_settings(None, None, None)
    elif confirmed:
        panel._keymap._on_reset_default(None, None, None)

    if confirmed:
        assert bindings == defaults
        assert reset_calls == [True]
        assert panel._keymap._last_state_key is None
        assert panel._keymap._pending_conflict is None
        assert {"profile_idx", "mode_idx", "is_capturing", "bindings_hint"} <= set(dirty)
    else:
        assert bindings == {"forward": "S", "backward": "Unbound", "left": "A"}
        assert reset_calls == []
        assert panel._keymap._last_state_key == ("old bindings",)
        assert dirty == []

    reset_input = confirmed and reset_scope == "all"
    assert state.zoom_speed == (11 if reset_input else 42)
    assert state.navigation_speed == (10 if reset_input else 73)
    assert panel._navigation_up_index() == ("0" if reset_input else "1")
    assert state.new_project_align_grid is reset_input
    assert state.navigation_up_axis == "data_z"
    assert state.navigation_up_calls == []


@pytest.mark.parametrize("section", ["general", "appearance", "interface", "mcp", "portal"])
def test_other_section_resets_preserve_key_bindings(preferences_panel_module, section):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._refresh_selection = lambda: None
    reset_calls = []
    module.lf.keymap.reset_to_default = lambda: reset_calls.append(True)
    module.lf.ui.set_theme = lambda _theme: None
    module.lf.ui.reset_layout = lambda: None
    module.lf.ui.set_scene_graph_selection_markers = lambda _enabled: None
    panel._keymap._last_state_key = ("custom bindings",)
    panel._set_zoom_speed(42)
    panel._set_navigation_speed(73)

    panel._reset_section(section)

    assert reset_calls == []
    assert panel._keymap._last_state_key == ("custom bindings",)
    assert state.zoom_speed == 42
    assert state.navigation_speed == 73


def test_navigation_defaults_do_not_modify_current_project(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._refresh_selection = lambda: None
    assert panel._navigation_up_index() == "0"
    panel._set_navigation_up_index("1")
    assert state.new_project_navigation_calls == [("data_z", True)]
    assert panel._navigation_up_index() == "1"
    panel._set_navigation_up_index("invalid")
    panel._set_navigation_up_index("2")
    assert len(state.new_project_navigation_calls) == 1
    panel._set_new_project_align_grid(False)
    assert state.new_project_navigation_calls[-1] == ("data_z", False)
    assert state.navigation_up_calls == []
    assert state.navigation_up_axis == "legacy_y"

def test_interface_reset_disables_external_windows_without_resetting_input(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    state.native_floating_windows = True
    calls = []
    module.lf.ui.reset_layout = lambda: calls.append('layout')
    module.lf.ui.clear_startup_sidebar = lambda: calls.append('clear_sidebar')
    module.lf.ui.set_scene_graph_selection_markers = lambda _enabled: None
    module.lf.keymap.reset_to_default = lambda: calls.append('input')
    panel._reset_section('interface')
    assert state.native_floating_windows is False
    assert calls == ['clear_sidebar', 'layout']


def test_appearance_reset_preserves_native_window_mode(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._refresh_selection = lambda: None
    state.native_floating_windows = False
    module.lf.ui.set_theme = lambda _theme: None
    panel._reset_section('appearance')
    assert state.native_floating_windows is False


def test_window_mode_change_refreshes_preferences_state(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    before = panel._state()
    state.native_floating_windows = False
    assert panel._state() != before
    dirty = []
    panel._handle = SimpleNamespace(dirty=dirty.append)
    panel._dirty_selection()
    assert 'native_floating_windows' in dirty


def test_save_on_close_change_invalidates_preferences_state(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    before = panel._state()
    module.lf.project_set_auto_save_on_close(True)
    assert state.auto_save_on_close is True
    assert panel._state() != before


def test_default_app_settings_preserve_registration_step(preferences_panel_module):
    module, _state = preferences_panel_module
    calls = []
    module.lf.ui.register_file_associations = lambda: calls.append("register")
    module.lf.ui.open_file_association_settings = lambda: calls.append("open")
    module.PreferencesPanel._open_default_app_settings()
    assert calls == ["register", "open"]


def test_marketplace_preferences_apply_and_reset_independently(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._set_marketplace_preference("view", "card")
    panel._set_marketplace_preference("git_checkout", True)
    assert state.marketplace_preferences == {"view": "card", "git_checkout": True}
    assert len(state.marketplace_notifications) == 2
    panel._reset_section("marketplace")
    assert state.marketplace_preferences == {"view": "list", "git_checkout": False}
    assert len(state.marketplace_notifications) == 3
    assert state.set_language_calls == []


@pytest.mark.parametrize("git_available", [False, True])
def test_marketplace_git_checkout_requires_git_without_changing_saved_choice(preferences_panel_module, monkeypatch, git_available):
    module, state = preferences_panel_module
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/git" if git_available else None)
    panel = module.PreferencesPanel()
    assert panel._marketplace_git_available is git_available
    panel._set_marketplace_preference("git_checkout", True)
    assert state.marketplace_preferences["git_checkout"] is git_available
    state.marketplace_preferences["git_checkout"] = True
    reopened = module.PreferencesPanel()
    assert reopened._marketplace_git_available is git_available
    assert state.marketplace_preferences["git_checkout"] is True
    reopened._set_marketplace_preference("git_checkout", False)
    assert state.marketplace_preferences["git_checkout"] is False


def test_splat_drag_preferences_convert_ui_values_and_respect_safe_mode(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._set_splat_drag_preference("sh_degree", "2")
    assert state.splat_drag_preferences["sh_degree"] == 2
    assert panel._get_splat_drag_preference("sh_degree") == "2"
    panel._mcp_safe_mode = True
    panel._set_splat_drag_preference("sh_degree", "1")
    assert state.splat_drag_preferences["sh_degree"] == 2


@pytest.mark.parametrize("axis", [-2, 2, -3, 3])
def test_drag_up_axis_accepts_the_four_maintainer_options(preferences_panel_module, axis):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._set_splat_drag_preference("up_axis", str(axis))
    assert state.splat_drag_preferences["up_axis"] == axis
    assert panel._get_splat_drag_preference("up_axis") == str(axis)


@pytest.mark.parametrize("axis", ["0", "1", "-1", "4", "not-an-axis", None])
def test_drag_up_axis_rejects_unsupported_values(preferences_panel_module, axis):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._set_splat_drag_preference("up_axis", axis)
    assert state.splat_drag_preferences["up_axis"] == -2


def test_drag_up_axis_respects_safe_mode(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._mcp_safe_mode = True
    panel._set_splat_drag_preference("up_axis", "3")
    assert state.splat_drag_preferences["up_axis"] == -2


@pytest.mark.parametrize("key,value", [("format", "spz"), ("sh_degree", 1), ("up_axis", 3)])
def test_external_drag_preference_changes_refresh_open_panel(preferences_panel_module, key, value):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._last_state = panel._state()
    state.splat_drag_preferences[key] = value
    assert panel._state() != panel._last_state
    dirty = []
    panel._handle = SimpleNamespace(dirty=dirty.append)
    panel._dirty_selection()
    assert {"splat_drag_format", "splat_drag_sh_degree", "splat_drag_up_axis"} <= set(dirty)


@pytest.mark.parametrize("safe_mode", [False, True])
def test_interface_reset_refreshes_drag_controls_and_respects_safe_mode(preferences_panel_module, safe_mode):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._mcp_safe_mode = safe_mode
    customized = {"format": "spz", "sh_degree": 1, "up_axis": 3}
    state.splat_drag_preferences.update(customized)
    module.lf.ui.set_scene_graph_selection_markers = lambda _enabled: None
    events = []
    panel._refresh_selection = lambda: events.append("refresh")
    module.lf.ui.reset_layout = lambda: events.append("layout")
    panel._reset_section("interface")
    expected = customized if safe_mode else {"format": "ply", "sh_degree": 3, "up_axis": -2}
    assert state.splat_drag_preferences == expected
    assert events == ["refresh", "layout"]


def test_quick_export_offers_only_ply_and_spz_with_plain_translated_labels():
    from xml.etree import ElementTree
    root = Path(__file__).resolve().parents[2]
    document = ElementTree.parse(root / "src/visualizer/gui/rmlui/resources/preferences.rml")
    select = next(element for element in document.iter("select")
                  if element.get("data-value") == "splat_drag_format")
    assert [option.get("value") for option in select] == ["ply", "spz"]
    for path in (root / "src/visualizer/gui/resources/locales").glob("*.json"):
        preferences = json.loads(path.read_text(encoding="utf-8"))
        assert preferences["preferences.drag_ply"] == "PLY", path.name
        assert preferences["preferences.drag_spz"] == "SPZ", path.name


@pytest.mark.parametrize("value", ["-1", "4", "invalid", None])
def test_drag_sh_limit_rejects_invalid_ui_values(preferences_panel_module, value):
    module, state = preferences_panel_module
    module.PreferencesPanel()._set_splat_drag_preference("sh_degree", value)
    assert state.splat_drag_preferences["sh_degree"] == 3


def test_drag_settings_use_one_sh_cap_and_localized_tooltips():
    from xml.etree import ElementTree
    root = Path(__file__).resolve().parents[2]
    document = ElementTree.parse(root / "src/visualizer/gui/rmlui/resources/preferences.rml")
    cap = next(element for element in document.iter("select")
               if element.get("data-value") == "splat_drag_sh_degree")
    assert [option.get("value") for option in cap] == ["0", "1", "2", "3"]
    text = (root / "src/visualizer/gui/rmlui/resources/preferences.rml").read_text(encoding="utf-8")
    assert "splat_drag_maximum_quality" not in text
    assert "splat_drag_ply_sh" not in text and "splat_drag_spz_sh" not in text
    tooltip_keys = {"preferences.drag_format_hint", "preferences.drag_sh_hint", "preferences.drag_axes_hint"}
    assert tooltip_keys <= {element.get("data-tooltip") for element in document.iter()}
    inline = " ".join("".join(element.itertext()) for element in document.iter("span")
                      if "preferences-description" in element.get("class", ""))
    assert not any(key in inline for key in tooltip_keys)
    marker_hint = "preferences.scene_graph_selection_markers_description"
    marker_row = next(element for element in document.iter("div")
                      if element.get("data-tooltip") == marker_hint)
    marker = next(element for element in marker_row.iter("input")
                  if element.get("data-checked") == "scene_graph_selection_markers")
    assert marker.get("data-attrif-disabled") == "mcp_safe_mode"
    assert marker_hint not in inline
    axes = next(element for element in document.iter()
                if element.get("data-tooltip") == "preferences.drag_axes_hint")
    assert axes.get("data-if") == "splat_drag_format == 'ply'"
    selector = next(element for element in axes.iter("select"))
    assert selector.get("data-value") == "splat_drag_up_axis"
    assert [option.get("value") for option in selector] == ["-2", "2", "-3", "3"]
    assert selector[0].text == "@tr:preferences.drag_axis_default"
    assert selector.get("data-attrif-disabled") == "mcp_safe_mode"
    for path in (root / "src/visualizer/gui/resources/locales").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert all(data.get(key) for key in tooltip_keys), path.name
        assert not any("SOG" in data[key].upper() for key in tooltip_keys), path.name
        assert "-Y" in data["preferences.drag_axis_default"], path.name
        assert data["preferences"]["scene_graph_selection_markers_description"].strip(), path.name


def test_camera_repeat_numeric_preference_and_step_buttons(preferences_panel_module, monkeypatch):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._refresh_selection = lambda: None
    writes = []

    def set_rate(value, persist=True):
        state.camera_browse_repeat_rate = float(value)
        writes.append((float(value), persist))

    monkeypatch.setattr(module.lf.ui, "set_camera_browse_repeat_rate", set_rate)
    panel._set_camera_browse_repeat(False)
    assert not state.camera_browse_repeat
    panel._set_camera_browse_repeat_rate("12")
    assert state.camera_browse_repeat_rate == 5
    assert writes == []
    assert panel._get_camera_repeat_rate_draft() == "12"
    assert panel._commit_camera_repeat_rate()
    assert state.camera_browse_repeat_rate == 12
    assert writes == [(12, True)]
    for invalid in ("", "abc", "nan", "inf", "0", "21"):
        panel._set_camera_browse_repeat_rate(invalid)
        assert panel._get_camera_repeat_rate_draft() == invalid
        assert not panel._commit_camera_repeat_rate()
        assert panel._get_camera_repeat_rate_draft() == "12"
    assert writes == [(12, True)]
    panel._step_camera_repeat_rate(None, None, [-1])
    assert state.camera_browse_repeat_rate == 11
    panel._step_camera_repeat_rate(None, None, [1])
    assert state.camera_browse_repeat_rate == 12
    panel._set_camera_browse_repeat_rate("20")
    panel._step_camera_repeat_rate(None, None, [1])
    assert state.camera_browse_repeat_rate == 20
    panel._set_camera_browse_repeat_rate("1")
    panel._step_camera_repeat_rate(None, None, [-1])
    assert state.camera_browse_repeat_rate == 1
    panel._reset_section("input")
    assert state.camera_browse_repeat
    assert state.camera_browse_repeat_rate == 5.0


def test_camera_repeat_draft_survives_refresh_and_commits_on_enter(preferences_panel_module):
    module, state = preferences_panel_module
    panel = module.PreferencesPanel()
    panel._refresh_selection = lambda: None
    panel._set_camera_browse_repeat_rate("")
    assert panel._get_camera_repeat_rate_draft() == ""
    assert state.camera_browse_repeat_rate == 5
    on_change = module.draft_change_handler(panel._set_camera_browse_repeat_rate, panel._commit_camera_repeat_rate)
    event = SimpleNamespace(get_bool_parameter=lambda key, default: False)
    on_change(None, event, ["1"])
    assert panel._get_camera_repeat_rate_draft() == "1"
    assert state.camera_browse_repeat_rate == 5
    event.get_bool_parameter = lambda key, default: True
    on_change(None, event, ["1"])
    assert state.camera_browse_repeat_rate == 1
    panel._set_camera_browse_repeat_rate("12")
    panel._cancel_camera_repeat_rate()
    assert panel._get_camera_repeat_rate_draft() == "1"
    assert state.camera_browse_repeat_rate == 1


def test_draft_change_handler_keeps_draft_until_enter(preferences_panel_module):
    module, _state = preferences_panel_module
    calls = []
    on_change = module.draft_change_handler(lambda value: calls.append(value), lambda: calls.append("commit"))
    typing = SimpleNamespace(get_bool_parameter=lambda key, default: False)
    enter = SimpleNamespace(get_bool_parameter=lambda key, default: key == "linebreak")
    on_change(None, typing, ["4"])
    on_change(None, typing, [])
    on_change(None, enter, ["45"])
    assert calls == ["4", "45", "commit"]
