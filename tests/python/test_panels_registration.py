# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for per-step isolation in register_builtin_panels()."""

from enum import IntEnum, IntFlag
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_recording_lf(monkeypatch):
    """Install a recording ``lichtfeld`` stub sufficient for builtin panel imports."""
    registered = []
    enabled = []
    step_side_effects = []

    panel_space = SimpleNamespace(
        SIDE_PANEL="SIDE_PANEL",
        FLOATING="FLOATING",
        VIEWPORT_OVERLAY="VIEWPORT_OVERLAY",
        MAIN_PANEL_TAB="MAIN_PANEL_TAB",
        SCENE_HEADER="SCENE_HEADER",
        STATUS_BAR="STATUS_BAR",
        BOTTOM_DOCK="BOTTOM_DOCK",
        LEFT_DOCK="LEFT_DOCK",
    )
    panel_height_mode = SimpleNamespace(FILL="fill", CONTENT="content")
    panel_option = SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED", HIDE_HEADER="HIDE_HEADER")

    class Panel:
        pass

    class Operator:
        @classmethod
        def _class_id(cls):
            return getattr(cls, "bl_idname", cls.__name__)

    class WindowFlags(IntFlag):
        NoTitleBar = 1
        NoResize = 2
        NoMove = 4
        NoScrollbar = 8
        NoInputs = 16
        NoBackground = 32
        NoFocusOnAppearing = 64
        NoBringToFrontOnFocus = 128

    ui_layout = SimpleNamespace(WindowFlags=WindowFlags)

    tool_mode_names = (
        "GLOBAL",
        "SELECTION",
        "TRANSLATE",
        "ROTATE",
        "SCALE",
        "ALIGN",
        "CROP_BOX",
    )
    action_names = (
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
    )
    tool_mode = IntEnum("ToolMode", {name: i for i, name in enumerate(tool_mode_names)})
    action_enum = IntEnum("Action", {name: i for i, name in enumerate(action_names)})

    class KeymapAction:
        def __getattr__(self, name):
            members = action_enum.__members__
            if name in members:
                return members[name]
            return SimpleNamespace(name=name, value=abs(hash(name)) % 100000)

    def register_class(cls):
        registered.append(getattr(cls, "__name__", str(cls)))

    def set_panel_enabled(panel_id, enabled_flag):
        enabled.append((panel_id, enabled_flag))

    lf_stub = ModuleType("lichtfeld")
    lf_stub.register_class = register_class
    lf_stub.unregister_class = lambda cls: None
    lf_stub.ui = SimpleNamespace(
        Panel=Panel,
        Operator=Operator,
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        UILayout=ui_layout,
        tr=lambda key: key,
        theme=lambda: SimpleNamespace(),
        set_panel_enabled=set_panel_enabled,
        set_panel_parent=lambda *args, **kwargs: None,
        set_panel_label=lambda *args, **kwargs: True,
        get_current_language=lambda: "en",
        request_redraw=lambda *args, **kwargs: None,
        is_windows_platform=lambda: False,
        toggle_system_console=lambda: None,
        add_hook=lambda *args, **kwargs: step_side_effects.append(("hook", args)),
        remove_hook=lambda *args, **kwargs: None,
        on_open_camera_preview=lambda cb: step_side_effects.append(("camera_preview", cb)),
        on_show_dataset_load_popup=lambda cb: None,
        on_show_resume_checkpoint_popup=lambda cb: None,
        on_request_exit=lambda cb: None,
        set_cancel_operator_callback=lambda cb: None,
        set_save_asset_callback=lambda cb: None,
        ops=SimpleNamespace(cancel_modal=lambda: None),
        execute_operator=lambda *args, **kwargs: None,
        get_content_type=lambda: "empty",
        add_keyframe=lambda: None,
        update_keyframe=lambda: None,
        play_pause=lambda: None,
        confirm_dialog=lambda *args, **kwargs: None,
        open_dataset_folder_dialog=lambda: None,
        open_ply_file_dialog=lambda *args: None,
        open_mesh_file_dialog=lambda *args: None,
        open_checkpoint_file_dialog=lambda: None,
        open_json_file_dialog=lambda: None,
        save_json_file_dialog=lambda *args: None,
        set_exit_popup_open=lambda *args: None,
        is_sequencer_visible=lambda: False,
        get_sequencer_state=lambda: None,
        get_import_state=lambda: {},
        get_video_export_state=lambda: {},
        dismiss_import=lambda: None,
        cancel_video_export=lambda: None,
        is_scene_empty=lambda: True,
        is_drag_hovering=lambda: False,
        is_startup_visible=lambda: False,
        get_time=lambda: 0.0,
        rml=SimpleNamespace(get_document=lambda *_a, **_k: None),
    )
    lf_stub.keymap = SimpleNamespace(
        ToolMode=tool_mode,
        Action=KeymapAction(),
        is_bound=lambda *args, **kwargs: False,
        get_trigger_description=lambda *args, **kwargs: "",
        get_available_profiles=lambda: ["Default"],
        get_current_profile=lambda: "Default",
        get_tool_mode_name=lambda mode: str(mode),
        get_action_name=lambda value: str(value),
        get_trigger=lambda *args, **kwargs: None,
        set_trigger_binding=lambda *args, **kwargs: True,
        find_conflict_for_action=lambda *args, **kwargs: None,
        get_allowed_trigger_kinds=lambda *args, **kwargs: ["key"],
        is_capturing=lambda: False,
        is_waiting_for_double_click=lambda: False,
        load_profile=lambda name: None,
        save_profile=lambda name: None,
        reset_to_default=lambda: None,
        export_profile=lambda path: None,
        import_profile=lambda path: None,
        start_capture=lambda *args: None,
        cancel_capture=lambda: None,
        clear_binding=lambda *args: None,
        get_captured_trigger=lambda: None,
    )
    lf_stub.undo = SimpleNamespace(
        undo=lambda: None,
        redo=lambda: None,
        can_undo=lambda: False,
        can_redo=lambda: False,
    )
    lf_stub.scene = SimpleNamespace(
        NodeType=IntEnum("NodeType", {"MESH": 10, "SPLAT": 20})
    )
    lf_stub.get_scene = lambda: None
    lf_stub.get_render_settings = lambda: SimpleNamespace(
        focal_length_mm=50.0, prop_info=lambda prop_id: {"name": prop_id}
    )
    lf_stub.get_current_view = lambda: SimpleNamespace(width=1920, height=1080)
    lf_stub.get_selected_node_name = lambda: ""
    lf_stub.get_vulkan_capabilities = lambda: {}
    lf_stub.log = SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warn=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    lf_stub.has_scene = lambda: False

    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)

    # Drop previously imported plugin modules so they rebind against the stub.
    for name in list(sys.modules):
        if name == "lfs_plugins" or name.startswith("lfs_plugins."):
            del sys.modules[name]

    state = SimpleNamespace(
        registered=registered,
        enabled=enabled,
        side_effects=step_side_effects,
        lf=lf_stub,
    )
    return state


@pytest.fixture
def panels_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    state = _install_recording_lf(monkeypatch)
    module = import_module("lfs_plugins.panels")
    return module, state


def test_isolation_mid_step_failure_still_registers_later_panels(panels_module, monkeypatch):
    """A mid-list step failure must not abort later steps; still return True."""
    module, state = panels_module

    original_build = module._build_builtin_panel_steps

    def build_with_broken_mid(lf):
        steps = original_build(lf)
        names = [name for name, _ in steps]
        broken_name = "input_settings_panel"
        assert broken_name in names, f"expected {broken_name} in {names}"
        broken_index = names.index(broken_name)

        def boom():
            raise AttributeError("simulated input_settings_panel failure")

        steps[broken_index] = (broken_name, boom)
        return steps

    monkeypatch.setattr(module, "_build_builtin_panel_steps", build_with_broken_mid)

    result = module.register_builtin_panels()
    assert result is True

    # Steps after the broken one still ran.
    assert "Mesh2SplatPanel" in state.registered
    assert "PluginMarketplacePanel" in state.registered
    assert "AssetManagerPanel" in state.registered
    assert any(effect[0] == "hook" for effect in state.side_effects)

    # Broken step did not register InputSettingsPanel.
    assert "InputSettingsPanel" not in state.registered

    # Earlier step still registered.
    assert "RenderingPanel" in state.registered


def test_all_good_registers_in_order_with_rendering_first(panels_module):
    """Happy path: all real steps succeed, RenderingPanel is first register_class."""
    module, state = panels_module

    result = module.register_builtin_panels()
    assert result is True

    assert state.registered, "expected at least one register_class call"
    assert state.registered[0] == "RenderingPanel"

    expected_panels = [
        "RenderingPanel",
        "TrainingPanel",
        "DatasetImportPanel",
        "ResumeCheckpointPanel",
        "URLImportPanel",
        "ExportPanel",
        "AboutPanel",
        "AccountPanel",
        "BugReportPanel",
        "GettingStartedPanel",
        "ImagePreviewPanel",
        "HistogramPanel",
        "ScriptsPanel",
        "InputSettingsPanel",
        "Mesh2SplatPanel",
        "PluginMarketplacePanel",
        "AssetManagerPanel",
    ]
    for panel_name in expected_panels:
        assert panel_name in state.registered, f"missing register_class for {panel_name}"

    # Overlays step should have registered its hook.
    assert any(effect[0] == "hook" for effect in state.side_effects)

    # Image preview callback hookup is part of image_preview_panel step.
    assert any(effect[0] == "camera_preview" for effect in state.side_effects)


def test_lichtfeld_import_failure_returns_false_and_skips_steps(monkeypatch):
    """If import lichtfeld fails, return False and do not run any steps."""
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    # Ensure a clean panels import without pulling marketplace eagerly.
    for name in list(sys.modules):
        if name == "lfs_plugins" or name.startswith("lfs_plugins."):
            del sys.modules[name]

    # No usable lichtfeld module.
    monkeypatch.setitem(sys.modules, "lichtfeld", None)

    # Import panels while lichtfeld is broken: panels itself must still import.
    module = import_module("lfs_plugins.panels")

    step_calls = []

    def boom_build(_lf):
        step_calls.append("built")
        return [("never", lambda: step_calls.append("ran"))]

    monkeypatch.setattr(module, "_build_builtin_panel_steps", boom_build)

    # Force import lichtfeld inside register_builtin_panels to fail.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lichtfeld":
            raise ModuleNotFoundError("No module named 'lichtfeld'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = module.register_builtin_panels()
    assert result is False
    assert step_calls == []


def test_step_runner_isolation_with_synthetic_steps(monkeypatch):
    """Step-runner isolation when the step list itself is fully synthetic."""
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    state = _install_recording_lf(monkeypatch)
    module = import_module("lfs_plugins.panels")

    executed = []

    def make_step(name, *, fail=False):
        def _step():
            if fail:
                raise RuntimeError(f"{name} boom")
            executed.append(name)
            # Mimic a panel registration so assertions can use state.registered too.
            state.lf.register_class(type(name, (), {}))

        return _step

    def synthetic_build(lf):
        return [
            ("first", make_step("first")),
            ("broken", make_step("broken", fail=True)),
            ("after_broken", make_step("after_broken")),
            ("last", make_step("last")),
        ]

    monkeypatch.setattr(module, "_build_builtin_panel_steps", synthetic_build)

    result = module.register_builtin_panels()
    assert result is True
    assert executed == ["first", "after_broken", "last"]
    assert "after_broken" in state.registered
    assert "last" in state.registered
    assert "broken" not in state.registered
