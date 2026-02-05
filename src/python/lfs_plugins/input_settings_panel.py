# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Input settings panel for keyboard/mouse binding configuration."""

import lichtfeld as lf
from .types import Panel


class InputSettingsPanel(Panel):
    """Input Settings panel - floating window for configuring keybindings."""

    idname = "lfs.input_settings"
    label = "Input Settings"
    space = "FLOATING"
    order = 100
    options = {"DEFAULT_CLOSED"}

    TOOL_MODES = [
        lf.keymap.ToolMode.GLOBAL,
        lf.keymap.ToolMode.SELECTION,
        lf.keymap.ToolMode.BRUSH,
        lf.keymap.ToolMode.ALIGN,
        lf.keymap.ToolMode.CROP_BOX,
    ]

    BINDING_SECTIONS = {
        "navigation": [
            lf.keymap.Action.CAMERA_ORBIT,
            lf.keymap.Action.CAMERA_PAN,
            lf.keymap.Action.CAMERA_ZOOM,
            lf.keymap.Action.CAMERA_SET_PIVOT,
            lf.keymap.Action.CAMERA_MOVE_FORWARD,
            lf.keymap.Action.CAMERA_MOVE_BACKWARD,
            lf.keymap.Action.CAMERA_MOVE_LEFT,
            lf.keymap.Action.CAMERA_MOVE_RIGHT,
            lf.keymap.Action.CAMERA_MOVE_UP,
            lf.keymap.Action.CAMERA_MOVE_DOWN,
            lf.keymap.Action.CAMERA_SPEED_UP,
            lf.keymap.Action.CAMERA_SPEED_DOWN,
            lf.keymap.Action.ZOOM_SPEED_UP,
            lf.keymap.Action.ZOOM_SPEED_DOWN,
        ],
        "navigation_global": [
            lf.keymap.Action.CAMERA_RESET_HOME,
            lf.keymap.Action.CAMERA_NEXT_VIEW,
            lf.keymap.Action.CAMERA_PREV_VIEW,
        ],
        "selection": [
            lf.keymap.Action.SELECTION_REPLACE,
            lf.keymap.Action.SELECTION_ADD,
            lf.keymap.Action.SELECTION_REMOVE,
        ],
        "selection_global": [
            lf.keymap.Action.SELECT_MODE_CENTERS,
            lf.keymap.Action.SELECT_MODE_RECTANGLE,
            lf.keymap.Action.SELECT_MODE_POLYGON,
            lf.keymap.Action.SELECT_MODE_LASSO,
            lf.keymap.Action.SELECT_MODE_RINGS,
        ],
        "depth": [
            lf.keymap.Action.TOGGLE_DEPTH_MODE,
            lf.keymap.Action.DEPTH_ADJUST_FAR,
            lf.keymap.Action.DEPTH_ADJUST_SIDE,
        ],
        "brush": [
            lf.keymap.Action.CYCLE_BRUSH_MODE,
            lf.keymap.Action.BRUSH_RESIZE,
        ],
        "crop_box": [
            lf.keymap.Action.APPLY_CROP_BOX,
        ],
        "editing": [
            lf.keymap.Action.UNDO,
            lf.keymap.Action.REDO,
            lf.keymap.Action.COPY_SELECTION,
            lf.keymap.Action.PASTE_SELECTION,
            lf.keymap.Action.INVERT_SELECTION,
            lf.keymap.Action.DESELECT_ALL,
        ],
        "view_global": [
            lf.keymap.Action.TOGGLE_SPLIT_VIEW,
            lf.keymap.Action.TOGGLE_GT_COMPARISON,
            lf.keymap.Action.CYCLE_PLY,
            lf.keymap.Action.CYCLE_SELECTION_VIS,
        ],
    }

    def __init__(self):
        self._selected_mode = lf.keymap.ToolMode.GLOBAL
        self._rebinding_action = None
        self._rebinding_mode = None

    def draw(self, layout):
        layout.text_colored("INPUT SETTINGS", (0.4, 0.6, 1.0, 1.0))
        layout.spacing()
        layout.separator()
        layout.spacing()

        profiles = lf.keymap.get_available_profiles()
        current = lf.keymap.get_current_profile()
        current_idx = profiles.index(current) if current in profiles else 0
        is_rebinding = lf.keymap.is_capturing()

        if is_rebinding:
            layout.begin_disabled()

        layout.label("Active Profile:")
        layout.same_line()
        changed, new_idx = layout.combo("##profile", current_idx, profiles)
        if changed and new_idx != current_idx:
            lf.keymap.load_profile(profiles[new_idx])

        if is_rebinding:
            layout.end_disabled()

        layout.spacing()
        layout.spacing()

        layout.text_colored("Tool Mode", (0.6, 0.6, 0.6, 1.0))
        layout.text_disabled("Select the tool mode to configure bindings for")
        layout.spacing()

        mode_names = [lf.keymap.get_tool_mode_name(m) for m in self.TOOL_MODES]
        current_mode_idx = self.TOOL_MODES.index(self._selected_mode) if self._selected_mode in self.TOOL_MODES else 0

        if is_rebinding:
            layout.begin_disabled()

        changed, new_mode_idx = layout.combo("##toolmode", current_mode_idx, mode_names)
        if changed and new_mode_idx != current_mode_idx:
            self._selected_mode = self.TOOL_MODES[new_mode_idx]

        if is_rebinding:
            layout.end_disabled()

        layout.spacing()
        layout.spacing()

        layout.text_colored("Current Bindings", (0.6, 0.6, 0.6, 1.0))
        if self._selected_mode == lf.keymap.ToolMode.GLOBAL:
            layout.text_disabled("Global bindings apply when no tool is active")
        else:
            layout.text_disabled("Tool-specific bindings override global bindings")
        layout.spacing()

        mode = self._selected_mode

        if layout.begin_table("bindings_table", 3):
            layout.table_setup_column("Action", 180)
            layout.table_setup_column("Binding", -1)
            layout.table_setup_column("", 70)
            layout.table_headers_row()

            self._draw_section_header(layout, "Navigation")
            for action in self.BINDING_SECTIONS["navigation"]:
                self._draw_binding_row(layout, action, mode)

            if mode == lf.keymap.ToolMode.GLOBAL:
                for action in self.BINDING_SECTIONS["navigation_global"]:
                    self._draw_binding_row(layout, action, mode)

            if mode in (lf.keymap.ToolMode.GLOBAL, lf.keymap.ToolMode.SELECTION, lf.keymap.ToolMode.BRUSH):
                self._draw_section_header(layout, "Selection")
                for action in self.BINDING_SECTIONS["selection"]:
                    self._draw_binding_row(layout, action, mode)

                if mode == lf.keymap.ToolMode.GLOBAL:
                    for action in self.BINDING_SECTIONS["selection_global"]:
                        self._draw_binding_row(layout, action, mode)

                if mode in (lf.keymap.ToolMode.GLOBAL, lf.keymap.ToolMode.SELECTION):
                    for action in self.BINDING_SECTIONS["depth"]:
                        self._draw_binding_row(layout, action, mode)

            if mode == lf.keymap.ToolMode.BRUSH:
                self._draw_section_header(layout, "Brush")
                for action in self.BINDING_SECTIONS["brush"]:
                    self._draw_binding_row(layout, action, mode)

            if mode == lf.keymap.ToolMode.CROP_BOX:
                self._draw_section_header(layout, "Crop Box")
                for action in self.BINDING_SECTIONS["crop_box"]:
                    self._draw_binding_row(layout, action, mode)

            self._draw_section_header(layout, "Editing")
            if mode in (lf.keymap.ToolMode.GLOBAL, lf.keymap.ToolMode.TRANSLATE,
                        lf.keymap.ToolMode.ROTATE, lf.keymap.ToolMode.SCALE):
                self._draw_binding_row(layout, lf.keymap.Action.DELETE_NODE, mode)
            else:
                self._draw_binding_row(layout, lf.keymap.Action.DELETE_SELECTED, mode)

            for action in self.BINDING_SECTIONS["editing"]:
                self._draw_binding_row(layout, action, mode)

            if mode == lf.keymap.ToolMode.GLOBAL:
                self._draw_section_header(layout, "View")
                for action in self.BINDING_SECTIONS["view_global"]:
                    self._draw_binding_row(layout, action, mode)

            layout.end_table()

        layout.spacing()
        layout.separator()
        layout.spacing()

        layout.push_style_color("button", (0.2, 0.6, 0.2, 0.7))
        if layout.button("Save Profile"):
            lf.keymap.save_profile(lf.keymap.get_current_profile())
        layout.pop_style_color()

        layout.same_line()

        layout.push_style_color("button", (0.6, 0.2, 0.2, 0.7))
        if layout.button("Reset to Default"):
            lf.keymap.reset_to_default()
        layout.pop_style_color()

        layout.spacing()

        layout.push_style_color("button", (0.3, 0.3, 0.5, 0.7))
        if layout.button("Export"):
            path = lf.ui.save_file_dialog("Export Input Bindings", "json")
            if path:
                lf.keymap.export_profile(path)
        layout.pop_style_color()

        layout.same_line()

        layout.push_style_color("button", (0.3, 0.3, 0.5, 0.7))
        if layout.button("Import"):
            path = lf.ui.open_file_dialog("Import Input Bindings", "json")
            if path:
                lf.keymap.import_profile(path)
        layout.pop_style_color()

        layout.spacing()
        layout.text_disabled("Save to persist custom bindings")
        layout.text_disabled("Tip: Double-click to bind double-click action")

    def _draw_section_header(self, layout, title):
        layout.table_next_row()
        layout.table_next_column()
        layout.table_set_bg_color(0, (0.2, 0.3, 0.5, 0.2))
        layout.text_colored(title, (0.5, 0.7, 1.0, 1.0))
        layout.table_next_column()
        layout.table_next_column()

    def _draw_binding_row(self, layout, action, mode):
        is_rebinding = (lf.keymap.is_capturing() and
                        self._rebinding_action == action and
                        self._rebinding_mode == mode)

        layout.table_next_row()
        layout.table_next_column()
        layout.label(lf.keymap.get_action_name(action))

        layout.table_next_column()
        if is_rebinding:
            if lf.keymap.is_waiting_for_double_click():
                layout.text_colored("Click again for double-click...", (1.0, 0.8, 0.3, 1.0))
            else:
                layout.text_colored("Press key or click...", (1.0, 0.8, 0.3, 1.0))
        else:
            desc = lf.keymap.get_trigger_description(action, mode)
            layout.text_colored(desc, (0.4, 0.6, 1.0, 1.0))

        layout.table_next_column()
        unique_id = action.value * 100 + mode.value

        if is_rebinding:
            layout.push_style_color("button", (0.6, 0.2, 0.2, 0.8))
            if layout.button(f"Cancel##{unique_id}"):
                lf.keymap.cancel_capture()
                self._rebinding_action = None
                self._rebinding_mode = None
            layout.pop_style_color()
        else:
            layout.push_style_color("button", (0.2, 0.4, 0.6, 0.8))
            if layout.button(f"Rebind##{unique_id}"):
                self._rebinding_action = action
                self._rebinding_mode = mode
                lf.keymap.start_capture(mode, action)
            layout.pop_style_color()

        if is_rebinding:
            trigger = lf.keymap.get_captured_trigger()
            if trigger is not None:
                self._rebinding_action = None
                self._rebinding_mode = None


