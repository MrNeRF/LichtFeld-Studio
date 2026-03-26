# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Input settings panel for keyboard/mouse binding configuration."""

import lichtfeld as lf
from .types import Panel


class InputSettingsPanel(Panel):
    id = "lfs.input_settings"
    label = "Input Settings"
    space = lf.ui.PanelSpace.FLOATING
    order = 100
    template = "rmlui/input_settings.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (500, 0)
    options = {lf.ui.PanelOption.DEFAULT_CLOSED}
    update_interval_ms = 50

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
            lf.keymap.Action.TOGGLE_SELECTION_DEPTH_FILTER,
            lf.keymap.Action.TOGGLE_SELECTION_CROP_FILTER,
            lf.keymap.Action.DEPTH_ADJUST_FAR,
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
        self._selected_mode_idx = 0
        self._rebinding_action = None
        self._rebinding_mode = None
        self._handle = None
        self._last_profiles = []
        self._last_state_key = None
        self._last_lang = ""
        self._last_current_profile = ""
        self._last_display_h = 0
        self._last_capturing = None

    # ── Data model ────────────────────────────────────────────

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("input_settings")
        if model is None:
            return

        model.bind_func("panel_label", lambda: lf.ui.tr("input_settings.title"))
        model.bind_func("bindings_hint", self._get_bindings_hint)
        model.bind_func("is_capturing", lambda: lf.keymap.is_capturing())

        model.bind(
            "profile_idx",
            lambda: str(self._get_profile_idx()),
            self._set_profile_idx,
        )
        model.bind(
            "mode_idx",
            lambda: str(self._selected_mode_idx),
            self._set_mode_idx,
        )

        model.bind_event("save_profile", self._on_save_profile)
        model.bind_event("reset_default", self._on_reset_default)
        model.bind_event("export_profile", self._on_export_profile)
        model.bind_event("import_profile", self._on_import_profile)
        model.bind_record_list("profiles")
        model.bind_record_list("tool_modes")
        model.bind_record_list("binding_rows")

        self._handle = model.get_handle()

    def _get_profile_idx(self):
        profiles = lf.keymap.get_available_profiles()
        current = lf.keymap.get_current_profile()
        return profiles.index(current) if current in profiles else 0

    def _set_profile_idx(self, v):
        try:
            idx = int(v)
        except (ValueError, TypeError):
            return
        profiles = lf.keymap.get_available_profiles()
        if 0 <= idx < len(profiles):
            lf.keymap.load_profile(profiles[idx])
            self._last_state_key = None

    def _set_mode_idx(self, v):
        try:
            idx = int(v)
        except (ValueError, TypeError):
            return
        if 0 <= idx < len(self.TOOL_MODES) and idx != self._selected_mode_idx:
            self._selected_mode_idx = idx
            self._last_state_key = None

    # ── Events ────────────────────────────────────────────────

    def _on_save_profile(self, _handle, _ev, _args):
        lf.keymap.save_profile(lf.keymap.get_current_profile())

    def _on_reset_default(self, _handle, _ev, _args):
        lf.keymap.reset_to_default()
        self._last_state_key = None

    def _on_export_profile(self, _handle, _ev, _args):
        tr = lf.ui.tr
        path = lf.ui.save_file_dialog(tr("input_settings.export_dialog_title"), "json")
        if path:
            lf.keymap.export_profile(path)

    def _on_import_profile(self, _handle, _ev, _args):
        tr = lf.ui.tr
        path = lf.ui.open_file_dialog(tr("input_settings.import_dialog_title"), "json")
        if path:
            lf.keymap.import_profile(path)
            self._last_state_key = None

    # ── Lifecycle ─────────────────────────────────────────────

    def on_mount(self, doc):
        super().on_mount(doc)
        self._last_lang = lf.ui.get_current_language()
        self._last_current_profile = lf.keymap.get_current_profile()

        self._rebuild_profile_records()
        self._rebuild_mode_records()
        self._rebuild_binding_rows(self.TOOL_MODES[self._selected_mode_idx])

        table_el = doc.get_element_by_id("bindings-table")
        if table_el:
            table_el.add_event_listener("click", self._on_table_click)

    def on_update(self, doc):
        self._update_max_height(doc)

        current_lang = lf.ui.get_current_language()
        profiles = lf.keymap.get_available_profiles()
        if profiles != self._last_profiles:
            self._last_profiles = list(profiles)
            self._rebuild_profile_records()
            self._dirty_model("profile_idx")

        is_capturing = lf.keymap.is_capturing()
        mode = self.TOOL_MODES[self._selected_mode_idx]

        if is_capturing and self._rebinding_action is not None:
            trigger = lf.keymap.get_captured_trigger()
            if trigger is not None:
                self._rebinding_action = None
                self._rebinding_mode = None

        current_profile = lf.keymap.get_current_profile()
        state_key = (
            self._selected_mode_idx,
            self._rebinding_action,
            is_capturing,
            current_profile,
            current_lang,
        )
        if state_key != self._last_state_key:
            self._last_state_key = state_key
            self._rebuild_binding_rows(mode)
            self._dirty_model("bindings_hint")

        if current_lang != self._last_lang:
            self._last_lang = current_lang
            self._rebuild_mode_records()
            self._rebuild_profile_records()
            self._dirty_model()

        if current_profile != self._last_current_profile:
            self._last_current_profile = current_profile
            self._dirty_model("profile_idx")

        if is_capturing != self._last_capturing:
            self._last_capturing = is_capturing
            self._dirty_model("is_capturing")

    def _update_max_height(self, doc):
        try:
            _, display_h = lf.ui.get_display_size()
        except (RuntimeError, AttributeError):
            return
        if display_h == self._last_display_h:
            return
        self._last_display_h = display_h
        max_h = int(display_h * 2 / 3)
        wrap = doc.get_element_by_id("content-wrap")
        if wrap:
            wrap.set_property("max-height", f"{max_h}dp")

    # ── Retained model updates ────────────────────────────────

    def _dirty_model(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            return
        for field in fields:
            self._handle.dirty(field)

    def _get_bindings_hint(self):
        tr = lf.ui.tr
        mode = self.TOOL_MODES[self._selected_mode_idx]
        if mode == lf.keymap.ToolMode.GLOBAL:
            return tr("input_settings.global_bindings_hint")
        return tr("input_settings.tool_bindings_hint")

    def _rebuild_profile_records(self):
        if not self._handle:
            return
        profiles = lf.keymap.get_available_profiles()
        self._last_profiles = list(profiles)
        self._handle.update_record_list(
            "profiles",
            [{"index": str(i), "label": name} for i, name in enumerate(profiles)],
        )

    def _rebuild_mode_records(self):
        if not self._handle:
            return
        self._handle.update_record_list(
            "tool_modes",
            [
                {"index": str(i), "label": lf.keymap.get_tool_mode_name(mode)}
                for i, mode in enumerate(self.TOOL_MODES)
            ],
        )

    def _append_binding_section(self, rows, title, actions, mode):
        rows.append({
            "is_section": True,
            "section_title": title,
        })
        for action in actions:
            rows.append(self._binding_row_record(action, mode))

    def _rebuild_binding_rows(self, mode):
        if not self._handle:
            return

        tr = lf.ui.tr
        rows = []

        self._append_binding_section(
            rows, tr("input_settings.section.navigation"),
            self.BINDING_SECTIONS["navigation"], mode)

        if mode == lf.keymap.ToolMode.GLOBAL:
            for action in self.BINDING_SECTIONS["navigation_global"]:
                rows.append(self._binding_row_record(action, mode))

        if mode in (lf.keymap.ToolMode.GLOBAL, lf.keymap.ToolMode.SELECTION, lf.keymap.ToolMode.BRUSH):
            self._append_binding_section(
                rows, tr("input_settings.section.selection"),
                self.BINDING_SECTIONS["selection"], mode)

            if mode == lf.keymap.ToolMode.GLOBAL:
                for action in self.BINDING_SECTIONS["selection_global"]:
                    rows.append(self._binding_row_record(action, mode))

            if mode == lf.keymap.ToolMode.SELECTION:
                for action in self.BINDING_SECTIONS["depth"]:
                    rows.append(self._binding_row_record(action, mode))

        if mode == lf.keymap.ToolMode.BRUSH:
            self._append_binding_section(
                rows, tr("input_settings.section.brush"),
                self.BINDING_SECTIONS["brush"], mode)

        if mode == lf.keymap.ToolMode.CROP_BOX:
            self._append_binding_section(
                rows, tr("input_settings.section.crop_box"),
                self.BINDING_SECTIONS["crop_box"], mode)

        rows.append({
            "is_section": True,
            "section_title": tr("input_settings.section.editing"),
        })
        if mode in (lf.keymap.ToolMode.GLOBAL, lf.keymap.ToolMode.TRANSLATE,
                    lf.keymap.ToolMode.ROTATE, lf.keymap.ToolMode.SCALE):
            rows.append(self._binding_row_record(lf.keymap.Action.DELETE_NODE, mode))
        else:
            rows.append(self._binding_row_record(lf.keymap.Action.DELETE_SELECTED, mode))

        for action in self.BINDING_SECTIONS["editing"]:
            rows.append(self._binding_row_record(action, mode))

        if mode == lf.keymap.ToolMode.GLOBAL:
            self._append_binding_section(
                rows, tr("input_settings.section.view"),
                self.BINDING_SECTIONS["view_global"], mode)

        self._handle.update_record_list("binding_rows", rows)

    def _binding_row_record(self, action, mode):
        tr = lf.ui.tr
        is_rebinding = (lf.keymap.is_capturing() and
                        self._rebinding_action == action and
                        self._rebinding_mode == mode)

        action_name = lf.keymap.get_action_name(action)
        action_val = str(action.value)
        mode_val = str(mode.value)

        if is_rebinding:
            if lf.keymap.is_waiting_for_double_click():
                desc_text = tr("input_settings.click_again_double")
            else:
                desc_text = tr("input_settings.press_key_or_click")
            desc_class = "is-binding-desc is-capturing"
            button_action = "cancel"
            button_label = tr("input_settings.cancel")
            button_class = "btn--error"
        else:
            desc_text = lf.keymap.get_trigger_description(action, mode)
            desc_class = "is-binding-desc"
            button_action = "rebind"
            button_label = tr("input_settings.rebind")
            button_class = "btn--primary"

        return {
            "is_section": False,
            "section_title": "",
            "action_name": action_name,
            "desc_text": desc_text,
            "desc_class": desc_class,
            "button_action": button_action,
            "button_label": button_label,
            "button_class": button_class,
            "action_id": action_val,
            "mode_id": mode_val,
        }

    # ── Event delegation ──────────────────────────────────────

    def _on_table_click(self, ev):
        target = ev.target()
        if target is None:
            return

        btn_action, action_id, mode_id = self._find_btn_action(target)
        if not btn_action:
            return

        try:
            action = lf.keymap.Action(action_id)
            mode = lf.keymap.ToolMode(mode_id)
        except (ValueError, KeyError):
            return

        if btn_action == "rebind":
            self._rebinding_action = action
            self._rebinding_mode = mode
            lf.keymap.start_capture(mode, action)
            self._last_state_key = None
        elif btn_action == "cancel":
            lf.keymap.cancel_capture()
            self._rebinding_action = None
            self._rebinding_mode = None
            self._last_state_key = None

    def _find_btn_action(self, element):
        while element is not None:
            action = element.get_attribute("data-btn-action")
            if action:
                aid_str = element.get_attribute("data-action-id")
                mid_str = element.get_attribute("data-mode-id")
                if not aid_str or not mid_str:
                    return None, None, None
                try:
                    return action, int(aid_str), int(mid_str)
                except (ValueError, TypeError):
                    return None, None, None
            element = element.parent()
        return None, None, None
