# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Input settings panel for keyboard/mouse binding configuration."""

import lichtfeld as lf
from .types import RmlPanel


def _xml_escape(text):
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


class InputSettingsPanel(RmlPanel):
    idname = "lfs.input_settings"
    label = "Input Settings"
    space = "FLOATING"
    order = 100
    rml_template = "rmlui/input_settings.rml"
    rml_height_mode = "content"
    initial_width = 500
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
        self._selected_mode_idx = 0
        self._rebinding_action = None
        self._rebinding_mode = None
        self._doc = None
        self._handle = None
        self._last_profiles = []
        self._last_state_key = None
        self._last_display_h = 0
        self._last_hint_mode = None
        self._last_capturing = None

    # ── Data model ────────────────────────────────────────────

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("input_settings")
        if model is None:
            return

        tr = lf.ui.tr

        model.bind_func("panel_label", lambda: tr("input_settings.title"))
        model.bind_func("profile_label", lambda: tr("input_settings.active_profile"))
        model.bind_func("tool_mode_label", lambda: tr("input_settings.tool_mode"))
        model.bind_func("tool_mode_hint", lambda: tr("input_settings.select_tool_mode"))
        model.bind_func("bindings_label", lambda: tr("input_settings.current_bindings"))
        model.bind_func("save_label", lambda: tr("input_settings.save_current_profile"))
        model.bind_func("reset_label", lambda: tr("input_settings.reset_to_default"))
        model.bind_func("export_label", lambda: tr("input_settings.export"))
        model.bind_func("import_label", lambda: tr("input_settings.import"))
        model.bind_func("save_hint", lambda: tr("input_settings.save_hint"))
        model.bind_func("double_click_hint", lambda: tr("input_settings.double_click_hint"))

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

    def on_load(self, doc):
        super().on_load(doc)
        self._doc = doc

        self._populate_profile_select()
        self._populate_mode_select()

        table_el = doc.get_element_by_id("bindings-table")
        if table_el:
            table_el.add_event_listener("click", self._on_table_click)

    def on_update(self, doc):
        self._update_max_height(doc)

        profiles = lf.keymap.get_available_profiles()
        if profiles != self._last_profiles:
            self._last_profiles = list(profiles)
            self._populate_profile_select()

        is_capturing = lf.keymap.is_capturing()
        mode = self.TOOL_MODES[self._selected_mode_idx]

        if is_capturing and self._rebinding_action is not None:
            trigger = lf.keymap.get_captured_trigger()
            if trigger is not None:
                self._rebinding_action = None
                self._rebinding_mode = None

        current_profile = lf.keymap.get_current_profile()
        state_key = (self._selected_mode_idx, self._rebinding_action, is_capturing, current_profile)
        if state_key != self._last_state_key:
            self._last_state_key = state_key
            self._rebuild_bindings_table(doc, mode)

        if mode != self._last_hint_mode:
            self._last_hint_mode = mode
            self._update_hint(doc, mode)

        if is_capturing != self._last_capturing:
            self._last_capturing = is_capturing
            self._update_disabled_state(doc, is_capturing)

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

    # ── DOM manipulation ──────────────────────────────────────

    def _populate_profile_select(self):
        select = self._doc.get_element_by_id("profile-select") if self._doc else None
        if not select:
            return
        profiles = lf.keymap.get_available_profiles()
        parts = [f'<option value="{i}">{_xml_escape(name)}</option>'
                 for i, name in enumerate(profiles)]
        select.set_inner_rml("".join(parts))

    def _populate_mode_select(self):
        select = self._doc.get_element_by_id("mode-select") if self._doc else None
        if not select:
            return
        parts = [f'<option value="{i}">{_xml_escape(lf.keymap.get_tool_mode_name(m))}</option>'
                 for i, m in enumerate(self.TOOL_MODES)]
        select.set_inner_rml("".join(parts))

    def _update_hint(self, doc, mode):
        hint_el = doc.get_element_by_id("bindings-hint")
        if not hint_el:
            return
        tr = lf.ui.tr
        if mode == lf.keymap.ToolMode.GLOBAL:
            hint_el.set_inner_rml(_xml_escape(tr("input_settings.global_bindings_hint")))
        else:
            hint_el.set_inner_rml(_xml_escape(tr("input_settings.tool_bindings_hint")))

    def _update_disabled_state(self, doc, is_capturing):
        for eid in ("profile-select", "mode-select"):
            el = doc.get_element_by_id(eid)
            if not el:
                continue
            if is_capturing:
                el.set_class("is-disabled-overlay", True)
            else:
                el.set_class("is-disabled-overlay", False)

    def _rebuild_bindings_table(self, doc, mode):
        table_el = doc.get_element_by_id("bindings-table")
        if not table_el:
            return
        table_el.set_inner_rml(self._build_bindings_rml(mode))

    def _build_bindings_rml(self, mode):
        tr = lf.ui.tr
        parts = []

        parts.append(self._section_rml(tr("input_settings.section.navigation")))
        for action in self.BINDING_SECTIONS["navigation"]:
            parts.append(self._binding_row_rml(action, mode))

        if mode == lf.keymap.ToolMode.GLOBAL:
            for action in self.BINDING_SECTIONS["navigation_global"]:
                parts.append(self._binding_row_rml(action, mode))

        if mode in (lf.keymap.ToolMode.GLOBAL, lf.keymap.ToolMode.SELECTION, lf.keymap.ToolMode.BRUSH):
            parts.append(self._section_rml(tr("input_settings.section.selection")))
            for action in self.BINDING_SECTIONS["selection"]:
                parts.append(self._binding_row_rml(action, mode))

            if mode == lf.keymap.ToolMode.GLOBAL:
                for action in self.BINDING_SECTIONS["selection_global"]:
                    parts.append(self._binding_row_rml(action, mode))

            if mode in (lf.keymap.ToolMode.GLOBAL, lf.keymap.ToolMode.SELECTION):
                for action in self.BINDING_SECTIONS["depth"]:
                    parts.append(self._binding_row_rml(action, mode))

        if mode == lf.keymap.ToolMode.BRUSH:
            parts.append(self._section_rml(tr("input_settings.section.brush")))
            for action in self.BINDING_SECTIONS["brush"]:
                parts.append(self._binding_row_rml(action, mode))

        if mode == lf.keymap.ToolMode.CROP_BOX:
            parts.append(self._section_rml(tr("input_settings.section.crop_box")))
            for action in self.BINDING_SECTIONS["crop_box"]:
                parts.append(self._binding_row_rml(action, mode))

        parts.append(self._section_rml(tr("input_settings.section.editing")))
        if mode in (lf.keymap.ToolMode.GLOBAL, lf.keymap.ToolMode.TRANSLATE,
                    lf.keymap.ToolMode.ROTATE, lf.keymap.ToolMode.SCALE):
            parts.append(self._binding_row_rml(lf.keymap.Action.DELETE_NODE, mode))
        else:
            parts.append(self._binding_row_rml(lf.keymap.Action.DELETE_SELECTED, mode))

        for action in self.BINDING_SECTIONS["editing"]:
            parts.append(self._binding_row_rml(action, mode))

        if mode == lf.keymap.ToolMode.GLOBAL:
            parts.append(self._section_rml(tr("input_settings.section.view")))
            for action in self.BINDING_SECTIONS["view_global"]:
                parts.append(self._binding_row_rml(action, mode))

        return "".join(parts)

    def _section_rml(self, title):
        return f'<span class="is-binding-section">{_xml_escape(title)}</span>'

    def _binding_row_rml(self, action, mode):
        tr = lf.ui.tr
        is_rebinding = (lf.keymap.is_capturing() and
                        self._rebinding_action == action and
                        self._rebinding_mode == mode)

        action_name = _xml_escape(lf.keymap.get_action_name(action))
        action_val = action.value
        mode_val = mode.value

        if is_rebinding:
            if lf.keymap.is_waiting_for_double_click():
                desc_text = _xml_escape(tr("input_settings.click_again_double"))
            else:
                desc_text = _xml_escape(tr("input_settings.press_key_or_click"))
            desc_class = "is-binding-desc is-capturing"
            btn = (f'<button class="btn btn--error is-rebind-btn" '
                   f'data-btn-action="cancel" data-action-id="{action_val}" data-mode-id="{mode_val}">'
                   f'{_xml_escape(tr("input_settings.cancel"))}</button>')
        else:
            desc_text = _xml_escape(lf.keymap.get_trigger_description(action, mode))
            desc_class = "is-binding-desc"
            btn = (f'<button class="btn btn--primary is-rebind-btn" '
                   f'data-btn-action="rebind" data-action-id="{action_val}" data-mode-id="{mode_val}">'
                   f'{_xml_escape(tr("input_settings.rebind"))}</button>')

        return (f'<div class="is-binding-row" data-action-id="{action_val}" data-mode-id="{mode_val}">'
                f'<span class="is-action-name">{action_name}</span>'
                f'<span class="{desc_class}">{desc_text}</span>'
                f'{btn}'
                f'</div>')

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
        for _ in range(6):
            if element is None:
                return None, None, None
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
            p = element.parent()
            if p is None:
                return None, None, None
            element = p
        return None, None, None
