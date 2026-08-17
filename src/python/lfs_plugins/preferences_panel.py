# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Application-level appearance and language preferences."""

import lichtfeld as lf

from .types import Panel

__lfs_panel_classes__ = ["PreferencesPanel"]
__lfs_panel_ids__ = ["lfs.preferences"]


class PreferencesPanel(Panel):
    """Floating home for application-level preferences."""

    id = "lfs.preferences"
    label = "Preferences"
    space = lf.ui.PanelSpace.FLOATING
    order = 100
    template = "rmlui/preferences.rml"
    height_mode = lf.ui.PanelHeightMode.FILL
    size = (780, 360)
    options = {lf.ui.PanelOption.DEFAULT_CLOSED}
    update_policy = "dirty"

    SCALE_OPTIONS = (
        (0.0, "menu.view.ui_scale.auto"),
        (1.0, "100%"),
        (1.25, "125%"),
        (1.5, "150%"),
        (1.75, "175%"),
        (2.0, "200%"),
    )

    NAVIGATION_OPTIONS = (
        ("orbit", "preferences.navigation_orbit"),
        ("trackball", "preferences.navigation_trackball"),
        ("fpv", "preferences.navigation_fpv"),
        ("drone", "preferences.navigation_drone"),
    )

    EXPANDABLE_SECTIONS = (
        "language",
        "appearance",
        "navigation",
        "view_snap",
        "interface",
    )

    def __init__(self):
        self._handle = None
        self._theme_catalog = []
        self._language_catalog = []
        self._last_state = None
        self._section = "general"
        self._expanded_sections = set(self.EXPANDABLE_SECTIONS)

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("preferences")
        if model is None:
            return

        model.bind_func("panel_label", lambda: lf.ui.tr("preferences.title"))
        model.bind_func("show_general", lambda: self._section == "general")
        model.bind_func("show_appearance", lambda: self._section == "appearance")
        model.bind_func("show_input", lambda: self._section == "input")
        model.bind_func("show_interface", lambda: self._section == "interface")
        model.bind_func("show_section_reset", lambda: True)
        model.bind_func("reset_section_label", self._reset_section_label)
        for section in self.EXPANDABLE_SECTIONS:
            model.bind_func(
                f"{section}_expanded",
                lambda section=section: section in self._expanded_sections,
            )
        model.bind("theme_idx", self._theme_index, self._set_theme_index)
        model.bind("scale_idx", self._scale_index, self._set_scale_index)
        model.bind("language_idx", self._language_index, self._set_language_index)
        model.bind("navigation_idx", self._navigation_index, self._set_navigation_index)
        model.bind("view_snap", lf.get_camera_view_snap_enabled, self._set_view_snap)
        model.bind("remember_navigation", lf.ui.remember_camera_navigation, self._set_remember_navigation)
        model.bind("remember_view_snap", lf.ui.remember_camera_view_snap, self._set_remember_view_snap)
        model.bind_event("close", self._on_close)
        model.bind_event("reset_current_section", self._on_reset_current_section)
        model.bind_event("reset_all_settings", self._on_reset_all_settings)
        model.bind_event("show_general", lambda *_: self._set_section("general"))
        model.bind_event("show_appearance", lambda *_: self._set_section("appearance"))
        model.bind_event("show_input", lambda *_: self._set_section("input"))
        model.bind_event("show_interface", lambda *_: self._set_section("interface"))
        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_record_list("themes")
        model.bind_record_list("scales")
        model.bind_record_list("languages")
        model.bind_record_list("navigation_modes")
        self._handle = model.get_handle()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._expanded_sections = set(self.EXPANDABLE_SECTIONS)
        self._dirty_expanded_sections()
        self._rebuild_records()
        self._last_state = self._state()
        self._refresh_selection()

    def on_unmount(self, doc):
        self._handle = None
        doc.remove_data_model("preferences")

    def on_update(self, doc):
        state = self._state()
        if state == self._last_state:
            return
        self._last_state = state
        self._dirty_selection()

    def _state(self):
        return (
            lf.ui.get_theme(),
            float(lf.ui.get_ui_scale_preference()),
            lf.ui.get_current_language(),
            lf.get_camera_navigation_mode(),
            lf.get_camera_view_snap_enabled(),
            lf.ui.remember_camera_navigation(),
            lf.ui.remember_camera_view_snap(),
        )

    def _rebuild_records(self):
        self._theme_catalog = sorted(
            lf.ui.themes(),
            key=lambda theme: (theme.get("order", 0), theme.get("name", theme.get("id", ""))),
        )
        self._language_catalog = list(lf.ui.get_languages())
        if not self._handle:
            return
        self._handle.update_record_list(
            "themes",
            [
                {
                    "index": str(index),
                    "label": lf.ui.tr(theme.get("label_key") or theme.get("name") or theme["id"]),
                }
                for index, theme in enumerate(self._theme_catalog)
            ],
        )
        self._handle.update_record_list(
            "scales",
            [
                {
                    "index": str(index),
                    "label": lf.ui.tr(label) if scale == 0.0 else label,
                }
                for index, (scale, label) in enumerate(self.SCALE_OPTIONS)
            ],
        )
        self._handle.update_record_list(
            "languages",
            [
                {"index": str(index), "label": name}
                for index, (_code, name) in enumerate(self._language_catalog)
            ],
        )
        self._handle.update_record_list(
            "navigation_modes",
            [
                {"index": str(index), "label": lf.ui.tr(label)}
                for index, (_mode, label) in enumerate(self.NAVIGATION_OPTIONS)
            ],
        )

    def _theme_index(self):
        current = lf.ui.get_theme()
        for index, theme in enumerate(self._theme_catalog):
            if theme["id"] == current:
                return str(index)
        return "0"

    def _set_theme_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self._theme_catalog):
            lf.ui.set_theme(self._theme_catalog[index]["id"])
            self._refresh_selection()

    def _scale_index(self):
        preference = float(lf.ui.get_ui_scale_preference())
        for index, (scale, _label) in enumerate(self.SCALE_OPTIONS):
            if abs(preference - scale) < 0.01:
                return str(index)
        return "0"

    def _set_scale_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self.SCALE_OPTIONS):
            lf.ui.set_ui_scale(self.SCALE_OPTIONS[index][0])
            self._refresh_selection()

    def _language_index(self):
        current = lf.ui.get_current_language()
        for index, (code, _name) in enumerate(self._language_catalog):
            if code == current:
                return str(index)
        return "0"

    def _set_language_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self._language_catalog):
            language = self._language_catalog[index][0]
            if language == lf.ui.get_current_language():
                return
            lf.ui.set_language(language)
            self._refresh_selection()

    def _navigation_index(self):
        current = lf.get_camera_navigation_mode()
        for index, (mode, _label) in enumerate(self.NAVIGATION_OPTIONS):
            if mode == current:
                return str(index)
        return "0"

    def _set_navigation_index(self, value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self.NAVIGATION_OPTIONS):
            lf.set_camera_navigation_mode(self.NAVIGATION_OPTIONS[index][0])
            self._refresh_selection()

    def _set_view_snap(self, enabled):
        lf.set_camera_view_snap_enabled(bool(enabled))
        self._refresh_selection()

    def _set_remember_navigation(self, enabled):
        lf.ui.set_remember_camera_navigation(bool(enabled))
        if enabled:
            lf.set_camera_navigation_mode(lf.get_camera_navigation_mode())
        self._refresh_selection()

    def _set_remember_view_snap(self, enabled):
        lf.ui.set_remember_camera_view_snap(bool(enabled))
        if enabled:
            lf.set_camera_view_snap_enabled(lf.get_camera_view_snap_enabled())
        self._refresh_selection()

    def _on_close(self, _handle, _event, _args):
        lf.ui.set_panel_enabled(self.id, False)

    def _set_section(self, section):
        if self._section == section:
            return
        self._section = section
        if self._handle:
            for name in ("show_general", "show_appearance", "show_input", "show_interface",
                          "show_section_reset", "reset_section_label"):
                self._handle.dirty(name)

    def _on_toggle_section(self, _handle, _event, args):
        if not args:
            return
        section = str(args[0])
        if section not in self.EXPANDABLE_SECTIONS:
            return
        if section in self._expanded_sections:
            self._expanded_sections.remove(section)
        else:
            self._expanded_sections.add(section)
        if self._handle:
            self._handle.dirty(f"{section}_expanded")

    def _dirty_expanded_sections(self):
        if not self._handle:
            return
        for section in self.EXPANDABLE_SECTIONS:
            self._handle.dirty(f"{section}_expanded")

    def _reset_section_label(self):
        return lf.ui.tr("preferences.reset_current_section")

    def _on_reset_current_section(self, _handle, _event, _args):
        reset_label = self._reset_section_label()
        section_name = lf.ui.tr(f"preferences.{self._section}")

        def _on_result(button):
            if button != reset_label:
                return
            error = self._reset_section()
            if error:
                lf.ui.message_dialog(
                    reset_label,
                    f"{lf.ui.tr('preferences.reset_section_failed')} {error}",
                    "error",
                )
                return
            lf.ui.message_dialog(
                reset_label,
                lf.ui.tr("preferences.reset_section_success"),
            )

        lf.ui.confirm_dialog(
            f"{reset_label}: {section_name}",
            f"{lf.ui.tr('preferences.reset_section_confirmation')} {section_name}.",
            [lf.ui.tr("common.cancel"), reset_label],
            _on_result,
        )

    def _on_reset_all_settings(self, _handle, _event, _args):
        reset_label = lf.ui.tr("preferences.reset_all_settings")

        def _on_result(button):
            if button != reset_label:
                return
            errors = [self._reset_section(section) for section in ("general", "appearance", "input", "interface")]
            errors.append(lf.ui.reset_window_state())
            error = next((item for item in errors if item), None)
            if error:
                lf.ui.message_dialog(
                    reset_label,
                    f"{lf.ui.tr('preferences.reset_section_failed')} {error}",
                    "error",
                )
                return
            self._refresh_selection()
            lf.ui.message_dialog(reset_label, lf.ui.tr("preferences.reset_all_settings_success"))

        lf.ui.confirm_dialog(
            reset_label,
            lf.ui.tr("preferences.reset_all_settings_confirmation"),
            [lf.ui.tr("common.cancel"), reset_label],
            _on_result,
        )

    def _reset_section(self, section=None):
        section = section or self._section
        if section == "general":
            lf.ui.set_language("en")
        elif section == "appearance":
            lf.ui.set_theme("dark")
            lf.ui.set_ui_scale(0.0)
        elif section == "input":
            lf.ui.set_remember_camera_navigation(False)
            lf.ui.set_remember_camera_view_snap(False)
            lf.set_camera_navigation_mode("orbit")
            lf.set_camera_view_snap_enabled(False)
        elif section == "interface":
            return lf.ui.reset_layout()
        self._refresh_selection()
        return None

    def _refresh_selection(self):
        self._last_state = self._state()
        self._dirty_selection()

    def _dirty_selection(self):
        if self._handle:
            self._handle.dirty("theme_idx")
            self._handle.dirty("scale_idx")
            self._handle.dirty("language_idx")
            self._handle.dirty("navigation_idx")
            self._handle.dirty("view_snap")
            self._handle.dirty("remember_navigation")
            self._handle.dirty("remember_view_snap")
