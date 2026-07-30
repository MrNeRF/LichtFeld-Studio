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
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (420, 0)
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

    def __init__(self):
        self._handle = None
        self._theme_catalog = []
        self._language_catalog = []
        self._last_state = None

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("preferences")
        if model is None:
            return

        model.bind_func("panel_label", lambda: lf.ui.tr("preferences.title"))
        model.bind("theme_idx", self._theme_index, self._set_theme_index)
        model.bind("scale_idx", self._scale_index, self._set_scale_index)
        model.bind("language_idx", self._language_index, self._set_language_index)
        model.bind_event("close", self._on_close)
        model.bind_event("reset_layout", self._on_reset_layout)
        model.bind_record_list("themes")
        model.bind_record_list("scales")
        model.bind_record_list("languages")
        self._handle = model.get_handle()

    def on_mount(self, doc):
        super().on_mount(doc)
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
        if self._handle:
            self._handle.dirty("theme_idx")
            self._handle.dirty("scale_idx")
            self._handle.dirty("language_idx")

    def _state(self):
        return (
            lf.ui.get_theme(),
            float(lf.ui.get_ui_scale_preference()),
            lf.ui.get_current_language(),
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
            lf.ui.set_language(self._language_catalog[index][0])
            self._refresh_selection()

    def _on_close(self, _handle, _event, _args):
        lf.ui.set_panel_enabled(self.id, False)

    def _on_reset_layout(self, _handle, _event, _args):
        reset_label = lf.ui.tr("preferences.reset_layout")

        def _on_result(button):
            if button != reset_label:
                return
            error = lf.ui.reset_layout()
            if error:
                lf.ui.message_dialog(
                    reset_label,
                    f"{lf.ui.tr('preferences.reset_layout_failed')} {error}",
                    "error",
                )
                return
            lf.ui.message_dialog(
                reset_label,
                lf.ui.tr("preferences.reset_layout_success"),
            )

        lf.ui.confirm_dialog(
            reset_label,
            lf.ui.tr("preferences.reset_layout_confirmation"),
            [lf.ui.tr("common.cancel"), reset_label],
            _on_result,
        )

    def _refresh_selection(self):
        self._last_state = self._state()
        if self._handle:
            self._handle.dirty("theme_idx")
            self._handle.dirty("scale_idx")
            self._handle.dirty("language_idx")
