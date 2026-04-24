# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""View menu implementation."""

import lichtfeld as lf
from .layouts.menus import menu_action, menu_separator, register_menu, menu_submenu, menu_toggle

__lfs_menu_classes__ = ["ViewMenu"]


def _tr_fallback(key: str, fallback: str) -> str:
    result = lf.ui.tr(key)
    if result and result != key:
        return result
    return fallback


@register_menu
class ViewMenu:
    """View menu for the menu bar."""

    label = "menu.view"
    location = "MENU_BAR"
    order = 30

    _SCALE_OPTIONS = (
        (0.0, "menu.view.ui_scale.auto"),
        (1.0, "100%"),
        (1.25, "125%"),
        (1.5, "150%"),
        (1.75, "175%"),
        (2.0, "200%"),
    )

    def menu_items(self):
        tr = lf.ui.tr
        current_theme = lf.ui.get_theme()
        theme_catalog = sorted(
            lf.ui.themes(),
            key=lambda theme: (theme.get("order", 0), theme.get("name", theme.get("id", ""))),
        )
        theme_items = [
            menu_toggle(
                tr(theme.get("label_key") or theme.get("name") or theme["id"]),
                lambda theme_id=theme["id"]: lf.ui.set_theme(theme_id),
                current_theme == theme["id"],
            )
            for theme in theme_catalog
        ]

        pref = lf.ui.get_ui_scale_preference()
        scale_items = []
        for scale_val, label_key in self._SCALE_OPTIONS:
            label = tr(label_key) if scale_val == 0.0 else label_key
            scale_items.append(
                menu_toggle(
                    label,
                    lambda scale=scale_val: lf.ui.set_ui_scale(scale),
                    abs(pref - scale_val) < 0.01,
                )
            )

        return [
            menu_submenu(tr("menu.view.theme"), theme_items),
            menu_submenu(tr("menu.view.ui_scale"), scale_items),
            menu_separator(),
            menu_action(_tr_fallback("image_preview.reset_view", "Reset View"), lf.reset_camera),
            menu_action(_tr_fallback("main_panel.console", "Console"), lf.ui.toggle_system_console),
        ]



def register():
    pass


def unregister():
    pass
