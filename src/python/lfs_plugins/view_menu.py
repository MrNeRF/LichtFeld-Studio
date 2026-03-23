# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""View menu implementation."""

import lichtfeld as lf
from .layouts.menus import register_menu, menu_submenu, menu_toggle


@register_menu
class ViewMenu:
    """View menu for the menu bar."""

    label = "menu.view"
    location = "MENU_BAR"
    order = 30
    _THEME_OPTIONS = (
        ("dark", "menu.view.theme.dark"),
        ("light", "menu.view.theme.light"),
        ("gruvbox", "menu.view.theme.gruvbox"),
        ("catppuccin_mocha", "menu.view.theme.catppuccin_mocha"),
        ("catppuccin_latte", "menu.view.theme.catppuccin_latte"),
        ("nord", "menu.view.theme.nord"),
    )

    _SCALE_OPTIONS = (
        (0.0, "menu.view.ui_scale.auto"),
        (1.0, "100%"),
        (1.25, "125%"),
        (1.5, "150%"),
        (1.75, "175%"),
        (2.0, "200%"),
    )

    @staticmethod
    def _normalize_theme_name(name: str) -> str:
        normalized = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "gruvbox_dark": "gruvbox",
            "catppuccin_dark": "catppuccin_mocha",
            "catppuccin_light": "catppuccin_latte",
        }
        return aliases.get(normalized, normalized)

    def menu_items(self):
        tr = lf.ui.tr
        current_theme = self._normalize_theme_name(lf.ui.get_theme())
        theme_items = [
            menu_toggle(
                tr(label_key),
                lambda theme=theme_id: lf.ui.set_theme(theme),
                current_theme == theme_id,
            )
            for theme_id, label_key in self._THEME_OPTIONS
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
        ]



def register():
    pass


def unregister():
    pass
