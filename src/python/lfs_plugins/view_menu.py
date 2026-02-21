# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""View menu implementation."""

import lichtfeld as lf
from .layouts.menus import register_menu


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

    @staticmethod
    def _normalize_theme_name(name: str) -> str:
        normalized = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "gruvbox_dark": "gruvbox",
            "catppuccin_dark": "catppuccin_mocha",
            "catppuccin_light": "catppuccin_latte",
        }
        return aliases.get(normalized, normalized)

    def draw(self, layout):
        tr = lf.ui.tr

        if layout.begin_menu(tr("menu.view.theme")):
            current_theme = self._normalize_theme_name(lf.ui.get_theme())
            for theme_id, label_key in self._THEME_OPTIONS:
                is_current = current_theme == theme_id
                if layout.menu_item_toggle(tr(label_key), "", is_current):
                    lf.ui.set_theme(theme_id)
            layout.end_menu()

        layout.separator()

        if layout.menu_item_shortcut(tr("menu.view.python_console"), "Ctrl+`"):
            lf.ui.show_python_console()

        if layout.menu_item(tr("menu.view.plugin_marketplace")):
            lf.ui.set_panel_enabled("lfs.plugin_marketplace", True)



def register():
    pass


def unregister():
    pass
