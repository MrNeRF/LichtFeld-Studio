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

    def draw(self, layout):
        tr = lf.ui.tr

        if layout.begin_menu(tr("menu.view.theme")):
            is_dark = lf.ui.get_theme() == "Dark"
            if layout.menu_item_toggle(tr("menu.view.theme.dark"), "", is_dark):
                lf.ui.set_theme("dark")
            if layout.menu_item_toggle(tr("menu.view.theme.light"), "", not is_dark):
                lf.ui.set_theme("light")
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
