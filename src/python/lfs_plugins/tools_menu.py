# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tools menu implementation."""

import lichtfeld as lf
from .layouts.menus import register_menu, menu_action, menu_separator

__lfs_menu_classes__ = ["ToolsMenu"]


@register_menu
class ToolsMenu:
    """Tools menu for the menu bar."""

    label = "menu.tools"
    location = "MENU_BAR"
    order = 25

    def menu_items(self):
        tr = lf.ui.tr
        return [
            menu_action(
                tr("menu.tools.asset_manager"),
                lambda: lf.ui.set_panel_enabled("lfs.asset_manager", True),
            ),
            menu_separator(),
            menu_action(
                tr("menu.tools.python_console"),
                lf.ui.show_python_console,
                shortcut="Ctrl+`",
            ),
            menu_action(
                tr("menu.tools.plugin_marketplace"),
                lambda: lf.ui.set_panel_enabled("lfs.plugin_marketplace", True),
            ),
        ]


def register():
    pass


def unregister():
    pass
