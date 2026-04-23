# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Edit menu implementation."""

import lichtfeld as lf
from .layouts.menus import register_menu, menu_action, menu_separator

__lfs_menu_classes__ = ["EditMenu"]


@register_menu
class EditMenu:
    """Edit menu for the menu bar."""

    label = "menu.edit"
    location = "MENU_BAR"
    order = 20

    def menu_items(self):
        return [
            menu_action(
                "Undo",
                lf.undo.undo,
                shortcut="Ctrl+Z",
                enabled=lf.undo.can_undo(),
            ),
            menu_action(
                "Redo",
                lf.undo.redo,
                shortcut="Ctrl+Shift+Z",
                enabled=lf.undo.can_redo(),
            ),
            menu_separator(),
            menu_action(
                lf.ui.tr("menu.edit.input_settings"),
                lambda: lf.ui.set_panel_enabled("lfs.input_settings", True),
            ),
        ]


def register():
    pass


def unregister():
    pass
