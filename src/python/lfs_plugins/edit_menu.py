# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Edit menu implementation."""

import lichtfeld as lf
from .layouts.menus import register_menu, menu_action


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
        ]


def register():
    pass


def unregister():
    pass
