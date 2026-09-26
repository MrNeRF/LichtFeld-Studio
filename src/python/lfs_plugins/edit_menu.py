# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Edit menu implementation."""

import lichtfeld as lf
from .layouts.menus import keymap_shortcut, register_menu, menu_action, menu_separator

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
                shortcut=keymap_shortcut(lf.keymap.Action.UNDO),
                enabled=lf.undo.can_undo(),
            ),
            menu_action(
                "Redo",
                lf.undo.redo,
                shortcut=keymap_shortcut(lf.keymap.Action.REDO),
                enabled=lf.undo.can_redo(),
            ),
            menu_separator(),
            menu_action(
                lf.ui.tr("menu.edit.preferences"),
                lambda: lf.ui.set_panel_enabled("lfs.preferences", True),
                shortcut=keymap_shortcut(lf.keymap.Action.OPEN_PREFERENCES),
            ),
        ]


def register():
    pass


def unregister():
    pass
