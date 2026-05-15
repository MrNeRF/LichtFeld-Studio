# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Edit menu implementation."""

import lichtfeld as lf
from .layouts.menus import register_menu, menu_action, menu_separator

__lfs_menu_classes__ = ["EditMenu"]


def _shortcut(action, fallback):
    try:
        shortcut = lf.keymap.get_trigger_description(action, lf.keymap.ToolMode.GLOBAL)
        return "" if shortcut == "Unbound" else shortcut
    except (AttributeError, RuntimeError, TypeError):
        return fallback


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
                shortcut=_shortcut(lf.keymap.Action.UNDO, "Ctrl+Z"),
                enabled=lf.undo.can_undo(),
            ),
            menu_action(
                "Redo",
                lf.undo.redo,
                shortcut=_shortcut(lf.keymap.Action.REDO, "Ctrl+Shift+Z"),
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
