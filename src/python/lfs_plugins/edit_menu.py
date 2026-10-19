# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Edit menu implementation."""

import lichtfeld as lf
from .select_menu import _can_edit_selection, _has_gaussian_selection, _has_gaussian_clipboard
from .layouts.menus import keymap_shortcut, register_menu, menu_action, menu_separator

__lfs_menu_classes__ = ["EditMenu"]


@register_menu
class EditMenu:
    """Edit menu for the menu bar."""

    label = "menu.edit"
    location = "MENU_BAR"
    order = 20

    def menu_items(self):
        tr = lf.ui.tr
        can_edit = _can_edit_selection()
        has_selection = _has_gaussian_selection()
        has_clipboard = _has_gaussian_clipboard()
        action = lf.keymap.Action
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
                tr("menu.select.cut_selection"),
                lf.ui.cut_gaussian_selection,
                shortcut=keymap_shortcut(action.CUT_SELECTION),
                enabled=can_edit and has_selection,
            ),
            menu_action(
                tr("menu.select.copy_selection"),
                lf.ui.copy_gaussian_selection,
                shortcut=keymap_shortcut(action.COPY_SELECTION),
                enabled=can_edit and has_selection,
            ),
            menu_action(
                tr("menu.select.paste_selection"),
                lf.ui.paste_gaussian_selection,
                shortcut=keymap_shortcut(action.PASTE_SELECTION),
                enabled=can_edit and has_clipboard,
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
