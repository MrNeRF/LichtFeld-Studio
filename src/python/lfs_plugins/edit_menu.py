# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Edit menu implementation."""

import lichtfeld as lf
from .layouts.menus import register_menu, menu_action, menu_separator, menu_submenu, menu_toggle


@register_menu
class EditMenu:
    """Edit menu for the menu bar."""

    label = "menu.edit"
    location = "MENU_BAR"
    order = 20

    def menu_items(self):
        current = lf.ui.get_current_language()
        undo_name = lf.undo.get_undo_name()
        redo_name = lf.undo.get_redo_name()
        undo_label = f"Undo {undo_name}" if undo_name else "Undo"
        redo_label = f"Redo {redo_name}" if redo_name else "Redo"
        undo_history_items = [
            menu_action(name, lambda steps=index + 1: lf.undo.jump("undo", steps))
            for index, name in enumerate(lf.undo.undo_names())
        ] or [menu_action("Nothing to undo", lambda: None, enabled=False)]
        redo_history_items = [
            menu_action(name, lambda steps=index + 1: lf.undo.jump("redo", steps))
            for index, name in enumerate(lf.undo.redo_names())
        ] or [menu_action("Nothing to redo", lambda: None, enabled=False)]
        language_items = [
            menu_toggle(
                lang_name,
                lambda code=lang_code: lf.ui.set_language(code),
                lang_code == current,
            )
            for lang_code, lang_name in lf.ui.get_languages()
        ]

        return [
            menu_action(
                undo_label,
                lf.undo.undo,
                shortcut="Ctrl+Z",
                enabled=lf.undo.can_undo(),
            ),
            menu_action(
                redo_label,
                lf.undo.redo,
                shortcut="Ctrl+Shift+Z",
                enabled=lf.undo.can_redo(),
            ),
            menu_submenu("Undo Stack", undo_history_items),
            menu_submenu("Redo Stack", redo_history_items),
            menu_separator(),
            menu_action(
                "History",
                lambda: lf.ui.set_panel_enabled("lfs.history", True),
                shortcut="Ctrl+Alt+H",
            ),
            menu_separator(),
            menu_action(
                lf.ui.tr("menu.edit.input_settings"),
                lambda: lf.ui.set_panel_enabled("lfs.input_settings", True),
            ),
            menu_separator(),
            menu_submenu(lf.ui.tr("preferences.language"), language_items),
        ]


def register():
    pass


def unregister():
    pass
