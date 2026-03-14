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
        undo_api = getattr(lf, "undo", None)
        undo_name = undo_api.get_undo_name() if undo_api else ""
        redo_name = undo_api.get_redo_name() if undo_api else ""
        undo_label = f"Undo {undo_name}" if undo_name else "Undo"
        redo_label = f"Redo {redo_name}" if redo_name else "Redo"
        undo_history_items = [
            menu_action(name, lambda: None, enabled=False)
            for name in (undo_api.undo_names() if undo_api else [])
        ] or [menu_action("Nothing to undo", lambda: None, enabled=False)]
        redo_history_items = [
            menu_action(name, lambda: None, enabled=False)
            for name in (undo_api.redo_names() if undo_api else [])
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
                undo_api.undo if undo_api else (lambda: None),
                shortcut="Ctrl+Z",
                enabled=undo_api.can_undo() if undo_api else False,
            ),
            menu_action(
                redo_label,
                undo_api.redo if undo_api else (lambda: None),
                shortcut="Ctrl+Shift+Z",
                enabled=undo_api.can_redo() if undo_api else False,
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
