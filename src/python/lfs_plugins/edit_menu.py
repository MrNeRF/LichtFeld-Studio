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
