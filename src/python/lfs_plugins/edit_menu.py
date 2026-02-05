# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Edit menu implementation."""

import lichtfeld as lf
from .layouts.menus import register_menu


@register_menu
class EditMenu:
    """Edit menu for the menu bar."""

    label = "menu.edit"
    location = "MENU_BAR"
    order = 20

    def draw(self, layout):
        if layout.menu_item(lf.ui.tr("menu.edit.input_settings")):
            lf.ui.set_panel_enabled("lfs.input_settings", True)

        layout.separator()

        if layout.begin_menu(lf.ui.tr("preferences.language")):
            current = lf.ui.get_current_language()
            for lang_code, lang_name in lf.ui.get_languages():
                is_selected = lang_code == current
                if layout.menu_item_toggle(lang_name, "", is_selected):
                    lf.ui.set_language(lang_code)
            layout.end_menu()


def register():
    pass


def unregister():
    pass
