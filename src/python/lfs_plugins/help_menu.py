# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Help menu implementation using Blender-style operators."""

import lichtfeld as lf
from .types import Operator
from .layouts.menus import register_menu


class GettingStartedOperator(Operator):
    label = "menu.help.getting_started"
    description = "Show the Getting Started guide"

    def execute(self, context) -> set:
        lf.ui.set_panel_enabled("lfs.getting_started", True)
        return {"FINISHED"}


class AboutOperator(Operator):
    label = "menu.help.about"
    description = "Show About dialog"

    def execute(self, context) -> set:
        lf.ui.set_panel_enabled("lfs.about", True)
        return {"FINISHED"}


@register_menu
class HelpMenu:
    """Help menu for the menu bar."""

    label = "menu.help"
    location = "MENU_BAR"
    order = 100

    def draw(self, layout):
        layout.operator_(GettingStartedOperator._class_id())
        layout.separator()
        layout.operator_(AboutOperator._class_id())


_operator_classes = [
    GettingStartedOperator,
    AboutOperator,
]


def register():
    for cls in _operator_classes:
        lf.register_class(cls)


def unregister():
    for cls in reversed(_operator_classes):
        lf.unregister_class(cls)
