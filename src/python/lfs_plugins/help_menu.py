# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Help menu implementation using Blender-style operators."""

import lichtfeld as lf
from .types import Operator
from .layouts.menus import register_menu, menu_operator, menu_separator

__lfs_menu_classes__ = ["HelpMenu"]


class GettingStartedOperator(Operator):
    label = "menu.help.getting_started"
    description = "Show the Getting Started guide"

    def execute(self, context) -> set:
        lf.ui.set_panel_enabled("lfs.getting_started", True)
        return {"FINISHED"}


class SetDefaultAppOperator(Operator):
    label = "file_association.menu_register"
    description = "Open Windows default app settings for splat files (.ply, .sog, .spz, .usd, .usda, .usdc, .usdz)"

    def execute(self, context) -> set:
        lf.ui.register_file_associations()
        lf.ui.open_file_association_settings()
        return {"FINISHED"}


class UnsetDefaultAppOperator(Operator):
    label = "file_association.menu_unregister"
    description = "Remove file type associations"

    def execute(self, context) -> set:
        lf.ui.unregister_file_associations()
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

    def menu_items(self):
        items = [menu_operator(GettingStartedOperator)]
        if lf.ui.is_windows_platform():
            items.append(menu_separator())
            items.append(menu_operator(SetDefaultAppOperator))
        items.append(menu_separator())
        items.append(menu_operator(AboutOperator))
        return items


_operator_classes = [
    GettingStartedOperator,
    SetDefaultAppOperator,
    UnsetDefaultAppOperator,
    AboutOperator,
]


def register():
    for cls in _operator_classes:
        lf.register_class(cls)


def unregister():
    for cls in reversed(_operator_classes):
        lf.unregister_class(cls)
