# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Select menu implementation."""

import lichtfeld as lf
from .layouts.menus import register_menu, menu_action, menu_separator

__lfs_menu_classes__ = ["SelectMenu"]


def _shortcut(action, fallback):
    try:
        if not lf.keymap.is_bound(action, lf.keymap.ToolMode.GLOBAL):
            return ""
        return lf.keymap.get_trigger_description(action, lf.keymap.ToolMode.GLOBAL)
    except (AttributeError, RuntimeError, TypeError):
        return fallback


def _can_edit_selection():
    try:
        return bool(lf.ui.can_edit_gaussian_selection())
    except (AttributeError, RuntimeError, TypeError):
        context = lf.ui.context()
        return bool(getattr(context, "num_gaussians", 0)) and not bool(getattr(context, "is_training", False))


def _has_gaussian_selection():
    try:
        return bool(lf.ui.has_gaussian_selection())
    except (AttributeError, RuntimeError, TypeError):
        selected = lf.ui.context().selected_gaussians
        return selected is not None


def _has_gaussian_clipboard():
    try:
        return bool(lf.ui.has_gaussian_clipboard())
    except (AttributeError, RuntimeError, TypeError):
        return False


def _is_selection_tool_active():
    try:
        return lf.ui.get_active_tool() == "builtin.select"
    except (AttributeError, RuntimeError, TypeError):
        return False


@register_menu
class SelectMenu:
    """Select menu for Gaussian selection actions."""

    label = "menu.select"
    location = "MENU_BAR"
    order = 22

    def menu_items(self):
        tr = lf.ui.tr
        can_edit = _can_edit_selection()
        has_selection = _has_gaussian_selection()
        has_clipboard = _has_gaussian_clipboard()
        selection_tool_active = _is_selection_tool_active()
        action = lf.keymap.Action

        return [
            menu_action(
                tr("menu.select.copy_selection"),
                lf.ui.copy_gaussian_selection,
                shortcut=_shortcut(action.COPY_SELECTION, "Ctrl+C"),
                enabled=can_edit and has_selection,
            ),
            menu_action(
                tr("menu.select.cut_selection"),
                lf.ui.cut_gaussian_selection,
                shortcut=_shortcut(action.CUT_SELECTION, "Ctrl+X"),
                enabled=can_edit and has_selection,
            ),
            menu_action(
                tr("menu.select.paste_selection"),
                lf.ui.paste_gaussian_selection,
                shortcut=_shortcut(action.PASTE_SELECTION, "Ctrl+V"),
                enabled=can_edit and has_clipboard,
            ),
            menu_separator(),
            menu_action(
                tr("menu.select.invert_selection"),
                lf.ui.invert_gaussian_selection,
                shortcut=_shortcut(action.INVERT_SELECTION, "Ctrl+I"),
                enabled=can_edit,
            ),
            menu_action(
                tr("menu.select.select_all"),
                lf.ui.select_all_gaussians,
                shortcut=_shortcut(action.SELECT_ALL, "Ctrl+A"),
                enabled=can_edit and selection_tool_active,
            ),
            menu_action(
                tr("menu.select.deselect_all"),
                lf.ui.deselect_all_gaussians,
                shortcut=_shortcut(action.DESELECT_ALL, "Ctrl+D"),
                enabled=can_edit and has_selection,
            ),
        ]


def register():
    pass


def unregister():
    pass
