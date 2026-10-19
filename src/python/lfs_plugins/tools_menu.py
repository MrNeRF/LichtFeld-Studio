# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tools menu implementation."""

import lichtfeld as lf
import sys
from .layouts.menus import keymap_shortcut, register_menu, menu_action, menu_operator, menu_separator, menu_submenu

__lfs_menu_classes__ = ["ToolsMenu"]


def _open_gallery_scope() -> None:
    lf.ui.set_panel_enabled("lfs.asset_manager", True)
    get_panel_object = getattr(lf.ui, "get_panel_object", None)
    panel = get_panel_object("lfs.asset_manager") if callable(get_panel_object) else None
    select_gallery_scope = getattr(panel, "select_gallery_scope", None)
    if callable(select_gallery_scope):
        select_gallery_scope()


def _open_projects_scope() -> None:
    lf.ui.set_panel_enabled("lfs.asset_manager", True)
    get_panel_object = getattr(lf.ui, "get_panel_object", None)
    panel = get_panel_object("lfs.asset_manager") if callable(get_panel_object) else None
    select_projects_scope = getattr(panel, "select_projects_scope", None)
    if callable(select_projects_scope):
        select_projects_scope()


@register_menu
class ToolsMenu:
    """Tools menu for the menu bar."""

    label = "menu.tools"
    location = "MENU_BAR"
    order = 40

    def menu_items(self):
        tr = lf.ui.tr
        developer_items = [
            menu_action(
                tr("menu.tools.python_console"),
                lf.ui.show_python_console,
                shortcut=keymap_shortcut(lf.keymap.Action.TOGGLE_PYTHON_CONSOLE),
            ),
        ]
        if sys.platform == "win32":
            developer_items.append(menu_action(
                tr("menu.tools.system_console"), lf.ui.toggle_system_console,
            ))
        return [
            menu_operator(
                "lfs_plugins.file_menu.Mesh2SplatOperator",
                label=tr("menu.file.mesh_to_splat"),
            ),
            menu_operator(
                "lfs_plugins.file_menu.ExtractVideoFramesOperator",
                label=tr("menu.file.extract_video_frames"),
            ),
            menu_action(
                tr("rendering_panel.section_simplify"),
                lambda: lf.ui.set_panel_enabled("lfs.splat_simplify", True),
            ),
            menu_separator(),
            menu_submenu(tr("menu.tools.developer"), developer_items),
            menu_action(
                tr("menu.tools.plugin_marketplace"),
                lambda: lf.ui.set_panel_enabled("lfs.plugin_marketplace", True),
            ),
        ]


def register():
    pass


def unregister():
    pass
