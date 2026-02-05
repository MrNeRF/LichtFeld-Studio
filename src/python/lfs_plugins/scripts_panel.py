# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Python scripts management panel."""

import lichtfeld as lf
from .types import Panel


class ScriptsPanel(Panel):
    """Python Scripts panel - floating window for managing loaded scripts."""

    idname = "lfs.scripts"
    label = "Python Scripts"
    space = "FLOATING"
    order = 200
    options = {"DEFAULT_CLOSED"}

    def __init__(self):
        pass

    def draw(self, layout):
        scripts = lf.scripts.get_scripts()

        if layout.begin_menu_bar():
            if layout.begin_menu("Actions"):
                if layout.menu_item("Reload All", enabled=len(scripts) > 0):
                    self._reload_all()
                layout.separator()
                if layout.menu_item("Clear All"):
                    lf.scripts.clear()
                layout.end_menu()
            layout.end_menu_bar()

        if not scripts:
            layout.text_disabled("No Python scripts loaded.")
            layout.text_disabled("Use --python-script <path> to load scripts.")
            return

        layout.label("Loaded Scripts:")
        layout.separator()

        for i, script in enumerate(scripts):
            layout.push_id(i)

            enabled = script["enabled"]
            _, new_enabled = layout.checkbox("##enabled", enabled)
            if new_enabled != enabled:
                lf.scripts.set_script_enabled(i, new_enabled)

            layout.same_line()

            if script["has_error"]:
                layout.push_style_color("text", (1.0, 0.4, 0.4, 1.0))
            elif not script["enabled"]:
                layout.push_style_color("text", (0.5, 0.5, 0.5, 1.0))
            else:
                layout.push_style_color("text", (0.5, 1.0, 0.5, 1.0))

            filename = script["path"].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            layout.text(filename)
            layout.pop_style_color()

            if layout.is_item_hovered():
                layout.begin_tooltip()
                layout.text(f"Path: {script['path']}")
                if script["has_error"]:
                    layout.separator()
                    layout.text_colored(f"Error: {script['error_message']}", (1.0, 0.4, 0.4, 1.0))
                layout.end_tooltip()

            layout.same_line()
            layout.set_cursor_pos_x(layout.get_window_width() - 80)
            if layout.small_button("Reload"):
                lf.scripts.set_script_error(i, "")
                if script["enabled"]:
                    result = lf.scripts.run([script["path"]])
                    if not result["success"]:
                        lf.scripts.set_script_error(i, result["error"])

            layout.pop_id()

    def _reload_all(self):
        lf.scripts.clear_errors()
        enabled_paths = lf.scripts.get_enabled_paths()
        if enabled_paths:
            result = lf.scripts.run(enabled_paths)
            if not result["success"]:
                scripts = lf.scripts.get_scripts()
                for i, script in enumerate(scripts):
                    if script["enabled"]:
                        lf.scripts.set_script_error(i, result["error"])


