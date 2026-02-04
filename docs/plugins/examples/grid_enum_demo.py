# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Grid flow and prop_enum usage."""

import lichtfeld as lf
from lfs_plugins.types import Panel


class GridEnumSettings:
    mode = "fast"


class GridEnumPanel(Panel):
    label = "Grid & Enum Demo"
    space = "SIDE_PANEL"
    order = 301

    def __init__(self):
        self.settings = GridEnumSettings()

    def draw(self, layout):
        with layout.row() as row:
            row.prop_enum(self.settings, "mode", "fast", "Fast")
            row.prop_enum(self.settings, "mode", "balanced", "Balanced")
            row.prop_enum(self.settings, "mode", "quality", "Quality")

        layout.separator()

        items = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
        with layout.grid_flow(columns=3) as grid:
            for item in items:
                with grid.box() as cell:
                    cell.label(item)
                    cell.button("Select##" + item)


_classes = [GridEnumPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
