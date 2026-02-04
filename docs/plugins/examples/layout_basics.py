# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Basic layout composition with row, column, box, and split."""

import lichtfeld as lf
from lfs_plugins.types import Panel


class LayoutBasicsPanel(Panel):
    label = "Layout Basics"
    space = "SIDE_PANEL"
    order = 300

    def __init__(self):
        self.opacity = 1.0
        self.threshold = 0.5
        self.name = "Untitled"
        self.is_active = True

    def draw(self, layout):
        with layout.row() as row:
            row.button("Action A")
            row.button("Action B")
            row.button("Action C")

        with layout.box() as box:
            box.heading("Settings")
            changed, self.opacity = box.slider_float("Opacity", self.opacity, 0.0, 1.0)
            changed, self.threshold = box.slider_float("Threshold", self.threshold, 0.0, 1.0)

        with layout.split(0.3) as split:
            split.label("Name")
            changed, self.name = split.input_text("##name", self.name)

        with layout.column() as col:
            col.enabled = self.is_active
            changed, self.opacity = col.slider_float("Opacity##col", self.opacity, 0.0, 1.0)
            with col.row() as row:
                row.button("Apply")
                row.button("Cancel")


_classes = [LayoutBasicsPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
