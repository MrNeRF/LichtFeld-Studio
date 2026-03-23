# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Basic layout composition with row, column, box, and split."""

import lichtfeld as lf


class LayoutBasicsPanel(lf.ui.Panel):
    label = "Layout Basics"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 300

    def __init__(self):
        self.opacity = 1.0
        self.threshold = 0.5
        self.name = "Untitled"
        self.is_active = True

    def draw(self, ui):
        with ui.row() as row:
            row.button("Action A")
            row.button("Action B")
            row.button("Action C")

        with ui.box() as box:
            box.heading("Settings")
            changed, self.opacity = box.slider_float("Opacity", self.opacity, 0.0, 1.0)
            changed, self.threshold = box.slider_float("Threshold", self.threshold, 0.0, 1.0)

        with ui.split(0.3) as split:
            split.label("Name")
            changed, self.name = split.input_text("##name", self.name)

        with ui.column() as col:
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
