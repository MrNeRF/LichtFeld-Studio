# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deep nesting with state cascading and alert."""

import lichtfeld as lf
from lfs_plugins.types import Panel


class NestedLayoutPanel(Panel):
    label = "Nested Layout"
    space = "SIDE_PANEL"
    order = 302

    def __init__(self):
        self.value = 0.5
        self.threshold = 0.8
        self.enabled = True

    def draw(self, layout):
        with layout.box() as outer:
            outer.heading("Outer Box")

            with outer.column() as col:
                col.enabled = self.enabled

                with col.split(0.4) as split:
                    split.label("Value")
                    changed, self.value = split.slider_float("##val", self.value, 0.0, 1.0)

                with col.row() as row:
                    row.button("Reset")
                    row.button("Apply")

                with col.column() as alert_col:
                    alert_col.alert = self.value > self.threshold
                    changed, self.threshold = alert_col.slider_float(
                        "Threshold", self.threshold, 0.0, 1.0
                    )
                    alert_col.label("Normal text after alert")


_classes = [NestedLayoutPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
