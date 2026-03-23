"""Minimal LichtFeld plugin.

Copy this directory to ~/.lichtfeld/plugins/hello_world/ and add a pyproject.toml.
"""

import lichtfeld as lf


class HelloPanel(lf.ui.Panel):
    label = "Hello World"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 200

    def draw(self, ui):
        ui.label("Hello from my plugin!")
        if ui.button("Greet"):
            lf.log.info("Hello, LichtFeld!")


_classes = [HelloPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
