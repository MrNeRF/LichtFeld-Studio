"""Minimal LichtFeld plugin.

Copy this directory to ~/.lichtfeld/plugins/hello_world/ and add a pyproject.toml.
"""

import lichtfeld as lf
from lfs_plugins.types import Panel


class HelloPanel(Panel):
    label = "Hello World"
    space = "SIDE_PANEL"
    order = 200

    def draw(self, layout):
        layout.label("Hello from my plugin!")
        if layout.button("Greet"):
            lf.log.info("Hello, LichtFeld!")


_classes = [HelloPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
