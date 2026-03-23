"""Hybrid panel example plugin."""

import lichtfeld as lf

from .panels.main_panel import HybridPanel

_classes = [HybridPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
