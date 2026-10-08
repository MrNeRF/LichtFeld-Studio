# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collapsible section animations must end in the same state as an instant toggle."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


@pytest.fixture
def widgets(monkeypatch):
    source_python = Path(__file__).parent.parent.parent / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))
    sys.modules.pop("lfs_plugins.rml_widgets", None)
    sys.modules.pop("lfs_plugins", None)
    monkeypatch.setitem(sys.modules, "lichtfeld", ModuleType("lichtfeld"))
    return import_module("lfs_plugins.rml_widgets")


class _SectionStub:
    def __init__(self):
        self.classes = set()
        self.attributes = {}
        self.properties = {}
        self.listeners = {}
        self.client_height = 240
        self.scroll_height = 240

    def set_class(self, name, enabled):
        if enabled:
            self.classes.add(name)
        else:
            self.classes.discard(name)

    def set_attribute(self, name, value):
        self.attributes[name] = value

    def get_attribute(self, name, default=""):
        return self.attributes.get(name, default)

    def set_property(self, name, value):
        self.properties[name] = value

    def remove_property(self, name):
        self.properties.pop(name, None)

    def animate(self, prop, target, *_args, **_kwargs):
        self.properties[prop] = target
        return True

    def add_event_listener(self, event, callback):
        self.listeners.setdefault(event, []).append(callback)

    def end_animation(self, prop):
        event = SimpleNamespace(get_parameter=lambda name, default="": prop if name == "property" else default)
        for callback in self.listeners.get("animationend", []):
            callback(event)


def test_collapse_ends_in_the_steady_collapsed_state(widgets):
    # Catches a collapse that leaves the section laid out at zero height with inline styles.
    section = _SectionStub()
    widgets.animate_section_toggle(section, False)

    section.end_animation("opacity")
    assert "collapsed" not in section.classes

    section.end_animation("max-height")
    assert "collapsed" in section.classes
    assert "max-height" not in section.properties
    assert "opacity" not in section.properties
    assert "pointer-events" not in section.properties


def test_reexpanding_before_the_collapse_ends_stays_expanded(widgets):
    section = _SectionStub()
    widgets.animate_section_toggle(section, False)
    widgets.animate_section_toggle(section, True)

    section.end_animation("max-height")

    assert "collapsed" not in section.classes


def test_repeated_collapses_register_one_finish_listener(widgets):
    section = _SectionStub()
    for _ in range(3):
        widgets.animate_section_toggle(section, False)
        widgets.animate_section_toggle(section, True)

    assert len(section.listeners["animationend"]) == 1


class _ArrowStub:
    def __init__(self):
        self.classes = set()
        self.text_writes = 0

    def set_class(self, name, enabled):
        if enabled:
            self.classes.add(name)
        else:
            self.classes.discard(name)

    def set_text(self, _text):
        self.text_writes += 1


def test_collapse_keeps_the_arrow_text_node(widgets):
    # Catches rewriting the arrow glyph: the collapse finishes inside the RmlUi update,
    # after layout, so a new text node would reach Render without a font face.
    section = _SectionStub()
    arrow = _ArrowStub()
    widgets.animate_section_toggle(section, False, arrow)
    section.end_animation("max-height")
    widgets.animate_section_toggle(section, True, arrow)

    assert arrow.text_writes == 0
    assert arrow.classes == {"is-expanded"}
