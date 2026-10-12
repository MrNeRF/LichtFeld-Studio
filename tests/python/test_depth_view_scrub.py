# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Depth range cancellation and shared numeric scrub guards."""

from importlib import import_module
from types import SimpleNamespace

import pytest


class Element:
    absolute_width = 100.0
    absolute_left = 0.0

    def __init__(self):
        self.attrs = {}
        self.classes = set()
        self.text = ""

    def set_class(self, name, value):
        if value:
            self.classes.add(name)
        else:
            self.classes.discard(name)

    def get_attribute(self, name, default=""):
        return self.attrs.get(name, default)

    def set_attribute(self, name, value):
        self.attrs[name] = value

    def get_inner_rml(self):
        return self.text

    def set_text(self, text):
        self.text = text

    def set_property(self, name, value):
        self.attrs[name] = value

    def focus(self):
        pass

    def select(self):
        pass


class Event:
    def __init__(self, **params):
        self.params = params
        self.stopped = False

    def get_parameter(self, name, default):
        return self.params.get(name, default)

    def get_bool_parameter(self, name, default):
        return self.params.get(name, default)

    def stop_propagation(self):
        self.stopped = True


def attach_fields(controller):
    module = import_module("lfs_plugins.scrub_fields")
    for prop, spec in controller._specs.items():
        controller._fields[prop] = module._ScrubFieldState(
            prop, spec, Element(), Element(), Element(), Element())
    return module


@pytest.fixture
def depth(monkeypatch):
    module = import_module("lfs_plugins.depth_view_controls")
    values = [5.0, 200.0]
    lf = SimpleNamespace(
        get_depth_view_range=lambda: tuple(values),
        set_depth_view_range=lambda near, far: values.__setitem__(slice(None), [near, far]),
        get_depth_view_mode=lambda: "palette", has_scene=lambda: True,
    )
    monkeypatch.setattr(module, "lf", lf)
    panel = module.DepthViewControlsController()
    panel._refresh_state()
    keys = attach_fields(panel._scrub_fields)
    return panel, values, keys


@pytest.mark.parametrize("field,initial,x", [
    ("near", [5.0, 200.0], 80),
    ("far", [5000.0, 6000.0], 5),
    ("near", [1.0, 1.001], 80),
    ("far", [9999.999, 10000.0], 5),
])
def test_escape_restores_both_depth_endpoints(depth, field, initial, x):
    panel, values, keys = depth
    values[:] = initial
    panel._refresh_state()
    scrub = panel._scrub_fields
    prop = f"depth_view_{field}_value"
    scrub._on_field_mousedown(prop, Event(mouse_x=50))
    scrub._on_body_mousemove(Event(mouse_x=x))
    assert all(a != b for a, b in zip(values, initial))
    event = Event(key_identifier=keys.KI_ESCAPE)
    scrub._on_keydown(event)
    assert values == initial
    assert event.stopped
    scrub._on_body_mousemove(Event(mouse_x=x + 10))
    scrub._on_body_mouseup(Event())
    assert values == initial
    assert not scrub._fields[prop].editing


@pytest.mark.parametrize("field,x", [("near", 80), ("far", 0)])
def test_depth_drag_release_keeps_existing_pair_adjustment(depth, field, x):
    panel, values, _ = depth
    prop = f"depth_view_{field}_value"
    scrub = panel._scrub_fields
    scrub._on_field_mousedown(prop, Event(mouse_x=50))
    scrub._on_body_mousemove(Event(mouse_x=x))
    expected = list(values)
    assert expected != [5.0, 200.0]
    assert expected[1] >= expected[0] + 0.01
    scrub._on_body_mouseup(Event())
    assert values == expected


def test_depth_text_commit_and_cancel_keep_original_semantics(depth):
    panel, values, keys = depth
    scrub = panel._scrub_fields
    prop = "depth_view_near_value"
    state = scrub._fields[prop]
    state.input_el.blur = lambda: scrub._on_input_blur(prop, Event())
    scrub._on_field_mousedown(prop, Event(mouse_x=50))
    scrub._on_body_mouseup(Event())
    state.input_el.set_attribute("value", "42")
    scrub._on_keydown(Event(key_identifier=keys.KI_ESCAPE))
    assert values == [5.0, 200.0]
    assert not state.editing
    scrub._on_field_mousedown(prop, Event(mouse_x=50))
    scrub._on_body_mouseup(Event())
    state.input_el.set_attribute("value", "42")
    scrub._on_input_change(prop, Event(linebreak=True))
    assert values == [42.0, 200.0]
    assert not state.editing


def test_independent_scalar_drag_cancel_release_and_threshold():
    module = import_module("lfs_plugins.scrub_fields")
    values = {"scale": 0.5}
    scrub = module.ScrubFieldController(
        {"scale": module.ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f")},
        values.__getitem__, values.__setitem__)
    attach_fields(scrub)
    scrub._on_field_mousedown("scale", Event(mouse_x=50))
    scrub._on_body_mousemove(Event(mouse_x=52))
    assert values["scale"] == 0.5
    scrub._on_body_mousemove(Event(mouse_x=83))
    assert values["scale"] == pytest.approx(0.83, abs=1e-15)
    scrub._on_keydown(Event(key_identifier=module.KI_ESCAPE))
    assert values["scale"] == 0.5
    scrub._on_body_mouseup(Event())
    scrub._on_field_mousedown("scale", Event(mouse_x=50))
    scrub._on_body_mousemove(Event(mouse_x=83))
    scrub._on_body_mouseup(Event())
    assert values["scale"] == pytest.approx(0.83, abs=1e-15)
