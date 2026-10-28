# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Depth range cancellation and shared numeric scrub guards."""

from importlib import import_module
import sys
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


def test_scrub_drag_acceleration_modifiers_and_mid_drag_change():
    module = import_module("lfs_plugins.scrub_fields")
    rows = [
        ("plain", [(10, {})], 50),
        ("shift", [(5, {"shift_key": True})], 90),
        ("ctrl", [(5, {"ctrl_key": True})], 90),
        ("change", [(5, {}), (2, {"shift_key": True}), (1, {"ctrl_key": True})], 75),
    ]
    for _, moves, expected in rows:
        values = {"speed": 40}
        scrub = module.ScrubFieldController(
            {"speed": module.ScrubFieldSpec(0.0, 100.0, 1.0, "%d", data_type=int)},
            values.__getitem__, values.__setitem__)
        attach_fields(scrub)
        scrub._on_field_mousedown("speed", Event(mouse_x=40))
        mouse_x = 40
        for delta, modifiers in moves:
            mouse_x += delta
            scrub._on_body_mousemove(Event(mouse_x=mouse_x, **modifiers))
        scrub._on_body_mouseup(Event())
        assert values["speed"] == expected


@pytest.mark.parametrize("language,text,expected", [
    ("de", "0,5", 0.5),
    ("fr", "0,5", 0.5),
    ("es", "0,5", 0.5),
    ("it", "0,5", 0.5),
    ("nl", "0,5", 0.5),
    ("pl", "0,5", 0.5),
    ("de", "0.5", 0.5),
    ("fr", "0.5", 0.5),
    ("en", "0.5", 0.5),
    ("en", "0,5", 1.0),
])
def test_numeric_text_uses_ui_decimal_separator(monkeypatch, language, text, expected):
    module = import_module("lfs_plugins.scrub_fields")
    monkeypatch.setitem(sys.modules, "lichtfeld", SimpleNamespace(
        ui=SimpleNamespace(get_current_language=lambda: language)))
    values = {"scale": 1.0}
    scrub = module.ScrubFieldController(
        {"scale": module.ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f")},
        values.__getitem__, values.__setitem__)
    attach_fields(scrub)
    state = scrub._fields["scale"]
    state.editing = True
    state.input_el.set_attribute("value", text)

    scrub._commit_edit_value(state)

    assert values["scale"] == expected


def test_numeric_text_commits_when_tab_moves_focus():
    module = import_module("lfs_plugins.scrub_fields")
    values = {"scale": 1.0}
    scrub = module.ScrubFieldController(
        {"scale": module.ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f")},
        values.__getitem__, values.__setitem__)
    attach_fields(scrub)
    state = scrub._fields["scale"]
    state.editing = True
    state.input_el.set_attribute("value", "0.5")
    state.input_el.blur = lambda: scrub._on_input_blur("scale", Event())

    event = Event(key_identifier=70)  # Rml::Input::KI_TAB
    scrub._on_keydown(event)
    state.input_el.blur()

    assert values["scale"] == 0.5
    assert not state.editing
    assert not event.stopped

def test_numeric_text_commits_when_mouse_click_moves_focus_away():
    module = import_module("lfs_plugins.scrub_fields")
    values = {"scale": 1.0}
    scrub = module.ScrubFieldController(
        {"scale": module.ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f")},
        values.__getitem__, values.__setitem__)
    attach_fields(scrub)
    state = scrub._fields["scale"]
    state.editing = True
    state.input_el.set_attribute("value", "0.5")

    # A mouse click on another control causes blur without a key event.
    state.input_el.blur = lambda: scrub._on_input_blur("scale", Event())
    state.input_el.blur()

    assert values["scale"] == 0.5
    assert not state.editing


def test_escape_then_mouse_click_away_keeps_numeric_edit_cancelled():
    module = import_module("lfs_plugins.scrub_fields")
    values = {"scale": 1.0}
    scrub = module.ScrubFieldController(
        {"scale": module.ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f")},
        values.__getitem__, values.__setitem__)
    attach_fields(scrub)
    state = scrub._fields["scale"]
    state.editing = True
    state.edit_value_before = 1.0
    state.input_el.set_attribute("value", "0.5")
    state.input_el.blur = lambda: scrub._on_input_blur("scale", Event())

    scrub._on_keydown(Event(key_identifier=module.KI_ESCAPE))
    # The click-away blur after Escape must not reapply the discarded text.
    state.input_el.blur()

    assert values["scale"] == 1.0
    assert not state.editing

def test_escapecancel_in_focused_numeric_text_restores_the_value():
    module = import_module("lfs_plugins.scrub_fields")
    values = {"scale": 1.0}
    scrub = module.ScrubFieldController(
        {"scale": module.ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f")},
        values.__getitem__, values.__setitem__)
    attach_fields(scrub)
    state = scrub._fields["scale"]
    state.editing = True
    state.edit_value_before = 1.0
    state.input_el.set_attribute("value", "0.5")

    # The host sends escapecancel to the focused field, then blurs it.
    scrub._on_input_escape("scale", Event())
    scrub._on_input_blur("scale", Event())

    assert values["scale"] == 1.0
    assert not state.editing


@pytest.mark.parametrize("text,expected", [
    ("1e309", 100),
    ("inf", 100),
    ("-inf", 1),
    ("999999999999999999999999999999", 100),
    ("12", 12),
    ("nan", 11),
])
def test_integer_scrub_handles_out_of_range_text_without_callback_exception(text, expected):
    module = import_module("lfs_plugins.scrub_fields")
    values = {"speed": 11}
    scrub = module.ScrubFieldController(
        {"speed": module.ScrubFieldSpec(1, 100, 1, "%d", data_type=int)},
        values.__getitem__, values.__setitem__)
    attach_fields(scrub)
    state = scrub._fields["speed"]
    state.editing = True
    state.input_el.set_attribute("value", text)
    state.input_el.blur = lambda: scrub._on_input_blur("speed", Event())

    scrub._on_input_change("speed", Event(linebreak=True))

    assert values["speed"] == expected
    assert not state.editing


def test_float_scrub_keeps_prior_value_for_non_finite_text():
    module = import_module("lfs_plugins.scrub_fields")
    values = {"scale": 0.5}
    scrub = module.ScrubFieldController(
        {"scale": module.ScrubFieldSpec(0.25, 1.0, 0.01, "%.2f")},
        values.__getitem__,
        values.__setitem__,
    )
    attach_fields(scrub)
    state = scrub._fields["scale"]
    state.input_el.blur = lambda: scrub._on_input_blur("scale", Event())

    for text in ("nan", "NaN", "inf", "-inf", "abc", ""):
        values["scale"] = 0.5
        state.editing = True
        state.edit_value_before = 0.5
        state.input_el.set_attribute("value", text)
        scrub._on_input_change("scale", Event(linebreak=True))
        assert values["scale"] == 0.5, text

