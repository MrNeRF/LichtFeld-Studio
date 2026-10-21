# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Keyboard editing through the shared scrub controller's mounted listeners."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mounted_scrubs():
    from lfs_plugins.scrub_fields import ScrubFieldController, ScrubFieldSpec

    focused = []

    def element():
        node = MagicMock()
        node.attrs = {}
        node.listeners = {}
        node.get_attribute.side_effect = lambda name, default="": node.attrs.get(name, default)
        node.set_attribute.side_effect = node.attrs.__setitem__
        node.add_event_listener.side_effect = node.listeners.__setitem__
        node.append_child.side_effect = lambda _: element()
        node.focus.side_effect = lambda: focused.append(node)
        node.blur.side_effect = lambda: node.listeners.get("blur", lambda _: None)(None)
        return node

    props = ("histogram_bin_count", "compare_x_bin_count", "compare_y_bin_count")
    values = dict.fromkeys(props, 32)
    committed = []
    controller = ScrubFieldController(
        {prop: ScrubFieldSpec(16 if prop == props[0] else 8, 128 if prop == props[0] else 48, 1, "%d", int)
         for prop in props},
        values.__getitem__, values.__setitem__,
        commit_value=lambda prop, value: committed.append((prop, value)),
    )
    doc = element()
    doc.create_element.side_effect = lambda _: element()
    ranges = [element() for _ in props]
    for prop, node in zip(props, ranges):
        node.attrs["data-value"] = prop
    doc.query_selector_all.return_value = ranges
    controller.mount(doc)
    return controller, values, committed, focused


def key_event(key):
    return SimpleNamespace(get_parameter=lambda name, default="": key if name == "key_identifier" else default,
                           stop_propagation=MagicMock())


@pytest.mark.parametrize("prop", ["histogram_bin_count", "compare_x_bin_count", "compare_y_bin_count"])
@pytest.mark.parametrize("activation", ["KI_RETURN", "KI_SPACE"])
def test_keyboard_can_open_commit_cancel_and_tab_between_bin_scrubs(mounted_scrubs, prop, activation):
    from lfs_plugins import rml_keys as keys

    controller, values, committed, focused = mounted_scrubs
    state = controller._fields[prop]
    activate = key_event(getattr(keys, activation))
    state.field.listeners["keydown"](activate)
    assert state.editing
    assert focused[-1] is state.input_el
    state.input_el.select.assert_called()
    activate.stop_propagation.assert_called_once()

    state.input_el.attrs["value"] = "40"
    change = SimpleNamespace(get_bool_parameter=lambda name, default=False: name == "linebreak")
    state.input_el.listeners["change"](change)
    assert values[prop] == 40
    assert committed == [(prop, 40)]
    assert not state.editing
    assert focused[-1] is state.field

    state.field.listeners["keydown"](key_event(keys.KI_RETURN))
    state.input_el.attrs["value"] = "45"
    state.input_el.listeners["escapecancel"](key_event(keys.KI_ESCAPE))
    state.input_el.blur()  # Native text input blurs after escapecancel.
    assert values[prop] == 40
    assert not state.editing
    assert focused[-1] is state.field

    state.field.listeners["keydown"](key_event(keys.KI_SPACE))
    state.input_el.attrs["value"] = "36"
    tab = key_event(keys.KI_TAB)
    controller._on_keydown(tab)
    tab.stop_propagation.assert_not_called()
    focused.clear()
    state.input_el.blur()  # Tab moves focus to the next control.
    assert values[prop] == 36
    assert not state.editing
    assert focused == []
    assert committed == [(prop, 40), (prop, 36)]


def test_other_keys_and_editor_keydown_do_not_reopen_scrubs(mounted_scrubs):
    from lfs_plugins import rml_keys as keys

    controller, _, _, _ = mounted_scrubs
    state = controller._fields["histogram_bin_count"]
    state.field.listeners["keydown"](key_event(keys.KI_A))
    assert not state.editing
    state.field.listeners["keydown"](key_event(keys.KI_SPACE))
    state.input_el.attrs["value"] = "42"
    state.field.listeners["keydown"](key_event(keys.KI_SPACE))
    assert state.input_el.attrs["value"] == "42"


def test_mouse_drag_does_not_leave_the_scrub_field_with_keyboard_focus(mounted_scrubs):
    controller, values, committed, focused = mounted_scrubs
    state = controller._fields["histogram_bin_count"]

    def mouse(x):
        params = {"button": "0", "mouse_x": str(x)}
        return SimpleNamespace(get_parameter=lambda name, default="": params.get(name, default),
                               get_bool_parameter=lambda name, default=False: default,
                               stop_propagation=MagicMock())

    state.field.listeners["mousedown"](mouse(10))
    controller._on_body_mousemove(mouse(60))
    controller._on_body_mouseup(mouse(60))
    assert values["histogram_bin_count"] != 32
    assert committed == [("histogram_bin_count", values["histogram_bin_count"])]
    assert not state.editing
    # The press focused the field; Space after the drag belongs to the viewport.
    assert focused[-1] is controller._doc


def test_committed_reset_discards_stale_text_before_blur(mounted_scrubs):
    controller, values, _committed, _focused = mounted_scrubs
    prop = "histogram_bin_count"
    state = controller._fields[prop]
    controller._enter_edit(state)
    state.input_el.set_attribute("value", "120")
    controller.prepare_reset(prop)
    values[prop] = 56
    controller.sync_all()
    controller._on_input_blur(prop, None)
    assert values[prop] == 56 and not state.editing
