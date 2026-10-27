# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


@pytest.fixture
def section(monkeypatch):
    state = dict(available=True, devices=[], selected=0, generation=0, axes=[], buttons=[], pressed=[])
    selected = []
    ui = SimpleNamespace(
        tr=lambda key: "{name} · #{id}" if key == "gamepad.device_label" else key,
        get_gamepad_test_state=lambda: dict(state),
        set_gamepad_test_device=lambda value: selected.append(value),
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", SimpleNamespace(ui=ui))
    path = Path(__file__).resolve().parents[2] / "src/python/lfs_plugins/gamepad_preferences.py"
    spec = importlib.util.spec_from_file_location("gamepad_preferences_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subject = module.GamepadTestSection()
    records, dirty = {}, []
    subject.handle = SimpleNamespace(
        update_record_list=lambda name, rows: records.update({name: rows}),
        dirty=dirty.append,
    )
    return subject, state, selected, records, dirty


def test_empty_state_and_no_calls_to_camera_api(section):
    subject, state, selected, records, dirty = section
    subject.sync()
    assert subject.status() == "gamepad.empty"
    assert records["gamepad_test_devices"] == [{"id": "0", "label": "gamepad.none"}]
    assert subject.axes() == "—"
    assert not selected
    dirty.clear()
    subject.sync()
    assert dirty == []


def test_diagnostics_are_consumed_once_and_shared_with_binding_capture(section, monkeypatch):
    subject, state, _selected, _records, _dirty = section
    ui = sys.modules['lichtfeld'].ui
    reads = []
    state.update(selected=7, raw_pressed=[True])
    def take():
        reads.append('take')
        snapshot = dict(state)
        state['raw_pressed'] = [False]
        return snapshot
    monkeypatch.setattr(ui, 'take_gamepad_test_state', take, raising=False)
    monkeypatch.setattr(ui, 'get_gamepad_test_state', lambda: pytest.fail('use the consuming reader'))
    captured = []
    subject.bindings = SimpleNamespace(sync=lambda: captured.append(subject.state['raw_pressed']))
    subject.sync()
    subject.sync()
    assert reads == ['take', 'take']
    assert captured == [[True], [False]]


def test_identical_names_stay_distinct_and_removal_clears_sample(section):
    subject, state, selected, records, dirty = section
    state.update(devices=[dict(id=11, name="Pad", mapped=True), dict(id=22, name="Pad", mapped=True)],
                 selected=11, generation=1, axes=[0.5, 0, 0, 0, 0, 1], pressed=["south"])
    subject.sync()
    assert len({row["label"] for row in records["gamepad_test_devices"]}) == 3
    assert "LX +0.50" in subject.axes()
    assert subject.buttons() == "south"
    state.update(selected=0, generation=0, axes=[], pressed=[])
    subject.sync()
    assert subject.buttons() == "—"
    assert subject.axes() == "—"
    assert selected == []


def test_selection_is_explicit_and_invalid_ids_are_not_sent(section):
    subject, state, selected, records, dirty = section
    for value in ("not-an-id", -1, 2**32):
        subject.select(value)
    assert selected == []
    subject.select("22")
    subject.select("0")
    assert selected == [22, 0]


def test_mapping_and_backend_errors_remain_visible(section):
    subject, state, selected, records, dirty = section
    state.update(devices=[dict(id=11, name="Pad", mapped=False)])
    subject.sync()
    assert "gamepad.unmapped" in records["gamepad_test_devices"][1]["label"]
    state.update(available=False, error="backend failed")
    subject.sync()
    assert subject.status() == "backend failed"


def test_leaving_input_preferences_stops_diagnostic_sampling(section):
    subject, state, selected, records, dirty = section
    state.update(selected=11, generation=1)
    subject.sync()
    subject.deactivate()
    subject.deactivate()
    assert selected == [0]
    assert subject.state == {}


def test_leaving_the_page_discards_unfinished_device_selection(section):
    subject, _state, selected, _records, _dirty = section
    subject._selection_intent.add("test")
    subject.deactivate()
    subject._select_from_input("test", "22")
    assert selected == []


def test_short_press_expires_and_missing_axis_is_not_reported_as_zero(section, monkeypatch):
    subject, state, selected, records, dirty = section
    now = [10.0]
    # The class function retains its module globals even without module registration.
    monkeypatch.setitem(subject.sync.__globals__, "time", SimpleNamespace(monotonic=lambda: now[0]))
    state.update(selected=11, generation=1, axes=[0] * 6, has_axes=[True, False, True, True, True, True], pressed=["south"])
    subject.sync()
    state["pressed"] = []
    subject.sync()
    assert subject.buttons() == "south"
    assert records["gamepad_test_button_rows"] == [{"label": "south"}]
    assert "LY —" in subject.axes()
    now[0] += 0.3
    subject.sync()
    assert subject.buttons() == "—"
    assert records["gamepad_test_button_rows"] == []


def test_disabled_navigation_is_quiet_but_errors_remain_visible(section):
    subject, state, _selected, _records, _dirty = section
    subject.sync()
    assert not subject.show_navigation_status()
    subject.navigation_error = "configuration failed"
    assert subject.show_navigation_status()
    assert subject.navigation_status() == "configuration failed"
    subject.navigation_error = ""
    state.update(available=False, error="backend failed")
    subject.sync()
    assert subject.show_navigation_status()
    assert subject.navigation_status() == "backend failed"


def test_diagnostic_rows_keep_each_axis_separate_and_mark_missing_controls(section):
    subject, state, _selected, _records, _dirty = section
    state.update(selected=11, axes=[0.5, -0.25, 0, 0.9, 0, 1], has_axes=[True, True, True, False, True, True])
    subject.sync()
    rows = subject.axis_rows()
    assert [row["label"] for row in rows] == ["LX", "LY", "RX", "RY", "LT", "RT"]
    assert [row["group_start"] for row in rows] == [True, False, True, False, True, False]
    assert rows[0]["value"] == "+0.50" and rows[0]["magnitude"] == 0.5
    assert rows[1]["value"] == "-0.25" and rows[1]["magnitude"] == 0.25
    assert rows[3]["value"] == "—" and not rows[3]["available"]
    state.update(selected=0)
    subject.sync()
    assert all(row["value"] == "—" and row["magnitude"] == 0 for row in subject.axis_rows())


def test_sidebar_reset_and_diagnostic_labels_exist_in_every_locale():
    root = Path(__file__).resolve().parents[2] / "src/visualizer/gui/resources/locales"
    keys = ["preferences." + key for key in ("reset_page", "reset_page_confirmation", "reset_all_scope", "reset_safe_mode_note", "open_gamepad_test", "back_to_gamepad")]
    keys += ["input_settings.reset_scope", "gamepad.left_stick", "gamepad.right_stick", "gamepad.triggers"]
    for path in root.glob("*.json"):
        values = json.loads(path.read_text(encoding="utf-8"))
        assert all(values.get(key) for key in keys), path.name


def test_controller_diagnostics_are_localized_in_every_shipped_language():
    root = Path(__file__).resolve().parents[2] / "src/visualizer/gui/resources/locales"
    required = {"gamepad." + key for key in ("title", "help", "none", "empty", "unavailable", "unmapped", "axes", "buttons")}
    for path in root.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert all(data.get(key) for key in required), path.name


def test_navigation_configuration_is_independent_of_diagnostic_selection(section, monkeypatch):
    subject, state, selected, records, dirty = section
    ui = subject.sync.__globals__["lf"].ui
    navigation = dict(enabled=True, selected=22, dead_zone=0.2, look_speed=2.0, move_speed=1.2,
                      status="viewport", mode="walk")
    changes = []
    monkeypatch.setattr(ui, "get_gamepad_navigation_state", lambda: dict(navigation), raising=False)
    monkeypatch.setattr(ui, "set_gamepad_navigation", lambda *args: changes.append(args) or True, raising=False)
    state.update(selected=11, generation=1)
    subject.sync()
    assert subject.navigation_status() == "gamepad.navigation.viewport"
    subject.configure(look_speed=3.0)
    assert changes == [(True, 22, 0.2, 3.0, 1.2)]
    subject.deactivate()
    assert selected == [0]
    assert subject.navigation["selected"] == 22
    assert subject.navigation["enabled"]


def test_native_configuration_failure_is_shown_without_changing_settings(section, monkeypatch):
    subject, state, selected, records, dirty = section
    ui = subject.sync.__globals__["lf"].ui
    monkeypatch.setattr(ui, "set_gamepad_navigation", lambda *args: False, raising=False)
    subject.configure(enabled=True, selected=99)
    assert subject.navigation_status() == "gamepad.unavailable"
    assert "gamepad_navigation_status" in dirty


def test_navigation_status_and_presets_exist_in_every_locale():
    root = Path(__file__).resolve().parents[2] / "src/visualizer/gui/resources/locales"
    keys = {"gamepad.navigation." + state for state in
            ("disabled", "no_device", "suspended", "projection", "mapping", "viewport", "neutral", "ready", "moving")}
    keys.update("gamepad.indicator." + state for state in ("disabled", "no_device", "suspended", "projection", "mapping", "viewport", "neutral", "left", "right"))
    keys.update("gamepad.binding." + key for key in ("shoulder_preset", "shoulder_preset_all", "shoulder_preset_mode", "apply_preset"))
    keys.update("gamepad.preset." + mode for mode in ("orbit", "trackball", "fpv", "drone", "walk"))
    for path in root.glob("*.json"):
        values = json.loads(path.read_text(encoding="utf-8"))
        assert all(values.get(key) for key in keys), path.name


def test_option_rebuild_does_not_replace_native_navigation_selection(section, monkeypatch):
    subject, state, selected, records, dirty = section
    ui = subject.sync.__globals__["lf"].ui
    navigation = dict(enabled=True, selected=22, dead_zone=0.2, look_speed=2.0, move_speed=1.2)
    changes = []
    monkeypatch.setattr(ui, "get_gamepad_navigation_state", lambda: dict(navigation), raising=False)
    monkeypatch.setattr(ui, "set_gamepad_navigation", lambda *args: changes.append(args) or True, raising=False)
    listeners = {channel: {} for channel in ("test", "navigation")}
    elements = {"gamepad-" + channel + "-device": SimpleNamespace(
        add_event_listener=lambda event, callback, c=channel: listeners[c].update({event: callback})
    ) for channel in listeners}
    subject.mount(SimpleNamespace(get_element_by_id=elements.get))
    subject.sync()
    subject._select_from_input("navigation", "0")
    subject._select_from_input("test", "0")
    assert not changes and not selected
    assert subject.navigation["selected"] == 22
    listeners["navigation"]["mousedown"](None)
    subject._select_from_input("navigation", "11")
    assert changes == [(True, 11, 0.2, 2.0, 1.2)]
    subject._select_from_input("navigation", "0")
    assert len(changes) == 1
    listeners["test"]["keydown"](None)
    subject._select_from_input("test", "22")
    assert selected == [22]
    listeners["navigation"]["keydown"](None)
    listeners["navigation"]["blur"](None)
    subject._select_from_input("navigation", "0")
    assert len(changes) == 1

def test_reset_disables_navigation_but_preserves_connected_device_choice(section, monkeypatch):
    subject, state, selected, records, dirty = section
    ui = subject.sync.__globals__['lf'].ui
    subject.navigation = dict(enabled=True, selected=22, dead_zone=.3, look_speed=2., move_speed=3.)
    calls = []
    monkeypatch.setattr(ui, 'set_gamepad_navigation', lambda *args: calls.append(args) or True, raising=False)
    assert subject.reset() is None
    assert calls == [(False, 22, .15, 1.5, 1.)]


def test_general_reset_uses_library_reset_without_touching_device_selection(section, monkeypatch):
    subject, _state, _selected, _records, _dirty = section
    calls = []
    monkeypatch.setattr(sys.modules["lichtfeld"].ui, "set_gamepad_navigation", lambda *args: calls.append(args), raising=False)
    monkeypatch.setattr(sys.modules["lichtfeld"].ui, "reset_gamepad_profiles", lambda: calls.append("library") or True, raising=False)
    assert subject.reset(all_profiles=True) is None
    assert calls == ["library"]


def test_slider_gesture_is_coalesced_and_final_value_is_flushed_on_close(section, monkeypatch):
    subject, _state, _selected, _records, _dirty = section
    ui = sys.modules['lichtfeld'].ui
    navigation = dict(enabled=True, selected=7, dead_zone=.15, look_speed=1.5, move_speed=1., profile_revision=1)
    calls = []
    now = [10.]
    monkeypatch.setitem(subject.sync.__globals__, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(ui, 'get_gamepad_navigation_state', lambda: dict(navigation), raising=False)
    def apply(enabled, selected, dead_zone, look_speed, move_speed):
        calls.append((enabled, selected, dead_zone, look_speed, move_speed))
        navigation.update(dead_zone=dead_zone, look_speed=look_speed, move_speed=move_speed)
        navigation['profile_revision'] += 1
        return True
    monkeypatch.setattr(ui, 'set_gamepad_navigation', apply, raising=False)
    subject.sync()
    for value in (.2, .25, .3):
        subject.stage_tuning('dead_zone', value)
        subject.sync()
    assert calls == []
    assert subject.navigation['dead_zone'] == .3
    now[0] += .3
    subject.sync()
    assert calls == [(True, 7, .3, 1.5, 1.)]
    subject.stage_tuning('move_speed', 2.)
    subject.close()
    assert calls[-1] == (True, 7, .3, 1.5, 2.)


def test_slider_draft_cannot_overwrite_a_new_profile_revision(section, monkeypatch):
    subject, _state, _selected, _records, _dirty = section
    ui = sys.modules['lichtfeld'].ui
    navigation = dict(enabled=False, selected=7, profile_revision=1, move_speed=1.)
    calls = []
    monkeypatch.setattr(ui, 'get_gamepad_navigation_state', lambda: dict(navigation), raising=False)
    monkeypatch.setattr(ui, 'set_gamepad_navigation', lambda *args: calls.append(args), raising=False)
    subject.sync()
    subject.stage_tuning('move_speed', 3.)
    navigation.update(profile_revision=2, move_speed=2.)
    subject.close()
    assert not calls
