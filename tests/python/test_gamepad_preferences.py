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
        tr=lambda key: key,
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
    assert "LY —" in subject.axes()
    now[0] += 0.3
    subject.sync()
    assert subject.buttons() == "—"


def test_controller_diagnostics_are_localized_in_every_shipped_language():
    root = Path(__file__).resolve().parents[2] / "src/visualizer/gui/resources/locales"
    required = {"gamepad." + key for key in ("title", "help", "none", "empty", "unavailable", "unmapped", "axes", "buttons")}
    for path in root.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert all(data.get(key) for key in required), path.name
