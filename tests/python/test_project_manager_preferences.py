# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the canonical Project Manager preference adapter."""

import json
from types import SimpleNamespace

import pytest

from lfs_plugins import project_manager_preferences as preferences


@pytest.fixture
def canonical_store(monkeypatch):
    store = SimpleNamespace(
        preferences={"defaultView": "remember", "rememberState": True},
        state_json="{}",
        reset_count=0,
    )
    monkeypatch.setattr(
        preferences.lf.ui,
        "get_project_manager_preferences",
        lambda: dict(store.preferences),
        raising=False,
    )
    monkeypatch.setattr(
        preferences.lf.ui,
        "set_project_manager_default_view",
        lambda value: store.preferences.__setitem__("defaultView", value),
        raising=False,
    )
    monkeypatch.setattr(
        preferences.lf.ui,
        "set_project_manager_remember_state",
        lambda value: store.preferences.__setitem__("rememberState", value),
        raising=False,
    )
    monkeypatch.setattr(
        preferences.lf.ui,
        "get_project_manager_state",
        lambda: store.state_json,
        raising=False,
    )
    monkeypatch.setattr(
        preferences.lf.ui,
        "set_project_manager_state",
        lambda value: setattr(store, "state_json", value),
        raising=False,
    )

    def reset():
        store.preferences = dict(preferences.DEFAULTS)
        store.state_json = "{}"
        store.reset_count += 1

    monkeypatch.setattr(
        preferences.lf.ui,
        "reset_project_manager_preferences",
        reset,
        raising=False,
    )
    return store


def test_invalid_native_values_fall_back_to_defaults(canonical_store):
    canonical_store.preferences = {"defaultView": "unsupported", "rememberState": "yes"}
    canonical_store.state_json = "not json"

    assert preferences.read_preferences() == preferences.DEFAULTS
    assert preferences.read_state() == {}


def test_preferences_and_state_round_trip_through_canonical_api(canonical_store):
    preferences.set_preference("defaultView", "gallery")
    preferences.set_preference("rememberState", False)
    preferences.set_state({"view_mode": "list", "navigator_width": 312.5})

    assert preferences.read_preferences() == {
        "defaultView": "gallery",
        "rememberState": False,
    }
    assert preferences.read_state() == {
        "view_mode": "list",
        "navigator_width": 312.5,
    }
    assert json.loads(canonical_store.state_json)["view_mode"] == "list"


def test_reset_is_delegated_to_canonical_store(canonical_store):
    preferences.set_preference("defaultView", "list")
    preferences.set_state({"view_mode": "list"})

    preferences.reset_preferences()

    assert canonical_store.reset_count == 1
    assert preferences.read_preferences() == preferences.DEFAULTS
    assert preferences.read_state() == {}


@pytest.mark.parametrize(
    ("key", "value"),
    [("defaultView", "tiles"), ("rememberState", 1), ("unknown", True)],
)
def test_invalid_preferences_are_rejected(canonical_store, key, value):
    with pytest.raises(ValueError):
        preferences.set_preference(key, value)


def test_non_finite_state_is_rejected_before_reaching_native_store(canonical_store):
    with pytest.raises(ValueError):
        preferences.set_state({"navigator_width": float("nan")})
