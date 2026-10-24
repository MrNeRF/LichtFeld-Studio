# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Context changes and live edits must never affect the wrong collider/body."""

from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace

import pytest
from test_toolbar_hooks import _install_stub_modules


class Doc:
    def get_element_by_id(self, _id):
        return None

    def query_selector_all(self, _selector):
        return []

    def add_event_listener(self, *_args):
        pass


@pytest.fixture
def controls(monkeypatch):
    _install_stub_modules(monkeypatch)
    module = import_module("lfs_plugins.collision_rendering")
    target = dict(uuid="a", name="First", opacity=.4, cutaway=1., locked=False, visible=True)
    state = dict(available=True, walking=False, height=1.8, radius=.2, speed=2.5, step=.25, target=None, targets=[target])
    writes, selections = [], []

    def write_walk(**values):
        if values["height"] > 2.4:
            raise RuntimeError("Ceiling clearance is too low")
        writes.append(("walk", values))
        state.update(values)

    def write_preview(uuid, **values):
        writes.append((uuid, values))
        state["target"].update(values)

    monkeypatch.setattr(module, "RuntimeState", SimpleNamespace(language_generation=SimpleNamespace(value=0)))
    monkeypatch.setattr(module, "lf", SimpleNamespace(
        get_collision_controls=lambda: deepcopy(state), set_walk_settings=write_walk,
        set_collision_preview=write_preview, select_node=selections.append,
        ui=SimpleNamespace(tr=lambda key: key),
    ))
    panel = module.CollisionRenderingControls()
    panel.mount(Doc())
    return panel, state, target, writes, selections


def test_rendering_controls_keep_the_collider_without_a_scene_selection(controls):
    panel, state, target, _, _ = controls
    panel.update(Doc())
    assert panel._target_uuid == "a"
    assert panel.update(Doc()) is None
    state["walking"] = True
    panel.update(Doc())
    assert panel._state["walking"]
    state["target"] = None
    panel.update(Doc())
    assert panel._target_uuid == "a"
    state["walking"] = False
    state["targets"] = []
    state["available"] = False
    panel.update(Doc())
    assert not panel._state["target"] and not panel._state["available"]


def test_preview_edit_uses_uuid_and_percentages(controls):
    panel, state, target, writes, _ = controls
    state["target"] = target
    panel.update(Doc())
    panel._set_value("collision_preview_opacity", 65)
    assert writes == [("a", {"opacity": .65, "cutaway": 1.})]
    assert panel._get_value("collision_preview_opacity") == 65
    assert state["height"] == 1.8


def test_unsafe_capsule_edit_preserves_body_and_reports_error(controls):
    panel, state, _, writes, _ = controls
    state["walking"] = True
    panel.update(Doc())
    panel._set_value("collision_walk_height", 2.7)
    assert writes == [] and state["height"] == 1.8
    assert panel._get_value("collision_walk_height") == 1.8
    assert "Ceiling" in panel._error
    panel._set_value("collision_walk_height", 2.)
    assert state["height"] == 2. and panel._error == ""
    assert writes[0][1] == dict(height=2., radius=.2, speed=2.5, step=.25)


def test_context_change_cancels_drafts_before_target_swap(controls, monkeypatch):
    panel, state, target, writes, _ = controls
    state["target"] = target
    panel.update(Doc())
    cancelled = []
    monkeypatch.setattr(panel._scrubs, "prepare_reset", lambda prop: cancelled.append((prop, panel._target_uuid)))
    state["target"] = dict(target, uuid="b", name="Second")
    state["targets"] = [state["target"]]
    panel.update(Doc())
    assert len(cancelled) == 2 and all(old == "a" for _, old in cancelled)
    assert writes == []
    panel._set_value("collision_preview_cutaway", 20)
    assert writes[-1] == ("b", {"opacity": .4, "cutaway": .2})


def test_locked_and_hidden_colliders_remain_explicit(controls):
    panel, state, target, writes, _ = controls
    state["target"] = dict(target, locked=True)
    panel.update(Doc())
    panel._set_value("collision_preview_opacity", 20)
    assert writes == [] and "locked" in panel._error
    assert "locked" in panel._status()
    state["target"] = dict(target, visible=False)
    panel.update(Doc())
    assert panel._state["target"] and "hidden" in panel._status()


def test_dropdown_resolves_fresh_name_and_ignores_deleted_uuid(controls):
    panel, state, target, _, selections = controls
    state["target"] = target
    panel.update(Doc())
    state["targets"] = [dict(target, name="Renamed")]
    panel._select_target("a")
    assert selections == ["Renamed"]
    state["targets"] = []
    panel._select_target("a")
    assert selections == ["Renamed"]


def test_edits_preserve_native_changes_since_last_card_refresh(controls):
    panel, state, target, writes, _ = controls
    state["walking"] = True
    state["target"] = target
    panel.update(Doc())
    state["height"] = 2.1
    panel._set_value("collision_walk_speed", 3.5)
    assert writes[-1][1]["height"] == 2.1
    state["targets"][0]["cutaway"] = .3
    panel._set_value("collision_preview_opacity", 50)
    assert writes[-1] == ("a", {"opacity": .5, "cutaway": .3})


def test_surface_only_scene_keeps_walk_available_without_voxel_preview(controls):
    panel, state, _, _, _ = controls
    state.update(available=True, targets=[], target=None)
    panel.update(Doc())
    assert panel._state["available"]
    assert not panel._state["target"] and not panel._targets
    assert "preview_unavailable" in panel._status()


def test_walk_checkbox_reports_native_rejection_and_can_leave_walk(controls, monkeypatch):
    panel, state, _, _, _ = controls
    panel.update(Doc())
    module = import_module("lfs_plugins.collision_rendering")
    modes = []

    def set_mode(mode):
        modes.append(mode)
        if mode == "walk":
            raise RuntimeError("Load a collision box or collision surface to enable Walk mode")
        state["walking"] = False

    monkeypatch.setattr(module.lf, "set_camera_navigation_mode", set_mode, raising=False)
    panel._set_walk_enabled(True)
    assert "Load a collision" in panel._error
    assert not panel._state["walking"]
    panel._set_walk_enabled(False)
    assert modes == ["walk", "fpv"] and panel._error == ""
