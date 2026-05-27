# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for viewport selection controls."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_lf_stub(monkeypatch):
    state = SimpleNamespace(
        active_tool="builtin.select",
        active_submode="rectangle",
        has_scene=True,
        has_selection=True,
        depth_enabled=False,
        depth_near=0.25,
        depth_far=7.5,
        depth_width=1.35,
        depth_calls=[],
        stage_calls=[],
        undo_available=True,
        redo_available=True,
        undo_calls=0,
        redo_calls=0,
    )

    class _SceneStub:
        def has_selection(self):
            return state.has_selection

    class _StageStub:
        def __init__(self, name):
            self._name = name

        def execute(self):
            state.stage_calls.append(self._name)
            return {"ok": True, "error": ""}

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        get_active_tool=lambda: state.active_tool,
        get_active_submode=lambda: state.active_submode,
        message_dialog=lambda *_args, **_kwargs: None,
    )
    lf_stub.has_scene = lambda: state.has_scene
    lf_stub.get_scene = lambda: _SceneStub() if state.has_scene else None

    def _set_depth_filter_range(enabled, near, far, width):
        state.depth_enabled = bool(enabled)
        state.depth_near = float(near)
        state.depth_far = float(far)
        state.depth_width = float(width)
        state.depth_calls.append((state.depth_enabled, state.depth_near, state.depth_far, state.depth_width))

    lf_stub.selection = SimpleNamespace(
        get_depth_filter_range=lambda: (
            state.depth_enabled,
            state.depth_near,
            state.depth_far,
            state.depth_width,
        ),
        set_depth_filter_range=_set_depth_filter_range,
    )
    lf_stub.pipeline = SimpleNamespace(
        edit=SimpleNamespace(delete_=lambda: _StageStub("edit.delete")),
        select=SimpleNamespace(
            all=lambda: _StageStub("select.all"),
            invert=lambda: _StageStub("select.invert"),
            none=lambda: _StageStub("select.none"),
        ),
    )
    lf_stub.undo = SimpleNamespace(
        can_undo=lambda: state.undo_available,
        can_redo=lambda: state.redo_available,
        undo=lambda: setattr(state, "undo_calls", state.undo_calls + 1) or True,
        redo=lambda: setattr(state, "redo_calls", state.redo_calls + 1) or True,
    )

    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return state


class _DataModelHandleStub:
    def __init__(self):
        self.dirty_calls = []

    def dirty(self, name):
        self.dirty_calls.append(name)


class _DataModelStub:
    def __init__(self):
        self.bound_binds = {}
        self.bound_funcs = {}
        self.bound_events = {}
        self.handle = _DataModelHandleStub()

    def bind(self, name, getter, setter):
        self.bound_binds[name] = (getter, setter)

    def bind_func(self, name, getter):
        self.bound_funcs[name] = getter

    def bind_event(self, name, callback):
        self.bound_events[name] = callback

    def get_handle(self):
        return self.handle


class _ElementStub:
    def __init__(self):
        self.classes = set()
        self.attributes = {}
        self.listeners = []

    def set_class(self, name, active):
        if active:
            self.classes.add(name)
        else:
            self.classes.discard(name)

    def add_event_listener(self, name, callback):
        self.listeners.append((name, callback))

    def get_attribute(self, name, default=""):
        return self.attributes.get(name, default)

    def set_attribute(self, name, value):
        self.attributes[name] = value


class _DocumentStub:
    def __init__(self):
        self.wrap = _ElementStub()
        self.near = _ElementStub()
        self.far = _ElementStub()

    def get_element_by_id(self, element_id):
        if element_id == "selection-block":
            return self.wrap
        if element_id == "selection-depth-near":
            return self.near
        if element_id == "selection-depth-far":
            return self.far
        return None


@pytest.fixture
def selection_controls_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.selection_controls", None)
    sys.modules.pop("lfs_plugins", None)
    state = _install_lf_stub(monkeypatch)
    module = import_module("lfs_plugins.selection_controls")
    return module, state


def test_selection_controls_show_for_selection_modes(selection_controls_module):
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    model = _DataModelStub()
    doc = _DocumentStub()

    panel.bind_model(model)
    panel.mount(doc)
    panel.update(doc)

    assert "hidden" not in doc.wrap.classes
    assert model.bound_funcs["selection_mode_label"]() == "Rectangle"
    assert model.bound_funcs["selection_has_scene"]() is True
    assert model.bound_funcs["selection_has_selection"]() is True
    assert model.bound_funcs["selection_can_undo"]() is True
    assert model.bound_binds["selection_depth_near_str"][0]() == "0.25"
    assert model.bound_binds["selection_depth_far_str"][0]() == "7.50"
    assert model.bound_funcs["selection_depth_near_slider_min"]() == "0.000"
    assert model.bound_funcs["selection_depth_near_slider_max"]() == "7.490"
    assert model.bound_funcs["selection_depth_far_slider_min"]() == "0.260"
    assert model.bound_funcs["selection_depth_far_slider_max"]() == "27.500"

    state.active_submode = "lasso"
    panel.update(doc)

    assert model.bound_funcs["selection_mode_label"]() == "Lasso"
    assert "selection_mode_label" in model.handle.dirty_calls


def test_selection_depth_toggle_and_sliders_use_selection_api(selection_controls_module):
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    model = _DataModelStub()

    panel.bind_model(model)
    panel.update(_DocumentStub())

    model.bound_events["selection_action"](None, None, ["toggle_depth"])
    assert state.depth_calls[-1] == (True, 0.25, 7.5, 1.35)

    model.bound_binds["selection_depth_near_str"][1]("1.5")
    assert state.depth_calls[-1] == (True, 1.5, 7.5, 1.35)

    model.bound_binds["selection_depth_far_str"][1]("2.0")
    assert state.depth_calls[-1] == (True, 1.5, 2.0, 1.35)


def test_selection_actions_use_undoable_pipeline_and_history(selection_controls_module):
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    model = _DataModelStub()

    panel.bind_model(model)

    model.bound_events["selection_action"](None, None, ["delete"])
    model.bound_events["selection_action"](None, None, ["select_all"])
    model.bound_events["selection_action"](None, None, ["invert"])
    model.bound_events["selection_action"](None, None, ["unselect"])
    model.bound_events["selection_action"](None, None, ["undo"])
    model.bound_events["selection_action"](None, None, ["redo"])

    assert state.stage_calls == ["edit.delete", "select.all", "select.invert", "select.none"]
    assert state.undo_calls == 1
    assert state.redo_calls == 1


def test_selection_controls_hide_when_selection_tool_is_inactive(selection_controls_module):
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    doc = _DocumentStub()

    state.active_tool = "builtin.translate"
    panel.mount(doc)
    doc.wrap.classes.discard("hidden")

    panel.update(doc)

    assert "hidden" in doc.wrap.classes
