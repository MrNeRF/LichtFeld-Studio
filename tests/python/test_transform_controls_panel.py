# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for visualizer-world transform controls."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _translation_matrix(x, y, z):
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        float(x), float(y), float(z), 1.0,
    ]


def _translation_from_matrix(matrix):
    return [float(matrix[12]), float(matrix[13]), float(matrix[14])]


def _decompose_transform(matrix):
    translation = _translation_from_matrix(matrix)
    return {
        "translation": translation,
        "rotation_quat": [0.0, 0.0, 0.0, 1.0],
        "rotation_euler": [0.0, 0.0, 0.0],
        "rotation_euler_deg": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }


def _install_lf_stub(monkeypatch):
    state = SimpleNamespace(
        active_tool="builtin.translate",
        selected_names=[],
        local_transforms={},
        visualizer_world_transforms={},
        selection_visualizer_world_center=None,
        selection_world_center=None,
        set_visualizer_world_calls=[],
        set_local_calls=[],
        op_calls=[],
    )

    class _Panel:
        def on_mount(self, _doc):
            pass

    panel_space = SimpleNamespace(
        SIDE_PANEL="SIDE_PANEL",
        FLOATING="FLOATING",
        VIEWPORT_OVERLAY="VIEWPORT_OVERLAY",
        MAIN_PANEL_TAB="MAIN_PANEL_TAB",
        SCENE_HEADER="SCENE_HEADER",
        STATUS_BAR="STATUS_BAR",
    )
    panel_height_mode = SimpleNamespace(FILL="fill", CONTENT="content")

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        Panel=_Panel,
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        tr=lambda key: key,
        get_active_tool=lambda: state.active_tool,
        is_ctrl_down=lambda: False,
        set_panel_parent=lambda _panel_id, _parent: None,
    )
    lf_stub.ops = SimpleNamespace(
        invoke=lambda operator_id, **kwargs: state.op_calls.append((operator_id, kwargs))
    )
    lf_stub.get_selected_node_names = lambda: list(state.selected_names)
    lf_stub.get_selection_visualizer_world_center = lambda: (
        list(state.selection_visualizer_world_center)
        if state.selection_visualizer_world_center is not None else None
    )
    lf_stub.get_selection_world_center = lambda: (
        list(state.selection_world_center)
        if state.selection_world_center is not None else None
    )
    lf_stub.get_node_transform = lambda name: state.local_transforms.get(name)
    lf_stub.get_node_visualizer_world_transform = lambda name: state.visualizer_world_transforms.get(name)

    def _set_node_transform(name, matrix):
        state.set_local_calls.append((name, list(matrix)))
        state.local_transforms[name] = list(matrix)

    def _set_node_visualizer_world_transform(name, matrix):
        state.set_visualizer_world_calls.append((name, list(matrix)))
        state.visualizer_world_transforms[name] = list(matrix)

    lf_stub.set_node_transform = _set_node_transform
    lf_stub.set_node_visualizer_world_transform = _set_node_visualizer_world_transform
    lf_stub.decompose_transform = _decompose_transform
    lf_stub.compose_transform = lambda translation, euler_deg, scale: _translation_matrix(*translation)
    lf_stub.register_class = lambda _cls: None

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
        self.listeners = []

    def set_class(self, name, active):
        if active:
            self.classes.add(name)
        else:
            self.classes.discard(name)

    def add_event_listener(self, name, callback):
        self.listeners.append((name, callback))


class _DocumentStub:
    def __init__(self):
        self.body = _ElementStub()
        self.wrap = _ElementStub()

    def get_element_by_id(self, element_id):
        if element_id in {"body", "overlay-body"}:
            return self.body
        if element_id == "transform-block":
            return self.wrap
        return None


@pytest.fixture
def transform_controls_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.transform_controls", None)
    sys.modules.pop("lfs_plugins", None)
    state = _install_lf_stub(monkeypatch)
    module = import_module("lfs_plugins.transform_controls")
    return module, state


def test_transform_controls_single_node_reads_visualizer_world_transform(transform_controls_module):
    module, state = transform_controls_module
    panel = module.TransformControlsController()
    panel._selected = ["target"]

    state.local_transforms["target"] = _translation_matrix(1.0, 2.0, 3.0)
    state.visualizer_world_transforms["target"] = _translation_matrix(1.0, -2.0, -3.0)

    panel._update_single_node()

    assert panel._trans == [1.0, -2.0, -3.0]


def test_transform_controls_single_node_writes_visualizer_world_transform(transform_controls_module):
    module, state = transform_controls_module
    panel = module.TransformControlsController()
    panel._selected = ["target"]
    panel._active_tool = "builtin.translate"
    panel._trans = [4.0, -5.0, -6.0]
    panel._euler = [0.0, 0.0, 0.0]
    panel._scale = [1.0, 1.0, 1.0]

    state.visualizer_world_transforms["target"] = _translation_matrix(1.0, -2.0, -3.0)

    panel._apply_single_transform()

    assert state.set_local_calls == []
    assert state.set_visualizer_world_calls == [("target", _translation_matrix(4.0, -5.0, -6.0))]


def test_transform_controls_multi_translate_uses_visualizer_world_space(transform_controls_module):
    module, state = transform_controls_module
    panel = module.TransformControlsController()
    panel._selected = ["left", "right"]
    panel._active_tool = "builtin.translate"

    state.selection_visualizer_world_center = [10.0, -20.0, -30.0]
    state.local_transforms["left"] = _translation_matrix(1.0, 2.0, 3.0)
    state.local_transforms["right"] = _translation_matrix(4.0, 5.0, 6.0)
    state.visualizer_world_transforms["left"] = _translation_matrix(10.0, -20.0, -30.0)
    state.visualizer_world_transforms["right"] = _translation_matrix(15.0, -24.0, -33.0)

    panel._begin_edit()
    panel._trans = [11.0, -18.0, -29.0]
    panel._state.display_translation = panel._trans.copy()

    panel._apply_multi_transform("builtin.translate")

    assert state.set_local_calls == []
    assert state.set_visualizer_world_calls == [
        ("left", _translation_matrix(11.0, -18.0, -29.0)),
        ("right", _translation_matrix(16.0, -22.0, -32.0)),
    ]


def test_transform_controls_hide_overlay_when_tool_is_inactive(transform_controls_module):
    module, state = transform_controls_module
    panel = module.TransformControlsController()
    state.active_tool = ""
    state.selected_names = ["target"]

    doc = _DocumentStub()

    panel.mount(doc)
    assert "hidden" in doc.wrap.classes

    doc.wrap.classes.discard("hidden")
    panel.update(doc)

    assert "hidden" in doc.wrap.classes


def test_transform_controls_bake_commits_active_edit(transform_controls_module):
    module, state = transform_controls_module
    panel = module.TransformControlsController()
    panel._selected = ["target"]

    old_transform = _translation_matrix(1.0, 2.0, 3.0)
    new_transform = _translation_matrix(4.0, 5.0, 6.0)
    state.local_transforms["target"] = new_transform
    panel._state.editing_active = True
    panel._state.editing_node_names = ["target"]
    panel._state.transforms_before_edit = [old_transform]

    panel._on_action(None, None, ["bake"])

    assert state.op_calls == [
        (
            "transform.apply_batch",
            {
                "node_names": ["target"],
                "old_transforms": [old_transform],
            },
        )
    ]
    assert panel._state.editing_active is False
