# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for visualizer-world transform controls."""

from importlib import import_module
import math
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


def _rotation_matrix(euler_deg):
    x, y, z = (math.radians(value) for value in euler_deg)
    cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
    sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
    return [
        [cy * cz, -cy * sz, sy],
        [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
        [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
    ]


def _multiply3(lhs, rhs):
    return [[sum(lhs[i][k] * rhs[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _transform_matrix(translation, euler_deg, scale):
    rotation = _rotation_matrix(euler_deg)
    rows = [[rotation[r][c] * scale[c] for c in range(3)] + [translation[r]] for r in range(3)]
    rows.append([0.0, 0.0, 0.0, 1.0])
    return [rows[r][c] for c in range(4) for r in range(4)]


def _matrix_rows(matrix):
    return [[matrix[c * 4 + r] for c in range(4)] for r in range(4)]


def _multiply4(lhs, rhs):
    return [[sum(lhs[i][k] * rhs[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _around_pivot_matrix(euler_deg, pivot):
    rotation = _rotation_matrix(euler_deg)
    rows = [[rotation[r][c] for c in range(3)] + [0.0] for r in range(3)]
    rows.append([0.0, 0.0, 0.0, 1.0])
    for r in range(3):
        rows[r][3] = pivot[r] - sum(rotation[r][c] * pivot[c] for c in range(3))
    return rows


def _frobenius_delta(lhs, rhs):
    return math.sqrt(sum((lhs[i] - rhs[i]) ** 2 for i in range(16)))


def _reference_old_multi_rotation(originals, decompositions, delta_deg, pivot):
    """The pre-fix Euler-addition implementation, retained as a guard oracle."""
    delta_rotation = _rotation_matrix(delta_deg)
    result = []
    for original, decomp in zip(originals, decompositions):
        pos = decomp["translation"]
        rel = [pos[j] - pivot[j] for j in range(3)]
        new_pos = [
            pivot[row] + sum(delta_rotation[row][col] * rel[col] for col in range(3))
            for row in range(3)
        ]
        new_euler = [decomp["rotation_euler_deg"][j] + delta_deg[j] for j in range(3)]
        result.append(_transform_matrix(new_pos, new_euler, decomp["scale"]))
    return result


def _multi_rotate_panel(module, state, orientations, translations=None, scales=None):
    names = [f"node_{i}" for i in range(len(orientations))]
    translations = translations or [[float(i * 2 - 2), 0.5 * i, -0.25 * i] for i in range(len(names))]
    scales = scales or [[0.2 + 0.1 * i, 0.3 + 0.1 * i, 0.4 + 0.1 * i] for i in range(len(names))]
    originals = [
        _transform_matrix(translations[i], orientations[i], scales[i]) for i in range(len(names))
    ]
    decompositions = {
        tuple(matrix): {
            "translation": list(translations[i]),
            "rotation_euler_deg": list(orientations[i]),
            "scale": list(scales[i]),
        }
        for i, matrix in enumerate(originals)
    }
    module.lf.decompose_transform = lambda matrix: decompositions[tuple(matrix)]
    module.lf.compose_transform = _transform_matrix
    panel = module.TransformControlsController()
    panel._selected = names
    panel._active_tool = "builtin.rotate"
    panel._euler = [0.0, 0.0, 0.0]
    panel._state.multi_editing_active = True
    panel._state.multi_node_names = names
    panel._state.multi_visualizer_world_transforms_before = originals
    panel._state.pivot_world = [sum(pos[axis] for pos in translations) / len(names) for axis in range(3)]
    return panel, originals, [decompositions[tuple(matrix)] for matrix in originals]


def _apply_rotation_input(panel, euler_deg):
    for axis, value in enumerate(euler_deg):
        panel._set_value("rot", axis, str(value))


def _reference_old_single_rotation(translation, euler_deg, scale):
    return _transform_matrix(translation, euler_deg, scale)


def _reference_previous_multi_transform(tool, originals, decompositions, delta, pivot):
    """Behavior before Individual-mode handling was added to the controller."""
    result = []
    for original, decomp in zip(originals, decompositions):
        if tool == "builtin.rotate":
            delta_matrix = _around_pivot_matrix(delta, pivot)
            rows = _multiply4(delta_matrix, _matrix_rows(original))
            result.append([rows[r][c] for c in range(4) for r in range(4)])
            continue

        pos = decomp["translation"]
        rel = [pos[j] - pivot[j] for j in range(3)]
        new_pos = [pivot[j] + rel[j] * delta[j] for j in range(3)]
        scale = [decomp["scale"][j] * delta[j] for j in range(3)]
        result.append(_transform_matrix(new_pos, decomp["rotation_euler_deg"], scale))
    return result


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
        multi_transform_mode=0,
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
        get_multi_transform_mode=lambda: state.multi_transform_mode,
        MULTI_TRANSFORM_MODE_SELECTION=0,
        MULTI_TRANSFORM_MODE_INDIVIDUAL=1,
    )
    lf_stub.ops = SimpleNamespace(
        invoke=lambda operator_id, **kwargs: state.op_calls.append((operator_id, kwargs))
    )
    lf_stub.can_transform_selection = lambda: getattr(state, "editable", True)
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


def test_transform_controls_multi_rotation_composes_existing_orientation(transform_controls_module):
    module, state = transform_controls_module
    panel = module.TransformControlsController()
    panel._selected = ["left", "right"]
    panel._active_tool = "builtin.rotate"
    panel._state.multi_node_names = ["left", "right"]
    panel._state.pivot_world = [0.0, 0.0, 0.0]
    panel._state.display_euler = [0.0, 0.0, 90.0]
    original_euler = [30.0, 20.0, 10.0]
    for position in (-2.0, 2.0):
        matrix = _transform_matrix([position, 0.0, 0.0], original_euler, [1.0, 1.0, 1.0])
        panel._state.multi_visualizer_world_transforms_before.append(matrix)
    original = _translation_matrix(-2.0, 0.0, 0.0)
    original_decomp = {
        "translation": [-2.0, 0.0, 0.0],
        "rotation_euler_deg": original_euler,
        "scale": [1.0, 1.0, 1.0],
    }
    module.lf.decompose_transform = lambda _matrix: original_decomp
    panel._apply_multi_transform("builtin.rotate")

    expected = _multiply3(_rotation_matrix([0.0, 0.0, 90.0]), _rotation_matrix(original_euler))
    actual_matrix = state.visualizer_world_transforms["left"]
    actual = [[actual_matrix[col * 4 + row] for col in range(3)] for row in range(3)]
    assert [value for row in actual for value in row] == pytest.approx(
        [value for row in expected for value in row], abs=1e-7
    )


def test_transform_controls_multi_rotation_sweeps_continuously_through_euler_wraps(transform_controls_module):
    module, state = transform_controls_module
    orientations = [[17.0, 23.0, 11.0], [-34.0, 4.0, 28.0], [5.0, -19.0, 73.0]]
    panel, originals, decompositions = _multi_rotate_panel(module, state, orientations)
    pivot = panel._state.pivot_world
    angles = [0.0] + [step * 0.5 for step in range(1, 361)]
    angles += [step * 0.5 for step in range(359, -361, -1)]

    for axis in range(3):
        panel._euler = [0.0, 0.0, 0.0]
        panel._set_value("rot", axis, "0")
        previous = {name: list(matrix) for name, matrix in zip(panel._selected, originals)}
        previous_angle = 0.0

        for angle in angles[1:]:
            panel._set_value("rot", axis, str(angle))
            euler = [0.0, 0.0, 0.0]
            euler[axis] = angle
            for name, original, decomp in zip(panel._selected, originals, decompositions):
                actual = state.visualizer_world_transforms[name]
                delta = _around_pivot_matrix(euler, pivot)
                expected_rows = _multiply4(delta, _matrix_rows(original))
                expected = [expected_rows[r][c] for c in range(4) for r in range(4)]
                assert actual == pytest.approx(expected, abs=1e-5)

                step_radians = math.radians(abs(angle - previous_angle))
                basis_norm = math.sqrt(sum(value * value for value in original[:12]))
                radius = math.sqrt(sum((decomp["translation"][j] - pivot[j]) ** 2 for j in range(3)))
                assert _frobenius_delta(actual, previous[name]) <= (
                    2.0 * step_radians * (basis_norm + radius) + 1e-8
                )
                previous[name] = list(actual)
            previous_angle = angle

        assert -90.0 in angles and 90.0 in angles and -180.0 in angles and 180.0 in angles


def test_transform_controls_multi_rotation_preserves_old_correct_cases(transform_controls_module):
    module, state = transform_controls_module
    cases = (
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [23.0, -11.0, 48.0]),
        ([[0.0, 0.0, 30.0], [0.0, 0.0, -15.0]], [0.0, 0.0, 42.0]),
        ([[17.0, 23.0, 11.0], [-34.0, 4.0, 28.0]], [0.0, 0.0, 0.0]),
    )
    for orientations, delta in cases:
        panel, originals, decompositions = _multi_rotate_panel(module, state, orientations)
        _apply_rotation_input(panel, delta)
        actual = [state.visualizer_world_transforms[name] for name in panel._selected]
        reference = _reference_old_multi_rotation(
            originals, decompositions, delta, panel._state.pivot_world
        )
        for result, old_result, original in zip(actual, reference, originals):
            assert result == pytest.approx(old_result, abs=1e-6)
            if delta == [0.0, 0.0, 0.0]:
                assert result == original

    # The single-node route remains the old absolute-Euler compose behavior.
    panel = module.TransformControlsController()
    panel._selected = ["single"]
    panel._active_tool = "builtin.rotate"
    original = _transform_matrix([1.0, 2.0, 3.0], [13.0, -21.0, 34.0], [0.5, 0.75, 1.25])
    decomp = {
        "translation": [1.0, 2.0, 3.0],
        "rotation_quat": [0.0, 0.0, 0.0, 1.0],
        "rotation_euler_deg": [13.0, -21.0, 34.0],
        "scale": [0.5, 0.75, 1.25],
    }
    module.lf.decompose_transform = lambda _matrix: decomp
    state.visualizer_world_transforms["single"] = original
    panel._euler = [71.0, -2.0, 19.0]
    panel._apply_single_transform()
    assert state.visualizer_world_transforms["single"] == pytest.approx(
        _reference_old_single_rotation(decomp["translation"], panel._euler, decomp["scale"]), abs=1e-6
    )


def test_transform_controls_multi_rotation_preserves_sheared_world_basis(transform_controls_module):
    module, state = transform_controls_module
    names = ["left", "right"]
    original_rows = [
        [[1.0, 0.2, 0.3, -2.0], [0.4, 1.5, 0.1, 0.5], [0.0, 0.2, 1.2, 0.0], [0, 0, 0, 1]],
        [[0.8, 0.3, 0.1, 2.0], [0.2, 1.4, 0.4, -0.5], [0.1, 0.0, 1.1, 0.0], [0, 0, 0, 1]],
    ]
    originals = [[rows[r][c] for c in range(4) for r in range(4)] for rows in original_rows]
    panel = module.TransformControlsController()
    panel._selected = names
    panel._active_tool = "builtin.rotate"
    panel._state.multi_editing_active = True
    panel._state.multi_node_names = names
    panel._state.multi_visualizer_world_transforms_before = originals
    panel._state.pivot_world = [0.0, 0.0, 0.0]
    module.lf.decompose_transform = lambda _matrix: pytest.fail("rotation must preserve the full basis")
    module.lf.compose_transform = lambda *_args: pytest.fail("rotation must not rebuild a TRS matrix")

    panel._set_value("rot", 2, "15")

    delta = _around_pivot_matrix([0.0, 0.0, 15.0], [0.0, 0.0, 0.0])
    for name, original in zip(names, originals):
        expected_rows = _multiply4(delta, _matrix_rows(original))
        expected = [expected_rows[r][c] for c in range(4) for r in range(4)]
        assert state.visualizer_world_transforms[name] == pytest.approx(expected, abs=1e-8)


def test_transform_controls_multi_translation_and_scale_are_continuous(transform_controls_module):
    module, state = transform_controls_module
    orientations = [[17.0, 23.0, 11.0], [-34.0, 4.0, 28.0]]
    names = ["node_0", "node_1"]
    translations = [[-2.0, 0.5, 0.0], [2.0, -0.5, 0.0]]
    scales = [[0.2, 0.3, 0.4], [0.5, 0.4, 0.3]]
    originals = [_transform_matrix(translations[i], orientations[i], scales[i]) for i in range(2)]
    decompositions = {
        tuple(matrix): {"translation": translations[i], "rotation_euler_deg": orientations[i], "scale": scales[i]}
        for i, matrix in enumerate(originals)
    }
    module.lf.decompose_transform = lambda matrix: decompositions[tuple(matrix)]
    module.lf.compose_transform = _transform_matrix

    def make_panel(tool):
        panel = module.TransformControlsController()
        panel._selected = names
        panel._active_tool = tool
        panel._state.multi_editing_active = True
        panel._state.multi_node_names = names
        panel._state.multi_visualizer_world_transforms_before = originals
        panel._state.pivot_world = [0.0, 0.0, 0.0]
        return panel

    panel = make_panel("builtin.translate")
    panel._trans = [0.0, 0.0, 0.0]
    previous = {name: list(matrix) for name, matrix in zip(names, originals)}
    for value in [step * 0.5 for step in range(11)]:
        panel._set_value("pos", 0, str(value))
        for name, original in zip(names, originals):
            actual = state.visualizer_world_transforms[name]
            expected = list(original)
            expected[12] += value
            assert actual == pytest.approx(expected, abs=1e-8)
            assert _frobenius_delta(actual, previous[name]) <= 0.5 + 1e-8
            previous[name] = list(actual)

    panel = make_panel("builtin.scale")
    panel._scale = [1.0, 1.0, 1.0]
    previous = {name: list(matrix) for name, matrix in zip(names, originals)}
    previous_scale = 1.0
    for scale in [1.0 + step * 0.01 for step in range(11)]:
        panel._set_uniform_scale(str(scale))
        for name, original, translation in zip(names, originals, translations):
            actual = state.visualizer_world_transforms[name]
            expected = [value * scale if i < 12 else value for i, value in enumerate(original)]
            for axis in range(3):
                expected[12 + axis] = translation[axis] * scale
            assert actual == pytest.approx(expected, abs=1e-8)
            basis_norm = math.sqrt(sum(value * value for value in original[:12]))
            radius = math.sqrt(sum(value * value for value in translation))
            assert _frobenius_delta(actual, previous[name]) <= (
                1.01 * abs(scale - previous_scale) * (basis_norm + radius) + 1e-8
            )
            previous[name] = list(actual)
        previous_scale = scale


def test_transform_controls_individual_scale_preserves_each_origin(transform_controls_module):
    module, state = transform_controls_module
    panel = module.TransformControlsController()
    panel._selected = ["left", "right"]
    panel._active_tool = "builtin.scale"
    panel._state.multi_node_names = ["left", "right"]
    panel._state.pivot_world = [0.0, 0.0, 0.0]
    panel._state.display_scale = [1.5, 1.5, 1.5]
    state.multi_transform_mode = module.lf.ui.MULTI_TRANSFORM_MODE_INDIVIDUAL
    decompositions = {
        -2.0: {"translation": [-2.0, 0.0, 0.0], "rotation_euler_deg": [0.0] * 3, "scale": [1.0] * 3},
        2.0: {"translation": [2.0, 0.0, 0.0], "rotation_euler_deg": [0.0] * 3, "scale": [1.0] * 3},
    }
    panel._state.multi_visualizer_world_transforms_before = [
        _translation_matrix(-2.0, 0.0, 0.0),
        _translation_matrix(2.0, 0.0, 0.0),
    ]
    module.lf.decompose_transform = lambda _matrix: pytest.fail("Individual scale must preserve the full matrix")
    module.lf.compose_transform = lambda *_args: pytest.fail("Individual scale must not rebuild a TRS matrix")

    panel._apply_multi_transform("builtin.scale")

    for name, original in zip(panel._selected, panel._state.multi_visualizer_world_transforms_before):
        actual = state.visualizer_world_transforms[name]
        expected = list(original)
        for offset in (0, 4, 8):
            expected[offset] *= 1.5
            expected[offset + 1] *= 1.5
            expected[offset + 2] *= 1.5
        assert actual == pytest.approx(expected, abs=1e-8)
        assert actual[12:15] == original[12:15]


def test_transform_controls_individual_rotation_sweeps_continuously_about_each_origin(transform_controls_module):
    module, state = transform_controls_module
    orientations = [[17.0, 23.0, 11.0], [-34.0, 4.0, 28.0], [5.0, -19.0, 73.0]]
    panel, originals, _decompositions = _multi_rotate_panel(module, state, orientations)
    state.multi_transform_mode = module.lf.ui.MULTI_TRANSFORM_MODE_INDIVIDUAL
    angles = [step * 0.5 for step in range(1, 361)]
    angles += [step * 0.5 for step in range(359, -361, -1)]

    for axis in range(3):
        panel._euler = [0.0, 0.0, 0.0]
        panel._set_value("rot", axis, "0")
        previous = {name: list(matrix) for name, matrix in zip(panel._selected, originals)}
        previous_angle = 0.0

        for angle in angles:
            panel._set_value("rot", axis, str(angle))
            euler = [0.0, 0.0, 0.0]
            euler[axis] = angle
            for name, original in zip(panel._selected, originals):
                pivot_at_node = [original[12], original[13], original[14]]
                delta = _around_pivot_matrix(euler, pivot_at_node)
                expected_rows = _multiply4(delta, _matrix_rows(original))
                expected = [expected_rows[r][c] for c in range(4) for r in range(4)]
                actual = state.visualizer_world_transforms[name]
                assert actual == pytest.approx(expected, abs=1e-5)
                assert actual[12:15] == pytest.approx(original[12:15], abs=1e-8)

                step_radians = math.radians(abs(angle - previous_angle))
                basis_norm = math.sqrt(sum(value * value for value in original[:12]))
                assert _frobenius_delta(actual, previous[name]) <= 2.0 * step_radians * basis_norm + 1e-8
                previous[name] = list(actual)
            previous_angle = angle


def test_transform_controls_individual_scale_sweeps_continuously_and_keeps_shear(transform_controls_module):
    module, state = transform_controls_module
    panel, originals, decompositions = _multi_rotate_panel(
        module,
        state,
        [[17.0, 23.0, 11.0], [-34.0, 4.0, 28.0], [5.0, -19.0, 73.0]],
        translations=[[-2.0, 0.5, 0.0], [0.25, -1.0, 2.0], [3.0, 1.5, -0.5]],
    )
    panel._active_tool = "builtin.scale"
    state.multi_transform_mode = module.lf.ui.MULTI_TRANSFORM_MODE_INDIVIDUAL
    sheared = [
        1.0, 0.4, 0.0, 0.0,
        0.2, 1.5, 0.2, 0.0,
        0.3, 0.1, 1.2, 0.0,
        -2.0, 0.5, 0.0, 1.0,
    ]
    originals[0] = sheared
    panel._state.multi_visualizer_world_transforms_before[0] = sheared
    module.lf.decompose_transform = lambda _matrix: pytest.fail("Individual scale must not decompose")
    module.lf.compose_transform = lambda *_args: pytest.fail("Individual scale must not compose TRS")

    factors = [1.0 - step * 0.01 for step in range(90, -1, -1)]
    factors += [1.01 + step * 0.01 for step in range(200)]
    previous = {name: list(matrix) for name, matrix in zip(panel._selected, originals)}
    previous_factor = 1.0
    for factor in factors:
        panel._set_uniform_scale(str(factor))
        for name, original in zip(panel._selected, originals):
            expected = list(original)
            for offset in (0, 4, 8):
                expected[offset] *= factor
                expected[offset + 1] *= factor
                expected[offset + 2] *= factor
            actual = state.visualizer_world_transforms[name]
            assert actual == pytest.approx(expected, abs=1e-8)
            assert actual[12:15] == original[12:15]
            basis_norm = math.sqrt(sum(value * value for value in original[:12]))
            assert _frobenius_delta(actual, previous[name]) <= abs(factor - previous_factor) * basis_norm + 1e-8
            previous[name] = list(actual)
        previous_factor = factor
    assert factors[0] == pytest.approx(0.1)
    assert factors[-1] == pytest.approx(3.0)


def test_transform_controls_shared_multi_transforms_match_previous_implementation(transform_controls_module):
    module, state = transform_controls_module
    orientations = [[17.0, 23.0, 11.0], [-34.0, 4.0, 28.0], [5.0, -19.0, 73.0]]
    panel, originals, decompositions = _multi_rotate_panel(module, state, orientations)
    state.multi_transform_mode = module.lf.ui.MULTI_TRANSFORM_MODE_SELECTION

    _apply_rotation_input(panel, [16.0, -22.0, 43.0])
    expected = _reference_previous_multi_transform(
        "builtin.rotate", originals, decompositions, [16.0, -22.0, 43.0], panel._state.pivot_world
    )
    for name, result in zip(panel._selected, expected):
        assert state.visualizer_world_transforms[name] == pytest.approx(result, abs=1e-6)

    panel._active_tool = "builtin.scale"
    panel._scale = [1.0, 1.0, 1.0]
    panel._set_uniform_scale("1.35")
    expected = _reference_previous_multi_transform(
        "builtin.scale", originals, decompositions, [1.35] * 3, panel._state.pivot_world
    )
    for name, result in zip(panel._selected, expected):
        assert state.visualizer_world_transforms[name] == pytest.approx(result, abs=1e-6)


def test_transform_controls_individual_identity_and_translation_modes_are_unchanged(transform_controls_module):
    module, state = transform_controls_module
    orientations = [[17.0, 23.0, 11.0], [-34.0, 4.0, 28.0]]
    panel, originals, _decompositions = _multi_rotate_panel(module, state, orientations)
    for mode in (module.lf.ui.MULTI_TRANSFORM_MODE_SELECTION, module.lf.ui.MULTI_TRANSFORM_MODE_INDIVIDUAL):
        state.multi_transform_mode = mode
        panel._active_tool = "builtin.rotate"
        panel._euler = [0.0, 0.0, 0.0]
        panel._set_value("rot", 0, "0")
        for name, original in zip(panel._selected, originals):
            assert state.visualizer_world_transforms[name] == original

        panel._active_tool = "builtin.scale"
        panel._scale = [1.0, 1.0, 1.0]
        panel._set_uniform_scale("1")
        for name, original in zip(panel._selected, originals):
            assert state.visualizer_world_transforms[name] == original

    results = []
    for mode in (module.lf.ui.MULTI_TRANSFORM_MODE_SELECTION, module.lf.ui.MULTI_TRANSFORM_MODE_INDIVIDUAL):
        state.multi_transform_mode = mode
        panel._active_tool = "builtin.translate"
        panel._trans = [0.0, 0.0, 0.0]
        panel._set_value("pos", 1, "2.5")
        results.append([state.visualizer_world_transforms[name] for name in panel._selected])
    assert results[0] == results[1]

    panel._selected = ["single"]
    panel._active_tool = "builtin.scale"
    original = _transform_matrix([1.0, 2.0, 3.0], [13.0, -21.0, 34.0], [0.5, 0.75, 1.25])
    decomp = {
        "translation": [1.0, 2.0, 3.0],
        "rotation_quat": [0.0, 0.0, 0.0, 1.0],
        "rotation_euler_deg": [13.0, -21.0, 34.0],
        "scale": [0.5, 0.75, 1.25],
    }
    module.lf.get_node_visualizer_world_transform = lambda _name: original
    module.lf.decompose_transform = lambda _matrix: decomp
    module.lf.compose_transform = _transform_matrix
    state.visualizer_world_transforms["single"] = original
    panel._trans = [1.0, 2.0, 3.0]
    panel._euler = [1.0, 2.0, 3.0]
    panel._scale = [1.5, 1.25, 0.5]
    panel._apply_single_transform()
    assert state.visualizer_world_transforms["single"] == pytest.approx(
        _transform_matrix(panel._trans, decomp["rotation_euler_deg"], panel._scale), abs=1e-6
    )


def _sheared_transform():
    rows = [
        [1.0, 0.2, 0.3, 1.0],
        [0.4, 1.5, 0.1, 2.0],
        [0.0, 0.2, 1.2, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return [rows[r][c] for c in range(4) for r in range(4)]


def _single_local_edit(module, state, tool):
    original = _sheared_transform()
    displayed = module._flip_yz_rows(original)
    decomp = {
        "translation": _translation_from_matrix(displayed),
        "rotation_quat": [0.0, 0.0, 0.0, 1.0],
        "rotation_euler_deg": [0.0, 0.0, 0.0],
        "scale": [2.0, 1.0, 1.0],
    }
    module.lf.decompose_transform = lambda _matrix: decomp
    module.lf.compose_transform = _transform_matrix
    state.local_transforms["child"] = original
    panel = module.TransformControlsController()
    panel._selected = ["child"]
    panel._active_tool = tool
    panel._transform_space = 0
    panel._trans = list(decomp["translation"])
    panel._euler = list(decomp["rotation_euler_deg"])
    panel._scale = list(decomp["scale"])
    return panel, original, displayed


def test_transform_controls_single_local_translation_keeps_sheared_basis(transform_controls_module):
    module, state = transform_controls_module
    panel, original, displayed = _single_local_edit(module, state, "builtin.translate")

    panel._set_value("pos", 0, "2")

    expected_display = list(displayed)
    expected_display[12] = 2.0
    assert state.local_transforms["child"] == pytest.approx(
        module._flip_yz_rows(expected_display), abs=1e-8
    )
    assert state.local_transforms["child"][:12] == original[:12]


def test_transform_controls_single_local_rotation_and_scale_keep_shear(transform_controls_module):
    module, state = transform_controls_module
    panel, _original, displayed = _single_local_edit(module, state, "builtin.rotate")
    panel._set_value("rot", 2, "45")
    rotation = _rotation_matrix([0.0, 0.0, 45.0])
    expected_rotation = list(displayed)
    for col in range(3):
        vec = [displayed[col * 4 + row] for row in range(3)]
        for row in range(3):
            expected_rotation[col * 4 + row] = sum(rotation[row][axis] * vec[axis] for axis in range(3))
    assert state.local_transforms["child"] == pytest.approx(
        module._flip_yz_rows(expected_rotation), abs=1e-8
    )

    panel, _original, displayed = _single_local_edit(module, state, "builtin.scale")
    panel._set_value("scale", 0, "3")
    expected_scale = list(displayed)
    for row in range(3):
        expected_scale[row] *= 1.5
    assert state.local_transforms["child"] == pytest.approx(
        module._flip_yz_rows(expected_scale), abs=1e-8
    )


def test_transform_controls_multi_shared_translate_and_scale_keep_shear(transform_controls_module):
    module, state = transform_controls_module
    names = ["left", "right"]
    originals = [_sheared_transform(), _sheared_transform()]
    originals[0][12:15] = [-2.0, 0.5, 1.0]
    originals[1][12:15] = [3.0, -0.5, 2.0]
    module.lf.decompose_transform = lambda _matrix: pytest.fail("Multi-node edits must keep the full basis")
    module.lf.compose_transform = lambda *_args: pytest.fail("Multi-node edits must not rebuild TRS")

    def panel_for(tool):
        panel = module.TransformControlsController()
        panel._selected = names
        panel._active_tool = tool
        panel._state.multi_editing_active = True
        panel._state.multi_node_names = names
        panel._state.multi_visualizer_world_transforms_before = [list(matrix) for matrix in originals]
        panel._state.pivot_world = [0.5, 0.0, 0.0]
        panel._trans = [0.5, 0.0, 0.0]
        panel._scale = [1.0, 1.0, 1.0]
        state.multi_transform_mode = module.lf.ui.MULTI_TRANSFORM_MODE_SELECTION
        return panel

    panel = panel_for("builtin.translate")
    panel._set_value("pos", 1, "2")
    for name, original in zip(names, originals):
        expected = list(original)
        expected[13] += 2.0
        assert state.visualizer_world_transforms[name] == pytest.approx(expected, abs=1e-8)

    panel = panel_for("builtin.scale")
    for axis, factor in enumerate((1.5, 0.75, 2.0)):
        panel._set_value("scale", axis, str(factor))
    factors = [1.5, 0.75, 2.0]
    pivot = panel._state.pivot_world
    for name, original in zip(names, originals):
        expected = list(original)
        for col in range(3):
            for row in range(3):
                expected[col * 4 + row] *= factors[col]
        for axis in range(3):
            expected[12 + axis] = pivot[axis] + (original[12 + axis] - pivot[axis]) * factors[axis]
        assert state.visualizer_world_transforms[name] == pytest.approx(expected, abs=1e-8)


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


def test_transform_controls_keeps_other_numeric_bindings_live_while_focused(transform_controls_module):
    module, _state = transform_controls_module
    panel = module.TransformControlsController()
    panel._handle = _DataModelHandleStub()

    # This is the original _dirty_all binding contract. Keep every field
    # refresh when no edit has focus.
    previous_bindings = [
        name
        for axis in ("x", "y", "z")
        for name in (
            f"transform_pos_{axis}_str",
            f"transform_rot_{axis}_str",
            f"transform_scale_{axis}_str",
        )
    ] + [
        "transform_scale_u_str",
        "transform_reset_label",
        "transform_bake_label",
        "transform_editable",
        "transform_show_translate",
        "transform_show_rotate",
        "transform_show_scale",
        "transform_show_actions",
    ]
    panel._dirty_all()
    assert panel._handle.dirty_calls == previous_bindings

    panel._handle.dirty_calls.clear()
    panel._focused_input_property = "transform_rot_y_str"
    panel._dirty_all()
    assert panel._handle.dirty_calls == [
        name for name in previous_bindings if name != "transform_rot_y_str"
    ]

    panel._handle.dirty_calls.clear()
    panel._on_input_blur(None, "transform_rot_y_str")
    panel._dirty_all()
    assert panel._handle.dirty_calls == previous_bindings


def test_transform_controls_keeps_typed_numeric_text_until_blur(transform_controls_module):
    module, _state = transform_controls_module
    panel = module.TransformControlsController()
    model = _DataModelStub()
    panel.bind_model(model)

    # Previously correct display formatting remains unchanged when not editing.
    panel._trans = [1.25, -2.5, 0.0]
    panel._euler = [10.0, -20.0, 30.0]
    panel._scale = [0.125, 2.0, 1.0]
    assert model.bound_binds["transform_pos_x_str"][0]() == "1.250"
    assert model.bound_binds["transform_rot_y_str"][0]() == "-20.0"
    assert model.bound_binds["transform_scale_u_str"][0]() == "1.042"

    panel._focused_input_property = "transform_rot_y_str"
    model.bound_binds["transform_rot_y_str"][1]("-20")
    assert model.bound_binds["transform_rot_y_str"][0]() == "-20"
    model.bound_binds["transform_rot_y_str"][1]("-")
    assert model.bound_binds["transform_rot_y_str"][0]() == "-"

    panel._focused_input_property = "transform_scale_u_str"
    model.bound_binds["transform_scale_u_str"][1]("0.01")
    assert model.bound_binds["transform_scale_u_str"][0]() == "0.01"

    panel._on_input_blur(None, "transform_scale_u_str")
    assert model.bound_binds["transform_scale_u_str"][0]() == "0.010"


def test_transform_controls_rejects_nonfinite_numeric_edits(transform_controls_module):
    module, _state = transform_controls_module
    panel = module.TransformControlsController()
    model = _DataModelStub()
    panel.bind_model(model)
    panel._trans = [1.25, -2.5, 0.0]
    panel._scale = [0.125, 2.0, 1.0]

    model.bound_binds["transform_pos_x_str"][1]("nan")
    model.bound_binds["transform_scale_u_str"][1]("inf")

    assert panel._trans == [1.25, -2.5, 0.0]
    assert panel._scale == [0.125, 2.0, 1.0]


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


def test_transform_controls_locked_selection_is_read_only(transform_controls_module):
    module, state = transform_controls_module
    panel = module.TransformControlsController()
    model = _DataModelStub()
    panel.bind_model(model)
    doc = _DocumentStub()
    state.editable = False
    panel.update(doc)
    assert model.bound_funcs["transform_editable"]() is False
    before = dict(state.local_transforms)
    panel._set_value("pos", 0, "4.0")
    panel._set_uniform_scale("2.0")
    assert state.local_transforms == before
    assert not panel._state.editing_active
    state.editable = True
    panel.update(doc)
    assert model.bound_funcs["transform_editable"]() is True
