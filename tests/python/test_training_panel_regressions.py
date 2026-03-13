# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for retained training panel status bindings."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_lf_stub(monkeypatch):
    panel_space = SimpleNamespace(
        SIDE_PANEL="SIDE_PANEL",
        FLOATING="FLOATING",
        VIEWPORT_OVERLAY="VIEWPORT_OVERLAY",
        MAIN_PANEL_TAB="MAIN_PANEL_TAB",
        SCENE_HEADER="SCENE_HEADER",
        STATUS_BAR="STATUS_BAR",
    )
    panel_height_mode = SimpleNamespace(FILL="fill", CONTENT="content")
    panel_option = SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED", HIDE_HEADER="HIDE_HEADER")
    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
    )
    lf_stub.optimization_params = lambda: None
    lf_stub.dataset_params = lambda: None
    lf_stub.loss_buffer = lambda: []
    lf_stub.push_loss_to_element = lambda _element, _data: (0.0, 0.0)
    lf_stub.get_render_settings = lambda: None
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return lf_stub


@pytest.fixture
def training_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))
    sys.modules.pop("lfs_plugins.training_panel", None)
    sys.modules.pop("lfs_plugins", None)
    _install_lf_stub(monkeypatch)
    return import_module("lfs_plugins.training_panel")


class _HandleStub:
    def __init__(self):
        self.dirty_fields = []
        self.dirty_all_count = 0

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_all_count += 1


def _make_signal(value):
    return SimpleNamespace(value=value)


class _ParamsStub:
    def __init__(self):
        self.iterations = 1234
        self.means_lr = 0.25
        self.steps_scaler = 1.0
        self.start_refine = 500
        self.stop_refine = 15000
        self.refine_every = 100
        self.reset_every = 3000
        self.sh_degree_interval = 1000
        self.ppisp_controller_activation_step = 5678

    def has_params(self):
        return True

    def apply_step_scaling(self, value):
        scale = value / self.steps_scaler if self.steps_scaler else value
        self.steps_scaler = value
        self.iterations = int(self.iterations * scale)
        self.start_refine = int(self.start_refine * scale)
        self.stop_refine = int(self.stop_refine * scale)

    def set(self, prop, value):
        setattr(self, prop, value)


class _DatasetStub:
    def __init__(self):
        self.max_width = 2048

    def has_params(self):
        return True


class _ModelStub:
    def __init__(self):
        self.bindings = {}

    def bind(self, name, getter, setter):
        self.bindings[name] = (getter, setter)


def test_training_panel_progress_updates_bound_value(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()

    monkeypatch.setattr(
        training_panel_module,
        "AppState",
        SimpleNamespace(
            iteration=_make_signal(25),
            max_iterations=_make_signal(100),
        ),
    )

    assert panel._update_progress() is True
    assert panel._progress_value == "0.25"
    assert panel._handle.dirty_fields == ["progress_value"]


def test_training_panel_loss_graph_updates_bound_labels(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    panel._loss_graph_el = object()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            loss_buffer=lambda: [1.0, 0.5, 0.25],
            push_loss_to_element=lambda _element, _data: (0.25, 1.0),
            ui=SimpleNamespace(tr=lambda key: key),
        ),
    )

    assert panel._update_loss_graph() is True
    assert panel._loss_label == "status.loss: 0.2500"
    assert panel._loss_tick_max == "1.00"
    assert panel._loss_tick_mid == "0.62"
    assert panel._loss_tick_min == "0.25"
    assert panel._handle.dirty_fields == [
        "loss_label",
        "loss_tick_max",
        "loss_tick_mid",
        "loss_tick_min",
    ]


def test_training_panel_loss_graph_clears_bound_labels(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    panel._loss_graph_el = object()
    panel._last_loss_signature = (3, 0.25)
    panel._loss_label = "status.loss: 0.2500"
    panel._loss_tick_max = "1.00"
    panel._loss_tick_mid = "0.62"
    panel._loss_tick_min = "0.25"

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            loss_buffer=lambda: [],
            push_loss_to_element=lambda _element, _data: (0.0, 0.0),
            ui=SimpleNamespace(tr=lambda key: key),
        ),
    )

    assert panel._update_loss_graph() is True
    assert panel._loss_label == ""
    assert panel._loss_tick_max == ""
    assert panel._loss_tick_mid == ""
    assert panel._loss_tick_min == ""
    assert panel._handle.dirty_fields == [
        "loss_label",
        "loss_tick_max",
        "loss_tick_mid",
        "loss_tick_min",
    ]


def test_numeric_parser_normalizes_integer_commas_and_keeps_float_validation(training_panel_module):
    assert training_panel_module._parse_num("1,234", int) == "1234"
    assert training_panel_module._parse_num("1,5", int) == "15"
    assert training_panel_module._parse_num("1,234.5", float) == "1234.5"

    with pytest.raises(ValueError):
        training_panel_module._parse_num("0,0001", float)

    with pytest.raises(ValueError):
        training_panel_module._parse_num("1,5", float)


def test_integer_commas_are_normalized_while_float_decimal_commas_still_fail(training_panel_module, monkeypatch):
    params = _ParamsStub()
    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(optimization_params=lambda: params),
    )

    panel = training_panel_module.TrainingPanel()

    assert panel._set_num_prop("iterations", "1,234", int, 1, None) is True
    assert params.iterations == 1234

    assert panel._set_num_prop("iterations", "1,5", int, 1, None) is True
    assert params.iterations == 15

    assert panel._set_num_prop("iterations", "1,0001,00", int, 1, None) is True
    assert params.iterations == 1000100

    assert panel._set_num_prop("means_lr", "0,0001", float, 0, None) is False
    assert params.means_lr == 0.25


@pytest.mark.parametrize(
    ("binding_name", "expected_text"),
    [
        ("iterations_str", "1,234"),
        ("ppisp_activation_step_str", "5,678"),
        ("max_width_str", "2,048"),
        ("new_step_str", "7,000"),
    ],
)
def test_cleared_numeric_fields_restore_model_value(training_panel_module, monkeypatch, binding_name, expected_text):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    model = _ModelStub()
    params = _ParamsStub()
    dataset = _DatasetStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
        ),
    )

    panel._bind_num_props(model, lambda: params, lambda: dataset)

    getter, setter = model.bindings[binding_name]
    setter("")
    assert getter() == ""

    panel._commit_number_input_key(binding_name)
    assert getter() == expected_text


@pytest.mark.parametrize(
    ("binding_name", "input_text", "expected_text"),
    [
        ("iterations_str", "300000", "300,000"),
        ("ppisp_activation_step_str", "300000", "300,000"),
        ("max_width_str", "3000", "3,000"),
        ("new_step_str", "300000", "300,000"),
    ],
)
def test_committed_numeric_fields_reformat_and_dirty(
    training_panel_module, monkeypatch, binding_name, input_text, expected_text
):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    model = _ModelStub()
    params = _ParamsStub()
    dataset = _DatasetStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
        ),
    )

    panel._bind_num_props(model, lambda: params, lambda: dataset)

    getter, setter = model.bindings[binding_name]
    setter(input_text)
    assert getter() == input_text
    assert panel._handle.dirty_fields == []

    panel._commit_number_input_key(binding_name)
    assert getter() == expected_text
    assert panel._handle.dirty_fields == [binding_name]


@pytest.mark.parametrize(
    ("binding_name", "input_text", "expected_text"),
    [
        ("iterations_str", "1,11110", "111,110"),
        ("ppisp_activation_step_str", "56,7800", "567,800"),
        ("max_width_str", "2,0,4,8", "2,048"),
        ("new_step_str", "70,0000", "700,000"),
    ],
)
def test_integer_fields_strip_arbitrary_commas_and_reformat(
    training_panel_module, monkeypatch, binding_name, input_text, expected_text
):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    model = _ModelStub()
    params = _ParamsStub()
    dataset = _DatasetStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
        ),
    )

    panel._bind_num_props(model, lambda: params, lambda: dataset)

    getter, setter = model.bindings[binding_name]
    setter(input_text)

    panel._commit_number_input_key(binding_name)
    assert getter() == expected_text
    assert panel._handle.dirty_fields == [binding_name]


@pytest.mark.parametrize(
    ("binding_name", "buffer_text", "expected_text"),
    [
        ("iterations_str", "1a1110", "1,234"),
        ("ppisp_activation_step_str", "56x7800", "5,678"),
        ("max_width_str", "20x48", "2,048"),
        ("new_step_str", "70x000", "7,000"),
    ],
)
def test_invalid_numeric_commit_restores_canonical_value(
    training_panel_module, monkeypatch, binding_name, buffer_text, expected_text
):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    params = _ParamsStub()
    dataset = _DatasetStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
        ),
    )

    panel._text_bufs[binding_name] = buffer_text
    panel._commit_number_input_key(binding_name)

    assert panel._text_bufs[binding_name] == expected_text
    assert panel._handle.dirty_fields == [binding_name]


def test_steps_scaler_syncs_dependent_text_bufs(training_panel_module, monkeypatch):
    """Issue #970: steps_scaler change must refresh all dependent param buffers."""
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    params = _ParamsStub()
    params.iterations = 30000
    params.start_refine = 500
    params.stop_refine = 15000
    params.steps_scaler = 1.0
    dataset = _DatasetStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
        ),
    )

    model = _ModelStub()
    panel._bind_num_props(model, lambda: params, lambda: dataset)

    assert panel._set_num_prop("steps_scaler", "2.0", float, 0.01, None) is True
    assert params.steps_scaler == 2.0
    assert params.iterations == 60000
    assert panel._text_bufs["iterations_str"] == "60,000"
    assert panel._handle.dirty_all_count >= 1


def test_set_bool_prop_hasattr_guard(training_panel_module, monkeypatch):
    """Issue #972: _set_bool_prop must not crash on missing attributes."""
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    params = _ParamsStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            get_render_settings=lambda: None,
        ),
    )

    panel._set_bool_prop("nonexistent_property", True)
    assert not hasattr(params, "nonexistent_property")
