# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for retained training panel status bindings."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_lf_stub(monkeypatch):
    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
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

    def dirty(self, name):
        self.dirty_fields.append(name)


def _make_signal(value):
    return SimpleNamespace(value=value)


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
