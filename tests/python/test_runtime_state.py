# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the public RuntimeState plugin facade."""

import pytest

from lfs_plugins.ui import AppState, AppStore, NativeAppStore, RuntimeState
from lfs_plugins.ui import store as store_module
from lfs_plugins.ui.signals import ComputedSignal, Signal
from lfs_plugins.ui.store import StateSignal


@pytest.fixture(autouse=True)
def fallback_runtime_state(monkeypatch):
    monkeypatch.setattr(store_module, "_native_store", lambda: None)
    RuntimeState.reset()
    yield
    RuntimeState.reset()


class TestRuntimeState:
    """RuntimeState exposes the documented app-state fields."""

    def test_training_signals_exist(self):
        assert hasattr(RuntimeState, "is_training")
        assert hasattr(RuntimeState, "trainer_state")
        assert hasattr(RuntimeState, "has_trainer")
        assert hasattr(RuntimeState, "iteration")
        assert hasattr(RuntimeState, "max_iterations")
        assert hasattr(RuntimeState, "loss")
        assert hasattr(RuntimeState, "psnr")
        assert hasattr(RuntimeState, "num_gaussians")

    def test_scene_signals_exist(self):
        assert hasattr(RuntimeState, "has_scene")
        assert hasattr(RuntimeState, "scene_generation")
        assert hasattr(RuntimeState, "scene_path")

    def test_selection_signals_exist(self):
        assert hasattr(RuntimeState, "has_selection")
        assert hasattr(RuntimeState, "selection_count")
        assert hasattr(RuntimeState, "selection_generation")

    def test_viewport_signals_exist(self):
        assert hasattr(RuntimeState, "viewport_width")
        assert hasattr(RuntimeState, "viewport_height")

    def test_native_fields_are_state_signals(self):
        assert isinstance(RuntimeState.is_training, StateSignal)
        assert isinstance(RuntimeState.trainer_state, StateSignal)
        assert isinstance(RuntimeState.iteration, StateSignal)
        assert isinstance(RuntimeState.loss, StateSignal)

    def test_python_only_compatibility_fields_are_signals(self):
        assert isinstance(RuntimeState.has_scene, Signal)
        assert isinstance(RuntimeState.has_selection, Signal)
        assert isinstance(RuntimeState.viewport_width, Signal)

    def test_computed_fields_are_computed_signals(self):
        assert isinstance(RuntimeState.psnr, ComputedSignal)
        assert isinstance(RuntimeState.training_progress, ComputedSignal)
        assert isinstance(RuntimeState.can_start_training, ComputedSignal)

    def test_compatibility_aliases_point_to_runtime_state(self):
        assert AppState is RuntimeState
        assert AppStore is RuntimeState
        assert NativeAppStore is RuntimeState

    def test_default_values(self):
        assert RuntimeState.is_training.value is False
        assert RuntimeState.trainer_state.value == "idle"
        assert RuntimeState.has_trainer.value is False
        assert RuntimeState.has_scene.value is False
        assert RuntimeState.has_selection.value is False
        assert RuntimeState.total_iterations.value == 0

    def test_reset_restores_defaults(self):
        RuntimeState.is_training.value = True
        RuntimeState.iteration.value = 1000
        RuntimeState.total_iterations.value = 7000
        RuntimeState.has_scene.value = True

        RuntimeState.reset()

        assert RuntimeState.is_training.value is False
        assert RuntimeState.iteration.value == 0
        assert RuntimeState.total_iterations.value == 0
        assert RuntimeState.has_scene.value is False

    def test_legacy_training_aliases_share_native_fields(self):
        RuntimeState.iteration.value = 123
        RuntimeState.total_iterations.value = 7000
        RuntimeState.training_state.value = "running"
        RuntimeState.trainer_loaded.value = True

        assert RuntimeState.max_iterations.value == 7000
        assert RuntimeState.trainer_state.value == "running"
        assert RuntimeState.has_trainer.value is True

    def test_psnr_compatibility_field_uses_eval_psnr(self):
        RuntimeState.eval_psnr.value = None
        assert RuntimeState.psnr.value == 0.0

        RuntimeState.eval_psnr.value = 29.5
        assert RuntimeState.psnr.value == pytest.approx(29.5)

    def test_training_progress_computed(self):
        RuntimeState.iteration.value = 1500
        RuntimeState.max_iterations.value = 30000

        assert 0.04 < RuntimeState.training_progress.value < 0.06

    def test_can_start_training_computed(self):
        RuntimeState.has_trainer.value = True
        RuntimeState.trainer_state.value = "ready"
        assert RuntimeState.can_start_training.value is True

        RuntimeState.trainer_state.value = "running"
        assert RuntimeState.can_start_training.value is False


class TestRuntimeStateSubscription:
    """RuntimeState fields keep the signal subscription contract."""

    def test_subscribe_to_training(self):
        notified = []

        unsub = RuntimeState.is_training.subscribe(notified.append)
        RuntimeState.is_training.value = True

        assert notified == [True]
        unsub()

    def test_subscribe_to_iteration(self):
        notified = []

        unsub = RuntimeState.iteration.subscribe(notified.append)
        RuntimeState.iteration.value = 100

        assert notified == [100]
        unsub()

    def test_multiple_subscribers(self):
        notified_a = []
        notified_b = []

        unsub_a = RuntimeState.has_scene.subscribe(notified_a.append)
        unsub_b = RuntimeState.has_scene.subscribe(notified_b.append)
        RuntimeState.has_scene.value = True

        assert notified_a == [True]
        assert notified_b == [True]
        unsub_a()
        unsub_b()
