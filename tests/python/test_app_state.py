# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the AppState singleton."""

import sys

import pytest

sys.path.insert(0, "build/python")

from lfs_plugins.ui.state import AppState
from lfs_plugins.ui.signals import Signal, ThrottledSignal


class TestAppState:
    """Tests for AppState signal definitions."""

    def test_training_signals_exist(self):
        """AppState should have training signals."""
        assert hasattr(AppState, "is_training")
        assert hasattr(AppState, "trainer_state")
        assert hasattr(AppState, "has_trainer")
        assert hasattr(AppState, "iteration")
        assert hasattr(AppState, "max_iterations")
        assert hasattr(AppState, "loss")
        assert hasattr(AppState, "psnr")
        assert hasattr(AppState, "num_gaussians")

    def test_scene_signals_exist(self):
        """AppState should have scene signals."""
        assert hasattr(AppState, "has_scene")
        assert hasattr(AppState, "scene_generation")
        assert hasattr(AppState, "scene_path")

    def test_selection_signals_exist(self):
        """AppState should have selection signals."""
        assert hasattr(AppState, "has_selection")
        assert hasattr(AppState, "selection_count")
        assert hasattr(AppState, "selection_generation")

    def test_viewport_signals_exist(self):
        """AppState should have viewport signals."""
        assert hasattr(AppState, "viewport_width")
        assert hasattr(AppState, "viewport_height")

    def test_signals_are_correct_types(self):
        """Signals should be of correct types."""
        assert isinstance(AppState.is_training, Signal)
        assert isinstance(AppState.trainer_state, Signal)
        assert isinstance(AppState.iteration, Signal)
        assert isinstance(AppState.loss, Signal)

    def test_default_values(self):
        """AppState should have sensible defaults."""
        AppState.reset()
        assert AppState.is_training.value is False
        assert AppState.trainer_state.value == "idle"
        assert AppState.has_trainer.value is False
        assert AppState.has_scene.value is False
        assert AppState.has_selection.value is False

    def test_reset_restores_defaults(self):
        """reset() should restore default values."""
        AppState.is_training.value = True
        AppState.iteration.value = 1000
        AppState.reset()
        assert AppState.is_training.value is False
        assert AppState.iteration.value == 0

    def test_computed_signals_exist(self):
        """Computed signals should be created."""
        assert hasattr(AppState, "training_progress")
        assert hasattr(AppState, "can_start_training")

    def test_training_progress_computed(self):
        """training_progress should compute from iteration/max_iterations."""
        AppState.reset()
        AppState.iteration.value = 1500
        AppState.max_iterations.value = 30000
        assert 0.04 < AppState.training_progress.value < 0.06

    def test_can_start_training_computed(self):
        """can_start_training should depend on trainer state."""
        AppState.reset()
        AppState.has_trainer.value = True
        AppState.trainer_state.value = "ready"
        assert AppState.can_start_training.value is True

        AppState.trainer_state.value = "running"
        assert AppState.can_start_training.value is False

    def test_signal_value_update(self):
        """Signal values should be directly settable."""
        AppState.reset()
        AppState.iteration.value = 999
        assert AppState.iteration.value == 999
        AppState.reset()


class TestAppStateSubscription:
    """Tests for subscribing to AppState signals."""

    def test_subscribe_to_training(self):
        """Should be able to subscribe to training state."""
        AppState.reset()
        notified = []

        def callback(v):
            notified.append(v)

        unsub = AppState.is_training.subscribe(callback)
        AppState.is_training.value = True
        assert notified == [True]
        unsub()

    def test_subscribe_to_iteration(self):
        """Should be able to subscribe to iteration (throttled)."""
        AppState.reset()
        notified = []

        def callback(v):
            notified.append(v)

        unsub = AppState.iteration.subscribe(callback)
        AppState.iteration.value = 100
        assert 100 in notified or len(notified) == 0
        unsub()

    def test_multiple_subscribers(self):
        """Multiple components can subscribe to same signal."""
        AppState.reset()
        notified_a = []
        notified_b = []

        def callback_a(v):
            notified_a.append(v)

        def callback_b(v):
            notified_b.append(v)

        unsub_a = AppState.has_scene.subscribe(callback_a)
        unsub_b = AppState.has_scene.subscribe(callback_b)
        AppState.has_scene.value = True
        assert True in notified_a
        assert True in notified_b
        unsub_a()
        unsub_b()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
