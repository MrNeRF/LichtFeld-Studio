# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Application state as reactive signals.

AppState provides centralized access to all application state via signals.
Panels read from these signals instead of polling the C++ runtime.

The C++ runtime updates these signals when state changes, pushing updates
to Python rather than requiring Python to poll.

Example:
    from lfs_plugins.ui import AppState

    # Read state
    if AppState.is_training.value:
        print(f"Iteration: {AppState.iteration.value}")

    # Subscribe to changes
    AppState.iteration.subscribe(lambda v: print(f"Now at {v}"))
"""

from __future__ import annotations

from .signals import Signal, ComputedSignal


class AppState:
    """Centralized application state as reactive signals.

    All state is exposed as class-level signals. Components read these signals
    and subscribe to changes instead of polling.

    Training State:
        is_training: Whether training is currently running
        trainer_state: Current trainer state string ("idle", "ready", "running", "paused", "stopping", "completed", "stopped", "error")
        has_trainer: Whether a trainer is loaded
        iteration: Current training iteration (C++ throttled)
        max_iterations: Maximum iterations target
        loss: Current loss value (C++ throttled)
        psnr: Current PSNR value
        num_gaussians: Current gaussian count (C++ throttled)

    Scene State:
        has_scene: Whether a scene is loaded
        scene_generation: Increments when scene changes (for cache invalidation)
        scene_path: Path to current scene file

    Selection State:
        has_selection: Whether any gaussians are selected
        selection_count: Number of selected gaussians
        selection_generation: Increments when selection changes

    Tool State:
        active_tool: Current toolbar/operator id
        transform_space: Current transform space id
        pivot_mode: Current transform pivot id

    Viewport State:
        viewport_width: Viewport width in pixels
        viewport_height: Viewport height in pixels

    Application State:
        is_headless: Whether running without GUI
    """

    # Training state - regular signals (C++ handles throttling)
    is_training = Signal(False, "is_training")
    trainer_state = Signal("idle", "trainer_state")
    has_trainer = Signal(False, "has_trainer")
    iteration = Signal(0, "iteration")
    max_iterations = Signal(30000, "max_iterations")
    loss = Signal(0.0, "loss")
    psnr = Signal(0.0, "psnr")
    num_gaussians = Signal(0, "num_gaussians")

    # Scene state
    has_scene = Signal(False, "has_scene")
    scene_generation = Signal(0, "scene_generation")
    scene_path = Signal("", "scene_path")

    # Selection state
    has_selection = Signal(False, "has_selection")
    selection_count = Signal(0, "selection_count")
    selection_generation = Signal(0, "selection_generation")

    # Tool state
    active_tool = Signal("", "active_tool")
    transform_space = Signal(1, "transform_space")
    pivot_mode = Signal(0, "pivot_mode")

    # Viewport state
    viewport_width = Signal(0, "viewport_width")
    viewport_height = Signal(0, "viewport_height")

    # Application state
    is_headless = Signal(False, "is_headless")

    # Derived state
    @classmethod
    def create_computed_signals(cls) -> None:
        """Create computed signals that depend on other signals."""
        cls.training_progress = ComputedSignal(
            lambda: cls.iteration.value / cls.max_iterations.value
            if cls.max_iterations.value > 0
            else 0.0,
            [cls.iteration, cls.max_iterations],
        )

        cls.can_start_training = ComputedSignal(
            lambda: cls.has_trainer.value and cls.trainer_state.value in ("idle", "ready"),
            [cls.has_trainer, cls.trainer_state],
        )

    @classmethod
    def reset(cls) -> None:
        """Reset all state to defaults. Used during shutdown or testing."""
        cls.is_training.value = False
        cls.trainer_state.value = "idle"
        cls.has_trainer.value = False
        cls.iteration.value = 0
        cls.max_iterations.value = 30000
        cls.loss.value = 0.0
        cls.psnr.value = 0.0
        cls.num_gaussians.value = 0
        cls.has_scene.value = False
        cls.scene_generation.value = 0
        cls.scene_path.value = ""
        cls.has_selection.value = False
        cls.selection_count.value = 0
        cls.selection_generation.value = 0
        cls.active_tool.value = ""
        cls.transform_space.value = 1
        cls.pivot_mode.value = 0


# Initialize computed signals
AppState.create_computed_signals()
