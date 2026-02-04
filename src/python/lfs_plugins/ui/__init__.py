# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""UI framework with reactive signals for state management.

Usage:
    from lfs_plugins.ui import Signal, AppState

    if AppState.is_training.value:
        print(f"Iteration: {AppState.iteration.value}")
"""

from .signals import Signal, ComputedSignal, ThrottledSignal, Batch
from .subscription_registry import SubscriptionRegistry
from .state import AppState

__all__ = [
    "Signal",
    "ComputedSignal",
    "ThrottledSignal",
    "Batch",
    "SubscriptionRegistry",
    "AppState",
]
