# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""UI framework with reactive signals for state management.

Usage:
    from lfs_plugins.ui import RuntimeState

    if RuntimeState.is_training.value:
        print(f"Iteration: {RuntimeState.iteration.value}")
"""

from .signals import Signal, ComputedSignal, ThrottledSignal, Batch
from .subscription_registry import SubscriptionRegistry
from .store import (
    AppState,
    AppStore,
    NativeAppStore,
    PanelStateBinding,
    PanelStoreBinding,
    RuntimeState,
    StateSignal,
    StoreSignal,
    batch_updates,
    invalidate_panel,
    native_value,
)

__all__ = [
    "Signal",
    "ComputedSignal",
    "ThrottledSignal",
    "Batch",
    "SubscriptionRegistry",
    "RuntimeState",
    "StateSignal",
    "PanelStateBinding",
    "AppState",
    "AppStore",
    "NativeAppStore",
    "PanelStoreBinding",
    "StoreSignal",
    "batch_updates",
    "invalidate_panel",
    "native_value",
]
