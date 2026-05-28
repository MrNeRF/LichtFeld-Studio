# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility module for the legacy AppState name."""

from __future__ import annotations

from .store import RuntimeState

AppState = RuntimeState

__all__ = ["AppState", "RuntimeState"]
