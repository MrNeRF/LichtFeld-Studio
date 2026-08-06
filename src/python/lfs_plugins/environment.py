# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Environment access for the Python plugin runtime."""

import os
from typing import Optional


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def value(name: str) -> Optional[str]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag using the native runtime spellings."""
    raw = value(name)
    if raw is None:
        return default
    normalized = raw.casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default
