# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Project Manager adapter for the canonical application preferences store."""

from __future__ import annotations

import json
from typing import Any

import lichtfeld as lf


DEFAULTS = {
    "defaultView": "remember",
    "openAtStartup": True,
    "rememberState": True,
}

def _validate(key: str, value: Any) -> Any:
    if key == "defaultView":
        text = str(value)
        if text not in {"remember", "gallery", "list"}:
            raise ValueError("Unsupported Project Manager default view")
        return text
    if key in {"openAtStartup", "rememberState"} and isinstance(value, bool):
        return value
    raise ValueError("Unknown Project Manager preference")


def read_preferences() -> dict[str, Any]:
    result = DEFAULTS.copy()
    try:
        document = lf.ui.get_project_manager_preferences()
    except AttributeError:
        return result
    if not isinstance(document, dict):
        return result
    for key in DEFAULTS:
        if key not in document:
            continue
        try:
            result[key] = _validate(key, document[key])
        except (ValueError, TypeError):
            pass
    return result


def read_state() -> dict[str, Any]:
    try:
        state = json.loads(lf.ui.get_project_manager_state())
    except (AttributeError, TypeError, ValueError):
        return {}
    return dict(state) if isinstance(state, dict) else {}


def set_preference(key: str, value: Any) -> None:
    validated = _validate(key, value)
    if key == "defaultView":
        lf.ui.set_project_manager_default_view(validated)
    elif key == "openAtStartup":
        lf.ui.set_project_manager_open_at_startup(validated)
    else:
        lf.ui.set_project_manager_remember_state(validated)


def set_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise TypeError("Project Manager state must be an object")
    lf.ui.set_project_manager_state(
        json.dumps(state, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    )


def reset_preferences() -> None:
    lf.ui.reset_project_manager_preferences()
