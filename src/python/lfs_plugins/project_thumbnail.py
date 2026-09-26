# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Availability checks shared by Projects and the File menu."""

from pathlib import Path

import lichtfeld as lf


def active_project_path() -> str:
    poll = getattr(lf, "project_poll_write", None)
    if not callable(poll):
        return ""
    try:
        state = poll()
        return str(state.get("path") or "") if isinstance(state, dict) else ""
    except Exception:
        return ""


def is_active_project_path(path: str) -> bool:
    active_path = active_project_path()
    if not path or not active_path:
        return False
    try:
        return Path(path).resolve() == Path(active_path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def has_renderable_project_viewport(path: str) -> bool:
    if not is_active_project_path(path):
        return False
    try:
        scene_getter = getattr(lf, "get_render_scene", None)
        exporter = getattr(lf, "export_viewport_image", None)
        if not callable(scene_getter) or not callable(exporter):
            return False
        scene = scene_getter()
        return scene is not None and int(getattr(scene, "total_gaussian_count", 0) or 0) > 0
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
