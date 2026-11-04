# SPDX-License-Identifier: GPL-3.0-or-later
"""Localized session labels, with no filesystem or model access."""
from pathlib import PurePath, PureWindowsPath
import math
from ..localization import safe_format

TERMINAL = {"completed", "stopped", "failed", "superseded"}
LOCALE_KEY = {"active": "external_training.state_training", "training": "external_training.state_training",
              "paused": "external_training.state_paused", "starting": "external_training.state_preparing",
              "preparing": "external_training.state_preparing", "connecting": "external_training.state_connecting",
              "finalizing": "external_training.state_finalizing", "completed": "external_training.state_completed",
              "stopped": "external_training.state_stopped", "failed": "external_training.state_failed",
              "superseded": "external_training.state_replaced", "disconnected": "external_training.state_unavailable",
              "preview error": "external_training.state_preview_error"}


def _tr(key, *args, **values):
    import lichtfeld as lf
    return safe_format(lf.ui.tr(key), *args, **values)


def title(record):
    label = str(record.get("label") or "").strip()
    project = record.get("project")
    if project and label in {"", "Trainer", "Blender training", "dataset", "LichtFeld trainer"}:
        path = PureWindowsPath(project) if "\\" in project else PurePath(project)
        return path.stem
    return label or _tr("external_training.unnamed")


def state_label(state):
    return _tr(LOCALE_KEY.get(state, "external_training.state_unknown"))


def can_open(record):
    return bool(record and record.get("active"))


def option_label(record):
    status = state_label(record.get("state")) if record.get("active") else _tr("external_training.state_unavailable")
    return f"{title(record)} · {status} · {record['id'][:6]}"


def preview_heading(status, stopped, error, has_snapshot):
    if stopped:
        return _tr("external_training.heading_disconnected" if has_snapshot else "external_training.heading_closed")
    if error:
        return _tr("external_training.heading_update_problem" if has_snapshot else "external_training.heading_load_failed")
    state = status.get("state", "connecting")
    if state == "disconnected":
        return _tr("external_training.heading_unresponsive" if status.get("liveness") == "unresponsive" else "external_training.heading_unavailable")
    if state in TERMINAL:
        return _tr("external_training.heading_ended", state=state_label(state))
    if state == "preparing preview":
        return _tr("external_training.heading_setup")
    if state == "connecting":
        return _tr("external_training.heading_reconnecting" if has_snapshot else "external_training.heading_connecting")
    return _tr("external_training.heading_connected", state=state_label(state))


def preview_quality(status):
    snapshot = status.get("snapshot") or {}
    count, degree = snapshot.get("count"), snapshot.get("sh_degree")
    if type(count) is not int or not 0 < count <= 10000000 or type(degree) is not int or not 0 <= degree <= 3:
        return _tr("external_training.quality_limited")
    total = status.get("model_count")
    points = _tr("external_training.points_total", count=count, total=total) if type(total) is int and total >= count else f"{count:,}"
    return _tr("external_training.quality", points=points, degree=str(degree))


def preview_cost(status):
    values = [status.get("preview_capture_seconds"), status.get("preview_publish_seconds")]
    if any(type(value) not in (int, float) or not 0 <= value <= 3600 or not math.isfinite(value) for value in values):
        return ""
    return _tr("external_training.cost", capture=f"{values[0]*1000:.1f}", publish=f"{values[1]*1000:.1f}")
