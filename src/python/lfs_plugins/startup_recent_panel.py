# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Startup recent-projects chooser (no silent session restore)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import lichtfeld as lf

from .rml_keys import KI_ESCAPE
from .training_confirm import _project_has_path
from .types import Panel

__lfs_panel_classes__ = ["StartupRecentPanel"]
__lfs_panel_ids__ = ["lfs.startup_recent"]

_MAX_ROWS = 5
_PATH_DISPLAY_MAX = 64


def elide_middle(path: str, max_len: int = _PATH_DISPLAY_MAX) -> str:
    """Elide the middle of a path when it exceeds max_len."""
    text = str(path or "")
    if len(text) <= max_len:
        return text
    if max_len < 8:
        return text[:max_len]
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return f"{text[:left]}...{text[-right:]}"


def display_name_for_path(path: str) -> str:
    stem = Path(path).stem
    return stem or Path(path).name or path


def format_mtime(path: str) -> str:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return "—"
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
    except (OverflowError, ValueError, OSError):
        return "—"


def recovery_disposition(path: str) -> str:
    inspect = getattr(lf, "project_autosave_recovery_disposition", None)
    if not callable(inspect):
        return "none"
    try:
        value = inspect(path)
    except Exception:
        return "none"
    return str(value or "none").lower()


def should_show_startup_recent() -> bool:
    """True when blank session and MRU has at least one entry."""
    recent_fn = getattr(lf, "project_recent_files", None)
    if not callable(recent_fn):
        return False
    try:
        recent = list(recent_fn() or [])
    except Exception:
        return False
    if not recent:
        return False
    if _project_has_path():
        return False
    is_empty = getattr(lf.ui, "is_scene_empty", None)
    if callable(is_empty):
        try:
            if not is_empty():
                return False
        except Exception:
            pass
    return True


def try_show_startup_recent() -> bool:
    """Enable the panel when the blank-session + MRU policy allows it."""
    if not should_show_startup_recent():
        return False
    lf.ui.set_panel_enabled("lfs.startup_recent", True)
    return True


def build_project_rows(paths, *, max_rows: int = _MAX_ROWS) -> list[dict]:
    """Build record-list rows for the top MRU paths (newest first)."""
    rows = []
    for index, path in enumerate(list(paths or [])[:max_rows]):
        path_text = str(path)
        disposition = recovery_disposition(path_text)
        recoverable = disposition == "offer"
        rows.append(
            {
                "index": index,
                "path": path_text,
                "display_name": display_name_for_path(path_text),
                "path_display": elide_middle(path_text),
                "last_opened": format_mtime(path_text),
                "recoverable": recoverable,
                "show_recoverable": recoverable,
            }
        )
    return rows


class StartupRecentPanel(Panel):
    """Floating chooser for the five most recent projects at startup."""

    id = "lfs.startup_recent"
    label = "Recent Projects"
    space = lf.ui.PanelSpace.FLOATING
    order = 5
    options = {lf.ui.PanelOption.DEFAULT_CLOSED}
    template = "rmlui/startup_recent_panel.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (520, 0)
    update_policy = "dirty"

    def __init__(self):
        super().__init__()
        self._handle = None
        self._rows: list[dict] = []
        self._show_crash_notice = False
        self._dismissed = False

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("startup_recent")
        if model is None:
            return

        model.bind_func(
            "panel_label",
            lambda: lf.ui.tr("startup_recent.title"),
        )
        model.bind_func(
            "show_crash_notice",
            lambda: self._show_crash_notice,
        )
        model.bind_func(
            "crash_notice",
            lambda: lf.ui.tr("startup_recent.crash_notice"),
        )
        model.bind_func(
            "recoverable_badge",
            lambda: lf.ui.tr("startup_recent.recoverable_badge"),
        )
        model.bind_func(
            "open_label",
            lambda: lf.ui.tr("startup_recent.open"),
        )
        model.bind_func(
            "open_other_label",
            lambda: lf.ui.tr("startup_recent.open_other"),
        )
        model.bind_func(
            "start_blank_label",
            lambda: lf.ui.tr("startup_recent.start_blank"),
        )
        model.bind_func(
            "last_opened_label",
            lambda: lf.ui.tr("startup_recent.last_opened"),
        )
        model.bind_record_list("projects")
        model.bind_event("open_other", self._on_open_other)
        model.bind_event("start_blank", self._on_start_blank)
        self._handle = model.get_handle()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._dismissed = False
        doc.add_event_listener("keydown", self._on_keydown)
        # Title-bar close is Start Blank (Panel base wires #close-btn).
        projects_list = doc.get_element_by_id("startup-recent-list")
        if projects_list:
            projects_list.add_event_listener("click", self._on_projects_click)
        self._refresh_rows(force=True)

    def on_unmount(self, doc):
        if self._handle is not None:
            try:
                doc.remove_data_model("startup_recent")
            except Exception:
                pass
        self._handle = None

    def on_update(self, doc):
        del doc
        if self._dismissed:
            return False
        if not should_show_startup_recent():
            self._dismiss()
            return False
        return self._refresh_rows(force=False)

    def _refresh_rows(self, force: bool = False) -> bool:
        if not self._handle:
            return False
        recent_fn = getattr(lf, "project_recent_files", lambda: [])
        try:
            paths = list(recent_fn() or [])
        except Exception:
            paths = []
        rows = build_project_rows(paths, max_rows=_MAX_ROWS)
        signature = tuple(
            (row["path"], row["recoverable"], row["last_opened"]) for row in rows
        )
        prev = tuple(
            (row["path"], row["recoverable"], row["last_opened"]) for row in self._rows
        )
        if not force and signature == prev:
            return False
        self._rows = rows
        self._show_crash_notice = bool(rows) and bool(rows[0].get("recoverable"))
        self._handle.update_record_list("projects", rows)
        self._handle.dirty_all()
        return True

    def _find_with_attr(self, element, attr, stop=None):
        while element is not None and element != stop:
            if element.has_attribute(attr):
                return element
            element = element.parent()
        return None

    def _on_projects_click(self, event):
        container = event.current_target()
        target = self._find_with_attr(event.target(), "data-open-index", container)
        if target is None:
            return
        try:
            index = int(target.get_attribute("data-open-index", ""))
        except (TypeError, ValueError):
            return
        if not (0 <= index < len(self._rows)):
            return
        event.stop_propagation()
        path = self._rows[index]["path"]
        self._open_path(path)

    def _open_path(self, path: str) -> None:
        path_text = str(path)
        name = display_name_for_path(path_text)
        missing_message = lf.ui.tr("startup_recent.project_missing").format(name=name)

        # Pre-check existence so a deleted MRU entry does not raise into RmlUI.
        if not Path(path_text).is_file():
            self._report_open_error(missing_message)
            return

        try:
            outcome = lf.project_open(path_text, True)
        except Exception as exc:
            # NotFoundError subclasses FileNotFoundError; other open failures
            # surface their error text on the same user-visible path.
            if isinstance(exc, FileNotFoundError):
                message = missing_message
            else:
                message = str(exc).strip() or missing_message
            self._report_open_error(message)
            return

        # Keep the chooser open while a crash-recovery prompt is pending.
        recovery_pending = getattr(
            getattr(lf, "ProjectOpenOutcome", None),
            "RECOVERY_PROMPT_PENDING",
            None,
        )
        if recovery_pending is not None and outcome == recovery_pending:
            return

        self._dismiss()

    def _report_open_error(self, message: str) -> None:
        try:
            lf.ui.confirm_dialog(
                lf.ui.tr("startup_recent.title"),
                message,
                [lf.ui.tr("common.ok")],
                lambda _button: None,
            )
        except Exception as exc:
            print(f"startup_recent: failed to show open-error dialog: {exc}")

    def _on_open_other(self, _handle, _event, _args):
        # Empty path opens the existing project file dialog.
        lf.project_open("", True)
        # If the dialog was cancelled, stay blank with chooser still useful;
        # if a project path appears, on_update will dismiss.
        if not should_show_startup_recent():
            self._dismiss()

    def _on_start_blank(self, _handle, _event, _args):
        self._dismiss()

    def _on_keydown(self, event):
        key = int(event.get_parameter("key_identifier", "0"))
        if key == KI_ESCAPE:
            self._dismiss()
            event.stop_propagation()

    def _dismiss(self):
        if self._dismissed:
            return
        self._dismissed = True
        try:
            lf.ui.set_panel_enabled(self.id, False)
        except Exception:
            pass
