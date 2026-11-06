# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stable retained controls for the single external training preview."""

import os
from concurrent.futures import ThreadPoolExecutor

import lichtfeld as lf
from lichtfeld_sessions.transport import discover
from .external_training import ExternalTraining
from .types import Panel

_current = None


class ExternalTrainingPanel(Panel):
    id = "lfs.external_training"
    label = "external_training.title"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 30
    template = "rmlui/external_training.rml"
    height_mode = lf.ui.PanelHeightMode.FILL
    update_policy = "interval"
    update_interval_ms = 500

    def __init__(self):
        self._handle = None
        self._executor = None
        self._scan = None
        self._sessions = []
        self._selected = ""
        self._error = ""
        self._signature = None

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("external_training")
        if model is None:
            return
        model.bind_record_list("sessions")
        model.bind("selected", lambda: self._selected, lambda value: setattr(self, "_selected", str(value)))
        model.bind_func("has_sessions", lambda: bool(self._sessions))
        model.bind_func("has_preview", lambda: _current is not None)
        model.bind_func("has_snapshot", lambda: _current is not None and _current.shown is not None)
        model.bind_func("connected", lambda: _current is not None and not _current.stop.is_set())
        model.bind_func("dataset_open", lambda: lf.ui.get_content_type() == "dataset")
        model.bind_func("can_connect", self._can_connect)
        model.bind_func("has_error", lambda: bool(self._message()))
        model.bind_func("error", self._message)
        model.bind_func("current_title", lambda: (_current.record.get("label") or _current.record["id"]) if _current else "")
        model.bind_func("progress", self._progress)
        model.bind_func("preview", self._preview)
        model.bind("show_cameras", self._cameras_shown, self._show_cameras)
        for name, callback in (("refresh", self._refresh), ("connect", self._connect),
                               ("disconnect", self._disconnect), ("remove", self._remove), ("frame", self._frame)):
            model.bind_event(name, lambda _handle, _event, _args, callback=callback: callback())
        self._handle = model.get_handle()

    def on_mount(self, doc):
        self._refresh()

    def on_unmount(self, doc):
        # Hiding the panel leaves the connected observer running.
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = self._scan = None
        doc.remove_data_model("external_training")
        self._handle = None

    def _refresh(self):
        if self._scan:
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trainer-discovery")
        self._scan = self._executor.submit(discover)

    def on_update(self, doc):
        if self._scan and self._scan.done():
            try:
                self._sessions = [record for record in self._scan.result() if record["pid"] != os.getpid()]
                if not any(record["id"] == self._selected for record in self._sessions):
                    self._selected = self._sessions[0]["id"] if self._sessions else ""
                self._handle.update_record_list("sessions", [
                    {"id": record["id"], "label": (record.get("label") or record["id"]) + f" · {record['pid']}"}
                    for record in self._sessions])
                self._error = ""
            except Exception as error:
                self._error = str(error)
            self._scan = None
        if _current and (not _current.scene.is_valid() or not _current.scene.get_node_by_uuid(_current.uuid)):
            self._remove()
        signature = (self._selected, tuple(record["id"] for record in self._sessions), self._error,
                     _current.uuid if _current else None, _current.stop.is_set() if _current else True,
                     self._progress(), self._preview(), self._message(), self._cameras_shown(), lf.ui.get_content_type())
        if signature == self._signature:
            return False
        self._signature = signature
        if self._handle:
            self._handle.dirty_all()
        return True

    def _connect(self):
        global _current
        if not self._can_connect():
            return
        record = next((record for record in self._sessions if record["id"] == self._selected), None)
        if record is None:
            return
        try:
            _current = ExternalTraining(record)
            self._error = ""
        except Exception as error:
            self._error = str(error)

    def _disconnect(self):
        if _current:
            _current.close()

    def _can_connect(self):
        return (bool(self._selected) and lf.ui.get_content_type() != "dataset"
                and not (_current and not _current.stop.is_set() and _current.record["id"] == self._selected))

    def _remove(self):
        global _current
        if _current:
            _current.close(remove=True)
            _current = None

    def _frame(self):
        if _current and _current.scene.is_valid():
            node = _current.scene.get_node_by_uuid(_current.uuid)
            if node:
                lf.select_node(node.name)
                lf.focus_selection()

    def _cameras_shown(self):
        settings = lf.get_render_settings()
        return bool(settings and settings.show_camera_frustums)

    def _message(self):
        return self._error or ((_current.connection_error or _current.error or _current.geometry_error) if _current else "")

    def _show_cameras(self, value):
        settings = lf.get_render_settings()
        if settings:
            settings.show_camera_frustums = bool(value)

    def _progress(self):
        if not _current:
            return ""
        status = _current.status
        key = {"training": "status.training", "paused": "status.paused", "preparing": "status.loading",
               "completed": "status.complete", "stopped": "status.stopped", "failed": "status.error"}.get(status.get("state"), "status.ready")
        return lf.ui.tr(key) + " · " + lf.ui.tr("external_training.progress").format(
            step=f"{status.get('iteration', 0):,}", total=f"{status.get('iterations', 0):,}")

    def _preview(self):
        if _current and _current.preparing_dependencies:
            return lf.ui.tr("external_training.dependencies")
        if not _current or not _current.shown:
            if _current and _current.stop.is_set():
                return lf.ui.tr("status.empty")
            return lf.ui.tr("external_training.waiting")
        shown = _current.shown
        return lf.ui.tr("external_training.snapshot").format(
            step=f"{shown['iteration']:,}", count=f"{shown['count']:,}", degree=shown["sh_degree"])
