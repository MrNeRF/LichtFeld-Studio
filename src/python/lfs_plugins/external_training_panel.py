# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stable retained controls for the single external training preview."""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import lichtfeld as lf
from lichtfeld_sessions import ensure_numpy, geometry
from lichtfeld_sessions.transport import TERMINAL, discover, resolve_result, validate_result
from .external_training import ExternalTraining, presentation_transform
from .types import Panel

_current = None
_log = logging.getLogger(__name__)


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
        self._result_error = ""
        self._signature = None
        self._result_future = None
        self._result_target = None
        self._result_load = None
        self._result_cancel = threading.Event()
        self._result_status = None
        self._result_descriptor = None
        self._result_invalid = False

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
        model.bind_func("saved_result", self._saved_result)
        model.bind_func("has_saved_result", lambda: bool(self._saved_result()))
        model.bind_func("can_import_result", self._can_import_result)
        model.bind_func("import_label", lambda: lf.ui.tr("status.loading") if self._result_future or self._result_load else lf.ui.tr("external_training.import_result"))
        model.bind("show_cameras", self._cameras_shown, self._show_cameras)
        for name, callback in (("refresh", self._refresh), ("connect", self._connect),
                               ("disconnect", self._disconnect), ("remove", self._remove), ("frame", self._frame),
                               ("import_result", self._import_result)):
            model.bind_event(name, lambda _handle, _event, _args, callback=callback: callback())
        self._handle = model.get_handle()

    def on_mount(self, doc):
        self._refresh()

    def on_unmount(self, doc):
        # Hiding the panel leaves the connected observer running.
        self._result_cancel.set()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._result_future = self._result_target = None
        self._executor = self._scan = None
        doc.remove_data_model("external_training")
        self._handle = None

    def _refresh(self):
        if self._scan:
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="trainer-discovery")
        self._scan = self._executor.submit(discover)

    def on_update(self, doc):
        self._refresh_saved_result()
        if self._result_future and self._result_future.done():
            try:
                path, transform, degree, iteration = self._result_future.result()
            except Exception as error:
                if not self._result_cancel.is_set():
                    _log.warning("Saved model import failed: %s", error)
                    if isinstance(error, ValueError):
                        self._result_error = lf.ui.tr("external_training.result_invalid")
                    elif isinstance(error, OSError):
                        self._result_error = lf.ui.tr("external_training.result_unreadable")
                    else:
                        self._result_error = str(error)
            else:
                if (not self._result_cancel.is_set() and _current is self._result_target
                        and _current.scene.get_node_by_uuid(_current.uuid)):
                    try:
                        if lf.ui.get_import_state().get("active"):
                            self._result_error = lf.ui.tr("external_training.import_busy")
                        else:
                            name = lf.ui.tr("external_training.result_name").format(name=_current.record.get("label") or _current.record["id"])
                            base = f"{name[:40]} · {iteration}"
                            group, counter = base, 1
                            while _current.scene.get_node(group):
                                group = f"{base} ({counter})"
                                counter += 1
                            lf.load_gallery_scene([{"path": str(path), "name": name[:512],
                                                    "transform": transform.tolist(), "shDegree": degree}], group)
                            self._result_load = (_current, group)
                    except Exception as error:
                        _log.warning("Saved model could not be added to the scene: %s", error)
                        self._result_error = lf.ui.tr("external_training.import_busy" if lf.ui.get_import_state().get("active")
                                                      else "progress.import_failed_title")
            finally:
                self._result_future = self._result_target = None
        if self._result_load:
            state = lf.ui.get_import_state()
            target, group = self._result_load
            if _current is not target:
                self._result_load = None
            elif not state.get("active"):
                if state.get("success") and target.scene.get_node(group):
                    node = target.scene.get_node_by_uuid(target.uuid)
                    if node:
                        node.set("visible", False)
                elif state.get("error"):
                    _log.warning("Saved model import failed: %s", state["error"])
                    self._result_error = lf.ui.tr("progress.import_failed_title")
                self._result_load = None
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
                     self._progress(), self._preview(), self._saved_result(), self._can_import_result(),
                     self._message(), self._cameras_shown(), lf.ui.get_content_type())
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
            self._result_cancel.set()
            _current = ExternalTraining(record)
            self._error = ""
            self._result_error = ""
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
        self._result_cancel.set()
        self._result_error = ""
        self._result_load = None
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
        return self._error or self._result_error or ((_current.connection_error or _current.error or _current.geometry_error) if _current else "")

    def _show_cameras(self, value):
        settings = lf.get_render_settings()
        if settings:
            settings.show_camera_frustums = bool(value)

    def _progress(self):
        if not _current:
            return ""
        status = _current.status
        key = {"training": "status.training", "paused": "status.paused", "preparing": "status.loading",
               "stopping": "status.stopping", "finalizing": "status.loading",
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

    def _refresh_saved_result(self):
        status = _current.status if _current else None
        if status is self._result_status:
            return
        self._result_status = status
        self._result_descriptor = None
        self._result_invalid = False
        result = status.get("result") if status else None
        if not isinstance(result, dict) or "root" not in result:
            return
        try:
            self._result_descriptor = validate_result(result)
        except ValueError:
            self._result_invalid = True

    def _saved_result(self):
        if self._result_invalid:
            return lf.ui.tr("external_training.result_invalid")
        result = self._result_descriptor
        if result is None:
            return ""
        key = "external_training.result_best" if result["kind"] == "best" else "external_training.result_last"
        return lf.ui.tr(key).format(step=f"{result.get('iteration', 0):,}",
                                   count=f"{result.get('count', 0):,}", degree=result.get("sh_degree", 0))

    def _can_import_result(self):
        self._refresh_saved_result()
        return bool(_current and self._result_status.get("state") in TERMINAL
                    and self._result_descriptor is not None
                    and not self._result_future and not self._result_load and not lf.ui.get_import_state().get("active")
                    and lf.ui.get_content_type() != "dataset")

    def _import_result(self):
        if not self._can_import_result():
            return
        self._result_error = ""
        self._result_target = _current
        status = self._result_status
        self._result_cancel = threading.Event()
        cancel = self._result_cancel.is_set

        def prepare():
            np = ensure_numpy(lf, cancel)
            path, value = resolve_result(status, cancel)
            transform = presentation_transform(value, np) @ geometry.matrix(value.get("model_to_data", geometry.IDENTITY))
            if cancel():
                raise InterruptedError("Saved model import cancelled")
            result = status["result"]
            return path, np.asarray(transform, dtype=np.float32), result["sh_degree"], result["iteration"]

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="trainer-discovery")
        self._result_future = self._executor.submit(prepare)
