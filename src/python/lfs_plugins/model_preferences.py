# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Presentation of the native model cache in application preferences."""

from _lfs_path_io import Path
import time
import weakref

import lichtfeld as lf

from .asset_format import format_size


class ModelPreferencesSection:
    def __init__(self):
        self.handle = None
        self.rows = []
        self.job = {}
        self.message = ""
        self.error = ""
        self._next_refresh = 0.0
        self._records = None
        self._last_status = None
        self._confirmation_token = object()

    def bind(self, model):
        self._records = None
        self._last_status = None
        self._next_refresh = 0.0
        self.cancel_pending_confirmation()
        model.bind_record_list("managed_models")
        model.bind_func("model_operation_message", lambda: self.message)
        model.bind_func("model_operation_error", lambda: bool(self.error) and
                        not any(row["id"] == self.job.get("model") for row in self.rows))
        model.bind_func("model_operation_detail", lambda: self.error)
        model.bind_event("model_action", self._action)
        model.bind_event("model_cancel", self._cancel)

    def sync(self, *, force=False):
        if self.handle is None or (not force and time.monotonic() < self._next_refresh):
            return
        self._next_refresh = time.monotonic() + 1.0
        try:
            snapshot = lf.ui.model_cache_snapshot()
        except RuntimeError as error:
            self.job = {}
            self.error = str(error)
            self.message = lf.ui.tr("preferences.models.failed")
            self._publish()
            return
        self.job = dict(snapshot.get("job", {}))
        self.rows = list(snapshot.get("models", []))
        tr = lf.ui.tr
        records = []
        self.error = self.message = ""
        for row in self.rows:
            present, busy = bool(row.get("present")), bool(row.get("busy"))
            previous = next((record for record in self._records or () if record["id"] == str(row["id"])), {})
            # Cache inspection avoids waiting on a running operation. Keep only
            # its last rendered metadata until the native snapshot supplies it.
            if busy:
                present = bool(previous.get("has_path"))
            detail = self._error_text(row.get("error", ""), row.get("error_code", ""))
            message = tr("preferences.models." + row["message"])
            if row.get("show_progress"):
                percent = round(max(0.0, min(1.0, float(row["progress"]))) * 100)
                message = tr("gallery.status.progress").format(stage=message, percent=percent)
            if row["id"] == self.job.get("model"):
                self.error, self.message = detail, message
            records.append({
                "id": str(row["id"]), "name": row["name"], "category": tr(row["category_key"]),
                "status": message, "detail": detail, "failed": row["status"] == "failed",
                "running": bool(row.get("running")), "canceling": bool(row.get("canceling")),
                "size": previous.get("size", "") if busy else format_size(row.get("bytes", 0)) if present else "",
                "path": str(row.get("path", "")) if present else "", "has_path": present,
                "available": row["status"] == "installed", "working": busy or bool(row.get("running")),
                "can_download": bool(row["can_download"]), "can_verify": bool(row["can_verify"]),
                "can_remove": bool(row["can_remove"]), "can_open": present,
            })
        if not any(row["id"] == self.job.get("model") for row in self.rows):
            self.error = self._error_text(self.job.get("error", ""), self.job.get("error_code", ""))
            self.message = tr("preferences.models.failed") if self.error else ""
        if records != self._records:
            self.handle.update_record_list("managed_models", records)
            self.handle.dirty("managed_models")
            self._records = records
        name = next((row["name"] for row in self.rows if row["id"] == self.job.get("model")), "")
        if name and self.message:
            self.message = tr("preferences.models.operation").format(name=name, message=self.message)
        self._publish()

    @staticmethod
    def _error_text(error, code):
        key = {"FailedPrecondition": "preferences.models.locked", "Unsupported": "preferences.models.not_file"}.get(code)
        return lf.ui.tr(key) if error and key else str(error)

    def _publish(self):
        if self.handle is not None:
            status = (self.message, self.error, bool(self.job.get("running")), bool(self.job.get("cancel_requested")))
            if status == self._last_status:
                return
            self._last_status = status
            for key in ("model_operation_message", "model_operation_error", "model_operation_detail"):
                self.handle.dirty(key)

    def _start(self, model, action):
        if self.handle is None:
            return
        error_message = ""
        try:
            if not lf.ui.model_cache_start(model, action):
                error_message = lf.ui.tr("preferences.models.busy")
        except (RuntimeError, ValueError) as error:
            error_message = str(error)
        self.sync(force=True)
        if error_message and not self.error:
            self.error = error_message
            self.message = lf.ui.tr("preferences.models.failed")
            self._publish()

    def _action(self, _handle, _event, args):
        if not args or len(args) != 2:
            return
        model = int(args[0])
        action = str(args[1])
        record = next((record for record in self._records or () if record["id"] == str(model)), None)
        if record is None or not record.get("can_" + action, False):
            return
        row = next((row for row in self.rows if row["id"] == model), None)
        if row is None:
            return
        if action == "open":
            lf.ui.open_url(str(Path(row["path"]).parent))
        elif action == "remove":
            remove = lf.ui.tr("preferences.models.remove")
            owner = weakref.ref(self)
            token = self._confirmation_token

            def confirmed(button):
                section = owner()
                if section is not None and section.handle is not None and section._confirmation_token is token and button == remove:
                    section._start(model, "remove")

            lf.ui.confirm_dialog(
                remove, lf.ui.tr("preferences.models.remove_confirm").format(name=row["name"]),
                [lf.ui.tr("common.cancel"), remove], confirmed,
            )
        elif action in ("download", "verify"):
            self._start(model, action)

    def cancel_pending_confirmation(self):
        self._confirmation_token = object()

    def _cancel(self, *_args):
        if self.job.get("running"):
            lf.ui.model_cache_cancel(int(self.job["id"]))
            self.sync(force=True)
