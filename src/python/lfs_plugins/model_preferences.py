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
        running = bool(self.job.get("running"))
        tr = lf.ui.tr
        self.error = str(self.job.get("error", "")) if not running and not self.job.get("canceled") else ""
        if self.error and self.job.get("error_code") == "FailedPrecondition":
            self.error = tr("preferences.models.locked")
        elif self.error and self.job.get("error_code") == "Unsupported":
            self.error = tr("preferences.models.not_file")
        if running:
            stage = str(self.job.get("stage", ""))
            key = "preferences.models.canceling" if self.job.get("cancel_requested") else {
                1: "preferences.models.verifying", 2: "preferences.models.removing",
            }.get(self.job.get("action"), "preferences.models.working")
            if not self.job.get("cancel_requested"):
                if stage.startswith("downloading"):
                    key = "preferences.models.downloading"
                elif stage.startswith("verifying"):
                    key = "preferences.models.verifying"
            self.message = tr(key)
            if stage.endswith("_progress") and not self.job.get("cancel_requested"):
                percent = round(max(0.0, min(1.0, float(self.job.get("progress", 0)))) * 100)
                self.message = tr("gallery.status.progress").format(stage=self.message, percent=percent)
        elif self.error:
            self.message = tr("preferences.models.failed")
            if self.job.get("error_code") in ("FailedPrecondition", "Unsupported"):
                self.message += " " + self.error
        elif self.job.get("canceled"):
            self.message = tr("preferences.models.canceled")
        elif self.job.get("completed"):
            key = {0: "preferences.models.ready", 1: "preferences.models.verified", 2: "preferences.models.removed"}.get(self.job.get("action"), "preferences.models.ready")
            self.message = tr(key)
        else:
            self.message = ""
        records = []
        for row in self.rows:
            present = bool(row.get("present"))
            busy = bool(row.get("busy"))
            previous = next((record for record in self._records or () if record["id"] == str(row["id"])), {})
            # Busy snapshots intentionally omit file metadata. Preserve only the
            # last rendered path/size so an operation does not collapse its card.
            if busy:
                present = bool(previous.get("has_path"))
            in_use = bool(row.get("in_use"))
            status = "in_use" if in_use else "working" if busy else "installed" if present else "missing"
            row_error = str(row.get("error", ""))
            if row_error and row.get("error_code") == "Unsupported":
                row_error = tr("preferences.models.not_file")
            owns_job = row["id"] == self.job.get("model")
            relevant = owns_job and (running or bool(self.error) and (present or self.job.get("action") != 1) or bool(self.job.get("canceled")) or
                                    (bool(self.job.get("completed")) and
                                     (present if self.job.get("action") in (0, 1) else not present)))
            records.append({
                "id": str(row["id"]), "name": row["name"],
                "category": tr(row["category_key"]),
                "status": self.message if relevant else tr("preferences.models.failed" if row_error else "preferences.models." + status),
                "detail": self.error if relevant else row_error,
                "failed": bool(self.error) if relevant else bool(row_error),
                "running": owns_job and running,
                "canceling": owns_job and bool(self.job.get("cancel_requested")),
                "size": previous.get("size", "") if busy else format_size(row.get("bytes", 0)) if present else "",
                "path": str(row.get("path", "")) if present else "",
                "has_path": present,
                "available": present and not busy and not row_error and not (owns_job and bool(self.error)),
                "working": busy,
                "can_download": not running and not busy and not in_use and not row_error,
                "can_verify": present and not running and not busy and not row_error,
                "can_remove": present and not running and not busy and not in_use and not row_error,
                "can_open": present,
            })
        if records != self._records:
            self.handle.update_record_list("managed_models", records)
            self.handle.dirty("managed_models")
            self._records = records
        name = next((row["name"] for row in self.rows if row["id"] == self.job.get("model")), "")
        if name and self.message:
            self.message = tr("preferences.models.operation").format(name=name, message=self.message)
        self._publish()

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
