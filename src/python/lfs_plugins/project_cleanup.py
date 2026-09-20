# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Preview and run atomic project cleanup without blocking the UI."""
from html import escape
from pathlib import Path
import threading

import lichtfeld as lf

_active = None


def _tr(key):
    return lf.ui.tr("project_cleanup." + key)


def _size(value):
    return f"{value / (1024 * 1024):,.1f} MB"


def _error_message(reason):
    text = str(reason)
    for line in text.splitlines():
        if line.strip().startswith("user_message:"):
            return line.split("user_message:", 1)[1].strip()
    return text


class ProjectCleanup:
    def __init__(self):
        self.path = ""
        self.plan = None
        self.finished = False
        self.cleaning = False
        self.destination = ""
        self.before_bytes = 0
        self.key = "project-cleanup"

    def error(self, reason):
        if self.finished:
            return
        self.finished = True
        message = _error_message(reason)
        if not lf.ui.form_dialog_update(self.key, [{"label": lf.ui.tr("common.ok")}], self.body(message)):
            lf.ui.message_dialog(_tr("title"), message, "error")

    def start(self):
        try:
            state = lf.project_poll_write()
            if state.get("running"):
                raise RuntimeError(_tr("wait"))
            if lf.is_training_active():
                raise RuntimeError(_tr("stop_training"))
            self.path = str(state.get("path") or "")
            if not self.path:
                raise RuntimeError(_tr("save_first"))
            if lf.project_is_dirty():
                def answer(button):
                    if button != lf.ui.tr("common.save"):
                        self.finished = True
                        return
                    try:
                        if not lf.project_save(wait=False, regenerate_preview=False):
                            raise RuntimeError(_tr("save_failed"))
                        self.schedule(self.saved)
                    except Exception as exc:
                        self.error(exc)
                lf.ui.confirm_dialog(_tr("title"), _tr("save_first"),
                    [lf.ui.tr("common.save"), lf.ui.tr("common.cancel")], answer)
            else:
                self.inspect()
        except Exception as exc:
            self.error(exc)

    def schedule(self, callback):
        timer = threading.Timer(0.1, lambda: lf.ui.schedule_on_ui_thread(callback))
        timer.daemon = True
        timer.start()

    def current(self):
        state = lf.project_poll_write()
        if str(state.get("path") or "") != self.path or lf.is_training_active():
            raise RuntimeError(_tr("changed"))
        return state

    def saved(self):
        try:
            state = self.current()
            if state.get("running"):
                self.schedule(self.saved)
                return
            if state.get("error") or lf.project_is_dirty():
                raise RuntimeError(state.get("error") or _tr("save_failed"))
            self.inspect()
        except Exception as exc:
            self.error(exc)

    def inspect(self):
        lf.ui.form_dialog(self.key, _tr("title"), self.body(_tr("reading")),
            [{"label": lf.ui.tr("common.cancel")}], self.choose)
        def worker():
            try:
                plan = lf.io.plan_reduce_size(self.path)
                details = lf.io.inspect_project_details(self.path)
                if str(details.card.commit_uuid) != str(plan.input_commit_uuid):
                    raise RuntimeError(_tr("changed"))
                saves = max(0, len(details.save_history) - 1)
                lf.ui.schedule_on_ui_thread(lambda: self.review(plan, saves))
            except Exception as exc:
                reason = str(exc)
                lf.ui.schedule_on_ui_thread(lambda: self.error(reason))
        threading.Thread(target=worker, daemon=True, name="ProjectCleanupPreview").start()

    def review(self, plan, saves):
        try:
            if self.finished:
                return
            if self.current().get("running") or lf.project_is_dirty():
                raise RuntimeError(_tr("changed"))
            self.plan = plan
            old_checkpoints = sum(not checkpoint.scng_bound for checkpoint in plan.retained_checkpoints)
            estimate = max(0, plan.physical_size - plan.drop_checkpoints.projected_size)
            message = _tr("summary").format(saves=saves, checkpoints=old_checkpoints, size=_size(estimate))
            lf.ui.form_dialog_update(self.key,
                [{"label": _tr("clean_here"), "style": "warning"}, {"label": _tr("copy")},
                 {"label": lf.ui.tr("common.cancel")}], self.body(message))
        except Exception as exc:
            self.error(exc)

    def choose(self, button, *_values):
        if button not in {_tr("clean_here"), _tr("copy")}:
            self.finished = True
            return
        try:
            if self.current().get("running") or lf.project_is_dirty():
                raise RuntimeError(_tr("changed"))
            if button == _tr("copy"):
                source = Path(self.path)
                self.destination = lf.ui.save_project_file_dialog(source.stem + "-clean.licht", str(source.parent))
                if not self.destination:
                    self.finished = True
                    return
                if Path(self.destination).exists():
                    raise RuntimeError(_tr("new_destination"))
            self.before_bytes = self.plan.physical_size
            lf.project_clean(self.destination, str(self.plan.input_commit_uuid))
            self.cleaning = True
            # The modal also prevents edits while the durable snapshot is cleaned.
            lf.ui.form_dialog(self.key, _tr("title"), self.body(_tr("working")),
                [{"label": lf.ui.tr("common.cancel")}], self.cancel)
            self.schedule(self.poll)
        except Exception as exc:
            self.error(exc)

    @staticmethod
    def body(message):
        return '<div class="modal-note">' + escape(str(message)).replace("\n", "<br/>") + '</div>'

    def cancel(self, *_args):
        if self.cleaning:
            lf.project_cancel_cleanup()
            # Dismissing the old modal must not unlock editing while the worker
            # is still finishing or rolling back its atomic write.
            lf.ui.form_dialog(self.key, _tr("title"), self.body(_tr("canceling")),
                [{"label": lf.ui.tr("common.cancel"), "disabled": True}], self.cancel)
        else:
            self.finished = True

    def poll(self):
        try:
            state = lf.project_poll_write()
            if state.get("running"):
                self.schedule(self.poll)
                return
            self.cleaning = False
            if state.get("error"):
                message = _error_message(state["error"])
            else:
                output = self.destination or self.path
                freed = max(0, self.before_bytes - Path(output).stat().st_size)
                message = _tr("done").format(size=_size(freed), path=output)
            self.finished = True
            if not lf.ui.form_dialog_update(self.key, [{"label": lf.ui.tr("common.ok")}], self.body(message)):
                lf.ui.message_dialog(_tr("title"), message)
        except Exception as exc:
            self.cleaning = False
            self.error(exc)


def open_project_cleanup():
    global _active
    if _active is not None and not _active.finished:
        return
    _active = ProjectCleanup()
    _active.start()
