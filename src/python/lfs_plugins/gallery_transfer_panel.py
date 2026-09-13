# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Persistent, nonmodal progress for native gallery operations."""
import lichtfeld as lf
from .panels import panel_class
from .types import Panel
from .asset_format import format_size

__lfs_panel_classes__ = ["GalleryTransferPanel"]
__lfs_panel_ids__ = ["lfs.gallery_transfer"]


def tr(key, **values):
    from .localization import safe_format
    full_key = "gallery.transfer." + key
    return safe_format(lf.ui.tr(full_key), **values)


def transfer_rows(snapshot, history_limit=30):
    pending, history = [], []
    for job in snapshot.get("jobs", []):
        if job.get("retired"):
            continue
        status = job["status"]
        done, total = job.get("completed", 0), job.get("total", 0)
        processing = bool(job.get("serverProcessing"))
        phase = "processing" if processing and status == "running" else (
            "downloading" if job.get("kind") == "download" else "uploading") if status == "running" else (
            "interrupted" if job.get("interrupted") and status == "paused" else status)
        row = {"id": job["id"], "title": job.get("metadata", {}).get("title", ""),
               "direction": "↓" if job.get("kind") == "download" else "↑",
               "bytes": (format_size(total) if status == "completed" else format_size(done) if status == "canceled"
                         else tr("bytes", done=format_size(done), total=format_size(total))),
               "phase": tr("phase." + phase), "reason": job.get("message", "") if status in ("error", "conflict") else "",
               "progress": 100 if status == "completed" else min(100, 100 * done / max(1, total)),
               "can_pause": status == "running", "can_resume": status in ("paused", "error", "queued") and not snapshot.get("busy"),
               "can_cancel": status not in ("completed", "canceled")}
        (history if status in ("completed", "canceled") else pending).append(row)
    if snapshot.get("phase", "idle") != "idle":
        pending.insert(0, {"id": "native", "title": tr("title"), "direction": "↓" if snapshot["phase"] == "applying" else "↑",
            "bytes": "", "phase": tr("phase." + snapshot["phase"]), "reason": "",
            "progress": snapshot.get("preparationProgress", 0), "can_pause": False,
            "can_resume": False, "can_cancel": True})
    if snapshot.get("batchQueued"):
        pending.insert(0, {"id": "batch-queue", "title": tr("batch", count=snapshot["batchQueued"]),
            "direction": "↑", "bytes": "", "phase": tr("phase.queued"), "reason": "", "progress": 0,
            "can_pause": False, "can_resume": False, "can_cancel": False})
    return pending + list(reversed(history))[:history_limit]


@panel_class("gallery_transfer")
class GalleryTransferPanel(Panel):
    def __init__(self):
        super().__init__()
        self._handle = None
        self._state = {}
        self._unsubscribe = None
        self._history_limit = 30
        self._owner = None

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("gallery_transfer")
        if model is None:
            return
        model.bind_func("panel_label", lambda: tr("title"))
        model.bind_func("header", lambda: tr("header", active=int(self._state.get("phase", "idle") != "idle") + sum(j.get("status") in ("running", "queued") for j in self._state.get("jobs", [])),
            paused=sum(j.get("status") == "paused" for j in self._state.get("jobs", []))))
        model.bind_func("interrupted", lambda: sum(j.get("status") == "paused" and j.get("interrupted", False) for j in self._state.get("jobs", [])))
        model.bind_func("recovery", lambda: tr("recovery", count=sum(j.get("status") == "paused" and j.get("interrupted", False) for j in self._state.get("jobs", []))))
        model.bind_func("message", lambda: self._state.get("message", ""))
        for key in ("pause", "resume", "cancel", "resume_all", "clear_finished", "show_older", "gallery", "empty"):
            translation_key = "action." + key
            model.bind_func(key + "_label", lambda k=translation_key: tr(k))
        model.bind_func("has_jobs", lambda: bool(self._state.get("jobs")) or self._state.get("phase", "idle") != "idle")
        model.bind_record_list("jobs")
        for action in ("pause", "resume", "cancel", "resume_all", "clear_finished", "show_older", "gallery"):
            model.bind_event(action, lambda _h, _e, args, a=action: self._action(a, args))
        self._handle = model.get_handle()

    def _changed(self, state):
        self._state = state
        if self._handle:
            self._handle.update_record_list("jobs", transfer_rows(state, self._history_limit))
            self._handle.dirty_all()
        lf.ui.request_redraw()

    def _action(self, action, args=()):
        if action == "gallery":
            lf.ui.set_panel_enabled("lfs.asset_manager", True)
            panel = lf.ui.get_panel_object("lfs.asset_manager")
            if panel:
                panel.focus_gallery()
        elif action == "show_older":
            self._history_limit += 30
            self._changed(self._state)
        elif self._owner:
            identifier = args[0] if args else None
            if identifier == "native":
                self._owner.command("pause")
            else:
                try:
                    self._owner.command(action, identifier)
                except Exception as exc:
                    from .gallery_messages import localize_message
                    self._changed(dict(self._state, message=localize_message(str(exc))))

    def on_mount(self, doc):
        super().on_mount(doc)
        from .gallery_controller import get_gallery_controller
        self._owner = get_gallery_controller()
        self._unsubscribe = self._owner.subscribe(self._changed)

    def on_unmount(self, doc):
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        self._handle = None
        super().on_unmount(doc)
