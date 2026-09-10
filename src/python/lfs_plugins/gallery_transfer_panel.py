# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Persistent, nonmodal progress for native gallery operations."""
import lichtfeld as lf
from .panels import panel_class
from .types import Panel

__lfs_panel_classes__ = ["GalleryTransferPanel"]
__lfs_panel_ids__ = ["lfs.gallery_transfer"]


def transfer_state(owner):
    state = owner.service.snapshot()
    result = dict(title="Gallery transfer", message="No active transfer", progress=0,
                  detail="", can_pause=False, can_resume=False, resume_id="", pause_label="Pause")
    if state["identity"] != owner._identity:
        result["message"] = "Account changed. Open your gallery to continue."
        return result
    if owner._save_pending or owner._export_pending or owner._import_pending:
        result.update(title=owner._title or "Gallery transfer", message=owner._message or "Preparing scene",
                      progress=owner._export_progress, can_pause=owner._can_pause(), pause_label="Cancel preparation")
        return result
    jobs = state["jobs"]
    if owner._message and not any(j["status"] == "running" for j in jobs):
        result.update(title=owner._title or "Gallery transfer", message=owner._message)
        return result
    job = next((j for j in reversed(jobs) if j["status"] == "running"), jobs[-1] if jobs else None)
    if job is None:
        return result
    total, done = max(1, job["total"]), job["completed"]
    processing = bool(job.get("serverProcessing"))
    status = job["status"]
    message = job["message"]
    if status == "completed":
        done = job["total"]
        message = "Ready in your gallery" if job.get("kind") != "download" else "Downloaded. Open your gallery to import."
    elif processing and status == "running":
        message = "Upload received · " + message
    result.update(title=job["metadata"]["title"], message=message,
                  progress=100 if status == "completed" else min(100, 100 * done / total),
                  detail=f"{min(100, 100 * done / total):.0f}% · {done / 1048576:.1f} / {job['total'] / 1048576:.1f} MB" + (" checked" if processing else ""),
                  can_pause=status == "running", pause_label="Stop waiting" if processing else "Pause transfer",
                  can_resume=not state["busy"] and status in ("paused", "error", "queued"), resume_id=job["id"])
    return result


@panel_class("gallery_transfer")
class GalleryTransferPanel(Panel):
    def __init__(self):
        super().__init__()
        self._handle = None
        self._state = {}
        self._owner = None
        self._key = None

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("gallery_transfer")
        if model is None:
            return
        model.bind_func("panel_label", lambda: "Gallery transfer")
        for field in ("title", "message", "progress", "detail", "can_pause", "can_resume", "pause_label"):
            model.bind_func(field, lambda f=field: self._state.get(f, False if f.startswith("can_") else 0 if f == "progress" else ""))
        model.bind_event("pause", lambda *_: self._action("pause"))
        model.bind_event("resume", lambda *_: self._action("resume", [self._state["resume_id"]]))
        model.bind_event("gallery", lambda *_: self._open_gallery())
        self._handle = model.get_handle()

    def _open_gallery(self):
        lf.ui.set_panel_enabled("lfs.gallery", True)
        lf.ui.set_panel_enabled("lfs.gallery_transfer", False)

    def _action(self, action, args=None):
        if self._owner:
            self._owner._dispatch(action, args or [])

    def on_update(self, doc):
        owner = lf.ui.get_panel_object("lfs.gallery")
        if hasattr(owner, "_load"):
            owner = owner._load()
        self._owner = owner
        key = (owner.service.state_key(), owner._message, owner._export_progress,
               bool(owner._save_pending), bool(owner._export_pending), bool(owner._import_pending))
        if key == self._key:
            return False
        self._key = key
        current = transfer_state(owner)
        if current != self._state:
            self._state = current
            if self._handle:
                self._handle.dirty_all()
            return True
        return False

    def on_unmount(self, doc):
        self._handle = None
        super().on_unmount(doc)
