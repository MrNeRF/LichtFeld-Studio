# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Persistent, nonmodal progress for native gallery operations."""
import lichtfeld as lf
from functools import partial
from .gallery_messages import tr as gallery_tr

tr = partial(gallery_tr, prefix="gallery.transfer.")
from .panels import panel_class
from .types import Panel
from .gallery_transfer_ui import transfer_rows

__lfs_panel_classes__ = ["GalleryTransferPanel"]
__lfs_panel_ids__ = ["lfs.gallery_transfer"]


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
