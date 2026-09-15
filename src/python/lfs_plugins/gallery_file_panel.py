# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""File-scoped gallery review, separate from the asset selection and transfers."""
from copy import deepcopy
from functools import partial
from pathlib import Path

import lichtfeld as lf

from .gallery_messages import localize_message, tr as gallery_tr
from .panels import panel_class
from .types import Panel

tr = partial(gallery_tr, prefix="asset_manager.gallery.")
__lfs_panel_classes__ = ["GalleryFilePanel"]
__lfs_panel_ids__ = ["lfs.gallery_file"]


def open_gallery_file_panel(**review):
    panel = lf.ui.get_panel_object("lfs.gallery_file")
    if panel:
        panel.show(**review)


@panel_class("gallery_file")
class GalleryFilePanel(Panel):
    def __init__(self):
        super().__init__()
        self._review = None
        self._fields = {}
        self._state = {}
        self._error = ""
        self._handle = None
        self._unsubscribe = None
        self._submitting = False

    def poll(self, _context):
        return self._review is not None

    def show(self, *, controller, asset, scene, action, fields, includes="", quota="",
             warning="", publish_new=False, open_after=False, on_done=None):
        identity = controller.service.identity()
        key = (identity, asset["id"], action, publish_new, open_after)
        if self._review and self._review["key"] == key:
            return
        self._finish(False)
        self._review = dict(controller=controller, asset=deepcopy(asset), scene=deepcopy(scene),
                            action=action, identity=identity, key=key, includes=includes,
                            quota=quota, warning=warning, publish_new=publish_new,
                            open_after=open_after, on_done=on_done)
        self._fields = dict(fields)
        self._error = ""
        if self._unsubscribe:
            self._unsubscribe()
        self._unsubscribe = controller.subscribe(self._changed)
        self._dirty()
        lf.ui.set_panel_enabled(self.id, True)

    def _changed(self, state):
        self._state = state
        if self._review and state.get("identity") != self._review["identity"]:
            self._close(False)
        self._dirty()

    def _dirty(self):
        if self._handle:
            self._handle.dirty_all()
        lf.ui.request_redraw()

    def _set(self, name, value):
        value = str(value)
        if name == "upload_format" and value not in ("studio", "sog", "ssog", "spz"):
            return
        if name == "visibility" and value not in ("private", "public"):
            return
        self._fields[name] = value
        self._error = ""
        self._dirty()

    def _is_pull(self):
        return bool(self._review and self._review["action"] == "pull")

    def _can_submit(self):
        if not self._review or self._submitting or self._state.get("busy"):
            return False
        if not self._state.get("signed_in") or self._state.get("relink_required"):
            return False
        if self._is_pull():
            return bool(self._fields.get("pull_folder", "").strip() and self._fields.get("pull_name", "").strip())
        return 0 < len(self._fields.get("title", "").strip()) <= 120 and len(self._fields.get("description", "")) <= 5000

    def _submit_label(self):
        if self._is_pull():
            return tr("action.pull_open" if self._review["open_after"] else "action.pull")
        if self._review and self._review["action"] == "update":
            return tr("action.update")
        return tr("review.publish_public" if self._fields.get("visibility") == "public" else "review.publish_private")

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("gallery_file")
        if model is None:
            return
        for name in ("title", "description", "visibility", "upload_format", "pull_folder", "pull_name"):
            model.bind(name, lambda n=name: self._fields.get(n, ""), lambda v, n=name: self._set(n, v))
        values = {
            "panel_label": lambda: tr("dialog.download" if self._is_pull() else "dialog.upload"),
            "file_name": lambda: (self._review or {}).get("asset", {}).get("name", ""),
            "is_pull": self._is_pull,
            "show_format": lambda: not self._is_pull(),
            "can_submit": self._can_submit,
            "submit_label": self._submit_label,
            "format_hint": lambda: tr("format." + self._fields.get("upload_format", "sog") + "_hint"),
            "includes": lambda: (self._review or {}).get("includes", ""),
            "quota": lambda: (self._review or {}).get("quota", ""),
            "warning": lambda: (self._review or {}).get("warning", ""),
            "error": lambda: self._error,
            "has_error": lambda: bool(self._error),
            "waiting": lambda: bool(self._state.get("busy")),
            "disconnected": lambda: not bool(self._state.get("signed_in")) or bool(self._state.get("relink_required")),
        }
        for name, getter in values.items():
            model.bind_func(name, getter)
        for key in ("review.title", "review.description", "review.visibility", "review.upload_as", "review.private", "review.public",
                    "format.studio", "format.sog", "format.ssog", "format.spz", "info.folder", "info.filename", "action.cancel"):
            model.bind_func("g_" + key.replace(".", "_"), lambda k=key: tr(k))
        model.bind_event("submit", lambda _h, _e, _args: self._submit())
        model.bind_event("cancel", lambda _h, _e, _args: self._close(False))
        self._handle = model.get_handle()

    def _submit(self):
        if not self._can_submit():
            return
        review = self._review
        controller = review["controller"]
        if controller.service.identity() != review["identity"]:
            self._close(False)
            return
        self._submitting = True
        try:
            if self._is_pull():
                name = self._fields["pull_name"].strip()
                if name in (".", "..") or "/" in name or "\\" in name:
                    raise ValueError(tr("error.destination"))
                controller.pull_asset(review["asset"], review["scene"],
                                      str(Path(self._fields["pull_folder"].strip()) / name),
                                      open_after=review["open_after"])
            else:
                details = {k: self._fields[k].strip() for k in ("title", "description", "visibility")}
                controller.upload_format = self._fields["upload_format"]
                controller.publish_asset(review["asset"], details, self._fields["upload_format"],
                                         update=review["action"] == "update", publish_as_new=review["publish_new"])
            self._close(True)
        except Exception as exc:
            self._error = localize_message(str(exc))
        finally:
            self._submitting = False
            self._dirty()

    def _finish(self, submitted):
        review, self._review = self._review, None
        if review and review["on_done"]:
            review["on_done"](submitted)

    def _close(self, submitted):
        self._finish(submitted)
        lf.ui.set_panel_enabled(self.id, False)

    def on_mount(self, doc):
        close_btn = doc.get_element_by_id("close-btn")
        if close_btn:
            close_btn.add_event_listener("click", lambda _event: self._close(False))
        from . import rml_widgets
        from .rml_keys import KI_ESCAPE, KI_RETURN

        def keydown(event):
            key = int(event.get_parameter("key_identifier", "0"))
            if key == KI_ESCAPE:
                self._close(False)
                event.stop_propagation()
            elif key == KI_RETURN and event.target().tag_name != "textarea":
                self._submit()
                event.stop_propagation()

        doc.add_event_listener("keydown", keydown)
        title = doc.get_element_by_id("gallery-file-title")
        if title and not self._is_pull():
            rml_widgets.bind_select_all_on_focus(title)
            title.focus()

    def on_unmount(self, doc):
        self._finish(False)
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        self._handle = None
        super().on_unmount(doc)
