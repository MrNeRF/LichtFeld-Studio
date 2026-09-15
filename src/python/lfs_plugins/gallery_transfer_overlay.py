# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Nonmodal transfer queue anchored inside the viewport."""
from __future__ import annotations

import lichtfeld as lf

from .gallery_transfer_ui import tr
from .ui import RuntimeState


class GalleryTransferOverlay:
    def __init__(self):
        self._handle = None
        self._state = {}
        self._visible = False
        self._collapsed = False
        self._last_signature = None
        self._message = ""

    def reset(self):
        self._handle = None
        self._last_signature = None

    def bind_model(self, model):
        model.bind_record_list("gallery_transfer_rows")
        bindings = {
            "visible": lambda: self._visible,
            "expanded": lambda: not self._collapsed,
            "empty": lambda: not self._state.get("rows"),
            "header": self._header,
            "message": lambda: self._message or self._state.get("message", ""),
            "message_error": lambda: bool(self._message),
            "toggle_label": lambda: tr("action.expand" if self._collapsed else "action.collapse"),
            "toggle_icon": lambda: "+" if self._collapsed else "−",
            "close_label": lambda: lf.ui.tr("common.close"),
        }
        for name, getter in bindings.items():
            model.bind_func("gallery_transfer_" + name, getter)
        for action in ("pause", "resume", "cancel", "details", "clear_finished", "empty"):
            model.bind_func("gallery_transfer_" + action + "_label", lambda a=action: tr("action." + a))
        model.bind_event("gallery_transfer_action", self._action)
        self._handle = model.get_handle()
        self._last_signature = None

    def _header(self):
        rows = self._state.get("rows", ())
        return tr("queue_header", active=sum(r["status"] == "running" for r in rows),
                  queued=sum(r["status"] == "queued" for r in rows))

    def update(self):
        state = RuntimeState.gallery_transfers.value
        signature = (state, RuntimeState.language_generation.value)
        if not self._handle or signature == self._last_signature:
            return False
        if state.get("identity") != self._state.get("identity"):
            self._visible = self._collapsed = False
            self._message = ""
            self._state = {}
        previous_ids = {r["id"] for r in self._state.get("rows", ())}
        if any(r["id"] not in previous_ids and r["status"] not in ("completed", "canceled") for r in state.get("rows", ())):
            self._visible = True
            self._collapsed = False
        self._state = state
        self._last_signature = signature
        self._handle.update_record_list("gallery_transfer_rows", state.get("rows", []))
        return True

    def show(self):
        self._visible = True
        self._collapsed = False
        if self._handle:
            self._handle.dirty_all()
        lf.ui.request_redraw()

    def _action(self, _handle, _event, args):
        if not args:
            return
        action = str(args[0])
        if action == "toggle":
            self._collapsed = not self._collapsed
        elif action == "close":
            self._visible = False
        elif action == "details":
            lf.ui.set_panel_enabled("lfs.gallery_transfer", True)
        else:
            from .gallery_controller import get_gallery_controller
            from .gallery_messages import localize_message
            identifier = str(args[1]) if len(args) > 1 else None
            try:
                get_gallery_controller().command("pause" if identifier == "native" else action,
                                                 None if identifier == "native" else identifier)
                self._message = ""
            except Exception as exc:
                self._message = localize_message(str(exc))
        if self._handle:
            self._handle.dirty_all()
        lf.ui.request_redraw()
