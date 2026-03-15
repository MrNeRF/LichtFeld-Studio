# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Floating undo/redo history panel."""

from __future__ import annotations

import lichtfeld as lf

from .types import Panel


def _format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB")
    amount = float(max(value, 0))
    unit_index = 0
    while amount >= 1024.0 and unit_index < len(units) - 1:
        amount /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(amount)} {units[unit_index]}"
    return f"{amount:.1f} {units[unit_index]}"


class HistoryPanel(Panel):
    id = "lfs.history"
    label = "History"
    space = lf.ui.PanelSpace.FLOATING
    order = 16
    template = "rmlui/history_panel.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (440, 0)
    # Keep history surfacing responsive enough that subscription-triggered redraws
    # land inside an update window and refresh the panel on the next frame.
    update_interval_ms = 50

    def __init__(self):
        self._handle = None
        self._subscription_id = 0
        self._last_state_key = None
        self._undo_label = "Undo"
        self._redo_label = "Redo"
        self._summary_text = "No history yet"
        self._transaction_label = ""
        self._empty_text = "Nothing recorded yet"
        self._has_undo = False
        self._has_redo = False
        self._show_transaction = False
        self._can_clear = False

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("history_panel")
        if model is None:
            return

        model.bind_func("panel_label", lambda: "History")
        model.bind_func("undo_label", lambda: self._undo_label)
        model.bind_func("redo_label", lambda: self._redo_label)
        model.bind_func("summary_text", lambda: self._summary_text)
        model.bind_func("transaction_label", lambda: self._transaction_label)
        model.bind_func("show_transaction", lambda: self._show_transaction)
        model.bind_func("empty_text", lambda: self._empty_text)
        model.bind_func("has_undo", lambda: self._has_undo)
        model.bind_func("has_redo", lambda: self._has_redo)
        model.bind_func("can_clear", lambda: self._can_clear)
        model.bind_event("do_undo", self._on_undo)
        model.bind_event("do_redo", self._on_redo)
        model.bind_event("clear_history", self._on_clear)
        model.bind_event("jump_to", self._on_jump_to)
        model.bind_record_list("undo_items")
        model.bind_record_list("redo_items")
        self._handle = model.get_handle()

    def on_mount(self, doc):
        del doc
        self._last_state_key = None
        self._subscription_id = int(lf.undo.subscribe(self._on_history_changed) or 0)
        self._refresh(force=True)

    def on_update(self, doc):
        del doc
        # History subscriptions cover most edits, but a periodic poll keeps the panel
        # honest when the UI misses a callback for residency-only changes.
        return self._refresh(force=False)

    def on_scene_changed(self, doc):
        del doc
        self._last_state_key = None
        self._refresh(force=True)

    def on_unmount(self, doc):
        if self._subscription_id:
            lf.undo.unsubscribe(self._subscription_id)
        self._subscription_id = 0
        doc.remove_data_model("history_panel")
        self._handle = None

    def _on_undo(self, _handle=None, _ev=None, _args=None):
        if lf.undo.can_undo():
            lf.undo.undo()
            self._last_state_key = None

    def _on_redo(self, _handle=None, _ev=None, _args=None):
        if lf.undo.can_redo():
            lf.undo.redo()
            self._last_state_key = None

    def _on_clear(self, _handle=None, _ev=None, _args=None):
        if self._can_clear:
            lf.undo.clear()
            self._last_state_key = None

    def _on_jump_to(self, _handle=None, _ev=None, args=None):
        if not args or len(args) < 2:
            return
        stack = str(args[0])
        count = max(0, int(args[1]))
        if count <= 0:
            return
        lf.undo.jump(stack, count)
        self._last_state_key = None

    def _on_history_changed(self):
        self._last_state_key = None
        # Undo observers may be notified off the UI thread; defer the actual
        # Rml data-model mutation to on_update and only wake the render loop here.
        self._request_redraw()

    def _request_redraw(self):
        request_redraw = getattr(getattr(lf, "ui", None), "request_redraw", None)
        if callable(request_redraw):
            request_redraw()

    def _refresh(self, force: bool) -> bool:
        if not self._handle:
            return False

        state = lf.undo.stack()
        undo_items = state.get("undo", [])
        redo_items = state.get("redo", [])
        state_key = (
            tuple(
                (item.get("id", ""), item.get("label", ""), item.get("source", ""),
                 item.get("scope", ""), int(item.get("estimated_bytes", 0)),
                 int(item.get("cpu_bytes", 0)), int(item.get("gpu_bytes", 0)))
                for item in undo_items
            ),
            tuple(
                (item.get("id", ""), item.get("label", ""), item.get("source", ""),
                 item.get("scope", ""), int(item.get("estimated_bytes", 0)),
                 int(item.get("cpu_bytes", 0)), int(item.get("gpu_bytes", 0)))
                for item in redo_items
            ),
            bool(state.get("transaction_active", False)),
            int(state.get("transaction_depth", 0)),
            state.get("transaction_name", ""),
            int(state.get("total_bytes", 0)),
            int(state.get("total_cpu_bytes", 0)),
            int(state.get("total_gpu_bytes", 0)),
        )

        if not force and state_key == self._last_state_key:
            return False

        self._last_state_key = state_key
        self._has_undo = bool(undo_items)
        self._has_redo = bool(redo_items)
        self._undo_label = f"Undo: {undo_items[0]['label']}" if undo_items else "Undo"
        self._redo_label = f"Redo: {redo_items[0]['label']}" if redo_items else "Redo"
        total_bytes = int(state.get("total_bytes", 0))
        total_gpu_bytes = int(state.get("total_gpu_bytes", total_bytes))
        if not undo_items and not redo_items:
            self._summary_text = "No history yet"
        elif total_gpu_bytes < total_bytes:
            self._summary_text = (
                f"{len(undo_items)} undo / {len(redo_items)} redo · "
                f"{_format_bytes(total_bytes)} total · {_format_bytes(total_gpu_bytes)} GPU"
            )
        else:
            self._summary_text = f"{len(undo_items)} undo / {len(redo_items)} redo · {_format_bytes(total_bytes)}"
        transaction_name = state.get("transaction_name", "") or "Grouped changes"
        transaction_depth = int(state.get("transaction_depth", 0))
        self._show_transaction = bool(state.get("transaction_active", False))
        self._can_clear = self._has_undo or self._has_redo or self._show_transaction
        self._transaction_label = (
            f"Transaction active: {transaction_name} (depth {transaction_depth})"
            if self._show_transaction
            else ""
        )
        self._empty_text = (
            "Nothing recorded yet" if not undo_items and not redo_items else "No entries in this stack"
        )

        self._handle.update_record_list("undo_items", self._build_rows(undo_items, kind="undo"))
        self._handle.update_record_list("redo_items", self._build_rows(redo_items, kind="redo"))
        self._handle.dirty("undo_label")
        self._handle.dirty("redo_label")
        self._handle.dirty("summary_text")
        self._handle.dirty("transaction_label")
        self._handle.dirty("show_transaction")
        self._handle.dirty("empty_text")
        self._handle.dirty("has_undo")
        self._handle.dirty("has_redo")
        self._handle.dirty("can_clear")
        self._request_redraw()
        return True

    def _build_rows(self, items, kind: str):
        rows = []
        for index, item in enumerate(items):
            estimated_bytes = int(item.get("estimated_bytes", 0))
            gpu_bytes = int(item.get("gpu_bytes", estimated_bytes))
            if gpu_bytes < estimated_bytes:
                size_meta = f"{_format_bytes(estimated_bytes)} · GPU {_format_bytes(gpu_bytes)}"
            else:
                size_meta = _format_bytes(estimated_bytes)
            scope_text = str(item.get("scope", "general")).replace("_", " ")
            source_text = str(item.get("source", "system"))
            next_label = "NEXT UNDO" if kind == "undo" else "NEXT REDO"
            rows.append(
                {
                    "label": item.get("label", ""),
                    "title_line": f"● {item.get('label', '')}",
                    "stack_line": (
                        f"{next_label} · Top of stack"
                        if index == 0
                        else f"{scope_text} · {source_text}"
                    ),
                    "detail_line": (
                        f"{scope_text} · {source_text} · Size: {size_meta}"
                        if index == 0
                        else f"Size: {size_meta}"
                    ),
                    "scope": scope_text,
                    "source": source_text,
                    "size": size_meta,
                    "is_next": index == 0,
                    "kind": kind,
                    "steps": index + 1,
                }
            )
        return rows
