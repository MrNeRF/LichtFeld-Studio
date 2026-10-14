# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compact navigation for the shared undo history."""

from html import escape

import lichtfeld as lf

from .panels import panel_class
from .types import Panel
from .rml_keys import KI_UP, KI_DOWN, KI_HOME, KI_END, KI_RETURN, KI_SPACE

__lfs_panel_classes__ = ["HistoryPanel"]
__lfs_panel_ids__ = ["lfs.history"]

PAGE_SIZE = 100
LABEL_CHARS = 256


def row_markup(name, steps, generation, stack="undo"):
    # Bound text before RmlUi layout, including tooltips. Retained names stay intact.
    name = name or lf.ui.tr("runtime.untitled_change")
    name = name.replace("\n", " ").replace("\r", " ")
    name = name if len(name) <= LABEL_CHARS else name[:LABEL_CHARS] + "…"
    classes = "btn btn--ghost history-row" + (" is-next" if steps == 1 else "")
    return (f'<button type="button" class="{classes}" '
            f'data-steps="{steps}" data-generation="{generation}" data-tooltip="scene.history_note">'
            f'<span class="history-index text-muted">{steps}</span>'
            f'<span class="history-label">{escape(name)}</span>'
            f'<img class="history-icon" src="../icon/{stack}.png"/></button>')


@panel_class("history")
class HistoryPanel(Panel):
    def __init__(self):
        super().__init__()
        self._doc = None
        self._handle = None
        self._stamp = None
        self._rows_stamp = None
        self._generation = None
        self._names = {"undo": [], "redo": []}
        self._stack = "undo"
        self._page = 0
        self._feedback = ""

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("history_panel")
        if model is not None:
            model.bind_func("panel_label", lambda: lf.ui.tr("scene.history"))
            self._handle = model.get_handle()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        self._stamp = None
        self._rows_stamp = None
        for name in ("undo", "redo", "clear", "undo-tab", "redo-tab", "newer", "older"):
            doc.get_element_by_id("history-" + name).add_event_listener(
                "click", lambda event, action=name: self._action(action))
        rows = doc.get_element_by_id("history-rows")
        rows.add_event_listener("click", self._on_row)
        rows.add_event_listener("keydown", self._on_key)
        self.on_update(doc)

    def on_unmount(self, doc):
        if doc:
            doc.remove_data_model("history_panel")
        self._doc = None
        self._handle = None
        self._stamp = None
        self._rows_stamp = None
        self._generation = None
        self._names = {"undo": [], "redo": []}

    def _disabled(self, name, disabled):
        element = self._doc.get_element_by_id("history-" + name)
        if disabled:
            element.set_attribute("disabled", "disabled")
        else:
            element.remove_attribute("disabled")

    def on_update(self, doc):
        generation = lf.undo.generation()
        language = lf.ui.get_current_language()
        active = lf.undo.has_active_transaction()
        stamp = (generation, language, active, self._stack, self._page, self._feedback)
        if stamp == self._stamp:
            return
        if generation != self._generation:
            self._names = {"undo": lf.undo.undo_names(), "redo": lf.undo.redo_names()}
            self._generation = generation
            self._page = 0
        names = self._names[self._stack]
        pages = max(1, (len(names) + PAGE_SIZE - 1) // PAGE_SIZE)
        self._page = min(self._page, pages - 1)
        tr = lf.ui.tr
        for stack in ("undo", "redo"):
            tab = doc.get_element_by_id("history-" + stack + "-tab")
            tab_text = tr('scene.' + stack + '_stack')
            count_text = format(len(self._names[stack]), 'n')
            tab.set_inner_rml(f'<span class="history-tab-label">{escape(tab_text)}</span>'
                              f'<span class="history-tab-count">{count_text}</span>')
            tab.set_class("is-active", stack == self._stack)
            self._disabled(stack, active or not self._names[stack])
        self._disabled("clear", active or not any(self._names.values()))
        self._disabled("newer", self._page == 0)
        self._disabled("older", self._page + 1 >= pages)
        first = self._page * PAGE_SIZE
        rows_stamp = (generation, language, self._stack, self._page)
        if rows_stamp != self._rows_stamp:
            doc.get_element_by_id("history-rows").set_inner_rml("".join(
                row_markup(name, first + i + 1, generation, self._stack)
                for i, name in enumerate(names[first:first + PAGE_SIZE])))
            doc.get_element_by_id("history-scroll").scroll_top = 0
            self._rows_stamp = rows_stamp
        empty = doc.get_element_by_id("history-empty")
        empty.set_inner_rml(escape(tr("scene.no_entries_in_stack")))
        empty.set_property("display", "none" if names else "block")
        page_text = f"{self._page + 1:n} / {pages:n}"
        doc.get_element_by_id("history-page").set_inner_rml(page_text)
        feedback = doc.get_element_by_id("history-feedback")
        feedback.set_inner_rml(escape(self._feedback))
        feedback.set_property("display", "block" if self._feedback else "none")
        for row in doc.get_element_by_id("history-rows").children():
            if active:
                row.set_attribute("disabled", "disabled")
            else:
                row.remove_attribute("disabled")
        self._stamp = (generation, language, active, self._stack, self._page, self._feedback)

    def _refresh(self):
        if self._doc:
            self.on_update(self._doc)
            lf.ui.request_redraw()

    def _jump(self, stack, count):
        if lf.undo.has_active_transaction():
            return
        result = lf.undo.jump(stack, count)
        self._feedback = result.get("error", "") if not result.get("success") else ""
        self._refresh()

    def _action(self, action):
        self._feedback = ""
        if action.endswith("-tab"):
            self._stack = action[:-4]
            self._page = 0
        elif action in ("older", "newer"):
            self._page = max(0, self._page + (1 if action == "older" else -1))
        elif action == "clear":
            if not lf.undo.has_active_transaction():
                lf.undo.clear()
        elif action in ("undo", "redo"):
            self._jump(action, 1)
        self._refresh()

    def _row_target(self, event):
        target = event.target()
        while target and target.id != "history-rows":
            if target.get_attribute("data-steps", ""):
                return target
            target = target.parent()
        return None

    def _on_row(self, event):
        row = self._row_target(event)
        if row is None:
            return
        generation = int(row.get_attribute("data-generation", "-1"))
        if generation != lf.undo.generation():
            self._refresh()
            return
        self._jump(self._stack, int(row.get_attribute("data-steps", "0")))
        event.stop_propagation()

    def _on_key(self, event):
        key = int(event.get_parameter("key_identifier", "0"))
        if key in (KI_RETURN, KI_SPACE):
            self._on_row(event)
            return
        if key not in (KI_UP, KI_DOWN, KI_HOME, KI_END):
            return
        row = self._row_target(event)
        if row is None:
            return
        rows = self._doc.get_element_by_id("history-rows").children()
        index = int(row.get_attribute("data-steps", "1")) - 1 - self._page * PAGE_SIZE
        index = (0 if key == KI_HOME else len(rows) - 1 if key == KI_END
                 else max(0, min(len(rows) - 1, index + (1 if key == KI_DOWN else -1))))
        rows[index].focus()
        rows[index].scroll_into_view(False)
        event.stop_propagation()
