# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compact navigation for the shared undo history."""

from html import escape
import math

import lichtfeld as lf

from .panels import panel_class
from .types import Panel
from .rml_keys import KI_UP, KI_DOWN, KI_HOME, KI_END, KI_RETURN, KI_SPACE

__lfs_panel_classes__ = ["HistoryPanel"]
__lfs_panel_ids__ = ["lfs.history"]

PAGE_SIZE = 25
LABEL_CHARS = 256


def operation_label(name):
    return lf.ui.tr("history.batch_transform") if name == "transform.batch" else name


def row_markup(name, steps, generation, stack="undo"):
    # Bound text before RmlUi layout, including tooltips. Retained names stay intact.
    name = operation_label(name) or lf.ui.tr("runtime.untitled_change")
    name = name.replace("\n", " ").replace("\r", " ")
    name = name if len(name) <= LABEL_CHARS else name[:LABEL_CHARS] + "…"
    classes = "history-entry" + (" is-next" if steps == 1 else "")
    return (f'<div class="{classes}" data-steps="{steps}" data-generation="{generation}">'
            f'<div class="history-entry-header">'
            f'<button type="button" class="history-disclosure" data-details="{steps}" '
            f'data-generation="{generation}" title="{escape(lf.ui.tr("history.details"), quote=True)}">'
            f'<img class="history-chevron" src="../icon/dropdown-arrow.png"/></button>'
            f'<button type="button" class="btn btn--ghost history-jump" '
            f'data-steps="{steps}" data-generation="{generation}" data-tooltip="scene.history_note">'
            f'<span class="history-index text-muted">{steps}</span>'
            f'<span class="history-label">{escape(name)}</span>'
            f'<img class="history-icon" src="../icon/{stack}.png"/></button></div>'
            f'<div id="history-detail-{steps}" class="history-details" style="display: none;"></div></div>')


def _vector_text(values):
    return ", ".join(format(float(value), ".6g") for value in values)


def _detail_line(key, before, after):
    caption = lf.ui.tr(key).rstrip(":")
    return (f'<div class="history-detail-line"><span class="history-detail-property">{escape(caption)}</span>'
            f'<span class="history-detail-value">{escape(_vector_text(before))}</span>'
            f'<span class="history-detail-value">{escape(_vector_text(after))}</span></div>')


def _orthogonal(matrix):
    columns = [matrix[i:i + 3] for i in (0, 4, 8)]
    lengths = [math.sqrt(sum(value * value for value in column)) for column in columns]
    if not all(math.isfinite(length) and length > 1e-8 for length in lengths):
        return False
    return all(abs(sum(a * b for a, b in zip(columns[i], columns[j])))
               <= 1e-5 * lengths[i] * lengths[j] for i, j in ((0, 1), (0, 2), (1, 2)))


def details_markup(description):
    tr = lf.ui.tr
    title = operation_label(description.get("label", ""))[:LABEL_CHARS]
    result = f'<div class="history-detail-title">{escape(title)}</div>'
    transforms = description.get("transforms", [])
    if not transforms:
        return result + f'<div class="text-muted">{escape(tr("history.no_details"))}</div>'
    result += (f'<div class="history-detail-line text-muted"><span class="history-detail-property"></span>'
               f'<span class="history-detail-value">{escape(tr("history.before"))}</span>'
               f'<span class="history-detail-value">{escape(tr("history.after"))}</span></div>')
    for transform in transforms[:25]:
        target = transform["target"][:LABEL_CHARS]
        result += f'<div class="history-detail-target">{escape(target)}</div>'
        before, after = transform["before"], transform["after"]
        if _orthogonal(before) and _orthogonal(after):
            before_trs = lf.decompose_transform(before)
            after_trs = lf.decompose_transform(after)
            for name, key in (("translation", "transform.position"),
                              ("rotation_euler_deg", "history.rotation_degrees"),
                              ("scale", "transform.scale")):
                if before_trs[name] != after_trs[name]:
                    result += _detail_line(key, before_trs[name], after_trs[name])
        else:
            for row in range(4):
                result += _detail_line("history.matrix", before[row::4], after[row::4])
    if len(transforms) > 25:
        more_text = tr("history.more_targets").format(count=len(transforms) - 25)
        result += f'<div class="text-muted">{escape(more_text)}</div>'
    return result


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
        self._expanded = {}

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
            self._expanded.clear()
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
            for steps, description in self._expanded.items():
                self._show_details(steps, description)
        empty = doc.get_element_by_id("history-empty")
        empty.set_inner_rml(escape(tr("scene.no_entries_in_stack")))
        empty.set_property("display", "none" if names else "block")
        page_text = f"{self._page + 1:n} / {pages:n}"
        doc.get_element_by_id("history-page").set_inner_rml(page_text)
        feedback = doc.get_element_by_id("history-feedback")
        feedback.set_inner_rml(escape(self._feedback))
        feedback.set_property("display", "block" if self._feedback else "none")
        for row in doc.get_element_by_id("history-rows").query_selector_all(".history-jump"):
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
            self._expanded.clear()
        elif action in ("older", "newer"):
            self._page = max(0, self._page + (1 if action == "older" else -1))
        elif action == "clear":
            if not lf.undo.has_active_transaction():
                lf.undo.clear()
        elif action in ("undo", "redo"):
            self._jump(action, 1)
        self._refresh()

    def _show_details(self, steps, description):
        detail = self._doc.get_element_by_id(f"history-detail-{steps}")
        if detail:
            detail.set_inner_rml(details_markup(description))
            detail.set_property("display", "block")
            detail.parent().set_class("is-expanded", True)

    def _toggle_details(self, steps, generation):
        if generation != lf.undo.generation():
            self._refresh()
            return
        if steps in self._expanded:
            del self._expanded[steps]
            detail = self._doc.get_element_by_id(f"history-detail-{steps}")
            detail.set_property("display", "none")
            detail.parent().set_class("is-expanded", False)
        else:
            description = lf.undo.details(self._stack, steps, generation)
            if description is None:
                self._refresh()
                return
            self._expanded[steps] = description
            self._show_details(steps, description)
        lf.ui.request_redraw()

    def _row_target(self, event):
        target = event.target()
        while target and target.id != "history-rows":
            if target.get_attribute("data-steps", ""):
                return target
            target = target.parent()
        return None

    def _on_row(self, event):
        target = event.target()
        while target and target.id != "history-rows":
            if "history-details" in target.get_attribute("class", "").split():
                event.stop_propagation()
                return
            steps = target.get_attribute("data-details", "")
            if steps:
                self._toggle_details(int(steps), int(target.get_attribute("data-generation", "-1")))
                event.stop_propagation()
                return
            target = target.parent()
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
        rows = self._doc.get_element_by_id("history-rows").query_selector_all(".history-jump")
        index = int(row.get_attribute("data-steps", "1")) - 1 - self._page * PAGE_SIZE
        index = (0 if key == KI_HOME else len(rows) - 1 if key == KI_END
                 else max(0, min(len(rows) - 1, index + (1 if key == KI_DOWN else -1))))
        rows[index].focus()
        rows[index].scroll_into_view(False)
        event.stop_propagation()
