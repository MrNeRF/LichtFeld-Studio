# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, searchable session logs in an independent utility window."""

from datetime import datetime
from bisect import bisect_left
from html import escape
from itertools import islice
from _lfs_path_io import Path
import re
import time

import lichtfeld as lf

from .panels import panel_class
from . import rml_widgets as w
from .types import Panel
from .rml_keys import KI_A, KI_C, KI_ESCAPE

__lfs_panel_classes__ = ["LoggingPanel"]
__lfs_panel_ids__ = ["lfs.logging"]

LEVELS = ("trace", "debug", "info", "performance", "warn", "error", "critical", "off")
PAGE_SIZES = (100, 250, 500, 1000)
# A row shows at most this much of a message; copy and export keep all of it.
# RmlUi lays out a long unbroken message so slowly that the GUI stops presenting.
ROW_MESSAGE_CHARS = 2048


def merge_snapshot(previous, update):
    """Apply an incremental tail and prune entries evicted by count/byte limits."""
    if previous is None or not update.get("incremental", False):
        return update
    first_sequence = update["first_sequence"]
    first = bisect_left(previous["entries"], first_sequence, key=lambda entry: entry["sequence"])
    entries = previous["entries"][first:]
    if not update["entry_count"]:
        entries = []
    entries.extend(update["entries"])
    return dict(update, entries=entries)


def matching_entries(entries, query="", source="", level=-1):
    """Match all space-separated search terms, a source substring, and severity."""
    terms = query.casefold().split()
    source = source.casefold().strip()
    if not terms and not source and level == -1:
        return entries
    result = []
    for entry in entries:
        severity = int(entry["level"])
        if level == -2 and severity < 4:
            continue
        if level >= 0 and severity != level:
            continue
        if not terms and not source:
            result.append(entry)
            continue
        location = f"{entry['file'] or 'runtime'}:{entry['line']}".casefold()
        if source and source not in location:
            continue
        searchable = f"{location} {entry['message']}".casefold()
        if all(term in searchable for term in terms):
            result.append(entry)
    return result


def entry_time(entry):
    return datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S.%f")[:-3]


def entries_text(entries):
    return "".join(
        f"[{entry_time(entry)}] [{LEVELS[entry['level']]}] "
        f"{entry['file'] or 'runtime'}:{entry['line']} {entry['message']}\n"
        for entry in entries
    )


def row_message(message):
    return message if len(message) <= ROW_MESSAGE_CHARS else message[:ROW_MESSAGE_CHARS] + "\u2026"


def entry_contents(entry):
    level = LEVELS[entry["level"]]
    label_key = f"scene.log_{level}"
    location = f"{entry['file'] or 'runtime'}:{entry['line']}"
    return (
        f'<span class="log-time text-muted">{entry_time(entry)}</span>'
        f'<span class="log-level">{escape(lf.ui.tr(label_key))}</span>'
        f'<span class="log-source text-muted" title="{escape(location, quote=True)}">'
        f'{escape(location)}</span>'
        f'<span class="log-message">{escape(row_message(entry["message"]))}</span>'
    )


def entry_markup(entry):
    return f'<div data-log-sequence="{entry["sequence"]}" class="log-row log-row--{LEVELS[entry["level"]]}">{entry_contents(entry)}</div>'


@panel_class("logging")
class LoggingPanel(Panel):
    def __init__(self):
        super().__init__()
        self._doc = None
        self._handle = None
        self._snapshot = None
        self._query = ""
        self._source = ""
        self._level = "-1"
        self._capture = str(lf.log.capture_level())
        self._limit = "5000"
        self._last_buffer_limit = None
        self._editing_limit = False
        self._escape_revert = w.EscapeRevertController()
        self._page_size = "250"
        self._paused = False
        self._wrap = True
        self._page = 0
        self._summary = ""
        self._feedback = ""
        self._feedback_deadline = 0.0
        self._dirty = True
        self._matched = []
        self._matches_dirty = True
        self._last_language = ""
        self._row_nodes = []
        self._row_locale = ""
        self._selected_sequences = set()
        self._rendered_selection = set()
        self._selection_anchor = None
        self._drag_anchor = None
        self._drag_base = set()
        self._drag_last = None
        self._dragging = False

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("logging_panel")
        if model is None:
            return
        model.bind_func("panel_label", lambda: lf.ui.tr("scene.logging"))
        for name in ("query", "source", "level", "page_size"):
            model.bind(name, lambda n=name: getattr(self, "_" + n),
                       lambda value, n=name: self._set_filter(n, value))
        model.bind("capture", lambda: self._capture, self._set_capture)
        model.bind("limit", lambda: self._limit, self._set_limit)
        model.bind("wrap", lambda: self._wrap, self._set_wrap)
        model.bind_func("paused", lambda: self._paused)
        model.bind_func("summary", lambda: self._summary)
        model.bind_func("feedback", lambda: self._feedback)
        for name in ("pause", "clear", "reset_filters", "copy", "export", "older", "newer"):
            model.bind_event(name, getattr(self, "_on_" + name))
        self._handle = model.get_handle()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        self._row_nodes = []
        self._rendered_selection = set()
        rows = doc.get_element_by_id("log-rows")
        rows.add_event_listener("mousedown", self._on_selection_down)
        doc.add_event_listener("mousemove", self._on_selection_move)
        doc.add_event_listener("mouseup", self._on_selection_up)
        doc.get_element_by_id("log-scroll").add_event_listener("keydown", self._on_selection_key)
        w.bind_committed_text_input(
            doc.get_element_by_id("log-limit"), "limit", commit=self._commit_limit,
            escape_revert=self._escape_revert, capture=lambda: self._limit, restore=self._restore_limit,
            on_focus=lambda key: setattr(self, "_editing_limit", True),
            on_blur=lambda key: setattr(self, "_editing_limit", False),
        )
        self._dirty = True
        self.on_update(doc)

    def on_unmount(self, doc):
        if doc:
            doc.remove_data_model("logging_panel")
        self._doc = None
        self._handle = None
        self._row_nodes = []
        self._dragging = False

    def _invalidate(self):
        self._dirty = True
        if self._handle:
            self._handle.dirty_all()
        if self._doc:
            self.on_update(self._doc)
        lf.ui.request_redraw()

    def _set_filter(self, name, value):
        value = str(value)
        if name == "level" and value not in {"-2", "-1", *(str(i) for i in range(7))}:
            return
        if name == "page_size" and value not in {str(i) for i in PAGE_SIZES}:
            return
        if value == getattr(self, "_" + name):
            return
        setattr(self, "_" + name, value)
        if name in {"query", "source", "level"}:
            self._matches_dirty = True
        self._page = 0
        self._invalidate()

    def _set_capture(self, value):
        level = int(value)
        if not 0 <= level <= 7:
            return
        lf.log.set_capture_level(level)
        self._capture = str(level)
        self._invalidate()

    def _set_limit(self, value):
        # Editing a number must not trim the buffer halfway through typing it.
        self._limit = str(value)

    def _restore_limit(self, snapshot):
        # Escape cancels the edit; the blur that follows commits the unchanged value.
        self._limit = str(snapshot)
        if self._handle:
            self._handle.dirty("limit")

    def _commit_limit(self, *_):
        # Accept digit grouping in any locale ("50,000", "50.000", "50 000").
        text = self._limit.strip()
        number = re.fullmatch(r"\d+|\d{1,3}(?:[,.'\u00a0\u202f ]\d{3})+", text, re.ASCII)
        limit = int(re.sub(r"\D", "", text, flags=re.ASCII)) if number else -1
        # Blank, negative or too large input must not clear or trim the buffer.
        if not 0 <= limit <= 50000:
            self._feedback = lf.ui.tr("logging.limit_error")
            self._feedback_deadline = 0.0
            self._invalidate()
            return
        lf.log.set_buffer_limit(limit)
        self._limit = str(limit)
        self._last_buffer_limit = limit
        # Apply destructive retention changes even while the display is paused.
        self._snapshot = lf.log.snapshot()
        self._matches_dirty = True
        self._page = 0
        self._feedback = ""
        self._invalidate()

    def _set_wrap(self, value):
        self._wrap = bool(value)
        self._invalidate()

    def _on_pause(self, *_):
        self._paused = not self._paused
        self._page = 0
        self._invalidate()

    def _on_clear(self, *_):
        lf.log.clear_buffer()
        self._snapshot = lf.log.snapshot()
        self._matches_dirty = True
        self._page = 0
        self._feedback = ""
        self._invalidate()

    def _on_reset_filters(self, *_):
        self._query = self._source = ""
        self._level = "-1"
        self._matches_dirty = True
        self._page = 0
        self._invalidate()

    def _refresh_matches(self):
        if not self._matches_dirty:
            return
        entries = self._snapshot["entries"] if self._snapshot else []
        self._matched = matching_entries(entries, self._query, self._source, int(self._level))
        self._matches_dirty = False
        if self._selected_sequences:
            self._selected_sequences.intersection_update(entry["sequence"] for entry in self._matched)
        if self._selection_anchor not in self._selected_sequences:
            self._selection_anchor = None

    def _append_matches(self, update):
        """Prune evictions and evaluate filters only against the new snapshot tail."""
        first = bisect_left(self._matched, update["first_sequence"], key=lambda entry: entry["sequence"])
        self._matched = self._matched[first:] if update["entry_count"] else []
        if update["entries"]:
            self._matched.extend(matching_entries(update["entries"], self._query, self._source, int(self._level)))
        self._selected_sequences = {sequence for sequence in self._selected_sequences
                                    if update["entry_count"] and sequence >= update["first_sequence"]}
        if self._selection_anchor is not None and self._selection_anchor < update["first_sequence"]:
            self._selection_anchor = None

    def _selection_sequence(self, event):
        if self._doc is None:
            return None
        row = w.find_ancestor_with_attribute(event.target(), "data-log-sequence", self._doc)
        return int(row.get_attribute("data-log-sequence", "0")) if row is not None else None

    def _selection_range(self, anchor, sequence):
        low, high = sorted((anchor, sequence))
        start = bisect_left(self._matched, low, key=lambda entry: entry["sequence"])
        selected = set()
        for entry in islice(self._matched, start, None):
            if entry["sequence"] > high:
                break
            selected.add(entry["sequence"])
        return selected

    def _sync_selection(self):
        if not self._selected_sequences and not self._rendered_selection:
            return
        visible = {sequence for sequence, node in self._row_nodes if sequence in self._selected_sequences}
        changed = visible ^ self._rendered_selection
        if not changed:
            return
        for sequence, node in self._row_nodes:
            if sequence in changed:
                node.set_class("is-selected", sequence in visible)
        self._rendered_selection = visible
        lf.ui.request_redraw()

    def _on_selection_down(self, event):
        if int(event.get_parameter("button", "0")) != 0:
            return
        sequence = self._selection_sequence(event)
        if sequence is None:
            return
        additive = any(event.get_bool_parameter(key, False) for key in ("ctrl_key", "meta_key", "command_key"))
        extend = event.get_bool_parameter("shift_key", False) and self._selection_anchor is not None
        self._drag_base = self._selected_sequences.copy() if additive else set()
        self._drag_anchor = self._selection_anchor if extend else sequence
        if extend:
            self._selected_sequences = self._drag_base | self._selection_range(self._drag_anchor, sequence)
        elif additive:
            self._selected_sequences.symmetric_difference_update({sequence})
        else:
            self._selected_sequences = {sequence}
        if not extend:
            self._selection_anchor = sequence
        self._dragging = True
        self._drag_last = sequence
        self._doc.get_element_by_id("log-scroll").focus()
        self._sync_selection()
        event.stop_propagation()

    def _on_selection_move(self, event):
        if not self._dragging:
            return
        sequence = self._selection_sequence(event)
        if sequence is None or sequence == self._drag_last:
            return
        self._drag_last = sequence
        self._selected_sequences = self._drag_base | self._selection_range(self._drag_anchor, sequence)
        self._sync_selection()

    def _on_selection_up(self, event):
        if not self._dragging or int(event.get_parameter("button", "0")) != 0:
            return
        self._dragging = False
        self._invalidate()

    def _on_selection_key(self, event):
        key = int(event.get_parameter("key_identifier", "0"))
        control = any(event.get_bool_parameter(name, False) for name in ("ctrl_key", "meta_key", "command_key"))
        if control and key == KI_C and self._selected_sequences:
            selected = [entry for entry in reversed(self._matched) if entry["sequence"] in self._selected_sequences]
            lf.ui.set_clipboard_text(entries_text(selected))
        elif control and key == KI_A:
            self._selected_sequences = {entry["sequence"] for entry in self._matched}
            self._sync_selection()
        elif key == KI_ESCAPE and self._selected_sequences:
            self._selected_sequences.clear()
            self._selection_anchor = None
            self._sync_selection()
        else:
            return
        event.stop_propagation()

    def _on_copy(self, *_):
        self._refresh_matches()
        if not self._matched:
            return
        lf.ui.set_clipboard_text(entries_text(self._matched))
        self._feedback = lf.ui.tr("logging.copied").format(count=len(self._matched))
        self._feedback_deadline = time.monotonic() + 4.0
        self._invalidate()

    def _on_export(self, *_):
        self._refresh_matches()
        if not self._matched:
            return
        path = lf.ui.save_text_file_dialog("lichtfeld_logs")
        if not path:
            return
        try:
            Path(path).write_text(entries_text(self._matched), encoding="utf-8")
            self._feedback = lf.ui.tr("logging.exported").format(count=len(self._matched))
            self._feedback_deadline = time.monotonic() + 4.0
        except OSError as error:
            self._feedback = lf.ui.tr("logging.export_failed").format(error=str(error))
            self._feedback_deadline = 0.0
        self._invalidate()

    def _on_older(self, *_):
        self._refresh_matches()
        if (self._page + 1) * int(self._page_size) < len(self._matched):
            self._paused = True
            self._page += 1
            self._invalidate()

    def _on_newer(self, *_):
        if self._page:
            self._page -= 1
            self._invalidate()

    def on_update(self, doc):
        if self._feedback and self._feedback_deadline and time.monotonic() >= self._feedback_deadline:
            self._feedback = ""
            self._feedback_deadline = 0.0
            self._dirty = True
        language = lf.ui.get_current_language()
        if language != self._last_language:
            self._last_language = language
            self._dirty = True
        capture = str(lf.log.capture_level())
        if capture != self._capture:
            self._capture = capture
            self._dirty = True
        if (not self._paused and not self._dragging) or self._snapshot is None:
            previous = self._snapshot["generation"] if self._snapshot else -1
            snapshot = lf.log.snapshot(previous)
            if snapshot is not None:
                self._snapshot = merge_snapshot(self._snapshot, snapshot)
                if not snapshot.get("incremental", False):
                    self._matches_dirty = True
                if snapshot["limit"] != self._last_buffer_limit:
                    self._matches_dirty = True
                    self._last_buffer_limit = snapshot["limit"]
                    if not self._editing_limit:
                        self._limit = str(snapshot["limit"])
                if not self._matches_dirty:
                    self._append_matches(snapshot)
                self._dirty = True
        if not self._dirty or self._snapshot is None:
            return
        self._dirty = False
        self._refresh_matches()
        page_size = int(self._page_size)
        self._page = min(self._page, max(0, (len(self._matched) - 1) // page_size))
        start = self._page * page_size
        visible = list(islice(reversed(self._matched), start, start + page_size))
        rows = doc.get_element_by_id("log-rows")
        if rows:
            self._sync_rows(rows, visible, language)
            self._sync_selection()
            rows.set_class("log-rows--nowrap", not self._wrap)
        empty = doc.get_element_by_id("log-empty")
        if empty:
            empty.set_class("hidden", bool(self._matched))
        for name, disabled in (("log-newer", self._page == 0),
                               ("log-older", start + page_size >= len(self._matched)),
                               ("log-copy", not self._matched), ("log-export", not self._matched)):
            button = doc.get_element_by_id(name)
            if button:
                if disabled:
                    button.set_attribute("disabled", "disabled")
                else:
                    button.remove_attribute("disabled")
        self._summary = lf.ui.tr("logging.summary").format(
            shown=len(visible), matched=len(self._matched), retained=len(self._snapshot["entries"]),
            limit=self._snapshot["limit"], discarded=self._snapshot["discarded"],
            page=self._page + 1, pages=max(1, (len(self._matched) + page_size - 1) // page_size),
        )
        if self._snapshot["discarded"]:
            self._summary += " · " + lf.ui.tr("logging.discarded").format(discarded=self._snapshot["discarded"])
        if self._handle:
            self._handle.dirty_all()

    def _sync_rows(self, rows, visible, language):
        """Keep unchanged DOM rows; prepend only new entries on the live page."""
        old_sequences = [sequence for sequence, node in self._row_nodes]
        sequences = [entry["sequence"] for entry in visible]
        if sequences == old_sequences and language == self._row_locale:
            return
        overlap = sequences.index(old_sequences[0]) if old_sequences and old_sequences[0] in sequences else -1
        if language == self._row_locale and overlap >= 0 and sequences[overlap:] == old_sequences[:len(sequences) - overlap]:
            reference = self._row_nodes[0][1]
            added = []
            for entry in reversed(visible[:overlap]):
                node = rows.insert_before("div", reference)
                node.set_attribute("class", f'log-row log-row--{LEVELS[entry["level"]]}')
                node.set_attribute("data-log-sequence", str(entry["sequence"]))
                node.set_inner_rml(entry_contents(entry))
                reference = node
                added.insert(0, (entry["sequence"], node))
            self._row_nodes = added + self._row_nodes
            while len(self._row_nodes) > len(visible):
                rows.remove_child(self._row_nodes.pop()[1])
        else:
            rows.set_inner_rml("".join(entry_markup(entry) for entry in visible))
            self._rendered_selection = set()
            self._row_nodes = list(zip(sequences, rows.children()))
        self._row_locale = language
