# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, searchable session logs in an independent utility window."""

from datetime import datetime
from html import escape
from pathlib import Path

import lichtfeld as lf

from .panels import panel_class
from .types import Panel

__lfs_panel_classes__ = ["LoggingPanel"]
__lfs_panel_ids__ = ["lfs.logging"]

LEVELS = ("trace", "debug", "info", "performance", "warn", "error", "critical", "off")
PAGE_SIZES = (100, 250, 500, 1000)
BUFFER_LIMITS = (1000, 5000, 10000, 50000)


def matching_entries(entries, query="", source="", level=-1):
    """Match all space-separated search terms, a source substring, and severity."""
    terms = query.casefold().split()
    source = source.casefold().strip()
    result = []
    for entry in entries:
        severity = int(entry["level"])
        if level == -2 and severity < 4:
            continue
        if level >= 0 and severity != level:
            continue
        location = f"{entry['file']}:{entry['line']}".casefold()
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


def entry_markup(entry):
    level = LEVELS[entry["level"]]
    label_key = f"scene.log_{level}"
    location = f"{entry['file'] or 'runtime'}:{entry['line']}"
    return (
        f'<div class="log-row log-row--{level}">'
        f'<span class="log-time text-muted">{entry_time(entry)}</span>'
        f'<span class="log-level">{escape(lf.ui.tr(label_key))}</span>'
        f'<span class="log-source text-muted" title="{escape(location, quote=True)}">'
        f'{escape(location)}</span>'
        f'<span class="log-message">{escape(entry["message"])}</span></div>'
    )


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
        self._page_size = "250"
        self._paused = False
        self._wrap = True
        self._page = 0
        self._summary = ""
        self._feedback = ""
        self._dirty = True
        self._matched = []
        self._last_language = ""

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
        self._doc = doc
        self._dirty = True
        self.on_update(doc)

    def on_unmount(self, doc):
        if doc:
            doc.remove_data_model("logging_panel")
        self._doc = None
        self._handle = None

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
        limit = int(value)
        if limit not in BUFFER_LIMITS:
            return
        lf.log.set_buffer_limit(limit)
        self._limit = str(limit)
        # Apply destructive retention changes even while the display is paused.
        self._snapshot = lf.log.snapshot()
        self._page = 0
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
        self._page = 0
        self._feedback = lf.ui.tr("logging.cleared")
        self._invalidate()

    def _on_reset_filters(self, *_):
        self._query = self._source = ""
        self._level = "-1"
        self._page = 0
        self._invalidate()

    def _refresh_matches(self):
        entries = self._snapshot["entries"] if self._snapshot else []
        self._matched = matching_entries(entries, self._query, self._source, int(self._level))

    def _on_copy(self, *_):
        self._refresh_matches()
        if not self._matched:
            return
        lf.ui.set_clipboard_text(entries_text(self._matched))
        self._feedback = lf.ui.tr("logging.copied").format(count=len(self._matched))
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
        except OSError as error:
            self._feedback = lf.ui.tr("logging.export_failed").format(error=str(error))
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
        language = lf.ui.get_current_language()
        if language != self._last_language:
            self._last_language = language
            self._dirty = True
        capture = str(lf.log.capture_level())
        if capture != self._capture:
            self._capture = capture
            self._dirty = True
        if not self._paused or self._snapshot is None:
            previous = self._snapshot["generation"] if self._snapshot else -1
            snapshot = lf.log.snapshot(previous)
            if snapshot is not None:
                self._snapshot = snapshot
                self._limit = str(snapshot["limit"])
                self._dirty = True
        if not self._dirty or self._snapshot is None:
            return
        self._dirty = False
        self._refresh_matches()
        page_size = int(self._page_size)
        self._page = min(self._page, max(0, (len(self._matched) - 1) // page_size))
        latest = list(reversed(self._matched))
        start = self._page * page_size
        visible = latest[start:start + page_size]
        rows = doc.get_element_by_id("log-rows")
        if rows:
            rows.set_inner_rml("".join(entry_markup(entry) for entry in visible))
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
        if self._handle:
            self._handle.dirty_all()
