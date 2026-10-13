# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Log viewer filtering, paging, export scope, and display/capture separation."""
import importlib
import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest


@pytest.fixture
def logging_module(monkeypatch):
    ui = SimpleNamespace(
        PanelSpace=SimpleNamespace(FLOATING="FLOATING"),
        PanelHeightMode=SimpleNamespace(FILL="FILL"),
        PanelOption=SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED"),
        tr=lambda key: key, request_redraw=lambda: None,
        get_current_language=lambda: "en", set_clipboard_text=lambda text: None,
    )
    log = SimpleNamespace(capture_level=lambda: 2, snapshot=lambda *args: None)
    monkeypatch.setitem(sys.modules, "lichtfeld", SimpleNamespace(ui=ui, log=log))
    monkeypatch.delitem(sys.modules, "lfs_plugins.logging_panel", raising=False)
    return importlib.import_module("lfs_plugins.logging_panel")


def entry(sequence, message, level=2, file="loader.cpp", line=10):
    return dict(sequence=sequence, timestamp=1234567890.0, level=level,
                file=file, line=line, message=message)


def test_search_combines_terms_source_and_severity(logging_module):
    entries = [entry(1, "Loaded forest", 2), entry(2, "FOREST decode failed", 5),
               entry(3, "forest decode warning", 4, "render.cpp"), entry(4, "decode failed", 5)]
    match = logging_module.matching_entries
    assert [e["sequence"] for e in match(entries, "forest DECODE", "loader", -2)] == [2]
    assert [e["sequence"] for e in match(entries, level=4)] == [3]
    assert [e["sequence"] for e in match(entries, query="loader.cpp:10")] == [1, 2, 4]
    assert match(entries, query="[invalid regex") == []


def test_markup_escapes_messages_sources_and_translated_labels(logging_module):
    logging_module.lf.ui.tr = lambda key: "<Warning>"
    markup = logging_module.entry_markup(entry(1, '<script>&"', 4, 'a"<b>.cpp'))
    assert "<script>" not in markup
    assert "&lt;script&gt;&amp;" in markup
    assert 'title="a&quot;&lt;b&gt;.cpp:10"' in markup
    assert "&lt;Warning&gt;" in markup


class Element:
    def __init__(self):
        self.markup = ""
        self.attrs = {}
        self.nodes = []
        self.listeners = {}
    def set_inner_rml(self, value):
        self.markup = value
        self.nodes = []
        for child in ET.fromstring("<root>" + value + "</root>"):
            node = Element()
            node.markup = ET.tostring(child, encoding="unicode")
            self.nodes.append(node)
    def children(self): return self.nodes
    def insert_before(self, tag, reference):
        node = Element()
        self.nodes.insert(self.nodes.index(reference), node)
        return node
    def remove_child(self, node): self.nodes.remove(node)
    def get_attribute(self, name, default=""): return self.attrs.get(name, default)
    def add_event_listener(self, name, callback): self.listeners[name] = callback
    def select(self): return True
    def set_class(self, name, value): pass
    def set_attribute(self, name, value):
        assert isinstance(value, str)
        self.attrs[name] = value
    def remove_attribute(self, name): self.attrs.pop(name, None)


class Document:
    def __init__(self): self.elements = {}
    def get_element_by_id(self, name): return self.elements.setdefault(name, Element())


def make_panel(module, count=300):
    snapshot = dict(generation=1, limit=5000, discarded=0,
                    entries=[entry(i, f"message-{i}") for i in range(count)])
    module.lf.log.snapshot = lambda previous=-1: snapshot if previous != snapshot["generation"] else None
    panel = module.LoggingPanel()
    doc = Document()
    panel.on_mount(doc)
    return panel, doc, snapshot


def test_page_bound_and_pause_keeps_snapshot_stable(logging_module):
    panel, doc, snapshot = make_panel(logging_module)
    assert doc.elements["log-rows"].markup.count('class="log-row ') == 250
    panel._on_older()
    assert panel._paused and panel._page == 1
    assert doc.elements["log-rows"].markup.count('class="log-row ') == 50
    next_snapshot = dict(snapshot, generation=2, entries=[entry(1000, "new log")])
    logging_module.lf.log.snapshot = lambda *_: next_snapshot
    panel.on_update(doc)
    assert panel._snapshot is snapshot
    panel._on_pause()
    assert panel._snapshot is next_snapshot
    assert "new log" in doc.elements["log-rows"].markup


def test_display_filter_does_not_change_capture_level(logging_module):
    panel, doc, _ = make_panel(logging_module)
    changes = []
    logging_module.lf.log.set_capture_level = changes.append
    panel._set_filter("level", "5")
    assert changes == []
    assert doc.elements["log-rows"].markup == ""
    panel._set_capture("1")
    assert changes == [1]


def test_copy_and_export_cover_all_matches_not_just_page(logging_module, tmp_path):
    panel, doc, _ = make_panel(logging_module)
    copied = []
    logging_module.lf.ui.set_clipboard_text = copied.append
    logging_module.lf.ui.tr = lambda key: "{count}" if key in ("logging.copied", "logging.exported") else key
    panel._on_copy()
    assert len(copied[0].splitlines()) == 300
    path = tmp_path / "logs.txt"
    logging_module.lf.ui.save_text_file_dialog = lambda name: str(path)
    panel._on_export()
    assert path.read_text() == copied[0]
    panel._set_filter("query", "message-299")
    panel._on_copy()
    assert len(copied[-1].splitlines()) == 1


def test_idle_poll_does_not_rebuild_rows(logging_module):
    panel, doc, _ = make_panel(logging_module)
    element = doc.elements["log-rows"]
    element.set_inner_rml = lambda value: pytest.fail("idle update rebuilt rows")
    panel.on_update(doc)


def test_incremental_snapshot_prunes_evicted_rows_and_handles_clear(logging_module):
    previous = dict(generation=3, entries=[entry(i, str(i)) for i in (1, 2, 3)])
    update = dict(generation=4, incremental=True, first_sequence=2, entry_count=3,
                  entries=[entry(4, "4")])
    merged = logging_module.merge_snapshot(previous, update)
    assert [row["sequence"] for row in merged["entries"]] == [2, 3, 4]
    trimmed = logging_module.merge_snapshot(merged, dict(update, generation=5, first_sequence=3,
                                                         entry_count=2, entries=[]))
    assert [row["sequence"] for row in trimmed["entries"]] == [3, 4]
    cleared = logging_module.merge_snapshot(trimmed, dict(update, generation=6, first_sequence=0,
                                                         entry_count=0, entries=[]))
    assert cleared["entries"] == []
    assert previous["entries"][0]["sequence"] == 1


def test_retention_edit_commits_on_enter_or_blur_and_preserves_partial_input(logging_module):
    panel, doc, snapshot = make_panel(logging_module)
    applied = []
    logging_module.lf.log.set_buffer_limit = applied.append
    panel._set_limit("50")
    panel.on_update(doc)
    assert panel._limit == "50"
    assert applied == []
    field = doc.elements["log-limit"]
    panel._set_limit("00")
    panel.on_update(doc)
    assert panel._limit == "00"
    assert applied == []
    field.listeners["change"](SimpleNamespace(get_bool_parameter=lambda *args: False))
    assert applied == []
    field.listeners["change"](SimpleNamespace(get_bool_parameter=lambda *args: True))
    assert applied == [0]
    assert panel._limit == "0"
    panel._set_limit("5,000")
    field.listeners["blur"](None)
    assert applied == [0, 5000]
    assert panel._limit == "5000"
    assert panel._page_size == "250"


def test_live_tail_reuses_unchanged_row_nodes(logging_module):
    panel, doc, snapshot = make_panel(logging_module)
    original_nodes = list(panel._row_nodes)
    updated = [entry(300, "new tail"), *reversed(snapshot["entries"][-249:])]
    panel._sync_rows(doc.elements["log-rows"], updated, "en")
    assert len(panel._row_nodes) == 250
    assert panel._row_nodes[0][0] == 300
    assert panel._row_nodes[1][1] is original_nodes[0][1]
    assert original_nodes[-1][1] not in doc.elements["log-rows"].nodes
