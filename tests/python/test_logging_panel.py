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


def test_search_matches_displayed_runtime_source(logging_module):
    entries = [entry(1, "source-less message", file="", line=0)]
    assert logging_module.matching_entries(entries, source="runtime:0") == entries
    assert logging_module.matching_entries(entries, query="runtime") == entries


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
            node.attrs = dict(child.attrib)
            self.nodes.append(node)
    def children(self): return self.nodes
    def insert_before(self, tag, reference):
        node = Element()
        self.nodes.insert(self.nodes.index(reference), node)
        return node
    def remove_child(self, node): self.nodes.remove(node)
    def get_attribute(self, name, default=""): return self.attrs.get(name, default)
    def has_attribute(self, name): return name in self.attrs
    def parent(self): return self
    def focus(self): return True
    def add_event_listener(self, name, callback):
        previous = self.listeners.get(name)
        self.listeners[name] = callback if previous is None else lambda event: (previous(event), callback(event))
    def select(self): return True
    def set_class(self, name, value): pass
    def set_attribute(self, name, value):
        assert isinstance(value, str)
        self.attrs[name] = value
    def remove_attribute(self, name): self.attrs.pop(name, None)


class Document:
    def __init__(self):
        self.elements = {}
        self.listeners = {}
    def add_event_listener(self, name, callback): self.listeners[name] = callback
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


def test_escape_cancels_a_retention_edit(logging_module):
    panel, doc, _snapshot = make_panel(logging_module)
    applied = []
    logging_module.lf.log.set_buffer_limit = applied.append
    field = doc.elements["log-limit"]
    field.listeners["focus"](None)
    panel._set_limit("3")
    field.listeners["escapecancel"](SimpleNamespace(stop_propagation=lambda: None))
    field.listeners["blur"](None)
    assert panel._limit == "5000"
    assert 3 not in applied


def test_blank_or_out_of_range_retention_is_rejected(logging_module):
    panel, doc, _snapshot = make_panel(logging_module)
    applied = []
    logging_module.lf.log.set_buffer_limit = applied.append
    field = doc.elements["log-limit"]
    for text in ("", "-4", "50001"):
        panel._feedback = ""
        panel._set_limit(text)
        field.listeners["blur"](None)
        assert panel._feedback, text
    assert applied == []


def test_long_messages_are_cut_in_rows_but_copied_whole(logging_module):
    message = "A" * 70000
    long_entry = entry(1, message)
    markup = logging_module.entry_contents(long_entry)
    assert "A" * logging_module.ROW_MESSAGE_CHARS + "\u2026" in markup
    assert len(markup) < logging_module.ROW_MESSAGE_CHARS + 1000
    assert message in logging_module.entries_text([long_entry])


def test_grouped_retention_values_commit_in_any_locale(logging_module):
    panel, doc, _snapshot = make_panel(logging_module)
    applied = []
    logging_module.lf.log.set_buffer_limit = applied.append
    field = doc.elements["log-limit"]
    for text in ("50.000", "1 000", "2,500", "1.0", "\u00b2"):
        panel._set_limit(text)
        field.listeners["blur"](None)
    assert applied == [50000, 1000, 2500]


def test_live_tail_reuses_unchanged_row_nodes(logging_module):
    panel, doc, snapshot = make_panel(logging_module)
    original_nodes = list(panel._row_nodes)
    updated = [entry(300, "new tail"), *reversed(snapshot["entries"][-249:])]
    panel._sync_rows(doc.elements["log-rows"], updated, "en")
    assert len(panel._row_nodes) == 250
    assert panel._row_nodes[0][0] == 300
    assert panel._row_nodes[1][1] is original_nodes[0][1]
    assert original_nodes[-1][1] not in doc.elements["log-rows"].nodes


@pytest.mark.parametrize('name,value', [('query', 'message-2'), ('source', 'loader'), ('level', '2')])
def test_filtered_live_updates_match_only_new_entries(logging_module, monkeypatch, name, value):
    panel, doc, snapshot = make_panel(logging_module)
    panel._set_filter(name, value)
    original = logging_module.matching_entries
    calls = []
    def tracked(entries, *args):
        calls.append([row['sequence'] for row in entries])
        return original(entries, *args)
    monkeypatch.setattr(logging_module, 'matching_entries', tracked)
    tail = [entry(300, 'message-299 tail'), entry(301, 'unrelated', 5, 'other.cpp')]
    update = dict(generation=2, limit=5000, discarded=2, incremental=True,
                  first_sequence=2, entry_count=300, entries=tail)
    logging_module.lf.log.snapshot = lambda *_: update
    panel.on_update(doc)
    assert calls == [[300, 301]]
    retained = snapshot['entries'][2:] + tail
    assert panel._matched == original(retained, panel._query, panel._source, int(panel._level))
    calls.clear()
    logging_module.lf.log.snapshot = lambda *_: None
    panel._set_wrap(False)
    panel._set_filter('page_size', '100')
    panel._on_copy()
    assert calls == []
    panel._set_filter('query', 'unrelated' if name != 'query' else 'tail')
    assert calls == [[row['sequence'] for row in retained]]


def test_match_cache_rebuilds_on_limit_change_and_clear(logging_module, monkeypatch):
    panel, doc, snapshot = make_panel(logging_module)
    panel._set_filter('query', 'message')
    original = logging_module.matching_entries
    calls = []
    def tracked(entries, *args):
        calls.append([row['sequence'] for row in entries])
        return original(entries, *args)
    monkeypatch.setattr(logging_module, 'matching_entries', tracked)
    update = dict(generation=2, limit=100, discarded=200, incremental=True,
                  first_sequence=200, entry_count=100, entries=[])
    logging_module.lf.log.snapshot = lambda *_: update
    panel.on_update(doc)
    assert calls == [list(range(200, 300))]
    calls.clear()
    empty = dict(generation=3, limit=100, discarded=0, entries=[])
    logging_module.lf.log.clear_buffer = lambda: None
    logging_module.lf.log.snapshot = lambda *_: empty
    panel._on_clear()
    assert calls == [[]]
    assert panel._matched == []


class SelectionEvent:
    def __init__(self, node=None, **parameters):
        self.node = node
        self.parameters = parameters
        self.stopped = False
    def target(self): return self.node
    def get_parameter(self, name, default=""): return str(self.parameters.get(name, default))
    def get_bool_parameter(self, name, default=False): return bool(self.parameters.get(name, default))
    def stop_propagation(self): self.stopped = True


def select_row(panel, index, **modifiers):
    node = panel._row_nodes[index][1]
    panel._on_selection_down(SelectionEvent(node, button=0, **modifiers))
    panel._on_selection_up(SelectionEvent(button=0))


def test_click_ctrl_shift_and_drag_select_rows_and_copy_in_display_order(logging_module):
    panel, doc, _ = make_panel(logging_module, count=6)
    copied = []
    logging_module.lf.ui.set_clipboard_text = copied.append
    select_row(panel, 1)
    assert panel._selected_sequences == {4}
    select_row(panel, 3, ctrl_key=True)
    assert panel._selected_sequences == {2, 4}
    select_row(panel, 1, ctrl_key=True)
    assert panel._selected_sequences == {2}
    select_row(panel, 0)
    select_row(panel, 2, shift_key=True)
    assert panel._selected_sequences == {3, 4, 5}
    event = SelectionEvent(key_identifier=logging_module.KI_C, ctrl_key=True)
    panel._on_selection_key(event)
    assert event.stopped
    assert copied[-1] == logging_module.entries_text([panel._matched[i] for i in (5, 4, 3)])
    # Drag selects the inclusive range and freezes new arrivals until release.
    panel._on_selection_down(SelectionEvent(panel._row_nodes[4][1], button=0))
    panel._on_selection_move(SelectionEvent(panel._row_nodes[1][1]))
    assert panel._selected_sequences == {1, 2, 3, 4}
    logging_module.lf.log.snapshot = lambda *_: pytest.fail('polled logs during selection drag')
    panel.on_update(doc)
    logging_module.lf.log.snapshot = lambda *_: None
    doc.listeners['mouseup'](SelectionEvent(button=0))
    assert not panel._dragging
    panel._on_selection_key(SelectionEvent(key_identifier=logging_module.KI_ESCAPE))
    assert panel._selected_sequences == set()


def test_selection_survives_new_rows_and_prunes_evictions_and_filter_changes(logging_module):
    panel, doc, _ = make_panel(logging_module, count=6)
    select_row(panel, 0)
    select_row(panel, 5, ctrl_key=True)
    update = dict(generation=2, limit=5000, discarded=1, incremental=True,
                  first_sequence=1, entry_count=6, entries=[entry(6, 'new')])
    logging_module.lf.log.snapshot = lambda *_: update
    panel.on_update(doc)
    assert panel._selected_sequences == {5}
    assert panel._row_nodes[0][0] == 6
    logging_module.lf.log.snapshot = lambda *_: None
    panel._set_filter('query', 'new')
    assert panel._selected_sequences == set()
    select_row(panel, 0)
    panel._on_selection_key(SelectionEvent(key_identifier=logging_module.KI_A, ctrl_key=True))
    assert panel._selected_sequences == {6}
    update = dict(update, generation=3, first_sequence=0, entry_count=0, entries=[])
    logging_module.lf.log.snapshot = lambda *_: update
    panel.on_update(doc)
    assert panel._selected_sequences == set()


def test_row_copy_shortcuts_are_scoped_to_log_scroll_and_ignore_other_keys(logging_module):
    panel, doc, _ = make_panel(logging_module, count=2)
    assert 'keydown' in doc.elements['log-scroll'].listeners
    assert 'keydown' not in doc.listeners
    copied = []
    logging_module.lf.ui.set_clipboard_text = copied.append
    select_row(panel, 0)
    event = SelectionEvent(key_identifier=logging_module.KI_C)
    panel._on_selection_key(event)
    assert not copied and not event.stopped
    right = SelectionEvent(panel._row_nodes[1][1], button=1)
    panel._on_selection_down(right)
    assert panel._selected_sequences == {1}


def test_unchanged_selection_does_not_restyle_rows(logging_module):
    panel, doc, _ = make_panel(logging_module, count=4)
    select_row(panel, 1)
    for sequence, node in panel._row_nodes:
        node.set_class = lambda *args: pytest.fail('restyled an unchanged row')
    panel._sync_selection()
    # An empty selection also has no DOM work on repeated updates.
    panel._selected_sequences.clear()
    panel._rendered_selection.clear()
    panel._sync_selection()
