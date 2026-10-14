# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""History navigation safety, bounded rendering, and lifecycle regressions."""
import importlib
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


class Element:
    def __init__(self, id="", attrs=None, parent=None):
        self.id, self.attrs, self._parent = id, attrs or {}, parent
        self.markup, self.nodes, self.listeners = "", [], {}
        self.classes, self.props = {}, {}
        self.writes, self.focused = 0, False
        self.scroll_top = 0

    def set_inner_rml(self, value):
        self.markup = value
        self.writes += 1
        self.nodes = [Element(attrs=dict(node.attrib), parent=self)
                      for node in ET.fromstring("<root>" + value + "</root>")]

    def set_attribute(self, key, value): self.attrs[key] = value
    def get_attribute(self, key, default=""): return self.attrs.get(key, default)
    def remove_attribute(self, key): self.attrs.pop(key, None)
    def set_class(self, key, value): self.classes[key] = value
    def set_property(self, key, value): self.props[key] = value
    def add_event_listener(self, key, callback): self.listeners[key] = callback
    def parent(self): return self._parent
    def children(self): return self.nodes
    def focus(self): self.focused = True
    def scroll_into_view(self, align): pass


class Document:
    def __init__(self): self.elements = {}
    def get_element_by_id(self, name): return self.elements.setdefault(name, Element(name))
    def remove_data_model(self, name): pass


@pytest.fixture
def history(monkeypatch):
    state = SimpleNamespace(generation=1, undo=[f"edit {i}" for i in range(205)],
                            redo=["undone edit"], active=False, calls=[], reads=0,
                            result={"success": True, "error": ""}, enabled=[])
    def names(stack):
        state.reads += 1
        return list(getattr(state, stack))
    def jump(stack, count):
        state.calls.append((stack, count))
        state.generation += 1
        return state.result
    def clear():
        state.calls.append("clear")
        state.undo, state.redo = [], []
        state.generation += 1
    ui = SimpleNamespace(
        PanelSpace=SimpleNamespace(FLOATING="FLOATING"),
        PanelHeightMode=SimpleNamespace(FILL="FILL"),
        PanelOption=SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED"),
        tr=lambda key: key, get_current_language=lambda: "en", request_redraw=lambda: None,
        set_panel_enabled=lambda *args: state.enabled.append(args),
    )
    undo = SimpleNamespace(generation=lambda: state.generation,
                           undo_names=lambda: names("undo"), redo_names=lambda: names("redo"),
                           has_active_transaction=lambda: state.active, jump=jump, clear=clear)
    monkeypatch.setitem(sys.modules, "lichtfeld", SimpleNamespace(ui=ui, undo=undo))
    # Match the existing panel-test isolation convention.
    monkeypatch.delitem(sys.modules, "lfs_plugins.history_panel", raising=False)
    mod = importlib.import_module("lfs_plugins.history_panel")
    panel, doc = mod.HistoryPanel(), Document()
    panel.on_mount(doc)
    return mod, panel, doc, state


def event(target, key="0"):
    return SimpleNamespace(target=lambda: target, get_parameter=lambda *_: key,
                           stop_propagation=lambda: None)


def test_paging_keeps_absolute_jump_counts_and_bounds_dom(history):
    mod, panel, doc, state = history
    rows = doc.elements["history-rows"]
    assert len(rows.nodes) == mod.PAGE_SIZE
    panel._action("older")
    assert rows.nodes[0].attrs["data-steps"] == "26"
    # Clicking a child must route to the containing button.
    panel._on_row(event(Element(parent=rows.nodes[4])))
    assert state.calls == [("undo", 30)]
    while "disabled" not in doc.elements["history-older"].attrs:
        panel._action("older")
    assert len(rows.nodes) == 5
    assert "disabled" in doc.elements["history-older"].attrs


def test_stale_click_refreshes_without_playback(history):
    _, panel, doc, state = history
    row = doc.elements["history-rows"].nodes[2]
    state.undo.insert(0, "new edit")
    state.generation += 1
    panel._on_row(event(row))
    assert state.calls == []
    assert panel._generation == state.generation


def test_idle_refresh_does_not_read_stacks_or_rebuild_dom(history):
    _, panel, doc, state = history
    writes, reads = doc.elements["history-rows"].writes, state.reads
    for _ in range(10): panel.on_update(doc)
    assert doc.elements["history-rows"].writes == writes
    assert state.reads == reads


def test_active_transaction_blocks_clear_and_playback(history):
    _, panel, doc, state = history
    state.active = True
    panel.on_update(doc)
    panel._action("undo")
    panel._action("clear")
    panel._on_row(event(doc.elements["history-rows"].nodes[0]))
    assert not state.calls
    assert "disabled" in doc.elements["history-clear"].attrs


def test_redo_navigation_and_failure_feedback(history):
    _, panel, doc, state = history
    panel._action("redo-tab")
    state.result = {"success": False, "error": "<playback failed>"}
    panel._on_row(event(doc.elements["history-rows"].nodes[0]))
    assert state.calls == [("redo", 1)]
    assert doc.elements["history-feedback"].markup == "&lt;playback failed&gt;"


def test_long_labels_are_bounded_escaped_and_single_line(history):
    mod, _, _, _ = history
    markup = mod.row_markup('<script>\n' + 'x' * 70000, 1, 1)
    assert '<script>' not in markup and '&lt;script&gt;' in markup
    assert len(markup) < 600
    assert '<br' not in markup


def test_close_button_and_reopen_refresh(history):
    _, panel, doc, state = history
    doc.elements["close-btn"].listeners["click"](event(None))
    assert state.enabled == [("lfs.history", False)]
    panel.on_unmount(doc)
    assert panel._doc is None
    state.undo = []
    state.generation += 1
    reopened = Document()
    panel.on_mount(reopened)
    assert reopened.elements["history-empty"].props["display"] == "block"


def test_arrow_navigation_moves_focus_without_playback(history):
    mod, panel, doc, state = history
    rows = doc.elements["history-rows"].nodes
    panel._on_key(event(rows[0], str(mod.KI_DOWN)))
    assert rows[1].focused and not state.calls
    panel._on_key(event(rows[1], str(mod.KI_END)))
    assert rows[-1].focused


def test_keyboard_activates_the_focused_row(history):
    mod, panel, doc, state = history
    panel._on_key(event(doc.elements["history-rows"].nodes[2], str(mod.KI_RETURN)))
    assert state.calls == [("undo", 3)]


def test_floating_title_is_bound_and_buttons_have_layout_children(history):
    mod, panel, _, _ = history
    bindings = {}
    model = SimpleNamespace(bind_func=lambda key, getter: bindings.update({key: getter}),
                            get_handle=lambda: object())
    panel.on_bind_model(SimpleNamespace(create_data_model=lambda name: model))
    assert bindings["panel_label"]() == "scene.history"
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    template = ET.parse(root / "src/visualizer/gui/rmlui/resources/history_panel.rml")
    assert template.find("body").get("data-model") == "history_panel"
    for button in template.iter("button"):
        if button.get("id") not in ("history-undo-tab", "history-redo-tab"):
            assert button.find("span") is not None


def test_new_page_appears_at_26_and_past_100_entries(history):
    _, panel, doc, state = history
    for count, pages in ((25, 1), (26, 2), (100, 4), (101, 5)):
        state.undo = [f"edit {i}" for i in range(count)]
        state.generation += 1
        panel.on_update(doc)
        assert len(doc.elements["history-rows"].nodes) == 25
        assert doc.elements["history-page"].markup == f"1 / {pages}"
        assert ("disabled" in doc.elements["history-older"].attrs) == (pages == 1)
    for _ in range(4): panel._action("older")
    assert len(doc.elements["history-rows"].nodes) == 1
    assert doc.elements["history-rows"].nodes[0].attrs["data-steps"] == "101"
    panel._action("newer")
    assert doc.elements["history-page"].markup == "4 / 5"
