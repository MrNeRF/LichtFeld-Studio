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
        def build(node, parent):
            element = Element(node.attrib.get("id", ""), dict(node.attrib), parent)
            element.nodes = [build(child, element) for child in node]
            return element
        self.nodes = [build(node, self) for node in ET.fromstring("<root>" + value + "</root>")]

    def query_selector_all(self, selector):
        result = []
        for node in self.nodes:
            if selector[1:] in node.attrs.get("class", "").split():
                result.append(node)
            result.extend(node.query_selector_all(selector))
        return result


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
    def get_element_by_id(self, name):
        def find(element):
            if element.id == name: return element
            for child in element.nodes:
                match = find(child)
                if match: return match
            return None
        for element in self.elements.values():
            match = find(element)
            if match: return match
        return self.elements.setdefault(name, Element(name))
    def remove_data_model(self, name): pass


@pytest.fixture
def history(monkeypatch):
    state = SimpleNamespace(generation=1, undo=[f"edit {i}" for i in range(205)],
                            redo=["undone edit"], active=False, calls=[], reads=0,
                            result={"success": True, "error": ""}, enabled=[], detail_reads=[])
    def stack():
        state.reads += 1
        return {direction: [{"id": name, "label": name, "source": "python", "scope": "custom"}
                            for name in getattr(state, direction)] for direction in ("undo", "redo")}
    def names(stack):
        state.reads += 1
        return list(getattr(state, stack))
    def details(stack, steps, generation):
        state.detail_reads.append((stack, steps, generation))
        return {"label": getattr(state, stack)[steps - 1], "transforms": []}
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
    undo = SimpleNamespace(generation=lambda: state.generation, stack=stack,
                           undo_names=lambda: names("undo"), redo_names=lambda: names("redo"),
                           has_active_transaction=lambda: state.active, jump=jump, clear=clear, details=details)
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
    assert len(markup) < 1800
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
    assert doc.elements["history-rows"].query_selector_all(".history-jump")[1].focused and not state.calls
    panel._on_key(event(rows[1], str(mod.KI_END)))
    assert doc.elements["history-rows"].query_selector_all(".history-jump")[-1].focused


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


def test_disclosure_reads_details_without_playback_and_closes(history):
    _, panel, doc, state = history
    disclosure = doc.elements["history-rows"].nodes[1].query_selector_all(".history-disclosure")[0]
    panel._on_row(event(disclosure))
    assert state.calls == []
    assert state.detail_reads == [("undo", 2, 1)]
    detail = doc.get_element_by_id("history-detail-2")
    assert detail.props["display"] == "block"
    panel._on_row(event(detail.nodes[0]))
    assert not state.calls
    panel._on_row(event(disclosure))
    assert detail.props["display"] == "none"
    assert len(state.detail_reads) == 1


def test_expanded_details_are_discarded_when_history_changes(history):
    _, panel, doc, state = history
    panel._toggle_details(1, state.generation)
    assert panel._expanded
    state.generation += 1
    panel.on_update(doc)
    assert not panel._expanded
    reads = list(state.detail_reads)
    panel._toggle_details(1, state.generation - 1)
    assert state.detail_reads == reads


def test_transform_details_show_affected_node_and_before_after_values(history):
    mod, _, _, _ = history
    identity = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
    moved = identity[:]
    moved[12] = 3
    mod.lf.decompose_transform = lambda matrix: {
        "translation": matrix[12:15], "rotation_euler_deg": [0,0,0], "scale": [1,1,1]}
    markup = mod.details_markup({"label": "Apply batch transform", "transforms": [
        {"target": "<model>", "before": identity, "after": moved}]})
    assert '&lt;model&gt;' in markup
    assert '0, 0, 0' in markup and '3, 0, 0' in markup
    assert 'history.before' in markup and 'history.after' in markup
    assert 'transform.position' in markup
    assert 'transform.scale' not in markup


def test_shear_details_keep_exact_matrix_values(history):
    mod, _, _, _ = history
    identity = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
    sheared = identity[:]
    sheared[4] = .25
    markup = mod.details_markup({"label": "Shear", "transforms": [
        {"target": "model", "before": identity, "after": sheared}]})
    assert 'history.matrix' in markup and '0.25' in markup


def test_operation_icons_reuse_assets_and_distinguish_edit_families(history):
    mod, _, _, _ = history
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "src/visualizer/gui/assets/icon"
    for key, icon in mod._OPERATION_PRESENTATION.values():
        assert (root / icon).is_file(), icon
    for icon in mod._SCOPE_ICONS.values():
        assert (root / icon).is_file(), icon
    assert mod.operation_icon("select.stroke") == "selection.png"
    assert mod.operation_icon("edit.delete") == "scene/trash.png"
    assert mod.operation_icon("transform.rotate") == "rotation.png"
    assert mod.operation_icon("Rename Node") == "scene/group.png"
    assert mod.operation_icon("Custom action", {"source": "python"}) == "puzzle.png"
    assert mod.operation_label("select.stroke") == "history.operation.selection_stroke"


def test_non_transform_details_show_recorded_fields_and_localized_values(history):
    mod, _, _, _ = history
    markup = mod.details_markup({"id": "select.stroke", "label": "select.stroke", "source": "ui",
                                "scope": "selection", "fields": [
        {"target": "<Group 1>", "property": "history.field.selected_count", "before": "2", "after": "8"},
        {"property": "history.field.selection_mode", "after": "@tr:history.value.replace"}], "transforms": []})
    assert "history.operation.selection_stroke" in markup
    assert "&lt;Group 1&gt;" in markup
    assert "history.field.selected_count" in markup and ">2<" in markup and ">8<" in markup
    assert "history.value.replace" in markup and "@tr:" not in markup
    assert "history.no_details" not in markup


def test_python_metadata_is_available_without_introspecting_callbacks(history):
    mod, _, _, _ = history
    markup = mod.details_markup({"id": "plugin.custom", "label": "User label", "source": "python", "scope": "custom"})
    assert "plugin.custom" in markup and "User label" in markup
    assert "history.source_values.python" in markup and "history.scope_values.custom" in markup
    assert "history.no_details" not in markup


def test_more_details_reveals_recorded_data_without_playback_or_refetch(history):
    _, panel, doc, state = history
    panel._expanded[1] = {"label": "Large edit", "fields": [
        {"property": "history.field.property", "after": f"field {i}"} for i in range(125)]}
    panel._show_details(1, panel._expanded[1])
    detail = doc.get_element_by_id("history-detail-1")
    assert "field 124" not in detail.markup
    more = detail.query_selector_all(".history-more-details")[0]
    panel._on_row(event(more))
    assert "field 124" in detail.markup
    assert not state.calls and not state.detail_reads


def test_large_unbroken_field_values_are_bounded_and_escaped(history):
    mod, _, _, _ = history
    markup = mod.details_markup({"label": "test", "fields": [
        {"property": "<caption>", "after": "<script>" + "X" * 70000}]})
    assert "&lt;caption&gt;" in markup and "&lt;script&gt;" in markup
    assert len(markup) < 2000


def test_presentation_keeps_unknown_user_labels_and_localizes_composed_operations(history):
    mod, _, _, _ = history
    assert mod.operation_label("My custom task") == "My custom task"
    assert mod.operation_label("Translate | Delete") == "history.operation.translate | history.operation.delete"
