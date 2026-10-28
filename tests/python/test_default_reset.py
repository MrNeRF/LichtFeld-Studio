# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests of delegated resets, including disabled ancestors and teardown."""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


class Element:
    def __init__(self, parent=None, classes=(), attrs=None):
        self._parent = parent
        self.classes = set(classes)
        self.attrs = dict(attrs or {})
        self.controls = []
        self.animations = []
        self.listeners = {}

    def parent(self): return self._parent
    def has_attribute(self, key): return key in self.attrs
    def get_attribute(self, key, default=""): return self.attrs.get(key, default)
    def set_attribute(self, key, value): self.attrs[key] = value
    def is_class_set(self, key): return key in self.classes
    def query_selector_all(self, _selector): return self.controls
    def add_event_listener(self, name, callback): self.listeners[name] = callback
    def get_property(self, key): return "rgb(190,190,190)" if key == "color" else "rgba(0,0,0,0)"
    def animate(self, *args): self.animations.append(args); return True


@pytest.fixture
def setup_reset(monkeypatch):
    lf = ModuleType("lichtfeld")
    lf.ui = SimpleNamespace(tr=lambda key: {"common.reset_default_hint": "Restore: {value}",
        "common.yes": "Yes", "common.no": "No"}.get(key, key),
        theme=lambda: SimpleNamespace(palette=SimpleNamespace(primary=(.2,.6,.8,1))))
    monkeypatch.setitem(sys.modules, "lichtfeld", lf)
    path = Path(__file__).resolve().parents[2]/"src/python/lfs_plugins/default_reset.py"
    spec = importlib.util.spec_from_file_location("reset_under_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    doc = Element(); wrapper = Element(doc); row = Element(wrapper, ("setting-row",))
    label = Element(row, attrs={"data-reset-prop": "amount"})
    control = Element(row); row.controls = [control]
    current = {"value": 8, "default": 2, "accept": True}; writes = []
    def apply(prop, value):
        writes.append((prop, value))
        if not current["accept"]: return False
        current["value"] = value
        return True
    controller = module.LabelDefaultReset(lambda prop: current["default"], apply)
    controller.mount(doc)
    event = SimpleNamespace(target=lambda: label, get_parameter=lambda key, default: default,
                            stop_propagation=lambda: writes.append("stopped"))
    return SimpleNamespace(controller=controller, doc=doc, wrapper=wrapper, row=row, label=label,
                           control=control, current=current, writes=writes, event=event)


@pytest.mark.parametrize("value", [0, False, 2.5, "0", (.1,.2,.3)])
def test_reset_accepts_falsey_and_non_numeric_defaults(setup_reset, value):
    s = setup_reset; s.current["default"] = value
    s.doc.listeners["dblclick"](s.event)
    assert s.current["value"] == value
    assert s.writes == [("amount", value), "stopped"]
    assert len(s.label.animations) == len(s.control.animations) == 1


@pytest.mark.parametrize("where", ["wrapper", "row", "label", "control"])
@pytest.mark.parametrize("lock", ["disabled", "readonly", "aria-disabled", "disabled-overlay", "disabled-class", "hidden"])
def test_reset_never_bypasses_locks(setup_reset, where, lock):
    s = setup_reset; node = getattr(s, where)
    if lock in ("disabled-overlay", "disabled-class", "hidden"): node.classes.add("disabled" if lock == "disabled-class" else lock)
    else: node.attrs[lock] = "true" if lock == "aria-disabled" else ""
    s.doc.listeners["dblclick"](s.event)
    assert not s.writes and not s.label.animations and s.current["value"] == 8


def test_rejected_value_has_no_success_flash(setup_reset):
    s = setup_reset; s.current["accept"] = False
    s.doc.listeners["dblclick"](s.event)
    assert s.writes == [("amount", 2)] and not s.control.animations


def test_double_clicking_input_or_unannotated_label_does_not_reset(setup_reset):
    s = setup_reset; s.event.target = lambda: s.control
    s.doc.listeners["dblclick"](s.event)
    assert not s.writes
    s.label.attrs.clear(); s.event.target = lambda: s.label
    s.doc.listeners["dblclick"](s.event)
    assert not s.writes


def test_child_of_label_and_dynamic_default(setup_reset):
    s = setup_reset; child = Element(s.label); s.event.target = lambda: child
    s.current["default"] = 42
    s.doc.listeners["mouseover"](s.event)
    assert s.label.attrs["title"] == "Restore: 42"
    s.current["default"] = 17
    s.doc.listeners["dblclick"](s.event)
    assert s.current["value"] == 17


def test_unmount_and_old_document_events_are_inert(setup_reset):
    s = setup_reset; s.controller.unmount()
    s.doc.listeners["dblclick"](s.event)
    assert not s.writes
    s.controller.mount(Element())
    s.doc.listeners["dblclick"](s.event)
    assert not s.writes


def test_repeated_reset_does_not_add_listeners_or_change_layout(setup_reset):
    s = setup_reset
    for _ in range(12): s.doc.listeners["dblclick"](s.event)
    assert set(s.doc.listeners) == {"dblclick", "mouseover"}
    assert len(s.control.animations) == 12
    assert all(a[0] == "background-color" and a[2] == .24 and a[-1] is True for a in s.control.animations)


def test_right_double_click_and_missing_default_are_inert(setup_reset):
    s = setup_reset; s.event.get_parameter = lambda *_: "1"
    s.doc.listeners["dblclick"](s.event)
    assert not s.writes
    s.event.get_parameter = lambda key, default: default; s.current["default"] = None
    s.doc.listeners["dblclick"](s.event)
    assert not s.writes


def test_float32_rounding_is_not_a_rejected_reset(monkeypatch):
    lf = ModuleType("lichtfeld")
    monkeypatch.setitem(sys.modules, "lichtfeld", lf)
    path = Path(__file__).resolve().parents[2]/"src/python/lfs_plugins/default_reset.py"
    spec = importlib.util.spec_from_file_location("reset_values_under_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.values_match(.10000000149011612, .1)
    assert module.values_match([.20000000298,.3,.4], (.2,.3,.4))
    assert not module.values_match(2,3)
    assert not module.values_match(float("nan"),0)


def test_all_declared_reset_targets_are_labels_with_explicit_adapters():
    import re
    resources = Path(__file__).resolve().parents[2]/"src/visualizer/gui/rmlui/resources"
    expected = {"rendering.rml", "training.rml", "preferences.rml", "mesh2splat_panel.rml",
                "histogram_panel.rml", "splat_simplify.rml", "export_panel.rml"}
    found = set()
    for path in resources.glob("*.rml"):
        for match in re.finditer(r'<([a-z]+)\b[^>]*data-(?:attr-data-)?reset-prop="([^"]+)"[^>]*>',
                                 path.read_text(encoding="utf-8")):
            assert match[1] in ("span", "label"), (path.name, match[0])
            assert match[2] not in {"delete", "apply", "start", "convert"} and not match[2].endswith("_path"), (path.name, match[2])
            found.add(path.name)
    assert found == expected


def test_document_listener_does_not_keep_panel_alive(setup_reset):
    import weakref
    s = setup_reset
    class Panel:
        def default(self, prop): return 2
        def apply(self, prop, value): return True
    panel = Panel()
    reference = weakref.ref(panel)
    controller = type(s.controller)(panel.default, panel.apply)
    controller.mount(s.doc)
    del panel
    assert reference() is None
    s.doc.listeners["dblclick"](s.event)
    assert not s.label.animations


def test_new_bound_label_inherits_reset_without_handwritten_event(setup_reset):
    s = setup_reset
    s.label.attrs.clear()
    s.label.classes.add("prop-label")
    s.control.attrs["data-value"] = "amount"
    s.doc.listeners["dblclick"](s.event)
    assert s.current["value"] == 2
    assert not s.label.has_attribute("data-reset-prop")


def test_inferred_unknown_property_and_expressions_are_not_reset(setup_reset):
    s = setup_reset
    s.label.attrs.clear(); s.label.classes.add("prop-label")
    s.controller._default_value = lambda prop: None
    s.control.attrs["data-value"] = "output_path"
    s.doc.listeners["dblclick"](s.event)
    assert not s.writes and not s.label.has_attribute("data-reset-prop")
    s.controller._default_value = lambda prop: 2
    s.control.attrs["data-value"] = "row.text"
    s.doc.listeners["dblclick"](s.event)
    assert not s.writes


def test_remounting_the_same_document_does_not_duplicate_listeners(setup_reset):
    s = setup_reset
    added = []
    s.doc.add_event_listener = lambda *args: added.append(args)
    s.controller.mount(s.doc)
    s.controller.unmount()
    s.controller.mount(s.doc)
    assert not added
    s.doc.listeners["dblclick"](s.event)
    assert s.writes == [("amount", 2), "stopped"]


def test_inferred_binding_is_resolved_again_when_a_row_is_reused(setup_reset):
    s = setup_reset
    s.label.attrs.clear()
    s.label.classes.add("prop-label")
    s.control.attrs["data-value"] = "amount"
    s.controller._default_value = lambda prop: {"amount": 2, "other": 7}.get(prop)
    s.doc.listeners["mouseover"](s.event)
    s.control.attrs["data-value"] = "other"
    s.doc.listeners["dblclick"](s.event)
    assert s.writes == [("other", 7), "stopped"]
    assert s.current["value"] == 7


def test_inferred_binding_does_not_survive_an_expression_or_unknown_property(setup_reset):
    s = setup_reset
    s.label.attrs.clear()
    s.label.classes.add("prop-label")
    s.control.attrs["data-value"] = "amount"
    s.controller._default_value = lambda prop: 2 if prop == "amount" else None
    s.doc.listeners["mouseover"](s.event)
    for prop in ("row.text", "unknown"):
        s.control.attrs["data-value"] = prop
        s.doc.listeners["dblclick"](s.event)
    assert not s.writes and not s.label.has_attribute("data-reset-prop")


def test_hover_uses_literal_title_and_preserves_translated_row_help(setup_reset):
    s = setup_reset
    s.label.classes.add("setting-row__label-col")
    s.row.attrs["data-tooltip"] = "common.yes"
    child = Element(s.label)
    s.event.target = lambda: child
    for default in (2, 17):
        s.current["default"] = default
        s.doc.listeners["mouseover"](s.event)
        assert "data-tooltip" not in s.label.attrs
        assert s.row.attrs["data-tooltip"] == "common.yes"
        assert s.label.attrs["title"] == f"Yes\nRestore: {default}"
        # Native resolution checks translated data-tooltip keys, then the
        # literal title on that element, before climbing to the row.
        element = child
        resolved = ""
        while element is not None:
            key = element.get_attribute("data-tooltip", "").strip()
            translated = {"common.yes": "Yes"}.get(key, key)
            if key and translated != key:
                resolved = translated
                break
            if element.get_attribute("title", ""):
                resolved = element.get_attribute("title")
                break
            element = element.parent()
        assert resolved == f"Yes\nRestore: {default}"


@pytest.mark.parametrize("value,display", [
    (5_000_000, "5,000,000"), (30_000, "30,000"), (-12_345, "-12,345"),
    (0, "0"), (True, "Yes"), (False, "No"), (.125, "0.125"), ("2048", "2048"),
])
def test_reset_hint_formats_integers_like_setting_fields(setup_reset, value, display):
    s = setup_reset
    s.current["default"] = value
    s.doc.listeners["mouseover"](s.event)
    assert s.label.attrs["title"] == f"Restore: {display}"
