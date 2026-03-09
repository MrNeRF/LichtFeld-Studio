# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the plugin API surface during the Rml transition."""

from importlib import import_module
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _source_python_path(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "src" / "python"))


def test_plugin_package_exports_rml_first_types(monkeypatch):
    monkeypatch.delitem(sys.modules, "lfs_plugins", raising=False)

    module = import_module("lfs_plugins")

    assert module.Panel.__name__ == "Panel"
    assert module.RmlPanel.__name__ == "RmlPanel"
    assert module.Menu.__name__ == "Menu"


def test_menu_base_exposes_schema_fallback(monkeypatch):
    monkeypatch.delitem(sys.modules, "lfs_plugins.types", raising=False)

    types_mod = import_module("lfs_plugins.types")

    assert types_mod.Menu().menu_items() == []


def test_rml_widgets_collapsible_uses_text_for_arrow(monkeypatch):
    monkeypatch.delitem(sys.modules, "lfs_plugins.rml_widgets", raising=False)

    widgets = import_module("lfs_plugins.rml_widgets")

    class _ElementStub:
        def __init__(self, tag="div"):
            self.tag = tag
            self.attrs = {}
            self.classes = ""
            self.text = None
            self.inner_rml = None
            self.children_list = []

        def append_child(self, tag):
            child = _ElementStub(tag)
            self.children_list.append(child)
            return child

        def set_id(self, value):
            self.attrs["id"] = value

        def set_class_names(self, value):
            self.classes = value

        def set_attribute(self, name, value):
            self.attrs[name] = value

        def set_text(self, value):
            self.text = value

        def set_inner_rml(self, value):
            self.inner_rml = value

        def set_class(self, _name, _enabled):
            return None

        def remove_property(self, _name):
            return None

        def set_property(self, _name, _value):
            return None

        @property
        def client_height(self):
            return 0

        @property
        def scroll_height(self):
            return 0

    container = _ElementStub()
    header, _content = widgets.collapsible(container, "advanced", title="Advanced", open=True)

    arrow = header.children_list[0]
    assert arrow.text == chr(0x25B6)
    assert arrow.inner_rml is None
