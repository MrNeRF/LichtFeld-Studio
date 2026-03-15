# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for declarative built-in menu schemas."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _source_python_path(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "src" / "python"))


def _install_lichtfeld_stub(monkeypatch):
    state = {
        "theme": "dark",
        "ui_scale": 1.0,
        "python_console_shown": 0,
        "undo_called": False,
        "redo_called": False,
    }

    ui = SimpleNamespace(
        tr=lambda key: f"tr:{key}",
        get_theme=lambda: state["theme"],
        set_theme=lambda theme: state.__setitem__("theme", theme),
        get_ui_scale_preference=lambda: state["ui_scale"],
        set_ui_scale=lambda scale: state.__setitem__("ui_scale", scale),
        show_python_console=lambda: state.__setitem__("python_console_shown", state["python_console_shown"] + 1),
        is_windows_platform=lambda: False,
        are_file_associations_registered=lambda: False,
    )

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = ui
    lf_stub.undo = SimpleNamespace(
        can_undo=lambda: True,
        can_redo=lambda: False,
        undo=lambda: state.__setitem__("undo_called", True),
        redo=lambda: state.__setitem__("redo_called", True),
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return state


def test_menu_helpers_and_builtin_schemas(monkeypatch):
    monkeypatch.delitem(sys.modules, "lfs_plugins", raising=False)
    monkeypatch.delitem(sys.modules, "lfs_plugins.layouts.menus", raising=False)
    monkeypatch.delitem(sys.modules, "lfs_plugins.edit_menu", raising=False)
    monkeypatch.delitem(sys.modules, "lfs_plugins.view_menu", raising=False)

    state = _install_lichtfeld_stub(monkeypatch)

    menus_mod = import_module("lfs_plugins.layouts.menus")
    edit_mod = import_module("lfs_plugins.edit_menu")
    view_mod = import_module("lfs_plugins.view_menu")

    class _OperatorStub:
        label = "demo.operator"

        @classmethod
        def _class_id(cls):
            return "demo.Operator"

    operator_item = menus_mod.menu_operator(_OperatorStub)
    assert operator_item == {
        "type": "operator",
        "operator_id": "demo.Operator",
        "label": "tr:demo.operator",
    }

    edit_items = edit_mod.EditMenu().menu_items()
    assert len(edit_items) == 2
    assert edit_items[0]["type"] == "item"
    assert edit_items[0]["label"] == "Undo"
    edit_items[0]["callback"]()
    assert state["undo_called"] is True
    assert edit_items[1]["type"] == "item"
    assert edit_items[1]["label"] == "Redo"
    assert edit_items[1]["enabled"] is False
    assert all(item["label"] in {"Undo", "Redo"} for item in edit_items)

    view_items = view_mod.ViewMenu().menu_items()
    assert view_items[0]["type"] == "submenu"
    assert view_items[1]["type"] == "submenu"
    assert view_items[3]["shortcut"] == "Ctrl+`"
    view_items[3]["callback"]()
    assert state["python_console_shown"] == 1
