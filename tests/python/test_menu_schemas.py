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
        "copy_called": False,
        "cut_called": False,
        "paste_called": False,
        "invert_called": False,
        "select_all_called": False,
        "deselect_all_called": False,
        "can_edit_gaussian_selection": True,
        "has_gaussian_selection": True,
        "has_gaussian_clipboard": False,
        "active_tool": "builtin.select",
    }

    ui = SimpleNamespace(
        tr=lambda key: f"tr:{key}",
        themes=lambda: [
            {"id": "dark", "name": "Dark", "label_key": "menu.view.theme.dark", "order": 0},
            {"id": "light", "name": "Light", "label_key": "menu.view.theme.light", "order": 1},
        ],
        get_theme=lambda: state["theme"],
        set_theme=lambda theme: state.__setitem__("theme", theme),
        get_ui_scale_preference=lambda: state["ui_scale"],
        set_ui_scale=lambda scale: state.__setitem__("ui_scale", scale),
        show_python_console=lambda: state.__setitem__("python_console_shown", state["python_console_shown"] + 1),
        toggle_system_console=lambda: state.__setitem__("python_console_shown", state["python_console_shown"] + 1),
        set_panel_enabled=lambda _panel_id, _enabled: None,
        is_windows_platform=lambda: False,
        are_file_associations_registered=lambda: False,
        can_edit_gaussian_selection=lambda: state["can_edit_gaussian_selection"],
        has_gaussian_selection=lambda: state["has_gaussian_selection"],
        has_gaussian_clipboard=lambda: state["has_gaussian_clipboard"],
        get_active_tool=lambda: state["active_tool"],
        copy_gaussian_selection=lambda: state.__setitem__("copy_called", True),
        cut_gaussian_selection=lambda: state.__setitem__("cut_called", True),
        paste_gaussian_selection=lambda: state.__setitem__("paste_called", True),
        invert_gaussian_selection=lambda: state.__setitem__("invert_called", True),
        select_all_gaussians=lambda: state.__setitem__("select_all_called", True),
        deselect_all_gaussians=lambda: state.__setitem__("deselect_all_called", True),
    )
    keymap = SimpleNamespace(
        Action=SimpleNamespace(
            UNDO="undo",
            REDO="redo",
            COPY_SELECTION="copy_selection",
            CUT_SELECTION="cut_selection",
            PASTE_SELECTION="paste_selection",
            INVERT_SELECTION="invert_selection",
            SELECT_ALL="select_all",
            DESELECT_ALL="deselect_all",
        ),
        ToolMode=SimpleNamespace(GLOBAL="global"),
        is_bound=lambda _action, _mode: True,
        get_trigger_description=lambda action, _mode: {
            "undo": "Ctrl+Z",
            "redo": "Ctrl+Shift+Z",
            "copy_selection": "Ctrl+C",
            "cut_selection": "Ctrl+X",
            "paste_selection": "Ctrl+V",
            "invert_selection": "Ctrl+I",
            "select_all": "Ctrl+A",
            "deselect_all": "Ctrl+D",
        }.get(action, "Unbound"),
    )

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = ui
    lf_stub.keymap = keymap
    lf_stub.reset_camera = lambda: None
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
    monkeypatch.delitem(sys.modules, "lfs_plugins.select_menu", raising=False)
    monkeypatch.delitem(sys.modules, "lfs_plugins.tools_menu", raising=False)
    monkeypatch.delitem(sys.modules, "lfs_plugins.view_menu", raising=False)

    state = _install_lichtfeld_stub(monkeypatch)

    menus_mod = import_module("lfs_plugins.layouts.menus")
    edit_mod = import_module("lfs_plugins.edit_menu")
    select_mod = import_module("lfs_plugins.select_menu")
    tools_mod = import_module("lfs_plugins.tools_menu")
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
    assert len(edit_items) == 4
    assert edit_items[0]["type"] == "item"
    assert edit_items[0]["label"] == "Undo"
    edit_items[0]["callback"]()
    assert state["undo_called"] is True
    assert edit_items[1]["type"] == "item"
    assert edit_items[1]["label"] == "Redo"
    assert edit_items[1]["enabled"] is False
    assert edit_items[2]["type"] == "separator"
    assert edit_items[3]["label"] == "tr:menu.edit.input_settings"

    assert edit_mod.EditMenu.order < select_mod.SelectMenu.order < tools_mod.ToolsMenu.order

    select_items = select_mod.SelectMenu().menu_items()
    assert len(select_items) == 7
    assert select_items[0]["label"] == "tr:menu.select.copy_selection"
    assert select_items[0]["shortcut"] == "Ctrl+C"
    assert select_items[0]["enabled"] is True
    select_items[0]["callback"]()
    assert state["copy_called"] is True
    assert select_items[1]["label"] == "tr:menu.select.cut_selection"
    assert select_items[1]["shortcut"] == "Ctrl+X"
    assert select_items[1]["enabled"] is True
    select_items[1]["callback"]()
    assert state["cut_called"] is True
    assert select_items[2]["label"] == "tr:menu.select.paste_selection"
    assert select_items[2]["shortcut"] == "Ctrl+V"
    assert select_items[2]["enabled"] is False
    assert select_items[3]["type"] == "separator"
    assert select_items[4]["label"] == "tr:menu.select.invert_selection"
    assert select_items[4]["enabled"] is True
    select_items[4]["callback"]()
    assert state["invert_called"] is True
    assert select_items[5]["label"] == "tr:menu.select.select_all"
    assert select_items[5]["shortcut"] == "Ctrl+A"
    select_items[5]["callback"]()
    assert state["select_all_called"] is True
    assert select_items[6]["label"] == "tr:menu.select.deselect_all"
    assert select_items[6]["shortcut"] == "Ctrl+D"
    assert select_items[6]["enabled"] is True
    select_items[6]["callback"]()
    assert state["deselect_all_called"] is True

    state["has_gaussian_selection"] = False
    state["has_gaussian_clipboard"] = True
    select_items = select_mod.SelectMenu().menu_items()
    assert select_items[0]["enabled"] is False
    assert select_items[1]["enabled"] is False
    assert select_items[2]["enabled"] is True
    assert select_items[6]["enabled"] is False

    state["active_tool"] = "builtin.translate"
    select_items = select_mod.SelectMenu().menu_items()
    assert select_items[4]["enabled"] is True
    assert select_items[5]["enabled"] is False

    state["active_tool"] = "builtin.select"
    state["can_edit_gaussian_selection"] = False
    select_items = select_mod.SelectMenu().menu_items()
    assert [item.get("enabled") for item in select_items if item["type"] == "item"] == [False] * 6

    view_items = view_mod.ViewMenu().menu_items()
    assert view_items[0]["type"] == "submenu"
    assert view_items[1]["type"] == "submenu"
    assert view_items[0]["items"][0]["selected"] is True
    assert view_items[1]["items"][1]["label"] == "100%"
    assert view_items[4]["label"] == "tr:main_panel.console"
    view_items[4]["callback"]()
    assert state["python_console_shown"] == 1
