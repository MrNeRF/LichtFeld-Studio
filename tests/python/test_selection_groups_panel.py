# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the selection groups Rml panel data model."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_lf_stub(monkeypatch):
    panel_space = SimpleNamespace(
        SIDE_PANEL="SIDE_PANEL",
        FLOATING="FLOATING",
        VIEWPORT_OVERLAY="VIEWPORT_OVERLAY",
        MAIN_PANEL_TAB="MAIN_PANEL_TAB",
        SCENE_HEADER="SCENE_HEADER",
        STATUS_BAR="STATUS_BAR",
    )
    panel_height_mode = SimpleNamespace(FILL="fill", CONTENT="content")
    panel_option = SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED", HIDE_HEADER="HIDE_HEADER")
    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
        get_current_language=lambda: "en",
        get_active_tool=lambda: "builtin.select",
        poll_context_menu=lambda: None,
    )
    lf_stub.get_scene = lambda: None
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return lf_stub


@pytest.fixture
def selection_groups_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))
    sys.modules.pop("lfs_plugins.selection_groups", None)
    sys.modules.pop("lfs_plugins", None)
    _install_lf_stub(monkeypatch)
    module = import_module("lfs_plugins.selection_groups")
    module.AppState.reset()
    return module


class _HandleStub:
    def __init__(self):
        self.records = {}
        self.dirty_fields = []

    def update_record_list(self, name, rows):
        self.records[name] = rows

    def dirty(self, name):
        self.dirty_fields.append(name)


def _make_group(group_id, name, count, locked, color):
    return SimpleNamespace(id=group_id, name=name, count=count, locked=locked, color=color)


class _ElementStub:
    def __init__(self):
        self.classes = []

    def set_class(self, name, enabled):
        self.classes.append((name, enabled))


class _DocStub:
    def __init__(self):
        self.content_wrap = _ElementStub()

    def get_element_by_id(self, element_id):
        if element_id == "content-wrap":
            return self.content_wrap
        return None


def _make_panel_lf(scene):
    return SimpleNamespace(
        get_scene=lambda: scene,
        ui=SimpleNamespace(
            get_active_tool=lambda: "builtin.select",
            poll_context_menu=lambda: None,
        ),
    )


def test_selection_groups_builds_record_list(selection_groups_module):
    panel = selection_groups_module.SelectionGroupsPanel()
    panel._handle = _HandleStub()

    groups = [
        _make_group(1, "Foreground", 5, False, (1.0, 0.0, 0.0)),
        _make_group(2, "Background", 3, True, (0.0, 0.5, 1.0)),
    ]
    scene = SimpleNamespace(
        active_selection_group=2,
        selection_groups=lambda: groups,
        update_selection_group_counts=lambda: None,
    )

    selection_groups_module.lf = SimpleNamespace(get_scene=lambda: scene)

    panel._rebuild_groups()

    assert panel._handle.records["groups"] == [
        {
            "gid": "1",
            "active": False,
            "lock_sprite": "icon-unlocked",
            "color_css": "rgb(255,0,0)",
            "label": "Foreground (5)",
        },
        {
            "gid": "2",
            "active": True,
            "lock_sprite": "icon-locked",
            "color_css": "rgb(0,127,255)",
            "label": "Background (3)",
        },
    ]


def test_selection_groups_marks_empty_state_dirty(selection_groups_module):
    panel = selection_groups_module.SelectionGroupsPanel()
    panel._handle = _HandleStub()
    panel._has_groups = True

    scene = SimpleNamespace(
        active_selection_group=-1,
        selection_groups=lambda: [],
        update_selection_group_counts=lambda: None,
    )

    selection_groups_module.lf = SimpleNamespace(get_scene=lambda: scene)

    panel._rebuild_groups()

    assert panel._handle.records["groups"] == []
    assert "show_empty_message" in panel._handle.dirty_fields


def test_selection_groups_on_update_skips_unchanged_count_poll(selection_groups_module):
    panel = selection_groups_module.SelectionGroupsPanel()
    panel._handle = _HandleStub()

    count_updates = 0

    def update_counts():
        nonlocal count_updates
        count_updates += 1

    groups = [_make_group(1, "Foreground", 5, False, (1.0, 0.0, 0.0))]
    scene = SimpleNamespace(
        active_selection_group=1,
        selection_groups=lambda: groups,
        update_selection_group_counts=update_counts,
    )
    selection_groups_module.lf = _make_panel_lf(scene)

    doc = _DocStub()
    panel.on_update(doc)
    panel.on_update(doc)

    assert count_updates == 1

    selection_groups_module.AppState.selection_generation.value += 1
    panel.on_update(doc)

    assert count_updates == 2
