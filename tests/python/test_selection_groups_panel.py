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
    module.RuntimeState.scene_generation.value = 0
    module.RuntimeState.selection_generation.value = 0
    module.RuntimeState.active_tool.value = "builtin.select"
    return module


class _HandleStub:
    def __init__(self):
        self.records = {}
        self.dirty_fields = []

    def update_record_list(self, name, rows):
        self.records[name] = rows

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_fields.append("__all__")

    def request_update(self):
        self.dirty_fields.append("__update__")


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
    context_menu_state = SimpleNamespace(items=None, callback=None)

    def show_context_menu(items, _sx, _sy, on_action=None):
        context_menu_state.items = items
        context_menu_state.callback = on_action

    return SimpleNamespace(
        get_scene=lambda: scene,
        ui=SimpleNamespace(
            get_active_tool=lambda: "builtin.select",
            poll_context_menu=lambda: None,
            show_context_menu=show_context_menu,
            get_mouse_screen_pos=lambda: (120.0, 220.0),
            tr=lambda key: key,
        ),
        context_menu_state=context_menu_state,
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


def test_selection_groups_uses_dirty_update_policy(selection_groups_module):
    assert selection_groups_module.SelectionGroupsPanel.update_policy == "dirty"
    assert "update_interval_ms" not in selection_groups_module.SelectionGroupsPanel.__dict__


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

    selection_groups_module.RuntimeState.selection_generation.value += 1
    panel.on_update(doc)

    assert count_updates == 2


def test_selection_groups_store_update_invalidates_dirty_panel(selection_groups_module):
    panel = selection_groups_module.SelectionGroupsPanel()
    panel._handle = _HandleStub()

    panel._subscribe_reactive_state()
    try:
        selection_groups_module.RuntimeState.selection_generation.value += 1

        assert "__update__" in panel._handle.dirty_fields
    finally:
        panel._unsubscribe_reactive_state()


def test_selection_groups_context_menu_uses_callback_without_poll(selection_groups_module):
    panel = selection_groups_module.SelectionGroupsPanel()
    panel._handle = _HandleStub()
    groups = [_make_group(1, "Foreground", 5, False, (1.0, 0.0, 0.0))]

    def remove_group(group_id):
        groups[:] = [group for group in groups if group.id != group_id]

    scene = SimpleNamespace(
        active_selection_group=1,
        selection_groups=lambda: groups,
        update_selection_group_counts=lambda: None,
        remove_selection_group=remove_group,
    )
    selection_groups_module.lf = _make_panel_lf(scene)

    panel._show_context_menu(1, SimpleNamespace())
    assert callable(selection_groups_module.lf.context_menu_state.callback)

    selection_groups_module.lf.context_menu_state.callback("delete")

    assert groups == []
    assert panel._handle.records["groups"] == []
