# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the selection group list in the Rendering panel."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_lf_stub(monkeypatch):
    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(tr=lambda key: key, get_current_language=lambda: "en")
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


def _make_scene(groups, active=1, update_counts=lambda: None, **methods):
    return SimpleNamespace(
        active_selection_group=active,
        selection_groups=lambda: groups,
        update_selection_group_counts=update_counts,
        **methods,
    )


def _make_lf(scene):
    context_menu_state = SimpleNamespace(items=None, callback=None)

    def show_context_menu(items, _sx, _sy, on_action=None):
        context_menu_state.items = items
        context_menu_state.callback = on_action

    return SimpleNamespace(
        get_scene=lambda: scene,
        ui=SimpleNamespace(
            show_context_menu=show_context_menu,
            get_mouse_screen_pos=lambda: (120.0, 220.0),
            tr=lambda key: key,
        ),
        context_menu_state=context_menu_state,
    )


def _make_section(module):
    section = module.SelectionGroupsSection(lambda _gid, _event: None)
    section.attach(_HandleStub())
    return section


def test_selection_groups_builds_record_list(selection_groups_module):
    section = _make_section(selection_groups_module)
    groups = [
        _make_group(1, "Foreground", 5, False, (1.0, 0.0, 0.0)),
        _make_group(2, "Background", 3, True, (0.0, 0.5, 1.0)),
    ]
    selection_groups_module.lf = _make_lf(_make_scene(groups, active=2))

    assert section.sync() is True

    assert section._handle.records["selection_groups"] == [
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
    section = _make_section(selection_groups_module)
    section._has_groups = True
    selection_groups_module.lf = _make_lf(_make_scene([], active=-1))

    section.sync()

    assert section._handle.records["selection_groups"] == []
    assert "show_no_selection_groups" in section._handle.dirty_fields


def test_selection_groups_counts_once_per_selection_change(selection_groups_module):
    section = _make_section(selection_groups_module)
    count_updates = 0

    def update_counts():
        nonlocal count_updates
        count_updates += 1

    groups = [_make_group(1, "Foreground", 5, False, (1.0, 0.0, 0.0))]
    selection_groups_module.lf = _make_lf(_make_scene(groups, update_counts=update_counts))

    assert section.sync() is True
    assert section.sync() is False
    assert count_updates == 1

    selection_groups_module.RuntimeState.selection_generation.value += 1
    section.sync()

    assert count_updates == 2


def test_selection_groups_hidden_list_skips_host_count(selection_groups_module):
    # Catches a collapsed section that still copies the selection mask on every stroke.
    section = _make_section(selection_groups_module)
    count_updates = 0

    def update_counts():
        nonlocal count_updates
        count_updates += 1

    groups = [_make_group(1, "Foreground", 5, False, (1.0, 0.0, 0.0))]
    selection_groups_module.lf = _make_lf(_make_scene(groups, update_counts=update_counts))

    selection_groups_module.RuntimeState.selection_generation.value += 1
    assert section.sync(visible=False) is False
    assert count_updates == 0

    assert section.sync() is True
    assert count_updates == 1


def test_selection_groups_color_edit_updates_scene_and_row(selection_groups_module):
    section = _make_section(selection_groups_module)
    groups = [_make_group(1, "Foreground", 5, False, (1.0, 0.0, 0.0))]

    def set_color(group_id, color):
        next(g for g in groups if g.id == group_id).color = color

    selection_groups_module.lf = _make_lf(_make_scene(groups, set_selection_group_color=set_color))
    section.sync()

    section.set_group_color(1, (0.0, 1.0, 0.0))

    assert section.group_color(1) == (0.0, 1.0, 0.0)
    assert section._handle.records["selection_groups"][0]["color_css"] == "rgb(0,255,0)"


def test_selection_groups_context_menu_deletes_group(selection_groups_module):
    section = _make_section(selection_groups_module)
    groups = [_make_group(1, "Foreground", 5, False, (1.0, 0.0, 0.0))]

    def remove_group(group_id):
        groups[:] = [group for group in groups if group.id != group_id]

    selection_groups_module.lf = _make_lf(_make_scene(groups, remove_selection_group=remove_group))

    section._show_context_menu(1)
    assert callable(selection_groups_module.lf.context_menu_state.callback)

    selection_groups_module.lf.context_menu_state.callback("delete")

    assert groups == []
    assert section._handle.records["selection_groups"] == []
