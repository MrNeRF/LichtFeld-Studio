# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the startup recent-projects chooser panel."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_lf_stub(monkeypatch, tmp_path, *, recent_paths=None, dispositions=None):
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

    state = SimpleNamespace(
        recent_paths=list(recent_paths or []),
        dispositions=dict(dispositions or {}),
        project_open_calls=[],
        panel_enabled_calls=[],
        has_path=False,
        scene_empty=True,
    )

    def tr(key):
        return key

    class Panel:
        id = ""
        label = ""
        space = panel_space.FLOATING
        order = 100
        options = set()
        template = ""
        height_mode = panel_height_mode.CONTENT
        size = None
        update_policy = "interval"

        def on_bind_model(self, ctx):
            del ctx

        def on_mount(self, doc):
            del doc

        def on_unmount(self, doc):
            del doc

        def on_update(self, doc):
            del doc

    ui = SimpleNamespace(
        Panel=Panel,
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=tr,
        set_panel_enabled=lambda panel_id, enabled: state.panel_enabled_calls.append(
            (panel_id, enabled)
        ),
        is_scene_empty=lambda: bool(state.scene_empty),
        schedule_on_ui_thread=None,
    )

    lf = ModuleType("lichtfeld")
    lf.ui = ui
    lf.project_recent_files = lambda: list(state.recent_paths)
    lf.project_autosave_recovery_disposition = lambda path: state.dispositions.get(
        str(path), "none"
    )
    lf.project_open = lambda path="", discard_changes=False: state.project_open_calls.append(
        {"path": path, "discard_changes": discard_changes}
    )
    lf.project_has_path = lambda: bool(state.has_path)

    monkeypatch.setitem(sys.modules, "lichtfeld", lf)
    return state


@pytest.fixture
def startup_recent_module(monkeypatch, tmp_path):
    for name in [n for n in list(sys.modules) if n == "lfs_plugins" or n.startswith("lfs_plugins.")]:
        sys.modules.pop(name, None)
    state = _install_lf_stub(monkeypatch, tmp_path)
    module = import_module("lfs_plugins.startup_recent_panel")
    return module, state


def test_build_rows_caps_at_five_newest_first(startup_recent_module, tmp_path):
    module, _state = startup_recent_module
    paths = []
    for i in range(7):
        p = tmp_path / f"project_{i}.licht"
        p.write_text("x")
        paths.append(str(p))
    newest_first = list(reversed(paths))
    rows = module.build_project_rows(newest_first)
    assert len(rows) == 5
    assert [row["path"] for row in rows] == newest_first[:5]
    assert rows[0]["display_name"] == "project_6"
    assert rows[0]["index"] == 0


def test_open_action_calls_project_open_and_dismisses(startup_recent_module, tmp_path):
    module, state = startup_recent_module
    path = tmp_path / "scene.licht"
    path.write_text("x")
    state.recent_paths = [str(path)]

    panel = module.StartupRecentPanel()
    panel._handle = SimpleNamespace(
        update_record_list=lambda *a, **k: None,
        dirty_all=lambda: None,
    )
    panel._rows = module.build_project_rows(state.recent_paths)
    panel._open_path(str(path))

    assert state.project_open_calls == [{"path": str(path), "discard_changes": True}]
    assert ("lfs.startup_recent", False) in state.panel_enabled_calls


def test_start_blank_dismisses_without_opening(startup_recent_module, tmp_path):
    module, state = startup_recent_module
    path = tmp_path / "scene.licht"
    path.write_text("x")
    state.recent_paths = [str(path)]

    panel = module.StartupRecentPanel()
    panel._handle = SimpleNamespace(
        update_record_list=lambda *a, **k: None,
        dirty_all=lambda: None,
    )
    panel._on_start_blank(None, None, None)

    assert state.project_open_calls == []
    assert ("lfs.startup_recent", False) in state.panel_enabled_calls


def test_recovery_offer_badges_row_and_crash_notice(startup_recent_module, tmp_path):
    module, state = startup_recent_module
    top = tmp_path / "latest.licht"
    other = tmp_path / "older.licht"
    top.write_text("x")
    other.write_text("y")
    state.recent_paths = [str(top), str(other)]
    state.dispositions = {str(top): "offer", str(other): "none"}

    rows = module.build_project_rows(state.recent_paths)
    assert rows[0]["recoverable"] is True
    assert rows[0]["show_recoverable"] is True
    assert rows[1]["recoverable"] is False

    panel = module.StartupRecentPanel()
    updates = []

    def update_record_list(name, items):
        updates.append((name, list(items)))

    panel._handle = SimpleNamespace(
        update_record_list=update_record_list,
        dirty_all=lambda: None,
    )
    panel._refresh_rows(force=True)
    assert panel._show_crash_notice is True
    assert updates and updates[-1][0] == "projects"
    assert updates[-1][1][0]["recoverable"] is True


def test_should_show_requires_mru_and_blank_session(startup_recent_module, tmp_path):
    module, state = startup_recent_module
    assert module.should_show_startup_recent() is False

    path = tmp_path / "a.licht"
    path.write_text("x")
    state.recent_paths = [str(path)]
    assert module.should_show_startup_recent() is True

    state.has_path = True
    assert module.should_show_startup_recent() is False
    state.has_path = False
    state.scene_empty = False
    assert module.should_show_startup_recent() is False


def test_elide_middle_shortens_long_paths(startup_recent_module):
    module, _state = startup_recent_module
    long_path = "/home/user/very/long/directory/structure/project_name.licht"
    elided = module.elide_middle(long_path, max_len=40)
    assert len(elided) <= 40
    assert elided.startswith("/home")
    assert elided.endswith(".licht")
    assert "..." in elided
