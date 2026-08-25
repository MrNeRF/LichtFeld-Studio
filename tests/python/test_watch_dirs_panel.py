# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavior tests for the .licht watched-directory dialog."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


@pytest.fixture
def watch_dirs_panel_module(monkeypatch, tmp_path):
    source_python = Path(__file__).resolve().parents[2] / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    state = SimpleNamespace(
        enabled=[],
        browse_path=str(tmp_path / "new-watch-root"),
        scheduled=[],
    )
    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=SimpleNamespace(FLOATING="FLOATING"),
        PanelHeightMode=SimpleNamespace(CONTENT="CONTENT"),
        PanelOption=SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED"),
        tr=lambda key: key,
        set_panel_enabled=lambda panel_id, enabled: state.enabled.append(
            (panel_id, enabled)
        ),
        open_folder_dialog=lambda _title, _start_dir: state.browse_path,
        schedule_on_ui_thread=lambda callback: state.scheduled.append(callback) or callback(),
    )
    lf_stub.register_class = lambda _panel: None
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    sys.modules.pop("lfs_plugins.watch_dirs_panel", None)
    sys.modules.pop("lfs_plugins", None)
    return import_module("lfs_plugins.watch_dirs_panel"), state


class _Index:
    def __init__(self, folder_id, watch_dirs):
        self.folders = {folder_id: {"id": folder_id, "name": "Projects"}}
        self._watch_dirs = list(watch_dirs)
        self.saved = []

    def get_watch_dirs(self, folder_id):
        assert folder_id in self.folders
        return list(self._watch_dirs)

    def set_watch_dirs(self, folder_id, paths):
        self.saved.append((folder_id, list(paths)))
        self._watch_dirs = list(paths)
        return True


class _ImmediateThread:
    def __init__(self, *, target, args, **_kwargs):
        self._target = target
        self._args = args

    def start(self):
        self._target(*self._args)

    def is_alive(self):
        return False

    def join(self):
        pass


def test_dialog_loads_adds_and_removes_watch_directories(
    watch_dirs_panel_module, tmp_path
):
    module, state = watch_dirs_panel_module
    existing = tmp_path / "existing"
    existing.mkdir()
    index = _Index("projects", [str(existing)])
    panel = module.WatchDirsDialogPanel()

    assert panel.show(index, "projects") is True
    assert panel._watch_dirs == [str(existing)]
    assert state.enabled[-1] == ("lfs.watch_dirs_dialog", True)

    state.browse_path = str(existing)
    panel._on_browse_add()
    assert panel._watch_dirs == [str(existing)]

    state.browse_path = str(tmp_path / "new")
    panel._on_browse_add()
    assert panel._watch_dirs == [str(existing), str(tmp_path / "new")]

    panel._on_remove_dir(args=["0"])
    assert panel._watch_dirs == [str(tmp_path / "new")]


def test_save_persists_then_scans_and_refreshes(
    watch_dirs_panel_module, monkeypatch, tmp_path
):
    module, state = watch_dirs_panel_module
    watched = tmp_path / "watched"
    watched.mkdir()
    index = _Index("projects", [str(watched)])
    panel = module.WatchDirsDialogPanel()
    refreshes = []
    scans = []
    panel.show(index, "projects", lambda: refreshes.append(True))

    monkeypatch.setattr(module.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        module,
        "scan_watch_directories",
        lambda scan_index, folder_id, directories, _cancel_event: scans.append(
            (scan_index, folder_id, directories)
        )
        or module.WatchScanResult(discovered=1, added=1),
    )

    panel._on_save()

    assert index.saved == [("projects", [str(watched)])]
    assert scans == [(index, "projects", [str(watched)])]
    assert refreshes == [True]
    assert len(state.scheduled) == 1
    assert panel._scan_done is True
    assert panel._scan_status == "watch_dirs.scan_complete_summary"
    assert state.enabled[-1] == ("lfs.watch_dirs_dialog", True)

    panel._on_save()
    assert state.enabled[-1] == ("lfs.watch_dirs_dialog", False)


def test_saving_empty_list_clears_watch_directories_without_scan(
    watch_dirs_panel_module, monkeypatch, tmp_path
):
    module, _state = watch_dirs_panel_module
    index = _Index("projects", [str(tmp_path / "old")])
    panel = module.WatchDirsDialogPanel()
    refreshes = []
    panel.show(index, "projects", lambda: refreshes.append(True))
    panel._watch_dirs = []
    monkeypatch.setattr(
        module,
        "scan_watch_directories",
        lambda *_args: pytest.fail("empty watch list must not start a scan"),
    )

    panel._on_save()

    assert index.saved == [("projects", [])]
    assert refreshes == [True]


def test_cancel_stops_and_joins_active_scan(watch_dirs_panel_module):
    module, state = watch_dirs_panel_module
    panel = module.WatchDirsDialogPanel()
    cancel_event = module.threading.Event()
    joined = []
    panel._scan_active = True
    panel._scan_cancel_event = cancel_event
    panel._scan_thread = SimpleNamespace(
        is_alive=lambda: True,
        join=lambda: joined.append(True),
    )

    panel._on_cancel()

    assert cancel_event.is_set() is True
    assert joined == [True]
    assert panel._scan_active is False
    assert state.enabled[-1] == ("lfs.watch_dirs_dialog", False)
