# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused contracts for File/Open Recent labels and tooltips."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_file_menu(monkeypatch, recent_paths=()):
    for module_name in list(sys.modules):
        if module_name == "lfs_plugins" or module_name.startswith("lfs_plugins."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    package = ModuleType("lfs_plugins")
    package.__path__ = [str(PROJECT_ROOT / "src" / "python" / "lfs_plugins")]
    monkeypatch.setitem(sys.modules, "lfs_plugins", package)

    def tr(key):
        if key == "menu.file.recent_entry":
            return "{name} — {parent}"
        return f"tr:{key}"

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(tr=tr)
    lf_stub.project_recent_files = lambda: list(recent_paths)
    lf_stub.project_clear_recent_files = lambda: None
    lf_stub.project_auto_save_on_close_enabled = lambda: False
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)

    class Operator:
        @classmethod
        def _class_id(cls):
            return f"{cls.__module__}.{cls.__qualname__}"

    types_stub = ModuleType("lfs_plugins.types")
    types_stub.Operator = Operator
    monkeypatch.setitem(sys.modules, "lfs_plugins.types", types_stub)

    asset_stub = ModuleType("lfs_plugins.asset_manager_integration")
    asset_stub.register_catalog_asset_path = lambda *_args, **_kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "lfs_plugins.asset_manager_integration",
        asset_stub,
    )

    imports_stub = ModuleType("lfs_plugins.import_panels")
    imports_stub.open_dataset_import_panel = lambda _path: None
    imports_stub.open_resume_checkpoint_panel = lambda _path: None
    monkeypatch.setitem(sys.modules, "lfs_plugins.import_panels", imports_stub)

    return import_module("lfs_plugins.file_menu")


def test_recent_project_entry_uses_compact_parent_hint_and_full_path_tooltip(
    monkeypatch,
):
    recent_path = "/home/paja/scans/garden/project.licht"
    file_menu = _load_file_menu(monkeypatch, [recent_path])
    tr = file_menu.lf.ui.tr

    assert file_menu.format_recent_project_entry(recent_path, tr) == (
        "project.licht — scans/garden",
        recent_path,
    )
    assert file_menu.format_recent_project_entry("/tmp/project.licht", tr) == (
        "project.licht — tmp",
        "/tmp/project.licht",
    )
    assert file_menu.format_recent_project_entry("/project.licht", tr) == (
        "project.licht",
        "/project.licht",
    )

    windows_path = r"C:\Users\paja\scans\garden\project.licht"
    assert file_menu.format_recent_project_entry(windows_path, tr) == (
        "project.licht — scans/garden",
        windows_path,
    )

    recent_item = file_menu.FileMenu().menu_items()[2]["items"][0]
    assert recent_item["label"] == "project.licht — scans/garden"
    assert recent_item["tooltip"] == recent_path


def test_open_recent_submenu_appends_clear_only_when_entries_exist(monkeypatch):
    recent_path = "/home/paja/scans/garden/project.licht"
    file_menu = _load_file_menu(monkeypatch, [recent_path])
    cleared = []
    file_menu.lf.project_clear_recent_files = lambda: cleared.append(True)

    populated = file_menu.FileMenu().menu_items()[2]["items"]
    assert populated[0]["label"] == "project.licht — scans/garden"
    assert populated[1]["type"] == "separator"
    assert populated[2]["label"] == "tr:menu.file.clear_recent_projects"
    assert populated[2]["enabled"] is True
    populated[2]["callback"]()
    assert cleared == [True]

    file_menu.lf.project_recent_files = lambda: []
    empty = file_menu.FileMenu().menu_items()[2]["items"]
    assert len(empty) == 1
    assert empty[0]["label"] == "tr:menu.file.no_recent_projects"
    assert empty[0]["enabled"] is False
    assert empty[0].get("type", "item") == "item"

