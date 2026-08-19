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
        if key == "menu.file.recent_missing_message":
            return "{path} missing"
        return f"tr:{key}"

    opened = []
    removed = []
    confirm_dialogs = []
    message_dialogs = []

    def project_open(path, discard=False):
        opened.append((path, discard))

    def project_remove_recent_file(path):
        removed.append(path)

    def confirm_dialog(title, message, buttons, callback=None):
        confirm_dialogs.append((title, message, buttons, callback))

    def message_dialog(title, message, style=None):
        message_dialogs.append((title, message, style))

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        tr=tr,
        confirm_dialog=confirm_dialog,
        message_dialog=message_dialog,
    )
    lf_stub.project_recent_files = lambda: list(recent_paths)
    lf_stub.project_clear_recent_files = lambda: None
    lf_stub.project_auto_save_on_close_enabled = lambda: False
    lf_stub.project_is_dirty = lambda: False
    lf_stub.project_has_path = lambda: False
    lf_stub.project_open = project_open
    lf_stub.project_remove_recent_file = project_remove_recent_file
    lf_stub.project_open_calls = opened
    lf_stub.project_remove_recent_file_calls = removed
    lf_stub.confirm_dialogs = confirm_dialogs
    lf_stub.message_dialogs = message_dialogs
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


def _recent_item_callback(file_menu):
    return file_menu.FileMenu().menu_items()[2]["items"][0]["callback"]


def test_open_recent_missing_path_offers_remove(monkeypatch):
    missing_path = "/no/such/recent/project.licht"
    file_menu = _load_file_menu(monkeypatch, [missing_path])

    _recent_item_callback(file_menu)()

    assert file_menu.lf.project_open_calls == []
    assert len(file_menu.lf.confirm_dialogs) == 1
    title, message, buttons, callback = file_menu.lf.confirm_dialogs[0]
    assert title == "tr:menu.file.recent_missing_title"
    assert buttons == [
        "tr:menu.file.remove_from_recent",
        "tr:common.cancel",
    ]
    assert missing_path in message

    callback("tr:common.cancel")
    assert file_menu.lf.project_remove_recent_file_calls == []

    callback("tr:menu.file.remove_from_recent")
    assert file_menu.lf.project_remove_recent_file_calls == [missing_path]


def test_open_recent_existing_path_opens_without_dialog(monkeypatch, tmp_path):
    project = tmp_path / "existing.licht"
    project.write_bytes(b"")
    path = str(project)
    file_menu = _load_file_menu(monkeypatch, [path])

    _recent_item_callback(file_menu)()

    assert file_menu.lf.project_open_calls == [(path, True)]
    assert file_menu.lf.confirm_dialogs == []
    assert file_menu.lf.message_dialogs == []


def test_open_recent_existing_file_not_found_offers_remove(monkeypatch, tmp_path):
    project = tmp_path / "gone.licht"
    project.write_bytes(b"")
    path = str(project)
    file_menu = _load_file_menu(monkeypatch, [path])

    def raise_not_found(selected, discard=False):
        raise FileNotFoundError(selected)

    file_menu.lf.project_open = raise_not_found
    _recent_item_callback(file_menu)()

    assert len(file_menu.lf.confirm_dialogs) == 1
    title, _message, buttons, _callback = file_menu.lf.confirm_dialogs[0]
    assert title == "tr:menu.file.recent_missing_title"
    assert buttons == [
        "tr:menu.file.remove_from_recent",
        "tr:common.cancel",
    ]
    assert file_menu.lf.message_dialogs == []


def test_open_recent_existing_other_error_shows_message(monkeypatch, tmp_path):
    project = tmp_path / "broken.licht"
    project.write_bytes(b"")
    path = str(project)
    file_menu = _load_file_menu(monkeypatch, [path])

    def raise_boom(selected, discard=False):
        raise RuntimeError("boom")

    file_menu.lf.project_open = raise_boom
    _recent_item_callback(file_menu)()

    assert file_menu.lf.confirm_dialogs == []
    assert len(file_menu.lf.message_dialogs) == 1
    _title, message, style = file_menu.lf.message_dialogs[0]
    assert style == "error"
    assert "boom" in message


def _compact_project_item(file_menu):
    for item in file_menu.FileMenu().menu_items():
        if str(item.get("operator_id", "")).endswith("CompactProjectOperator"):
            return item
    raise AssertionError("CompactProjectOperator menu entry missing")


def test_compact_project_enabled_only_with_durable_path(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)

    file_menu.lf.project_has_path = lambda: False
    disabled = _compact_project_item(file_menu)
    assert disabled["enabled"] is False

    file_menu.lf.project_has_path = lambda: True
    enabled = _compact_project_item(file_menu)
    assert "enabled" not in enabled or enabled["enabled"] is True

    def raise_missing():
        raise RuntimeError("project_has_path unavailable")

    file_menu.lf.project_has_path = raise_missing
    guarded = _compact_project_item(file_menu)
    assert guarded["enabled"] is False

