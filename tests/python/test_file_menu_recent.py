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
    opened_stop_training = []
    opened_keep_asset_manager = []
    new_projects = []
    removed = []
    confirm_dialogs = []
    message_dialogs = []
    warnings = []
    training_active = False

    def project_open(
        path,
        discard=False,
        stop_training=False,
        keep_asset_manager_open=False,
    ):
        opened.append((path, discard))
        opened_stop_training.append(stop_training)
        opened_keep_asset_manager.append(keep_asset_manager_open)

    def new_project(discard=False, stop_training=False):
        new_projects.append((discard, stop_training))

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
        open_dataset_folder_dialog=lambda: "",
        open_ply_file_dialog=lambda _path: "",
        open_mesh_file_dialog=lambda _path: "",
        open_checkpoint_file_dialog=lambda: "",
        open_json_file_dialog=lambda: "",
    )
    lf_stub.log = SimpleNamespace(warn=warnings.append)
    lf_stub.project_recent_files = lambda: list(recent_paths)
    lf_stub.project_clear_recent_files = lambda: None
    lf_stub.project_auto_save_on_close_enabled = lambda: False
    lf_stub.project_is_dirty = lambda: False
    lf_stub.project_has_path = lambda: False
    lf_stub.project_open = project_open
    lf_stub.new_project = new_project
    lf_stub.is_training_active = lambda: training_active
    lf_stub.project_remove_recent_file = project_remove_recent_file
    lf_stub.project_open_calls = opened
    lf_stub.project_open_stop_training = opened_stop_training
    lf_stub.project_open_keep_asset_manager = opened_keep_asset_manager
    lf_stub.new_project_calls = new_projects
    lf_stub.project_remove_recent_file_calls = removed
    lf_stub.confirm_dialogs = confirm_dialogs
    lf_stub.message_dialogs = message_dialogs
    lf_stub.warning_messages = warnings
    loaded = []

    def load_file(*args, **kwargs):
        loaded.append((args, kwargs))

    lf_stub.load_file = load_file
    lf_stub.load_file_calls = loaded
    lf_stub.project_save = lambda *args, **kwargs: True
    lf_stub.project_save_as = lambda *args, **kwargs: True
    lf_stub.load_config_file = lambda *_args, **_kwargs: None
    lf_stub.is_dataset_path = lambda _path: True
    lf_stub.read_checkpoint_header = lambda _path: object()
    lf_stub.read_checkpoint_params = lambda _path: object()
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)

    class Operator:
        @classmethod
        def _class_id(cls):
            return f"{cls.__module__}.{cls.__qualname__}"

    types_stub = ModuleType("lfs_plugins.types")
    types_stub.Operator = Operator
    monkeypatch.setitem(sys.modules, "lfs_plugins.types", types_stub)

    imports_stub = ModuleType("lfs_plugins.import_panels")
    imports_stub.open_dataset_import_panel = lambda _path: True
    imports_stub.open_resume_checkpoint_panel = lambda _path: True
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


def test_open_project_with_confirmation_handles_dirty_project(monkeypatch):
    path = "/tmp/catalog-project.licht"
    file_menu = _load_file_menu(monkeypatch)
    file_menu.lf.project_is_dirty = lambda: True

    file_menu.open_project_with_confirmation(path)

    assert file_menu.lf.project_open_calls == []
    assert len(file_menu.lf.confirm_dialogs) == 1
    title, _message, buttons, callback = file_menu.lf.confirm_dialogs[0]
    assert title == "tr:menu.file.open_project"
    assert buttons == [
        "tr:menu.file.save_project_as",
        "tr:unsaved_work.continue_without_saving",
        "tr:common.cancel",
    ]

    callback("tr:common.cancel")
    assert file_menu.lf.project_open_calls == []

    callback("tr:unsaved_work.continue_without_saving")
    assert file_menu.lf.project_open_calls == [(path, True)]


def test_asset_manager_open_can_keep_panel_open(monkeypatch):
    path = "/tmp/catalog-project.licht"
    file_menu = _load_file_menu(monkeypatch)

    file_menu.open_project_with_confirmation(
        path,
        keep_asset_manager_open=True,
    )

    assert file_menu.lf.project_open_calls == [(path, True)]
    assert file_menu.lf.project_open_keep_asset_manager == [True]


def test_open_project_with_confirmation_reports_open_error(monkeypatch):
    path = "/tmp/broken-catalog-project.licht"
    file_menu = _load_file_menu(monkeypatch)

    def raise_boom(_path, _discard=False, _stop_training=False):
        raise RuntimeError("open failed")

    file_menu.lf.project_open = raise_boom
    file_menu.open_project_with_confirmation(path)

    assert file_menu.lf.message_dialogs == [
        ("tr:menu.file.open_project", "open failed", "error")
    ]


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


def test_imports_are_grouped_before_exports(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)
    items = file_menu.FileMenu().menu_items()

    import_index = next(
        index
        for index, item in enumerate(items)
        if item.get("type") == "submenu"
        and item.get("label") == "tr:menu.file.import"
    )
    import_items = items[import_index]["items"]
    operator_names = [
        item["operator_id"].rsplit(".", 1)[-1]
        for item in import_items
        if item.get("type") == "operator"
    ]

    assert operator_names == [
        "ImportDatasetOperator",
        "ImportPlyOperator",
        "ImportMeshOperator",
        "ImportCheckpointOperator",
        "ImportConfigOperator",
    ]
    assert import_items[-2]["type"] == "separator"
    assert items[import_index + 1]["operator_id"].endswith("ExportOperator")
    assert items[import_index + 2]["operator_id"].endswith(
        "ExportConfigOperator"
    )


def test_unrecognized_dataset_reports_modal_and_warning(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)
    selected = "/tmp/not-a-dataset"
    file_menu.lf.ui.open_dataset_folder_dialog = lambda: selected
    file_menu.lf.is_dataset_path = lambda _path: False

    result = file_menu.ImportDatasetOperator().execute(None)

    assert result == {"CANCELLED"}
    assert len(file_menu.lf.message_dialogs) == 1
    title, message, style = file_menu.lf.message_dialogs[0]
    assert title == "tr:menu.file.import_failed"
    assert message == "tr:menu.file.dataset_not_recognized"
    assert style == "error"
    assert file_menu.lf.warning_messages == [
        "Import rejected: path='/tmp/not-a-dataset', "
        "reason='dataset format was not recognized'"
    ]


def test_unrecognized_checkpoint_reports_modal_and_warning(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)
    selected = "/tmp/not-a-checkpoint.ckpt"
    file_menu.lf.ui.open_checkpoint_file_dialog = lambda: selected
    file_menu.lf.read_checkpoint_header = lambda _path: None

    result = file_menu.ImportCheckpointOperator().execute(None)

    assert result == {"CANCELLED"}
    assert len(file_menu.lf.message_dialogs) == 1
    title, message, style = file_menu.lf.message_dialogs[0]
    assert title == "tr:menu.file.import_failed"
    assert message == "tr:menu.file.checkpoint_not_recognized"
    assert style == "error"
    assert "checkpoint format was not recognized" in (
        file_menu.lf.warning_messages[0]
    )


def test_checkpoint_preflight_reads_only_the_header(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)
    selected = "/tmp/large.resume"
    header_reads = []
    file_menu.lf.ui.open_checkpoint_file_dialog = lambda: selected
    file_menu.lf.read_checkpoint_header = lambda path: header_reads.append(path) or object()

    def unexpected_parameter_read(_path):
        raise AssertionError("checkpoint parameters belong to the retained import panel")

    file_menu.lf.read_checkpoint_params = unexpected_parameter_read

    result = file_menu.ImportCheckpointOperator().execute(None)

    assert result == {"FINISHED"}
    assert header_reads == [selected]
    assert file_menu.lf.message_dialogs == []
    assert file_menu.lf.warning_messages == []


def test_immediate_import_error_reports_reason(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)
    selected = "/tmp/broken.ply"
    file_menu.lf.ui.open_ply_file_dialog = lambda _path: selected

    def fail_load(*_args, **_kwargs):
        raise RuntimeError("load failed")

    file_menu.lf.load_file = fail_load

    result = file_menu.ImportPlyOperator().execute(None)

    assert result == {"CANCELLED"}
    assert len(file_menu.lf.message_dialogs) == 1
    title, message, style = file_menu.lf.message_dialogs[0]
    assert title == "tr:menu.file.import_failed"
    assert message == "tr:menu.file.import_failed_message"
    assert style == "error"
    assert "load failed" in file_menu.lf.warning_messages[0]


def test_new_project_while_training_prompts_instead_of_switching(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)
    file_menu.lf.is_training_active = lambda: True

    file_menu.NewProjectOperator().execute(None)

    assert file_menu.lf.new_project_calls == []
    assert len(file_menu.lf.confirm_dialogs) == 1
    title, message, buttons, callback = file_menu.lf.confirm_dialogs[0]
    assert title == "tr:project_switch.stop_training_title"
    assert message == "tr:project_switch.stop_training_message"
    assert buttons == ["tr:common.yes", "tr:common.no"]

    callback("tr:common.no")
    assert file_menu.lf.new_project_calls == []

    callback("tr:common.yes")
    assert file_menu.lf.new_project_calls == [(True, True)]


def test_open_recent_while_training_prompts_instead_of_opening(
    monkeypatch, tmp_path
):
    project = tmp_path / "training.licht"
    project.write_bytes(b"")
    path = str(project)
    file_menu = _load_file_menu(monkeypatch, [path])
    file_menu.lf.is_training_active = lambda: True

    _recent_item_callback(file_menu)()

    assert file_menu.lf.project_open_calls == []
    assert file_menu.lf.message_dialogs == []
    assert len(file_menu.lf.confirm_dialogs) == 1
    title, _message, buttons, callback = file_menu.lf.confirm_dialogs[0]
    assert title == "tr:project_switch.stop_training_title"
    assert buttons == ["tr:common.yes", "tr:common.no"]

    callback("tr:common.no")
    assert file_menu.lf.project_open_calls == []
    assert file_menu.lf.project_open_stop_training == []

    callback("tr:common.yes")
    assert file_menu.lf.project_open_calls == [(path, True)]
    assert file_menu.lf.project_open_stop_training == [True]


def test_stop_training_confirmation_yes_retries_with_stop_flag(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)

    file_menu._show_stop_training_confirmation(True, "", True)
    assert len(file_menu.lf.confirm_dialogs) == 1
    _title, _message, buttons, callback = file_menu.lf.confirm_dialogs[0]
    assert buttons == ["tr:common.yes", "tr:common.no"]

    callback("tr:common.no")
    assert file_menu.lf.new_project_calls == []

    callback("tr:common.yes")
    assert file_menu.lf.new_project_calls == [(True, True)]

    file_menu._show_stop_training_confirmation(
        False, "/tmp/other.licht", False
    )
    _title, _message, _buttons, open_callback = file_menu.lf.confirm_dialogs[1]
    open_callback("tr:common.yes")
    assert file_menu.lf.project_open_calls == [("/tmp/other.licht", False)]
    assert file_menu.lf.project_open_stop_training == [True]


def test_drag_open_confirmation_preserves_asset_manager(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)

    file_menu._show_project_switch_confirmation(
        False, "/tmp/dragged.licht", True
    )

    assert file_menu.lf.project_open_calls == [
        ("/tmp/dragged.licht", True)
    ]
    assert file_menu.lf.project_open_keep_asset_manager == [True]


def test_new_project_dirty_offers_save_continue_cancel(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)
    file_menu.lf.project_is_dirty = lambda: True
    file_menu.lf.project_has_path = lambda: True

    file_menu.NewProjectOperator().execute(None)

    assert file_menu.lf.new_project_calls == []
    assert len(file_menu.lf.confirm_dialogs) == 1
    title, message, buttons, callback = file_menu.lf.confirm_dialogs[0]
    assert title == "tr:menu.file.new_project"
    assert message == "tr:exit_popup.unsaved_warning"
    assert buttons == [
        "tr:common.save",
        "tr:unsaved_work.continue_without_saving",
        "tr:common.cancel",
    ]

    callback("tr:common.cancel")
    assert file_menu.lf.new_project_calls == []

    callback("tr:unsaved_work.continue_without_saving")
    assert file_menu.lf.new_project_calls == [(True, False)]


def test_load_file_confirmation_title_for_splat_and_dataset(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)
    file_menu.lf.project_is_dirty = lambda: True
    file_menu.lf.project_has_path = lambda: True

    file_menu._show_load_file_confirmation(["/tmp/a.ply"], False, False)
    splat_title, _splat_message, _splat_buttons, _splat_cb = (
        file_menu.lf.confirm_dialogs[0]
    )
    assert splat_title == "tr:unsaved_work.title"

    file_menu._show_load_file_confirmation(["/tmp/dataset"], True, False)
    dataset_title, _dataset_message, _dataset_buttons, _dataset_cb = (
        file_menu.lf.confirm_dialogs[1]
    )
    assert dataset_title == "tr:load_dataset_popup.save_title"


def test_load_file_confirmation_reissues_in_order_with_replace_on_first(
    monkeypatch,
):
    file_menu = _load_file_menu(monkeypatch)

    file_menu._show_load_file_confirmation(
        ["/tmp/a.ply", "/tmp/b.ply"], False, True
    )

    assert file_menu.lf.load_file_calls == [
        (
            ("/tmp/a.ply",),
            {
                "is_dataset": False,
                "discard_changes": True,
                "replace": True,
                "stop_training": False,
            },
        ),
        (
            ("/tmp/b.ply",),
            {
                "is_dataset": False,
                "discard_changes": True,
                "replace": False,
                "stop_training": False,
            },
        ),
    ]
