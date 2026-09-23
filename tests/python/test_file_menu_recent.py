# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused contracts for File/Open Recent labels and tooltips."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


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
            return "{name} ({parent})"
        if key == "menu.file.recent_missing_message":
            return "{path} missing"
        return f"tr:{key}"

    opened = []
    opened_stop_training = []
    opened_keep_asset_manager = []
    new_projects = []
    embed_calls = []
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
    created = []

    def project_create(path, discard_changes=False, stop_training=False, overwrite=False):
        created.append((path, discard_changes, stop_training, overwrite))
        return True

    lf_stub.project_create = project_create
    lf_stub.project_create_calls = created
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
    lf_stub.embed_calls = embed_calls
    loaded = []

    def load_file(*args, **kwargs):
        loaded.append((args, kwargs))

    lf_stub.load_file = load_file
    lf_stub.load_file_calls = loaded
    lf_stub.project_save = lambda *args, **kwargs: True
    lf_stub.project_save_as = lambda *args, **kwargs: True
    lf_stub.project_can_embed_dataset = lambda: False
    lf_stub.project_embed_dataset = lambda: embed_calls.append(True)
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
    imports_stub.open_new_project_panel = lambda _path: True
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
        "project (scans/garden)",
        recent_path,
    )
    assert file_menu.format_recent_project_entry("/tmp/project.licht", tr) == (
        "project (tmp)",
        "/tmp/project.licht",
    )
    assert file_menu.format_recent_project_entry("/project.licht", tr) == (
        "project",
        "/project.licht",
    )

    windows_path = r"C:\Users\paja\scans\garden\project.licht"
    assert file_menu.format_recent_project_entry(windows_path, tr) == (
        "project (scans/garden)",
        windows_path,
    )

    recent_item = file_menu.FileMenu().menu_items()[2]["items"][0]
    assert recent_item["label"] == "project (scans/garden)"
    assert recent_item["tooltip"] == recent_path


def test_open_recent_submenu_appends_clear_only_when_entries_exist(monkeypatch):
    recent_path = "/home/paja/scans/garden/project.licht"
    file_menu = _load_file_menu(monkeypatch, [recent_path])
    cleared = []
    file_menu.lf.project_clear_recent_files = lambda: cleared.append(True)

    populated = file_menu.FileMenu().menu_items()[2]["items"]
    assert populated[0]["label"] == "project (scans/garden)"
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


def test_file_menu_publishes_current_project_from_review_without_asset_index(
    monkeypatch, tmp_path
):
    project = tmp_path / "project.licht"
    project.write_bytes(b"saved")
    file_menu = _load_file_menu(monkeypatch)
    file_menu.lf.project_has_path = lambda: True
    file_menu.lf.project_poll_write = lambda: {"path": str(project)}
    file_menu.lf.io = SimpleNamespace(
        inspect_project_card=lambda path: SimpleNamespace(
            project_uuid="project-id",
            commit_uuid="commit-id",
            file_uuid="file-id",
            title=None,
            physical_file_size=5,
            has_preview=False,
        )
    )
    controller = SimpleNamespace(
        upload_format="sog",
        service=SimpleNamespace(identity=lambda: ("https://gallery.test", "account")),
        snapshot=lambda: {"links": {}, "scenes": []},
    )
    opened = []
    gallery_panel = ModuleType("lfs_plugins.gallery_file_panel")
    gallery_panel.open_gallery_file_panel = lambda **review: opened.append(review)
    monkeypatch.setitem(sys.modules, "lfs_plugins.gallery_file_panel", gallery_panel)
    controller_module = ModuleType("lfs_plugins.gallery_controller")
    controller_module.get_gallery_controller = lambda: controller
    monkeypatch.setitem(sys.modules, "lfs_plugins.gallery_controller", controller_module)

    item = next(
        item for item in file_menu.FileMenu().menu_items()
        if item.get("label") == "tr:menu.file.publish_to_gallery"
    )
    assert item["enabled"] is True
    item["callback"]()

    assert len(opened) == 1
    review = opened[0]
    assert review["asset"]["id"] == "project-id"
    assert review["asset"]["path"] == str(project.resolve())
    assert review["action"] == "publish"
    assert review["fields"]["title"] == "project"
    assert review["fields"]["upload_format"] == "sog"
    assert review["expected_project_path"] == str(project.resolve())


@pytest.mark.parametrize(
    "state_name,expected",
    [("local", "update"), ("remote", "apply"), ("diverged", "resolve"),
     ("unknown", "check"), ("equal", "update")],
)
def test_file_menu_linked_project_uses_gallery_primary_action(monkeypatch, tmp_path, state_name, expected):
    project = tmp_path / "project.licht"
    project.write_bytes(b"saved")
    file_menu = _load_file_menu(monkeypatch)
    file_menu.lf.project_has_path = lambda: True
    file_menu.lf.project_poll_write = lambda: {"path": str(project)}
    file_menu.lf.io = SimpleNamespace(
        inspect_project_card=lambda _path: SimpleNamespace(
            project_uuid="project-id", commit_uuid="commit-id", file_uuid="file-id",
            title="Project", physical_file_size=5, has_preview=False,
        )
    )
    scene = {"id": "scene-id", "title": "Project", "contentLength": 10}
    link = {"sceneId": "scene-id"}
    calls = []
    controller = SimpleNamespace(
        upload_format="sog",
        service=SimpleNamespace(identity=lambda: ("https://gallery.test", "account"), busy=False),
        snapshot=lambda: {"links": {"project-id": link}, "scenes": [scene]},
        refresh=lambda: calls.append(("check",)),
        resolve_asset=lambda asset, details, **kwargs: calls.append(("resolve", kwargs)),
        open_portal=lambda selected, action: calls.append((action, selected["id"])),
        _schedule_poll=lambda: calls.append(("schedule",)),
    )
    controller_module = ModuleType("lfs_plugins.gallery_controller")
    controller_module.get_gallery_controller = lambda: controller
    controller_module.asset_sync_state = lambda *args, **kwargs: {
        "relationship": "linked", "linked": True, "freshness": state_name,
        "state": state_name, "activity": "idle", "active": False,
        "job": {}, "sceneReady": True, "presentationChanged": False,
    }
    monkeypatch.setitem(sys.modules, "lfs_plugins.gallery_controller", controller_module)
    opened = []
    gallery_panel = ModuleType("lfs_plugins.gallery_file_panel")
    gallery_panel.open_gallery_file_panel = lambda **review: opened.append(review)
    monkeypatch.setitem(sys.modules, "lfs_plugins.gallery_file_panel", gallery_panel)

    item = next(item for item in file_menu.FileMenu().menu_items()
                if item.get("label") == "tr:menu.file.publish_to_gallery")
    item["callback"]()

    if expected == "update":
        assert len(opened) == 1 and opened[0]["action"] == "update"
        assert calls == []
    elif expected == "apply":
        assert calls == [("resolve", {"apply_only": True})] and opened == []
    elif expected == "resolve":
        assert calls == [("resolve", {"apply_only": False})] and opened == []
    elif expected == "check":
        assert calls == [("check",)] and opened == []
        assert file_menu.lf.message_dialogs == []
        assert callable(controller._after_service)

        controller_module.asset_sync_state = lambda *args, **kwargs: {
            "relationship": "linked", "linked": True, "freshness": "local",
            "state": "local", "activity": "idle", "active": False,
            "job": {}, "sceneReady": True, "presentationChanged": False,
        }
        controller._after_service()
        assert len(opened) == 1 and opened[0]["action"] == "update"


def _linked_file_publish_harness(monkeypatch, tmp_path, snapshot):
    """File → Publish for a linked saved project, using real asset_sync_state."""
    project = tmp_path / "project.licht"
    project.write_bytes(b"saved")
    file_menu = _load_file_menu(monkeypatch)
    file_menu.lf.project_has_path = lambda: True
    file_menu.lf.project_poll_write = lambda: {"path": str(project)}
    file_menu.lf.io = SimpleNamespace(
        inspect_project_card=lambda _path: SimpleNamespace(
            project_uuid="project-id", commit_uuid="commit-id", file_uuid="file-id",
            title="Project", physical_file_size=5, has_preview=False,
        )
    )
    calls = []
    controller = SimpleNamespace(
        upload_format="sog",
        service=SimpleNamespace(identity=lambda: ("https://gallery.test", "account"), busy=False),
        snapshot=lambda: snapshot,
        refresh=lambda: calls.append("refresh"),
        resolve_asset=lambda asset, details, **kwargs: calls.append(("resolve", kwargs)),
        _schedule_poll=lambda: None,
    )
    from lfs_plugins import gallery_controller as controller_module
    sync_calls = []
    real_sync = controller_module.asset_sync_state

    def capturing_sync(*args, **kwargs):
        sync_calls.append((args, kwargs))
        return real_sync(*args, **kwargs)

    monkeypatch.setattr(controller_module, "get_gallery_controller", lambda: controller)
    monkeypatch.setattr(controller_module, "asset_sync_state", capturing_sync)
    opened = []
    gallery_panel = ModuleType("lfs_plugins.gallery_file_panel")
    gallery_panel.open_gallery_file_panel = lambda **review: opened.append(review)
    monkeypatch.setitem(sys.modules, "lfs_plugins.gallery_file_panel", gallery_panel)

    def invoke():
        item = next(
            item for item in file_menu.FileMenu().menu_items()
            if item.get("label") == "tr:menu.file.publish_to_gallery"
        )
        item["callback"]()

    return SimpleNamespace(
        file_menu=file_menu,
        controller=controller,
        opened=opened,
        calls=calls,
        sync_calls=sync_calls,
        invoke=invoke,
        project=project,
    )


def _sync_scene_and_checked(sync_calls):
    args, kwargs = sync_calls[-1]
    return args[2], kwargs["checked"]


def test_file_menu_linked_missing_scene_refreshes_before_checked(monkeypatch, tmp_path):
    link = {"sceneId": "scene-id", "commitUuid": "commit-id"}
    snapshot = {
        "links": {"project-id": link},
        "scenes": [{"id": "other-scene", "title": "Other"}],
        "jobs": [],
    }
    harness = _linked_file_publish_harness(monkeypatch, tmp_path, snapshot)
    harness.invoke()

    assert harness.calls == ["refresh"]
    assert harness.opened == []
    assert harness.file_menu.lf.message_dialogs == []
    assert callable(harness.controller._after_service)
    scene, checked = _sync_scene_and_checked(harness.sync_calls)
    assert scene is None
    assert checked is False
    assert snapshot["links"]["project-id"] is link


def test_file_menu_linked_checked_missing_scene_opens_publish_again_review(
    monkeypatch, tmp_path
):
    link = {"sceneId": "scene-id", "commitUuid": "commit-id"}
    snapshot = {
        "links": {"project-id": link},
        "scenes": [{"id": "other-scene", "title": "Other"}],
        "jobs": [],
        "checkedAt": 1,
        "established": True,
    }
    harness = _linked_file_publish_harness(monkeypatch, tmp_path, snapshot)
    harness.invoke()

    assert harness.calls == []
    assert harness.file_menu.lf.message_dialogs == []
    assert len(harness.opened) == 1
    review = harness.opened[0]
    assert review["action"] == "publish"
    assert review["publish_new"] is True
    assert review["scene"] is None
    assert review["asset"]["id"] == "project-id"
    assert review["expected_project_path"] == str(harness.project.resolve())
    scene, checked = _sync_scene_and_checked(harness.sync_calls)
    assert scene is None
    assert checked is True
    assert snapshot["links"]["project-id"] is link


def test_file_menu_linked_missing_scene_after_refresh_can_publish_again(
    monkeypatch, tmp_path
):
    link = {"sceneId": "scene-id", "commitUuid": "commit-id"}
    snapshot = {
        "links": {"project-id": link},
        "scenes": [{"id": "other-scene", "title": "Other"}],
        "jobs": [],
    }
    harness = _linked_file_publish_harness(monkeypatch, tmp_path, snapshot)
    harness.invoke()
    assert harness.calls == ["refresh"]
    assert harness.opened == []

    snapshot["checkedAt"] = 1
    snapshot["established"] = True
    harness.controller._after_service()

    assert len(harness.opened) == 1
    review = harness.opened[0]
    assert review["action"] == "publish"
    assert review["publish_new"] is True
    assert review["scene"] is None
    assert harness.file_menu.lf.message_dialogs == []
    scene, checked = _sync_scene_and_checked(harness.sync_calls)
    assert scene is None
    assert checked is True
    assert snapshot["links"]["project-id"] is link


def test_file_menu_linked_missing_scene_does_not_publish_when_listing_stays_unchecked(
    monkeypatch, tmp_path
):
    snapshot = {
        "links": {"project-id": {"sceneId": "scene-id", "commitUuid": "commit-id"}},
        "scenes": [],
        "jobs": [],
    }
    harness = _linked_file_publish_harness(monkeypatch, tmp_path, snapshot)
    harness.invoke()
    harness.controller._after_service()

    assert harness.opened == []
    assert harness.file_menu.lf.message_dialogs == []
    assert harness.calls == ["refresh"]
    scene, checked = _sync_scene_and_checked(harness.sync_calls)
    assert scene is None
    assert checked is False


def test_file_menu_linked_remote_deleted_marker_opens_publish_again_review(
    monkeypatch, tmp_path
):
    link = {"sceneId": "scene-id", "commitUuid": "old", "remoteDeleted": True}
    snapshot = {
        "links": {"project-id": link},
        "scenes": [],
        "jobs": [],
    }
    harness = _linked_file_publish_harness(monkeypatch, tmp_path, snapshot)
    harness.invoke()

    assert harness.calls == []
    assert harness.file_menu.lf.message_dialogs == []
    assert len(harness.opened) == 1
    review = harness.opened[0]
    assert review["action"] == "publish"
    assert review["publish_new"] is True
    assert review["scene"] is None
    assert snapshot["links"]["project-id"] is link


def test_file_menu_linked_alive_scene_opens_update_review(monkeypatch, tmp_path):
    scene = {
        "id": "scene-id",
        "title": "Project",
        "contentRevision": "c1",
        "metadataRevision": "m1",
        "status": "ready",
    }
    link = {
        "sceneId": "scene-id",
        "commitUuid": "old-commit",
        "contentRevision": "c1",
        "metadataRevision": "m1",
    }
    snapshot = {
        "links": {"project-id": link},
        "scenes": [scene],
        "jobs": [],
        "checkedAt": 1,
        "established": True,
    }
    harness = _linked_file_publish_harness(monkeypatch, tmp_path, snapshot)
    harness.invoke()

    assert harness.calls == []
    assert harness.file_menu.lf.message_dialogs == []
    assert len(harness.opened) == 1
    review = harness.opened[0]
    assert review["action"] == "update"
    assert review["publish_new"] is False
    assert review["scene"] is scene
    matched, checked = _sync_scene_and_checked(harness.sync_calls)
    assert matched is scene
    assert checked is True


def test_file_menu_publish_is_disabled_for_unsaved_project(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)
    file_menu.lf.project_has_path = lambda: False

    item = next(
        item for item in file_menu.FileMenu().menu_items()
        if item.get("label") == "tr:menu.file.publish_to_gallery"
    )

    assert item["enabled"] is False
    item["callback"]()
    assert file_menu.lf.message_dialogs


def test_file_menu_publish_rejects_missing_saved_project(monkeypatch, tmp_path):
    missing = tmp_path / "missing.licht"
    file_menu = _load_file_menu(monkeypatch)
    file_menu.lf.project_has_path = lambda: True
    file_menu.lf.project_poll_write = lambda: {"path": str(missing)}
    opened = []
    gallery_panel = ModuleType("lfs_plugins.gallery_file_panel")
    gallery_panel.open_gallery_file_panel = lambda **review: opened.append(review)
    monkeypatch.setitem(sys.modules, "lfs_plugins.gallery_file_panel", gallery_panel)

    item = next(
        item for item in file_menu.FileMenu().menu_items()
        if item.get("label") == "tr:menu.file.publish_to_gallery"
    )
    item["callback"]()

    assert opened == []
    assert file_menu.lf.message_dialogs
    assert "missing.licht" in file_menu.lf.message_dialogs[-1][1]


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


def test_embed_dataset_operator_requires_external_incomplete_dataset(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)
    assert file_menu.EmbedDatasetOperator.poll(None) is False
    file_menu.lf.project_can_embed_dataset = lambda: True
    assert file_menu.EmbedDatasetOperator.poll(None) is True
    assert file_menu.EmbedDatasetOperator().execute(None) == {"FINISHED"}
    assert file_menu.lf.embed_calls == [True]


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


def test_new_project_opens_unsaved_workspace_without_create_dialog(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)
    dialogs = []
    monkeypatch.setattr(import_module("lfs_plugins.import_panels"),
                        "open_new_project_panel", dialogs.append)

    assert file_menu.NewProjectOperator().execute(None) == {"FINISHED"}

    assert file_menu.lf.new_project_calls == [(True, False)]
    assert file_menu.lf.project_create_calls == []
    assert file_menu.lf.confirm_dialogs == []
    assert dialogs == []


@pytest.mark.parametrize("has_path", [False, True])
@pytest.mark.parametrize("choice", ["save", "discard", "cancel", "save_failed"])
def test_new_project_protects_unsaved_work(monkeypatch, has_path, choice):
    file_menu = _load_file_menu(monkeypatch)
    file_menu.lf.project_is_dirty = lambda: True
    file_menu.lf.project_has_path = lambda: has_path
    saves = []

    def save(*args, **kwargs):
        saves.append((args, kwargs))
        assert file_menu.lf.new_project_calls == []
        if choice == "save_failed":
            return False
        file_menu.lf.project_has_path = lambda: True
        return True

    file_menu.lf.project_save = save
    file_menu.lf.project_save_as = save
    file_menu.NewProjectOperator().execute(None)
    assert file_menu.lf.new_project_calls == []
    assert len(file_menu.lf.confirm_dialogs) == 1
    title, _message, buttons, callback = file_menu.lf.confirm_dialogs[0]
    save_label = "tr:common.save" if has_path else "tr:menu.file.save_project_as"
    assert title == "tr:menu.file.new_project"
    assert buttons == [save_label, "tr:unsaved_work.continue_without_saving", "tr:common.cancel"]
    callback({"save": save_label, "save_failed": save_label,
              "discard": buttons[1], "cancel": buttons[2]}[choice])
    assert file_menu.lf.new_project_calls == ([(True, False)] if choice in ("save", "discard") else [])
    assert len(saves) == int(choice in ("save", "save_failed"))
    if saves:
        assert saves[0] == ((() if has_path else ("",)), {"wait": True})
    assert file_menu.lf.project_create_calls == []


@pytest.mark.parametrize("stop", [False, True])
def test_new_project_asks_before_stopping_training(monkeypatch, stop):
    file_menu = _load_file_menu(monkeypatch)
    file_menu.lf.is_training_active = lambda: True
    file_menu.NewProjectOperator().execute(None)
    assert file_menu.lf.new_project_calls == []
    title, _message, buttons, callback = file_menu.lf.confirm_dialogs[0]
    assert title == "tr:project_switch.stop_training_title"
    assert buttons == ["tr:common.yes", "tr:common.no"]
    callback(buttons[0 if stop else 1])
    assert file_menu.lf.new_project_calls == ([(True, True)] if stop else [])
    assert file_menu.lf.project_create_calls == []


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


def test_create_path_confirmations_preserve_overwrite_authorization(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)

    file_menu._show_project_switch_confirmation(
        True, "", False, "/tmp/existing.licht", True
    )
    assert file_menu.lf.project_create_calls == [
        ("/tmp/existing.licht", True, False, True)
    ]

    file_menu._show_stop_training_confirmation(
        True, "", True, False, "/tmp/existing.licht", True
    )
    _title, _message, _buttons, stop_callback = file_menu.lf.confirm_dialogs[0]
    stop_callback("tr:common.yes")
    assert file_menu.lf.project_create_calls[-1] == (
        "/tmp/existing.licht",
        True,
        True,
        True,
    )


def test_drag_open_confirmation_preserves_asset_manager(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)

    file_menu._show_project_switch_confirmation(
        False, "/tmp/dragged.licht", True
    )

    assert file_menu.lf.project_open_calls == [
        ("/tmp/dragged.licht", True)
    ]
    assert file_menu.lf.project_open_keep_asset_manager == [True]


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


def test_splat_picker_imports_ssog_as_splat(monkeypatch):
    file_menu = _load_file_menu(monkeypatch)
    selected = "/tmp/garden.ssog"
    file_menu.lf.ui.open_ply_file_dialog = lambda _default: selected
    assert file_menu.ImportPlyOperator().execute(None) == {"FINISHED"}
    assert file_menu.lf.load_file_calls == [((selected,), {"is_dataset": False})]


def test_menu_bar_transfer_operator_shows_overlay(monkeypatch):
    import ast
    file_menu = _load_file_menu(monkeypatch)
    calls = []
    overlays = ModuleType('lfs_plugins.overlays')
    overlays.show_gallery_transfers = lambda: calls.append('overlay')
    monkeypatch.setitem(sys.modules, 'lfs_plugins.overlays', overlays)
    monkeypatch.setattr(file_menu.lf.ui, 'set_panel_enabled', lambda panel_id, enabled: calls.append((panel_id, enabled)), raising=False)
    path = PROJECT_ROOT / 'src/python/lfs_plugins/help_menu.py'
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GalleryTransfersOperator')
    scope = {'Operator': object, '__package__': 'lfs_plugins', '__name__': 'lfs_plugins.help_menu'}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    assert scope['GalleryTransfersOperator'].description == 'Show gallery transfers'
    assert scope['GalleryTransfersOperator']().execute(None) == {'FINISHED'}
    assert calls == ['overlay']
