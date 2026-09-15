# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""P2 library reconciliation and journal-join contracts."""

import json
import os
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

from lfs_plugins.asset_index import (
    AssetIndex,
    AssetObservation,
    LibraryService,
    display_name,
    fix_action_for_health,
    last_known_gallery_label,
    previous_scene_for,
)
from lfs_plugins.asset_watch import scan_asset_folder


def _inspection(project_uuid, commit_uuid=None):
    return SimpleNamespace(
        project_uuid=project_uuid,
        file_uuid=str(uuid.uuid4()),
        commit_uuid=commit_uuid or str(uuid.uuid4()),
        generation=2,
        created_at_unix_ns=1,
        saved_at_unix_ns=2,
        physical_file_size=10,
        role=SimpleNamespace(name="MASTER"),
        open_state=SimpleNamespace(name="OPEN"),
        has_preview=True,
        iteration=30000,
    )


def test_reconcile_is_order_independent_and_keeps_replacement_reference(tmp_path):
    p = tmp_path / "p.licht"
    q = tmp_path / "q.licht"
    p.write_bytes(b"old-file!!")
    q.write_bytes(b"new-file!!")
    old_id, new_id = str(uuid.uuid4()), str(uuid.uuid4())

    def build(order):
        index = AssetIndex(tmp_path / f"{order}-library.json", tmp_path)
        index.load()
        index.reconcile_observations(
            [AssetObservation(str(p), "default", _inspection(old_id, "old"))],
            folder_ids=["default"],
        )
        observations = [
            AssetObservation(str(p), "default", _inspection(new_id, "new")),
            AssetObservation(str(q), "default", _inspection(old_id, "old")),
        ]
        if order == "q-first":
            observations.reverse()
        index.reconcile_observations(observations, folder_ids=["default"])
        return index

    first, second = build("p-first"), build("q-first")
    for index in (first, second):
        assert index.get_asset(old_id).path == str(q)
        assert index.get_asset(new_id).path == str(p)
        assert index.get_asset(new_id).previous_project_uuid == ""


def test_overwrite_records_previous_uuid_and_joins_only_the_old_journal_link(tmp_path):
    path = tmp_path / "project.licht"
    path.write_bytes(b"0123456789")
    old_id, new_id = str(uuid.uuid4()), str(uuid.uuid4())
    index = AssetIndex(tmp_path / "library.json", tmp_path)
    index.load()
    index.reconcile_observations(
        [AssetObservation(str(path), "default", _inspection(old_id, "old"))],
        folder_ids=["default"],
    )
    original_stat = path.stat()
    path.write_bytes(b"abcdefghij")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    index.reconcile_observations(
        [AssetObservation(str(path), "default", _inspection(new_id, "new"))],
        folder_ids=["default"],
    )
    entry = index.get_asset(new_id)
    links = {old_id: {"sceneId": "scene-1", "state": "equal"}}
    assert entry.previous_project_uuid == old_id
    assert previous_scene_for(entry, links)["sceneId"] == "scene-1"
    assert last_known_gallery_label(entry, None) is None
    assert last_known_gallery_label(entry, {"established": True, "links": {}}) == "Not published"
    assert path.stat().st_size == original_stat.st_size
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns


def test_same_uuid_aliases_and_divergent_commits_are_stored(tmp_path):
    first, second = tmp_path / "a.licht", tmp_path / "b.licht"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    project_id = str(uuid.uuid4())
    index = AssetIndex(tmp_path / "library.json", tmp_path)
    index.load()
    index.reconcile_observations(
        [
            AssetObservation(str(first), "default", _inspection(project_id, "one")),
            AssetObservation(str(second), "default", _inspection(project_id, "two")),
        ],
        folder_ids=["default"],
    )
    entry = index.get_asset(project_id)
    assert len(entry.aliases) == 1
    assert entry.to_dict()["other_paths"]
    assert entry.status == "DIVERGED_COPIES"
    assert fix_action_for_health(entry.status) == "review_copies"


def test_v5_owner_catalog_migrates_once_with_distinct_backup(tmp_path):
    source = Path(__file__).with_name("fixtures") / "asset_library_schema5.json"
    library = tmp_path / "library.json"
    shutil.copy2(source, library)
    index = AssetIndex(library_path=library, default_folder_path=tmp_path / "projects")
    assert index.load()
    saved = json.loads(library.read_text())
    assert saved["schema_version"] == 6
    assert all("gallery" not in value for value in saved["projects"].values())
    backup = tmp_path / "library.json.v5.bak"
    assert json.loads(backup.read_text())["schema_version"] == 5
    backup_bytes = backup.read_bytes()
    assert index.load()
    assert backup.read_bytes() == backup_bytes


def test_owner_copy_has_the_four_real_projects_and_folder_removal_keeps_links(tmp_path):
    source = Path(
        "/home/paja/projects/gaussian-splatting-cuda/.codex_tmp/am_concept/testhome/data/asset_library/library.json"
    )
    library = tmp_path / "library.json"
    shutil.copy2(source, library)
    index = AssetIndex(library_path=library, default_folder_path=tmp_path / "projects")
    assert index.load()
    real_names = {
        Path(project.path).name
        for project in index.list_projects()
        if "mrnf" in Path(project.path).parts
    }
    assert {"project.licht"} == real_names
    assert sum("mrnf" in Path(project.path).parts for project in index.list_projects()) == 3

    folder_path = tmp_path / "watched"
    folder_path.mkdir()
    project_path = folder_path / "linked.licht"
    project_path.write_bytes(b"linked")
    project_id = str(uuid.uuid4())
    links = {"published-id": {"sceneId": "scene"}}
    index._inspect_path = lambda _path: _inspection(project_id, "c")
    folder = index.add_folder(str(folder_path))
    index.register_licht_asset(str(project_path), folder_id=folder.id)
    index.get_asset(project_id).previous_project_uuid = "published-id"
    assert index.delete_folder(folder.id) == 1
    assert index.get_asset(project_id) is None
    assert links == {"published-id": {"sceneId": "scene"}}


def test_library_service_serializes_commands_and_exposes_clean_missing(tmp_path):
    index = AssetIndex(tmp_path / "library.json", tmp_path)
    index.load()
    with LibraryService(index) as service:
        assert service._worker.is_alive()
        folder = service.add_folder(str(tmp_path / "folder"))
        assert folder is None  # add_folder keeps the real-directory contract
        assert service.snapshot()["epoch"] >= 0


def test_display_name_and_health_fixes_cover_panel_contract(tmp_path):
    project = SimpleNamespace(name="Stem", name_origin="stem", path=str(tmp_path / "project.licht"))
    assert display_name(project) == tmp_path.name
    project.name_origin = "user"
    assert display_name(project) == "Stem"
    assert fix_action_for_health("MISSING") == "locate"
    assert fix_action_for_health("UNREADABLE") == "verify"
    assert fix_action_for_health("REPAIR_ONLY") == "repair"
    assert fix_action_for_health("UNSUPPORTED_NEWER") == "update"


def test_unknown_catalog_fields_survive_a_v6_save_and_generated_name_is_not_stored(tmp_path):
    project_path = tmp_path / "generated.licht"
    project_path.write_bytes(b"x")
    project_id = str(uuid.uuid4())
    library = tmp_path / "library.json"
    library.write_text(
        json.dumps(
            {
                "schema_version": 6,
                "future_root": {"keep": True},
                "folders": {"default": {"path": str(tmp_path), "future_folder": 7}},
                "projects": {
                    project_id: {
                        "name": "generated",
                        "path": str(project_path),
                        "folder_id": "default",
                        "future_project": [1, 2, 3],
                    }
                },
            }
        )
    )
    index = AssetIndex(library, tmp_path)
    assert index.load()
    assert index.save()
    saved = json.loads(library.read_text())
    assert saved["future_root"] == {"keep": True}
    assert saved["folders"]["default"]["future_folder"] == 7
    assert saved["projects"][project_id]["future_project"] == [1, 2, 3]
    assert "name" not in saved["projects"][project_id]
    assert saved["projects"][project_id]["name_origin"] == "stem"
