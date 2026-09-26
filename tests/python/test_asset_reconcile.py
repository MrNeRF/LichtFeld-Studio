# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""P2 library reconciliation and journal-join contracts."""

import json
import os
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from lfs_plugins.asset_index import (
    AssetIndex,
    AssetObservation,
    display_name,
    fix_action_for_health,
    last_known_gallery_label,
    previous_scene_for,
)


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


def _observe(index, observations, **kwargs):
    inspections = {str(Path(item.path).resolve()): item.inspection for item in observations}
    index._inspect_path = lambda path: inspections[str(Path(path).resolve())]
    return index.reconcile_observations(observations, **kwargs)


def test_reconcile_is_order_independent_and_keeps_replacement_reference(tmp_path):
    p = tmp_path / "p.licht"
    q = tmp_path / "q.licht"
    p.write_bytes(b"old-file!!")
    q.write_bytes(b"new-file!!")
    old_id, new_id = str(uuid.uuid4()), str(uuid.uuid4())

    def build(order):
        index = AssetIndex(tmp_path / f"{order}-library.json", tmp_path)
        index.load()
        _observe(index,
            [AssetObservation(str(p), "default", _inspection(old_id, "old"))],
            folder_ids=["default"],
        )
        observations = [
            AssetObservation(str(p), "default", _inspection(new_id, "new")),
            AssetObservation(str(q), "default", _inspection(old_id, "old")),
        ]
        if order == "q-first":
            observations.reverse()
        _observe(index, observations, folder_ids=["default"])
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
    _observe(index,
        [AssetObservation(str(path), "default", _inspection(old_id, "old"))],
        folder_ids=["default"],
    )
    original_stat = path.stat()
    path.write_bytes(b"abcdefghij")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    _observe(index,
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


def _copies_fixture(tmp_path):
    first, second = tmp_path / "a.licht", tmp_path / "b.licht"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    project_id = str(uuid.uuid4())
    observations = [
        AssetObservation(str(first), "default", _inspection(project_id, "one")),
        AssetObservation(str(second), "default", _inspection(project_id, "two")),
    ]
    index = AssetIndex(tmp_path / "library.json", tmp_path)
    index.load()
    return index, observations, project_id


def _stat(path):
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _copy_entries(index, project_id):
    return [entry for entry in index.list_projects() if entry.id != project_id]


def test_files_sharing_a_project_uuid_get_their_own_entries(tmp_path):
    index, observations, project_id = _copies_fixture(tmp_path)

    result = _observe(index, observations, folder_ids=["default"])
    _observe(index, observations, folder_ids=["default"])

    primary = index.get_asset(project_id)
    [copy] = _copy_entries(index, project_id)
    assert (result["added"], result["copies"]) == (2, 1)
    assert (primary.path, primary.commit_uuid, primary.status) == (observations[0].path, "one", "AVAILABLE")
    assert (copy.path, copy.commit_uuid, copy.status) == (observations[1].path, "two", "AVAILABLE")
    assert copy.project_uuid == project_id
    assert copy.to_dict()["copy_of"] == project_id
    assert primary.to_dict()["copy_of"] == ""
    assert len(index.list_projects()) == 2


def test_a_first_scan_treats_the_oldest_file_as_the_original(tmp_path):
    index, observations, project_id = _copies_fixture(tmp_path)
    older, newer = observations[1], observations[0]
    older = AssetObservation(older.path, "default", older.inspection,
                             stat_identity={**_stat(older.path), "st_ctime_ns": 1})
    newer = AssetObservation(newer.path, "default", newer.inspection,
                             stat_identity={**_stat(newer.path), "st_ctime_ns": 2})

    _observe(index, [newer, older], folder_ids=["default"])

    assert index.get_asset(project_id).path == older.path


def test_a_copy_found_in_a_later_scan_batch_does_not_move_the_original(tmp_path):
    index, observations, project_id = _copies_fixture(tmp_path)

    _observe(index, observations[:1], folder_ids=["default"])
    _observe(index, observations[1:], folder_ids=["default"])

    assert index.get_asset(project_id).path == observations[0].path
    assert [copy.path for copy in _copy_entries(index, project_id)] == [observations[1].path]


def test_copies_survive_a_catalog_reload(tmp_path):
    index, observations, project_id = _copies_fixture(tmp_path)
    _observe(index, observations, folder_ids=["default"])
    [copy] = _copy_entries(index, project_id)

    reloaded = AssetIndex(tmp_path / "library.json", tmp_path)
    assert reloaded.load()

    assert reloaded.get_asset(project_id).path == observations[0].path
    restored = reloaded.get_asset(copy.id)
    assert (restored.path, restored.project_uuid) == (observations[1].path, project_id)
    assert restored.to_dict()["copy_of"] == project_id


def test_a_copy_takes_over_when_the_original_file_is_gone(tmp_path):
    index, observations, project_id = _copies_fixture(tmp_path)
    _observe(index, observations, folder_ids=["default"])
    primary = index.get_asset(project_id)
    primary.name, primary.name_origin, primary.pinned = "Mine", "user", True
    primary.previous_project_uuid = "replaced-project"
    Path(observations[0].path).unlink()

    _observe(index, observations[1:], folder_ids=["default"])

    [entry] = index.list_projects()
    assert entry.id == project_id
    assert entry.path == observations[1].path
    assert (entry.name, entry.name_origin, entry.pinned) == ("Mine", "user", True)
    assert entry.previous_project_uuid == "replaced-project"
    assert entry.to_dict()["copy_of"] == ""


def test_removing_the_original_entry_promotes_its_copy(tmp_path):
    index, observations, project_id = _copies_fixture(tmp_path)
    _observe(index, observations, folder_ids=["default"])

    assert index.delete_asset(project_id)

    [entry] = index.list_projects()
    assert (entry.id, entry.path) == (project_id, observations[1].path)


def test_a_moved_original_keeps_its_entry(tmp_path):
    index, observations, project_id = _copies_fixture(tmp_path)
    _observe(index, observations[:1], folder_ids=["default"])
    moved = tmp_path / "moved.licht"
    Path(observations[0].path).rename(moved)

    _observe(index, [AssetObservation(str(moved), "default", observations[0].inspection)],
             folder_ids=["default"])

    [entry] = index.list_projects()
    assert (entry.id, entry.path) == (project_id, str(moved))


def test_gallery_publishing_is_blocked_for_copies():
    from lfs_plugins.gallery_actions import gallery_eligibility

    assert gallery_eligibility({"copy_of": "project"}, {})["reasons"][0] == "copy"
    assert "copy" not in gallery_eligibility({"copy_of": ""}, {})["reasons"]


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


@pytest.mark.parametrize(
    ("name", "name_origin", "expected"),
    [
        ("Named by user", "user", "Named by user"),
        ("Legacy catalog name", "", "Legacy catalog name"),
        ("Stem", "stem", "project"),
    ],
)
def test_display_name_prefers_explicit_name_then_filename(
    tmp_path, name, name_origin, expected
):
    project = SimpleNamespace(
        name=name,
        name_origin=name_origin,
        path=str(tmp_path / "assets" / "project.licht"),
    )

    assert display_name(project) == expected


def test_health_fixes_cover_panel_contract():
    assert fix_action_for_health("MISSING") == "locate"
    assert fix_action_for_health("UNREADABLE") == "verify"
    assert fix_action_for_health("REPAIR_ONLY") == "repair"
    assert fix_action_for_health("UNSUPPORTED_NEWER") == "update"


def test_project_file_does_not_inherit_parent_directory_as_display_name(tmp_path):
    project = SimpleNamespace(
        name="project",
        name_origin="stem",
        path=str(tmp_path / "assets" / "project.licht"),
    )

    assert display_name(project) == "project"


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
