# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for .licht discovery in real Asset Manager folders."""

import json
import os
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from lfs_plugins.asset_index import AssetIndex
from lfs_plugins import asset_watch
from lfs_plugins.asset_watch import (
    AssetFolderScanProgress,
    scan_all_asset_folders,
    scan_asset_folder,
    verify_catalog_projects,
)


def _inspection(project_uuid: str):
    return SimpleNamespace(
        project_uuid=project_uuid,
        file_uuid=str(uuid.uuid4()),
        commit_uuid=str(uuid.uuid4()),
        generation=1,
        created_at_unix_ns=100,
        saved_at_unix_ns=200,
        physical_file_size=1234,
        role=SimpleNamespace(name="MASTER"),
        open_state=SimpleNamespace(name="OPEN"),
        has_preview=False,
    )


def test_real_folder_mapping_is_normalized_and_persisted(tmp_path: Path):
    default = tmp_path / "default"
    selected = tmp_path / "selected"
    default.mkdir()
    selected.mkdir()
    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path, default_folder_path=default)
    index.load()

    folder = index.add_folder(str(selected))
    assert folder is not None
    assert folder.path == str(selected.resolve())

    reloaded = AssetIndex(library_path=library_path, default_folder_path=default)
    assert reloaded.load() is True
    assert reloaded.folders[folder.id]["path"] == str(selected.resolve())


def test_non_recursive_folder_policy_is_persisted_and_reloaded(tmp_path: Path):
    default = tmp_path / "default"
    selected = tmp_path / "selected"
    default.mkdir()
    selected.mkdir()
    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path, default_folder_path=default)
    index.load()

    folder = index.add_folder(str(selected), recursive=False)

    assert folder is not None
    stored = json.loads(library_path.read_text(encoding="utf-8"))
    assert stored["folders"][folder.id]["recursive"] is False
    reloaded = AssetIndex(library_path=library_path, default_folder_path=default)
    assert reloaded.load() is True
    assert reloaded.folders[folder.id]["recursive"] is False


def test_global_scan_respects_non_recursive_folder_policy(
    monkeypatch, tmp_path: Path
):
    default = tmp_path / "default"
    selected = tmp_path / "selected"
    nested = selected / "nested"
    default.mkdir()
    nested.mkdir(parents=True)
    top_level = selected / "top-level.licht"
    nested_project = nested / "nested.licht"
    top_level.write_bytes(b"top")
    nested_project.write_bytes(b"nested")
    inspection = _inspection(str(uuid.uuid4()))
    monkeypatch.setattr(
        AssetIndex,
        "_inspect_path",
        staticmethod(lambda _path: inspection),
    )
    index = AssetIndex(
        library_path=tmp_path / "library.json",
        default_folder_path=default,
    )
    index.load()
    folder = index.add_folder(str(selected), recursive=False)
    assert folder is not None

    reloaded = AssetIndex(
        library_path=tmp_path / "library.json",
        default_folder_path=default,
    )
    assert reloaded.load() is True
    result = scan_all_asset_folders(reloaded)

    assert result.discovered == 1
    assert {project["path"] for project in reloaded.assets.values()} == {
        str(top_level)
    }


def test_global_scan_assigns_new_project_to_most_specific_root(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    project = nested / "project.licht"
    project.write_bytes(b"container")

    class _Index:
        folders = {
            "broad": {"path": str(tmp_path)},
            "specific": {"path": str(nested)},
        }

        def __init__(self):
            self.paths = []

        def register_licht_asset(self, path, *, folder_id, adopt_existing, save):
            self.paths.append((path, folder_id, adopt_existing, save))
            return SimpleNamespace(id=path), True

        def save(self):
            return True

    index = _Index()
    result = scan_all_asset_folders(index)

    assert index.paths == [(str(project), "specific", False, False)]
    assert result.discovered == 1
    assert result.added == 1
    assert result.failed == 0


def _age_directories(*directories: Path) -> None:
    timestamp = time.time() - 10.0
    for directory in directories:
        os.utime(directory, ns=(int(timestamp * 1_000_000_000),) * 2)


@pytest.mark.parametrize(
    ("initial_recursive", "updated_recursive", "initial_names", "updated_names"),
    [
        (False, True, {"top-level.licht"}, {"top-level.licht", "nested.licht"}),
        (True, False, {"top-level.licht", "nested.licht"}, {"top-level.licht"}),
    ],
)
def test_scan_cache_refreshes_when_folder_depth_changes_after_reload(
    monkeypatch,
    tmp_path: Path,
    initial_recursive: bool,
    updated_recursive: bool,
    initial_names: set[str],
    updated_names: set[str],
):
    default = tmp_path / "default"
    selected = tmp_path / "selected"
    nested = selected / "nested"
    default.mkdir()
    nested.mkdir(parents=True)
    top_level = selected / "top-level.licht"
    nested_project = nested / "nested.licht"
    top_level.write_bytes(b"top")
    nested_project.write_bytes(b"nested")
    _age_directories(default, selected, nested)

    inspections = {
        top_level.name: _inspection(str(uuid.uuid4())),
        nested_project.name: _inspection(str(uuid.uuid4())),
    }
    monkeypatch.setattr(
        AssetIndex,
        "_inspect_path",
        staticmethod(lambda path: inspections[Path(path).name]),
    )
    storage = tmp_path / "storage"
    monkeypatch.setenv("LFS_ASSET_MANAGER_DIR", str(storage))
    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path, default_folder_path=default)
    index.load()
    folder = index.add_folder(str(selected), recursive=initial_recursive)
    assert folder is not None

    first = scan_all_asset_folders(index)

    assert first.failed == 0
    assert {Path(project["path"]).name for project in index.assets.values()} == initial_names
    cache_path = storage / asset_watch.SCAN_CACHE_FILENAME
    first_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert first_cache[asset_watch._directory_key(selected)]["recursive"] is initial_recursive

    assert index.add_folder(str(selected), recursive=updated_recursive) is not None
    reloaded = AssetIndex(library_path=library_path, default_folder_path=default)
    assert reloaded.load() is True

    second = scan_all_asset_folders(reloaded)

    assert second.failed == 0
    assert {Path(project["path"]).name for project in reloaded.assets.values()} == updated_names
    second_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert second_cache[asset_watch._directory_key(selected)]["recursive"] is updated_recursive


def test_scan_cache_rejects_entries_without_a_saved_depth_policy(
    monkeypatch,
    tmp_path: Path,
):
    default = tmp_path / "default"
    selected = tmp_path / "selected"
    nested = selected / "nested"
    default.mkdir()
    nested.mkdir(parents=True)
    nested_project = nested / "nested.licht"
    nested_project.write_bytes(b"nested")
    _age_directories(default, selected, nested)

    inspection = _inspection(str(uuid.uuid4()))
    monkeypatch.setattr(
        AssetIndex,
        "_inspect_path",
        staticmethod(lambda _path: inspection),
    )
    storage = tmp_path / "storage"
    storage.mkdir()
    monkeypatch.setenv("LFS_ASSET_MANAGER_DIR", str(storage))
    cache_path = storage / asset_watch.SCAN_CACHE_FILENAME
    cache_path.write_text(
        json.dumps(
            {
                asset_watch._directory_key(selected): {
                    "mtime_ns": selected.stat().st_mtime_ns,
                    "dirs": [],
                    "licht": [],
                }
            }
        ),
        encoding="utf-8",
    )
    index = AssetIndex(
        library_path=tmp_path / "library.json",
        default_folder_path=default,
    )
    index.load()
    assert index.add_folder(str(selected), recursive=True) is not None

    result = scan_all_asset_folders(index)

    assert result.failed == 0
    assert {project["path"] for project in index.assets.values()} == {
        str(nested_project)
    }
    rewritten_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert rewritten_cache[asset_watch._directory_key(selected)]["recursive"] is True


def test_nonrecursive_parent_preserves_projects_owned_by_a_nested_folder(
    monkeypatch,
    tmp_path: Path,
):
    default = tmp_path / "default"
    parent = tmp_path / "parent"
    nested = parent / "nested"
    default.mkdir()
    nested.mkdir(parents=True)
    nested_project = nested / "nested.licht"
    nested_project.write_bytes(b"nested")
    _age_directories(default, parent, nested)

    inspection = _inspection(str(uuid.uuid4()))
    monkeypatch.setattr(
        AssetIndex,
        "_inspect_path",
        staticmethod(lambda _path: inspection),
    )
    monkeypatch.setenv("LFS_ASSET_MANAGER_DIR", str(tmp_path / "storage"))
    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path, default_folder_path=default)
    index.load()
    parent_folder = index.add_folder(str(parent), recursive=True)
    nested_folder = index.add_folder(str(nested), recursive=True)
    assert parent_folder is not None
    assert nested_folder is not None
    assert scan_all_asset_folders(index).failed == 0
    assert index.get_asset(inspection.project_uuid).folder_id == nested_folder.id

    assert index.add_folder(str(parent), recursive=False) is not None
    reloaded = AssetIndex(library_path=library_path, default_folder_path=default)
    assert reloaded.load() is True

    result = scan_all_asset_folders(reloaded)

    assert result.failed == 0
    project = reloaded.get_asset(inspection.project_uuid)
    assert project is not None
    assert project.folder_id == nested_folder.id


def test_cached_scan_reinspects_in_place_overwrite_with_cleared_status(
    monkeypatch, tmp_path: Path
):
    root = tmp_path / "watched"
    root.mkdir()
    project_path = root / "project.licht"
    project_path.write_bytes(b"old")
    old_uuid = str(uuid.uuid4())
    new_uuid = str(uuid.uuid4())
    new_inspection = _inspection(new_uuid)
    storage = tmp_path / "storage"
    old_inspection = _inspection(old_uuid)
    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(lambda _path: old_inspection))
    monkeypatch.setenv("LFS_ASSET_MANAGER_DIR", str(storage))
    index = AssetIndex(
        library_path=tmp_path / "library.json",
        default_folder_path=root,
    )
    index.load()
    project, created = index.register_licht_asset(
        str(project_path), inspection=_inspection(old_uuid)
    )
    assert created is True
    _age_directories(root)
    assert scan_asset_folder(index, "default", str(root)).already_cataloged == 1

    project_path.write_bytes(b"new")
    index._clear_runtime(project, "IDENTITY_MISMATCH", "stale project")
    monkeypatch.setattr(
        AssetIndex, "_inspect_path", staticmethod(lambda _path: new_inspection)
    )

    result = scan_asset_folder(index, "default", str(root))

    assert result.added == 1
    assert old_uuid not in index.assets
    assert new_uuid in index.assets


def test_single_folder_scan_does_not_steal_projects_from_more_specific_folder(
    monkeypatch, tmp_path: Path
):
    parent = tmp_path / "parent"
    nested = parent / "nested"
    nested.mkdir(parents=True)
    parent_project = parent / "parent.licht"
    nested_owned = nested / "owned.licht"
    nested_new = nested / "new.licht"
    parent_project.write_bytes(b"parent")
    nested_owned.write_bytes(b"owned")
    nested_new.write_bytes(b"new")
    inspections = {
        parent_project.name: _inspection(str(uuid.uuid4())),
        nested_owned.name: _inspection(str(uuid.uuid4())),
        nested_new.name: _inspection(str(uuid.uuid4())),
    }
    monkeypatch.setattr(
        AssetIndex,
        "_inspect_path",
        staticmethod(lambda path: inspections[Path(path).name]),
    )

    index = AssetIndex(
        library_path=tmp_path / "library.json",
        default_folder_path=tmp_path / "default",
    )
    index.load()
    nested_folder = index.add_folder(str(nested))
    parent_folder = index.add_folder(str(parent))
    owned, created = index.register_licht_asset(str(nested_owned))
    assert created is True
    assert owned is not None
    assert owned.folder_id == nested_folder.id

    result = scan_asset_folder(index, parent_folder.id, str(parent))

    assert result.cancelled is False
    by_name = {Path(asset["path"]).name: asset for asset in index.assets.values()}
    assert by_name["owned.licht"]["folder_id"] == nested_folder.id
    assert by_name["new.licht"]["folder_id"] == nested_folder.id
    assert by_name["parent.licht"]["folder_id"] == parent_folder.id


def test_scan_reregisters_identity_mismatch_with_cleared_metadata(
    monkeypatch, tmp_path: Path
):
    project = tmp_path / "project.licht"
    project.write_bytes(b"container")
    old_uuid = str(uuid.uuid4())
    new_uuid = str(uuid.uuid4())
    new_inspection = _inspection(new_uuid)
    monkeypatch.setattr(
        AssetIndex,
        "_inspect_path",
        staticmethod(lambda _path: _inspection(old_uuid)),
    )

    index = AssetIndex(
        library_path=tmp_path / "library.json",
        default_folder_path=tmp_path,
    )
    index.load()
    old_project, created = index.register_licht_asset(
        str(project), inspection=_inspection(old_uuid)
    )
    assert created is True
    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(lambda _path: new_inspection))
    index._clear_runtime(old_project, "IDENTITY_MISMATCH", "stale project")
    assert old_project.path_size_bytes == 0
    assert old_project.path_mtime_ns == 0

    result = scan_asset_folder(index, "default", str(tmp_path))

    assert result.added == 1
    assert result.already_cataloged == 0
    assert old_uuid not in index.assets
    assert new_uuid in index.assets
    assert index.assets[new_uuid]["status"] == "AVAILABLE"


def _write_licht_tree(tmp_path: Path, count: int):
    inspections = {}
    paths = []
    for index in range(count):
        path = tmp_path / f"p{index:02d}.licht"
        path.write_bytes(b"container")
        paths.append(path)
        inspections[path.name] = _inspection(str(uuid.uuid4()))
    return paths, inspections


def test_precancelled_scan_keeps_catalog_and_disk_unchanged(monkeypatch, tmp_path: Path):
    (tmp_path / "project.licht").write_bytes(b"container")
    cancel_event = threading.Event()
    index = AssetIndex(
        library_path=tmp_path / "library.json",
        default_folder_path=tmp_path,
    )
    index.load()
    original = index.library_path.read_bytes()
    monkeypatch.setattr(asset_watch.os, "scandir", lambda _root: pytest.fail("Canceled scan enumerated files"))
    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(lambda _path: pytest.fail("Canceled scan inspected a file")))

    cancel_event.set()
    result = scan_asset_folder(index, "default", str(tmp_path), cancel_event)

    assert result.cancelled is True
    assert len(index.list_projects()) == 0
    assert index.library_path.read_bytes() == original
    reloaded = AssetIndex(
        library_path=tmp_path / "library.json",
        default_folder_path=tmp_path,
    )
    assert reloaded.load() is True
    assert len(reloaded.list_projects()) == 0


def test_scan_reports_discovery_progress_before_reconciliation(monkeypatch, tmp_path: Path):
    _, inspections = _write_licht_tree(tmp_path, 4)
    monkeypatch.setattr(
        AssetIndex,
        "_inspect_path",
        staticmethod(lambda path: inspections[Path(path).name]),
    )
    snapshots = []
    before_commit = []
    index = AssetIndex(
        library_path=tmp_path / "library.json",
        default_folder_path=tmp_path,
    )
    index.load()
    progress = AssetFolderScanProgress()
    original_commit = asset_watch._commit_registration_batch

    def tracked_commit(index_arg, batch, cancel_event, **kwargs):
        before_commit.append(progress.snapshot())
        result = original_commit(index_arg, batch, cancel_event, **kwargs)
        snapshots.append(progress.snapshot())
        return result

    monkeypatch.setattr(asset_watch, "_commit_registration_batch", tracked_commit)
    result = scan_asset_folder(
        index, "default", str(tmp_path), progress=progress
    )

    assert result.added == 4
    assert before_commit and before_commit[0][1] == 4
    assert snapshots
    assert snapshots[-1][1] == 4
    directories, projects, root = progress.snapshot()
    assert projects == 4
    assert directories >= 1
    assert Path(root).resolve() == tmp_path.resolve()


def test_verify_catalog_projects_runs_in_batches(monkeypatch, tmp_path: Path):
    paths, inspections = _write_licht_tree(tmp_path, 5)
    monkeypatch.setattr(
        AssetIndex,
        "_inspect_path",
        staticmethod(lambda path: inspections[Path(path).name]),
    )
    library_path = tmp_path / "library.json"
    projects = {}
    for path in paths:
        project_uuid = inspections[path.name].project_uuid
        projects[project_uuid] = {
            "name": path.stem,
            "path": str(path),
            "folder_id": "default",
        }
    library_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "folders": {"default": {"path": str(tmp_path)}},
                "projects": projects,
            }
        ),
        encoding="utf-8",
    )
    index = AssetIndex(library_path=library_path, default_folder_path=tmp_path)
    assert index.load() is True
    assert {project.status for project in index.list_projects()} == {"READING"}

    batch_sizes = []
    original = index.verify_projects_batch

    def tracked(asset_ids):
        batch_sizes.append(len(asset_ids))
        return original(asset_ids)

    index.verify_projects_batch = tracked
    verified = verify_catalog_projects(index, batch_size=2, interval_s=60.0)

    assert verified == 5
    assert batch_sizes == [2, 2, 1]
    assert {project.status for project in index.list_projects()} == {"AVAILABLE"}


def test_verify_catalog_projects_prioritizes_visible_window():
    order = []

    class Index:
        def list_projects(self):
            return [SimpleNamespace(id=str(i)) for i in range(5)]

        def verify_projects_batch(self, ids):
            order.extend(ids)
            return len(ids)

    assert verify_catalog_projects(
        Index(), visible_asset_ids=["3", "1"], batch_size=1, interval_s=0
    ) == 5
    assert order == ["3", "1", "0", "2", "4"]


@pytest.mark.parametrize("scan_all", [False, True])
@pytest.mark.parametrize("remaining", [0, 1])
def test_rescan_persists_removed_files_without_new_projects(tmp_path, monkeypatch, scan_all, remaining):
    root = tmp_path / "watched"
    root.mkdir()
    paths = [root / "first.licht", root / "second.licht"]
    inspections = {path.name: _inspection(str(uuid.uuid4())) for path in paths}
    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(lambda path: inspections[Path(path).name]))
    monkeypatch.setenv("LFS_ASSET_MANAGER_DIR", str(tmp_path / "cache"))
    index = AssetIndex(library_path=tmp_path / "library.json", default_folder_path=root)
    index.load()
    for path in paths:
        path.write_bytes(b"project")
    assert scan_asset_folder(index, "default", str(root)).added == 2
    for path in paths[remaining:]:
        path.unlink()
    result = scan_all_asset_folders(index) if scan_all else scan_asset_folder(index, "default", str(root))
    assert result.failed == 0
    for path in paths[remaining:]:
        assert index.get_asset(inspections[path.name].project_uuid).status == "MISSING"
    reloaded = AssetIndex(library_path=tmp_path / "library.json", default_folder_path=root)
    assert reloaded.load()
    for path in paths[remaining:]:
        assert reloaded.get_asset(inspections[path.name].project_uuid).status == "MISSING"


@pytest.mark.parametrize("scan_all", [False, True])
def test_scan_save_failure_restores_catalog(tmp_path, monkeypatch, scan_all):
    root = tmp_path / "watched"
    root.mkdir()
    path = root / "new.licht"
    path.write_bytes(b"project")
    inspection = _inspection(str(uuid.uuid4()))
    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(lambda path: inspection))
    monkeypatch.setenv("LFS_ASSET_MANAGER_DIR", str(tmp_path / "cache"))
    index = AssetIndex(library_path=tmp_path / "library.json", default_folder_path=root)
    index.load()
    monkeypatch.setattr(index, "save", lambda: False)
    result = scan_all_asset_folders(index) if scan_all else scan_asset_folder(index, "default", str(root))
    assert result.failed > 0 and result.added == 0
    assert index.get_asset(inspection.project_uuid) is None


@pytest.mark.parametrize("scan_all", [False, True])
def test_cancel_during_discovery_does_not_commit_partial_scan(tmp_path, monkeypatch, scan_all):
    root = tmp_path / "watched"
    root.mkdir()
    path = root / "new.licht"
    path.write_bytes(b"project")
    inspection = _inspection(str(uuid.uuid4()))
    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(lambda path: inspection))
    monkeypatch.setenv("LFS_ASSET_MANAGER_DIR", str(tmp_path / "cache"))
    index = AssetIndex(library_path=tmp_path / "library.json", default_folder_path=root)
    index.load()
    cancel = threading.Event()
    def discover(*args, **kwargs):
        yield str(path)
        cancel.set()
    monkeypatch.setattr(asset_watch, "iter_licht_projects", discover)
    result = scan_all_asset_folders(index, cancel) if scan_all else scan_asset_folder(index, "default", str(root), cancel)
    assert result.cancelled
    assert result.added == 0
    assert index.get_asset(inspection.project_uuid) is None


def test_scan_without_folder_roots_does_not_reconcile_unrelated_projects():
    index = SimpleNamespace(
        folders={"legacy": {"path": ""}},
        _inspect_path=lambda path: None,
        reconcile_observations=lambda *args, **kwargs: pytest.fail("No folder was scanned"),
    )
    assert scan_all_asset_folders(index) == asset_watch.AssetFolderScanResult()
