# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for .licht discovery in real Asset Manager folders."""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from lfs_plugins.asset_index import AssetIndex
from lfs_plugins import asset_watch
from lfs_plugins.asset_watch import (
    AssetFolderScanProgress,
    discover_licht_projects,
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


def test_discovery_recurses_and_returns_only_licht_files(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "first.licht"
    second = nested / "SECOND.LICHT"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    (tmp_path / "directory.licht").mkdir()

    discovered = discover_licht_projects(str(tmp_path))

    assert discovered == [str(first.resolve()), str(second.resolve())]


def test_scan_registers_projects_from_one_real_folder(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "first.licht"
    second = nested / "second.licht"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    class _Index:
        def __init__(self):
            self.paths = []

        def register_licht_asset(self, path, *, folder_id, adopt_existing, save):
            assert adopt_existing is False
            assert save is False
            self.paths.append((path, folder_id))
            return SimpleNamespace(id=path), len(self.paths) == 1

        def save(self):
            return True

    index = _Index()
    result = scan_asset_folder(
        index,
        "projects",
        str(tmp_path),
    )

    assert index.paths == [
        (str(first.resolve()), "projects"),
        (str(second.resolve()), "projects"),
    ]
    assert result.discovered == 2
    assert result.added == 1
    assert result.already_cataloged == 1
    assert result.failed == 0


def test_real_folder_mapping_is_normalized_and_persisted(tmp_path: Path):
    default = tmp_path / "default"
    selected = tmp_path / "selected"
    default.mkdir()
    selected.mkdir()
    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path, default_folder_path=default)
    index.ensure_default_catalog()

    folder = index.add_folder(str(selected))
    assert folder is not None
    assert folder.path == str(selected.resolve())

    reloaded = AssetIndex(library_path=library_path, default_folder_path=default)
    assert reloaded.load() is True
    assert reloaded.folders[folder.id]["path"] == str(selected.resolve())


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


def test_missing_asset_folder_discovers_nothing(tmp_path: Path):
    assert discover_licht_projects(str(tmp_path / "missing")) == []


def test_discovery_does_not_find_licht_directly_inside_hidden_or_pruned_directory_names(
    tmp_path: Path,
):
    visible = tmp_path / "visible.licht"
    visible.write_bytes(b"visible")
    hidden = tmp_path / ".cache"
    hidden.mkdir()
    (hidden / "nested.licht").write_bytes(b"hidden")
    for directory_name in (
        "node_modules",
        "CMakeFiles",
        "vcpkg_installed",
        "site-packages",
        "__pycache__",
    ):
        directory = tmp_path / directory_name
        directory.mkdir()
        (directory / "hidden.licht").write_bytes(b"hidden")

    assert discover_licht_projects(str(tmp_path)) == [str(visible.resolve())]


def test_discovery_reports_progress_counters(tmp_path: Path):
    nested = tmp_path / "keep"
    nested.mkdir()
    project = nested / "project.licht"
    project.write_bytes(b"project")
    progress = AssetFolderScanProgress()

    discovered = discover_licht_projects(str(tmp_path), progress=progress)

    directories, projects, root = progress.snapshot()
    assert discovered == [str(project.resolve())]
    assert projects == 1
    assert directories >= 2
    assert Path(root).resolve() == tmp_path.resolve()


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
    index.ensure_default_catalog()
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


def test_discovery_prunes_dataset_directories(tmp_path: Path):
    visible = tmp_path / "visible.licht"
    visible.write_bytes(b"visible")
    for directory_name in (
        "sparse",
        "dense",
        "masks",
        "stereo",
        "depth",
        "images",
        "__pycache__",
    ):
        directory = tmp_path / directory_name
        directory.mkdir()
        (directory / "hidden.licht").write_bytes(b"hidden")

    assert discover_licht_projects(str(tmp_path)) == [str(visible.resolve())]


def test_scan_skips_inspection_for_unchanged_cataloged_path(tmp_path: Path):
    project = tmp_path / "project.licht"
    project.write_bytes(b"container")

    class _Index:
        def find_asset_by_path(self, path):
            assert path == str(project.resolve())
            return SimpleNamespace(status="AVAILABLE")

        def register_licht_asset(self, *_args, **_kwargs):
            raise AssertionError("unchanged cataloged path must not be inspected")

        def save(self):
            raise AssertionError("unchanged catalog must not be saved")

    result = scan_asset_folder(_Index(), "projects", str(tmp_path))

    assert result.discovered == 1
    assert result.already_cataloged == 1
    assert result.added == 0


def test_cancelled_scan_rolls_back_discovered_projects(tmp_path: Path):
    first = tmp_path / "first.licht"
    second = tmp_path / "second.licht"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    cancel_event = threading.Event()

    class _Index:
        def __init__(self):
            self.paths = []

        def _snapshot_state(self):
            return list(self.paths)

        def _restore_state(self, snapshot):
            self.paths = snapshot

        def register_licht_asset(self, path, **_kwargs):
            self.paths.append(path)
            cancel_event.set()
            return SimpleNamespace(id=path), True

        def save(self):
            raise AssertionError("cancelled scan must not save")

    index = _Index()
    result = scan_asset_folder(
        index,
        "projects",
        str(tmp_path),
        cancel_event,
    )

    assert result.cancelled is True
    assert index.paths == []


def test_discovery_warns_once_when_folder_has_more_than_10000_directories(
    monkeypatch, tmp_path: Path, caplog
):
    def fake_walk(_root, topdown=True, onerror=None, followlinks=False):
        for index in range(10001):
            yield str(tmp_path / f"dir-{index}"), [], []

    monkeypatch.setattr(os, "walk", fake_walk)

    with caplog.at_level(logging.WARNING, logger="lfs_plugins.asset_watch"):
        discovered = discover_licht_projects(str(tmp_path))

    assert discovered == []
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "very large (>10000 directories)" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert str(tmp_path) in warnings[0].getMessage()


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _write_licht_tree(tmp_path: Path, count: int):
    inspections = {}
    paths = []
    for index in range(count):
        path = tmp_path / f"p{index:02d}.licht"
        path.write_bytes(b"container")
        paths.append(path)
        inspections[path.name] = _inspection(str(uuid.uuid4()))
    return paths, inspections


def test_scan_streams_first_batch_before_walk_finishes(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(asset_watch, "SCAN_BATCH_SIZE", 2)
    monkeypatch.setattr(asset_watch, "SCAN_BATCH_INTERVAL_S", 60.0)
    paths, inspections = _write_licht_tree(tmp_path, 8)
    monkeypatch.setattr(
        AssetIndex,
        "_inspect_path",
        staticmethod(lambda path: inspections[Path(path).name]),
    )
    walk_finished = threading.Event()
    past_first_batch = threading.Event()

    def fake_walk(_root, topdown=True, onerror=None, followlinks=False):
        names = [path.name for path in paths]
        for index, name in enumerate(names):
            yield str(tmp_path), [], [name]
            if index == 1:
                past_first_batch.set()
            time.sleep(0.03)
        walk_finished.set()

    monkeypatch.setattr(os, "walk", fake_walk)
    index = AssetIndex(
        library_path=tmp_path / "library.json",
        default_folder_path=tmp_path,
    )
    index.ensure_default_catalog()
    result_holder = {}

    def run_scan():
        result_holder["result"] = scan_asset_folder(
            index, "default", str(tmp_path)
        )

    thread = threading.Thread(target=run_scan)
    thread.start()
    assert past_first_batch.wait(timeout=2.0)
    assert _wait_until(lambda: len(index.list_projects()) >= 2)
    assert walk_finished.is_set() is False
    thread.join(timeout=2.0)
    assert thread.is_alive() is False
    assert len(index.list_projects()) == 8
    assert result_holder["result"].added == 8
    assert result_holder["result"].cancelled is False


def test_cancelled_scan_keeps_committed_batches(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(asset_watch, "SCAN_BATCH_SIZE", 2)
    monkeypatch.setattr(asset_watch, "SCAN_BATCH_INTERVAL_S", 60.0)
    paths, inspections = _write_licht_tree(tmp_path, 6)
    monkeypatch.setattr(
        AssetIndex,
        "_inspect_path",
        staticmethod(lambda path: inspections[Path(path).name]),
    )
    cancel_event = threading.Event()

    def fake_walk(_root, topdown=True, onerror=None, followlinks=False):
        names = [path.name for path in paths]
        for index, name in enumerate(names):
            yield str(tmp_path), [], [name]
            if index == 3:
                cancel_event.wait(timeout=2.0)

    monkeypatch.setattr(os, "walk", fake_walk)
    index = AssetIndex(
        library_path=tmp_path / "library.json",
        default_folder_path=tmp_path,
    )
    index.ensure_default_catalog()

    def cancel_after_two_batches():
        if _wait_until(lambda: len(index.list_projects()) >= 4):
            cancel_event.set()

    waiter = threading.Thread(target=cancel_after_two_batches)
    waiter.start()
    result = scan_asset_folder(index, "default", str(tmp_path), cancel_event)
    waiter.join(timeout=2.0)

    assert result.cancelled is True
    assert len(index.list_projects()) == 4
    reloaded = AssetIndex(
        library_path=tmp_path / "library.json",
        default_folder_path=tmp_path,
    )
    assert reloaded.load() is True
    assert len(reloaded.list_projects()) == 4


def test_scan_progress_updates_while_batching(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(asset_watch, "SCAN_BATCH_SIZE", 2)
    monkeypatch.setattr(asset_watch, "SCAN_BATCH_INTERVAL_S", 60.0)
    paths, inspections = _write_licht_tree(tmp_path, 4)
    monkeypatch.setattr(
        AssetIndex,
        "_inspect_path",
        staticmethod(lambda path: inspections[Path(path).name]),
    )
    snapshots = []

    def fake_walk(_root, topdown=True, onerror=None, followlinks=False):
        names = [path.name for path in paths]
        for name in names:
            yield str(tmp_path), [], [name]

    monkeypatch.setattr(os, "walk", fake_walk)
    index = AssetIndex(
        library_path=tmp_path / "library.json",
        default_folder_path=tmp_path,
    )
    index.ensure_default_catalog()
    progress = AssetFolderScanProgress()
    original_commit = asset_watch._commit_registration_batch

    def tracked_commit(index_arg, batch, cancel_event):
        result = original_commit(index_arg, batch, cancel_event)
        snapshots.append(progress.snapshot())
        return result

    monkeypatch.setattr(asset_watch, "_commit_registration_batch", tracked_commit)
    result = scan_asset_folder(
        index, "default", str(tmp_path), progress=progress
    )

    assert result.added == 4
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
    assert {project.status for project in index.list_projects()} == {"UNVERIFIED"}

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
