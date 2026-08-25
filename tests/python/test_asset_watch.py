# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for .licht-only watched-directory discovery."""

import os
import threading
from pathlib import Path
from types import SimpleNamespace

from lfs_plugins.asset_index import AssetIndex, DEFAULT_FOLDER_ID
from lfs_plugins.asset_watch import (
    discover_licht_projects,
    scan_all_watch_directories,
    scan_watch_directories,
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


def test_scan_deduplicates_overlapping_roots_before_registration(tmp_path: Path):
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
    result = scan_watch_directories(
        index,
        "projects",
        [str(tmp_path), str(nested)],
    )

    assert index.paths == [
        (str(first.resolve()), "projects"),
        (str(second.resolve()), "projects"),
    ]
    assert result.discovered == 2
    assert result.added == 1
    assert result.already_cataloged == 1
    assert result.failed == 0


def test_watch_directories_are_normalized_deduplicated_and_persisted(tmp_path: Path):
    watched = tmp_path / "watched"
    watched.mkdir()
    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path)
    index.ensure_default_catalog()

    relative = os.path.relpath(watched, Path.cwd())
    assert index.set_watch_dirs(DEFAULT_FOLDER_ID, [relative, str(watched), ""]) is True

    reloaded = AssetIndex(library_path=library_path)
    assert reloaded.load() is True
    assert reloaded.get_watch_dirs(DEFAULT_FOLDER_ID) == [str(watched.resolve())]
    assert reloaded.folders[DEFAULT_FOLDER_ID]["watch_directories"] == [
        str(watched.resolve())
    ]


def test_global_scan_assigns_new_project_to_most_specific_root(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    project = nested / "project.licht"
    project.write_bytes(b"container")

    class _Index:
        folders = {
            "broad": {"watch_directories": [str(tmp_path)]},
            "specific": {"watch_directories": [str(nested)]},
        }

        def __init__(self):
            self.paths = []

        def register_licht_asset(self, path, *, folder_id, adopt_existing, save):
            self.paths.append((path, folder_id, adopt_existing, save))
            return SimpleNamespace(id=path), True

        def save(self):
            return True

    index = _Index()
    result = scan_all_watch_directories(index)

    assert index.paths == [(str(project), "specific", False, False)]
    assert result.discovered == 1
    assert result.added == 1
    assert result.failed == 0


def test_missing_watched_directory_discovers_nothing(tmp_path: Path):
    assert discover_licht_projects(str(tmp_path / "missing")) == []


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

    result = scan_watch_directories(_Index(), "projects", [str(tmp_path)])

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
    result = scan_watch_directories(
        index,
        "projects",
        [str(tmp_path)],
        cancel_event,
    )

    assert result.cancelled is True
    assert index.paths == []
