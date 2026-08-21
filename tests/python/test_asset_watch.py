# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for .licht-only watched-directory discovery."""

import os
from pathlib import Path
from types import SimpleNamespace

from lfs_plugins.asset_index import AssetIndex, DEFAULT_FOLDER_ID
from lfs_plugins.asset_watch import discover_licht_projects, scan_watch_directories


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

        def register_licht_asset(self, path, *, folder_id):
            self.paths.append((path, folder_id))
            return SimpleNamespace(id=path), len(self.paths) == 1

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


def test_missing_watched_directory_discovers_nothing(tmp_path: Path):
    assert discover_licht_projects(str(tmp_path / "missing")) == []
