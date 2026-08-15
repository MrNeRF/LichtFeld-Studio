# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for legacy library.json schema migration (projects -> folders)."""

import json
import time
from pathlib import Path

from lfs_plugins.asset_index import (
    DEFAULT_FOLDER_ID,
    LIBRARY_VERSION,
    AssetIndex,
)


def _legacy_library(
    project_id: str = "proj-1",
    scene_id: str = "scene-1",
    asset_id: str = "asset-1",
) -> dict:
    """Build a pre-#1265 library.json payload keyed by projects/project_id."""
    return {
        "version": LIBRARY_VERSION,
        "created_at": "2024-01-01T00:00:00",
        "modified_at": "2024-01-01T00:00:00",
        "projects": {
            project_id: {
                "id": project_id,
                "name": "Legacy Project",
                "description": "A project from the old schema",
                "created_at": "2024-01-01T00:00:00",
                "modified_at": "2024-01-01T00:00:00",
                "scene_ids": [scene_id],
                "tags": ["legacy"],
                "notes": "",
                "thumbnail_asset_id": None,
                "watch_directories": [],
            }
        },
        "scenes": {
            scene_id: {
                "id": scene_id,
                "project_id": project_id,
                "name": "Legacy Scene",
                "description": "",
                "created_at": "2024-01-01T00:00:00",
                "modified_at": "2024-01-01T00:00:00",
                "dataset_asset_id": None,
                "tags": [],
                "notes": "",
                "thumbnail_asset_id": None,
            }
        },
        "assets": {
            asset_id: {
                "id": asset_id,
                "project_id": project_id,
                "scene_id": scene_id,
                "name": "Legacy Asset",
                "type": "dataset",
                "role": "source",
                "path": "data",
                "absolute_path": "/tmp/data",
                "created_at": "2024-01-01T00:00:00",
                "modified_at": "2024-01-01T00:00:00",
                "file_size_bytes": 0,
                "tags": [],
                "thumbnail_path": None,
                "geometry_metadata": {},
                "dataset_metadata": {},
                "transform_metadata": {},
                "exists": True,
            }
        },
        "collections": {},
        "tags": {},
    }


def _modern_library(
    folder_id: str = "folder-1",
    scene_id: str = "scene-1",
    asset_id: str = "asset-1",
) -> dict:
    """Build a current-schema library.json payload."""
    return {
        "version": LIBRARY_VERSION,
        "created_at": "2024-06-01T00:00:00",
        "modified_at": "2024-06-01T00:00:00",
        "folders": {
            folder_id: {
                "id": folder_id,
                "name": "Modern Folder",
                "description": "",
                "created_at": "2024-06-01T00:00:00",
                "modified_at": "2024-06-01T00:00:00",
                "scene_ids": [scene_id],
                "tags": [],
                "notes": "",
                "thumbnail_asset_id": None,
                "watch_directories": [],
            }
        },
        "scenes": {
            scene_id: {
                "id": scene_id,
                "folder_id": folder_id,
                "name": "Modern Scene",
                "description": "",
                "created_at": "2024-06-01T00:00:00",
                "modified_at": "2024-06-01T00:00:00",
                "dataset_asset_id": None,
                "tags": [],
                "notes": "",
                "thumbnail_asset_id": None,
            }
        },
        "assets": {
            asset_id: {
                "id": asset_id,
                "folder_id": folder_id,
                "scene_id": scene_id,
                "name": "Modern Asset",
                "type": "dataset",
                "role": "source",
                "path": "data",
                "absolute_path": "/tmp/data",
                "created_at": "2024-06-01T00:00:00",
                "modified_at": "2024-06-01T00:00:00",
                "file_size_bytes": 0,
                "tags": [],
                "thumbnail_path": None,
                "geometry_metadata": {},
                "dataset_metadata": {},
                "transform_metadata": {},
                "exists": True,
            }
        },
        "collections": {},
        "tags": {},
    }


def _write_library(path: Path, payload: dict) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    return text


def test_legacy_file_migrates(tmp_path: Path):
    library_path = tmp_path / "library.json"
    project_id = "proj-legacy"
    legacy = _legacy_library(project_id=project_id)
    original_text = _write_library(library_path, legacy)

    index = AssetIndex(library_path=library_path)
    assert index.load() is True

    assert project_id in index._folders
    assert index._folders[project_id].name == "Legacy Project"
    assert index._scenes["scene-1"].folder_id == project_id
    assert index._assets["asset-1"].folder_id == project_id

    on_disk = json.loads(library_path.read_text(encoding="utf-8"))
    assert "folders" in on_disk
    assert "projects" not in on_disk
    assert project_id in on_disk["folders"]
    assert on_disk["scenes"]["scene-1"]["folder_id"] == project_id
    assert "project_id" not in on_disk["scenes"]["scene-1"]
    assert on_disk["assets"]["asset-1"]["folder_id"] == project_id
    assert "project_id" not in on_disk["assets"]["asset-1"]

    bak_path = library_path.with_suffix(".json.bak")
    assert bak_path.is_file()
    bak_data = json.loads(bak_path.read_text(encoding="utf-8"))
    assert "projects" in bak_data
    assert "folders" not in bak_data
    assert bak_data["scenes"]["scene-1"].get("project_id") == project_id
    assert bak_data["assets"]["asset-1"].get("project_id") == project_id
    # Backup holds the original legacy content (key shape).
    assert bak_data["projects"][project_id]["id"] == project_id
    assert "folder_id" not in bak_data["scenes"]["scene-1"]
    # original_text is the exact bytes we wrote before load/save.
    assert json.loads(original_text)["projects"][project_id]["id"] == project_id


def test_scene_without_folder_linkage_gets_default(tmp_path: Path):
    library_path = tmp_path / "library.json"
    # Scene missing both folder_id and project_id.
    payload = _legacy_library()
    scene = payload["scenes"]["scene-1"]
    scene.pop("project_id", None)
    scene.pop("folder_id", None)
    assert "project_id" not in scene
    assert "folder_id" not in scene
    _write_library(library_path, payload)

    index = AssetIndex(library_path=library_path)
    assert index.load() is True
    assert index._scenes["scene-1"].folder_id == DEFAULT_FOLDER_ID


def test_modern_file_untouched(tmp_path: Path):
    library_path = tmp_path / "library.json"
    modern = _modern_library()
    original_text = _write_library(library_path, modern)
    # Ensure mtime is stable enough to compare after a fast load.
    time.sleep(0.05)
    mtime_before = library_path.stat().st_mtime_ns

    index = AssetIndex(library_path=library_path)
    assert index.load() is True

    assert library_path.read_text(encoding="utf-8") == original_text
    assert library_path.stat().st_mtime_ns == mtime_before
    assert not library_path.with_suffix(".json.bak").exists()

    assert "folder-1" in index._folders
    assert index._scenes["scene-1"].folder_id == "folder-1"
    assert index._assets["asset-1"].folder_id == "folder-1"


def test_migrated_index_round_trips(tmp_path: Path):
    library_path = tmp_path / "library.json"
    project_id = "proj-roundtrip"
    _write_library(library_path, _legacy_library(project_id=project_id))

    index = AssetIndex(library_path=library_path)
    assert index.load() is True
    assert index.save() is True

    reloaded = AssetIndex(library_path=library_path)
    assert reloaded.load() is True
    assert project_id in reloaded._folders
    assert reloaded._folders[project_id].name == "Legacy Project"
    assert reloaded._scenes["scene-1"].folder_id == project_id
    assert reloaded._assets["asset-1"].folder_id == project_id
    assert reloaded._scenes["scene-1"].name == "Legacy Scene"
    assert reloaded._assets["asset-1"].name == "Legacy Asset"

    on_disk = json.loads(library_path.read_text(encoding="utf-8"))
    assert "folders" in on_disk
    assert "projects" not in on_disk


def test_migration_save_failure_logs_and_load_still_succeeds(
    tmp_path: Path, monkeypatch, caplog
):
    """In-memory migration remains valid if persist fails; load returns True."""
    import logging

    from lfs_plugins import asset_index as asset_index_module

    library_path = tmp_path / "library.json"
    project_id = "proj-save-fail"
    _write_library(library_path, _legacy_library(project_id=project_id))

    index = AssetIndex(library_path=library_path)

    def failing_save(self):
        return False

    monkeypatch.setattr(AssetIndex, "save", failing_save)

    with caplog.at_level(logging.ERROR, logger=asset_index_module.__name__):
        assert index.load() is True

    # In-memory catalog is still migrated.
    assert project_id in index._folders
    assert index._scenes["scene-1"].folder_id == project_id
    assert index._assets["asset-1"].folder_id == project_id

    # Stale legacy file remains on disk when save fails.
    on_disk = json.loads(library_path.read_text(encoding="utf-8"))
    assert "projects" in on_disk
    assert "folders" not in on_disk

    error_msgs = [
        rec.getMessage() for rec in caplog.records if rec.levelno >= logging.ERROR
    ]
    assert any(
        "Failed to persist migrated library.json" in msg and str(library_path) in msg
        for msg in error_msgs
    ), error_msgs
