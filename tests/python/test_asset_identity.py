# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager `.licht` project identity regressions."""

import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from lfs_plugins.asset_index import AssetIndex
from lfs_plugins.asset_watch import scan_watch_directories


ROOT = Path(__file__).resolve().parents[2]


def _inspection(
    project_uuid: str,
    *,
    file_uuid: str | None = None,
    commit_uuid: str | None = None,
    generation: int = 1,
    has_preview: bool = True,
):
    return SimpleNamespace(
        project_uuid=project_uuid,
        file_uuid=file_uuid or str(uuid.uuid4()),
        commit_uuid=commit_uuid or str(uuid.uuid4()),
        generation=generation,
        created_at_unix_ns=100,
        saved_at_unix_ns=200,
        physical_file_size=1234,
        role=SimpleNamespace(name="MASTER"),
        open_state=SimpleNamespace(name="OPEN"),
        has_preview=has_preview,
    )


def _install_inspections(monkeypatch, inspections):
    def inspect(path):
        value = inspections[Path(path).name]
        return value() if callable(value) else value

    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(inspect))


def test_catalog_uses_project_uuid_and_persists_only_locator_fields(monkeypatch, tmp_path: Path):
    first_path = tmp_path / "first.licht"
    copied_path = tmp_path / "copy.licht"
    first_path.write_bytes(b"first container")
    shutil.copy2(first_path, copied_path)
    project_uuid = str(uuid.uuid4())
    _install_inspections(
        monkeypatch,
        {
            first_path.name: _inspection(project_uuid),
            copied_path.name: _inspection(project_uuid),
        },
    )

    index = AssetIndex(library_path=tmp_path / "library.json")
    index.ensure_default_catalog()

    first, first_created = index.register_licht_asset(str(first_path), name="My project")
    duplicate, duplicate_created = index.register_licht_asset(str(copied_path))

    assert first is not None
    assert first_created is True
    assert duplicate_created is False
    assert duplicate.id == first.id
    assert len(index.list_projects()) == 1
    assert duplicate.path == str(copied_path)
    assert duplicate.name == "My project"

    catalog = json.loads((tmp_path / "library.json").read_text(encoding="utf-8"))
    assert set(catalog) == {"schema_version", "folders", "projects"}
    assert catalog["schema_version"] == 2
    assert set(catalog["folders"]["default"]) == {"name", "watch_directories"}
    assert catalog["projects"][first.id] == {
        "name": "My project",
        "path": str(copied_path),
        "folder_id": "default",
    }


def test_catalog_rejects_non_licht_paths(tmp_path: Path):
    unsupported = tmp_path / "unsupported.txt"
    unsupported.write_bytes(b"not a LichtFeld project")
    index = AssetIndex(library_path=tmp_path / "library.json")
    index.ensure_default_catalog()

    asset, created = index.register_licht_asset(str(unsupported))

    assert asset is None
    assert created is False
    assert index.list_projects() == []


def test_project_commit_changes_do_not_change_catalog_identity(monkeypatch, tmp_path: Path):
    project = tmp_path / "project.licht"
    project.write_bytes(b"project container")
    project_uuid = str(uuid.uuid4())
    inspection = _inspection(project_uuid, generation=1)
    inspections = {project.name: lambda: inspection}
    _install_inspections(monkeypatch, inspections)

    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path)
    index.ensure_default_catalog()
    licht_asset, _ = index.register_licht_asset(str(project))

    new_commit_uuid = str(uuid.uuid4())
    inspection = _inspection(
        project_uuid,
        commit_uuid=new_commit_uuid,
        generation=2,
    )
    verified = index.verify_asset(licht_asset.id)
    assert verified.commit_uuid == new_commit_uuid
    assert verified.generation == 2
    assert verified.available is True

    assert [asset.id for asset in index.list_projects()] == [licht_asset.id]
    stored = json.loads(library_path.read_text(encoding="utf-8"))["projects"]
    assert set(stored) == {project_uuid}
    assert "commit_uuid" not in stored[project_uuid]


def test_v2_load_rewrites_records_to_the_exact_minimal_schema(monkeypatch, tmp_path: Path):
    project = tmp_path / "project.licht"
    project.write_bytes(b"project container")
    project_uuid = str(uuid.uuid4())
    _install_inspections(monkeypatch, {project.name: _inspection(project_uuid)})
    library_path = tmp_path / "library.json"
    library_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "obsolete_root_field": True,
                "folders": {
                    "custom": {
                        "name": "Custom",
                        "watch_directories": [str(tmp_path), str(tmp_path)],
                        "obsolete_folder_field": True,
                    }
                },
                "projects": {
                    project_uuid: {
                        "name": "Project",
                        "path": str(project),
                        "folder_id": "custom",
                        "obsolete_project_field": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    index = AssetIndex(library_path=library_path)
    assert index.load() is True

    assert json.loads(library_path.read_text(encoding="utf-8")) == {
        "schema_version": 2,
        "folders": {
            "custom": {
                "name": "Custom",
                "watch_directories": [str(tmp_path)],
            },
            "default": {"name": "Default", "watch_directories": []},
        },
        "projects": {
            project_uuid: {
                "name": "Project",
                "path": str(project),
                "folder_id": "custom",
            }
        },
    }


def test_deleting_last_project_keeps_default_import_folder(monkeypatch, tmp_path: Path):
    first = tmp_path / "first.licht"
    second = tmp_path / "second.licht"
    first.write_bytes(b"first project")
    second.write_bytes(b"second project")
    _install_inspections(
        monkeypatch,
        {
            first.name: _inspection(str(uuid.uuid4())),
            second.name: _inspection(str(uuid.uuid4())),
        },
    )

    index = AssetIndex(library_path=tmp_path / "library.json")
    index.ensure_default_catalog()
    registered, _ = index.register_licht_asset(str(first))

    assert index.delete_asset(registered.id) is True
    replacement, created = index.register_licht_asset(str(second))
    assert created is True
    assert replacement is not None


def test_watched_scan_does_not_replace_a_live_explicit_locator(monkeypatch, tmp_path: Path):
    watched = tmp_path / "watched"
    nested = watched / "nested"
    nested.mkdir(parents=True)
    first = watched / "a.licht"
    duplicate = watched / "b.licht"
    second = nested / "c.LICHT"
    first.write_bytes(b"first project")
    shutil.copy2(first, duplicate)
    second.write_bytes(b"second project")
    first_uuid = str(uuid.uuid4())
    _install_inspections(
        monkeypatch,
        {
            first.name: _inspection(first_uuid),
            duplicate.name: _inspection(first_uuid),
            second.name: _inspection(str(uuid.uuid4())),
        },
    )

    index = AssetIndex(library_path=tmp_path / "library.json")
    index.ensure_default_catalog()
    result = scan_watch_directories(index, "default", [str(watched)])

    assert result.discovered == 3
    assert result.added == 2
    assert result.already_cataloged == 1
    assert result.failed == 0
    assert len(index.list_projects()) == 2
    assert index.get_asset(first_uuid).path == str(first)
    assert all(Path(asset.path).suffix.lower() == ".licht" for asset in index.list_projects())


def test_relink_requires_the_same_project_uuid(monkeypatch, tmp_path: Path):
    original = tmp_path / "original.licht"
    same_project = tmp_path / "same.licht"
    other_project = tmp_path / "other.licht"
    for path in (original, same_project, other_project):
        path.write_bytes(b"container")
    project_uuid = str(uuid.uuid4())
    _install_inspections(
        monkeypatch,
        {
            original.name: _inspection(project_uuid),
            same_project.name: _inspection(project_uuid),
            other_project.name: _inspection(str(uuid.uuid4())),
        },
    )
    index = AssetIndex(library_path=tmp_path / "library.json")
    index.ensure_default_catalog()
    project, _ = index.register_licht_asset(str(original))

    assert index.relink_asset(project.id, str(other_project)) is False
    assert project.path == str(original)
    assert index.relink_asset(project.id, str(same_project)) is True
    assert project.path == str(same_project)


def test_legacy_catalog_migration_keeps_only_names_paths_folders_and_watch_roots(
    monkeypatch, tmp_path: Path
):
    project_path = tmp_path / "legacy.licht"
    missing_path = tmp_path / "missing.licht"
    project_path.write_bytes(b"container")
    project_uuid = str(uuid.uuid4())
    _install_inspections(monkeypatch, {project_path.name: _inspection(project_uuid)})
    library_path = tmp_path / "library.json"
    library_path.write_text(
        json.dumps(
            {
                "version": "1.2.0",
                "created_at": "obsolete",
                "folders": {
                    "default": {
                        "id": "default",
                        "name": "Default",
                        "description": "drop",
                        "watch_directories": [str(tmp_path)],
                    }
                },
                "scenes": {"old": {"name": "drop"}},
                "assets": {
                    "old-id": {
                        "id": "old-id",
                        "name": "Custom legacy name",
                        "absolute_path": str(project_path),
                        "path": str(project_path),
                        "folder_id": "default",
                        "fingerprint": {"drop": True},
                        "notes": "drop",
                    },
                    "missing": {
                        "name": "Missing",
                        "absolute_path": str(missing_path),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    index = AssetIndex(library_path=library_path)
    assert index.load() is True
    migrated = json.loads(library_path.read_text(encoding="utf-8"))

    assert migrated["schema_version"] == 2
    assert migrated["folders"] == {
        "default": {
            "name": "Default",
            "watch_directories": [str(tmp_path)],
        }
    }
    assert migrated["projects"][project_uuid] == {
        "name": "Custom legacy name",
        "path": str(project_path),
        "folder_id": "default",
    }
    missing = next(
        project
        for project in migrated["projects"].values()
        if project["path"] == str(missing_path)
    )
    assert missing == {
        "name": "Missing",
        "path": str(missing_path),
        "folder_id": "default",
    }


def test_legacy_migration_preserves_original_backup(monkeypatch, tmp_path: Path):
    project_path = tmp_path / "legacy.licht"
    project_path.write_bytes(b"container")
    project_uuid = str(uuid.uuid4())
    _install_inspections(monkeypatch, {project_path.name: _inspection(project_uuid)})
    library_path = tmp_path / "library.json"
    legacy = {
        "folders": {"default": {"name": "Default"}},
        "assets": {"old": {"name": "Old", "absolute_path": str(project_path)}},
    }
    original = json.dumps(legacy, indent=2) + "\n"
    library_path.write_text(original, encoding="utf-8")

    assert AssetIndex(library_path=library_path).load() is True

    assert library_path.with_suffix(".json.bak").read_text(encoding="utf-8") == original


def test_pre_1265_projects_are_migrated_as_folders(monkeypatch, tmp_path: Path):
    project_path = tmp_path / "legacy.licht"
    project_path.write_bytes(b"container")
    project_uuid = str(uuid.uuid4())
    _install_inspections(monkeypatch, {project_path.name: _inspection(project_uuid)})
    library_path = tmp_path / "library.json"
    library_path.write_text(
        json.dumps(
            {
                "projects": {
                    "legacy-folder": {
                        "name": "Legacy folder",
                        "watch_directories": [str(tmp_path)],
                    }
                },
                "assets": {
                    "old": {
                        "name": "Legacy project",
                        "absolute_path": str(project_path),
                        "project_id": "legacy-folder",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    index = AssetIndex(library_path=library_path)
    assert index.load() is True

    assert index.folders["legacy-folder"]["name"] == "Legacy folder"
    assert index.assets[project_uuid]["folder_id"] == "legacy-folder"


def test_failed_v2_load_restores_previous_catalog(monkeypatch, tmp_path: Path):
    original_path = tmp_path / "original.licht"
    replacement_path = tmp_path / "replacement.licht"
    original_path.write_bytes(b"original")
    replacement_path.write_bytes(b"replacement")
    original_uuid = str(uuid.uuid4())
    replacement_uuid = str(uuid.uuid4())
    _install_inspections(
        monkeypatch,
        {
            original_path.name: _inspection(original_uuid),
            replacement_path.name: _inspection(replacement_uuid),
        },
    )
    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path)
    index.ensure_default_catalog()
    index.register_licht_asset(str(original_path))
    before = index.assets
    library_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "folders": {"default": {"name": "Default", "watch_directories": []}},
                "projects": {
                    replacement_uuid: {
                        "name": "Replacement",
                        "path": str(replacement_path),
                        "folder_id": "default",
                    },
                    "not-a-uuid": {
                        "name": "Broken",
                        "path": str(tmp_path / "broken.licht"),
                        "folder_id": "default",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    assert index.load() is False
    assert index.assets == before


def test_failed_mutations_restore_in_memory_catalog(monkeypatch, tmp_path: Path):
    original_path = tmp_path / "original.licht"
    relink_path = tmp_path / "relink.licht"
    new_path = tmp_path / "new.licht"
    for path in (original_path, relink_path, new_path):
        path.write_bytes(b"container")
    project_uuid = str(uuid.uuid4())
    _install_inspections(
        monkeypatch,
        {
            original_path.name: _inspection(project_uuid),
            relink_path.name: _inspection(project_uuid),
            new_path.name: _inspection(str(uuid.uuid4())),
        },
    )
    index = AssetIndex(library_path=tmp_path / "library.json")
    index.ensure_default_catalog()
    folder = index.create_folder("Projects")
    project, _ = index.register_licht_asset(
        str(original_path), folder_id=folder.id
    )
    before_folders = index.folders
    before_assets = index.assets
    monkeypatch.setattr(index, "save", lambda: False)

    assert index.create_folder("Unsaved") is None
    assert index.update_folder(folder.id, name="Unsaved rename") is None
    assert index.update_asset(project.id, name="Unsaved project") is None
    assert index.relink_asset(project.id, str(relink_path)) is False
    assert index.delete_folder(folder.id) is False
    assert index.register_licht_asset(str(new_path)) == (None, False)

    assert index.folders == before_folders
    assert index.assets == before_assets


def test_watched_duplicate_does_not_adopt_when_locator_is_offline(
    monkeypatch, tmp_path: Path
):
    original = tmp_path / "original.licht"
    watched = tmp_path / "watched"
    watched.mkdir()
    duplicate = watched / "duplicate.licht"
    original.write_bytes(b"container")
    duplicate.write_bytes(b"container")
    project_uuid = str(uuid.uuid4())
    _install_inspections(
        monkeypatch,
        {
            original.name: _inspection(project_uuid),
            duplicate.name: _inspection(project_uuid),
        },
    )
    index = AssetIndex(library_path=tmp_path / "library.json")
    index.ensure_default_catalog()
    project, _ = index.register_licht_asset(str(original))
    original.unlink()

    result = scan_watch_directories(index, "default", [str(watched)])

    assert result.already_cataloged == 1
    assert index.get_asset(project.id).path == str(original)
    assert index.get_asset(project.id).status == "MISSING"


def test_storage_resolution_falls_back_from_unwritable_native_path(
    monkeypatch, tmp_path: Path
):
    from lfs_plugins import asset_index as asset_index_module

    native = tmp_path / "unwritable" / "asset_library"
    native.mkdir(parents=True)
    (native / "library.json").write_text('{"schema_version": 2}', encoding="utf-8")
    appdata = tmp_path / "appdata"
    fallback = appdata / "LichtFeldStudio" / "asset_manager"
    monkeypatch.delenv("LFS_ASSET_MANAGER_DIR", raising=False)
    monkeypatch.setenv("LFS_RESOLVED_ASSET_LIBRARY_DIR", str(native))
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.delenv("LFS_SAFE_MODE", raising=False)
    monkeypatch.setattr(
        asset_index_module,
        "_path_accepts_writes",
        lambda path: path == fallback,
    )

    assert asset_index_module.resolve_asset_manager_storage_path() == fallback
    assert (fallback / "library.json").read_text(encoding="utf-8") == (
        '{"schema_version": 2}'
    )


def test_safe_mode_storage_resolution_does_not_probe(monkeypatch, tmp_path: Path):
    from lfs_plugins import asset_index as asset_index_module

    native = tmp_path / "native" / "asset_library"
    monkeypatch.delenv("LFS_ASSET_MANAGER_DIR", raising=False)
    monkeypatch.setenv("LFS_RESOLVED_ASSET_LIBRARY_DIR", str(native))
    monkeypatch.setenv("LFS_SAFE_MODE", "1")
    monkeypatch.setattr(
        asset_index_module,
        "_path_accepts_writes",
        lambda _path: pytest.fail("safe mode must not probe storage"),
    )

    assert asset_index_module.resolve_asset_manager_storage_path() == native


def test_default_catalog_migrates_from_appdata_location(monkeypatch, tmp_path: Path):
    project_path = tmp_path / "legacy.licht"
    project_path.write_bytes(b"container")
    project_uuid = str(uuid.uuid4())
    _install_inspections(monkeypatch, {project_path.name: _inspection(project_uuid)})
    native = tmp_path / "native" / "asset_library"
    appdata = tmp_path / "appdata"
    legacy_library = appdata / "LichtFeldStudio" / "asset_manager" / "library.json"
    legacy_library.parent.mkdir(parents=True)
    legacy_library.write_text(
        json.dumps(
            {
                "folders": {"default": {"name": "Default"}},
                "assets": {
                    "legacy": {
                        "name": "Legacy",
                        "absolute_path": str(project_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("LFS_ASSET_MANAGER_DIR", raising=False)
    monkeypatch.delenv("LFS_SAFE_MODE", raising=False)
    monkeypatch.setenv("LFS_RESOLVED_ASSET_LIBRARY_DIR", str(native))
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    index = AssetIndex()
    assert index.load() is True

    assert index.library_path == native / "library.json"
    assert index.assets[project_uuid]["path"] == str(project_path)
    assert json.loads(
        index.library_path.with_suffix(".json.bak").read_text(encoding="utf-8")
    )["assets"]["legacy"]["name"] == "Legacy"


def test_asset_library_binding_returns_canonical_path(lf):
    assert Path(lf.io.asset_library_dir()).name == "asset_library"


def test_asset_manager_ui_exposes_only_project_import_and_open_actions():
    rml = (
        ROOT / "src/visualizer/gui/rmlui/resources/asset_manager.rml"
    ).read_text(encoding="utf-8")
    panel_source = (
        ROOT / "src/python/lfs_plugins/asset_manager_panel.py"
    ).read_text(encoding="utf-8")

    assert 'data-event-click="on_import_project"' in rml
    assert 'data-asset-action="load"' in rml
    assert 'data-folder-action="menu"' in rml
    assert '"action": "watch_dirs"' in panel_source
    assert rml.count('data-event-click="on_import_project"') == 1
