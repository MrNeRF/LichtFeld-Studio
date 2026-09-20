# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager `.licht` project identity regressions."""

import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from lfs_plugins.asset_index import AssetIndex, Project
from lfs_plugins.asset_watch import scan_asset_folder


@pytest.mark.parametrize("operation", [
    "register", "relink", "rename", "viewing_copy", "delete", "delete_batch",
    "delete_folder", "default_folder", "reconcile", "verify", "verify_batch", "clean_missing",
])
def test_library_rechecks_identity_at_json_replacement(monkeypatch, tmp_path, operation):
    from lfs_plugins import asset_index

    folder = tmp_path / "项目"
    folder.mkdir()
    path = folder / "é.licht"
    path.write_bytes(b"original project")
    original = _inspection(str(uuid.uuid4()))
    current = original
    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(lambda _path: current))
    index = AssetIndex(tmp_path / "library.json", tmp_path / "default")
    assert index.load()
    project, _ = index.register_licht_asset(str(path))
    alternate = folder / "new.licht"
    shutil.copyfile(path, alternate)
    if operation == "clean_missing":
        path.unlink()
    before = index.library_path.read_bytes()
    rows = index.assets.copy()
    real_fsync = asset_index.os.fsync

    def replace_identity(fd):
        nonlocal current
        real_fsync(fd)
        current = _inspection(str(uuid.uuid4()))
        if operation == "clean_missing":
            path.write_bytes(b"a new project appeared")

    monkeypatch.setattr(asset_index.os, "fsync", replace_identity)
    actions = {
        "register": lambda: index.register_licht_asset(str(alternate), inspection=original),
        "relink": lambda: index.relink_asset(project.id, str(alternate)),
        "rename": lambda: index.update_asset(project.id, name="New title"),
        "viewing_copy": lambda: index.update_asset(project.id, viewing_copy=True),
        "delete": lambda: index.delete_asset(project.id),
        "delete_batch": lambda: index.delete_assets([project.id]),
        "delete_folder": lambda: index.delete_folder(project.folder_id),
        "default_folder": lambda: index.set_default_folder_path(str(tmp_path / "new-default")),
        "reconcile": lambda: index.reconcile_observations([
            asset_index.AssetObservation(str(path), project.folder_id, original)]),
        "verify": lambda: index.verify_asset(project.id),
        "verify_batch": lambda: index.verify_projects_batch([project.id]),
        "clean_missing": lambda: index.clean_missing_entries(project.folder_id),
    }
    actions[operation]()
    assert "changed" in index.last_error
    assert index.library_path.read_bytes() == before
    assert index.assets == rows


def test_library_rejects_redirected_project_and_library_paths(monkeypatch, tmp_path):
    path = tmp_path / "项目.licht"
    path.write_bytes(b"original project")
    other = tmp_path / "other.licht"
    shutil.copyfile(path, other)
    alias = tmp_path / "别名.licht"
    alias.symlink_to(path)
    inspection = _inspection(str(uuid.uuid4()))
    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(lambda _path: inspection))
    index = AssetIndex(tmp_path / "library.json", tmp_path)
    assert index.load()
    project, _ = index.register_licht_asset(str(alias))
    assert index.find_asset_by_path(str(path)).id == project.id
    before = index.library_path.read_bytes()
    index.update_asset(project.id, save=False, name="Pending")
    alias.unlink()
    alias.symlink_to(other)
    assert not index.save()
    assert "path changed" in index.last_error and index.library_path.read_bytes() == before

    redirected = tmp_path / "other-library.json"
    redirected.write_bytes(before)
    index.library_path.unlink()
    index.library_path.symlink_to(redirected)
    assert not index.save()
    assert "library path changed" in index.last_error and redirected.read_bytes() == before


def test_reconciliation_refuses_an_alias_redirected_after_observation(monkeypatch, tmp_path):
    from lfs_plugins.asset_index import AssetObservation

    path, other, alias = (tmp_path / name for name in ("a.licht", "b.licht", "alias.licht"))
    path.write_bytes(b"same identity")
    shutil.copyfile(path, other)
    alias.symlink_to(path)
    inspection = _inspection(str(uuid.uuid4()))
    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(lambda _path: inspection))
    index = AssetIndex(tmp_path / "library.json", tmp_path)
    assert index.load()
    observation = AssetObservation(str(alias), "default", inspection)
    before = index.library_path.read_bytes()
    alias.unlink()
    alias.symlink_to(other)
    assert index.reconcile_observations([observation])["failed"] == 1
    assert "path changed" in index.last_error and index.library_path.read_bytes() == before


@pytest.mark.parametrize("operation", ["verify", "verify_batch", "reconcile_all", "scan_batch"])
def test_library_keeps_path_identity_from_before_inspection(monkeypatch, tmp_path, operation):
    from lfs_plugins import asset_watch

    path, other, alias = (tmp_path / name for name in ("a.licht", "b.licht", "别名.licht"))
    path.write_bytes(b"project")
    shutil.copyfile(path, other)
    alias.symlink_to(path)
    inspection = _inspection(str(uuid.uuid4()))
    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(lambda *args: inspection))
    index = AssetIndex(tmp_path / "library.json", tmp_path)
    assert index.load()
    project, _ = index.register_licht_asset(str(alias))
    before = index.library_path.read_bytes()
    swapped = False

    def inspect(*args):
        nonlocal swapped
        if not swapped:
            swapped = True
            alias.unlink()
            alias.symlink_to(other)
        return inspection

    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(inspect))
    if operation == "verify":
        index.verify_asset(project.id)
    elif operation == "verify_batch":
        index.verify_projects_batch([project.id])
    elif operation == "reconcile_all":
        index.reconcile_all()
    else:
        monkeypatch.setattr(asset_watch, "_known_path_is_unchanged", lambda *args: False)
        asset_watch._commit_registration_batch(index, [(str(alias), "default")], None)
        assert not index.save()
    assert swapped and "identity or path changed" in index.last_error
    assert index.library_path.read_bytes() == before


@pytest.mark.parametrize("operation", ["verify", "verify_batch", "reconcile_all", "reconcile_observations"])
@pytest.mark.parametrize("health", ["missing", "unreadable"])
def test_library_health_write_refuses_a_replaced_project(monkeypatch, tmp_path, operation, health):
    from lfs_plugins import asset_index

    path = tmp_path / "项目.licht"
    path.write_bytes(b"project")
    inspection = _inspection(str(uuid.uuid4()))
    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(lambda *args: inspection))
    index = AssetIndex(tmp_path / "library.json", tmp_path)
    assert index.load()
    project, _ = index.register_licht_asset(str(path))
    before = index.library_path.read_bytes()
    rows = index.assets.copy()
    if health == "missing":
        path.unlink()
    else:
        def unreadable(*args):
            raise ValueError("Damaged project")
        monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(unreadable))
    observation = asset_index.AssetObservation(str(path), "default", error="Could not read the project")
    original_fsync = asset_index.os.fsync

    def replace(fd):
        original_fsync(fd)
        path.write_bytes(b"a replacement project")

    monkeypatch.setattr(asset_index.os, "fsync", replace)
    if operation == "verify":
        index.verify_asset(project.id)
    elif operation == "verify_batch":
        index.verify_projects_batch([project.id])
    elif operation == "reconcile_all":
        index.reconcile_all()
    else:
        index.reconcile_observations([observation])
    assert "identity or path changed" in index.last_error
    assert index.library_path.read_bytes() == before and index.assets == rows


def test_library_writes_preserve_unicode_symlinks(monkeypatch, tmp_path):
    path = tmp_path / "项目-é.licht"
    path.write_bytes(b"project")
    inspection = _inspection(str(uuid.uuid4()))
    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(lambda _path: inspection))
    library = tmp_path / "library.json"
    assert AssetIndex(library, tmp_path).load()
    alias = tmp_path / "图书馆.json"
    alias.symlink_to(library)
    index = AssetIndex(alias, tmp_path)
    assert index.load()
    project, _ = index.register_licht_asset(str(path), name="项目 é")
    assert alias.is_symlink()
    assert json.loads(library.read_text())["projects"][project.id]["name"] == "项目 é"


def _inspection(
    project_uuid: str,
    *,
    file_uuid: str | None = None,
    commit_uuid: str | None = None,
    generation: int = 1,
    has_preview: bool = True,
    preview_width: int = 640,
    preview_height: int = 360,
    open_state: str = "OPEN",
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
        open_state=SimpleNamespace(name=open_state),
        has_preview=has_preview,
        preview_width=preview_width,
        preview_height=preview_height,
    )


def _legacy_backup_path(library_path: Path) -> Path:
    return library_path.with_name(library_path.name + ".legacy.bak")


def _install_inspections(monkeypatch, inspections):
    def inspect(path):
        value = inspections[Path(path).name]
        return value() if callable(value) else value

    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(inspect))


def _cached_record(path: Path, project_uuid: str, inspection=None):
    inspection = inspection or _inspection(project_uuid)
    return {
        "name": path.stem,
        "path": str(path),
        "folder_id": "default",
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "fallback_preview_path": "",
        "file_uuid": inspection.file_uuid,
        "commit_uuid": inspection.commit_uuid,
        "generation": inspection.generation,
        "created_at_unix_ns": inspection.created_at_unix_ns,
        "saved_at_unix_ns": inspection.saved_at_unix_ns,
        "file_size_bytes": inspection.physical_file_size,
        "role": "MASTER",
        "open_state": inspection.open_state.name,
        "has_preview": inspection.has_preview,
        "preview_width": inspection.preview_width,
        "preview_height": inspection.preview_height,
        "status": "AVAILABLE",
    }


def test_catalog_uses_project_uuid_and_persists_inspection_fields(monkeypatch, tmp_path: Path):
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
    index.load()

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
    assert catalog["schema_version"] == 6
    assert catalog["folders"]["default"] == {"path": str(tmp_path)}
    assert catalog["projects"][first.id] == duplicate.to_storage_dict()
    assert {
        "file_uuid",
        "commit_uuid",
        "generation",
        "created_at_unix_ns",
        "saved_at_unix_ns",
        "file_size_bytes",
        "role",
        "open_state",
        "has_preview",
        "preview_width",
        "preview_height",
        "status",
    }.issubset(catalog["projects"][first.id])


def test_catalog_rejects_non_licht_paths(tmp_path: Path):
    unsupported = tmp_path / "unsupported.txt"
    unsupported.write_bytes(b"not a LichtFeld project")
    index = AssetIndex(library_path=tmp_path / "library.json")
    index.load()

    asset, created = index.register_licht_asset(str(unsupported))

    assert asset is None
    assert created is False
    assert index.list_projects() == []


def test_projects_are_assigned_by_real_directory_not_virtual_folder_id(
    monkeypatch, tmp_path: Path
):
    default = tmp_path / "default"
    selected = default / "selected"
    selected.mkdir(parents=True)
    project_path = selected / "project.licht"
    project_path.write_bytes(b"project container")
    project_uuid = str(uuid.uuid4())
    _install_inspections(monkeypatch, {project_path.name: _inspection(project_uuid)})
    library_path = tmp_path / "catalog" / "library.json"
    index = AssetIndex(library_path=library_path, default_folder_path=default)
    index.load()
    selected_folder = index.add_folder(str(selected))

    project, created = index.register_licht_asset(
        str(project_path), folder_id="default"
    )

    assert created is True
    assert selected_folder is not None
    assert project.folder_id == selected_folder.id
    assert index.update_asset(project.id, folder_id="default") is None
    stored = json.loads(library_path.read_text(encoding="utf-8"))
    assert stored["folders"] == {
        "default": {"path": str(default)},
        selected_folder.id: {"path": str(selected)},
    }


def test_failed_project_inspection_does_not_leave_an_implicit_folder(
    monkeypatch, tmp_path: Path
):
    default = tmp_path / "default"
    outside = tmp_path / "outside"
    outside.mkdir()
    project_path = outside / "broken.licht"
    project_path.write_bytes(b"broken container")

    def fail_inspection(_path):
        raise ValueError("broken")

    monkeypatch.setattr(
        AssetIndex,
        "_inspect_path",
        staticmethod(fail_inspection),
    )
    index = AssetIndex(
        library_path=tmp_path / "catalog" / "library.json",
        default_folder_path=default,
    )
    index.load()

    with pytest.raises(ValueError, match="broken"):
        index.register_licht_asset(str(project_path))

    assert set(index.folders) == {"default"}


def test_changing_default_directory_preserves_old_real_folder_mapping(
    monkeypatch, tmp_path: Path
):
    old_default = tmp_path / "old-default"
    new_default = tmp_path / "new-default"
    old_default.mkdir()
    project_path = old_default / "project.licht"
    project_path.write_bytes(b"project container")
    project_uuid = str(uuid.uuid4())
    _install_inspections(monkeypatch, {project_path.name: _inspection(project_uuid)})
    index = AssetIndex(
        library_path=tmp_path / "catalog" / "library.json",
        default_folder_path=old_default,
    )
    index.load()
    project, _ = index.register_licht_asset(str(project_path))

    assert index.set_default_folder_path(str(new_default)) is True

    folders = index.folders
    old_mapping = next(
        folder
        for folder in folders.values()
        if folder["id"] != "default" and folder["path"] == str(old_default)
    )
    assert new_default.is_dir()
    assert folders["default"]["path"] == str(new_default)
    assert index.get_asset(project.id).folder_id == old_mapping["id"]


def test_existing_folder_mapping_becomes_default_without_duplicate(tmp_path: Path):
    old_default = tmp_path / "old-default"
    new_default = tmp_path / "new-default"
    old_default.mkdir()
    new_default.mkdir()
    index = AssetIndex(
        library_path=tmp_path / "catalog" / "library.json",
        default_folder_path=old_default,
    )
    index.load()
    added = index.add_folder(str(new_default))
    assert added is not None

    assert index.set_default_folder_path(str(new_default)) is True

    assert index.folders == {
        "default": {
            "id": "default",
            "name": new_default.name,
            "path": str(new_default),
            "is_default": True,
        }
    }


def test_project_commit_changes_do_not_change_catalog_identity(monkeypatch, tmp_path: Path):
    project = tmp_path / "project.licht"
    project.write_bytes(b"project container")
    project_uuid = str(uuid.uuid4())
    inspection = _inspection(project_uuid, generation=1)
    inspections = {project.name: lambda: inspection}
    _install_inspections(monkeypatch, inspections)

    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path)
    index.load()
    licht_asset, _ = index.register_licht_asset(str(project))

    new_commit_uuid = str(uuid.uuid4())
    inspection = _inspection(
        project_uuid,
        commit_uuid=new_commit_uuid,
        generation=2,
    )
    # A real project save changes the file metadata; the cache must not be
    # bypassed merely because this test swapped its inspection stub.
    project.write_bytes(b"project container updated")
    verified = index.verify_asset(licht_asset.id)
    assert verified.commit_uuid == new_commit_uuid
    assert verified.generation == 2
    assert verified.available is True

    assert [asset.id for asset in index.list_projects()] == [licht_asset.id]
    stored = json.loads(library_path.read_text(encoding="utf-8"))["projects"]
    assert set(stored) == {project_uuid}
    assert stored[project_uuid]["commit_uuid"] == new_commit_uuid
    assert stored[project_uuid]["generation"] == 2


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

    migrated = json.loads(library_path.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 6
    assert migrated["folders"] == {"default": {"path": str(tmp_path)}}
    assert migrated["projects"][project_uuid] == index.get_asset(project_uuid).to_storage_dict()


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
    index.load()
    registered, _ = index.register_licht_asset(str(first))

    assert index.delete_asset(registered.id) is True
    replacement, created = index.register_licht_asset(str(second))
    assert created is True
    assert replacement is not None


def test_folder_scan_does_not_replace_a_live_explicit_locator(monkeypatch, tmp_path: Path):
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
    index.load()
    result = scan_asset_folder(index, "default", str(watched))

    assert result.discovered == 3
    assert result.added == 2
    assert result.already_cataloged == 1
    assert result.failed == 0
    assert len(index.list_projects()) == 2
    assert index.get_asset(first_uuid).path == str(first)
    assert index.get_asset(first_uuid).relocation_candidate == ""
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
    index.load()
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

    assert migrated["schema_version"] == 6
    assert migrated["folders"] == {"default": {"path": str(tmp_path)}}
    migrated_project = index.get_asset(project_uuid)
    assert migrated_project is not None
    assert migrated["projects"][project_uuid] == migrated_project.to_storage_dict()
    missing_record = next(
        project
        for project in migrated["projects"].values()
        if project["path"] == str(missing_path)
    )
    missing_project = next(
        project for project in index.list_projects() if project.path == str(missing_path)
    )
    assert missing_record == missing_project.to_storage_dict()


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

    index = AssetIndex(library_path=library_path)
    assert index.load() is True

    legacy_backup = _legacy_backup_path(library_path)
    assert legacy_backup.read_text(encoding="utf-8") == original
    assert library_path.with_suffix(".json.bak").read_text(encoding="utf-8") == original
    assert index.save() is True
    assert index.save() is True
    assert legacy_backup.read_text(encoding="utf-8") == original


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

    assert index.folders["default"]["path"] == str(tmp_path)
    assert index.assets[project_uuid]["folder_id"] == "default"


def test_v3_load_skips_bad_project_rows_and_migrates_valid_rows(monkeypatch, tmp_path: Path):
    good_path = tmp_path / "good.licht"
    good_path.write_bytes(b"good")
    good_uuid = str(uuid.uuid4())
    duplicate_uuid = str(uuid.uuid4())
    empty_uuid = str(uuid.uuid4())
    non_licht_uuid = str(uuid.uuid4())
    uppercase_uuid = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"
    _install_inspections(monkeypatch, {good_path.name: _inspection(good_uuid)})
    library_path = tmp_path / "library.json"
    payload = {
        "schema_version": 3,
        "folders": {"default": {"path": str(tmp_path)}},
        "projects": {
            good_uuid: {
                "name": "Good",
                "path": str(good_path),
                "folder_id": "default",
            },
            "not-a-uuid": {
                "name": "Broken",
                "path": str(tmp_path / "broken.licht"),
                "folder_id": "default",
            },
            uppercase_uuid: {
                "name": "Upper",
                "path": str(tmp_path / "upper.licht"),
                "folder_id": "default",
            },
            non_licht_uuid: {
                "name": "Text",
                "path": str(tmp_path / "notes.txt"),
                "folder_id": "default",
            },
            duplicate_uuid: {
                "name": "Duplicate",
                "path": str(good_path),
                "folder_id": "default",
            },
            empty_uuid: {
                "name": "Empty",
                "path": "",
                "folder_id": "default",
            },
            str(uuid.uuid4()): "not-an-object",
        },
    }
    original = json.dumps(payload, indent=2) + "\n"
    library_path.write_text(original, encoding="utf-8")

    index = AssetIndex(library_path=library_path)
    assert index.load() is True

    assert [project.id for project in index.list_projects()] == [good_uuid]
    assert len(index.load_issues) == 6
    assert any("not-a-uuid" in issue for issue in index.load_issues)
    assert any(uppercase_uuid in issue for issue in index.load_issues)
    assert any(str(tmp_path / "notes.txt") in issue for issue in index.load_issues)
    assert any(str(good_path) in issue and "duplicate" in issue.casefold() for issue in index.load_issues)
    assert any(empty_uuid in issue for issue in index.load_issues)
    assert any("not an object" in issue for issue in index.load_issues)
    migrated = json.loads(library_path.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 6
    assert migrated["projects"][good_uuid] == index.get_asset(good_uuid).to_storage_dict()


def test_v3_load_leaves_cached_rows_unverified_without_inspecting(
    monkeypatch, tmp_path: Path
):
    project_path = tmp_path / "garden.licht"
    project_path.write_bytes(b"garden")
    project_uuid = str(uuid.uuid4())
    inspect_calls = []

    def inspect(_path):
        inspect_calls.append(_path)
        raise AssertionError("v3 load must not inspect catalog rows")

    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(inspect))
    library_path = tmp_path / "library.json"
    library_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "folders": {"default": {"path": str(tmp_path)}},
                "projects": {
                    project_uuid: {
                        "name": "Garden",
                        "path": str(project_path),
                        "folder_id": "default",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    index = AssetIndex(library_path=library_path)
    assert index.load() is True

    project = index.get_asset(project_uuid)
    assert project is not None
    assert project.status == "READING"
    assert project.name == "Garden"
    assert project.path == str(project_path)
    assert project.exists is True
    assert project.available is False
    assert inspect_calls == []

    monkeypatch.setattr(
        AssetIndex, "_inspect_path", staticmethod(lambda _path: _inspection(project_uuid))
    )
    verified = index.verify_asset(project_uuid)
    assert verified is not None
    assert verified.status == "AVAILABLE"
    assert verified.available is True



def test_v4_restored_inspection_is_verified_before_becoming_available(
    monkeypatch, tmp_path: Path
):
    project_path = tmp_path / "cached.licht"
    project_path.write_bytes(b"cached project")
    project_uuid = str(uuid.uuid4())
    inspection = _inspection(project_uuid, commit_uuid="cached-commit")
    library_path = tmp_path / "library.json"
    library_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "folders": {"default": {"path": str(tmp_path)}},
                "projects": {
                    project_uuid: _cached_record(project_path, project_uuid, inspection)
                },
                "directory_mtimes": {},
            }
        ),
        encoding="utf-8",
    )
    inspect_calls = []

    def inspect(*_args):
        inspect_calls.append(True)
        return inspection

    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(inspect))
    index = AssetIndex(library_path=library_path)
    assert index.load() is True
    assert "directory_mtimes" in json.loads(library_path.read_text(encoding="utf-8"))

    project = index.get_asset(project_uuid)
    assert project is not None
    assert project.inspection_restored is True
    assert index.verify_projects_batch([project_uuid]) == 1
    assert project.status == "AVAILABLE"
    assert project.available is True
    assert project.has_preview is True
    assert project.preview_width == 640
    assert project.preview_height == 360
    assert project.commit_uuid == "cached-commit"
    assert project.inspection_verified is True
    assert inspect_calls == [True, True]  # Inspection and the guarded library write.


def test_identity_mismatch_preserves_inspected_file_size_and_clears_path_stat(
    monkeypatch, tmp_path: Path
):
    project_path = tmp_path / "mismatch.licht"
    project_path.write_bytes(b"small container")
    project_uuid = str(uuid.uuid4())
    mismatch = _inspection(str(uuid.uuid4()))
    mismatch.physical_file_size = 500 * 1024 * 1024
    inspection = _inspection(project_uuid)
    monkeypatch.setattr(
        AssetIndex,
        "_inspect_path",
        staticmethod(lambda _path: inspection),
    )
    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path)
    index.load()
    project, _ = index.register_licht_asset(str(project_path))

    project_path.write_bytes(b"changed container")
    inspection = mismatch
    result = index.verify_asset(project.id)

    assert result.status == "IDENTITY_MISMATCH"
    assert result.file_size_bytes == mismatch.physical_file_size
    assert result.path_size_bytes == 0
    assert result.path_mtime_ns == 0
    stored = json.loads(library_path.read_text(encoding="utf-8"))["projects"][project.id]
    assert stored["file_size_bytes"] == mismatch.physical_file_size
    assert stored["size"] == 0
    assert stored["mtime_ns"] == 0


@pytest.mark.parametrize("status", ["MISSING", "UNREADABLE", "IDENTITY_MISMATCH", "UNSUPPORTED"])
def test_cleared_inspection_status_and_exists_round_trip(
    monkeypatch, tmp_path: Path, status: str
):
    project_path = tmp_path / "cleared.licht"
    project_path.write_bytes(b"container")
    project_uuid = str(uuid.uuid4())
    monkeypatch.setattr(
        AssetIndex, "_inspect_path", staticmethod(lambda _path: _inspection(project_uuid))
    )
    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path)
    index.load()
    project, _ = index.register_licht_asset(str(project_path))
    index._clear_runtime(project, status, "saved diagnostic")
    assert index.save()

    restored_index = AssetIndex(library_path=library_path)
    assert restored_index.load()
    restored = restored_index.get_asset(project.id)

    assert restored.status == status
    assert restored.exists is (status != "MISSING")
    assert restored.available is False


def test_v4_cached_inspection_with_changed_stat_refreshes_fields(
    monkeypatch, tmp_path: Path
):
    project_path = tmp_path / "changed.licht"
    project_path.write_bytes(b"changed project")
    project_uuid = str(uuid.uuid4())
    old = _inspection(project_uuid, commit_uuid="old-commit")
    refreshed = _inspection(project_uuid, commit_uuid="new-commit", generation=2)
    record = _cached_record(project_path, project_uuid, old)
    record["mtime_ns"] += 1
    library_path = tmp_path / "library.json"
    library_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "folders": {"default": {"path": str(tmp_path)}},
                "projects": {project_uuid: record},
                "directory_mtimes": {},
            }
        ),
        encoding="utf-8",
    )
    inspect_calls = []

    def inspect(_path, _resolve_fallback=True):
        inspect_calls.append(True)
        return refreshed

    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(inspect))
    index = AssetIndex(library_path=library_path)
    assert index.load() is True
    assert index.verify_projects_batch([project_uuid]) == 1

    project = index.get_asset(project_uuid)
    assert project is not None
    assert project.commit_uuid == "new-commit"
    assert project.generation == 2
    assert project.status == "AVAILABLE"
    assert project.inspection_restored is False
    assert project.inspection_verified is True
    assert len(inspect_calls) == 2  # Inspection and the guarded library write.


def test_v4_cached_inspection_missing_field_is_inspected(
    monkeypatch, tmp_path: Path
):
    project_path = tmp_path / "incomplete.licht"
    project_path.write_bytes(b"incomplete project")
    project_uuid = str(uuid.uuid4())
    cached = _inspection(project_uuid, commit_uuid="cached-commit")
    refreshed = _inspection(project_uuid, commit_uuid="refreshed-commit")
    record = _cached_record(project_path, project_uuid, cached)
    del record["commit_uuid"]
    library_path = tmp_path / "library.json"
    library_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "folders": {"default": {"path": str(tmp_path)}},
                "projects": {project_uuid: record},
                "directory_mtimes": {},
            }
        ),
        encoding="utf-8",
    )
    inspect_calls = []

    def inspect(_path, _resolve_fallback=True):
        inspect_calls.append(True)
        return refreshed

    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(inspect))
    index = AssetIndex(library_path=library_path)
    assert index.load() is True
    assert index.verify_projects_batch([project_uuid]) == 1

    project = index.get_asset(project_uuid)
    assert project is not None
    assert project.commit_uuid == "refreshed-commit"
    assert project.inspection_verified is True
    assert len(inspect_calls) == 2  # Inspection and the guarded library write.


def test_malformed_v3_catalog_restores_previous_catalog(monkeypatch, tmp_path: Path):
    original_path = tmp_path / "original.licht"
    original_path.write_bytes(b"original")
    original_uuid = str(uuid.uuid4())
    _install_inspections(monkeypatch, {original_path.name: _inspection(original_uuid)})
    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path)
    index.load()
    index.register_licht_asset(str(original_path))
    before = index.assets
    disk_before = library_path.read_text(encoding="utf-8")
    library_path.write_text(
        json.dumps({"schema_version": 3, "folders": {"default": {"path": str(tmp_path)}}}),
        encoding="utf-8",
    )

    assert index.load() is False
    assert index.assets == before
    assert index.load_issues == []
    assert library_path.read_text(encoding="utf-8") != disk_before


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
    index.load()
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
    folder_path = tmp_path / "Projects"
    folder_path.mkdir()
    unsaved_folder_path = tmp_path / "Unsaved"
    unsaved_folder_path.mkdir()
    original_path = folder_path / "original.licht"
    relink_path = folder_path / "relink.licht"
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
    index.load()
    folder = index.add_folder(str(folder_path))
    project, _ = index.register_licht_asset(
        str(original_path), folder_id=folder.id
    )
    before_folders = index.folders
    before_assets = index.assets
    monkeypatch.setattr(index, "save", lambda: False)

    assert index.add_folder(str(unsaved_folder_path)) is None
    assert index.update_asset(project.id, name="Unsaved project") is None
    assert index.relink_asset(project.id, str(relink_path)) is False
    assert index.delete_folder(folder.id) == 0
    assert index.register_licht_asset(str(new_path)) == (None, False)

    assert index.folders == before_folders
    assert index.assets == before_assets


def test_folder_scan_duplicate_does_not_adopt_when_locator_is_offline(
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
    index.load()
    project, _ = index.register_licht_asset(str(original))
    original.unlink()

    library_path = tmp_path / "library.json"
    result = scan_asset_folder(index, "default", str(watched))

    relocated = index.get_asset(project.id)
    assert result.already_cataloged >= 1
    assert relocated.path == str(duplicate)
    assert relocated.status == "AVAILABLE"
    assert relocated.relocation_candidate == ""
    assert "relocation_candidate" not in json.loads(library_path.read_text(encoding="utf-8"))[
        "projects"
    ][project.id]


def test_storage_resolution_keeps_unwritable_native_path(monkeypatch, tmp_path):
    from lfs_plugins import asset_index

    native = tmp_path / "blocked"
    native.write_text("not a directory")
    monkeypatch.delenv("LFS_ASSET_MANAGER_DIR", raising=False)
    monkeypatch.setenv("LFS_RESOLVED_ASSET_LIBRARY_DIR", str(native))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    assert asset_index.resolve_asset_manager_storage_path() == native
    with pytest.raises(OSError):
        AssetIndex(library_path=native / "library.json")
    assert not (tmp_path / "appdata").exists()


def test_safe_mode_storage_resolution_does_not_write(monkeypatch, tmp_path):
    from lfs_plugins import asset_index

    native = tmp_path / "native" / "asset_library"
    monkeypatch.delenv("LFS_ASSET_MANAGER_DIR", raising=False)
    monkeypatch.setenv("LFS_RESOLVED_ASSET_LIBRARY_DIR", str(native))
    monkeypatch.setenv("LFS_SAFE_MODE", "1")
    assert asset_index.resolve_asset_manager_storage_path() == native
    assert not native.exists()


def test_preview_capture_stays_home_and_cleans_failure(monkeypatch, tmp_path):
    from lfs_plugins import asset_storage

    monkeypatch.setenv("LFS_HOME", str(tmp_path / "Lichtfeld 日本語"))
    with pytest.raises(RuntimeError, match="capture failed"):
        with asset_storage.preview_capture("project") as target:
            assert target.parent == asset_storage.lichtfeld_home() / "cache" / "previews"
            target.write_bytes(b"png")
            raise RuntimeError("capture failed")
    assert not target.exists()


def test_preview_cache_evicts_oldest_and_removed_project(monkeypatch, tmp_path):
    import os
    from lfs_plugins import asset_storage

    monkeypatch.setenv("LFS_HOME", str(tmp_path))
    monkeypatch.setattr(asset_storage, "PREVIEW_CACHE_BYTES", 8)
    directory = tmp_path / "cache" / "previews"
    directory.mkdir(parents=True)
    paths = [directory / (asset_storage._preview_prefix(key) + "old.png") for key in ("old", "keep", "removed")]
    for index, path in enumerate(paths):
        path.write_bytes(b"1234")
        os.utime(path, ns=(index + 1, index + 1))
    asset_storage.prune_previews()
    assert not paths[0].exists()
    assert paths[1].exists() and paths[2].exists()
    asset_storage.prune_previews(["removed"])
    assert paths[1].exists() and not paths[2].exists()


def test_safe_mode_does_not_create_default_asset_directory(monkeypatch, tmp_path: Path):
    missing = tmp_path / "missing-assets"
    monkeypatch.setenv("LFS_SAFE_MODE", "1")

    AssetIndex(
        library_path=tmp_path / "catalog" / "library.json",
        default_folder_path=missing,
    )

    assert missing.exists() is False


def test_default_catalog_migrates_from_appdata_location(monkeypatch, tmp_path: Path):
    project_path = tmp_path / "legacy.licht"
    project_path.write_bytes(b"container")
    project_uuid = str(uuid.uuid4())
    _install_inspections(monkeypatch, {project_path.name: _inspection(project_uuid)})
    native = tmp_path / "native" / "asset_library"
    appdata = tmp_path / "appdata"
    legacy_library = appdata / "LichtFeldStudio" / "asset_manager" / "library.json"
    legacy_library.parent.mkdir(parents=True)
    original = json.dumps(
        {
            "folders": {"default": {"name": "Default"}},
            "assets": {
                "legacy": {
                    "name": "Legacy",
                    "absolute_path": str(project_path),
                }
            },
        }
    )
    legacy_library.write_text(original, encoding="utf-8")
    monkeypatch.delenv("LFS_ASSET_MANAGER_DIR", raising=False)
    monkeypatch.delenv("LFS_SAFE_MODE", raising=False)
    monkeypatch.setenv("LFS_RESOLVED_ASSET_LIBRARY_DIR", str(native))
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    index = AssetIndex()
    assert index.load() is True

    assert index.library_path == native / "library.json"
    assert index.assets[project_uuid]["path"] == str(project_path)
    legacy_backup = _legacy_backup_path(index.library_path)
    assert json.loads(legacy_backup.read_text(encoding="utf-8"))["assets"]["legacy"][
        "name"
    ] == "Legacy"
    assert legacy_backup.read_text(encoding="utf-8") == original
    assert index.save() is True
    assert index.save() is True
    assert legacy_backup.read_text(encoding="utf-8") == original


def test_inspection_maps_repair_only_and_unsupported_newer_status(
    monkeypatch, tmp_path: Path
):
    repair = tmp_path / "repair.licht"
    newer = tmp_path / "newer.licht"
    other = tmp_path / "other.licht"
    for path in (repair, newer, other):
        path.write_bytes(b"container")
    repair_uuid = str(uuid.uuid4())
    newer_uuid = str(uuid.uuid4())
    other_uuid = str(uuid.uuid4())
    _install_inspections(
        monkeypatch,
        {
            repair.name: _inspection(repair_uuid, open_state="REPAIR_ONLY"),
            newer.name: _inspection(newer_uuid, open_state="UNSUPPORTED_NEWER"),
            other.name: _inspection(other_uuid, open_state="UNKNOWN_FUTURE"),
        },
    )
    index = AssetIndex(library_path=tmp_path / "library.json")
    index.load()

    repair_project, _ = index.register_licht_asset(str(repair))
    newer_project, _ = index.register_licht_asset(str(newer))
    other_project, _ = index.register_licht_asset(str(other))

    assert repair_project.status == "REPAIR_ONLY"
    assert repair_project.available is False
    assert newer_project.status == "UNSUPPORTED_NEWER"
    assert newer_project.available is False
    assert other_project.status == "UNSUPPORTED"
    assert other_project.available is False
    stored = json.loads((tmp_path / "library.json").read_text(encoding="utf-8"))
    assert stored["projects"][repair_uuid]["status"] == "REPAIR_ONLY"


def test_asset_snapshot_is_cached_but_explicit_verify_reinspects(monkeypatch, tmp_path: Path):
    project_path = tmp_path / "project.licht"
    project_path.write_bytes(b"container")
    project_uuid = str(uuid.uuid4())
    calls = []

    def inspect(_path):
        calls.append(1)
        return _inspection(project_uuid)

    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(inspect))
    index = AssetIndex(library_path=tmp_path / "library.json")
    index.load()
    project, _ = index.register_licht_asset(str(project_path))
    assert project is not None
    snapshot = index.assets
    assert index.assets is snapshot
    calls_before_verify = len(calls)
    index.verify_asset(project.id)
    assert len(calls) == calls_before_verify + 2
    index.update_asset(project.id, save=False, name="Renamed")
    assert index.assets is not snapshot


def test_fallback_preview_path_is_cached_in_catalog(monkeypatch, tmp_path: Path):
    project_path = tmp_path / "garden.licht"
    project_path.write_bytes(b"container")
    project_uuid = str(uuid.uuid4())
    fallback = str(tmp_path / "images" / "000.png")
    inspection = _inspection(project_uuid, has_preview=False)
    inspection.fallback_preview_path = fallback
    _install_inspections(monkeypatch, {project_path.name: inspection})
    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path)
    index.load()

    project, created = index.register_licht_asset(str(project_path))

    assert created is True
    assert project.fallback_preview_path == fallback
    assert project.to_dict()["fallback_preview_path"] == fallback
    assert project.to_storage_dict()["fallback_preview_path"] == fallback
    stored = json.loads(library_path.read_text(encoding="utf-8"))
    assert stored["projects"][project.id]["fallback_preview_path"] == fallback

    missing_field = Project(
        project_uuid=str(uuid.uuid4()),
        name="Bare",
        path=str(tmp_path / "bare.licht"),
        folder_id="default",
        fallback_preview_path=fallback,
    )
    index._apply_inspection(missing_field, _inspection(missing_field.project_uuid))
    assert not hasattr(_inspection(missing_field.project_uuid), "fallback_preview_path")
    assert missing_field.fallback_preview_path == ""

    project_path.unlink()
    cleared = index.verify_asset(project.id)
    assert cleared.fallback_preview_path == ""
    assert cleared.to_dict()["fallback_preview_path"] == ""
