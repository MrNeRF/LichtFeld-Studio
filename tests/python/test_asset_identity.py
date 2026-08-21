# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager .licht identity and fingerprint binding regressions."""

import json
import os
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

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


def test_fingerprint_bindings_report_typed_dispositions(lf, tmp_path: Path):
    project = tmp_path / "scene.licht"
    project.write_bytes(b"licht-project-content")

    expected = lf.io.fingerprint_path(project)
    assert expected.kind == lf.io.FingerprintKind.FILE
    assert len(expected.head_xxh3.to_hex()) == 32

    matched = lf.io.check_fingerprint(project, expected)
    assert matched.matches is True
    assert matched.disposition == lf.io.FingerprintDisposition.MATCH_FAST_PATH

    stat = project.stat()
    os.utime(project, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    refreshed = lf.io.check_fingerprint(project, expected)
    assert refreshed.matches is True
    assert refreshed.disposition == lf.io.FingerprintDisposition.MATCH_MTIME_REFRESHED
    assert refreshed.observed is not None

    project.write_bytes(b"different-content-now")
    mismatch = lf.io.check_fingerprint(project, expected)
    assert mismatch.matches is False
    assert mismatch.disposition == lf.io.FingerprintDisposition.CONTENT_MISMATCH


def test_references_chapter_bindings_verify_and_relink(lf, tmp_path: Path):
    original = tmp_path / "original.licht"
    relocated = tmp_path / "relocated.licht"
    original.write_bytes(b"same-project")
    shutil.copy2(original, relocated)

    record = lf.io.ReferenceRecord()
    record.uuid = str(uuid.uuid4())
    record.key = "asset-manager-project"
    record.kind = "project"
    original_locator = lf.io.ReferenceLocator()
    original_locator.preferred = str(original)
    original_locator.base = lf.io.LocatorBase.ABSOLUTE
    record.locator = original_locator
    record.fingerprint = lf.io.fingerprint_path(original)

    chapter = lf.io.ReferencesChapter()
    chapter.upsert(record)
    assert chapter.find(record.uuid).key == record.key

    check = chapter.verify_and_refresh(record.uuid, original)
    assert check.matches is True

    locator = lf.io.ReferenceLocator()
    locator.preferred = str(relocated)
    locator.base = lf.io.LocatorBase.ABSOLUTE
    chapter.relink(record.uuid, locator, relocated)
    assert chapter.find(record.uuid).locator.preferred == str(relocated)
    assert "asset-manager-project" in chapter.to_json()


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

    assert migrated == {
        "schema_version": 2,
        "folders": {
            "default": {
                "name": "Default",
                "watch_directories": [str(tmp_path)],
            }
        },
        "projects": {
            project_uuid: {
                "name": "Custom legacy name",
                "path": str(project_path),
                "folder_id": "default",
            }
        },
    }


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
