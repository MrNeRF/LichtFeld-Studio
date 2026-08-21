# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager .licht identity and fingerprint binding regressions."""

import json
import os
import shutil
import uuid
from pathlib import Path

from lfs_plugins.asset_index import AssetIndex
from lfs_plugins.asset_watch import scan_watch_directories


ROOT = Path(__file__).resolve().parents[2]


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


def test_catalog_deduplicates_licht_projects_by_content(lf, tmp_path: Path):
    first_path = tmp_path / "first.licht"
    copied_path = tmp_path / "copy.licht"
    first_path.write_bytes(b"one durable project identity")
    shutil.copy2(first_path, copied_path)

    index = AssetIndex(library_path=tmp_path / "library.json")
    index.ensure_default_catalog()

    first, first_created = index.register_licht_asset(str(first_path))
    duplicate, duplicate_created = index.register_licht_asset(str(copied_path))

    assert first is not None
    assert first_created is True
    assert duplicate_created is False
    assert duplicate.id == first.id
    assert len(index.list_projects()) == 1
    assert duplicate.absolute_path == str(first_path)

    catalog = json.loads((tmp_path / "library.json").read_text(encoding="utf-8"))
    stored = catalog["assets"][first.id]
    assert "type" not in stored
    assert "role" not in stored
    assert "tags" not in stored


def test_catalog_rejects_non_licht_paths(lf, tmp_path: Path):
    unsupported = tmp_path / "unsupported.txt"
    unsupported.write_bytes(b"not a LichtFeld project")
    index = AssetIndex(library_path=tmp_path / "library.json")
    index.ensure_default_catalog()

    asset, created = index.register_licht_asset(str(unsupported))

    assert asset is None
    assert created is False
    assert index.list_projects() == []


def test_catalog_verifies_project_content_changes(lf, tmp_path: Path):
    project = tmp_path / "project.licht"
    project.write_bytes(b"original project bytes")

    library_path = tmp_path / "library.json"
    index = AssetIndex(library_path=library_path)
    index.ensure_default_catalog()
    licht_asset, _ = index.register_licht_asset(str(project))

    project.write_bytes(b"changed project bytes!")
    verified = index.verify_asset(licht_asset.id)
    assert verified.verification_disposition == "CONTENT_MISMATCH"
    assert verified.exists is False

    assert [asset.id for asset in index.list_projects()] == [licht_asset.id]


def test_deleting_last_project_keeps_default_import_folder(lf, tmp_path: Path):
    first = tmp_path / "first.licht"
    second = tmp_path / "second.licht"
    first.write_bytes(b"first project")
    second.write_bytes(b"second project")

    index = AssetIndex(library_path=tmp_path / "library.json")
    index.ensure_default_catalog()
    registered, _ = index.register_licht_asset(str(first))

    assert index.delete_asset(registered.id) is True
    replacement, created = index.register_licht_asset(str(second))
    assert created is True
    assert replacement is not None


def test_watched_directory_scan_registers_only_licht_projects(lf, tmp_path: Path):
    watched = tmp_path / "watched"
    nested = watched / "nested"
    nested.mkdir(parents=True)
    first = watched / "a.licht"
    duplicate = watched / "b.licht"
    second = nested / "c.LICHT"
    first.write_bytes(b"first project")
    shutil.copy2(first, duplicate)
    second.write_bytes(b"second project")
    (watched / "legacy.ply").write_bytes(b"legacy splat")
    (nested / "checkpoint.ckpt").write_bytes(b"legacy checkpoint")

    index = AssetIndex(library_path=tmp_path / "library.json")
    index.ensure_default_catalog()
    result = scan_watch_directories(index, "default", [str(watched)])

    assert result.discovered == 3
    assert result.added == 2
    assert result.already_cataloged == 1
    assert result.failed == 0
    assert len(index.list_projects()) == 2
    assert all(Path(asset.absolute_path).suffix.lower() == ".licht" for asset in index.list_projects())


def test_asset_library_binding_returns_canonical_path(lf):
    assert Path(lf.io.asset_library_dir()).name == "asset_library"


def test_asset_manager_ui_exposes_only_project_import_and_open_actions():
    rml = (
        ROOT / "src/visualizer/gui/rmlui/resources/asset_manager.rml"
    ).read_text(encoding="utf-8")

    assert 'data-event-click="on_import_project"' in rml
    assert 'data-asset-action="load"' in rml
    assert 'data-folder-action="watch_dirs"' in rml
    assert rml.count('data-event-click="on_import_project"') == 1
