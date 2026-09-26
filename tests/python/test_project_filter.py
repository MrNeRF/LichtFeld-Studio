# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Projects filters use durable facts from native project inspection."""

import json
import uuid
from types import SimpleNamespace

from lfs_plugins.asset_index import AssetIndex
from lfs_plugins.asset_watch import scan_asset_folder


def test_catalog_persists_filter_facts_and_refreshes_old_entries(monkeypatch, tmp_path):
    path = tmp_path / "project-a.licht"
    path.touch()
    facts = SimpleNamespace(
        project_uuid=str(uuid.uuid4()),
        file_uuid=str(uuid.uuid4()),
        commit_uuid=str(uuid.uuid4()),
        generation=1,
        created_at_unix_ns=100,
        saved_at_unix_ns=200,
        physical_file_size=path.stat().st_size,
        role=SimpleNamespace(name="MASTER"),
        open_state=SimpleNamespace(name="OPEN"),
        has_preview=False,
        has_checkpoint=True,
        has_dataset=True,
    )
    calls = []

    def inspect(*_args):
        calls.append(True)
        return facts

    monkeypatch.setattr(AssetIndex, "_inspect_path", staticmethod(inspect))
    library = tmp_path / "library.json"
    index = AssetIndex(library_path=library, default_folder_path=tmp_path)
    assert index.load()
    project, _ = index.register_licht_asset(str(path))
    assert project.inspection["has_checkpoint"] is True
    assert project.inspection["has_dataset"] is True
    stored = json.loads(library.read_text())
    assert stored["projects"][project.id]["inspection"]["has_checkpoint"] is True

    stored["projects"][project.id]["inspection"] = {"version": 1}
    library.write_text(json.dumps(stored))
    calls.clear()
    old = AssetIndex(library_path=library, default_folder_path=tmp_path)
    assert old.load()
    assert old.verify_projects_batch([project.id]) == 1
    refreshed = old.get_asset(project.id)
    assert refreshed.inspection["has_checkpoint"] is True
    assert refreshed.inspection["has_dataset"] is True
    assert calls

    stored = json.loads(library.read_text())
    stored["projects"][project.id]["inspection"] = {"version": 1}
    library.write_text(json.dumps(stored))
    calls.clear()
    old = AssetIndex(library_path=library, default_folder_path=tmp_path)
    assert old.load()
    scan_asset_folder(old, "default", str(tmp_path))
    assert old.get_asset(project.id).inspection["has_checkpoint"] is True
    assert old.get_asset(project.id).inspection["has_dataset"] is True
    assert calls
