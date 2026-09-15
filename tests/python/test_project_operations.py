"""Exercise closed-file project operations through the native Python bindings."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


def _fixture() -> Path:
    configured = os.environ.get("LFS_PROJECT_OPERATIONS_FIXTURE")
    if configured:
        return Path(configured)
    for parent in Path(__file__).parents:
        candidate = parent / ".codex_tmp/am_concept/testhome/projects/bonsai.licht"
        if candidate.is_file():
            return candidate
    return Path(__file__).parents[2] / "missing-project.licht"


@pytest.fixture(scope="module")
def native_io():
    try:
        from lichtfeld import io
    except ImportError as error:
        pytest.skip(f"native lichtfeld.io is unavailable: {error}")
    return io


def test_closed_file_operations(native_io, tmp_path):
    source = _fixture()
    if not source.is_file():
        pytest.skip(f"operations fixture is unavailable: {source}")
    path = tmp_path / "copy.licht"
    shutil.copy2(source, path)

    canceled = native_io.verify_project_file(path, cancel=lambda: True)
    assert canceled.status is native_io.ProjectVerificationStatus.CANCELED
    verified = native_io.verify_project_file(path)
    assert verified.status is native_io.ProjectVerificationStatus.VERIFIED

    plan = native_io.plan_reduce_size(path)
    assert plan.physical_size > 0
    assert plan.input_commit_uuid
    reduced = native_io.reduce_size(
        path, {"drop_unbound_checkpoints": False, "drop_embedded_dataset": False}
    )
    assert reduced.card.physical_file_size > 0
    assert reduced.recovery_copy.is_file()

    restored = native_io.restore_save(path, 1, tmp_path / "restored.licht")
    assert restored.project_uuid != native_io.inspect_project_card(path).project_uuid

    png = b"\x89PNG\r\n\x1a\n"
    native_io.set_project_preview(path, png)
    native_io.set_project_license(path, "CC-BY-4.0", "Python test")
    titled = native_io.set_project_title(path, "Python operation copy")
    assert titled.title == "Python operation copy"
    assert native_io.inspect_project_details(path).license.identifier == "CC-BY-4.0"

    compacted = native_io.compact_project_file(path)
    assert "older save points" in compacted.diagnostic
    native_io.clear_project_license(path)
    assert native_io.inspect_project_details(path).license is None


def test_contents_removals_persist_until_compaction(native_io, tmp_path):
    import base64
    import json
    source = _fixture()
    if not source.is_file():
        pytest.skip(f'operations fixture is unavailable: {source}')
    path = tmp_path / 'contents.licht'
    shutil.copy2(source,path)
    native_io.set_project_title(path,'Before removal')
    native_io.set_project_license(path,'CC-BY-4.0','Credit: Studio')
    png=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aGNcAAAAASUVORK5CYII=')
    native_io.set_project_preview(path,png)
    details=native_io.inspect_project_details(path)
    current=details.card.generation
    with pytest.raises(Exception,match='current save'):
        native_io.reduce_size(path,dict(compact=False,drop_unbound_checkpoints=False,save_generation=current))
    oldest=details.save_history[0].generation
    size=path.stat().st_size
    reduced=native_io.reduce_size(path,dict(compact=False,drop_unbound_checkpoints=False,save_generation=oldest))
    assert reduced.recovery_copy.is_file() and reduced.bytes_reclaimed==0
    assert path.stat().st_size>=size
    pending=json.loads(native_io.inspect_project_details(path).manifest['contents_removals'])['rows']
    assert pending[0]['id']==f'save:{oldest}'
    with pytest.raises(Exception,match='removed'):
        native_io.restore_save(path,oldest,tmp_path/'removed-save.licht')
    native_io.reduce_size(path,dict(compact=False,drop_unbound_checkpoints=False,drop_thumbnail=True))
    assert not native_io.inspect_project_details(path).card.has_preview
    native_io.clear_project_license(path)
    details=native_io.inspect_project_details(path)
    assert details.license is None
    assert {r['kind'] for r in json.loads(details.manifest['contents_removals'])['rows']}=={'save','thumbnail','license'}
    assert native_io.verify_project_file(path).status is native_io.ProjectVerificationStatus.VERIFIED
    before=path.stat().st_size
    native_io.compact_project_file(path)
    assert path.stat().st_size<before
    assert 'contents_removals' not in native_io.inspect_project_details(path).manifest


def test_restore_replaces_same_path_with_new_identity(native_io,tmp_path):
    source=_fixture()
    if not source.is_file():pytest.skip(f'operations fixture is unavailable: {source}')
    path=tmp_path/'restore.licht'
    shutil.copy2(source,path)
    native_io.set_project_title(path,'Old title')
    old=native_io.inspect_project_card(path)
    native_io.set_project_title(path,'New title')
    restored=native_io.restore_save(path,old.generation,path)
    assert restored.project_uuid!=old.project_uuid
    assert restored.title=='Old title'
    assert native_io.verify_project_file(path).status is native_io.ProjectVerificationStatus.VERIFIED
    assert len(list(tmp_path.glob('*.licht')))==1
    assert len(list(tmp_path.glob('*.bak')))==1


def test_removing_bound_checkpoint_preserves_visible_model(native_io, tmp_path):
    import json

    configured = os.environ.get("LFS_PROJECT_CHECKPOINT_FIXTURE")
    source = Path(configured) if configured else _fixture().parent / "mrnf/stump/project.licht"
    if not source.is_file():
        pytest.skip(f"checkpoint fixture is unavailable: {source}")
    path = tmp_path / "checkpoint.licht"
    shutil.copy2(source, path)
    before = native_io.inspect_project_details(path)
    checkpoint = next(cp for cp in before.retained_checkpoints if cp.retained and cp.binds_scene_graph)

    result = native_io.reduce_size(path, {
        "compact": False,
        "drop_unbound_checkpoints": False,
        "checkpoint_uuid": str(checkpoint.instance_uuid),
    })
    after = native_io.inspect_project_details(path)
    assert result.checkpoints_removed == 1
    assert result.bytes_reclaimed == 0
    assert result.recovery_copy.is_file()
    assert after.scene_graph.training_node_id is None
    assert after.scene_graph.node_counts_by_type == before.scene_graph.node_counts_by_type
    assert not any(cp.retained and cp.instance_uuid == checkpoint.instance_uuid for cp in after.retained_checkpoints)
    assert any(str(part.fourcc) == "SPLT" and part.stored_bytes > 0 for part in after.chapters)
    removed = json.loads(after.manifest["contents_removals"])["rows"]
    assert any(row["id"] == f"checkpoint:{checkpoint.instance_uuid}" for row in removed)
    assert native_io.verify_project_file(path).status is native_io.ProjectVerificationStatus.VERIFIED


def test_rebinding_repeatedly_keeps_recovery_copies_out_of_catalog(native_io, tmp_path):
    configured = os.environ.get("LFS_PROJECT_CHECKPOINT_FIXTURE")
    source = Path(configured) if configured else _fixture().parent / "mrnf/stump/project.licht"
    if not source.is_file():
        pytest.skip(f"checkpoint fixture is unavailable: {source}")
    path = tmp_path / "resume.licht"
    shutil.copy2(source, path)
    details = native_io.inspect_project_details(path)
    checkpoint = next(cp for cp in details.retained_checkpoints if cp.retained and cp.binds_scene_graph)
    first = native_io.rebind_checkpoint(path, str(checkpoint.instance_uuid))
    second = native_io.rebind_checkpoint(path, str(checkpoint.instance_uuid))
    assert first.commit_uuid != second.commit_uuid
    assert first.project_uuid == second.project_uuid == details.card.project_uuid
    assert len(list(tmp_path.glob("*.licht"))) == 1
    assert len(list(tmp_path.glob("*.bak"))) == 2
    assert native_io.verify_project_file(path).status is native_io.ProjectVerificationStatus.VERIFIED


def test_chosen_thumbnail_survives_other_contents_edits(native_io, tmp_path):
    import struct
    import zlib

    def png(rgb):
        def chunk(kind, payload):
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

        return (b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(b"\x00" + bytes(rgb) * 2))
                + chunk(b"IEND", b""))

    source = _fixture()
    if not source.is_file():
        pytest.skip(f"operations fixture is unavailable: {source}")
    path = tmp_path / "preview.licht"
    shutil.copy2(source, path)
    dataset = tmp_path / "dataset"
    for folder in ("images", "images_4"):
        images = dataset / folder
        images.mkdir(parents=True)
        (images / "first.png").write_bytes(png((255, 0, 0)))
    native_io.set_dataset_reference(path, dataset, True)
    native_io.preview_from_first_dataset_image(path)
    chosen = png((0, 0, 255))
    assert native_io.read_preview(path) != chosen

    native_io.set_project_preview(path, chosen)
    assert native_io.read_preview(path) == chosen
    native_io.set_project_license(path, "CC-BY-4.0", "Credit: Studio")
    assert native_io.read_preview(path) == chosen
    native_io.reduce_size(path, {
        "compact": False,
        "drop_unbound_checkpoints": False,
        "drop_thumbnail": True,
    })
    assert not native_io.inspect_project_card(path).has_preview
    native_io.clear_project_license(path)
    assert not native_io.inspect_project_card(path).has_preview
    assert native_io.verify_project_file(path).status is native_io.ProjectVerificationStatus.VERIFIED
