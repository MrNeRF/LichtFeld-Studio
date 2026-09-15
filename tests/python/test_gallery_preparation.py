# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
import io
import json
from pathlib import Path
import struct
import threading
import uuid

import pytest

from lfs_plugins import gallery_preparation, gallery_sync
from lfs_plugins.portal_gallery import GalleryTransferCanceled
IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
from test_gallery_sync import Client, connected, finish

def staging(root):
    import shutil
    directory = root / (str(uuid.uuid4()) + ".scene")
    fixture = Path(__file__).parents[1] / 'data' / 'portable-sog.licht'
    gallery_preparation.unpack_project(root, fixture, directory)
    shutil.copyfile(fixture, directory / 'project.licht')
    return directory

def native_service(root, monkeypatch):
    monkeypatch.setattr(Client, "_request", lambda self, *args: {"storageHosts": ["portal.example"],
        "id": self.account.owner, "gallerySyncVersion": 1, "revisionDomains": 1, "sourceFormats": ["ply", "licht"]})
    return connected(root, monkeypatch)

def uploaded():
    return {"scene": {"contentRevision": "new", "metadataRevision": "new", "id": "remote-scene", "revision": "new", "title": "Scene", "sourceFormat": "licht"}}

def test_transfer_resume_does_not_repackage_and_discard_cleans_staging(tmp_path, monkeypatch):
    service = native_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    monkeypatch.setattr(Client, "upload", lambda *args, **kwargs: (_ for _ in ()).throw(GalleryTransferCanceled()), raising=False)
    identifier = service.queue_prepared_upload( directory, {"title": "Scene", "viewerSettings": {"environment": {"exposure": -1.25, "rotation": 123}}}, "project")
    finish(service)
    job = service.snapshot()["jobs"][0]
    assert job["packaged"] and directory.exists()
    service.resume(identifier)
    finish(service)
    assert service.snapshot()["jobs"][0]["status"] == "paused"
    service.discard(identifier)
    finish(service)
    assert not directory.exists() and not Path(job["path"]).exists()

@pytest.mark.parametrize("damage", ["escape", "symlink", "extra", "duplicate", "destination"])
def test_untrusted_preparation_never_uploads_or_removes_unrelated_files(tmp_path, monkeypatch, damage):
    service = native_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    private = tmp_path / "private.txt"
    private.write_bytes(b"keep private")
    manifest = directory / "manifest.json"
    data = json.loads(manifest.read_text())
    if damage == "escape":
        data["nodes"][0]["path"] = "../private.txt"
        manifest.write_text(json.dumps(data))
    elif damage == "symlink":
        (directory / "0.sog").unlink()
        (directory / "0.sog").symlink_to(private)
    elif damage == "extra":
        (directory / "private.txt").write_bytes(b"unrecognized file")
    elif damage == "duplicate":
        manifest.write_text('{"version":1,"version":1,"nodes":[]}')
    elif damage == "destination":
        directory.with_suffix(".licht").symlink_to(private)
    monkeypatch.setattr(Client, "upload", lambda *args, **kwargs: pytest.fail("Invalid staging must not upload"), raising=False)
    service.queue_prepared_upload( directory, {"title": "Scene", "viewerSettings": {"environment": {"exposure": -1.25, "rotation": 123}}}, "project")
    finish(service)
    assert service.snapshot()["jobs"][0]["status"] == "error"
    assert private.read_bytes() == b"keep private"
    assert not service.snapshot()["links"]

def test_project_preparation_requires_advertised_capability(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="does not support gallery sync"):
        service.queue_prepared_upload( staging(tmp_path), {"title": "Scene", "viewerSettings": {"environment": {"exposure": -1.25, "rotation": 123}}}, "project")

def test_queue_is_durable_before_worker_start_and_recovers_start_failure(tmp_path, monkeypatch):
    service = native_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    monkeypatch.setattr(service, "resume", lambda _: (_ for _ in ()).throw(RuntimeError("thread unavailable")))
    identifier = service.queue_prepared_upload( directory, {"title": "Scene", "viewerSettings": {"environment": {"exposure": -1.25, "rotation": 123}}}, "project")
    saved = json.loads((tmp_path / "sync.json").read_text())
    jobs = [j for bucket in saved["accounts"].values() for j in bucket["jobs"]]
    assert len(jobs) == 1 and jobs[0]["id"] == identifier and jobs[0]["status"] == "queued"
    assert directory.exists()
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    monkeypatch.setattr(Client, "upload", lambda *args, **kwargs: uploaded(), raising=False)
    restarted.resume(identifier)
    finish(restarted)
    assert restarted.snapshot()["jobs"][0]["status"] == "completed"

def test_changed_journal_refuses_handoff_before_accepting_the_snapshot(tmp_path, monkeypatch):
    service = native_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    journal = tmp_path / "sync.json"
    journal.write_text(journal.read_text() + "\n")
    with pytest.raises(ValueError, match="Another LichtFeld Studio window updated"):
        service.queue_prepared_upload( directory, {"title": "Scene", "viewerSettings": {"environment": {"exposure": -1.25, "rotation": 123}}}, "project")
    assert not any(bucket["jobs"] for bucket in service._data["accounts"].values())
    assert directory.exists()  # Caller retains ownership when queueing raises.

@pytest.mark.parametrize("redirect", ["outside", "symlink"])
def test_resumed_packaged_upload_revalidates_ownership_before_reading(tmp_path, monkeypatch, redirect):
    service = native_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    monkeypatch.setattr(Client, "upload", lambda *args, **kwargs: (_ for _ in ()).throw(GalleryTransferCanceled()), raising=False)
    identifier = service.queue_prepared_upload( directory, {"title": "Scene", "viewerSettings": {"environment": {"exposure": -1.25, "rotation": 123}}}, "project")
    finish(service)
    private = tmp_path / "private.licht"
    private.write_bytes(b"never upload")
    saved = json.loads((tmp_path / "sync.json").read_text())
    job = next(j for bucket in saved["accounts"].values() for j in bucket["jobs"])
    assert job["packaged"]
    if redirect == "outside":
        job["path"] = str(private)
    else:
        Path(job["path"]).unlink()
        Path(job["path"]).symlink_to(private)
    (tmp_path / "sync.json").write_text(json.dumps(saved))
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    monkeypatch.setattr(Client, "upload", lambda *args, **kwargs: pytest.fail("Must reject before reading upload"))
    restarted.resume(identifier)
    finish(restarted)
    assert restarted.snapshot()["jobs"][0]["status"] == "error"
    assert private.read_bytes() == b"never upload"

def test_discard_reports_files_kept_when_staging_contains_unrecognized_data(tmp_path, monkeypatch):
    service = native_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    kept = directory / "unrecognized.txt"
    kept.write_text("keep for review")
    identifier = service.queue_prepared_upload( directory, {"title": "Scene", "viewerSettings": {"environment": {"exposure": -1.25, "rotation": 123}}}, "project")
    finish(service)
    service.discard(identifier)
    finish(service)
    job = service.snapshot()["jobs"][0]
    assert job["status"] == "canceled" and job["cleanupPending"]
    assert "files were kept" in job["message"] and "recovery folder" in job["message"]
    assert kept.read_text() == "keep for review"

def test_cleanup_failure_cannot_turn_published_upload_into_retry(tmp_path, monkeypatch):
    service = native_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    monkeypatch.setattr(Client, "upload", lambda *args, **kwargs: uploaded(), raising=False)
    monkeypatch.setattr(service, "_retire_export", lambda job: (_ for _ in ()).throw(RuntimeError("cleanup interrupted")))
    service.queue_prepared_upload( directory, {"title": "Scene", "viewerSettings": {"environment": {"exposure": -1.25, "rotation": 123}}}, "project")
    finish(service)
    job = service.snapshot()["jobs"][0]
    assert job["status"] == "completed" and job["cleanupPending"]
    assert service.snapshot()["links"]["project"]["sceneId"] == "remote-scene"
    assert directory.exists()
    with pytest.raises(ValueError):
        service.resume(job["id"])

def test_current_publishing_requires_a_fresh_native_project(tmp_path, monkeypatch):
    service = native_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    (directory / "project.licht").unlink()
    with pytest.raises(ValueError, match="fresh .licht"):
        service.queue_prepared_upload(directory, {"title": "Scene", "viewerSettings": {"environment": {"exposure": -1.25, "rotation": 123}}}, "project")
    assert service.snapshot()["jobs"] == []

def test_current_publishing_uploads_only_the_fresh_licht(tmp_path, monkeypatch):
    service = native_service(tmp_path, monkeypatch)
    service._source_formats = ['licht']
    fixture = Path(__file__).parents[1] / 'data' / 'portable-sog.licht'
    directory = tmp_path / (str(uuid.uuid4()) + '.scene')
    gallery_preparation.unpack_project(tmp_path, fixture, directory)
    original = fixture.read_bytes()
    (directory / 'project.licht').write_bytes(original)
    def upload(self, path, metadata, **kwargs):
        assert Path(path).suffix == '.licht'
        assert Path(path).read_bytes() == original
        return {'scene': {"contentRevision": 'new', "metadataRevision": 'new', 'id': 'remote-scene', 'revision': 'new', 'title': 'Scene', 'sourceFormat': 'licht'}}
    monkeypatch.setattr(Client, 'upload', upload, raising=False)
    service.queue_prepared_upload(directory, {'title': 'Scene', 'viewerSettings': {'environment': {'exposure': -1.25, 'rotation': 123}}}, 'project')
    finish(service)
    job = service.snapshot()['jobs'][0]
    assert job['status'] == 'completed', job['message']
    assert not directory.exists() and not Path(job['path']).exists()

def test_staging_accepts_spz_sidecar_payload(tmp_path):
    directory = tmp_path / (str(uuid.uuid4()) + ".scene")
    directory.mkdir(mode=0o700)
    payload = struct.pack('<III BBBB I 12s', 0x5053474e, 4, 1, 0, 12, 0, 1, 32, b'\x00' * 12)
    payload += struct.pack('<QQ', 0, 9)
    (directory / "0.spz").write_bytes(payload)
    (directory / "manifest.json").write_text(json.dumps({
        "version": 1,
        "nodes": [{"path": "0.spz", "transform": IDENTITY, "shDegree": 0}],
    }))
    nodes, total = gallery_preparation.read_staging(tmp_path, directory)
    assert nodes[0]["path"].name == "0.spz"
    assert total == len(payload)

def test_saved_publication_metadata_uses_embedded_hdr_without_live_view(tmp_path):
    import shutil
    import uuid
    from pathlib import Path
    from lfs_plugins import gallery_preparation
    staging = tmp_path / (str(uuid.uuid4()) + '.scene')
    staging.mkdir()
    shutil.copyfile(Path(__file__).parents[1] / 'data' / 'portable-sog.licht', staging / 'project.licht')
    result = gallery_preparation.publication_view_metadata(tmp_path, staging)
    assert result['environment'] == {'exposure': -1.25, 'rotation': 123.0}
    assert result['exposure'] == pytest.approx(0.7)
    assert result['camera']['position'] == [-2.0, 2.0, -6.0]
    assert result['verticalFov'] is True
