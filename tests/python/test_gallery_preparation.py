# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
import io
import json
from pathlib import Path
import threading
import uuid

import pytest

from lfs_plugins import gallery_bundle, gallery_preparation, gallery_sync
from lfs_plugins.portal_gallery import GalleryTransferCanceled
from test_gallery_bundle import IDENTITY, SHEAR, ply
from test_gallery_sync import Client, connected, finish


def queue_legacy_staging(service, directory, metadata, project_id):
    # Existing journals from earlier versions still recover their saved LFSG
    # staging, but the current publishing entrypoint requires a fresh .licht.
    return service.queue_upload(directory.with_suffix(".lfsg"), metadata, project_id,
                                owned_export=True, preparation=str(directory))


def download_bundle(root):
    directory = staging(root)
    nodes, _ = gallery_preparation.read_staging(root, directory)
    target = root / (str(uuid.uuid4()) + ".lfsg")
    gallery_bundle.write_bundle(target, nodes)
    return target


def test_unpack_preserves_geometry_transforms_and_degrees(tmp_path):
    source = download_bundle(tmp_path)
    destination = tmp_path / (str(uuid.uuid4()) + ".scene")
    progress = []
    gallery_preparation.unpack_bundle(tmp_path, source, destination, progress=lambda *p: progress.append(p))
    nodes, total = gallery_preparation.read_staging(tmp_path, destination)
    assert [n["transform"] for n in nodes] == [IDENTITY, SHEAR]
    assert [n["shDegree"] for n in nodes] == [0, 1]
    assert [n["path"].read_bytes() for n in nodes] == [ply(count=1), ply(count=2)]
    assert progress[-1] == (total, total)
    assert destination.stat().st_mode & 0o777 == 0o700
    assert source.exists()


def test_unpack_cancel_removes_partial_import_but_keeps_download(tmp_path):
    source = download_bundle(tmp_path)
    destination = tmp_path / (str(uuid.uuid4()) + ".scene")
    def cancel(done, total):
        if done:
            raise GalleryTransferCanceled()
    with pytest.raises(GalleryTransferCanceled):
        gallery_preparation.unpack_bundle(tmp_path, source, destination, progress=cancel)
    assert not destination.exists() and source.exists()


def test_unpack_corrupt_second_object_never_publishes_first(tmp_path):
    source = download_bundle(tmp_path)
    with source.open("rb") as stream, gallery_bundle.open_bundle(stream) as bundle:
        offset, length, _ = bundle.node_storage(1)
    with source.open("r+b") as stream:
        stream.seek(offset + length - 1)
        original = stream.read(1)
        stream.seek(-1, 1)
        stream.write(bytes([original[0] ^ 1]))
    destination = tmp_path / (str(uuid.uuid4()) + ".scene")
    with pytest.raises(ValueError):
        gallery_preparation.unpack_bundle(tmp_path, source, destination)
    assert not destination.exists() and source.exists()


def test_unpack_never_overwrites_existing_import(tmp_path):
    source = download_bundle(tmp_path)
    destination = staging(tmp_path)
    previous = (destination / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        gallery_preparation.unpack_bundle(tmp_path, source, destination)
    assert (destination / "manifest.json").read_bytes() == previous


def test_bundle_download_staging_and_owned_cleanup(tmp_path, monkeypatch):
    from test_gallery_sync import cleanup_download
    service = bundle_service(tmp_path, monkeypatch)
    job, path, old_stage = cleanup_download(service)
    path.unlink()
    old_stage.unlink()
    job.pop("stagedImport")
    source = download_bundle(tmp_path)
    destination = path.with_suffix(".lfsg")
    source.rename(destination)
    job["path"] = str(destination)
    identifier = service.stage_download(job["id"])
    finish(service)
    stage = job["stagedImport"]
    assert stage["id"] == identifier and stage["state"] == "ready", stage
    nodes, _ = gallery_preparation.read_staging(tmp_path / "imports", stage["path"])
    assert len(nodes) == 2
    previous = Path(stage["path"])
    service.stage_download(job["id"])
    finish(service)
    stage = job["stagedImport"]
    assert stage["state"] == "ready" and not previous.exists()
    assert len(list((tmp_path / "imports").glob("*.scene"))) == 1
    service.clear_finished([job["id"]])
    finish(service)
    assert not destination.exists() and not Path(stage["path"]).exists()


def staging(root):
    directory = root / (str(uuid.uuid4()) + ".scene")
    directory.mkdir(mode=0o700)
    nodes = []
    for number, transform in enumerate((IDENTITY, SHEAR)):
        (directory / f"{number}.ply").write_bytes(ply(count=number + 1))
        nodes.append({"path": f"{number}.ply", "transform": transform, "shDegree": number})
    (directory / "manifest.json").write_text(json.dumps({"version": 1, "nodes": nodes}))
    return directory


def bundle_service(root, monkeypatch):
    monkeypatch.setattr(Client, "_request", lambda self, *args: {
        "id": self.account.owner, "gallerySyncVersion": 1, "sourceFormats": ["ply", "lfsg"]})
    return connected(root, monkeypatch)


def uploaded():
    return {"scene": {"id": "remote-scene", "revision": "new", "title": "Scene", "sourceFormat": "lfsg"}}


def test_packages_native_nodes_off_ui_thread_and_retires_only_owned_files(tmp_path, monkeypatch):
    service = bundle_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    unrelated = staging(tmp_path)
    main_thread = threading.get_ident()

    def upload(self, path, metadata, **kwargs):
        assert threading.get_ident() != main_thread
        assert metadata == {"title": "Scene", "viewerSettings": {"shDegree": 1}}
        with Path(path).open("rb") as source, gallery_bundle.open_bundle(source) as bundle:
            assert [n["transform"] for n in bundle.manifest["nodes"]] == [IDENTITY, SHEAR]
            assert [n["shDegree"] for n in bundle.manifest["nodes"]] == [0, 1]
            for number in range(2):
                data = io.BytesIO()
                bundle.copy_node(number, data)
                assert data.getvalue() == ply(count=number + 1)
        return uploaded()

    monkeypatch.setattr(Client, "upload", upload, raising=False)
    queue_legacy_staging(service, directory, {"title": "Scene", "viewerSettings": {"shDegree": 1}}, "project")
    finish(service)
    job = service.snapshot()["jobs"][0]
    assert job["status"] == "completed", job["message"]
    assert service.snapshot()["links"]["project"]["sceneId"] == "remote-scene"
    assert not directory.exists() and not Path(job["path"]).exists()
    assert (unrelated / "0.ply").exists()


def test_packaging_pause_survives_restart_and_reuses_saved_scene(tmp_path, monkeypatch):
    service = bundle_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    real_write = gallery_bundle.write_bundle

    def paused(destination, nodes, *, progress):
        def callback(done):
            service.pause()
            progress(done)
        return real_write(destination, nodes, progress=callback)

    monkeypatch.setattr(gallery_bundle, "write_bundle", paused)
    monkeypatch.setattr(Client, "upload", lambda *args, **kwargs: pytest.fail("Paused preparation must not upload"), raising=False)
    identifier = queue_legacy_staging(service, directory, {"title": "Scene"}, "project")
    finish(service)
    job = service.snapshot()["jobs"][0]
    assert job["status"] == "paused" and "Preparation paused" in job["message"]
    assert directory.exists() and not Path(job["path"]).exists()
    assert not list(tmp_path.glob(".gallery-bundle-*"))
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    monkeypatch.setattr(gallery_bundle, "write_bundle", real_write)
    monkeypatch.setattr(Client, "upload", lambda *args, **kwargs: uploaded())
    restarted.resume(identifier)
    finish(restarted)
    assert restarted.snapshot()["jobs"][0]["status"] == "completed"
    assert not directory.exists()


def test_transfer_resume_does_not_repackage_and_discard_cleans_staging(tmp_path, monkeypatch):
    service = bundle_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    monkeypatch.setattr(Client, "upload", lambda *args, **kwargs: (_ for _ in ()).throw(GalleryTransferCanceled()), raising=False)
    identifier = queue_legacy_staging(service, directory, {"title": "Scene"}, "project")
    finish(service)
    job = service.snapshot()["jobs"][0]
    assert job["packaged"] and directory.exists()
    monkeypatch.setattr(gallery_bundle, "write_bundle", lambda *args, **kwargs: pytest.fail("Must reuse completed package"))
    service.resume(identifier)
    finish(service)
    assert service.snapshot()["jobs"][0]["status"] == "paused"
    service.discard(identifier)
    finish(service)
    assert not directory.exists() and not Path(job["path"]).exists()


@pytest.mark.parametrize("damage", ["escape", "symlink", "extra", "duplicate", "destination"])
def test_untrusted_preparation_never_uploads_or_removes_unrelated_files(tmp_path, monkeypatch, damage):
    service = bundle_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    private = tmp_path / "private.txt"
    private.write_bytes(b"keep private")
    manifest = directory / "manifest.json"
    data = json.loads(manifest.read_text())
    if damage == "escape":
        data["nodes"][0]["path"] = "../private.txt"
        manifest.write_text(json.dumps(data))
    elif damage == "symlink":
        (directory / "0.ply").unlink()
        (directory / "0.ply").symlink_to(private)
    elif damage == "extra":
        (directory / "private.txt").write_bytes(b"unrecognized file")
    elif damage == "duplicate":
        manifest.write_text('{"version":1,"version":1,"nodes":[]}')
    elif damage == "destination":
        directory.with_suffix(".lfsg").symlink_to(private)
    monkeypatch.setattr(Client, "upload", lambda *args, **kwargs: pytest.fail("Invalid staging must not upload"), raising=False)
    queue_legacy_staging(service, directory, {"title": "Scene"}, "project")
    finish(service)
    assert service.snapshot()["jobs"][0]["status"] == "error"
    assert private.read_bytes() == b"keep private"
    assert not service.snapshot()["links"]


def test_account_switch_during_packaging_never_uploads_to_new_account(tmp_path, monkeypatch):
    service = bundle_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    real_write = gallery_bundle.write_bundle

    def switched(destination, nodes, *, progress):
        def callback(done):
            service.account.email = "another@example.com"
            progress(done)
        return real_write(destination, nodes, progress=callback)

    monkeypatch.setattr(gallery_bundle, "write_bundle", switched)
    monkeypatch.setattr(Client, "upload", lambda *args, **kwargs: pytest.fail("Account switch must stop upload"), raising=False)
    queue_legacy_staging(service, directory, {"title": "Scene"}, "project")
    finish(service)
    assert not service.snapshot()["jobs"]
    assert directory.exists() and not directory.with_suffix(".lfsg").exists()


def test_bundle_preparation_requires_advertised_capability(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="cannot receive"):
        queue_legacy_staging(service, staging(tmp_path), {"title": "Scene"}, "project")


def test_queue_is_durable_before_worker_start_and_recovers_start_failure(tmp_path, monkeypatch):
    service = bundle_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    monkeypatch.setattr(service, "resume", lambda _: (_ for _ in ()).throw(RuntimeError("thread unavailable")))
    identifier = queue_legacy_staging(service, directory, {"title": "Scene"}, "project")
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
    service = bundle_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    journal = tmp_path / "sync.json"
    journal.write_text(journal.read_text() + "\n")
    with pytest.raises(ValueError, match="Another Studio window updated"):
        queue_legacy_staging(service, directory, {"title": "Scene"}, "project")
    assert not any(bucket["jobs"] for bucket in service._data["accounts"].values())
    assert directory.exists()  # Caller retains ownership when queueing raises.


@pytest.mark.parametrize("redirect", ["outside", "symlink"])
def test_resumed_packaged_upload_revalidates_ownership_before_reading(tmp_path, monkeypatch, redirect):
    service = bundle_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    monkeypatch.setattr(Client, "upload", lambda *args, **kwargs: (_ for _ in ()).throw(GalleryTransferCanceled()), raising=False)
    identifier = queue_legacy_staging(service, directory, {"title": "Scene"}, "project")
    finish(service)
    private = tmp_path / "private.lfsg"
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
    service = bundle_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    kept = directory / "unrecognized.txt"
    kept.write_text("keep for review")
    identifier = queue_legacy_staging(service, directory, {"title": "Scene"}, "project")
    finish(service)
    service.discard(identifier)
    finish(service)
    job = service.snapshot()["jobs"][0]
    assert job["status"] == "canceled" and job["cleanupPending"]
    assert "files were kept" in job["message"] and "recovery folder" in job["message"]
    assert kept.read_text() == "keep for review"


def test_cleanup_failure_cannot_turn_published_upload_into_retry(tmp_path, monkeypatch):
    service = bundle_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    monkeypatch.setattr(Client, "upload", lambda *args, **kwargs: uploaded(), raising=False)
    monkeypatch.setattr(service, "_retire_export", lambda job: (_ for _ in ()).throw(RuntimeError("cleanup interrupted")))
    queue_legacy_staging(service, directory, {"title": "Scene"}, "project")
    finish(service)
    job = service.snapshot()["jobs"][0]
    assert job["status"] == "completed" and job["cleanupPending"]
    assert service.snapshot()["links"]["project"]["sceneId"] == "remote-scene"
    assert directory.exists()
    with pytest.raises(ValueError):
        service.resume(job["id"])


def test_hdr_import_asset_survives_transfer_cleanup(tmp_path, monkeypatch):
    import struct
    from test_gallery_sync import cleanup_download
    service = bundle_service(tmp_path, monkeypatch)
    job, path, old_stage = cleanup_download(service)
    path.unlink(); old_stage.unlink(); job.pop('stagedImport')
    directory = staging(tmp_path)
    nodes, _ = gallery_preparation.read_staging(tmp_path, directory)
    pixels = b'LFSENV1\0' + struct.pack('<II3f', 1, 1, .25, 1, 5)
    background = tmp_path / 'private.lfsenv'
    background.write_bytes(pixels)
    destination = path.with_suffix('.lfsg')
    gallery_bundle.write_bundle(destination, nodes, environment=background)
    job['path'] = str(destination)
    job['result']['viewerSettings'] = {'environment': {'exposure': -1, 'rotation': 10}}
    service.stage_download(job['id']); finish(service)
    assert job['stagedImport']['state'] == 'ready', job['stagedImport']
    retained = service.environment_path(job)
    assert retained.read_bytes() == pixels and retained.stat().st_mode & 0o777 == 0o600
    service.clear_finished([job['id']]); finish(service)
    assert not destination.exists() and retained.read_bytes() == pixels


def test_current_publishing_requires_a_fresh_native_project(tmp_path, monkeypatch):
    service = bundle_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    with pytest.raises(ValueError, match="fresh .licht"):
        service.queue_prepared_upload(directory, {"title": "Scene"}, "project")
    assert service.snapshot()["jobs"] == []


def test_current_publishing_uploads_only_the_fresh_licht(tmp_path, monkeypatch):
    service = bundle_service(tmp_path, monkeypatch)
    service._source_formats = ['licht']
    fixture = Path(__file__).parents[1] / 'data' / 'portable-sog.licht'
    directory = tmp_path / (str(uuid.uuid4()) + '.scene')
    gallery_preparation.unpack_project(tmp_path, fixture, directory)
    original = fixture.read_bytes()
    (directory / 'project.licht').write_bytes(original)
    def upload(self, path, metadata, **kwargs):
        assert Path(path).suffix == '.licht'
        assert Path(path).read_bytes() == original
        return {'scene': {'id': 'remote-scene', 'revision': 'new', 'title': 'Scene', 'sourceFormat': 'licht'}}
    monkeypatch.setattr(Client, 'upload', upload, raising=False)
    service.queue_prepared_upload(directory, {'title': 'Scene', 'viewerSettings': {'environment': {'exposure': -1.25, 'rotation': 123}}}, 'project')
    finish(service)
    job = service.snapshot()['jobs'][0]
    assert job['status'] == 'completed', job['message']
    assert not directory.exists() and not Path(job['path']).exists()
