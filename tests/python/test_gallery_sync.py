# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
import threading
from types import SimpleNamespace

import pytest

from lfs_plugins import gallery_sync
from lfs_plugins.portal_gallery import GalleryTransferCanceled, GalleryProcessingPaused


class Account:
    base_url = "https://portal.example"
    email = "one@example.com"
    owner = "one"

    def snapshot(self):
        return SimpleNamespace(signed_in=True, email=self.email, connected_since="session")


class Client:
    def __init__(self, account, **kwargs):
        self.account = account

    def _request(self, *args):
        return {"id": self.account.owner, "gallerySyncVersion": 1}

    def list_scenes(self):
        return []

    def scene(self, scene_id):
        return {"id": scene_id, "revision": "remote-new"}


def finish(service):
    service._thread.join(3)
    assert not service.busy


def connected(tmp_path, monkeypatch):
    monkeypatch.setattr(gallery_sync, "PortalGalleryClient", Client)
    service = gallery_sync.GallerySync(Account(), tmp_path)
    service.refresh()
    finish(service)
    # Persistence tests start with an explicitly seeded journal; browsing no
    # longer creates or rewrites this file.
    service._save()
    return service


def test_identity_reads_current_account_without_traversing_private_history(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    expected = service.snapshot()["identity"]
    class UncopyableHistory:
        def __deepcopy__(self, memo):
            raise AssertionError("Identity checks must not copy history")
    service._bucket()["jobs"] = [UncopyableHistory()]
    assert service.identity() == expected
    service.account.email = "two@example.com"
    assert service.identity() == ("https://portal.example", "two@example.com", "session", True)
    assert service.snapshot()["jobs"] == []  # Previous account stays inaccessible.


def test_resume_after_restart_reuses_checkpoint_and_links_project(tmp_path, monkeypatch):
    def pause(self, path, metadata, **kwargs):
        kwargs["on_checkpoint"]({"uploadId": "pending", "idempotencyKey": "stable"})
        kwargs["on_progress"](4, 8)
        raise GalleryTransferCanceled()
    monkeypatch.setattr(Client, "upload", pause, raising=False)
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / "scene.ply"
    path.write_bytes(b"ply-data")
    job = service.queue_upload(path, {"title": "Example"}, "project-uuid")
    finish(service)
    assert service.snapshot()["jobs"][0]["status"] == "paused"
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    def resume(self, path, metadata, **kwargs):
        assert kwargs["checkpoint"] == {"uploadId": "pending", "idempotencyKey": "stable"}
        return {"scene": {"id": "remote-scene", "revision": "new", "title": "Example"}}
    monkeypatch.setattr(Client, "upload", resume)
    restarted.resume(job)
    finish(restarted)
    state = restarted.snapshot()
    assert state["links"]["project-uuid"]["sceneId"] == "remote-scene"
    assert state["jobs"][0]["status"] == "completed"
    assert json.loads((tmp_path / "sync.json").read_text())["accounts"]


def test_server_processing_is_visible_and_never_linked_before_completion(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / "scene.ply"
    path.write_bytes(b"ply-data")
    def upload(*args, **kwargs):
        kwargs["on_processing"]({"stage": "validating", "completed": 4, "total": 8})
        state = service.snapshot()
        assert state["jobs"][0]["message"] == "Checking scene"
        assert state["jobs"][0]["serverProcessing"]
        assert not state["links"]
        raise GalleryProcessingPaused()
    monkeypatch.setattr(Client, "upload", upload, raising=False)
    service.queue_upload(path, {"title": "Scene"}, "project")
    finish(service)
    state = service.snapshot()
    assert state["jobs"][0]["status"] == "paused"
    assert "portal may continue" in state["jobs"][0]["message"]
    assert not state["links"]


def test_account_switch_hides_and_cannot_resume_previous_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(Client, "upload", lambda *args, **kw: (_ for _ in ()).throw(GalleryTransferCanceled()), raising=False)
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / "scene.ply"
    path.write_bytes(b"ply-data")
    job = service.queue_upload(path, {"title": "Private"}, "project")
    finish(service)
    service.account.email, service.account.owner = "two@example.com", "two"
    assert service.snapshot()["jobs"] == []
    with pytest.raises(ValueError, match="account changed"):
        service.resume(job)
    service.refresh()
    finish(service)
    assert service.snapshot()["jobs"] == []
    assert service.snapshot()["links"] == {}


def _download_scene():
    return {"id": "scene", "revision": "r1", "sourceFormat": "spz", "title": "Garden", "contentLength": 8}


def test_download_survives_launch_failure_and_restart(tmp_path, monkeypatch):
    from pathlib import Path
    service = connected(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "resume", lambda *_: (_ for _ in ()).throw(RuntimeError("thread start failed")))
    service.download(_download_scene())
    job = service.snapshot()["jobs"][0]
    assert job["kind"] == "download" and job["status"] == "paused"
    assert job["sceneId"] == "scene" and job["revision"] == "r1"
    persisted = json.loads((tmp_path / "sync.json").read_text())
    saved = next(item for bucket in persisted["accounts"].values() for item in bucket["jobs"])
    assert saved["id"] == job["id"] and saved["sceneId"] == "scene" and saved["status"] == "queued"
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    kept = restarted.snapshot()["jobs"][0]
    assert kept["id"] == job["id"] and kept["status"] == "paused" and kept["sceneId"] == "scene"
    def download(self, scene_id, destination, **kwargs):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"payload!")
        return {"id": scene_id, "revision": "r1", "title": "Garden"}
    monkeypatch.setattr(Client, "download", download, raising=False)
    restarted.resume(kept["id"])
    finish(restarted)
    assert restarted.snapshot()["jobs"][0]["status"] == "completed"


def test_download_journal_write_failure_rolls_back(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    original = (tmp_path / "sync.json").read_bytes()
    monkeypatch.setattr(service, "_save", lambda: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        service.download(_download_scene())
    assert service.snapshot()["jobs"] == []
    assert (tmp_path / "sync.json").read_bytes() == original


def test_download_rejects_stale_journal_before_queueing(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    journal = tmp_path / "sync.json"
    different = {"version": 1, "accounts": {"other": {"jobs": [], "links": {}}}}
    journal.write_text(json.dumps(different))
    monkeypatch.setattr(service, "resume", lambda *_: pytest.fail("Stale downloads must not start"))
    with pytest.raises(ValueError, match="Another LichtFeld Studio window"):
        service.download(_download_scene())
    assert json.loads(journal.read_text()) == different
    assert service.snapshot()["jobs"] == []


def test_upload_cannot_silently_retarget_project_link(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    service._bucket()["links"]["project"] = {"sceneId": "original", "revision": "r"}
    path = tmp_path / "scene.ply"
    path.write_bytes(b"ply-data")
    for metadata in ({"title": "New"}, {"title": "Other", "replaceSceneId": "other"}):
        with pytest.raises(ValueError, match="linked to another"):
            service.queue_upload(path, metadata, "project")
    assert service.snapshot()["links"]["project"]["sceneId"] == "original"
    assert service.snapshot()["jobs"] == []


def test_discard_failure_keeps_recovery_record(tmp_path, monkeypatch):
    def upload(*args, **kwargs):
        kwargs["on_checkpoint"]({"uploadId": "pending"})
        raise GalleryTransferCanceled()
    monkeypatch.setattr(Client, "upload", upload, raising=False)
    monkeypatch.setattr(Client, "cancel_upload", lambda *args: (_ for _ in ()).throw(OSError()), raising=False)
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / "scene.ply"
    path.write_bytes(b"ply-data")
    job = service.queue_upload(path, {"title": "Example"}, "project")
    finish(service)
    with pytest.raises(ValueError, match="already has a transfer"):
        service.queue_upload(path, {"title": "Duplicate"}, "project")
    with pytest.raises(ValueError, match="Discard"):
        service.unlink("project")
    service.discard(job)
    finish(service)
    assert service.snapshot()["jobs"][0]["status"] == "paused"
    assert path.read_bytes() == b"ply-data"


@pytest.mark.parametrize("contents", ["broken", "null", '{"version":1,"accounts":{"one":null}}',
    '{"version":1,"accounts":{"one":{"jobs":[null],"links":{}}}}',
    '{"version":1,"accounts":{},"accounts":{}}', '{"version":1,"accounts":{},"bad":NaN}',
    '{"version":1,"accounts":{},"bad":1e999}'])
def test_corrupt_journal_is_never_replaced_with_empty_state(tmp_path, monkeypatch, contents):
    path = tmp_path / "sync.json"
    path.write_text(contents)
    monkeypatch.setattr(gallery_sync, "PortalGalleryClient", lambda *a, **kw: pytest.fail("Damaged records must not reach the network"))
    service = gallery_sync.GallerySync(Account(), tmp_path)
    assert service.snapshot()["storage_issue"]
    assert not service.snapshot()["connected"]
    assert service.snapshot()["jobs"] == []
    service.refresh()
    finish(service)
    with pytest.raises(ValueError, match="saved gallery links"):
        service.queue_upload(tmp_path / "scene.ply", {"title": "Duplicate"}, "project")
    assert path.read_text() == contents


def test_repaired_journal_retries_without_restart_and_preserves_pending_key(tmp_path, monkeypatch):
    monkeypatch.setattr(Client, "upload", lambda *a, **kw: (_ for _ in ()).throw(GalleryTransferCanceled()), raising=False)
    original = connected(tmp_path, monkeypatch)
    path = tmp_path / "scene.ply"
    path.write_bytes(b"ply-data")
    original.queue_upload(path, {"title": "Pending"}, "project")
    finish(original)
    data = json.loads((tmp_path / "sync.json").read_text())
    job = next(iter(data["accounts"].values()))["jobs"][0]
    job.update(status="running", checkpoint={"idempotencyKey": "original-key", "uploadId": "original-upload"})
    saved = json.dumps(data)
    (tmp_path / "sync.json").write_text("damaged")
    service = gallery_sync.GallerySync(original.account, tmp_path)
    assert service.snapshot()["storage_issue"]
    (tmp_path / "sync.json").write_text(saved)
    service.refresh()
    finish(service)
    state = service.snapshot()
    assert state["connected"] and not state["storage_issue"]
    assert state["jobs"][0]["status"] == "paused"
    assert state["jobs"][0]["checkpoint"] == job["checkpoint"]
    assert path.read_bytes() == b"ply-data"


def test_missing_or_oversized_journal_cannot_start_empty(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    journal = tmp_path / "sync.json"
    saved = journal.read_bytes()
    journal.unlink()
    service.refresh()
    finish(service)
    assert service.snapshot()["storage_issue"] and not journal.exists()
    journal.write_bytes(saved)
    monkeypatch.setattr(gallery_sync, "MAX_JOURNAL_BYTES", len(saved) - 1)
    service.refresh()
    finish(service)
    assert service.snapshot()["storage_issue"] and journal.read_bytes() == saved


def test_empty_profile_never_binds_a_gallery_account(tmp_path, monkeypatch):
    monkeypatch.setattr(gallery_sync, "PortalGalleryClient", Client)
    account = Account()
    account.email = ""
    service = gallery_sync.GallerySync(account, tmp_path)
    service.refresh()
    finish(service)
    assert service._owner is None
    assert not service.snapshot()["connected"]
    assert not (tmp_path / "sync.json").exists()


@pytest.mark.parametrize("change", ["email", "base_url"])
def test_refresh_cannot_install_previous_account_results_after_switch(tmp_path, monkeypatch, change):
    service = connected(tmp_path, monkeypatch)
    original = (tmp_path / "sync.json").read_bytes()

    def list_scenes(client):
        setattr(client.account, change, "two@example.com" if change == "email" else "https://second.example")
        return [{"id": "old-account-scene"}]

    monkeypatch.setattr(Client, "list_scenes", list_scenes)
    service.refresh()
    finish(service)
    assert not service.snapshot()["connected"]
    assert service.snapshot()["scenes"] == []
    assert (tmp_path / "sync.json").read_bytes() == original


def test_other_process_journal_change_cannot_be_overwritten(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    journal = tmp_path / "sync.json"
    different = {"version": 1, "accounts": {"other": {"jobs": [], "links": {}}}}
    journal.write_text(json.dumps(different))
    monkeypatch.setattr(Client, "update", lambda *a, **kw: pytest.fail("Stale mutations must not reach the network"), raising=False)
    service.edit("scene", "old-revision", {"title": "Old edit"})
    finish(service)
    assert "Another LichtFeld Studio window" in service.message
    assert not service.snapshot()["connected"]
    assert json.loads(journal.read_text()) == different
    service.refresh()
    finish(service)
    assert service.snapshot()["connected"]
    assert "other" in service._data["accounts"]
    migrated = json.loads(journal.read_text())
    assert migrated == different  # A refresh migrates in memory, without rewriting a peer journal.
    assert service._data["version"] == 2
    assert service._data["accounts"]["other"] == different["accounts"]["other"]


def test_refresh_waits_for_other_window_then_loads_completed_transfer(tmp_path, monkeypatch):
    first = connected(tmp_path, monkeypatch)
    second = connected(tmp_path, monkeypatch)
    started, release = threading.Event(), threading.Event()

    def upload(*args, **kwargs):
        kwargs["on_checkpoint"]({"idempotencyKey": "stable"})
        started.set()
        assert release.wait(3)
        return {"scene": {"id": "remote", "revision": "new"}}

    monkeypatch.setattr(Client, "upload", upload, raising=False)
    path = tmp_path / "scene.ply"
    path.write_bytes(b"ply-data")
    first.queue_upload(path, {"title": "First window"}, "project")
    assert started.wait(3)
    try:
        second.refresh()
        assert second.busy
    finally:
        release.set()
    finish(first)
    finish(second)
    state = second.snapshot()
    assert state["links"]["project"]["sceneId"] == "remote"
    assert len(state["jobs"]) == 1 and state["jobs"][0]["status"] == "completed"
    assert state["jobs"][0]["checkpoint"]["idempotencyKey"] == "stable"


def test_refresh_preserves_other_account_keys_and_resume_writes_reloaded_job(tmp_path, monkeypatch):
    def pause(client, *args, **kwargs):
        kwargs["on_checkpoint"]({"uploadId": client.account.owner, "idempotencyKey": client.account.owner + "-key"})
        raise GalleryTransferCanceled()

    monkeypatch.setattr(Client, "upload", pause, raising=False)
    first = connected(tmp_path, monkeypatch)
    path = tmp_path / "scene.ply"
    path.write_bytes(b"ply-data")
    first_id = first.queue_upload(path, {"title": "First private title"}, "project-one")
    finish(first)
    other_account = Account()
    other_account.owner, other_account.email = "two", "two@example.com"
    other = gallery_sync.GallerySync(other_account, tmp_path)
    other.refresh()
    finish(other)
    other.queue_upload(path, {"title": "Second private title"}, "project-two")
    finish(other)
    other_data = other.snapshot()["jobs"]
    first.refresh()
    finish(first)
    assert [j["metadata"]["title"] for j in first.snapshot()["jobs"]] == ["First private title"]

    def resume(client, *args, **kwargs):
        assert kwargs["checkpoint"] == {"uploadId": "one", "idempotencyKey": "one-key"}
        kwargs["on_checkpoint"]({"uploadId": "one", "idempotencyKey": "one-key", "verified": True})
        raise GalleryTransferCanceled()

    monkeypatch.setattr(Client, "upload", resume)
    first.resume(first_id)
    finish(first)
    data = json.loads((tmp_path / "sync.json").read_text())
    all_jobs = [job for bucket in data["accounts"].values() for job in bucket["jobs"]]
    assert len(all_jobs) == 2
    assert next(job for job in all_jobs if job["project"] == "project-two") == other_data[0]
    assert next(job for job in all_jobs if job["id"] == first_id)["checkpoint"]["verified"]


def test_completed_snapshot_cleanup_never_removes_external_source(tmp_path, monkeypatch):
    import uuid
    monkeypatch.setattr(Client, "upload", lambda *a, **kw: {"scene": {"id": "scene", "revision": "r"}}, raising=False)
    service = connected(tmp_path, monkeypatch)
    snapshot = tmp_path / (str(uuid.uuid4()) + ".ply")
    snapshot.write_bytes(b"ply-data")
    service.queue_upload(snapshot, {"title": "Owned snapshot"}, "one", owned_export=True)
    finish(service)
    assert not snapshot.exists()
    external = tmp_path / (str(uuid.uuid4()) + ".ply")
    external.write_bytes(b"user-data")
    service.queue_upload(external, {"title": "User file"}, "two")
    finish(service)
    assert external.read_bytes() == b"user-data"


def test_native_view_space_round_trip():
    from lfs_plugins.gallery_view import viewer_vector
    for point in ((1, 2, 3), (0, -1, .5), (1e8, 0, 0)):
        assert viewer_vector(viewer_vector(point)) == list(point)


def downloaded_job(service):
    job = {"id": "download", "project": "", "kind": "download", "status": "completed",
        "result": {"id": "scene", "revision": "remote-new"}}
    service._bucket()["jobs"].append(job)
    service._bucket()["links"]["project"] = {"sceneId": "scene", "revision": "old"}
    return job


def cleanup_download(service, *, backup=False):
    from uuid import uuid4
    identifier, stage_id = str(uuid4()), str(uuid4())
    path = service.root / "downloads" / (identifier + ".ply")
    stage = service.root / "imports" / (stage_id + ".ply")
    for item in (path, stage):
        item.parent.mkdir(exist_ok=True)
        item.write_bytes(b"downloaded bytes")
    job = dict(id=identifier, kind="download", project="project", status="completed", path=str(path),
        sceneId="scene", revision="r", result={"id": "scene", "revision": "r", "title": "Scene"},
        metadata={"title": "Scene"}, checkpoint=None, completed=16, total=16, message="Downloaded",
        stagedImport={"id": stage_id, "state": "ready", "path": str(stage)})
    if backup:
        saved = service.root / "backups" / (str(uuid4()) + ".licht")
        saved.parent.mkdir(exist_ok=True)
        saved.write_bytes(b"independent saved project")
        job["localUpdate"] = {"id": saved.stem, "state": "ready", "backupPath": str(saved)}
    service._bucket()["jobs"].append(job)
    service._bucket()["links"]["project"] = {"sceneId": "scene", "revision": "r"}
    service._save()
    return job, path, stage


def test_clear_download_preserves_backup_and_project_link(tmp_path, monkeypatch):
    from pathlib import Path
    service = connected(tmp_path, monkeypatch)
    job, path, stage = cleanup_download(service, backup=True)
    backup = Path(job["localUpdate"]["backupPath"])
    service.clear_finished([job["id"]])
    finish(service)
    assert not path.exists() and not stage.exists()
    assert backup.read_bytes() == b"independent saved project"
    assert service.snapshot()["links"]["project"] == {"sceneId": "scene", "revision": "r"}
    kept = service.snapshot()["jobs"][0]
    assert kept["retired"] and kept["path"] == "" and "stagedImport" not in kept
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    assert restarted.snapshot()["jobs"][0] == kept


def test_clear_history_never_deletes_external_upload_source(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / "my-original.ply"
    path.write_bytes(b"user source")
    monkeypatch.setattr(Client, "upload", lambda *a, **kw: {"scene": {"id": "scene", "revision": "r"}}, raising=False)
    identifier = service.queue_upload(path, {"title": "Original"}, "project")
    finish(service)
    service.clear_finished([identifier])
    finish(service)
    assert path.read_bytes() == b"user source"
    assert service.snapshot()["jobs"] == [] and service.snapshot()["links"]["project"]["sceneId"] == "scene"


def test_native_use_excludes_cleanup_in_another_window(tmp_path, monkeypatch):
    first = connected(tmp_path, monkeypatch)
    job, path, stage = cleanup_download(first)
    second = gallery_sync.GallerySync(first.account, tmp_path)
    second.refresh()
    finish(second)
    with first.local_use(job["id"]):
        second.clear_finished([job["id"]])
        finish(second)
        assert "using a downloaded scene" in second.message
        assert path.exists() and stage.exists()
        assert not second.snapshot()["jobs"][0].get("cleanupPending")
    second.clear_finished([job["id"]])
    finish(second)
    assert not path.exists() and second.snapshot()["jobs"] == []
    with pytest.raises(ValueError, match="Another LichtFeld Studio window"):
        with first.local_use(job["id"]):
            pytest.fail("Stale imports must not reach native code")


def test_partial_cleanup_is_persisted_and_retries_after_restart(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    job, path, stage = cleanup_download(service)
    original = service._unlink_temporary

    def fail_stage(item):
        if item == stage:
            raise OSError("simulated file in use")
        original(item)

    monkeypatch.setattr(service, "_unlink_temporary", fail_stage)
    service.clear_finished([job["id"]])
    finish(service)
    assert not path.exists() and stage.exists()
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    assert restarted.snapshot()["jobs"][0]["cleanupPending"]
    with pytest.raises(ValueError, match="no longer ready"):
        with restarted.local_use(job["id"]):
            pytest.fail("An interrupted cleanup cannot be imported")
    restarted.clear_finished([job["id"]])
    finish(restarted)
    assert not stage.exists() and restarted.snapshot()["jobs"] == []


def test_cleanup_commit_failure_keeps_retry_record(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    job, path, stage = cleanup_download(service)
    original, writes = service._save, []

    def fail_commit():
        writes.append(True)
        if len(writes) == 2:
            raise OSError("simulated journal commit failure")
        original()

    monkeypatch.setattr(service, "_save", fail_commit)
    service.clear_finished([job["id"]])
    finish(service)
    assert not path.exists() and not stage.exists()
    assert service.snapshot()["jobs"][0]["cleanupPending"]
    monkeypatch.setattr(service, "_save", original)
    service.clear_finished([job["id"]])
    finish(service)
    assert service.snapshot()["jobs"] == []


def test_batch_cleanup_can_resume_after_first_download_was_removed(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    first, path_one, stage_one = cleanup_download(service)
    second, path_two, stage_two = cleanup_download(service, backup=True)
    original = service._unlink_temporary

    def fail_second(path):
        if path == path_two:
            raise OSError("simulated interruption between downloads")
        original(path)

    monkeypatch.setattr(service, "_unlink_temporary", fail_second)
    service.clear_finished([first["id"], second["id"]])
    finish(service)
    assert not path_one.exists() and not stage_one.exists()
    assert path_two.exists() and stage_two.exists()
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    assert all(j["cleanupPending"] for j in restarted.snapshot()["jobs"])
    restarted.clear_finished([first["id"], second["id"]])
    finish(restarted)
    assert not path_two.exists() and not stage_two.exists()
    assert len(restarted.snapshot()["jobs"]) == 1 and restarted.snapshot()["jobs"][0]["retired"]


@pytest.mark.parametrize("redirect", ["directory", "file", "other_account", "recovery_copy"])
def test_cleanup_refuses_redirected_or_other_account_references(tmp_path, monkeypatch, redirect):
    import copy
    service = connected(tmp_path, monkeypatch)
    job, path, stage = cleanup_download(service)
    if redirect == "directory":
        moved = tmp_path / "user-files"
        path.parent.rename(moved)
        path.parent.symlink_to(moved, target_is_directory=True)
    elif redirect == "file":
        moved = tmp_path / "original.ply"
        path.rename(moved)
        path.symlink_to(moved)
    elif redirect == "other_account":
        other = copy.deepcopy(job)
        other["id"] = "other-account-transfer"
        service._data["accounts"]["another-account"] = {"jobs": [other], "links": {}}
        service._save()
    else:
        job["localUpdate"] = {"backupPath": str(path)}
        service._save()
    service.clear_finished([job["id"]])
    finish(service)
    assert path.read_bytes() == b"downloaded bytes" and stage.exists()
    assert not service.snapshot()["jobs"][0].get("cleanupPending")


def test_local_update_keeps_verified_recovery_copy_before_changing_link(tmp_path, monkeypatch):
    import hashlib
    service = connected(tmp_path, monkeypatch)
    job = downloaded_job(service)
    source = tmp_path / "project.licht"
    source.write_bytes(b"saved local project")
    service.prepare_local_update(job["id"], "project", source, gallery_sync.file_stamp(source))
    finish(service)
    record = job["localUpdate"]
    assert record["state"] == "ready"
    from pathlib import Path
    assert Path(record["backupPath"]).read_bytes() == source.read_bytes()
    assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert service.snapshot()["links"]["project"]["revision"] == "old"


def test_local_update_rejects_changed_source_and_different_account(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    job = downloaded_job(service)
    source = tmp_path / "project.licht"
    source.write_bytes(b"original")
    stamp = gallery_sync.file_stamp(source)
    source.write_bytes(b"changed local version")
    service.prepare_local_update(job["id"], "project", source, stamp)
    finish(service)
    assert job["localUpdate"]["state"] == "failed"
    assert not list((tmp_path / "backups").glob("*.licht"))
    assert source.read_bytes() == b"changed local version"
    service.account.email = "different@example.com"
    with pytest.raises(ValueError, match="account changed"):
        service.prepare_local_update(job["id"], "project", source, gallery_sync.file_stamp(source))


def test_download_cannot_retarget_an_existing_project_link(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    job = downloaded_job(service)
    service._bucket()["links"]["project"]["sceneId"] = "different-scene"
    with pytest.raises(ValueError, match="different gallery item"):
        service.link_download(job["id"], "project")


def test_local_update_rejects_a_download_superseded_on_the_portal(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    job = downloaded_job(service)
    source = tmp_path / "project.licht"
    source.write_bytes(b"local work")
    monkeypatch.setattr(Client, "scene", lambda *_: {"revision": "newer-remote"})
    service.prepare_local_update(job["id"], "project", source, gallery_sync.file_stamp(source))
    finish(service)
    assert job["localUpdate"]["state"] == "failed"
    assert "changed since this download" in job["localUpdate"]["message"]
    assert service.snapshot()["links"]["project"]["revision"] == "old"


def test_staged_import_is_unique_and_keeps_download_intact_without_hard_links(tmp_path, monkeypatch):
    from pathlib import Path
    service = connected(tmp_path, monkeypatch)
    job, source, previous_stage = cleanup_download(service)
    previous_stage.unlink()
    job.pop("stagedImport")
    source.write_bytes(b"downloaded splats")
    job["path"] = str(source)
    monkeypatch.setattr(gallery_sync.os, "link", lambda *_: (_ for _ in ()).throw(OSError("unsupported")))
    service.stage_download(job["id"])
    finish(service)
    staged = Path(job["stagedImport"]["path"])
    assert staged != source and staged.suffix == ".ply"
    assert staged.read_bytes() == source.read_bytes()
    staged.unlink()
    assert source.read_bytes() == b"downloaded splats"


def test_failed_link_save_is_not_reported_as_a_completed_update(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    job = downloaded_job(service)
    monkeypatch.setattr(service, "_save", lambda: (_ for _ in ()).throw(OSError("disk full")))
    operation = service.link_download(job["id"], "project")
    finish(service)
    assert job["linkOperation"]["id"] == operation
    assert job["linkOperation"]["state"] == "failed"
    assert service.snapshot()["links"]["project"]["revision"] == "old"
    assert job["project"] == ""


def test_reviewed_conflict_is_persisted_before_reusing_upload(tmp_path, monkeypatch):
    from lfs_plugins.portal_account import PortalHTTPError
    remote = {"id": "scene", "revision": "reviewed", "title": "Remote title", "description": "Remote edit",
        "visibility": "private", "viewerSettings": {"antialiasing": True}}
    def upload(*args, **kwargs):
        if not kwargs["checkpoint"]:
            kwargs["on_checkpoint"]({"uploadId": "pending", "idempotencyKey": "same-upload"})
            raise PortalHTTPError(409, "sync_conflict")
        record = kwargs["checkpoint"]
        assert record["rebase"]["baseRevision"] == "reviewed"
        assert record["rebase"]["metadata"]["description"] == "Remote edit"
        assert "reviewed" in (tmp_path / "sync.json").read_text()
        return {"scene": remote}
    monkeypatch.setattr(Client, "upload", upload, raising=False)
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / "scene.ply"
    path.write_bytes(b"ply-data")
    job = service.queue_upload(path, {"title": "Local title", "replaceSceneId": "scene"}, "project")
    finish(service)
    assert service.snapshot()["jobs"][0]["status"] == "conflict"
    service.resolve_conflict(job, remote)
    finish(service)
    assert service.snapshot()["links"]["project"]["revision"] == "reviewed"
    assert service.snapshot()["jobs"][0]["status"] == "completed"


def test_reviewed_create_conflict_uses_new_request_before_any_parts(tmp_path, monkeypatch):
    from lfs_plugins.portal_account import PortalHTTPError
    remote = {"id": "scene", "revision": "reviewed", "title": "Remote title", "description": "Remote edit",
        "visibility": "private", "viewerSettings": {"antialiasing": True}}
    calls = []
    def upload(self, path, metadata, **kwargs):
        calls.append(dict(metadata))
        if len(calls) == 1:
            kwargs["on_checkpoint"]({"idempotencyKey": "rejected-before-create"})
            raise PortalHTTPError(409, "sync_conflict")
        assert kwargs["checkpoint"] is None
        assert metadata["baseRevision"] == "reviewed"
        assert metadata["description"] == "Remote edit"
        assert "reviewed" in (tmp_path / "sync.json").read_text()
        return {"scene": remote}
    monkeypatch.setattr(Client, "upload", upload, raising=False)
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / "scene.ply"
    path.write_bytes(b"ply-data")
    job = service.queue_upload(path, {"title": "Old", "replaceSceneId": "scene", "baseRevision": "stale"}, "project")
    finish(service)
    service.resolve_conflict(job, remote)
    finish(service)
    assert len(calls) == 2
    assert service.snapshot()["jobs"][0]["status"] == "completed"


@pytest.mark.parametrize('suffix', ['sog', 'ssog', 'spz'])
def test_compressed_studio_snapshot_uploads_and_retires_only_its_owned_file(tmp_path, monkeypatch, suffix):
    import uuid
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / f'{uuid.uuid4()}.{suffix}'
    path.write_bytes(b'compressed-snapshot')
    calls = []
    def upload(self, source, metadata, **kwargs):
        from pathlib import Path
        calls.append(Path(source).read_bytes())
        return {'scene':{'id':'compressed', 'revision':'r', 'sourceFormat':suffix}}
    monkeypatch.setattr(Client, 'upload', upload, raising=False)
    identifier = service.queue_upload(path, {'title':'Compressed'}, 'project', owned_export=True)
    finish(service)
    assert calls == [b'compressed-snapshot']
    assert service.snapshot()['jobs'][-1]['status'] == 'completed'
    assert not path.exists()
    assert service.snapshot()['links']['project']['sceneId'] == 'compressed'


def test_sync_camera_track_patches_only_path_updates_revision_and_keeps_conflict(tmp_path, monkeypatch):
    from lfs_plugins.portal_account import PortalHTTPError
    path = {"version": 1, "keyframes": [{"t": 0}], "duration": 2, "loopMode": "once", "playbackSpeed": 1}
    merged = {
        "id": "scene", "revision": "next", "title": "Keep title", "description": "Keep description",
        "visibility": "private",
        "viewerSettings": {"camera": {"fov": 50}, "exposure": 2, "cameraPath": path},
    }
    calls = []

    def update(self, scene_id, revision, **metadata):
        calls.append((scene_id, revision, metadata))
        assert list(metadata) == ["viewerSettings"]
        assert list(metadata["viewerSettings"]) == ["cameraPath"]
        return merged

    monkeypatch.setattr(Client, "update", update, raising=False)
    service = connected(tmp_path, monkeypatch)
    service.scenes = [{"id": "scene", "revision": "old", "title": "Keep title"}]
    service._bucket()["links"]["project"] = {
        "sceneId": "scene", "revision": "old",
        "metadata": {"title": "Keep title", "viewerSettings": {"camera": {"fov": 50}, "exposure": 2}},
    }
    version_before = service.version
    service.send_camera_track("scene", "old", path)
    assert service.busy
    assert service.message == "Sending the gallery camera track…"
    finish(service)
    assert calls == [("scene", "old", {"viewerSettings": {"cameraPath": path}})]
    state = service.snapshot()
    assert state["links"]["project"]["revision"] == "next"
    assert state["scenes"][0]["revision"] == "next"
    assert state["links"]["project"]["metadata"]["title"] == "Keep title"
    assert state["links"]["project"]["metadata"]["viewerSettings"]["camera"] == {"fov": 50}
    assert state["links"]["project"]["metadata"]["viewerSettings"]["exposure"] == 2
    assert "camera track" in state["message"].lower()
    assert state["version"] > version_before

    def conflict(*args, **kwargs):
        raise PortalHTTPError(409, "sync_conflict")

    monkeypatch.setattr(Client, "update", conflict)
    before = json.loads(json.dumps(service.snapshot()["links"]))
    service.send_camera_track("scene", "old", None)
    assert service.message == "Sending the gallery camera track…"
    finish(service)
    assert service.snapshot()["links"] == before
    assert "changed" in service.message.lower()
    assert "review both versions" in service.message.lower()


def test_sync_camera_track_sends_explicit_null_without_other_metadata(tmp_path, monkeypatch):
    calls = []

    def update(self, scene_id, revision, **metadata):
        calls.append((scene_id, revision, metadata))
        return {"id": scene_id, "revision": "cleared", "title": "Keep",
                "viewerSettings": {"camera": {"fov": 40}, "cameraPath": None}}

    monkeypatch.setattr(Client, "update", update, raising=False)
    service = connected(tmp_path, monkeypatch)
    service.scenes = [{"id": "scene", "revision": "old"}]
    service._bucket()["links"]["project"] = {"sceneId": "scene", "revision": "old", "metadata": {"title": "Keep"}}
    service.send_camera_track("scene", "old", None)
    finish(service)
    assert calls == [("scene", "old", {"viewerSettings": {"cameraPath": None}})]
    assert service.snapshot()["links"]["project"]["revision"] == "cleared"
    assert service.snapshot()["links"]["project"]["metadata"]["viewerSettings"]["cameraPath"] is None


def test_fetch_camera_track_gets_path_without_changing_link_revision(tmp_path, monkeypatch):
    path = {"version": 1, "keyframes": [{"t": 1}], "duration": 4, "loopMode": "loop", "playbackSpeed": 1}
    remote = {"id": "scene", "revision": "web-editor", "title": "Keep title",
        "viewerSettings": {"camera": {"fov": 40}, "exposure": 3, "cameraPath": path}}
    calls = []

    def scene(self, scene_id):
        calls.append(("GET", scene_id))
        return remote

    monkeypatch.setattr(Client, "scene", scene, raising=False)
    monkeypatch.setattr(Client, "update", lambda *a, **k: pytest.fail("track fetch must not PATCH"), raising=False)
    monkeypatch.setattr(Client, "upload", lambda *a, **k: pytest.fail("track fetch must not upload"), raising=False)
    service = connected(tmp_path, monkeypatch)
    service.scenes = [{"id": "scene", "revision": "old", "title": "Keep title"}]
    service._bucket()["links"]["project"] = {
        "sceneId": "scene", "revision": "geometry-rev",
        "metadata": {"title": "Keep title", "viewerSettings": {"camera": {"fov": 40}}},
    }
    operation = service.fetch_camera_track("scene")
    assert service.busy
    assert service.message == "Getting the gallery camera track…"
    finish(service)
    assert calls == [("GET", "scene")]
    fetch = service.snapshot()["trackFetch"]
    assert fetch["id"] == operation and fetch["state"] == "ready"
    assert fetch["cameraPath"] == path and fetch["cameraPath"] is not path
    assert fetch["revision"] == "web-editor"
    assert service.snapshot()["links"]["project"]["revision"] == "geometry-rev"
    assert service.snapshot()["scenes"][0]["revision"] == "web-editor"
    assert "received" in service.snapshot()["message"].lower()


def test_fetch_camera_track_null_and_missing_path_are_explicit_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(Client, "scene", lambda self, scene_id: {"id": scene_id, "revision": "r",
        "viewerSettings": {"cameraPath": None}}, raising=False)
    service = connected(tmp_path, monkeypatch)
    service._bucket()["links"]["project"] = {"sceneId": "scene", "revision": "old"}
    service.fetch_camera_track("scene")
    finish(service)
    assert service.snapshot()["trackFetch"]["cameraPath"] is None
    assert service.snapshot()["links"]["project"]["revision"] == "old"

    monkeypatch.setattr(Client, "scene", lambda self, scene_id: {"id": scene_id, "revision": "r2"}, raising=False)
    service.fetch_camera_track("scene")
    finish(service)
    assert service.snapshot()["trackFetch"]["cameraPath"] is None


def test_fetch_camera_track_conflict_does_not_mutate_link(tmp_path, monkeypatch):
    from lfs_plugins.portal_account import PortalHTTPError

    def scene(*args, **kwargs):
        raise PortalHTTPError(409, "sync_conflict")

    monkeypatch.setattr(Client, "scene", scene, raising=False)
    service = connected(tmp_path, monkeypatch)
    service._bucket()["links"]["project"] = {"sceneId": "scene", "revision": "old", "metadata": {"title": "Keep"}}
    service.fetch_camera_track("scene")
    finish(service)
    assert service.snapshot()["trackFetch"]["state"] == "failed"
    assert service.snapshot()["links"]["project"]["revision"] == "old"
    assert "changed" in service.message.lower()


def test_v1_migration_preserves_links_and_does_not_invent_commit(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    bucket = service._bucket()
    bucket['links']['project'] = {'sceneId':'scene','revision':'r1','metadata':{'title':'Kept'}}
    service._data['version'] = 1
    service._save()
    recovered = gallery_sync.GallerySync(service.account, tmp_path)
    recovered.refresh()
    finish(recovered)
    link = recovered.snapshot()['links']['project']
    assert link['commitUuid'] == '' and link['sharedFields'] == {}
    assert link['metadata']['title'] == 'Kept'
    assert recovered._data['version'] == 2


def test_presentation_revision_is_adopted_but_shared_change_keeps_guard(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    base = dict(id='scene',revision='old',title='Title',description='',visibility='private',viewerSettings={'cameraPath':None})
    service._bucket()['links']['project'] = gallery_sync.exchange_link(base,'commit')
    remote = dict(base,revision='cover',presentation={'cover':'different'})
    monkeypatch.setattr(Client,'scene',lambda *_: dict(remote))
    assert service._write_revision(Client(service.account),'scene','old') == 'cover'
    remote['title'] = 'Changed on portal'
    assert service._write_revision(Client(service.account),'scene','old') == 'old'


def test_completed_upload_records_exact_prepared_commit(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    path = tmp_path/'scene.ply'
    path.write_bytes(b'ply-data')
    remote = dict(id='scene',revision='new',title='Example',description='',visibility='private',viewerSettings={})
    received=[]
    def upload(_client, _path, metadata, **kwargs):
        received.append(dict(metadata))
        return {'scene':remote}
    monkeypatch.setattr(Client,'upload',upload,raising=False)
    service.queue_upload(path,{'title':'Example','_commitUuid':'prepared-commit','_uploadFormat':'sog'},'project')
    finish(service)
    link=service.snapshot()['links']['project']
    assert link['commitUuid']=='prepared-commit' and link['uploadFormat']=='sog'
    assert link['sharedFields']==gallery_sync.shared_fields(remote)
    assert link['exchangedAt'] > 0
    assert received==[{'title':'Example'}]


def test_publish_as_new_keeps_old_pair_until_success(tmp_path, monkeypatch):
    service=connected(tmp_path,monkeypatch)
    old=dict(id='old',revision='r1',title='Old')
    service._bucket()['links']['project']=gallery_sync.exchange_link(old,'saved')
    path=tmp_path/'scene.ply';path.write_bytes(b'ply-data')
    def paused(*args,**kwargs): raise GalleryTransferCanceled()
    monkeypatch.setattr(Client,'upload',paused,raising=False)
    job=service.queue_upload(path,{'title':'New','_publishAsNew':True},'project')
    finish(service)
    assert service.snapshot()['links']['project']['sceneId']=='old'
    monkeypatch.setattr(Client,'upload',lambda *_a,**_k:{'scene':dict(id='new',revision='r2',title='New')})
    service.resume(job);finish(service)
    assert service.snapshot()['links']['project']['sceneId']=='new'


def test_pull_undo_checks_backup_digest_and_later_local_save(tmp_path, monkeypatch):
    import hashlib
    service=connected(tmp_path,monkeypatch)
    target=tmp_path/'local.licht';target.write_bytes(b'updated')
    backup=tmp_path/'backups'/'old.licht';backup.parent.mkdir();backup.write_bytes(b'original')
    job=downloaded_job(service)
    job['localUpdate']={'backupPath':str(backup),'sha256':hashlib.sha256(b'original').hexdigest()}
    service._save()
    stamp=gallery_sync.file_stamp(target)
    target.write_bytes(b'later-save')
    service.restore_local_backup(target,backup,stamp);finish(service)
    assert target.read_bytes()==b'later-save'
    service.restore_local_backup(target,backup,gallery_sync.file_stamp(target));finish(service)
    assert target.read_bytes()==b'original' and backup.read_bytes()==b'original'


def test_C2_camera_send_records_shared_baseline_and_exchange_times(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    before = {'id':'scene','revision':'old','title':'Title','description':'','visibility':'private','viewerSettings':{}}
    link = gallery_sync.exchange_link(before, 'commit')
    link.update(exchangedAt=1, checkedAt=1)
    service._bucket()['links']['project'] = link
    service._save()
    track = {'version':1,'duration':2,'keyframes':[{'t':0}]}
    after = dict(before, revision='new', viewerSettings={'cameraPath':track})
    monkeypatch.setattr(Client, 'update', lambda *args, **kwargs: after, raising=False)
    service.send_camera_track('scene','old',track)
    finish(service)
    current = service.snapshot()['links']['project']
    assert current['sharedFields'] == gallery_sync.shared_fields(after)
    assert current['commitUuid'] == 'commit'
    assert current['exchangedAt'] > 1 and current['checkedAt'] > 1


def test_D3_refresh_preserves_journal_bytes_and_does_not_interrupt_peer_jobs(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    # Construct the peer before an owner writes a running checkpoint.
    peer = gallery_sync.GallerySync(Account(), tmp_path)
    service._bucket()['jobs'] = [{'id':'job','kind':'upload','project':'p','path':'/source.ply',
        'metadata':{'title':'Title'},'status':'running','completed':0,'total':1,'checkpoint':None,'message':''}]
    service._save()
    journal = tmp_path/'sync.json'
    before, stamp = journal.read_bytes(), journal.stat().st_mtime_ns
    peer.refresh(); finish(peer)
    assert peer.snapshot()['jobs'][0]['status'] == 'running'
    assert not peer.snapshot()['jobs'][0].get('interrupted')
    assert journal.read_bytes() == before and journal.stat().st_mtime_ns == stamp
    # The owner's next write still has a valid guard after another window browses.
    service._save()
    assert not service.snapshot()['storage_issue']


@pytest.mark.parametrize('failure', ['missing', 'digest', 'journal_after_restore'])
def test_D1_restore_worker_reports_failure_and_actual_file_replacement(tmp_path, monkeypatch, failure):
    import hashlib
    service = connected(tmp_path, monkeypatch)
    target = tmp_path / 'local.licht'
    target.write_bytes(b'updated')
    backup = tmp_path / 'backups' / 'original.licht'
    backup.parent.mkdir()
    backup.write_bytes(b'original')
    job = downloaded_job(service)
    job['localUpdate'] = {'backupPath': str(backup), 'sha256': hashlib.sha256(b'original').hexdigest()}
    service._save()
    if failure == 'missing':
        backup.unlink()
    elif failure == 'digest':
        backup.write_bytes(b'corrupt')
    else:
        monkeypatch.setattr(service, '_save', lambda: (_ for _ in ()).throw(OSError('disk full')))
    operation = service.restore_local_backup(target, backup, gallery_sync.file_stamp(target))
    finish(service)
    result = service.snapshot()['undoRestore']
    assert result['id'] == operation
    assert result['state'] == ('restored' if failure == 'journal_after_restore' else 'failed')
    assert result['backupMissing'] == (failure == 'missing')
    assert target.read_bytes() == (b'original' if failure == 'journal_after_restore' else b'updated')


@pytest.mark.parametrize('resumed', [False, True])
def test_A5_finished_upload_records_all_bytes_even_without_final_progress(tmp_path, monkeypatch, resumed):
    service = connected(tmp_path, monkeypatch)
    source = tmp_path / 'scene.ply'
    source.write_bytes(b'x' * 137114)
    attempts = []
    def upload(client, path, metadata, **callbacks):
        attempts.append(callbacks['checkpoint'])
        if resumed and len(attempts) == 1:
            # All parts reached the portal before processing was paused; the
            # last stored counter is the processing count, not uploaded bytes.
            callbacks['on_checkpoint']({'uploadId': 'all-parts', 'idempotencyKey': 'stable'})
            callbacks['on_progress'](137114, 137114)
            callbacks['on_processing']({'stage': 'validating', 'completed': 0, 'total': 137114})
            raise GalleryProcessingPaused()
        # Idempotent resume can return the finished scene without sending parts
        # or emitting any further byte progress callback.
        return {'scene': {'id': 'scene', 'revision': 'ready'}}
    monkeypatch.setattr(Client, 'upload', upload, raising=False)
    identifier = service.queue_upload(source, {'title': 'Scene'}, 'project')
    finish(service)
    if resumed:
        assert service.snapshot()['jobs'][0]['status'] == 'paused'
        service.resume(identifier)
        finish(service)
        assert attempts[-1]['uploadId'] == 'all-parts'
    job = service.snapshot()['jobs'][0]
    assert job['status'] == 'completed'
    assert job['completed'] == job['total'] == 137114
    saved = json.loads((tmp_path / 'sync.json').read_text())
    stored = next(bucket['jobs'][0] for bucket in saved['accounts'].values() if bucket['jobs'])
    assert stored['completed'] == stored['total'] == 137114


def test_saved_project_preparation_journals_source_commit_before_upload(tmp_path, monkeypatch):
    import uuid
    service = connected(tmp_path, monkeypatch)
    service._source_formats = ['licht']
    staging = tmp_path / (str(uuid.uuid4()) + '.scene')
    staging.mkdir()
    (staging / 'project.licht').write_bytes(b'prepared file: packaging intentionally deferred')
    monkeypatch.setattr(service, 'resume', lambda *_: None)
    job_id = service.queue_prepared_upload(staging, {'title': 'Saved project', '_commitUuid': 'verified-source-commit',
        '_uploadFormat': 'ssog', '_contentStamp': 'saved-content'}, 'source-project')
    journal = json.loads((tmp_path / 'sync.json').read_text())
    jobs = [job for bucket in journal['accounts'].values() for job in bucket['jobs']]
    job = next(job for job in jobs if job['id'] == job_id)
    assert job['project'] == 'source-project'
    assert job['commitUuid'] == 'verified-source-commit'
    assert job['uploadFormat'] == 'ssog' and job['contentStamp'] == 'saved-content'
    assert job['preparation'] == str(staging)
    assert '_commitUuid' not in job['metadata']


def test_domain_exchange_tokens_survive_journal_reload(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    scene = {"id": "scene", "revision": "legacy", "contentRevision": "content", "metadataRevision": "metadata", "title": "Title"}
    service._bucket()["links"]["project"] = gallery_sync.exchange_link(scene, "commit")
    service._save()
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    link = restarted.snapshot()["links"]["project"]
    assert {key: link[key] for key in ("revision", "contentRevision", "metadataRevision")} == {
        "revision": "legacy", "contentRevision": "content", "metadataRevision": "metadata"}


def test_304_keeps_scene_cache_and_exchange_baseline(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    scene = {"id": "scene", "revision": "legacy", "title": "Title"}
    link = gallery_sync.exchange_link(scene, "commit")
    service._bucket()["links"]["project"] = link
    service.scenes = [scene]
    old = link["checkedAt"]
    service._list_etag = 'W/"cached"'
    def listing(self, etag=None):
        assert etag == 'W/"cached"'
        self.list_etag = etag
        return None
    monkeypatch.setattr(Client, "list_scenes", listing)
    monkeypatch.setattr(Client, "_request", lambda self, *args: {"id": "one", "gallerySyncVersion": 1, "revisionDomains": 1})
    service.refresh()
    finish(service)
    snap = service.snapshot()
    assert snap["revisionDomains"] == 1
    assert snap["scenes"] == [scene]
    assert snap["links"]["project"]["checkedAt"] >= old
    assert snap["links"]["project"]["exchangedAt"] == link["exchangedAt"]
    assert snap["checkedAt"] >= old
    assert service._list_etag == 'W/"cached"'


def _poster_scene():
    import uuid
    return {"id": str(uuid.uuid4()), "status": "ready", "posterRevision": "poster1", "thumbnailUrl": "https://portal.example/ignored"}


def test_poster_cache_bound_eviction_etag_change_and_sign_out(tmp_path, monkeypatch):
    from pathlib import Path
    service = connected(tmp_path, monkeypatch)
    from lfs_plugins import gallery_preferences
    assert gallery_preferences.read_preferences(tmp_path)["posterCacheMiB"] == 64
    monkeypatch.setattr(gallery_preferences, "read_preferences", lambda root: {"posterCacheMiB": 20 / (1024 * 1024)})
    scenes = [_poster_scene() for _ in range(3)]
    calls = []
    def thumbnail(scene_id, *, etag=None):
        calls.append((scene_id, etag))
        return 200, '"first"', b"0123456789"
    client = SimpleNamespace(thumbnail=thumbnail)
    identity = service.identity()
    service._cache_posters(client, scenes, identity)
    posters = service.snapshot()["posters"]
    assert len(posters) == 2 and scenes[0]["id"] not in posters
    assert sum(path.stat().st_size for path in (tmp_path / "posters").glob("*.png")) == 20
    selected = scenes[-1]
    old_path = Path(posters[selected["id"]])
    assert old_path.name == selected["id"] + "-poster1.png"
    selected["posterRevision"] = "poster2"
    def changed(scene_id, *, etag=None):
        assert etag == '"first"'
        return 200, '"changed"', b"new"
    service._cache_posters(SimpleNamespace(thumbnail=changed), [selected], identity)
    assert not old_path.exists()
    new_path = Path(service.snapshot()["posters"][selected["id"]])
    assert new_path.name.endswith("-poster2.png") and new_path.read_bytes() == b"new"
    def unchanged(scene_id, *, etag=None):
        assert etag == '"changed"'
        return 304, etag, b""
    revision_stamp = new_path.stat().st_mtime_ns
    service._cache_posters(SimpleNamespace(thumbnail=unchanged), [selected], identity)
    assert new_path.read_bytes() == b"new"
    assert new_path.stat().st_mtime_ns == revision_stamp
    monkeypatch.setattr(service.account, "snapshot", lambda: SimpleNamespace(signed_in=False, email="", connected_since=""))
    assert service.snapshot()["posters"] == {}
    assert not (tmp_path / "posters").exists()


def test_poster_response_cannot_repopulate_after_account_switch(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    scene = _poster_scene()
    def thumbnail(*args, **kwargs):
        service.account.email = "other@example.com"
        return 200, '"poster"', b"private image"
    service._cache_posters(SimpleNamespace(thumbnail=thumbnail), [scene], service.identity())
    assert service.snapshot()["posters"] == {}
    assert not (tmp_path / "posters").exists()


def test_metadata_update_keeps_unexchanged_content_baseline(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    service._revision_domains = 1
    scene = {"id": "scene", "revision": "old", "contentRevision": "original", "metadataRevision": "m1", "title": "Title"}
    service._bucket()["links"]["project"] = gallery_sync.exchange_link(scene, "commit")
    def update(self, scene_id, revision, **metadata):
        assert revision["contentRevision"] == "original"
        return {**scene, "revision": "new", "contentRevision": "remote-change", "metadataRevision": "m2", **metadata}
    monkeypatch.setattr(Client, "update", update, raising=False)
    service.edit("scene", "old", {"title": "Edited"})
    finish(service)
    link = service.snapshot()["links"]["project"]
    assert link["contentRevision"] == "original" and link["metadataRevision"] == "m2"


def test_explicitly_reviewed_revision_uses_current_domain_guards(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    service._revision_domains = 1
    old = {"id": "scene", "revision": "old", "contentRevision": "c1", "metadataRevision": "m1"}
    service._bucket()["links"]["project"] = gallery_sync.exchange_link(old)
    service.scenes = [{**old, "revision": "reviewed", "contentRevision": "c2", "metadataRevision": "m2"}]
    guards = service._write_revision(service._client(), "scene", "reviewed")
    assert guards == {"revision": "reviewed", "contentRevision": "c2", "metadataRevision": "m2"}
    assert service._write_revision(service._client(), "scene", "old")["contentRevision"] == "c1"
