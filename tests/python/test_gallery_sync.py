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
    assert "Another Studio window" in service.message
    assert not service.snapshot()["connected"]
    assert json.loads(journal.read_text()) == different
    service.refresh()
    finish(service)
    assert service.snapshot()["connected"]
    assert "other" in service._data["accounts"]
    assert json.loads(journal.read_text()) == different


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
    with pytest.raises(ValueError, match="Another Studio window"):
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


@pytest.mark.parametrize('suffix', ['sog', 'ssog'])
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
