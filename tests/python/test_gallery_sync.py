# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from lfs_plugins import gallery_sync
from lfs_plugins.portal_gallery import GalleryTransferCanceled, GalleryProcessingPaused

class Account:
    base_url = "https://portal.example"
    email = "one@example.com"
    owner = "one"
    busy = False

    def snapshot(self):
        return SimpleNamespace(signed_in=True, email=self.email, connected_since="session", linking=False)

class Client:
    def __init__(self, account, **kwargs):
        self.account = account

    def _request(self, *args):
        return {"storageHosts": ["portal.example"], "id": self.account.owner, "gallerySyncVersion": 1, "revisionDomains": 1}

    def list_scenes(self):
        return []

    def scene(self, scene_id):
        return {"contentRevision": "remote-new", "metadataRevision": "remote-new", "id": scene_id, "revision": "remote-new"}

def finish(service):
    service._thread.join(3)
    assert not service.busy

def connected(tmp_path, monkeypatch):
    monkeypatch.setattr(gallery_sync, "PortalGalleryClient", Client)
    monkeypatch.setattr(gallery_sync, "_project_uuid", lambda _path: "project")
    service = gallery_sync.GallerySync(Account(), tmp_path)
    service.refresh()
    finish(service)
    # Persistence tests start with an explicitly seeded journal; browsing no
    # longer creates or rewrites this file.
    service._save()
    return service


def test_relink_latches_automatic_refresh_but_manual_retry_is_allowed(tmp_path, monkeypatch):
    calls = []
    stages = []

    class RelinkClient(Client):
        def _request(self, *args):
            calls.append(args)
            raise gallery_sync.PortalHTTPError(403, "gallery_relink_required")

    monkeypatch.setattr(gallery_sync, "PortalGalleryClient", RelinkClient)
    monkeypatch.setattr(gallery_sync, "log_stage", lambda name, **values: stages.append((name, values)))
    monkeypatch.setattr(gallery_sync, "log_failure", lambda *_args, **_kwargs: pytest.fail(
        "An expected relink response must not emit an error traceback"
    ))
    service = gallery_sync.GallerySync(Account(), tmp_path)

    service.refresh()
    finish(service)
    assert service.snapshot()["relink_required"] is True
    assert len(calls) == 1
    assert stages == [("relink_required", {"operation": "refresh"})]

    service.refresh()
    assert len(calls) == 1

    service.refresh(force=True)
    finish(service)
    assert len(calls) == 2


@pytest.mark.parametrize("in_worker", [False, True])
@pytest.mark.parametrize("missing", ["email", "connected_since"])
def test_refresh_reports_settled_incomplete_account_without_worker_failure(tmp_path, monkeypatch, in_worker, missing):
    failures = []
    snap = SimpleNamespace(signed_in=True, email="one@example.com", connected_since="session", linking=False, error="")
    account = Account()
    account.snapshot = lambda: snap
    service = gallery_sync.GallerySync(account, tmp_path)
    service.message = ""
    service._refresh_ok = True
    version = service.version
    monkeypatch.setattr(gallery_sync, "log_failure", lambda *args, **kwargs: failures.append((args, kwargs)))
    if in_worker:
        launch = service._launch
        def changed(action, **kwargs):
            setattr(snap, missing, "")
            launch(action, **kwargs)
        monkeypatch.setattr(service, "_launch", changed)
    else:
        setattr(snap, missing, "")

    service.refresh(force=True)
    if in_worker:
        finish(service)
    else:
        assert service._thread is None

    state = service.snapshot()
    assert state["message"] == "projects.gallery.error.account_loading"
    assert state["actionFailure"]["message"] == state["message"]
    assert state["actionFailure"]["identity"] == service.identity()
    assert state["refresh_ok"] is False
    assert state["version"] > version
    assert failures == []


@pytest.mark.parametrize("locale", ["en", "de", "es", "fr", "it", "ja", "ko", "nl", "pl", "zh"])
def test_incomplete_account_sentinel_resolves_through_translation(monkeypatch, locale):
    import lichtfeld as lf
    from lfs_plugins.gallery_messages import localize_message

    path = Path(__file__).resolve().parents[2] / "src/visualizer/gui/resources/locales" / f"{locale}.json"
    translations = json.loads(path.read_text())
    calls = []
    def translate(key):
        calls.append(key)
        return translations[key]
    monkeypatch.setattr(lf.ui, "tr", translate)
    message = "projects.gallery.error.account_loading"
    assert localize_message(message) == translations[message]
    assert translations[message] != translations["projects.gallery.error.access"]
    assert calls == [message]


@pytest.mark.parametrize("in_worker", [False, True])
@pytest.mark.parametrize("pending", ["linking", "busy"])
def test_refresh_waits_silently_for_complete_account_details(tmp_path, monkeypatch, in_worker, pending):
    failures = []

    class LoadingAccount(Account):
        email = ""

        def snapshot(self):
            return SimpleNamespace(signed_in=True, email=self.email, connected_since="", linking=pending == "linking")

    monkeypatch.setattr(gallery_sync, "log_failure", lambda *args, **kwargs: failures.append((args, kwargs)))
    service = gallery_sync.GallerySync(LoadingAccount(), tmp_path)
    service.message = ""
    service.account.busy = pending == "busy"
    if in_worker:
        snapshot = service.account.snapshot
        service.account.snapshot = Account().snapshot
        launch = service._launch
        def changed(action, **kwargs):
            service.account.snapshot = snapshot
            launch(action, **kwargs)
        monkeypatch.setattr(service, "_launch", changed)

    service.refresh(force=True)

    if in_worker:
        finish(service)
    else:
        assert service._thread is None
    assert service.snapshot()["message"] == ""
    assert service.snapshot()["actionFailure"] is None
    assert failures == []

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


def _native_local_project(tmp_path, monkeypatch):
    from pathlib import Path
    from lichtfeld import io

    service = connected(tmp_path / "sync", monkeypatch)
    from lfs_plugins import asset_index
    monkeypatch.setattr(asset_index, "resolve_default_asset_directory", lambda: tmp_path / "assets")
    monkeypatch.setattr(gallery_sync, "_project_uuid", lambda path: str(io.inspect_project_card(path).project_uuid))
    path = tmp_path / "项目-é.licht"
    path.write_bytes((Path(__file__).parents[1] / "data" / "portable-sog.licht").read_bytes())
    card = io.inspect_project_card(path)
    job = downloaded_job(service)
    service._bucket()["links"][str(card.project_uuid)] = service._bucket()["links"].pop("project")
    return service, job, path, card, io


def test_gallery_apply_backup_rechecks_selected_project_id(tmp_path, monkeypatch):
    service, job, path, card, io = _native_local_project(tmp_path, monkeypatch)
    replacement = tmp_path / "replacement.licht"
    io.restore_save(path, 1, replacement)
    path.write_bytes(replacement.read_bytes())
    before = path.read_bytes()
    service.prepare_local_update(job["id"], str(card.project_uuid), path, gallery_sync.file_stamp(path))
    finish(service)
    assert job["localUpdate"]["state"] == "failed"
    assert "identity changed" in job["localUpdate"]["message"]
    assert path.read_bytes() == before and not job["localUpdate"].get("backupPath")


def test_gallery_backup_refuses_destination_that_appears_during_copy(tmp_path, monkeypatch):
    from contextlib import contextmanager

    service, job, path, card, _ = _native_local_project(tmp_path, monkeypatch)
    original = gallery_sync.tempfile.NamedTemporaryFile
    occupied = []

    @contextmanager
    def appeared(**kwargs):
        with original(**kwargs) as output:
            yield output
        if kwargs.get("dir") == service.root / "backups":
            destination = service.root / "backups" / (job["localUpdate"]["id"] + ".licht")
            destination.write_bytes(b"another project")
            occupied.append(destination)

    monkeypatch.setattr(gallery_sync.tempfile, "NamedTemporaryFile", appeared)
    service.prepare_local_update(job["id"], str(card.project_uuid), path, gallery_sync.file_stamp(path))
    finish(service)
    assert job["localUpdate"]["state"] == "failed"
    assert "identity or path changed" in job["localUpdate"]["message"]
    assert occupied[0].read_bytes() == b"another project"


def test_gallery_undo_refuses_a_different_project_with_a_fresh_stamp(tmp_path, monkeypatch):
    service, job, path, card, io = _native_local_project(tmp_path, monkeypatch)
    service.prepare_local_update(job["id"], str(card.project_uuid), path, gallery_sync.file_stamp(path))
    finish(service)
    backup = job["localUpdate"]["backupPath"]
    replacement = tmp_path / "replacement.licht"
    io.restore_save(path, 1, replacement)
    path.write_bytes(replacement.read_bytes())
    before = path.read_bytes()
    service.restore_local_backup(path, backup, gallery_sync.file_stamp(path))
    finish(service)
    assert service._undo_restore["state"] == "failed"
    assert "identity changed" in service._undo_restore["message"]
    assert path.read_bytes() == before


def test_gallery_undo_preserves_unicode_alias(tmp_path, monkeypatch):
    service, job, path, card, io = _native_local_project(tmp_path, monkeypatch)
    original = path.read_bytes()
    alias = tmp_path / "别名-é.licht"
    alias.symlink_to(path)
    service.prepare_local_update(job["id"], str(card.project_uuid), alias, gallery_sync.file_stamp(alias))
    finish(service)
    io.set_project_title(path, "Edited")
    service.restore_local_backup(alias, job["localUpdate"]["backupPath"], gallery_sync.file_stamp(alias))
    finish(service)
    assert service._undo_restore["state"] == "restored"
    assert alias.is_symlink() and path.read_bytes() == original


@pytest.mark.parametrize("swap", ["source", "destination", "directory", "none"])
def test_download_staging_refuses_changed_source_or_destination(tmp_path, monkeypatch, swap):
    import shutil
    import uuid
    from pathlib import Path
    from lfs_plugins import gallery_preparation

    service, job, path, card, io = _native_local_project(tmp_path, monkeypatch)
    identifier = str(uuid.uuid4())
    source = service.root / "downloads" / (identifier + ".licht")
    source.parent.mkdir()
    shutil.copyfile(path, source)
    destination = tmp_path / "输出" / "项目.licht"
    destination.parent.mkdir()
    job.update(id=identifier, path=str(source), total=source.stat().st_size,
               destination=str(destination), destinationPath=str(destination.resolve()),
               downloadProject=str(card.project_uuid))
    job["result"]["title"] = "Downloaded"
    replacement = tmp_path / "replacement.licht"
    io.restore_save(path, 1, replacement)
    other_bytes = replacement.read_bytes()
    original_unpack = gallery_preparation.unpack_project
    original_restore = io.restore_save

    def unpack(*args, **kwargs):
        result = original_unpack(*args, **kwargs)
        if swap == "source":
            source.write_bytes(other_bytes)
        elif swap == "directory":
            redirected = tmp_path / "redirected"
            redirected.mkdir()
            destination.parent.rename(tmp_path / "original-output")
            destination.parent.symlink_to(redirected, target_is_directory=True)
        return result

    def restore(source_path, generation, output):
        Path(output).write_bytes(other_bytes)
        return original_restore(source_path, generation, output)

    monkeypatch.setattr(gallery_preparation, "unpack_project", unpack)
    if swap == "destination":
        monkeypatch.setattr(io, "restore_save", restore)
    service.stage_download(identifier)
    finish(service)
    stage = job["stagedImport"]
    if swap == "none":
        assert stage["state"] == "ready", stage.get("message")
        assert stage["projectId"] == str(io.inspect_project_card(destination).project_uuid)
        assert stage["projectId"] != str(card.project_uuid)
        assert stage["projectStamp"] == gallery_sync.file_stamp(destination)
        return
    assert stage["state"] == "failed"
    assert ("already exists" if swap == "destination" else "changed") in stage["message"]
    if swap == "destination":
        assert destination.read_bytes() == other_bytes
    else:
        assert not destination.exists()


@pytest.mark.parametrize("operation", ["download_link", "settings_link"])
@pytest.mark.parametrize("swap", ["identity", "path"])
def test_gallery_link_rechecks_project_at_journal_replacement(tmp_path, monkeypatch, operation, swap):
    import copy
    import os

    service, job, path, card, io = _native_local_project(tmp_path, monkeypatch)
    project_id = str(card.project_uuid)
    alias = tmp_path / "别名.licht"
    alias.symlink_to(path)
    service.prepare_local_update(job["id"], project_id, alias, gallery_sync.file_stamp(alias))
    finish(service)
    job["project"] = project_id
    service._save()
    original_link = copy.deepcopy(service._bucket()["links"][project_id])
    original_update = copy.deepcopy(job["localUpdate"])
    saved_journal = service._journal.read_bytes()
    replacement = tmp_path / "replacement.licht"
    if swap == "identity":
        io.restore_save(path, 1, replacement)
    else:
        replacement.write_bytes(path.read_bytes())
    original_fsync = gallery_sync.os.fsync
    swapped = False

    def fsync(fd):
        nonlocal swapped
        original_fsync(fd)
        if swapped:
            return
        swapped = True
        if swap == "identity":
            os.replace(replacement, path)
        else:
            alias.unlink()
            alias.symlink_to(replacement)

    monkeypatch.setattr(gallery_sync.os, "fsync", fsync)
    if operation == "download_link":
        service.link_download(job["id"], project_id, "new", project_path=alias)
    else:
        service.finish_settings_update(job["id"], "new", gallery_sync.file_stamp(alias), {})
    finish(service)
    assert swapped
    assert "identity or path changed" in service.snapshot()["actionFailure"]["message"]
    assert service._bucket()["links"][project_id] == original_link
    assert job["localUpdate"] == original_update
    assert service._journal.read_bytes() == saved_journal


def test_metadata_failure_survives_refresh_and_stays_with_its_account(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    def fail():
        raise ValueError("The cover changed. Check the gallery before trying again.")
    service._launch_metadata(fail)
    finish(service)
    failure = service.snapshot()["actionFailure"]
    assert failure["message"].startswith("The cover changed.")
    service.refresh()
    finish(service)
    assert service.snapshot()["actionFailure"] == failure
    service._launch_metadata(fail)
    finish(service)
    assert service.snapshot()["actionFailure"]["id"] != failure["id"]
    service.account.email = "two@example.com"
    assert service.snapshot()["actionFailure"] is None

def test_resume_after_restart_reuses_checkpoint_and_links_project(tmp_path, monkeypatch):
    def pause(self, path, metadata, **kwargs):
        kwargs["on_checkpoint"]({"uploadId": "pending", "idempotencyKey": "stable"})
        kwargs["on_progress"](4, 8)
        raise GalleryTransferCanceled()
    monkeypatch.setattr(Client, "upload", pause, raising=False)
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / "scene.licht"
    path.write_bytes(b"ply-data")
    job = service.queue_upload(path, {"title": "Example"}, "project-uuid")
    finish(service)
    assert service.snapshot()["jobs"][0]["status"] == "paused"
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    def resume(self, path, metadata, **kwargs):
        assert kwargs["checkpoint"] == {"uploadId": "pending", "idempotencyKey": "stable"}
        return {"scene": {"contentRevision": "new", "metadataRevision": "new", "id": "remote-scene", "revision": "new", "title": "Example"}}
    monkeypatch.setattr(Client, "upload", resume)
    restarted.resume(job)
    finish(restarted)
    state = restarted.snapshot()
    assert state["links"]["project-uuid"]["sceneId"] == "remote-scene"
    assert state["jobs"][0]["status"] == "completed"
    assert json.loads((tmp_path / "sync.json").read_text())["accounts"]

def test_server_processing_is_visible_and_never_linked_before_completion(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / "scene.licht"
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
    path = tmp_path / "scene.licht"
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
    return {"contentRevision": "r1", "metadataRevision": "r1", "id": "scene", "revision": "r1", "sourceFormat": "licht", "title": "Garden", "contentLength": 8}

def test_download_survives_launch_failure_and_restart(tmp_path, monkeypatch):
    from pathlib import Path
    service = connected(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "resume", lambda *_: (_ for _ in ()).throw(RuntimeError("thread start failed")))
    service.download(_download_scene())
    job = service.snapshot()["jobs"][0]
    assert job["kind"] == "download" and job["status"] == "paused"
    assert job["sceneId"] == "scene" and "revision" not in job
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
        return {"contentRevision": "r1", "metadataRevision": "r1", "id": scene_id, "revision": "r1", "title": "Garden"}
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
    different = {"version": 2, "accounts": {"other": {"jobs": [], "links": {}}}}
    journal.write_text(json.dumps(different))
    monkeypatch.setattr(service, "resume", lambda *_: pytest.fail("Stale downloads must not start"))
    with pytest.raises(ValueError, match="Another LichtFeld Studio window"):
        service.download(_download_scene())
    assert json.loads(journal.read_text()) == different
    assert service.snapshot()["jobs"] == []

def test_upload_cannot_silently_retarget_project_link(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    service._bucket()["links"]["project"] = {"contentRevision": "r", "metadataRevision": "r", "sceneId": "original", "revision": "r"}
    path = tmp_path / "scene.licht"
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
    path = tmp_path / "scene.licht"
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
        service.queue_upload(tmp_path / "scene.licht", {"title": "Duplicate"}, "project")
    assert path.read_text() == contents

def test_repaired_journal_retries_without_restart_and_preserves_pending_key(tmp_path, monkeypatch):
    monkeypatch.setattr(Client, "upload", lambda *a, **kw: (_ for _ in ()).throw(GalleryTransferCanceled()), raising=False)
    original = connected(tmp_path, monkeypatch)
    path = tmp_path / "scene.licht"
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
    assert service._thread is None
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
    different = {"version": 2, "accounts": {"other": {"jobs": [], "links": {}}}}
    journal.write_text(json.dumps(different))
    monkeypatch.setattr(Client, "update", lambda *a, **kw: pytest.fail("Stale mutations must not reach the network"), raising=False)
    service.edit("scene", {"contentRevision": "old", "metadataRevision": "old"}, {"title": "Old edit"})
    finish(service)
    assert "Another LichtFeld Studio window" in service.message
    assert not service.snapshot()["connected"]
    assert json.loads(journal.read_text()) == different
    service.refresh()
    finish(service)
    assert service.snapshot()["connected"]
    assert "other" in service._data["accounts"]
    persisted = json.loads(journal.read_text())
    assert persisted == different  # A refresh does not rewrite a peer journal.
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
        return {"scene": {"contentRevision": "new", "metadataRevision": "new", "id": "remote", "revision": "new"}}

    monkeypatch.setattr(Client, "upload", upload, raising=False)
    path = tmp_path / "scene.licht"
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
    path = tmp_path / "scene.licht"
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
    monkeypatch.setattr(Client, "upload", lambda *a, **kw: {"scene": {"contentRevision": "r", "metadataRevision": "r", "id": "scene", "revision": "r"}}, raising=False)
    service = connected(tmp_path, monkeypatch)
    snapshot = tmp_path / (str(uuid.uuid4()) + ".licht")
    snapshot.write_bytes(b"ply-data")
    service.queue_upload(snapshot, {"title": "Owned snapshot"}, "one", owned_export=True)
    finish(service)
    assert not snapshot.exists()
    external = tmp_path / (str(uuid.uuid4()) + ".licht")
    external.write_bytes(b"user-data")
    service.queue_upload(external, {"title": "User file"}, "two")
    finish(service)
    assert external.read_bytes() == b"user-data"

def downloaded_job(service):
    path = service.root / 'download.licht'
    path.write_bytes(b'download')
    job = {"contentRevision": "remote-new", "metadataRevision": "remote-new", "id": "download", "project": "", "kind": "download", "status": "completed",
        "path": str(path), "total": 8, "completed": 8, "metadata": {"title": "Downloaded"},
        "message": "Downloaded", "checkpoint": None, "sceneId": "scene", "revision": "remote-new",
        "result": {"contentRevision": "remote-new", "metadataRevision": "remote-new", "id": "scene", "revision": "remote-new"}}
    service._bucket()["jobs"].append(job)
    service._bucket()["links"]["project"] = {"contentRevision": "old", "metadataRevision": "old", "sceneId": "scene", "revision": "old"}
    return job

def cleanup_download(service, *, backup=False):
    from uuid import uuid4
    identifier, stage_id = str(uuid4()), str(uuid4())
    path = service.root / "downloads" / (identifier + ".licht")
    stage = service.root / "imports" / (stage_id + ".licht")
    for item in (path, stage):
        item.parent.mkdir(exist_ok=True)
        item.write_bytes(b"downloaded bytes")
    job = dict(id=identifier, kind="download", project="project", status="completed", path=str(path),
        sceneId="scene", revision="r", result={"contentRevision": "r", "metadataRevision": "r", "id": "scene", "revision": "r", "title": "Scene"},
        metadata={"title": "Scene"}, checkpoint=None, completed=16, total=16, message="Downloaded",
        stagedImport={"id": stage_id, "state": "ready", "path": str(stage)}, contentRevision="r", metadataRevision="r")
    if backup:
        saved = service.root / "backups" / (str(uuid4()) + ".licht")
        saved.parent.mkdir(exist_ok=True)
        saved.write_bytes(b"independent saved project")
        job["localUpdate"] = {"id": saved.stem, "state": "ready", "backupPath": str(saved)}
    service._bucket()["jobs"].append(job)
    service._bucket()["links"]["project"] = {"contentRevision": "r", "metadataRevision": "r", "sceneId": "scene", "revision": "r"}
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
    assert service.snapshot()["links"]["project"] == {"contentRevision": "r", "metadataRevision": "r", "sceneId": "scene", "revision": "r"}
    kept = service.snapshot()["jobs"][0]
    assert kept["retired"] and kept["path"] == "" and "stagedImport" not in kept
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    assert restarted.snapshot()["jobs"][0] == kept

def test_clear_history_never_deletes_external_upload_source(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / "my-original.licht"
    path.write_bytes(b"user source")
    monkeypatch.setattr(Client, "upload", lambda *a, **kw: {"scene": {"contentRevision": "r", "metadataRevision": "r", "id": "scene", "revision": "r"}}, raising=False)
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
        moved = tmp_path / "original.licht"
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
    monkeypatch.setattr(Client, "scene", lambda *_: {"contentRevision": "newer-remote", "metadataRevision": "newer-remote", "revision": "newer-remote"})
    service.prepare_local_update(job["id"], "project", source, gallery_sync.file_stamp(source))
    finish(service)
    assert job["localUpdate"]["state"] == "failed"
    assert "changed since this download" in job["localUpdate"]["message"]
    assert service.snapshot()["links"]["project"]["revision"] == "old"

def test_failed_link_save_is_not_reported_as_a_completed_update(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    job = downloaded_job(service)
    path = tmp_path / "saved.licht"
    path.write_bytes(b"saved project")
    monkeypatch.setattr(service, "_save", lambda **kw: (_ for _ in ()).throw(OSError("disk full")))
    operation = service.link_download(job["id"], "project", project_path=path)
    finish(service)
    assert job["linkOperation"]["id"] == operation
    assert job["linkOperation"]["state"] == "failed"
    assert service.snapshot()["links"]["project"]["revision"] == "old"
    assert job["project"] == ""


def test_completed_upload_records_exact_prepared_commit(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    path = tmp_path/'scene.licht'
    path.write_bytes(b'ply-data')
    remote = dict(id='scene',revision='new',title='Example',description='',visibility='private',viewerSettings={}, contentRevision='new', metadataRevision='new')
    received=[]
    def upload(_client, _path, metadata, **kwargs):
        received.append(dict(metadata))
        return {'scene':remote}
    monkeypatch.setattr(Client,'upload',upload,raising=False)
    job = service.queue_upload(path,{'title':'Example','_commitUuid':'prepared-commit','_uploadFormat':'sog'},'project')
    finish(service)
    link=service.snapshot()['links']['project']
    assert link['commitUuid']=='prepared-commit' and link['uploadFormat']=='sog'
    assert link['sharedFields']==gallery_sync.shared_fields(remote)
    assert link['exchangedAt'] > 0
    assert received == [dict(title='Example', originProjectUuid='project', originCommitUuid='prepared-commit', clientMutationId=job)]

def test_publish_as_new_keeps_old_pair_until_success(tmp_path, monkeypatch):
    service=connected(tmp_path,monkeypatch)
    old=dict(id='old',revision='r1',title='Old', contentRevision='r1', metadataRevision='r1')
    service._bucket()['links']['project']=gallery_sync.exchange_link(old,'saved')
    path=tmp_path/'scene.licht';path.write_bytes(b'ply-data')
    def paused(*args,**kwargs): raise GalleryTransferCanceled()
    monkeypatch.setattr(Client,'upload',paused,raising=False)
    job=service.queue_upload(path,{'title':'New','_publishAsNew':True},'project')
    finish(service)
    assert service.snapshot()['links']['project']['sceneId']=='old'
    monkeypatch.setattr(Client,'upload',lambda *_a,**_k:{'scene':dict(id='new',revision='r2',title='New', contentRevision='r2', metadataRevision='r2')})
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

def test_D3_refresh_preserves_journal_bytes_and_does_not_interrupt_peer_jobs(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    # Construct the peer before an owner writes a running checkpoint.
    peer = gallery_sync.GallerySync(Account(), tmp_path)
    service._bucket()['jobs'] = [{'id':'job','kind':'upload','project':'p','path':'/source.licht',
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
    source = tmp_path / 'scene.licht'
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
        return {'scene': {"contentRevision": 'ready', "metadataRevision": 'ready', 'id': 'scene', 'revision': 'ready'}}
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
    assert {key: link[key] for key in ("contentRevision", "metadataRevision")} == {
        "contentRevision": "content", "metadataRevision": "metadata"}

def test_old_portal_walk_keeps_exchange_baseline(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    scene = {"contentRevision": "legacy", "metadataRevision": "legacy", "id": "scene", "revision": "legacy", "title": "Title"}
    link = gallery_sync.exchange_link(scene, "commit")
    service._bucket()["links"]["project"] = link
    service.scenes = [scene]
    old = link["checkedAt"]
    service._list_etag = 'W/"cached"'
    def listing(self, etag=None):
        assert etag is None
        self.list_etag = 'W/"fresh"'
        return [scene]
    monkeypatch.setattr(Client, "list_scenes", listing)
    monkeypatch.setattr(Client, "_request", lambda self, *args: {"storageHosts": ["portal.example"], "id": "one", "gallerySyncVersion": 1, "revisionDomains": 1})
    service.refresh()
    finish(service)
    snap = service.snapshot()
    assert snap["revisionDomains"] == 1
    assert snap["scenes"] == [scene]
    assert snap["links"]["project"]["checkedAt"] >= old
    assert snap["links"]["project"]["exchangedAt"] == link["exchangedAt"]
    assert snap["checkedAt"] >= old
    assert service._list_etag == 'W/"fresh"'

def test_304_keeps_scene_cache_and_exchange_baseline(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    scene = {"contentRevision": "legacy", "metadataRevision": "legacy", "id": "scene", "revision": "legacy", "title": "Title"}
    link = gallery_sync.exchange_link(scene, "commit")
    service._bucket()["links"]["project"] = link
    service.scenes = [scene]
    old = link["checkedAt"]
    service._list_etag = 'W/"cached"'
    # P4 accepts 304 only for an owner-wide listing with a known sequence.
    service._change_sequence = 10
    def unavailable(*_):
        raise gallery_sync.PortalHTTPError(404, "Not found")
    def listing(self, etag=None, *, owner_wide=False):
        assert etag == 'W/"cached"' and owner_wide
        self.list_etag = etag
        return None
    monkeypatch.setattr(Client, "changes_since", unavailable, raising=False)
    monkeypatch.setattr(Client, "list_scenes", listing)
    monkeypatch.setattr(Client, "_request", lambda self, *args: {"storageHosts": ["portal.example"], "id": "one", "gallerySyncVersion": 1, "revisionDomains": 1})
    service.refresh()
    finish(service)
    snap = service.snapshot()
    assert snap["refresh_ok"] and snap["revisionDomains"] == 1
    assert snap["scenes"] == [scene]
    assert snap["links"]["project"]["checkedAt"] >= old
    assert snap["links"]["project"]["exchangedAt"] == link["exchangedAt"]
    assert snap["links"]["project"]["commitUuid"] == "commit"
    assert snap["checkedAt"] >= old and snap["changeSequence"] == 10
    assert service._list_etag == 'W/"cached"'

    service._change_sequence = None
    monkeypatch.setattr(Client, "list_scenes", lambda self: [dict(id="old", title="Full old portal walk")])
    service.refresh()
    finish(service)
    assert service.snapshot()["scenes"][0]["id"] == "old"

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
    service.edit("scene", scene, {"title": "Edited"})
    finish(service)
    link = service.snapshot()["links"]["project"]
    assert link["contentRevision"] == "original" and link["metadataRevision"] == "m2"

def test_reviewed_domain_guards_are_used_without_cached_lookup(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    reviewed = {"contentRevision": "c1", "metadataRevision": "m1"}
    service.scenes = [{"id": "scene", "contentRevision": "c2", "metadataRevision": "m2"}]
    calls = []
    def update(self, scene_id, guards, **metadata):
        calls.append(guards)
        return {"id": scene_id, **guards, **metadata}
    monkeypatch.setattr(Client, "update", update, raising=False)
    service.edit("scene", reviewed, {"title": "Reviewed"})
    finish(service)
    assert calls == [reviewed]
    service.edit("scene", "broad-hash", {"title": "Invalid"})
    finish(service)
    assert calls == [reviewed]
    assert "revision tokens" in service.message


def test_incremental_check_and_expired_feed_full_walk(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    service._change_sequence = 10
    service.scenes = [dict(id="scene", title="Before")]
    def changes(self, since):
        assert since == 10
        self.change_sequence = 11
        return [dict(type="upsert", sceneId="scene", scene=dict(id="scene", title="After"))]
    monkeypatch.setattr(Client, "changes_since", changes, raising=False)
    service.refresh(); finish(service)
    assert service.snapshot()["scenes"][0]["title"] == "After"
    assert service.snapshot()["changeSequence"] == 11
    def expired(*_):
        raise gallery_sync.PortalHTTPError(409, "resync_required")
    monkeypatch.setattr(Client, "changes_since", expired)
    monkeypatch.setattr(Client, "list_scenes", lambda *_: [dict(id="full", title="Fresh")])
    service.refresh(); finish(service)
    assert service.snapshot()["scenes"][0]["id"] == "full"
    assert service.snapshot()["refresh_ok"]


def handoff_upload(service, tmp_path):
    old = dict(id="scene", title="Title", contentRevision="c", metadataRevision="m")
    service._bucket()["links"]["old"] = gallery_sync.exchange_link(old, "old-save")
    service._save()
    path = tmp_path / "prepared.licht"
    path.write_bytes(b"viewing-copy")
    handoff = dict(oldProject="old", newProject="new", sceneId="scene", origin=service._origin,
                   owner=service._owner, commitUuid="pinned-save", fileUuid="pinned-file",
                   baseRevisions=dict(content="c", metadata="m"), state="pending")
    metadata = dict(title="Title", replaceSceneId="scene", baseRevisions=dict(content="c", metadata="m"),
                    _commitUuid="pinned-save", _handoff=handoff)
    return path, metadata


def test_replacement_handoff_survives_restart_and_retires_only_old_link(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    path, metadata = handoff_upload(service, tmp_path)
    def pause(self, path, metadata, **kwargs):
        kwargs["on_checkpoint"]({"uploadId": "upload", "idempotencyKey": "stable"})
        raise GalleryTransferCanceled()
    monkeypatch.setattr(Client, "upload", pause, raising=False)
    identifier = service.queue_upload(path, metadata, "new")
    finish(service)
    assert set(service.snapshot()["links"]) == {"old"}
    saved = json.loads(service._journal.read_text())
    assert saved["version"] == 3
    assert gallery_sync._validate_journal(saved)
    saved["version"] = 2
    with pytest.raises(ValueError):
        gallery_sync._validate_journal(saved)
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh(); finish(restarted)
    monkeypatch.setattr(Client, "upload", lambda *_a, **_k: {"scene": dict(id="scene", title="Title", contentRevision="new-c", metadataRevision="m")})
    restarted.resume(identifier); finish(restarted)
    assert set(restarted.snapshot()["links"]) == {"new"}
    assert restarted.snapshot()["links"]["new"]["commitUuid"] == "pinned-save"
    assert restarted.snapshot()["jobs"][0]["handoff"]["state"] == "completed"
    assert gallery_sync._validate_journal(json.loads(restarted._journal.read_text()))


def test_handoff_rejects_changed_old_link_and_preserves_links_when_commit_fails(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    path, metadata = handoff_upload(service, tmp_path)
    metadata["_handoff"]["commitUuid"] = "wrong-save"
    with pytest.raises(ValueError, match="previous gallery link"):
        service.queue_upload(path, metadata, "new")
    metadata["_handoff"]["commitUuid"] = "pinned-save"
    real_save = service._save
    def save():
        if service._bucket()["jobs"] and service._bucket()["jobs"][0]["status"] == "completed":
            raise OSError("disk full")
        real_save()
    monkeypatch.setattr(service, "_save", save)
    monkeypatch.setattr(Client, "upload", lambda *_a, **_k: {"scene": dict(id="scene", title="Title", contentRevision="new-c", metadataRevision="m")}, raising=False)
    service.queue_upload(path, metadata, "new"); finish(service)
    assert set(service.snapshot()["links"]) == {"old"}
    assert service.snapshot()["jobs"][0]["handoff"]["state"] == "pending"




def test_downloaded_copy_does_not_block_original_project_updates(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    scene = dict(id="scene", title="Title", contentRevision="c", metadataRevision="m")
    service._bucket()["links"] = {
        "original": gallery_sync.exchange_link(scene, "saved"),
        "viewing-copy": dict(gallery_sync.exchange_link(scene, "copy-save"), viewingCopy=True)}
    service._save()
    path = tmp_path / "scene.licht"
    path.write_bytes(b"prepared copy")
    metadata = dict(title="Updated", replaceSceneId="scene", baseRevisions=dict(content="c", metadata="m"))
    with pytest.raises(ValueError, match="previous project"):
        service.queue_upload(path, metadata, "unlinked-new-project")
    monkeypatch.setattr(Client, "upload", lambda *_a, **_k: {"scene": dict(scene, contentRevision="c2")}, raising=False)
    service.queue_upload(path, metadata, "original"); finish(service)
    assert set(service.snapshot()["links"]) == {"original", "viewing-copy"}
    assert service.snapshot()["links"]["original"]["contentRevision"] == "c2"
    assert service.snapshot()["links"]["viewing-copy"]["contentRevision"] == "c"



def test_compact_changes_feed_fetches_view_settings_before_marking_checked(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    service._change_sequence = 10
    service.scenes = [dict(id="scene", status="ready", title="Before", viewerSettings={"exposure": 1})]
    def changes(self, since):
        self.change_sequence = 11
        return [dict(type="upsert", sceneId="scene", scene=dict(id="scene", status="ready", title="After"))]
    monkeypatch.setattr(Client, "changes_since", changes, raising=False)
    monkeypatch.setattr(Client, "scene", lambda *_: dict(id="scene", status="ready", title="After", viewerSettings={"exposure": 2}))
    service.refresh(); finish(service)
    assert service.snapshot()["scenes"][0]["viewerSettings"] == {"exposure": 2}
    assert service.snapshot()["refresh_ok"]
    checked = service.snapshot()["checkedAt"]
    monkeypatch.setattr(Client, "scene", lambda *_: (_ for _ in ()).throw(OSError("connection lost")))
    service.refresh(); finish(service)
    assert service.snapshot()["checkedAt"] == checked
    assert not service.snapshot()["refresh_ok"]



def test_metadata_publish_acknowledges_only_the_project_that_sent_it(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    scene = dict(id="scene", title="Before", contentRevision="c", metadataRevision="m", viewerSettings={})
    service._bucket()["links"] = {"original": gallery_sync.exchange_link(scene, "old"),
                                  "copy": gallery_sync.exchange_link(scene, "copy-save")}
    service._save()
    monkeypatch.setattr(Client, "update", lambda *_a, **_k: dict(scene, title="After", metadataRevision="m2"), raising=False)
    service.edit("scene", scene, {"title": "After"}, commit_uuid="new", project_id="original")
    finish(service)
    links = service.snapshot()["links"]
    assert links["original"]["commitUuid"] == "new" and links["original"]["metadataRevision"] == "m2"
    assert links["copy"]["commitUuid"] == "copy-save" and links["copy"]["metadataRevision"] == "m"



def test_settings_only_backup_and_undo_survive_restart(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    remote = dict(id="remote", title="Gallery title", contentRevision="c1", metadataRevision="m2",
                  viewerSettings={}, description="", visibility="private")
    monkeypatch.setattr(Client, "scene", lambda *args: remote)
    path = tmp_path / "master.licht"
    original = b"geometry and checkpoint before applying settings"
    path.write_bytes(original)
    service._bucket()["links"]["project"] = gallery_sync.exchange_link(dict(remote, metadataRevision="m1"), "before")
    service._save()
    job_id, backup_id = service.prepare_settings_update(remote, "project", str(path), gallery_sync.file_stamp(path))
    finish(service)
    update = service._job(job_id)["localUpdate"]
    assert update["id"] == backup_id and update["state"] == "ready"
    assert gallery_sync.Path(update["backupPath"]).read_bytes() == original
    path.write_bytes(original + b" settings")
    service.finish_settings_update(job_id, "after", gallery_sync.file_stamp(path), gallery_sync.shared_fields(remote))
    finish(service)
    assert service._job(job_id)["localUpdate"]["state"] == "applied"
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    assert restarted._job(job_id)["localUpdate"]["state"] == "applied"
    link = restarted.snapshot()["links"]["project"]
    assert link["commitUuid"] == "after" and link["contentRevision"] == "c1" and link["metadataRevision"] == "m2"
    restarted.restore_local_backup(str(path), update["backupPath"], gallery_sync.file_stamp(path))
    finish(restarted)
    assert path.read_bytes() == original

    restored = restarted.snapshot()["links"]["project"]
    assert restored["commitUuid"] == "before" and restored["metadataRevision"] == "m1"



def test_settings_only_refuses_remote_changes_during_apply(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    remote = dict(id="remote", title="Gallery title", contentRevision="c1", metadataRevision="m2")
    monkeypatch.setattr(Client, "scene", lambda *args: dict(remote))
    path = tmp_path / "master.licht"
    path.write_bytes(b"unchanged geometry and checkpoint")
    service._bucket()["links"]["project"] = gallery_sync.exchange_link(remote, "before")
    service._save()
    job, _ = service.prepare_settings_update(remote, "project", str(path), gallery_sync.file_stamp(path))
    finish(service)
    remote["metadataRevision"] = "m3"
    service.finish_settings_update(job, "after", gallery_sync.file_stamp(path), {})
    finish(service)
    assert service.snapshot()["links"]["project"]["commitUuid"] == "before"
    assert service._job(job)["localUpdate"]["state"] == "ready"
    assert "changed" in service.message



def test_replacement_choice_is_durable_before_native_preparation(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    path, metadata = handoff_upload(service, tmp_path)
    handoff = service.remember_replacement(metadata["_handoff"])
    assert not service.snapshot()["jobs"]
    assert set(service.snapshot()["links"]) == {"old"}
    saved = json.loads(service._journal.read_text())
    assert saved["version"] == 3
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh(); finish(restarted)
    assert restarted.remember_replacement(metadata["_handoff"]) == handoff
    with pytest.raises(ValueError, match="saved replacement choice"):
        restarted.queue_upload(path, dict(title="Old project update", replaceSceneId="scene"), "old")
    metadata["_handoff"] = handoff
    monkeypatch.setattr(Client, "upload", lambda *_a, **_k: {"scene": dict(id="scene", title="Title", contentRevision="new-c", metadataRevision="m")}, raising=False)
    restarted.queue_upload(path, metadata, "new"); finish(restarted)
    assert set(restarted.snapshot()["links"]) == {"new"}
    assert restarted.snapshot()["handoffIntents"] == {}



def test_reviewed_replacement_keeps_separate_local_and_gallery_guards(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    path, metadata = handoff_upload(service, tmp_path)
    raw = dict(metadata["_handoff"], oldLinkRevisions={"content": "c", "metadata": "m"},
               baseRevisions={"content": "new-remote-c", "metadata": "new-remote-m"})
    handoff = service.remember_replacement(raw)
    metadata.update(_handoff=handoff, baseRevisions=handoff["baseRevisions"])
    monkeypatch.setattr(Client, "upload", lambda *_a, **_k: {"scene": dict(id="scene", title="Title", contentRevision="published-c", metadataRevision="new-remote-m")}, raising=False)
    service.queue_upload(path, metadata, "new"); finish(service)
    assert set(service.snapshot()["links"]) == {"new"}


def test_publish_cover_adds_only_a_thumbnail_to_the_prepared_copy(tmp_path):
    import base64
    import io
    import shutil
    from pathlib import Path
    from lfs_plugins import gallery_preparation, portable_project
    original = Path(__file__).parents[1] / "data" / "portable-sog.licht"
    prepared = tmp_path / "project.licht"
    shutil.copyfile(original, prepared)
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jkWQAAAAASUVORK5CYII=")
    gallery_preparation.attach_preview(prepared, png)
    with original.open("rb") as before, prepared.open("rb") as after:
        a, b = portable_project.ProjectFile(before), portable_project.ProjectFile(after)
        assert b.chapters[b"THMB"] == png
        assert {k: v for k, v in b.chapters.items() if k != b"THMB"} == a.chapters
        for index in range(len(a.manifest["nodes"])):
            first, second = io.BytesIO(), io.BytesIO()
            a.copy_node(index, first)
            b.copy_node(index, second)
            assert first.getvalue() == second.getvalue()
    content = prepared.read_bytes()
    gallery_preparation.attach_preview(prepared, png)
    assert prepared.read_bytes() == content
    with pytest.raises(ValueError, match="PNG"):
        gallery_preparation.attach_preview(prepared, b"private text")
    assert prepared.read_bytes() == content


def test_settings_undo_survives_its_follow_up_upload(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    remote = dict(id="remote", title="Gallery title", contentRevision="c1", metadataRevision="m2",
                  viewerSettings={"exposure": 1.6}, description="", visibility="private")
    monkeypatch.setattr(Client, "scene", lambda *args: remote)
    path = tmp_path / "master.licht"
    path.write_bytes(b"original geometry and checkpoint")
    before = gallery_sync.exchange_link(dict(remote, metadataRevision="m1"), "before")
    service._bucket()["links"]["project"] = gallery_sync.copy.deepcopy(before)
    service._save()
    job_id, _ = service.prepare_settings_update(remote, "project", str(path), gallery_sync.file_stamp(path))
    finish(service)
    path.write_bytes(b"original geometry and checkpoint with reviewed settings")
    fields = dict(gallery_sync.shared_fields(remote), viewerSettings={"exposure": 1.0})
    service.finish_settings_update(job_id, "after", gallery_sync.file_stamp(path), fields, acknowledge=False)
    finish(service)
    published = dict(remote, **fields)
    published.update(contentRevision="c2", metadataRevision="m3")
    monkeypatch.setattr(Client, "upload", lambda *_a, **_k: {"scene": published}, raising=False)
    service.queue_upload(path, dict(fields, replaceSceneId="remote", baseRevisions={"content": "c1", "metadata": "m2"},
                                   _commitUuid="after"), "project")
    finish(service)
    update = service._job(job_id)["localUpdate"]
    # A saved undo record may predate the removal of visibility from shared fields.
    update["appliedLink"]["localFields"] = dict(fields, visibility="private")
    current = service.snapshot()["links"]["project"]
    assert gallery_sync.same_undo_link(current, update["appliedLink"])
    assert not gallery_sync.same_undo_link(dict(current, sharedFields=dict(fields, title="Changed")), update["appliedLink"])
    assert update["previousLink"] == before
    service.restore_local_backup(str(path), update["backupPath"], gallery_sync.file_stamp(path))
    finish(service)
    assert path.read_bytes() == b"original geometry and checkpoint"
    assert service.snapshot()["links"]["project"] == before


@pytest.mark.parametrize('paused', [False, True])
def test_owned_upload_failure_removes_copies_but_pause_keeps_them(tmp_path, monkeypatch, paused):
    import uuid
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / (str(uuid.uuid4()) + '.licht')
    path.write_bytes(b'prepared upload')
    def fail(*_args, **_kwargs):
        raise GalleryTransferCanceled() if paused else ValueError('upload marker')
    monkeypatch.setattr(Client, 'upload', fail, raising=False)
    identifier = service.queue_upload(path, {'title': 'Scene'}, 'project', owned_export=True)
    finish(service)
    job = service.snapshot()['jobs'][0]
    assert path.exists() is paused
    assert bool(job.get('requiresPreparation')) is not paused
    if not paused:
        assert job['preparedRemoved']
        with pytest.raises(ValueError, match='Review the project'):
            service.resume(identifier)


def test_startup_finishes_prepared_upload_cleanup_intent(tmp_path, monkeypatch):
    import uuid
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / (str(uuid.uuid4()) + '.licht')
    path.write_bytes(b'prepared upload')
    monkeypatch.setattr(Client, 'upload', lambda *_args, **_kwargs:
        (_ for _ in ()).throw(GalleryTransferCanceled()), raising=False)
    service.queue_upload(path, {'title': 'Scene'}, 'project', owned_export=True)
    finish(service)
    job = service._bucket()['jobs'][0]
    job.update(status='error', requiresPreparation=True)
    service._save()
    assert path.exists()
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    assert not path.exists()
    records = [job for bucket in restarted._data['accounts'].values() for job in bucket['jobs']]
    assert records[0]['preparedRemoved']


@pytest.mark.parametrize('kill_point', ['before_save', 'after_save'])
def test_handoff_sigkill_keeps_one_durable_link(tmp_path, monkeypatch, kill_point):
    import os
    import signal
    if not hasattr(os, 'fork'):
        pytest.skip('process kill fixture requires fork')
    service = connected(tmp_path, monkeypatch)
    path, metadata = handoff_upload(service, tmp_path)
    monkeypatch.setattr(Client, 'upload', lambda *_args, **_kwargs:
        {'scene': dict(id='scene', title='Title', contentRevision='new-c', metadataRevision='m')}, raising=False)
    child = os.fork()
    if child == 0:
        try:
            signal.alarm(15)
            metadata['_handoff'] = service.remember_replacement(metadata['_handoff'])
            save = service._save
            def kill_at_commit():
                if service._bucket()['jobs'] and service._bucket()['jobs'][0]['status'] == 'completed':
                    if kill_point == 'after_save':
                        save()
                    os.kill(os.getpid(), signal.SIGKILL)
                save()
            service._save = kill_at_commit
            service.queue_upload(path, metadata, 'new')
            finish(service)
        finally:
            os._exit(2)
    _, status = os.waitpid(child, 0)
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh(); finish(restarted)
    if kill_point == 'before_save':
        assert set(restarted.snapshot()['links']) == {'old'}
        job = restarted.snapshot()['jobs'][0]
        restarted.resume(job['id']); finish(restarted)
    assert set(restarted.snapshot()['links']) == {'new'}
    assert restarted.snapshot()['jobs'][0]['handoff']['state'] == 'completed'
    assert not restarted.snapshot().get('handoffIntents')


def test_server_private_visibility_is_preserved_but_not_shared():
    remote = dict(id="scene", title="Scene", description="", visibility="private",
        viewerSettings={}, contentRevision="c", metadataRevision="m")
    link = gallery_sync.exchange_link(remote, "saved")
    journal = dict(version=3, accounts={"owner": dict(links={"project": link}, jobs=[])})
    gallery_sync._validate_journal(journal)
    assert (link["sharedFields"], link["metadata"]) == (dict(title="Scene", description="", viewerSettings={}), remote)


@pytest.mark.parametrize("save_fails", [False, True])
def test_retrying_failed_settings_apply_keeps_backup_and_clears_old_failure(tmp_path, monkeypatch, save_fails):
    service = connected(tmp_path, monkeypatch)
    remote = dict(id="remote", title="Gallery title", contentRevision="c1", metadataRevision="m2")
    monkeypatch.setattr(Client, "scene", lambda *args: dict(remote))
    path = tmp_path / "master.licht"
    path.write_bytes(b"original project")
    service._bucket()["links"]["project"] = gallery_sync.exchange_link(remote, "before")
    service._save()
    first, _ = service.prepare_settings_update(remote, "project", str(path), gallery_sync.file_stamp(path))
    finish(service)
    recovery_path = Path(service._job(first)["localUpdate"]["backupPath"])
    service.fail_local_update(first, "Gallery changed; review again")
    finish(service)

    import copy
    import uuid
    unrelated = copy.deepcopy(service._job(first))
    unrelated.update(id=str(uuid.uuid4()), project="another-project", sceneId="another-scene")
    service._bucket()["jobs"].append(unrelated)
    retry, _ = service.prepare_settings_update(remote, "project", str(path), gallery_sync.file_stamp(path))
    finish(service)
    if save_fails:
        monkeypatch.setattr(service, "_save", lambda **kwargs: (_ for _ in ()).throw(OSError("journal write failed")))
    service.finish_settings_update(retry, "after", gallery_sync.file_stamp(path), {})
    finish(service)

    assert not unrelated.get("retired")
    assert recovery_path.read_bytes() == b"original project"
    assert bool(service._job(first).get("retired")) is not save_fails
    assert service._job(retry)["localUpdate"]["state"] == ("ready" if save_fails else "applied")
    assert service.snapshot()["links"]["project"]["commitUuid"] == ("before" if save_fails else "after")
    from lfs_plugins.gallery_controller import asset_sync_state
    facts = asset_sync_state(dict(id="project", path=str(path), commit_uuid="after"),
        service.snapshot()["links"]["project"], dict(remote, status="ready"), service.snapshot()["jobs"])
    assert (facts["action"] == "resolve") is save_fails
    if not save_fails:
        restarted = gallery_sync.GallerySync(service.account, tmp_path)
        records = [job for bucket in restarted._data["accounts"].values() for job in bucket["jobs"]]
        assert next(job for job in records if job["id"] == first)["retired"]
        assert recovery_path.read_bytes() == b"original project"


def test_local_only_resolution_keeps_unpublished_content_after_restart(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    remote = dict(id="remote", title="Gallery title", contentRevision="c1", metadataRevision="m2",
                  viewerSettings={}, description="", visibility="private")
    monkeypatch.setattr(Client, "scene", lambda *args: remote)
    path = tmp_path / "master.licht"
    path.write_bytes(b"locally edited geometry with checkpoint")
    service._bucket()["links"]["project"] = gallery_sync.exchange_link(dict(remote, metadataRevision="m1"), "published")
    service._save()
    job_id, _ = service.prepare_settings_update(remote, "project", str(path), gallery_sync.file_stamp(path))
    finish(service)
    service.finish_settings_update(job_id, "local-save", gallery_sync.file_stamp(path),
                                   gallery_sync.shared_fields(remote), preserve_local_content=True)
    finish(service)
    assert service._job(job_id)["localUpdate"]["state"] == "applied"
    restarted = gallery_sync.GallerySync(service.account, tmp_path)
    restarted.refresh()
    finish(restarted)
    link = restarted.snapshot()["links"]["project"]
    assert link["commitUuid"] == "published"
    assert link["metadataRevision"] == "m2"
    assert link["localFields"] == gallery_sync.shared_fields(remote)
    assert path.read_bytes() == b"locally edited geometry with checkpoint"


def test_gallery_content_keeps_chosen_local_settings_pending(tmp_path, monkeypatch):
    service = connected(tmp_path, monkeypatch)
    job = downloaded_job(service)
    path = tmp_path / "saved.licht"
    path.write_bytes(b"gallery geometry with local view")
    chosen = dict(title="My title", description="My notes", viewerSettings={"exposure": 2})
    service.link_download(job["id"], "project", "saved", project_path=path, local_fields=chosen)
    finish(service)
    assert job["linkOperation"]["state"] == "ready"
    link = service.snapshot()["links"]["project"]
    assert link["localFields"] == chosen
    assert link["sharedFields"] == gallery_sync.shared_fields(job["result"])
    assert link["localFields"] != link["sharedFields"]


@pytest.mark.parametrize("timeout_first", [False, True])
@pytest.mark.parametrize("restart_message", [False, True])
def test_download_preparation_tracks_progress_and_keep_waiting(tmp_path, monkeypatch, timeout_first, restart_message):
    from lfs_plugins.gallery_transfer_ui import transfer_phase
    from lfs_plugins.portal_gallery import GalleryProcessingTimeout
    import time

    service = connected(tmp_path, monkeypatch)
    attempts, deadlines = [], []
    def download(client, scene_id, path, **kwargs):
        attempts.append(scene_id)
        deadlines.append(client.processing_deadline)
        if client.processing_deadline is None:
            client.processing_deadline = time.time() + 60
        kwargs["on_processing"]({"stage": "preparing_download", "completed": 0, "total": 0})
        job = service.snapshot()["jobs"][0]
        assert job["status"] == "running" and not job["needsAttention"]
        assert transfer_phase(job) == "processing"
        saved = json.loads((service.root / "sync.json").read_text())
        saved_job = next(iter(saved["accounts"].values()))["jobs"][0]
        assert saved_job["processingDeadline"] == client.processing_deadline
        if timeout_first and len(attempts) == 1:
            raise GalleryProcessingTimeout("The viewing copy is being prepared. Keep waiting to check again.")
        if restart_message:
            kwargs["on_message"]("Restarting from zero")
        kwargs["on_progress"](8, 8)
        downloading = service.snapshot()["jobs"][0]
        assert transfer_phase(downloading) == "downloading"
        assert downloading["message"] == ("Restarting from zero" if restart_message else "Downloading")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"payload!")
        return _download_scene()
    monkeypatch.setattr(Client, "download", download, raising=False)
    service.download(_download_scene())
    finish(service)
    if timeout_first:
        waiting = service.snapshot()["jobs"][0]
        assert waiting["status"] == "paused" and waiting["needsAttention"]
        assert not Path(waiting["path"]).exists()
        service.resume(waiting["id"], keep_waiting=True)
        finish(service)
        assert deadlines[1] > waiting["processingDeadline"]
    done = service.snapshot()["jobs"][0]
    assert done["status"] == "completed", done
    assert not done["serverProcessing"] and not done["needsAttention"]
    assert transfer_phase(done) == "completed"
    assert len(attempts) == (2 if timeout_first else 1)
