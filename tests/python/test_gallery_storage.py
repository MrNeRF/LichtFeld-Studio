# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Storage admission and feedback through the client, durable queue and UI."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

from test_asset_manager_panel import panel_module

from lfs_plugins.gallery_storage import GalleryStorageError, STORAGE_SENTENCE, gallery_quota, storage_requirement
from lfs_plugins.portal_account import PortalHTTPError
from lfs_plugins.portal_gallery import PortalGalleryClient


@pytest.fixture
def translations(monkeypatch):
    strings = json.loads((Path(__file__).parents[2] / "src/visualizer/gui/resources/locales/en.json").read_text())
    strings.update({"projects." + key: value for key, value in strings["projects"].items()})
    monkeypatch.setitem(sys.modules, "lichtfeld", SimpleNamespace(ui=SimpleNamespace(tr=lambda key: strings.get(key, key))))
    return strings


def test_storage_sentence_and_unknown_portal_advice(translations):
    from lfs_plugins.gallery_sync import friendly_error
    from lfs_plugins.gallery_messages import localize_message
    message = localize_message(friendly_error(PortalHTTPError(400, STORAGE_SENTENCE)))
    assert message == translations["projects.gallery.error.gallery_storage"]
    assert "Canceled" not in message
    for sentence in ("Please cancel another operation first.", "Invalid format for this new portal feature.",
                     "This request failed. Cancel pending work and contact support."):
        assert localize_message(friendly_error(PortalHTTPError(400, sentence))) == sentence
    assert localize_message("Discarded") == translations["projects.gallery.info.canceled"]
    assert localize_message("Update canceled. Your local splats remain.") == translations["projects.gallery.info.canceled"]
    localized = translations["projects.gallery.error.gallery_storage_size"].format(needed="2.1 MB", free="1.9 MB", quota="2.0 MB")
    assert localize_message(localized) == localized


@pytest.mark.parametrize("source,replaced,needed", [("ssog", 0, 285_213_248),
    ("licht", 0, 281_213_248), ("sog", 100_000_000, 185_213_248), ("spz", 300_000_000, 4_000_000)])
def test_admission_math(source, replaced, needed):
    facts = dict(quotaBytes=300_000_000, usedBytes=19_600_000, reservedBytes=4_000_000, wrapOverheadBytes=4_000_000)
    assert gallery_quota(facts) == (300_000_000, 23_600_000, 276_400_000)
    assert storage_requirement(281_213_248, facts, source, replaced) == needed


@pytest.mark.parametrize("size,expected", [(999, "999 B"), (1000, "1.0 KB"), (1_000_000, "1.0 MB"),
    (1_000_000_000, "1.0 GB"), (281_213_248, "281 MB"), (None, "0.0 B"), (-1, "0.0 B")])
def test_decimal_sizes(translations, size, expected):
    from lfs_plugins.asset_format import format_size
    assert format_size(size) == expected


@pytest.mark.parametrize("replaced,reserved,quota,blocked", [(0, 20, 100, True), (40, 20, 100, False), (0, 0, 110, False)])
def test_preflight_before_create(tmp_path, replaced, reserved, quota, blocked):
    export = tmp_path / "scene.licht"
    export.write_bytes(b"x" * 70)
    calls = []
    scene_id = str(uuid.uuid4())

    def request(method, path, body=None):
        calls.append((method, path))
        if path.endswith("/me"):
            return dict(id="owner", gallerySyncVersion=1, revisionDomains=1, sourceFormats=["licht"], quotaBytes=quota,
                        usedBytes=40, reservedBytes=reserved, wrapOverheadBytes=4_000_000)
        if method == "GET":
            return dict(contentLength=replaced)
        return dict(id=str(uuid.uuid4()), status="completed", scene=dict(id=scene_id))

    client = PortalGalleryClient(SimpleNamespace(base_url="https://portal.example", request_json_authenticated=request))
    metadata = dict(title="Scene")
    if replaced:
        metadata.update(replaceSceneId=scene_id, baseRevisions=dict(content="c", metadata="m"))
    if blocked:
        with pytest.raises(GalleryStorageError):
            client.upload(export, metadata)
        assert all(method != "POST" for method, _ in calls)
    else:
        assert client.upload(export, metadata)["status"] == "completed"
        assert any(method == "POST" for method, _ in calls)


def test_server_rejection_refreshes_numbers(tmp_path, translations):
    from lfs_plugins.gallery_messages import localize_message
    from lfs_plugins.gallery_sync import friendly_error
    export = tmp_path / "scene.licht"
    export.write_bytes(b"x" * 70)
    reads = 0

    def request(method, path, body=None):
        nonlocal reads
        if method == "GET":
            reads += 1
            return dict(id="owner", gallerySyncVersion=1, revisionDomains=1, sourceFormats=["licht"], quotaBytes=100 if reads == 1 else 60,
                        usedBytes=20, reservedBytes=0)
        raise PortalHTTPError(400, STORAGE_SENTENCE)

    client = PortalGalleryClient(SimpleNamespace(base_url="https://portal.example", request_json_authenticated=request))
    with pytest.raises(GalleryStorageError) as failure:
        client.upload(export, dict(title="Scene"))
    assert reads == 2
    message = localize_message(friendly_error(failure.value))
    assert "70 B" in message and "40 B" in message and "60 B" in message


def test_publication_rejection_is_a_storage_error(tmp_path):
    export = tmp_path / "scene.licht"
    export.write_bytes(b"x" * 70)

    def request(method, path, body=None):
        if path.endswith("/me"):
            return dict(id="owner", gallerySyncVersion=1, revisionDomains=1, sourceFormats=["licht"],
                        quotaBytes=100, usedBytes=0, reservedBytes=0)
        return dict(id=str(uuid.uuid4()), status="failed", scene=dict(id="scene"),
                    processing=dict(stage="failed", message=STORAGE_SENTENCE, retryable=True))

    client = PortalGalleryClient(SimpleNamespace(base_url="https://portal.example", request_json_authenticated=request))
    with pytest.raises(GalleryStorageError):
        client.upload(export, dict(title="Scene"))


def test_lost_create_response_replays_key_without_double_counting_reservation(tmp_path):
    export = tmp_path / "scene.licht"
    export.write_bytes(b"x" * 70)
    checkpoints, creates = [], []

    def request(method, path, body=None):
        if method == "GET":
            return dict(id="owner", gallerySyncVersion=1, revisionDomains=1, sourceFormats=["licht"],
                        quotaBytes=100, usedBytes=0, reservedBytes=70 if creates else 0)
        creates.append(body["idempotencyKey"])
        if len(creates) == 1:
            raise ConnectionError("response lost after creation")
        return dict(id=str(uuid.uuid4()), status="completed", scene=dict(id="scene"))

    client = PortalGalleryClient(SimpleNamespace(base_url="https://portal.example", request_json_authenticated=request))
    with pytest.raises(ConnectionError):
        client.upload(export, dict(title="Scene"), on_checkpoint=checkpoints.append)
    result = client.upload(export, dict(title="Scene"), checkpoint=checkpoints[-1])
    assert result["status"] == "completed" and creates[0] == creates[1]


@pytest.mark.parametrize("server", [False, True])
def test_storage_failure_keeps_prepared_copy_across_restart(tmp_path, monkeypatch, server):
    from test_gallery_sync import connected, finish, Client
    service = connected(tmp_path, monkeypatch)
    path = tmp_path / (str(uuid.uuid4()) + ".licht")
    path.write_bytes(b"prepared upload")

    def fail(*args, **kwargs):
        raise PortalHTTPError(400, STORAGE_SENTENCE) if server else GalleryStorageError(100, 50, 300)

    monkeypatch.setattr(Client, "upload", fail, raising=False)
    identifier = service.queue_upload(path, dict(title="Scene"), "project", owned_export=True)
    finish(service)
    job = service.snapshot()["jobs"][0]
    assert job["status"] == "error" and not job.get("requiresPreparation") and path.exists()
    restarted = type(service)(service.account, tmp_path)
    assert path.exists()
    restarted.refresh()
    finish(restarted)
    monkeypatch.setattr(Client, "upload", lambda *a, **k: dict(scene=dict(id="scene", contentRevision="c", metadataRevision="m")), raising=False)
    restarted.resume(identifier)
    finish(restarted)
    assert restarted.snapshot()["jobs"][0]["status"] == "completed"


def test_reservation_row_and_review(translations):
    from lfs_plugins.gallery_transfer_ui import transfer_rows
    from lfs_plugins.gallery_actions import gallery_eligibility, prepared_upload_size
    job = dict(id="job", kind="upload", status="error", total=281_213_248, completed=0,
               checkpoint=dict(uploadId="upload", reservedBytes=181_213_248), message="Failure",
               packaged=True, project="project", commitUuid="commit", uploadFormat="ssog")
    facts = dict(jobs=[job], quotaBytes=300_000_000, usedBytes=19_600_000, reservedBytes=4_000_000, upload_format="ssog")
    row = transfer_rows(facts)[0]
    assert "181 MB" in row["detail"] and "until canceled" in row["detail"]
    entry = dict(id="project", commit_uuid="commit")
    assert prepared_upload_size(entry, facts) == 281_213_248
    eligibility = gallery_eligibility(entry, facts)
    assert eligibility["status"] == "blocked"
    assert "281 MB" in eligibility["reason"] and "276 MB" in eligibility["reason"]
    job["status"] = "canceled"
    assert "reserves" not in transfer_rows(facts)[0]["detail"]


def test_review_uses_prepared_bytes_and_explains_ssog(panel_module, monkeypatch):
    from lfs_plugins.gallery_file_panel import GalleryFilePanel
    strings = json.loads((Path(__file__).parents[2] / "src/visualizer/gui/resources/locales/en.json").read_text())
    strings.update({"projects." + key: value for key, value in strings["projects"].items()})
    monkeypatch.setattr(panel_module.lf.ui, "tr", lambda key: strings.get(key, key))
    panel = GalleryFilePanel()
    panel._fields = dict(upload_format="ssog")
    panel._review = dict(mode="publish", action="publish", asset=dict(publication=dict(
        checked=True, preparedBytes=281_213_248, estimatedBytes=48_000_000)))
    panel._state = dict(quotaBytes=300_000_000, usedBytes=19_600_000, reservedBytes=4_000_000)
    assert panel._upload_size_label() == "Upload size: 281 MB"
    assert "every LOD level" in panel._format_hint()
    assert "276 MB" in panel._eligibility_reason()


def test_storage_strings_exist_in_every_locale():
    import string
    locales = Path(__file__).parents[2] / "src/visualizer/gui/resources/locales"
    keys = ("error.gallery_storage", "error.gallery_storage_size", "info.reserves", "info.reserves_unknown",
            "info.storage_retry", "review.upload_size", "review.ssog_levels", "quota.used")
    reference = json.loads((locales / "en.json").read_text())
    placeholders = lambda text: {name for _, name, _, _ in string.Formatter().parse(text) if name}
    assert len(list(locales.glob("*.json"))) == 10
    for path in locales.glob("*.json"):
        values = json.loads(path.read_text())
        for suffix in keys:
            key = "projects.gallery." + suffix
            assert values[key] and placeholders(values[key]) == placeholders(reference[key]), (path, key)
