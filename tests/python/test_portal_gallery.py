# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Transfer boundaries: redirects, local files, account changes and pagination."""
import io
import hashlib
import os
from pathlib import Path
import threading
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from lfs_plugins import http, portal_gallery
from lfs_plugins.portal_account import PortalHTTPError, PortalProtocolError


@pytest.mark.parametrize("representation", [False, True], ids=["legacy", "representation"])
@pytest.mark.parametrize("swap", ["identity", "path", "appeared"])
def test_download_rechecks_destination_before_replacement(tmp_path, monkeypatch, representation, swap):
    from lichtfeld import io as native_io

    data = (Path(__file__).parents[1] / "data" / "portable-sog.licht").read_bytes()
    target = tmp_path / "项目-é.licht"
    target.write_bytes(data)
    replacement = tmp_path / "replacement.licht"
    native_io.restore_save(target, 1, replacement)
    replacement_bytes = replacement.read_bytes()
    if swap == "appeared":
        target.unlink()
    identifier = str(uuid.uuid4())
    scene = dict(id=identifier, contentRevision="c", metadataRevision="m", presentationRevision="p",
                 contentLength=len(data), sourceFormat="licht", viewerSettings={})
    choice = dict(representationId="pinned", size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                  format="licht", status="ready")

    def request(method, path, body=None, **kwargs):
        if path.endswith("download-options"):
            return {"representations": [choice]} if representation else {}
        if path.endswith("/download"):
            return {"url": "https://portal.lichtfeld.io/data", "scene": scene}
        return scene

    def response(method, path, **kwargs):
        start, end = map(int, kwargs["headers"]["Range"].removeprefix("bytes=").split("-"))
        return 206, {"Content-Range": f"bytes {start}-{end}/{len(data)}"}, data[start:end + 1]

    account = SimpleNamespace(base_url="https://portal.lichtfeld.io", request_json_authenticated=request,
                              request_response_authenticated=response)
    client = portal_gallery.PortalGalleryClient(account)
    client.max_file_bytes = len(data) * 2
    monkeypatch.setattr(portal_gallery, "urlopen", lambda *args, **kwargs: io.BytesIO(data))
    swapped = False

    def progress(*_args):
        nonlocal swapped
        if swapped:
            return
        swapped = True
        if swap == "path":
            target.unlink()
            target.symlink_to(replacement)
        else:
            os.replace(replacement, target)

    with pytest.raises(ValueError, match="identity or path changed"):
        client.download(identifier, target, on_progress=progress)
    assert swapped and target.read_bytes() == replacement_bytes

def test_upload_resumes_server_processing_without_sending_parts(tmp_path):
    identifier = str(uuid.uuid4())
    export = tmp_path / "scene.licht"
    export.write_bytes(b"ply\ndata")
    requests, states = [], []
    waiting = {"id": identifier, "status": "processing", "processing": {
        "stage": "validating", "bytesProcessed": 4, "totalBytes": 8}}
    completed = {"id": identifier, "status": "completed", "scene": {"contentRevision": "new", "metadataRevision": "new", "id": "scene", "revision": "new"}}
    def request(method, path, body=None):
        requests.append((method, path))
        if path.endswith("/me"):
            return {"storageHosts": ["portal.example"], "sourceFormats": ["licht"], "id": "owner", "gallerySyncVersion": 1, "revisionDomains": 1}
        if method == "POST" and path.endswith("/uploads"):
            return waiting
        assert method == "GET" and path.endswith(identifier)
        return completed
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url="https://portal.example", request_json_authenticated=request))
    cancel = SimpleNamespace(is_set=lambda: False, wait=lambda duration: False)
    assert client.upload(export, {"title": "Scene"}, cancel=cancel, on_processing=states.append) == completed
    assert states == [{"stage": "validating", "completed": 4, "total": 8}]
    assert len(requests) == 3

def test_stopping_processing_wait_preserves_checkpoint_and_does_not_cancel_server(tmp_path):
    identifier = str(uuid.uuid4())
    export = tmp_path / "scene.licht"
    export.write_bytes(b"ply\ndata")
    cancel, checkpoints, requests = threading.Event(), [], []
    def request(method, path, body=None):
        requests.append((method, path))
        if path.endswith("/me"):
            return {"storageHosts": ["portal.example"], "sourceFormats": ["licht"], "id": "owner", "gallerySyncVersion": 1, "revisionDomains": 1}
        return {"id": identifier, "status": "processing", "processing": {
            "stage": "queued", "bytesProcessed": 0, "totalBytes": 8}}
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url="https://portal.example", request_json_authenticated=request))
    with pytest.raises(portal_gallery.GalleryProcessingPaused):
        client.upload(export, {"title": "Scene"}, cancel=cancel, on_checkpoint=checkpoints.append,
                      on_processing=lambda state: cancel.set())
    assert checkpoints[-1]["uploadId"] == identifier
    assert len(requests) == 2 and export.exists()

@pytest.mark.parametrize("result", [
    {"status": "conflict"},
    {"status": "failed", "processing": {"retryable": False}},
    {"status": "processing", "processing": {"stage": "validating", "bytesProcessed": 9, "totalBytes": 8}},
])
def test_processing_conflicts_invalid_files_and_invalid_progress_are_not_success(result):
    identifier = str(uuid.uuid4())
    client = portal_gallery.PortalGalleryClient(SimpleNamespace())
    exception = PortalHTTPError if result["status"] == "conflict" else ValueError if result["status"] == "failed" else PortalProtocolError
    with pytest.raises(exception):
        client._await_processing({"id": identifier, **result}, identifier, 8, threading.Event(), lambda state: None)

def test_processing_poll_rejects_a_different_upload():
    identifier = str(uuid.uuid4())
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(request_json_authenticated=lambda *a: {
        "id": str(uuid.uuid4()), "status": "completed"}))
    with pytest.raises(PortalProtocolError, match="different upload"):
        client._await_processing({"id": identifier, "status": "processing", "processing": {
            "stage": "validating", "bytesProcessed": 4, "totalBytes": 8}}, identifier, 8,
            SimpleNamespace(wait=lambda duration: False), lambda state: None)

def test_authenticated_redirect_is_not_followed():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            self.send_response(302)
            self.send_header("Location", "/unexpected")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/account",
                                         headers={"Authorization": "Bearer private-test-token"})
        with pytest.raises(urllib.error.HTTPError) as error:
            http.urlopen(request, timeout=2, no_redirect=True)
        assert error.value.code == 302
        assert requests == [("/account", "Bearer private-test-token")]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()

@pytest.mark.parametrize("url", ["http://evil.example/data", "file:///tmp/file", "https://user:password@example.com/data", "https://example.com/data#fragment"])
def test_storage_url_rejects_unsafe_transports(url):
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url="https://portal.lichtfeld.io"))
    with pytest.raises(PortalProtocolError):
        client._storage_url(url)

def test_download_cancellation_preserves_existing_file(tmp_path, monkeypatch):
    identifier = str(uuid.uuid4())
    scene = {"contentRevision": "original", "metadataRevision": "original", "id": identifier, "contentLength": 8, "revision": "original"}
    account = SimpleNamespace(base_url="https://portal.lichtfeld.io", request_json_authenticated=lambda *a: {
        "storageHosts": ["portal.lichtfeld.io"], "revisionDomains": 1, "url": "https://portal.lichtfeld.io/scene.licht", "scene": scene})
    client = portal_gallery.PortalGalleryClient(account)
    monkeypatch.setattr(portal_gallery, "urlopen", lambda *a, **kw: io.BytesIO(b"ply\nnew!"))
    destination = tmp_path / "local.licht"
    destination.write_bytes(b"keep me")
    cancel = threading.Event()
    with pytest.raises(portal_gallery.GalleryTransferCanceled):
        client.download(identifier, destination, cancel=cancel, on_progress=lambda *a: cancel.set())
    assert destination.read_bytes() == b"keep me"
    assert list(tmp_path.iterdir()) == [destination]

def test_download_rejects_scene_changed_before_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(portal_gallery, "validate_download", lambda *args: None)  # Isolate revision validation.
    identifier = str(uuid.uuid4())
    def request(method, path, body):
        if path.endswith("/me"):
            return {"storageHosts": ["portal.lichtfeld.io"], "revisionDomains": 1}
        if path.endswith("/download"):
            return {"storageHosts": ["portal.lichtfeld.io"], "revisionDomains": 1, "url": "https://portal.lichtfeld.io/data", "scene": {"contentRevision": "old", "metadataRevision": "old", "id": identifier, "contentLength": 4, "revision": "old"}}
        return {"contentRevision": "changed", "metadataRevision": "changed", "revision": "changed"}
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url="https://portal.lichtfeld.io", request_json_authenticated=request))
    monkeypatch.setattr(portal_gallery, "urlopen", lambda *a, **kw: io.BytesIO(b"data"))
    destination = tmp_path / "local.licht"
    destination.write_bytes(b"keep")
    with pytest.raises(ValueError, match="changed"):
        client.download(identifier, destination)
    assert destination.read_bytes() == b"keep"

def test_repeated_pagination_cursor_is_rejected():
    account = SimpleNamespace(request_json_authenticated=lambda *a: {"scenes": [], "nextCursor": "repeated"})
    with pytest.raises(PortalProtocolError, match="pagination"):
        portal_gallery.PortalGalleryClient(account).list_scenes()

def test_old_portal_cannot_silently_create_instead_of_replace(tmp_path):
    requests = []
    def request(method, path, body):
        requests.append((method, path))
        return {"id": "old-portal-owner"}
    account = SimpleNamespace(base_url="https://portal.lichtfeld.io", request_json_authenticated=request)
    export = tmp_path / "scene.licht"
    export.write_bytes(b"ply\nexport")
    with pytest.raises(PortalProtocolError, match="does not support gallery sync"):
        portal_gallery.PortalGalleryClient(account).upload(export, {"title": "Edited", "replaceSceneId": str(uuid.uuid4())})
    assert requests == [("GET", "/api/gallery/v1/me")]

def _domain_client(changed=None):
    scene_id = str(uuid.uuid4())
    scene = {"id": scene_id, "revision": "reviewed", "contentRevision": "content-old", "metadataRevision": "metadata-old"}
    calls = []
    def request(method, path, body=None, **kwargs):
        import copy
        calls.append((method, path, copy.deepcopy(body)))
        if path.endswith("/me"):
            return {"storageHosts": ["portal.example"], "sourceFormats": ["licht"], "id": "owner", "gallerySyncVersion": 1, "revisionDomains": 1}
        writes = sum(call[0] != "GET" for call in calls)
        if changed and writes == 1:
            raise PortalHTTPError(409, "sync_conflict", detail={"changedDomains": changed,
                "currentRevisions": {"content": "content-new", "metadata": "metadata-new"}})
        return scene
    def response(method, path, *, body=None, **kwargs):
        request(method, path, body, **kwargs)
        return 204, {}, b""
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url="https://portal.example",
        request_json_authenticated=request, request_response_authenticated=response))
    return client, scene, calls

@pytest.mark.parametrize("operation,domains", [("metadata", {"metadata"}), ("camera", {"content", "metadata"}), ("delete", {"content", "metadata"})])
def test_capability_selects_exactly_one_guard_style(operation, domains):
    client, scene, calls = _domain_client()
    if operation == "delete":
        client.delete(scene["id"], scene)
    else:
        client.update(scene["id"], scene, **({"title": "Changed"} if operation == "metadata" else {"viewerSettings": {"cameraPath": None}}))
    body = calls[-1][2]
    assert set(body["baseRevisions"]) == domains
    assert "baseRevision" not in body

@pytest.mark.parametrize('operation', ['update', 'delete'])
def test_domain_conflict_preserves_details_without_retry(operation):
    changed = ['content', 'metadata']
    client, scene, calls = _domain_client(changed=changed)
    with pytest.raises(PortalHTTPError) as error:
        if operation == 'update':
            client.update(scene['id'], scene, title='Changed')
        else:
            client.delete(scene['id'], scene)
    assert error.value.detail['changedDomains'] == changed
    assert sum(call[0] != 'GET' for call in calls) == 1

def test_upload_create_uses_reviewed_revision_guards(tmp_path):
    scene_id, upload_id = str(uuid.uuid4()), str(uuid.uuid4())
    calls = []
    def request(method, path, body=None):
        calls.append((method, path, body))
        if path.endswith('/me'):
            return {'sourceFormats': ['licht'], 'id': 'owner', 'revisionDomains': 1, 'gallerySyncVersion': 1}
        assert path.endswith('/uploads')
        return {'id': upload_id, 'status': 'completed', 'scene': {'id': scene_id}}
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url='https://portal.example', request_json_authenticated=request))
    path = tmp_path / 'scene.licht'
    path.write_bytes(b'transport fixture')
    client.upload(path, {'title': 'Title', 'replaceSceneId': scene_id, 'baseRevisions': {'content': 'c', 'metadata': 'm'}})
    assert len(calls) == 2
    assert calls[-1][2]['baseRevisions'] == {'content': 'c', 'metadata': 'm'}
    assert 'baseRevision' not in calls[-1][2]

def test_listing_validator_only_on_first_page_and_304():
    import json
    calls = []
    responses = iter([(200, {"ETag": 'W/"first"'}, json.dumps({"scenes": [{"id": "a"}], "nextCursor": "cursor"}).encode()),
                      (200, {}, b'{"scenes":[{"id":"b"}],"nextCursor":null}'),
                      (304, {"ETag": 'W/"first"'}, b"")])
    def response(method, path, **kwargs):
        calls.append((path, kwargs["headers"]))
        return next(responses)
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(request_response_authenticated=response))
    assert client.list_scenes(etag='W/"old"', owner_wide=True) == [{"id": "a"}, {"id": "b"}]
    assert client.list_etag == 'W/"first"'
    assert client.list_scenes(etag=client.list_etag, owner_wide=True) is None
    assert calls == [(portal_gallery.API + "/splats", {"If-None-Match": 'W/"old"'}),
                     (portal_gallery.API + "/splats?cursor=cursor", {}),
                     (portal_gallery.API + "/splats", {"If-None-Match": 'W/"first"'})]

def test_thumbnail_uses_shared_bearer_refresh_and_handles_304(tmp_path, monkeypatch):
    from lfs_plugins import portal_account
    service = portal_account.PortalAccountService(base_url="https://portal.example", credentials_path=tmp_path / "credentials.json")
    credentials = SimpleNamespace(email="owner", connected_since="session", access_token="expired")
    monkeypatch.setattr(service, "_current_credentials", lambda: credentials)
    monkeypatch.setattr(service, "_assert_active_origin", lambda credentials: None)
    refreshes, calls = [], []
    def refresh(token, **kwargs):
        refreshes.append(token)
        credentials.access_token = "fresh"
        return "ok"
    monkeypatch.setattr(service, "_refresh_tokens", refresh)
    def urlopen(request, **kwargs):
        calls.append(request)
        assert kwargs["no_redirect"] is True
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"error":"invalid_token"}'))
        if len(calls) == 3:
            raise urllib.error.HTTPError(request.full_url, 304, "Not Modified", {"ETag": '"poster"'}, io.BytesIO())
        response = io.BytesIO(b"\x89PNG\r\n\x1a\nposter")
        response.status, response.headers = 200, {"ETag": '"poster"'}
        return response
    monkeypatch.setattr(portal_account, "urlopen", urlopen)
    client = portal_gallery.PortalGalleryClient(service, expected_session=("owner", "session"))
    identifier = str(uuid.uuid4())
    assert client.thumbnail(identifier)[1] == '"poster"'
    assert client.thumbnail(identifier, etag='"poster"') == (304, '"poster"', b"")
    assert refreshes == ["expired"]
    assert [request.get_header("Authorization") for request in calls] == ["Bearer expired", "Bearer fresh", "Bearer fresh"]
    assert calls[-1].get_header("If-none-match") == '"poster"'
    assert calls[-1].full_url.endswith(identifier + "/thumbnail?size=256")
    credentials.connected_since = "other-session"
    with pytest.raises(PortalProtocolError, match="account changed"):
        client.thumbnail(identifier)
    assert len(calls) == 3

def test_bounded_authenticated_response_rejects_oversized_thumbnail(tmp_path, monkeypatch):
    from lfs_plugins import portal_account
    service = portal_account.PortalAccountService(base_url="https://portal.example", credentials_path=tmp_path / "credentials.json")
    response = io.BytesIO(b"12345")
    response.status, response.headers = 200, {}
    monkeypatch.setattr(portal_account, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises(PortalProtocolError, match="size limit"):
        service._request_json("GET", "/thumbnail", response_options={"max_bytes": 4})

@pytest.mark.parametrize('background', [False, True])
def test_upload_conflicts_preserve_details_without_automatic_rebase(tmp_path, background):
    changed = ['content', 'metadata']
    identifier, scene_id = str(uuid.uuid4()), str(uuid.uuid4())
    calls = []
    parts = [{'partNumber': 1, 'etag': 'part', 'size': 8}]
    detail = {'changedDomains': changed, 'currentRevisions': {'content': 'c2', 'metadata': 'm2'}}
    conflict = {'id': identifier, 'status': 'conflict', 'conflict': {'error': 'sync_conflict', 'detail': detail}}
    def request(method, path, body=None):
        calls.append(path)
        if path.endswith('/me'):
            return {'sourceFormats': ['licht'], 'id': 'owner', 'revisionDomains': 1, 'gallerySyncVersion': 1}
        if path.endswith('/uploads'):
            return conflict if background else {'id': identifier, 'status': 'uploading', 'partSize': 8, 'uploadedParts': parts}
        assert path.endswith('/complete')
        assert body['parts'] == [{'partNumber': 1, 'etag': 'part'}]
        raise PortalHTTPError(409, 'sync_conflict', detail=detail)
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url='https://portal.example', request_json_authenticated=request))
    export = tmp_path / 'scene.licht'
    export.write_bytes(b'ply\ndata')
    with pytest.raises(PortalHTTPError) as error:
        client.upload(export, {'title': 'Title', 'replaceSceneId': scene_id, 'baseRevisions': {'content': 'c1', 'metadata': 'm1'}})
    assert error.value.detail == detail
    assert len(calls) == (2 if background else 3)

def test_exposure_only_patch_uses_metadata_guard():
    client, scene, calls = _domain_client()
    client.update(scene["id"], scene, viewerSettings={"exposure": 2})
    assert calls[-1][2]["baseRevisions"] == {"metadata": "metadata-old"}

def test_expired_listing_restarts_unconditionally_once():
    import json
    calls = []
    def response(method, path, **kwargs):
        calls.append((path, kwargs["headers"]))
        if "cursor=" in path:
            raise PortalHTTPError(400, "The gallery listing expired. Sync again.")
        return 200, {"ETag": 'W/"new"'}, json.dumps({"scenes": [], "nextCursor": "expired"}).encode()
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(request_response_authenticated=response))
    with pytest.raises(PortalHTTPError):
        client.list_scenes(etag='W/"old"', owner_wide=True)
    assert len(calls) == 4
    assert calls[0][1] == {"If-None-Match": 'W/"old"'}
    assert all(not headers for _, headers in calls[1:])

def test_cached_scene_tokens_cannot_override_an_older_review():
    client, scene, calls = _domain_client()
    with pytest.raises(PortalProtocolError, match="Missing gallery revision tokens"):
        client.update(scene["id"], "older-review", title="Edited")
    assert all(call[0] == "GET" for call in calls)


def test_old_portal_check_sees_title_changed_on_scene_101():
    import json
    calls, title = [], ["Before"]
    def response(method, path, **kwargs):
        calls.append((path, kwargs["headers"]))
        assert not kwargs["headers"]
        page = ({"scenes": [{"id": str(i)} for i in range(100)], "nextCursor": "page2"}
                if "cursor=" not in path else {"scenes": [{"id": "101", "title": title[0]}], "nextCursor": None})
        return 200, {"ETag": '"unchanged-page-one"'}, json.dumps(page).encode()
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(request_response_authenticated=response))
    assert client.list_scenes()[-1]["title"] == "Before"
    title[0] = "After"
    assert client.list_scenes(etag=client.list_etag)[-1]["title"] == "After"
    assert len(calls) == 4


def test_change_feed_walk_pins_sequence_and_rejects_repeated_cursor():
    pages = iter([
        {"changeSequence": 12, "nextSince": 11, "changes": [dict(changeSequence=11, type="upsert", sceneId="s", scene={"id": "s"})]},
        {"changeSequence": 12, "nextSince": None, "changes": [dict(changeSequence=12, type="delete", sceneId="s", scene={"id": "s"})]},
    ])
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(request_json_authenticated=lambda *_: next(pages)))
    assert len(client.changes_since(10)) == 2
    assert client.change_sequence == 12
    client._request = lambda *_: {"changeSequence": 12, "changes": [], "nextSince": 10}
    with pytest.raises(PortalProtocolError, match="pagination"):
        client.changes_since(10)


def test_origin_lookup_and_share_links_are_owner_scoped():
    identifier, calls = str(uuid.uuid4()), []
    link = {"id": "link", "url": "https://portal.example/gallery/share/example/", "expiresAt": None}
    active, status = False, None
    def request(method, path, body=None):
        calls.append((method, path, body))
        if "originProjectUuid" in path:
            return {"scenes": [], "nextCursor": None, "changeSequence": 1}
        if method == "GET":
            if status is not None:
                raise PortalHTTPError(status, "unavailable")
            return {"links": [dict(link, active=True)] if active else []}
        return link
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url="https://portal.example", request_json_authenticated=request))
    client.list_scenes(origin_project_uuid=identifier)
    assert calls.pop()[1].endswith("originProjectUuid=" + identifier)
    path = "/api/gallery/v1/splats/" + identifier
    assert client.share_link_details(identifier) == link
    assert calls == [("GET", path + "/share-links", None), ("POST", path + "/share-links", {"expiresIn": "never"})]
    calls.clear()
    active = True
    assert client.share_link_details(identifier) == dict(link, active=True)
    assert calls == [("GET", path + "/share-links", None)]
    for status in (404, 405):
        calls.clear()
        assert client.share_link_details(identifier) == link
        assert calls == [("GET", path + "/share-links", None), ("POST", path + "/share-link", {})]
    calls.clear()
    status = 403
    with pytest.raises(PortalHTTPError):
        client.share_link_details(identifier)
    assert calls == [("GET", path + "/share-links", None)]


def test_pinned_download_uses_authenticated_ranges_and_checks_digest(tmp_path, monkeypatch):
    import hashlib
    data, identifier = b"viewing copy", str(uuid.uuid4())
    scene = dict(id=identifier, contentRevision="c", metadataRevision="m", presentationRevision="p")
    choice = dict(representationId="project-pinned", size=len(data), sha256=hashlib.sha256(data).hexdigest(), format="licht", status="ready")
    calls = []
    def request(method, path, body=None, **kwargs):
        return {"representations": [choice]} if path.endswith("download-options") else scene
    def response(method, path, **kwargs):
        calls.append((path, kwargs))
        assert kwargs["headers"]["Range"] == f"bytes=0-{len(data)-1}"
        return 206, {"Content-Range": f"bytes 0-{len(data)-1}/{len(data)}"}, data
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(request_json_authenticated=request, request_response_authenticated=response), expected_session=("test", "session"))
    client.max_file_bytes = 1024
    monkeypatch.setattr(portal_gallery, "validate_download", lambda *_: None)
    target = tmp_path / "new.licht"
    client.download(identifier, target)
    assert target.read_bytes() == data
    assert calls[0][1]["expected_session"] == ("test", "session")
    choice["sha256"] = "0" * 64
    with pytest.raises(portal_gallery.GalleryTransferInvalid, match="checksum"):
        client.download(identifier, tmp_path / "bad.licht")
    assert not (tmp_path / "bad.licht").exists()


def test_pinned_download_storage_redirect_never_forwards_account_token(tmp_path, monkeypatch):
    import hashlib
    data, identifier = b"viewing copy", str(uuid.uuid4())
    scene = dict(id=identifier, contentRevision="c", metadataRevision="m", presentationRevision="p")
    choice = dict(representationId="pinned", size=len(data), sha256=hashlib.sha256(data).hexdigest(), format="licht", status="ready")
    location = ["https://storage.example/viewing.licht?signature=test"]
    def request(method, path, body=None, **kwargs):
        return {"representations": [choice]} if path.endswith("download-options") else scene
    def response(method, path, **kwargs):
        assert kwargs["allow_redirect"] is True
        return 302, {"Location": location[0]}, b""
    def storage(request, **kwargs):
        assert request.get_header("Authorization") is None
        assert request.get_header("Range") == f"bytes=0-{len(data)-1}"
        assert kwargs["no_redirect"] is True
        result = io.BytesIO(data)
        result.status = 206
        result.headers = {"Content-Range": f"bytes 0-{len(data)-1}/{len(data)}"}
        return result
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url="https://portal.example", request_json_authenticated=request, request_response_authenticated=response))
    client.storage_hosts = ["storage.example"]
    client.max_file_bytes = 1024
    monkeypatch.setattr(portal_gallery, "urlopen", storage)
    monkeypatch.setattr(portal_gallery, "validate_download", lambda *_: None)
    target = tmp_path / "new.licht"
    client.download(identifier, target)
    assert target.read_bytes() == data
    location[0] = "https://untrusted.example/data"
    with pytest.raises(PortalProtocolError):
        client.download(identifier, tmp_path / "unsafe.licht")


@pytest.mark.parametrize('method,key,attempts', [('GET', False, 4), ('HEAD', False, 1),
    ('POST', False, 1), ('POST', True, 4)])
def test_account_requests_retry_only_when_idempotent(monkeypatch, method, key, attempts):
    from lfs_plugins import portal_account, portal_retry
    service = object.__new__(portal_account.PortalAccountService)
    service._current_credentials = lambda: None
    calls = []
    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise PortalHTTPError(503, 'busy')
    service._request_json_once = fail
    monkeypatch.setattr(portal_retry.time, 'sleep', lambda _delay: None)
    with pytest.raises(PortalHTTPError):
        service._request_json(method, '/uploads/id/complete', {'idempotencyKey': 'key'} if key else None, timeout=7)
    assert len(calls) == attempts
    assert all(kwargs['timeout'] == 7 for _args, kwargs in calls)


def test_expired_authorization_refresh_does_not_repeat_a_post(monkeypatch):
    from lfs_plugins.portal_account import PortalAccountService
    service = object.__new__(PortalAccountService)
    credentials = SimpleNamespace(access_token='old', connection_enabled=True)
    service._current_credentials = lambda: credentials
    service.snapshot = lambda: SimpleNamespace(disconnecting=False)
    calls = []
    def fail(*args, **_kwargs):
        calls.append(args[0])
        raise PortalHTTPError(401, 'invalid_token')
    service._request_with_bearer = fail
    refreshed = []
    service._refresh_tokens = lambda token, **_kwargs: refreshed.append(token) or 'ok'
    with pytest.raises(PortalHTTPError) as failure:
        service._authenticated_request('POST', '/uploads')
    assert failure.value.status == 401
    assert calls == ['POST'] and refreshed == ['old']


@pytest.mark.parametrize('status,key', [(401, 'authorization_expired'), (403, 'access'),
    (404, 'not_found'), (409, 'http_conflict'), (413, 'too_large'), (429, 'portal_busy'), (500, 'server')])
def test_http_status_reasons_remain_distinct_at_the_ui(monkeypatch, status, key):
    import json
    import sys
    from pathlib import Path
    from lfs_plugins.gallery_sync import friendly_error
    from lfs_plugins.gallery_messages import localize_message
    translations = json.loads((Path(__file__).parents[2] / 'src/visualizer/gui/resources/locales/en.json').read_text())
    monkeypatch.setitem(sys.modules, 'lichtfeld', SimpleNamespace(ui=SimpleNamespace(tr=lambda key: translations.get(key, key))))
    expected = translations['projects.gallery.error.' + key]
    assert localize_message(friendly_error(PortalHTTPError(status, 'reason'))) == expected
    assert localize_message(friendly_error(urllib.error.HTTPError('https://host', status, 'reason', {}, io.BytesIO()))) == expected


def test_gallery_notice_never_contains_a_python_traceback(monkeypatch):
    import sys
    from lfs_plugins.gallery_messages import localize_message
    monkeypatch.setitem(sys.modules, 'lichtfeld', SimpleNamespace(ui=SimpleNamespace(tr=lambda key: key)))
    text = 'Traceback (most recent call last):\n  File "/private/path.py", line 7\nRuntimeError: request marker'
    assert localize_message(text) == 'request marker'


@pytest.mark.parametrize("status", ["failed", "preparing", "ready", "unknown"])
def test_project_representation_failure_is_not_indefinite_processing(tmp_path, monkeypatch, status):
    identifier = str(uuid.uuid4())
    choice = dict(format="licht", status=status, representationId="pinned", size=4)
    requests = []
    def request(method, path, body=None):
        requests.append(path)
        assert path.endswith("/download-options")
        return {"representations": [choice]}
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url="https://portal.example", request_json_authenticated=request))
    client.max_file_bytes = 100
    client.processing_deadline = portal_gallery.time.time() - 1
    used = []
    monkeypatch.setattr(client, "_download_representation", lambda *args: used.append(args[1]) or {"id": identifier})
    if status == "ready":
        assert client.download(identifier, tmp_path / "project.licht") == {"id": identifier}
        assert used == [choice]
    else:
        with pytest.raises(PortalProtocolError if status == "unknown" else ValueError) as failure:
            client.download(identifier, tmp_path / "project.licht")
        assert isinstance(failure.value, portal_gallery.GalleryProcessingTimeout) is (status == "preparing")
        if status == "failed":
            assert "could not prepare" in str(failure.value)
        elif status == "unknown":
            assert isinstance(failure.value, PortalProtocolError)
        assert not used
    assert len(requests) == 1
    assert not (tmp_path / "project.licht").exists()


@pytest.mark.parametrize("outcome", ["ready", "failed", "cancel", "account_changed"])
def test_new_gallery_project_waits_for_preparation(tmp_path, monkeypatch, outcome):
    data = (Path(__file__).parents[1] / "data" / "portable-sog.licht").read_bytes()
    identifier = str(uuid.uuid4())
    scene = dict(id=identifier, contentRevision="content", metadataRevision="metadata",
                 presentationRevision="story", sourceFormat="licht", contentLength=len(data))
    calls, waits = [], []
    def request(method, path, body=None, **kwargs):
        assert kwargs["expected_session"] == ("owner", "session")
        if path.endswith("download-options"):
            calls.append(path)
            if len(calls) > 1 and outcome == "account_changed":
                raise RuntimeError("The account changed")
            status = "preparing" if len(calls) < 3 else outcome
            return {"sceneId": identifier, "representations": [dict(
                format="licht", status=status, representationId="project-pinned",
                size=len(data) if status == "ready" else None,
                sha256=hashlib.sha256(data).hexdigest() if status == "ready" else None)]}
        return scene
    def response(method, path, **kwargs):
        assert len(calls) == 3
        start, end = map(int, kwargs["headers"]["Range"].removeprefix("bytes=").split("-"))
        return 206, {"Content-Range": f"bytes {start}-{end}/{len(data)}"}, data[start:end + 1]
    cancel = SimpleNamespace(is_set=lambda: False,
                             wait=lambda seconds: waits.append(seconds) or outcome == "cancel")
    account = SimpleNamespace(base_url="https://portal.example", request_json_authenticated=request,
                              request_response_authenticated=response)
    client = portal_gallery.PortalGalleryClient(account, expected_session=("owner", "session"))
    client.max_file_bytes = len(data) * 2
    target = tmp_path / "local-copy.licht"
    if outcome == "ready":
        assert client.download(identifier, target, cancel=cancel) == scene
        assert target.read_bytes() == data
    else:
        error, message = {
            "failed": (ValueError, "could not prepare"),
            "cancel": (portal_gallery.GalleryTransferCanceled, None),
            "account_changed": (RuntimeError, "account changed"),
        }[outcome]
        with pytest.raises(error, match=message):
            client.download(identifier, target, cancel=cancel)
        assert not target.exists()
        assert not list(tmp_path.glob("*.part"))
    assert waits == ([1] if outcome in ("cancel", "account_changed") else [1, 1])


def test_project_preparation_stops_at_its_deadline(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(portal_gallery.time, "time", lambda: clock[0])
    monkeypatch.setattr(portal_gallery.time, "monotonic", lambda: clock[0])
    identifier = str(uuid.uuid4())
    calls, waits, states = [], [], []
    def request(method, path, body=None):
        calls.append(path)
        assert len(calls) <= 3, "Preparation must have a bounded wait"
        return {"representations": [{"format": "licht", "status": "preparing"}]}
    def wait(seconds):
        waits.append(seconds)
        clock[0] += 1
        return False
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(request_json_authenticated=request))
    client.max_file_bytes = 100
    client.processing_deadline = 1002.0
    with pytest.raises(portal_gallery.GalleryProcessingTimeout):
        client.download(identifier, tmp_path / "local.licht",
                        cancel=SimpleNamespace(is_set=lambda: False, wait=wait), on_processing=states.append)
    assert len(calls) == 3 and waits == [1, 1]
    assert client.processing_deadline == 1002.0
    assert all(state == {"stage": "preparing_download", "completed": 0, "total": 0} for state in states)
    assert not list(tmp_path.iterdir())
