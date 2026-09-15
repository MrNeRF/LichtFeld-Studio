# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Transfer boundaries: redirects, local files, account changes and pagination."""
import io
import threading
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from lfs_plugins import http, portal_gallery
from lfs_plugins.portal_account import PortalHTTPError, PortalProtocolError

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

def _domain_client(version=1, changed=None):
    scene_id = str(uuid.uuid4())
    scene = {"id": scene_id, "revision": "reviewed", "contentRevision": "content-old", "metadataRevision": "metadata-old"}
    calls = []
    def request(method, path, body=None, **kwargs):
        import copy
        calls.append((method, path, copy.deepcopy(body)))
        if path.endswith("/me"):
            return {"storageHosts": ["portal.example"], "sourceFormats": ["licht"], "id": "owner", "gallerySyncVersion": 1, "revisionDomains": 1, **({"revisionDomains": version} if version else {})}
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

@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("operation,domains", [("metadata", {"metadata"}), ("camera", {"content", "metadata"}), ("delete", {"content", "metadata"})])
def test_capability_selects_exactly_one_guard_style(version, operation, domains):
    client, scene, calls = _domain_client(version)
    if operation == "delete":
        client.delete(scene["id"], scene)
    else:
        client.update(scene["id"], scene, **({"title": "Changed"} if operation == "metadata" else {"viewerSettings": {"cameraPath": None}}))
    body = calls[-1][2]
    assert set(body["baseRevisions"]) == domains
    assert "baseRevision" not in body

@pytest.mark.parametrize('changed', [['content'], ['metadata']])
@pytest.mark.parametrize('operation', ['update', 'delete'])
def test_domain_conflict_preserves_details_without_retry(changed, operation):
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
    assert client.list_scenes(etag='W/"old"') == [{"id": "a"}, {"id": "b"}]
    assert client.list_etag == 'W/"first"'
    assert client.list_scenes(etag=client.list_etag) is None
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

@pytest.mark.parametrize('changed', [['content'], ['metadata']])
@pytest.mark.parametrize('background', [False, True])
def test_upload_conflicts_preserve_details_without_automatic_rebase(tmp_path, changed, background):
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
        client.list_scenes(etag='W/"old"')
    assert len(calls) == 4
    assert calls[0][1] == {"If-None-Match": 'W/"old"'}
    assert all(not headers for _, headers in calls[1:])

def test_cached_scene_tokens_cannot_override_an_older_review():
    client, scene, calls = _domain_client()
    with pytest.raises(PortalProtocolError, match="Missing gallery revision tokens"):
        client.update(scene["id"], "older-review", title="Edited")
    assert all(call[0] == "GET" for call in calls)
