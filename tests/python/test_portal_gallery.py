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
    export = tmp_path / "scene.ply"
    export.write_bytes(b"ply\ndata")
    requests, states = [], []
    waiting = {"id": identifier, "status": "processing", "processing": {
        "stage": "validating", "bytesProcessed": 4, "totalBytes": 8}}
    completed = {"id": identifier, "status": "completed", "scene": {"id": "scene", "revision": "new"}}
    def request(method, path, body=None):
        requests.append((method, path))
        if path.endswith("/me"):
            return {"id": "owner", "gallerySyncVersion": 1}
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
    export = tmp_path / "scene.ply"
    export.write_bytes(b"ply\ndata")
    cancel, checkpoints, requests = threading.Event(), [], []
    def request(method, path, body=None):
        requests.append((method, path))
        if path.endswith("/me"):
            return {"id": "owner", "gallerySyncVersion": 1}
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
    scene = {"id": identifier, "contentLength": 8, "revision": "original"}
    account = SimpleNamespace(base_url="https://portal.lichtfeld.io", request_json_authenticated=lambda *a: {
        "url": "https://storage.example/scene.ply", "scene": scene})
    client = portal_gallery.PortalGalleryClient(account)
    monkeypatch.setattr(portal_gallery, "urlopen", lambda *a, **kw: io.BytesIO(b"ply\nnew!"))
    destination = tmp_path / "local.ply"
    destination.write_bytes(b"keep me")
    cancel = threading.Event()
    with pytest.raises(portal_gallery.GalleryTransferCanceled):
        client.download(identifier, destination, cancel=cancel, on_progress=lambda *a: cancel.set())
    assert destination.read_bytes() == b"keep me"
    assert list(tmp_path.iterdir()) == [destination]


def test_download_rejects_scene_changed_before_publish(tmp_path, monkeypatch):
    identifier = str(uuid.uuid4())
    def request(method, path, body):
        if path.endswith("/download"):
            return {"url": "https://storage.example/data", "scene": {"id": identifier, "contentLength": 4, "revision": "old"}}
        return {"revision": "changed"}
    client = portal_gallery.PortalGalleryClient(SimpleNamespace(base_url="https://portal.lichtfeld.io", request_json_authenticated=request))
    monkeypatch.setattr(portal_gallery, "urlopen", lambda *a, **kw: io.BytesIO(b"data"))
    destination = tmp_path / "local.ply"
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
    export = tmp_path / "scene.ply"
    export.write_bytes(b"ply\nexport")
    with pytest.raises(PortalProtocolError, match="needs an update"):
        portal_gallery.PortalGalleryClient(account).upload(export, {"title": "Edited", "replaceSceneId": str(uuid.uuid4())})
    assert requests == [("GET", "/api/gallery/v1/me")]
