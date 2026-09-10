# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Resumable gallery transfers using the existing Studio account connection.

No bearer or signed URL is stored in a transfer checkpoint. Call from a worker;
progress and checkpoint callbacks let the panel publish state on the UI thread.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from .http import urlopen
from .portal_account import PortalHTTPError, PortalProtocolError

API = "/api/gallery/v1"


class GalleryTransferCanceled(RuntimeError):
    pass


class GalleryProcessingPaused(GalleryTransferCanceled):
    pass


def _identifier(value):
    return str(uuid.UUID(str(value)))


def _fingerprint(path, cancel=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if cancel is not None and cancel.is_set():
                raise GalleryTransferCanceled("Upload paused")
            digest.update(chunk)
    return digest.hexdigest()


class PortalGalleryClient:
    def __init__(self, account, *, expected_session=None):
        self.account = account
        self.expected_session = expected_session

    def _request(self, method, path, body=None):
        kwargs = {"expected_session": self.expected_session} if self.expected_session is not None else {}
        return self.account.request_json_authenticated(method, API + path, body, **kwargs)

    def list_scenes(self):
        result, cursor, seen = [], None, set()
        while True:
            query = "?" + urllib.parse.urlencode({"cursor": cursor}) if cursor else ""
            payload = self._request("GET", f"/splats{query}")
            scenes = payload.get("scenes")
            if not isinstance(scenes, list):
                raise PortalProtocolError("Invalid gallery list")
            result.extend(scenes)
            cursor = payload.get("nextCursor")
            if cursor is None:
                break
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise PortalProtocolError("Invalid gallery pagination")
            seen.add(cursor)
        return result

    def scene(self, scene_id):
        return self._request("GET", f"/splats/{_identifier(scene_id)}")

    def update(self, scene_id, revision, **metadata):
        return self._request("PATCH", f"/splats/{_identifier(scene_id)}", {**metadata, "baseRevision": revision})

    def delete(self, scene_id, revision):
        return self._request("DELETE", f"/splats/{_identifier(scene_id)}", {"baseRevision": revision})

    def cancel_upload(self, upload_id):
        return self._request("POST", f"/splats/uploads/{_identifier(upload_id)}/cancel", {})

    def _storage_url(self, url):
        parsed = urllib.parse.urlsplit(url)
        local = self.account.base_url.startswith("http://127.0.0.1:") and url.startswith(self.account.base_url + "/")
        if (parsed.scheme != "https" and not local) or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise PortalProtocolError("Invalid gallery storage URL")
        return url

    def download(self, scene_id, destination, *, on_progress=lambda completed, total: None, cancel=None):
        cancel = cancel or threading.Event()
        payload = self._request("GET", f"/splats/{_identifier(scene_id)}/download")
        scene = payload["scene"]
        total = scene["contentLength"]
        if type(total) is not int or total <= 0:
            raise PortalProtocolError("Invalid gallery download size")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".gallery-", delete=False) as output:
                temporary = Path(output.name)
                with urlopen(self._storage_url(payload["url"]), timeout=120, no_redirect=True) as response:
                    completed = 0
                    while True:
                        if cancel.is_set():
                            raise GalleryTransferCanceled("Download canceled")
                        chunk = response.read(min(1024 * 1024, total - completed + 1))
                        if not chunk:
                            break
                        completed += len(chunk)
                        if completed > total:
                            raise PortalProtocolError("Gallery download exceeds its declared size")
                        output.write(chunk)
                        on_progress(completed, total)
                    if completed != total:
                        raise PortalProtocolError("Gallery download was incomplete")
                output.flush()
                os.fsync(output.fileno())
            if self.scene(scene_id)["revision"] != scene["revision"]:
                raise ValueError("The gallery scene changed while downloading. Sync again.")
            os.replace(temporary, destination)
            return scene
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)

    def _await_processing(self, upload, upload_id, size, cancel, on_processing):
        while upload.get("status") == "processing":
            if _identifier(upload.get("id")) != upload_id:
                raise PortalProtocolError("The portal returned a different upload.")
            state = upload.get("processing")
            if (not isinstance(state, dict) or state.get("stage") not in ("queued", "assembling", "validating", "publishing")
                    or type(state.get("totalBytes")) is not int or state["totalBytes"] != size
                    or type(state.get("bytesProcessed")) is not int or not 0 <= state["bytesProcessed"] <= size):
                raise PortalProtocolError("Invalid portal processing status")
            on_processing({"stage": state["stage"], "completed": state["bytesProcessed"], "total": size})
            if cancel.wait(1):
                raise GalleryProcessingPaused("Stopped waiting for portal processing")
            upload = self._request("GET", f"/splats/uploads/{upload_id}")
            if _identifier(upload.get("id")) != upload_id:
                raise PortalProtocolError("The portal returned a different upload.")
        if upload.get("status") == "conflict":
            raise PortalHTTPError(409, "sync_conflict")
        if upload.get("status") == "failed":
            if upload.get("processing", {}).get("retryable"):
                raise ValueError("The portal could not finish checking this upload. Resume to retry.")
            raise ValueError("The uploaded scene is incomplete or invalid. Export it again and start a new upload.")
        if upload.get("status") in ("created", "uploading"):
            raise ValueError("Some upload parts need to be sent again. Resume the upload.")
        if upload.get("status") != "completed":
            raise PortalProtocolError("The portal did not complete this upload.")
        return upload

    def upload(self, path, metadata, *, checkpoint=None, on_checkpoint=lambda value: None,
               on_progress=lambda completed, total: None, on_processing=lambda state: None, cancel=None):
        """Upload an immutable export; a canceled worker can resume its checkpoint."""
        path = Path(path)
        cancel = cancel or threading.Event()

        def check_canceled():
            if cancel.is_set():
                raise GalleryTransferCanceled("Upload paused")

        check_canceled()
        size = path.stat().st_size
        if size <= 0 or path.suffix.lower() not in (".ply", ".sog", ".ssog", ".lfsg"):
            raise ValueError("Choose a nonempty PLY, SOG, SSOG or Studio scene export")
        fingerprint = _fingerprint(path, cancel)
        capabilities = self._request("GET", "/me")
        if capabilities.get("gallerySyncVersion") != 1:
            raise PortalProtocolError("This portal needs an update before Studio gallery sync is available.")
        if path.suffix.lower() == ".lfsg" and "lfsg" not in capabilities.get("sourceFormats", []):
            raise PortalProtocolError("This portal does not support Studio scene uploads yet.")
        if metadata.get("viewerSettings", {}).get("environment") and not capabilities.get("hdrBackgrounds"):
            raise PortalProtocolError("This portal needs an update before it can display HDR backgrounds.")
        identity = capabilities["id"]
        request = {**metadata, "sourceFormat": path.suffix.lower()[1:], "contentLength": size}
        if checkpoint is None:
            checkpoint = {"origin": self.account.base_url, "owner": identity, "sha256": fingerprint,
                          "request": request, "idempotencyKey": str(uuid.uuid4())}
            on_checkpoint(dict(checkpoint))
        else:
            checkpoint = dict(checkpoint)
            if any(checkpoint.get(name) != value for name, value in {
                    "origin": self.account.base_url, "owner": identity, "sha256": fingerprint, "request": request}.items()):
                raise ValueError("The account, export or details changed. Start a new upload.")
        check_canceled()
        upload = self._request("POST", "/splats/uploads", {**request, "idempotencyKey": checkpoint["idempotencyKey"]})
        upload_id = _identifier(upload["id"])
        checkpoint["uploadId"] = upload_id
        on_checkpoint(dict(checkpoint))
        if upload.get("status") == "completed":
            on_progress(size, size)
            return upload
        if checkpoint.get("rebase"):
            upload = self._request("POST", f"/splats/uploads/{upload_id}/rebase", checkpoint["rebase"])
            if upload.get("status") == "completed":
                on_progress(size, size)
                return upload
        if upload.get("status") == "failed" and upload.get("processing", {}).get("retryable"):
            upload = self._request("POST", f"/splats/uploads/{upload_id}/complete", {"parts": []})
        if upload.get("status") in ("processing", "conflict", "failed"):
            return self._await_processing(upload, upload_id, size, cancel, on_processing)
        if upload.get("status") not in ("created", "uploading"):
            raise ValueError("This upload was canceled. Start a new upload.")
        part_size = upload.get("partSize")
        if type(part_size) is not int or not 1 <= part_size <= 64 * 1024 * 1024:
            raise PortalProtocolError("Invalid gallery upload part size")
        part_count = (size + part_size - 1) // part_size
        parts = {part["partNumber"]: part for part in upload.get("uploadedParts", [])}
        completed = 0
        with path.open("rb") as stream:
            for number in range(1, part_count + 1):
                check_canceled()
                length = min(part_size, size - (number - 1) * part_size)
                if number in parts and parts[number].get("size") == length:
                    completed += length
                    on_progress(completed, size)
                    continue
                signed = self._request("POST", f"/splats/uploads/{upload_id}/part-upload-urls", {"parts": [number]})
                urls = signed.get("urls", [])
                if len(urls) != 1 or urls[0].get("partNumber") != number:
                    raise PortalProtocolError("Invalid gallery upload URL response")
                url = self._storage_url(urls[0]["url"])
                stream.seek((number - 1) * part_size)
                data = stream.read(length)
                if len(data) != length:
                    raise ValueError("The export changed during upload")
                # Signed storage requests never receive account Authorization.
                req = urllib.request.Request(url, data=data, method="PUT", headers={"Content-Type": "application/octet-stream"})
                with urlopen(req, timeout=120, no_redirect=True) as response:
                    etag = response.headers.get("ETag")
                    if not etag or not 200 <= response.status < 300:
                        raise PortalProtocolError("Storage did not acknowledge the upload part")
                parts[number] = {"partNumber": number, "etag": etag, "size": length}
                completed += length
                on_progress(completed, size)
                on_checkpoint(dict(checkpoint))
        check_canceled()
        if path.stat().st_size != size or _fingerprint(path, cancel) != fingerprint:
            raise ValueError("The export changed during upload. Export again before retrying.")
        result = self._request("POST", f"/splats/uploads/{upload_id}/complete", {"parts": [
            {"partNumber": n, "etag": parts[n]["etag"]} for n in range(1, part_count + 1)]})
        return self._await_processing(result, upload_id, size, cancel, on_processing)
