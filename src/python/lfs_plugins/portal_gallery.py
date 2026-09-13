# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Resumable gallery transfers using the existing Studio account connection.

No bearer or signed URL is stored in a transfer checkpoint. Call from a worker;
progress and checkpoint callbacks let the panel publish state on the UI thread.
"""
from __future__ import annotations

import hashlib
import json
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
    def __init__(self, account, *, expected_session=None, revision_domains=None):
        self.account = account
        self.expected_session = expected_session
        self.revision_domains = revision_domains
        self.list_etag = None
        self.scene_tokens = {}

    def _request(self, method, path, body=None):
        kwargs = {"expected_session": self.expected_session} if self.expected_session is not None else {}
        result = self.account.request_json_authenticated(method, API + path, body, **kwargs)
        if path == "/me":
            version = result.get("revisionDomains", 0)
            self.revision_domains = version if type(version) is int else 0
        return result

    def _response(self, path, *, etag=None, max_bytes=4 * 1024 * 1024):
        return self.account.request_response_authenticated("GET", API + path,
            headers={"If-None-Match": etag} if etag else {}, max_bytes=max_bytes,
            expected_session=self.expected_session)

    @staticmethod
    def _etag(headers):
        return next((v for k, v in headers.items() if k.lower() == "etag"), None)

    def thumbnail(self, scene_id, *, etag=None, size=256):
        if size not in (256, 512):
            raise ValueError("Gallery thumbnail size must be 256 or 512")
        status, headers, data = self._response(f"/splats/{_identifier(scene_id)}/thumbnail?size={size}", etag=etag)
        tag = self._etag(headers)
        if status != 304 and (not data.startswith(b"\x89PNG\r\n\x1a\n") or not tag or tag.startswith("W/")):
            raise PortalProtocolError("Invalid gallery thumbnail")
        return status, tag, data

    def guards(self, scene_id, revision, domains):
        if self.revision_domains is None:
            self._request("GET", "/me")
        if self.revision_domains < 1:
            return {"baseRevision": revision.get("revision") if isinstance(revision, dict) else revision}
        cached = self.scene_tokens.get(scene_id, {})
        scene = revision if isinstance(revision, dict) else cached if cached.get("revision") == revision else {}
        if not all(scene.get(name + "Revision") for name in domains):
            # A legacy link has no domain baseline. Adopt tokens only for the exact
            # broad revision the user reviewed; otherwise require another review.
            current = self.scene(scene_id)
            legacy = revision.get("revision") if isinstance(revision, dict) else revision
            if current.get("revision") != legacy:
                raise PortalHTTPError(409, "sync_conflict")
            scene = current
        tokens = {name: scene.get(name + "Revision") for name in domains}
        if not all(isinstance(token, str) and token for token in tokens.values()):
            raise PortalProtocolError("Missing gallery revision tokens")
        return {"baseRevisions": tokens}

    def _guarded_request(self, method, path, body):
        try:
            return self._request(method, path, body)
        except PortalHTTPError as exc:
            retry = self._retry_guards(exc, body)
            if retry is None:
                raise
            # Deliberately outside the try: a second conflict is surfaced.
            body.update(retry)
            return self._request(method, path, body)

    def _retry_guards(self, exc, body):
        guards = body.get("baseRevisions")
        detail = exc.detail or {}
        changed, current = detail.get("changedDomains"), detail.get("currentRevisions")
        if (exc.status != 409 or not guards or not isinstance(changed, list) or not changed
                or any(name not in ("content", "metadata", "presentation") for name in changed)
                or "content" in changed or set(changed).intersection(guards)
                or not isinstance(current, dict)
                or not all(isinstance(current.get(name), str) and current[name] for name in guards)):
            return None
        return {**body, "baseRevisions": {name: current[name] for name in guards}}

    def list_scenes(self, etag=None):
        try:
            return self._list_pages(etag)
        except PortalHTTPError as exc:
            if exc.status != 400 or "gallery listing expired" not in exc.error.lower():
                raise
            return self._list_pages(None)

    def _list_pages(self, etag):
        result, cursor, seen = [], None, set()
        while True:
            query = "?" + urllib.parse.urlencode({"cursor": cursor}) if cursor else ""
            if callable(getattr(self.account, "request_response_authenticated", None)):
                status, headers, raw = self._response(f"/splats{query}", etag=etag if not cursor else None,
                                                       max_bytes=32 * 1024 * 1024)
                if not cursor:
                    self.list_etag = self._etag(headers)
                    if status == 304:
                        return None
                try:
                    payload = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    raise PortalProtocolError("Invalid gallery list") from None
            else:
                payload = self._request("GET", f"/splats{query}")
            if not isinstance(payload, dict):
                raise PortalProtocolError("Invalid gallery list")
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
        view = metadata.get("viewerSettings", {})
        domains = ("content", "metadata") if any(key in view for key in ("camera", "cameraPath", "environment")) else ("metadata",)
        return self._guarded_request("PATCH", f"/splats/{_identifier(scene_id)}",
                                     {**metadata, **self.guards(scene_id, revision, domains)})

    def delete(self, scene_id, revision):
        return self._guarded_request("DELETE", f"/splats/{_identifier(scene_id)}",
                                     self.guards(scene_id, revision, ("content", "metadata")))

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
            current = self.scene(scene_id)
            fields = ("contentRevision", "metadataRevision") if all(scene.get(k) and current.get(k) for k in ("contentRevision", "metadataRevision")) else ("revision",)
            if any(current[k] != scene[k] for k in fields):
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
            raise PortalHTTPError(409, "sync_conflict", detail=(upload.get("conflict") or {}).get("detail"))
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
        if size <= 0 or path.suffix.lower() not in (".ply", ".sog", ".ssog", ".spz", ".lfsg", ".licht"):
            raise ValueError("Choose a nonempty PLY, SOG, SSOG, SPZ or .licht export")
        fingerprint = _fingerprint(path, cancel)
        capabilities = self._request("GET", "/me")
        if capabilities.get("gallerySyncVersion") != 1:
            raise PortalProtocolError("This portal needs an update before Studio gallery sync is available.")
        if path.suffix.lower() in (".lfsg", ".licht") and path.suffix.lower()[1:] not in capabilities.get("sourceFormats", []):
            raise PortalProtocolError("This portal needs an update before it can accept this .licht upload.")
        if metadata.get("viewerSettings", {}).get("environment") and not capabilities.get("hdrBackgrounds"):
            raise PortalProtocolError("This portal needs an update before it can display HDR backgrounds.")
        identity = capabilities["id"]
        metadata = dict(metadata)
        if metadata.get("replaceSceneId"):
            revision = metadata.pop("baseRevision", None)
            supplied = metadata.pop("baseRevisions", None)
            if self.revision_domains >= 1 and supplied:
                metadata["baseRevisions"] = supplied
            else:
                metadata.update(self.guards(metadata["replaceSceneId"], revision, ("content", "metadata")))
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
        create = {**request, "idempotencyKey": checkpoint["idempotencyKey"]}
        upload = self._guarded_request("POST", "/splats/uploads", create)
        checkpoint["request"] = {key: value for key, value in create.items() if key != "idempotencyKey"}
        upload_id = _identifier(upload["id"])
        checkpoint["uploadId"] = upload_id
        on_checkpoint(dict(checkpoint))
        if upload.get("status") == "completed":
            on_progress(size, size)
            return upload
        if checkpoint.get("rebase"):
            rebase = dict(checkpoint["rebase"])
            revision, supplied = rebase.pop("baseRevision", None), rebase.pop("baseRevisions", None)
            rebase.update({"baseRevisions": supplied} if self.revision_domains >= 1 and supplied else
                          self.guards(metadata["replaceSceneId"], revision, ("content", "metadata")))
            upload = self._guarded_request("POST", f"/splats/uploads/{upload_id}/rebase", rebase)
            checkpoint["rebase"] = rebase
            on_checkpoint(dict(checkpoint))
            if upload.get("status") == "completed":
                on_progress(size, size)
                return upload
        if upload.get("status") == "failed" and upload.get("processing", {}).get("retryable"):
            upload = self._request("POST", f"/splats/uploads/{upload_id}/complete", {"parts": []})
        def finish(result):
            try:
                return self._await_processing(result, upload_id, size, cancel, on_processing)
            except PortalHTTPError as exc:
                guards = checkpoint.get("rebase", checkpoint["request"])
                retry = self._retry_guards(exc, guards)
                if retry is None:
                    raise
                details = {key: metadata[key] for key in ("title", "description", "visibility", "viewerSettings") if key in metadata}
                rebase = {"baseRevisions": retry["baseRevisions"], "metadata": guards.get("metadata", details)}
                checkpoint["rebase"] = rebase
                on_checkpoint(dict(checkpoint))
                rebased = self._request("POST", f"/splats/uploads/{upload_id}/rebase", rebase)
                if rebased.get("status") in ("created", "uploading"):
                    rebased = self._request("POST", f"/splats/uploads/{upload_id}/complete", {"parts": [
                        {"partNumber": part["partNumber"], "etag": part["etag"]} for part in rebased.get("uploadedParts", [])]})
                return self._await_processing(rebased, upload_id, size, cancel, on_processing)

        if upload.get("status") in ("processing", "conflict", "failed"):
            return finish(upload)
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
        try:
            result = self._request("POST", f"/splats/uploads/{upload_id}/complete", {"parts": [
                {"partNumber": n, "etag": parts[n]["etag"]} for n in range(1, part_count + 1)]})
        except PortalHTTPError as exc:
            if exc.status != 409:
                raise
            result = {"id": upload_id, "status": "conflict", "conflict": {"detail": exc.detail}}
        return finish(result)
