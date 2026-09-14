# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Resumable gallery transfers using the existing Studio account connection.

No bearer or signed URL is stored in a transfer checkpoint. Call from a worker;
progress and checkpoint callbacks let the panel publish state on the UI thread.
"""
from __future__ import annotations

import hashlib
from http.client import IncompleteRead, RemoteDisconnected
import json
import logging
import os
import shutil
import time
import re
import urllib.error
import threading
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from .http import urlopen
from .portal_retry import retry_call
from .portal_account import _default_client_version
from .portal_account import PortalHTTPError, PortalProtocolError

API = "/api/gallery/v1"
UNSUPPORTED_PORTAL = "This portal version does not support gallery sync"


class GalleryTransferCanceled(RuntimeError):
    pass


class GalleryProcessingPaused(GalleryTransferCanceled):
    pass


class GalleryProcessingTimeout(ValueError):
    pass


PROCESSING_TIMEOUT = 15 * 60
DEFAULT_MAX_FILE_BYTES = 100 * 1024**3


class GalleryTransferInvalid(PortalProtocolError, ValueError):
    """A malformed transfer cannot be resumed against the same representation."""


def disk_preflight(allocations):
    """Sum staging, recovery and destination allocations per filesystem."""
    devices = {}
    for path, size in allocations:
        parent = Path(path).absolute().parent
        while not parent.exists():
            parent = parent.parent
        device = parent.stat().st_dev
        previous = devices.get(device, (parent, 0))
        devices[device] = (parent, previous[1] + size)
    for parent, required in devices.values():
        if shutil.disk_usage(parent).free < required:
            raise ValueError('Not enough disk space for the download, staging and recovery copy.')


def validate_download(path, extension, cancel=None):
    try:
        _validate_download(path, extension, cancel)
    except ValueError as exc:
        # Preserve the validator's exception type for callers that distinguish
        # container and archive failures, while giving the UI one damage reason.
        exc.args = (f"The downloaded file is damaged or was changed on the portal. {exc}",)
        raise


def _validate_download(path, extension, cancel=None):
    """Admit the native publishing subset and verify embedded bytes before use."""
    from .portable_project import ProjectFile

    class Sink:
        def write(self, value):
            if cancel is not None and cancel.is_set():
                raise GalleryTransferCanceled('Download paused')
            return len(value)

    if extension != '.licht':
        raise PortalProtocolError("Gallery downloads must be .licht files")
    with Path(path).open('rb') as stream:
        project = ProjectFile(stream)
        for index in range(len(project.manifest['nodes'])):
            project.copy_node(index, Sink())
        if 'environment' in project.manifest:
            project.copy_environment(Sink())


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


def domain_tokens(scene):
    keys = ("contentRevision", "metadataRevision")
    if not isinstance(scene, dict) or not all(isinstance(scene.get(key), str) and scene[key] for key in keys):
        raise PortalProtocolError("Missing gallery revision tokens")
    return {key: scene[key] for key in keys}


class PortalGalleryClient:
    def __init__(self, account, *, expected_session=None, revision_domains=None):
        self.account = account
        self.expected_session = expected_session
        self.revision_domains = revision_domains
        self.list_etag = None
        self.max_file_bytes = None
        self.storage_hosts = None
        self._logged_storage_hosts = set()
        self.processing_deadline = None
        self.user_agent = "LichtFeld-Studio/" + getattr(account, "_client_version", _default_client_version())

    def _request(self, method, path, body=None):
        kwargs = {"expected_session": self.expected_session} if self.expected_session is not None else {}
        if method == "DELETE":
            status, _, _ = self.account.request_response_authenticated(method, API + path, body=body, **kwargs)
            if status not in (200, 204):
                raise PortalProtocolError("The portal has not confirmed removal. Refresh the gallery and try again.")
            return {}
        result = self.account.request_json_authenticated(method, API + path, body, **kwargs)
        if path == "/me":
            version = result.get("revisionDomains", 0)
            self.revision_domains = version if type(version) is int else 0
            if self.revision_domains < 1:
                raise PortalProtocolError(UNSUPPORTED_PORTAL)
            self.max_file_bytes = result.get("maxFileBytes", DEFAULT_MAX_FILE_BYTES)
            self.storage_hosts = result.get("storageHosts")
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

    def guards(self, baseline, domains):
        if self.revision_domains is None:
            self._request("GET", "/me")
        if self.revision_domains < 1:
            raise PortalProtocolError(UNSUPPORTED_PORTAL)
        scene = domain_tokens(baseline)
        tokens = {name: scene.get(name + "Revision") for name in domains}
        if not all(isinstance(token, str) and token for token in tokens.values()):
            raise PortalProtocolError("Missing gallery revision tokens")
        return {"baseRevisions": tokens}

    def list_scenes(self, etag=None):
        try:
            return self._list_pages(etag)
        except PortalHTTPError as exc:
            if exc.status != 400 or "gallery listing expired" not in exc.error.lower():
                raise
            return self._list_pages(None)

    def share_link(self, scene_id):
        result = self._request("POST", f"/splats/{_identifier(scene_id)}/share-link", {})
        if not isinstance(result, dict) or not isinstance(result.get("url"), str):
            raise PortalProtocolError("Invalid gallery share link")
        from .portal_security import portal_url
        return portal_url(self.account.base_url, result["url"])

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

    def update(self, scene_id, baseline, **metadata):
        view = metadata.get("viewerSettings", {})
        domains = ("content", "metadata") if any(key in view for key in ("camera", "cameraPath", "environment")) else ("metadata",)
        return self._request("PATCH", f"/splats/{_identifier(scene_id)}",
                                     {**metadata, **self.guards(baseline, domains)})

    def delete(self, scene_id, baseline):
        return self._request("DELETE", f"/splats/{_identifier(scene_id)}",
                                     self.guards(baseline, ("content", "metadata")))

    def cancel_upload(self, upload_id):
        return self._request("POST", f"/splats/uploads/{_identifier(upload_id)}/cancel", {})

    def _storage_url(self, url):
        from .portal_security import storage_url
        try:
            result = storage_url(self.account.base_url, url, self.storage_hosts)
        except ValueError:
            raise PortalProtocolError("Unsafe portal URL") from None
        host = urllib.parse.urlsplit(result).netloc
        if host not in self._logged_storage_hosts:
            self._logged_storage_hosts.add(host)
            # Presigned paths/queries contain secrets; log only the authority.
            logging.getLogger(__name__).debug("Portal storage transfer host: %s", host)
        return result

    def download(self, scene_id, destination, *, on_progress=lambda completed, total: None, cancel=None,
                 checkpoint=None, on_checkpoint=lambda value: None, on_message=lambda message: None,
                 final_destination=None):
        cancel = cancel or threading.Event()
        if self.max_file_bytes is None:
            self._request("GET", "/me")
        payload = self._request("GET", f"/splats/{_identifier(scene_id)}/download")
        scene = payload["scene"]
        total = scene["contentLength"]
        if type(total) is not int or total <= 0:
            raise PortalProtocolError("Invalid gallery download size")
        if type(self.max_file_bytes) is not int or self.max_file_bytes <= 0 or total > self.max_file_bytes:
            raise PortalProtocolError("Gallery download exceeds the portal file-size limit")
        url = self._storage_url(payload["url"])
        destination = Path(destination)
        final = Path(final_destination) if final_destination else destination
        disk_preflight([(destination, total * 2), (final, total + (final.stat().st_size if final.exists() else 0))])
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        partial = destination.with_name('.' + destination.name + '.part')
        if partial.is_symlink() or destination.is_symlink():
            raise ValueError("Download destination was redirected")
        identity = {key: scene.get(key) for key in ('id', 'contentRevision', 'metadataRevision', 'contentLength')}
        saved = dict(checkpoint or {})
        etag = saved.get('representationId')
        resume = (isinstance(etag, str) and re.fullmatch(r'"[^"\r\n]+"', etag)
                  and saved.get('scene') == identity and partial.is_file() and 0 < partial.stat().st_size <= total)
        if not resume:
            if checkpoint or partial.exists():
                on_message('This download has no matching pinned representation. Restarting from zero.')
            partial.unlink(missing_ok=True)
            saved = {}
        offset = partial.stat().st_size if resume else 0
        keep_partial = bool(resume)
        try:
            def transfer():
                nonlocal offset, saved, keep_partial
                # A retry may reuse only bytes tied to the strong storage ETag.
                # A completed partial rechecks its final byte against the pinned ETag.
                offset = min(partial.stat().st_size, total - 1) if keep_partial and partial.exists() else 0
                headers = {'User-Agent': self.user_agent}
                tag = saved.get('representationId')
                if offset:
                    headers.update(Range=f'bytes={offset}-', **{'If-Range': tag})
                request = urllib.request.Request(url, headers=headers, method='GET')
                with urlopen(request, timeout=120, no_redirect=True) as response:
                    response_headers = getattr(response, 'headers', {})
                    status = getattr(response, 'status', 200)
                    current_tag = response_headers.get('ETag')
                    pinned = (isinstance(current_tag, str) and re.fullmatch(r'"[^"\r\n]+"', current_tag)
                              and response_headers.get('Accept-Ranges', '').lower() == 'bytes')
                    # Establish size consistency before opening staging or reading
                    # any body bytes. A later short stream is a lost connection.
                    content_range = response_headers.get('Content-Range', '')
                    range_match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                    content_length = response_headers.get('Content-Length')
                    if content_length is not None and not re.fullmatch(r'\d+', str(content_length)):
                        raise GalleryTransferInvalid('Invalid gallery download Content-Length')
                    if ((status == 206 and range_match and int(range_match[3]) != total)
                            or (content_length is not None and int(content_length) !=
                                (total - offset if status == 206 else total))):
                        raise GalleryTransferInvalid('Gallery download is incomplete or larger than the portal declared')
                    if offset and status == 200:
                        on_message('The portal restarted this download from zero because its representation changed.')
                        offset = 0
                    elif offset and (status != 206 or current_tag != tag or
                            response_headers.get('Content-Range') != f'bytes {offset}-{total-1}/{total}'):
                        keep_partial = False
                        raise PortalProtocolError('Invalid resumed gallery download representation')
                    elif not offset and status != 200:
                        raise PortalProtocolError('Unexpected gallery download response')
                    keep_partial = bool(pinned)
                    # Retain the attempted scene even without a pin so a later
                    # service retry can explain why it must restart from zero.
                    saved = {'scene': identity}
                    if pinned:
                        saved['representationId'] = current_tag
                    on_checkpoint(dict(saved))
                    digest = hashlib.sha256()
                    if offset:
                        with partial.open('rb') as existing:
                            remaining = offset
                            while remaining and (chunk := existing.read(min(1024 * 1024, remaining))):
                                digest.update(chunk)
                                remaining -= len(chunk)
                    completed = offset
                    if not offset:
                        if checkpoint or partial.exists():
                            on_progress(0, total)
                        if partial.exists():
                            on_message('The portal does not provide a matching pinned representation. Restarting from zero.')
                        partial.unlink(missing_ok=True)
                    flags = os.O_WRONLY | (os.O_APPEND if offset else os.O_CREAT | os.O_EXCL)
                    flags |= getattr(os, 'O_NOFOLLOW', 0)
                    fd = os.open(partial, flags, 0o600)
                    with os.fdopen(fd, 'ab' if offset else 'wb') as output:
                        if os.fstat(output.fileno()).st_nlink != 1:
                            raise ValueError('Partial download is linked to another file')
                        if offset:
                            os.ftruncate(output.fileno(), offset)
                        try:
                            while True:
                                if cancel.is_set():
                                    raise GalleryTransferCanceled('Download paused')
                                chunk = response.read(min(1024 * 1024, total - completed + 1))
                                if not chunk:
                                    break
                                completed += len(chunk)
                                if completed > total:
                                    raise GalleryTransferInvalid('Gallery download exceeds its declared size')
                                output.write(chunk)
                                digest.update(chunk)
                                on_progress(completed, total)
                            if completed != total:
                                raise ConnectionError('Gallery download connection closed before completion')
                        finally:
                            output.flush()
                            os.fsync(output.fileno())
                    return digest.hexdigest()
            # Only an explicit Resume may continue or restart a download.
            checksum = retry_call(transfer, idempotent=False)
            try:
                validate_download(partial, destination.suffix.lower(), cancel)
            except (ValueError, PortalProtocolError) as exc:
                raise GalleryTransferInvalid(str(exc)) from exc
            saved['sha256'] = checksum
            on_checkpoint(dict(saved))
            try:
                current = self.scene(scene_id)
                current_tokens = domain_tokens(current)
            except (OSError, PortalHTTPError, PortalProtocolError) as exc:
                raise ConnectionError("The portal could not verify the download. Resume to check again.") from exc
            if (current_tokens['contentRevision'] != domain_tokens(scene)['contentRevision']
                    or current.get('viewerSettings') != scene.get('viewerSettings')):
                raise ValueError("The gallery scene changed while downloading. Sync again.")
            if current_tokens['metadataRevision'] != scene['metadataRevision']:
                scene = {**scene, **{key: current[key] for key in
                         ('title', 'description', 'visibility', 'metadataRevision') if key in current}}
            os.replace(partial, destination)
            return scene
        except (GalleryTransferCanceled, IncompleteRead, TimeoutError, ConnectionError, urllib.error.URLError) as exc:
            if not keep_partial:
                partial.unlink(missing_ok=True)
            if isinstance(getattr(exc, 'reason', exc), (IncompleteRead, RemoteDisconnected)):
                raise ConnectionError('Gallery download connection closed before completion') from exc
            raise
        except Exception:
            partial.unlink(missing_ok=True)
            raise

    def _await_processing(self, upload, upload_id, size, cancel, on_processing):
        deadline = self.processing_deadline or (time.time() + PROCESSING_TIMEOUT)
        explicit_deadline = self.processing_deadline is not None
        monotonic_deadline = time.monotonic() + max(0, min(PROCESSING_TIMEOUT, deadline - time.time()))
        while upload.get("status") == "processing":
            if _identifier(upload.get("id")) != upload_id:
                raise PortalProtocolError("The portal returned a different upload.")
            state = upload.get("processing")
            if (not isinstance(state, dict) or state.get("stage") not in ("queued", "assembling", "validating", "publishing")
                    or type(state.get("totalBytes")) is not int or state["totalBytes"] != size
                    or type(state.get("bytesProcessed")) is not int or not 0 <= state["bytesProcessed"] <= size):
                raise PortalProtocolError("Invalid portal processing status")
            started = state.get('startedAt')
            if started is not None and not explicit_deadline:
                try:
                    from datetime import datetime
                    started = float(started) if isinstance(started, (int, float)) else datetime.fromisoformat(started.replace('Z', '+00:00')).timestamp()
                    deadline = min(deadline, started + PROCESSING_TIMEOUT)
                    monotonic_deadline = min(monotonic_deadline, time.monotonic() + max(0, deadline - time.time()))
                except (ValueError, TypeError, OverflowError):
                    raise PortalProtocolError('Invalid portal processing start time') from None
            self.processing_deadline = deadline
            on_processing({"stage": state["stage"], "completed": state["bytesProcessed"], "total": size})
            if time.time() >= deadline or time.monotonic() >= monotonic_deadline:
                raise GalleryProcessingTimeout('Portal is taking longer than expected · Retry / Keep waiting')
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
        resuming = bool(checkpoint and checkpoint.get("uploadId"))

        def check_canceled():
            if cancel.is_set():
                raise GalleryTransferCanceled("Upload paused")

        check_canceled()
        size = path.stat().st_size
        if size <= 0 or path.suffix.lower() != ".licht":
            raise ValueError("Choose a nonempty .licht export")
        fingerprint = _fingerprint(path, cancel)
        capabilities = self._request("GET", "/me")
        if capabilities.get("gallerySyncVersion") != 1:
            raise PortalProtocolError(UNSUPPORTED_PORTAL)
        if "licht" not in capabilities.get("sourceFormats", []):
            raise PortalProtocolError(UNSUPPORTED_PORTAL)
        if metadata.get("viewerSettings", {}).get("environment") and not capabilities.get("hdrBackgrounds"):
            raise PortalProtocolError(UNSUPPORTED_PORTAL)
        identity = capabilities["id"]
        metadata = dict(metadata)
        if metadata.get("replaceSceneId"):
            tokens = metadata.get("baseRevisions", {})
            if not all(isinstance(tokens.get(name), str) and tokens[name] for name in ("content", "metadata")):
                raise PortalProtocolError("Missing gallery revision tokens")
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
        upload = self._request("POST", "/splats/uploads", create)
        checkpoint["request"] = {key: value for key, value in create.items() if key != "idempotencyKey"}
        upload_id = _identifier(upload["id"])
        checkpoint["uploadId"] = upload_id
        on_checkpoint(dict(checkpoint))
        # The create response can be an idempotent replay. Storage is the
        # authority for parts acknowledged before the last local checkpoint.
        if resuming:
            upload = self._request("GET", f"/splats/uploads/{upload_id}")
        if upload.get("status") == "completed":
            on_progress(size, size)
            return upload
        if upload.get("status") == "failed" and upload.get("processing", {}).get("retryable"):
            upload = self._request("POST", f"/splats/uploads/{upload_id}/complete", {"idempotencyKey": checkpoint["idempotencyKey"], "parts": []})
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
                stream.seek((number - 1) * part_size)
                data = stream.read(length)
                if len(data) != length:
                    raise ValueError("The export changed during upload")
                def put_part():
                    signed = self._request("POST", f"/splats/uploads/{upload_id}/part-upload-urls", {"parts": [number]})
                    urls = signed.get("urls", [])
                    if len(urls) != 1 or urls[0].get("partNumber") != number:
                        raise PortalProtocolError("Invalid gallery upload URL response")
                    req = urllib.request.Request(self._storage_url(urls[0]["url"]), data=data, method="PUT",
                        headers={"Content-Type": "application/octet-stream", "User-Agent": self.user_agent})
                    def send():
                        with urlopen(req, timeout=120, no_redirect=True) as response:
                            tag = response.headers.get("ETag")
                            if not tag or not 200 <= response.status < 300:
                                raise GalleryTransferInvalid("Storage did not acknowledge the upload part")
                            return tag
                    return retry_call(send, idempotent=True)
                for renewal in range(2):
                    try:
                        etag = put_part()
                        break
                    except urllib.error.HTTPError as exc:
                        if exc.code not in (401, 403) or renewal:
                            raise
                        exc.close()  # Renew an expired presigned URL once, same part bytes.
                parts[number] = {"partNumber": number, "etag": etag, "size": length}
                completed += length
                on_progress(completed, size)
                on_checkpoint(dict(checkpoint))
        check_canceled()
        if path.stat().st_size != size or _fingerprint(path, cancel) != fingerprint:
            raise ValueError("The export changed during upload. Export again before retrying.")
        try:
            result = self._request("POST", f"/splats/uploads/{upload_id}/complete", {"idempotencyKey": checkpoint["idempotencyKey"], "parts": [
                {"partNumber": n, "etag": parts[n]["etag"]} for n in range(1, part_count + 1)]})
        except PortalHTTPError as exc:
            if exc.status != 409:
                raise
            result = {"id": upload_id, "status": "conflict", "conflict": {"detail": exc.detail}}
        return self._await_processing(result, upload_id, size, cancel, on_processing)
