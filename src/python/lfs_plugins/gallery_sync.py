# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Persistent gallery work queue. Network and disk transfers run off the UI thread."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
from http.client import IncompleteRead, RemoteDisconnected
import json
import math
import os
import shutil
import tempfile
import threading
import uuid
import time
import urllib.error
from pathlib import Path

from .portal_account import PortalHTTPError, PortalProtocolError, _locked_sidecar
from .portal_gallery import (PortalGalleryClient, GalleryTransferCanceled, GalleryProcessingPaused,
    GalleryProcessingTimeout, GalleryTransferInvalid, PROCESSING_TIMEOUT, DEFAULT_MAX_FILE_BYTES, disk_preflight, domain_tokens, UNSUPPORTED_PORTAL)
from .portal_retry import transfer_attempts, is_transient
from .portal_security import redact, safe_filename
from .credential_storage import FileBackend
from . import gallery_validation, gallery_preparation


MAX_JOURNAL_BYTES = 32 * 1024 * 1024


def shared_fields(scene):
    """Fields shared with LichtFeld Studio; cover, highlights and broad revision excluded."""
    return copy.deepcopy({key: scene.get(key, {} if key == "viewerSettings" else "")
                          for key in ("title", "description", "visibility", "viewerSettings")})


def exchange_link(scene, commit_uuid=""):
    now = time.time()
    return {"sceneId": scene["id"], **domain_tokens(scene),
            "metadata": copy.deepcopy(scene), "sharedFields": shared_fields(scene),
            "commitUuid": commit_uuid, "exchangedAt": now, "checkedAt": now}
JOURNAL_RECOVERY_MESSAGE = (
    "LichtFeld Studio couldn't read your saved gallery links and transfers. "
    "Your files have been kept. Open the recovery folder for help, then retry."
)
JOURNAL_CHANGED_MESSAGE = "Another LichtFeld Studio window updated gallery sync. Refresh to load its changes before continuing."


def _journal_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate sync record key")
        value[key] = item
    return value


def _journal_number(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("Non-finite sync value")
    return value


def _validate_journal(data):
    """Reject damaged records as a whole; dropping one can lose an upload key."""
    def require(condition):
        if not condition:
            raise ValueError("Invalid sync record")

    def optional_text(record, keys):
        require(all(key not in record or isinstance(record[key], str) for key in keys))

    def guards(record):
        if "baseRevisions" in record:
            tokens = record["baseRevisions"]
            require(isinstance(tokens, dict) and bool(tokens) and not tokens.keys() - {"content", "metadata"}
                    and all(isinstance(value, str) and bool(value) for value in tokens.values()))

    require(isinstance(data, dict) and type(data.get("version")) is int and data["version"] == 2
        and isinstance(data.get("accounts"), dict))
    for bucket in data["accounts"].values():
        require(isinstance(bucket, dict) and isinstance(bucket.get("links"), dict)
            and isinstance(bucket.get("jobs"), list))
        for link in bucket["links"].values():
            require(isinstance(link, dict) and all(isinstance(link.get(key), str) for key in ("sceneId",)))
            require(isinstance(link.get("metadata", {}), dict))
            optional_text(link, ("commitUuid",))
            require(all(isinstance(link.get(key), str) and link[key] for key in ("contentRevision", "metadataRevision")))
            require(isinstance(link.get("sharedFields", {}), dict))
            for key in ("exchangedAt", "checkedAt"):
                require(key not in link or (type(link[key]) in (float, int) and math.isfinite(link[key]) and link[key] >= 0))
            optional_text(link.get("metadata", {}), ("id", "title", "description", "visibility"))
        identifiers = set()
        for job in bucket["jobs"]:
            require(isinstance(job, dict))
            require(all(isinstance(job.get(key), str) for key in ("id", "project", "path", "message")))
            require(job["id"] and job["id"] not in identifiers)
            identifiers.add(job["id"])
            require(job.get("status") in ("queued", "running", "paused", "error", "conflict", "completed", "canceled"))
            require(job.get("kind", "upload") in ("upload", "download"))
            for key in ('createdAt', 'finishedAt', 'processingDeadline'):
                require(key not in job or (type(job[key]) in (int, float) and math.isfinite(job[key]) and job[key] >= 0))
            require('attempts' not in job or (type(job['attempts']) is int and job['attempts'] >= 0))
            for key in ("serverProcessing", "packaged", "needsAttention", "retryable"):
                require(key not in job or type(job[key]) is bool)
            if "preparation" in job:
                require(isinstance(job["preparation"], str) and job.get("kind", "upload") == "upload"
                        and job.get("ownedExport") is True and Path(job["path"]).suffix == ".licht")
            require(all(type(job.get(key)) is int and 0 <= job[key] <= 2**63-1 for key in ("completed", "total")))
            require(isinstance(job.get("metadata"), dict) and isinstance(job["metadata"].get("title"), str))
            optional_text(job["metadata"], ("description", "visibility", "replaceSceneId"))
            guards(job["metadata"])
            require(job.get("checkpoint") is None or isinstance(job["checkpoint"], dict))
            if job.get("checkpoint") is not None:
                optional_text(job["checkpoint"], ("origin", "sha256", "idempotencyKey", "uploadId"))
                require(all(key not in job["checkpoint"] or isinstance(job["checkpoint"][key], dict) for key in ("request", "rebase")))
                for key in ("request", "rebase"):
                    guards(job["checkpoint"].get(key, {}))
            for key in ("result", "localUpdate", "stagedImport", "linkOperation"):
                require(key not in job or isinstance(job[key], dict))
            for key in ("localUpdate", "stagedImport", "linkOperation"):
                optional_text(job.get(key, {}), ("id", "state", "path", "backupPath", "message", "sha256", "project"))
            if "result" in job:
                require(all(isinstance(job["result"].get(key), str) for key in ("id",)))
                optional_text(job["result"], ("title", "description", "visibility", "sourceFormat"))
            if job.get("kind") == "download":
                require(all(isinstance(job.get(key), str) for key in ("sceneId",)))
                require(job["status"] != "completed" or "result" in job)
    return data


def friendly_error(exc):
    if isinstance(exc, PortalHTTPError):
        if exc.error == "gallery_relink_required":
            return "Use the Portal button to approve gallery access. Your local work is safe."
        if exc.status == 400 and exc.error in (
                "Invalid portable LichtFeld project.", "Project checksum failed.",
                "Embedded project asset checksum failed."):
            return "The downloaded file is damaged or was changed on the portal."
        return {
            401: "Sign in again, then resume the transfer.",
            403: "Gallery access is unavailable for this account. Check your account on the portal.",
            404: "This gallery item is no longer available. Refresh the gallery.",
            409: "The gallery item changed. Refresh and review both versions before replacing it.",
            429: "The portal is busy. Wait a moment, then resume.",
        }.get(exc.status, "The portal could not finish this operation. Your local work is safe. Retry when ready.")
    if isinstance(exc, (ValueError, PortalProtocolError)):
        return redact(exc)
    return "The connection or local storage was interrupted. Check your connection and disk space, then resume."


def file_stamp(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


class GallerySync:
    def __init__(self, account, root):
        self.account = account
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._persist_lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread = None
        self._operation = None
        self._session = None
        self._origin = None
        self._owner = None
        self._source_formats = []
        self._max_file_bytes = DEFAULT_MAX_FILE_BYTES
        self._storage_hosts = None
        self._quota_bytes = None
        self._used_bytes = None
        self._completion = None
        self._unsupported_identity = None
        self._revision_domains = 0
        self._list_etag = None
        self._checked_at = 0
        self._poster_entries = {}
        self._poster_identity = None
        self._refresh_ok = False
        self._relink_identity = None
        self.scenes = []
        self._undo_restore = {}
        self.message = "Refresh to connect your gallery."
        self.version = 0
        self._data = {"version": 2, "accounts": {}}
        self._journal = self.root / "sync.json"
        self._disk_digest = None
        self._journal_seen = False
        self._journal_problem = False
        self._stale = False
        try:
            try:
                with _locked_sidecar(self.root / "sync.lock", blocking=False):
                    self._reload_journal(recover_interrupted=True)
            except OSError:
                self._reload_journal()
        except (OSError, ValueError, RecursionError):
            self._journal_problem = True
            self.message = JOURNAL_RECOVERY_MESSAGE

    def _journal_bytes(self):
        try:
            with self._journal.open("rb") as file:
                self._journal_seen = True
                encoded = file.read(MAX_JOURNAL_BYTES + 1)
        except FileNotFoundError:
            if self._journal_seen:
                raise ValueError("The saved sync record is missing")
            return None
        if len(encoded) > MAX_JOURNAL_BYTES:
            raise ValueError("The saved sync record is too large")
        return encoded

    def _reload_journal(self, *, recover_interrupted=False):
        encoded = self._journal_bytes()
        digest = hashlib.sha256(encoded).hexdigest() if encoded is not None else None
        if encoded is None:
            data = {"version": 2, "accounts": {}}
        else:
            data = _validate_journal(json.loads(encoded, object_pairs_hook=_journal_object,
                parse_float=_journal_number, parse_constant=_journal_number))
        with self._lock:
            if digest != self._disk_digest or self._journal_problem:
                for bucket in data["accounts"].values():
                    for job in bucket["jobs"]:
                        if (recover_interrupted or self._journal_problem) and job["status"] in ("queued", "running"):
                            job.update(status="paused", interrupted=True, message="Interrupted. Resume when ready.")
                self._data = data
            self._disk_digest = digest
            self._journal_problem = self._stale = False

    def _save(self):
        with self._persist_lock:
            with self._lock:
                self._check_journal_ready()
                self._prune_jobs()
                encoded = json.dumps(self._data, allow_nan=False)
                if len(encoded.encode()) > MAX_JOURNAL_BYTES:
                    raise ValueError("Gallery transfer history is too large to save. Keep the recovery folder for help.")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=self.root, delete=False) as file:
                    temporary = Path(file.name)
                    file.write(encoded)
                    file.flush()
                    os.fsync(file.fileno())
                previous = self._journal_bytes()
                if previous is not None:
                    FileBackend(self._journal.with_suffix(".json.bak")).write(previous)
                os.replace(temporary, self._journal)
                self._journal_seen = True
                self._disk_digest = hashlib.sha256(encoded.encode()).hexdigest()
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
            with self._lock:
                self.version += 1

    def _prune_jobs(self):
        now = time.time()
        for bucket in self._data['accounts'].values():
            for job in bucket['jobs']:
                job.setdefault('createdAt', now)
                if job['status'] in ('completed', 'canceled'):
                    job.setdefault('finishedAt', now)
            terminal = sorted((job for job in reversed(bucket['jobs']) if job['status'] in ('completed', 'canceled')),
                              key=lambda job: job['finishedAt'], reverse=True)
            recent = {job['id'] for job in terminal[:200] if job['finishedAt'] >= now - 30 * 86400}
            bucket['jobs'][:] = [job for job in bucket['jobs'] if job['status'] not in ('completed', 'canceled')
                or job['id'] in recent or self._owns_recovery_path(job)]

    @staticmethod
    def _owns_recovery_path(job):
        # Retain durable ownership until explicit cleanup removes the records.
        def owns(value):
            if not isinstance(value, dict):
                return False
            return any((bool(item) and (key in ('backupPath', 'recoveryPath', 'downloadPath')
                        or key == 'path' and (value is not job or job.get('kind') == 'download')))
                       or isinstance(item, dict) and owns(item) for key, item in value.items())
        return owns(job)

    def _journal_digest(self):
        encoded = self._journal_bytes()
        return hashlib.sha256(encoded).hexdigest() if encoded is not None else None

    def _check_journal_ready(self):
        if self._journal_problem:
            raise ValueError(JOURNAL_RECOVERY_MESSAGE)
        if self._stale:
            raise ValueError(JOURNAL_CHANGED_MESSAGE)


    def identity(self):
        """Read the account boundary without copying the scene/transfer history."""
        snap = self.account.snapshot()
        return self.account.base_url, snap.email, snap.connected_since, snap.signed_in

    def _bucket(self):
        if not self._owner:
            raise ValueError("Refresh the gallery before continuing.")
        key = hashlib.sha256(json.dumps([self._origin, self._owner]).encode()).hexdigest()
        return self._data["accounts"].setdefault(key, {"links": {}, "jobs": []})

    @property
    def busy(self):
        return bool(self._thread and self._thread.is_alive())

    @property
    def metadata_busy(self):
        return self.busy and self._operation != "refresh"

    def snapshot(self):
        with self._lock:
            snap = self.account.snapshot()
            self._check_poster_account(snap)
            same = bool(not self._journal_problem and not self._stale and snap.signed_in and snap.email and snap.connected_since and self._origin == self.account.base_url
                and self._session == (snap.email, snap.connected_since))
            bucket = self._bucket() if same and self._owner else {"jobs": [], "links": {}}
            return copy.deepcopy({"scenes": self.scenes if same else [], **bucket,
                "identity": (self.account.base_url, snap.email, snap.connected_since, snap.signed_in),
                "signed_in": snap.signed_in, "email": snap.email if snap.signed_in else "",
                "display_name": getattr(snap, "display_name", "") if snap.signed_in else "",
                "message": self.message if self._journal_problem or self._stale or same or (snap.signed_in and self._owner is None) else "Sign in and refresh to connect your gallery.",
                "storage_issue": self._journal_problem,
                "refresh_ok": self._refresh_ok,
                "relink_required": snap.signed_in and self._relink_identity == self.identity(),
                "unsupported": self._unsupported_identity == self.identity(),
                "source_formats": self._source_formats if same else [],
                "owner": self._owner if same else None,
                "quotaBytes": self._quota_bytes if same else None,
                "usedBytes": self._used_bytes if same else None,
                "completion": self._completion if same else None,
                "revisionDomains": self._revision_domains if same else 0,
                "checkedAt": self._checked_at if same else 0,
                "posters": {key: value["path"] for key, value in self._poster_entries.items()} if same else {},
                "undoRestore": self._undo_restore if (snap.signed_in and self._session == (snap.email, snap.connected_since)
                    and self._origin == self.account.base_url) else {},
                "busy": self.busy, "connected": bool(same and self._owner), "version": self.version})

    def _client(self):
        self._check_journal_ready()
        if self._unsupported_identity == self.identity():
            raise PortalProtocolError(UNSUPPORTED_PORTAL)
        snap = self.account.snapshot()
        if not snap.signed_in or not snap.email or not snap.connected_since or self._origin != self.account.base_url or self._session != (snap.email, snap.connected_since):
            raise ValueError("The account changed. Refresh the gallery before continuing.")
        client = PortalGalleryClient(self.account, expected_session=self._session, revision_domains=self._revision_domains)
        client.max_file_bytes = self._max_file_bytes
        client.storage_hosts = self._storage_hosts
        return client

    def _launch(self, action, *, reload_journal=False, operation=None):
        with self._lock:
            previous = self._thread if self.busy else None
            if previous and not (operation == "metadata" and self._operation == "refresh"):
                raise ValueError("Wait for the current operation or pause it first.")
            previous_operation, previous_cancel = self._operation, self._cancel
            self._cancel = threading.Event()
            identity = self.identity()

            def worker():
                try:
                    if previous:
                        previous.join()
                    # Serialize across Studio processes without blocking the UI.
                    with _locked_sidecar(self.root / "sync.lock"):
                        try:
                            if reload_journal:
                                self._reload_journal()
                            elif self._journal_digest() != self._disk_digest:
                                with self._lock:
                                    self._stale = True
                        except (OSError, ValueError, RecursionError):
                            with self._lock:
                                self._journal_problem = True
                            raise ValueError(JOURNAL_RECOVERY_MESSAGE) from None
                        self._check_journal_ready()
                        action()
                except Exception as exc:
                    with self._lock:
                        self._relink_identity = identity if isinstance(exc, PortalHTTPError) and exc.error == "gallery_relink_required" else None
                        self.message = friendly_error(exc)
                finally:
                    with self._lock:
                        self.version += 1

            self._thread = threading.Thread(target=worker, daemon=True, name="GallerySync")
            self._operation = operation
            try:
                self._thread.start()
            except Exception:
                self._thread, self._operation, self._cancel = previous, previous_operation, previous_cancel
                raise

    def _launch_metadata(self, action):
        # Refresh may reload the journal. Resolve clients/buckets only after it
        # finishes, and never apply a queued action to a newly signed-in account.
        identity = self.identity()
        self._client()
        def checked():
            if self.identity() != identity:
                raise ValueError("The account changed. Refresh the gallery before continuing.")
            action()
        self._launch(checked, operation="metadata")

    def refresh(self):
        if self._unsupported_identity == self.identity():
            return
        self._unsupported_identity = None
        self._refresh_ok = False
        def action():
            snap = self.account.snapshot()
            if not snap.signed_in:
                raise ValueError("Sign in with your LichtFeld account first.")
            if not snap.email or not snap.connected_since:
                raise ValueError("Your account details are still loading. Refresh your account, then retry.")
            session = (snap.email, snap.connected_since)
            origin = self.account.base_url
            client = PortalGalleryClient(self.account, expected_session=session)
            try:
                capabilities = client._request("GET", "/me")
                if (capabilities.get("gallerySyncVersion") != 1
                        or type(capabilities.get("revisionDomains")) is not int or capabilities["revisionDomains"] < 1):
                    raise PortalProtocolError(UNSUPPORTED_PORTAL)
            except PortalProtocolError as exc:
                if str(exc) == UNSUPPORTED_PORTAL:
                    self._unsupported_identity = (origin, *session, True)
                    self._owner = None
                    self.scenes = []
                raise
            with self._lock:
                self._check_poster_account(snap)
                same = self._session == session and self._origin == origin and self._owner == capabilities["id"]
                etag = self._list_etag if same else None
            scenes = client.list_scenes(etag=etag) if etag else client.list_scenes()
            with self._lock:
                current = self.account.snapshot()
                if not current.signed_in or (current.email, current.connected_since) != session or self.account.base_url != origin:
                    raise ValueError("The account changed. Refresh the gallery before continuing.")
                self._session, self._owner = session, capabilities["id"]
                self._origin = origin
                self._completion = None if not same else self._completion
                self._source_formats = capabilities.get("sourceFormats", [])
                self._max_file_bytes = capabilities.get("maxFileBytes", DEFAULT_MAX_FILE_BYTES)
                self._storage_hosts = copy.deepcopy(capabilities.get("storageHosts"))
                self._quota_bytes = capabilities.get("quotaBytes")
                self._used_bytes = capabilities.get("usedBytes")
                version = capabilities.get("revisionDomains", 0)
                self._revision_domains = version if type(version) is int else 0
                self._list_etag = getattr(client, "list_etag", None) or (etag if scenes is None else None)
                self._checked_at = time.time()
                if scenes is not None:
                    self.scenes = scenes
                for link in self._bucket()["links"].values():
                    link["checkedAt"] = self._checked_at
                self._refresh_ok = True
                self._relink_identity = None
                self._unsupported_identity = None
                self.message = "Gallery is up to date."
            self._cache_posters(client, self.scenes, (origin, *session, True))
            # Recovered jobs are persisted by the next actual mutation.
        self._launch(action, reload_journal=True, operation="refresh")

    def _check_poster_account(self, snap):
        identity = (self.account.base_url, snap.email, snap.connected_since, snap.signed_in)
        with self._lock:
            if identity != self._poster_identity or not snap.signed_in:
                shutil.rmtree(self.root / "posters", ignore_errors=True)
                self._poster_entries.clear()
                self._storage_hosts = None
                self._poster_identity = identity
                self._list_etag = None
                self.scenes = []
                self._undo_restore = {}
                self._completion = None
                self._checked_at = 0
                self._refresh_ok = False

    def _trim_poster_cache(self, limit):
        files = sorted((self.root / "posters").glob("*.png"), key=lambda p: p.stat().st_atime_ns)
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= limit:
                break
            total -= path.stat().st_size
            path.unlink(missing_ok=True)
        self._poster_entries = {key: value for key, value in self._poster_entries.items()
                                if Path(value["path"]).is_file()}

    def _cache_posters(self, client, scenes, identity):
        """Worker-only authenticated cache; no signed URL or bearer is persisted."""
        from .gallery_preferences import read_preferences
        cache_limit = read_preferences(self.root)["posterCacheMiB"] * 1024 * 1024
        folder = self.root / "posters"
        live = {scene["id"] for scene in scenes if scene.get("thumbnailUrl") and scene.get("status") == "ready"}
        with self._lock:
            try:
                self._trim_poster_cache(cache_limit)
            except OSError:
                pass  # Poster I/O must not fail a successful scene listing.
            for key in set(self._poster_entries) - live:
                Path(self._poster_entries.pop(key)["path"]).unlink(missing_ok=True)
        for scene in scenes:
            if scene["id"] not in live or self._cancel.is_set():
                continue
            try:
                scene_id = str(uuid.UUID(scene["id"]))
                revision = scene.get("posterRevision", "")
                if not isinstance(revision, str) or not revision or len(revision) > 128 or any(
                        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in revision):
                    continue
                with self._lock:
                    entry = self._poster_entries.get(scene_id, {})
                    old = Path(entry["path"]) if entry else None
                    etag = entry.get("etag") if old and old.is_file() else None
                status, tag, data = client.thumbnail(scene_id, etag=etag)
                with self._lock:
                    if self.identity() != identity:
                        self._check_poster_account(self.account.snapshot())
                        return
                    if status == 304 and old and old.is_file():
                        # LRU access must not change the image decoder's revision.
                        os.utime(old, ns=(time.time_ns(), old.stat().st_mtime_ns))
                        continue
                    if status == 304:
                        continue
                    folder.mkdir(parents=True, exist_ok=True)
                    # Invalidate the old texture source even if a server changes
                    # bytes/ETag without changing the advertised poster token.
                    for stale in folder.glob(scene_id + "-*.png"):
                        stale.unlink(missing_ok=True)
                    destination = folder / f"{scene_id}-{revision}.png"
                    destination.write_bytes(data)
                    os.utime(destination, ns=(time.time_ns(), destination.stat().st_mtime_ns))
                    self._poster_entries[scene_id] = {"path": str(destination), "etag": tag}
                    self._trim_poster_cache(cache_limit)
            except (OSError, ValueError, PortalHTTPError, PortalProtocolError):
                # A missing/malformed poster does not prevent scene synchronization.
                with self._lock:
                    entry = self._poster_entries.pop(scene["id"], None)
                    if entry:
                        Path(entry["path"]).unlink(missing_ok=True)

    def queue_prepared_upload(self, staging, metadata, project_id):
        staging = gallery_preparation.staging_path(self.root, staging)
        if not (staging / "project.licht").is_file():
            raise ValueError("Prepare a fresh .licht file in LichtFeld Studio before uploading.")
        return self.queue_upload(staging.with_suffix(".licht"), metadata, project_id,
                                 owned_export=True, preparation=str(staging))

    def queue_upload(self, export_path, metadata, project_id, *, owned_export=False, preparation=None):
        with self._lock:
            self._client()
            if self.busy:
                raise ValueError("Wait for the current operation or pause it first.")
            if preparation is not None and Path(export_path).suffix[1:] not in self._source_formats:
                raise ValueError(UNSUPPORTED_PORTAL)
            linked = self._bucket()["links"].get(project_id)
            if linked and metadata.get("replaceSceneId") != linked["sceneId"] and not metadata.get("_publishAsNew"):
                raise ValueError("This project is linked to another gallery item. Select its linked item or unlink the project first.")
            jobs = self._bucket()["jobs"]
            if any(j["project"] == project_id and j["status"] not in ("completed", "canceled") for j in jobs):
                raise ValueError("This project already has a transfer. Resume or discard it first.")
            job = {"id": str(uuid.uuid4()), "project": project_id, "path": str(export_path),
                "kind": "upload",
                "ownedExport": owned_export,
                "metadata": copy.deepcopy(metadata), "checkpoint": None, "status": "queued",
                "completed": 0, "total": 0 if preparation is not None else Path(export_path).stat().st_size,
                "message": "Ready to prepare" if preparation is not None else "Ready to upload"}
            job["commitUuid"] = job["metadata"].pop("_commitUuid", "")
            job["uploadFormat"] = job["metadata"].pop("_uploadFormat", "studio")
            job["contentStamp"] = job["metadata"].pop("_contentStamp", "")
            job["publishAsNew"] = job["metadata"].pop("_publishAsNew", False)
            if preparation is not None:
                job.update(preparation=preparation, packaged=False)
            guard = _locked_sidecar(self.root / "sync.lock", blocking=False)
            try:
                guard.__enter__()
            except OSError:
                raise ValueError("Another LichtFeld Studio window is updating gallery sync. Try again when it finishes.") from None
            try:
                if self._journal_digest() != self._disk_digest:
                    self._stale = True
                    raise ValueError(JOURNAL_CHANGED_MESSAGE)
                jobs.append(job)
                try:
                    self._save()
                except Exception:
                    jobs.remove(job)
                    raise
            finally:
                guard.__exit__(None, None, None)
        # The ownership handoff is now durable. A thread-start failure must keep
        # the queued snapshot for resume instead of making the panel delete it.
        try:
            self.resume(job["id"])
        except Exception:
            with self._lock:
                job.update(status="paused", message="Scene prepared. Resume when ready to upload.")
                self.message = job["message"]
                self.version += 1
        return job["id"]

    def download(self, scene, *, destination=None):
        with self._lock:
            self._client()
            if self.busy:
                raise ValueError("Wait for the current transfer or pause it first.")
            if scene["sourceFormat"] != "licht":
                raise ValueError("This scene format cannot be opened in LichtFeld Studio.")
            total = scene.get('contentLength')
            if (type(total) is not int or total <= 0 or type(self._max_file_bytes) is not int
                    or total > self._max_file_bytes):
                raise PortalProtocolError('Gallery download exceeds the portal file-size limit')
            identifier = str(uuid.uuid4())
            path = self.root / "downloads" / (identifier + "." + scene["sourceFormat"])
            jobs = self._bucket()["jobs"]
            job = {"id": identifier, "project": "", "kind": "download",
                "path": str(path), "metadata": {"title": scene["title"]}, "sceneId": scene["id"],
                "checkpoint": None, "status": "queued", "completed": 0,
                "total": scene["contentLength"], "message": "Ready to download"}
            if destination:
                target = Path(destination)
                if (not target.is_absolute() or target.name != safe_filename(target.stem) or target.exists()
                        or not target.parent.is_dir() or any(part == ".." for part in target.parts)):
                    raise ValueError("Choose an unused .licht filename in an existing folder.")
                job["destination"] = str(target)
            final_path = self._download_destination(job)
            disk_preflight([(path, total * 2), (final_path, total)])
            guard = _locked_sidecar(self.root / "sync.lock", blocking=False)
            try:
                guard.__enter__()
            except OSError:
                raise ValueError("Another LichtFeld Studio window is updating gallery sync. Try again when it finishes.") from None
            try:
                if self._journal_digest() != self._disk_digest:
                    self._stale = True
                    raise ValueError(JOURNAL_CHANGED_MESSAGE)
                jobs.append(job)
                try:
                    self._save()
                except Exception:
                    jobs.remove(job)
                    raise
            finally:
                guard.__exit__(None, None, None)
        try:
            self.resume(job["id"])
        except Exception:
            with self._lock:
                job.update(status="paused", message="Ready to download. Resume when ready.")
                self.message = job["message"]
                self.version += 1

    def _download_destination(self, job):
        if job.get('destination'):
            return Path(job['destination'])
        if Path(job['path']).suffix == '.licht':
            from .asset_index import resolve_default_asset_directory
            return resolve_default_asset_directory() / ('Gallery-' + job['id'] + '.licht')
        return Path(job['path'])

    def _job(self, job_id):
        return next(j for j in self._bucket()["jobs"] if j["id"] == job_id)


    def resume(self, job_id, *, keep_waiting=False):
        with self._lock:
            client = self._client()
            job = self._job(job_id)
            if job["status"] in ("completed", "canceled"):
                raise ValueError("This transfer is already finished.")
            if job.get("retryable") is False:
                raise ValueError(job["message"])
            bucket = self._bucket()
            extend_processing = keep_waiting or job.get("needsAttention", False)

        def action():
            def checkpoint(value):
                with self._lock:
                    job["checkpoint"] = value
                    request = value.get("request", {})
                    if "baseRevisions" in request:
                        job["metadata"]["baseRevisions"] = copy.deepcopy(request["baseRevisions"])
                self._save()

            def progress(done, total):
                self._client()
                with self._lock:
                    job.update(completed=done, total=total)
                    self.version += 1

            def processing(state):
                label = {"queued": "Waiting for the portal", "assembling": "Assembling upload",
                    "validating": "Checking scene", "publishing": "Publishing scene"}[state["stage"]]
                with self._lock:
                    job.update(serverProcessing=True, completed=state["completed"], total=state["total"], message=label)
                    if getattr(client, 'processing_deadline', None) and job.get('processingDeadline') != client.processing_deadline:
                        job['processingDeadline'] = client.processing_deadline
                        self._save()
                    self.message = label
                    self.version += 1

            def complete_upload(result):
                self._client()
                scene = result["scene"]
                with self._lock:
                    self._completion = {"id": str(uuid.uuid4()), "kind": "publish", "scene": copy.deepcopy(scene)}
                    bucket["links"][job["project"]] = exchange_link(scene, job.get("commitUuid", ""))
                    bucket["links"][job["project"]]["uploadFormat"] = job.get("uploadFormat", "studio")
                    bucket["links"][job["project"]]["contentStamp"] = job.get("contentStamp", "")
                    job.update(status="completed", completed=job["total"], serverProcessing=False, message="Uploaded", result=scene)
                    self.scenes = [s for s in self.scenes if s["id"] != scene["id"]] + [scene]
                    self.message = "Upload complete. Review Story on portal after this content change." if job["metadata"].get("replaceSceneId") else "Upload complete."
                self._save()
                self._retire_export(job)

            try:
                with self._lock:
                    if extend_processing:
                        job["processingDeadline"] = time.time() + PROCESSING_TIMEOUT
                    job.update(status="running", interrupted=False, needsAttention=False, serverProcessing=False, message="Downloading" if job.get("kind") == "download" else "Uploading")
                    self.message = job["message"]
                self._save()
                if keep_waiting and (job.get('checkpoint') or {}).get('uploadId') and job.get('kind') != 'download':
                    upload_id = str(uuid.UUID(job['checkpoint']['uploadId']))
                    client.processing_deadline = job.get('processingDeadline')
                    result = client._await_processing(client._request('GET', f'/splats/uploads/{upload_id}'),
                        upload_id, job['total'], self._cancel, processing)
                    complete_upload(result)
                    return
                if job.get("ownedExport"):
                    self._cleanup_paths(job, {})  # Apply ownership checks before reading, including resumed packages.
                if job.get("preparation") and not job.get("packaged"):
                    self._client()
                    staging = gallery_preparation.staging_path(self.root, job["preparation"])
                    destination = Path(job["path"]).absolute()
                    if destination != staging.with_suffix(".licht") or destination.is_symlink():
                        raise ValueError("Scene preparation no longer matches its transfer. Keep it for recovery.")
                    nodes, total = gallery_preparation.read_staging(self.root, job["preparation"])
                    with self._lock:
                        job.update(completed=0, total=total, message="Preparing scene package")
                        self.message = job["message"]
                        self.version += 1

                    def packaging_progress(done):
                        self._client()
                        if self._cancel.is_set():
                            raise GalleryTransferCanceled()
                        progress(done, total)

                    packaging_progress(0)
                    background = staging / "environment.lfsenv"
                    if background.exists() != bool(job["metadata"].get("viewerSettings", {}).get("environment")):
                        raise ValueError("The HDR background changed. Prepare the scene again before uploading.")
                    from .portable_project import ProjectFile
                    source_path = staging / "project.licht"
                    total = source_path.stat().st_size
                    Path(job["path"]).unlink(missing_ok=True)
                    with source_path.open("rb") as source:
                        ProjectFile(source)  # Admit only the fresh native publishing subset.
                        stamp = gallery_validation._stamp(source)
                        source.seek(0)
                        with Path(job["path"]).open("xb") as output:
                            copied = 0
                            while chunk := source.read(gallery_validation.CHUNK_BYTES):
                                output.write(chunk)
                                copied += len(chunk)
                                if copied > stamp[2]: raise ValueError("The prepared project changed.")
                                packaging_progress(copied)
                            output.flush()
                            os.fsync(output.fileno())
                        if copied != stamp[2] or gallery_validation._stamp(source) != stamp:
                            raise ValueError("The prepared project changed. Prepare it again.")
                    packaging_progress(total)
                    with self._lock:
                        job.update(packaged=True, completed=0, total=Path(job["path"]).stat().st_size, message="Uploading")
                        self.message = job["message"]
                    self._save()
                if job.get("kind") == "download":
                    def download_message(message):
                        with self._lock:
                            job['message'] = message
                            self.message = message
                            self.version += 1
                    scene = client.download(job["sceneId"], job["path"], on_progress=progress, cancel=self._cancel,
                        checkpoint=job.get('checkpoint'), on_checkpoint=checkpoint, on_message=download_message,
                        final_destination=self._download_destination(job))
                    self._client()
                    with self._lock:
                        self._completion = {"id": str(uuid.uuid4()), "kind": "download"}
                        job.update(status="completed", sha256=(job.get("checkpoint") or {}).get("sha256", ""), result=scene, message="Downloaded. Open as a new project when ready.")
                        self.message = job["message"]
                    self._save()
                    return
                client.processing_deadline = job.get('processingDeadline')
                result = client.upload(job["path"], job["metadata"], checkpoint=job["checkpoint"],
                    on_checkpoint=checkpoint, on_progress=progress, on_processing=processing, cancel=self._cancel)
                complete_upload(result)
            except Exception as exc:
                if job["status"] == "completed":
                    with self._lock:
                        job.update(cleanupPending=True, message=("Download complete. Its local files were kept; refresh to check the saved transfer."
                            if job.get("kind") == "download" else "Upload complete. Temporary files were kept; refresh before cleaning them up."))
                        self.message = job["message"]
                    self._save()
                    return
                if isinstance(getattr(exc, 'reason', exc), (IncompleteRead, RemoteDisconnected)):
                    exc = (ConnectionError("Gallery download connection closed before completion")
                        if job.get("kind") == "download" else GalleryTransferInvalid(
                            "The portal closed the upload without acknowledging it. Start a new upload."))
                elif isinstance(exc, PortalHTTPError) and exc.status == 400 and exc.error in (
                        "Invalid portable LichtFeld project.", "Project checksum failed.",
                        "Embedded project asset checksum failed."):
                    exc = GalleryTransferInvalid(friendly_error(exc))
                with self._lock:
                    job.update(status="paused" if isinstance(exc, GalleryTransferCanceled) else "conflict"
                        if isinstance(exc, PortalHTTPError) and exc.status == 409 else "error", message=friendly_error(exc))
                    if (is_transient(exc) and not isinstance(exc, (PortalHTTPError, urllib.error.HTTPError))
                            and not self._cancel.is_set()):
                        job.update(status="paused", message="Paused (connection lost)")
                    elif self._cancel.is_set():
                        job.update(status="paused", message="Paused. Resume when ready.")
                    if isinstance(exc, GalleryTransferInvalid):
                        job.update(status="error", retryable=False)
                    if isinstance(exc, GalleryTransferCanceled):
                        job["message"] = (("Paused. The pinned download will resume from its saved bytes." if (job.get('checkpoint') or {}).get('representationId')
                            else "Paused. The download will restart from zero because the portal has no pinned representation.") if job.get("kind") == "download"
                            else "Paused. Uploaded parts will be reused when you resume.")
                    if isinstance(exc, GalleryProcessingTimeout):
                        job.update(needsAttention=True, message=str(exc))
                    if isinstance(exc, GalleryProcessingPaused):
                        job["message"] = "Stopped waiting. The portal may continue checking this upload. Resume to check its status."
                    elif isinstance(exc, GalleryTransferCanceled) and job.get("preparation") and not job.get("packaged"):
                        job["message"] = "Preparation paused. Resume to prepare the saved scene and upload it."
                    self.message = job["message"]
                self._save()
        def attempted():
            self._client()
            with self._lock:
                job['attempts'] = job.get('attempts', 0) + 1
            self._save()

        def run():
            with transfer_attempts(attempted, self._cancel):
                action()
        self._launch(run)

    def link_download(self, job_id, project_id, commit_uuid=""):
        self._client()
        job, bucket = self._job(job_id), self._bucket()
        if job.get("kind") != "download" or job["status"] != "completed" or job.get("retired") or job.get("cleanupPending"):
            raise ValueError("Finish downloading this scene first.")
        linked = bucket["links"].get(project_id)
        if linked and linked["sceneId"] != job["result"]["id"]:
            raise ValueError("This project is linked to a different gallery item.")
        operation = str(uuid.uuid4())
        def action():
            previous, previous_project = copy.deepcopy(bucket["links"].get(project_id)), job["project"]
            try:
                self._client()
                with self._lock:
                    scene = job["result"]
                    bucket["links"][project_id] = exchange_link(scene, commit_uuid)
                    job["project"] = project_id
                    job["linkOperation"] = {"id": operation, "state": "ready"}
                self._save()
            except Exception as exc:
                with self._lock:
                    if previous is None:
                        bucket["links"].pop(project_id, None)
                    else:
                        bucket["links"][project_id] = previous
                    job["project"] = previous_project
                    job["linkOperation"] = {"id": operation, "state": "failed", "message": friendly_error(exc)}
                raise
        self._launch(action)
        return operation

    def prepare_local_update(self, job_id, project_id, project_path, expected_stamp):
        """Keep the saved local version before the UI replaces its visible splats."""
        self._client()
        job, bucket = self._job(job_id), self._bucket()
        linked = bucket["links"].get(project_id)
        if (job.get("kind") != "download" or job["status"] != "completed" or job.get("retired") or job.get("cleanupPending") or
                not linked or linked["sceneId"] != job["result"]["id"]):
            raise ValueError("Download the current project's linked gallery item first.")
        if any(j["project"] == project_id and j["status"] not in ("completed", "canceled") for j in bucket["jobs"]):
            raise ValueError("Finish or discard this project's pending transfer before updating it.")
        update_id = str(uuid.uuid4())
        record = {"id": update_id, "state": "preparing", "project": project_id,
            "path": str(project_path), "sourceStamp": list(expected_stamp)}

        def action():
            temporary = None
            try:
                with self._lock:
                    job["localUpdate"] = record
                    self.message = "Keeping a recovery copy of your local project…"
                self._save()
                remote = self._client().scene(job["result"]["id"])
                if domain_tokens(remote) != domain_tokens(job["result"]):
                    raise ValueError("The gallery item changed since this download. Download its latest version before updating.")
                directory = self.root / "backups"
                backup = directory / (update_id + ".licht")
                disk_preflight([(backup, Path(project_path).stat().st_size),
                                (project_path, expected_stamp[2] + job['total'])])
                directory.mkdir(mode=0o700, exist_ok=True)
                if file_stamp(project_path) != expected_stamp:
                    raise ValueError("The local project changed. Review it before updating.")
                digest = hashlib.sha256()
                with open(project_path, "rb") as source, tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                    temporary = Path(output.name)
                    while block := source.read(4 * 1024 * 1024):
                        if self._cancel.is_set():
                            raise GalleryTransferCanceled()
                        digest.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                if file_stamp(project_path) != expected_stamp:
                    raise ValueError("The local project changed while making its recovery copy. Try again.")
                self._client()
                if self._cancel.is_set():
                    raise GalleryTransferCanceled()
                os.replace(temporary, backup)
                temporary = None
                with self._lock:
                    record.update(state="ready", backupPath=str(backup), sha256=digest.hexdigest())
            except Exception as exc:
                with self._lock:
                    record.update(state="failed", message="Update canceled. Your local splats remain." if isinstance(exc, GalleryTransferCanceled) else friendly_error(exc))
                    self.message = record["message"]
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
                self._save()
        self._launch(action)
        return update_id

    def stage_download(self, job_id):
        """Give a native import a unique name so it cannot be confused with local nodes."""
        self._client()
        job = self._job(job_id)
        if job.get("kind") != "download" or job["status"] != "completed" or job.get("retired") or job.get("cleanupPending"):
            raise ValueError("Finish downloading this scene first.")
        identifier = str(uuid.uuid4())
        record = {"id": identifier, "state": "preparing"}

        def action():
            target = self.root / "imports" / (identifier + ".scene")
            # Retire the previous preview while its ownership record still exists.
            # Repeated updates must not orphan a directory on every attempt.
            try:
                self._client()
                previous_paths = self._cleanup_paths(job, {})
                for path in previous_paths[1:]:  # The first path is the kept download.
                    self._unlink_temporary(path)
            except Exception as exc:
                with self._lock:
                    previous = job.setdefault("stagedImport", record)
                    previous.update(state="failed", message=friendly_error(exc))
                self._save()
                return
            with self._lock:
                job["stagedImport"] = record
            retained_asset = None
            retained_project = None
            try:
                disk_preflight([(target, Path(job['path']).stat().st_size),
                                (self._download_destination(job), Path(job['path']).stat().st_size)])
                target.parent.mkdir(mode=0o700, exist_ok=True)
                with self._lock:
                    record["path"] = str(target)
                self._save()  # Keep partial preparations discoverable after a restart.
                def progress(done, total):
                    self._client()
                    if self._cancel.is_set():
                        raise GalleryTransferCanceled()
                    with self._lock:
                        record.update(completed=done, total=total)
                        self.message = f"Checking downloaded scene… {int(100 * done / max(1, total))}%"
                        self.version += 1
                gallery_preparation.unpack_project(target.parent, job["path"], target, progress=progress)
                from .asset_index import resolve_default_asset_directory
                assets = resolve_default_asset_directory()
                assets.mkdir(parents=True, exist_ok=True)
                project_path = Path(job["destination"]) if job.get("destination") else assets / ("Gallery-" + identifier + ".licht")
                if project_path.exists():
                    raise ValueError("The destination already exists. Choose another file name.")
                with open(job["path"], "rb") as source, project_path.open("xb") as output:
                    retained_project = project_path
                    while chunk := source.read(gallery_validation.CHUNK_BYTES):
                        if self._cancel.is_set(): raise GalleryTransferCanceled()
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                with self._lock:
                    record["projectPath"] = str(project_path)
                background = target / "environment.lfsenv"
                if background.exists():
                    # Keep a private asset independently of disposable import
                    # staging. Saved projects/recovery copies reference it.
                    assets = self.root / "environments"
                    assets.mkdir(mode=0o700, exist_ok=True)
                    if assets.is_symlink() or getattr(assets, "is_junction", lambda: False)():
                        raise ValueError("The HDR asset folder was redirected.")
                    asset = assets / (identifier + ".lfsenv")
                    with background.open("rb") as source, asset.open("xb") as output:
                        retained_asset = asset
                        while chunk := source.read(gallery_validation.CHUNK_BYTES):
                            if self._cancel.is_set():
                                raise GalleryTransferCanceled()
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    asset.chmod(0o600)
                    with self._lock:
                        record["environmentPath"] = str(asset)
                self._client()
                if self._cancel.is_set():
                    raise GalleryTransferCanceled()
                with self._lock:
                    record.update(state="ready", path=str(target))
            except Exception as exc:
                if retained_project is not None:
                    retained_project.unlink(missing_ok=True)
                if retained_asset is not None:
                    retained_asset.unlink(missing_ok=True)
                for path in gallery_preparation.staging_files(target.parent, target):
                    self._unlink_temporary(path)
                with self._lock:
                    record.update(state="failed", message="Preparation canceled. Your download was kept." if isinstance(exc, GalleryTransferCanceled) else friendly_error(exc))
            self._save()
        self._launch(action)
        return identifier

    def environment_path(self, job):
        if not job.get("result", {}).get("viewerSettings", {}).get("environment"):
            return None
        with self._lock:
            stage = dict(self._job(job["id"]).get("stagedImport", {}))
        expected_id = job.get("_opening", job.get("_update", {})).get("stage_id")
        if stage.get("state") != "ready" or (expected_id is not None and stage.get("id") != expected_id):
            raise ValueError("The HDR background preparation changed. Open the download again.")
        identifier = str(uuid.UUID(stage.get("id", "")))
        expected = self.root / "environments" / (identifier + ".lfsenv")
        if (stage.get("environmentPath") != str(expected) or not expected.is_file()
                or any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                       for p in (expected.parent, expected))):
            raise ValueError("Download the HDR background again before opening this scene.")
        return expected


    def pause(self):
        self._cancel.set()

    def restore_local_backup(self, path, backup, expected_stamp):
        """Undo one completed pull without overwriting a later saved project."""
        self._client()
        target, source = Path(path), Path(backup)
        record = next((j.get("localUpdate", {}) for j in self._bucket()["jobs"]
                       if j.get("localUpdate", {}).get("backupPath") == str(source)), None)
        if not record or source.parent != self.root / "backups":
            raise ValueError("The saved backup is no longer available.")
        operation = {"id": str(uuid.uuid4()), "state": "running"}
        self._undo_restore = operation
        def action():
            temporary = None
            restored = False
            try:
                if file_stamp(target) != expected_stamp:
                    raise ValueError("The local project changed. Keep the backup and review both files.")
                digest = hashlib.sha256()
                with source.open("rb") as incoming, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                    temporary = Path(output.name)
                    while chunk := incoming.read(gallery_validation.CHUNK_BYTES):
                        self._client()
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                self._client()
                if digest.hexdigest() != record.get("sha256") or file_stamp(target) != expected_stamp:
                    raise ValueError("The backup or local project changed. No file was restored.")
                os.replace(temporary, target)
                restored = True
                record["undoRestored"] = True
                self._save()
                with self._lock:
                    operation.update(state="restored")
            except Exception as exc:
                with self._lock:
                    # Once replace succeeded, retrying would overwrite a restored
                    # file; a later journal failure does not undo that success.
                    operation.update(state="restored" if restored else "failed", message=friendly_error(exc),
                                     backupMissing=not source.is_file())
                raise
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
        self._launch(action)
        return operation["id"]

    @contextmanager
    def local_use(self, job_id):
        """Protect a downloaded source for the complete native import lifetime."""
        guard = _locked_sidecar(self.root / "local-use.lock", blocking=False)
        try:
            guard.__enter__()
        except OSError:
            raise ValueError("Another LichtFeld Studio window is opening or clearing a download. Try again when it finishes.") from None
        try:
            self._client()
            try:
                current = self._journal_digest()
            except (OSError, ValueError):
                self._journal_problem = True
                raise ValueError(JOURNAL_RECOVERY_MESSAGE) from None
            if current != self._disk_digest:
                self._stale = True
                raise ValueError(JOURNAL_CHANGED_MESSAGE)
            job = self._job(job_id)
            if (job.get("kind") != "download" or job["status"] != "completed" or
                    job.get("cleanupPending") or job.get("retired") or not Path(job["path"]).is_file()):
                raise ValueError("This download is no longer ready to open. Refresh or download it again.")
            yield
        finally:
            guard.__exit__(None, None, None)

    def _cleanup_references(self):
        references = {}
        for bucket in self._data["accounts"].values():
            for job in bucket["jobs"]:
                for value, backup in ((job.get("path"), False),
                        (job.get("preparation"), False),
                        (job.get("stagedImport", {}).get("path"), False),
                        (job.get("localUpdate", {}).get("backupPath"), True)):
                    if value:
                        references.setdefault(Path(value).resolve(), []).append((job, backup))
        return references

    def _cleanup_paths(self, job, references):
        paths = []
        root = self.root.absolute()

        def owned(value, directory, identifier=None):
            path = Path(value).absolute()
            if path.parent != directory or path.suffix not in (".licht",):
                raise ValueError("A transfer file is outside its saved temporary folder. Keep it for recovery.")
            try:
                uuid.UUID(path.stem)
            except ValueError:
                raise ValueError("A transfer file has an unexpected name. Keep it for recovery.") from None
            if identifier is not None and path.stem != identifier:
                raise ValueError("A transfer file no longer matches its saved record. Keep it for recovery.")
            if directory.is_symlink() or getattr(directory, "is_junction", lambda: False)() or path.is_symlink():
                raise ValueError("A transfer file was redirected. Keep it for recovery.")
            paths.append(path)

        if job.get("kind") == "download":
            owned(job["path"], root / "downloads", job["id"])
            partial = Path(job['path']).with_name('.' + Path(job['path']).name + '.part')
            if partial.exists() or partial.is_symlink():
                if partial.is_symlink() or not partial.is_file():
                    raise ValueError('A partial download was redirected. Keep it for recovery.')
                paths.append(partial)
            stage = job.get("stagedImport", {})
            if stage.get("path"):
                if Path(stage["path"]).suffix == ".scene":
                    directory = gallery_preparation.staging_path(root / "imports", stage["path"])
                    if directory.stem != stage["id"]:
                        raise ValueError("The import preparation no longer matches its record. Keep it for recovery.")
                    paths.extend(gallery_preparation.staging_files(root / "imports", directory))
                else:
                    owned(stage["path"], root / "imports", stage["id"])
        elif job.get("ownedExport"):
            if Path(job["path"]).suffix not in (".licht",):
                raise ValueError("The saved export is not a prepared gallery upload. Keep it for recovery.")
            owned(job["path"], root)
            if job.get("preparation"):
                directory = gallery_preparation.staging_path(root, job["preparation"])
                if Path(job["path"]).absolute() != directory.with_suffix(".licht"):
                    raise ValueError("Scene preparation no longer matches its transfer. Keep it for recovery.")
                paths.extend(gallery_preparation.staging_files(root, directory))
        for path in paths:
            for other, backup in references.get(path.resolve(), []):
                if backup:
                    raise ValueError("A recovery copy still uses this file. Keep it for recovery.")
                if other is not job:
                    raise ValueError("Another transfer still uses this temporary file. Keep it for recovery.")
        return paths

    @staticmethod
    def _unlink_temporary(path):
        if os.name == "nt":
            # The directory and file reparse checks are performed by _cleanup_paths.
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink(missing_ok=True)
            return
        try:
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return
        try:
            try:
                if path.is_dir():
                    os.rmdir(path.name, dir_fd=directory)
                else:
                    os.unlink(path.name, dir_fd=directory)
            except FileNotFoundError:
                pass
        finally:
            os.close(directory)

    def clear_finished(self, job_ids):
        """Clear reviewed account-local history and owned files, retaining recovery."""
        self._client()
        bucket = self._bucket()
        identifiers = tuple(dict.fromkeys(job_ids))

        def action():
            guard = _locked_sidecar(self.root / "local-use.lock", blocking=False)
            try:
                guard.__enter__()
            except OSError:
                raise ValueError("A LichtFeld Studio window is using a downloaded scene. Clear transfers after it finishes.") from None
            try:
                available = {j["id"]: j for j in bucket["jobs"]}
                if any(identifier not in available for identifier in identifiers):
                    raise ValueError("The transfer list changed. Refresh and review it before clearing.")
                jobs = [available[identifier] for identifier in identifiers]
                if any(j["status"] not in ("completed", "canceled") or j.get("retired") for j in jobs):
                    raise ValueError("Only finished transfers can be cleared. Refresh and review the list.")
                references = self._cleanup_references()
                planned = [(job, self._cleanup_paths(job, references)) for job in jobs]
                replacements = {}
                for job in jobs:
                    if job.get("localUpdate", {}).get("backupPath"):
                        kept = {key: copy.deepcopy(job[key]) for key in
                            ("id", "project", "kind", "status", "sceneId", "localUpdate") if key in job}
                        kept.update(retired=True, path="", checkpoint=None, completed=0, total=0,
                            metadata={"title": job["metadata"]["title"]}, message="Transfer cleared. Recovery copy kept.")
                        if "result" in job:
                            kept["result"] = {key: job["result"][key] for key in ("id", "contentRevision", "metadataRevision", "title") if key in job["result"]}
                        replacements[job["id"]] = kept
                self._client()
                with self._lock:
                    for job in jobs:
                        job.update(cleanupPending=True, message="Clearing temporary transfer files…")
                self._save()  # Persist the whole intent before unlinking any source.
                for job, paths in planned:
                    self._client()
                    for path in paths:
                        self._unlink_temporary(path)
                with self._lock:
                    previous = bucket["jobs"]
                    cleared = set(identifiers)
                    bucket["jobs"] = [replacements.get(j["id"], j) for j in previous
                        if j["id"] not in cleared or j["id"] in replacements]
                try:
                    self._save()
                except Exception:
                    with self._lock:
                        bucket["jobs"] = previous
                    raise
                with self._lock:
                    self.message = "Finished transfers cleared. Project links and recovery copies kept."
            finally:
                guard.__exit__(None, None, None)
        self._launch(action)

    def _retire_export(self, job):
        # Never remove a user-supplied file; only snapshots created inside our spool.
        if not job.get("ownedExport"):
            return
        try:
            for path in self._cleanup_paths(job, self._cleanup_references()):
                self._unlink_temporary(path)
        except (ValueError, OSError):
            with self._lock:
                job.update(cleanupPending=True, message=("Upload complete." if job["status"] == "completed" else "Upload discarded.")
                    + " Some temporary files were kept. Open the recovery folder to review them.")
                self.message = job["message"]
            self._save()

    def discard(self, job_id):
        client = self._client()
        job = self._job(job_id)
        if job["status"] == "completed":
            raise ValueError("This transfer is complete. Use Remove from gallery to delete it.")

        def action():
            if (job.get("checkpoint") or {}).get("uploadId"):
                client.cancel_upload(job["checkpoint"]["uploadId"])
            with self._lock:
                job.update(status="canceled", message="Discarded")
            self._save()
            self._retire_export(job)
        self._launch(action)

    def unlink(self, project_id):
        def check_pending():
            if any(j["project"] == project_id and j["status"] not in ("completed", "canceled") for j in self._bucket()["jobs"]):
                raise ValueError("Discard the pending transfer before unlinking.")
        with self._lock:
            self._client()
            check_pending()
        def action():
            self._client()
            with self._lock:
                bucket = self._bucket()
                check_pending()
                bucket["links"].pop(project_id, None)
            self._save()
        self._launch_metadata(action)

    def edit(self, scene_id, baseline, metadata, *, commit_uuid=None, content_stamp=None):
        baseline = copy.deepcopy(baseline)
        metadata = copy.deepcopy(metadata)
        def action():
            client = self._client()
            bucket = self._bucket()
            scene = client.update(scene_id, domain_tokens(baseline), **metadata)
            with self._lock:
                self.scenes = [scene if s["id"] == scene_id else s for s in self.scenes]
                self.message = "Gallery details saved."
                self._completion = {"id": str(uuid.uuid4()), "kind": "publish", "scene": copy.deepcopy(scene)}
                for link in bucket["links"].values():
                    if link["sceneId"] == scene_id:
                        # A metadata exchange does not exchange remote geometry.
                        tokens = domain_tokens(scene)
                        if link.get("contentRevision"):
                            tokens["contentRevision"] = link["contentRevision"]
                        link.update(**tokens, metadata=copy.deepcopy(scene),
                                    sharedFields=shared_fields(scene), exchangedAt=time.time(), checkedAt=time.time())
                        if commit_uuid:
                            link["commitUuid"] = commit_uuid
                        if content_stamp:
                            link["contentStamp"] = content_stamp
            self._save()
        self._launch_metadata(action)


    def remove(self, scene_id, baseline):
        baseline = copy.deepcopy(baseline)
        title = next((s.get("title", "") for s in self.scenes if s["id"] == scene_id), "")
        def action():
            client = self._client()
            bucket = self._bucket()
            if any(j["metadata"].get("replaceSceneId") == scene_id and j["status"] not in ("completed", "canceled") for j in bucket["jobs"]):
                raise ValueError("Discard the unfinished replacement before removing this gallery item.")
            guard = domain_tokens(baseline)
            try:
                client.delete(scene_id, guard)
            except PortalHTTPError as exc:
                if exc.status != 404:
                    raise
            with self._lock:
                self.scenes = [s for s in self.scenes if s["id"] != scene_id]
                self._list_etag = None
                self.message = "Removed from gallery. Local projects are unchanged."
                self._completion = {"id": str(uuid.uuid4()), "kind": "remove", "title": title}
                for link in bucket["links"].values():
                    if link["sceneId"] == scene_id:
                        link["remoteDeleted"] = True
            self._save()
        self._launch_metadata(action)


_service = None


def get_gallery_sync():
    global _service
    if _service is None:
        from .asset_index import resolve_asset_manager_storage_path
        from .portal_account import get_portal_account_service
        _service = GallerySync(get_portal_account_service(), resolve_asset_manager_storage_path() / "gallery")
    return _service
