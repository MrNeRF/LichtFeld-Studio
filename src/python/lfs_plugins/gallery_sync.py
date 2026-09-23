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
    GalleryProcessingTimeout, GalleryTransferInvalid, PROCESSING_TIMEOUT, DEFAULT_MAX_FILE_BYTES, disk_preflight, domain_tokens, UNSUPPORTED_PORTAL, _fingerprint)
from .portal_retry import transfer_attempts, is_transient
from .portal_security import redact, safe_filename
from .credential_storage import FileBackend
from .project_identity import ProjectPathIdentity
from . import gallery_validation, gallery_preparation
from .gallery_logging import failure as log_failure, safe_url, safe_text, stage as log_stage


MAX_JOURNAL_BYTES = 32 * 1024 * 1024


def shared_fields(scene):
    """Fields shared with LichtFeld Studio; cover, highlights and broad revision excluded."""
    return copy.deepcopy({key: scene.get(key, {} if key == "viewerSettings" else "")
                          for key in ("title", "description", "viewerSettings")})


def exchange_link(scene, commit_uuid=""):
    now = time.time()
    return {"sceneId": scene["id"], **domain_tokens(scene),
            "acknowledgedPresentationRevision": scene.get("presentationRevision", ""),
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

    require(isinstance(data, dict) and type(data.get("version")) is int and data["version"] in (2, 3)
        and isinstance(data.get("accounts"), dict))
    for bucket in data["accounts"].values():
        require(isinstance(bucket, dict) and isinstance(bucket.get("links"), dict)
            and isinstance(bucket.get("jobs"), list))
        require(isinstance(bucket.get("unlinkedProjects", []), list)
            and all(isinstance(project_id, str) and project_id for project_id in bucket.get("unlinkedProjects", [])))
        intents = bucket.get("handoffIntents", {})
        require(isinstance(intents, dict) and (not intents or data["version"] == 3))
        for identifier, handoff in intents.items():
            require(isinstance(handoff, dict) and handoff.get("id") == identifier)
            require(all(isinstance(handoff.get(key), str) and handoff[key] for key in
                ("id", "oldProject", "newProject", "sceneId", "origin", "owner", "commitUuid", "fileUuid")))
            require(handoff["oldProject"] != handoff["newProject"] and handoff.get("state") == "pending")
            guards(handoff)
            require(set(handoff.get("baseRevisions", {})) == {"content", "metadata"})
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
            for key in ("serverProcessing", "packaged", "needsAttention", "retryable", "requiresPreparation", "preparedRemoved"):
                require(key not in job or type(job[key]) is bool)
            if "preparation" in job:
                require(isinstance(job["preparation"], str) and job.get("kind", "upload") == "upload"
                        and job.get("ownedExport") is True and Path(job["path"]).suffix == ".licht")
            require(all(type(job.get(key)) is int and 0 <= job[key] <= 2**63-1 for key in ("completed", "total")))
            require(isinstance(job.get("metadata"), dict) and isinstance(job["metadata"].get("title"), str))
            optional_text(job["metadata"], ("description", "visibility", "replaceSceneId"))
            optional_text(job, ("failureReason", "previewPng", "destinationPath", "downloadProject"))
            # Older writers saved full viewport PNGs here. The whole journal is
            # already byte-bounded; one oversized cover must not hide every link
            # and upload key. Validate/resize its image when preparing that job.
            if "handoff" in job:
                handoff = job["handoff"]
                require(data["version"] == 3 and isinstance(handoff, dict))
                require(all(isinstance(handoff.get(key), str) and handoff[key] for key in
                    ("oldProject", "newProject", "sceneId", "origin", "owner", "commitUuid", "fileUuid")))
                require(handoff["oldProject"] != handoff["newProject"] == job["project"])
                require(handoff.get("state") in ("pending", "completed"))
                guards(handoff)
                require(set(handoff.get("baseRevisions", {})) == {"content", "metadata"})
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
    import lichtfeld as lf

    if isinstance(exc, getattr(lf, "Error", ())):
        return redact(exc.user_message)
    status = getattr(exc, "status", getattr(exc, "code", None))
    if isinstance(exc, PortalHTTPError):
        if exc.error == "gallery_relink_required":
            return "Use the Portal button to approve gallery access. Your local work is safe."
        if exc.status == 400 and exc.error in (
                "Invalid portable LichtFeld project.", "Project checksum failed.",
                "Embedded project asset checksum failed."):
            return "The downloaded file is damaged or was changed on the portal."
    if isinstance(status, int):
        key = {401: "authorization_expired", 403: "access", 404: "not_found",
               409: "http_conflict", 413: "too_large", 429: "portal_busy"}.get(status)
        if key is None and 500 <= status <= 599:
            key = "server"
        if key:
            return "projects.gallery.error." + key
    if isinstance(exc, (ValueError, PortalProtocolError)):
        return redact(exc)
    return "The connection or local storage was interrupted. Check your connection and disk space, then resume."


def file_stamp(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, str(Path(path).resolve())]


def _project_uuid(path):
    import lichtfeld as lf
    return str(lf.io.inspect_project_card(path).project_uuid)


def _require_project(path, project_id):
    if _project_uuid(path) != project_id:
        raise ValueError(f"The project identity changed at {path}. Refresh Projects and try again.")


class GallerySync:
    def __init__(self, account, root):
        self.account = account
        self.root = Path(root).resolve()
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
        self._reserved_bytes = None
        self._hdr_backgrounds = None
        self._change_sequence = None
        self._completion = None
        self._action_failure = None
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
                        update = job.get("localUpdate", {})
                        if recover_interrupted and not job.get("retired") and update.get("state") in ("preparing", "ready", "applying"):
                            update["interrupted"] = True
                            job["message"] = "Applying Gallery changes was interrupted. Review the recovery copy before continuing."
                self._data = data
            self._disk_digest = digest
            self._journal_problem = self._stale = False
        if recover_interrupted:
            pending_cleanup = [job for bucket in self._data["accounts"].values() for job in bucket["jobs"]
                               if job["status"] in ("error", "completed", "canceled") and job.get("ownedExport")
                               and (not job.get("preparedRemoved") or job.get("cleanupPending"))]
            for job in pending_cleanup:
                if job["status"] == "error":
                    job.update(requiresPreparation=True, retryable=True)
            if pending_cleanup:
                self._save()
                for job in pending_cleanup:
                    self._retire_export(job)

    def _save(self, *, project_checks=()):
        with self._persist_lock:
            with self._lock:
                self._check_journal_ready()
                self._prune_jobs()
                _validate_journal(self._data)
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
                for check in project_checks:
                    path_identity, project_id = check[:2]
                    path_identity.validate()
                    _require_project(path_identity.path, project_id)
                    path_identity.validate()
                    if len(check) > 2 and file_stamp(path_identity.path) != check[2]:
                        raise ValueError("The local project changed. Review it before updating.")
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
                "reservedBytes": self._reserved_bytes if same else None,
                "hdrBackgrounds": self._hdr_backgrounds if same else None,
                "changeSequence": self._change_sequence if same else None,
                "established": bool(same and self._owner and self._checked_at),
                "completion": self._completion if same else None,
                "actionFailure": self._action_failure if self._action_failure and self._action_failure["identity"] == self.identity() else None,
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
                    relink_required = (
                        isinstance(exc, PortalHTTPError)
                        and exc.error == "gallery_relink_required"
                    )
                    if relink_required:
                        log_stage("relink_required", operation=operation or "transfer")
                    else:
                        log_failure("worker", exc, operation=operation or "transfer")
                    with self._lock:
                        self._relink_identity = identity if relink_required else None
                        self.message = friendly_error(exc)
                        self._action_failure = dict(id=str(uuid.uuid4()), identity=identity, message=self.message)
                finally:
                    with self._lock:
                        self.version += 1

            self._thread = threading.Thread(target=worker, daemon=True, name="GallerySync")
            self._operation = operation
            try:
                self._thread.start()
            except Exception as exc:
                log_failure("worker_start", exc, operation=operation or "transfer")
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

    def _report_incomplete_account(self, snap):
        if snap.linking or self.account.busy:
            return
        with self._lock:
            self.message = "projects.gallery.error.account_loading"
            self._action_failure = dict(id=str(uuid.uuid4()), identity=self.identity(), message=self.message)
            self._refresh_ok = False
            # Publish direct returns too; workers also advance version on completion.
            self.version += 1

    def refresh(self, *, force=False):
        if self.busy and self._operation == "refresh":
            return
        if not force and self._relink_identity == self.identity():
            return
        if self._unsupported_identity == self.identity():
            return
        snap = self.account.snapshot()
        if snap.signed_in and (not snap.email or not snap.connected_since):
            self._report_incomplete_account(snap)
            return
        self._unsupported_identity = None
        self._refresh_ok = False
        def action():
            snap = self.account.snapshot()
            if not snap.signed_in:
                raise ValueError("Sign in with your LichtFeld account first.")
            if not snap.email or not snap.connected_since:
                self._report_incomplete_account(snap)
                return
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
                self._session, self._owner, self._origin = session, capabilities["id"], origin
                bucket = self._bucket()
                linked_scene_ids = {
                    link.get("sceneId") for link in bucket["links"].values()
                    if isinstance(link, dict) and link.get("sceneId") and not link.get("remoteDeleted")
                }
                cache_key = hashlib.sha256(json.dumps([origin, self._owner]).encode()).hexdigest()
                cache_file = FileBackend(self.root / ("listing-" + cache_key + ".json"))
                cache = {}
                if not same:
                    try:
                        raw = cache_file.read()
                        cache = json.loads(raw) if raw and len(raw) <= MAX_JOURNAL_BYTES else {}
                        if (not isinstance(cache, dict) or cache.get("owner") != self._owner
                                or cache.get("origin") != origin or not isinstance(cache.get("scenes"), list)):
                            cache = {}
                    except (OSError, ValueError):
                        cache = {}
                    self._quota_bytes = self._used_bytes = self._reserved_bytes = None
                    self._source_formats = []
                cached = self.scenes if same else copy.deepcopy(cache.get("scenes", []))
                sequence = self._change_sequence if same else cache.get("changeSequence")
                if not same:
                    etag = cache.get("ownerEtag") if type(sequence) is int else None
                if not same:
                    self.scenes = cached
                    self._checked_at = cache.get("checkedAt", 0)
                self.version += 1
            scenes = None
            if type(sequence) is int:
                try:
                    events = client.changes_since(sequence)
                    by_id = {scene["id"]: scene for scene in cached}
                    for event in events:
                        if event["type"] == "delete":
                            by_id.pop(event["sceneId"], None)
                        else:
                            by_id[event["sceneId"]] = event["scene"]
                    scenes = list(by_id.values())
                except PortalHTTPError as exc:
                    if not (exc.status in (404, 405) or exc.status == 409 and exc.error == "resync_required"):
                        raise
                    if exc.status in (404, 405) and etag:
                        # Only a previous changeSequence response proves that
                        # this ETag covers the whole owner listing.
                        scenes = client.list_scenes(etag=etag, owner_wide=True)
                        if scenes is None:
                            scenes = cached
                            client.change_sequence = sequence
            if scenes is None:
                # This also handles an old portal whose first-page ETag does
                # not describe the later pages.
                scenes = client.list_scenes()
            # A stale incremental cache can legitimately have no events while
            # still missing a scene that is referenced by the local journal.
            # Re-walk the owner listing before declaring those projects removed.
            listed_scene_ids = {scene.get("id") for scene in scenes if isinstance(scene, dict)}
            if linked_scene_ids - listed_scene_ids:
                scenes = client.list_scenes()
            # The changes feed may contain compact scene summaries. Resolve
            # them before comparing or applying authored view settings. This
            # also repairs a cache written by an earlier client.
            scenes = [client.scene(scene["id"]) if scene.get("status") == "ready"
                      and "viewerSettings" not in scene else scene for scene in scenes]
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
                self._reserved_bytes = capabilities.get("reservedBytes")
                self._hdr_backgrounds = capabilities.get("hdrBackgrounds")
                self._change_sequence = getattr(client, "change_sequence", None)
                version = capabilities.get("revisionDomains", 0)
                self._revision_domains = version if type(version) is int else 0
                self._list_etag = getattr(client, "list_etag", None) or (etag if self._change_sequence == sequence else None)
                self._checked_at = time.time()
                if scenes is not None:
                    self.scenes = scenes
                bucket = self._bucket()
                recovered = self._origin_publication_links(self.scenes, bucket["links"], bucket.get("unlinkedProjects", ()))
                for link in bucket["links"].values():
                    link["checkedAt"] = self._checked_at
                if recovered:
                    self._save()
                self._refresh_ok = True
                self._relink_identity = None
                self._unsupported_identity = None
                self.message = "Gallery is up to date."
                cache = dict(scenes=copy.deepcopy(self.scenes), checkedAt=self._checked_at,
                             changeSequence=self._change_sequence, ownerEtag=self._list_etag,
                             origin=origin, owner=self._owner)
            cache_file.write(json.dumps(cache, allow_nan=False).encode())
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
            except OSError as exc:
                log_failure("poster_cache_trim", exc)
                # Poster I/O must not fail a successful scene listing.
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
            except (OSError, ValueError, PortalHTTPError, PortalProtocolError) as exc:
                log_failure("poster_cache", exc, scene_id=scene["id"])
                # A missing/malformed poster does not prevent scene synchronization.
                with self._lock:
                    entry = self._poster_entries.pop(scene["id"], None)
                    if entry:
                        Path(entry["path"]).unlink(missing_ok=True)

    @staticmethod
    def _origin_publication_links(scenes, links, unlinked_projects=()):
        """Recover unambiguous local-project links from an owner listing."""
        candidates = {}
        for scene in scenes:
            project_id = scene.get("originProjectUuid")
            if scene.get("status") != "ready" or not isinstance(project_id, str) or not project_id:
                continue
            candidates.setdefault(project_id, []).append(scene)

        recovered = {}
        for project_id, matches in candidates.items():
            if project_id in links or project_id in unlinked_projects or len(matches) != 1:
                continue
            scene = matches[0]
            if not all(isinstance(scene.get(key), str) and scene[key]
                       for key in ("id", "contentRevision", "metadataRevision")):
                continue
            link = exchange_link(scene, scene.get("originCommitUuid") or "")
            source_format = scene.get("sourceFormat")
            if isinstance(source_format, str) and source_format:
                link["uploadFormat"] = source_format
            links[project_id] = link
            recovered[project_id] = link
        return recovered

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
            if any(metadata.get("_handoff", {}).get("id") != intent["id"]
                   and (project_id in (intent["oldProject"], intent["newProject"])
                        or metadata.get("replaceSceneId") == intent["sceneId"])
                   for intent in self._bucket().get("handoffIntents", {}).values()):
                raise ValueError("Review the saved replacement choice before publishing either project again.")
            if any(j.get("handoff", {}).get("state") == "pending" and j["status"] not in ("completed", "canceled")
                   and (project_id in (j["handoff"]["oldProject"], j["handoff"]["newProject"])
                        or metadata.get("replaceSceneId") == j["handoff"]["sceneId"]) for j in jobs):
                raise ValueError("Finish the replacement handoff before publishing either project again.")
            if any(j["project"] == project_id and j["status"] not in ("completed", "canceled") for j in jobs):
                raise ValueError("This project already has a transfer. Resume or discard it first.")
            job = {"id": str(uuid.uuid4()), "project": project_id, "path": str(export_path),
                "kind": "upload",
                "ownedExport": owned_export,
                "metadata": copy.deepcopy(metadata), "checkpoint": None, "status": "queued",
                "completed": 0, "total": 0 if preparation is not None else Path(export_path).stat().st_size,
                "message": "Ready to prepare" if preparation is not None else "Ready to upload"}
            log_stage("upload_queued", project_id=project_id, path=export_path,
                      size=job["total"], format=Path(export_path).suffix.lower().lstrip("."),
                      account_origin=safe_url(self.account.base_url))
            job["commitUuid"] = job["metadata"].pop("_commitUuid", "")
            job["uploadFormat"] = job["metadata"].pop("_uploadFormat", "studio")
            job["contentStamp"] = job["metadata"].pop("_contentStamp", "")
            job["publishAsNew"] = job["metadata"].pop("_publishAsNew", False)
            preview = job["metadata"].pop("_previewPng", None)
            if preview is not None:
                import base64
                job["previewPng"] = base64.b64encode(gallery_preparation.publication_preview(
                    base64.b64decode(preview, validate=True))).decode("ascii")
            handoff = job["metadata"].pop("_handoff", None)
            target = job["metadata"].get("replaceSceneId")
            if target and not handoff and (linked or {}).get("sceneId") != target and any(key != project_id and value["sceneId"] == target
                    for key, value in self._bucket()["links"].items()):
                raise ValueError("The published scene belongs to the previous project. Review the replacement first.")
            if handoff:
                job["handoff"] = copy.deepcopy(handoff)
                job["metadata"]["originFileUuid"] = handoff["fileUuid"]
                self._check_handoff(job)
            job["metadata"].setdefault("originProjectUuid", project_id)
            if job["commitUuid"]:
                job["metadata"].setdefault("originCommitUuid", job["commitUuid"])
            job["metadata"].setdefault("clientMutationId", job["id"])
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
                previous_version = self._data["version"]
                if handoff:
                    self._data["version"] = 3
                try:
                    self._save()
                except Exception:
                    jobs.remove(job)
                    self._data["version"] = previous_version
                    raise
            finally:
                guard.__exit__(None, None, None)
        # The ownership handoff is now durable. A thread-start failure must keep
        # the queued snapshot for resume instead of making the panel delete it.
        try:
            self.resume(job["id"])
        except Exception as exc:
            log_failure("upload_launch", exc, project_id=project_id)
            with self._lock:
                job.update(status="paused", message="Scene prepared. Resume when ready to upload.")
                self.message = job["message"]
                self.version += 1
        return job["id"]

    def _check_handoff(self, job):
        handoff = job.get("handoff")
        if not handoff:
            return
        bucket = self._bucket()
        if handoff.get("id") and bucket.get("handoffIntents", {}).get(handoff["id"]) != handoff:
            raise ValueError("The saved replacement choice changed. Review the replacement again.")
        old = bucket["links"].get(handoff.get("oldProject"), {})
        tokens = {name: old.get(name + "Revision") for name in ("content", "metadata")}
        if (handoff.get("origin") != self._origin or handoff.get("owner") != self._owner
                or handoff.get("newProject") != job["project"] or handoff.get("oldProject") == job["project"]
                or old.get("sceneId") != handoff.get("sceneId") or tokens != handoff.get("oldLinkRevisions", handoff.get("baseRevisions"))
                or job["metadata"].get("replaceSceneId") != handoff.get("sceneId")
                or job["metadata"].get("baseRevisions") != handoff.get("baseRevisions")
                or not handoff.get("commitUuid") or handoff["commitUuid"] != job.get("commitUuid")
                or not handoff.get("fileUuid") or job["project"] in bucket["links"]):
            raise ValueError("The previous gallery link changed. Review the replacement again.")
        if any(other["id"] != job["id"] and other["status"] not in ("completed", "canceled")
                and (other.get("project") in (handoff["oldProject"], job["project"])
                    or other.get("metadata", {}).get("replaceSceneId") == handoff["sceneId"])
                for other in bucket["jobs"]):
            raise ValueError("Finish the previous project's transfer before replacing its published scene.")

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
            job["destinationPath"] = str(final_path.resolve())
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
        except Exception as exc:
            log_failure("download_launch", exc, scene_id=scene.get("id", ""))
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
            if job.get("requiresPreparation"):
                raise ValueError("The prepared upload was removed after failure. Review the project to start a fresh upload.")
            if job.get("retryable") is False:
                raise ValueError(job["message"])
            bucket = self._bucket()
            identity = self.identity()
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
                    if job.get("kind") == "download" and job.get("serverProcessing"):
                        job["serverProcessing"] = False
                        if job["message"] == "Preparing viewing copy":
                            job["message"] = "Downloading"
                            self.message = job["message"]
                    self.version += 1

            def processing(state):
                label = {"queued": "Waiting for the portal", "assembling": "Assembling upload",
                    "validating": "Checking scene", "publishing": "Publishing scene",
                    "preparing_download": "Preparing viewing copy"}[state["stage"]]
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
                cover_preview = (job.get("previewPng") if job["metadata"].get("useEmbeddedPreview")
                    and job["metadata"].get("replaceSceneId") and not job.get("handoff") else None)
                linked = bucket["links"].get(job["project"])
                with self._lock:
                    self._check_handoff(job)
                    if job.get("handoff") and scene["id"] != job["handoff"]["sceneId"]:
                        raise ValueError("The Gallery returned a different replacement scene. The previous link was kept.")
                    previous_links = copy.deepcopy(bucket["links"])
                    previous_unlinked = list(bucket.get("unlinkedProjects", ()))
                    previous_job = copy.deepcopy(job)
                    previous_scenes = copy.deepcopy(self.scenes)
                    previous_intents = copy.deepcopy(bucket.get("handoffIntents", {}))
                    previous_undo = []
                    self._completion = {"id": str(uuid.uuid4()), "kind": "publish", "scene": copy.deepcopy(scene)}
                    bucket["links"][job["project"]] = exchange_link(scene, job.get("commitUuid", ""))
                    bucket["unlinkedProjects"] = [project_id for project_id in previous_unlinked
                                                   if project_id != job["project"]]
                    bucket["links"][job["project"]]["uploadFormat"] = job.get("uploadFormat", "studio")
                    bucket["links"][job["project"]]["contentStamp"] = job.get("contentStamp", "")
                    for history in bucket["jobs"]:
                        update = history.get("localUpdate", {})
                        if (history.get("project") == job["project"] and update.get("state") == "applied"
                                and not update.get("undoRestored") and update.get("appliedCommit") == job.get("commitUuid")
                                and tuple(update.get("appliedIdentity", ())) == self.identity()):
                            previous_undo.append((update, copy.deepcopy(update.get("appliedLink"))))
                            update["appliedLink"] = copy.deepcopy(bucket["links"][job["project"]])
                    job.update(status="completed", completed=job["total"], serverProcessing=False, message="Uploaded", result=scene)
                    job.pop("previewPng", None)
                    if job.get("handoff"):
                        bucket["links"].pop(job["handoff"]["oldProject"])
                        bucket.get("handoffIntents", {}).pop(job["handoff"].get("id"), None)
                        job["handoff"]["state"] = "completed"
                    self.scenes = [s for s in self.scenes if s["id"] != scene["id"]] + [scene]
                    self.message = "Upload complete. Review Story on portal after this content change." if job["metadata"].get("replaceSceneId") else "Upload complete."
                try:
                    self._save()
                except Exception:
                    with self._lock:
                        bucket["links"] = previous_links
                        bucket["unlinkedProjects"] = previous_unlinked
                        self.scenes = previous_scenes
                        bucket["handoffIntents"] = previous_intents
                        for update, applied_link in previous_undo:
                            update["appliedLink"] = applied_link
                        job.clear()
                        job.update(previous_job)
                        self._completion = None
                    raise
                self._retire_export(job)
                log_stage("link_saved", scene_id=scene["id"],
                          content_revision=scene.get("contentRevision", ""),
                          metadata_revision=scene.get("metadataRevision", ""),
                          project_id=job["project"])
                if cover_preview:
                    try:
                        if not linked or linked["sceneId"] != scene["id"]:
                            raise ValueError("The Gallery link changed. Check gallery before setting its cover.")
                        import base64
                        client.set_cover(scene["id"], scene, base64.b64decode(cover_preview, validate=True))
                        updated = client.scene(scene["id"])
                        with self._lock:
                            self.scenes = [updated if item["id"] == scene["id"] else item for item in self.scenes]
                            link = bucket["links"][job["project"]]
                            link["metadata"] = copy.deepcopy(updated)
                            link["acknowledgedPresentationRevision"] = updated.get("presentationRevision", "")
                            job["result"] = updated
                            self._completion["scene"] = copy.deepcopy(updated)
                        self._save()
                    except Exception as exc:
                        log_failure("cover_after_upload", exc, project_id=job["project"])
                        with self._lock:
                            self.message = friendly_error(exc)
                            self._action_failure = dict(id=str(uuid.uuid4()), identity=identity, message=self.message)

            try:
                self._check_handoff(job)
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
                    destination = Path(job["path"]).resolve()
                    if destination != staging.with_suffix(".licht") or destination.is_symlink():
                        raise ValueError("Scene preparation no longer matches its transfer. Keep it for recovery.")
                    nodes, total = gallery_preparation.read_staging(self.root, job["preparation"])
                    preparation_started = time.monotonic()
                    log_stage("preparation_start", node_count=len(nodes), payload_bytes=total,
                              staging_path=staging, project_id=job["project"])
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
                    if job.get("previewPng"):
                        import base64
                        gallery_preparation.attach_preview(source_path, base64.b64decode(job["previewPng"], validate=True), cancel=self._cancel)
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
                    staged_path = Path(job["path"])
                    log_stage("preparation_end", node_count=len(nodes),
                              payload_bytes=staged_path.stat().st_size, staging_path=staging,
                              elapsed_ms=f"{(time.monotonic() - preparation_started) * 1000:.1f}",
                              project_id=job["project"])
                    log_stage("export_staged", path=staged_path, bytes=staged_path.stat().st_size,
                              sha256=_fingerprint(staged_path), project_id=job["project"])
                    self._save()
                client.processing_deadline = job.get('processingDeadline')
                if job.get("kind") == "download":
                    def download_message(message):
                        with self._lock:
                            job['message'] = message
                            self.message = message
                            self.version += 1
                    scene = client.download(job["sceneId"], job["path"], on_progress=progress, cancel=self._cancel,
                        checkpoint=job.get('checkpoint'), on_checkpoint=checkpoint, on_message=download_message,
                        final_destination=self._download_destination(job), on_processing=processing)
                    self._client()
                    with self._lock:
                        self._completion = {"id": str(uuid.uuid4()), "kind": "download"}
                        job.update(status="completed", serverProcessing=False,
                                   sha256=(job.get("checkpoint") or {}).get("sha256", ""),
                                   downloadProject=_project_uuid(job["path"]), result=scene,
                                   message="Downloaded. Open as a new project when ready.")
                        self.message = job["message"]
                    log_stage("download_complete", scene_id=scene["id"],
                              bytes=job.get("total", 0), status="completed")
                    self._save()
                    return
                if not job.get("preparation"):
                    upload_path = Path(job["path"])
                    log_stage("export_staged", path=upload_path, bytes=upload_path.stat().st_size,
                              sha256=_fingerprint(upload_path), project_id=job["project"])
                result = client.upload(job["path"], job["metadata"], checkpoint=job["checkpoint"],
                    on_checkpoint=checkpoint, on_progress=progress, on_processing=processing, cancel=self._cancel)
                complete_upload(result)
            except Exception as exc:
                log_failure("transfer", exc, job_id=job.get("id", ""), kind=job.get("kind", "upload"))
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
                    job["failureReason"] = safe_text(f"{type(exc).__name__}: {exc}")
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
                        job["message"] = (("Paused. The pinned download will resume from its saved bytes." if ((job.get('checkpoint') or {}).get('representationId') or (job.get('checkpoint') or {}).get('apiRepresentationId'))
                            else "Paused. The download will restart from zero because the portal has no pinned representation.") if job.get("kind") == "download"
                            else "Paused. Uploaded parts will be reused when you resume.")
                    if isinstance(exc, GalleryProcessingTimeout):
                        job.update(status="paused", serverProcessing=True, needsAttention=True, message=str(exc))
                    if isinstance(exc, GalleryProcessingPaused):
                        job["message"] = "Stopped waiting. The portal may continue checking this upload. Resume to check its status."
                    elif isinstance(exc, GalleryTransferCanceled) and job.get("preparation") and not job.get("packaged"):
                        job["message"] = "Preparation paused. Resume to prepare the saved scene and upload it."
                    self.message = job["message"]
                    if job["status"] == "error" and job.get("ownedExport"):
                        job.update(requiresPreparation=True, retryable=True)
                self._save()
                if job.get("requiresPreparation"):
                    self._retire_export(job)
        def attempted():
            self._client()
            with self._lock:
                job['attempts'] = job.get('attempts', 0) + 1
            self._save()

        def run():
            with transfer_attempts(attempted, self._cancel):
                action()
        self._launch(run)

    def link_download(self, job_id, project_id, commit_uuid="", *, project_path=None, local_fields=None):
        self._client()
        local_fields = copy.deepcopy(local_fields)
        job, bucket = self._job(job_id), self._bucket()
        if job.get("kind") != "download" or job["status"] != "completed" or job.get("retired") or job.get("cleanupPending"):
            raise ValueError("Finish downloading this scene first.")
        linked = bucket["links"].get(project_id)
        if linked and linked["sceneId"] != job["result"]["id"]:
            raise ValueError("This project is linked to a different gallery item.")
        operation = str(uuid.uuid4())
        project_path = project_path or job.get("localUpdate", {}).get("path") or job.get("stagedImport", {}).get("projectPath")
        if not project_path:
            raise ValueError("The saved project path is missing. Prepare the download again before linking.")
        path_identity = ProjectPathIdentity.capture(project_path)
        def action():
            previous, previous_project = copy.deepcopy(bucket["links"].get(project_id)), job["project"]
            previous_unlinked = list(bucket.get("unlinkedProjects", ()))
            previous_update = copy.deepcopy(job.get("localUpdate"))
            try:
                self._client()
                with self._lock:
                    scene = job["result"]
                    bucket["links"][project_id] = exchange_link(scene, commit_uuid)
                    bucket["unlinkedProjects"] = [identifier for identifier in previous_unlinked
                                                   if identifier != project_id]
                    if local_fields is not None:
                        bucket["links"][project_id]["localFields"] = local_fields
                    if job.get("localUpdate", {}).get("backupPath"):
                        update = job["localUpdate"]
                        update.update(state="applied", appliedStamp=file_stamp(update["path"]), appliedCommit=commit_uuid,
                                      appliedIdentity=list(self.identity()), appliedLink=copy.deepcopy(bucket["links"][project_id]))
                    else:
                        bucket["links"][project_id]["viewingCopy"] = True
                    job["project"] = project_id
                    job["linkOperation"] = {"id": operation, "state": "ready"}
                with self._supersede_failed_local_updates(job):
                    self._save(project_checks=((path_identity, project_id),))
            except Exception as exc:
                log_failure("link_saved", exc, project_id=project_id, operation_id=operation)
                with self._lock:
                    if previous is None:
                        bucket["links"].pop(project_id, None)
                    else:
                        bucket["links"][project_id] = previous
                    bucket["unlinkedProjects"] = previous_unlinked
                    job["project"] = previous_project
                    if previous_update is not None:
                        job["localUpdate"] = previous_update
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
        source_identity = ProjectPathIdentity.capture(project_path)
        backup = self.root / "backups" / (update_id + ".licht")
        backup_identity = ProjectPathIdentity.capture(backup)
        record = {"id": update_id, "state": "preparing", "project": project_id,
            "path": str(project_path), "sourceStamp": list(expected_stamp), "previousLink": copy.deepcopy(linked)}

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
                if backup.exists():
                    raise ValueError("The recovery copy destination already exists. Try again.")
                disk_preflight([(backup, Path(project_path).stat().st_size),
                                (project_path, expected_stamp[2] + job['total'])])
                directory.mkdir(mode=0o700, exist_ok=True)
                if file_stamp(project_path) != expected_stamp:
                    raise ValueError("The local project changed. Review it before updating.")
                source_identity.validate()
                _require_project(project_path, project_id)
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
                source_identity.validate()
                _require_project(project_path, project_id)
                backup_identity.validate()
                os.replace(temporary, backup)
                temporary = None
                with self._lock:
                    record.update(state="ready", backupPath=str(backup), sha256=digest.hexdigest())
            except Exception as exc:
                log_failure("local_backup", exc, project_id=project_id, job_id=job_id)
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
        source_identity = ProjectPathIdentity.capture(job["path"])
        source_project = job.get("downloadProject") or _project_uuid(job["path"])
        from .asset_index import resolve_default_asset_directory
        project_path = (Path(job["destination"]) if job.get("destination") else
                        resolve_default_asset_directory() / ("Gallery-" + identifier + ".licht"))
        destination_identity = ProjectPathIdentity.capture(project_path)

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
                log_failure("download_staging_cleanup", exc, job_id=job["id"])
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
                if project_path.exists():
                    raise ValueError("The destination already exists. Choose another file name.")
                import lichtfeld as lf
                self._client()
                # A download is a new project each time, even when the same
                # representation is downloaded twice on this machine.
                source_identity.validate()
                _require_project(job["path"], source_project)
                destination_identity.validate()
                planned_destination = Path(job.get("destinationPath", str(project_path.resolve())))
                if ((job.get("destination") and planned_destination != project_path.resolve()) or
                        planned_destination.parent != project_path.resolve().parent):
                    raise ValueError("The download destination path changed. Choose it again.")
                restored = lf.io.restore_save(job["path"], 1, project_path)
                retained_project = ProjectPathIdentity.capture(project_path)
                with self._lock:
                    record["projectPath"] = str(project_path)
                    record["projectId"] = str(restored.project_uuid)
                    record["projectStamp"] = file_stamp(project_path)
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
                log_failure("download_staging", exc, job_id=job["id"])
                if retained_project is not None:
                    try:
                        retained_project.validate()
                        retained_project.path.unlink(missing_ok=True)
                    except (OSError, ValueError) as cleanup_error:
                        log_failure("download_staging_cleanup", cleanup_error, job_id=job["id"])
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
        target_identity = ProjectPathIdentity.capture(target)
        backup_identity = ProjectPathIdentity.capture(source)
        record = next((j.get("localUpdate", {}) for j in self._bucket()["jobs"]
                       if j.get("localUpdate", {}).get("backupPath") and
                       Path(j["localUpdate"]["backupPath"]).resolve() == source.resolve()), None)
        if not record or source.resolve().parent != (self.root / "backups").resolve():
            raise ValueError("The saved backup is no longer available.")
        if record.get("appliedIdentity") and tuple(record["appliedIdentity"]) != self.identity():
            raise ValueError("The account changed. Keep the backup for recovery.")
        if record.get("appliedLink"):
            current = self._bucket()["links"].get(record["project"], {})
            if not same_undo_link(current, record["appliedLink"]):
                raise ValueError("The Gallery link changed. Keep the backup for recovery.")
        operation = {"id": str(uuid.uuid4()), "state": "running"}
        self._undo_restore = operation
        def action():
            temporary = None
            restored = False
            try:
                if file_stamp(target) != expected_stamp:
                    raise ValueError("The local project changed. Keep the backup and review both files.")
                digest = hashlib.sha256()
                with source.open("rb") as incoming, tempfile.NamedTemporaryFile(dir=target_identity.canonical_path.parent, delete=False) as output:
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
                target_identity.validate()
                backup_identity.validate()
                expected_project = record.get("project") or _project_uuid(source)
                _require_project(source, expected_project)
                _require_project(target, expected_project)
                os.replace(temporary, target_identity.canonical_path)
                restored = True
                record["undoRestored"] = True
                if record.get("previousLink") and record.get("appliedLink"):
                    self._bucket()["links"][record["project"]] = copy.deepcopy(record["previousLink"])
                self._save()
                with self._lock:
                    operation.update(state="restored")
            except Exception as exc:
                log_failure("download_restore", exc, operation_id=operation["id"])
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
        root = self.root
        if root.resolve() != root:
            raise ValueError("The transfer folder was redirected. Keep it for recovery.")

        def owned(value, directory, identifier=None):
            path = Path(value).absolute()
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise ValueError("A transfer file was redirected. Keep it for recovery.")
            path = path.resolve()
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
                if Path(job["path"]).resolve() != directory.with_suffix(".licht"):
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
                except Exception as exc:
                    log_failure("clear_transfers", exc)
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
            job.pop("cleanupPending", None)
            job["preparedRemoved"] = True
            job.pop("previewPng", None)
            self._save()
            log_stage("export_cleanup", job_id=job["id"], path=job["path"], status=job["status"])
        except (ValueError, OSError) as exc:
            log_failure("export_cleanup", exc, job_id=job["id"])
            with self._lock:
                job.update(cleanupPending=True, message=("Upload complete." if job["status"] == "completed" else "Upload failed." if job["status"] == "error" else "Upload discarded.")
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
                self._bucket().get("handoffIntents", {}).pop(job.get("handoff", {}).get("id"), None)
            self._save()
            self._retire_export(job)
        self._launch(action)

    def unlink(self, project_id):
        def check_pending():
            if any((j["project"] == project_id or j.get("handoff", {}).get("oldProject") == project_id)
                    and j["status"] not in ("completed", "canceled") for j in self._bucket()["jobs"]):
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
                unlinked = bucket.setdefault("unlinkedProjects", [])
                if project_id not in unlinked:
                    unlinked.append(project_id)
                bucket["handoffIntents"] = {key: value for key, value in bucket.get("handoffIntents", {}).items()
                                           if project_id not in (value["oldProject"], value["newProject"])}
            self._save()
        self._launch_metadata(action)

    def set_local_details(self, project_id, title, description):
        with self._lock:
            self._check_journal_ready()
            bucket = self._bucket()
            if any(job["project"] == project_id and job["status"] not in ("completed", "canceled")
                   for job in bucket["jobs"]):
                raise ValueError("Finish or discard this project's pending transfer before updating it.")
            link = bucket["links"].get(project_id)
            if link is None:
                raise ValueError("The Gallery link changed. Review it again.")
            previous = copy.deepcopy(link.get("localFields"))
            had_local = "localFields" in link
            fields = copy.deepcopy(link.get("localFields") or link.get("sharedFields") or shared_fields(link.get("metadata", {})))
            fields.update(title=str(title), description=str(description))
            link["localFields"] = fields
            try:
                self._save()
            except Exception:
                if had_local:
                    link["localFields"] = previous
                else:
                    link.pop("localFields", None)
                raise

    def edit(self, scene_id, baseline, metadata, *, commit_uuid=None, content_stamp=None, project_id=None, cover_png=None):
        baseline = copy.deepcopy(baseline)
        metadata = copy.deepcopy(metadata)
        def action():
            client = self._client()
            bucket = self._bucket()
            if cover_png is not None:
                link = bucket["links"].get(project_id)
                if not link or link["sceneId"] != scene_id:
                    raise ValueError("The Gallery link changed. Check gallery before setting its cover.")
            scene = client.update(scene_id, domain_tokens(baseline), **metadata)
            with self._lock:
                self.scenes = [scene if s["id"] == scene_id else s for s in self.scenes]
                self.message = "Gallery details saved."
                self._completion = {"id": str(uuid.uuid4()), "kind": "publish", "scene": copy.deepcopy(scene)}
                linked_projects = [key for key, value in bucket["links"].items() if value["sceneId"] == scene_id]
                acknowledged = project_id or (linked_projects[0] if len(linked_projects) == 1 else None)
                for identifier, link in bucket["links"].items():
                    if identifier == acknowledged and link["sceneId"] == scene_id:
                        # A metadata exchange does not exchange remote geometry.
                        tokens = domain_tokens(scene)
                        if link.get("contentRevision"):
                            tokens["contentRevision"] = link["contentRevision"]
                        link.update(**tokens, metadata=copy.deepcopy(scene),
                                    sharedFields=shared_fields(scene), exchangedAt=time.time(), checkedAt=time.time())
                        link.pop("localFields", None)
                        if commit_uuid:
                            link["commitUuid"] = commit_uuid
                        if content_stamp:
                            link["contentStamp"] = content_stamp
            self._save()
            if cover_png is not None:
                client.set_cover(scene_id, scene, cover_png)
                updated = client.scene(scene_id)
                with self._lock:
                    self.scenes = [updated if item["id"] == scene_id else item for item in self.scenes]
                    bucket["links"][project_id]["metadata"] = copy.deepcopy(updated)
                    bucket["links"][project_id]["acknowledgedPresentationRevision"] = updated.get("presentationRevision", "")
                    self._completion["scene"] = copy.deepcopy(updated)
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


    def fail_local_update(self, job_id, reason):
        def action():
            job = self._job(job_id)
            job.setdefault("localUpdate", {}).update(state="failed", message=reason)
            job["message"] = reason
            self._save()
        self._launch_metadata(action)

    @contextmanager
    def _supersede_failed_local_updates(self, current):
        """A successful retry clears old failures without deleting recovery files."""
        previous = []
        for job in self._bucket()["jobs"]:
            update = job.get("localUpdate", {})
            if (job is not current and not job.get("retired")
                    and job.get("project") == current.get("project")
                    and job.get("sceneId") == current.get("sceneId")
                    and job.get("status") == "completed"
                    and (update.get("state") == "failed" or update.get("interrupted"))):
                previous.append((job, job.get("retired")))
                job["retired"] = True
        try:
            yield
        except Exception:
            for job, retired in previous:
                if retired is None:
                    job.pop("retired", None)
                else:
                    job["retired"] = retired
            raise



    def finish_settings_update(self, job_id, commit_uuid, stamp, fields, *, acknowledge=True, preserve_local_content=False):
        fields = copy.deepcopy(fields)
        def action():
            bucket = self._bucket()
            job = self._job(job_id)
            update = job.get("localUpdate", {})
            if update.get("state") != "ready" or not update.get("backupPath") or file_stamp(update["path"]) != stamp:
                raise ValueError("The local project changed. Its recovery copy was kept.")
            link = bucket["links"].get(job["project"])
            if not link or link["sceneId"] != job["sceneId"]:
                raise ValueError("The Gallery link changed. Its recovery copy was kept.")
            path_identity = ProjectPathIdentity.capture(update["path"])
            remote = self._client().scene(job["sceneId"])
            if domain_tokens(remote) != domain_tokens(job["result"]):
                raise ValueError("The gallery item changed while applying settings. Its recovery copy was kept.")
            _require_project(update["path"], job["project"])
            before_link, before_update = copy.deepcopy(link), copy.deepcopy(update)
            link["localFields"] = fields
            if acknowledge:
                link.update(metadataRevision=job["result"]["metadataRevision"],
                            sharedFields=shared_fields(job["result"]),
                            metadata=copy.deepcopy(job["result"]), exchangedAt=time.time())
                if not preserve_local_content:
                    link["commitUuid"] = commit_uuid
            update.update(state="applied", appliedStamp=list(stamp), appliedCommit=commit_uuid,
                          appliedIdentity=list(self.identity()), appliedLink=copy.deepcopy(link))
            job.update(message="Gallery changes applied. Recovery copy kept.")
            self.message = job["message"]
            try:
                with self._supersede_failed_local_updates(job):
                    self._save(project_checks=((path_identity, job["project"]),))
            except Exception:
                link.clear()
                link.update(before_link)
                update.clear()
                update.update(before_update)
                raise
        self._launch_metadata(action)


    def acknowledge_gallery_text(self, scene, project_id, path, stamp, reviewed_link):
        scene = copy.deepcopy(scene)
        reviewed_link = copy.deepcopy(reviewed_link)
        identity = self.identity()
        def action():
            bucket = self._bucket()
            link = bucket["links"].get(project_id)
            if (not link or any(link.get(key) != reviewed_link.get(key) for key in
                    ("sceneId", "contentRevision", "metadataRevision", "commitUuid", "sharedFields", "localFields"))
                    or link["sceneId"] != scene["id"]):
                raise ValueError("The previous gallery link changed. Review it again.")
            if any(not job.get("retired") and (job.get("status") not in ("completed", "canceled")
                    or job.get("localUpdate", {}).get("state") in ("preparing", "ready", "failed")
                    or job.get("localUpdate", {}).get("interrupted"))
                    and (job.get("project") == project_id or job.get("sceneId") == scene["id"])
                    for job in bucket["jobs"]):
                raise ValueError("This project already has a transfer. Resume or discard it first.")
            if any(project_id in (intent["oldProject"], intent["newProject"])
                    for intent in bucket.get("handoffIntents", {}).values()):
                raise ValueError("This project already has a transfer. Resume or discard it first.")
            if file_stamp(path) != stamp:
                raise ValueError("The local project changed. Review it before updating.")
            path_identity = ProjectPathIdentity.capture(path)
            _require_project(path, project_id)
            remote = self._client().scene(scene["id"])
            if self.identity() != identity:
                raise ValueError("The account changed. Refresh the gallery before continuing.")
            if domain_tokens(remote) != domain_tokens(scene):
                raise ValueError("The previous gallery link changed. Review it again.")
            if file_stamp(path) != stamp:
                raise ValueError("The local project changed. Review it before updating.")
            before_link = copy.deepcopy(link)
            local_fields = {key: value for key, value in link.get("localFields", {}).items()
                            if key not in ("title", "description") and value != link.get("sharedFields", {}).get(key)}
            link.update(metadataRevision=scene["metadataRevision"], sharedFields=shared_fields(remote),
                        metadata=copy.deepcopy(remote), exchangedAt=time.time(), checkedAt=time.time())
            if local_fields:
                link["localFields"] = local_fields
            else:
                link.pop("localFields", None)
            try:
                self._save(project_checks=((path_identity, project_id, stamp),))
            except Exception:
                link.clear()
                link.update(before_link)
                raise
            self.message = "projects.gallery.info.applied"
        self._launch_metadata(action)



    def prepare_settings_environment(self, job_id):
        def action():
            job = self._job(job_id)
            job["settingsEnvironment"] = {"state": "preparing"}
            self._save()
            target = self.root / "environments" / (str(uuid.UUID(job_id)) + ".lfsenv")
            temporary = target.with_suffix(".tmp")
            try:
                client = self._client()
                remote = client.scene(job["sceneId"])
                if domain_tokens(remote) != domain_tokens(job["result"]):
                    raise ValueError("The Gallery background changed. Check gallery before applying it.")
                def progress(done, total):
                    self._client()
                    with self._lock:
                        job.update(completed=done, total=total)
                        self.version += 1
                client.download(job["sceneId"], job["path"], cancel=self._cancel, on_progress=progress)
                target.parent.mkdir(mode=0o700, exist_ok=True)
                if target.parent.is_symlink() or target.is_symlink() or temporary.is_symlink():
                    raise ValueError("The HDR background folder was redirected.")
                from .portable_project import ProjectFile
                with open(job["path"], "rb") as source, temporary.open("wb") as output:
                    ProjectFile(source).copy_environment(output)
                    output.flush()
                    os.fsync(output.fileno())
                self._client()
                os.replace(temporary, target)
                job["settingsEnvironment"] = {"state": "ready", "path": str(target)}
            except Exception as exc:
                job["settingsEnvironment"] = {"state": "failed", "message": friendly_error(exc)}
                raise
            finally:
                temporary.unlink(missing_ok=True)
                self._save()
        self._launch_metadata(action)



    def prepare_settings_update(self, scene, project_id, path, stamp):
        self._client()
        if self.busy:
            raise ValueError("Wait for the current operation before applying Gallery changes.")
        identifier = str(uuid.uuid4())
        job = dict(id=identifier, project=project_id, kind="download", sceneId=scene["id"],
                   path=str(self.root / "downloads" / (identifier + ".licht")),
                   metadata={"title": scene["title"]}, checkpoint=None, status="completed",
                   completed=0, total=0, message="Applying Gallery settings", result=copy.deepcopy(scene),
                   settingsOnly=True)
        jobs = self._bucket()["jobs"]
        jobs.append(job)
        try:
            operation = self.prepare_local_update(identifier, project_id, path, stamp)
        except Exception:
            jobs.remove(job)
            raise
        return identifier, operation


    def acknowledge_replacement(self, project_id, old_project, scene_id):
        def action():
            bucket = self._bucket()
            if bucket["links"].get(old_project, {}).get("sceneId") != scene_id:
                raise ValueError("The previous gallery link changed. Review it again.")
            self._data["version"] = 3
            bucket.setdefault("replacementAcknowledgments", {})[project_id] = {
                "oldProject": old_project, "sceneId": scene_id}
            bucket["handoffIntents"] = {key: value for key, value in bucket.get("handoffIntents", {}).items()
                                       if value["newProject"] != project_id}
            self._save()
        self._launch_metadata(action)



    def remember_replacement(self, handoff):
        """Persist the reviewed identity change before native preparation starts."""
        with self._lock:
            self._client()
            if self.busy:
                raise ValueError("Wait for the current operation before replacing the published scene.")
            bucket = self._bucket()
            for existing in bucket.get("handoffIntents", {}).values():
                if {key: value for key, value in existing.items() if key != "id"} == handoff:
                    return copy.deepcopy(existing)
            intent = dict(handoff, id=str(uuid.uuid4()))
            probe = dict(id=intent["id"], project=intent["newProject"], commitUuid=intent["commitUuid"],
                         handoff=handoff, metadata=dict(replaceSceneId=intent["sceneId"], baseRevisions=intent["baseRevisions"]))
            self._check_handoff(probe)
            with _locked_sidecar(self.root / "sync.lock", blocking=False):
                if self._journal_digest() != self._disk_digest:
                    self._stale = True
                    raise ValueError(JOURNAL_CHANGED_MESSAGE)
                previous, version = copy.deepcopy(bucket.get("handoffIntents", {})), self._data["version"]
                intents = {key: value for key, value in previous.items() if value["newProject"] != intent["newProject"]}
                intents[intent["id"]] = intent
                bucket["handoffIntents"], self._data["version"] = intents, 3
                try:
                    self._save()
                except Exception:
                    bucket["handoffIntents"], self._data["version"] = previous, version
                    raise
            return copy.deepcopy(intent)


    def acknowledge_presentation(self, project_id, scene):
        def action():
            link = self._bucket()["links"].get(project_id)
            if link and link["sceneId"] == scene["id"]:
                link["acknowledgedPresentationRevision"] = scene.get("presentationRevision", "")
                self._save()
        self._launch_metadata(action)



    def set_cover(self, project_id, scene, png):
        scene = copy.deepcopy(scene)
        def action():
            client = self._client()
            link = self._bucket()["links"].get(project_id)
            if not link or link["sceneId"] != scene["id"]:
                raise ValueError("The Gallery link changed. Check gallery before setting its cover.")
            client.set_cover(scene["id"], scene, png)
            updated = client.scene(scene["id"])
            self.scenes = [item for item in self.scenes if item["id"] != scene["id"]] + [updated]
            link["acknowledgedPresentationRevision"] = updated.get("presentationRevision", "")
            self._save()
            self.message = "projects.gallery.info.cover"
        self._launch_metadata(action)


    def find_publications(self, project_id):
        """Recover one unambiguous origin link after authenticating its owner."""
        def action():
            scenes = self._client().list_scenes(origin_project_uuid=project_id)
            scenes = [scene for scene in scenes if scene.get("originProjectUuid") == project_id
                      and scene.get("status") == "ready"]
            bucket = self._bucket()
            recovered = self._origin_publication_links(scenes, bucket["links"], bucket.get("unlinkedProjects", ()))
            if recovered:
                scene = next(iter(recovered.values()))["metadata"]
                self.scenes = [item for item in self.scenes if item["id"] != scene["id"]] + [scene]
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


def same_undo_link(current, applied):
    def fields(link):
        saved = link.get("localFields") or link.get("sharedFields")
        # Links written by older builds still include visibility in their saved fields.
        return {k: v for k, v in saved.items() if k != "visibility"} if saved is not None else None
    return (all(current.get(key) == applied.get(key) for key in ("sceneId", "commitUuid"))
            and fields(current) == fields(applied))
