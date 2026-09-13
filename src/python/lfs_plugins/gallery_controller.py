# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gallery publishing and recovery from Asset Manager."""
from __future__ import annotations

import uuid
import time
import re
import threading
import copy
from pathlib import Path

import lichtfeld as lf

from .gallery_sync import get_gallery_sync, friendly_error, file_stamp
from .gallery_view import capture_camera_path, capture_view, restore_camera_path, restore_view
from . import gallery_preparation

def tr(key, **values):
    from .localization import safe_format
    full_key = "asset_manager.gallery." + key
    return safe_format(lf.ui.tr(full_key), **values)

class GalleryController:
    def __init__(self):
        self.service = get_gallery_sync()
        self._handle = None
        self._version = None
        self._state = self.service.snapshot()
        self._identity = self._state["identity"]
        self._scene = None
        self._confirm = None
        self._title = ""
        self._description = ""
        self._visibility = "private"
        self._upload_format = "studio"
        self._message = ""
        self._import_pending = None
        self._save_pending = None
        self._export_pending = None
        self._project_export_fallback = None
        self._export_cancelled = False
        self._export_identity = None
        self._export_progress = 0
        self._phase_poll_scheduled = False
        self._import_started = None
        self._import_detached = False
        self._project_link = None
        self._reset_account_ui = False
        self._history_limit = 30
        self._native_use = None
        self._progress_pending = False
        self._suppress_transfer_progress = False
        self._track_pull_pending = None
        self._subscribers = {}
        self._timer = None
        self._last_notification = 0.0
        self._last_snapshot = None
        self._next_refresh = 0.0
        self._refresh_pending = False
        self._backoff = 5.0
        self.checked_at = 0.0
        self.offline = False
        self._operation_project = None
        self._pull_requests = {}
        self._cancel_requests = set()
        self._resume_queue = []
        self._open_continuation = None
        self._pull_overrides = None
        self._undo_pull = None
        self._reupload_reason = None
        self._pulled_project = None
        self._allow_metadata_patch = False
        self._decision_pending = False
        self._last_canceled = False
        self._publish_as_new = False
        self._closed = False
        self._update_queue = []
        self._batch_rows = []
        self._batch_retries = {}
        self._batch_approval = None
        self._batch_current = None

    def close(self):
        """Release timers/subscribers when the plugin runtime shuts down."""
        self._closed = True
        if self._timer:
            self._timer.cancel()
            self._timer = None
        self._subscribers.clear()

    def preferences(self):
        from .gallery_preferences import read_preferences
        return read_preferences(getattr(self.service, "root", None))

    @property
    def upload_format(self):
        return self.preferences()["uploadFormat"]

    @upload_format.setter
    def upload_format(self, value):
        from .gallery_preferences import set_preference
        set_preference("uploadFormat", value, self.service.root)
        self._upload_format = value

    def update_all(self, assets):
        """Queue a reviewed account-bound batch; each item reuses publish validation."""
        self._check_identity()
        if self._panel_busy() or self._update_queue:
            raise ValueError(tr("error.busy"))
        state = self.service.snapshot()
        identity = self.service.identity()
        entries = []
        public = {}
        for asset in assets:
            link = state["links"].get(asset["id"])
            scene = next((s for s in state["scenes"] if link and s["id"] == link["sceneId"]), None)
            facts = asset_sync_state(asset, link, scene, state["jobs"])
            if facts["action"] != "update" or facts["freshness"] != "local":
                continue
            entries.append({"asset": copy.deepcopy(asset), "scene": copy.deepcopy(scene),
                            "identity": identity, "format": self.upload_format})
            if scene.get("visibility") == "public":
                public[scene["id"]] = scene.get("revision")
        def start():
            if self.service.identity() != identity:
                return
            self._batch_approval = (identity, public)
            self._update_queue = entries
            self._advance_update_all()
            self._schedule_tick()
        if public:
            self.confirm_action("confirm.update_all", "\n".join(e["scene"].get("title", "") for e in entries
                if e["scene"]["id"] in public), start)
        else:
            start()

    def _advance_update_all(self):
        if self._panel_busy() or self._open_continuation:
            return
        if self._batch_current:
            entry, previous_message = self._batch_current
            self._batch_current = None
            if entry["identity"] == self.service.identity():
                state = self.service.snapshot()
                has_job = any(j.get("project") == entry["asset"]["id"] and j["id"] not in entry.get("job_ids", ()) for j in state["jobs"])
                completed = state.get("completion") and state["completion"].get("id") != entry.get("completion_id")
                if not has_job and not completed and self._message and self._message != previous_message:
                    self._record_batch_failure(entry, self._message)
            if self._last_canceled:
                self._update_queue = []
        if not self._update_queue:
            self._batch_approval = None
            return
        entry = self._update_queue.pop(0)
        if entry["identity"] != self.service.identity():
            self._update_queue = []
            self._batch_approval = None
            self._batch_rows = []
            return
        asset, scene = entry["asset"], entry["scene"]
        try:
            state = self.service.snapshot()
            current = next((s for s in state["scenes"] if s["id"] == scene["id"]), None)
            if not current or current.get("revision") != scene.get("revision"):
                raise ValueError(tr("error.refresh"))
            entry["job_ids"] = {j["id"] for j in state["jobs"]}
            entry["completion_id"] = (state.get("completion") or {}).get("id")
            self._batch_current = (entry, self._message)
            self.publish_asset(asset, {k: scene.get(k, "") for k in ("title", "description", "visibility")},
                               entry["format"], update=True)
        except Exception as exc:
            self._batch_current = None
            self._record_batch_failure(entry, str(exc))

    def _record_batch_failure(self, entry, message):
        from .gallery_messages import localize_message
        identifier = "batch:" + str(uuid.uuid4())
        self._batch_retries[identifier] = entry
        self._batch_rows.append({"id": identifier, "project": entry["asset"]["id"],
            "status": "error", "kind": "upload", "metadata": {"title": entry["scene"].get("title", "")},
            "message": localize_message(message), "batchFailure": True})

    def command(self, name, job_id=None):
        if job_id and job_id.startswith("batch:"):
            if name in ("cancel", "resume"):
                entry = self._batch_retries.get(job_id)
                if name == "resume" and entry:
                    self._check_identity()
                    if entry["identity"] != self.service.identity():
                        return
                    if self._panel_busy() or self._update_queue:
                        raise ValueError(tr("error.busy"))
                    # Retry enters the normal confirmation/validation path again.
                    self.publish_asset(entry["asset"], {k: entry["scene"].get(k, "") for k in
                        ("title", "description", "visibility")}, entry["format"], update=True)
                    self._batch_current = (entry, self._message)
                self._batch_rows = [row for row in self._batch_rows if row["id"] != job_id]
                self._batch_retries.pop(job_id, None)
                self._schedule_tick()
            return
        if name == "pause":
            self._update_queue = []
            self._batch_approval = None
            self._action_pause()
        elif name == "cancel":
            if self.service.busy:
                self._cancel_requests.add(job_id)
                if any(j["id"] == job_id and j["status"] == "running" for j in self.service.snapshot()["jobs"]):
                    self.service.pause()
            else:
                self.service.discard(job_id)
        elif name == "resume_all":
            self._resume_queue = [j["id"] for j in self.service.snapshot()["jobs"]
                                  if j["status"] in ("paused", "error", "queued")]
        elif name == "clear_finished":
            self.service.clear_finished(tuple(self._clearable_jobs()))
        else:
            self._dispatch(name, [job_id] if job_id else [])
        self._schedule_tick()

    def publish_asset(self, asset, details, upload_format, *, update=False, publish_as_new=False, on_fallback=None):
        self._check_identity()
        self._refresh_model()
        if self._panel_busy():
            raise ValueError(tr("error.busy"))
        poll = lf.project_poll_write()
        if not poll.get("path") or Path(poll["path"]).resolve() != Path(asset["path"]).resolve():
            self._publish_closed_asset(asset, details, upload_format, update=update,
                                       publish_as_new=publish_as_new, on_fallback=on_fallback)
            return
        project, path = self._project_identity()
        if project != asset["id"] or Path(path).resolve() != Path(asset["path"]).resolve():
            raise ValueError(tr("error.project_changed"))
        self._operation_project = project
        self._reupload_reason = None
        self._last_canceled = False
        self._publish_as_new = publish_as_new
        self._scene = None
        if update:
            link = self._state["links"].get(project)
            self._scene = next((s for s in self._state["scenes"] if link and s["id"] == link["sceneId"]), None)
            if self._scene is None:
                raise ValueError(tr("error.refresh"))
            facts = asset_sync_state(asset, link, self._scene)
            if facts["freshness"] in ("diverged", "remote", "unknown"):
                self.resolve_asset(asset, details)
                return
        self._title, self._description, self._visibility = (details[k] for k in ("title", "description", "visibility"))
        self._upload_format = upload_format
        self._allow_metadata_patch = update
        self.upload_format = upload_format
        if update and asset.get("commit_uuid") == link.get("commitUuid") and not lf.project_is_dirty():
            metadata = self._details()
            metadata["viewerSettings"] = capture_view(lf)
            self.edit_scene(self._scene, metadata)
        else:
            self._action_publish()
            self._show_confirmation()
        self._schedule_tick()

    def _publish_closed_asset(self, asset, details, upload_format, *, update, publish_as_new, on_fallback):
        """Prepare saved content without consulting the current scene or view."""
        prepare = getattr(lf, "prepare_gallery_project", None)
        if not callable(prepare):
            if on_fallback:
                on_fallback()
                return
            raise ValueError(tr("error.project_changed"))
        if upload_format not in ("studio", "sog", "ssog", "spz"):
            raise ValueError("Choose a supported upload format.")
        if "licht" not in self._state.get("source_formats", []):
            raise ValueError("Update the portal connection before publishing .licht files.")
        project_id, path = asset["id"], asset["path"]
        info = lf.io.inspect_project(path)
        if str(info.project_uuid) != project_id:
            raise ValueError(tr("error.project_changed"))
        if any(j["project"] == project_id and j["status"] not in ("completed", "canceled") for j in self._state["jobs"]):
            raise ValueError("This project already has an upload. Resume or discard it first.")
        link = self._state["links"].get(project_id)
        scene = next((s for s in self._state["scenes"] if link and s["id"] == link["sceneId"]), None) if update else None
        if update and scene is None:
            raise ValueError(tr("error.refresh"))
        if update and asset_sync_state(asset, link, scene)["freshness"] in ("diverged", "remote", "unknown"):
            # Review must resolve remote write guards before any replacement.
            raise ValueError(tr("error.refresh"))
        if link and not update and not publish_as_new:
            raise ValueError("This project is linked to a gallery item. Select it to replace, or unlink before publishing a new item.")
        self._title, self._description, self._visibility = (details[k] for k in ("title", "description", "visibility"))
        metadata = self._details()
        metadata["viewerSettings"] = {}  # Portal imports VIEW/SEQR from the prepared .licht.
        metadata["_uploadFormat"] = upload_format
        if publish_as_new:
            metadata["_publishAsNew"] = True
        if scene:
            metadata.update(replaceSceneId=scene["id"], baseRevision=scene["revision"])
        expected_commit = str(asset.get("commit_uuid") or getattr(info, "commit_uuid", ""))
        identity = self.service.identity()
        self._operation_project = project_id
        self._last_canceled = False
        self._reupload_reason = None
        self.upload_format = upload_format

        def start():
            if self.service.identity() != identity:
                return
            if lf.ui.get_export_state().get("active"):
                raise ValueError("Wait for the current export to finish before uploading.")
            from .gallery_project_facts import saved_content_stamp
            metadata["_contentStamp"] = saved_content_stamp(path)
            export = self.service.root / (str(uuid.uuid4()) + ".scene")
            self._export_cancelled = False
            self._export_identity = identity
            self._export_progress = 0
            try:
                prepare(path, str(export), "ply" if upload_format == "studio" else upload_format, expected_commit)
            except Exception as exc:
                if on_fallback and self._project_export_can_fallback(str(exc)):
                    on_fallback()
                    return
                raise
            self._project_export_fallback = (on_fallback, expected_commit)
            self._export_pending = (export, metadata, project_id, time.monotonic())
            self._schedule_phase_poll()

        self._public_confirmation(scene, start, details=metadata, defer=True)
        self._show_confirmation()
        self._schedule_tick()

    @staticmethod
    def _project_export_can_fallback(error):
        return error.startswith(("gallery_project_not_supported:", "gallery_project_commit_mismatch:",
                                 "gallery_project_payload_unavailable:", "gallery_project_no_splats:"))

    def edit_scene(self, scene, details, *, on_started=None):
        """The one metadata editor path for local links and gallery-only items."""
        self._check_identity()
        if self._panel_busy():
            raise ValueError(tr("error.busy"))
        metadata = copy.deepcopy(details)
        if ("title" in metadata and (not metadata["title"].strip() or len(metadata["title"].strip()) > 120)
                or len(metadata.get("description", "")) > 5000
                or metadata.get("visibility", "private") not in ("private", "public")):
            raise ValueError(tr("error.details"))
        if "title" in metadata:
            metadata["title"] = metadata["title"].strip()
        scene = copy.deepcopy(scene)
        identity = self._identity
        def apply():
            if self.service.identity() != identity:
                return
            self.service.edit(scene["id"], scene["revision"], metadata)
            if on_started:
                on_started()
            self._schedule_tick()
        self._public_confirmation(scene, apply, details=metadata)

    def _public_confirmation(self, scene, action, *, details=None, defer=False):
        """One public-visibility policy for PATCH and upload commands."""
        details = details or {}
        scene = scene or {}
        was_public = scene.get("visibility") == "public"
        visibility = details.get("visibility", scene.get("visibility", "private"))
        if visibility == "public" and self.preferences()["askBeforePublic"] and not self._batch_public_approved(scene, details):
            key = "confirm.public_update" if was_public else "confirm.public"
            title = details.get("title", scene.get("title", ""))
            self._confirm = (tr(key, title=title), action, tr("action.update" if was_public else "action.submit"))
            if not defer:
                self._show_confirmation()
        else:
            action()

    def _batch_public_approved(self, scene, details):
        approval = getattr(self, "_batch_approval", None)
        return bool(approval and approval[0] == self.service.identity()
                    and approval[1].get(scene.get("id")) == scene.get("revision")
                    and scene.get("visibility") == "public"
                    and details.get("visibility", "public") == "public")

    def resolve_asset(self, asset, details):
        """Review each differing shared group before accepting a new write guard."""
        self._refresh_model()
        link = self._state["links"].get(asset["id"])
        scene = next((s for s in self._state["scenes"] if link and s["id"] == link["sceneId"]), None)
        if not scene:
            raise ValueError(tr("error.refresh"))
        current = lf.project_poll_write().get("path")
        if not current or Path(current).resolve() != Path(asset["path"]).resolve():
            from .training_confirm import confirm_discard_work_then
            identity = self._identity
            def open_selected(stop_training):
                lf.project_open(asset["path"], True, stop_training, keep_asset_manager_open=True)
                self._open_continuation = (asset["path"], identity, lambda: self.resolve_asset(asset, details))
                self._schedule_tick()
            confirm_discard_work_then(tr("action.resolve"), open_selected)
            return
        local = dict(details, viewerSettings=capture_view(lf))
        remote = copy.deepcopy(scene)
        baseline = link.get("sharedFields", {})
        groups = []
        for name, keys in (("text", ("title", "description")), ("visibility", ("visibility",))):
            if any(local.get(k) != remote.get(k) for k in keys):
                groups.append((name, keys))
        local_view, remote_view = local["viewerSettings"], remote.get("viewerSettings", {})
        if {k: v for k, v in local_view.items() if k != "cameraPath"} != {k: v for k, v in remote_view.items() if k != "cameraPath"}:
            groups.append(("view", ()))
        if local_view.get("cameraPath") != remote_view.get("cameraPath"):
            groups.append(("track", ()))
        if not link.get("commitUuid") or asset.get("commit_uuid") != link["commitUuid"]:
            groups.append(("content", ()))
        if not groups:
            self._message = tr("state.unknown")
            return
        identity = self._identity
        decisions = {}
        expected_project = self._project_identity()

        def choose(index):
            if self.service.identity() != identity or self._project_identity() != expected_project:
                return
            if index == len(groups):
                self._decision_pending = False
                apply()
                return
            name, keys = groups[index]
            buttons = [tr("action.cancel"), tr("conflict.mine"), tr("conflict.portal")]
            if name == "track" and local_view.get("cameraPath") and remote_view.get("cameraPath"):
                buttons.append(tr("conflict.both"))
            def selected(button):
                if button not in buttons[1:]:
                    self._decision_pending = False
                    self._last_canceled = True
                    return
                decisions[name] = "mine" if button == buttons[1] else "portal" if button == buttons[2] else "both"
                choose(index + 1)
            self._decision_pending = True
            lf.ui.confirm_dialog(tr("action.resolve"), tr("conflict.group", title=scene["title"], group=tr("conflict." + name)), buttons, selected)

        def apply():
            metadata = copy.deepcopy(local)
            for name, keys in groups:
                if decisions[name] == "portal":
                    for key in keys:
                        metadata[key] = remote.get(key, "")
            view = copy.deepcopy(remote_view if decisions.get("view") == "portal" else local_view)
            track = copy.deepcopy(remote_view.get("cameraPath") if decisions.get("track") == "portal" else local_view.get("cameraPath"))
            if decisions.get("track") == "both":
                track = combine_camera_tracks(local_view["cameraPath"], remote_view["cameraPath"])
            view["cameraPath"] = track
            metadata["viewerSettings"] = view
            if decisions.get("content") == "portal":
                self.pull_asset(asset, scene)
                # The staged apply consumes the reviewed view; metadata PATCH
                # waits until the backup and local save have completed.
                self._pull_overrides = (scene["id"], metadata, identity)
            elif "content" in decisions:
                metadata.update(replaceSceneId=scene["id"], baseRevision=scene["revision"])
                self._operation_project = asset["id"]
                self._publish(metadata, expected_project=expected_project, upload_format=self.upload_format)
            else:
                self.service.edit(scene["id"], scene["revision"], metadata)
            self._schedule_tick()
        choose(0)

    def confirm_action(self, key, title, continuation):
        """Native confirmation shared by Asset Manager actions."""
        self._check_identity()
        if self._panel_busy():
            raise ValueError(tr("error.busy"))
        self._confirm = (tr(key, title=title), continuation, tr("action.submit"))
        self._show_confirmation()

    def _show_confirmation(self):
        if not self._confirm:
            return
        message, action, label = self._confirm
        identity = self._identity
        self._confirm = None
        self._decision_pending = True
        def selected(button):
            self._decision_pending = False
            self._last_canceled = button != label
            if button == label and self.service.identity() == identity:
                try:
                    if self._panel_busy():
                        raise ValueError(tr("error.busy"))
                    action()
                except Exception as exc:
                    from .gallery_messages import localize_message
                    self._message = localize_message(str(exc))
                self._schedule_tick()
        lf.ui.confirm_dialog(tr("sidebar.title"), message, [tr("action.cancel"), label], selected)

    @staticmethod
    def safe_filename(title):
        return (re.sub(r"[^\w -]", "", title).strip(" .")[:70] or "Gallery") + ".licht"

    def open_portal(self, scene, action="open"):
        if not scene:
            return
        url = self.service.account.base_url + "/gallery/scenes/" + str(uuid.UUID(scene["id"])) + "/"
        if action == "copy" and scene.get("visibility") == "public" and scene.get("viewerUrl"):
            lf.ui.set_clipboard_text(scene["viewerUrl"])
        else:
            tab = "manage" if action == "copy" else action if action in ("story", "display", "manage") else "story"
            lf.ui.open_url(url + "?tab=" + tab)

    def pull_asset(self, asset, scene, destination=None, *, open_after=False):
        self._check_identity()
        if self._panel_busy():
            raise ValueError(tr("error.busy"))
        if destination:
            target = Path(destination).expanduser().absolute()
            if target.suffix.lower() != ".licht" or not target.parent.is_dir() or target.exists():
                raise ValueError(tr("error.destination"))
        else:
            target = None
        self.service.download(scene, destination=str(target) if target else None)
        job = self.service.snapshot()["jobs"][-1]
        self._pull_requests[job["id"]] = (dict(asset, _pull_open=bool(open_after)), self._identity)
        self._operation_project = asset["id"]
        self._schedule_tick()

    def _finish_pulls(self):
        if self.service.busy or self.phase() != "idle":
            return
        for job in self._state["jobs"]:
            request = self._pull_requests.get(job["id"])
            if not request or job["status"] != "completed":
                continue
            asset, identity = self._pull_requests.pop(job["id"])
            if identity != self.service.identity():
                return
            if asset.get("remote_only") or not asset.get("exists", True):
                if asset.get("_pull_open"):
                    self._import_download(job)
                else:
                    self._register_download(job, identity)
            else:
                def apply(job=job):
                    self._refresh_model()
                    self._action_update_local(job["id"])
                    self._show_confirmation()
                current = lf.project_poll_write().get("path")
                if current and Path(current).resolve() == Path(asset["path"]).resolve():
                    apply()
                else:
                    from .training_confirm import confirm_discard_work_then
                    def opened(stop_training):
                        lf.project_open(asset["path"], True, stop_training, keep_asset_manager_open=True)
                        self._open_continuation = (asset["path"], identity, apply)
                    confirm_discard_work_then(tr("action.pull"), opened)
            break

    def subscribe(self, callback, *, visible=True):
        """Subscribe on the UI thread. Timers only post work to that thread."""
        self._subscribers[callback] = visible
        self._check_identity()
        initial = self.snapshot()
        callback(initial)
        self._last_snapshot = copy.deepcopy(initial)
        self._last_notification = time.monotonic()
        if visible:
            self._next_refresh = min(self._next_refresh, time.monotonic() + 30)
        self._schedule_tick()
        def unsubscribe():
            self._subscribers.pop(callback, None)
            if not any(self._subscribers.values()) and not self.offline:
                self._next_refresh = time.monotonic() + self.preferences()["refreshMinutes"] * 60
        return unsubscribe

    def snapshot(self):
        from .gallery_messages import localize_message
        state = self.service.snapshot()
        state["jobs"] = list(state.get("jobs", [])) + copy.deepcopy(self._batch_rows)
        state["batchQueued"] = len(self._update_queue)
        for job in state.get("jobs", []):
            job["message"] = localize_message(job.get("message", ""))
        return dict(state, checkedAt=self.checked_at, offline=self.offline,
                    message=localize_message(self._message or state.get("message", "")),
                    accountFlow=self._account_flow(), phase=self.phase(), preparationProgress=self._export_progress,
                    undoPull=copy.deepcopy(self._undo_pull), pulledProject=copy.deepcopy(self._pulled_project), reuploadReason=copy.deepcopy(self._reupload_reason))

    def _account_flow(self):
        account = getattr(self.service, "account", None)
        if account is None:
            return {}
        snap = account.snapshot()
        return {key: getattr(snap, key, default) for key, default in (
            ("linking", False), ("user_code", ""), ("verification_uri_complete", ""),
            ("countdown_seconds", 0), ("error", ""))}

    def undo_pull(self):
        pending = self._undo_pull
        if not pending or pending["identity"] != self.service.identity():
            return
        if lf.project_is_dirty() or self._panel_busy():
            raise ValueError(tr("error.project_changed"))
        try:
            pending["operation"] = self.service.restore_local_backup(pending["path"], pending["backup"], pending["stamp"])
        except Exception as exc:
            self._record_undo_failure(pending, str(exc))
        else:
            self._after_service = self._finish_undo_pull
        self._schedule_tick()

    def _finish_undo_pull(self):
        pending = self._undo_pull
        if not pending or pending["identity"] != self.service.identity():
            return
        from .gallery_messages import localize_message
        state = self.service.snapshot()
        result = state.get("undoRestore", {})
        if result.get("id") == pending.get("operation") and result.get("state") == "restored":
            self._undo_pull = None
            self._message = localize_message(result.get("message", ""))
            current = lf.project_poll_write().get("path")
            if not lf.project_is_dirty() and current and Path(current).resolve() == Path(pending["path"]).resolve():
                lf.project_open(pending["path"], True, False, keep_asset_manager_open=True)
            return
        self._record_undo_failure(pending, result.get("message") or state.get("message"), result.get("backupMissing", False))

    def _record_undo_failure(self, pending, message, backup_missing=False):
        from .gallery_messages import localize_message
        pending.pop("operation", None)
        pending["attempt"] = pending.get("attempt", 0) + 1
        pending["backupMissing"] = backup_missing or not Path(pending["backup"]).is_file()
        pending["error"] = (tr("error.backup") if pending["backupMissing"] else
            localize_message(message or tr("error.failed")))
        self._message = pending["error"]

    def phase(self):
        if self._import_pending:
            return "applying"
        if self._export_pending or self._save_pending:
            return "preparing"
        return "idle"

    def refresh(self):
        if not self.service.busy:
            self._refresh_pending = True
            self.service.refresh()
            self._schedule_tick()

    def _schedule_tick(self):
        if self._closed or self._timer is not None:
            return
        self._timer = threading.Timer(0.1 if self.service.busy or self.phase() != "idle" else 1.0,
            lambda: lf.ui.schedule_on_ui_thread(self._tick))
        self._timer.daemon = True
        self._timer.start()

    def _tick(self):
        self._timer = None
        if self._closed:
            return
        try:
            self._tick_body()
            self._last_poll_error = None
        except Exception as exc:
            from .gallery_messages import report_poll_error
            self._message = report_poll_error(self, exc, "Gallery polling failed")
        finally:
            self._schedule_tick()

    def _tick_body(self):
        self._check_identity()
        now = time.monotonic()
        if self._refresh_pending and not self.service.busy:
            self._refresh_pending = False
            state = self.service.snapshot()
            # A failed refresh retains its account-scoped cache.
            self.offline = not state.get("refresh_ok", state.get("connected", False))
            if self.offline:
                self._next_refresh = now + self._backoff
                self._backoff = min(300.0, self._backoff * 2)
            else:
                self.checked_at = time.time()
                self._backoff = 5.0
                self._next_refresh = now + (30 if any(self._subscribers.values()) else self.preferences()["refreshMinutes"] * 60)
        if now >= self._next_refresh and not self._refresh_pending and not self.service.busy:
            if self.service.snapshot().get("signed_in"):
                self.refresh()
        self._refresh_model()
        if self._open_continuation:
            path, identity, continuation = self._open_continuation
            if identity != self.service.identity():
                self._open_continuation = None
            elif lf.project_poll_write().get("path") == path and not lf.ui.get_import_state().get("active"):
                self._open_continuation = None
                continuation()
        if not self.service.busy:
            after = getattr(self, "_after_service", None)
            if after:
                self._after_service = None
                after()
            if self._cancel_requests:
                job_id = self._cancel_requests.pop()
                job = next((j for j in self.service.snapshot()["jobs"] if j["id"] == job_id), None)
                if job and job["status"] not in ("completed", "canceled"):
                    self.service.discard(job_id)
            elif self._resume_queue:
                self.service.resume(self._resume_queue.pop(0))
            else:
                try:
                    self._finish_pulls()
                except Exception as exc:
                    self._message = friendly_error(exc)
        self._advance_update_all()
        snapshot = self.snapshot()
        if snapshot != self._last_snapshot and now - self._last_notification >= 0.1:
            self._last_snapshot = copy.deepcopy(snapshot)
            self._last_notification = now
            self._publish_runtime_state(snapshot)
            for callback in tuple(self._subscribers):
                if callback in self._subscribers:
                    try:
                        callback(copy.deepcopy(snapshot))
                    except Exception as exc:
                        lf.log.error(f"Gallery subscriber failed: {exc}")

    def _publish_runtime_state(self, snapshot):
        """Feed the other lane's optional native status signal from this owner."""
        from .ui import RuntimeState
        signal = getattr(RuntimeState, "gallery_state", None)
        if signal is None:
            return
        jobs = snapshot.get("jobs", [])
        active = [j for j in jobs if j["status"] in ("running", "queued")]
        up = sum(j.get("kind") != "download" for j in active)
        down = len(active) - up
        attention = sum(j["status"] in ("conflict", "error") for j in jobs)
        total = sum(j.get("total", 0) for j in active)
        percent = int(100 * sum(j.get("completed", 0) for j in active) / total) if total else -1
        signal.value = dict(signed_in=snapshot.get("signed_in", False), active_uploads=up, active_downloads=down,
            paused=sum(j["status"] == "paused" for j in jobs), attention=attention,
            percent=min(100, percent), label=tr("sidebar.title"),
            tooltip=tr("sidebar.aggregate", uploads=up, downloads=down, attention=attention),
            tone="attention" if attention else "busy" if active else "idle", epoch=snapshot.get("version", 0))

    def link_for(self, project_uuid):
        return self.service.snapshot().get("links", {}).get(project_uuid)

    def when_project_open(self, path, identity, continuation):
        self._open_continuation = (str(path), identity, continuation)
        self._schedule_tick()

    def _dispatch(self, name, args):
        self._progress_pending = False
        if name not in ("confirm_action", "pause", "cancel_action", "transfer_progress"):
            self._suppress_transfer_progress = False
        try:
            if self._check_identity() and name not in ("account", "refresh"):
                raise ValueError("The account changed. Review your gallery before continuing.")
            self._message = ""
            self._release_native_use()
            if (self.service.busy or self._save_pending or self._import_pending or self._export_pending or self._native_use or self._track_pull_pending) and name not in ("pause", "account", "cancel_action", "transfer_progress"):
                raise ValueError("Wait for the operation to finish or pause the transfer.")
            getattr(self, "_action_"+name)(*args)
            self._progress_pending = (name in ("publish", "confirm_action", "download", "resume", "resolve", "import", "update_local")
                and not self._suppress_transfer_progress)
        except Exception as exc:
            self._message = friendly_error(exc)
        self._release_native_use()
        self._refresh_model()
        self._maybe_show_transfer_progress()

    def _maybe_show_transfer_progress(self):
        if not self._progress_pending:
            return
        if self._can_pause():
            self._progress_pending = False
            self._action_transfer_progress()
        elif not self.service.busy:
            self._progress_pending = False

    def _action_transfer_progress(self):
        lf.ui.set_panel_enabled("lfs.gallery", False)
        lf.ui.set_panel_enabled("lfs.gallery_transfer", True)

    def _panel_busy(self):
        return self.service.busy or self._decision_pending or bool(self._export_pending or self._import_pending or self._save_pending or self._native_use or self._track_pull_pending)

    def _can_pause(self):
        if self._export_pending or self._save_pending or self._track_pull_pending:
            return True
        job = self._import_pending
        if job:
            if job.get("_save_new") or job.get("_link") or job.get("_update", {}).get("phase") in ("save_updated", "linking"):
                return False
            return bool(job.get("_update") or job.get("_bundle") or job.get("_register"))
        return any(j["status"] == "running" for j in self._state["jobs"])

    def _check_identity(self):
        identity = self.service.identity()
        if identity == self._identity:
            return False
        self._identity = identity
        self._reupload_reason = None
        self._pulled_project = None
        self._update_queue = []
        self._batch_rows = []
        self._batch_retries = {}
        self._batch_approval = None
        self._batch_current = None
        self._pull_requests.clear()
        self._cancel_requests.clear()
        self._resume_queue.clear()
        self._open_continuation = None
        self._pull_overrides = None
        self._undo_pull = None
        self._after_service = None
        self.checked_at = 0
        self.offline = False
        self._next_refresh = 0
        self.service.pause()
        self._scene = self._confirm = None
        self._title = self._description = self._search = ""
        self._visibility = "private"
        self._history_limit = 30
        self._reset_account_ui = True
        self._progress_pending = False
        self._suppress_transfer_progress = False
        if self._track_pull_pending:
            self._track_pull_pending["canceled"] = True
            self._track_pull_pending = None
        if self._export_pending:
            self._cancel_own_export()
            self._export_cancelled = True
        if self._save_pending:
            self._save_pending["canceled"] = True
        if self._import_pending:
            # Keep tracking owned native work, but never register or link the
            # previous account's download after a switch.
            self._import_detached = True
        self._message = "Account changed. Refresh to load your gallery."
        if self._import_pending:
            self._message = "Account changed. Finishing the local import without linking it to this account."
        return True


    def _clearable_jobs(self):
        return [j["id"] for j in self._state["jobs"] if j["status"] in ("completed", "canceled") and not j.get("retired")]

    def _visible_jobs(self):
        pending = [j for j in self._state["jobs"] if j["status"] not in ("completed", "canceled")]
        history = [j for j in reversed(self._state["jobs"]) if j["status"] in ("completed", "canceled")]
        return pending + history[:self._history_limit]

    def _action_more_history(self):
        self._history_limit += 30

    def _action_clear_finished(self):
        identifiers = tuple(self._clearable_jobs())
        if identifiers:
            self._confirm = (tr("confirm.clear_finished", count=len(identifiers)),
                lambda: self.service.clear_finished(identifiers), lf.ui.tr("gallery.transfer.action.clear_finished"))

    def _acquire_native_use(self, job_id):
        self._release_native_use()
        if self._native_use is not None:
            raise ValueError("Finish the current gallery import first.")
        guard = self.service.local_use(job_id)
        guard.__enter__()
        self._native_use = guard

    def _release_native_use(self):
        if self._native_use is None or self._import_pending:
            return
        if self.service.busy or lf.ui.get_import_state().get("active"):
            self._schedule_phase_poll()
            return
        guard, self._native_use = self._native_use, None
        guard.__exit__(None, None, None)
        if self._handle:
            self._handle.dirty_all()

    def _refresh_model(self):
        self._check_identity()
        self._state = self.service.snapshot()
        self._project_link = None
        try:
            project_id, _ = self._project_identity()
            self._project_link = self._state["links"].get(project_id)
        except Exception:
            pass  # An unsaved or unavailable project has no selectable link.
        # Operation snapshots are pinned by the initiating command. The Asset
        # Manager owns its editable fields and refreshes clean drafts itself.

    def _schedule_phase_poll(self):
        # Native export/import must finish even when the user closes this panel.
        # Only scheduling runs on the timer; all app access stays on the UI thread.
        if self._phase_poll_scheduled or not (self._export_pending or self._import_pending or self._save_pending or self._native_use or self._track_pull_pending):
            return
        self._phase_poll_scheduled = True
        timer = threading.Timer(0.2, lambda: lf.ui.schedule_on_ui_thread(self._poll_phases))
        timer.daemon = True
        timer.start()

    def _poll_phases(self):
        self._phase_poll_scheduled = False
        self._advance_phases()
        self._schedule_phase_poll()

    def _advance_phases(self):
        try:
            self._check_identity()
            if self._save_pending:
                self._finish_current_project_save()
                if self._save_pending:
                    return
            if self._track_pull_pending:
                self._finish_camera_track_pull()
                if self._save_pending or (self._track_pull_pending and self._track_pull_pending.get("phase") == "fetching"):
                    return
            if self._export_pending:
                self._finish_export()
            if self._import_pending:
                self._finish_import()
                if self._import_pending:
                    native = lf.ui.get_import_state()
                    # The model refreshes when the service version changes.
                    # Progress painting must not clone the full history per frame.
                    current = next((j for j in self._state["jobs"] if j["id"] == self._import_pending["id"]), {})
                    staged = current.get("stagedImport", {})
                    self._export_progress = (100 * native.get("progress", 0) if native.get("active") else
                        min(100, 100 * staged.get("completed", 0) / max(1, staged.get("total", 0))))
                    if self._handle:
                        self._handle.dirty_all()
        except Exception as exc:
            self._discard_update_preview()
            self._export_pending = self._import_pending = self._save_pending = self._track_pull_pending = None
            self._message = friendly_error(exc)
            self._refresh_model()
        finally:
            self._release_native_use()
            if self._handle:
                self._handle.dirty_all()

    def _action_refresh(self):
        self.service.refresh()

    def _discard_update_preview(self):
        update = (self._import_pending or {}).get("_update", {})
        if update.get("phase") not in ("save_before_backup", "backup") or not update.get("incoming"):
            return
        try:
            if self._project_identity() != update["project"] or lf.ui.get_import_state().get("active"):
                return
            scene = lf.get_scene()
            incoming = scene.get_node_by_uuid(update["incoming"])
            if incoming is not None:
                scene.remove_node(incoming.name)
        except Exception:
            # Never hide the original failure or touch another project's nodes.
            pass

    def _action_show_recovery_folder(self):
        lf.ui.reveal_in_file_manager(str(self.service.root))

    def _action_account(self):
        lf.ui.set_panel_enabled("lfs.account", True)


    def _details(self):
        title = self._title.strip()
        if not title or len(title) > 120 or len(self._description) > 5000:
            raise ValueError(tr("error.details"))
        return {"title": title, "description": self._description, "visibility": self._visibility}

    def _project_identity(self):
        if not lf.project_has_path():
            raise ValueError("Save the current project first so gallery updates stay linked to it.")
        path = lf.project_poll_write()["path"]
        return str(lf.io.inspect_project(path).project_uuid), path

    def _save_current_project(self, continuation):
        project = self._project_identity()
        identity = self.service.identity()
        poll = lf.project_poll_write()
        if poll.get("running"):
            raise ValueError("Wait for the current project save before continuing.")
        generation = poll["generation"]
        if not lf.project_save(wait=False, regenerate_preview=False):
            raise ValueError("The project could not be saved. Resolve the save error before uploading.")
        self._save_pending = {"project": project, "identity": identity, "generation": generation + 1,
            "continuation": continuation}
        self._export_progress = 0
        self._message = "Saving your current project…"
        self._schedule_phase_poll()

    def _finish_current_project_save(self):
        pending = self._save_pending
        poll = lf.project_poll_write()
        if poll.get("running"):
            return
        self._save_pending = None
        if poll.get("error"):
            self._track_pull_pending = None
            raise ValueError("The project could not be saved. Your gallery operation was stopped; resolve the save error before retrying.")
        if pending.get("canceled") or self.service.identity() != pending["identity"]:
            self._message = "Project save finished. The gallery operation was canceled."
            self._track_pull_pending = None
            return
        if (poll.get("generation") != pending["generation"] or self._project_identity() != pending["project"] or lf.project_is_dirty()):
            raise ValueError("The project changed while saving. Your gallery operation was stopped; review your work and try again.")
        pending["continuation"]()

    def _action_publish(self):
        project = self._project_identity()
        metadata = self._details()
        if self._publish_as_new:
            metadata["_publishAsNew"] = True
        upload_format = self._upload_format
        if upload_format not in ("studio", "sog", "ssog", "spz"):
            raise ValueError("Choose a supported upload format.")
        metadata["viewerSettings"] = capture_view(lf)
        environment_source = str(lf.get_render_settings().environment_map_path) if metadata["viewerSettings"].get("environment") else None
        scene = dict(self._scene) if self._scene else None
        if scene:
            metadata.update(replaceSceneId=scene["id"], baseRevision=scene["revision"])
        self._public_confirmation(scene,
            lambda: self._publish(metadata, expected_project=project, environment_source=environment_source, upload_format=upload_format),
            details=metadata, defer=True)

    def _publish(self, metadata, *, expected_project=None, environment_source=None, upload_format="studio"):
        identity = self.service.identity()
        project_id, path = self._project_identity()
        if expected_project is not None and (project_id, path) != expected_project:
            raise ValueError("The current project changed. Review its gallery details before uploading.")
        pending = [j for j in self._state["jobs"] if j["project"] == project_id and j["status"] not in ("completed", "canceled")]
        if pending:
            raise ValueError("This project already has an upload. Resume or discard it first.")
        linked = self._state["links"].get(project_id)
        if linked and metadata.get("replaceSceneId") != linked["sceneId"] and not metadata.get("_publishAsNew"):
            raise ValueError("This project is linked to a gallery item. Select it to replace, or unlink before publishing a new item.")
        nodes = [n.name for n in self._visible_splats()]
        if not nodes:
            raise ValueError("There are no visible splats to upload.")
        self._save_current_project(lambda: self._publish_saved(metadata, project_id, path, identity, environment_source, upload_format))

    def _publish_saved(self, metadata, project_id, path, identity, environment_source=None, upload_format="studio"):
        if self.service.identity() != identity or self._project_identity() != (project_id, path):
            raise ValueError("The account or current project changed while saving. Review it before uploading.")
        from .gallery_project_facts import saved_content_stamp
        content_stamp = saved_content_stamp(path)
        linked = self.service.snapshot().get("links", {}).get(project_id, {})
        if self._allow_metadata_patch and (not content_stamp or not linked.get("contentStamp")):
            self._reupload_reason = {"project": project_id, "message": tr("info.reupload_encoding")}
        if (self._allow_metadata_patch and content_stamp and content_stamp == linked.get("contentStamp")
                and metadata.get("replaceSceneId") == linked.get("sceneId")):
            details = {k: v for k, v in metadata.items() if k in ("title", "description", "visibility", "viewerSettings")}
            self.service.edit(linked["sceneId"], metadata["baseRevision"], details,
                commit_uuid=str(lf.io.inspect_project(path).commit_uuid), content_stamp=content_stamp)
            self._allow_metadata_patch = False
            return
        nodes = [n.name for n in self._visible_splats()]
        if not nodes:
            raise ValueError("There are no visible splats to upload.")
        if upload_format not in ("studio", "sog", "ssog", "spz"):
            raise ValueError("Choose a supported upload format.")
        if "licht" not in self.service.snapshot().get("source_formats", []):
            raise ValueError("Update the portal connection before publishing .licht files.")
        environment = metadata.get("viewerSettings", {}).get("environment")
        if environment:
            settings = lf.get_render_settings()
            if (settings.environment_mode != "EQUIRECTANGULAR" or str(settings.environment_map_path) != environment_source
                    or float(settings.environment_exposure) != environment["exposure"]
                    or float(settings.environment_rotation_degrees) != environment["rotation"]):
                raise ValueError("The HDR background changed. Review the current view and try uploading again.")
        export = self.service.root / (str(uuid.uuid4()) + ".scene")
        metadata = dict(metadata)
        metadata["_commitUuid"] = str(getattr(lf.io.inspect_project(path), "commit_uuid", ""))
        metadata["_uploadFormat"] = upload_format
        metadata["_contentStamp"] = content_stamp
        self._message = "Preparing the current scene for upload…"
        self._refresh_model()
        if lf.ui.get_export_state().get("active"):
            raise ValueError("Wait for the current export to finish before uploading.")
        self._export_cancelled = False
        self._project_export_fallback = None
        self._export_identity = identity
        self._export_progress = 0
        lf.prepare_gallery_scene(str(export), "ply" if upload_format == "studio" else upload_format)
        if self.service.identity() != identity:
            self._export_cancelled = True
        self._export_pending = (export, metadata, project_id, time.monotonic())
        self._schedule_phase_poll()

    def _owns_export(self, state):
        return bool(self._export_pending and state.get("path")
            and Path(state["path"]) == Path(self._export_pending[0]))

    def _cancel_own_export(self):
        state = lf.ui.get_export_state()
        if self._owns_export(state) and state.get("active"):
            lf.ui.cancel_export()

    def _remove_preparation(self, export):
        if Path(export).suffix == ".scene":
            for path in gallery_preparation.staging_files(self.service.root, export):
                self.service._unlink_temporary(path)
        else:
            Path(export).unlink(missing_ok=True)

    def _finish_export(self):
        export, metadata, project_id, started = self._export_pending
        if self._export_identity is not None and self.service.identity() != self._export_identity:
            self._export_cancelled = True
        state = lf.ui.get_export_state()
        if not self._owns_export(state):
            self._export_pending = None
            self._project_export_fallback = None
            self._remove_preparation(export)
            self._message = "The scene preparation status changed. Please prepare your upload again."
            self._refresh_model()
            return
        if state.get("active"):
            progress = max(0, min(100, int(float(state.get("progress", 0)) * 100)))
            if progress != self._export_progress:
                self._export_progress = progress
                self._message = "Canceling scene preparation…" if self._export_cancelled else f"Preparing scene for upload… {progress}%"
                self._refresh_model()
            return
        outcome = state.get("outcome")
        if self._export_cancelled or outcome in ("failed", "cancelled"):
            fallback = self._project_export_fallback
            self._project_export_fallback = None
            self._export_pending = None
            self._remove_preparation(export)
            error = str(state.get("error", ""))
            if (not self._export_cancelled and outcome == "failed" and fallback and fallback[0]
                    and self._project_export_can_fallback(error)):
                fallback[0]()
            else:
                self._message = "Scene preparation canceled." if self._export_cancelled or outcome == "cancelled" else (error or "Scene preparation failed. Check the export status and try again.")
            self._refresh_model()
        elif outcome == "completed" and export.exists():
            self._export_pending = None
            source = self._project_export_fallback
            self._project_export_fallback = None
            try:
                if source is not None:
                    commit = str(state.get("commit_uuid", ""))
                    if not commit or (source[1] and commit != source[1]):
                        raise ValueError("The prepared project commit does not match the reviewed version.")
                    metadata["_commitUuid"] = commit
                    metadata["viewerSettings"] = gallery_preparation.publication_view_metadata(self.service.root, export)
                self.service.queue_prepared_upload(export, metadata, project_id)
                self._message = ""
            except Exception as exc:
                try:
                    self._remove_preparation(export)
                    self._message = friendly_error(exc)
                except (OSError, ValueError):
                    self._message = "The upload could not be queued. Temporary files were kept; open the recovery folder to review them."
            self._refresh_model()
        elif time.monotonic() - started > 60:
            self._project_export_fallback = None
            self._export_pending = None
            self._remove_preparation(export)
            self._message = "Studio could not prepare the scene. Check the export status and try again."
            self._refresh_model()


    def _camera_track_block_reason(self):
        if not self._scene:
            return "Select the gallery item linked to this LichtFeld Studio project first."
        try:
            project_id, _ = self._project_identity()
        except ValueError as exc:
            return str(exc)
        link = self._state["links"].get(project_id)
        if not link:
            return "This LichtFeld Studio project is not linked to a gallery item. Upload it first, then send or get its camera track."
        if link["sceneId"] != self._scene["id"]:
            return "Select the gallery item linked to this LichtFeld Studio project to send or get its camera track."
        return None

    def _can_sync_camera_track(self):
        return self._camera_track_block_reason() is None

    def _camera_track_send_revision(self, scene):
        """Send uses the live gallery revision; dirty form fields stay on `_scene` for Edit."""
        state = self.service.snapshot()
        current = next((item for item in state.get("scenes") or []
            if item.get("id") == scene["id"] and item.get("status") == "ready"), None)
        if current is None or not isinstance(current.get("revision"), str):
            raise ValueError("The selected gallery item changed. The camera track was not sent.")
        project_id, _ = self._project_identity()
        link = (state.get("links") or {}).get(project_id)
        if not link or link.get("sceneId") != scene["id"]:
            raise ValueError("Select the gallery item linked to this LichtFeld Studio project to send or get its camera track.")
        return current["revision"]

    def _camera_track_help(self):
        reason = self._camera_track_block_reason()
        if reason:
            return reason
        return "Send or get playback cameras for the current LichtFeld Studio project."

    def _action_send_camera_track(self):
        reason = self._camera_track_block_reason()
        if reason:
            raise ValueError(reason)
        project = self._project_identity()
        identity = self.service.identity()
        scene = dict(self._scene)
        track = capture_camera_path(lf)
        title = scene["title"]
        if track is None:
            message = tr("confirm.track_clear", title=title)
            label = tr("action.track_clear")
        else:
            message = tr("confirm.track_send", title=title)
            label = tr("action.track_send")
        self._suppress_transfer_progress = True
        self._confirm = (message, lambda: self._start_camera_track_send(scene, project, identity, track), label)

    def _start_camera_track_send(self, scene, project, identity, track):
        if self.service.identity() != identity or self._project_identity() != project:
            raise ValueError("The account or current project changed. Review its gallery details before sending the camera track.")
        reason = self._camera_track_block_reason()
        if reason:
            raise ValueError(reason)
        if capture_camera_path(lf) != track:
            raise ValueError("The camera track changed. Review it in LichtFeld Studio and try again.")
        self._save_current_project(lambda: self._send_camera_track_saved(scene, project, identity, track))

    def _send_camera_track_saved(self, scene, project, identity, track):
        if self.service.identity() != identity or self._project_identity() != project:
            raise ValueError("The account or current project changed while saving. The camera track was not sent.")
        if capture_camera_path(lf) != track:
            raise ValueError("The camera track changed while saving. Review it in LichtFeld Studio and try again.")
        reason = self._camera_track_block_reason()
        if reason:
            raise ValueError(reason)
        if not self._scene or self._scene["id"] != scene["id"]:
            raise ValueError("The selected gallery item changed. The camera track was not sent.")
        self._message = ""
        self.service.send_camera_track(scene["id"], self._camera_track_send_revision(scene), track)
        self._suppress_transfer_progress = False
        self._refresh_model()

    def _action_get_camera_track(self):
        reason = self._camera_track_block_reason()
        if reason:
            raise ValueError(reason)
        project = self._project_identity()
        identity = self.service.identity()
        scene = dict(self._scene)
        local = capture_camera_path(lf)
        self._suppress_transfer_progress = True
        self._confirm = (
            tr("confirm.track_get", title=scene["title"]),
            lambda: self._start_camera_track_pull(scene, project, identity, local),
            tr("action.track_get"))

    def _start_camera_track_pull(self, scene, project, identity, local):
        if self.service.identity() != identity or self._project_identity() != project:
            raise ValueError("The account or current project changed. Review its gallery details before getting the camera track.")
        reason = self._camera_track_block_reason()
        if reason:
            raise ValueError(reason)
        if capture_camera_path(lf) != local:
            raise ValueError("The camera track changed. Review it in LichtFeld Studio and try again.")
        if not self._scene or self._scene["id"] != scene["id"]:
            raise ValueError("The selected gallery item changed. The camera track was not applied.")
        self._save_current_project(lambda: self._camera_track_pull_saved(scene, project, identity, local))

    def _camera_track_pull_saved(self, scene, project, identity, local):
        if self.service.identity() != identity or self._project_identity() != project:
            raise ValueError("The account or current project changed while saving. The camera track was not applied.")
        if capture_camera_path(lf) != local:
            raise ValueError("The camera track changed while saving. The gallery track was not applied.")
        reason = self._camera_track_block_reason()
        if reason:
            raise ValueError(reason)
        if not self._scene or self._scene["id"] != scene["id"]:
            raise ValueError("The selected gallery item changed. The camera track was not applied.")
        self._message = ""
        operation = self.service.fetch_camera_track(scene["id"])
        self._track_pull_pending = {"id": operation, "scene": scene, "project": project,
            "identity": identity, "local": local, "phase": "fetching"}
        self._suppress_transfer_progress = False
        self._refresh_model()
        if not self.service.busy:
            self._finish_camera_track_pull()
        else:
            self._schedule_phase_poll()

    def _finish_camera_track_pull(self):
        pending = self._track_pull_pending
        if not pending:
            return
        if pending.get("canceled"):
            self._track_pull_pending = None
            self._message = "Getting the camera track was canceled."
            return
        if pending.get("phase") != "fetching":
            return
        if self.service.identity() != pending["identity"] or self._project_identity() != pending["project"]:
            self._track_pull_pending = None
            raise ValueError("The account or current project changed. The gallery camera track was not applied.")
        if self.service.busy:
            return
        fetch = (self.service.snapshot().get("trackFetch") or {})
        if fetch.get("id") != pending["id"]:
            self._track_pull_pending = None
            raise ValueError("The gallery camera track request changed. Try again.")
        if fetch.get("state") == "canceled":
            self._track_pull_pending = None
            self._message = "Getting the camera track was canceled."
            return
        if fetch.get("state") != "ready":
            self._track_pull_pending = None
            status = fetch.get("message") or getattr(self.service, "message", None) or self.service.snapshot().get("message")
            raise ValueError(status or "The gallery camera track could not be retrieved.")
        if fetch.get("sceneId") != pending["scene"]["id"] or not self._scene or self._scene["id"] != pending["scene"]["id"]:
            self._track_pull_pending = None
            raise ValueError("The selected gallery item changed. The camera track was not applied.")
        reason = self._camera_track_block_reason()
        if reason:
            self._track_pull_pending = None
            raise ValueError(reason)
        if capture_camera_path(lf) != pending["local"]:
            self._track_pull_pending = None
            raise ValueError("The camera track changed while retrieving it. Review it in LichtFeld Studio and try again.")
        remote = fetch.get("cameraPath")
        restore_camera_path(lf, remote)
        applied = capture_camera_path(lf)
        if remote is None:
            if applied is not None:
                self._track_pull_pending = None
                raise ValueError("LichtFeld Studio could not restore this camera path.")
        elif applied is None:
            self._track_pull_pending = None
            raise ValueError("LichtFeld Studio could not restore this camera path.")
        pending["applied"] = applied
        pending["phase"] = "save_after"
        self._save_current_project(lambda: self._camera_track_pull_applied_saved(pending))

    def _camera_track_pull_applied_saved(self, pending):
        self._track_pull_pending = None
        if pending.get("canceled") or self.service.identity() != pending["identity"]:
            self._message = "Project save finished. The gallery operation was canceled."
            return
        if self._project_identity() != pending["project"]:
            raise ValueError("The current project changed while saving. Review the camera track before trying again.")
        if capture_camera_path(lf) != pending.get("applied"):
            raise ValueError("The camera track changed while saving. Review it in LichtFeld Studio and try again.")
        self._message = "Gallery camera track applied."
        self._refresh_model()


    def _action_resume(self, job_id):
        self.service.resume(job_id)


    def _action_pause(self):
        if self._import_pending and self._import_pending.get("_register"):
            self._import_pending["_register"]["canceled"] = True
        if self._save_pending:
            self._save_pending["canceled"] = True
        if self._track_pull_pending:
            self._track_pull_pending["canceled"] = True
            if not self._save_pending:
                self._track_pull_pending = None
                self._message = "Getting the camera track was canceled."
        if self._export_pending:
            self._cancel_own_export()
            self._export_cancelled = True
            self._message = "Canceling scene preparation…"
        if self._import_pending and self._import_pending.get("_update"):
            update = self._import_pending["_update"]
            update["canceled"] = True
            if update["phase"] == "importing" and Path(update.get("path", "")).suffix == ".scene":
                lf.ui.cancel_gallery_import()
        if self._import_pending and self._import_pending.get("_bundle"):
            bundle = self._import_pending["_bundle"]
            bundle["canceled"] = True
            if bundle["phase"] == "importing":
                lf.ui.cancel_gallery_import()
        self.service.pause()

    def _action_download(self):
        if self._scene:
            self.service.download(dict(self._scene))


    def _import_download(self, job):
        identity = self.service.identity()
        if lf.is_training_active():
            raise ValueError("Stop training before opening another project.")
        if lf.project_is_dirty() or lf.project_has_path():
            self._save_current_project(lambda: self._open_download(job, identity))
            return
        elif lf.get_scene().get_nodes():
            raise ValueError("Save the current project before opening the downloaded scene.")
        self._open_download(job, identity)

    def _register_download(self, job, identity):
        """Keep and link a portable project without changing the open document."""
        if Path(job["path"]).suffix != ".licht":
            raise ValueError(tr("error.format"))
        if identity != self.service.identity():
            raise ValueError(tr("error.account_changed"))
        self._acquire_native_use(job["id"])
        try:
            stage_id = self.service.stage_download(job["id"])
        except Exception:
            self._release_native_use()
            raise
        self._import_pending = dict(job, _accountIdentity=identity,
            _register={"stage_id": stage_id, "phase": "staging"})
        self._import_detached = False
        self._message = tr("state.downloading", percent=100)
        self._schedule_phase_poll()

    def _finish_register_download(self, job):
        pending = job["_register"]
        if self.service.busy:
            return
        if pending.get("canceled") or self._import_detached or job["_accountIdentity"] != self.service.identity():
            self._import_pending = None
            self._message = tr("error.account_changed" if self._import_detached else "info.canceled")
            return
        current = next((j for j in self.service.snapshot()["jobs"] if j["id"] == job["id"]), {})
        if pending["phase"] == "staging":
            stage = current.get("stagedImport", {})
            if stage.get("id") != pending["stage_id"] or stage.get("state") != "ready":
                raise ValueError(stage.get("message") or tr("error.failed"))
            path = stage["projectPath"]
            from .asset_index import AssetIndex
            index = AssetIndex()
            if not index.load():
                raise ValueError(tr("error.storage"))
            inspection = lf.io.inspect_project(path)
            previous = index.get_asset(str(inspection.project_uuid))
            if previous and Path(previous.path).resolve() != Path(path).resolve() and Path(previous.path).exists():
                # Never move an existing catalog entry to an unrelated copy.
                raise ValueError(tr("error.link"))
            project, _ = index.register_licht_asset(path, name=job["result"]["title"], inspection=inspection)
            if project is None:
                raise ValueError(tr("error.storage"))
            pending.update(phase="linking", path=path, project=str(inspection.project_uuid),
                operation=self._link_saved_download(job["id"], path))
            return
        operation = current.get("linkOperation", {})
        self._import_pending = None
        if operation.get("id") != pending["operation"] or operation.get("state") != "ready":
            raise ValueError(operation.get("message") or tr("error.link"))
        self._pulled_project = {"id": pending["project"], "path": pending["path"], "jobId": job["id"]}
        self._message = tr("info.pulled")
        self._refresh_model()

    def _open_download(self, job, identity):
        if self.service.identity() != identity:
            raise ValueError("The account changed while saving. Review your gallery before opening the download.")
        if lf.is_training_active() or lf.ui.get_import_state().get("active"):
            raise ValueError("Finish training or the current import before opening the download.")
        self._acquire_native_use(job["id"])
        if Path(job["path"]).suffix in (".lfsg", ".licht"):
            stage_id = self.service.stage_download(job["id"])
            self._import_pending = dict(job, _accountIdentity=identity,
                _bundle={"phase": "staging", "stage_id": stage_id, "scene": lf.get_scene()})
            self._import_detached = False
            self._import_started = time.monotonic()
            self._message = "Checking downloaded scene…"
            self._schedule_phase_poll()
            return
        lf.new_project()
        if lf.get_scene().get_nodes():
            raise ValueError("The current project could not be closed. Your download is ready to open later.")
        if self.service.identity() != identity:
            raise ValueError("The account changed. Review your gallery before opening the download.")
        lf.load_file(job["path"])
        self._import_pending = dict(job, _accountIdentity=identity)
        self._import_detached = self.service.identity() != identity
        self._import_started = time.monotonic()
        self._message = "Opening downloaded scene…"
        self._schedule_phase_poll()

    def _finish_import(self):
        job = self._import_pending
        if job.get("_register"):
            self._finish_register_download(job)
            return
        if job.get("_accountIdentity") is not None and self.service.identity() != job["_accountIdentity"]:
            self._import_detached = True
        if job.get("_native_project"):
            expected = job["_native_project"]
            if self._import_detached:
                self._import_pending = None
                self._message = "Account changed. The downloaded project is kept locally."
                return
            current_path = lf.project_poll_write().get("path")
            if (not current_path or Path(current_path).resolve() != Path(expected["path"]).resolve()
                    or lf.get_scene().total_gaussian_count != expected["count"]):
                if time.monotonic() - self._import_started > 180:
                    self._import_pending = None
                    self._message = "Project loading did not finish. The downloaded .licht file is kept."
                return
            from .asset_index import AssetIndex
            index = AssetIndex()
            if not index.load(): raise ValueError("Could not open the Asset Manager catalog.")
            project, _ = index.register_licht_asset(expected["path"], name=job["result"]["title"])
            if project is None: raise ValueError("The project opened but could not be added to Asset Manager.")
            restore_view(lf, job["result"].get("viewerSettings", {}), environment_path=self.service.environment_path(job))
            operation = self._link_saved_download(job["id"], expected["path"])
            job.pop("_native_project")
            job["_link"] = operation
            job["_registered_project"] = {"id": str(project.project_uuid), "path": expected["path"], "jobId": job["id"]}
            self._message = "Project opened. Saving its gallery link…"
            return
        if job.get("_update"):
            self._finish_local_update(job)
            return
        if job.get("_save_new"):
            self._finish_new_project_save(job)
            return
        bundle = job.get("_bundle")
        if bundle and bundle["phase"] == "staging":
            if self.service.busy:
                return
            if self._import_detached or bundle.get("canceled") or not bundle["scene"].is_valid() or lf.project_is_dirty():
                self._import_pending = None
                self._message = "Account or project changed. Your download is kept; open it again when ready."
                return
            if lf.is_training_active() or lf.ui.get_import_state().get("active"):
                raise ValueError("Finish training or the current import before opening this download. Your download is kept.")
            current = next(j for j in self.service.snapshot()["jobs"] if j["id"] == job["id"])
            stage = current.get("stagedImport", {})
            if stage.get("id") != bundle["stage_id"] or stage.get("state") != "ready":
                raise ValueError(stage.get("message") or "The downloaded scene could not be prepared.")
            if Path(job["path"]).suffix == ".licht":
                from .portable_project import ProjectFile
                with open(stage["projectPath"], "rb") as source:
                    prepared = ProjectFile(source)
                    count = sum(node["count"] for node in prepared.manifest["nodes"])
                lf.project_open(stage["projectPath"], keep_asset_manager_open=True)
                job["_native_project"] = {"path": stage["projectPath"], "count": count}
                bundle["phase"] = "opened"
                self._import_started = time.monotonic()
                self._message = "Opening .licht project…"
                return
            nodes = self._bundle_nodes(stage["path"])
            lf.new_project()
            if lf.get_scene().get_nodes():
                raise ValueError("The current project could not be closed. Your download is kept.")
            lf.load_gallery_scene(nodes, Path(stage["path"]).stem)
            job["_native_path"] = stage["path"]
            bundle["phase"] = "importing"
            bundle["scene"] = lf.get_scene()
            self._import_started = time.monotonic()
            self._message = "Opening downloaded scene…"
            return
        if bundle and not job.get("_link") and bundle["phase"] == "importing" and lf.project_has_path():
            self._import_pending = None
            self._message = "Project changed. Your download is kept."
            return
        if job.get("_link"):
            if self._import_detached:
                self._import_pending = None
                self._message = "The download is saved in Asset Manager. Account changed; refresh your gallery to check its link."
                self._refresh_model()
                return
            if self.service.busy:
                return
            current = next((j for j in self.service.snapshot()["jobs"] if j["id"] == job["id"]), {})
            operation = current.get("linkOperation", {})
            self._import_pending = None
            if operation.get("id") == job["_link"] and operation.get("state") == "ready":
                self._message = "Downloaded scene saved and linked in Asset Manager."
                self._pulled_project = job.get("_registered_project")
            else:
                self._message = "The download is saved in Asset Manager, but its gallery link could not be saved. Refresh your gallery before continuing."
            self._refresh_model()
            return
        state = lf.ui.get_import_state()
        if state.get("active"):
            return
        node = lf.get_scene().get_node(Path(job.get("_native_path", job["path"])).stem)
        if bundle and bundle.get("canceled"):
            if node is not None:
                lf.get_scene().remove_node(node.name)
            self._import_pending = None
            self._message = "Import canceled. Your download was kept."
            self._refresh_model()
            return
        if node is None:
            if state.get("error") or time.monotonic() - self._import_started > 60:
                self._import_pending = None
                self._message = "Studio could not open the download. The downloaded file has been kept."
            return
        self._import_pending = None
        # The gallery owns this completed import and supplies its completion UI.
        lf.ui.dismiss_import()
        if self._import_detached:
            self._message = "The previous account’s download is open locally and has not been linked. Save it before closing."
            self._refresh_model()
            return
        try:
            restore_view(lf, job["result"].get("viewerSettings", {}), environment_path=self.service.environment_path(job))
            from .asset_index import resolve_default_asset_directory
            directory = resolve_default_asset_directory()
            directory.mkdir(parents=True, exist_ok=True)
            title = job["result"]["title"]
            if bundle and lf.get_scene().get_node(title) is None:
                lf.get_scene().rename_node(node.name, title)
            filename = re.sub(r"[^\w -]", "", title).strip(" .")[:70] or "Gallery splat"
            path = Path(job["destination"]) if job.get("destination") else directory / (filename + "-" + str(uuid.uuid4())[:8] + ".licht")
            if path.exists():
                raise ValueError(tr("error.destination"))
            poll = lf.project_poll_write()
            if poll.get("running"):
                raise ValueError("Another project save is running. Your download is open; save it before closing.")
            if not lf.project_save_as(str(path), wait=False):
                raise ValueError("The downloaded scene is open, but could not be saved as a project. Save it before closing.")
            self._import_pending = dict(job, _save_new={"path": str(path), "title": title,
                "generation": poll["generation"] + 1})
            self._message = "Saving the downloaded project…"
            self._schedule_phase_poll()
        except Exception as exc:
            self._message = friendly_error(exc)
        self._refresh_model()

    def _link_saved_download(self, job_id, path):
        inspected = lf.io.inspect_project(path)
        commit = str(getattr(inspected, "commit_uuid", ""))
        args = (job_id, str(inspected.project_uuid))
        return self.service.link_download(*args, commit) if commit else self.service.link_download(*args)

    def _finish_new_project_save(self, job):
        saved = job["_save_new"]
        poll = lf.project_poll_write()
        if poll.get("running"):
            return
        self._import_pending = None
        if poll.get("error"):
            raise ValueError("The downloaded scene could not be saved as a project. Its download is kept; save your open scene before closing.")
        if (poll.get("path") != saved["path"] or poll.get("generation") != saved["generation"] or lf.project_is_dirty()):
            raise ValueError("The current project changed while saving. Your download is kept; review the saved project before linking it.")
        from .asset_index import AssetIndex
        index = AssetIndex()
        if not index.load():
            raise ValueError("Project saved, but the Asset Manager catalog could not be loaded.")
        project, _ = index.register_licht_asset(saved["path"], name=saved["title"])
        if project is None:
            raise ValueError("Project saved, but could not be added to Asset Manager. Open the saved project to retry.")
        identifier = str(lf.io.inspect_project(saved["path"]).project_uuid)
        if self._import_detached or self.service.identity() != job.get("_accountIdentity", self._identity):
            self._message = "The download is saved in Asset Manager. Account changed; it has not been linked to this account."
        else:
            operation = self._link_saved_download(job["id"], saved["path"])
            linked_job = dict(job, _link=operation)
            linked_job.pop("_save_new")
            self._import_pending = linked_job
            self._message = "Downloaded scene saved. Saving its gallery link…"
            self._schedule_phase_poll()
        self._refresh_model()

    def _action_update_local(self, job_id):
        job = next(j for j in self._state["jobs"] if j["id"] == job_id)
        project = self._project_identity()
        link = self._state["links"].get(project[0])
        if not link or link["sceneId"] != job.get("result", {}).get("id"):
            raise ValueError("This download is not linked to the current project.")
        self._confirm = (tr("confirm.pull", title=Path(project[1]).stem),
            lambda: self._begin_local_update(job, project), tr("action.pull"))

    def _bundle_nodes(self, path):
        nodes, _ = gallery_preparation.read_staging(self.service.root / "imports", path)
        return [dict(node, path=str(node["path"])) for node in nodes]

    @staticmethod
    def _visible_splats():
        scene = lf.get_scene()
        # Consolidation stores geometry together and releases per-node models.
        # Visibility and node identity remain available without copying geometry.
        return [node for node in scene.get_nodes()
            if node.type == lf.scene.NodeType.SPLAT and scene.is_node_effectively_visible(node.id)]

    def _begin_local_update(self, job, project):
        if self._pull_overrides and self._pull_overrides[0] == job.get("result", {}).get("id"):
            job = copy.deepcopy(job)
            job["result"]["viewerSettings"] = copy.deepcopy(self._pull_overrides[1]["viewerSettings"])
        if self._project_identity() != project:
            raise ValueError("The current project changed. Review it before updating.")
        if lf.is_training_active() or lf.ui.get_import_state().get("active"):
            raise ValueError("Finish training or the current import before updating this project.")
        if any(n.locked for n in self._visible_splats()):
            raise ValueError("Unlock the visible splats before updating this project.")
        if job.get("kind") != "download" or job["status"] != "completed":
            raise ValueError("Finish downloading this scene first.")
        self._acquire_native_use(job["id"])
        stage_id = self.service.stage_download(job["id"])
        self._import_pending = dict(job, _accountIdentity=self._identity,
            _update={"project": project, "phase": "staging", "stage_id": stage_id})
        self._import_detached = False
        self._import_started = time.monotonic()
        self._message = "Preparing the gallery update…"
        self._schedule_phase_poll()

    def _finish_local_update(self, job):
        update = job["_update"]
        project = update["project"]
        if self._save_pending:
            return
        if self._project_identity() != project:
            self._import_pending = None
            self._message = ("Project changed after the gallery update. Its recovery copy was kept; review the saved project before linking it."
                if update["phase"] in ("save_updated", "linking") else "Project changed. The gallery update was not applied.")
            return
        if update["phase"] == "save_updated":
            self._finish_update_save(job)
            return
        # Wait for an owned native import to finish before hiding its preview.
        if update["phase"] == "importing" and lf.ui.get_import_state().get("active"):
            return
        scene = lf.get_scene()
        incoming = scene.get_node(Path(update["path"]).stem) if update.get("path") else None
        if incoming is not None and update["phase"] == "importing":
            lf.set_node_visibility(incoming.name, False)
        if self._import_detached or update.get("canceled"):
            if incoming is not None and Path(update.get("path", "")).suffix == ".scene" and update["phase"] in ("importing", "save_before_backup", "backup"):
                scene.remove_node(incoming.name)
            self._import_pending = None
            self._message = ("The project was updated; its recovery copy was kept. Refresh the gallery to check its link."
                if update["phase"] == "linking" else "Update canceled. Your existing local splats remain.")
            return
        if self.service.busy:
            return
        current_job = next(j for j in self.service.snapshot()["jobs"] if j["id"] == job["id"])
        if update["phase"] == "linking":
            linked = current_job.get("linkOperation", {})
            if linked.get("id") != update["link_operation"] or linked.get("state") != "ready":
                raise ValueError("The project was updated, but its gallery link could not be saved. Your recovery copy is available; refresh the gallery before continuing.")
            self._import_pending = None
            if current_job.get("localUpdate", {}).get("backupPath"):
                self._undo_pull = {"path": project[1], "backup": current_job["localUpdate"]["backupPath"],
                    "stamp": file_stamp(project[1]), "identity": self._identity}
            if self._pull_overrides:
                scene_id, metadata, identity = self._pull_overrides
                self._pull_overrides = None
                if identity == self._identity:
                    self.service.edit(scene_id, current_job["result"]["revision"], metadata)
            self._message = "Linked project updated. Your previous local work is kept in its recovery copy."
            self._refresh_model()
            return
        if update["phase"] == "staging":
            stage = current_job.get("stagedImport", {})
            if stage.get("id") != update["stage_id"] or stage.get("state") != "ready":
                raise ValueError(stage.get("message") or self.service.message)
            update["path"] = stage["path"]
            if scene.get_node(Path(stage["path"]).stem):
                raise ValueError("The update preview already exists. Download the gallery item again.")
            if Path(stage["path"]).suffix == ".scene":
                lf.load_gallery_scene(self._bundle_nodes(stage["path"]), Path(stage["path"]).stem, hidden=True)
            else:
                lf.load_file(stage["path"])
            update["phase"] = "importing"
            self._import_started = time.monotonic()
            return
        if update["phase"] == "importing":
            if incoming is None:
                if lf.ui.get_import_state().get("error") or time.monotonic() - self._import_started > 60:
                    raise ValueError("Studio could not open the gallery update. Your local splats remain.")
                return
            lf.ui.dismiss_import()
            update["incoming"] = incoming.uuid
            update["old_nodes"] = [n.uuid for n in self._visible_splats() if n.uuid != incoming.uuid]
            update["phase"] = "save_before_backup"
            self._save_current_project(lambda: self._prepare_update_backup(job))
            return
        backup = current_job.get("localUpdate", {})
        if backup.get("id") != update["backup_id"] or backup.get("state") != "ready":
            raise ValueError(backup.get("message") or self.service.message)
        if (lf.project_is_dirty() or lf.project_poll_write()["generation"] != update["generation"] or
                file_stamp(project[1]) != update["stamp"]):
            raise ValueError("Your local project changed during preparation. Its splats were kept. Review it and try the update again.")
        incoming = scene.get_node_by_uuid(update["incoming"])
        if incoming is None:
            raise ValueError("The downloaded preview changed. Your local splats were kept.")
        update["phase"] = "applying"
        try:
            self._apply_local_update(scene, incoming, job, update)
        except Exception as exc:
            self._recover_failed_update(update, exc)
        update["phase"] = "save_updated"
        self._message = "Saving the updated project…"

    def _prepare_update_backup(self, job):
        update = job["_update"]
        project = update["project"]
        update["generation"] = lf.project_poll_write()["generation"]
        update["stamp"] = file_stamp(project[1])
        update["backup_id"] = self.service.prepare_local_update(job["id"], project[0], project[1], update["stamp"])
        update["phase"] = "backup"
        self._message = "Keeping a recovery copy before replacing local splats…"

    def _recover_failed_update(self, update, exc):
        recovery = "Your recovery copy is available under Show recovery copy."
        try:
            project = update["project"]
            # Reopen only the unchanged generation that preceded the update.
            if self._project_identity() == project and file_stamp(project[1]) == update["stamp"]:
                lf.project_open(project[1], discard_changes=True, keep_asset_manager_open=True)
                recovery = "Your saved local project is being reopened. Its recovery copy is also available."
        except Exception:
            pass
        raise ValueError("The gallery update could not be completed. " + recovery) from exc

    def _finish_update_save(self, job):
        update = job["_update"]
        poll = lf.project_poll_write()
        if poll.get("running"):
            return
        if poll.get("error"):
            self._recover_failed_update(update, ValueError(poll["error"]))
        if (poll.get("generation") != update["generation"] + 1 or
                self._project_identity() != update["project"] or lf.project_is_dirty()):
            raise ValueError("The saved project changed during the gallery update. Your recovery copy was kept; review the current project before linking it.")
        if self._import_detached or update.get("canceled"):
            self._import_pending = None
            self._message = "The project was updated and its recovery copy was kept. Account changed; it has not been linked to this account."
            return
        update["phase"] = "linking"
        try:
            update["link_operation"] = self._link_saved_download(job["id"], update["project"][1])
        except Exception as exc:
            raise ValueError("The project was updated and its recovery copy was kept, but the gallery link could not be saved. Refresh your gallery before continuing.") from exc
        self._message = "Project updated. Saving its gallery link…"

    def _apply_local_update(self, scene, incoming, job, update):
        # A saved recovery copy exists, and no edits have occurred since it was made.
        restore_view(lf, job["result"].get("viewerSettings", {}), environment_path=self.service.environment_path(job))
        # Native remove_node(keep_children=True) keeps child-local transforms.
        # Reparent retained children explicitly first to preserve their world pose.
        removed_ids = set(update["old_nodes"])
        for node_id in update["old_nodes"]:
            node = scene.get_node_by_uuid(node_id)
            if node is not None:
                for child_id in list(node.children):
                    child = scene.get_node_by_id(child_id)
                    if child is not None and child.uuid not in removed_ids:
                        if not scene.reparent(child_id, node.parent_id):
                            raise ValueError("A child of a replaced splat could not be preserved. Unlock it before retrying; your recovery copy is available.")
        for node_id in update["old_nodes"]:
            node = scene.get_node_by_uuid(node_id)
            if node is not None:
                scene.remove_node(node.name, keep_children=True)
        lf.set_node_visibility(incoming.name, True)
        title = job["result"]["title"]
        if scene.get_node(title) is not None:
            title += " (gallery " + incoming.uuid[:8] + ")"
        scene.rename_node(incoming.name, title)
        if not lf.project_save(wait=False):
            raise ValueError("The updated project could not be saved. Your recovery copy is available under Show recovery copy.")

    def _action_show_backup(self, job_id):
        job = next(j for j in self._state["jobs"] if j["id"] == job_id)
        path = Path(job["localUpdate"]["backupPath"]).resolve()
        if path.parent != (self.service.root / "backups").resolve() or not path.is_file():
            raise ValueError("The recovery copy is no longer available at its saved location.")
        lf.ui.reveal_in_file_manager(str(path))

    def _action_confirm_action(self):
        if self._confirm:
            action = self._confirm[1]
            self._confirm = None
            action()

    def _action_cancel_action(self):
        self._confirm = None
        self._suppress_transfer_progress = False


_controller = None

def get_gallery_controller():
    global _controller
    if _controller is None:
        _controller = GalleryController()
    return _controller


def asset_sync_state(project=None, link=None, scene=None, jobs=(), *, checked=False,
                     phase="idle", storage_issue=False, cached_projection=None):
    """Three independent facts and one deterministic badge; never compare clocks."""
    project = project or {}
    identifier = project.get("project_uuid", project.get("id", ""))
    relationship = "linked" if link else "unlinked" if project else "remote_only"
    if project.get("status") in ("IDENTITY_CONFLICT", "IDENTITY_MISMATCH", "DUPLICATE", "AMBIGUOUS"):
        relationship = "identity_ambiguous"
    elif link and (link.get("remoteDeleted") or (scene and scene.get("status") == "deleted") or (checked and scene is None)):
        relationship = "remote_deleted"
    elif link and not project.get("exists", True):
        relationship = "local_missing"
    freshness = "unknown"
    if link and link.get("commitUuid") and project.get("commit_uuid") and scene and (link.get("sharedFields") or all(link.get(key) and scene.get(key) for key in ("contentRevision", "metadataRevision"))):
        from .gallery_sync import shared_fields
        local = project["commit_uuid"] != link["commitUuid"]
        domains = ("contentRevision", "metadataRevision")
        remote = (any(scene[key] != link[key] for key in domains)
                  if all(scene.get(key) and link.get(key) for key in domains)
                  else shared_fields(scene) != link["sharedFields"])
        freshness = "diverged" if local and remote else "local" if local else "remote" if remote else "equal"
    scene_id = (link or scene or {}).get("sceneId", (scene or {}).get("id"))
    matching = [j for j in jobs if j.get("status") not in ("completed", "canceled") and
                ((identifier and j.get("project") == identifier) or (scene_id and
                (j.get("sceneId") == scene_id or j.get("metadata", {}).get("replaceSceneId") == scene_id)))]
    job = matching[-1] if matching else {}
    activity = phase
    if job:
        status = job.get("status")
        activity = ("processing" if job.get("serverProcessing") else
                    "downloading" if job.get("kind") == "download" else "uploading") if status == "running" else (
                    "interrupted" if job.get("interrupted") else "paused") if status == "paused" else (
                    "error" if status in ("conflict", "error") else "queued")
        if status == "conflict":
            freshness = "diverged"
    active = activity in ("checking", "preparing", "queued", "uploading", "processing", "downloading", "applying")
    if freshness == "diverged":
        visible = "diverged"
    elif active:
        visible = activity
    elif storage_issue or relationship == "identity_ambiguous":
        visible = "error"
    elif activity in ("error", "paused", "interrupted"):
        visible = activity
    elif relationship in ("local_missing", "remote_deleted"):
        visible = relationship
    elif relationship == "linked":
        visible = freshness
    else:
        visible = relationship
    if cached_projection and not link and not scene and not jobs and not storage_issue and relationship == "unlinked":
        cached = cached_projection.get("state")
        if cached in {"equal", "local", "remote", "diverged", "unknown", "error", "paused", "interrupted", "local_missing", "remote_deleted"}:
            relationship = "linked"
            visible = cached
    icons = {"unlinked": "cloud", "equal": "cloud-check", "local": "cloud-up",
             "remote": "cloud-down", "diverged": "cloud-updown", "remote_only": "cloud-down",
             "queued": "cloud-dotted", "paused": "cloud-dotted", "interrupted": "cloud-dotted",
             "error": "cloud-bang", "local_missing": "cloud-bang", "remote_deleted": "cloud-strike",
             "unknown": "cloud-dotted"}
    tones = {"equal": "success", "local": "primary", "remote": "info", "remote_only": "info",
             "diverged": "warning", "error": "error", "local_missing": "warning", "remote_deleted": "warning",
             "interrupted": "warning", "processing": "info"}
    action = {"unlinked": "publish", "equal": "open", "local": "update", "remote": "pull",
              "diverged": "resolve", "remote_only": "pull", "local_missing": "pull",
              "remote_deleted": "publish_new", "unknown": "check", "error": "retry",
              "paused": "resume", "interrupted": "resume"}.get(visible, "")
    if visible == "remote_deleted" and not project.get("exists", True):
        action = "unlink"
    if relationship == "identity_ambiguous" or storage_issue or (cached_projection and not link):
        action = "check"
    return dict(relationship=relationship, freshness=freshness, activity=activity, state=visible,
                icon=icons.get(visible, "ring"), tone=tones.get(visible, "primary" if active else "text_dim"),
                action=action, active=active, jobId=job.get("id", ""), reason=job.get("message") or project.get("error", ""),
                progress=min(100, int(100 * job.get("completed", 0) / max(1, job.get("total", 0)))),
                attention=visible in ("diverged", "error", "local_missing", "remote_deleted"))


def combine_camera_tracks(mine, portal):
    """Keep both complete tracks in order in P0's single native camera path."""
    result = copy.deepcopy(mine)
    offset = float(mine.get("duration", 0))
    appended = copy.deepcopy(portal.get("keyframes", []))
    for frame in appended:
        frame["t"] = float(frame.get("t", frame.get("time", 0))) + offset
        frame.pop("time", None)
    result["keyframes"] = result.get("keyframes", []) + appended
    result["duration"] = offset + float(portal.get("duration", 0))
    return result
