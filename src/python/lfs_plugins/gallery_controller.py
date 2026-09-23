# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gallery publishing and recovery from Asset Manager."""
from __future__ import annotations

import uuid
import time
import threading
import copy
from pathlib import Path

import lichtfeld as lf
from .gallery_messages import tr, localize_message
from .gallery_actions import gallery_actions
from .gallery_transfer_ui import TransferEstimate, transfer_metrics, transfer_phase, transfer_rows

from .gallery_sync import get_gallery_sync, friendly_error, file_stamp
from .gallery_view import capture_view, restore_view
from .portal_gallery import domain_tokens, UNSUPPORTED_PORTAL
from . import gallery_preparation
from .portal_security import redact, safe_filename, checked_portal_url
from .gallery_logging import failure as log_failure, safe_url, stage as log_stage

class GalleryController:
    def __init__(self):
        self.service = get_gallery_sync()
        self._state = self.service.snapshot()
        self._identity = self._state["identity"]
        self._confirm = None
        self._message = ""
        self._failure_notice = ""
        self._import_pending = None
        self._save_pending = None
        self._export_pending = None
        self._prepared_commit = None
        self._export_cancelled = False
        self._export_identity = None
        self._export_progress = 0
        self._import_started = None
        self._import_detached = False
        self._native_use = None
        self._subscribers = {}
        self._timer = None
        self._last_snapshot = None
        self._transfer_estimate = TransferEstimate()
        self._job_transfer_estimates = {}
        self._transfer_ui_epoch = 0
        self._refresh_pending = False
        self._refresh_requested = False
        self._refresh_force_requested = False
        self._share_copy_pending = None
        self.checked_at = 0.0
        self.offline = False
        self._operation_project = None
        self._operation_title = ""
        self._pull_requests = {}
        self._cancel_requests = set()
        self._open_continuation = None
        self._pull_overrides = None
        self._undo_pull = None
        self._settings_pending = None
        self._reupload_reason = None
        self._pulled_project = None
        self._decision_pending = False
        self._last_canceled = False
        self._update_queue = []
        self._batch_rows = []
        self._batch_retries = {}
        self._batch_current = None
        self._preparation_failure = None
        from .ui import RuntimeState
        RuntimeState.account_state.subscribe(self._account_changed)
        self._publish_runtime_state(self.snapshot())

    def _account_changed(self, _state):
        def update():
            try:
                changed = self._check_identity()
                if changed and self.service.snapshot().get("signed_in"):
                    self.refresh()
                self._schedule_poll()
            except Exception as exc:
                log_failure("account_callback", exc)
                self._message = friendly_error(exc)
        lf.ui.schedule_on_ui_thread(update)

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

    def update_all(self, assets):
        """Queue a reviewed account-bound batch; each item reuses publish validation."""
        self._check_identity()
        if self._panel_busy() or self._update_queue:
            raise ValueError(tr("error.busy"))
        state = self.service.snapshot()
        identity = self.service.identity()
        entries = []
        for asset in assets:
            link = state["links"].get(asset["id"])
            scene = next((s for s in state["scenes"] if link and s["id"] == link["sceneId"]), None)
            facts = asset_sync_state(asset, link, scene, state["jobs"])
            if facts["action"] != "update" or facts["freshness"] != "local":
                continue
            entries.append({"asset": copy.deepcopy(asset), "scene": copy.deepcopy(scene),
                            "identity": identity, "format": self.upload_format})
        if self.service.identity() != identity:
            return
        self._update_queue = entries
        self._advance_update_all()
        self._schedule_poll()

    def _advance_update_all(self):
        if self._panel_busy() or self._open_continuation:
            return
        if self._batch_current:
            entry, previous_message = self._batch_current
            self._batch_current = None
            if entry["identity"] == self.service.identity():
                state = self.service.snapshot()
                jobs = [j for j in state["jobs"] if j.get("project") == entry["asset"]["id"] and j["id"] not in entry.get("job_ids", ())]
                has_job = bool(jobs)
                if any(j["status"] == "paused" for j in jobs):
                    self._update_queue = []
                completed = state.get("completion") and state["completion"].get("id") != entry.get("completion_id")
                if not has_job and not completed and self._message and self._message != previous_message:
                    self._record_batch_failure(entry, self._message)
            if self._last_canceled:
                self._update_queue = []
        if not self._update_queue:
            return
        entry = self._update_queue.pop(0)
        if entry["identity"] != self.service.identity():
            self._update_queue = []
            self._batch_rows = []
            return
        asset, scene = entry["asset"], entry["scene"]
        try:
            state = self.service.snapshot()
            current = next((s for s in state["scenes"] if s["id"] == scene["id"]), None)
            if not current or domain_tokens(current) != domain_tokens(scene):
                raise ValueError(tr("error.refresh"))
            entry["job_ids"] = {j["id"] for j in state["jobs"]}
            entry["completion_id"] = (state.get("completion") or {}).get("id")
            self._batch_current = (entry, self._message)
            self.publish_asset(asset, {k: scene.get(k, "") for k in ("title", "description")},
                               entry["format"], update=True)
        except Exception as exc:
            self._batch_current = None
            log_failure("batch_publish", exc, project_id=entry["asset"].get("id", ""))
            self._record_batch_failure(entry, str(exc))

    def _record_batch_failure(self, entry, message):
        from .gallery_messages import localize_message
        identifier = "batch:" + str(uuid.uuid4())
        self._batch_retries[identifier] = entry
        self._batch_rows.append({"id": identifier, "project": entry["asset"]["id"],
            "status": "error", "kind": "upload", "metadata": {"title": entry["scene"].get("title", "")},
            "message": localize_message(message), "batchFailure": True})

    def command(self, name, job_id=None):
        if job_id and job_id.startswith("queue:"):
            if name == "cancel":
                identifier = job_id.removeprefix("queue:")
                self._update_queue = [entry for entry in self._update_queue if entry["asset"]["id"] != identifier]
                self._schedule_poll()
            return
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
                        ("title", "description")}, entry["format"], update=True)
                    self._batch_current = (entry, self._message)
                self._batch_rows = [row for row in self._batch_rows if row["id"] != job_id]
                self._batch_retries.pop(job_id, None)
                self._schedule_poll()
            return
        if name == "pause":
            self._last_canceled = True
            self._update_queue = []
            self._action_pause()
        elif name == "cancel":
            if any(j["id"] == job_id and j.get("project") == self._operation_project
                   for j in self.service.snapshot()["jobs"]):
                self._last_canceled = True
            if self.service.busy:
                self._cancel_requests.add(job_id)
                if any(j["id"] == job_id and j["status"] == "running" for j in self.service.snapshot()["jobs"]):
                    self.service.pause()
            else:
                self.service.discard(job_id)
        elif name == "keep_waiting":
            self.service.resume(job_id, keep_waiting=True)
        elif name == "clear_finished":
            self.service.clear_finished(tuple(self._clearable_jobs()))
        else:
            self._dispatch(name, [job_id] if job_id else [])
        self._schedule_poll()

    def publish_asset(self, asset, details, upload_format, *, update=False, publish_as_new=False):
        self._check_identity()
        self._refresh_model()
        if self._panel_busy():
            raise ValueError(tr("error.busy"))
        poll = lf.project_poll_write()
        if not poll.get("path") or Path(poll["path"]).resolve() != Path(asset["path"]).resolve():
            self._publish_closed_asset(asset, details, upload_format, update=update,
                                       publish_as_new=publish_as_new)
            return
        project, path = self._project_identity()
        if project != asset["id"] or Path(path).resolve() != Path(asset["path"]).resolve():
            raise ValueError(tr("error.project_changed"))
        self._operation_project = project
        self._operation_title = details.get("title") or asset.get("name", "")
        self._reupload_reason = None
        self._last_canceled = False
        scene = None
        if update:
            link = self._state["links"].get(project)
            scene = next((s for s in self._state["scenes"] if link and s["id"] == link["sceneId"]), None)
            if scene is None:
                raise ValueError(tr("error.refresh"))
            facts = asset_sync_state(asset, link, scene)
            if facts["freshness"] in ("diverged", "remote", "unknown"):
                self.resolve_asset(asset, details)
                return
        self.upload_format = upload_format
        self._review_publish(scene, details, upload_format, publish_as_new, update=update)
        self._schedule_poll()

    def _publish_closed_asset(self, asset, details, upload_format, *, update, publish_as_new, handoff=None):
        """Prepare saved content without consulting the current scene or view."""
        if upload_format not in ("studio", "sog", "ssog", "spz"):
            raise ValueError("Choose a supported upload format.")
        if "licht" not in self._state.get("source_formats", []):
            raise ValueError(UNSUPPORTED_PORTAL)
        project_id, path = asset["id"], asset["path"]
        info = lf.io.inspect_project(path)
        if str(info.project_uuid) != project_id:
            raise ValueError(tr("error.project_changed"))
        if any(j["project"] == project_id and j["status"] not in ("completed", "canceled") for j in self._state["jobs"]):
            raise ValueError("This project already has an upload. Resume or discard it first.")
        link = self._state["links"].get(project_id)
        scene = next((s for s in self._state["scenes"] if link and s["id"] == link["sceneId"]), None) if update else None
        if handoff:
            scene = next((s for s in self._state["scenes"] if s["id"] == handoff["sceneId"]), None)
        if (update or handoff) and scene is None:
            raise ValueError(tr("error.refresh"))
        if update and asset_sync_state(asset, link, scene)["freshness"] in ("diverged", "remote", "unknown"):
            # Review must resolve remote write guards before any replacement.
            raise ValueError(tr("error.refresh"))
        if link and not update and not publish_as_new:
            raise ValueError("This project is linked to a gallery item. Select it to replace, or unlink before publishing a new item.")
        metadata = self._details(details)
        if getattr(info, "file_uuid", ""):
            metadata["originFileUuid"] = str(info.file_uuid)
        metadata["_uploadFormat"] = upload_format
        if publish_as_new:
            metadata["_publishAsNew"] = True
        if scene:
            metadata.update(replaceSceneId=scene["id"], baseRevisions={name: scene[name + "Revision"] for name in ("content", "metadata")})
        if handoff:
            metadata.update(_handoff=copy.deepcopy(handoff), replaceSceneId=handoff["sceneId"],
                            baseRevisions=copy.deepcopy(handoff["baseRevisions"]))
        expected_commit = str(asset.get("commit_uuid") or getattr(info, "commit_uuid", ""))
        identity = self.service.identity()
        self._operation_project = project_id
        self._operation_title = details.get("title") or asset.get("name", "")
        self._last_canceled = False
        self._reupload_reason = None
        self.upload_format = upload_format

        def start():
            self._preparation_failure = None
            if self.service.identity() != identity:
                return
            if lf.ui.get_export_state().get("active"):
                raise ValueError("Wait for the current export to finish before uploading.")
            if str(lf.io.inspect_project(path).commit_uuid) != expected_commit:
                raise ValueError(tr("error.project_changed"))
            if self._patch_saved_update(metadata, project_id, path, update=update):
                return
            self._pin_publish_preview(metadata, path, expected_commit)
            metadata["viewerSettings"] = {}  # Portal imports VIEW/SEQR from the prepared .licht.
            export = self.service.root / (str(uuid.uuid4()) + ".scene")
            self._export_cancelled = False
            self._export_identity = identity
            self._export_progress = 0
            lf.prepare_gallery_project(path, str(export), "ply" if upload_format == "studio" else upload_format, expected_commit)
            self._prepared_commit = expected_commit
            self._export_pending = (export, metadata, project_id, time.monotonic())
            self._schedule_poll()

        start()
        self._schedule_poll()

    def replace_published_asset(self, asset, scene, upload_format):
        self._check_identity()
        self._refresh_model()
        if self._panel_busy():
            raise ValueError(tr("error.busy"))
        old_project = asset.get("previous_project_uuid")
        old = self._state["links"].get(old_project, {})
        if old.get("sceneId") != scene.get("id") or asset["id"] in self._state["links"]:
            raise ValueError(tr("error.project_changed"))
        current = lf.project_poll_write().get("path")
        if current and Path(current).resolve() == Path(asset["path"]).resolve() and lf.project_is_dirty():
            raise ValueError(tr("error.project_changed"))
        info = lf.io.inspect_project(asset["path"])
        if str(info.project_uuid) != asset["id"] or str(info.commit_uuid) != asset.get("commit_uuid"):
            raise ValueError(tr("error.project_changed"))
        handoff = dict(oldProject=old_project, newProject=asset["id"], sceneId=scene["id"],
                       origin=self.service._origin, owner=self.service._owner,
                       commitUuid=str(info.commit_uuid), fileUuid=str(info.file_uuid), state="pending",
                       oldLinkRevisions={name: old[name + "Revision"] for name in ("content", "metadata")},
                       baseRevisions={name: scene[name + "Revision"] for name in ("content", "metadata")})
        handoff = self.service.remember_replacement(handoff)
        self._publish_closed_asset(asset, scene, upload_format, update=False, publish_as_new=False, handoff=handoff)

    def resolve_asset(self, asset, details, *, apply_only=False):
        """Review all differing parts together, before accepting any write guard."""
        self._check_identity()
        self._refresh_model()
        link = self._state["links"].get(asset["id"])
        scene = next((s for s in self._state["scenes"] if link and s["id"] == link["sceneId"]), None)
        if not scene:
            raise ValueError(tr("error.refresh"))
        current = lf.project_poll_write().get("path")
        current_open = bool(current and Path(current).resolve() == Path(asset["path"]).resolve())
        remote = copy.deepcopy(scene)
        if current_open:
            local = dict(details, viewerSettings=capture_view(lf))
            project = self._project_identity()
            reviewed_dirty = lf.project_is_dirty()
        else:
            local = stored_local_scene(link, details)
            project = (asset["id"], str(Path(asset["path"]).resolve()))
            reviewed_dirty = None
        groups = conflict_groups(asset, link, local, remote, apply_only=apply_only)
        if not groups:
            self._message = tr("state.equal")
            return
        identity = self._identity
        reviewed_link = copy.deepcopy(link)
        local_view, remote_view = local.get("viewerSettings") or {}, remote.get("viewerSettings", {})
        reviewed_stamp = file_stamp(asset["path"]) if Path(asset["path"]).is_file() else None

        def apply(decisions, *, local_only=False):
            publish = not (apply_only or local_only)

            text_only = (apply_only and decisions.get("text") == "gallery"
                and asset.get("commit_uuid") == link.get("commitUuid")
                and scene.get("contentRevision") == link.get("contentRevision")
                and not any(choice in ("gallery", "both") for key, choice in decisions.items() if key != "text"))
            if text_only:
                if any(entry["asset"]["id"] == asset["id"] for entry in self._update_queue):
                    raise ValueError(tr("error.project_changed"))
                self.service.acknowledge_gallery_text(remote, asset["id"], asset["path"], reviewed_stamp, reviewed_link)
                self._schedule_poll()
                return

            def run():
                if self.service.identity() != identity or self._project_identity() != project:
                    raise ValueError(tr("error.project_changed"))
                if (reviewed_stamp is not None and file_stamp(asset["path"]) != reviewed_stamp
                        or reviewed_dirty is not None and (
                            lf.project_is_dirty() != reviewed_dirty or capture_view(lf) != local_view)):
                    raise ValueError(tr("error.project_changed"))
                # A closed review only has the last Gallery snapshot. Preserve
                # the actual saved view loaded after approval, including tracks
                # that were not selected for replacement in the review.
                if reviewed_dirty is None and lf.project_is_dirty():
                    raise ValueError(tr("error.project_changed"))
                apply_local_view = capture_view(lf) if reviewed_dirty is None else local_view
                metadata = copy.deepcopy(local)
                if decisions.get("text") == "gallery":
                    for key in ("title", "description"):
                        metadata[key] = remote.get(key, "")
                view = copy.deepcopy(remote_view if decisions.get("view") == "gallery" else apply_local_view)
                track = copy.deepcopy(remote_view.get("cameraPath") if decisions.get("track") == "gallery" else apply_local_view.get("cameraPath"))
                if decisions.get("track") == "both":
                    if not apply_local_view.get("cameraPath"):
                        raise ValueError(tr("error.project_changed"))
                    track = combine_camera_tracks(apply_local_view["cameraPath"], remote_view["cameraPath"])
                if track is not None:
                    view["cameraPath"] = track
                else:
                    view.pop("cameraPath", None)
                metadata["viewerSettings"] = view

                def start():
                    if decisions.get("content") == "gallery":
                        local_environment_path = None
                        if decisions.get("view") != "gallery":
                            local_environment_path = str(lf.get_render_settings().environment_map_path) if view.get("environment") else ""
                        self._pull_overrides = (scene["id"], metadata, identity, publish, local_environment_path)
                        try:
                            self.pull_asset(asset, scene)
                        except Exception:
                            self._pull_overrides = None
                            raise
                    else:
                        self._begin_settings_apply(asset, scene, metadata,
                            publish=publish, replace_content="content" in decisions and publish,
                            preserve_local_content="content" in decisions,
                            use_gallery_environment=decisions.get("view") == "gallery" and bool(view.get("environment")))
                    self._schedule_poll()
                self._resolve_pending_uploads(asset["id"], scene["id"], identity, start)

            if self.service.identity() != identity:
                raise ValueError(tr("error.project_changed"))
            if reviewed_stamp is not None and file_stamp(asset["path"]) != reviewed_stamp:
                raise ValueError(tr("error.project_changed"))
            if reviewed_dirty is not None:
                current = lf.project_poll_write().get("path")
                if (not current or Path(current).resolve() != Path(project[1])
                        or self._project_identity() != project
                        or lf.project_is_dirty() != reviewed_dirty or capture_view(lf) != local_view):
                    raise ValueError(tr("error.project_changed"))
                run()
                return
            path = lf.project_poll_write().get("path")
            if not path or Path(path).resolve() != Path(asset["path"]).resolve():
                from .training_confirm import confirm_discard_work_then
                def open_selected(stop_training):
                    lf.project_open(asset["path"], True, stop_training, keep_asset_manager_open=True)
                    self._open_continuation = (asset["path"], identity, run)
                    self._schedule_poll()
                confirm_discard_work_then(tr("action.apply"), open_selected)
                return
            run()

        from .gallery_file_panel import open_gallery_file_panel
        self._decision_pending = True
        def closed(_submitted):
            self._decision_pending = False
            self._schedule_poll()
        try:
            open_gallery_file_panel(controller=self, asset=asset, scene=scene, action="conflict", fields={},
                mode="conflict", groups=groups, on_submit=apply, on_done=closed, apply_only=apply_only)
            self._decision_pending = True
        except Exception:
            self._decision_pending = False
            raise

    def _begin_settings_apply(self, asset, scene, metadata, *, publish=False, replace_content=False, use_gallery_environment=False, preserve_local_content=False):
        project = self._project_identity()
        identity = self.service.identity()
        if project[0] != asset["id"] or lf.is_training_active():
            raise ValueError(tr("error.project_changed"))
        def saved():
            stamp = file_stamp(project[1])
            job, backup = self.service.prepare_settings_update(scene, project[0], project[1], stamp)
            self._settings_pending = dict(project=project, identity=identity, stamp=stamp,
                scene=copy.deepcopy(scene), metadata=copy.deepcopy(metadata), job=job, backup=backup,
                phase="backup", publish=publish, replace_content=replace_content, use_gallery_environment=use_gallery_environment,
                preserve_local_content=preserve_local_content)
            self._operation_project, self._operation_title = project[0], metadata["title"]
            self._schedule_poll()
        if lf.project_is_dirty():
            self._save_current_project(saved, expected_project=project)
        else:
            saved()

    def _finish_settings_apply(self):
        pending = self._settings_pending
        if not pending or self.service.busy or self._save_pending:
            return
        if self.service.identity() != pending["identity"] or self._project_identity() != pending["project"]:
            raise ValueError(tr("error.project_changed"))
        job = next(j for j in self.service.snapshot()["jobs"] if j["id"] == pending["job"])
        backup = job.get("localUpdate", {})
        if pending["phase"] in ("backup", "environment"):
            if backup.get("id") != pending["backup"] or backup.get("state") != "ready":
                raise ValueError(backup.get("message") or tr("error.backup"))
            if file_stamp(pending["project"][1]) != pending["stamp"] or lf.project_is_dirty():
                raise ValueError(tr("error.project_changed"))
            if pending["phase"] == "backup" and pending.get("use_gallery_environment"):
                self.service.prepare_settings_environment(pending["job"])
                pending["phase"] = "environment"
                return
            metadata = pending["metadata"]
            environment = metadata["viewerSettings"].get("environment")
            environment_path = str(lf.get_render_settings().environment_map_path) if environment else None
            if pending["phase"] == "environment":
                prepared = job.get("settingsEnvironment", {})
                if prepared.get("state") != "ready":
                    raise ValueError(prepared.get("message") or tr("error.failed"))
                environment_path = prepared["path"]
            pending["environment_path"] = environment_path or ""
            restore_view(lf, metadata["viewerSettings"], environment_path=environment_path)
            pending["phase"] = "saving"
            def saved():
                stamp = file_stamp(pending["project"][1])
                self.service.finish_settings_update(pending["job"],
                    str(lf.io.inspect_project(pending["project"][1]).commit_uuid), stamp,
                    pending["metadata"], acknowledge=not pending["publish"],
                    preserve_local_content=pending.get("preserve_local_content", False))
                pending.update(phase="linking", applied_stamp=stamp)
            self._save_current_project(saved, expected_project=pending["project"])
            return
        if pending["phase"] != "linking":
            return
        if backup.get("state") != "applied":
            raise ValueError(tr("error.failed"))
        self._settings_pending = None
        self._undo_pull = dict(path=pending["project"][1], backup=backup["backupPath"],
            stamp=pending["applied_stamp"], identity=pending["identity"], jobId=job["id"], project=pending["project"][0])
        if pending["publish"]:
            metadata = pending["metadata"]
            if pending["replace_content"]:
                metadata.update(replaceSceneId=pending["scene"]["id"], baseRevisions={name: pending["scene"][name + "Revision"] for name in ("content", "metadata")})
                self._publish_saved(metadata, pending["project"][0], pending["project"][1], pending["identity"],
                    upload_format=self.upload_format, environment_source=pending.get("environment_path", ""))
            else:
                self.service.edit(pending["scene"]["id"], domain_tokens(pending["scene"]), metadata,
                    commit_uuid=str(lf.io.inspect_project(pending["project"][1]).commit_uuid), project_id=pending["project"][0])
        self._message = tr("info.applied")

    def _fail_settings_apply(self, error):
        pending, self._settings_pending = self._settings_pending, None
        if pending and self.service.identity() == pending["identity"] and not self.service.busy:
            self.service.fail_local_update(pending["job"], friendly_error(error))
            # The remote item may have changed while the review was open.
            # Queue a fresh check after recording the failed apply.
            self.refresh(force=True)

    def _resolve_pending_uploads(self, project_id, scene_id, identity, continuation):
        """Retire only the failed replacement explicitly superseded by Resolve."""
        if self.service.identity() != identity:
            self._decision_pending = False
            return
        if self.service.busy:
            self._decision_pending = True
            def idle():
                self._decision_pending = False
                self._resolve_pending_uploads(project_id, scene_id, identity, continuation)
            self._after_service = idle
            return
        self._refresh_model()
        pending = next((j for j in self._state["jobs"] if j.get("project") == project_id
            and j.get("status") == "conflict" and j.get("metadata", {}).get("replaceSceneId") == scene_id), None)
        if pending:
            self.service.discard(pending["id"])
            self._decision_pending = True
            def retired():
                self._decision_pending = False
                current = next(j for j in self.service.snapshot()["jobs"] if j["id"] == pending["id"])
                if current["status"] != "canceled":
                    raise ValueError(current.get("message") or "The previous upload could not be canceled.")
                self._resolve_pending_uploads(project_id, scene_id, identity, continuation)
            self._after_service = retired
        else:
            continuation()

    def confirm_action(self, key, title, continuation):
        """Native confirmation shared by Asset Manager actions."""
        self._check_identity()
        if self._decision_pending:
            return
        metadata_only = key in ("confirm.remove", "confirm.unlink")
        if self._metadata_busy() if metadata_only else self._panel_busy():
            raise ValueError(tr("error.busy"))
        verb = {"confirm.remove": "action.remove", "confirm.unlink": "action.unlink"}.get(key, "action.submit")
        self._confirm = (tr(key, title=title), continuation, tr(verb).rstrip("…."))
        self._show_confirmation(metadata_only=metadata_only)

    def _show_confirmation(self, *, metadata_only=False):
        if not self._confirm:
            return
        message, action, label = self._confirm
        identity = self._identity
        self._confirm = None
        self._decision_pending = True
        def selected(button):
            nonlocal action
            if action is None:
                return
            continuation, action = action, None
            self._decision_pending = False
            self._last_canceled = button != label
            if button == label and self.service.identity() == identity:
                try:
                    if self._metadata_busy() if metadata_only else self._panel_busy():
                        raise ValueError(tr("error.busy"))
                    continuation()
                except Exception as exc:
                    from .gallery_messages import localize_message
                    log_failure("confirmation_callback", exc)
                    self._message = localize_message(str(exc))
                    self._failure_notice = self._message
                self._schedule_poll()
        lf.ui.confirm_dialog(tr("sidebar.title"), message, [tr("action.cancel"), label], selected)

    @staticmethod
    def safe_filename(title):
        return safe_filename(title)

    def open_portal(self, scene, action="open"):
        if not scene:
            return
        url = self.service.account.base_url + "/gallery/scenes/" + str(uuid.UUID(scene["id"])) + "/open/"
        if action == "copy":
            self._copy_share_link(scene)
        else:
            lf.ui.open_url(checked_portal_url(self.service.account, url))

    def _copy_share_link(self, scene):
        from .portal_gallery import PortalGalleryClient
        identity = self.service.identity()
        key = (identity, scene["id"])
        if self._share_copy_pending == key:
            return
        self._share_copy_pending = key
        account = self.service.account

        def copy_link():
            link, error = None, None
            try:
                link = PortalGalleryClient(account, expected_session=identity[1:3]).share_link_details(scene["id"])
            except Exception as exc:
                log_failure("share_link", exc, scene_id=scene["id"])
                error = friendly_error(exc)

            def finished():
                if self._share_copy_pending != key:
                    return
                self._share_copy_pending = None
                if self.service.identity() != identity:
                    return
                if error:
                    self._message = error
                    self._failure_notice = error
                else:
                    lf.ui.set_clipboard_text(checked_portal_url(account, link["url"]))
                    self._message = tr("share.expiry", prefix="projects.gallery.",
                        expiry=link.get("expiresAt") or tr("share.never", prefix="projects.gallery."))
                self._refresh_model()
                self._schedule_poll()
            lf.ui.schedule_on_ui_thread(finished)
        threading.Thread(target=copy_link, daemon=True, name="GalleryShareLink").start()

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
        self._operation_title = scene.get("title", "")
        self._schedule_poll()

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
                    if self._pull_overrides and self._pull_overrides[0] == job["result"]["id"]:
                        self._begin_local_update(job, self._project_identity())
                    else:
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

    def subscribe(self, callback):
        """Observe state without starting any network work."""
        self._subscribers[callback] = True
        self._check_identity()
        initial = self.snapshot()
        try:
            callback(initial)
        except Exception as exc:
            log_failure("subscriber_callback", exc)
            self._message = friendly_error(exc)
        self._last_snapshot = copy.deepcopy(initial)
        return lambda: self._subscribers.pop(callback, None)

    def snapshot(self):
        from .gallery_messages import localize_message
        state = self.service.snapshot()
        state["jobs"] = list(state.get("jobs", [])) + copy.deepcopy(self._batch_rows)
        queued_intents = {job.get("handoff", {}).get("id") for job in state["jobs"] if job.get("status") != "canceled"}
        for intent in state.get("handoffIntents", {}).values():
            if intent["id"] not in queued_intents and not (self._operation_project == intent["newProject"] and self.phase() == "preparing"):
                state["jobs"].append(dict(id="handoff:" + intent["id"], project=intent["newProject"], sceneId=intent["sceneId"],
                    kind="upload", status="paused", interrupted=True, handoffIntent=True, completed=0, total=0,
                    metadata={"title": tr("replacement.title")}, message=tr("state.interrupted")))
        state["batchQueued"] = len(self._update_queue)
        state["preparationFailure"] = dict(self._preparation_failure, nativePreparation=True) if self._preparation_failure else None
        state["jobs"].extend({"id": "queue:" + entry["asset"]["id"], "project": entry["asset"]["id"],
            "status": "queued", "kind": "upload", "batchQueued": True,
            "metadata": {"title": entry["scene"].get("title") or entry["asset"].get("name", "")}}
            for entry in self._update_queue)
        operation = next((j for j in state["jobs"] if j.get("project") == self._operation_project), {})
        state["operationTitle"] = self._operation_title or operation.get("metadata", {}).get("title", "")
        measured_ids = set()
        for job in state.get("jobs", []):
            job["message"] = localize_message(job.get("message", ""))
            stage = transfer_phase(job)
            job["transferDetail"] = ""
            if not job.get("retired") and stage in ("uploading", "downloading"):
                measured_ids.add(job["id"])
                estimate = self._job_transfer_estimates.setdefault(job["id"], TransferEstimate())
                job["transferDetail"] = transfer_metrics(*estimate.sample([job], stage))
        self._job_transfer_estimates = {key: value for key, value in self._job_transfer_estimates.items() if key in measured_ids}
        undo_history = self.undo_records(state)
        if self._undo_pull and self._undo_pull.get("jobId") and self._undo_pull["jobId"] not in undo_history:
            self._undo_pull = None
        if not self._undo_pull and undo_history:
            self._undo_pull = copy.deepcopy(next(reversed(undo_history.values())))
        return dict(state, checkedAt=self.checked_at or state.get("checkedAt", 0), offline=self.offline,
                    message=localize_message(self._message or state.get("message", "")),
                    actionError=localize_message(self._failure_notice),
                    actionErrorId=(state.get("actionFailure") or {}).get("id", ""),
                    phase=self.phase(), preparationProgress=self._export_progress,
                    undoPull=copy.deepcopy(self._undo_pull), undoHistory=undo_history,
                    operationProject=self._operation_project,
                    pulledProject=copy.deepcopy(self._pulled_project), reuploadReason=copy.deepcopy(self._reupload_reason))

    def undo_records(self, state=None):
        state = self.service.snapshot() if state is None else state
        records = {}
        if not state.get("signed_in") or not any(job.get("localUpdate", {}).get("state") == "applied" for job in state.get("jobs", [])):
            return records
        current = lf.project_poll_write().get("path")
        for job in state.get("jobs", []):
            update = job.get("localUpdate", {})
            if update.get("state") != "applied" or job.get("retired") or update.get("undoRestored"):
                continue
            if update.get("appliedIdentity") and tuple(update["appliedIdentity"]) != tuple(state["identity"]):
                continue
            if update.get("appliedLink"):
                from .gallery_sync import same_undo_link
                if not same_undo_link(state.get("links", {}).get(job["project"], {}), update["appliedLink"]):
                    continue
            try:
                path, stamp = update["path"], update["appliedStamp"]
                if file_stamp(path) != stamp or current and Path(current).resolve() == Path(path).resolve() and lf.project_is_dirty():
                    continue
            except (KeyError, OSError):
                continue
            records[job["id"]] = dict(path=path, stamp=stamp, backup=update["backupPath"],
                identity=state["identity"], jobId=job["id"], project=job["project"])
        return records

    def _account_linking(self):
        account = getattr(self.service, "account", None)
        return bool(account and getattr(account.snapshot(), "linking", False))

    def undo_pull(self, job_id=None):
        if job_id:
            self._undo_pull = self.undo_records().get(job_id)
        pending = self._undo_pull
        if not pending or pending["identity"] != self.service.identity():
            return
        if lf.project_is_dirty() or self._panel_busy():
            raise ValueError(tr("error.project_changed"))
        try:
            pending["operation"] = self.service.restore_local_backup(pending["path"], pending["backup"], pending["stamp"])
        except Exception as exc:
            log_failure("undo_pull", exc)
            self._record_undo_failure(pending, str(exc))
        else:
            self._after_service = self._finish_undo_pull
        self._schedule_poll()

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
        if self._import_pending or getattr(self, "_settings_pending", None):
            return "applying"
        if self._export_pending or self._save_pending:
            return "preparing"
        return "idle"

    def refresh(self, *, force=False):
        self._check_identity()
        if not self.service.snapshot().get("signed_in"):
            self._refresh_requested = False
            self._refresh_force_requested = False
            self._message = ""
            self._refresh_model()
            return
        self._refresh_requested = True
        self._refresh_force_requested = self._refresh_force_requested or force
        if self._refresh_pending and self.service.busy:
            self._schedule_poll()
            return
        if not self.service.busy:
            self._refresh_requested = False
            self._message = ""
            self._refresh_pending = True
            force = self._refresh_force_requested
            self._refresh_force_requested = False
            if force:
                self.service.refresh(force=True)
            else:
                self.service.refresh()
        self._schedule_poll()

    def _schedule_poll(self):
        if self._timer is not None:
            return
        self._timer = threading.Timer(0.1,
            lambda: lf.ui.schedule_on_ui_thread(self._poll))
        self._timer.daemon = True
        self._timer.start()

    def _poll(self):
        self._timer = None
        try:
            self._poll_body()
            self._last_poll_error = None
        except Exception as exc:
            from .gallery_messages import report_poll_error
            if self._last_poll_error != (type(exc).__name__, redact(exc)):
                log_failure("poll", exc)
            self._message = report_poll_error(self, exc, "Gallery polling failed")
            self._failure_notice = self._message
        finally:
            if self._work_pending():
                self._schedule_poll()

    def _work_pending(self):
        return bool(self.service.busy or self.phase() != "idle" or self._native_use
                    or self._open_continuation or self._refresh_pending or self._refresh_requested or self._cancel_requests
                    or self._update_queue or self._batch_current
                    or getattr(self, "_after_service", None) or self._account_linking())

    def _poll_body(self):
        self._check_identity()
        failure = self.service.snapshot().get("actionFailure")
        if failure and failure["id"] != getattr(self, "_reported_action_failure", None):
            self._reported_action_failure = failure["id"]
            self._failure_notice = failure["message"]
        self._advance_phases()
        try:
            self._finish_settings_apply()
        except Exception as exc:
            self._fail_settings_apply(exc)
            log_failure("apply_settings", exc)
            self._message = friendly_error(exc)
            self._failure_notice = self._message
        if self._refresh_requested and not self.service.busy:
            self.refresh(force=self._refresh_force_requested)
        if self._refresh_pending and not self.service.busy:
            self._refresh_pending = False
            state = self.service.snapshot()
            # A failed refresh retains its account-scoped cache.
            self.offline = not state.get("refresh_ok", state.get("connected", False))
            if not self.offline:
                self.checked_at = time.time()
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
            else:
                try:
                    self._finish_pulls()
                except Exception as exc:
                    log_failure("finish_pulls", exc)
                    self._message = friendly_error(exc)
                    self._failure_notice = self._message
        self._advance_update_all()
        completion = self.service.snapshot().get("completion")
        if completion and completion.get("id") != getattr(self, "_refreshed_completion", None) and not self._work_pending():
            self._refreshed_completion = completion["id"]
            self.refresh()
        snapshot = self.snapshot()
        self._publish_runtime_state(snapshot)
        if snapshot != self._last_snapshot:
            self._last_snapshot = copy.deepcopy(snapshot)
            for callback in tuple(self._subscribers):
                if callback in self._subscribers:
                    try:
                        callback(copy.deepcopy(snapshot))
                    except Exception as exc:
                        log_failure("subscriber_callback", exc)

    def _transfer_speed_and_eta(self, jobs, stage):
        return self._transfer_estimate.sample(jobs, stage)

    def _publish_runtime_state(self, snapshot):
        """Publish transfer status to the native UI."""
        from .ui import RuntimeState
        from .asset_format import format_size
        signal = getattr(RuntimeState, "gallery_state", None)
        if signal is None:
            return
        jobs = [j for j in snapshot.get("jobs", []) if not j.get("retired")]
        active = [j for j in jobs if j["status"] in ("running", "queued")]
        running = [j for j in active if j["status"] == "running"]
        phase = snapshot.get("phase", "idle")
        up = sum(j.get("kind") != "download" for j in active)
        down = len(active) - up
        up += phase == "preparing"
        down += phase == "applying"
        attention = sum(j["status"] in ("conflict", "error") for j in jobs)
        total = sum(j.get("total", 0) for j in running)
        done = sum(min(j.get("total", 0), max(0, j.get("completed", 0))) for j in running)
        percent = min(100, int(100 * done / total)) if total > 0 and running else -1
        stage = "queued"
        if phase != "idle":
            stage = phase
            percent = min(100, max(0, int(snapshot.get("preparationProgress", 0)))) if phase == "preparing" else -1
            if percent == 0:
                percent = -1
        elif running:
            if all(j.get("serverProcessing") for j in running):
                stage, percent = "processing", -1
            elif any(j.get("preparation") and not j.get("packaged") for j in running):
                stage = "preparing"
            else:
                stage = "downloading" if running[0].get("kind") == "download" else "uploading"

        stage_label = tr("phase." + stage, prefix="gallery.transfer.")
        label = stage_label if up or down else tr("sidebar.title")
        if percent >= 0 and (up or down):
            label = tr("gallery.status.progress", prefix="", stage=label, percent=percent)
        speed, remaining = self._transfer_speed_and_eta(running, stage)
        detail = transfer_metrics(speed, remaining) if stage in ("uploading", "downloading") else ""
        details = [label, detail, tr("sidebar.aggregate", uploads=up, downloads=down, attention=attention)]
        if running:
            details.insert(0, running[0].get("metadata", {}).get("title", ""))
            details.append(tr("bytes", prefix="gallery.transfer.", done=format_size(done), total=format_size(total)))
        if snapshot.get("message"):
            details.append(snapshot["message"])
        tooltip = "\n".join(filter(None, details))
        queue = dict(identity=snapshot.get("identity"), rows=transfer_rows(snapshot, history_limit=3),
                     message=snapshot.get("message", ""))
        if queue != RuntimeState.gallery_transfers.value:
            RuntimeState.gallery_transfers.value = queue
            self._transfer_ui_epoch += 1
        signal.value = dict(signed_in=snapshot.get("signed_in", False),
            relink_required=snapshot.get("relink_required", False),
            active_uploads=up, active_downloads=down,
            paused=sum(j["status"] == "paused" for j in jobs), attention=attention,
            percent=percent, label=label, detail=detail, tooltip=tooltip,
            tone="busy" if up or down else "attention" if attention else "idle", epoch=self._transfer_ui_epoch)


    def _dispatch(self, name, args):
        try:
            if self._check_identity():
                raise ValueError("The account changed. Review your gallery before continuing.")
            self._message = ""
            self._release_native_use()
            if self._panel_busy():
                raise ValueError("Wait for the operation to finish or pause the transfer.")
            actions = {"resume": self._action_resume, "show_recovery_folder": self._action_show_recovery_folder}
            actions[name](*args)
        except Exception as exc:
            log_failure("dispatch", exc, action=name)
            self._message = friendly_error(exc)
            self._failure_notice = self._message
        self._refresh_model()

    def _panel_busy(self):
        return self.service.busy or self._decision_pending or bool(self._export_pending or self._import_pending or self._save_pending or self._settings_pending or self._native_use)

    def _metadata_busy(self):
        self._release_native_use()
        return (getattr(self.service, "metadata_busy", self.service.busy) or self._decision_pending
                or bool(self._export_pending or self._import_pending or self._save_pending or self._settings_pending or self._native_use))

    def _check_identity(self):
        identity = self.service.identity()
        if identity == self._identity:
            return False
        self._identity = identity
        self._state = dict(self._state, scenes=[], links={}, jobs=[], posters={}, identity=identity)
        self._last_snapshot = None
        self._reupload_reason = None
        self._preparation_failure = None
        self._failure_notice = ""
        self._pulled_project = None
        self._update_queue = []
        self._batch_rows = []
        self._batch_retries = {}
        self._batch_current = None
        self._pull_requests.clear()
        self._cancel_requests.clear()
        self._open_continuation = None
        self._pull_overrides = None
        self._undo_pull = None
        self._settings_pending = None
        self._after_service = None
        self._decision_pending = False
        self.checked_at = 0
        self.offline = False
        self._refresh_pending = False
        self._refresh_requested = bool(identity[-1])
        self.service.pause()
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
        undo = self.undo_records()
        return [j["id"] for j in self._state["jobs"] if j["status"] in ("completed", "canceled") and not j.get("retired") and j["id"] not in undo
                and not j.get("localUpdate", {}).get("interrupted")]

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
        if getattr(self.service, "metadata_busy", self.service.busy) or lf.ui.get_import_state().get("active"):
            self._schedule_poll()
            return
        guard, self._native_use = self._native_use, None
        guard.__exit__(None, None, None)

    def _refresh_model(self):
        self._check_identity()
        self._state = self.service.snapshot()
    def _advance_phases(self):
        try:
            self._check_identity()
            if self._save_pending:
                self._finish_current_project_save()
                if self._save_pending:
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
        except Exception as exc:
            log_failure("advance_phases", exc)
            self._discard_update_preview()
            self._fail_settings_apply(exc)
            pending = self._import_pending
            if (pending and not self.service.busy and pending.get("_accountIdentity") == self.service.identity()
                    and callable(getattr(self.service, "fail_local_update", None))):
                self.service.fail_local_update(pending["id"], friendly_error(exc))
            self._export_pending = self._import_pending = self._save_pending = None
            self._message = friendly_error(exc)
            self._failure_notice = self._message
            self._refresh_model()
        finally:
            self._release_native_use()

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
        except Exception as exc:
            # Never hide the original failure or touch another project's nodes.
            log_failure("discard_update_preview", exc)

    def _action_show_recovery_folder(self):
        lf.ui.reveal_in_file_manager(str(self.service.root))

    @staticmethod
    def _details(details):
        title = details["title"].strip()
        if not title or len(title) > 120 or len(details["description"]) > 5000:
            raise ValueError(tr("error.details"))
        result = {"title": title, "description": details["description"]}
        if "useEmbeddedPreview" in details:
            result["useEmbeddedPreview"] = bool(details["useEmbeddedPreview"])
        return result

    def _project_identity(self):
        if not lf.project_has_path():
            raise ValueError("Save the current project first so gallery updates stay linked to it.")
        path = str(Path(lf.project_poll_write()["path"]).resolve())
        return str(lf.io.inspect_project(path).project_uuid), path

    def _save_current_project(self, continuation, *, expected_project=None):
        project = self._project_identity()
        if expected_project is not None and project != expected_project:
            raise ValueError("The project identity or path changed before saving. Refresh Projects and try again.")
        identity = self.service.identity()
        poll = lf.project_poll_write()
        if poll.get("running"):
            raise ValueError("Wait for the current project save before continuing.")
        generation = poll["generation"]
        # File operations can append a save before the open document rebases.
        saved_generation = getattr(lf.io.inspect_project(project[1]), "generation", None)
        if type(saved_generation) is int:
            generation = max(generation, saved_generation)
        if not lf.project_save(wait=False, regenerate_preview=False):
            raise ValueError("The project could not be saved. Resolve the save error before uploading.")
        self._save_pending = {"project": project, "identity": identity, "generation": generation + 1,
            "continuation": continuation}
        self._export_progress = 0
        self._message = "Saving your current project…"
        self._schedule_poll()

    def _finish_current_project_save(self):
        pending = self._save_pending
        poll = lf.project_poll_write()
        if poll.get("running"):
            return
        self._save_pending = None
        if poll.get("error"):
            if self._project_identity() != pending["project"]:
                raise ValueError("The project identity or path changed before saving. Refresh Projects and try again.")
            raise ValueError("The project could not be saved. Your gallery operation was stopped; resolve the save error before retrying.")
        if pending.get("canceled") or self.service.identity() != pending["identity"]:
            self._message = "Project save finished. The gallery operation was canceled."
            return
        if (poll.get("generation") != pending["generation"] or self._project_identity() != pending["project"] or lf.project_is_dirty()):
            raise ValueError("The project changed while saving. Your gallery operation was stopped; review your work and try again.")
        pending["continuation"]()

    def _review_publish(self, scene, details, upload_format, publish_as_new, *, update=False):
        project = self._project_identity()
        metadata = self._details(details)
        if publish_as_new:
            metadata["_publishAsNew"] = True
        if upload_format not in ("studio", "sog", "ssog", "spz"):
            raise ValueError("Choose a supported upload format.")
        metadata["viewerSettings"] = capture_view(lf)
        environment_source = str(lf.get_render_settings().environment_map_path) if metadata["viewerSettings"].get("environment") else None
        if scene:
            metadata.update(replaceSceneId=scene["id"], baseRevisions={name: scene[name + "Revision"] for name in ("content", "metadata")})
        self._publish(metadata, expected_project=project, environment_source=environment_source,
                      upload_format=upload_format, update=update)

    def _publish(self, metadata, *, expected_project=None, environment_source=None, upload_format="studio", update=False):
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
        self._preparation_failure = None
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = 0
        account = getattr(self.service, "account", None)
        log_stage("publish_requested", project_id=project_id, path=path, size=size,
                  format=upload_format, account_origin=safe_url(getattr(account, "base_url", "")),
                  update=update)
        self._save_current_project(lambda: self._publish_saved(metadata, project_id, path, identity,
                                                             environment_source, upload_format, update=update))

    def _patch_saved_update(self, metadata, project_id, path, *, update):
        from .gallery_project_facts import saved_content_stamp
        content_stamp = saved_content_stamp(path)
        metadata["_contentStamp"] = content_stamp
        linked = self.service.snapshot().get("links", {}).get(project_id, {})
        if update and (":" not in content_stamp or ":" not in linked.get("contentStamp", "")):
            self._reupload_reason = {"project": project_id, "message": tr("info.reupload_encoding")}
        # Live metadata carries the complete view; a closed PATCH must also
        # prove its saved VIEW/SEQR unchanged so it cannot lose local view edits.
        baseline = linked.get("contentStamp", "")
        comparable = lambda stamp: stamp.split(":", 1)[0] if "viewerSettings" in metadata else stamp
        if (update and content_stamp and ":" in content_stamp and ":" in baseline
                and comparable(content_stamp) == comparable(baseline)
                and metadata.get("replaceSceneId") == linked.get("sceneId")):
            details = {k: v for k, v in metadata.items() if k in ("title", "description", "viewerSettings")}
            self.service.edit(linked["sceneId"], {name + "Revision": token for name, token in metadata["baseRevisions"].items()}, details,
                commit_uuid=str(lf.io.inspect_project(path).commit_uuid), content_stamp=content_stamp, project_id=project_id)
            return True
        return False

    def _publish_saved(self, metadata, project_id, path, identity, environment_source=None, upload_format="studio", *, update=False):
        if self.service.identity() != identity or self._project_identity() != (project_id, path):
            raise ValueError("The account or current project changed while saving. Review it before uploading.")
        if self._patch_saved_update(metadata, project_id, path, update=update):
            return
        nodes = [n.name for n in self._visible_splats()]
        if not nodes:
            raise ValueError("There are no visible splats to upload.")
        if upload_format not in ("studio", "sog", "ssog", "spz"):
            raise ValueError("Choose a supported upload format.")
        if "licht" not in self.service.snapshot().get("source_formats", []):
            raise ValueError(UNSUPPORTED_PORTAL)
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
        file_uuid = str(getattr(lf.io.inspect_project(path), "file_uuid", ""))
        if file_uuid:
            metadata["originFileUuid"] = file_uuid
        metadata["_uploadFormat"] = upload_format
        self._message = "Preparing the current scene for upload…"
        self._refresh_model()
        if lf.ui.get_export_state().get("active"):
            raise ValueError("Wait for the current export to finish before uploading.")
        self._export_cancelled = False
        self._pin_publish_preview(metadata, path, metadata["_commitUuid"])
        self._prepared_commit = metadata["_commitUuid"]
        self._export_identity = identity
        self._export_progress = 0
        lf.prepare_gallery_project(path, str(export), "ply" if upload_format == "studio" else upload_format,
                                   self._prepared_commit)
        if self.service.identity() != identity:
            self._export_cancelled = True
        self._export_pending = (export, metadata, project_id, time.monotonic())
        self._schedule_poll()

    @staticmethod
    def _pin_publish_preview(metadata, path, commit):
        if metadata.get("useEmbeddedPreview"):
            import base64
            preview = lf.io.read_preview(path)
            if str(lf.io.inspect_project(path).commit_uuid) != commit:
                raise ValueError(tr("error.project_changed"))
            if not preview:
                raise ValueError(tr("error.preview"))
            metadata["_previewPng"] = base64.b64encode(gallery_preparation.publication_preview(preview)).decode("ascii")

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
        prepared_commit = self._prepared_commit
        if self._export_identity is not None and self.service.identity() != self._export_identity:
            self._export_cancelled = True
        state = lf.ui.get_export_state()
        if not self._owns_export(state):
            self._export_pending = None
            self._prepared_commit = None
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
            self._prepared_commit = None
            self._export_pending = None
            self._remove_preparation(export)
            error = str(state.get("error", ""))
            self._message = "Scene preparation canceled." if self._export_cancelled or outcome == "cancelled" else (error or "Scene preparation failed. Check the export status and try again.")
            if outcome == "failed" and not self._export_cancelled:
                self._preparation_failure = {"id": "preparation:" + project_id, "project": project_id,
                    "commitUuid": prepared_commit,
                    "status": "error", "kind": "upload", "metadata": {"title": self._operation_title},
                    "message": self._message, "failureReason": self._message}
                log_failure("native_preparation", RuntimeError(self._message), project_id=project_id)
            self._refresh_model()
        elif outcome == "completed" and export.exists():
            self._export_pending = None
            source = self._prepared_commit
            self._prepared_commit = None
            try:
                if source is not None:
                    commit = str(state.get("commit_uuid", ""))
                    if not commit or (source and commit != source):
                        raise ValueError("The prepared project commit does not match the reviewed version.")
                    metadata["_commitUuid"] = commit
                    saved_view = gallery_preparation.publication_view_metadata(self.service.root, export)
                    metadata["viewerSettings"] = saved_view | metadata.get("viewerSettings", {})
                self.service.queue_prepared_upload(export, metadata, project_id)
                self._message = ""
            except Exception as exc:
                log_failure("queue_after_preparation", exc, project_id=project_id)
                self._preparation_failure = {"id": "preparation:" + project_id, "project": project_id,
                    "commitUuid": prepared_commit,
                    "status": "error", "kind": "upload", "metadata": {"title": self._operation_title},
                    "message": friendly_error(exc), "failureReason": friendly_error(exc)}
                try:
                    self._remove_preparation(export)
                    self._message = friendly_error(exc)
                except (OSError, ValueError) as cleanup_exc:
                    log_failure("preparation_cleanup", cleanup_exc, project_id=project_id)
                    self._message = "The upload could not be queued. Temporary files were kept; open the recovery folder to review them."
            self._refresh_model()
        elif time.monotonic() - started > 60:
            self._prepared_commit = None
            self._export_pending = None
            self._remove_preparation(export)
            self._message = "LichtFeld Studio could not prepare the scene. Check the export status and try again."
            self._preparation_failure = {"id": "preparation:" + project_id, "project": project_id,
                "commitUuid": prepared_commit,
                "status": "error", "kind": "upload", "metadata": {"title": self._operation_title},
                "message": self._message, "failureReason": self._message}
            log_failure("native_preparation_timeout", TimeoutError(self._message), project_id=project_id)
            self._refresh_model()

    def _action_resume(self, job_id):
        job = next((job for job in self.service.snapshot()['jobs'] if job['id'] == job_id), {})
        if job.get('needsAttention'):
            identity = self.service.identity()
            labels = {action: tr('action.' + action) for action in ('cancel', 'retry', 'keep_waiting')}
            actions = {label: action for action, label in labels.items()}
            def selected(label):
                action = actions.get(label)
                if identity != self.service.identity() or self.service.busy or action not in ('retry', 'keep_waiting'):
                    return
                self.service.resume(job_id, **({'keep_waiting': True} if action == 'keep_waiting' else {}))
                self._schedule_poll()
            lf.ui.confirm_dialog(tr('state.error'), tr('confirm.processing_wait'), list(labels.values()), selected)
            return
        self.service.resume(job_id)
        self._schedule_poll()

    def _action_pause(self):
        if self._import_pending and self._import_pending.get("_register"):
            self._import_pending["_register"]["canceled"] = True
        if self._save_pending:
            self._save_pending["canceled"] = True
        if self._export_pending:
            self._cancel_own_export()
            self._export_cancelled = True
            self._message = "Canceling scene preparation…"
        if self._import_pending and self._import_pending.get("_update"):
            update = self._import_pending["_update"]
            update["canceled"] = True
            if update["phase"] == "importing" and Path(update.get("path", "")).suffix == ".scene":
                lf.ui.cancel_gallery_import()
        if self._import_pending and self._import_pending.get("_opening"):
            opening = self._import_pending["_opening"]
            opening["canceled"] = True
            if opening["phase"] == "importing":
                lf.ui.cancel_gallery_import()
        self.service.pause()

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
        except Exception as exc:
            log_failure("stage_download", exc, job_id=job["id"])
            self._release_native_use()
            raise
        self._import_pending = dict(job, _accountIdentity=identity,
            _register={"stage_id": stage_id, "phase": "staging"})
        self._import_detached = False
        self._message = tr("state.downloading", percent=100)
        self._schedule_poll()

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
            if file_stamp(path) != stage.get("projectStamp"):
                raise ValueError("The downloaded project identity or path changed. Prepare the download again.")
            inspection = lf.io.inspect_project(path)
            if str(inspection.project_uuid) != stage.get("projectId"):
                raise ValueError("The downloaded project identity changed. Prepare the download again.")
            previous = index.get_asset(str(inspection.project_uuid))
            if previous and Path(previous.path).resolve() != Path(path).resolve() and Path(previous.path).exists():
                # Never move an existing catalog entry to an unrelated copy.
                raise ValueError(tr("error.link"))
            project, _ = index.register_licht_asset(path, name=job["result"]["title"], inspection=inspection)
            if project is None:
                raise ValueError(index.last_error or tr("error.storage"))
            self._mark_viewing_copy(index, project)
            pending.update(phase="linking", path=path, project=str(inspection.project_uuid),
                operation=self._link_saved_download(job["id"], path, str(inspection.project_uuid)))
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
        stage_id = self.service.stage_download(job["id"])
        self._import_pending = dict(job, _accountIdentity=identity,
            _opening={"phase": "staging", "stage_id": stage_id, "scene": lf.get_scene()})
        self._import_detached = False
        self._import_started = time.monotonic()
        self._message = "Checking downloaded scene…"
        self._schedule_poll()
        return

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
            if file_stamp(expected["path"]) != expected["projectStamp"]:
                raise ValueError("The downloaded project identity or path changed. Prepare the download again.")
            inspection = lf.io.inspect_project(expected["path"])
            if str(inspection.project_uuid) != expected["projectId"]:
                raise ValueError("The downloaded project identity changed. Prepare the download again.")
            project, _ = index.register_licht_asset(expected["path"], name=job["result"]["title"], inspection=inspection)
            if project is None:
                raise ValueError(index.last_error or "The project opened but could not be added to Asset Manager.")
            self._mark_viewing_copy(index, project)
            restore_view(lf, job["result"].get("viewerSettings", {}), environment_path=self.service.environment_path(job))
            operation = self._link_saved_download(job["id"], expected["path"], expected["projectId"])
            job.pop("_native_project")
            job["_link"] = operation
            job["_registered_project"] = {"id": str(project.project_uuid), "path": expected["path"], "jobId": job["id"]}
            self._message = "Project opened. Saving its gallery link…"
            return
        if job.get("_update"):
            self._finish_local_update(job)
            return
        opening = job.get("_opening")
        if opening and opening["phase"] == "staging":
            if self.service.busy:
                return
            if self._import_detached or opening.get("canceled") or not opening["scene"].is_valid() or lf.project_is_dirty():
                self._import_pending = None
                self._message = "Account or project changed. Your download is kept; open it again when ready."
                return
            if lf.is_training_active() or lf.ui.get_import_state().get("active"):
                raise ValueError("Finish training or the current import before opening this download. Your download is kept.")
            current = next(j for j in self.service.snapshot()["jobs"] if j["id"] == job["id"])
            stage = current.get("stagedImport", {})
            if stage.get("id") != opening["stage_id"] or stage.get("state") != "ready":
                raise ValueError(stage.get("message") or "The downloaded scene could not be prepared.")
            from .portable_project import ProjectFile
            # The downloaded subset has the validated portable index. The
            # fresh local identity may use native index compression.
            with open(job["path"], "rb") as source:
                prepared = ProjectFile(source)
                count = sum(node["count"] for node in prepared.manifest["nodes"])
            lf.project_open(stage["projectPath"], keep_asset_manager_open=True)
            job["_native_project"] = {"path": stage["projectPath"], "count": count,
                "projectId": stage["projectId"], "projectStamp": stage["projectStamp"]}
            opening["phase"] = "opened"
            self._import_started = time.monotonic()
            self._message = "Opening .licht project…"
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
    @staticmethod
    def _mark_viewing_copy(index, project):
        if index.update_asset(project.id, viewing_copy=True) is None:
            raise ValueError(index.last_error or tr("error.storage"))

    def _link_saved_download(self, job_id, path, expected_project_id, *, local_fields=None):
        inspected = lf.io.inspect_project(path)
        if str(inspected.project_uuid) != expected_project_id:
            raise ValueError("The project identity changed before linking. Refresh Projects and try again.")
        commit = str(getattr(inspected, "commit_uuid", ""))
        args = (job_id, str(inspected.project_uuid))
        options = {"local_fields": local_fields} if local_fields is not None else {}
        return self.service.link_download(*args, commit, project_path=path, **options)

    def _action_update_local(self, job_id):
        job = next(j for j in self._state["jobs"] if j["id"] == job_id)
        project = self._project_identity()
        link = self._state["links"].get(project[0])
        if not link or link["sceneId"] != job.get("result", {}).get("id"):
            raise ValueError("This download is not linked to the current project.")
        self._confirm = (tr("confirm.pull", title=Path(project[1]).stem),
            lambda: self._begin_local_update(job, project), tr("action.pull"))

    def _staged_nodes(self, path):
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
            job["_local_fields"] = copy.deepcopy(self._pull_overrides[1])
            if self._pull_overrides[4] is not None:
                job["_local_environment_path"] = self._pull_overrides[4]
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
        self._schedule_poll()

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
                scene_id, metadata, identity, publish = self._pull_overrides[:4]
                self._pull_overrides = None
                if identity == self._identity and publish:
                    self.service.edit(scene_id, domain_tokens(current_job["result"]), metadata, project_id=project[0])
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
                lf.load_gallery_scene(self._staged_nodes(stage["path"]), Path(stage["path"]).stem, hidden=True)
            else:
                lf.load_file(stage["path"])
            update["phase"] = "importing"
            self._import_started = time.monotonic()
            return
        if update["phase"] == "importing":
            if incoming is None:
                if lf.ui.get_import_state().get("error") or time.monotonic() - self._import_started > 60:
                    raise ValueError("LichtFeld Studio could not open the gallery update. Your local splats remain.")
                return
            lf.ui.dismiss_import()
            update["incoming"] = incoming.uuid
            update["old_nodes"] = [n.uuid for n in self._visible_splats() if n.uuid != incoming.uuid]
            update["phase"] = "save_before_backup"
            self._save_current_project(lambda: self._prepare_update_backup(job), expected_project=project)
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
        recovery = "Your recovery copy is available in the recovery folder."
        try:
            project = update["project"]
            # Reopen only the unchanged generation that preceded the update.
            if self._project_identity() == project and file_stamp(project[1]) == update["stamp"]:
                lf.project_open(project[1], discard_changes=True, keep_asset_manager_open=True)
                recovery = "Your saved local project is being reopened. Its recovery copy is also available."
        except Exception as exc:
            log_failure("recover_failed_update", exc, project_id=update.get("project", ("", ""))[0])
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
            update["link_operation"] = self._link_saved_download(job["id"], update["project"][1], update["project"][0],
                **({"local_fields": job["_local_fields"]} if "_local_fields" in job else {}))
        except Exception as exc:
            log_failure("link_saved_download", exc, job_id=job["id"])
            raise ValueError("The project was updated and its recovery copy was kept, but the gallery link could not be saved. Refresh your gallery before continuing.") from exc
        self._message = "Project updated. Saving its gallery link…"

    def _apply_local_update(self, scene, incoming, job, update):
        # A saved recovery copy exists, and no edits have occurred since it was made.
        environment_path = (job["_local_environment_path"] if "_local_environment_path" in job
                            else self.service.environment_path(job))
        metadata = job.get("_local_fields", job["result"])
        restore_view(lf, metadata.get("viewerSettings", {}), environment_path=environment_path)
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
        title = metadata["title"]
        if scene.get_node(title) is not None:
            title += " (gallery " + incoming.uuid[:8] + ")"
        scene.rename_node(incoming.name, title)
        if self._project_identity() != update["project"] or file_stamp(update["project"][1]) != update["stamp"]:
            raise ValueError("The project identity or path changed before saving. Your recovery copy was kept.")
        if not lf.project_save(wait=False):
            raise ValueError("The updated project could not be saved. Your recovery copy is available in the recovery folder.")


_controller = None

_LOCAL_FILE_PROBLEM_STATUSES = {
    "MISSING",
    "READING",
    "UNVERIFIED",
    "UNREADABLE",
    "UNSUPPORTED",
    "REPAIR_ONLY",
    "UNSUPPORTED_NEWER",
}
_LOCAL_FILE_PROBLEM_LABELS = {
    "REPAIR_ONLY": "projects.status.needs_repair",
    "UNSUPPORTED_NEWER": "projects.status.newer_version",
}

def get_gallery_controller():
    global _controller
    if _controller is None:
        _controller = GalleryController()
    return _controller

def stored_local_scene(link, details=None):
    """Last saved Studio fields for a linked item, without opening the project."""
    fields = copy.deepcopy((link or {}).get("localFields") or (link or {}).get("sharedFields") or {})
    local = dict(details or {})
    for key in ("title", "description"):
        local.setdefault(key, fields.get(key, ""))
    local.setdefault("viewerSettings", fields.get("viewerSettings") or {})
    return local


def _change_labels(groups):
    return list(dict.fromkeys(name for row in groups for name in row.get("fields") or [row["label"]] if name))


def conflict_groups(asset, link, local, remote, *, apply_only=False):
    import json
    baseline = link.get("sharedFields", {})
    local_view, remote_view = local.get("viewerSettings", {}), remote.get("viewerSettings", {})
    parts = [
        ("text", {k: local.get(k, "") for k in ("title", "description")}, {k: remote.get(k, "") for k in ("title", "description")}, {k: baseline.get(k, "") for k in ("title", "description")}),
        ("view", {k: v for k, v in local_view.items() if k != "cameraPath"}, {k: v for k, v in remote_view.items() if k != "cameraPath"}, {k: v for k, v in baseline.get("viewerSettings", {}).items() if k != "cameraPath"}),
        ("track", local_view.get("cameraPath"), remote_view.get("cameraPath"), baseline.get("viewerSettings", {}).get("cameraPath")),
    ]
    rows = []
    def text(value):
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(", ", ": "))
    for identifier, mine, gallery, base in parts:
        if mine == gallery:
            continue
        mine_value, gallery_value = text(mine), text(gallery)
        values = tr("conflict.values", mine=mine_value, gallery=gallery_value)
        difference = values
        fields = [tr({"text": "conflict.text", "view": "conflict.view", "track": "conflict.track"}[identifier])]
        if identifier == "text":
            changed = [key for key in ("title", "description") if mine.get(key) != gallery.get(key)]
            fields = [tr("review.title" if key == "title" else "conflict.description") for key in changed]
            difference = tr("conflict.values", mine=" · ".join(text(mine.get(key, "")) for key in changed),
                            gallery=" · ".join(text(gallery.get(key, "")) for key in changed))
        elif identifier == "view":
            native_labels = {"exposure": "main_panel.color_exposure", "tonemapping": "main_panel.color_tonemapping",
                             "background": "main_panel.background", "antialiasing": "main_panel.mip_filter", "shDegree": "main_panel.sh_degree",
                             "renderMode": "main_panel.raster_backend", "environment": "main_panel.environment", "verticalFov": "main_panel.fov"}
            names = [lf.ui.tr(native_labels[key]) if key in native_labels else tr("conflict.camera" if key == "camera" else "conflict.view")
                     for key in sorted(set(mine) | set(gallery)) if mine.get(key) != gallery.get(key)]
            difference = tr("conflict.changed_settings", parts=", ".join(dict.fromkeys(names)))
        elif identifier == "track":
            difference = tr("conflict.track_counts", mine=len((mine or {}).get("keyframes", [])),
                            gallery=len((gallery or {}).get("keyframes", [])))
        rows.append(dict(id=identifier, label=tr({"text": "conflict.text", "view": "conflict.view", "track": "conflict.track"}[identifier]),
            mine_value=mine_value, gallery_value=gallery_value, fields=fields,
            difference=difference, values=values,
            choice="gallery" if apply_only or mine == base else "mine",
            can_both=identifier == "track" and bool(mine and gallery)))
    content_changed = (link.get("contentRevision") != remote.get("contentRevision")
        or not apply_only and (not link.get("commitUuid") or asset.get("commit_uuid") != link.get("commitUuid")))
    if content_changed:
        mine, gallery = tr("conflict.local_content"), tr("conflict.gallery_content")
        rows.append(dict(id="content", label=tr("conflict.content"), mine_value=mine, gallery_value=gallery,
            fields=[tr("conflict.content")],
            difference=tr("conflict.values", mine=mine, gallery=gallery), values=tr("conflict.values", mine=mine, gallery=gallery),
            choice="mine", can_both=False))
    return rows


def asset_sync_state(project=None, link=None, scene=None, jobs=(), *, checked=False,
                     phase="idle", storage_issue=False, cached_projection=None, established=True):
    """Three independent facts and one deterministic badge; never compare clocks."""
    project = project or {}
    identifier = project.get("project_uuid", project.get("id", ""))
    relationship = "linked" if link else "unlinked" if project else "remote_only"
    if project.get("status") in ("IDENTITY_CONFLICT", "IDENTITY_MISMATCH", "DUPLICATE", "AMBIGUOUS", "DIVERGED_COPIES"):
        relationship = "identity_ambiguous"
    elif project.get("status") in _LOCAL_FILE_PROBLEM_STATUSES:
        relationship = "local_file_problem"
    elif link and (link.get("remoteDeleted") or (scene and scene.get("status") == "deleted") or (checked and scene is None)):
        relationship = "remote_deleted"
    elif link and not project.get("exists", True):
        relationship = "local_missing"
    freshness = "unknown"
    if link and link.get("commitUuid") and project.get("commit_uuid") and scene and all(link.get(key) and scene.get(key) for key in ("contentRevision", "metadataRevision")):
        # Links written by older builds still include visibility in their saved fields.
        local = (project["commit_uuid"] != link["commitUuid"] or
                 "localFields" in link and {k: v for k, v in link["localFields"].items() if k != "visibility"} !=
                 {k: v for k, v in link.get("sharedFields", {}).items() if k != "visibility"})
        remote = any(scene[key] != link[key] for key in ("contentRevision", "metadataRevision"))
        freshness = "diverged" if local and remote else "local" if local else "remote" if remote else "equal"
    scene_id = (link or scene or {}).get("sceneId", (scene or {}).get("id"))
    matching = [j for j in jobs if (j.get("status") not in ("completed", "canceled") or j.get("localUpdate", {}).get("interrupted") or j.get("localUpdate", {}).get("state") == "failed"
                or j.get("settingsEnvironment", {}).get("state") == "preparing") and not j.get("retired") and
                ((identifier and j.get("project") == identifier) or (scene_id and
                (j.get("sceneId") == scene_id or j.get("metadata", {}).get("replaceSceneId") == scene_id)))]
    job = matching[-1] if matching else {}
    activity = phase
    if job:
        status = job.get("status")
        activity = "error" if status == "conflict" else transfer_phase(job)
        if status == "conflict":
            freshness = "diverged"
    remote_content = bool(link and scene and link.get("contentRevision") != scene.get("contentRevision"))
    presentation = bool(link and scene and scene.get("presentationRevision") and scene["presentationRevision"] != (link.get("acknowledgedPresentationRevision") or link.get("metadata", {}).get("presentationRevision", "")))
    active = activity in ("preparing", "queued", "uploading", "processing", "downloading", "applying")
    if relationship in ("identity_ambiguous", "local_file_problem"):
        visible = "error"
    elif active:
        visible = activity
    elif freshness == "diverged":
        visible = "diverged"
    elif activity in ("error", "paused", "interrupted"):
        visible = activity
    elif relationship in ("local_missing", "remote_deleted"):
        visible = relationship
    elif not established:
        visible = "not_checked"
    elif relationship == "linked":
        visible = "remote_content" if freshness == "remote" and remote_content else "presentation" if freshness == "equal" and presentation else freshness
    else:
        visible = relationship
    if cached_projection and not link and not scene and not jobs and not storage_issue and relationship == "unlinked":
        cached = cached_projection.get("state")
        if cached in {"equal", "local", "remote", "diverged", "unknown", "error", "paused", "interrupted", "local_missing", "remote_deleted"}:
            relationship = "linked"
            visible = cached
    icons = {"unlinked": "cloud", "equal": "cloud-check", "local": "cloud-up",
             "remote": "cloud-down", "diverged": "cloud-bang", "remote_only": "cloud-dotted",
             "queued": "ring", "paused": "pause", "interrupted": "pause",
             "error": "error", "local_missing": "cloud-bang", "remote_deleted": "cloud-strike",
             "unknown": "cloud-dotted", "not_checked": "cloud-dotted",
             "presentation": "cloud-check", "remote_content": "cloud-down"}
    tones = {"equal": "success", "local": "primary", "remote": "primary", "remote_only": "primary",
             "diverged": "warning", "error": "error", "local_missing": "warning", "remote_deleted": "text_dim",
             "interrupted": "text_dim", "processing": "primary", "remote_content": "primary"}
    reason = localize_message(job.get("message") or project.get("error", ""))
    if relationship == "local_file_problem":
        reason = project.get("error")
        if not reason:
            from .asset_manager_panel import tr as asset_tr
            label = _LOCAL_FILE_PROBLEM_LABELS.get(project["status"])
            reason = asset_tr(label) if label else project["status"]
    health = project.get("status", "")
    health_tone = ("warning" if health in ("MISSING", "IDENTITY_MISMATCH", "IDENTITY_CONFLICT", "DUPLICATE", "AMBIGUOUS")
                   else "error" if health in _LOCAL_FILE_PROBLEM_STATUSES else "")
    icon = "ring" if active else icons.get(visible, "cloud")
    if relationship == "local_file_problem":
        icon = "cloud-bang"
    change_fields, change_detail = [], ""
    if visible in ("remote", "remote_content", "diverged") and scene and link:
        groups = conflict_groups(project, link, stored_local_scene(link), scene,
                                 apply_only=visible != "diverged")
        change_fields = _change_labels(groups)
        change_detail = "\n".join(row["difference"] for row in groups if row.get("difference"))
    result = dict(relationship=relationship, freshness=freshness, activity=activity, state=visible,
                icon=icon, tone="primary" if active else tones.get(visible, "text_dim"),
                health_icon="bang" if health_tone else "", health_tone=health_tone,
                active=active, jobId=job.get("id", ""), job=job, reason=reason,
                change_fields=change_fields, change_summary=", ".join(change_fields),
                change_detail=change_detail,
                linked=bool(link), sceneReady=bool(scene and scene.get("status", "ready") == "ready"),
                established=established, cachedUnverified=bool(cached_projection and not link), storage_issue=storage_issue, viewingCopy=bool(project.get("viewing_copy") or (link or {}).get("viewingCopy")),
                remoteContent=remote_content, presentationChanged=presentation,
                progress=min(100, int(100 * job.get("completed", 0) / max(1, job.get("total", 0)))),
                attention=visible in ("diverged", "error", "interrupted", "local_missing", "remote_deleted")
                and project.get("status") not in ("READING", "UNSUPPORTED_NEWER"))
    actions = gallery_actions(project, result)
    result["action"] = actions[0]["id"] if actions else ""
    result["actions"] = actions
    return result

def combine_camera_tracks(mine, portal):
    """Keep both complete tracks in order in the native camera path."""
    result = copy.deepcopy(mine)
    offset = float(mine.get("duration", 0))
    appended = copy.deepcopy(portal.get("keyframes", []))
    for frame in appended:
        frame["time"] = float(frame["time"]) + offset
    result["keyframes"] = result.get("keyframes", []) + appended
    result["duration"] = offset + float(portal.get("duration", 0))
    return result
