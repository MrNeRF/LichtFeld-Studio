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
from .gallery_sync_steps import LocalUpdateSteps, DownloadOpenSteps, PublishSteps
from .gallery_sync_facts import asset_sync_state as _asset_sync_state, conflict_groups as _conflict_groups, stored_local_scene, combine_camera_tracks
from .gallery_view import capture_view, restore_view
from .portal_gallery import domain_tokens, UNSUPPORTED_PORTAL
from . import gallery_preparation
from .portal_security import redact, safe_filename, checked_portal_url
from .gallery_logging import failure as log_failure, safe_url, stage as log_stage

class GalleryController:
    @property
    def service(self):
        return self._service

    @service.setter
    def service(self, value):
        self._service = value
        for machine in (getattr(self, "_local_update_steps", None),
                        getattr(self, "_download_open_steps", None),
                        getattr(self, "_publish_steps", None)):
            if machine is not None:
                machine.service = value

    def __init__(self):
        self.service = get_gallery_sync()
        self._state = self.service.snapshot()
        self._identity = self._state["identity"]
        self._confirm = None
        self._ui_message = ""
        self._message_source = None
        self._failure_notice = ""
        self._save_pending = None
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
        self._undo_pull = None
        self._settings_pending = None
        self._decision_pending = False
        self._last_canceled = False
        self._update_queue = []
        self._batch_rows = []
        self._batch_retries = {}
        self._batch_current = None
        self._local_update_steps = LocalUpdateSteps(
            self.service, lf, time.monotonic,
            project_identity=lambda: self._project_identity(),
            current_identity=lambda: self._identity,
            visible_splats=lambda: self._visible_splats(),
            staged_nodes=lambda path: self._staged_nodes(path),
            acquire_native_use=lambda job_id: self._acquire_native_use(job_id),
            save_project=lambda continuation, **kwargs: self._save_current_project(continuation, **kwargs),
            save_pending=lambda: bool(self._save_pending),
            cancel_save=lambda: self._cancel_current_project_save(),
            link_saved_download=lambda *args, **kwargs: self._link_saved_download(*args, **kwargs),
            refresh_model=lambda: self._refresh_model(),
            schedule_poll=lambda: self._schedule_poll(),
            apply_update=lambda *args: self._apply_local_update(*args),
            recover_update=lambda *args: self._recover_failed_update(*args),
            set_undo_pull=lambda value: setattr(self, "_undo_pull", value),
            message_changed=self._select_step_message)
        self._download_open_steps = DownloadOpenSteps(
            self.service, lf, time.monotonic,
            acquire_native_use=lambda job_id: self._acquire_native_use(job_id),
            release_native_use=lambda: self._release_native_use(),
            mark_viewing_copy=lambda index, project: self._mark_viewing_copy(index, project),
            link_saved_download=lambda *args, **kwargs: self._link_saved_download(*args, **kwargs),
            refresh_model=lambda: self._refresh_model(),
            schedule_poll=lambda: self._schedule_poll(),
            message_changed=self._select_step_message)
        self._publish_steps = PublishSteps(
            self.service, lf, time.monotonic, lambda *args: asset_sync_state(*args),
            model_state=lambda: self._state,
            project_identity=lambda: self._project_identity(),
            visible_splats=lambda: self._visible_splats(),
            details=lambda value: self._details(value),
            save_project=lambda continuation: self._save_current_project(continuation),
            start_saved=lambda *args, **kwargs: self._publish_saved(*args, **kwargs),
            patch_saved_update=lambda *args, **kwargs: self._patch_saved_update(*args, **kwargs),
            pin_publish_preview=lambda *args: self._pin_publish_preview(*args),
            cleanup_preparation=lambda export: self._remove_preparation(export),
            refresh_model=lambda: self._refresh_model(),
            schedule_poll=lambda: self._schedule_poll(),
            set_operation=lambda project, title: self._set_publish_operation(project, title),
            operation_title=lambda: self._operation_title,
            set_last_canceled=lambda value: setattr(self, "_last_canceled", value),
            set_upload_format=lambda value: setattr(self, "upload_format", value),
            message_changed=self._select_step_message)
        from .ui import RuntimeState
        RuntimeState.account_state.subscribe(self._account_changed)
        self._publish_runtime_state(self.snapshot())

    def _select_step_message(self, machine):
        self._message_source = machine

    @property
    def _message(self):
        return self._message_source.message if self._message_source is not None else self._ui_message

    @_message.setter
    def _message(self, value):
        self._ui_message = value
        self._message_source = None

    def _set_publish_operation(self, project, title):
        self._operation_project = project
        self._operation_title = title

    def _active_import_steps(self):
        return self._local_update_steps if self._local_update_steps.pending else self._download_open_steps

    def _active_preparation_steps(self):
        if self._local_update_steps.pending:
            return self._local_update_steps
        if self._download_open_steps.pending:
            return self._download_open_steps
        return self._publish_steps

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
        self._publish_steps.reupload_reason = None
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
        self._review_publish(scene, details, upload_format, publish_as_new, update=update,
                             expected_commit=str(asset.get("commit_uuid") or getattr(lf.io.inspect_project(path), "commit_uuid", "")))
        self._schedule_poll()

    def publish_unlinked_scene(self, asset, details, upload_format):
        self._check_identity()
        self._refresh_model()
        if self._panel_busy() or lf.project_poll_write().get("path"):
            raise ValueError(tr("error.project_changed"))
        metadata = self._details(details)
        metadata["viewerSettings"] = capture_view(lf)
        metadata["useEmbeddedPreview"] = bool(details.get("useEmbeddedPreview"))
        self.upload_format = upload_format
        self._publish_steps.start_live(metadata, upload_format, unlinked=True)


    def _publish_closed_asset(self, asset, details, upload_format, *, update, publish_as_new, handoff=None):
        return self._publish_steps.start_closed(asset, details, upload_format, update=update, publish_as_new=publish_as_new, handoff=handoff)

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
        from .gallery_project_facts import saved_content_stamp
        saved_stamp = (saved_content_stamp(asset["path"])
                       if link.get("contentStamp") and not reviewed_dirty and Path(asset["path"]).is_file() else "")
        groups = conflict_groups(asset, link, local, remote, apply_only=apply_only,
                                 saved_stamp=saved_stamp, closed=not current_open)
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
                track_source = remote_view if decisions.get("track") == "gallery" else apply_local_view
                track = copy.deepcopy(track_source.get("cameraPath"))
                if decisions.get("track") == "both":
                    if not apply_local_view.get("cameraPath"):
                        raise ValueError(tr("error.project_changed"))
                    track = combine_camera_tracks(apply_local_view["cameraPath"], remote_view["cameraPath"])
                if track is not None:
                    view["cameraPath"] = track
                elif decisions.get("view") != "gallery":
                    view.pop("cameraPath", None)
                metadata["viewerSettings"] = view

                def start():
                    if decisions.get("content") == "gallery":
                        local_environment_path = None
                        if decisions.get("view") != "gallery":
                            local_environment_path = str(lf.get_render_settings().environment_map_path) if view.get("environment") else ""
                        self._local_update_steps.overrides = (scene["id"], metadata, identity, publish, local_environment_path)
                        try:
                            self.pull_asset(asset, scene)
                        except Exception:
                            self._local_update_steps.overrides = None
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
                    if self._local_update_steps.overrides and self._local_update_steps.overrides[0] == job["result"]["id"]:
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
        state["preparationFailure"] = dict(self._publish_steps.preparation_failure, nativePreparation=True) if self._publish_steps.preparation_failure else None
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
                    phase=self.phase(), preparationProgress=self._active_preparation_steps().progress,
                    undoPull=copy.deepcopy(self._undo_pull), undoHistory=undo_history,
                    operationProject=self._operation_project,
                    pulledProject=copy.deepcopy(self._download_open_steps.pulled_project), reuploadReason=copy.deepcopy(self._publish_steps.reupload_reason))

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
        if self._active_import_steps().pending or getattr(self, "_settings_pending", None):
            return "applying"
        if self._publish_steps.pending or self._save_pending:
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
        return self.service.busy or self._decision_pending or bool(self._publish_steps.pending or self._active_import_steps().pending or self._save_pending or self._settings_pending or self._native_use)

    def _metadata_busy(self):
        self._release_native_use()
        return (getattr(self.service, "metadata_busy", self.service.busy) or self._decision_pending
                or bool(self._publish_steps.pending or self._active_import_steps().pending or self._save_pending or self._settings_pending or self._native_use))

    def _check_identity(self):
        identity = self.service.identity()
        if identity == self._identity:
            return False
        self._identity = identity
        self._state = dict(self._state, scenes=[], links={}, jobs=[], posters={}, identity=identity)
        self._last_snapshot = None
        self._publish_steps.reupload_reason = None
        self._publish_steps.preparation_failure = None
        self._failure_notice = ""
        self._download_open_steps.pulled_project = None
        self._update_queue = []
        self._batch_rows = []
        self._batch_retries = {}
        self._batch_current = None
        self._pull_requests.clear()
        self._cancel_requests.clear()
        self._open_continuation = None
        self._local_update_steps.overrides = None
        self._undo_pull = None
        self._settings_pending = None
        self._after_service = None
        self._decision_pending = False
        self.checked_at = 0
        self.offline = False
        self._refresh_pending = False
        self._refresh_requested = bool(identity[-1])
        self.service.pause()
        if self._publish_steps.pending:
            self._cancel_own_export()
            self._publish_steps.cancelled = True
        if self._save_pending:
            self._save_pending["canceled"] = True
        if self._active_import_steps().pending:
            # Keep tracking owned native work, but never register or link the
            # previous account's download after a switch.
            self._active_import_steps().detached = True
        self._message = "Account changed. Refresh to load your gallery."
        if self._active_import_steps().pending:
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
        if self._native_use is None or self._active_import_steps().pending:
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
        update_pending = self._local_update_steps.pending
        try:
            self._check_identity()
            if self._save_pending:
                self._finish_current_project_save()
                if self._save_pending:
                    return
            if self._publish_steps.pending:
                self._finish_export()
            if self._active_import_steps().pending:
                self._finish_import()
                if self._active_import_steps().pending:
                    native = lf.ui.get_import_state()
                    # The model refreshes when the service version changes.
                    # Progress painting must not clone the full history per frame.
                    current = next((j for j in self._state["jobs"] if j["id"] == self._active_import_steps().pending["id"]), {})
                    staged = current.get("stagedImport", {})
                    self._active_preparation_steps().progress = (100 * native.get("progress", 0) if native.get("active") else
                        min(100, 100 * staged.get("completed", 0) / max(1, staged.get("total", 0))))
        except Exception as exc:
            log_failure("advance_phases", exc)
            self._discard_update_preview()
            self._fail_settings_apply(exc)
            pending = self._active_import_steps().pending
            if (pending and not self.service.busy and pending.get("_accountIdentity") == self.service.identity()
                    and callable(getattr(self.service, "fail_local_update", None))):
                self.service.fail_local_update(pending["id"], friendly_error(exc))
            self._publish_steps.pending = self._local_update_steps.pending = self._download_open_steps.pending = self._save_pending = None
            self._message = friendly_error(exc)
            self._failure_notice = self._message
            self._refresh_model()
        finally:
            if update_pending and self._local_update_steps.pending is None:
                update = update_pending.get("_update", {})
                try:
                    self.service.finish_update_download(update_pending["id"], update.get("stage_id"))
                except Exception as exc:
                    log_failure("update_download_cleanup", exc, job_id=update_pending["id"])
            self._release_native_use()

    def _discard_update_preview(self):
        return self._local_update_steps.discard_preview()

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
        self._active_preparation_steps().progress = 0
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

    def _cancel_current_project_save(self):
        if self._save_pending:
            self._save_pending["canceled"] = True

    def _review_publish(self, scene, details, upload_format, publish_as_new, *, update=False, expected_commit=None):
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
                      upload_format=upload_format, update=update,
                      save_project=bool(details.get("saveProject", True)), expected_commit=expected_commit)

    def _publish(self, metadata, *, expected_project=None, environment_source=None, upload_format="studio", update=False,
                 save_project=True, expected_commit=None):
        return self._publish_steps.start(metadata, expected_project=expected_project, environment_source=environment_source, upload_format=upload_format, update=update, save_project=save_project, expected_commit=expected_commit)

    def _patch_saved_update(self, metadata, project_id, path, *, update, expected_commit=None):
        return self._publish_steps.patch_saved_update(metadata, project_id, path, update=update, expected_commit=expected_commit)

    def _publish_saved(self, metadata, project_id, path, identity, environment_source=None, upload_format="studio", *, update=False,
                       expected_commit=None):
        return self._publish_steps.start_saved(metadata, project_id, path, identity, environment_source, upload_format, update=update, expected_commit=expected_commit)

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
        return self._publish_steps.owns_export(state)

    def _cancel_own_export(self):
        return self._publish_steps.cancel_export()

    def _remove_preparation(self, export):
        return self._publish_steps.remove_preparation(export)

    def _finish_export(self):
        return self._publish_steps.advance()

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
        if self._save_pending:
            self._save_pending["canceled"] = True
        if self._publish_steps.pending:
            self._cancel_own_export()
            self._publish_steps.cancelled = True
            self._message = "Canceling scene preparation…"
        self._local_update_steps.cancel()
        self._download_open_steps.cancel()
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
        return self._download_open_steps.start_keep(job, identity)

    def _finish_register_download(self, job):
        return self._download_open_steps.advance_keep(job)

    def _open_download(self, job, identity):
        return self._download_open_steps.start(job, identity)

    def _finish_import(self):
        if self._local_update_steps.pending:
            job = self._local_update_steps.pending
            if job.get("_accountIdentity") is not None and self.service.identity() != job["_accountIdentity"]:
                self._local_update_steps.detached = True
            return self._finish_local_update(job)
        return self._download_open_steps.advance()

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
        return self._local_update_steps.begin(job, project)

    def _finish_local_update(self, job):
        return self._local_update_steps.advance(job)

    def _prepare_update_backup(self, job):
        return self._local_update_steps.prepare_backup(job)

    def _recover_failed_update(self, update, exc):
        return self._local_update_steps.recover_failure(update, exc)

    def _finish_update_save(self, job):
        return self._local_update_steps.finish_save(job)

    def _apply_local_update(self, scene, incoming, job, update):
        return self._local_update_steps.apply(scene, incoming, job, update)

_controller = None

def get_gallery_controller():
    global _controller
    if _controller is None:
        _controller = GalleryController()
    return _controller

def conflict_groups(asset, link, local, remote, *, apply_only=False, saved_stamp="", closed=False):
    return _conflict_groups(asset, link, local, remote, apply_only=apply_only,
                            saved_stamp=saved_stamp, closed=closed, translate=tr)

def asset_sync_state(project=None, link=None, scene=None, jobs=(), *, checked=False,
                     phase="idle", storage_issue=False, cached_projection=None, established=True):
    return _asset_sync_state(project, link, scene, jobs, checked=checked, phase=phase,
                             storage_issue=storage_issue, cached_projection=cached_projection,
                             established=established, translate=tr)
