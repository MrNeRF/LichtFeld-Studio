# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native steps of a gallery update, ordered around a durable recovery copy."""
from __future__ import annotations

import copy
import uuid
import time
from pathlib import Path

import lichtfeld as lf
from .gallery_sync import file_stamp, friendly_error
from .gallery_view import restore_view
from .portal_gallery import domain_tokens, UNSUPPORTED_PORTAL
from .gallery_logging import failure as log_failure, safe_url, stage as log_stage
from .gallery_messages import tr
from . import gallery_preparation


class LocalUpdateSteps:
    """staging -> save_before_backup -> backup -> importing -> applying -> save_updated -> linking.

    The user's project is saved and backed up before a gallery preview enters
    the live scene. Old splats and the view are replaced only after the backup
    is ready and the saved generation and stamp still match. Cancel through
    backup leaves the saved project and its file untouched.
    Apply is synchronous; failure reopens that saved file. Cancel after the
    updated save keeps the updated project and recovery copy. A submitted link
    may still complete and must be checked by refreshing the gallery.
    """

    def __init__(self, service, app, clock, *, project_identity, current_identity,
                 visible_splats, staged_nodes, acquire_native_use, save_project,
                 save_pending, cancel_save, link_saved_download, refresh_model, schedule_poll,
                 apply_update, recover_update, set_undo_pull, message_changed):
        self.service = service
        self.app = app
        self.clock = clock
        self.project_identity = project_identity
        self.current_identity = current_identity
        self.visible_splats = visible_splats
        self.staged_nodes = staged_nodes
        self.acquire_native_use = acquire_native_use
        self.save_project = save_project
        self.save_pending = save_pending
        self.cancel_save = cancel_save
        self.link_saved_download = link_saved_download
        self.refresh_model = refresh_model
        self.schedule_poll = schedule_poll
        self.apply_update = apply_update
        self.recover_update = recover_update
        self.set_undo_pull = set_undo_pull
        self.message_changed = message_changed
        self.pending = None
        self.detached = False
        self.started = None
        self.overrides = None
        self.progress = 0
        self.save_in_progress = False
        self.__message = ""

    @property
    def message(self):
        return self.__message

    @property
    def _message(self):
        return self.__message

    @_message.setter
    def _message(self, value):
        self.__message = value
        self.message_changed(self)

    def cancel(self):
        if self.pending and self.pending.get("_update"):
            update = self.pending["_update"]
            update["canceled"] = True
            if update["phase"] == "save_before_backup" and self.save_in_progress:
                self.cancel_save()
            if update["phase"] == "importing" and Path(update.get("path", "")).suffix == ".scene":
                self.app.ui.cancel_gallery_import()

    def discard_preview(self):
        update = (self.pending or {}).get("_update", {})
        if update.get("phase") not in ("save_before_backup", "backup") or not update.get("incoming"):
            return
        try:
            if self.project_identity() != update["project"] or self.app.ui.get_import_state().get("active"):
                return
            scene = self.app.get_scene()
            incoming = scene.get_node_by_uuid(update["incoming"])
            if incoming is not None:
                scene.remove_node(incoming.name)
        except Exception as exc:
            # Never hide the original failure or touch another project's nodes.
            log_failure("discard_update_preview", exc)

    def begin(self, job, project):
        if self.overrides and self.overrides[0] == job.get("result", {}).get("id"):
            job = copy.deepcopy(job)
            job["_local_fields"] = copy.deepcopy(self.overrides[1])
            if self.overrides[4] is not None:
                job["_local_environment_path"] = self.overrides[4]
        if self.project_identity() != project:
            raise ValueError("The current project changed. Review it before updating.")
        if self.app.is_training_active() or self.app.ui.get_import_state().get("active"):
            raise ValueError("Finish training or the current import before updating this project.")
        if any(n.locked for n in self.visible_splats()):
            raise ValueError("Unlock the visible splats before updating this project.")
        if job.get("kind") != "download" or job["status"] != "completed":
            raise ValueError("Finish downloading this scene first.")
        self.acquire_native_use(job["id"])
        stage_id = self.service.stage_download(job["id"], for_update=True)
        self.pending = dict(job, _accountIdentity=self.current_identity(),
            _update={"project": project, "phase": "staging", "stage_id": stage_id})
        self.detached = False
        self.started = self.clock()
        self.save_in_progress = False
        self._message = "Preparing the gallery update…"
        self.schedule_poll()

    def advance(self, job):
        update = job["_update"]
        project = update["project"]
        if self.save_pending():
            return
        if self.project_identity() != project:
            self.pending = None
            self._message = ("Project changed after the gallery update. Its recovery copy was kept; review the saved project before linking it."
                if update["phase"] in ("save_updated", "linking") else "Project changed. The gallery update was not applied.")
            return
        if update["phase"] == "save_updated":
            self.finish_save(job)
            return
        # Wait for an owned native import to finish before hiding its preview.
        if update["phase"] == "importing" and self.app.ui.get_import_state().get("active"):
            return
        scene = self.app.get_scene()
        incoming = scene.get_node(Path(update["path"]).stem) if update.get("path") else None
        if incoming is not None and update["phase"] == "importing":
            self.app.set_node_visibility(incoming.name, False)
        if self.detached or update.get("canceled"):
            if incoming is not None and Path(update.get("path", "")).suffix == ".scene" and update["phase"] in ("importing", "save_before_backup", "backup"):
                scene.remove_node(incoming.name)
            self.pending = None
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
            self.pending = None
            if current_job.get("localUpdate", {}).get("backupPath"):
                self.set_undo_pull({"path": project[1], "backup": current_job["localUpdate"]["backupPath"],
                    "stamp": file_stamp(project[1]), "identity": self.current_identity()})
            if self.overrides:
                scene_id, metadata, identity, publish = self.overrides[:4]
                self.overrides = None
                if identity == self.current_identity() and publish:
                    self.service.edit(scene_id, domain_tokens(current_job["result"]), metadata, project_id=project[0])
            self._message = "Linked project updated. Your previous local work is kept in its recovery copy."
            self.refresh_model()
            return
        if update["phase"] == "staging":
            stage = current_job.get("stagedImport", {})
            if stage.get("id") != update["stage_id"] or stage.get("state") != "ready":
                raise ValueError(stage.get("message") or self.service.message)
            update["path"] = stage["path"]
            if scene.get_node(Path(stage["path"]).stem):
                raise ValueError("The update preview already exists. Download the gallery item again.")
            if Path(stage["path"]).suffix not in (".scene", ".licht"):
                raise ValueError("The downloaded project identity or path changed. Prepare the download again.")
            update["phase"] = "save_before_backup"
            self._message = "Saving your current project…"
            return
        if update["phase"] == "save_before_backup":
            if self.app.project_is_dirty():
                if not self.save_in_progress:
                    self.save_in_progress = True
                    self.save_project(lambda: self.prepare_backup(job), expected_project=project)
            else:
                self.prepare_backup(job)
            return
        if update["phase"] == "backup":
            backup = current_job.get("localUpdate", {})
            if backup.get("id") != update.get("backup_id") or backup.get("state") != "ready":
                raise ValueError(backup.get("message") or self.service.message)
            if (self.app.project_is_dirty() or self.app.project_poll_write()["generation"] != update["generation"] or
                    file_stamp(project[1]) != update["stamp"]):
                raise ValueError("Your local project changed during preparation. Its splats were kept. Review it and try the update again.")
            path = update["path"]
            if Path(path).suffix == ".scene":
                self.app.load_gallery_scene(self.staged_nodes(path), Path(path).stem, hidden=True)
            else:
                self.app.load_file(path)
            update["phase"] = "importing"
            self.started = self.clock()
            return
        if update["phase"] == "importing":
            if incoming is None:
                if self.app.ui.get_import_state().get("error") or self.clock() - self.started > 60:
                    raise ValueError("LichtFeld Studio could not open the gallery update. Your local splats remain.")
                return
            self.app.ui.dismiss_import()
            update["incoming"] = incoming.uuid
            update["old_nodes"] = [n.uuid for n in self.visible_splats() if n.uuid != incoming.uuid]
            update["phase"] = "applying"
        backup = current_job.get("localUpdate", {})
        if backup.get("id") != update["backup_id"] or backup.get("state") != "ready":
            raise ValueError(backup.get("message") or self.service.message)
        if (self.app.project_poll_write()["generation"] != update["generation"] or
                file_stamp(project[1]) != update["stamp"]):
            raise ValueError("Your local project changed during preparation. Its splats were kept. Review it and try the update again.")
        incoming = scene.get_node_by_uuid(update["incoming"])
        if incoming is None:
            raise ValueError("The downloaded preview changed. Your local splats were kept.")
        update["phase"] = "applying"
        try:
            self.apply_update(scene, incoming, job, update)
        except Exception as exc:
            self.recover_update(update, exc)
        update["phase"] = "save_updated"
        self._message = "Saving the updated project…"

    def prepare_backup(self, job):
        update = job["_update"]
        project = update["project"]
        update["generation"] = self.app.project_poll_write()["generation"]
        update["stamp"] = file_stamp(project[1])
        update["backup_id"] = self.service.prepare_local_update(job["id"], project[0], project[1], update["stamp"])
        self.save_in_progress = False
        update["phase"] = "backup"
        self._message = "Keeping a recovery copy before replacing local splats…"

    def recover_failure(self, update, exc):
        recovery = "Your recovery copy is available in the recovery folder."
        try:
            project = update["project"]
            # Reopen only the unchanged generation that preceded the update.
            if self.project_identity() == project and file_stamp(project[1]) == update["stamp"]:
                self.app.project_open(project[1], discard_changes=True, keep_asset_manager_open=True)
                recovery = "Your saved local project is being reopened. Its recovery copy is also available."
        except Exception as exc:
            log_failure("recover_failed_update", exc, project_id=update.get("project", ("", ""))[0])
            pass
        raise ValueError("The gallery update could not be completed. " + recovery) from exc

    def finish_save(self, job):
        update = job["_update"]
        poll = self.app.project_poll_write()
        if poll.get("running"):
            return
        if poll.get("error"):
            self.recover_update(update, ValueError(poll["error"]))
        if (poll.get("generation") != update["generation"] + 1 or
                self.project_identity() != update["project"] or self.app.project_is_dirty()):
            raise ValueError("The saved project changed during the gallery update. Your recovery copy was kept; review the current project before linking it.")
        if self.detached or update.get("canceled"):
            self.pending = None
            self._message = "The project was updated and its recovery copy was kept. Account changed; it has not been linked to this account."
            return
        update["phase"] = "linking"
        try:
            update["link_operation"] = self.link_saved_download(job["id"], update["project"][1], update["project"][0],
                **({"local_fields": job["_local_fields"]} if "_local_fields" in job else {}))
        except Exception as exc:
            log_failure("link_saved_download", exc, job_id=job["id"])
            raise ValueError("The project was updated and its recovery copy was kept, but the gallery link could not be saved. Refresh your gallery before continuing.") from exc
        self._message = "Project updated. Saving its gallery link…"

    def apply(self, scene, incoming, job, update):
        # A saved recovery copy exists, and no edits have occurred since it was made.
        environment_path = (job["_local_environment_path"] if "_local_environment_path" in job
                            else self.service.environment_path(job))
        metadata = job.get("_local_fields", job["result"])
        restore_view(self.app, metadata.get("viewerSettings", {}), environment_path=environment_path)
        # Native remove_node(keep_children=True) keeps child-local transforms.
        # Reparent retained children explicitly first to preserve their world pose.
        removed_ids = set(update["old_nodes"])
        old_groups = {}
        for node_id in update["old_nodes"]:
            node = scene.get_node_by_uuid(node_id)
            if node is not None:
                parent_id = node.parent_id
                depth = 0
                while parent_id != -1:
                    parent = scene.get_node_by_id(parent_id)
                    if parent is None:
                        break
                    if parent.type == lf.scene.NodeType.GROUP:
                        old_groups[parent_id] = max(depth, old_groups.get(parent_id, -1))
                    parent_id = parent.parent_id
                    depth += 1
                for child_id in list(node.children):
                    child = scene.get_node_by_id(child_id)
                    if child is not None and child.uuid not in removed_ids:
                        if not scene.reparent(child_id, node.parent_id):
                            raise ValueError("A child of a replaced splat could not be preserved. Unlock it before retrying; your recovery copy is available.")
        for node_id in update["old_nodes"]:
            node = scene.get_node_by_uuid(node_id)
            if node is not None:
                scene.remove_node(node.name, keep_children=True)
        for group_id in sorted(old_groups, key=old_groups.get):
            group = scene.get_node_by_id(group_id)
            if group is not None and group.type == lf.scene.NodeType.GROUP and not group.children:
                scene.remove_node(group.name)
        self.app.set_node_visibility(incoming.name, True)
        title = metadata["title"]
        if scene.get_node(title) is not None:
            title += " (gallery " + incoming.uuid[:8] + ")"
        scene.rename_node(incoming.name, title)
        if self.project_identity() != update["project"] or file_stamp(update["project"][1]) != update["stamp"]:
            raise ValueError("The project identity or path changed before saving. Your recovery copy was kept.")
        if not self.app.project_save(wait=False):
            raise ValueError("The updated project could not be saved. Your recovery copy is available in the recovery folder.")


class DownloadOpenSteps:
    """Open: staging -> opened -> registration -> linking; keep: staging -> linking.

    Cancel at staging leaves the download alone. Cancel after native open
    keeps the new local file but skips registration and linking. After link
    submission, the journal result determines what the UI reports. The keep
    path never changes the open document; late cancel cannot retract a link.
    """

    def __init__(self, service, app, clock, *, acquire_native_use, release_native_use,
                 mark_viewing_copy, link_saved_download,
                 refresh_model, schedule_poll, message_changed):
        self.service = service
        self.app = app
        self.clock = clock
        self.acquire_native_use = acquire_native_use
        self.release_native_use = release_native_use
        self.mark_viewing_copy = mark_viewing_copy
        self.link_saved_download = link_saved_download
        self.refresh_model = refresh_model
        self.schedule_poll = schedule_poll
        self.message_changed = message_changed
        self.pending = None
        self.detached = False
        self.started = None
        self.progress = 0
        self.pulled_project = None
        self.__message = ""

    @property
    def message(self):
        return self.__message

    @property
    def _message(self):
        return self.__message

    @_message.setter
    def _message(self, value):
        self.__message = value
        self.message_changed(self)

    def cancel(self):
        if self.pending and self.pending.get("_register"):
            self.pending["_register"]["canceled"] = True
        if self.pending and self.pending.get("_opening"):
            opening = self.pending["_opening"]
            opening["canceled"] = True
            if opening["phase"] == "importing":
                self.app.ui.cancel_gallery_import()

    def start(self, job, identity):
        if self.service.identity() != identity:
            raise ValueError("The account changed while saving. Review your gallery before opening the download.")
        if self.app.is_training_active() or self.app.ui.get_import_state().get("active"):
            raise ValueError("Finish training or the current import before opening the download.")
        self.acquire_native_use(job["id"])
        stage_id = self.service.stage_download(job["id"])
        self.pending = dict(job, _accountIdentity=identity,
            _opening={"phase": "staging", "stage_id": stage_id, "scene": self.app.get_scene()})
        self.detached = False
        self.started = self.clock()
        self._message = "Checking downloaded scene…"
        self.schedule_poll()
        return

    def advance(self):
        job = self.pending
        if job.get("_register"):
            self.advance_keep(job)
            return
        if job.get("_accountIdentity") is not None and self.service.identity() != job["_accountIdentity"]:
            self.detached = True
        if job.get("_native_project"):
            expected = job["_native_project"]
            canceled = job.get("_opening", {}).get("canceled")
            if self.detached or canceled:
                self.pending = None
                self._message = (tr("info.canceled") if canceled and not self.detached
                    else "Account changed. The downloaded project is kept locally.")
                return
            current_path = self.app.project_poll_write().get("path")
            if (not current_path or Path(current_path).resolve() != Path(expected["path"]).resolve()
                    or self.app.get_scene().total_gaussian_count != expected["count"]):
                if self.clock() - self.started > 180:
                    self.pending = None
                    self._message = "Project loading did not finish. The downloaded .licht file is kept."
                return
            from .asset_index import AssetIndex
            index = AssetIndex()
            if not index.load(): raise ValueError("Could not open the Asset Manager catalog.")
            if file_stamp(expected["path"]) != expected["projectStamp"]:
                raise ValueError("The downloaded project identity or path changed. Prepare the download again.")
            inspection = self.app.io.inspect_project(expected["path"])
            if str(inspection.project_uuid) != expected["projectId"]:
                raise ValueError("The downloaded project identity changed. Prepare the download again.")
            project, _ = index.register_licht_asset(expected["path"], name=job["result"]["title"], inspection=inspection)
            if project is None:
                raise ValueError(index.last_error or "The project opened but could not be added to Asset Manager.")
            self.mark_viewing_copy(index, project)
            restore_view(self.app, job["result"].get("viewerSettings", {}), environment_path=self.service.environment_path(job))
            operation = self.link_saved_download(job["id"], expected["path"], expected["projectId"])
            job.pop("_native_project")
            job["_link"] = operation
            job["_registered_project"] = {"id": str(project.project_uuid), "path": expected["path"], "jobId": job["id"]}
            self._message = "Project opened. Saving its gallery link…"
            return
        opening = job.get("_opening")
        if opening and opening["phase"] == "staging":
            if self.service.busy:
                return
            if self.detached or opening.get("canceled") or not opening["scene"].is_valid() or self.app.project_is_dirty():
                self.pending = None
                self._message = "Account or project changed. Your download is kept; open it again when ready."
                return
            if self.app.is_training_active() or self.app.ui.get_import_state().get("active"):
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
            self.app.project_open(stage["projectPath"], keep_asset_manager_open=True)
            job["_native_project"] = {"path": stage["projectPath"], "count": count,
                "projectId": stage["projectId"], "projectStamp": stage["projectStamp"]}
            opening["phase"] = "opened"
            self.started = self.clock()
            self._message = "Opening .licht project…"
            return
        if job.get("_link"):
            if self.detached:
                self.pending = None
                self._message = "The download is saved in Asset Manager. Account changed; refresh your gallery to check its link."
                self.refresh_model()
                return
            if self.service.busy:
                return
            current = next((j for j in self.service.snapshot()["jobs"] if j["id"] == job["id"]), {})
            operation = current.get("linkOperation", {})
            self.pending = None
            if operation.get("id") == job["_link"] and operation.get("state") == "ready":
                self._message = "Downloaded scene saved and linked in Asset Manager."
                self.pulled_project = job.get("_registered_project")
            else:
                self._message = "The download is saved in Asset Manager, but its gallery link could not be saved. Refresh your gallery before continuing."
            self.refresh_model()
            return

    def start_keep(self, job, identity):
        """Keep and link a portable project without changing the open document."""
        if Path(job["path"]).suffix != ".licht":
            raise ValueError(tr("error.format"))
        if identity != self.service.identity():
            raise ValueError(tr("error.account_changed"))
        self.acquire_native_use(job["id"])
        try:
            stage_id = self.service.stage_download(job["id"])
        except Exception as exc:
            log_failure("stage_download", exc, job_id=job["id"])
            self.release_native_use()
            raise
        self.pending = dict(job, _accountIdentity=identity,
            _register={"stage_id": stage_id, "phase": "staging"})
        self.detached = False
        self._message = tr("state.downloading", percent=100)
        self.schedule_poll()

    def advance_keep(self, job):
        pending = job["_register"]
        if self.service.busy:
            return
        if (pending.get("canceled") and pending["phase"] == "staging"
                or self.detached or job["_accountIdentity"] != self.service.identity()):
            self.pending = None
            self._message = tr("error.account_changed" if self.detached else "info.canceled")
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
            inspection = self.app.io.inspect_project(path)
            if str(inspection.project_uuid) != stage.get("projectId"):
                raise ValueError("The downloaded project identity changed. Prepare the download again.")
            previous = index.get_asset(str(inspection.project_uuid))
            if previous and Path(previous.path).resolve() != Path(path).resolve() and Path(previous.path).exists():
                # Never move an existing catalog entry to an unrelated copy.
                raise ValueError(tr("error.link"))
            project, _ = index.register_licht_asset(
                path, name=job["result"]["title"], inspection=inspection,
                pin=index.folder_id_for_path(path) is None,
            )
            if project is None:
                raise ValueError(index.last_error or tr("error.storage"))
            self.mark_viewing_copy(index, project)
            pending.update(phase="linking", path=path, project=str(inspection.project_uuid),
                operation=self.link_saved_download(job["id"], path, str(inspection.project_uuid)))
            return
        operation = current.get("linkOperation", {})
        self.pending = None
        if operation.get("id") != pending["operation"] or operation.get("state") != "ready":
            raise ValueError(operation.get("message") or tr("error.link"))
        self.pulled_project = {"id": pending["project"], "path": pending["path"], "jobId": job["id"]}
        self._message = tr("info.pulled")
        self.refresh_model()


class PublishSteps:
    """Review -> save or verify -> export -> queue upload.

    Cancel during save stops its continuation. Cancel during export stops only
    this operation's native export and removes its temporary preparation after
    it ends. A completed export is queued only after its commit matches the
    reviewed saved project and account. The journal owns the transfer after
    queueing; cancellation there follows the normal transfer path.
    """

    def __init__(self, service, app, clock, asset_sync_state, *, model_state, project_identity,
                 visible_splats, details, save_project, start_saved, patch_saved_update,
                 pin_publish_preview, cleanup_preparation, refresh_model, schedule_poll,
                 set_operation, operation_title, set_last_canceled, set_upload_format,
                 message_changed):
        self.service = service
        self.app = app
        self.clock = clock
        self.asset_sync_state = asset_sync_state
        self.model_state = model_state
        self.project_identity = project_identity
        self.visible_splats = visible_splats
        self.details = details
        self.save_project = save_project
        self.start_saved_port = start_saved
        self.patch_saved_update_port = patch_saved_update
        self.pin_publish_preview = pin_publish_preview
        self.cleanup_preparation = cleanup_preparation
        self.refresh_model = refresh_model
        self.schedule_poll = schedule_poll
        self.set_operation = set_operation
        self.operation_title = operation_title
        self.set_last_canceled = set_last_canceled
        self.set_upload_format = set_upload_format
        self.message_changed = message_changed
        self.pending = None
        self.prepared_commit = None
        self.cancelled = False
        self.identity = None
        self.progress = 0
        self.preparation_failure = None
        self.reupload_reason = None
        self.__message = ""

    @property
    def message(self):
        return self.__message

    @property
    def _message(self):
        return self.__message

    @_message.setter
    def _message(self, value):
        self.__message = value
        self.message_changed(self)

    def start_closed(self, asset, details, upload_format, *, update, publish_as_new, handoff=None):
        """Prepare saved content without consulting the current scene or view."""
        if upload_format not in ("studio", "sog", "ssog", "spz"):
            raise ValueError("Choose a supported upload format.")
        if "licht" not in self.model_state().get("source_formats", []):
            raise ValueError(UNSUPPORTED_PORTAL)
        project_id, path = asset["id"], asset["path"]
        info = self.app.io.inspect_project(path)
        if str(info.project_uuid) != project_id:
            raise ValueError(tr("error.project_changed"))
        if any(j["project"] == project_id and j["status"] not in ("completed", "canceled") for j in self.model_state()["jobs"]):
            raise ValueError("This project already has an upload. Resume or discard it first.")
        link = self.model_state()["links"].get(project_id)
        scene = next((s for s in self.model_state()["scenes"] if link and s["id"] == link["sceneId"]), None) if update else None
        if handoff:
            scene = next((s for s in self.model_state()["scenes"] if s["id"] == handoff["sceneId"]), None)
        if (update or handoff) and scene is None:
            raise ValueError(tr("error.refresh"))
        if update and self.asset_sync_state(asset, link, scene)["freshness"] in ("diverged", "remote", "unknown"):
            # Review must resolve remote write guards before any replacement.
            raise ValueError(tr("error.refresh"))
        if link and not update and not publish_as_new:
            raise ValueError("This project is linked to a gallery item. Select it to replace, or unlink before publishing a new item.")
        metadata = self.details(details)
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
        self.set_operation(project_id, details.get("title") or asset.get("name", ""))
        self.set_last_canceled(False)
        self.reupload_reason = None
        self.set_upload_format(upload_format)

        def start():
            self.preparation_failure = None
            if self.service.identity() != identity:
                return
            if self.app.ui.get_export_state().get("active"):
                raise ValueError("Wait for the current export to finish before uploading.")
            if str(self.app.io.inspect_project(path).commit_uuid) != expected_commit:
                raise ValueError(tr("error.project_changed"))
            if self.patch_saved_update_port(metadata, project_id, path, update=update):
                return
            self.pin_publish_preview(metadata, path, expected_commit)
            metadata["viewerSettings"] = {}  # The saved project supplies VIEW/SEQR.
            export = self.service.root / (str(uuid.uuid4()) + ".scene")
            self.cancelled = False
            self.identity = identity
            self.progress = 0
            self.app.prepare_gallery_project(path, str(export), "ply" if upload_format == "studio" else upload_format, expected_commit)
            self.prepared_commit = expected_commit
            self.pending = (export, metadata, project_id, self.clock())
            self.schedule_poll()

        start()
        self.schedule_poll()

    def start(self, metadata, *, expected_project=None, environment_source=None, upload_format="studio", update=False,
                 save_project=True, expected_commit=None):
        identity = self.service.identity()
        project_id, path = self.project_identity()
        if expected_project is not None and (project_id, path) != expected_project:
            raise ValueError("The current project changed. Review its gallery details before uploading.")
        pending = [j for j in self.model_state()["jobs"] if j["project"] == project_id and j["status"] not in ("completed", "canceled")]
        if pending:
            raise ValueError("This project already has an upload. Resume or discard it first.")
        linked = self.model_state()["links"].get(project_id)
        if linked and metadata.get("replaceSceneId") != linked["sceneId"] and not metadata.get("_publishAsNew"):
            raise ValueError("This project is linked to a gallery item. Select it to replace, or unlink before publishing a new item.")
        nodes = [n.name for n in self.visible_splats()]
        if not nodes:
            raise ValueError("There are no visible splats to upload.")
        self.preparation_failure = None
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = 0
        account = getattr(self.service, "account", None)
        log_stage("publish_requested", project_id=project_id, path=path, size=size,
                  format=upload_format, account_origin=safe_url(getattr(account, "base_url", "")),
                  update=update)
        if save_project:
            self.save_project(lambda: self.start_saved_port(metadata, project_id, path, identity,
                                                                 environment_source, upload_format, update=update))
        elif not self.app.project_is_dirty():
            self.start_saved_port(metadata, project_id, path, identity, environment_source, upload_format,
                                  update=update, expected_commit=expected_commit or str(self.app.io.inspect_project(path).commit_uuid))
        else:
            if self.app.project_poll_write().get("running"):
                raise ValueError("Wait for the current project save before continuing.")
            self.start_live(metadata, upload_format, project_id=project_id)

    def start_live(self, metadata, upload_format, *, project_id=None, unlinked=False):
        if upload_format not in ("studio", "sog", "ssog", "spz"):
            raise ValueError("Choose a supported upload format.")
        if "licht" not in self.service.snapshot().get("source_formats", []):
            raise ValueError(UNSUPPORTED_PORTAL)
        if not [node for node in self.visible_splats()]:
            raise ValueError("There are no visible splats to upload.")
        if self.app.ui.get_export_state().get("active"):
            raise ValueError("Wait for the current export to finish before uploading.")
        if unlinked and self.app.project_poll_write().get("path"):
            raise ValueError(tr("error.project_changed"))
        project_id = project_id or str(uuid.uuid4())
        metadata = dict(metadata, _uploadFormat=upload_format)
        if unlinked:
            metadata["_unlinked"] = True
        elif project_id:
            metadata["_liveSnapshot"] = True
        self.preparation_failure = None
        self.cancelled = False
        self.identity = self.service.identity()
        self.prepared_commit = None
        self.progress = 0
        self.set_operation(project_id, metadata["title"])
        export = self.service.root / (str(uuid.uuid4()) + ".scene")
        self.app.prepare_gallery_scene(str(export), "ply" if upload_format == "studio" else upload_format)
        self.pending = (export, metadata, project_id, self.clock())
        self.schedule_poll()

    def patch_saved_update(self, metadata, project_id, path, *, update, expected_commit=None):
        from .gallery_project_facts import saved_content_stamp
        content_stamp = saved_content_stamp(path)
        metadata["_contentStamp"] = content_stamp
        linked = self.service.snapshot().get("links", {}).get(project_id, {})
        if update and (":" not in content_stamp or ":" not in linked.get("contentStamp", "")):
            self.reupload_reason = {"project": project_id, "message": tr("info.reupload_encoding")}
        # Live metadata carries the complete view; a closed PATCH must also
        # prove its saved VIEW/SEQR unchanged so it cannot lose local view edits.
        baseline = linked.get("contentStamp", "")
        comparable = lambda stamp: stamp.split(":", 1)[0] if "viewerSettings" in metadata else stamp
        if (update and content_stamp and ":" in content_stamp and ":" in baseline
                and comparable(content_stamp) == comparable(baseline)
                and metadata.get("replaceSceneId") == linked.get("sceneId")):
            details = {k: v for k, v in metadata.items() if k in ("title", "description", "viewerSettings")}
            commit = str(self.app.io.inspect_project(path).commit_uuid)
            if expected_commit is not None and commit != expected_commit:
                raise ValueError(tr("error.project_changed"))
            cover = {}
            if metadata.get("useEmbeddedPreview"):
                import base64
                self.pin_publish_preview(metadata, path, commit)
                cover["cover_png"] = base64.b64decode(metadata["_previewPng"], validate=True)
            self.service.edit(linked["sceneId"], {name + "Revision": token for name, token in metadata["baseRevisions"].items()}, details,
                commit_uuid=commit, content_stamp=content_stamp, project_id=project_id, **cover)
            return True
        return False

    def start_saved(self, metadata, project_id, path, identity, environment_source=None, upload_format="studio", *, update=False,
                       expected_commit=None):
        if self.service.identity() != identity or self.project_identity() != (project_id, path):
            raise ValueError("The account or current project changed while saving. Review it before uploading.")
        inspection = self.app.io.inspect_project(path)
        if expected_commit is not None and str(inspection.commit_uuid) != expected_commit:
            raise ValueError(tr("error.project_changed"))
        if expected_commit is not None:
            references = self.app.io.inspect_project_details(path).references
            saved_environment = next((Path(ref.path).resolve() for ref in references if ref.kind == "environment_map"), None)
            live_environment = Path(environment_source).resolve() if metadata.get("viewerSettings", {}).get("environment") and environment_source else None
            if saved_environment != live_environment:
                raise ValueError(tr("error.save_hdr_first"))
        environment = metadata.get("viewerSettings", {}).get("environment")
        if environment:
            settings = self.app.get_render_settings()
            if (settings.environment_mode != "EQUIRECTANGULAR" or str(settings.environment_map_path) != environment_source
                    or float(settings.environment_exposure) != environment["exposure"]
                    or float(settings.environment_rotation_degrees) != environment["rotation"]):
                raise ValueError("The HDR background changed. Review the current view and try uploading again.")
        if self.patch_saved_update_port(metadata, project_id, path, update=update, expected_commit=expected_commit):
            return
        nodes = [n.name for n in self.visible_splats()]
        if not nodes:
            raise ValueError("There are no visible splats to upload.")
        if upload_format not in ("studio", "sog", "ssog", "spz"):
            raise ValueError("Choose a supported upload format.")
        if "licht" not in self.service.snapshot().get("source_formats", []):
            raise ValueError(UNSUPPORTED_PORTAL)
        export = self.service.root / (str(uuid.uuid4()) + ".scene")
        metadata = dict(metadata)
        metadata["_commitUuid"] = str(getattr(inspection, "commit_uuid", ""))
        file_uuid = str(getattr(inspection, "file_uuid", ""))
        if file_uuid:
            metadata["originFileUuid"] = file_uuid
        metadata["_uploadFormat"] = upload_format
        self._message = "Preparing the current scene for upload…"
        self.refresh_model()
        if self.app.ui.get_export_state().get("active"):
            raise ValueError("Wait for the current export to finish before uploading.")
        self.cancelled = False
        self.pin_publish_preview(metadata, path, metadata["_commitUuid"])
        self.prepared_commit = metadata["_commitUuid"]
        self.identity = identity
        self.progress = 0
        self.app.prepare_gallery_project(path, str(export), "ply" if upload_format == "studio" else upload_format,
                                   self.prepared_commit)
        if self.service.identity() != identity:
            self.cancelled = True
        self.pending = (export, metadata, project_id, self.clock())
        self.schedule_poll()

    def owns_export(self, state):
        return bool(self.pending and state.get("path")
            and Path(state["path"]) == Path(self.pending[0]))

    def cancel_export(self):
        state = self.app.ui.get_export_state()
        if self.owns_export(state) and state.get("active"):
            self.app.ui.cancel_export()

    def remove_preparation(self, export):
        if Path(export).suffix == ".scene":
            for path in gallery_preparation.staging_files(self.service.root, export):
                self.service._unlink_temporary(path)
        else:
            Path(export).unlink(missing_ok=True)

    def advance(self):
        export, metadata, project_id, started = self.pending
        prepared_commit = self.prepared_commit
        if self.identity is not None and self.service.identity() != self.identity:
            self.cancelled = True
        state = self.app.ui.get_export_state()
        if not self.owns_export(state):
            self.pending = None
            self.prepared_commit = None
            self.cleanup_preparation(export)
            self._message = "The scene preparation status changed. Please prepare your upload again."
            self.refresh_model()
            return
        if state.get("active"):
            progress = max(0, min(100, int(float(state.get("progress", 0)) * 100)))
            if progress != self.progress:
                self.progress = progress
                self._message = "Canceling scene preparation…" if self.cancelled else f"Preparing scene for upload… {progress}%"
                self.refresh_model()
            return
        outcome = state.get("outcome")
        if self.cancelled or outcome in ("failed", "cancelled"):
            self.prepared_commit = None
            self.pending = None
            self.cleanup_preparation(export)
            error = str(state.get("error", ""))
            self._message = "Scene preparation canceled." if self.cancelled or outcome == "cancelled" else (error or "Scene preparation failed. Check the export status and try again.")
            if outcome == "failed" and not self.cancelled:
                self.preparation_failure = {"id": "preparation:" + project_id, "project": project_id,
                    "commitUuid": prepared_commit,
                    "status": "error", "kind": "upload", "metadata": {"title": self.operation_title()},
                    "message": self._message, "failureReason": self._message}
                log_failure("native_preparation", RuntimeError(self._message), project_id=project_id)
            self.refresh_model()
        elif outcome == "completed" and export.exists():
            self.pending = None
            source = self.prepared_commit
            self.prepared_commit = None
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
                self.preparation_failure = {"id": "preparation:" + project_id, "project": project_id,
                    "commitUuid": prepared_commit,
                    "status": "error", "kind": "upload", "metadata": {"title": self.operation_title()},
                    "message": friendly_error(exc), "failureReason": friendly_error(exc)}
                try:
                    self.cleanup_preparation(export)
                    self._message = friendly_error(exc)
                except (OSError, ValueError) as cleanup_exc:
                    log_failure("preparation_cleanup", cleanup_exc, project_id=project_id)
                    self._message = "The upload could not be queued. Temporary files were kept; open the recovery folder to review them."
            self.refresh_model()
        elif self.clock() - started > 60:
            self.prepared_commit = None
            self.pending = None
            self.cleanup_preparation(export)
            self._message = "LichtFeld Studio could not prepare the scene. Check the export status and try again."
            self.preparation_failure = {"id": "preparation:" + project_id, "project": project_id,
                "commitUuid": prepared_commit,
                "status": "error", "kind": "upload", "metadata": {"title": self.operation_title()},
                "message": self._message, "failureReason": self._message}
            log_failure("native_preparation_timeout", TimeoutError(self._message), project_id=project_id)
            self.refresh_model()
