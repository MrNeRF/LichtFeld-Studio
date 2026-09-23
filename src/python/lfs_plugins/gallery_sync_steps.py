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
    """staging -> importing -> save_before_backup -> backup -> applying -> save_updated -> linking.

    Import may add a hidden preview, and save_before_backup may save the
    existing project. Old splats and the view are replaced only after the
    backup is ready and the saved generation and stamp still match. Cancel
    through backup removes the owned preview, leaving the baseline save.
    Apply is synchronous; failure reopens that saved file. Cancel after the
    updated save keeps the updated project and recovery copy. A submitted link
    may still complete and must be checked by refreshing the gallery.
    """

    def __init__(self, ui):
        self.ui = ui

    def cancel(self):
        ui = self.ui
        if ui._import_pending and ui._import_pending.get("_update"):
            update = ui._import_pending["_update"]
            update["canceled"] = True
            if update["phase"] == "importing" and Path(update.get("path", "")).suffix == ".scene":
                lf.ui.cancel_gallery_import()

    def discard_preview(self):
        ui = self.ui
        update = (ui._import_pending or {}).get("_update", {})
        if update.get("phase") not in ("save_before_backup", "backup") or not update.get("incoming"):
            return
        try:
            if ui._project_identity() != update["project"] or lf.ui.get_import_state().get("active"):
                return
            scene = lf.get_scene()
            incoming = scene.get_node_by_uuid(update["incoming"])
            if incoming is not None:
                scene.remove_node(incoming.name)
        except Exception as exc:
            # Never hide the original failure or touch another project's nodes.
            log_failure("discard_update_preview", exc)

    def begin(self, job, project):
        ui = self.ui
        if ui._pull_overrides and ui._pull_overrides[0] == job.get("result", {}).get("id"):
            job = copy.deepcopy(job)
            job["_local_fields"] = copy.deepcopy(ui._pull_overrides[1])
            if ui._pull_overrides[4] is not None:
                job["_local_environment_path"] = ui._pull_overrides[4]
        if ui._project_identity() != project:
            raise ValueError("The current project changed. Review it before updating.")
        if lf.is_training_active() or lf.ui.get_import_state().get("active"):
            raise ValueError("Finish training or the current import before updating this project.")
        if any(n.locked for n in ui._visible_splats()):
            raise ValueError("Unlock the visible splats before updating this project.")
        if job.get("kind") != "download" or job["status"] != "completed":
            raise ValueError("Finish downloading this scene first.")
        ui._acquire_native_use(job["id"])
        stage_id = ui.service.stage_download(job["id"])
        ui._import_pending = dict(job, _accountIdentity=ui._identity,
            _update={"project": project, "phase": "staging", "stage_id": stage_id})
        ui._import_detached = False
        ui._import_started = time.monotonic()
        ui._message = "Preparing the gallery update…"
        ui._schedule_poll()

    def advance(self, job):
        ui = self.ui
        update = job["_update"]
        project = update["project"]
        if ui._save_pending:
            return
        if ui._project_identity() != project:
            ui._import_pending = None
            ui._message = ("Project changed after the gallery update. Its recovery copy was kept; review the saved project before linking it."
                if update["phase"] in ("save_updated", "linking") else "Project changed. The gallery update was not applied.")
            return
        if update["phase"] == "save_updated":
            ui._finish_update_save(job)
            return
        # Wait for an owned native import to finish before hiding its preview.
        if update["phase"] == "importing" and lf.ui.get_import_state().get("active"):
            return
        scene = lf.get_scene()
        incoming = scene.get_node(Path(update["path"]).stem) if update.get("path") else None
        if incoming is not None and update["phase"] == "importing":
            lf.set_node_visibility(incoming.name, False)
        if ui._import_detached or update.get("canceled"):
            if incoming is not None and Path(update.get("path", "")).suffix == ".scene" and update["phase"] in ("importing", "save_before_backup", "backup"):
                scene.remove_node(incoming.name)
            ui._import_pending = None
            ui._message = ("The project was updated; its recovery copy was kept. Refresh the gallery to check its link."
                if update["phase"] == "linking" else "Update canceled. Your existing local splats remain.")
            return
        if ui.service.busy:
            return
        current_job = next(j for j in ui.service.snapshot()["jobs"] if j["id"] == job["id"])
        if update["phase"] == "linking":
            linked = current_job.get("linkOperation", {})
            if linked.get("id") != update["link_operation"] or linked.get("state") != "ready":
                raise ValueError("The project was updated, but its gallery link could not be saved. Your recovery copy is available; refresh the gallery before continuing.")
            ui._import_pending = None
            if current_job.get("localUpdate", {}).get("backupPath"):
                ui._undo_pull = {"path": project[1], "backup": current_job["localUpdate"]["backupPath"],
                    "stamp": file_stamp(project[1]), "identity": ui._identity}
            if ui._pull_overrides:
                scene_id, metadata, identity, publish = ui._pull_overrides[:4]
                ui._pull_overrides = None
                if identity == ui._identity and publish:
                    ui.service.edit(scene_id, domain_tokens(current_job["result"]), metadata, project_id=project[0])
            ui._message = "Linked project updated. Your previous local work is kept in its recovery copy."
            ui._refresh_model()
            return
        if update["phase"] == "staging":
            stage = current_job.get("stagedImport", {})
            if stage.get("id") != update["stage_id"] or stage.get("state") != "ready":
                raise ValueError(stage.get("message") or ui.service.message)
            update["path"] = stage["path"]
            if scene.get_node(Path(stage["path"]).stem):
                raise ValueError("The update preview already exists. Download the gallery item again.")
            if Path(stage["path"]).suffix == ".scene":
                lf.load_gallery_scene(ui._staged_nodes(stage["path"]), Path(stage["path"]).stem,
                    hidden=True)
            elif Path(stage["path"]).suffix == ".licht":
                lf.load_file(stage["path"])
            else:
                raise ValueError("The downloaded project identity or path changed. Prepare the download again.")
            update["phase"] = "importing"
            ui._import_started = time.monotonic()
            return
        if update["phase"] == "importing":
            if incoming is None:
                if lf.ui.get_import_state().get("error") or time.monotonic() - ui._import_started > 60:
                    raise ValueError("LichtFeld Studio could not open the gallery update. Your local splats remain.")
                return
            lf.ui.dismiss_import()
            update["incoming"] = incoming.uuid
            update["old_nodes"] = [n.uuid for n in ui._visible_splats() if n.uuid != incoming.uuid]
            update["phase"] = "save_before_backup"
            ui._save_current_project(lambda: ui._prepare_update_backup(job), expected_project=project)
            return
        backup = current_job.get("localUpdate", {})
        if backup.get("id") != update["backup_id"] or backup.get("state") != "ready":
            raise ValueError(backup.get("message") or ui.service.message)
        if (lf.project_is_dirty() or lf.project_poll_write()["generation"] != update["generation"] or
                file_stamp(project[1]) != update["stamp"]):
            raise ValueError("Your local project changed during preparation. Its splats were kept. Review it and try the update again.")
        incoming = scene.get_node_by_uuid(update["incoming"])
        if incoming is None:
            raise ValueError("The downloaded preview changed. Your local splats were kept.")
        update["phase"] = "applying"
        try:
            ui._apply_local_update(scene, incoming, job, update)
        except Exception as exc:
            ui._recover_failed_update(update, exc)
        update["phase"] = "save_updated"
        ui._message = "Saving the updated project…"

    def prepare_backup(self, job):
        ui = self.ui
        update = job["_update"]
        project = update["project"]
        update["generation"] = lf.project_poll_write()["generation"]
        update["stamp"] = file_stamp(project[1])
        update["backup_id"] = ui.service.prepare_local_update(job["id"], project[0], project[1], update["stamp"])
        update["phase"] = "backup"
        ui._message = "Keeping a recovery copy before replacing local splats…"

    def recover_failure(self, update, exc):
        ui = self.ui
        recovery = "Your recovery copy is available in the recovery folder."
        try:
            project = update["project"]
            # Reopen only the unchanged generation that preceded the update.
            if ui._project_identity() == project and file_stamp(project[1]) == update["stamp"]:
                lf.project_open(project[1], discard_changes=True, keep_asset_manager_open=True)
                recovery = "Your saved local project is being reopened. Its recovery copy is also available."
        except Exception as exc:
            log_failure("recover_failed_update", exc, project_id=update.get("project", ("", ""))[0])
            pass
        raise ValueError("The gallery update could not be completed. " + recovery) from exc

    def finish_save(self, job):
        ui = self.ui
        update = job["_update"]
        poll = lf.project_poll_write()
        if poll.get("running"):
            return
        if poll.get("error"):
            ui._recover_failed_update(update, ValueError(poll["error"]))
        if (poll.get("generation") != update["generation"] + 1 or
                ui._project_identity() != update["project"] or lf.project_is_dirty()):
            raise ValueError("The saved project changed during the gallery update. Your recovery copy was kept; review the current project before linking it.")
        if ui._import_detached or update.get("canceled"):
            ui._import_pending = None
            ui._message = "The project was updated and its recovery copy was kept. Account changed; it has not been linked to this account."
            return
        update["phase"] = "linking"
        try:
            update["link_operation"] = ui._link_saved_download(job["id"], update["project"][1], update["project"][0],
                **({"local_fields": job["_local_fields"]} if "_local_fields" in job else {}))
        except Exception as exc:
            log_failure("link_saved_download", exc, job_id=job["id"])
            raise ValueError("The project was updated and its recovery copy was kept, but the gallery link could not be saved. Refresh your gallery before continuing.") from exc
        ui._message = "Project updated. Saving its gallery link…"

    def apply(self, scene, incoming, job, update):
        ui = self.ui
        # A saved recovery copy exists, and no edits have occurred since it was made.
        environment_path = (job["_local_environment_path"] if "_local_environment_path" in job
                            else ui.service.environment_path(job))
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
        if ui._project_identity() != update["project"] or file_stamp(update["project"][1]) != update["stamp"]:
            raise ValueError("The project identity or path changed before saving. Your recovery copy was kept.")
        if not lf.project_save(wait=False):
            raise ValueError("The updated project could not be saved. Your recovery copy is available in the recovery folder.")


class DownloadOpenSteps:
    """Open: staging -> opened -> registration -> linking; keep: staging -> linking.

    Cancel at staging leaves the download alone. Cancel after native open
    keeps the new local file but skips registration and linking. After link
    submission, the journal result determines what the UI reports. The keep
    path never changes the open document; late cancel cannot retract a link.
    """

    def __init__(self, ui):
        self.ui = ui

    def cancel(self):
        ui = self.ui
        if ui._import_pending and ui._import_pending.get("_register"):
            ui._import_pending["_register"]["canceled"] = True
        if ui._import_pending and ui._import_pending.get("_opening"):
            opening = ui._import_pending["_opening"]
            opening["canceled"] = True
            if opening["phase"] == "importing":
                lf.ui.cancel_gallery_import()

    def start(self, job, identity):
        ui = self.ui
        if ui.service.identity() != identity:
            raise ValueError("The account changed while saving. Review your gallery before opening the download.")
        if lf.is_training_active() or lf.ui.get_import_state().get("active"):
            raise ValueError("Finish training or the current import before opening the download.")
        ui._acquire_native_use(job["id"])
        stage_id = ui.service.stage_download(job["id"])
        ui._import_pending = dict(job, _accountIdentity=identity,
            _opening={"phase": "staging", "stage_id": stage_id, "scene": lf.get_scene()})
        ui._import_detached = False
        ui._import_started = time.monotonic()
        ui._message = "Checking downloaded scene…"
        ui._schedule_poll()
        return

    def advance(self):
        ui = self.ui
        job = ui._import_pending
        if job.get("_register"):
            ui._finish_register_download(job)
            return
        if job.get("_accountIdentity") is not None and ui.service.identity() != job["_accountIdentity"]:
            ui._import_detached = True
        if job.get("_native_project"):
            expected = job["_native_project"]
            canceled = job.get("_opening", {}).get("canceled")
            if ui._import_detached or canceled:
                ui._import_pending = None
                ui._message = (tr("info.canceled") if canceled and not ui._import_detached
                    else "Account changed. The downloaded project is kept locally.")
                return
            current_path = lf.project_poll_write().get("path")
            if (not current_path or Path(current_path).resolve() != Path(expected["path"]).resolve()
                    or lf.get_scene().total_gaussian_count != expected["count"]):
                if time.monotonic() - ui._import_started > 180:
                    ui._import_pending = None
                    ui._message = "Project loading did not finish. The downloaded .licht file is kept."
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
            ui._mark_viewing_copy(index, project)
            restore_view(lf, job["result"].get("viewerSettings", {}), environment_path=ui.service.environment_path(job))
            operation = ui._link_saved_download(job["id"], expected["path"], expected["projectId"])
            job.pop("_native_project")
            job["_link"] = operation
            job["_registered_project"] = {"id": str(project.project_uuid), "path": expected["path"], "jobId": job["id"]}
            ui._message = "Project opened. Saving its gallery link…"
            return
        if job.get("_update"):
            ui._finish_local_update(job)
            return
        opening = job.get("_opening")
        if opening and opening["phase"] == "staging":
            if ui.service.busy:
                return
            if ui._import_detached or opening.get("canceled") or not opening["scene"].is_valid() or lf.project_is_dirty():
                ui._import_pending = None
                ui._message = "Account or project changed. Your download is kept; open it again when ready."
                return
            if lf.is_training_active() or lf.ui.get_import_state().get("active"):
                raise ValueError("Finish training or the current import before opening this download. Your download is kept.")
            current = next(j for j in ui.service.snapshot()["jobs"] if j["id"] == job["id"])
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
            ui._import_started = time.monotonic()
            ui._message = "Opening .licht project…"
            return
        if job.get("_link"):
            if ui._import_detached:
                ui._import_pending = None
                ui._message = "The download is saved in Asset Manager. Account changed; refresh your gallery to check its link."
                ui._refresh_model()
                return
            if ui.service.busy:
                return
            current = next((j for j in ui.service.snapshot()["jobs"] if j["id"] == job["id"]), {})
            operation = current.get("linkOperation", {})
            ui._import_pending = None
            if operation.get("id") == job["_link"] and operation.get("state") == "ready":
                ui._message = "Downloaded scene saved and linked in Asset Manager."
                ui._pulled_project = job.get("_registered_project")
            else:
                ui._message = "The download is saved in Asset Manager, but its gallery link could not be saved. Refresh your gallery before continuing."
            ui._refresh_model()
            return

    def start_keep(self, job, identity):
        """Keep and link a portable project without changing the open document."""
        ui = self.ui
        if Path(job["path"]).suffix != ".licht":
            raise ValueError(tr("error.format"))
        if identity != ui.service.identity():
            raise ValueError(tr("error.account_changed"))
        ui._acquire_native_use(job["id"])
        try:
            stage_id = ui.service.stage_download(job["id"])
        except Exception as exc:
            log_failure("stage_download", exc, job_id=job["id"])
            ui._release_native_use()
            raise
        ui._import_pending = dict(job, _accountIdentity=identity,
            _register={"stage_id": stage_id, "phase": "staging"})
        ui._import_detached = False
        ui._message = tr("state.downloading", percent=100)
        ui._schedule_poll()

    def advance_keep(self, job):
        ui = self.ui
        pending = job["_register"]
        if ui.service.busy:
            return
        if (pending.get("canceled") and pending["phase"] == "staging"
                or ui._import_detached or job["_accountIdentity"] != ui.service.identity()):
            ui._import_pending = None
            ui._message = tr("error.account_changed" if ui._import_detached else "info.canceled")
            return
        current = next((j for j in ui.service.snapshot()["jobs"] if j["id"] == job["id"]), {})
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
            ui._mark_viewing_copy(index, project)
            pending.update(phase="linking", path=path, project=str(inspection.project_uuid),
                operation=ui._link_saved_download(job["id"], path, str(inspection.project_uuid)))
            return
        operation = current.get("linkOperation", {})
        ui._import_pending = None
        if operation.get("id") != pending["operation"] or operation.get("state") != "ready":
            raise ValueError(operation.get("message") or tr("error.link"))
        ui._pulled_project = {"id": pending["project"], "path": pending["path"], "jobId": job["id"]}
        ui._message = tr("info.pulled")
        ui._refresh_model()


class PublishSteps:
    """Review -> save or verify -> export -> queue upload.

    Cancel during save stops its continuation. Cancel during export stops only
    this operation's native export and removes its temporary preparation after
    it ends. A completed export is queued only after its commit matches the
    reviewed saved project and account. The journal owns the transfer after
    queueing; cancellation there follows the normal transfer path.
    """

    def __init__(self, ui, asset_sync_state):
        self.ui = ui
        self.asset_sync_state = asset_sync_state

    def start_closed(self, asset, details, upload_format, *, update, publish_as_new, handoff=None):
        """Prepare saved content without consulting the current scene or view."""
        ui = self.ui
        if upload_format not in ("studio", "sog", "ssog", "spz"):
            raise ValueError("Choose a supported upload format.")
        if "licht" not in ui._state.get("source_formats", []):
            raise ValueError(UNSUPPORTED_PORTAL)
        project_id, path = asset["id"], asset["path"]
        info = lf.io.inspect_project(path)
        if str(info.project_uuid) != project_id:
            raise ValueError(tr("error.project_changed"))
        if any(j["project"] == project_id and j["status"] not in ("completed", "canceled") for j in ui._state["jobs"]):
            raise ValueError("This project already has an upload. Resume or discard it first.")
        link = ui._state["links"].get(project_id)
        scene = next((s for s in ui._state["scenes"] if link and s["id"] == link["sceneId"]), None) if update else None
        if handoff:
            scene = next((s for s in ui._state["scenes"] if s["id"] == handoff["sceneId"]), None)
        if (update or handoff) and scene is None:
            raise ValueError(tr("error.refresh"))
        if update and self.asset_sync_state(asset, link, scene)["freshness"] in ("diverged", "remote", "unknown"):
            # Review must resolve remote write guards before any replacement.
            raise ValueError(tr("error.refresh"))
        if link and not update and not publish_as_new:
            raise ValueError("This project is linked to a gallery item. Select it to replace, or unlink before publishing a new item.")
        metadata = ui._details(details)
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
        identity = ui.service.identity()
        ui._operation_project = project_id
        ui._operation_title = details.get("title") or asset.get("name", "")
        ui._last_canceled = False
        ui._reupload_reason = None
        ui.upload_format = upload_format

        def start():
            ui._preparation_failure = None
            if ui.service.identity() != identity:
                return
            if lf.ui.get_export_state().get("active"):
                raise ValueError("Wait for the current export to finish before uploading.")
            if str(lf.io.inspect_project(path).commit_uuid) != expected_commit:
                raise ValueError(tr("error.project_changed"))
            if ui._patch_saved_update(metadata, project_id, path, update=update):
                return
            ui._pin_publish_preview(metadata, path, expected_commit)
            metadata["viewerSettings"] = {}  # The saved project supplies VIEW/SEQR.
            export = ui.service.root / (str(uuid.uuid4()) + ".scene")
            ui._export_cancelled = False
            ui._export_identity = identity
            ui._export_progress = 0
            lf.prepare_gallery_project(path, str(export), "ply" if upload_format == "studio" else upload_format, expected_commit)
            ui._prepared_commit = expected_commit
            ui._export_pending = (export, metadata, project_id, time.monotonic())
            ui._schedule_poll()

        start()
        ui._schedule_poll()

    def start(self, metadata, *, expected_project=None, environment_source=None, upload_format="studio", update=False,
                 save_project=True, expected_commit=None):
        ui = self.ui
        identity = ui.service.identity()
        project_id, path = ui._project_identity()
        if expected_project is not None and (project_id, path) != expected_project:
            raise ValueError("The current project changed. Review its gallery details before uploading.")
        pending = [j for j in ui._state["jobs"] if j["project"] == project_id and j["status"] not in ("completed", "canceled")]
        if pending:
            raise ValueError("This project already has an upload. Resume or discard it first.")
        linked = ui._state["links"].get(project_id)
        if linked and metadata.get("replaceSceneId") != linked["sceneId"] and not metadata.get("_publishAsNew"):
            raise ValueError("This project is linked to a gallery item. Select it to replace, or unlink before publishing a new item.")
        nodes = [n.name for n in ui._visible_splats()]
        if not nodes:
            raise ValueError("There are no visible splats to upload.")
        ui._preparation_failure = None
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = 0
        account = getattr(ui.service, "account", None)
        log_stage("publish_requested", project_id=project_id, path=path, size=size,
                  format=upload_format, account_origin=safe_url(getattr(account, "base_url", "")),
                  update=update)
        if save_project:
            ui._save_current_project(lambda: ui._publish_saved(metadata, project_id, path, identity,
                                                                 environment_source, upload_format, update=update))
        else:
            if lf.project_poll_write().get("running"):
                raise ValueError("Wait for the current project save before continuing.")
            ui._publish_saved(metadata, project_id, path, identity, environment_source, upload_format,
                                update=update, expected_commit=expected_commit or str(lf.io.inspect_project(path).commit_uuid))

    def patch_saved_update(self, metadata, project_id, path, *, update, expected_commit=None):
        ui = self.ui
        from .gallery_project_facts import saved_content_stamp
        content_stamp = saved_content_stamp(path)
        metadata["_contentStamp"] = content_stamp
        linked = ui.service.snapshot().get("links", {}).get(project_id, {})
        if update and (":" not in content_stamp or ":" not in linked.get("contentStamp", "")):
            ui._reupload_reason = {"project": project_id, "message": tr("info.reupload_encoding")}
        # Live metadata carries the complete view; a closed PATCH must also
        # prove its saved VIEW/SEQR unchanged so it cannot lose local view edits.
        baseline = linked.get("contentStamp", "")
        comparable = lambda stamp: stamp.split(":", 1)[0] if "viewerSettings" in metadata else stamp
        if (update and content_stamp and ":" in content_stamp and ":" in baseline
                and comparable(content_stamp) == comparable(baseline)
                and metadata.get("replaceSceneId") == linked.get("sceneId")):
            details = {k: v for k, v in metadata.items() if k in ("title", "description", "viewerSettings")}
            commit = str(lf.io.inspect_project(path).commit_uuid)
            if expected_commit is not None and commit != expected_commit:
                raise ValueError(tr("error.project_changed"))
            cover = {}
            if metadata.get("useEmbeddedPreview"):
                import base64
                ui._pin_publish_preview(metadata, path, commit)
                cover["cover_png"] = base64.b64decode(metadata["_previewPng"], validate=True)
            ui.service.edit(linked["sceneId"], {name + "Revision": token for name, token in metadata["baseRevisions"].items()}, details,
                commit_uuid=commit, content_stamp=content_stamp, project_id=project_id, **cover)
            return True
        return False

    def start_saved(self, metadata, project_id, path, identity, environment_source=None, upload_format="studio", *, update=False,
                       expected_commit=None):
        ui = self.ui
        if ui.service.identity() != identity or ui._project_identity() != (project_id, path):
            raise ValueError("The account or current project changed while saving. Review it before uploading.")
        inspection = lf.io.inspect_project(path)
        if expected_commit is not None and str(inspection.commit_uuid) != expected_commit:
            raise ValueError(tr("error.project_changed"))
        if expected_commit is not None:
            references = lf.io.inspect_project_details(path).references
            saved_environment = next((Path(ref.path).resolve() for ref in references if ref.kind == "environment_map"), None)
            live_environment = Path(environment_source).resolve() if metadata.get("viewerSettings", {}).get("environment") and environment_source else None
            if saved_environment != live_environment:
                raise ValueError(tr("error.save_hdr_first"))
        environment = metadata.get("viewerSettings", {}).get("environment")
        if environment:
            settings = lf.get_render_settings()
            if (settings.environment_mode != "EQUIRECTANGULAR" or str(settings.environment_map_path) != environment_source
                    or float(settings.environment_exposure) != environment["exposure"]
                    or float(settings.environment_rotation_degrees) != environment["rotation"]):
                raise ValueError("The HDR background changed. Review the current view and try uploading again.")
        if ui._patch_saved_update(metadata, project_id, path, update=update, expected_commit=expected_commit):
            return
        nodes = [n.name for n in ui._visible_splats()]
        if not nodes:
            raise ValueError("There are no visible splats to upload.")
        if upload_format not in ("studio", "sog", "ssog", "spz"):
            raise ValueError("Choose a supported upload format.")
        if "licht" not in ui.service.snapshot().get("source_formats", []):
            raise ValueError(UNSUPPORTED_PORTAL)
        export = ui.service.root / (str(uuid.uuid4()) + ".scene")
        metadata = dict(metadata)
        metadata["_commitUuid"] = str(getattr(inspection, "commit_uuid", ""))
        file_uuid = str(getattr(inspection, "file_uuid", ""))
        if file_uuid:
            metadata["originFileUuid"] = file_uuid
        metadata["_uploadFormat"] = upload_format
        ui._message = "Preparing the current scene for upload…"
        ui._refresh_model()
        if lf.ui.get_export_state().get("active"):
            raise ValueError("Wait for the current export to finish before uploading.")
        ui._export_cancelled = False
        ui._pin_publish_preview(metadata, path, metadata["_commitUuid"])
        ui._prepared_commit = metadata["_commitUuid"]
        ui._export_identity = identity
        ui._export_progress = 0
        lf.prepare_gallery_project(path, str(export), "ply" if upload_format == "studio" else upload_format,
                                   ui._prepared_commit)
        if ui.service.identity() != identity:
            ui._export_cancelled = True
        ui._export_pending = (export, metadata, project_id, time.monotonic())
        ui._schedule_poll()

    def owns_export(self, state):
        ui = self.ui
        return bool(ui._export_pending and state.get("path")
            and Path(state["path"]) == Path(ui._export_pending[0]))

    def cancel_export(self):
        ui = self.ui
        state = lf.ui.get_export_state()
        if ui._owns_export(state) and state.get("active"):
            lf.ui.cancel_export()

    def remove_preparation(self, export):
        ui = self.ui
        if Path(export).suffix == ".scene":
            for path in gallery_preparation.staging_files(ui.service.root, export):
                ui.service._unlink_temporary(path)
        else:
            Path(export).unlink(missing_ok=True)

    def advance(self):
        ui = self.ui
        export, metadata, project_id, started = ui._export_pending
        prepared_commit = ui._prepared_commit
        if ui._export_identity is not None and ui.service.identity() != ui._export_identity:
            ui._export_cancelled = True
        state = lf.ui.get_export_state()
        if not ui._owns_export(state):
            ui._export_pending = None
            ui._prepared_commit = None
            ui._remove_preparation(export)
            ui._message = "The scene preparation status changed. Please prepare your upload again."
            ui._refresh_model()
            return
        if state.get("active"):
            progress = max(0, min(100, int(float(state.get("progress", 0)) * 100)))
            if progress != ui._export_progress:
                ui._export_progress = progress
                ui._message = "Canceling scene preparation…" if ui._export_cancelled else f"Preparing scene for upload… {progress}%"
                ui._refresh_model()
            return
        outcome = state.get("outcome")
        if ui._export_cancelled or outcome in ("failed", "cancelled"):
            ui._prepared_commit = None
            ui._export_pending = None
            ui._remove_preparation(export)
            error = str(state.get("error", ""))
            ui._message = "Scene preparation canceled." if ui._export_cancelled or outcome == "cancelled" else (error or "Scene preparation failed. Check the export status and try again.")
            if outcome == "failed" and not ui._export_cancelled:
                ui._preparation_failure = {"id": "preparation:" + project_id, "project": project_id,
                    "commitUuid": prepared_commit,
                    "status": "error", "kind": "upload", "metadata": {"title": ui._operation_title},
                    "message": ui._message, "failureReason": ui._message}
                log_failure("native_preparation", RuntimeError(ui._message), project_id=project_id)
            ui._refresh_model()
        elif outcome == "completed" and export.exists():
            ui._export_pending = None
            source = ui._prepared_commit
            ui._prepared_commit = None
            try:
                if source is not None:
                    commit = str(state.get("commit_uuid", ""))
                    if not commit or (source and commit != source):
                        raise ValueError("The prepared project commit does not match the reviewed version.")
                    metadata["_commitUuid"] = commit
                    saved_view = gallery_preparation.publication_view_metadata(ui.service.root, export)
                    metadata["viewerSettings"] = saved_view | metadata.get("viewerSettings", {})
                ui.service.queue_prepared_upload(export, metadata, project_id)
                ui._message = ""
            except Exception as exc:
                log_failure("queue_after_preparation", exc, project_id=project_id)
                ui._preparation_failure = {"id": "preparation:" + project_id, "project": project_id,
                    "commitUuid": prepared_commit,
                    "status": "error", "kind": "upload", "metadata": {"title": ui._operation_title},
                    "message": friendly_error(exc), "failureReason": friendly_error(exc)}
                try:
                    ui._remove_preparation(export)
                    ui._message = friendly_error(exc)
                except (OSError, ValueError) as cleanup_exc:
                    log_failure("preparation_cleanup", cleanup_exc, project_id=project_id)
                    ui._message = "The upload could not be queued. Temporary files were kept; open the recovery folder to review them."
            ui._refresh_model()
        elif time.monotonic() - started > 60:
            ui._prepared_commit = None
            ui._export_pending = None
            ui._remove_preparation(export)
            ui._message = "LichtFeld Studio could not prepare the scene. Check the export status and try again."
            ui._preparation_failure = {"id": "preparation:" + project_id, "project": project_id,
                "commitUuid": prepared_commit,
                "status": "error", "kind": "upload", "metadata": {"title": ui._operation_title},
                "message": ui._message, "failureReason": ui._message}
            log_failure("native_preparation_timeout", TimeoutError(ui._message), project_id=project_id)
            ui._refresh_model()
