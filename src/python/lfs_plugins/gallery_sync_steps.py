# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native steps of a gallery update, ordered around a durable recovery copy."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import lichtfeld as lf
from .gallery_sync import file_stamp
from .gallery_view import restore_view
from .portal_gallery import domain_tokens
from .gallery_logging import failure as log_failure


class LocalUpdateSteps:
    """Stage, import, save, back up, apply, save, and link a downloaded update.

    The preview may enter the open scene before backup, but saved local splats,
    view and project content may change only after the backup is ready and the
    saved generation and stamp still match. Cancellation through backup removes
    the owned preview. After the updated save, the recovery copy is retained and
    linking may be skipped; an apply failure reopens the unchanged saved file.
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
    """Stage and open a saved download, register it, then persist its link.

    Cancellation before native open leaves the download alone. Once opening
    starts, cancellation keeps the new local file but skips registration and
    linking. After link submission, the link result must be checked before the
    UI can report completion.
    """

    def __init__(self, ui):
        self.ui = ui

    def cancel(self):
        ui = self.ui
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
            if ui._import_detached or job.get("_opening", {}).get("canceled"):
                ui._import_pending = None
                ui._message = "Account changed. The downloaded project is kept locally."
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
