# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gallery publishing and recovery from Asset Manager."""
from __future__ import annotations

import uuid
import time
import re
import threading
from pathlib import Path

import lichtfeld as lf

from .gallery_sync import get_gallery_sync, friendly_error, file_stamp
from .gallery_view import capture_view, restore_view
from . import gallery_preparation
from .panels import panel_class
from .types import Panel

__lfs_panel_classes__ = ["GalleryPanel"]
__lfs_panel_ids__ = ["lfs.gallery"]


@panel_class("gallery")
class GalleryPanel(Panel):
    def __init__(self):
        super().__init__()
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
        self._export_cancelled = False
        self._export_identity = None
        self._export_progress = 0
        self._search = ""
        self._phase_poll_scheduled = False
        self._import_started = None
        self._import_detached = False
        self._project_link = None
        self._requested_project = None
        self._focus_project = False
        self._reset_account_ui = False
        self._history_limit = 30
        self._native_use = None
        self._progress_pending = False

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("gallery_sync")
        if model is None:
            return
        for field in ("title", "description", "visibility", "upload_format"):
            model.bind(field, lambda f=field: getattr(self, "_"+f),
                lambda value, f=field: setattr(self, "_"+f, str(value)))
        model.bind_func("panel_label", lambda: "Gallery")
        model.bind_func("message", lambda: self._message or self._state["message"])
        model.bind_func("recovery_job", lambda: next((j["id"] for j in reversed(self._state["jobs"])
            if j.get("localUpdate", {}).get("backupPath")), ""))
        model.bind_func("busy", lambda: self._state["busy"] or bool(self._export_pending or self._import_pending or self._save_pending or self._native_use))
        model.bind_func("connected", lambda: self._state["connected"])
        model.bind_func("storage_issue", lambda: self._state.get("storage_issue", False))
        model.bind_func("signed_in", lambda: self._state["signed_in"])
        model.bind_func("account_name", lambda: self._state["display_name"] or self._state["email"] or "Your LichtFeld account")
        model.bind_func("account_email", lambda: self._state["email"] if self._state["display_name"] else "")
        model.bind_func("account_action", lambda: "Manage account" if self._state["signed_in"] else "Sign in")
        model.bind_func("gallery_count", lambda: str(len(self._visible_scenes(False))))
        model.bind_func("empty_gallery", lambda: not self._visible_scenes(False))
        model.bind_func("no_matches", lambda: bool(self._visible_scenes(False)) and not self._visible_scenes())
        model.bind_func("has_jobs", lambda: bool(self._state["jobs"]))
        model.bind_func("can_clear_finished", lambda: bool(self._clearable_jobs()))
        model.bind_func("more_history", lambda: sum(j["status"] in ("completed", "canceled") for j in self._state["jobs"]) > self._history_limit)
        model.bind_func("editor_heading", lambda: "Gallery item details" if self._scene else "Publish your current scene")
        model.bind_func("visibility_hint", lambda: "Hidden from public galleries. Existing scene or gallery share links still work." if self._visibility == "private" else "Anyone can view this splat. It may appear in Explore.")
        model.bind_func("can_pause", self._can_pause)
        model.bind_func("pause_label", lambda: "Cancel gallery action" if self._save_pending else "Cancel import" if self._import_pending else "Cancel preparation" if self._export_pending else
            "Stop waiting" if any(j.get("serverProcessing") and j["status"] == "running" for j in self._state["jobs"]) else "Pause transfer")
        model.bind_func("preparing", lambda: bool(self._export_pending or self._import_pending or self._save_pending))
        model.bind_func("preparation_progress", lambda: self._export_progress)
        model.bind("search", lambda: self._search, self._set_search)
        model.bind_func("selected", lambda: bool(self._scene))
        model.bind_func("linked", lambda: bool(self._project_link))
        model.bind_func("needs_project", lambda: bool(self._requested_project))
        model.bind_func("requested_project_name", lambda: Path(self._requested_project).stem if self._requested_project else "")
        model.bind_func("current_project_name", lambda: Path(lf.project_poll_write().get("path") or "Unsaved project").stem)
        model.bind_func("linked_title", lambda: self._project_link.get("metadata", {}).get("title", "Gallery splat") if self._project_link else "")
        model.bind_func("confirm", lambda: bool(self._confirm))
        model.bind_func("confirm_text", lambda: self._confirm[0] if self._confirm else "")
        model.bind_func("confirm_label", lambda: self._confirm[2] if self._confirm else "")
        model.bind_func("publish_label", lambda: "Replace selected splat" if self._scene else "Upload new splat")
        model.bind_record_list("scenes")
        model.bind_record_list("jobs")
        for name in ("refresh", "account", "select", "new", "publish", "edit", "remove", "unlink",
                     "resume", "discard", "pause", "confirm_action", "cancel_action", "open", "download", "import", "resolve", "linked", "open_project", "use_current", "update_local", "show_backup", "show_recovery_folder", "clear_finished", "more_history", "transfer_progress"):
            model.bind_event(name, lambda _handle, _event, args, h=name: self._dispatch(h, args))
        self._handle = model.get_handle()

    def _dispatch(self, name, args):
        self._progress_pending = False
        try:
            if self._check_identity() and name not in ("account", "refresh"):
                raise ValueError("The account changed. Review your gallery before continuing.")
            self._message = ""
            self._release_native_use()
            if (self.service.busy or self._save_pending or self._import_pending or self._export_pending or self._native_use) and name not in ("pause", "account", "cancel_action", "transfer_progress"):
                raise ValueError("Wait for the operation to finish or pause the transfer.")
            getattr(self, "_action_"+name)(*args)
            self._progress_pending = name in ("publish", "confirm_action", "download", "resume", "resolve", "import", "update_local")
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

    def _can_pause(self):
        if self._export_pending or self._save_pending:
            return True
        job = self._import_pending
        if job:
            if job.get("_save_new") or job.get("_link") or job.get("_update", {}).get("phase") in ("save_updated", "linking"):
                return False
            return bool(job.get("_update") or job.get("_bundle"))
        return any(j["status"] == "running" for j in self._state["jobs"])

    def _check_identity(self):
        identity = self.service.identity()
        if identity == self._identity:
            return False
        self._identity = identity
        self.service.pause()
        self._scene = self._confirm = None
        self._title = self._description = self._search = ""
        self._visibility = "private"
        self._history_limit = 30
        self._requested_project = None
        self._focus_project = False
        self._reset_account_ui = True
        self._progress_pending = False
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

    def _visible_scenes(self, filtered=True):
        query = self._search.strip().casefold() if filtered else ""
        return [s for s in self._state["scenes"] if s["status"] == "ready"
            and (not query or query in s["title"].casefold())]

    def _set_search(self, value):
        self._search = str(value)
        self._refresh_model()

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
            self._confirm = (f"Clear {len(identifiers)} finished transfers and their temporary downloads? "
                "You can download gallery items again. Your saved projects, gallery items, links and recovery copies will remain.",
                lambda: self.service.clear_finished(identifiers), "Clear finished transfers")

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
        if self._requested_project:
            current_path = lf.project_poll_write().get("path")
            if current_path and Path(current_path).resolve() == Path(self._requested_project).resolve():
                self._requested_project = None
                self._focus_project = True
        if self._focus_project and self._state["connected"]:
            self._focus_project = False
            self._search = ""
            if self._project_link and any(s["id"] == self._project_link["sceneId"] for s in self._visible_scenes(False)):
                self._action_select(self._project_link["sceneId"])
            else:
                self._action_new()
        if self._scene:
            current = next((s for s in self._state["scenes"] if s["id"] == self._scene["id"] and s["status"] == "ready"), None)
            fields = ("title", "description", "visibility")
            if current is None:
                self._scene = self._confirm = None
            elif all(getattr(self, "_" + field) == self._scene[field] for field in fields):
                self._scene = current
                self._title, self._description, self._visibility = [current[field] for field in fields]
            elif all(getattr(self, "_" + field) == current[field] for field in fields):
                self._scene = current
        if self._handle:
            self._handle.update_record_list("scenes", [dict(s, selected=bool(self._scene and self._scene["id"] == s["id"]))
                for s in self._visible_scenes()])
            self._handle.update_record_list("jobs", [dict(j, title=j["metadata"]["title"], progress="Kept" if j.get("retired") else "100%" if j["status"] == "completed" else f'{100*j["completed"]/max(1,j["total"]):.0f}%',
                recovery_only=bool(j.get("retired")),
                progress_value=100 if j["status"] == "completed" else min(100, 100*j["completed"]/max(1,j["total"])),
                importable=j.get("kind") == "download" and j["status"] == "completed" and not (j.get("retired") or j.get("cleanupPending")),
                updatable=bool(j.get("kind") == "download" and j["status"] == "completed" and not (j.get("retired") or j.get("cleanupPending")) and self._project_link and self._project_link["sceneId"] == j.get("result", {}).get("id")),
                has_backup=bool(j.get("localUpdate", {}).get("backupPath")),
                conflicted=j["status"] == "conflict",
                resume_label="Restart download" if j.get("kind") == "download" else "Resume upload",
                discard_label="Discard download" if j.get("kind") == "download" else "Discard upload",
                direction="Recovery copy" if j.get("retired") else "Download" if j.get("kind") == "download" else "Upload",
                resumable=j["status"] in ("paused", "error"), discardable=j["status"] in ("paused", "error", "conflict"))
                for j in self._visible_jobs()])
            self._handle.dirty_all()

    def on_mount(self, doc):
        super().on_mount(doc)
        if not self._title:
            self._action_new()
        self._refresh_model()
        if self._state["signed_in"] and not self._state["connected"] and not self.service.busy:
            self.service.refresh()

    def on_update(self, doc):
        self._check_identity()
        reset_ui = self._reset_account_ui
        if reset_ui:
            self._reset_account_ui = False
            # A different user's gallery starts at the top without retaining
            # keyboard focus or scrolling to the previous user's edited field.
            for selector in ("#gallery-search", "#gallery-title", "#gallery-description", "#gallery-visibility"):
                element = doc.query_selector(selector)
                if element is not None:
                    element.blur()
            for selector in (".gallery-content", ".gallery-list"):
                element = doc.query_selector(selector)
                if element is not None:
                    element.scroll_top = 0
        if self._export_pending or self._import_pending or self._save_pending or self._native_use:
            self._advance_phases()
        key = (self.service.state_key(), lf.project_poll_write().get("path"))
        if key != self._version:
            snapshot = self.service.snapshot()
            if not snapshot["connected"]:
                self._scene = self._confirm = None
            self._version = key
            self._refresh_model()
            self._maybe_show_transfer_progress()
            return True
        return reset_ui

    def on_unmount(self, doc):
        self._handle = None
        super().on_unmount(doc)

    def _schedule_phase_poll(self):
        # Native export/import must finish even when the user closes this panel.
        # Only scheduling runs on the timer; all app access stays on the UI thread.
        if self._phase_poll_scheduled or not (self._export_pending or self._import_pending or self._save_pending or self._native_use):
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
            self._export_pending = self._import_pending = self._save_pending = None
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

    def focus_project(self, path=None):
        """Asset Manager supplies an explicit local source before publishing."""
        if self.service.busy or self._save_pending or self._import_pending or self._export_pending:
            self._message = "Finish or pause the current gallery operation before choosing another project."
            self._refresh_model()
            return
        self._check_identity()
        self._requested_project = str(Path(path).resolve()) if path else None
        self._focus_project = path is None
        self._confirm = None
        self._refresh_model()

    def _action_open_project(self):
        if self._requested_project:
            from .training_confirm import confirm_discard_work_then
            requested, identity = self._requested_project, self._identity

            def open_checked(stop_training):
                try:
                    self._check_identity()
                    if self._identity != identity or self._requested_project != requested:
                        raise ValueError("The account or selected project changed. Review your gallery before continuing.")
                    if self.service.busy or self._save_pending or self._import_pending or self._export_pending:
                        raise ValueError("Finish or pause the current gallery operation before opening another project.")
                    lf.project_open(requested, True, stop_training, True)
                except Exception as exc:
                    self._message = friendly_error(exc)
                self._refresh_model()

            confirm_discard_work_then("Open selected project", open_checked)

    def _action_use_current(self):
        self.focus_project()

    def _action_select(self, scene_id):
        self._scene = next(s for s in self._state["scenes"] if s["id"] == scene_id)
        self._title, self._description, self._visibility = [self._scene[k] for k in ("title", "description", "visibility")]
        self._confirm = None

    def _action_new(self):
        self._scene = self._confirm = None
        self._description, self._visibility = "", "private"
        path = lf.project_poll_write().get("path")
        self._title = Path(path).stem if path else "Untitled splat"

    def _action_linked(self):
        if self._project_link:
            scene_id = self._project_link["sceneId"]
            if not any(s["id"] == scene_id for s in self._visible_scenes(False)):
                raise ValueError("The linked gallery item is unavailable. Refresh your gallery, or unlink this project.")
            self._search = ""
            self._action_select(scene_id)

    def _details(self):
        title = self._title.strip()
        if not title or len(title) > 120 or len(self._description) > 5000:
            raise ValueError("Enter a title up to 120 characters and a description up to 5,000 characters.")
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
            raise ValueError("The project could not be saved. Your gallery operation was stopped; resolve the save error before retrying.")
        if pending.get("canceled") or self.service.identity() != pending["identity"]:
            self._message = "Project save finished. The gallery operation was canceled."
            return
        if (poll.get("generation") != pending["generation"] or self._project_identity() != pending["project"] or lf.project_is_dirty()):
            raise ValueError("The project changed while saving. Your gallery operation was stopped; review your work and try again.")
        pending["continuation"]()

    def _action_publish(self):
        if self._requested_project:
            raise ValueError("Open the selected Asset Manager project first, or choose Use current project.")
        project = self._project_identity()
        metadata = self._details()
        upload_format = self._upload_format
        if upload_format not in ("studio", "sog", "ssog"):
            raise ValueError("Choose a supported upload format.")
        metadata["viewerSettings"] = capture_view(lf)
        environment_source = str(lf.get_render_settings().environment_map_path) if metadata["viewerSettings"].get("environment") else None
        scene = dict(self._scene) if self._scene else None
        if scene:
            metadata.update(replaceSceneId=scene["id"], baseRevision=scene["revision"])
        title = scene["title"] if scene else metadata["title"]
        if scene or metadata["visibility"] == "public":
            message = (f'Replace “{title}” with the current visible splats?' if scene else
                f'Publish the current visible splats as “{title}”?')
            if metadata["visibility"] == "public":
                message += " Anyone can view it, and it may appear in Explore."
            self._confirm = (message, lambda: self._publish(metadata, expected_project=project, environment_source=environment_source, upload_format=upload_format), "Replace splat" if scene else "Publish splat")
        else:
            self._publish(metadata, expected_project=project, environment_source=environment_source, upload_format=upload_format)

    def _publish(self, metadata, *, expected_project=None, environment_source=None, upload_format="studio"):
        identity = self.service.identity()
        project_id, path = self._project_identity()
        if expected_project is not None and (project_id, path) != expected_project:
            raise ValueError("The current project changed. Review its gallery details before uploading.")
        pending = [j for j in self._state["jobs"] if j["project"] == project_id and j["status"] not in ("completed", "canceled")]
        if pending:
            raise ValueError("This project already has an upload. Resume or discard it first.")
        linked = self._state["links"].get(project_id)
        if linked and metadata.get("replaceSceneId") != linked["sceneId"]:
            raise ValueError("This project is linked to a gallery item. Select it to replace, or unlink before publishing a new item.")
        nodes = [n.name for n in self._visible_splats()]
        if not nodes:
            raise ValueError("There are no visible splats to upload.")
        self._save_current_project(lambda: self._publish_saved(metadata, project_id, path, identity, environment_source, upload_format))

    def _publish_saved(self, metadata, project_id, path, identity, environment_source=None, upload_format="studio"):
        if self.service.identity() != identity or self._project_identity() != (project_id, path):
            raise ValueError("The account or current project changed while saving. Review it before uploading.")
        nodes = [n.name for n in self._visible_splats()]
        if not nodes:
            raise ValueError("There are no visible splats to upload.")
        if upload_format not in ("studio", "sog", "ssog"):
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
        self._message = "Preparing the current scene for upload…"
        self._refresh_model()
        if lf.ui.get_export_state().get("active"):
            raise ValueError("Wait for the current export to finish before uploading.")
        self._export_cancelled = False
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
            self._export_pending = None
            self._remove_preparation(export)
            self._message = "Scene preparation canceled." if self._export_cancelled or outcome == "cancelled" else "Scene preparation failed. Check the export status and try again."
            self._refresh_model()
        elif outcome == "completed" and export.exists():
            self._export_pending = None
            try:
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
            self._export_pending = None
            self._remove_preparation(export)
            self._message = "Studio could not prepare the scene. Check the export status and try again."
            self._refresh_model()

    def _action_edit(self):
        if self._scene:
            scene, details = dict(self._scene), self._details()
            action = lambda: self.service.edit(scene["id"], scene["revision"], details)
            if scene["visibility"] != "public" and details["visibility"] == "public":
                self._confirm = (f'Make “{details["title"]}” public? Anyone can view it, and it may appear in Explore.', action, "Make public")
            else:
                action()

    def _action_remove(self):
        if self._scene:
            scene = dict(self._scene)
            self._confirm = (f'Remove “{scene["title"]}” from the gallery? Its share links will stop working. Your local project will remain.',
                lambda: self.service.remove(scene["id"], scene["revision"]), "Remove from gallery")

    def _action_unlink(self):
        project_id, _ = self._project_identity()
        self._confirm = ("Unlink this project? Both the local project and gallery item will remain.", lambda: self.service.unlink(project_id), "Unlink project")

    def _action_resume(self, job_id):
        self.service.resume(job_id)

    def _action_discard(self, job_id):
        job = next(j for j in self._state["jobs"] if j["id"] == job_id)
        kind = "download" if job.get("kind") == "download" else "upload"
        self._confirm = (f"Discard this unfinished {kind}? The gallery splat and local project will remain.", lambda: self.service.discard(job_id), f"Discard {kind}")

    def _action_resolve(self, job_id):
        job = next(j for j in self._state["jobs"] if j["id"] == job_id)
        scene = next((s for s in self._state["scenes"] if s["id"] == job["metadata"].get("replaceSceneId") and s["status"] == "ready"), None)
        if scene is None:
            raise ValueError("Refresh the gallery to review the latest version first.")
        self._confirm = (f'Replace the geometry of “{scene["title"]}” with your pending upload, keeping the gallery’s current description, visibility and view settings? Uploaded parts will be reused.',
            lambda: self.service.resolve_conflict(job_id, scene), "Replace and resume")

    def _action_pause(self):
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
        if self._import_pending and self._import_pending.get("_bundle"):
            bundle = self._import_pending["_bundle"]
            bundle["canceled"] = True
            if bundle["phase"] == "importing":
                lf.ui.cancel_gallery_import()
        self.service.pause()

    def _action_download(self):
        if self._scene:
            self.service.download(dict(self._scene))

    def _action_import(self, job_id):
        job = next(j for j in self._state["jobs"] if j["id"] == job_id)
        self._confirm = (f'Open “{job["result"]["title"]}” as a new local project? Your current project will be saved first.',
            lambda: self._import_download(job), "Open new project")

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
            operation = self.service.link_download(job["id"], str(lf.io.inspect_project(expected["path"]).project_uuid))
            job.pop("_native_project")
            job["_link"] = operation
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
                lf.project_open(stage["projectPath"])
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
            path = directory / (filename + "-" + str(uuid.uuid4())[:8] + ".licht")
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
            operation = self.service.link_download(job["id"], identifier)
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
        self._confirm = (f'Update “{Path(project[1]).stem}” from the downloaded gallery version? '
            "Its visible splats and camera settings will be replaced. A recovery copy will keep your local work; hidden splats and other scene objects will remain.",
            lambda: self._begin_local_update(job, project), "Update linked project")

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
                lf.project_open(project[1], discard_changes=True)
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
            update["link_operation"] = self.service.link_download(job["id"], update["project"][0])
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

    def _action_open(self):
        if self._scene:
            # Use our pinned origin and scene id, never an arbitrary server-provided URL.
            lf.ui.open_url(self.service.account.base_url + "/gallery/scenes/" + str(uuid.UUID(self._scene["id"])) + "/")
