# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gallery file actions and transfer status in the Asset Manager."""
from __future__ import annotations

import time
import copy
import threading
from pathlib import Path

import lichtfeld as lf
from .gallery_messages import tr

from .gallery_controller import asset_sync_state, get_gallery_controller

SCOPE_PUBLISHED = "__gallery__"
SCOPE_ATTENTION = "__gallery_attention__"
SCOPE_TRANSFERS = "__gallery_transfers__"
GALLERY_DRAG_PAYLOAD_TYPE = "application/x-lichtfeld-gallery-scene"
GALLERY_SCOPES = (SCOPE_PUBLISHED, SCOPE_ATTENTION)



def relative_time(timestamp, *, now=None):
    """Display elapsed time only; never use it to decide sync freshness."""
    elapsed = max(0, (time.time() if now is None else now) - timestamp)
    if elapsed < 60:
        return tr("time.just_now")
    if elapsed < 3600:
        return tr("time.minutes", count=int(elapsed // 60))
    if elapsed < 86400:
        return tr("time.hours", count=int(elapsed // 3600))
    if elapsed < 172800:
        return tr("time.yesterday")
    return tr("time.days", count=int(elapsed // 86400))


class GalleryAssetMixin:
    def _init_gallery(self):
        self._gallery_controller = None
        self._gallery_unsubscribe = None
        self._gallery_state = {"scenes": [], "links": {}, "jobs": [], "signed_in": False}
        self._gallery_upload_format = "sog"
        self._gallery_pull_folder = ""
        self._gallery_pull_name = ""
        self._gallery_batch = []
        self._gallery_batch_waiting = False
        self._gallery_batch_asset = None
        self._gallery_notice = ""
        self._gallery_undo = None
        self._gallery_undo_kind = ""
        self._gallery_undo_backup = None
        self._gallery_last_folder = None
        self._gallery_focus_path = None
        self._gallery_undo_timer = None
        self._gallery_pulled_job = None
        self._gallery_completion_id = None
        self._gallery_toast = None
        self._gallery_toast_timer = None

    def _controller(self):
        if self._gallery_controller is None:
            self._gallery_controller = get_gallery_controller()
            self._gallery_upload_format = self._gallery_controller.upload_format
        return self._gallery_controller

    def _subscribe_gallery(self):
        if self._gallery_unsubscribe is None:
            self._gallery_unsubscribe = self._controller().subscribe(self._gallery_changed)

    def _gallery_changed(self, snapshot):
        previous_identity = self._gallery_state.get("identity")
        previous = self._gallery_state
        self._gallery_state = snapshot
        if snapshot.get("relink_required"):
            self._gallery_notice = snapshot.get("message", "")
        if previous_identity != snapshot.get("identity"):
            self._gallery_undo = None
            self._gallery_batch = []
            self._gallery_batch_waiting = False
            self._gallery_batch_asset = None
            self._gallery_notice = ""
            self._gallery_pulled_job = None
            self._gallery_toast = None
            self._gallery_completion_id = (snapshot.get("completion") or {}).get("id")
        self._gallery_completions(previous, snapshot)
        pulled = snapshot.get("pulledProject")
        if pulled and pulled["jobId"] != self._gallery_pulled_job and self._asset_index:
            if self._asset_index.load():
                self._gallery_pulled_job = pulled["jobId"]
                self._select_folder_id(SCOPE_PUBLISHED)
                self._select_asset_id(pulled["id"])
                self._refresh_records(assets=True, folders=True)
                self._gallery_notice = tr("info.pulled")
        self._repair_selection()
        self._refresh_records(assets=True)
        rebuild = getattr(self._asset_index, "rebuild_gallery_projection", None)
        if callable(rebuild):
            # A projection contains no titles, tokens, or private scene cache.
            projection = {identifier: {"sceneId": link["sceneId"],
                "state": self._gallery_facts(asset)["state"], "checkedAt": link.get("checkedAt", 0)}
                for identifier, asset in self._asset_index_assets().items()
                if (link := snapshot.get("links", {}).get(identifier))}
            rebuild(projection)
        undo = snapshot.get("undoPull")
        undo_token = (undo.get("backup"), undo.get("attempt", 0)) if undo else None
        if undo and undo_token != self._gallery_undo_backup:
            self._gallery_undo_backup = undo_token
            self._gallery_notice = undo.get("error", "")
            if undo.get("backupMissing"):
                self._gallery_undo = None
            else:
                self._set_gallery_undo(self._controller().undo_pull, kind="pull")
        elif not undo and self._gallery_undo_kind == "pull":
            self._gallery_undo = None
        controller = self._gallery_controller
        if self._gallery_batch_waiting and controller and not controller._panel_busy():
            facts = self._gallery_facts(self._gallery_batch_asset or self._get_selected_asset() or {})
            self._gallery_batch_waiting = False
            if facts["activity"] in ("error", "paused", "interrupted") or facts["freshness"] == "diverged" or controller._last_canceled:
                self._gallery_batch = []
            elif self._gallery_batch:
                identifier, action = self._gallery_batch.pop(0)
                self._select_asset_id(identifier)
                self._begin_gallery_publish(self._get_selected_asset(), action)
        if self._handle:
            self._handle.dirty_all()
        self._request_model_update()

    def _has_gallery_link(self):
        asset = self._get_selected_asset() or {}
        return asset.get("id") in self._gallery_state.get("links", {}) or bool(self._gallery_scene(asset))

    def _gallery_scene(self, asset):
        scene_id = asset.get("scene_id") or self._gallery_state.get("links", {}).get(asset.get("id"), {}).get("sceneId")
        return next((s for s in self._gallery_state.get("scenes", []) if s.get("id") == scene_id), None)

    def _gallery_facts(self, asset):
        remote = asset.get("remote_only", False)
        link = self._gallery_state.get("links", {}).get(asset.get("id"))
        phase = "idle"
        controller = self._gallery_controller
        if controller and getattr(controller, "_operation_project", None) == asset.get("id"):
            phase = controller.phase()
        facts = asset_sync_state(None if remote else asset, link, self._gallery_scene(asset),
            self._gallery_state.get("jobs", ()), checked=bool(self._gallery_state.get("checkedAt")),
            storage_issue=self._gallery_state.get("storage_issue", False), phase=phase,
            cached_projection=asset.get("gallery") if "identity" not in self._gallery_state else None)
        return facts

    def _asset_with_poster(self, asset):
        scene = self._gallery_scene(asset) or {}
        poster = self._gallery_state.get("posters", {}).get(scene.get("id"), "")
        return {**asset, "poster_path": poster, "prefer_poster": self._selected_folder_id in GALLERY_SCOPES}

    def _gallery_remote_assets(self):
        linked = {link["sceneId"] for identifier, link in self._gallery_state.get("links", {}).items()
                  if identifier in self._asset_index_assets()}
        return {"remote:" + s["id"]: {"id": "remote:" + s["id"], "scene_id": s["id"],
                "remote_only": True, "name": s.get("title", ""), "path": "", "exists": True,
                "available": False, "file_size_bytes": s.get("contentLength", 0),
                "source_format": s.get("sourceFormat", "licht"), "has_preview": False}
                for s in self._gallery_state.get("scenes", []) if s.get("status") == "ready" and s["id"] not in linked}

    def _all_display_assets(self):
        return {**self._asset_index_assets(), **self._gallery_remote_assets()}

    def _gallery_rows(self, attention=False):
        transferring = {j.get("project") for j in self._gallery_state.get("jobs", ())
                        if not j.get("retired") and j.get("status") not in ("completed", "canceled")}
        controller = self._gallery_controller
        if controller and self._gallery_state.get("phase", "idle") != "idle":
            transferring.add(getattr(controller, "_operation_project", None))
        rows = [a for a in self._asset_index_assets().values()
                if a.get("id") in self._gallery_state.get("links", {}) or a.get("id") in transferring]
        rows += list(self._gallery_remote_assets().values())
        if attention:
            # Failed first publishes are also actionable, even before a link exists.
            rows = list(self._all_display_assets().values())
            rows = [a for a in rows if self._gallery_facts(a)["attention"]]
        return rows

    def _gallery_badge(self, asset):
        facts = self._gallery_facts(asset)
        state_key = "state." + facts["state"]
        label = tr(state_key, percent=facts["progress"])
        if facts["reason"] and facts["state"] == "error":
            label = tr("state.with_reason", state=label, reason=facts["reason"])
        if asset.get("remote_only"):
            label = tr("state.remote_detail", state=label, size=self._format_size(asset.get("file_size_bytes")),
                       format=asset.get("source_format", "licht").upper())
        job = next((j for j in self._gallery_state.get("jobs", ()) if j["id"] == facts["jobId"]), {})
        native = facts["activity"] in ("preparing", "applying") and not job
        can_cancel = native or bool(job and job.get("status") not in ("completed", "canceled"))
        can_pause = job.get("status") == "running"
        scene = self._gallery_scene(asset) or {}
        stored = bool(scene.get("status") == "ready" and self._gallery_state.get("connected")
                      and self._gallery_state.get("checkedAt") and not self._gallery_state.get("offline"))
        indeterminate = facts["active"] and (facts["activity"] in ("preparing", "processing", "applying")
                                            or (facts["activity"] in ("uploading", "downloading") and not job.get("total")))
        detail = job.get("transferDetail", "")
        byte_label = tr("bytes", prefix="gallery.transfer.", done=self._format_size(job.get("completed", 0)),
                        total=self._format_size(job["total"])) if facts["active"] and job.get("total") else ""
        return {"gallery_state": facts["state"], "gallery_label": label,
                "gallery_detail": detail, "gallery_bytes": byte_label,
                "gallery_has_bytes": bool(byte_label),
                "gallery_stored": stored,
                "gallery_stored_label": tr("state.stored") + " · " + self._gallery_checked_label() if stored else "",
                "gallery_can_pause": can_pause, "gallery_can_cancel": can_cancel,
                "gallery_action_persistent": can_cancel or facts["action"] in ("retry", "resume"),
                "gallery_has_controls": can_cancel or bool(facts["action"]),
                "gallery_tooltip": "\n".join(filter(None, (label, byte_label, detail))),
                "gallery_progress_width": f"{35 if indeterminate else facts['progress']}%",
                "gallery_indeterminate": indeterminate,
                "gallery_icon": "../icon/gallery-" + facts["icon"] + ".png",
                "gallery_tone": "gallery-tone-" + facts["tone"],
                "gallery_has_action": bool(facts["action"]), "gallery_action": facts["action"], "gallery_action_label": tr("action." + facts["action"]) if facts["action"] else "",
                "gallery_progress": facts["progress"], "gallery_active": facts["active"],
                "remote_only": bool(asset.get("remote_only"))}

    def select_gallery_scope(self):
        self.focus_gallery()

    def focus_gallery(self, path=None):
        self._gallery_focus_path = path
        self._select_folder_id(SCOPE_PUBLISHED)
        if path:
            target = Path(path).resolve()
            asset = next((a for a in self._asset_index_assets().values() if Path(a["path"]).resolve() == target), None)
            if asset:
                self._gallery_focus_path = None
                if asset["id"] not in self._gallery_state.get("links", {}):
                    self._select_folder_id("__all__")
                self._select_asset_id(asset["id"])
        self._request_model_update()

    def _gallery_counts(self):
        selected = [a for key in self._selected_asset_ids if (a := self._asset_dict(key)) and not a.get("remote_only")]
        return {"count": len(selected),
                "ready": sum(self._project_available(a) and self._gallery_facts(a)["relationship"] == "unlinked" for a in selected),
                "linked": sum(self._project_available(a) and self._gallery_facts(a)["relationship"] == "linked" for a in selected),
                "missing": sum(not self._project_available(a) for a in selected)}

    def _gallery_checked_label(self):
        if not self._gallery_state.get("signed_in"):
            return tr("sidebar.sign_in_hint")
        if self._gallery_state.get("offline"):
            return tr("sidebar.offline")
        checked = self._gallery_state.get("checkedAt", 0)
        return tr("sidebar.checked_relative", time=relative_time(checked)) if checked else tr("state.unknown")

    def _gallery_aggregate(self):
        jobs = self._gallery_state.get("jobs", [])
        uploads = sum(j.get("status") in ("queued", "running") and j.get("kind") != "download" for j in jobs)
        downloads = sum(j.get("status") in ("queued", "running") and j.get("kind") == "download" for j in jobs)
        phase = self._gallery_state.get("phase", "idle")
        uploads += phase == "preparing"
        downloads += phase == "applying"
        attention = len(self._gallery_rows(True))
        return (" · ".join(([f"{uploads}↑"] if uploads else []) + ([f"{downloads}↓"] if downloads else [])) or ("!" if attention else ""),
                tr("sidebar.aggregate", uploads=uploads, downloads=downloads, attention=attention))

    def _bind_gallery_model(self, model):
        values = {
            "gallery_supported": lambda: not self._gallery_state.get("unsupported", False),
            "gallery_signed_in": lambda: self._gallery_state.get("signed_in", False) and not self._gallery_state.get("relink_required", False),
            "gallery_account": lambda: self._gallery_state.get("display_name") or self._gallery_state.get("email", ""),
            "gallery_checked": self._gallery_checked_label,
            "gallery_quota": self._gallery_quota,
            "gallery_has_toast": lambda: bool(self._gallery_toast),
            "gallery_toast": lambda: (self._gallery_toast or {}).get("text", ""),
            "gallery_toast_portal": lambda: bool((self._gallery_toast or {}).get("scene")),
            "gallery_toast_open": lambda: bool((self._gallery_toast or {}).get("path")),
            "gallery_update_all_visible": lambda: self._selected_folder_id in GALLERY_SCOPES,
            "gallery_update_all_label": lambda: tr("action.update_all", count=len(self._gallery_update_candidates())),
            "gallery_update_all_enabled": lambda: bool(self._gallery_update_candidates()) and not self._gallery_state.get("busy") and self._gallery_state.get("phase", "idle") == "idle",
            "gallery_empty": lambda: self._selected_folder_id == SCOPE_PUBLISHED and self._gallery_state.get("connected", False) and not self._gallery_state.get("scenes"),
            "gallery_local_empty": lambda: self._selected_folder_id not in GALLERY_SCOPES and not self._asset_index_assets(),
            "gallery_empty_pull": lambda: bool(self._gallery_state.get("scenes")) and not self._asset_index_assets(),
            "gallery_published_count": lambda: len(self._gallery_rows()),
            "gallery_attention_count": lambda: len(self._gallery_rows(True)),
            "gallery_transfer_count": lambda: sum(j.get("status") not in ("completed", "canceled") for j in self._gallery_state.get("jobs", [])),
            "gallery_overlay": lambda: self._gallery_aggregate()[0],
            "gallery_tooltip": lambda: " · ".join(filter(None, (self._gallery_aggregate()[1], self._gallery_quota()))),
            "gallery_selected_state": lambda: self._gallery_badge(self._get_selected_asset())["gallery_label"] if self._get_selected_asset() else "",
            "gallery_remote": lambda: bool((self._get_selected_asset() or {}).get("remote_only")),
            "gallery_linked": lambda: self._has_gallery_link(),
            "gallery_notice": lambda: self._gallery_notice or self._gallery_state.get("message", ""),
            "gallery_has_notice": lambda: bool(self._gallery_notice or self._gallery_state.get("message", "")),
            "gallery_exchange_summary": lambda: " · ".join(filter(None, (self._gallery_published_summary(), self._gallery_checked_label()))),
            "gallery_selected_format": lambda: ((self._get_selected_asset() or {}).get("source_format") or "licht").upper(),
            "gallery_selected_visibility": lambda: tr("review." + (self._gallery_scene(self._get_selected_asset() or {}) or {}).get("visibility", "private")),
            "gallery_multi_summary": lambda: tr("multi.summary", **self._gallery_counts()),
            "gallery_publish_many": lambda: tr("multi.publish", count=self._gallery_counts()["ready"]),
            "gallery_update_many": lambda: tr("multi.update", count=self._gallery_counts()["linked"]),
            "gallery_can_publish_many": lambda: not self._gallery_state.get("unsupported") and self._gallery_counts()["ready"] > 0,
            "gallery_can_update_many": lambda: not self._gallery_state.get("unsupported") and self._gallery_counts()["linked"] > 0,
            "gallery_has_undo": lambda: bool(self._gallery_undo and time.monotonic() < self._gallery_undo[0]
                and not (self._gallery_state.get("undoPull") or {}).get("operation")),
            "gallery_has_recovery": lambda: bool((self._gallery_state.get("undoPull") or {}).get("backupMissing")),
        }
        for name, getter in values.items():
            model.bind_func(name, getter)
        for key in ("sidebar.title", "sidebar.published", "sidebar.attention", "sidebar.transfers",
                    "review.visibility", "action.open", "action.copy", "action.undo", "action.cancel",
                    "info.format", "state.remote_only", "action.open_local", "action.open_recovery"):
            model.bind_func("g_" + key.replace(".", "_"), lambda k=key: tr(k))
        model.bind_func("g_action_pause", lambda: tr("action.pause", prefix="gallery.transfer."))
        for action in ("toast_open", "toast_portal", "toast_copy", "update_all", "refresh", "undo",
                       "publish_many", "update_many", "open_recovery"):
            model.bind_event("gallery_" + action, lambda _h, _e, args, a=action: self._gallery_command(a, args))

    def _gallery_context_items(self, asset):
        facts = self._gallery_facts(asset)
        scene = self._gallery_scene(asset)
        items = []
        if facts["action"] and facts["action"] != "open" and (self._project_available(asset) or asset.get("remote_only") or facts["relationship"] == "local_missing"):
            items.append({"label": tr("action." + facts["action"]), "action": "gallery:" + facts["action"], "separator_before": True})
        if asset.get("remote_only"):
            items.append({"label": tr("action.pull_open"), "action": "gallery:pull_open"})
        if facts["relationship"] == "remote_deleted" and facts["action"] != "unlink":
            items.append({"label": tr("action.unlink"), "action": "gallery:unlink"})
        if scene:
            items += [{"label": tr("action.open"), "action": "gallery:open"}, {"label": tr("action.copy"), "action": "gallery:copy"}]
        badge = self._gallery_badge(asset)
        if badge["gallery_can_pause"]:
            items.append({"label": tr("action.pause", prefix="gallery.transfer."), "action": "gallery:pause_transfer"})
        if badge["gallery_can_cancel"]:
            items.append({"label": tr("action.cancel"), "action": "gallery:cancel_transfer"})
        if scene:
            items.append({"label": tr("action.delete_upstream"), "action": "gallery:remove", "separator_before": True})
        return items

    def _gallery_command(self, action, args=()):
        try:
            if (self._gallery_state.get("unsupported") and action not in
                    ("refresh", "open_recovery", "undo", "toast_open", "toast_portal", "toast_copy")):
                self._gallery_notice = tr("error.portal_version")
                return
            if action.startswith("toast_"):
                toast = self._gallery_toast or {}
                if toast.get("identity") != self._gallery_state.get("identity"):
                    return
                if action == "toast_open" and toast.get("path"):
                    from .file_menu import open_project_with_confirmation
                    open_project_with_confirmation(toast["path"], keep_asset_manager_open=True)
                elif toast.get("scene"):
                    self._controller().open_portal(toast["scene"], "copy" if action == "toast_copy" else "open")
                return
            if action == "update_all":
                self._controller().update_all(self._gallery_update_candidates())
                return
            if action == "refresh":
                self._controller().refresh()
                return
            if action == "open_recovery":
                self._controller().command("show_recovery_folder")
                return
            if action == "undo" and self._gallery_undo and time.monotonic() < self._gallery_undo[0]:
                self._gallery_undo[1]()
                if self._gallery_undo_kind != "pull":
                    self._gallery_undo = None
                return
            if action in ("publish_many", "update_many"):
                relationship = "unlinked" if action == "publish_many" else "linked"
                self._gallery_batch = [(a["id"], "publish" if action == "publish_many" else "update")
                    for key in sorted(self._selected_asset_ids) if (a := self._asset_dict(key))
                    and self._project_available(a) and self._gallery_facts(a)["relationship"] == relationship]
                if self._gallery_batch:
                    identifier, command = self._gallery_batch.pop(0)
                    self._select_asset_id(identifier)
                    self._gallery_command(command)
                return
            asset = self._get_selected_asset()
            if not asset:
                return
            facts = self._gallery_facts(asset)
            if action == "primary":
                action = self._selected_gallery_action()
            if action == "check":
                if facts["relationship"] == "linked":
                    self._controller().resolve_asset(asset, self._gallery_details())
                elif facts["relationship"] in ("identity_ambiguous", "local_file_problem") or self._gallery_state.get("storage_issue"):
                    folder_id = str(asset.get("folder_id") or "")
                    folder = self._asset_index_folders().get(folder_id, {})
                    directory = str(folder.get("path") or "").strip()
                    if folder_id and directory:
                        self.refresh_catalog(scan_folders=False)
                        self._scan_asset_folders(folder_id=folder_id, directory=directory)
                    else:
                        self._controller().refresh()
                else:
                    self._controller().refresh()
            elif action == "publish_new":
                self._confirm_gallery("confirm.publish_new", lambda: self._publish_as_new(asset))
            elif action in ("publish", "update"):
                if not asset.get("remote_only"):
                    self._open_gallery_review(asset, action)
            elif action == "resolve":
                self._controller().resolve_asset(asset, self._gallery_details())
            elif action in ("retry", "resume") and facts["jobId"]:
                self._controller().command("resume", facts["jobId"])
            elif action in ("pause_transfer", "cancel_transfer"):
                badge = self._gallery_badge(asset)
                allowed = badge["gallery_can_pause" if action == "pause_transfer" else "gallery_can_cancel"]
                if allowed:
                    command = "pause" if action == "pause_transfer" or not facts["jobId"] else "cancel"
                    self._controller().command(command, facts["jobId"] or None)
            elif action in ("pull", "pull_open"):
                self._pull_gallery_asset(asset, open_after=action == "pull_open")
            elif action in ("open", "copy"):
                self._controller().open_portal(self._gallery_scene(asset), action)
            elif action == "unlink":
                self._confirm_gallery("confirm.unlink", lambda: self._controller().service.unlink(asset["id"]))
            elif action == "remove":
                scene = self._gallery_scene(asset)
                self._confirm_gallery("confirm.remove", lambda: self._controller().service.remove(scene["id"], scene))
        except Exception as exc:
            from .gallery_messages import localize_message
            self._gallery_notice = localize_message(str(exc))
            if action == "undo" and self._gallery_undo_kind == "pull" and self._gallery_undo:
                self._set_gallery_undo(self._gallery_undo[1], kind="pull")
        finally:
            if self._handle:
                self._handle.dirty_all()
            self._request_model_update()

    def _gallery_drag_payload(self, asset):
        import uuid
        state = self._gallery_state
        scene = self._gallery_scene(asset)
        identity = state.get("identity")
        if not asset.get("remote_only") or not scene or not state.get("connected") or not state.get("owner") or not identity:
            return None
        try:
            scene_id = str(uuid.UUID(scene["id"]))
        except (ValueError, TypeError):
            return None
        return {"origin": identity[0], "owner": state["owner"], "sceneId": scene_id}

    def gallery_viewport_drop(self, payload):
        """Native hit-tested drop handoff. Treat the payload as untrusted identity data."""
        import json
        import uuid
        try:
            data = json.loads(payload)
            if not isinstance(data, dict) or set(data) != {"origin", "owner", "sceneId"}:
                return False
            state = self._gallery_state
            if (not state.get("connected") or not state.get("identity")
                    or data["origin"] != state["identity"][0] or data["owner"] != state.get("owner")):
                return False
            scene_id = str(uuid.UUID(data["sceneId"]))
            asset = self._gallery_remote_assets().get("remote:" + scene_id)
            if not asset:
                return False
            self._select_folder_id(SCOPE_PUBLISHED)
            self._select_asset_id(asset["id"])
            self._gallery_command("pull_open")
            return True
        except (ValueError, TypeError, KeyError):
            return False

    def _gallery_drop_asset(self, identifier, folder, identity):
        if identity != self._gallery_state.get("identity") or not self._gallery_state.get("connected"):
            return
        asset = self._asset_dict(identifier)
        if not asset:
            return
        self._select_asset_id(identifier)
        if asset.get("remote_only"):
            if folder == SCOPE_PUBLISHED:
                self._show_gallery_toast(tr("drop.already_published"))
                if self._handle:
                    self._handle.dirty_all()
                self._request_model_update()
                return
            target = self._asset_index_folders().get(folder)
            if not target:
                return
            self._gallery_last_folder = folder
            # The shared pull review and controller validate existence/overwrite.
            self._gallery_command("pull")
        elif folder == SCOPE_PUBLISHED:
            facts = self._gallery_facts(asset)
            if not self._project_available(asset):
                self._show_gallery_toast(tr("drop.review_first"))
            elif facts["state"] == "equal":
                self._show_gallery_toast(tr("drop.up_to_date"))
            elif facts["action"] in ("publish", "update"):
                self._gallery_command(facts["action"])
            else:
                self._show_gallery_toast(tr("drop.review_first"))
        if self._handle:
            self._handle.dirty_all()
        self._request_model_update()

    def _gallery_quota(self):
        quota, used = self._gallery_quota_values()
        if quota is None:
            return ""
        return tr("quota.used", used=f"{used / 1e9:.1f}", quota=f"{quota / 1e9:g}")

    def _gallery_quota_values(self):
        state = self._gallery_state
        quota, used = state.get("quotaBytes"), state.get("usedBytes")
        if type(quota) is not int or quota < 0 or type(used) is not int or used < 0:
            return None, 0
        return quota, used

    def _gallery_quota_warning(self):
        quota, used = self._gallery_quota_values()
        size = (self._get_selected_asset() or {}).get("file_size_bytes", 0)
        return tr("quota.warning") if quota is not None and size > max(0, quota - used) else ""

    def _gallery_update_candidates(self):
        return [a for a in self._asset_index_assets().values() if self._project_available(a)
                and self._gallery_facts(a)["freshness"] == "local"
                and self._gallery_facts(a)["action"] == "update"]

    def _show_gallery_toast(self, text, **actions):
        if self._gallery_toast_timer:
            self._gallery_toast_timer.cancel()
        toast = dict(text=text, identity=self._gallery_state.get("identity"), **actions)
        self._gallery_toast = toast
        generation = self._mount_generation
        def expire():
            if not self._panel_mounted or generation != self._mount_generation:
                return
            if self._gallery_toast is toast:
                self._gallery_toast = None
                if self._handle:
                    self._handle.dirty_all()
                self._request_model_update()
        self._gallery_toast_timer = threading.Timer(8, lambda: lf.ui.schedule_on_ui_thread(expire))
        self._gallery_toast_timer.daemon = True
        self._gallery_toast_timer.start()

    def _gallery_completions(self, previous, snapshot):
        completion = snapshot.get("completion")
        if completion and completion["id"] != self._gallery_completion_id:
            self._gallery_completion_id = completion["id"]
            if completion["kind"] == "remove":
                self._show_gallery_toast(tr("toast.removed", title=completion["title"]))
            elif completion.get("scene"):
                self._show_gallery_toast(tr("toast.published", title=completion["scene"].get("title", "")), scene=copy.deepcopy(completion["scene"]))
        pulled = snapshot.get("pulledProject")
        if pulled and pulled != previous.get("pulledProject"):
            path = Path(pulled["path"])
            asset = self._asset_dict(pulled["id"]) or {}
            self._show_gallery_toast(tr("toast.pulled", title=asset.get("name", path.stem), folder=path.parent.name), path=str(path))

    def _confirm_gallery(self, key, continuation):
        controller = self._controller()
        if controller._decision_pending:
            return
        self._gallery_notice = ""
        controller._message = ""
        controller.confirm_action(key, self._gallery_details()["title"], continuation)

    def _set_gallery_undo(self, action, *, kind="visibility"):
        self._gallery_undo_kind = kind
        if self._gallery_undo_timer:
            self._gallery_undo_timer.cancel()
        undo = (time.monotonic() + 8, action)
        self._gallery_undo = undo
        generation = self._mount_generation
        def expire():
            if not self._panel_mounted or generation != self._mount_generation:
                return
            if self._gallery_undo is undo:
                if kind == "pull" and (self._gallery_state.get("undoPull") or {}).get("operation"):
                    return
                self._gallery_undo = None
                if self._handle:
                    self._handle.dirty_all()
                    self._request_model_update()
        self._gallery_undo_timer = threading.Timer(8, lambda: lf.ui.schedule_on_ui_thread(expire))
        self._gallery_undo_timer.daemon = True
        self._gallery_undo_timer.start()

    def _gallery_details(self, asset=None):
        asset = asset if asset is not None else self._get_selected_asset() or {}
        scene = self._gallery_scene(asset) or {}
        return {"title": scene.get("title", asset.get("name", "")),
                "description": scene.get("description", ""), "visibility": scene.get("visibility", "private")}

    def _gallery_review_includes(self, *, publish_new=False):
        text = tr("review.includes", saved=self.get_selected_asset_modified())
        if self._gallery_scene(self._get_selected_asset() or {}) and not publish_new:
            text += " " + tr("review.cover_kept")
        return text

    def _gallery_published_summary(self):
        asset = self._get_selected_asset() or {}
        link = self._gallery_state.get("links", {}).get(asset.get("id"), {})
        if not link:
            return ""
        scene = self._gallery_scene(asset) or link.get("metadata", {})
        return tr("info.published_relative", format=link.get("uploadFormat", "licht").upper(),
            size=self._format_size(scene.get("contentLength", 0)),
            time=relative_time(link.get("exchangedAt") or time.time()))

    def _selected_gallery_action(self):
        if self._gallery_state.get("unsupported"):
            return ""
        asset = self._get_selected_asset()
        if not asset:
            return ""
        return self._gallery_facts(asset)["action"]

    def _begin_gallery_publish(self, asset, action):
        warning = self._gallery_quota_warning()
        if warning and not getattr(self, "_gallery_quota_ack", False):
            def proceed():
                self._gallery_quota_ack = True
                try:
                    self._begin_gallery_publish(asset, action)
                finally:
                    self._gallery_quota_ack = False
            self._confirm_gallery("quota.confirm", proceed)
            return
        controller = self._controller()
        self._gallery_batch_waiting = bool(self._gallery_batch)
        self._gallery_batch_asset = dict(asset)
        try:
            controller.publish_asset(asset, self._gallery_details(), self._gallery_upload_format,
                                     update=action == "update")
        except Exception:
            self._gallery_batch_waiting = False
            raise

    def _publish_as_new(self, asset):
        self._open_gallery_review(asset, "publish", publish_new=True)

    def _open_gallery_review(self, asset, action, *, open_after=False, publish_new=False):
        from .gallery_file_panel import open_gallery_file_panel
        controller = self._controller()
        self._gallery_upload_format = getattr(controller, "upload_format", self._gallery_upload_format)

        def done(submitted):
            self._gallery_batch_waiting = bool(submitted and self._gallery_batch)
            self._gallery_batch_asset = dict(asset) if submitted else None
            if submitted:
                self._gallery_upload_format = controller.upload_format
            if not submitted:
                self._gallery_batch = []
            self._request_model_update()

        fields = dict(self._gallery_details(asset), upload_format=self._gallery_upload_format,
                      pull_folder=self._gallery_pull_folder, pull_name=self._gallery_pull_name)
        if publish_new:
            fields["visibility"] = "private"
        open_gallery_file_panel(controller=controller, asset=asset,
            scene=self._gallery_scene(asset), action=action,
            fields=fields,
            includes=self._gallery_review_includes(publish_new=publish_new), quota=self._gallery_quota(),
            warning=self._gallery_quota_warning() if action != "pull" else "",
            publish_new=publish_new, open_after=open_after, on_done=done)

    def _pull_gallery_asset(self, asset, *, open_after=False):
        scene = self._gallery_scene(asset)
        if not scene:
            return
        if asset.get("remote_only") or not asset.get("exists", True):
            folder = self._asset_index_folders().get(self._gallery_last_folder) or self._asset_index_folders().get(self._default_folder_id(), {})
            self._gallery_pull_folder = folder.get("path", "")
            self._gallery_pull_name = self._controller().safe_filename(scene.get("title", ""))
            self._open_gallery_review(asset, "pull", open_after=open_after)
        else:
            self._controller().pull_asset(asset, scene)
