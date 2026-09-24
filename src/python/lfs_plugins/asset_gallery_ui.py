# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gallery file actions and transfer status in the Asset Manager."""
from __future__ import annotations

import time
import copy
import threading
from .gallery_logging import failure as log_failure
from pathlib import Path

import lichtfeld as lf
from .gallery_messages import tr, localize_message

from .gallery_controller import asset_sync_state, get_gallery_controller
from .gallery_sync_facts import needs_attention
from .gallery_actions import gallery_actions, gallery_quota
from .asset_index import display_name, last_known_gallery_label, previous_scene_for, resolve_default_asset_directory
from .portal_connection_ui import connection_action, connection_state

SCOPE_PUBLISHED = "__gallery__"
SCOPE_ATTENTION = "__gallery_attention__"
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
        self._gallery_rows_generation = 0
        self._gallery_upload_format = "sog"
        self._gallery_pull_folder = ""
        self._gallery_pull_name = ""
        self._gallery_batch = []
        self._gallery_batch_waiting = False
        self._gallery_batch_asset = None
        self._gallery_connection_resume = None
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

    def _portal_account_snapshot(self):
        controller = self._controller()
        account = getattr(getattr(controller, "service", None), "account", None)
        return account.snapshot() if account is not None else None

    def _portal_connection_state(self):
        return connection_state(self._portal_account_snapshot())

    def _controller(self):
        if self._gallery_controller is None:
            self._gallery_controller = get_gallery_controller()
            self._gallery_upload_format = self._gallery_controller.upload_format
        return self._gallery_controller

    def _subscribe_gallery(self):
        if self._gallery_unsubscribe is None:
            self._gallery_unsubscribe = self._controller().subscribe(self._gallery_changed)
            if self._gallery_state.get("signed_in"):
                self._controller().refresh()

    def _gallery_changed(self, snapshot):
        previous_identity = self._gallery_state.get("identity")
        previous = self._gallery_state
        if snapshot.get("message") in ("Gallery checked.", tr("info.checked")):
            snapshot = {**snapshot, "message": ""}
        self._gallery_state = snapshot
        self._gallery_rows_generation += 1
        if previous_identity != snapshot.get("identity"):
            self._gallery_undo = None
            self._gallery_batch = []
            self._gallery_batch_waiting = False
            self._gallery_batch_asset = None
            self._gallery_notice = ""
            self._gallery_pulled_job = None
            self._gallery_toast = None
            self._gallery_completion_id = (snapshot.get("completion") or {}).get("id")
        if snapshot.get("actionError") and (snapshot.get("actionError") != previous.get("actionError")
                or snapshot.get("actionErrorId") != previous.get("actionErrorId")):
            self._gallery_notice = snapshot["actionError"]
        if snapshot.get("relink_required"):
            # Identity changes reset transient notices above. Relink is durable
            # account state and must remain visible until access is approved.
            self._gallery_notice = snapshot.get("message", "")
        self._gallery_completions(previous, snapshot)
        pulled = snapshot.get("pulledProject")
        if pulled and pulled["jobId"] != self._gallery_pulled_job and self._asset_index:
            loaded = (
                self._library_command("load")
                if getattr(self, "_library_service", None)
                else self._asset_index.load()
            )
            if loaded:
                self._gallery_pulled_job = pulled["jobId"]
                self._select_folder_id("__all__")
                self._set_filter("published")
                self._select_asset_id(pulled["id"])
                self._refresh_records(assets=True, folders=True)
                self._gallery_notice = tr("info.pulled")
        self._repair_selection()
        self._refresh_records(assets=True)
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
        previous_failures = {job["id"]: job.get("message", "") for job in previous.get("jobs", [])
                             if job.get("status") == "error" or job.get("localUpdate", {}).get("state") == "failed"}
        failures = [job for job in snapshot.get("jobs", [])
                    if job.get("status") == "error" or job.get("localUpdate", {}).get("state") == "failed"]
        preparation = snapshot.get("preparationFailure")
        if preparation and preparation != previous.get("preparationFailure"):
            failures.append(preparation)
        for failure in failures:
            if failure.get("message") and previous_failures.get(failure["id"]) != failure["message"]:
                self._gallery_notice = localize_message(failure["message"])
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
        if self._gallery_connection_resume:
            if self._portal_connection_state() != "connected":
                self._gallery_connection_resume = None
            elif self._gallery_connection_ready():
                identifier, action = self._gallery_connection_resume
                self._gallery_connection_resume = None
                self._resume_gallery_action(identifier, action)
        if self._handle:
            self._handle.dirty_all()
        self._request_model_update()

    def _has_gallery_link(self):
        asset = self._get_selected_asset() or {}
        return self._gallery_project_id(asset) in self._gallery_state.get("links", {}) or bool(self._gallery_scene(asset))

    @staticmethod
    def _gallery_project_id(asset):
        """Use inspected identity for external Recent rows without cataloguing them."""
        if asset.get("recent_only"):
            return asset.get("native_project_uuid") or asset.get("id")
        return asset.get("id")

    def _gallery_scene(self, asset):
        project_id = self._gallery_project_id(asset)
        if project_id in self._gallery_state.get("unlinkedProjects", ()):
            return None
        scene_id = asset.get("scene_id") or self._gallery_state.get("links", {}).get(project_id, {}).get("sceneId")
        scene = next((s for s in self._gallery_state.get("scenes", []) if s.get("id") == scene_id), None)
        if scene is not None:
            return scene
        previous = previous_scene_for(asset, self._gallery_state)
        acknowledged = self._gallery_state.get("replacementAcknowledgments", {}).get(asset.get("id"), {})
        if previous and (acknowledged.get("oldProject") != asset.get("previous_project_uuid")
                         or acknowledged.get("sceneId") != previous.get("sceneId")):
            return next((s for s in self._gallery_state.get("scenes", []) if s.get("id") == previous["sceneId"]),
                        previous.get("metadata"))
        origins = [s for s in self._gallery_state.get("scenes", []) if s.get("originProjectUuid") == project_id
                   and s.get("status") == "ready"]
        return origins[0] if len(origins) == 1 else None

    def _gallery_facts(self, asset):
        remote = asset.get("remote_only", False)
        project_id = self._gallery_project_id(asset)
        link = self._gallery_state.get("links", {}).get(project_id)
        phase = "idle"
        controller = self._gallery_controller
        if controller and getattr(controller, "_operation_project", None) == project_id:
            phase = controller.phase()
        jobs = list(self._gallery_state.get("jobs", ()))
        failure = self._gallery_state.get("preparationFailure")
        if failure and (not failure.get("commitUuid") or failure["commitUuid"] == asset.get("commit_uuid")):
            jobs.append(failure)
        gallery_asset = ({**asset, "project_uuid": project_id}
                         if project_id != asset.get("id") else asset)
        explicitly_unlinked = project_id in self._gallery_state.get("unlinkedProjects", ())
        facts = asset_sync_state(None if remote else gallery_asset, link, self._gallery_scene(asset),
            jobs, checked=bool(self._gallery_state.get("checkedAt")),
            storage_issue=self._gallery_state.get("storage_issue", False), phase=phase,
            cached_projection=asset.get("gallery") if "identity" not in self._gallery_state and not explicitly_unlinked else None,
            established=self._gallery_state.get("established", self._gallery_state.get("connected", "identity" not in self._gallery_state)))
        for key in ("signed_in", "busy", "relink_required", "unsupported", "source_formats", "quotaBytes", "usedBytes", "reservedBytes", "hdrBackgrounds"):
            if key in self._gallery_state:
                facts[key] = self._gallery_state[key]
        facts["attention"] = needs_attention(facts)
        account_snapshot = self._portal_account_snapshot()
        facts["connection_state"] = connection_state(account_snapshot)
        previous = None if explicitly_unlinked else previous_scene_for(asset, self._gallery_state)
        acknowledged = self._gallery_state.get("replacementAcknowledgments", {}).get(asset.get("id"), {})
        if previous and not link and (acknowledged.get("oldProject") != asset.get("previous_project_uuid")
                or acknowledged.get("sceneId") != previous.get("sceneId")):
            facts.update(relationship="replaced", attention=True)
            if not facts["active"] and facts["activity"] not in ("interrupted", "paused", "error"):
                facts["state"] = "replaced"
        elif not link and self._gallery_scene(asset) and not remote and not previous:
            facts.update(state="unknown", originMatch=True)
        facts["replacedBytes"] = (self._gallery_scene(asset) or {}).get("contentLength", 0) if link else 0
        facts["jobs"] = jobs
        # Recent-only rows have a path and display name, but no catalog identity
        # that the Gallery controller can safely act on.
        facts["actions"] = [] if asset.get("recent_only") else gallery_actions(asset, facts)
        facts["action"] = facts["actions"][0]["id"] if facts["actions"] else ""
        return facts

    def _asset_with_poster(self, asset):
        scene = self._gallery_scene(asset) or {}
        poster = self._gallery_state.get("posters", {}).get(scene.get("id"), "")
        return {**asset, "poster_path": poster, "prefer_poster": self._selected_folder_id in GALLERY_SCOPES}

    def _gallery_remote_assets(self):
        linked = {link["sceneId"] for identifier, link in self._gallery_state.get("links", {}).items()
                  if identifier in self._asset_index_assets()}
        linked.update(scene["id"] for asset in self._asset_index_assets().values()
                      if (scene := self._gallery_scene(asset)) and scene.get("originProjectUuid") == asset["id"])
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
                if a.get("id") in self._gallery_state.get("links", {}) or a.get("id") in transferring
                or self._gallery_scene(a) is not None]
        rows += list(self._gallery_remote_assets().values())
        if attention:
            # Failed first publishes are also actionable, even before a link exists.
            rows = list(self._all_display_assets().values())
            rows = [a for a in rows if self._gallery_facts(a)["attention"]]
        return rows

    def _gallery_badge(self, asset):
        facts = self._gallery_facts(asset)
        from .gallery_messages import localize_message
        facts["reason"] = localize_message(facts["reason"]) if facts["reason"] else ""
        state_key = "state." + (facts["activity"] if facts["active"] else facts["state"])
        label = tr(state_key, percent=facts["progress"])
        relink_pending = bool(self._gallery_state.get("relink_required"))
        if relink_pending and not facts["active"]:
            # Relink is one account-level condition, not a fault on every row.
            label = ""
        known_label = last_known_gallery_label(asset, self._gallery_state)
        if (not relink_pending and self._gallery_project_id(asset) in self._gallery_state.get("links", {}) and known_label is None
                and not facts["active"] and facts["activity"] not in ("paused", "interrupted", "error")):
            label = tr("state.not_checked")
        if facts.get("viewingCopy"):
            label = tr("state.viewing_copy") + " · " + label
        if facts["reason"] and facts["state"] == "error":
            label = tr("state.with_reason", state=label, reason=facts["reason"])
        elif not facts["reason"] and facts.get("change_summary") and facts["state"] in ("remote", "diverged"):
            label = tr("state.with_reason", state=label, reason=facts["change_summary"])
        if facts["health_icon"]:
            label = getattr(self, "_project_status_label", lambda _asset: label)(asset)
        job = next((j for j in self._gallery_state.get("jobs", ()) if j["id"] == facts["jobId"]), {})
        actions = facts["actions"]
        verbs = {action["id"] for action in actions if action["enabled"]}
        primary = actions[0] if actions else {}
        can_cancel, can_pause = "cancel" in verbs, "pause" in verbs
        scene = self._gallery_scene(asset) or {}
        stored = bool(scene.get("status") == "ready" and self._gallery_state.get("connected")
                      and self._gallery_state.get("checkedAt") and not self._gallery_state.get("offline"))
        indeterminate = facts["active"] and (facts["activity"] in ("preparing", "processing", "applying")
                                            or (facts["activity"] in ("uploading", "downloading") and not job.get("total")))
        detail = job.get("transferDetail", "")
        byte_label = tr("bytes", prefix="gallery.transfer.", done=self._format_size(job.get("completed", 0)),
                        total=self._format_size(job["total"])) if facts["active"] and job.get("total") else ""
        gallery_action = primary.get("id", "")
        action_label = primary.get("label", "")
        change_reason = facts["reason"] or facts.get("change_detail") or facts.get("change_summary") or ""
        return {"gallery_state": facts["state"], "gallery_label": label,
                "gallery_reason": change_reason, "gallery_has_reason": bool(change_reason),
                "gallery_detail": detail, "gallery_bytes": byte_label,
                "gallery_has_bytes": bool(byte_label),
                "gallery_stored": stored,
                "gallery_stored_label": tr("state.stored") + " · " + self._gallery_checked_label() if stored else "",
                "gallery_can_pause": can_pause, "gallery_can_cancel": can_cancel,
                "gallery_action_persistent": can_cancel or gallery_action in ("retry", "resume"),
                "gallery_has_controls": can_cancel or bool(gallery_action),
                "gallery_tooltip": "\n".join(filter(None, (label, facts.get("change_detail") or facts["reason"], byte_label, detail, primary.get("reason")))),
                "health_badge": bool(facts["health_icon"]),
                "health_tone": "asset-health-" + facts["health_tone"],
                "gallery_ring": facts["active"],
                "gallery_progress_value": (0.25 if indeterminate else facts["progress"] / 100.0),
                "gallery_progress_width": f"{35 if indeterminate else facts['progress']}%",
                "gallery_indeterminate": indeterminate,
                "gallery_icon": "../icon/gallery-" + facts["icon"] + ".png",
                "gallery_tone": "gallery-tone-" + facts["tone"],
                "gallery_has_badge": not relink_pending and not facts["health_icon"] and not facts["active"],
                "gallery_has_action": bool(gallery_action), "gallery_action": gallery_action, "gallery_action_label": action_label,
                "gallery_action_enabled": primary.get("enabled", False), "gallery_action_reason": primary.get("reason", ""),
                "gallery_progress": facts["progress"], "gallery_active": facts["active"],
                "remote_only": bool(asset.get("remote_only"))}

    def select_gallery_scope(self):
        self.focus_gallery()

    def focus_gallery(self, path=None):
        self._gallery_focus_path = path
        self._select_folder_id("__all__")
        self._set_filter("published")
        if path:
            target = Path(path).resolve()
            asset = next((a for a in self._asset_index_assets().values() if Path(a["path"]).resolve() == target), None)
            if asset:
                self._gallery_focus_path = None
                if asset["id"] not in self._gallery_state.get("links", {}):
                    self._set_filter("all")
                self._select_asset_id(asset["id"])
        self._request_model_update()

    def _gallery_verb_enabled(self, asset, verb):
        if asset.get("recent_only"):
            return False
        return any(action["id"] == verb and action["enabled"] for action in self._gallery_facts(asset)["actions"])

    def _gallery_counts(self):
        visible = {
            str(a.get("id") or a.get("project_uuid") or "")
            for a in self._filtered_assets()
        }
        selected = [a for key in self._selected_asset_ids if key in visible
                    and (a := self._asset_dict(key)) and not a.get("remote_only")]
        return {"count": len(selected),
                "ready": sum(self._gallery_verb_enabled(a, "publish") for a in selected),
                "linked": sum(self._gallery_verb_enabled(a, "update") for a in selected),
                "missing": sum(not self._project_available(a) for a in selected)}

    def _gallery_checked_label(self):
        if self._portal_connection_state() != "connected" or self._gallery_state.get("relink_required"):
            return tr("sidebar.not_checked")
        if self._gallery_state.get("offline"):
            return tr("sidebar.offline")
        checked = self._gallery_state.get("checkedAt", 0)
        return tr("sidebar.checked_relative", time=relative_time(checked)) if checked else tr("sidebar.not_checked")

    def _gallery_check_label(self):
        state = self._portal_connection_state()
        if state != "connected":
            _action, label_key = connection_action(state)
            return lf.ui.tr(label_key)
        return lf.ui.tr("projects.action.check_gallery")

    def _resume_gallery_action(self, identifier, action):
        if self._portal_connection_state() != "connected":
            self._gallery_connection_resume = None
            return
        if not self._gallery_connection_ready():
            self._gallery_connection_resume = (identifier, action)
            self._controller()._schedule_poll()
            return
        asset = self._asset_dict(identifier)
        if asset:
            self._select_asset_id(identifier)
            self._gallery_command(action)

    def _gallery_connection_ready(self):
        controller = self._controller()
        return (self._gallery_state.get("signed_in") and self._gallery_state.get("checkedAt")
                and not controller._panel_busy()
                and not getattr(controller, "_refresh_pending", False)
                and not getattr(controller, "_refresh_requested", False))

    def _gallery_notice_text(self):
        from .gallery_messages import localize_message
        if self._gallery_notice:
            return localize_message(self._gallery_notice)
        if self._portal_connection_state() != "connected" or self._gallery_state.get("relink_required"):
            return ""
        return localize_message(self._gallery_state.get("message", ""))

    def _selected_gallery_badge_value(self, name):
        asset = self._get_selected_asset()
        if not asset:
            return False if name in ("gallery_has_badge", "gallery_ring", "health_badge", "gallery_has_reason") else 0.0 if name == "gallery_progress_value" else ""
        return self._gallery_badge(asset).get(name, "")

    def _bind_gallery_model(self, model):
        for name in ("gallery_icon", "gallery_tone", "gallery_tooltip", "gallery_reason", "gallery_has_reason", "gallery_has_badge", "gallery_ring", "gallery_progress_value", "health_badge", "health_tone"):
            model.bind_func("selected_" + name, lambda name=name: self._selected_gallery_badge_value(name))
        values = {
            "gallery_supported": lambda: not self._gallery_state.get("unsupported", False),
            "gallery_signed_in": lambda: self._portal_connection_state() == "connected" and not self._gallery_state.get("relink_required", False),
            "gallery_checked": self._gallery_checked_label,
            "gallery_needs_connection": lambda: connection_action(self._portal_connection_state())[0] is not None,
            "gallery_connection_action_label": lambda: lf.ui.tr(
                connection_action(self._portal_connection_state())[1]),
            "gallery_quota": self._gallery_quota,
            "gallery_has_toast": lambda: bool(self._gallery_toast),
            "gallery_toast": lambda: (self._gallery_toast or {}).get("text", ""),
            "gallery_toast_portal": lambda: bool((self._gallery_toast or {}).get("scene")),
            "gallery_toast_open": lambda: bool((self._gallery_toast or {}).get("path")),
            "gallery_update_all_visible": lambda: bool(self._gallery_update_candidates()),
            "gallery_update_all_label": lambda: tr("action.update_all", count=len(self._gallery_update_candidates())),
            "gallery_update_all_enabled": lambda: bool(self._gallery_update_candidates()) and not self._gallery_state.get("busy") and self._gallery_state.get("phase", "idle") == "idle",
            "gallery_empty": lambda: not self._backend_load_active and self._selected_folder_id == SCOPE_PUBLISHED and self._gallery_state.get("connected", False) and not self._gallery_state.get("scenes"),
            "gallery_local_empty": lambda: not self._backend_load_active and self._selected_folder_id not in GALLERY_SCOPES and not (self._all_display_assets() if self._selected_folder_id == "__all__" else self._asset_index_assets()),
            "gallery_empty_pull": lambda: bool(self._gallery_state.get("scenes")) and not self._asset_index_assets(),
            "gallery_published_count": lambda: len(self._gallery_rows()),
            "gallery_attention_count": lambda: len(self._gallery_rows(True)),
            "gallery_selected_reason": lambda: ((self._gallery_badge(self._get_selected_asset()).get("gallery_action_reason") or self._gallery_badge(self._get_selected_asset()).get("gallery_reason")) if self._get_selected_asset() else ""),
            "gallery_has_selected_reason": lambda: bool(self._get_selected_asset() and (
                self._gallery_badge(self._get_selected_asset()).get("gallery_action_reason")
                or self._gallery_badge(self._get_selected_asset()).get("gallery_reason"))),
            "gallery_selected_state": lambda: self._gallery_badge(self._get_selected_asset())["gallery_label"] if self._get_selected_asset() else "",
            "gallery_can_copy": lambda: self._gallery_verb_enabled(self._get_selected_asset() or {}, "copy"),
            "gallery_remote": lambda: bool((self._get_selected_asset() or {}).get("remote_only")),
            "gallery_linked": lambda: self._has_gallery_link(),
            "gallery_notice": self._gallery_notice_text,
            "gallery_has_notice": lambda: bool(self._gallery_notice_text()),
            "gallery_needs_recovery": lambda: self._gallery_state.get("storage_issue", False),
            "gallery_exchange_summary": lambda: " · ".join(filter(None, (self._gallery_published_summary(), self._gallery_checked_label()))),
            "gallery_selected_format": lambda: ((self._get_selected_asset() or {}).get("source_format") or "licht").upper(),
            "gallery_multi_summary": lambda: tr("multi.summary", **self._gallery_counts()),
            "gallery_publish_many": lambda: tr("multi.publish", count=self._gallery_counts()["ready"]),
            "gallery_update_many": lambda: tr("multi.update", count=self._gallery_counts()["linked"]),
            "gallery_can_publish_many": lambda: not self._gallery_state.get("unsupported") and self._gallery_counts()["ready"] > 0,
            "gallery_can_update_many": lambda: not self._gallery_state.get("unsupported") and self._gallery_counts()["linked"] > 0,
            "gallery_has_undo": lambda: bool(self._gallery_undo
                and not (self._gallery_state.get("undoPull") or {}).get("operation")),
            "gallery_has_recovery": lambda: bool((self._gallery_state.get("undoPull") or {}).get("backupMissing")),
        }
        for name, getter in values.items():
            model.bind_func(name, getter)
        for key in ("sidebar.title", "sidebar.published", "sidebar.attention",
                    "action.open", "action.copy",
                    "info.format", "state.remote_only", "action.open_local", "action.open_recovery"):
            model.bind_func("g_" + key.replace(".", "_"), lambda k=key: tr(k))
        for action in ("toast_open", "toast_portal", "toast_copy", "copy", "update_all", "refresh", "undo",
                       "publish_many", "update_many", "open_recovery"):
            model.bind_event("gallery_" + action, lambda _h, _e, args, a=action: self._gallery_command(a, args))
        model.bind_event("transfer_open_recovery", lambda _h, _e, args: self._transfer_command("open_recovery", args))

    def _transfer_command(self, action, args=()):
        identifier = args[0] if args else None
        if not identifier:
            return
        try:
            if action == "open_recovery":
                self._controller().command("show_recovery_folder")
        except Exception as exc:
            log_failure(action, exc, transfer=identifier)
            from .gallery_messages import localize_message
            self._gallery_notice = localize_message(str(exc))
            self._request_model_update()

    def _restart_failed_upload(self, asset, job):
        controller = self._controller()
        identity = controller.service.identity()
        asset_id, job_id = asset["id"], job["id"]

        def review():
            current = next((row for row in controller.service.snapshot()["jobs"] if row["id"] == job_id), {})
            if controller.service.identity() != identity or current.get("status") != "canceled":
                return
            if not self._select_asset_id(asset_id):
                return
            current_asset = self._asset_dict(asset_id)
            if job.get("handoff"):
                self._open_replacement_review(current_asset)
            else:
                self._open_gallery_review(current_asset, "update" if job.get("metadata", {}).get("replaceSceneId") else "publish")

        controller.service.discard(job_id)
        controller._after_service = review
        controller._schedule_poll()

    def _gallery_context_items(self, asset):
        return [{"label": action["label"], "action": "gallery:" + action["id"],
                 "enabled": action["enabled"], "tooltip": action["reason"],
                 "separator_before": index == 0 or action["id"] == "remove"}
                for index, action in enumerate(self._gallery_facts(asset)["actions"])]

    def _gallery_command(self, action, args=()):
        global_action = action.startswith("toast_") or action in {
            "update_all", "refresh", "open_recovery", "undo", "publish_many", "update_many",
        }
        if not global_action:
            asset = self._get_selected_asset()
            if asset and asset.get("recent_only"):
                return
        try:
            self._controller()._failure_notice = ""
            if not action.startswith("toast_"):
                self._dismiss_gallery_toast()
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
                account = getattr(getattr(self._controller(), "service", None), "account", None)
                if account is None:
                    self._controller().refresh(force=True)
                    return
                if self._portal_connection_state() != "connected":
                    account.run_after_connection(lambda: self._controller().refresh(force=True))
                else:
                    self._controller().refresh(force=True)
                return
            if action == "open_recovery":
                self._controller().command("show_recovery_folder")
                return
            if action == "undo" and self._gallery_undo:
                self._gallery_undo[1]()
                if self._gallery_undo_kind != "pull":
                    self._gallery_undo = None
                return
            if action in ("publish_many", "update_many"):
                relationship = "unlinked" if action == "publish_many" else "linked"
                self._gallery_batch = [(a["id"], "publish" if action == "publish_many" else "update")
                    for key in sorted(self._selected_asset_ids) if (a := self._asset_dict(key))
                    and self._gallery_verb_enabled(a, "publish" if action == "publish_many" else "update")]
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
            aliases = {"pause_transfer": "pause", "cancel_transfer": "cancel"}
            action = aliases.get(action, action)
            account = getattr(getattr(self._controller(), "service", None), "account", None)
            connection = connection_state(self._portal_account_snapshot())
            if account and action in ("check", "publish", "update", "publish_new", "publish_again") \
                    and connection != "connected":
                identifier = asset["id"]
                if account.run_after_connection(lambda: self._resume_gallery_action(identifier, action)):
                    self._gallery_notice = lf.ui.tr("projects.portal.connection_pending")
                return
            offered = next((item for item in facts["actions"] if item["id"] == action), None)
            self._gallery_notice = ""
            if offered and not offered["enabled"]:
                self._gallery_notice = offered["reason"]
                return
            if not offered and action not in ("check",):
                self._gallery_notice = tr("error.refresh")
                return
            if action == "check":
                if facts.get("originMatch"):
                    self._controller().service.find_publications(asset["id"])
                    self._controller()._schedule_poll()
                elif facts["relationship"] == "linked":
                    self._controller().refresh()
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
            elif action in ("publish_new", "publish_again"):
                self._publish_as_new(asset)
            elif action in ("publish", "update"):
                if not asset.get("remote_only"):
                    self._open_gallery_review(asset, action)
            elif action == "resolve":
                self._controller().resolve_asset(asset, self._gallery_details())
            elif action == "retry" and facts.get("job", {}).get("requiresPreparation"):
                self._restart_failed_upload(asset, facts["job"])
            elif action == "retry" and facts.get("job", {}).get("nativePreparation"):
                self._open_gallery_review(asset, "update" if facts.get("linked") else "publish")
            elif action in ("retry", "resume", "keep_waiting") and facts["jobId"]:
                self._controller().command("keep_waiting" if action == "keep_waiting" else "resume", facts["jobId"])
            elif action in ("pause", "cancel"):
                badge = self._gallery_badge(asset)
                allowed = badge["gallery_can_pause" if action == "pause" else "gallery_can_cancel"]
                if allowed:
                    command = "pause" if action == "pause" or not facts["jobId"] else "cancel"
                    self._controller().command(command, facts["jobId"] or None)
            elif action in ("pull", "pull_open"):
                self._pull_gallery_asset(asset, open_after=action == "pull_open")
            elif action == "apply":
                self._controller().resolve_asset(asset, self._gallery_details(), apply_only=True)
            elif action == "replace_review":
                self._open_replacement_review(asset)
            elif action == "unlink_previous":
                self._confirm_gallery("confirm.unlink", lambda: self._controller().service.unlink(asset["previous_project_uuid"]))
            elif action == "locate":
                self.on_locate_file()
            elif action in ("open", "copy"):
                self._controller().open_portal(self._gallery_scene(asset), action)
                if action == "open" and facts.get("presentationChanged") and not self._controller().service.busy:
                    self._controller().service.acknowledge_presentation(asset["id"], self._gallery_scene(asset))
                    self._controller()._schedule_poll()
            elif action == "unlink":
                self._confirm_gallery("confirm.unlink", lambda: self._controller().service.unlink(asset["id"]))
            elif action == "remove":
                scene = self._gallery_scene(asset)
                self._confirm_gallery("confirm.remove", lambda: self._controller().service.remove(scene["id"], scene))
        except Exception as exc:
            log_failure(action, exc, path=(self._get_selected_asset() or {}).get("path", ""))
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
            self._select_folder_id("__all__")
            self._set_filter("gallery")
            self._select_asset_id(asset["id"])
            self._gallery_command("pull_open")
            return True
        except (ValueError, TypeError, KeyError):
            return False

    def _gallery_drop_asset(self, identifier, folder, identity):
        if identity != self._gallery_state.get("identity") or not self._gallery_state.get("connected"):
            return
        asset = self._asset_dict(identifier)
        if not asset or not asset.get("remote_only"):
            return
        self._select_asset_id(identifier)
        target = self._asset_index_folders().get(folder)
        if not target:
            return
        self._gallery_last_folder = folder
        # The shared pull review and controller validate existence/overwrite.
        self._gallery_command("pull")
        if self._handle:
            self._handle.dirty_all()
        self._request_model_update()

    def _gallery_quota(self):
        quota, used, _ = gallery_quota(self._gallery_state)
        if quota is None:
            return ""
        return tr("quota.used", used=f"{used / 1e9:.1f}", quota=f"{quota / 1e9:g}")

    def _gallery_quota_warning(self):
        quota, used, _ = gallery_quota(self._gallery_state)
        size = (self._get_selected_asset() or {}).get("file_size_bytes", 0)
        return tr("quota.warning") if quota is not None and size > max(0, quota - used) else ""

    def _gallery_update_candidates(self):
        return [a for a in self._filtered_assets() if self._gallery_verb_enabled(a, "update")]

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
        self._gallery_toast_timer = threading.Timer(6, lambda: lf.ui.schedule_on_ui_thread(expire))
        self._gallery_toast_timer.daemon = True
        self._gallery_toast_timer.start()

    def _dismiss_gallery_toast(self):
        if self._gallery_toast_timer:
            self._gallery_toast_timer.cancel()
            self._gallery_toast_timer = None
        if self._gallery_toast is not None:
            self._gallery_toast = None
            if self._handle:
                self._handle.dirty_all()
            self._request_model_update()

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
            self._show_gallery_toast(tr("toast.pulled", title=display_name(asset), folder=path.parent.name), path=str(path))

    def _confirm_gallery(self, key, continuation):
        controller = self._controller()
        if controller._decision_pending:
            return
        self._gallery_notice = ""
        controller._message = ""
        controller.confirm_action(key, self._gallery_details()["title"], continuation)

    def _set_gallery_undo(self, action, *, kind):
        self._gallery_undo_kind = kind
        if self._gallery_undo_timer:
            self._gallery_undo_timer.cancel()
            self._gallery_undo_timer = None
        # Keep the completed operation in history until the user invokes Undo
        # or a newer operation replaces it.
        self._gallery_undo = (float("inf"), action)

    def _dismiss_gallery_undo(self):
        if self._gallery_undo_timer:
            self._gallery_undo_timer.cancel()
            self._gallery_undo_timer = None
        self._gallery_undo = None

    def _gallery_details(self, asset=None):
        asset = asset if asset is not None else self._get_selected_asset() or {}
        scene = self._gallery_scene(asset) or {}
        link = self._gallery_state.get("links", {}).get(asset.get("id"), {})
        draft = asset.get("gallery_details_draft")
        fields = link.get("localFields") or link.get("sharedFields") or scene or (draft if isinstance(draft, dict) else {})
        return {"title": fields.get("title", display_name(asset)),
                "description": fields.get("description", "")}

    def _open_replacement_review(self, asset):
        from .gallery_file_panel import open_gallery_file_panel
        controller, scene = self._controller(), self._gallery_scene(asset)
        identity = controller.service.identity()
        def chosen(action):
            if controller.service.identity() != identity:
                raise ValueError(tr("error.account_changed"))
            if action == "replace":
                controller.replace_published_asset(asset, scene, self._gallery_upload_format)
            elif action == "keep":
                controller.service.acknowledge_replacement(asset["id"], asset["previous_project_uuid"], scene["id"])
            elif action == "locate":
                selected = str(lf.ui.open_project_file_dialog(""))
                if selected:
                    info = lf.io.inspect_project(selected)
                    if str(info.project_uuid) != asset["previous_project_uuid"]:
                        raise ValueError(tr("error.project_changed"))
                    self._library_command("register_licht_asset", selected)
                    controller.service.acknowledge_replacement(asset["id"], asset["previous_project_uuid"], scene["id"])
            controller._schedule_poll()
        open_gallery_file_panel(controller=controller, asset=asset, scene=scene, action="replace_review",
                                fields={}, mode="replacement", on_submit=chosen)

    def _gallery_review_includes(self, *, publish_new=False):
        text = tr("review.includes", saved=self.get_selected_asset_modified())
        if self._gallery_scene(self._get_selected_asset() or {}) and not publish_new:
            text += " " + tr("review.cover_on_update")
        return text

    def _gallery_published_summary(self):
        asset = self._get_selected_asset() or {}
        link = self._gallery_state.get("links", {}).get(self._gallery_project_id(asset), {})
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
        if not asset or asset.get("recent_only"):
            return ""
        return self._gallery_badge(asset)["gallery_action"]

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
        except Exception as exc:
            log_failure("publish", exc, path=asset["path"])
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
        pull_folders = []
        if action == "pull":
            default_path = str(resolve_default_asset_directory())
            pull_folders.append({"name": tr("review.default_folder", name=Path(default_path).name or default_path), "path": default_path})
            for folder in self._asset_index_folders().values():
                path = str(folder.get("path") or "")
                if path and not any(Path(row["path"]) == Path(path) for row in pull_folders):
                    pull_folders.append({"name": str(folder.get("name") or Path(path).name), "path": path})
        asset = dict(asset)
        from .project_inspector import value
        details = getattr(self, "_inspection_by_asset", {}).get(asset["id"], {}).get("details")
        if details and str(value(value(details, "card"), "commit_uuid", "")) == asset.get("commit_uuid"):
            checkpoints = [item for item in value(details, "retained_checkpoints", [])
                           if value(item, "binds_scene_graph", False)]
            if checkpoints:
                latest = max(checkpoints, key=lambda item: value(item, "source_generation", 0))
                asset["publication"] = dict(asset.get("publication", {}), estimatedPoints=value(latest, "gaussians", 0),
                                            shDegree=value(latest, "sh_degree", 3))
        open_gallery_file_panel(controller=controller, asset=dict(asset, name=display_name(asset)),
            scene=self._gallery_scene(asset), action=action,
            fields=fields,
            includes=self._gallery_review_includes(publish_new=publish_new), quota=self._gallery_quota(),
            warning=self._gallery_quota_warning() if action != "pull" else "",
            publish_new=publish_new, open_after=open_after, on_done=done,
            pull_folders=pull_folders)

    def _pull_gallery_asset(self, asset, *, open_after=False):
        scene = self._gallery_scene(asset)
        if not scene:
            return
        self._gallery_pull_folder = str(resolve_default_asset_directory())
        self._gallery_pull_name = self._controller().safe_filename(scene.get("title", ""))
        self._open_gallery_review(dict(asset, remote_only=True), "pull", open_after=open_after)
