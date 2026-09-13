# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager's Gallery projection and inline review controls."""
from __future__ import annotations

import time
import copy
import threading
from pathlib import Path
from urllib.parse import quote

import lichtfeld as lf

from .gallery_controller import asset_sync_state, get_gallery_controller

SCOPE_PUBLISHED = "__gallery__"
SCOPE_ATTENTION = "__gallery_attention__"
SCOPE_TRANSFERS = "__gallery_transfers__"
GALLERY_SCOPES = (SCOPE_PUBLISHED, SCOPE_ATTENTION)


def tr(key, **values):
    from .localization import safe_format
    full_key = "asset_manager.gallery." + key
    return safe_format(lf.ui.tr(full_key), **values)


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
        self._gallery_expanded = False
        self._gallery_review = False
        self._gallery_more = False
        self._gallery_edit_id = None
        self._gallery_editor_scene = None
        self._gallery_title = ""
        self._gallery_description = ""
        self._gallery_visibility = "private"
        self._gallery_upload_format = "sog"
        self._gallery_pull_folder = ""
        self._gallery_pull_name = ""
        self._gallery_pull_review = False
        self._gallery_pending_publish = None
        self._gallery_batch = []
        self._gallery_batch_waiting = False
        self._gallery_notice = ""
        self._gallery_undo = None
        self._gallery_undo_kind = ""
        self._gallery_undo_backup = None
        self._gallery_last_folder = None
        self._gallery_focus_path = None
        self._gallery_undo_timer = None
        self._gallery_publish_new = False
        self._gallery_pull_open = False
        self._gallery_pulled_job = None

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
        self._gallery_state = snapshot
        if snapshot.get("relink_required"):
            self._gallery_notice = snapshot.get("message", "")
        if previous_identity != snapshot.get("identity"):
            self._gallery_edit_id = None
            self._gallery_review = self._gallery_pull_review = False
            self._gallery_pending_publish = None
            self._gallery_undo = None
            self._gallery_batch = []
            self._gallery_notice = ""
            self._gallery_pulled_job = None
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
        self._gallery_selection_changed()
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
        self._continue_gallery_publish()
        controller = self._gallery_controller
        if self._gallery_batch_waiting and controller and not controller._panel_busy() and not self._gallery_pending_publish:
            facts = self._gallery_facts(self._get_selected_asset() or {})
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
        return asset_sync_state(None if remote else asset, link, self._gallery_scene(asset),
            self._gallery_state.get("jobs", ()), checked=bool(self._gallery_state.get("checkedAt")),
            storage_issue=self._gallery_state.get("storage_issue", False), phase=phase,
            cached_projection=asset.get("gallery") if "identity" not in self._gallery_state else None)

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
        rows = [a for a in self._asset_index_assets().values()
                if a.get("id") in self._gallery_state.get("links", {})]
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
        return {"gallery_state": facts["state"], "gallery_label": label,
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
                self._gallery_expanded = True
        self._request_model_update()

    def _gallery_selection_changed(self):
        asset = self._get_selected_asset()
        identifier = asset.get("id") if asset else None
        scene = self._gallery_scene(asset) if asset else None
        if identifier == self._gallery_edit_id:
            previous = self._gallery_editor_scene
            fields = self._gallery_details()
            if scene and (previous is None or all(fields[k] == previous.get(k, "") for k in fields)):
                self._gallery_title = scene.get("title", "")
                self._gallery_description = scene.get("description", "")
                self._gallery_visibility = scene.get("visibility", "private")
                self._gallery_editor_scene = copy.deepcopy(scene)
            elif scene and all(fields[k] == scene.get(k, "") for k in fields):
                self._gallery_editor_scene = copy.deepcopy(scene)
            return
        self._gallery_edit_id = identifier
        self._gallery_review = self._gallery_pull_review = False
        self._gallery_more = False
        self._gallery_publish_new = False
        scene = self._gallery_scene(asset) if asset else None
        self._gallery_editor_scene = copy.deepcopy(scene)
        self._gallery_title = (scene or asset or {}).get("title", (asset or {}).get("name", ""))
        self._gallery_description = (scene or {}).get("description", "")
        self._gallery_visibility = (scene or {}).get("visibility", "private")
        self._gallery_expanded = bool(scene or (asset and asset["id"] in self._gallery_state.get("links", {})))

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
        for name in ("title", "description", "visibility", "upload_format", "pull_folder", "pull_name"):
            model.bind("gallery_" + name, lambda n=name: getattr(self, "_gallery_" + n),
                lambda value, n=name: self._set_gallery_field(n, str(value)))
        for name in ("review", "expanded", "more", "pull_review"):
            model.bind_func("gallery_" + name, lambda n=name: getattr(self, "_gallery_" + n))
        values = {
            "gallery_signed_in": lambda: self._gallery_state.get("signed_in", False) and not self._gallery_state.get("relink_required", False),
            "gallery_account": lambda: self._gallery_state.get("display_name") or self._gallery_state.get("email", ""),
            "gallery_checked": self._gallery_checked_label,
            "gallery_published_count": lambda: len(self._gallery_rows()),
            "gallery_attention_count": lambda: len(self._gallery_rows(True)),
            "gallery_transfer_count": lambda: sum(j.get("status") not in ("completed", "canceled") for j in self._gallery_state.get("jobs", [])),
            "gallery_overlay": lambda: self._gallery_aggregate()[0],
            "gallery_tooltip": lambda: self._gallery_aggregate()[1],
            "gallery_selected_state": lambda: self._gallery_badge(self._get_selected_asset())["gallery_label"] if self._get_selected_asset() else "",
            "gallery_reupload_reason": lambda: (self._gallery_state.get("reuploadReason") or {}).get("message", "")
                if (self._gallery_state.get("reuploadReason") or {}).get("project") == (self._get_selected_asset() or {}).get("id") else "",
            "gallery_selected_action": lambda: tr("action." + self._selected_gallery_action()) if self._selected_gallery_action() else "",
            "gallery_show_editor": lambda: self._gallery_review or bool(self._gallery_scene(self._get_selected_asset() or {})),
            "gallery_show_primary": lambda: not self._gallery_review and not self._gallery_pull_review,
            "gallery_has_action": lambda: bool(self._gallery_badge(self._get_selected_asset())["gallery_action"]) if self._get_selected_asset() else False,
            "gallery_collapsed_publish": lambda: not self._gallery_expanded and not bool(self._gallery_scene(self._get_selected_asset() or {})),
            "gallery_remote": lambda: bool((self._get_selected_asset() or {}).get("remote_only")),
            "gallery_linked": lambda: self._has_gallery_link(),
            "gallery_has_portal": lambda: bool(self._gallery_scene(self._get_selected_asset() or {})),
            "gallery_notice": lambda: self._gallery_notice or self._gallery_state.get("message", ""),
            "gallery_format_hint": lambda: tr("format." + self._gallery_upload_format + "_hint"),
            "gallery_publish_label": self._gallery_publish_label,
            "gallery_includes": lambda: tr("review.includes", saved=self.get_selected_asset_modified()),
            "gallery_published_summary": self._gallery_published_summary,
            "gallery_exchange_summary": lambda: " · ".join(filter(None, (self._gallery_published_summary(), self._gallery_checked_label()))),
            "gallery_selected_format": lambda: ((self._get_selected_asset() or {}).get("source_format") or "licht").upper(),
            "gallery_selected_visibility": lambda: tr("review." + (self._gallery_scene(self._get_selected_asset() or {}) or {}).get("visibility", "private")),
            "gallery_multi_summary": lambda: tr("multi.summary", **self._gallery_counts()),
            "gallery_publish_many": lambda: tr("multi.publish", count=self._gallery_counts()["ready"]),
            "gallery_update_many": lambda: tr("multi.update", count=self._gallery_counts()["linked"]),
            "gallery_can_publish_many": lambda: self._gallery_counts()["ready"] > 0,
            "gallery_can_update_many": lambda: self._gallery_counts()["linked"] > 0,
            "gallery_has_pulled_project": lambda: bool(self._gallery_state.get("pulledProject")),
            "gallery_pull_label": lambda: tr("action.pull_open" if self._gallery_pull_open else "action.pull"),
            "gallery_has_undo": lambda: bool(self._gallery_undo and time.monotonic() < self._gallery_undo[0]
                and not (self._gallery_state.get("undoPull") or {}).get("operation")),
            "gallery_has_recovery": lambda: bool((self._gallery_state.get("undoPull") or {}).get("backupMissing")),
        }
        for name, getter in values.items():
            model.bind_func(name, getter)
        for key in ("sidebar.title", "sidebar.sign_in", "sidebar.published", "sidebar.attention", "sidebar.transfers", "sidebar.refresh",
                    "review.title", "review.description", "review.visibility", "review.upload_as", "review.private", "review.public",
                    "format.studio", "format.sog", "format.ssog", "format.spz", "info.portal_hint", "info.folder", "info.filename",
                    "action.publish", "action.pull", "action.open", "action.copy", "action.more", "action.unlink", "action.remove", "action.undo",
                    "action.story", "action.display", "action.manage", "action.submit", "action.cancel", "action.grid", "action.list",
                    "info.format", "state.remote_only", "action.pull_open", "action.open_local", "action.open_recovery"):
            model.bind_func("g_" + key.replace(".", "_"), lambda k=key: tr(k))
        for action in ("refresh", "account", "transfers", "toggle", "primary", "publish", "pull", "open", "copy", "more",
                       "unlink", "remove", "story", "display", "manage", "undo", "publish_many", "update_many", "cancel", "pull_open", "open_local", "open_recovery"):
            model.bind_event("gallery_" + action, lambda _h, _e, args, a=action: self._gallery_command(a, args))

    def _set_gallery_field(self, name, value):
        if name == "upload_format":
            if value not in ("studio", "sog", "ssog", "spz"):
                return
            self._controller().upload_format = value
        if name == "visibility" and self._gallery_scene(self._get_selected_asset() or {}):
            self._change_gallery_visibility(value)
            return
        setattr(self, "_gallery_" + name, value)
        if self._handle:
            self._handle.dirty_all()

    def _gallery_publish_label(self):
        asset = self._get_selected_asset() or {}
        poll = getattr(lf, "project_poll_write", lambda: {})()
        if poll.get("path") == asset.get("path") and lf.project_is_dirty():
            return tr("action.save_publish")
        return tr("review.publish_public" if self._gallery_visibility == "public" else "review.publish_private")

    def _gallery_context_items(self, asset):
        facts = self._gallery_facts(asset)
        scene = self._gallery_scene(asset)
        items = []
        if facts["action"] and facts["action"] != "open" and (self._project_available(asset) or asset.get("remote_only") or facts["relationship"] == "local_missing"):
            items.append({"label": tr("action." + facts["action"]), "action": "gallery:" + facts["action"], "separator_before": True})
        if asset.get("remote_only"):
            items.append({"label": tr("action.pull_open"), "action": "gallery:pull_open"})
        if scene:
            items += [{"label": tr("action.open"), "action": "gallery:open"}, {"label": tr("action.copy"), "action": "gallery:copy"}]
        if asset.get("id") in self._gallery_state.get("links", {}):
            items.append({"label": tr("action.more"), "action": "gallery:context_more", "children": [
                {"label": tr("action.unlink"), "action": "gallery:unlink"},
                {"label": tr("action.publish_new"), "action": "gallery:publish_new"},
                *([{"label": tr("action.remove"), "action": "gallery:remove"}] if scene else [])]})
        return items

    def _gallery_command(self, action, args=()):
        try:
            if action == "account":
                lf.ui.set_panel_enabled("lfs.account", True)
                return
            if action == "transfers":
                self.on_open_gallery()
                return
            if action == "refresh":
                self._controller().refresh()
                return
            if action in ("toggle", "more"):
                field = "_gallery_expanded" if action == "toggle" else "_gallery_more"
                setattr(self, field, not getattr(self, field))
                return
            if action == "cancel":
                self._gallery_review = self._gallery_pull_review = False
                self._gallery_batch = []
                self._gallery_pending_publish = None
                return
            if action == "open_recovery":
                self._controller().command("show_recovery_folder")
                return
            if action == "open_local":
                pulled = self._gallery_state.get("pulledProject")
                if pulled:
                    from .file_menu import open_project_with_confirmation
                    open_project_with_confirmation(pulled["path"], keep_asset_manager_open=True)
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
                else:
                    self._controller().refresh()
            elif action == "publish_new":
                self._confirm_gallery("confirm.publish_new", lambda: self._publish_as_new(asset))
            elif action == "publish" and not self._gallery_review:
                self._gallery_expanded = self._gallery_review = True
            elif action in ("publish", "update"):
                if asset.get("remote_only"):
                    self._controller().edit_scene(self._gallery_editor_scene or self._gallery_scene(asset), self._gallery_details())
                else:
                    self._begin_gallery_publish(asset, action)
            elif action == "resolve":
                self._controller().resolve_asset(asset, self._gallery_details())
            elif action in ("retry", "resume") and facts["jobId"]:
                self._controller().command("resume", facts["jobId"])
            elif action in ("pull", "pull_open"):
                self._pull_gallery_asset(asset, open_after=action == "pull_open")
            elif action in ("open", "story", "display", "manage", "copy"):
                self._controller().open_portal(self._gallery_scene(asset), action)
            elif action == "context_more":
                items = self._gallery_context_items(asset)[-1].get("children", [])
                self._show_shared_context_menu(items, lambda command: self._handle_asset_context_action(command, asset["id"]))
            elif action == "unlink":
                self._confirm_gallery("confirm.unlink", lambda: self._controller().service.unlink(asset["id"]))
            elif action == "remove":
                scene = self._gallery_scene(asset)
                self._confirm_gallery("confirm.remove", lambda: self._controller().service.remove(scene["id"], scene["revision"]))
        except Exception as exc:
            from .gallery_messages import localize_message
            self._gallery_notice = localize_message(str(exc))
            if action == "undo" and self._gallery_undo_kind == "pull" and self._gallery_undo:
                self._set_gallery_undo(self._gallery_undo[1], kind="pull")
        finally:
            if self._handle:
                self._handle.dirty_all()
            self._request_model_update()

    def _confirm_gallery(self, key, continuation):
        self._controller().confirm_action(key, self._gallery_title, continuation)

    def _set_gallery_undo(self, action, *, kind="visibility"):
        self._gallery_undo_kind = kind
        if self._gallery_undo_timer:
            self._gallery_undo_timer.cancel()
        undo = (time.monotonic() + 8, action)
        self._gallery_undo = undo
        def expire():
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

    def _gallery_details(self):
        return {"title": self._gallery_title, "description": self._gallery_description, "visibility": self._gallery_visibility}

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
        asset = self._get_selected_asset()
        if not asset:
            return ""
        facts = self._gallery_facts(asset)
        scene = self._gallery_scene(asset)
        if scene and facts["state"] in ("equal", "remote_only"):
            if any(self._gallery_details()[key] != scene.get(key, "") for key in self._gallery_details()):
                return "update"
        return facts["action"]

    def _begin_gallery_publish(self, asset, action):
        controller = self._controller()
        pending = {"asset": dict(asset), "action": action, "details": self._gallery_details(),
                   "format": self._gallery_upload_format, "identity": controller.service.identity(),
                   "publish_new": self._gallery_publish_new}
        poll = getattr(lf, "project_poll_write", lambda: {})()
        if poll.get("path") and Path(poll["path"]).resolve() == Path(asset["path"]).resolve():
            controller.publish_asset(asset, pending["details"], pending["format"], update=action == "update",
                                     **({"publish_as_new": True} if pending["publish_new"] else {}))
            self._gallery_batch_waiting = bool(self._gallery_batch)
            self._gallery_review = False
            return
        from .training_confirm import confirm_discard_work_then
        def open_selected(stop_training):
            if controller.service.identity() != pending["identity"]:
                return
            self._gallery_pending_publish = pending
            lf.project_open(asset["path"], True, stop_training, True)
            when_open = getattr(controller, "when_project_open", None)
            if callable(when_open):
                when_open(asset["path"], pending["identity"], self._continue_gallery_publish)
        confirm_discard_work_then(tr("action.publish"), open_selected)

    def _continue_gallery_publish(self):
        pending = self._gallery_pending_publish
        if not pending:
            return
        poll = getattr(lf, "project_poll_write", lambda: {})()
        if not poll.get("path") or poll.get("running"):
            return
        if Path(poll["path"]).resolve() != Path(pending["asset"]["path"]).resolve():
            return
        if getattr(lf.ui, "get_import_state", lambda: {})().get("active"):
            return
        self._gallery_pending_publish = None
        if self._controller().service.identity() == pending["identity"]:
            self._controller().publish_asset(pending["asset"], pending["details"], pending["format"], update=pending["action"] == "update",
                                            **({"publish_as_new": True} if pending["publish_new"] else {}))
            self._gallery_batch_waiting = bool(self._gallery_batch)
            self._gallery_review = False

    def _publish_as_new(self, asset):
        self._gallery_publish_new = True
        self._gallery_visibility = "private"
        self._gallery_review = self._gallery_expanded = True

    def _change_gallery_visibility(self, value):
        if value not in ("private", "public") or value == self._gallery_visibility:
            return
        asset = self._get_selected_asset()
        scene = dict(self._gallery_editor_scene or self._gallery_scene(asset))
        def started():
            self._gallery_visibility = value
            if value == "private":
                identity = self._controller().service.identity()
                def undo():
                    if self._controller().service.identity() == identity:
                        current = self._gallery_scene(asset)
                        self._controller().edit_scene(current, {"visibility": "public"})
                self._set_gallery_undo(undo)
        self._controller().edit_scene(scene, {"visibility": value}, on_started=started)

    def _pull_gallery_asset(self, asset, *, open_after=False):
        scene = self._gallery_scene(asset)
        if not scene:
            return
        if asset.get("remote_only") or not asset.get("exists", True):
            if not self._gallery_pull_review:
                self._gallery_pull_open = open_after
                folder = self._asset_index_folders().get(self._gallery_last_folder) or self._asset_index_folders().get(self._default_folder_id(), {})
                self._gallery_pull_folder = folder.get("path", "")
                self._gallery_pull_name = self._controller().safe_filename(scene.get("title", ""))
                self._gallery_pull_review = self._gallery_expanded = True
                return
            self._controller().pull_asset(asset, scene, str(Path(self._gallery_pull_folder) / self._gallery_pull_name),
                open_after=self._gallery_pull_open or open_after)
            self._gallery_pull_review = False
        else:
            self._controller().pull_asset(asset, scene)
