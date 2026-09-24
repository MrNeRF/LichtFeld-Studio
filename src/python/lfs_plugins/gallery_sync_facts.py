# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gallery relationship, conflict, and status facts."""
from __future__ import annotations

import copy
import math

import lichtfeld as lf
from .gallery_actions import gallery_actions
from .gallery_messages import tr, localize_message
from .gallery_transfer_ui import transfer_phase

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

_VIEW_FLOAT_TOLERANCE = 1e-6


def needs_attention(facts):
    """Whether the project card presents a health or gallery attention state."""
    return bool(facts.get("health_icon") or facts.get("attention") or facts.get("relink_required"))


def _same_view(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_view(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same_view(x, y) for x, y in zip(a, b))
    if (type(a) in (int, float) and type(b) in (int, float)
            and (type(a) is float or type(b) is float)):
        return math.isclose(a, b, rel_tol=_VIEW_FLOAT_TOLERANCE,
                            abs_tol=_VIEW_FLOAT_TOLERANCE)
    return a == b


def _same_shared_fields(a, b):
    return (a.keys() == b.keys()
            and all(_same_view(a[key], b[key]) if key == "viewerSettings" else a[key] == b[key]
                    for key in a))

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


def conflict_groups(asset, link, local, remote, *, apply_only=False, saved_stamp="", closed=False, translate=tr):
    import json
    tr = translate
    baseline = link.get("sharedFields", {})
    baseline_stamp = link.get("contentStamp", "")
    comparable_stamps = ":" in saved_stamp and ":" in baseline_stamp
    saved_content, saved_view = saved_stamp.split(":", 1) if comparable_stamps else ("", "")
    baseline_content, baseline_view = baseline_stamp.split(":", 1) if comparable_stamps else ("", "")
    comparable_stamps = comparable_stamps and all((saved_content, saved_view, baseline_content, baseline_view))
    saved_view_changed = closed and comparable_stamps and saved_view != baseline_view
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
        same = _same_view(mine, gallery) if identifier in ("view", "track") else mine == gallery
        if same:
            continue
        mine_value = tr("conflict.saved_view_changed") if saved_view_changed and identifier in ("view", "track") else text(mine)
        gallery_value = text(gallery)
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
                     for key in sorted(set(mine) | set(gallery)) if not _same_view(mine.get(key), gallery.get(key))]
            difference = tr("conflict.changed_settings", parts=", ".join(dict.fromkeys(names)))
        elif identifier == "track":
            difference = tr("conflict.track_counts", mine=len((mine or {}).get("keyframes", [])),
                            gallery=len((gallery or {}).get("keyframes", [])))
        if saved_view_changed and identifier in ("view", "track"):
            difference = values
            fields = [tr("conflict.view" if identifier == "view" else "conflict.track")]
        rows.append(dict(id=identifier, label=tr({"text": "conflict.text", "view": "conflict.view", "track": "conflict.track"}[identifier]),
            mine_value=mine_value, gallery_value=gallery_value, fields=fields,
            difference=difference, values=values,
            choice="gallery" if apply_only or ((_same_view(mine, base) if identifier in ("view", "track") else mine == base)
                and not (saved_view_changed and identifier in ("view", "track"))) else "mine",
            can_both=identifier == "track" and bool(mine and gallery)))
    content_changed = (link.get("contentRevision") != remote.get("contentRevision")
        or (not apply_only and (not link.get("commitUuid") or asset.get("commit_uuid") != link.get("commitUuid"))
            and (not comparable_stamps or saved_content != baseline_content)))
    if content_changed:
        mine, gallery = tr("conflict.local_content"), tr("conflict.gallery_content")
        rows.append(dict(id="content", label=tr("conflict.content"), mine_value=mine, gallery_value=gallery,
            fields=[tr("conflict.content")],
            difference=tr("conflict.values", mine=mine, gallery=gallery), values=tr("conflict.values", mine=mine, gallery=gallery),
            choice="mine", can_both=False))
    return rows


def asset_sync_state(project=None, link=None, scene=None, jobs=(), *, checked=False,
                     phase="idle", storage_issue=False, cached_projection=None, established=True, translate=tr):
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
    if link and link.get("liveSnapshot"):
        freshness = "live_snapshot"
    if link and link.get("commitUuid") and project.get("commit_uuid") and scene and all(link.get(key) and scene.get(key) for key in ("contentRevision", "metadataRevision")):
        # Links written by older builds still include visibility in their saved fields.
        # Older applies could omit an explicit null camera track from local fields.
        local_fields = {k: v for k, v in link.get("localFields", {}).items() if k != "visibility"}
        gallery_fields = {k: v for k, v in link.get("sharedFields", {}).items() if k != "visibility"}
        local_view = local_fields.get("viewerSettings")
        gallery_view = gallery_fields.get("viewerSettings")
        if (isinstance(local_view, dict) and isinstance(gallery_view, dict)
                and "cameraPath" not in local_view and "cameraPath" in gallery_view
                and gallery_view["cameraPath"] is None):
            local_fields = dict(local_fields, viewerSettings=dict(local_view, cameraPath=None))
        local = (project["commit_uuid"] != link["commitUuid"] or
                 "localFields" in link and not _same_shared_fields(local_fields, gallery_fields))
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
                                 apply_only=visible != "diverged", translate=translate)
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
