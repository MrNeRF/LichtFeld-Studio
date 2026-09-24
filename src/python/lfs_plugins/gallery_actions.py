# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""One set of Gallery verbs for cards, rows, menus, review and transfers."""
from .gallery_messages import tr


FILE_PROBLEMS = {"MISSING", "READING", "UNVERIFIED", "UNREADABLE", "UNSUPPORTED", "REPAIR_ONLY",
                 "UNSUPPORTED_NEWER", "DIVERGED_COPIES", "IDENTITY_CONFLICT", "IDENTITY_MISMATCH",
                 "DUPLICATE", "AMBIGUOUS"}


def gallery_quota(facts):
    quota, used, reserved = (facts.get(key) for key in ("quotaBytes", "usedBytes", "reservedBytes"))
    if type(quota) is not int or quota < 0 or type(used) is not int or used < 0:
        return None, 0, None
    used += reserved if type(reserved) is int and reserved >= 0 else 0
    return quota, used, max(0, quota - used)


def gallery_eligibility(entry, facts):
    reasons = []
    if (facts.get("connection_state", "connected" if facts.get("signed_in", True) else "not_connected") != "connected"
            or facts.get("relink_required")):
        reasons.append("connect")
    if facts.get("busy") or facts.get("active"):
        reasons.append("busy")
    formats = facts.get("source_formats")
    if facts.get("unsupported") or (formats is not None and "licht" not in formats and facts.get("established")):
        reasons.append("format")
    if entry.get("has_hdr") and facts.get("hdrBackgrounds") is False:
        reasons.append("hdr")
    publication = dict(entry.get("publication", {}))
    commit = entry.get("commit_uuid")
    for job in reversed(facts.get("jobs", [])):
        if (commit and job.get("project") == entry.get("id") and job.get("commitUuid") == commit
                and job.get("kind") == "upload" and job.get("packaged")
                and job.get("status") == "completed"):
            publication["checked"] = True
            if not facts.get("upload_format") or facts["upload_format"] == job.get("uploadFormat"):
                publication["preparedBytes"] = job.get("total")
            break
    failure = facts.get("preparationFailure") or facts.get("job") or {}
    if (commit and failure.get("project") == entry.get("id") and failure.get("commitUuid") == commit):
        reason = {"gallery_project_no_splats": "no_splats",
                  "gallery_project_payload_unavailable": "external_payloads",
                  "gallery_project_not_supported": "format"}.get(failure.get("failureReason", "").split(":", 1)[0])
        if reason and reason not in reasons:
            reasons.append(reason)
    if publication.get("visibleSplats") == 0:
        reasons.append("no_splats")
    if publication.get("externalPayloads"):
        reasons.append("external_payloads")
    _, _, remaining = gallery_quota(facts)
    if remaining is not None:
        size = publication.get("preparedBytes")
        replaced = facts.get("replacedBytes", 0)
        replaced = replaced if type(replaced) is int and replaced >= 0 else 0
        if (type(size) is int and max(0, size - replaced) > remaining) or (remaining == 0 and not replaced):
            reasons.append("space")
    return {"status": "blocked" if reasons else "eligible" if publication.get("checked") else "not_checked",
            "reasons": reasons, "reason": "\n".join(tr("eligibility." + key) for key in reasons),
            "remainingBytes": remaining}


def gallery_actions(entry, facts):
    """Return ordered verb records, with the primary Gallery action first.

    Disabled records carry their visible explanation. File recovery precedes
    Gallery work; a transfer precedes the relationship it temporarily covers.
    """
    entry = entry or {}
    actions = []
    from .portal_connection_ui import connection_action

    connection_state = facts.get("connection_state", "connected")
    connection_action_id, connection_label_key = connection_action(connection_state)
    connected = (connection_action_id is None and connection_state == "connected"
                 and not facts.get("relink_required"))
    busy = bool(facts.get("busy"))
    eligibility = gallery_eligibility(entry, facts)

    def add(identifier, *, enabled=True, reason="", account=True, label=None):
        if any(item["id"] == identifier for item in actions):
            return
        if account and not connected:
            enabled, reason = False, tr("eligibility.connect")
        elif not enabled and not reason and busy:
            reason = tr("eligibility.busy")
        if label and label.startswith("portal."):
            import lichtfeld as lf
            label = lf.ui.tr(label)
        else:
            label = tr(label or "action." + identifier)
        actions.append(dict(id=identifier, enabled=bool(enabled), reason=reason,
                            label=label, primary=not actions))

    if ((not entry.get("exists", True) and facts.get("state") == "unlinked")
            or entry.get("status") in FILE_PROBLEMS
            or facts.get("relationship") in ("local_file_problem", "identity_ambiguous")):
        if entry.get("status") == "MISSING" or (not entry.get("exists", True) and entry.get("status") not in FILE_PROBLEMS):
            add("locate", account=False, label="local.locate")
        return actions
    if facts.get("storage_issue"):
        return actions
    if facts.get("state") == "live_snapshot":
        return actions
    if facts.get("cachedUnverified"):
        add("check", enabled=not busy)
        return actions
    job = facts.get("job") or {}
    activity = facts.get("activity", "idle")
    if facts.get("relationship") == "replaced" and (not job or job.get("handoffIntent")) and not facts.get("active"):
        add("replace_review", enabled=not busy)
        add("unlink_previous", enabled=not busy)
        return actions
    if job.get("localUpdate", {}).get("interrupted") or job.get("localUpdate", {}).get("state") == "failed":
        if facts.get("linked") and facts.get("sceneReady") and entry.get("exists", True):
            add("resolve", enabled=not busy)
        add("open_recovery", account=False)
        return actions
    if job.get("requiresPreparation"):
        add("retry", enabled=not busy)
        add("cancel", enabled=not busy)
        return actions
    if job.get("nativePreparation"):
        add("retry", enabled=not busy and eligibility["status"] != "blocked", reason=eligibility["reason"])
        return actions
    if activity == "applying":
        return actions
    if job.get("status") == "completed" and facts.get("undoAvailable"):
        add("undo", enabled=not busy)
        return actions
    if job and activity == "processing" and job.get("needsAttention"):
        add("keep_waiting", enabled=not busy)
        add("cancel", enabled=not busy)
        return actions
    if job.get("status") == "queued":
        if not job.get("batchQueued"):
            add("resume", enabled=not busy)
        add("cancel")
        return actions
    if facts.get("active"):
        if job.get("status") == "running":
            add("pause", label="transfer.pause")
        elif activity == "preparing":
            add("cancel")
        if job:
            add("cancel")
        return actions
    if facts.get("relationship") == "remote_deleted" and job.get("status") == "conflict":
        # A failed replacement cannot be resolved against an item that is gone.
        # Retire it explicitly before offering Publish again.
        add("cancel", enabled=not busy)
        return actions
    if facts.get("freshness") == "diverged":
        add("resolve", enabled=not busy)
    elif activity in ("error", "paused", "interrupted") and job:
        if job.get("retryable") is not False:
            add("retry" if activity == "error" else "resume", enabled=not busy)
        else:
            add("check", enabled=not busy)
        add("cancel", enabled=not busy)
        return actions
    else:
        state = facts.get("state", "unlinked")
        if state == "remote_deleted" and not entry.get("exists", True):
            add("unlink", enabled=not busy)
            return actions
        verb = {"unlinked": "publish", "equal": "open", "local": "update", "remote": "apply",
                "remote_content": "apply", "presentation": "open", "remote_only": "pull",
                "local_missing": "pull", "remote_deleted": "publish_again", "unknown": "check",
                "not_checked": "check"}.get(state)
        if (verb == "check" and state == "not_checked" and not connected
                and facts.get("relationship") == "unlinked" and not entry.get("remote_only")):
            # The row's connection action retains the intended publish. Its
            # review opens after the first account-specific Gallery check.
            verb = "publish"
        if facts.get("viewingCopy") and verb in ("update", "open", "apply"):
            add("publish_new", enabled=not busy and eligibility["status"] != "blocked", reason=eligibility["reason"])
        if verb:
            publishing = verb in ("publish", "update", "publish_again")
            blocked = [reason for reason in eligibility["reasons"] if reason != "connect"]
            label = connection_label_key if not connected else None
            add(verb, enabled=not busy and (not publishing or not blocked),
                reason="\n".join(tr("eligibility." + reason) for reason in blocked) if publishing else "",
                account=not publishing, label=label)
    if entry.get("remote_only"):
        add("pull_open", enabled=not busy)
    if facts.get("sceneReady"):
        if not entry.get("remote_only"):
            add("pull", enabled=not busy)
            add("pull_open", enabled=not busy)
        add("open")
        add("copy")
        add("remove", enabled=not busy)
    if facts.get("linked"):
        add("unlink", enabled=not busy)
    return actions
