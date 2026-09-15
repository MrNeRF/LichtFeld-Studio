# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure Python models for the Projects Inspector and closed-file actions.

The panel owns the worker and native calls.  This module deliberately contains
only cache keys, presentation shaping, and action decisions so those rules can
be tested without starting LichtFeld Studio.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional


def value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def inspection_cache_key(entry: Any) -> tuple[Any, ...]:
    """Return the stat and commit identity used by both inspection tiers."""
    stat = value(entry, "stat_identity", {}) or {}
    if not isinstance(stat, dict):
        stat = {}
    size = stat.get("size", value(entry, "file_size_bytes", value(entry, "size", 0)))
    mtime = stat.get("mtime_ns", value(entry, "path_mtime_ns", value(entry, "mtime_ns", 0)))
    return (
        int(size or 0),
        int(mtime or 0),
        int(stat.get("st_dev", 0) or 0),
        int(stat.get("st_ino", 0) or 0),
        str(value(entry, "commit_uuid", "") or ""),
    )


@dataclass
class InspectionCache:
    key: tuple[Any, ...]
    card: Any = None
    details: Any = None
    error: str = ""


class InspectionFactsPipeline:
    """Cancelable off-thread card/details inspection with identity caching.

    ``refresh`` is safe to call from the UI thread.  ``on_result`` is invoked
    from the worker and should therefore only enqueue a UI callback.  A caller
    can pass a scheduler to make that rule explicit in tests and in the panel.
    """

    def __init__(
        self,
        inspect_card: Callable[[str], Any],
        inspect_details: Callable[[str], Any],
        on_result: Callable[[str, str, Any, Optional[Exception]], None],
        *,
        scheduler: Optional[Callable[[Callable[[], None]], None]] = None,
        max_background_details: int = 1,
    ) -> None:
        import threading

        self._inspect_card = inspect_card
        self._inspect_details = inspect_details
        self._on_result = on_result
        self._scheduler = scheduler
        self._max_background_details = max(0, int(max_background_details))
        self._cache: dict[str, InspectionCache] = {}
        self._details_last_started: dict[str, float] = {}
        self._cancel: Optional[threading.Event] = None
        self._generation = 0
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    @property
    def cache(self) -> dict[str, InspectionCache]:
        return self._cache

    def cached(self, asset_id: str) -> Optional[InspectionCache]:
        return self._cache.get(str(asset_id))

    def invalidate(self, asset_id: Optional[str] = None) -> None:
        if asset_id is None:
            self._cache.clear()
            return
        self._cache.pop(str(asset_id), None)

    def cancel(self) -> None:
        with self._lock:
            if self._cancel is not None:
                self._cancel.set()
            self._generation += 1

    def refresh(self, entries: Iterable[Any], selected_id: str = "") -> None:
        import threading
        import time

        self.cancel()
        with self._lock:
            generation = self._generation
            cancel = threading.Event()
            self._cancel = cancel
        rows = []
        for entry in entries:
            asset_id = str(value(entry, "id", value(entry, "project_uuid", "")) or "")
            path = str(value(entry, "path", "") or "")
            if asset_id and path:
                rows.append((asset_id, path, inspection_cache_key(entry)))
        selected_id = str(selected_id or "")

        def worker() -> None:
            background_done = 0
            for asset_id, path, key in rows:
                if cancel.is_set():
                    return
                cached = self._cache.get(asset_id)
                if cached is None or cached.key != key or cached.card is None:
                    try:
                        card = self._inspect_card(path)
                    except Exception as exc:  # a damaged card is still a result
                        with self._lock:
                            if cancel.is_set():
                                return
                            self._cache[asset_id] = InspectionCache(key=key, error=str(exc))
                            self._deliver(asset_id, "card", None, exc, cancel)
                    else:
                        with self._lock:
                            if cancel.is_set():
                                return
                            current = self._cache.get(asset_id)
                            self._cache[asset_id] = InspectionCache(
                                key=key, card=card, details=current.details if current and current.key == key else None
                            )
                            self._deliver(asset_id, "card", card, None, cancel)
                if cancel.is_set():
                    return
                is_selected = asset_id == selected_id
                last = self._details_last_started.get(asset_id, 0.0)
                due = is_selected or time.monotonic() - last >= 0.5
                if not due or (not is_selected and background_done >= self._max_background_details):
                    continue
                cached = self._cache.get(asset_id)
                if cached is not None and cached.key == key and cached.details is not None:
                    continue
                self._details_last_started[asset_id] = time.monotonic()
                try:
                    details = self._inspect_details(path)
                except Exception as exc:
                    with self._lock:
                        if cancel.is_set():
                            return
                        current = self._cache.get(asset_id)
                        if current is not None and current.key == key:
                            current.error = str(exc)
                        self._deliver(asset_id, "details", None, exc, cancel)
                else:
                    # Cancellation must not leave a cached result whose UI
                    # notification was discarded. The next refresh would see
                    # the cache and skip the result forever (Reading...).
                    with self._lock:
                        if cancel.is_set():
                            return
                        current = self._cache.get(asset_id)
                        if current is None or current.key != key:
                            current = InspectionCache(key=key)
                            self._cache[asset_id] = current
                        current.details = details
                        self._deliver(asset_id, "details", details, None, cancel)
                    if not is_selected:
                        background_done += 1

        thread = threading.Thread(target=worker, daemon=True, name="ProjectsInspection")
        with self._lock:
            self._thread = thread
        thread.start()

    def _deliver(self, asset_id: str, kind: str, result: Any, error: Optional[Exception], cancel: Any) -> None:
        if cancel.is_set():
            return
        callback = lambda: self._on_result(asset_id, kind, result, error)
        if self._scheduler is not None:
            self._scheduler(callback)
        else:
            callback()

    def close(self) -> None:
        self.cancel()


def _latest_checkpoint(details: Any) -> Any:
    checkpoints = [checkpoint for checkpoint in value(details, "retained_checkpoints", []) or []
                   if value(checkpoint, "retained", True)]
    return max(checkpoints, key=lambda item: int(value(item, "iteration", 0) or 0), default=None)


def _latest_save(details: Any) -> Any:
    saves = list(value(details, "save_history", []) or [])
    return max(saves, key=lambda item: int(value(item, "sequence", 0) or 0), default=None)


def _format_path(path: Any) -> str:
    return str(path or "")


def details_rows(entry: Any, details: Any, *, format_size: Callable[[Any], str], format_time: Callable[[Any], str]) -> dict[str, Any]:
    """Shape native detail objects into empty-row-safe Inspector values."""
    card = value(details, "card", None)
    storage = value(details, "storage", None)
    params = value(details, "parameters", None)
    scene = value(details, "scene_graph", None)
    latest = _latest_checkpoint(details)
    latest_save = _latest_save(details)
    saves = list(value(details, "save_history", []) or [])
    references = list(value(details, "references", []) or [])
    metrics = value(details, "metrics", None)
    embedded = bool(value(params, "embedded_dataset_present", False))
    embedded_images = int(value(params, "embedded_images", 0) or 0)
    embedded_normals = int(value(params, "embedded_normals", 0) or 0)
    embedded_sparse = int(value(params, "embedded_sparse", 0) or 0)
    external = next((ref for ref in references if str(value(ref, "kind", "")).lower() in {"dataset", "images", "data"}), None)
    external_path = _format_path(value(external, "path", "")) if external else ""
    external_reachable = bool(value(external, "reachable", False)) if external else False
    has_samples = bool(value(metrics, "loss_samples", 0) or value(metrics, "psnr_samples", 0))
    dead_bytes = int(value(storage, "dead_bytes", 0) or 0)
    physical = int(value(storage, "physical_bytes", value(card, "physical_file_size", value(entry, "file_size_bytes", 0))) or 0)
    ratio = float(value(storage, "dead_ratio", (dead_bytes / physical if physical else 0.0)) or 0.0)
    iteration = value(card, "iteration", None)
    if iteration is None:
        iteration = value(latest, "iteration", None)
    model = {
        "saved": f"Save {int(value(latest_save, 'sequence', len(saves)) or len(saves))} of {len(saves)}" if saves else "",
        "saved_at": format_time(value(latest_save, "saved_at_unix_ns", value(card, "saved_at_unix_ns", 0))),
        "opened": format_time(value(entry, "last_opened_at_unix_ns", value(entry, "opened_at_unix_ns", 0))),
        "iteration": str(iteration) if iteration is not None else "",
        "strategy": str(value(params, "active_strategy", "") or ""),
        "resumable": bool(latest and value(latest, "binds_scene_graph", False)),
        "gaussians": f"{int(value(latest, 'gaussians', 0) or 0):,}" if latest and value(latest, "gaussians", 0) else "",
        "sh_degree": str(value(latest, "sh_degree", "") or "") if latest else "",
        "dataset": "",
        "dataset_path": external_path,
        "dataset_reachable": external_reachable,
        "embedded": embedded,
        "embedded_images": embedded_images,
        "embedded_normals": embedded_normals,
        "embedded_sparse": embedded_sparse,
        "embedded_complete": bool(value(params, "embedded_dataset_complete", False)),
        "dataset_node": str(value(scene, "dataset_node_name", "") or ""),
        "metrics": "",
        "license_identifier": str(value(value(details, "license", None), "identifier", "") or ""),
        "license_notice": str(value(value(details, "license", None), "notice", "") or ""),
        "title": str(value(card, "title", "") or ""),
        "physical_size": format_size(physical),
        "dead_bytes": format_size(dead_bytes),
        "reclaimable_percent": f"{ratio * 100.0:.1f}%",
        "saves": str(len(saves)),
        "autosave_newer": bool(value(details, "autosave_sidecar_present", False)),
        "chapter_count": str(len(list(value(details, "chapters", []) or []))),
        "has_metrics": has_samples,
        "metric_samples": str(int(value(metrics, "loss_samples", 0) or 0) + int(value(metrics, "psnr_samples", 0) or 0)) if has_samples else "",
    }
    if embedded:
        complete = "complete" if model["embedded_complete"] else "incomplete"
        model["dataset"] = f"embedded, {embedded_images} images, {embedded_normals} normals and {embedded_sparse} sparse, {complete}"
    elif external_path:
        model["dataset"] = f"{external_path} ({'reachable' if external_reachable else 'missing'})"
    if has_samples:
        model["metrics"] = f"{model['metric_samples']} samples"
    return model


def operation_actions(entry: Any) -> list[dict[str, Any]]:
    status = str(value(entry, "status", "READING") or "READING")
    if not value(entry, "path", "") or status in {"MISSING", "IDENTITY_MISMATCH", "UNREADABLE", "UNSUPPORTED_NEWER"}:
        return []
    if status == "REPAIR_ONLY":
        return [{"action": "repair", "label": "projects.action.repair"}]
    return [
        {"action": "contents", "label": "projects.contents.title"},
        {"action": "export_as", "label": "projects.action.export_as"},
        {"action": "update_thumbnail", "label": "projects.action.update_thumbnail"},
        {"action": "rename", "label": "projects.action.rename"},
    ]


def dialog_model(kind: str, *, entry: Any = None, details: Any = None) -> dict[str, Any]:
    """Return a stable model for each Inspector dialog kind."""
    kind = str(kind or "")
    base = {"kind": kind, "name": str(value(entry, "name", "") or ""), "path": str(value(entry, "path", "") or "")}
    if kind == "export_as":
        base.update({"format": "sog", "destination": "", "formats": ["ply", "sog", "ssog", "spz"]})
    elif kind == "update_thumbnail":
        base.update({"sources": ["viewport", "first_dataset", "first_embedded", "image_file"], "source": "first_dataset"})
    elif kind == "license":
        license_obj = value(details, "license", None)
        base.update(license_fields(license_obj))
    elif kind == "rename":
        base["title"] = str(value(value(details, "card", None), "title", "") or "")
    elif kind == "repair":
        base.update({"destination": "", "summary": ""})
    return base


LICENSES = (
    ("CC0-1.0", "cc0"), ("CC-BY-4.0", "by"), ("CC-BY-SA-4.0", "by_sa"),
    ("CC-BY-NC-4.0", "by_nc"), ("CC-BY-NC-SA-4.0", "by_nc_sa"),
    ("CC-BY-ND-4.0", "by_nd"), ("CC-BY-NC-ND-4.0", "by_nc_nd"),
    ("LicenseRef-Proprietary", "proprietary"), ("custom", "custom"),
)


def license_name(identifier: str, tr: Callable[[str], str]) -> str:
    key = next((key for ident, key in LICENSES if ident == identifier), None)
    return tr("projects.license." + key) if key else identifier.removeprefix("LicenseRef-")


def license_fields(license_obj: Any) -> dict[str, str]:
    identifier = str(value(license_obj, "identifier", "") or "")
    notice = str(value(license_obj, "notice", "") or "")
    known = {ident for ident, _key in LICENSES if ident != "custom"}
    text, sep, credit = notice.rpartition("\nCredit: ")
    if not sep:
        text, credit = ("", notice[8:]) if notice.startswith("Credit: ") else (notice, "")
    return dict(license_choice=identifier if identifier in known else "custom" if identifier else "CC-BY-4.0",
                license_name=identifier.removeprefix("LicenseRef-") if identifier not in known else "",
                license_text=text, attribution=credit)


def license_value(fields: dict[str, Any]) -> tuple[str, str]:
    choice = str(fields.get("license_choice") or "CC-BY-4.0")
    notice = ""
    if choice == "custom":
        name = str(fields.get("license_name") or "").strip()
        # SPDX LicenseRef suffixes allow letters, digits, dots, and hyphens.
        suffix = re.sub(r"[^A-Za-z0-9.-]", "", name)
        notice = str(fields.get("license_text") or "").strip()
        if not suffix or not notice:
            raise ValueError("projects.license.custom_required")
        choice = "LicenseRef-" + suffix
    elif choice not in {identifier for identifier, _ in LICENSES}:
        raise ValueError("projects.license.custom_required")
    credit = str(fields.get("attribution") or "").strip()
    if credit and choice not in {"CC0-1.0", "LicenseRef-Proprietary"}:
        notice = (notice + "\n" if notice else "") + "Credit: " + credit
    return choice, notice


def pending_removals(details: Any) -> list[dict[str, Any]]:
    raw = value(details, "manifest", {}).get("contents_removals", "")
    try:
        rows = json.loads(raw).get("rows", []) if raw else []
        return [row for row in rows if isinstance(row, dict)]
    except (ValueError, TypeError, AttributeError):
        return []


def contents_rows(entry: Any, details: Any, plan: Any = None, *,
                  tr: Callable[[str], str], format_size: Callable[[Any], str],
                  format_time: Callable[[Any], str], busy: bool = False) -> list[dict[str, Any]]:
    """One row per existing part, with durable removals and only relevant actions."""
    if details is None:
        return []
    rows = []
    pending = pending_removals(details)
    removed_ids = {str(row.get("id", "")) for row in pending}
    params = value(details, "parameters", None)
    saves = list(value(details, "save_history", []) or [])
    current = max((int(value(save, "generation", 0)) for save in saves), default=0)
    chapters = list(value(details, "chapters", []) or [])
    def size_of(code):
        return sum(int(value(part, "stored_bytes", 0)) for part in chapters if str(value(part, "fourcc", "")) == code)
    def row(id, kind, label, size=0, action="", action_label="", remove=False, secondary="", secondary_label="", **extra):
        result = dict(id=id, kind=kind, label=label, size=format_size(size) if size else "", bytes=int(size),
                      action=action, action_label=tr(action_label) if action_label else "",
                      action_tooltip=tr("projects.action.resume_training") if action == "resume" else tr(action_label) if action_label else "",
                      secondary=secondary, secondary_label=tr(secondary_label) if secondary_label else "",
                      removable=remove, disabled=busy, pending=False, remove_label=tr("projects.contents.remove"), **extra)
        rows.append(result)
        return result
    for index, save in reversed(list(enumerate(saves, 1))):
        generation = int(value(save, "generation", index))
        if "save:" + str(generation) in removed_ids:
            continue
        label = tr("projects.contents.save").format(number=index, total=len(saves))
        date = format_time(value(save, "saved_at_unix_ns", 0))
        if date: label += ", " + date
        if generation == current: label += ", " + tr("projects.contents.current")
        row("save:" + str(generation), "save", label, value(save, "bytes_added", 0),
            "restore" if generation != current else "", "projects.contents.restore" if generation != current else "",
            generation != current, generation=generation)
    checkpoints = list(value(details, "retained_checkpoints", []) or [])
    sizes = {str(value(cp, "instance_uuid", "")): int(value(cp, "bytes", 0)) for cp in value(plan, "retained_checkpoints", []) or []}
    strategy = str(value(params, "active_strategy", "") or "")
    for cp in checkpoints:
        if not value(cp, "retained", True): continue
        uuid = str(value(cp, "instance_uuid", ""))
        iteration = int(value(cp, "iteration", 0))
        label = tr("projects.contents.checkpoint").format(iteration=iteration)
        if strategy: label += ", " + strategy
        resumable = bool(value(cp, "header_reachable", True)) and bool(value(value(details, "scene_graph", None), "training_node_id", None))
        row("checkpoint:" + uuid, "checkpoint", label, sizes.get(uuid, 0),
            "resume" if resumable else "", "projects.contents.resume" if resumable else "", True,
            checkpoint_uuid=uuid, iteration=iteration, bound=bool(value(cp, "binds_scene_graph", False)))
    embedded = bool(value(params, "embedded_dataset_present", False))
    if embedded:
        count = int(value(params, "embedded_images", 0))
        r = row("dataset:embedded", "dataset", tr("projects.contents.dataset_embedded").format(count=count),
                sum(int(value(part, "bytes", 0)) for part in value(plan, "embedded_dataset", []) or []), remove=True)
        r["remove_disabled"] = not bool(value(value(plan, "drop_embedded_dataset", None), "allowed", False))
        r["remove_label"] = tr("projects.contents.dataset_kept") if r["remove_disabled"] else tr("projects.contents.remove")
    else:
        external = next((ref for ref in value(details, "references", []) or [] if str(value(ref, "kind", "")).lower() in {"dataset", "images", "data"}), None)
        dataset_node = str(value(value(details, "scene_graph", None), "dataset_node_name", "") or "")
        if external or dataset_node:
            reachable = bool(value(external, "reachable", False))
            label = tr("projects.contents.dataset_external").format(path=value(external, "path", "")).rstrip(", 、，")
            label += ", " + tr("projects.contents.reachable" if reachable else "projects.contents.missing")
            r = row("dataset:external", "external", label, action="embed", action_label="projects.contents.embed",
                    secondary="locate", secondary_label="projects.contents.locate")
            r["action_disabled"] = not reachable
    has_thumbnail = bool(value(value(details, "card", None), "has_preview", False))
    row("thumbnail", "thumbnail", tr("projects.contents.thumbnail" if has_thumbnail else "projects.contents.add_thumbnail"),
        size_of("THMB") if has_thumbnail else 0, "thumbnail", "projects.contents.update" if has_thumbnail else "projects.contents.add", has_thumbnail)
    metrics = value(details, "metrics", None)
    samples = int(value(metrics, "loss_samples", 0)) + int(value(metrics, "psnr_samples", 0))
    if samples:
        row("metrics", "metrics", tr("projects.contents.metrics").format(count=samples), size_of("METR"), remove=True)
    license_obj = value(details, "license", None)
    identifier = str(value(license_obj, "identifier", "") or "")
    license_bytes = len(identifier.encode("utf-8")) + len(str(value(license_obj, "notice", "") or "").encode("utf-8"))
    row("license", "license", tr("projects.contents.license").format(name=license_name(identifier, tr)) if identifier else tr("projects.contents.add_license"),
        license_bytes,
        action="license", action_label="projects.contents.change" if identifier else "projects.contents.add", remove=bool(identifier))
    for removed in pending:
        kind = removed.get("kind", "")
        if kind == "save": label = tr("projects.contents.save").format(number=removed.get("generation", ""), total=len(saves))
        elif kind == "checkpoint": label = tr("projects.contents.checkpoint").format(iteration=removed.get("iteration", 0))
        elif kind == "dataset": label = tr("projects.contents.dataset_embedded").format(count=removed.get("images", 0))
        elif kind == "metrics": label = tr("projects.contents.metrics").format(count=removed.get("samples", 0))
        elif kind == "license": label = tr("projects.contents.license").format(name=license_name(removed.get("identifier", ""), tr))
        else: label = tr("projects.contents.thumbnail")
        r = row("removed:" + str(len(rows)), kind, label + ", " + tr("projects.contents.removed"), removed.get("bytes", 0))
        r.update(pending=True, disabled=True)
    storage = value(details, "storage", None)
    ratio = float(value(storage, "dead_ratio", 0) or 0)
    if ratio >= 0.01:
        row("compact", "compact", tr("projects.contents.reclaimable").format(percent=f"{ratio * 100:.0f}"),
            value(storage, "dead_bytes", 0), "compact", "projects.contents.compact")
    for r in rows:
        r["has_action"] = bool(r["action"])
        r["has_secondary"] = bool(r["secondary"])
        r.setdefault("action_disabled", False)
        r.setdefault("remove_disabled", False)
        r["disabled"] = bool(r["disabled"] or not value(entry, "path", "") or str(value(entry, "status", "")) in {"MISSING", "UNREADABLE", "REPAIR_ONLY", "UNSUPPORTED_NEWER", "IDENTITY_MISMATCH"})
    return rows
