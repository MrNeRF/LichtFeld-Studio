# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure Python models for the Projects Inspector and closed-file actions.

The panel owns the worker and native calls.  This module deliberately contains
only cache keys, presentation shaping, and action decisions so those rules can
be tested without starting LichtFeld Studio.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from .localization import safe_format


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
        self._pending_deliveries: set[tuple[str, str]] = set()
        self._details_last_started: dict[str, float] = {}
        self._cancel: Optional[threading.Event] = None
        self._generation = 0
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    @property
    def cache(self) -> dict[str, InspectionCache]:
        return self._cache

    def cached(self, asset_id: str) -> Optional[InspectionCache]:
        with self._lock:
            return self._cache.get(str(asset_id))

    def invalidate(self, asset_id: Optional[str] = None) -> None:
        with self._lock:
            if asset_id is None:
                self._cache.clear()
                return
            self._cache.pop(str(asset_id), None)

    def cancel(self) -> None:
        with self._lock:
            if self._cancel is not None:
                self._cancel.set()
            self._generation += 1
            for asset_id, _kind in self._pending_deliveries:
                self._cache.pop(asset_id, None)
            self._pending_deliveries.clear()

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
                    except Exception as exc:
                        logging.getLogger(__name__).exception("Inspect project card failed path=%s", path)
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
                cached = self._cache.get(asset_id)
                if cached is None or cached.card is None:
                    continue
                # Cards classify missing, damaged and newer-version files
                # without throwing. Structural details require an open project.
                open_state = value(cached.card, "open_state", "OPEN")
                if str(value(open_state, "name", open_state)).rsplit(".", 1)[-1] != "OPEN":
                    continue
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
                    logging.getLogger(__name__).exception("Inspect project details failed path=%s", path)
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
        self._pending_deliveries.add((asset_id, kind))

        def callback():
            with self._lock:
                if cancel.is_set():
                    return
                self._pending_deliveries.discard((asset_id, kind))
            try:
                self._on_result(asset_id, kind, result, error)
            except Exception as exc:
                logging.getLogger(__name__).exception("Deliver project inspection failed project=%s kind=%s", asset_id, kind)
                with self._lock:
                    cached = self._cache.get(asset_id)
                    if cached is not None:
                        cached.error = str(exc)
                        cached.card = cached.details = None

        if self._scheduler is not None:
            try:
                self._scheduler(callback)
            except Exception as exc:
                logging.getLogger(__name__).exception("Schedule project inspection failed project=%s kind=%s", asset_id, kind)
                with self._lock:
                    cached = self._cache.get(asset_id)
                    if cached is not None:
                        cached.error = str(exc)
                        cached.card = cached.details = None
        else:
            callback()

    def close(self) -> None:
        self.cancel()


def _latest_checkpoint(details: Any) -> Any:
    checkpoints = [checkpoint for checkpoint in value(details, "retained_checkpoints", []) or []
                   if value(checkpoint, "retained", True)]
    return max(checkpoints, key=lambda item: int(value(item, "iteration", 0) or 0), default=None)


def _format_path(path: Any) -> str:
    return str(path or "")


def details_rows(entry: Any, details: Any, *, format_size: Callable[[Any], str], format_time: Callable[[Any], str]) -> dict[str, Any]:
    """Shape native detail objects into empty-row-safe Inspector values."""
    card = value(details, "card", None)
    storage = value(details, "storage", None)
    params = value(details, "parameters", None)
    scene = value(details, "scene_graph", None)
    latest = _latest_checkpoint(details)
    saves = save_groups(details)
    current_generation = int(value(card, "generation", 0) or 0)
    current_save = next((group for group in saves if group["generation"] == current_generation), saves[-1] if saves else None)
    latest_save = current_save["save"] if current_save else None
    save_count = sum(group["numbered"] for group in saves)
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
        "saved": f"Save {saves.index(current_save) + 1:,} of {save_count:,}" if current_save and current_save["numbered"] else "",
        "saved_at": format_time(value(latest_save, "saved_at_unix_ns", value(card, "saved_at_unix_ns", 0))),
        "opened": format_time(value(entry, "last_opened_at_unix_ns", value(entry, "opened_at_unix_ns", 0))),
        "iteration": f"{int(iteration):,}" if iteration is not None else "",
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
        "saves": f"{save_count:,}",
        "autosave_newer": bool(value(details, "autosave_sidecar_present", False)),
        "chapter_count": str(len(list(value(details, "chapters", []) or []))),
        "has_metrics": has_samples,
        "metric_samples": f"{int(value(metrics, 'loss_samples', 0) or 0) + int(value(metrics, 'psnr_samples', 0) or 0):,}" if has_samples else "",
    }
    if embedded:
        complete = "complete" if model["embedded_complete"] else "incomplete"
        model["dataset"] = f"embedded, {embedded_images:,} images, {embedded_normals:,} normals and {embedded_sparse:,} sparse, {complete}"
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
        {"action": "inspector", "label": "projects.inspector.title"},
        {"action": "export_as", "label": "projects.action.export_as"},
        {"action": "update_thumbnail", "label": "projects.action.update_thumbnail"},
        {"action": "rename", "label": "projects.action.rename"},
    ]


def thumbnail_source_options(
    entry: Any,
    *,
    dataset_available: bool = False,
    embedded_available: bool = False,
    viewport_available: bool = False,
) -> list[str]:
    """Return thumbnail sources that are available for this exact project."""
    options: list[str] = []
    if viewport_available:
        options.append("viewport")
    if dataset_available:
        options.append("first_dataset")
    if embedded_available:
        options.append("first_embedded")

    options.append("image_file")
    return options


def dialog_model(
    kind: str,
    *,
    entry: Any = None,
    details: Any = None,
    dataset_available: bool = False,
    embedded_available: bool = False,
    viewport_available: bool = False,
) -> dict[str, Any]:
    """Return a stable model for each Inspector dialog kind."""
    kind = str(kind or "")
    base = {"kind": kind, "name": str(value(entry, "name", "") or ""), "path": str(value(entry, "path", "") or "")}
    if kind == "export_as":
        base.update({"format": "sog", "destination": "", "formats": ["ply", "sog", "ssog", "spz"]})
    elif kind == "update_thumbnail":
        sources = thumbnail_source_options(
            entry,
            dataset_available=dataset_available,
            embedded_available=embedded_available,
            viewport_available=viewport_available,
        )
        preferred = next(
            (source for source in ("first_dataset", "first_embedded", "viewport") if source in sources),
            "image_file",
        )
        base.update({"sources": sources, "source": preferred})
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


def commit_kind(save: Any) -> str:
    kind = value(save, "kind", "EXPLICIT")
    return str(value(kind, "name", kind)).rsplit(".", 1)[-1].upper()


def save_groups(details: Any) -> list[dict[str, Any]]:
    """Keep the user state and its later Contents edits in one restore row."""
    groups = []
    for save in value(details, "save_history", []) or []:
        generation = int(value(save, "generation", len(groups) + 1))
        if commit_kind(save) in {"CONTENTS", "COMPACTION"}:
            source_generation = int(value(save, "source_save_generation", 0) or 0)
            group = next((item for item in reversed(groups)
                          if item["source_generation"] == source_generation), None)
            if group is None and groups:
                group = groups[-1]
            if group is None:
                # Compaction keeps the source save's facts in native inspection.
                # An older file without that provenance is an unnumbered state.
                source_date = int(value(save, "source_saved_at_unix_ns", 0) or 0)
                source = dict(saved_at_unix_ns=source_date,
                              kind=value(save, "source_save_kind", "EXPLICIT"),
                              checkpoint_iteration=value(save, "checkpoint_iteration"),
                              planned_iterations=value(save, "planned_iterations"),
                              strategy=value(save, "strategy", ""),
                              gaussians=value(save, "gaussians"))
                group = dict(save=source, source_generation=source_generation or generation,
                             generation=generation, edits=[], bytes=0, numbered=bool(source_date))
                groups.append(group)
            group["generation"] = generation
            group["edits"].append(save)
            group["bytes"] += int(value(save, "bytes_added", 0) or 0)
        else:
            groups.append(dict(save=save, source_generation=generation, generation=generation,
                               edits=[], bytes=int(value(save, "bytes_added", 0) or 0), numbered=True))
    return groups


def contents_rows(entry: Any, details: Any, plan: Any = None, *,
                  tr: Callable[[str], str], format_size: Callable[[Any], str],
                  format_time: Callable[[Any], str], busy: bool = False) -> list[dict[str, Any]]:
    """One row per existing part, with durable removals and only relevant actions."""
    if details is None:
        return []
    rows = []
    pending = pending_removals(details)
    params = value(details, "parameters", None)
    saves = save_groups(details)
    current = int(value(value(details, "card", None), "generation", 0) or
                  max((save["generation"] for save in saves), default=0))
    chapters = list(value(details, "chapters", []) or [])
    def size_of(code):
        return sum(int(value(part, "stored_bytes", 0)) for part in chapters if str(value(part, "fourcc", "")) == code)
    def row(id, kind, label, size=0, action="", action_label="", remove=False, secondary="", secondary_label="", **extra):
        result = dict(id=id, kind=kind, label=label, size=format_size(size) if size else "", bytes=int(size),
                      action=action, action_label=tr(action_label) if action_label else "",
                      action_tooltip=tr("projects.action.resume_training") if action == "resume" else tr(action_label) if action_label else "",
                      secondary=secondary, secondary_label=tr(secondary_label) if secondary_label else "",
                      removable=remove, disabled=busy, pending=False, remove_label=tr("projects.contents.remove"),
                      tooltip=label, detail="", current=False, autosave=False, undo=False, error="", **extra)
        rows.append(result)
        return result
    numbered = [save for save in saves if save["numbered"]]
    pending_saves = set()
    for index, group in reversed(list(enumerate(saves, 1))):
        save = group["save"]
        generation = group["generation"]
        removed = next((part for part in pending if part.get("kind") == "save" and
                        int(part.get("generation", 0)) in {generation, group["source_generation"]}), None)
        label = (safe_format(tr("projects.contents.save"), number=sum(item["numbered"] for item in saves[:index]), total=len(numbered))
                 if group["numbered"] else tr("projects.contents.saved_state"))
        date = format_time(value(save, "saved_at_unix_ns", 0))
        if date: label += ", " + date
        iteration = value(save, "checkpoint_iteration", None)
        planned = value(save, "planned_iterations", None)
        if iteration is not None:
            label += ", " + (tr("projects.contents.step_of").format(step=f"{iteration:,}", total=f"{planned:,}")
                              if planned else tr("projects.contents.step").format(step=f"{iteration:,}"))
        strategy = str(value(save, "strategy", "") or "")
        gaussians = value(save, "gaussians", None)
        if strategy: label += ", " + strategy
        if gaussians:
            count = f"{gaussians / 1_000_000:.1f} M" if gaussians >= 1_000_000 else f"{gaussians:,}"
            label += ", " + safe_format(tr("projects.contents.gaussians"), count=count)
        available = generation != current and removed is None
        r = row("save:" + str(group["source_generation"]), "save", label, group["bytes"],
                "restore" if available else "", "projects.contents.restore" if available else "",
                available, generation=generation)
        r.update(number=sum(item["numbered"] for item in saves[:index]), total=len(numbered))
        r.update(current=generation == current, autosave=commit_kind(save) == "AUTOSAVE")
        if index == len(saves):
            r["tooltip"] = label + "\n" + tr("projects.contents.save_explanation")
        edits = []
        for edit in group["edits"]:
            operation = str(value(edit, "operation", "") or
                            ("compacted" if commit_kind(edit) == "COMPACTION" else "changed"))
            edits.append(tr("projects.contents.edited").format(
                time=format_time(value(edit, "saved_at_unix_ns", 0)).rsplit(" ", 1)[-1],
                change=tr("projects.contents.edit." + operation)))
        r["detail"] = "; ".join(dict.fromkeys(edits))
        if removed:
            pending_saves.add(str(removed.get("id", "")))
            r.update(pending=True, undo=True, removal_id=removed["id"],
                     detail=tr("projects.contents.removed"))
    # Compaction rewrites physical chunk order; display order follows iteration.
    checkpoints = sorted(value(details, "retained_checkpoints", []) or [],
                         key=lambda cp: (int(value(cp, "iteration", 0)),
                                         str(value(cp, "instance_uuid", ""))))
    sizes = {str(value(cp, "instance_uuid", "")): int(value(cp, "bytes", 0)) for cp in value(plan, "retained_checkpoints", []) or []}
    strategy = str(value(params, "active_strategy", "") or "")
    for cp in checkpoints:
        if not value(cp, "retained", True): continue
        uuid = str(value(cp, "instance_uuid", ""))
        iteration = int(value(cp, "iteration", 0))
        label = safe_format(tr("projects.contents.checkpoint"), iteration=iteration)
        if strategy: label += ", " + strategy
        resumable = bool(value(cp, "header_reachable", True)) and bool(value(value(details, "scene_graph", None), "training_node_id", None))
        row("checkpoint:" + uuid, "checkpoint", label, sizes.get(uuid, 0),
            "resume" if resumable else "", "projects.contents.resume" if resumable else "", True,
            checkpoint_uuid=uuid, iteration=iteration, bound=bool(value(cp, "binds_scene_graph", False)))
    embedded = bool(value(params, "embedded_dataset_present", False))
    if embedded:
        count = int(value(params, "embedded_images", 0))
        r = row("dataset:embedded", "dataset", safe_format(tr("projects.contents.dataset_embedded"), count=count),
                sum(int(value(part, "bytes", 0)) for part in value(plan, "embedded_dataset", []) or []), remove=True, images=count)
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
        row("metrics", "metrics", safe_format(tr("projects.contents.metrics"), count=samples), size_of("METR"), remove=True)
    license_obj = value(details, "license", None)
    identifier = str(value(license_obj, "identifier", "") or "")
    license_bytes = len(identifier.encode("utf-8")) + len(str(value(license_obj, "notice", "") or "").encode("utf-8"))
    row("license", "license", tr("projects.contents.license").format(name=license_name(identifier, tr)) if identifier else tr("projects.contents.add_license"),
        license_bytes,
        action="license", action_label="projects.contents.change" if identifier else "projects.contents.add", remove=bool(identifier))
    for removed in pending:
        kind = removed.get("kind", "")
        # Payload removals already changed the active generation. Only old saves
        # remain addressable until Compact, and get a dim row with Undo.
        if kind != "save" or str(removed.get("id", "")) in pending_saves:
            continue
        label = tr("projects.contents.saved_state")
        date = format_time(removed.get("date", 0))
        if date:
            label += ", " + date
        r = row("removed:" + str(len(rows)), kind, label, removed.get("bytes", 0))
        r.update(pending=True, undo=True, removal_id=removed["id"], detail=tr("projects.contents.removed"))
    storage = value(details, "storage", None)
    ratio = float(value(storage, "dead_ratio", 0) or 0)
    if ratio >= 0.01 or pending:
        compact = row("compact", "compact", tr("projects.contents.reclaimable").format(percent=f"{ratio * 100:.0f}"),
                      value(storage, "dead_bytes", 0), "compact", "projects.contents.compact")
        if pending:
            compact["detail"] = safe_format(tr("projects.contents.compact_removals"), count=len(pending))
    for r in rows:
        r["has_action"] = bool(r["action"])
        r["has_secondary"] = bool(r["secondary"])
        r["has_detail"] = bool(r["detail"])
        r["has_error"] = bool(r["error"])
        r.setdefault("action_disabled", False)
        r.setdefault("remove_disabled", False)
        r.setdefault("removal_id", "")
        r["disabled"] = bool(r["disabled"] or not value(entry, "path", "") or str(value(entry, "status", "")) in {"MISSING", "UNREADABLE", "REPAIR_ONLY", "UNSUPPORTED_NEWER", "IDENTITY_MISMATCH"})
    return rows
