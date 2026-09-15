# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure Python models for the Projects Inspector and closed-file actions.

The panel owns the worker and native calls.  This module deliberately contains
only cache keys, presentation shaping, and action decisions so those rules can
be tested without starting LichtFeld Studio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional


def value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def enum_name(obj: Any) -> str:
    return str(getattr(obj, "name", obj) or "").upper()


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
    checkpoints = list(value(details, "retained_checkpoints", []) or [])
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


def reduce_plan_rows(plan: Any, *, format_size: Callable[[Any], str]) -> dict[str, Any]:
    checkpoints = list(value(plan, "retained_checkpoints", []) or [])
    payloads = list(value(plan, "embedded_dataset", []) or [])
    drop_cp = value(plan, "drop_checkpoints", None)
    drop_data = value(plan, "drop_embedded_dataset", None)
    compact = value(plan, "compact", None)
    return {
        "physical_size": format_size(value(plan, "physical_size", 0)),
        "reclaimable_bytes": format_size(
            int(value(drop_cp, "reclaimable_bytes", 0) or 0)
            + int(value(drop_data, "reclaimable_bytes", 0) or 0)
            + int(value(compact, "reclaimable_bytes", 0) or 0)
        ),
        "checkpoints": [{"iteration": value(row, "iteration", 0), "bytes": format_size(value(row, "bytes", 0)), "locked": bool(value(row, "scng_bound", False))} for row in checkpoints],
        "dataset": [{"path": str(value(row, "rel_path", "") or ""), "kind": str(value(row, "kind", "") or ""), "bytes": format_size(value(row, "bytes", 0)), "validated": bool(value(row, "external_replacement_validated", False))} for row in payloads],
        "drop_checkpoints_allowed": bool(value(drop_cp, "allowed", False)),
        "drop_dataset_allowed": bool(value(drop_data, "allowed", False)),
        "projected_checkpoint_size": format_size(value(drop_cp, "projected_size", 0)),
        "projected_dataset_size": format_size(value(drop_data, "projected_size", 0)),
        "projected_compact_size": format_size(value(compact, "projected_size", 0)),
    }


def operation_actions(entry: Any, details: Any = None, *, has_operation: bool = False) -> list[dict[str, Any]]:
    """Single action table shared by Inspector and context menus."""
    status = str(value(entry, "status", "READING") or "READING")
    path = str(value(entry, "path", "") or "")
    rows: list[dict[str, Any]] = []
    if not path or status in {"MISSING", "IDENTITY_MISMATCH", "UNREADABLE", "UNSUPPORTED_NEWER"}:
        return rows
    if status == "REPAIR_ONLY":
        return [{"action": "repair", "label": "projects.action.repair"}]
    if path and status not in {"MISSING", "UNREADABLE", "REPAIR_ONLY", "UNSUPPORTED_NEWER"}:
        rows.append({"action": "save_history", "label": "projects.action.save_history"})
    if details is not None:
        storage = value(details, "storage", None)
        reclaimable = float(value(storage, "dead_ratio", 0.0) or 0.0)
        if reclaimable > 0.10:
            rows.append({"action": "reduce_size", "label": "projects.action.reduce_size"})
        params = value(details, "parameters", None)
        if not bool(value(params, "embedded_dataset_present", False)) and any(str(value(ref, "kind", "")).lower() in {"dataset", "images", "data"} for ref in value(details, "references", []) or []):
            rows.append({"action": "embed_dataset", "label": "projects.action.embed_dataset"})
        if any(not bool(value(ref, "reachable", False)) for ref in value(details, "references", []) or []):
            rows.append({"action": "locate_dataset", "label": "projects.action.locate_dataset"})
    rows.extend([
        {"action": "export_as", "label": "projects.action.export_as"},
        {"action": "update_thumbnail", "label": "projects.action.update_thumbnail"},
        {"action": "set_license", "label": "projects.action.set_license"},
        {"action": "rename", "label": "projects.action.rename"},
    ])
    return rows


def dialog_model(kind: str, *, entry: Any = None, details: Any = None, plan: Any = None, format_size: Callable[[Any], str] = str, format_time: Callable[[Any], str] = str) -> dict[str, Any]:
    """Return a stable model for each Inspector dialog kind."""
    kind = str(kind or "")
    base = {"kind": kind, "name": str(value(entry, "name", "") or ""), "path": str(value(entry, "path", "") or ""), "rows": []}
    if kind == "save_history":
        base["rows"] = [{"kind": str(value(row, "kind", "") or "").rsplit(".", 1)[-1].lower(), "date": format_time(value(row, "saved_at_unix_ns", 0)), "iteration": str(value(row, "checkpoint_iteration", "")) if value(row, "checkpoint_iteration", None) is not None else "", "bytes_added": format_size(value(row, "bytes_added", 0)), "generation": int(value(row, "generation", 0) or 0), "holds_checkpoint": bool(value(row, "holds_checkpoint", False))} for row in value(details, "save_history", []) or []]
        base["generation"] = max((row["generation"] for row in base["rows"]), default=0)
    elif kind == "reduce_size":
        base.update(reduce_plan_rows(plan, format_size=format_size) if plan is not None else {})
        base["confirm_is_destructive"] = True
    elif kind == "export_as":
        base.update({"format": "sog", "destination": "", "formats": ["ply", "sog", "ssog", "spz"]})
    elif kind == "update_thumbnail":
        base.update({"sources": ["viewport", "first_dataset", "first_embedded", "image_file"], "source": "first_dataset"})
    elif kind == "set_license":
        license_obj = value(details, "license", None)
        base.update({"identifier": str(value(license_obj, "identifier", "") or ""), "notice": str(value(license_obj, "notice", "") or "")})
    elif kind == "rename":
        base["title"] = str(value(value(details, "card", None), "title", "") or "")
    elif kind == "repair":
        base.update({"destination": "", "summary": ""})
    return base
