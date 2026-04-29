# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared Asset Manager integration helpers.

These helpers keep non-panel code paths writing to the same JSON catalog that the
Asset Manager panel renders from.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import lichtfeld as lf

try:
    from .asset_index import AssetIndex
    from .asset_scanner import AssetScanner
    from .asset_thumbnails import AssetThumbnails

    ASSET_MANAGER_BACKEND_AVAILABLE = True
except ImportError:
    AssetIndex = None
    AssetScanner = None
    AssetThumbnails = None
    ASSET_MANAGER_BACKEND_AVAILABLE = False

_logger = logging.getLogger(__name__)
_active_panel = None
_DEFAULT_STORAGE_PATH = Path.home() / ".lichtfeld" / "asset_manager"


def set_active_asset_manager_panel(panel) -> None:
    global _active_panel
    _active_panel = panel


def clear_active_asset_manager_panel(panel) -> None:
    global _active_panel
    if _active_panel is panel:
        _active_panel = None


def get_asset_manager_panel():
    return _active_panel


def _storage_path() -> Path:
    _DEFAULT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_STORAGE_PATH


def load_asset_index(asset_index: Optional[AssetIndex] = None) -> Optional[AssetIndex]:
    if not ASSET_MANAGER_BACKEND_AVAILABLE:
        return None
    if asset_index is not None:
        return asset_index
    index = AssetIndex(library_path=_storage_path() / "library.json")
    index.load()
    return index


def load_scanner(scanner: Optional[AssetScanner] = None) -> Optional[AssetScanner]:
    if not ASSET_MANAGER_BACKEND_AVAILABLE:
        return None
    return scanner or AssetScanner()


def load_thumbnails(
    thumbnails: Optional[AssetThumbnails] = None,
) -> Optional[AssetThumbnails]:
    if not ASSET_MANAGER_BACKEND_AVAILABLE:
        return None
    return thumbnails or AssetThumbnails(_storage_path() / "thumbnails")


def metadata_to_asset_kwargs(metadata: dict[str, Any]) -> dict[str, Any]:
    format_specific = metadata.get("format_specific", {}) or {}
    asset_type = metadata.get("type") or "unknown"

    kwargs: dict[str, Any] = {
        "file_size_bytes": metadata.get("size_bytes", 0),
        "created_at": metadata.get("created"),
        "modified_at": metadata.get("modified"),
    }

    if asset_type in ("ply", "rad", "sog", "spz", "mesh"):
        kwargs["geometry_metadata"] = format_specific
    elif asset_type == "checkpoint":
        kwargs["training_metadata"] = format_specific
    elif asset_type == "dataset":
        kwargs["dataset_metadata"] = format_specific
    elif asset_type in ("video", "mp4", "mov"):
        normalized_video = dict(format_specific)
        resolution = normalized_video.pop("resolution", None)
        if resolution and "x" in resolution:
            width, height = resolution.split("x", 1)
            try:
                normalized_video["width"] = int(width)
                normalized_video["height"] = int(height)
            except ValueError:
                pass
        duration = normalized_video.pop("duration", None)
        if duration is not None:
            normalized_video["duration_seconds"] = duration
        kwargs["video_metadata"] = normalized_video

    return kwargs


def _generate_thumbnail(
    index: AssetIndex,
    asset,
    thumbnails: Optional[AssetThumbnails] = None,
) -> None:
    if thumbnails is None or asset is None:
        return
    try:
        thumb_path = thumbnails.generate_placeholder(asset.type, asset.id)
        index.update_asset(asset.id, thumbnail_path=str(thumb_path))
    except Exception as exc:
        _logger.debug("Failed to generate thumbnail for %s: %s", asset.id, exc)


def derive_project_scene_names(dataset_path: str) -> tuple[str, str]:
    normalized = os.path.normpath(dataset_path)
    scene_name = os.path.basename(normalized) or "Untitled Dataset"
    parent_dir = os.path.basename(os.path.dirname(normalized))
    project_name = parent_dir if parent_dir and parent_dir != "." else scene_name
    return project_name, scene_name


def resolve_latest_run_id(index: AssetIndex, scene_id: Optional[str]) -> Optional[str]:
    if not scene_id:
        return None
    runs = index.list_runs(scene_id=scene_id)
    if not runs:
        return None

    def _sort_key(run) -> tuple[str, str]:
        return (
            getattr(run, "completed_at", "") or getattr(run, "modified_at", ""),
            getattr(run, "created_at", ""),
        )

    runs.sort(key=_sort_key, reverse=True)
    return runs[0].id


def backfill_scene_provenance(index: AssetIndex, scene_id: Optional[str]) -> bool:
    run_id = resolve_latest_run_id(index, scene_id)
    if not run_id or not scene_id:
        return False

    updated = False
    for asset in index.list_assets(scene_id=scene_id):
        if asset.run_id:
            continue
        if asset.role not in {"trained_output", "training_checkpoint"}:
            continue
        if index.update_asset(asset.id, run_id=run_id) is not None:
            updated = True
    return updated


def ensure_dataset_catalog_context(
    dataset_path: str,
    *,
    asset_index: Optional[AssetIndex] = None,
    scanner: Optional[AssetScanner] = None,
    thumbnails: Optional[AssetThumbnails] = None,
) -> dict[str, Optional[str]]:
    if not dataset_path:
        return {"project_id": None, "scene_id": None, "asset_id": None}

    index = load_asset_index(asset_index)
    scan = load_scanner(scanner)
    thumbs = load_thumbnails(thumbnails)
    if index is None:
        return {"project_id": None, "scene_id": None, "asset_id": None}

    normalized_path = os.path.abspath(dataset_path)
    project_name, scene_name = derive_project_scene_names(normalized_path)
    project = index.find_or_create_project(project_name)
    scene = index.find_or_create_scene(project.id, scene_name) if project else None
    project_id = project.id if project else None
    scene_id = scene.id if scene else None
    existing = index.find_asset_by_path(normalized_path)

    asset = existing
    if asset is None or asset.type != "dataset":
        metadata = scan.scan_file(normalized_path) if scan else {}
        asset_kwargs = metadata_to_asset_kwargs(metadata)
        asset = index.create_asset(
            project_id=project_id,
            name=Path(normalized_path).name,
            type="dataset",
            path=normalized_path,
            absolute_path=normalized_path,
            scene_id=scene_id,
            role="source_dataset",
            **asset_kwargs,
        )
        if asset is not None:
            _generate_thumbnail(index, asset, thumbs)
    else:
        update_kwargs: dict[str, Any] = {
            "project_id": project_id or asset.project_id,
            "scene_id": scene_id or asset.scene_id,
            "name": Path(normalized_path).name or asset.name,
            "role": asset.role or "source_dataset",
        }
        if scan is not None and os.path.exists(normalized_path):
            metadata = scan.scan_file(normalized_path)
            update_kwargs.update(metadata_to_asset_kwargs(metadata))
        asset = index.update_asset(asset.id, **update_kwargs) or asset

    if scene_id and asset is not None:
        index.update_scene(scene_id, dataset_asset_id=asset.id)

    return {
        "project_id": project_id,
        "scene_id": scene_id,
        "asset_id": asset.id if asset is not None else None,
    }


def register_catalog_asset_path(
    path: str,
    *,
    is_dataset: bool = False,
    asset_type: Optional[str] = None,
    role: Optional[str] = None,
    select: bool = False,
    asset_index: Optional[AssetIndex] = None,
    scanner: Optional[AssetScanner] = None,
    thumbnails: Optional[AssetThumbnails] = None,
) -> Optional[Any]:
    if not path or not ASSET_MANAGER_BACKEND_AVAILABLE:
        return None

    normalized_path = os.path.abspath(path)
    index = load_asset_index(asset_index)
    scan = load_scanner(scanner)
    thumbs = load_thumbnails(thumbnails)
    if index is None:
        return None

    if is_dataset:
        context = ensure_dataset_catalog_context(
            normalized_path,
            asset_index=index,
            scanner=scan,
            thumbnails=thumbs,
        )
        asset_id = context.get("asset_id")
        asset = index.get_asset(asset_id) if asset_id else None
        if asset is not None and select:
            select_asset_in_active_panel(
                asset.id,
                project_id=context.get("project_id"),
                scene_id=context.get("scene_id"),
            )
        return asset

    dataset_context = {"project_id": None, "scene_id": None, "asset_id": None}
    dataset_params = None
    try:
        dataset_params = lf.dataset_params()
    except Exception:
        dataset_params = None

    if dataset_params and dataset_params.has_params() and dataset_params.data_path:
        dataset_context = ensure_dataset_catalog_context(
            dataset_params.data_path,
            asset_index=index,
            scanner=scan,
            thumbnails=thumbs,
        )

    project_id = dataset_context.get("project_id")
    scene_id = dataset_context.get("scene_id")
    run_id = None
    if role in {"trained_output", "training_checkpoint"}:
        run_id = resolve_latest_run_id(index, scene_id)

    if project_id is None:
        project = index.find_or_create_project("Imported Assets")
        project_id = project.id if project else None

    metadata = scan.scan_file(normalized_path) if scan else {}
    asset_kwargs = metadata_to_asset_kwargs(metadata)
    detected_type = asset_type or metadata.get("type") or Path(normalized_path).suffix.lstrip(".").lower() or "unknown"
    detected_role = role or metadata.get("role") or "reference"

    asset = index.create_asset(
        project_id=project_id,
        name=Path(normalized_path).name,
        type=detected_type,
        path=normalized_path,
        absolute_path=normalized_path,
        scene_id=scene_id,
        run_id=run_id,
        role=detected_role,
        **asset_kwargs,
    )
    if asset is not None:
        _generate_thumbnail(index, asset, thumbs)

    if asset is not None and select:
        select_asset_in_active_panel(
            asset.id,
            project_id=project_id,
            scene_id=scene_id,
            run_id=run_id,
        )
    elif asset is not None:
        refresh_active_panel()

    return asset


def refresh_active_panel() -> None:
    panel = get_asset_manager_panel()
    if panel is None:
        return
    try:
        panel.refresh_catalog()
    except Exception:
        _logger.debug("Failed to refresh active Asset Manager panel", exc_info=True)


def select_asset_in_active_panel(
    asset_id: str,
    *,
    project_id: Optional[str] = None,
    scene_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    panel = get_asset_manager_panel()
    if panel is None:
        return

    try:
        panel._selected_asset_ids = {asset_id}
        if project_id is not None:
            panel._selected_project_id = project_id
        if scene_id is not None:
            panel._selected_scene_id = scene_id
        panel._selected_run_id = run_id
        panel._update_selection_type()
        panel.refresh_catalog()
    except Exception:
        _logger.debug("Failed to update active Asset Manager selection", exc_info=True)
