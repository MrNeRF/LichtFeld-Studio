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
    # Always use "Default" project for all imported assets
    project = index.find_or_create_project("Default")
    # Use derived scene name for organization within the Default project
    _project_name, scene_name = derive_project_scene_names(normalized_path)
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
        # Always use "Default" project for imported assets
        project = index.find_or_create_project("Default")
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


def _safe_node_name(node, fallback: str = "") -> str:
    try:
        return str(node.get("name"))
    except Exception:
        return fallback


def _resolve_transform_target(scene, node_name: str) -> tuple[str, str, str] | None:
    node = scene.get_node(node_name)
    if node is None:
        return None

    allowed_types = {"SPLAT", "POINTCLOUD", "MESH", "DATASET", "GROUP"}
    node_type = getattr(getattr(node, "type", None), "name", "")
    if node_type not in allowed_types:
        return None

    if node_type in {"SPLAT", "POINTCLOUD", "MESH"}:
        expected_transform_name = f"{node_name}_transform"
        for child_id in getattr(node, "children", []):
            child = scene.get_node_by_id(child_id)
            if child is None:
                continue
            if getattr(getattr(child, "type", None), "name", "") != "GROUP":
                continue
            child_name = _safe_node_name(child)
            if child_name == expected_transform_name:
                return child_name, node_name, node_type
        return node_name, node_name, node_type

    # Dataset assets should be anchored on the dataset node itself.
    # Using the helper group here can miss transforms applied on the dataset root.
    if node_type == "DATASET":
        return node_name, node_name, node_type

    if node_type == "GROUP" and getattr(node, "parent_id", -1) != -1:
        parent = scene.get_node_by_id(node.parent_id)
        if parent is not None:
            parent_name = _safe_node_name(parent)
            parent_type = getattr(getattr(parent, "type", None), "name", "")
            if parent_type in {"SPLAT", "POINTCLOUD", "MESH", "DATASET"}:
                if _safe_node_name(node) == f"{parent_name}_transform":
                    # Transform helpers are not standalone assets; save the parent asset.
                    return parent_name, parent_name, parent_type

    return node_name, node_name, node_type


def _extract_transform_metadata(node_name: str) -> dict[str, Any] | None:
    # Persist local transform for round-tripping via set_node_transform().
    # World-space is stored only as auxiliary metadata.
    local_matrix = lf.get_node_transform(node_name)
    if local_matrix is None:
        return None
    world_matrix = lf.get_node_visualizer_world_transform(node_name)
    decomp = lf.decompose_transform(local_matrix) or {}
    return {
        "matrix": list(local_matrix),
        "world_matrix": list(world_matrix) if world_matrix is not None else None,
        "translation": decomp.get("translation", [0.0, 0.0, 0.0]),
        "rotation_euler_deg": decomp.get("rotation_euler_deg", [0.0, 0.0, 0.0]),
        "rotation_quat": decomp.get("rotation_quat", [0.0, 0.0, 0.0, 1.0]),
        "scale": decomp.get("scale", [1.0, 1.0, 1.0]),
    }


def _collect_subtree_transform_metadata(scene, root_name: str) -> dict[str, dict[str, Any]]:
    root = scene.get_node(root_name)
    if root is None:
        return {}

    result: dict[str, dict[str, Any]] = {}
    stack = [root]

    while stack:
        current = stack.pop()
        current_name = _safe_node_name(current)
        if not current_name:
            continue

        local_matrix = lf.get_node_transform(current_name)
        world_matrix = lf.get_node_visualizer_world_transform(current_name)
        result[current_name] = {
            "local_matrix": list(local_matrix) if local_matrix is not None else None,
            "world_matrix": list(world_matrix) if world_matrix is not None else None,
        }

        for child_id in getattr(current, "children", []):
            child = scene.get_node_by_id(child_id)
            if child is not None:
                stack.append(child)

    return result


def _asset_type_from_node_type(node_type: str) -> str:
    type_mapping = {
        "SPLAT": "ply_3dgs",
        "POINTCLOUD": "ply_pcl",
        "MESH": "mesh",
        "DATASET": "dataset",
        "GROUP": "group",
    }
    return type_mapping.get(node_type, "unknown")


def _best_existing_asset_match(index: AssetIndex, transform_name: str, geometry_name: str) -> dict[str, Any] | None:
    for asset in index.assets.values():
        if asset.get("name") in {geometry_name, transform_name}:
            return asset

    for asset in index.assets.values():
        geo = asset.get("geometry_metadata", {}) or {}
        if geo.get("transform_node_name") == transform_name:
            return asset
        if geo.get("scene_node_name") == geometry_name:
            return asset

    return None


def _infer_source_path(index: AssetIndex, transform_name: str, geometry_name: str) -> tuple[str, str]:
    for candidate_name in (geometry_name, transform_name):
        try:
            node_path = lf.get_node_source_path(candidate_name)
            if node_path:
                normalized = os.path.abspath(node_path)
                if os.path.exists(normalized):
                    return normalized, normalized
        except Exception:
            pass

    for asset in index.assets.values():
        abs_path = asset.get("absolute_path") or ""
        if not abs_path or abs_path.startswith("scene://"):
            continue
        stem = Path(abs_path).stem
        if stem in {transform_name, geometry_name}:
            return asset.get("path") or abs_path, abs_path
    return "", ""


def save_asset_to_catalog(node_name: str) -> bool:
    if not ASSET_MANAGER_BACKEND_AVAILABLE:
        return False

    try:
        scene = lf.get_scene()
        if scene is None:
            _logger.warning("Cannot save asset: no scene available")
            return False

        target = _resolve_transform_target(scene, node_name)
        if target is None:
            _logger.warning("Cannot save asset: unsupported or missing node '%s'", node_name)
            return False

        transform_name, geometry_name, geometry_type = target
        if geometry_type not in {"SPLAT", "POINTCLOUD", "MESH", "DATASET", "GROUP"}:
            _logger.warning("Cannot save asset: unsupported type '%s' for node '%s'", geometry_type, geometry_name)
            return False
        transform_metadata = _extract_transform_metadata(transform_name)
        if transform_metadata is None:
            _logger.warning("Cannot save asset: missing transform for '%s'", transform_name)
            return False

        index = load_asset_index()
        if index is None:
            _logger.warning("Cannot save asset: asset index not available")
            return False

        geometry_metadata = {
            "transform_node_name": transform_name,
            "scene_node_name": geometry_name,
            "scene_node_type": geometry_type,
        }
        if geometry_type in {"GROUP", "DATASET"}:
            geometry_metadata["subtree_transforms"] = _collect_subtree_transform_metadata(scene, geometry_name)
        rel_path, abs_path = _infer_source_path(index, transform_name, geometry_name)

        existing_asset = _best_existing_asset_match(index, transform_name, geometry_name)
        if existing_asset is None and abs_path:
            matched_by_path = index.find_asset_by_path(abs_path)
            if matched_by_path is not None:
                existing_asset = matched_by_path.to_dict()
        if existing_asset is not None:
            update_kwargs: dict[str, Any] = {
                "name": geometry_name,
                "transform_metadata": transform_metadata,
                "geometry_metadata": {**(existing_asset.get("geometry_metadata", {}) or {}), **geometry_metadata},
            }
            if abs_path:
                update_kwargs["path"] = rel_path
                update_kwargs["absolute_path"] = abs_path
            updated = index.update_asset(
                existing_asset["id"],
                **update_kwargs,
            )
            if updated is not None:
                refresh_active_panel()
                return True
            return False

        project = index.find_or_create_project("Default")
        if project is None:
            _logger.warning("Cannot save asset: failed to create project")
            return False

        if not abs_path and geometry_type in {"GROUP", "DATASET"}:
            rel_path = f"scene://{geometry_name}"
            abs_path = rel_path

        if not abs_path:
            _logger.warning(
                "Cannot save asset '%s': missing source path for scene node '%s'",
                transform_name,
                geometry_name,
            )
            return False
        created = index.create_asset(
            project_id=project.id,
            name=geometry_name,
            type=_asset_type_from_node_type(geometry_type),
            path=rel_path,
            absolute_path=abs_path,
            role="scene_reference",
            geometry_metadata=geometry_metadata,
            transform_metadata=transform_metadata,
        )
        if created is not None:
            refresh_active_panel()
            return True
        return False

    except Exception as exc:
        _logger.error("Failed to save asset '%s': %s", node_name, exc, exc_info=True)
        return False


# Register callbacks with C++ runtime on module load
def _register_save_callbacks():
    """Register save asset callbacks with the C++ runtime."""
    try:
        lf.ui.set_save_asset_callback(save_asset_to_catalog)
        _logger.info("Registered save asset callback with C++ runtime")
    except Exception as e:
        _logger.error(f"Failed to register save asset callback: {e}", exc_info=True)


# Auto-register on module load
_register_save_callbacks()
