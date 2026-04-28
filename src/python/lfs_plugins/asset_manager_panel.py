# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager panel for browsing and managing Gaussian Splatting assets."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

import lichtfeld as lf

_logger = logging.getLogger(__name__)
from .types import Panel

# Import backend components (to be implemented)
try:
    from .asset_index import AssetIndex, Project, Scene, TrainingRun, Asset
    from .asset_scanner import AssetScanner
    from .asset_thumbnails import AssetThumbnails

    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    AssetIndex = None
    AssetScanner = None
    AssetThumbnails = None

__lfs_panel_classes__ = ["AssetManagerPanel"]
__lfs_panel_ids__ = ["lfs.asset_manager"]


class AssetManagerPanel(Panel):
    """Floating Asset Manager window for browsing splats, videos, and exports."""

    SORT_MODES = ("recent", "name", "size", "type")
    LOADABLE_TYPES = {"ply", "rad", "sog", "spz", "checkpoint", "dataset"}

    id = "lfs.asset_manager"
    label = "Asset Manager"
    space = lf.ui.PanelSpace.FLOATING
    order = 20
    template = "rmlui/asset_manager.rml"
    height_mode = lf.ui.PanelHeightMode.FILL
    size = (980, 620)
    update_interval_ms = 500

    # Storage path for asset manager data
    STORAGE_PATH = Path.home() / ".lichtfeld" / "asset_manager"

    def __init__(self):
        self._handle = None
        self._doc = None

        # Backend components
        self._asset_index: Optional[Any] = None
        self._asset_scanner: Optional[Any] = None
        self._asset_thumbnails: Optional[Any] = None

        # UI state
        self._selected_asset_ids: Set[str] = set()
        self._selected_project_id: Optional[str] = None
        self._selected_scene_id: Optional[str] = None
        self._selected_run_id: Optional[str] = None
        self._active_filter: str = "all"
        self._active_tab: str = "info"  # info, parameters, history
        self._view_mode: str = "gallery"  # gallery, list
        self._sort_mode: str = "recent"  # recent, name, size, type
        self._search_query: str = ""
        self._pending_tag_name: str = ""

        # Selection type for info panel display
        self._selection_type: str = "none"  # none, asset, run, multiple

        # Import menu state
        self._import_menu_open: bool = False
        self._library_mtime: float = 0.0
        self._updating_selection_details: bool = False

    # ── Initialization ────────────────────────────────────────

    def _initialize_backend(self):
        """Initialize backend components."""
        if not BACKEND_AVAILABLE:
            return False

        try:
            # Ensure storage directory exists
            self.STORAGE_PATH.mkdir(parents=True, exist_ok=True)

            # Initialize components
            self._asset_thumbnails = AssetThumbnails(self.STORAGE_PATH / "thumbnails")
            self._asset_scanner = AssetScanner()
            self._asset_index = AssetIndex(
                library_path=self.STORAGE_PATH / "library.json",
            )
            return True
        except Exception as e:
            _logger.warning(f"Failed to initialize asset manager backend: {e}")
            return False

    # ── Data model ────────────────────────────────────────────

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("asset_manager")
        if model is None:
            return

        # Basic properties
        model.bind_func("panel_label", lambda: "Asset Manager")
        model.bind_func("search_query", self.get_search_query)
        model.bind_func("asset_count", self.get_asset_count)
        model.bind_func("selected_count", self.get_selected_count)
        model.bind_func("selected_total_size", self.get_selected_total_size)
        model.bind_func("project_count", self.get_project_count)
        model.bind_func("scene_count", self.get_scene_count)
        model.bind_func("collection_count", self.get_collection_count)

        # Formatted strings for display (must be bound before getting handle)
        model.bind_func(
            "selected_count_text", lambda: f"{self.get_selected_count()} selected"
        )
        model.bind_func(
            "selected_total_text", lambda: f"Total: {self.get_selected_total_size()}"
        )

        # View state
        model.bind_func("view_mode", self.get_view_mode)
        model.bind_func("is_gallery_view", lambda: self._view_mode == "gallery")
        model.bind_func("is_list_view", lambda: self._view_mode == "list")
        model.bind_func("sort_mode", self.get_sort_mode)
        model.bind_func("sort_label", self.get_sort_label)
        model.bind_func("pending_tag_name", self.get_pending_tag_name)

        # Active states
        model.bind_func("active_filter", self.get_active_filter)
        model.bind_func("active_tab", self.get_active_tab)
        model.bind_func("selection_type", self.get_selection_type)

        # Import menu state
        model.bind_func("import_menu_open", self.get_import_menu_open)

        # Selected IDs for UI conditionals
        model.bind_func("selected_project_id", self.get_selected_project_id)
        model.bind_func("selected_scene_id", self.get_selected_scene_id)
        model.bind_func("selected_run_id", self.get_selected_run_id)

        # Selected asset properties (flattened bind_func pattern)
        model.bind_func("selected_asset_name", self.get_selected_asset_name)
        model.bind_func("selected_asset_type", self.get_selected_asset_type)
        model.bind_func("selected_asset_role", self.get_selected_asset_role)
        model.bind_func("selected_asset_project_name", self.get_selected_asset_project_name)
        model.bind_func("selected_asset_scene_name", self.get_selected_asset_scene_name)
        model.bind_func("selected_asset_run_name", self.get_selected_asset_run_name)
        model.bind_func("selected_asset_path", self.get_selected_asset_path)
        model.bind_func("selected_asset_size", self.get_selected_asset_size)
        model.bind_func("selected_asset_points", self.get_selected_asset_points)
        model.bind_func("selected_asset_resolution", self.get_selected_asset_resolution)
        model.bind_func("selected_asset_duration", self.get_selected_asset_duration)
        model.bind_func("selected_asset_created", self.get_selected_asset_created)
        model.bind_func("selected_asset_modified", self.get_selected_asset_modified)
        model.bind_func(
            "selected_asset_is_favorite", self.get_selected_asset_is_favorite
        )
        model.bind_func(
            "selected_asset_has_training_provenance",
            self.get_selected_asset_has_training_provenance,
        )
        model.bind_func(
            "selected_asset_source_dataset", self.get_selected_asset_source_dataset
        )
        model.bind_func(
            "selected_asset_training_run_id", self.get_selected_asset_training_run_id
        )
        model.bind_func("selected_asset_iterations", self.get_selected_asset_iterations)
        model.bind_func("selected_asset_sh_degree", self.get_selected_asset_sh_degree)
        model.bind_func("selected_asset_train_time", self.get_selected_asset_train_time)
        model.bind_func("selected_asset_optimizer", self.get_selected_asset_optimizer)
        model.bind_func(
            "selected_asset_learning_rate", self.get_selected_asset_learning_rate
        )
        model.bind_func("selected_asset_batch_size", self.get_selected_asset_batch_size)
        model.bind_func(
            "selected_asset_has_geometry_metadata",
            self.get_selected_asset_has_geometry_metadata,
        )
        model.bind_func(
            "selected_asset_has_dataset_metadata",
            self.get_selected_asset_has_dataset_metadata,
        )
        model.bind_func(
            "selected_asset_dataset_image_count",
            self.get_selected_asset_dataset_image_count,
        )
        model.bind_func(
            "selected_asset_dataset_image_root",
            self.get_selected_asset_dataset_image_root,
        )
        model.bind_func(
            "selected_asset_dataset_masks",
            self.get_selected_asset_dataset_masks,
        )
        model.bind_func(
            "selected_asset_dataset_sparse_model",
            self.get_selected_asset_dataset_sparse_model,
        )
        model.bind_func(
            "selected_asset_dataset_camera_count",
            self.get_selected_asset_dataset_camera_count,
        )
        model.bind_func(
            "selected_asset_dataset_database",
            self.get_selected_asset_dataset_database,
        )
        model.bind_func(
            "selected_asset_bounding_box", self.get_selected_asset_bounding_box
        )
        model.bind_func("selected_asset_center", self.get_selected_asset_center)
        model.bind_func("selected_asset_scale", self.get_selected_asset_scale)
        model.bind_func(
            "selected_asset_file_missing", self.get_selected_asset_file_missing
        )
        model.bind_func(
            "selected_asset_expected_path", self.get_selected_asset_expected_path
        )
        model.bind_func(
            "selected_asset_preview_class", self.get_selected_asset_preview_class
        )
        model.bind_func(
            "selected_asset_preview_label", self.get_selected_asset_preview_label
        )
        model.bind_func("selected_asset_pill_class", self.get_selected_asset_pill_class)
        model.bind_func("selected_asset_type_label", self.get_selected_asset_type_label)

        # Selected run properties (flattened)
        model.bind_func("selected_run_name", self.get_selected_run_name)
        model.bind_func("selected_run_status", self.get_selected_run_status)
        model.bind_func("selected_run_started", self.get_selected_run_started)
        model.bind_func("selected_run_completed", self.get_selected_run_completed)
        model.bind_func("selected_run_duration", self.get_selected_run_duration)

        # Selected scene properties (flattened)
        model.bind_func("selected_scene_name", self.get_selected_scene_name)
        model.bind_func(
            "selected_scene_project_name", self.get_selected_scene_project_name
        )
        model.bind_func(
            "selected_scene_asset_count", self.get_selected_scene_asset_count
        )
        model.bind_func("selected_scene_created", self.get_selected_scene_created)
        model.bind_func("selected_scene_modified", self.get_selected_scene_modified)

        # Selected project properties (flattened)
        model.bind_func("selected_project_name", self.get_selected_project_name)
        model.bind_func(
            "selected_project_scene_count", self.get_selected_project_scene_count
        )
        model.bind_func(
            "selected_project_total_assets", self.get_selected_project_total_assets
        )
        model.bind_func("selected_project_path", self.get_selected_project_path)
        model.bind_func("selected_project_created", self.get_selected_project_created)
        model.bind_func("selected_project_modified", self.get_selected_project_modified)

        # Record lists for data-for loops (main lists)
        model.bind_record_list("projects")
        model.bind_record_list("scenes")
        model.bind_record_list("filters")
        model.bind_record_list("tags")
        model.bind_record_list("collections")
        model.bind_record_list("assets")
        model.bind_record_list("selected_asset_tags")

        # Record lists for nested struct lists
        model.bind_record_list("selected_run_parameters")
        model.bind_record_list("selected_run_artifacts")
        model.bind_record_list("selected_scene_assets")
        model.bind_record_list("selected_project_scenes")

        self._handle = model.get_handle()

        # Initialize record lists
        self._update_all_record_lists()

        # Event handlers
        model.bind_event("set_filter", self.set_filter)
        model.bind_event("set_tab", self.set_tab)
        model.bind_event("set_view_mode", self.set_view_mode)
        model.bind_event("cycle_sort_mode", self.cycle_sort_mode)
        model.bind_event("toggle_asset_selection", self.toggle_asset_selection)
        model.bind_event("on_search", self.on_search)
        model.bind_event("on_import_asset", self.on_import_asset)
        model.bind_event("on_import_dataset", self.on_import_dataset)
        model.bind_event("on_import_folder", self.on_import_folder)
        model.bind_event("on_load_selected", self.on_load_selected)
        model.bind_event("on_remove_from_catalog", self.on_remove_from_catalog)
        model.bind_event("on_toggle_favorite", self.on_toggle_favorite)
        model.bind_event("select_project", self.select_project)
        model.bind_event("select_scene", self.select_scene)
        model.bind_event("toggle_import_menu", self.toggle_import_menu)
        model.bind_event("on_import_checkpoint", self.on_import_checkpoint)
        model.bind_event("on_locate_file", self.on_locate_file)
        model.bind_event("select_artifact", self.select_artifact)
        model.bind_event("select_asset", self.select_asset_by_id)
        model.bind_event("on_export_selected", self.on_export_selected)
        model.bind_event("on_load_asset", self.on_load_asset)
        model.bind_event("on_remove_asset", self.on_remove_asset)
        model.bind_event("on_pending_tag_change", self.on_pending_tag_change)
        model.bind_event("on_add_tag", self.on_add_tag)
        model.bind_event("on_remove_tag", self.on_remove_tag)

    # ── Data Retrieval Methods ─────────────────────────────────

    def get_search_query(self) -> str:
        return self._search_query

    def get_asset_count(self) -> int:
        return len(self.get_filtered_assets())

    def get_project_count(self) -> int:
        if self._asset_index and hasattr(self._asset_index, "projects"):
            return len(self._asset_index.projects)
        return 0

    def get_scene_count(self) -> int:
        if self._asset_index and hasattr(self._asset_index, "scenes"):
            if self._selected_project_id:
                return len(
                    [
                        s
                        for s in self._asset_index.scenes.values()
                        if s.get("project_id") == self._selected_project_id
                    ]
                )
            return len(self._asset_index.scenes)
        return 0

    def get_collection_count(self) -> int:
        if self._asset_index and hasattr(self._asset_index, "collections"):
            return len(self._asset_index.collections)
        return 0

    def get_selected_count(self) -> int:
        self._reconcile_selection()
        return len(self._selected_asset_ids)

    def get_selected_total_size(self) -> str:
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return "0 MB"

        self._reconcile_selection()
        total_bytes = 0
        for asset_id in self._selected_asset_ids:
            asset = self._asset_index.assets.get(asset_id)
            if asset:
                total_bytes += asset.get("file_size_bytes", 0)
        return self._format_size(total_bytes)

    def get_view_mode(self) -> str:
        return self._view_mode

    def get_sort_mode(self) -> str:
        return self._sort_mode

    def get_sort_label(self) -> str:
        labels = {
            "recent": "Sort by: Recent",
            "name": "Sort by: Name",
            "size": "Sort by: File Size",
            "type": "Sort by: Type",
        }
        return labels.get(self._sort_mode, "Sort by: Recent")

    def get_active_filter(self) -> str:
        return self._active_filter

    def get_active_tab(self) -> str:
        return self._active_tab

    def get_selection_type(self) -> str:
        return self._selection_type

    def get_import_menu_open(self) -> bool:
        return self._import_menu_open

    def get_pending_tag_name(self) -> str:
        return self._pending_tag_name

    def get_selected_project_id(self) -> Optional[str]:
        return self._selected_project_id

    def get_selected_scene_id(self) -> Optional[str]:
        return self._selected_scene_id

    def get_selected_run_id(self) -> Optional[str]:
        return self._selected_run_id

    def _format_size(self, file_size_bytes: int) -> str:
        if file_size_bytes >= 1024**3:
            return f"{file_size_bytes / (1024**3):.2f} GB"
        if file_size_bytes >= 1024**2:
            return f"{file_size_bytes / (1024**2):.1f} MB"
        if file_size_bytes >= 1024:
            return f"{file_size_bytes / 1024:.1f} KB"
        return f"{file_size_bytes} B"

    def _reconcile_selection(self) -> None:
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            self._selected_asset_ids.clear()
            self._update_selection_type()
            return
        valid_ids = set(self._asset_index.assets.keys())
        if not self._selected_asset_ids.issubset(valid_ids):
            self._selected_asset_ids.intersection_update(valid_ids)
            self._update_selection_type()

    def _get_asset_relationship_names(self, asset: Dict[str, Any]):
        project_name = ""
        scene_name = ""
        run_name = ""

        if self._asset_index and hasattr(self._asset_index, "projects"):
            project_name = self._asset_index.projects.get(
                asset.get("project_id"), {}
            ).get("name", "")
        if self._asset_index and hasattr(self._asset_index, "scenes"):
            scene_name = self._asset_index.scenes.get(asset.get("scene_id"), {}).get(
                "name", ""
            )
        if self._asset_index and hasattr(self._asset_index, "runs"):
            run_name = self._asset_index.runs.get(asset.get("run_id"), {}).get(
                "name", ""
            )

        return project_name, scene_name, run_name

    def _get_asset_display_fields(
        self,
        asset: Dict[str, Any],
        project_name: str,
        scene_name: str,
        run_name: str,
    ) -> Dict[str, str]:
        asset_name = asset.get("name", "Unnamed")
        role_label = asset.get("role", "").replace("_", " ").title()
        display_name = scene_name or asset_name or "Unnamed"

        if asset_name and asset_name != display_name:
            display_subtitle = asset_name
        elif run_name:
            display_subtitle = run_name
        elif project_name:
            display_subtitle = project_name
        else:
            display_subtitle = role_label

        context_parts = []
        if project_name and project_name != display_subtitle:
            context_parts.append(project_name)
        if run_name and run_name not in (display_name, display_subtitle):
            context_parts.append(run_name)

        context_label = " / ".join(context_parts)
        if role_label:
            context_label = (
                f"{context_label} - {role_label}" if context_label else role_label
            )

        return {
            "display_name": display_name,
            "display_subtitle": display_subtitle,
            "context_label": context_label,
        }

    def get_filtered_assets(self) -> List[Dict[str, Any]]:
        """Return assets filtered by search query, active filter, and selections."""
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return []

        assets = []
        for _asset_id, asset in self._asset_index.assets.items():
            if self._selected_project_id and asset.get("project_id") != self._selected_project_id:
                continue
            if self._selected_scene_id and asset.get("scene_id") != self._selected_scene_id:
                continue

            if self._active_filter.startswith("tag:"):
                tag_name = self._active_filter.split(":", 1)[1]
                if tag_name not in asset.get("tags", []):
                    continue
            elif self._active_filter.startswith("collection:"):
                collection_id = self._active_filter.split(":", 1)[1]
                if collection_id not in asset.get("collection_ids", []):
                    continue
            elif self._active_filter == "recent":
                if not self._is_recent_asset(asset):
                    continue
            elif self._active_filter == "splat":
                if asset.get("type") not in ("ply", "rad", "sog", "spz"):
                    continue
            elif self._active_filter == "video":
                if asset.get("type") not in ("mp4", "mov", "video"):
                    continue
            elif self._active_filter == "checkpoint":
                if asset.get("type") != "checkpoint":
                    continue
            elif self._active_filter == "dataset":
                if asset.get("type") != "dataset" and asset.get("role") != "source_dataset":
                    continue
            elif self._active_filter == "trained":
                if asset.get("role") != "trained_output":
                    continue
            elif self._active_filter == "favorites":
                if not asset.get("is_favorite", False):
                    continue
            elif self._active_filter == "missing":
                if asset.get("exists", True):
                    continue

            if self._search_query and not self._asset_matches_query(asset, self._search_query):
                continue

            assets.append(self._format_asset_for_ui(asset))

        return self._sort_assets(assets)

    def _asset_matches_query(self, asset: Dict[str, Any], query: str) -> bool:
        query_l = query.strip().lower()
        if not query_l:
            return True

        project_name, scene_name, run_name = self._get_asset_relationship_names(asset)

        searchable = " ".join(
            [
                asset.get("name", ""),
                asset.get("type", ""),
                asset.get("role", ""),
                asset.get("path", ""),
                asset.get("absolute_path", ""),
                asset.get("notes", ""),
                " ".join(asset.get("tags", [])),
                project_name,
                scene_name,
                run_name,
            ]
        ).lower()
        return query_l in searchable

    def _is_recent_asset(self, asset: Dict[str, Any]) -> bool:
        try:
            from datetime import datetime, timedelta

            modified_at = asset.get("modified_at", "")
            if not modified_at:
                return False
            modified = datetime.fromisoformat(modified_at.replace("Z", "+00:00"))
            return modified >= datetime.now(modified.tzinfo) - timedelta(days=30)
        except Exception:
            return False

    def _sort_assets(self, assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort assets based on current sort mode."""
        if self._sort_mode == "name":
            return sorted(assets, key=lambda a: a.get("name", "").lower())
        if self._sort_mode == "size":
            return sorted(
                assets, key=lambda a: a.get("file_size_bytes", 0), reverse=True
            )
        if self._sort_mode == "type":
            return sorted(assets, key=lambda a: a.get("type", "").lower())
        return sorted(assets, key=lambda a: a.get("modified_at", ""), reverse=True)

    def _format_asset_for_ui(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        """Format asset data for UI display."""
        asset_id = asset.get("id", "")
        asset_type = asset.get("type", "")
        file_size_bytes = asset.get("file_size_bytes", 0)

        # Format size string
        size_str = self._format_size(file_size_bytes)

        # Get geometry metadata
        geom = asset.get("geometry_metadata", {}) or {}
        gaussian_count = geom.get("gaussian_count", 0)
        dataset_meta = asset.get("dataset_metadata", {}) or {}

        # Format gaussian count
        if asset_type == "dataset":
            image_count = dataset_meta.get("image_count", 0)
            if image_count >= 1_000_000:
                points_str = f"{image_count / 1_000_000:.2f}M images"
            elif image_count >= 1_000:
                points_str = f"{image_count / 1_000:.1f}K images"
            else:
                points_str = f"{image_count} images" if image_count else ""
        elif gaussian_count >= 1_000_000:
            points_str = f"{gaussian_count / 1_000_000:.2f}M Gaussians"
        elif gaussian_count >= 1_000:
            points_str = f"{gaussian_count / 1_000:.1f}K Gaussians"
        else:
            points_str = f"{gaussian_count} Gaussians" if gaussian_count else ""

        # Determine thumbnail class based on type
        thumb_classes = {
            "ply": "asset-thumb-splat",
            "rad": "asset-thumb-splat",
            "sog": "asset-thumb-splat",
            "spz": "asset-thumb-splat",
            "checkpoint": "asset-thumb-checkpoint",
            "mp4": "asset-thumb-video",
            "mov": "asset-thumb-video",
            "dataset": "asset-thumb-dataset",
        }
        thumb_class = thumb_classes.get(asset_type, "asset-thumb-default")

        project_name, scene_name, run_name = self._get_asset_relationship_names(asset)
        display_fields = self._get_asset_display_fields(
            asset, project_name, scene_name, run_name
        )

        return {
            "id": asset_id,
            "name": asset.get("name", "Unnamed"),
            "display_name": display_fields["display_name"],
            "display_subtitle": display_fields["display_subtitle"],
            "context_label": display_fields["context_label"],
            "type": asset_type,
            "role": asset.get("role", ""),
            "type_label": asset_type.upper() if asset_type else "",
            "role_label": asset.get("role", "").replace("_", " ").title(),
            "size_label": size_str,
            "file_size_bytes": file_size_bytes,
            "points_label": points_str,
            "gaussian_count": gaussian_count,
            "tags": asset.get("tags", []),
            "thumb_class": thumb_class,
            "thumb_label": asset_type.upper() if asset_type else "ASSET",
            "pill_class": f"asset-pill-{asset_type}" if asset_type else "",
            "is_favorite": asset.get("is_favorite", False),
            "is_selected": asset_id in self._selected_asset_ids,
            "exists": asset.get("exists", True),
            "status_label": "Missing" if not asset.get("exists", True) else "Available",
            "can_load": asset_type in self.LOADABLE_TYPES and asset.get("exists", True),
            "project_id": asset.get("project_id"),
            "scene_id": asset.get("scene_id"),
            "run_id": asset.get("run_id"),
            "project_name": project_name,
            "scene_name": scene_name,
            "run_name": run_name,
            "modified_at": asset.get("modified_at", ""),
            "modified_label": self._format_timestamp(asset.get("modified_at", "")),
            "thumbnail_path": asset.get("thumbnail_path"),
        }

    def get_project_list(self) -> List[Dict[str, Any]]:
        """Return list of projects with scene counts for UI."""
        if not self._asset_index or not hasattr(self._asset_index, "projects"):
            return []

        projects = []
        for project_id, project in self._asset_index.projects.items():
            scene_count = len(project.get("scene_ids", []))
            projects.append(
                {
                    "id": project_id,
                    "name": project.get("name", "Unnamed Project"),
                    "description": project.get("description", ""),
                    "scene_count": scene_count,
                    "is_selected": project_id == self._selected_project_id,
                    "thumbnail_asset_id": project.get("thumbnail_asset_id"),
                }
            )

        return sorted(projects, key=lambda p: p["name"].lower())

    def get_scene_list(self) -> List[Dict[str, Any]]:
        """Return list of scenes for selected project."""
        if not self._asset_index or not hasattr(self._asset_index, "scenes"):
            return []

        if not self._selected_project_id:
            return []

        scenes = []
        for scene_id, scene in self._asset_index.scenes.items():
            if scene.get("project_id") != self._selected_project_id:
                continue

            run_count = len(scene.get("run_ids", []))
            scenes.append(
                {
                    "id": scene_id,
                    "name": scene.get("name", "Unnamed Scene"),
                    "description": scene.get("description", ""),
                    "run_count": run_count,
                    "is_selected": scene_id == self._selected_scene_id,
                    "thumbnail_asset_id": scene.get("thumbnail_asset_id"),
                }
            )

        return sorted(scenes, key=lambda s: s["name"].lower())

    def get_filter_list(self) -> List[Dict[str, Any]]:
        """Return list of filter categories with counts."""
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return self._get_default_filters()

        assets = list(self._asset_index.assets.values())

        # Count by filter
        all_count = len(assets)
        splat_count = sum(
            1 for a in assets if a.get("type") in ("ply", "rad", "sog", "spz")
        )
        video_count = sum(1 for a in assets if a.get("type") in ("mp4", "mov", "video"))
        checkpoint_count = sum(1 for a in assets if a.get("type") == "checkpoint")
        dataset_count = sum(1 for a in assets if a.get("role") == "source_dataset")
        trained_count = sum(1 for a in assets if a.get("role") == "trained_output")
        fav_count = sum(1 for a in assets if a.get("is_favorite", False))
        missing_count = sum(1 for a in assets if not a.get("exists", True))

        # Recent count (last 30 days)
        import datetime

        cutoff = (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat()
        recent_count = sum(1 for a in assets if a.get("modified_at", "") > cutoff)

        filters = [
            {"id": "all", "label": "All Assets", "count": all_count},
            {"id": "recent", "label": "Recent", "count": recent_count},
            {"id": "splat", "label": "Splats", "count": splat_count},
            {"id": "video", "label": "Videos", "count": video_count},
            {"id": "checkpoint", "label": "Checkpoints", "count": checkpoint_count},
            {"id": "dataset", "label": "Datasets", "count": dataset_count},
            {"id": "trained", "label": "Trained Outputs", "count": trained_count},
            {"id": "favorites", "label": "Favorites", "count": fav_count},
            {"id": "missing", "label": "Missing Files", "count": missing_count},
        ]

        return filters

    def _get_default_filters(self) -> List[Dict[str, Any]]:
        """Return default filter list when backend unavailable."""
        return [
            {"id": "all", "label": "All Assets", "count": 0},
            {"id": "recent", "label": "Recent", "count": 0},
            {"id": "splat", "label": "Splats", "count": 0},
            {"id": "video", "label": "Videos", "count": 0},
            {"id": "checkpoint", "label": "Checkpoints", "count": 0},
            {"id": "dataset", "label": "Datasets", "count": 0},
            {"id": "trained", "label": "Trained Outputs", "count": 0},
            {"id": "favorites", "label": "Favorites", "count": 0},
            {"id": "missing", "label": "Missing Files", "count": 0},
        ]

    def get_tag_list(self) -> List[Dict[str, Any]]:
        """Return list of tags with counts."""
        if not self._asset_index or not hasattr(self._asset_index, "tags"):
            return []

        tags = []
        for tag_id, tag_data in self._asset_index.tags.items():
            tags.append(
                {
                    "id": f"tag:{tag_id}",
                    "label": tag_data.get("label", tag_id),
                    "count": tag_data.get("count", 0),
                    "is_selected": self._active_filter == f"tag:{tag_id}",
                }
            )

        return sorted(tags, key=lambda t: t["label"].lower())

    def get_collection_list(self) -> List[Dict[str, Any]]:
        """Return list of collections with counts."""
        if not self._asset_index or not hasattr(self._asset_index, "collections"):
            return []

        collections = []
        for coll_id, coll_data in self._asset_index.collections.items():
            collections.append(
                {
                    "id": f"collection:{coll_id}",
                    "label": coll_data.get("name", "Unnamed"),
                    "count": len(coll_data.get("asset_ids", [])),
                    "is_selected": self._active_filter == f"collection:{coll_id}",
                }
            )

        return sorted(collections, key=lambda c: c["label"].lower())

    def get_selected_asset_struct(self) -> Dict[str, Any]:
        """Return selected asset as a struct for RML data binding."""
        if not self._selected_asset_ids or len(self._selected_asset_ids) != 1:
            return self._get_empty_asset_struct()

        asset_id = list(self._selected_asset_ids)[0]
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return self._get_empty_asset_struct()

        asset = self._asset_index.assets.get(asset_id)
        if not asset:
            return self._get_empty_asset_struct()

        return self._build_asset_struct(asset)

    def _get_empty_asset_struct(self) -> Dict[str, Any]:
        """Return empty asset struct with default values."""
        return {
            "id": "",
            "name": "",
            "type": "",
            "role": "",
            "path": "",
            "size": "",
            "points": "",
            "resolution": "",
            "duration": "",
            "created": "",
            "modified": "",
            "is_favorite": False,
            "has_training_provenance": False,
            "source_dataset": "",
            "training_run_id": "",
            "iterations": "",
            "sh_degree": "",
            "train_time": "",
            "has_geometry_metadata": False,
            "bounding_box": "",
            "center": "",
            "scale": "",
            "file_missing": False,
            "expected_path": "",
            "preview_class": "asset-thumb-default",
            "preview_label": "Preview",
            "pill_class": "",
            "type_label": "",
        }

    def _build_asset_struct(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete asset struct from asset data."""
        asset_id = asset.get("id", "")
        asset_type = asset.get("type", "")
        file_path = asset.get("absolute_path") or asset.get("path", "")

        # Format timestamps
        created_at = asset.get("created_at", "")
        modified_at = asset.get("modified_at", "")
        created_str = self._format_timestamp(created_at) if created_at else ""
        modified_str = self._format_timestamp(modified_at) if modified_at else ""

        # Get geometry metadata
        geom = asset.get("geometry_metadata", {}) or {}
        gaussian_count = geom.get("gaussian_count", 0)

        # Format points
        if gaussian_count >= 1_000_000:
            points_str = f"{gaussian_count / 1_000_000:.2f}M"
        elif gaussian_count >= 1_000:
            points_str = f"{gaussian_count / 1_000:.1f}K"
        else:
            points_str = str(gaussian_count)

        # Format size
        file_size_bytes = asset.get("file_size_bytes", 0)
        if file_size_bytes >= 1024**3:
            size_str = f"{file_size_bytes / (1024**3):.2f} GB"
        elif file_size_bytes >= 1024**2:
            size_str = f"{file_size_bytes / (1024**2):.1f} MB"
        elif file_size_bytes >= 1024:
            size_str = f"{file_size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{file_size_bytes} B"

        # Check training provenance
        run_id = asset.get("run_id", "")
        has_training_provenance = bool(run_id)

        # Get training info
        iterations = ""
        sh_degree = ""
        train_time = ""
        source_dataset = ""

        if has_training_provenance and self._asset_index:
            run = getattr(self._asset_index, "runs", {}).get(run_id)
            if run:
                params = run.get("parameters", {})
                sh_degree = str(params.get("sh_degree", ""))
                iterations = str(
                    params.get(
                        "iterations",
                        run.get("metrics", {}).get("final_iteration", ""),
                    )
                )
                # Calculate training duration
                started = run.get("created_at", "")
                completed = run.get("completed_at", "") or run.get("modified_at", "")
                if started and completed:
                    train_time = self._format_duration(started, completed)

                # Find source dataset for the scene
                scene_id = run.get("scene_id")
                if scene_id and hasattr(self._asset_index, "scenes"):
                    scene = self._asset_index.scenes.get(scene_id)
                    if scene:
                        dataset_id = scene.get("dataset_asset_id")
                        if dataset_id and hasattr(self._asset_index, "assets"):
                            dataset_asset = self._asset_index.assets.get(dataset_id)
                            if dataset_asset:
                                source_dataset = dataset_asset.get("name", "")

        # Check geometry metadata
        has_geometry_metadata = bool(geom)
        bbox = geom.get("bounding_box", {})
        if bbox:
            min_val = bbox.get("min", [0, 0, 0])
            max_val = bbox.get("max", [0, 0, 0])
            bbox_str = f"[{min_val}, {max_val}]"
        else:
            bbox_str = ""

        center = geom.get("center", [0, 0, 0])
        center_str = (
            f"{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}" if center else ""
        )

        scale = geom.get("scale", 1.0)
        scale_str = f"{scale:.2f}" if scale else "1.0"

        # Check if file exists
        file_exists = asset.get("exists", True)
        file_missing = not file_exists

        # Preview class
        thumb_classes = {
            "ply": "asset-thumb-splat",
            "rad": "asset-thumb-splat",
            "sog": "asset-thumb-splat",
            "spz": "asset-thumb-splat",
            "checkpoint": "asset-thumb-checkpoint",
            "mp4": "asset-thumb-video",
            "mov": "asset-thumb-video",
            "video": "asset-thumb-video",
            "dataset": "asset-thumb-dataset",
        }
        preview_class = thumb_classes.get(asset_type, "asset-thumb-default")

        # Video resolution and duration
        resolution = ""
        duration = ""
        if asset_type in ("mp4", "mov", "video"):
            video_meta = asset.get("video_metadata", {}) or {}
            width = video_meta.get("width", 0)
            height = video_meta.get("height", 0)
            if width and height:
                resolution = f"{width}x{height}"
            duration_secs = video_meta.get("duration_seconds", 0)
            if duration_secs:
                mins = int(duration_secs // 60)
                secs = int(duration_secs % 60)
                duration = f"{mins:02d}:{secs:02d}"

        return {
            "id": asset_id,
            "name": asset.get("name", "Unnamed"),
            "type": asset_type.upper() if asset_type else "",
            "role": asset.get("role", "").replace("_", " ").title(),
            "path": file_path,
            "size": size_str,
            "points": points_str if gaussian_count > 0 else "",
            "resolution": resolution,
            "duration": duration,
            "created": created_str,
            "modified": modified_str,
            "is_favorite": asset.get("is_favorite", False),
            "has_training_provenance": has_training_provenance,
            "source_dataset": source_dataset,
            "training_run_id": run_id,
            "iterations": iterations,
            "sh_degree": sh_degree,
            "train_time": train_time,
            "has_geometry_metadata": has_geometry_metadata,
            "bounding_box": bbox_str,
            "center": center_str,
            "scale": scale_str,
            "file_missing": file_missing,
            "expected_path": file_path if file_missing else "",
            "preview_class": preview_class,
            "preview_label": asset_type.upper() if asset_type else "Asset",
            "pill_class": f"asset-pill-{asset_type}" if asset_type else "",
            "type_label": asset_type.upper() if asset_type else "",
        }

    def get_selected_run_struct(self) -> Dict[str, Any]:
        """Return selected run as a struct for RML data binding."""
        if not self._selected_run_id:
            return self._get_empty_run_struct()

        if not self._asset_index or not hasattr(self._asset_index, "runs"):
            return self._get_empty_run_struct()

        run = self._asset_index.runs.get(self._selected_run_id)
        if not run:
            return self._get_empty_run_struct()

        return self._build_run_struct(run)

    def _get_empty_run_struct(self) -> Dict[str, Any]:
        """Return empty run struct with default values."""
        return {
            "id": "",
            "name": "",
            "status": "",
            "started": "",
            "completed": "",
            "duration": "",
            "parameters": [],
            "artifacts": [],
        }

    def _build_run_struct(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete run struct from run data."""
        # Format timestamps
        started_at = run.get("created_at", "")
        completed_at = run.get("completed_at", "") or run.get("modified_at", "")
        started_str = self._format_timestamp(started_at) if started_at else ""
        completed_str = self._format_timestamp(completed_at) if completed_at else ""

        # Calculate duration
        duration_str = ""
        if started_at and completed_at:
            duration_str = self._format_duration(started_at, completed_at)

        # Build parameters list
        params = run.get("parameters", {})
        parameters_list = [
            {"name": k.replace("_", " ").title(), "value": str(v)}
            for k, v in params.items()
        ]

        # Build artifacts list
        artifact_ids = run.get("artifact_asset_ids", [])
        artifacts_list = []
        for artifact_id in artifact_ids:
            if hasattr(self._asset_index, "assets"):
                artifact = self._asset_index.assets.get(artifact_id)
                if artifact:
                    artifacts_list.append(
                        {
                            "id": artifact_id,
                            "name": artifact.get("name", "Unnamed"),
                            "type": artifact.get("type", "").upper(),
                        }
                    )

        return {
            "id": run.get("id", ""),
            "name": run.get("name", "Unnamed Run"),
            "status": run.get("status", "").capitalize(),
            "started": started_str,
            "completed": completed_str,
            "duration": duration_str,
            "parameters": parameters_list,
            "artifacts": artifacts_list,
        }

    def get_selected_scene_struct(self) -> Dict[str, Any]:
        """Return selected scene as a struct for RML data binding."""
        if not self._selected_scene_id:
            return self._get_empty_scene_struct()

        if not self._asset_index or not hasattr(self._asset_index, "scenes"):
            return self._get_empty_scene_struct()

        scene = self._asset_index.scenes.get(self._selected_scene_id)
        if not scene:
            return self._get_empty_scene_struct()

        return self._build_scene_struct(scene)

    def _get_empty_scene_struct(self) -> Dict[str, Any]:
        """Return empty scene struct with default values."""
        return {
            "id": "",
            "name": "",
            "project_name": "",
            "asset_count": 0,
            "created": "",
            "modified": "",
            "assets": [],
        }

    def _build_scene_struct(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete scene struct from scene data."""
        scene_id = scene.get("id", "")

        # Get project name
        project_id = scene.get("project_id", "")
        project_name = ""
        if project_id and self._asset_index and hasattr(self._asset_index, "projects"):
            project = self._asset_index.projects.get(project_id)
            if project:
                project_name = project.get("name", "")

        # Count assets in scene
        asset_count = 0
        scene_assets = []
        if self._asset_index and hasattr(self._asset_index, "assets"):
            for asset_id, asset in self._asset_index.assets.items():
                if asset.get("scene_id") == scene_id:
                    asset_count += 1
                    scene_assets.append(
                        {
                            "id": asset_id,
                            "name": asset.get("name", "Unnamed"),
                            "type": asset.get("type", "").upper(),
                        }
                    )

        # Format timestamps
        created_at = scene.get("created_at", "")
        modified_at = scene.get("modified_at", "")
        created_str = self._format_timestamp(created_at) if created_at else ""
        modified_str = self._format_timestamp(modified_at) if modified_at else ""

        return {
            "id": scene_id,
            "name": scene.get("name", "Unnamed Scene"),
            "project_name": project_name,
            "asset_count": asset_count,
            "created": created_str,
            "modified": modified_str,
            "assets": scene_assets,
        }

    def get_selected_project_struct(self) -> Dict[str, Any]:
        """Return selected project as a struct for RML data binding."""
        if not self._selected_project_id:
            return self._get_empty_project_struct()

        if not self._asset_index or not hasattr(self._asset_index, "projects"):
            return self._get_empty_project_struct()

        project = self._asset_index.projects.get(self._selected_project_id)
        if not project:
            return self._get_empty_project_struct()

        return self._build_project_struct(project)

    def _get_empty_project_struct(self) -> Dict[str, Any]:
        """Return empty project struct with default values."""
        return {
            "id": "",
            "name": "",
            "scene_count": 0,
            "total_assets": 0,
            "path": "",
            "created": "",
            "modified": "",
            "scenes": [],
        }

    def _build_project_struct(self, project: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete project struct from project data."""
        project_id = project.get("id", "")

        # Count scenes and assets
        scene_ids = project.get("scene_ids", [])
        scene_count = len(scene_ids)

        total_assets = 0
        project_scenes = []

        if self._asset_index:
            for scene_id in scene_ids:
                if hasattr(self._asset_index, "scenes"):
                    scene = self._asset_index.scenes.get(scene_id)
                    if scene:
                        # Count assets for this scene
                        scene_asset_count = 0
                        if hasattr(self._asset_index, "assets"):
                            for asset in self._asset_index.assets.values():
                                if asset.get("scene_id") == scene_id:
                                    scene_asset_count += 1
                                    total_assets += 1

                        project_scenes.append(
                            {
                                "id": scene_id,
                                "name": scene.get("name", "Unnamed Scene"),
                                "asset_count": scene_asset_count,
                            }
                        )

        # Format timestamps
        created_at = project.get("created_at", "")
        modified_at = project.get("modified_at", "")
        created_str = self._format_timestamp(created_at) if created_at else ""
        modified_str = self._format_timestamp(modified_at) if modified_at else ""

        # Get project path
        project_path = project.get("path", "")

        return {
            "id": project_id,
            "name": project.get("name", "Unnamed Project"),
            "scene_count": scene_count,
            "total_assets": total_assets,
            "path": project_path,
            "created": created_str,
            "modified": modified_str,
            "scenes": project_scenes,
        }

    # ── Flattened Selected Asset Getters ─────────────────────

    def _get_selected_asset(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected single asset, if any."""
        if not self._selected_asset_ids or len(self._selected_asset_ids) != 1:
            return None
        asset_id = list(self._selected_asset_ids)[0]
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return None
        return self._asset_index.assets.get(asset_id)

    def get_selected_asset_name(self) -> str:
        asset = self._get_selected_asset()
        return asset.get("name", "") if asset else ""

    def get_selected_asset_type(self) -> str:
        asset = self._get_selected_asset()
        asset_type = asset.get("type", "") if asset else ""
        return asset_type.upper() if asset_type else ""

    def get_selected_asset_role(self) -> str:
        asset = self._get_selected_asset()
        role = asset.get("role", "") if asset else ""
        return role.replace("_", " ").title() if role else ""

    def get_selected_asset_project_name(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        project_name, _scene_name, _run_name = self._get_asset_relationship_names(asset)
        return project_name

    def get_selected_asset_scene_name(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        _project_name, scene_name, _run_name = self._get_asset_relationship_names(asset)
        return scene_name

    def get_selected_asset_run_name(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        _project_name, _scene_name, run_name = self._get_asset_relationship_names(asset)
        return run_name

    def get_selected_asset_path(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        return asset.get("absolute_path") or asset.get("path", "")

    def get_selected_asset_size(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        return self._format_size(asset.get("file_size_bytes", 0))

    def get_selected_asset_points(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        geom = asset.get("geometry_metadata", {}) or {}
        gaussian_count = geom.get("gaussian_count", 0)
        if gaussian_count >= 1_000_000:
            return f"{gaussian_count / 1_000_000:.2f}M"
        elif gaussian_count >= 1_000:
            return f"{gaussian_count / 1_000:.1f}K"
        return str(gaussian_count) if gaussian_count > 0 else ""

    def get_selected_asset_resolution(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        asset_type = asset.get("type", "")
        if asset_type in ("mp4", "mov", "video"):
            video_meta = asset.get("video_metadata", {}) or {}
            width = video_meta.get("width", 0)
            height = video_meta.get("height", 0)
            if width and height:
                return f"{width}x{height}"
        return ""

    def get_selected_asset_duration(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        asset_type = asset.get("type", "")
        if asset_type in ("mp4", "mov", "video"):
            video_meta = asset.get("video_metadata", {}) or {}
            duration_secs = video_meta.get("duration_seconds", 0)
            if duration_secs:
                mins = int(duration_secs // 60)
                secs = int(duration_secs % 60)
                return f"{mins:02d}:{secs:02d}"
        return ""

    def get_selected_asset_created(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        created_at = asset.get("created_at", "")
        return self._format_timestamp(created_at) if created_at else ""

    def get_selected_asset_modified(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        modified_at = asset.get("modified_at", "")
        return self._format_timestamp(modified_at) if modified_at else ""

    def get_selected_asset_is_favorite(self) -> bool:
        asset = self._get_selected_asset()
        return asset.get("is_favorite", False) if asset else False

    def get_selected_asset_has_training_provenance(self) -> bool:
        asset = self._get_selected_asset()
        if not asset:
            return False
        return bool(asset.get("run_id", ""))

    def get_selected_asset_source_dataset(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return "--"
        run_id = asset.get("run_id", "")
        if not run_id or not self._asset_index:
            return "--"
        run = getattr(self._asset_index, "runs", {}).get(run_id)
        if not run:
            return "--"
        scene_id = run.get("scene_id")
        if not scene_id or not hasattr(self._asset_index, "scenes"):
            return "--"
        scene = self._asset_index.scenes.get(scene_id)
        if not scene:
            return "--"
        dataset_id = scene.get("dataset_asset_id")
        if not dataset_id or not hasattr(self._asset_index, "assets"):
            return "--"
        dataset_asset = self._asset_index.assets.get(dataset_id)
        return dataset_asset.get("name", "--") if dataset_asset else "--"

    def get_selected_asset_training_run_id(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return "--"
        run_id = asset.get("run_id", "")
        return run_id if run_id else "--"

    def get_selected_asset_iterations(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return "--"
        run_id = asset.get("run_id", "")
        if not run_id or not self._asset_index:
            return "--"
        run = getattr(self._asset_index, "runs", {}).get(run_id)
        if not run:
            return "--"
        params = run.get("parameters", {})
        iterations = params.get("iterations", run.get("metrics", {}).get("final_iteration", ""))
        return str(iterations) if iterations else "--"

    def get_selected_asset_sh_degree(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return "--"
        run_id = asset.get("run_id", "")
        if not run_id or not self._asset_index:
            return "--"
        run = getattr(self._asset_index, "runs", {}).get(run_id)
        if not run:
            return "--"
        params = run.get("parameters", {})
        sh_degree = params.get("sh_degree", "")
        return str(sh_degree) if sh_degree else "--"

    def get_selected_asset_train_time(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return "--"
        run_id = asset.get("run_id", "")
        if not run_id or not self._asset_index:
            return "--"
        run = getattr(self._asset_index, "runs", {}).get(run_id)
        if not run:
            return "--"
        started = run.get("created_at", "")
        completed = run.get("completed_at", "") or run.get("modified_at", "")
        if started and completed:
            return self._format_duration(started, completed)
        return "--"

    def get_selected_asset_optimizer(self) -> str:
        """Get optimizer name for selected asset's training run."""
        asset = self._get_selected_asset()
        if not asset:
            return "--"
        run_id = asset.get("run_id", "")
        if not run_id or not self._asset_index:
            return "--"
        run = getattr(self._asset_index, "runs", {}).get(run_id)
        if not run:
            return "--"
        params = run.get("parameters", {})
        optimizer = params.get("optimizer", params.get("opt_type", ""))
        return str(optimizer) if optimizer else "--"

    def get_selected_asset_learning_rate(self) -> str:
        """Get learning rate for selected asset's training run."""
        asset = self._get_selected_asset()
        if not asset:
            return "--"
        run_id = asset.get("run_id", "")
        if not run_id or not self._asset_index:
            return "--"
        run = getattr(self._asset_index, "runs", {}).get(run_id)
        if not run:
            return "--"
        params = run.get("parameters", {})
        lr = params.get("learning_rate", params.get("lr", ""))
        if isinstance(lr, (int, float)):
            return f"{lr:.6f}"
        return str(lr) if lr else "--"

    def get_selected_asset_batch_size(self) -> str:
        """Get batch size for selected asset's training run."""
        asset = self._get_selected_asset()
        if not asset:
            return "--"
        run_id = asset.get("run_id", "")
        if not run_id or not self._asset_index:
            return "--"
        run = getattr(self._asset_index, "runs", {}).get(run_id)
        if not run:
            return "--"
        params = run.get("parameters", {})
        batch_size = params.get("batch_size", params.get("bs", ""))
        return str(batch_size) if batch_size else "--"

    def get_selected_asset_has_geometry_metadata(self) -> bool:
        asset = self._get_selected_asset()
        if not asset:
            return False
        geom = asset.get("geometry_metadata", {}) or {}
        return bool(geom)

    def get_selected_asset_has_dataset_metadata(self) -> bool:
        asset = self._get_selected_asset()
        if not asset:
            return False
        dataset_meta = asset.get("dataset_metadata", {}) or {}
        return asset.get("type") == "dataset" or bool(dataset_meta)

    def get_selected_asset_dataset_image_count(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        dataset_meta = asset.get("dataset_metadata", {}) or {}
        image_count = dataset_meta.get("image_count", 0)
        return str(image_count) if image_count or asset.get("type") == "dataset" else ""

    def get_selected_asset_dataset_image_root(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        dataset_meta = asset.get("dataset_metadata", {}) or {}
        image_root = dataset_meta.get("image_root", "")
        return image_root or "."

    def get_selected_asset_dataset_masks(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        dataset_meta = asset.get("dataset_metadata", {}) or {}
        mask_count = dataset_meta.get("mask_count", 0)
        if mask_count:
            return f"Yes ({mask_count})"
        return "Yes" if dataset_meta.get("has_masks") else "No"

    def get_selected_asset_dataset_sparse_model(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        dataset_meta = asset.get("dataset_metadata", {}) or {}
        return "Yes" if dataset_meta.get("sparse_model") else "No"

    def get_selected_asset_dataset_camera_count(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        dataset_meta = asset.get("dataset_metadata", {}) or {}
        camera_count = dataset_meta.get("camera_count")
        if camera_count is None:
            return "--"
        return str(camera_count)

    def get_selected_asset_dataset_database(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        dataset_meta = asset.get("dataset_metadata", {}) or {}
        return "Yes" if dataset_meta.get("database_present") else "No"

    def get_selected_asset_bounding_box(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        geom = asset.get("geometry_metadata", {}) or {}
        bbox = geom.get("bounding_box", {})
        if bbox:
            min_val = bbox.get("min", [0, 0, 0])
            max_val = bbox.get("max", [0, 0, 0])
            return f"[{min_val}, {max_val}]"
        return ""

    def get_selected_asset_center(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        geom = asset.get("geometry_metadata", {}) or {}
        center = geom.get("center", [0, 0, 0])
        if center:
            return f"{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}"
        return ""

    def get_selected_asset_scale(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        geom = asset.get("geometry_metadata", {}) or {}
        scale = geom.get("scale", 1.0)
        return f"{scale:.2f}" if scale else "1.0"

    def get_selected_asset_file_missing(self) -> bool:
        asset = self._get_selected_asset()
        if not asset:
            return False
        return not asset.get("exists", True)

    def get_selected_asset_expected_path(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        file_exists = asset.get("exists", True)
        if file_exists:
            return ""
        return asset.get("absolute_path") or asset.get("path", "")

    def get_selected_asset_preview_class(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return "asset-thumb-default"
        asset_type = asset.get("type", "")
        thumb_classes = {
            "ply": "asset-thumb-splat",
            "rad": "asset-thumb-splat",
            "sog": "asset-thumb-splat",
            "spz": "asset-thumb-splat",
            "checkpoint": "asset-thumb-checkpoint",
            "mp4": "asset-thumb-video",
            "mov": "asset-thumb-video",
            "video": "asset-thumb-video",
            "dataset": "asset-thumb-dataset",
        }
        return thumb_classes.get(asset_type, "asset-thumb-default")

    def get_selected_asset_preview_label(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return "Asset"
        asset_type = asset.get("type", "")
        return asset_type.upper() if asset_type else "Asset"

    def get_selected_asset_pill_class(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        asset_type = asset.get("type", "")
        return f"asset-pill-{asset_type}" if asset_type else ""

    def get_selected_asset_type_label(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        asset_type = asset.get("type", "")
        return asset_type.upper() if asset_type else ""

    # ── Flattened Selected Run Getters ───────────────────────

    def _get_selected_run(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected run, if any."""
        if not self._selected_run_id:
            return None
        if not self._asset_index or not hasattr(self._asset_index, "runs"):
            return None
        return self._asset_index.runs.get(self._selected_run_id)

    def get_selected_run_name(self) -> str:
        run = self._get_selected_run()
        return run.get("name", "") if run else ""

    def get_selected_run_status(self) -> str:
        run = self._get_selected_run()
        status = run.get("status", "") if run else ""
        return status.capitalize() if status else ""

    def get_selected_run_started(self) -> str:
        run = self._get_selected_run()
        if not run:
            return ""
        started_at = run.get("created_at", "")
        return self._format_timestamp(started_at) if started_at else ""

    def get_selected_run_completed(self) -> str:
        run = self._get_selected_run()
        if not run:
            return ""
        completed_at = run.get("completed_at", "") or run.get("modified_at", "")
        return self._format_timestamp(completed_at) if completed_at else ""

    def get_selected_run_duration(self) -> str:
        run = self._get_selected_run()
        if not run:
            return ""
        started_at = run.get("created_at", "")
        completed_at = run.get("completed_at", "") or run.get("modified_at", "")
        if started_at and completed_at:
            return self._format_duration(started_at, completed_at)
        return ""

    # ── Flattened Selected Scene Getters ───────────────────────

    def _get_selected_scene(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected scene, if any."""
        if not self._selected_scene_id:
            return None
        if not self._asset_index or not hasattr(self._asset_index, "scenes"):
            return None
        return self._asset_index.scenes.get(self._selected_scene_id)

    def get_selected_scene_name(self) -> str:
        scene = self._get_selected_scene()
        return scene.get("name", "") if scene else ""

    def get_selected_scene_project_name(self) -> str:
        scene = self._get_selected_scene()
        if not scene:
            return ""
        project_id = scene.get("project_id", "")
        if not project_id or not self._asset_index:
            return ""
        project = getattr(self._asset_index, "projects", {}).get(project_id)
        return project.get("name", "") if project else ""

    def get_selected_scene_asset_count(self) -> int:
        scene = self._get_selected_scene()
        if not scene or not self._asset_index:
            return 0
        scene_id = scene.get("id", "")
        if not scene_id or not hasattr(self._asset_index, "assets"):
            return 0
        return sum(
            1
            for asset in self._asset_index.assets.values()
            if asset.get("scene_id") == scene_id
        )

    def get_selected_scene_created(self) -> str:
        scene = self._get_selected_scene()
        if not scene:
            return ""
        created_at = scene.get("created_at", "")
        return self._format_timestamp(created_at) if created_at else ""

    def get_selected_scene_modified(self) -> str:
        scene = self._get_selected_scene()
        if not scene:
            return ""
        modified_at = scene.get("modified_at", "")
        return self._format_timestamp(modified_at) if modified_at else ""

    # ── Flattened Selected Project Getters ─────────────────────

    def _get_selected_project(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected project, if any."""
        if not self._selected_project_id:
            return None
        if not self._asset_index or not hasattr(self._asset_index, "projects"):
            return None
        return self._asset_index.projects.get(self._selected_project_id)

    def get_selected_project_name(self) -> str:
        project = self._get_selected_project()
        return project.get("name", "") if project else ""

    def get_selected_project_scene_count(self) -> int:
        project = self._get_selected_project()
        if not project:
            return 0
        return len(project.get("scene_ids", []))

    def get_selected_project_total_assets(self) -> int:
        project = self._get_selected_project()
        if not project or not self._asset_index:
            return 0
        scene_ids = set(project.get("scene_ids", []))
        if not scene_ids or not hasattr(self._asset_index, "assets"):
            return 0
        return sum(
            1
            for asset in self._asset_index.assets.values()
            if asset.get("scene_id") in scene_ids
        )

    def get_selected_project_path(self) -> str:
        project = self._get_selected_project()
        return project.get("path", "") if project else ""

    def get_selected_project_created(self) -> str:
        project = self._get_selected_project()
        if not project:
            return ""
        created_at = project.get("created_at", "")
        return self._format_timestamp(created_at) if created_at else ""

    def get_selected_project_modified(self) -> str:
        project = self._get_selected_project()
        if not project:
            return ""
        modified_at = project.get("modified_at", "")
        return self._format_timestamp(modified_at) if modified_at else ""

    def _format_timestamp(self, timestamp: str) -> str:
        """Format ISO timestamp to readable string."""
        if not timestamp:
            return ""
        try:
            import datetime

            dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%b %d, %Y, %H:%M")
        except Exception:
            return timestamp

    def _format_duration(self, start: str, end: str) -> str:
        """Format duration between two ISO timestamps."""
        try:
            import datetime

            start_dt = datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = datetime.datetime.fromisoformat(end.replace("Z", "+00:00"))
            duration = end_dt - start_dt
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except Exception:
            return ""

    def get_selected_assets_info(self) -> Dict[str, Any]:
        """Return metadata about current selection for info panel."""
        if not self._selected_asset_ids:
            return {"type": "none", "assets": []}

        if len(self._selected_asset_ids) == 1:
            asset_id = list(self._selected_asset_ids)[0]
            if self._asset_index and hasattr(self._asset_index, "assets"):
                asset = self._asset_index.assets.get(asset_id)
                if asset:
                    return {
                        "type": "asset",
                        "asset": self._format_asset_for_ui(asset),
                    }

        return {
            "type": "multiple",
            "count": len(self._selected_asset_ids),
            "total_size": self.get_selected_total_size(),
        }

    def _ensure_import_project(self, default_name: str = "Imported Assets") -> Optional[str]:
        if self._selected_project_id:
            return self._selected_project_id
        if not self._asset_index:
            return None
        project = self._asset_index.find_or_create_project(default_name)
        self._selected_project_id = project.id
        return project.id

    def _metadata_to_asset_kwargs(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        format_specific = metadata.get("format_specific", {}) or {}
        asset_type = metadata.get("type") or "unknown"

        kwargs: Dict[str, Any] = {
            "file_size_bytes": metadata.get("size_bytes", 0),
            "created_at": metadata.get("created"),
            "modified_at": metadata.get("modified"),
        }

        if asset_type in ("ply", "rad", "sog", "spz"):
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

    def _generate_asset_thumbnail(self, asset: Any) -> None:
        if not self._asset_thumbnails or not asset:
            return
        try:
            thumb_path = self._asset_thumbnails.generate_placeholder(asset.type, asset.id)
            self._asset_index.update_asset(asset.id, thumbnail_path=str(thumb_path))
        except Exception as exc:
            _logger.debug(f"Failed to generate thumbnail for {asset.id}: {exc}")

    def _asset_needs_metadata_sync(self, asset: Dict[str, Any]) -> bool:
        asset_type = asset.get("type", "")
        file_path = asset.get("absolute_path") or asset.get("path", "")
        if not file_path or not os.path.exists(file_path):
            return False

        if asset_type == "dataset":
            dataset_meta = asset.get("dataset_metadata", {}) or {}
            return (
                asset.get("file_size_bytes", 0) <= 0
                or "image_count" not in dataset_meta
                or "mask_count" not in dataset_meta
                or "database_present" not in dataset_meta
                or "image_root" not in dataset_meta
            )

        if asset_type in ("ply", "rad", "sog", "spz"):
            return not (asset.get("geometry_metadata", {}) or {})
        if asset_type == "checkpoint":
            return not (asset.get("training_metadata", {}) or {})
        if asset_type in ("video", "mp4", "mov"):
            return not (asset.get("video_metadata", {}) or {})

        return asset.get("file_size_bytes", 0) <= 0

    def _sync_existing_asset_metadata(self) -> bool:
        if not self._asset_index or not self._asset_scanner:
            return False

        updated_any = False
        for asset_id, asset in list(self._asset_index.assets.items()):
            if not self._asset_needs_metadata_sync(asset):
                continue

            file_path = asset.get("absolute_path") or asset.get("path", "")
            try:
                metadata = self._asset_scanner.scan_file(file_path)
            except Exception as exc:
                _logger.debug(f"Failed to rescan asset metadata for {file_path}: {exc}")
                continue

            update_kwargs = self._metadata_to_asset_kwargs(metadata)
            size_bytes = metadata.get("size_bytes")
            if size_bytes is not None and size_bytes != asset.get("file_size_bytes", 0):
                update_kwargs["file_size_bytes"] = size_bytes

            modified_at = metadata.get("modified")
            if modified_at and modified_at != asset.get("modified_at"):
                update_kwargs["modified_at"] = modified_at

            created_at = metadata.get("created")
            if created_at and not asset.get("created_at"):
                update_kwargs["created_at"] = created_at

            if update_kwargs:
                self._asset_index.update_asset(asset_id, **update_kwargs)
                updated_any = True

        return updated_any

    def _scan_and_register_asset(
        self,
        path: str,
        *,
        project_id: Optional[str],
        scene_id: Optional[str],
        run_id: Optional[str] = None,
        fallback_role: str = "reference",
        override_type: Optional[str] = None,
        override_role: Optional[str] = None,
    ):
        metadata = self._asset_scanner.scan_file(path) if self._asset_scanner else {}
        asset_kwargs = self._metadata_to_asset_kwargs(metadata)
        asset_type = override_type or asset_kwargs.pop("type", None) or "unknown"
        role = override_role or asset_kwargs.pop("role", None) or fallback_role

        asset = self._asset_index.create_asset(
            project_id=project_id,
            name=Path(path).name,
            type=asset_type,
            path=path,
            absolute_path=path,
            scene_id=scene_id,
            run_id=run_id,
            role=role,
            **asset_kwargs,
        )
        if asset:
            self._generate_asset_thumbnail(asset)
        return asset

    # ── Event Handlers ────────────────────────────────────────

    def set_filter(self, _handle, _ev, args):
        """Set the active filter."""
        if not args:
            return
        filter_id = str(args[0])
        self._active_filter = filter_id
        self._dirty_model("active_filter", "assets", "asset_count", "tags", "collections")

    def set_tab(self, _handle, _ev, args):
        """Set the active info tab."""
        if not args:
            return
        tab_id = str(args[0])
        self._active_tab = tab_id
        self._dirty_model("active_tab")

    def set_view_mode(self, _handle, _ev, args):
        """Set the view mode (gallery or list)."""
        if not args:
            return
        mode = str(args[0])
        self._view_mode = mode
        self._dirty_model("view_mode", "is_gallery_view", "is_list_view", "assets")

    def cycle_sort_mode(self, _handle, _ev, args):
        """Cycle through supported sort modes."""
        try:
            current_index = self.SORT_MODES.index(self._sort_mode)
        except ValueError:
            current_index = 0
        self._sort_mode = self.SORT_MODES[(current_index + 1) % len(self.SORT_MODES)]
        self._dirty_model("sort_mode", "sort_label", "assets")

    def toggle_asset_selection(self, _handle, _ev, args):
        """Toggle selection state of an asset."""
        if not args:
            return
        asset_id = str(args[0])

        # Handle Ctrl/Cmd multi-select via args[1] if provided
        multi_select = len(args) > 1 and bool(args[1])

        if multi_select:
            if asset_id in self._selected_asset_ids:
                self._selected_asset_ids.remove(asset_id)
            else:
                self._selected_asset_ids.add(asset_id)
        else:
            if self._selected_asset_ids == {asset_id}:
                self._selected_asset_ids.clear()
            else:
                self._selected_asset_ids = {asset_id}

        self._update_selection_type()
        self._dirty_model(
            "assets",
            "selected_count",
            "selected_total_size",
            "selection_type",
            "selected_asset_name",
            "selected_asset_type",
            "selected_asset_role",
        )

    def _update_selection_type(self):
        """Update selection type based on current selection."""
        if not self._selected_asset_ids:
            self._selection_type = "none"
        elif len(self._selected_asset_ids) == 1:
            self._selection_type = "asset"
        else:
            self._selection_type = "multiple"

    def on_search(self, _handle, _ev, args):
        """Handle search input change."""
        if args and len(args) > 0:
            self._search_query = str(args[0])
        self._dirty_model("search_query", "assets", "asset_count")

    def on_pending_tag_change(self, _handle, _ev, args):
        """Update the pending tag input buffer."""
        self._pending_tag_name = str(args[0]) if args else ""
        self._dirty_model("pending_tag_name")

    def on_add_tag(self, _handle, _ev, args):
        """Add the pending tag to the currently selected asset."""
        asset = self._get_selected_asset()
        if not asset or not self._asset_index:
            return
        tag = self._pending_tag_name.strip()
        if not tag:
            return
        self._asset_index.add_tag_to_asset(asset["id"], tag)
        self._pending_tag_name = ""
        self.refresh_catalog()
        self._dirty_model("pending_tag_name", "tags", "assets", "selected_asset_tags")

    def on_remove_tag(self, _handle, _ev, args):
        """Remove a tag from the currently selected asset."""
        asset = self._get_selected_asset()
        if not asset or not self._asset_index or not args:
            return
        tag = str(args[0]).strip()
        if not tag:
            return
        self._asset_index.remove_tag_from_asset(asset["id"], tag)
        self.refresh_catalog()
        self._dirty_model("tags", "assets", "selected_asset_tags")

    def on_import_asset(self, _handle, _ev, args):
        """Import a single asset file."""
        if not self._asset_index:
            _logger.warning("Asset index not initialized")
            return

        # Open file dialog for Gaussian splat files (includes .ply, .sog, .spz, .usd)
        file_path = lf.ui.open_ply_file_dialog("")

        if not file_path:
            return

        try:
            project_id = self._ensure_import_project()
            asset = self._scan_and_register_asset(
                file_path,
                project_id=project_id,
                scene_id=self._selected_scene_id,
                fallback_role="reference",
            )
            self._import_menu_open = False

            # Refresh UI
            self.refresh_catalog()
            self._dirty_model("import_menu_open")

            if asset:
                _logger.info(f"Imported asset: {asset.name}")

        except Exception as e:
            _logger.error(f"Failed to import asset: {e}")

    def on_import_dataset(self, _handle, _ev, args):
        """Import a dataset folder."""
        if not self._asset_index:
            _logger.warning("Asset index not initialized")
            return

        # Open folder dialog for datasets
        folder_path = lf.ui.open_dataset_folder_dialog()

        if not folder_path:
            return

        try:
            # Validate dataset structure
            dataset_info = self._asset_scanner.validate_dataset(folder_path)

            if not dataset_info.get("is_valid", False):
                _logger.warning(f"Invalid dataset structure: {folder_path}")
                return

            # Create or select project/scene
            project_id = self._ensure_import_project("Imported Datasets")
            scene_id = self._selected_scene_id

            if not scene_id:
                # Create scene for dataset
                scene = self._asset_index.create_scene(
                    project_id=project_id, name=Path(folder_path).name
                )
                scene_id = scene.id
                self._selected_scene_id = scene_id

            asset = self._scan_and_register_asset(
                folder_path,
                project_id=project_id,
                scene_id=scene_id,
                fallback_role="source_dataset",
                override_type="dataset",
                override_role="source_dataset",
            )

            # Link dataset to scene
            if asset:
                self._asset_index.update_scene(scene_id, dataset_asset_id=asset.id)
            self._import_menu_open = False

            # Refresh UI
            self.refresh_catalog()
            self._dirty_model("import_menu_open")

            if asset:
                _logger.info(f"Imported dataset: {asset.name}")

        except Exception as e:
            _logger.error(f"Failed to import dataset: {e}")

    def on_import_folder(self, _handle, _ev, args):
        """Scan folder and import found assets."""
        if not self._asset_index:
            _logger.warning("Asset index not initialized")
            return

        # Open folder dialog
        folder_path = lf.ui.open_folder_dialog("Select Folder to Scan", "")

        if not folder_path:
            return

        try:
            # Scan folder for assets
            found_assets = self._asset_scanner.scan_folder(folder_path)

            if not found_assets:
                _logger.info(f"No assets found in: {folder_path}")
                return

            # Show confirmation dialog with found assets
            # For now, auto-import all found assets
            imported_count = 0
            for file_info in found_assets:
                try:
                    file_path = file_info["path"]
                    asset = self._scan_and_register_asset(
                        file_path,
                        project_id=self._ensure_import_project(),
                        scene_id=self._selected_scene_id,
                        fallback_role=file_info.get("role", "reference"),
                        override_type=file_info.get("type"),
                    )
                    if asset:
                        imported_count += 1
                except Exception as e:
                    _logger.warning(f"Failed to import {file_info.get('path')}: {e}")
            self._import_menu_open = False

            # Refresh UI
            self.refresh_catalog()
            self._dirty_model("import_menu_open")

            _logger.info(f"Imported {imported_count} assets from folder")

        except Exception as e:
            _logger.error(f"Failed to import folder: {e}")

    def on_load_selected(self, _handle, _ev, args):
        """Load selected asset(s) into the viewer."""
        if not self._selected_asset_ids:
            return

        for asset_id in self._selected_asset_ids:
            if not self._asset_index or not hasattr(self._asset_index, "assets"):
                continue

            asset = self._asset_index.assets.get(asset_id)
            if not asset:
                continue

            file_path = asset.get("absolute_path") or asset.get("path")
            if not file_path or not os.path.exists(file_path):
                _logger.warning(f"Asset file not found: {file_path}")
                continue

            try:
                if asset.get("type") not in self.LOADABLE_TYPES:
                    continue
                # Load based on asset type
                asset_type = asset.get("type", "")
                if asset_type == "dataset":
                    # Datasets need special loading with output path
                    output_path = asset.get("output_path") or str(
                        Path(file_path) / "output"
                    )
                    lf.load_file(
                        file_path,
                        is_dataset=True,
                        output_path=output_path,
                    )
                else:
                    # Regular mesh/splat file loading
                    lf.load_file(file_path)
                _logger.info(f"Loaded asset: {asset.get('name')}")
            except Exception as e:
                _logger.error(f"Failed to load asset {asset_id}: {e}")

    def on_remove_from_catalog(self, _handle, _ev, args):
        """Remove selected assets from catalog (not from disk)."""
        if not self._selected_asset_ids:
            return

        if not self._asset_index:
            return

        removed_count = 0
        for asset_id in list(self._selected_asset_ids):
            try:
                self._asset_index.remove_asset(asset_id)
                removed_count += 1
            except Exception as e:
                _logger.warning(f"Failed to remove asset {asset_id}: {e}")

        # Clear selection
        self._selected_asset_ids.clear()
        self._update_selection_type()

        # Save catalog
        self._asset_index.save()

        # Refresh UI
        self.refresh_catalog()

        _logger.info(f"Removed {removed_count} assets from catalog")

    def on_toggle_favorite(self, _handle, _ev, args):
        """Toggle favorite status of selected asset(s)."""
        if not args:
            return

        asset_id = str(args[0])

        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return

        asset = self._asset_index.assets.get(asset_id)
        if not asset:
            return

        new_state = not asset.get("is_favorite", False)

        try:
            self._asset_index.update_asset(asset_id, is_favorite=new_state)
            self._asset_index.save()
            self.refresh_catalog()
        except Exception as e:
            _logger.error(f"Failed to toggle favorite: {e}")

    def select_project(self, _handle, _ev, args):
        """Select a project to filter scenes and assets."""
        if not args:
            return
        project_id = str(args[0])

        self._selected_project_id = project_id if project_id != "all" else None
        self._selected_scene_id = None  # Clear scene selection when project changes
        self._selected_run_id = None
        self._selected_asset_ids.clear()
        self._selection_type = "project" if self._selected_project_id else "none"

        self._dirty_model(
            "projects",
            "scenes",
            "assets",
            "asset_count",
            "scene_count",
            "project_count",
            "selected_count",
            "selected_total_size",
            "selection_type",
            "selected_project_name",
            "selected_project_scene_count",
            "selected_project_total_assets",
            "selected_project_path",
            "selected_project_created",
            "selected_project_modified",
            "selected_project_scenes",
        )

    def select_scene(self, _handle, _ev, args):
        """Select a scene to filter assets."""
        if not args:
            return
        scene_id = str(args[0])

        self._selected_scene_id = scene_id if scene_id != "all" else None
        self._selected_run_id = None
        self._selected_asset_ids.clear()
        self._selection_type = "scene" if self._selected_scene_id else "none"

        self._dirty_model(
            "scenes",
            "assets",
            "asset_count",
            "selected_count",
            "selected_total_size",
            "selection_type",
            "selected_scene_name",
            "selected_scene_project_name",
            "selected_scene_asset_count",
            "selected_scene_created",
            "selected_scene_modified",
            "selected_scene_assets",
        )

    def toggle_import_menu(self, _handle, _ev, args):
        """Toggle the import dropdown menu."""
        self._import_menu_open = not self._import_menu_open
        self._dirty_model("import_menu_open")

    def on_import_checkpoint(self, _handle, _ev, args):
        """Import a checkpoint file."""
        if not self._asset_index:
            _logger.warning("Asset index not initialized")
            return

        # Open file dialog for checkpoint
        file_path = lf.ui.open_checkpoint_file_dialog()

        if not file_path:
            return

        try:
            asset = self._scan_and_register_asset(
                file_path,
                project_id=self._ensure_import_project(),
                scene_id=self._selected_scene_id,
                fallback_role="training_checkpoint",
                override_type="checkpoint",
                override_role="training_checkpoint",
            )
            self._import_menu_open = False

            # Refresh UI
            self.refresh_catalog()
            self._dirty_model("import_menu_open")

            if asset:
                _logger.info(f"Imported checkpoint: {asset.name}")

        except Exception as e:
            _logger.error(f"Failed to import checkpoint: {e}")

    def on_locate_file(self, _handle, _ev, args):
        """Open file dialog to locate missing file."""
        if not self._selected_asset_ids or len(self._selected_asset_ids) != 1:
            return

        asset_id = list(self._selected_asset_ids)[0]
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return

        asset = self._asset_index.assets.get(asset_id)
        if not asset:
            return

        # Open file dialog - use ply dialog as it supports multiple asset formats
        file_path = lf.ui.open_ply_file_dialog("")

        if not file_path:
            return

        try:
            # Update asset path
            self._asset_index.update_asset(
                asset_id,
                path=file_path,
                absolute_path=os.path.abspath(file_path),
                exists=True,
            )
            self.refresh_catalog()
            _logger.info(f"Updated asset path: {asset.get('name', 'unknown')}")
        except Exception as e:
            _logger.error(f"Failed to locate file: {e}")

    def select_artifact(self, _handle, _ev, args):
        """Select an artifact from a run."""
        if not args:
            return
        artifact_id = str(args[0])

        # Select the artifact as an asset
        self._selected_asset_ids = {artifact_id}
        self._selection_type = "asset"
        self._dirty_model(
            "assets",
            "selected_count",
            "selected_total_size",
            "selection_type",
            "selected_asset_name",
        )

    def select_asset_by_id(self, _handle, _ev, args):
        """Select an asset by ID."""
        if not args:
            return
        asset_id = str(args[0])

        self._selected_asset_ids = {asset_id}
        self._selection_type = "asset"
        self._dirty_model(
            "assets",
            "selected_count",
            "selected_total_size",
            "selection_type",
            "selected_asset_name",
        )

    def on_export_selected(self, _handle, _ev, args):
        """Export selected assets."""
        if not self._selected_asset_ids:
            return

        # TODO: Implement export dialog and logic
        _logger.info(f"Export requested for {len(self._selected_asset_ids)} assets")

    def on_load_asset(self, _handle, _ev, args):
        """Load a specific asset by ID into the viewer."""
        if not args:
            return
        asset_id = str(args[0])

        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            _logger.warning("Asset index not initialized")
            return

        asset = self._asset_index.assets.get(asset_id)
        if not asset:
            _logger.warning(f"Asset not found: {asset_id}")
            return
        if asset.get("type") not in self.LOADABLE_TYPES:
            _logger.warning(f"Asset type is not loadable: {asset.get('type')}")
            return

        file_path = asset.get("absolute_path") or asset.get("path")
        if not file_path or not os.path.exists(file_path):
            _logger.warning(f"Asset file not found: {file_path}")
            return

        try:
            # Load based on asset type
            asset_type = asset.get("type", "")
            if asset_type == "dataset":
                # Datasets need special loading with output path
                output_path = asset.get("output_path") or str(
                    Path(file_path) / "output"
                )
                lf.load_file(
                    file_path,
                    is_dataset=True,
                    output_path=output_path,
                )
            else:
                # Regular mesh/splat file loading
                lf.load_file(file_path)
            _logger.info(f"Loaded asset: {asset.get('name', 'unknown')}")

            # Select the loaded asset
            self._selected_asset_ids = {asset_id}
            self._selection_type = "asset"
            self._dirty_model(
                "assets",
                "selected_count",
                "selected_total_size",
                "selection_type",
                "selected_asset_name",
            )
        except Exception as e:
            _logger.error(f"Failed to load asset {asset_id}: {e}")

    def on_remove_asset(self, _handle, _ev, args):
        """Remove a specific asset from the catalog by ID."""
        if not args:
            return
        asset_id = str(args[0])

        if not self._asset_index:
            _logger.warning("Asset index not initialized")
            return

        try:
            # Remove asset from catalog
            if hasattr(self._asset_index, "delete_asset"):
                self._asset_index.delete_asset(asset_id)
            elif hasattr(self._asset_index, "remove_asset"):
                self._asset_index.remove_asset(asset_id)
            else:
                _logger.warning("Asset index does not support asset deletion")
                return

            # Remove from selection if selected
            if asset_id in self._selected_asset_ids:
                self._selected_asset_ids.discard(asset_id)
                self._update_selection_type()

            # Save catalog
            self._asset_index.save()

            # Refresh UI
            self.refresh_catalog()

            _logger.info(f"Removed asset from catalog: {asset_id}")
        except Exception as e:
            _logger.error(f"Failed to remove asset {asset_id}: {e}")

    # ── Lifecycle ─────────────────────────────────────────────

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc

        # Initialize backend
        backend_ok = self._initialize_backend()
        if not backend_ok:
            _logger.warning(
                "Asset Manager backend not available - running in limited mode"
            )

        # Load index
        if self._asset_index and hasattr(self._asset_index, "load"):
            try:
                self._asset_index.load()
                if self._sync_existing_asset_metadata() and self._asset_index.library_path.exists():
                    self._library_mtime = self._asset_index.library_path.stat().st_mtime
                if self._asset_index.library_path.exists():
                    self._library_mtime = self._asset_index.library_path.stat().st_mtime
            except Exception as e:
                _logger.warning(f"Failed to load asset index: {e}")

        # Initial refresh
        self._update_all_record_lists()

    def on_update(self, doc):
        """Periodic update - check for missing files."""
        if not self._asset_index:
            return False

        try:
            library_path = self._asset_index.library_path
            if library_path.exists():
                current_mtime = library_path.stat().st_mtime
                if current_mtime > self._library_mtime:
                    self._asset_index.load()
                    if self._sync_existing_asset_metadata() and library_path.exists():
                        current_mtime = library_path.stat().st_mtime
                    self._library_mtime = current_mtime
                    self.refresh_catalog()
                    return False

            if hasattr(self._asset_index, "mark_missing_files"):
                previous_missing = sum(
                    1 for asset in self._asset_index.assets.values() if not asset.get("exists", True)
                )
                current_missing, _total = self._asset_index.mark_missing_files()
                if current_missing != previous_missing:
                    if library_path.exists():
                        self._library_mtime = library_path.stat().st_mtime
                    self.refresh_catalog()
        except Exception:
            pass

        return False

    def on_unmount(self, doc):
        """Save index on unmount."""
        if self._asset_index and hasattr(self._asset_index, "save"):
            try:
                self._asset_index.save()
            except Exception as e:
                _logger.warning(f"Failed to save asset index: {e}")

        doc.remove_data_model("asset_manager")
        self._handle = None
        self._doc = None

    # ── Integration Hooks (Stubs) ─────────────────────────────

    def on_training_started(
        self, project_name: str, scene_name: str, parameters: Dict[str, Any]
    ) -> Optional[str]:
        """Called when training starts - create run entry.

        Returns:
            Run ID if created, None otherwise.
        """
        if not self._asset_index:
            return None

        try:
            # Create or get project
            project = self._asset_index.find_or_create_project(project_name)
            project_id = project.id

            # Create or get scene
            scene = self._asset_index.find_or_create_scene(project_id, scene_name)
            scene_id = scene.id

            # Create training run
            run = self._asset_index.create_run(
                project_id=project_id,
                scene_id=scene_id,
                name=f"Run {len(scene.run_ids) + 1}",
                status="training",
                parameters=parameters,
            )
            if not run:
                return None

            self._asset_index.save()

            # Update UI if panel is open
            self._selected_project_id = project_id
            self._selected_scene_id = scene_id
            self._selected_run_id = run.id
            self.refresh_catalog()

            return run.id

        except Exception as e:
            _logger.error(f"Failed to create training run entry: {e}")
            return None

    def on_checkpoint_saved(
        self, run_id: str, checkpoint_path: str, iteration: int
    ) -> Optional[str]:
        """Called when checkpoint is saved - add checkpoint asset.

        Returns:
            Asset ID if created, None otherwise.
        """
        if not self._asset_index:
            return None

        try:
            run = (
                self._asset_index.runs.get(run_id)
                if hasattr(self._asset_index, "runs")
                else None
            )
            if not run:
                return None

            asset = self._scan_and_register_asset(
                checkpoint_path,
                project_id=run.get("project_id"),
                scene_id=run.get("scene_id"),
                run_id=run_id,
                fallback_role="training_checkpoint",
                override_type="checkpoint",
                override_role="training_checkpoint",
            )

            if asset:
                training_metadata = dict(asset.training_metadata or {})
                training_metadata["iteration"] = iteration
                self._asset_index.update_asset(
                    asset.id,
                    training_metadata=training_metadata,
                )
                self._asset_index.save()

                # Refresh UI
                self.refresh_catalog()

                return asset.id
            return None

        except Exception as e:
            _logger.error(f"Failed to register checkpoint: {e}")
            return None

    def on_training_completed(
        self, run_id: str, metrics: Optional[Dict[str, Any]] = None
    ):
        """Called when training completes - update run status."""
        if not self._asset_index:
            return

        try:
            self._asset_index.update_run(
                run_id,
                status="completed",
                metrics=metrics or {},
                completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            self._asset_index.save()

            # Refresh UI
            self.refresh_catalog()

        except Exception as e:
            _logger.error(f"Failed to update training completion: {e}")

    def on_export_generated(
        self,
        file_path: str,
        export_type: str,
        project_id: Optional[str] = None,
        scene_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[str]:
        """Called when export is generated - register export asset.

        Args:
            file_path: Path to exported file
            export_type: Type of export (ply, rad, sog, spz, mp4, etc.)
            project_id: Optional associated project
            scene_id: Optional associated scene
            run_id: Optional associated training run

        Returns:
            Asset ID if created, None otherwise.
        """
        if not self._asset_index:
            return None

        try:
            # Determine role
            role = "export"
            if run_id:
                role = "trained_output"

            asset = self._scan_and_register_asset(
                project_id=project_id,
                path=file_path,
                scene_id=scene_id,
                run_id=run_id,
                fallback_role=role,
                override_type=export_type,
                override_role=role,
            )

            self._asset_index.save()

            # Refresh UI if panel is open
            self.refresh_catalog()

            return asset.id if asset else None

        except Exception as e:
            _logger.error(f"Failed to register export: {e}")
            return None

    # ── Helper Methods ─────────────────────────────────────────

    def refresh_catalog(self):
        """Refresh all catalog data in the UI."""
        self._reconcile_selection()
        self._update_all_record_lists()
        if self._handle:
            for field in (
                "asset_count",
                "project_count",
                "scene_count",
                "selected_count",
                "selected_total_size",
                "selected_count_text",
                "selected_total_text",
                "selection_type",
                "sort_label",
            ):
                self._handle.dirty(field)

    def _update_all_record_lists(self):
        """Update all record lists in the data model."""
        if not self._handle:
            return

        self._handle.update_record_list("projects", self.get_project_list())
        self._handle.update_record_list("scenes", self.get_scene_list())
        self._handle.update_record_list("filters", self.get_filter_list())
        self._handle.update_record_list("tags", self.get_tag_list())
        self._handle.update_record_list("collections", self.get_collection_list())
        self._handle.update_record_list("assets", self.get_filtered_assets())

        # Update selection-specific record lists
        self._update_selection_details()

    def _update_selection_details(self):
        """Update record lists for selected run, scene, and project."""
        if not self._handle or self._updating_selection_details:
            return
        self._updating_selection_details = True
        try:
            # Update selected run parameters and artifacts
            run = self._get_selected_run()
            if run:
                params = run.get("parameters", {})
                parameters_list = []
                param_order = [
                    ("iterations", "Iterations"),
                    ("sh_degree", "SH Degree"),
                    ("optimizer", "Optimizer"),
                    ("opt_type", "Optimizer Type"),
                    ("learning_rate", "Learning Rate"),
                    ("lr", "Learning Rate"),
                    ("batch_size", "Batch Size"),
                    ("bs", "Batch Size"),
                    ("resolution", "Resolution"),
                    ("densify_from_iter", "Densify From"),
                    ("densify_until_iter", "Densify Until"),
                    ("densification_interval", "Densify Interval"),
                    ("opacity_reset_interval", "Opacity Reset"),
                    ("lambda_dssim", "DSSIM Weight"),
                ]

                added_params = set()
                for key, label in param_order:
                    if key in params and key not in added_params:
                        value = params[key]
                        if isinstance(value, float):
                            value_str = f"{value:.6f}" if value < 0.001 else f"{value:.4f}"
                        else:
                            value_str = str(value)
                        parameters_list.append({"name": label, "value": value_str})
                        added_params.add(key)

                for k, v in params.items():
                    if k not in added_params:
                        parameters_list.append(
                            {"name": k.replace("_", " ").title(), "value": str(v)}
                        )

                self._handle.update_record_list("selected_run_parameters", parameters_list)

                artifact_ids = run.get("artifact_asset_ids", [])
                artifacts_list = []
                for artifact_id in artifact_ids:
                    if self._asset_index and hasattr(self._asset_index, "assets"):
                        artifact = self._asset_index.assets.get(artifact_id)
                        if artifact:
                            artifacts_list.append(
                                {
                                    "id": artifact_id,
                                    "name": artifact.get("name", "Unnamed"),
                                    "type": artifact.get("type", "").upper(),
                                }
                            )
                self._handle.update_record_list("selected_run_artifacts", artifacts_list)
            else:
                self._handle.update_record_list("selected_run_parameters", [])
                self._handle.update_record_list("selected_run_artifacts", [])

            scene = self._get_selected_scene()
            if scene:
                scene_id = scene.get("id", "")
                scene_assets = []
                if self._asset_index and hasattr(self._asset_index, "assets"):
                    for asset_id, asset in self._asset_index.assets.items():
                        if asset.get("scene_id") == scene_id:
                            scene_assets.append(
                                {
                                    "id": asset_id,
                                    "name": asset.get("name", "Unnamed"),
                                    "type": asset.get("type", "").upper(),
                                }
                            )
                self._handle.update_record_list("selected_scene_assets", scene_assets)
            else:
                self._handle.update_record_list("selected_scene_assets", [])

            project = self._get_selected_project()
            if project:
                scene_ids = project.get("scene_ids", [])
                project_scenes = []
                if self._asset_index and hasattr(self._asset_index, "scenes"):
                    for scene_id in scene_ids:
                        scene_data = self._asset_index.scenes.get(scene_id)
                        if scene_data:
                            scene_asset_count = 0
                            if hasattr(self._asset_index, "assets"):
                                for asset in self._asset_index.assets.values():
                                    if asset.get("scene_id") == scene_id:
                                        scene_asset_count += 1
                            project_scenes.append(
                                {
                                    "id": scene_id,
                                    "name": scene_data.get("name", "Unnamed Scene"),
                                    "asset_count": scene_asset_count,
                                }
                            )
                self._handle.update_record_list("selected_project_scenes", project_scenes)
            else:
                self._handle.update_record_list("selected_project_scenes", [])

            selected_asset = self._get_selected_asset()
            if selected_asset:
                self._handle.update_record_list(
                    "selected_asset_tags",
                    [{"value": tag} for tag in selected_asset.get("tags", [])],
                )
            else:
                self._handle.update_record_list("selected_asset_tags", [])

            if self._selection_type == "asset":
                for field in (
                    "selected_asset_name",
                    "selected_asset_type",
                    "selected_asset_role",
                    "selected_asset_project_name",
                    "selected_asset_scene_name",
                    "selected_asset_run_name",
                    "selected_asset_path",
                    "selected_asset_size",
                    "selected_asset_points",
                    "selected_asset_resolution",
                    "selected_asset_duration",
                    "selected_asset_created",
                    "selected_asset_modified",
                    "selected_asset_is_favorite",
                    "selected_asset_source_dataset",
                    "selected_asset_training_run_id",
                    "selected_asset_iterations",
                    "selected_asset_sh_degree",
                    "selected_asset_optimizer",
                    "selected_asset_learning_rate",
                    "selected_asset_batch_size",
                    "selected_asset_train_time",
                    "selected_asset_has_dataset_metadata",
                    "selected_asset_dataset_image_count",
                    "selected_asset_dataset_image_root",
                    "selected_asset_dataset_masks",
                    "selected_asset_dataset_sparse_model",
                    "selected_asset_dataset_camera_count",
                    "selected_asset_dataset_database",
                    "selected_asset_bounding_box",
                    "selected_asset_center",
                    "selected_asset_scale",
                    "selected_asset_file_missing",
                    "selected_asset_expected_path",
                    "selected_asset_preview_class",
                    "selected_asset_preview_label",
                    "selected_asset_pill_class",
                    "selected_asset_type_label",
                    "pending_tag_name",
                ):
                    self._handle.dirty(field)
        finally:
            self._updating_selection_details = False

    def _dirty_model(self, *fields):
        """Mark fields as dirty to trigger UI refresh."""
        if not self._handle:
            return

        if not fields:
            self._handle.dirty_all()
            self._update_all_record_lists()
            return

        # Check if any selection-related fields are being dirtied
        selection_fields = {
            "selection_type",
            "selected_asset",
            "selected_asset_name",
            "selected_asset_type",
            "selected_asset_role",
            "selected_asset_path",
            "selected_run",
            "selected_run_name",
            "selected_run_status",
            "selected_scene",
            "selected_scene_name",
            "selected_scene_project_name",
            "selected_scene_asset_count",
            "selected_scene_assets",
            "selected_project",
            "selected_project_name",
            "selected_project_scene_count",
            "selected_project_total_assets",
            "selected_project_scenes",
            "selected_asset_tags",
        }
        needs_selection_update = any(f in selection_fields for f in fields)

        for field in fields:
            self._handle.dirty(field)
            # Update record lists when they change
            if field in (
                "projects",
                "scenes",
                "filters",
                "tags",
                "collections",
                "assets",
                "selected_asset_tags",
            ):
                list_map = {
                    "projects": self.get_project_list,
                    "scenes": self.get_scene_list,
                    "filters": self.get_filter_list,
                    "tags": self.get_tag_list,
                    "collections": self.get_collection_list,
                    "assets": self.get_filtered_assets,
                    "selected_asset_tags": lambda: [
                        {"value": tag}
                        for tag in (self._get_selected_asset() or {}).get("tags", [])
                    ],
                }
                if field in list_map:
                    self._handle.update_record_list(field, list_map[field]())

        # Update selection-specific record lists if needed
        if needs_selection_update and not self._updating_selection_details:
            self._update_selection_details()
