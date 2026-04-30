# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager panel for browsing and managing Gaussian Splatting assets."""

import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

import lichtfeld as lf

from . import rml_widgets
from .asset_manager_integration import (
    backfill_scene_provenance,
    clear_active_asset_manager_panel,
    ensure_dataset_catalog_context,
    set_active_asset_manager_panel,
)
from .types import Panel

_logger = logging.getLogger(__name__)

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

def tr(key):
    result = lf.ui.tr(key)
    if result == key:
        # Strip prefix for fallback
        if key.startswith("asset_manager."):
            return key.split(".")[-1].replace("_", " ").title()
    return result

__lfs_panel_classes__ = ["AssetManagerPanel"]
__lfs_panel_ids__ = ["lfs.asset_manager"]


class AssetManagerPanel(Panel):
    """Floating Asset Manager window for browsing splats, videos, and exports."""

    SORT_MODES = ("recent", "name", "size", "type")
    LOADABLE_TYPES = {"ply_3dgs", "ply_pcl", "rad", "sog", "spz", "checkpoint", "dataset", "mesh", "usd"}

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
        self._active_filters: Set[str] = set()  # Multi-select: empty = show all
        self._active_tab: str = "info"  # info, parameters, history
        self._view_mode: str = "gallery"  # gallery, list
        self._sort_mode: str = "recent"  # recent, name, size, type
        self._search_query: str = ""
        self._pending_tag_name: str = ""

        # Track which asset has its dropdown menu open
        self._open_menu_asset_id: Optional[str] = None

        # Track which project has its dropdown menu open
        self._open_menu_project_id: Optional[str] = None

        # Selection type for info panel display
        self._selection_type: str = "none"  # none, asset, run, multiple

        # Import menu state
        self._import_menu_open: bool = False
        self._library_mtime: float = 0.0
        self._updating_selection_details: bool = False
        self._pending_transform_applications: List[Dict[str, Any]] = []

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
        model.bind_func("has_selection", lambda: self.get_selected_count() > 0)
        model.bind_func("has_multi_selection", lambda: self.get_selected_count() > 1)
        model.bind_func("has_loadable_selection", self.get_has_loadable_selection)
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
        model.bind_func("active_filters", self.get_active_filters)
        model.bind_func("active_tab", self.get_active_tab)
        model.bind_func("selection_type", self.get_selection_type)
        model.bind_func("show_selection_none", lambda: self._selection_type == "none")
        model.bind_func("show_selection_asset", lambda: self._selection_type == "asset")
        model.bind_func("show_selection_run", lambda: self._selection_type == "run")
        model.bind_func("show_selection_scene", lambda: self._selection_type == "scene")
        model.bind_func(
            "show_selection_project", lambda: self._selection_type == "project"
        )
        model.bind_func(
            "show_selection_multiple", lambda: self._selection_type == "multiple"
        )
        model.bind_func("show_info_tab", lambda: self._active_tab == "info")
        model.bind_func("show_parameters_tab", lambda: self._active_tab == "parameters")

        # Import menu state
        model.bind_func("import_menu_open", self.get_import_menu_open)

        # Move menu projects list (for hover submenu)
        model.bind_record_list("move_menu_projects")

        # Selected IDs for UI conditionals
        model.bind_func("selected_project_id", self.get_selected_project_id)
        model.bind_func("selected_scene_id", self.get_selected_scene_id)
        model.bind_func("selected_run_id", self.get_selected_run_id)

        # Selected asset properties (flattened bind_func pattern)
        model.bind_func("selected_asset_name", self.get_selected_asset_name)
        model.bind_func("selected_asset_type", self.get_selected_asset_type)
        model.bind_func("selected_asset_role", self.get_selected_asset_role)
        model.bind_func(
            "selected_asset_project_name", self.get_selected_asset_project_name
        )
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
        model.bind_func("selected_asset_has_runs", self.get_selected_asset_has_runs)
        model.bind_func(
            "selected_asset_show_no_training", self.get_selected_asset_show_no_training
        )
        model.bind_func(
            "selected_asset_source_dataset", self.get_selected_asset_source_dataset
        )
        model.bind_func("selected_asset_can_load", self.get_selected_asset_can_load)
        model.bind_func(
            "selected_asset_training_run_id", self.get_selected_asset_training_run_id
        )
        model.bind_func("selected_asset_iterations", self.get_selected_asset_iterations)
        model.bind_func("selected_asset_sh_degree", self.get_selected_asset_sh_degree)
        model.bind_func(
            "selected_asset_gaussian_count", self.get_selected_asset_gaussian_count
        )
        model.bind_func("selected_asset_image_count", self.get_selected_asset_image_count)
        model.bind_func("selected_asset_strategy", self.get_selected_asset_strategy)
        model.bind_func(
            "selected_asset_steps_scaler", self.get_selected_asset_steps_scaler
        )
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
            "selected_asset_dataset_initial_points",
            self.get_selected_asset_dataset_initial_points,
        )
        model.bind_func(
            "selected_asset_bounding_box", self.get_selected_asset_bounding_box
        )
        model.bind_func("selected_asset_center", self.get_selected_asset_center)
        model.bind_func("selected_asset_scale", self.get_selected_asset_scale)
        model.bind_func(
            "selected_asset_has_transform_metadata",
            self.get_selected_asset_has_transform_metadata,
        )
        model.bind_func(
            "selected_asset_transform_translation", self.get_selected_asset_transform_translation
        )
        model.bind_func(
            "selected_asset_transform_rotation", self.get_selected_asset_transform_rotation
        )
        model.bind_func(
            "selected_asset_transform_scaling", self.get_selected_asset_transform_scaling
        )
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

        # UI Labels (for i18n)
        model.bind_func("search_icon_label", lambda: tr("asset_manager.toolbar.search_icon"))
        model.bind_func("search_placeholder", lambda: tr("asset_manager.toolbar.search_placeholder"))
        model.bind_func("gallery_label", lambda: tr("asset_manager.toolbar.view_gallery"))
        model.bind_func("list_label", lambda: tr("asset_manager.toolbar.view_list"))
        model.bind_func("import_label", lambda: tr("asset_manager.toolbar.import"))
        model.bind_func("import_asset_label", lambda: tr("asset_manager.import_menu.import_asset"))
        model.bind_func("import_dataset_label", lambda: tr("asset_manager.import_menu.import_dataset"))
        model.bind_func("import_checkpoint_label", lambda: tr("asset_manager.import_menu.import_checkpoint"))
        model.bind_func("summary_assets_label", lambda: tr("asset_manager.summary.assets"))
        model.bind_func("summary_projects_label", lambda: tr("asset_manager.summary.projects"))
        model.bind_func("summary_scenes_label", lambda: tr("asset_manager.summary.scenes"))
        model.bind_func("summary_selected_label", lambda: tr("asset_manager.summary.selected"))
        model.bind_func("projects_title", lambda: tr("asset_manager.sidebar.projects"))
        model.bind_func("scenes_title", lambda: tr("asset_manager.sidebar.scenes"))
        model.bind_func("filters_title", lambda: tr("asset_manager.sidebar.filters"))
        model.bind_func("filter_hint", lambda: tr("asset_manager.sidebar.filter_hint"))
        model.bind_func("gallery_title", lambda: tr("asset_manager.toolbar.view_gallery"))
        model.bind_func("list_title", lambda: tr("asset_manager.toolbar.view_list"))
        model.bind_func("rename_project_label", lambda: tr("asset_manager.action.rename_project"))
        model.bind_func("delete_project_label", lambda: tr("asset_manager.action.delete_project"))
        model.bind_func("load_button_label", lambda: tr("asset_manager.action.load"))
        model.bind_func("rename_label", lambda: tr("asset_manager.action.rename"))
        model.bind_func("move_to_project_label", lambda: tr("asset_manager.action.move_to_project"))
        model.bind_func("new_project_label", lambda: tr("asset_manager.action.new_project"))
        model.bind_func("show_in_folder_label", lambda: tr("asset_manager.action.show_in_folder"))
        model.bind_func("remove_label", lambda: tr("asset_manager.action.remove"))
        model.bind_func("col_name_label", lambda: tr("asset_manager.property.name"))
        model.bind_func("col_type_label", lambda: tr("asset_manager.property.type"))
        model.bind_func("col_role_label", lambda: tr("asset_manager.property.role"))
        model.bind_func("col_project_label", lambda: tr("asset_manager.property.project"))
        model.bind_func("col_scene_label", lambda: tr("asset_manager.property.scene"))
        model.bind_func("col_run_label", lambda: tr("asset_manager.property.run"))
        model.bind_func("col_size_label", lambda: tr("asset_manager.property.size"))
        model.bind_func("col_modified_label", lambda: tr("asset_manager.property.modified"))
        model.bind_func("col_status_label", lambda: tr("asset_manager.property.status"))
        model.bind_func("info_tab_label", lambda: tr("asset_manager.info_panel.info"))
        model.bind_func("parameters_tab_label", lambda: tr("asset_manager.info_panel.parameters"))
        model.bind_func("select_item_hint", lambda: tr("asset_manager.status.select_item"))
        model.bind_func("asset_details_title", lambda: tr("asset_manager.info_panel.asset_details"))
        model.bind_func("prop_project_label", lambda: tr("asset_manager.property.project"))
        model.bind_func("prop_scene_label", lambda: tr("asset_manager.property.scene"))
        model.bind_func("prop_run_label", lambda: tr("asset_manager.property.run"))
        model.bind_func("prop_points_label", lambda: tr("asset_manager.property.points"))
        model.bind_func("prop_sh_degree_label", lambda: tr("asset_manager.property.sh_degree"))
        model.bind_func("prop_size_label", lambda: tr("asset_manager.property.size"))
        model.bind_func("prop_path_label", lambda: tr("asset_manager.property.path"))
        model.bind_func("prop_created_label", lambda: tr("asset_manager.property.created"))
        model.bind_func("prop_modified_label", lambda: tr("asset_manager.property.modified"))
        model.bind_func("prop_resolution_label", lambda: tr("asset_manager.property.resolution"))
        model.bind_func("prop_duration_label", lambda: tr("asset_manager.property.duration"))
        model.bind_func("dataset_details_title", lambda: tr("asset_manager.info_panel.dataset_details"))
        model.bind_func("prop_images_label", lambda: tr("asset_manager.property.images"))
        model.bind_func("prop_image_root_label", lambda: tr("asset_manager.property.image_root"))
        model.bind_func("prop_masks_label", lambda: tr("asset_manager.property.masks"))
        model.bind_func("prop_sparse_model_label", lambda: tr("asset_manager.property.sparse_model"))
        model.bind_func("prop_cameras_label", lambda: tr("asset_manager.property.cameras"))
        model.bind_func("prop_initial_points_label", lambda: tr("asset_manager.property.initial_points"))
        model.bind_func("tags_title", lambda: tr("asset_manager.sidebar.tags"))
        model.bind_func("remove_tag_label", lambda: tr("asset_manager.action.remove"))
        model.bind_func("add_tag_placeholder", lambda: tr("asset_manager.action.add_tag"))
        model.bind_func("add_tag_button_label", lambda: tr("asset_manager.action.add_tag"))
        model.bind_func("geometry_metadata_title", lambda: tr("asset_manager.info_panel.geometry_metadata"))
        model.bind_func("prop_bounding_box_label", lambda: tr("asset_manager.geometry.bounding_box"))
        model.bind_func("prop_center_label", lambda: tr("asset_manager.geometry.center"))
        model.bind_func("prop_scale_label", lambda: tr("asset_manager.geometry.scale"))
        model.bind_func("transform_info_title", lambda: tr("asset_manager.info_panel.transform_information"))
        model.bind_func("prop_translation_label", lambda: tr("asset_manager.property.translation"))
        model.bind_func("prop_rotation_label", lambda: tr("asset_manager.property.rotation"))
        model.bind_func("prop_scaling_label", lambda: tr("asset_manager.property.scaling"))
        model.bind_func("file_not_found_title", lambda: tr("asset_manager.info_panel.file_not_found"))
        model.bind_func("prop_expected_path_label", lambda: tr("asset_manager.property.expected_path"))
        model.bind_func("locate_file_button_label", lambda: tr("asset_manager.action.locate_file"))
        model.bind_func("training_runs_title", lambda: tr("asset_manager.info_panel.training_runs"))
        model.bind_func("training_summary_title", lambda: tr("asset_manager.info_panel.training_summary"))
        model.bind_func("prop_gaussians_label", lambda: tr("asset_manager.property.gaussians"))
        model.bind_func("prop_strategy_label", lambda: tr("asset_manager.property.strategy"))
        model.bind_func("prop_steps_scaler_label", lambda: tr("asset_manager.property.steps_scaler"))
        model.bind_func("prop_iterations_label", lambda: tr("asset_manager.training.iterations"))
        model.bind_func("no_training_provenance_msg", lambda: tr("asset_manager.info_panel.no_training_provenance"))
        model.bind_func("run_pill_label", lambda: tr("asset_manager.type.run"))
        model.bind_func("run_details_title", lambda: tr("asset_manager.info_panel.run_details"))
        model.bind_func("prop_status_label", lambda: tr("asset_manager.property.status"))
        model.bind_func("prop_started_label", lambda: tr("asset_manager.property.started"))
        model.bind_func("prop_completed_label", lambda: tr("asset_manager.property.completed"))
        model.bind_func("artifacts_title", lambda: tr("asset_manager.info_panel.artifacts"))
        model.bind_func("scene_pill_label", lambda: tr("asset_manager.type.scene"))
        model.bind_func("scene_details_title", lambda: tr("asset_manager.info_panel.scene_details"))
        model.bind_func("prop_assets_label", lambda: tr("asset_manager.property.assets"))
        model.bind_func("scene_assets_title", lambda: tr("asset_manager.info_panel.scenes"))
        model.bind_func("project_pill_label", lambda: tr("asset_manager.type.project"))
        model.bind_func("project_details_title", lambda: tr("asset_manager.info_panel.project_details"))
        model.bind_func("prop_scenes_label", lambda: tr("asset_manager.property.scenes"))
        model.bind_func("prop_total_assets_label", lambda: tr("asset_manager.summary.total"))
        model.bind_func("scenes_list_title", lambda: tr("asset_manager.sidebar.scenes"))
        model.bind_func("multi_select_hint", lambda: tr("asset_manager.hint.multi_select"))

        # Record lists for data-for loops (main lists)
        model.bind_record_list("projects")
        model.bind_record_list("scenes")
        model.bind_record_list("filters")
        model.bind_record_list("tags")
        model.bind_record_list("collections")
        model.bind_record_list("assets")
        model.bind_record_list("selected_asset_tags")
        model.bind_record_list("selected_asset_runs")

        # Record lists for nested struct lists
        model.bind_record_list("selected_run_parameters")
        model.bind_record_list("selected_run_artifacts")
        model.bind_record_list("selected_scene_assets")
        model.bind_record_list("selected_project_scenes")

        self._handle = model.get_handle()

        # Initialize record lists
        self._update_all_record_lists()

        # Event handlers
        model.bind_event("toggle_filter", self.toggle_filter)
        model.bind_event("set_tab", self.set_tab)
        model.bind_event("set_view_mode", self.set_view_mode)
        model.bind_event("cycle_sort_mode", self.cycle_sort_mode)
        model.bind_event("toggle_asset_selection", self.toggle_asset_selection)
        model.bind_event("on_search", self.on_search)
        model.bind_event("on_import_asset", self.on_import_asset)
        model.bind_event("on_import_dataset", self.on_import_dataset)
        model.bind_event("on_load_selected", self.on_load_selected)
        model.bind_event("on_remove_from_catalog", self.on_remove_from_catalog)
        model.bind_event("on_toggle_favorite", self.on_toggle_favorite)
        model.bind_event("select_project", self.select_project)
        model.bind_event("select_scene", self.select_scene)
        model.bind_event("select_run", self.select_run)
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
            return sum(
                1
                for project_id in self._asset_index.projects
                if self._project_has_content(project_id)
            )
        return 0

    def get_scene_count(self) -> int:
        if self._asset_index and hasattr(self._asset_index, "scenes"):
            return sum(
                1
                for scene_id, scene in self._asset_index.scenes.items()
                if self._scene_has_content(scene_id)
                and (
                    not self._selected_project_id
                    or scene.get("project_id") == self._selected_project_id
                )
            )
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

    def get_has_loadable_selection(self) -> bool:
        if not self._selected_asset_ids or not self._asset_index:
            return False
        assets = getattr(self._asset_index, "assets", {})
        for asset_id in self._selected_asset_ids:
            asset = assets.get(asset_id)
            if (
                asset
                and asset.get("exists", True)
                and asset.get("type") in self.LOADABLE_TYPES
            ):
                return True
        return False

    def get_view_mode(self) -> str:
        return self._view_mode

    def get_sort_mode(self) -> str:
        return self._sort_mode

    def get_sort_label(self) -> str:
        labels = {
            "recent": tr("asset_manager.toolbar.sort_by_recent"),
            "name": tr("asset_manager.toolbar.sort_by_name"),
            "size": tr("asset_manager.toolbar.sort_by_size"),
            "type": tr("asset_manager.toolbar.sort_by_type"),
        }
        return labels.get(self._sort_mode, tr("asset_manager.toolbar.sort_by_recent"))

    def get_active_filters(self) -> Set[str]:
        return self._active_filters

    def get_active_tab(self) -> str:
        return self._active_tab

    def get_selection_type(self) -> str:
        return self._selection_type

    def get_import_menu_open(self) -> bool:
        return self._import_menu_open

    def get_move_menu_projects(self) -> List[Dict[str, str]]:
        """Get projects for the currently open move menu."""
        if not self._open_menu_asset_id or not self._asset_index:
            return []

        asset = self._asset_index.assets.get(self._open_menu_asset_id)
        if not asset:
            return []

        return self._get_available_projects_for_asset(asset)

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
            return f"{file_size_bytes / (1024**3):.2f} {tr('asset_manager.unit.gb')}"
        if file_size_bytes >= 1024**2:
            return f"{file_size_bytes / (1024**2):.1f} {tr('asset_manager.unit.mb')}"
        if file_size_bytes >= 1024:
            return f"{file_size_bytes / 1024:.1f} {tr('asset_manager.unit.kb')}"
        return f"{file_size_bytes} {tr('asset_manager.unit.b')}"

    def _ellipsize_path(self, path: str, max_chars: int = 56) -> str:
        if not path or len(path) <= max_chars:
            return path
        keep = max(8, (max_chars - 3) // 2)
        return f"{path[:keep]}...{path[-keep:]}"

    def _reconcile_selection(self) -> None:
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            self._selected_asset_ids.clear()
            self._selected_project_id = None
            self._selected_scene_id = None
            self._selected_run_id = None
            self._update_selection_type()
            return
        if (
            self._selected_project_id
            and self._selected_project_id
            not in getattr(self._asset_index, "projects", {})
        ):
            self._selected_project_id = None
        if (
            self._selected_scene_id
            and self._selected_scene_id not in getattr(self._asset_index, "scenes", {})
        ):
            self._selected_scene_id = None
        if (
            self._selected_run_id
            and self._selected_run_id not in getattr(self._asset_index, "runs", {})
        ):
            self._selected_run_id = None
        valid_ids = set(self._asset_index.assets.keys())
        if not self._selected_asset_ids.issubset(valid_ids):
            self._selected_asset_ids.intersection_update(valid_ids)
            self._update_selection_type()
        if not self._selected_asset_ids:
            if self._selection_type == "run" and not self._selected_run_id:
                self._selection_type = "none"
            elif self._selection_type == "scene" and not self._selected_scene_id:
                self._selection_type = "none"
            elif self._selection_type == "project" and not self._selected_project_id:
                self._selection_type = "none"

    def _scene_asset_count(self, scene_id: str) -> int:
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return 0
        return sum(
            1
            for asset in self._asset_index.assets.values()
            if asset.get("scene_id") == scene_id
        )

    def _scene_run_count(self, scene_id: str) -> int:
        if not self._asset_index or not hasattr(self._asset_index, "runs"):
            return 0
        return sum(
            1
            for run in self._asset_index.runs.values()
            if run.get("scene_id") == scene_id
        )

    def _scene_has_content(self, scene_id: str) -> bool:
        return self._scene_asset_count(scene_id) > 0 or self._scene_run_count(scene_id) > 0

    def _project_asset_count(self, project_id: str) -> int:
        """Count total assets in a project."""
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return 0
        return sum(
            1
            for asset in self._asset_index.assets.values()
            if asset.get("project_id") == project_id
        )

    def _project_has_content(self, project_id: str) -> bool:
        if not self._asset_index:
            return False
        if hasattr(self._asset_index, "assets") and any(
            asset.get("project_id") == project_id
            for asset in self._asset_index.assets.values()
        ):
            return True
        if not hasattr(self._asset_index, "scenes"):
            return False
        return any(
            scene.get("project_id") == project_id and self._scene_has_content(scene_id)
            for scene_id, scene in self._asset_index.scenes.items()
        )

    def _ensure_default_project(self) -> None:
        """Ensure a 'Default' project always exists."""
        if not self._asset_index or not hasattr(self._asset_index, "projects"):
            return

        # Check if Default project exists
        default_project_name = tr("asset_manager.default_project_name")
        has_default = any(
            proj.get("name") == default_project_name
            for proj in self._asset_index.projects.values()
        )

        if not has_default:
            try:
                self._asset_index.create_project(name=default_project_name)
                self._log_info(tr("asset_manager.msg.created_default"))
            except Exception as e:
                self._log_error(tr("asset_manager.msg.failed_create_default"), e)

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

    def _asset_display_title(self, asset: Dict[str, Any]) -> str:
        # Prioritize custom name if set by user
        custom_name = asset.get("name", "").strip()
        if custom_name:
            return custom_name

        # Fall back to filename from path
        file_path = asset.get("absolute_path") or asset.get("path") or ""
        if file_path:
            try:
                leaf = Path(os.path.normpath(file_path)).name
                if leaf:
                    return leaf
            except Exception:
                pass

        return tr("asset_manager.unnamed")

    def _get_asset_display_fields(
        self,
        asset: Dict[str, Any],
        project_name: str,
        scene_name: str,
        run_name: str,
    ) -> Dict[str, str]:
        asset_name = asset.get("name", "Unnamed")
        role_label = asset.get("role", "").replace("_", " ").title()
        display_name = self._asset_display_title(asset)

        if scene_name and scene_name != display_name:
            display_subtitle = scene_name
        elif run_name:
            display_subtitle = run_name
        elif project_name:
            display_subtitle = project_name
        elif asset_name and asset_name != display_name:
            display_subtitle = asset_name
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
            if (
                self._selected_project_id
                and asset.get("project_id") != self._selected_project_id
            ):
                continue
            if (
                self._selected_scene_id
                and asset.get("scene_id") != self._selected_scene_id
            ):
                continue

            # Multi-select filter logic: if any filters selected, asset must match at least one
            if self._active_filters:
                matches_filter = False

                # Splat filter: 3DGS PLY files, SOG files, and legacy PLY (Gaussian splats)
                if "splat" in self._active_filters:
                    if asset.get("type") in ("ply_3dgs", "sog", "ply"):
                        matches_filter = True

                # PCL filter: Regular point cloud PLY files
                if "pcl" in self._active_filters:
                    if asset.get("type") == "ply_pcl":
                        matches_filter = True

                # Dataset filter: source datasets
                if "dataset" in self._active_filters:
                    if asset.get("type") == "dataset" or asset.get("role") == "source_dataset":
                        matches_filter = True

                # Checkpoint filter: training checkpoints
                if "checkpoint" in self._active_filters:
                    if asset.get("type") == "checkpoint":
                        matches_filter = True

                if not matches_filter:
                    continue

            if self._search_query and not self._asset_matches_query(
                asset, self._search_query
            ):
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
            "ply_3dgs": "asset-thumb-splat",
            "ply_pcl": "asset-thumb-splat",
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

        # Format type label for display
        type_labels = {
            "ply_3dgs": tr("asset_manager.type.splat"),
            "ply_pcl": tr("asset_manager.type.pcl"),
            "ply": tr("asset_manager.type.splat"),  # Legacy PLY type
            "rad": tr("asset_manager.type.rad"),
            "sog": tr("asset_manager.type.sog"),
            "spz": tr("asset_manager.type.spz"),
            "checkpoint": tr("asset_manager.type.checkpoint"),
            "dataset": tr("asset_manager.type.dataset"),
            "mesh": tr("asset_manager.type.mesh"),
            "usd": tr("asset_manager.type.usd"),
            "mp4": tr("asset_manager.type.video"),
            "mov": tr("asset_manager.type.video"),
        }
        type_label = type_labels.get(asset_type, asset_type.upper() if asset_type else "")

        return {
            "id": asset_id,
            "name": asset.get("name", "Unnamed"),
            "display_name": display_fields["display_name"],
            "display_subtitle": display_fields["display_subtitle"],
            "context_label": display_fields["context_label"],
            "type": asset_type,
            "role": asset.get("role", ""),
            "type_label": type_label,
            "role_label": asset.get("role", "").replace("_", " ").title(),
            "size_label": size_str,
            "file_size_bytes": file_size_bytes,
            "points_label": points_str,
            "gaussian_count": gaussian_count,
            # Record-list rows only support scalar fields in the current RML bridge.
            "tags_label": ", ".join(asset.get("tags", [])) if asset.get("tags") else "",
            "thumb_class": thumb_class,
            "thumb_label": asset_type.upper() if asset_type else tr("asset_manager.type.asset"),
            "pill_class": f"asset-pill-{asset_type}" if asset_type else "",
            "is_favorite": asset.get("is_favorite", False),
            "is_selected": asset_id in self._selected_asset_ids,
            "exists": asset.get("exists", True),
            "status_label": tr("asset_manager.status.missing") if not asset.get("exists", True) else tr("asset_manager.status.available"),
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
            "menu_open": asset_id == self._open_menu_asset_id,
        }

    def get_project_list(self) -> List[Dict[str, Any]]:
        """Return list of projects with asset counts for UI."""
        if not self._asset_index or not hasattr(self._asset_index, "projects"):
            return []

        # Ensure Default project always exists
        self._ensure_default_project()

        projects = []
        for project_id, project in self._asset_index.projects.items():
            # Show all projects, even empty ones (user must manually delete)
            asset_count = self._project_asset_count(project_id)
            projects.append(
                {
                    "id": project_id,
                    "name": project.get("name", tr("asset_manager.unnamed_project")),
                    "description": project.get("description", ""),
                    "scene_count": asset_count,  # Now shows asset count instead of scene count
                    "is_selected": project_id == self._selected_project_id,
                    "thumbnail_asset_id": project.get("thumbnail_asset_id"),
                    "menu_open": project_id == self._open_menu_project_id,
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
            # Show all scenes, even empty ones (user must manually delete)
            run_count = self._scene_run_count(scene_id)
            asset_count = self._scene_asset_count(scene_id)
            scenes.append(
                {
                    "id": scene_id,
                    "name": scene.get("name", tr("asset_manager.unnamed_scene")),
                    "description": scene.get("description", ""),
                    "run_count": run_count,
                    "asset_count": asset_count,
                    "is_selected": scene_id == self._selected_scene_id,
                    "thumbnail_asset_id": scene.get("thumbnail_asset_id"),
                }
            )

        return sorted(scenes, key=lambda s: s["name"].lower())

    def get_filter_list(self) -> List[Dict[str, Any]]:
        """Return list of filter categories with counts (multi-select checkboxes)."""
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return self._get_default_filters()

        assets = list(self._asset_index.assets.values())

        # Count by filter (Splat, PCL, Dataset, Checkpoint)
        # Splat: 3DGS PLY files (ply_3dgs), SOG files, and legacy PLY
        splat_count = sum(1 for a in assets if a.get("type") in ("ply_3dgs", "sog", "ply"))
        # PCL: Regular point cloud PLY files (ply_pcl)
        pcl_count = sum(1 for a in assets if a.get("type") == "ply_pcl")
        checkpoint_count = sum(1 for a in assets if a.get("type") == "checkpoint")
        dataset_count = sum(
            1
            for a in assets
            if a.get("type") == "dataset" or a.get("role") == "source_dataset"
        )

        filters = [
            {
                "id": "splat",
                "label": "Splat",
                "count": splat_count,
                "is_selected": "splat" in self._active_filters,
            },
            {
                "id": "pcl",
                "label": "PCL",
                "count": pcl_count,
                "is_selected": "pcl" in self._active_filters,
            },
            {
                "id": "dataset",
                "label": "Dataset",
                "count": dataset_count,
                "is_selected": "dataset" in self._active_filters,
            },
            {
                "id": "checkpoint",
                "label": "Checkpoint",
                "count": checkpoint_count,
                "is_selected": "checkpoint" in self._active_filters,
            },
        ]

        return filters

    def _get_default_filters(self) -> List[Dict[str, Any]]:
        """Return default filter list when backend unavailable."""
        return [
            {"id": "splat", "label": tr("asset_manager.filter.splat"), "count": 0, "is_selected": False},
            {"id": "pcl", "label": tr("asset_manager.filter.pcl"), "count": 0, "is_selected": False},
            {"id": "dataset", "label": tr("asset_manager.filter.dataset"), "count": 0, "is_selected": False},
            {"id": "checkpoint", "label": tr("asset_manager.filter.checkpoint"), "count": 0, "is_selected": False},
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
                    "is_selected": f"tag:{tag_id}" in self._active_filters,
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
                    "label": coll_data.get("name", tr("asset_manager.unnamed")),
                    "count": len(coll_data.get("asset_ids", [])),
                    "is_selected": f"collection:{coll_id}" in self._active_filters,
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
            "preview_label": tr("asset_manager.preview"),
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
            "ply_3dgs": "asset-thumb-splat",
            "ply_pcl": "asset-thumb-splat",
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

        # Format type for display in info panel
        type_display_names = {
            "ply_3dgs": tr("asset_manager.type.gaussian_splat"),
            "ply_pcl": tr("asset_manager.type.point_cloud"),
            "ply": tr("asset_manager.type.gaussian_splat"),  # Legacy PLY type
            "rad": tr("asset_manager.type.rad"),
            "sog": tr("asset_manager.type.sog"),
            "spz": tr("asset_manager.type.spz"),
            "checkpoint": tr("asset_manager.type.checkpoint"),
            "dataset": tr("asset_manager.type.dataset"),
            "mesh": tr("asset_manager.type.mesh"),
            "usd": tr("asset_manager.type.usd"),
            "mp4": tr("asset_manager.type.video"),
            "mov": tr("asset_manager.type.video"),
        }
        type_display = type_display_names.get(asset_type, asset_type.upper() if asset_type else "")

        return {
            "id": asset_id,
            "name": asset.get("name", "Unnamed"),
            "type": type_display,
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
            "preview_label": type_display_names.get(asset_type, asset_type.upper() if asset_type else tr("asset_manager.type.asset")),
            "pill_class": f"asset-pill-{asset_type.replace('_', '-')}" if asset_type else "",
            "type_label": type_display_names.get(asset_type, asset_type.upper() if asset_type else ""),
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
            "name": run.get("name", tr("asset_manager.unnamed_run")),
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
            "name": scene.get("name", tr("asset_manager.unnamed_scene")),
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
                                "name": scene.get("name", tr("asset_manager.unnamed_scene")),
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
            "name": project.get("name", tr("asset_manager.unnamed_project")),
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
        return self._asset_display_title(asset) if asset else ""

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
        path = asset.get("absolute_path") or asset.get("path", "")
        return self._ellipsize_path(path)

    def get_selected_asset_size(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        return self._format_size(asset.get("file_size_bytes", 0))

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

    def get_selected_asset_has_runs(self) -> bool:
        return bool(self._runs_for_asset_context(self._get_selected_asset()))

    def get_selected_asset_show_no_training(self) -> bool:
        return (
            not self.get_selected_asset_has_training_provenance()
            and not self.get_selected_asset_has_runs()
        )

    def _get_training_run_for_asset(
        self, asset: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        asset = asset or self._get_selected_asset()
        if not asset or not self._asset_index:
            return None
        run_id = asset.get("run_id", "")
        if not run_id:
            return None
        return getattr(self._asset_index, "runs", {}).get(run_id)

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

    def get_selected_asset_can_load(self) -> bool:
        asset = self._get_selected_asset()
        if not asset:
            return False
        asset_type = asset.get("type", "")
        return asset_type in self.LOADABLE_TYPES

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
        run = self._get_training_run_for_asset(asset)
        if not run:
            return "--"
        params = run.get("parameters", {})
        iterations = params.get(
            "iterations", run.get("metrics", {}).get("final_iteration", "")
        )
        return str(iterations) if iterations else "--"

    def get_selected_asset_sh_degree(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        # Only show SH Degree for splat files, not regular point clouds
        asset_type = asset.get("type", "")
        if asset_type not in ("ply_3dgs", "sog", "spz", "rad"):
            return ""  # Not a splat file, don't show SH Degree
        # First check geometry metadata for Splat/PCL files
        geom = asset.get("geometry_metadata", {}) or {}
        sh_degree = geom.get("sh_degree")
        if sh_degree is not None:
            return str(sh_degree)
        # Fall back to training run parameters
        run = self._get_training_run_for_asset(asset)
        if not run:
            return ""
        params = run.get("parameters", {})
        sh_degree = params.get("sh_degree", "")
        return str(sh_degree) if sh_degree else ""

    def get_selected_asset_gaussian_count(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return "--"
        geom = asset.get("geometry_metadata", {}) or {}
        training_meta = asset.get("training_metadata", {}) or {}
        gaussian_count = (
            geom.get("gaussian_count")
            or geom.get("num_gaussians")
            or training_meta.get("num_gaussians")
        )
        run = self._get_training_run_for_asset(asset)
        if not gaussian_count and run:
            gaussian_count = run.get("metrics", {}).get("final_gaussians")
        if not gaussian_count:
            return "--"
        try:
            return f"{int(gaussian_count):,}"
        except (TypeError, ValueError):
            return str(gaussian_count)

    def get_selected_asset_image_count(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return "--"
        dataset_meta = asset.get("dataset_metadata", {}) or {}
        image_count = dataset_meta.get("image_count")
        run = self._get_training_run_for_asset(asset)
        if not image_count and run:
            image_count = run.get("parameters", {}).get("image_count")
            if not image_count and self._asset_index:
                source_id = run.get("source_dataset_id")
                source_asset = getattr(self._asset_index, "assets", {}).get(source_id)
                if source_asset:
                    image_count = (
                        source_asset.get("dataset_metadata", {}) or {}
                    ).get("image_count")
        if not image_count:
            return "--"
        try:
            return f"{int(image_count):,}"
        except (TypeError, ValueError):
            return str(image_count)

    def get_selected_asset_strategy(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return "--"
        run = self._get_training_run_for_asset(asset)
        if not run:
            return "--"
        strategy = run.get("parameters", {}).get("strategy", "")
        return str(strategy) if strategy else "--"

    def get_selected_asset_steps_scaler(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return "--"
        run = self._get_training_run_for_asset(asset)
        if not run:
            return "--"
        scaler = run.get("parameters", {}).get("steps_scaler", "")
        if scaler in ("", None):
            return "--"
        try:
            return f"{float(scaler):.2f}"
        except (TypeError, ValueError):
            return str(scaler)

    def get_selected_asset_train_time(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return "--"
        run = self._get_training_run_for_asset(asset)
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
        return str(mask_count)

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

    def get_selected_asset_dataset_initial_points(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        dataset_meta = asset.get("dataset_metadata", {}) or {}
        initial_points = dataset_meta.get("initial_points")
        if initial_points is None:
            return ""
        if initial_points >= 1_000_000:
            return f"{initial_points / 1_000_000:.2f}M"
        elif initial_points >= 1_000:
            return f"{initial_points / 1_000:.1f}K"
        return str(initial_points)

    def get_selected_asset_points(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        asset_type = asset.get("type", "")
        # For datasets, show initial points from COLMAP
        if asset_type == "dataset":
            dataset_meta = asset.get("dataset_metadata", {}) or {}
            points = dataset_meta.get("initial_points")
            if points is None:
                return ""
            if points >= 1_000_000:
                return f"{points / 1_000_000:.2f}M"
            elif points >= 1_000:
                return f"{points / 1_000:.1f}K"
            return str(points)
        # For geometry files (PLY, SOG, etc.), show gaussian count
        geom = asset.get("geometry_metadata", {}) or {}
        gaussian_count = geom.get("gaussian_count") or 0
        if gaussian_count >= 1_000_000:
            return f"{gaussian_count / 1_000_000:.2f}M"
        elif gaussian_count >= 1_000:
            return f"{gaussian_count / 1_000:.1f}K"
        return str(gaussian_count) if gaussian_count > 0 else ""

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

    def get_selected_asset_has_transform_metadata(self) -> bool:
        asset = self._get_selected_asset()
        if not asset:
            return False
        transform_meta = asset.get("transform_metadata", {}) or {}
        return bool(transform_meta)

    def get_selected_asset_transform_translation(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        transform_meta = asset.get("transform_metadata", {}) or {}
        translation = transform_meta.get("translation", [0.0, 0.0, 0.0])
        if translation and len(translation) >= 3:
            return f"{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}"
        return "0.000, 0.000, 0.000"

    def get_selected_asset_transform_rotation(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        transform_meta = asset.get("transform_metadata", {}) or {}
        # Prefer euler degrees if available
        euler_deg = transform_meta.get("rotation_euler_deg", [0.0, 0.0, 0.0])
        if euler_deg and len(euler_deg) >= 3:
            return f"{euler_deg[0]:.2f}°, {euler_deg[1]:.2f}°, {euler_deg[2]:.2f}°"
        # Fallback to quaternion
        quat = transform_meta.get("rotation_quat", [0.0, 0.0, 0.0, 1.0])
        if quat and len(quat) >= 4:
            return f"quat({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})"
        return "0.00°, 0.00°, 0.00°"

    def get_selected_asset_transform_scaling(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        transform_meta = asset.get("transform_metadata", {}) or {}
        scale = transform_meta.get("scale", [1.0, 1.0, 1.0])
        if scale and len(scale) >= 3:
            return f"{scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f}"
        return "1.000, 1.000, 1.000"

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
            "ply_3dgs": "asset-thumb-splat",
            "ply_pcl": "asset-thumb-splat",
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
        type_labels = {
            "ply_3dgs": "SPLAT",
            "ply_pcl": "PCL",
            "ply": "SPLAT",  # Legacy PLY type
            "rad": "RAD",
            "sog": "SOG",
            "spz": "SPZ",
            "checkpoint": "CKPT",
            "dataset": "DATASET",
            "mesh": "MESH",
            "usd": "USD",
            "mp4": "VIDEO",
            "mov": "VIDEO",
        }
        return type_labels.get(asset_type, asset_type.upper() if asset_type else "Asset")

    def get_selected_asset_pill_class(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        asset_type = asset.get("type", "")
        return f"asset-pill-{asset_type.replace('_', '-')}" if asset_type else ""

    def get_selected_asset_type_label(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        asset_type = asset.get("type", "")
        type_labels = {
            "ply_3dgs": "SPLAT",
            "ply_pcl": "PCL",
            "ply": "SPLAT",  # Legacy PLY type
            "rad": "RAD",
            "sog": "SOG",
            "spz": "SPZ",
            "checkpoint": "CKPT",
            "dataset": "DATASET",
            "mesh": "MESH",
            "usd": "USD",
            "mp4": "VIDEO",
            "mov": "VIDEO",
        }
        return type_labels.get(asset_type, asset_type.upper() if asset_type else "")

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

    def _runs_for_asset_context(
        self, asset: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not asset or not self._asset_index or not hasattr(self._asset_index, "runs"):
            return []

        runs = list(self._asset_index.runs.values())
        runs = [r for r in runs if r is not None]  # Filter out None values
        asset_id = asset.get("id")
        scene_id = asset.get("scene_id")
        run_id = asset.get("run_id")
        if run_id:
            return [run for run in runs if run.get("id") == run_id]

        if asset.get("type") == "dataset" or asset.get("role") == "source_dataset":
            return [
                run
                for run in runs
                if run.get("source_dataset_id") == asset_id
                or (scene_id and run.get("scene_id") == scene_id)
            ]

        if scene_id:
            return [run for run in runs if run.get("scene_id") == scene_id]
        return []

    def _run_parameter_value(self, run: Dict[str, Any], key: str):
        params = run.get("parameters", {}) or {}
        metrics = run.get("metrics", {}) or {}
        if key == "gaussians":
            return (
                metrics.get("final_gaussians")
                or params.get("max_gaussians")
                or params.get("max_cap")
            )
        if key == "images":
            image_count = params.get("image_count")
            if not image_count and self._asset_index:
                source_id = run.get("source_dataset_id")
                source_asset = getattr(self._asset_index, "assets", {}).get(source_id)
                if source_asset:
                    image_count = (
                        source_asset.get("dataset_metadata", {}) or {}
                    ).get("image_count")
            return image_count
        if key == "iterations":
            return params.get("iterations") or metrics.get("final_iteration")
        if key == "sh_degree":
            return params.get("sh_degree")
        if key == "strategy":
            return (
                params.get("strategy")
                or params.get("optimizer")
                or params.get("opt_type")
            )
        if key == "steps_scaler":
            return params.get("steps_scaler")
        return None

    def _format_compact_value(self, value, *, decimals: Optional[int] = None) -> str:
        if value in (None, ""):
            return "--"
        if decimals is not None:
            try:
                return f"{float(value):.{decimals}f}"
            except (TypeError, ValueError):
                return str(value)
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return str(value)

    def _format_run_parameters_for_ui(
        self, run: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        return [
            {
                "name": "Gaussians",
                "value": self._format_compact_value(
                    self._run_parameter_value(run, "gaussians")
                ),
            },
            {
                "name": "Images",
                "value": self._format_compact_value(
                    self._run_parameter_value(run, "images")
                ),
            },
            {
                "name": "SH Degree",
                "value": self._format_compact_value(
                    self._run_parameter_value(run, "sh_degree")
                ),
            },
            {
                "name": "Strategy",
                "value": self._format_compact_value(
                    self._run_parameter_value(run, "strategy")
                ),
            },
            {
                "name": "Steps Scaler",
                "value": self._format_compact_value(
                    self._run_parameter_value(run, "steps_scaler"), decimals=2
                ),
            },
            {
                "name": "Iterations",
                "value": self._format_compact_value(
                    self._run_parameter_value(run, "iterations")
                ),
            },
        ]

    def _format_run_for_ui(self, run: Dict[str, Any]) -> Dict[str, str]:
        details = [
            f"{self._format_compact_value(self._run_parameter_value(run, 'gaussians'))} gaussians",
            f"{self._format_compact_value(self._run_parameter_value(run, 'images'))} images",
            f"SH {self._format_compact_value(self._run_parameter_value(run, 'sh_degree'))}",
            f"{self._format_compact_value(self._run_parameter_value(run, 'strategy'))}",
            f"{self._format_compact_value(self._run_parameter_value(run, 'steps_scaler'), decimals=2)}x",
            f"{self._format_compact_value(self._run_parameter_value(run, 'iterations'))} iters",
        ]
        return {
            "id": run.get("id", ""),
            "name": run.get("name", "Run"),
            "status": run.get("status", "").capitalize(),
            "summary": " / ".join(details),
            "artifact_count": str(len(run.get("artifact_asset_ids", []) or [])),
        }

    def _ensure_import_project(
        self, default_name: str = "Default"
    ) -> Optional[str]:
        # Always import to "Default" project, never create a new one automatically
        if not self._asset_index:
            return None
        # Ensure Default project exists
        self._ensure_default_project()
        # Find the Default project
        for pid, proj in self._asset_index.projects.items():
            if proj.get("name") == "Default":
                self._selected_project_id = pid
                return pid
        return None

    def _metadata_to_asset_kwargs(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        format_specific = metadata.get("format_specific", {}) or {}
        asset_type = metadata.get("type") or "unknown"

        kwargs: Dict[str, Any] = {
            "type": asset_type,
            "file_size_bytes": metadata.get("size_bytes", 0),
            "created_at": metadata.get("created"),
            "modified_at": metadata.get("modified"),
        }

        if asset_type in ("ply_3dgs", "ply_pcl", "ply", "rad", "sog", "spz"):
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
            thumb_path = self._asset_thumbnails.generate_placeholder(
                asset.type, asset.id
            )
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

        if asset_type in ("ply_3dgs", "ply_pcl", "ply", "rad", "sog", "spz"):
            geom_meta = asset.get("geometry_metadata", {}) or {}
            # Need sync if empty or if gaussian_count is not present
            return not geom_meta or geom_meta.get("gaussian_count") is None
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
        # Always pop type and role from kwargs to avoid duplicate keyword argument error
        kwargs_type = asset_kwargs.pop("type", None)
        kwargs_role = asset_kwargs.pop("role", None)
        asset_type = override_type or kwargs_type or "unknown"
        role = override_role or kwargs_role or fallback_role

        # Final safety: ensure type and role are not in kwargs (they're passed explicitly)
        asset_kwargs.pop("type", None)
        asset_kwargs.pop("role", None)

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

    def _log_info(self, message: str, *args) -> None:
        if args:
            message = message % args
        try:
            lf.log.info(message)
        except Exception:
            _logger.info(message)

    def _log_warn(self, message: str, *args) -> None:
        if args:
            message = message % args
        try:
            lf.log.warn(message)
        except Exception:
            _logger.warning(message)

    def _log_error(self, message: str, *args) -> None:
        if args:
            message = message % args
        try:
            lf.log.error(message)
        except Exception:
            _logger.error(message)

    # ── Event Handlers ────────────────────────────────────────

    def toggle_filter(self, _handle, _ev, args):
        """Toggle a filter on/off (multi-select)."""
        if not args:
            return
        filter_id = str(args[0])

        # Toggle the filter in the set
        if filter_id in self._active_filters:
            self._active_filters.discard(filter_id)
        else:
            self._active_filters.add(filter_id)

        self._dirty_model(
            "active_filters", "filters", "assets", "asset_count"
        )

    def set_tab(self, _handle, _ev, args):
        """Set the active info tab."""
        if not args:
            return
        self._set_active_tab(str(args[0]))

    def _set_active_tab(self, tab_id: str) -> None:
        if tab_id not in ("info", "parameters"):
            return
        if self._active_tab == tab_id:
            return
        self._active_tab = tab_id
        self._dirty_model("active_tab", "show_info_tab", "show_parameters_tab")

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
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")

        # Handle Ctrl/Cmd multi-select via args[1] if provided
        multi_select = len(args) > 1 and bool(args[1])
        self._select_asset_id(
            asset_id,
            toggle=True,
            multi_select=multi_select,
        )

    def _selection_visibility_fields(self):
        return (
            "selection_type",
            "show_selection_none",
            "show_selection_asset",
            "show_selection_run",
            "show_selection_scene",
            "show_selection_project",
            "show_selection_multiple",
            "show_info_tab",
            "show_parameters_tab",
            "has_selection",
            "has_multi_selection",
            "has_loadable_selection",
        )

    def _select_asset_id(
        self,
        asset_id: str,
        *,
        toggle: bool = False,
        multi_select: bool = False,
    ) -> bool:
        if not asset_id:
            self._log_warn(
                "Asset Manager click ignored: no asset id resolved from event/DOM"
            )
            return False

        asset = None
        if self._asset_index and hasattr(self._asset_index, "assets"):
            asset = self._asset_index.assets.get(asset_id)
        if asset is None:
            available = []
            if self._asset_index and hasattr(self._asset_index, "assets"):
                available = list(self._asset_index.assets.keys())[:10]
            self._log_warn(
                "Asset Manager click resolved asset_id=%s but asset is missing "
                "from index. sample_ids=%s",
                asset_id,
                available,
            )
            return False

        self._selected_run_id = None

        if multi_select:
            if asset_id in self._selected_asset_ids:
                self._selected_asset_ids.remove(asset_id)
            else:
                self._selected_asset_ids.add(asset_id)
        elif toggle and self._selected_asset_ids == {asset_id}:
            self._selected_asset_ids.clear()
        else:
            self._selected_asset_ids = {asset_id}

        self._update_selection_type()
        self.refresh_catalog()
        return True

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
        """Import a single asset file (point clouds, splats, meshes, etc.)."""
        if not self._asset_index:
            _logger.warning("Asset index not initialized")
            return

        # Try point cloud/splat dialog first (PLY, SOG, SPZ, USD formats)
        file_path = lf.ui.open_ply_file_dialog("")

        # If user cancelled, try mesh dialog (OBJ, FBX, GLTF, etc.)
        if not file_path:
            file_path = lf.ui.open_mesh_file_dialog("")

        if not file_path:
            return

        try:
            project_id = self._ensure_import_project()

            # Detect asset type and role
            asset_type = None
            fallback_role = "reference"
            path_lower = file_path.lower()

            if path_lower.endswith(('.obj', '.fbx', '.gltf', '.glb', '.stl', '.dae', '.3ds')):
                asset_type = "mesh"
                fallback_role = "reference"
            elif path_lower.endswith('.ply'):
                # PLY files will be auto-detected as ply_3dgs or ply_pcl by the scanner
                asset_type = None  # Let scanner detect
                if 'point_cloud' in file_path.lower() or 'initial' in file_path.lower():
                    fallback_role = "initial_point_cloud"
                else:
                    fallback_role = "trained_output"
            elif path_lower.endswith(('.sog', '.spz')):
                asset_type = path_lower.split('.')[-1].replace('sog', 'sog').replace('spz', 'spz')
                if 'point_cloud' in file_path.lower() or 'initial' in file_path.lower():
                    fallback_role = "initial_point_cloud"
                else:
                    fallback_role = "trained_output"
            elif path_lower.endswith(('.usd', '.usda', '.usdc', '.usdz')):
                asset_type = "usd"
                fallback_role = "reference"

            asset = self._scan_and_register_asset(
                file_path,
                project_id=project_id,
                scene_id=self._selected_scene_id,
                fallback_role=fallback_role,
                override_type=asset_type,
            )
            self._import_menu_open = False

            # Auto-select the newly imported asset
            if asset:
                self._selected_asset_ids.add(asset.id)
                self._update_selection_type()

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

            context = ensure_dataset_catalog_context(
                folder_path,
                asset_index=self._asset_index,
                scanner=self._asset_scanner,
                thumbnails=self._asset_thumbnails,
            )
            project_id = context.get("project_id")
            scene_id = context.get("scene_id")
            asset_id = context.get("asset_id")
            asset = self._asset_index.get_asset(asset_id) if asset_id else None

            # Link dataset to scene
            if asset:
                # Auto-select the newly imported dataset to show its info
                # Add to selection instead of replacing (allow multiple imports)
                self._selected_asset_ids.add(asset.id)
                # Preserve user's existing project/scene filters - don't change them
                # The dataset will appear in the catalog based on current filters
                self._update_selection_type()
            self._import_menu_open = False

            # Refresh UI
            self.refresh_catalog()
            self._dirty_model("import_menu_open")
            self._update_selection_details()

            if asset:
                _logger.info(f"Imported dataset: {asset.name}")

        except Exception as e:
            _logger.error(f"Failed to import dataset: {e}")

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
                    transform_node_name = self._load_asset_with_hierarchy(file_path)
                    self._apply_asset_transform(asset, transform_node_name)
                _logger.info(f"Loaded asset: {asset.get('name')}")
            except Exception as e:
                _logger.error(f"Failed to load asset {asset_id}: {e}")

    def _delete_asset_from_catalog(self, asset_id: str) -> bool:
        if not self._asset_index:
            return False
        if hasattr(self._asset_index, "delete_asset"):
            return bool(self._asset_index.delete_asset(asset_id))
        if hasattr(self._asset_index, "remove_asset"):
            return bool(self._asset_index.remove_asset(asset_id))
        return False

    def on_remove_from_catalog(self, _handle, _ev, args):
        """Remove selected assets from catalog (not from disk)."""
        if not self._selected_asset_ids:
            return

        removed_count = 0
        for asset_id in list(self._selected_asset_ids):
            try:
                if self._delete_asset_from_catalog(asset_id):
                    removed_count += 1
            except Exception as e:
                _logger.warning(f"Failed to remove asset {asset_id}: {e}")

        # Clear selection
        self._selected_asset_ids.clear()
        self._update_selection_type()

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
        project_id = self._resolve_event_value(args, _ev, "data-project-id")
        self._select_project_id(project_id)

    def _select_project_id(self, project_id: str) -> bool:
        if not project_id:
            return False
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
            *self._selection_visibility_fields(),
        )
        return True

    def select_scene(self, _handle, _ev, args):
        """Select a scene to filter assets."""
        scene_id = self._resolve_event_value(args, _ev, "data-scene-id")
        self._select_scene_id(scene_id)

    def _select_scene_id(self, scene_id: str) -> bool:
        if not scene_id:
            return False
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
            *self._selection_visibility_fields(),
        )
        return True

    def select_run(self, _handle, _ev, args):
        """Select a training run to inspect its parameters and artifacts."""
        run_id = self._resolve_event_value(args, _ev, "data-run-id")
        self._select_run_id(run_id)

    def _select_run_id(self, run_id: str) -> bool:
        if not run_id or not self._asset_index or not hasattr(self._asset_index, "runs"):
            return False
        run = self._asset_index.runs.get(run_id)
        if not run:
            return False

        self._selected_run_id = run_id
        self._selected_project_id = run.get("project_id")
        self._selected_scene_id = run.get("scene_id")
        self._selected_asset_ids.clear()
        self._selection_type = "run"

        self._dirty_model(
            "selected_run_id",
            "selected_run_name",
            "selected_run_status",
            "selected_run_started",
            "selected_run_completed",
            "selected_run_duration",
            "selected_run_parameters",
            "selected_run_artifacts",
            "assets",
            "selected_count",
            "selected_total_size",
            "selected_count_text",
            "selected_total_text",
            *self._selection_visibility_fields(),
        )
        return True

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
        artifact_id = self._resolve_event_value(args, _ev, "data-asset-id")
        self._select_asset_id(artifact_id)

    def select_asset_by_id(self, _handle, _ev, args):
        """Select an asset by ID."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        self._select_asset_id(asset_id)

    def on_export_selected(self, _handle, _ev, args):
        """Export selected assets."""
        if not self._selected_asset_ids:
            return

        # TODO: Implement export dialog and logic
        _logger.info(f"Export requested for {len(self._selected_asset_ids)} assets")

    def on_load_asset(self, _handle, _ev, args):
        """Load a specific asset by ID into the viewer."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        if not asset_id:
            return

        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            self._log_warn("Asset index not initialized")
            return

        asset = self._asset_index.assets.get(asset_id)
        if not asset:
            self._log_warn("Asset not found: %s", asset_id)
            return
        if asset.get("type") not in self.LOADABLE_TYPES:
            self._log_warn("Asset type is not loadable: %s", asset.get("type"))
            return

        file_path = asset.get("absolute_path") or asset.get("path")
        if not file_path or not os.path.exists(file_path):
            self._log_warn("Asset file not found: %s", file_path)
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
                self._queue_pending_transform_application(asset)
            else:
                # Regular mesh/splat file loading
                transform_node_name = self._load_asset_with_hierarchy(file_path)
                self._apply_asset_transform(asset, transform_node_name)
            self._log_info("Loaded asset: %s", asset.get("name", "unknown"))

            # Select the loaded asset
            self._selected_asset_ids = {asset_id}
            self._selection_type = "asset"
            self.refresh_catalog()
        except Exception as e:
            self._log_error("Failed to load asset %s: %s", asset_id, e)

    def _node_name(self, node: Any) -> str:
        try:
            return str(node.get("name"))
        except Exception:
            return ""

    def _load_asset_with_hierarchy(self, file_path: str) -> Optional[str]:
        scene = lf.get_scene()
        before_ids = {node.id for node in scene.get_nodes()} if scene is not None else set()
        lf.load_file(file_path)
        scene = lf.get_scene()
        if scene is None:
            return None

        new_nodes = [node for node in scene.get_nodes() if node.id not in before_ids]
        by_id = {node.id: node for node in new_nodes}
        for node in new_nodes:
            if getattr(getattr(node, "type", None), "name", "") != "GROUP":
                continue
            parent = by_id.get(getattr(node, "parent_id", -1))
            if parent is None:
                continue
            parent_name = self._node_name(parent)
            if self._node_name(node) == f"{parent_name}_transform":
                return self._node_name(node)
        for node in new_nodes:
            if getattr(getattr(node, "type", None), "name", "") == "GROUP" and getattr(node, "parent_id", -1) == -1:
                return self._node_name(node)
        for node in new_nodes:
            if getattr(node, "parent_id", -1) == -1:
                return self._node_name(node)
        return None

    def _quat_to_euler_deg(self, quat: Any) -> Optional[List[float]]:
        try:
            x, y, z, w = [float(v) for v in quat[:4]]
            n = (x * x + y * y + z * z + w * w) ** 0.5
            if n == 0.0:
                return [0.0, 0.0, 0.0]
            x, y, z, w = x / n, y / n, z / n, w / n

            t0 = 2.0 * (w * x + y * z)
            t1 = 1.0 - 2.0 * (x * x + y * y)
            roll = math.atan2(t0, t1)

            t2 = 2.0 * (w * y - z * x)
            t2 = 1.0 if t2 > 1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch = math.asin(t2)

            t3 = 2.0 * (w * z + x * y)
            t4 = 1.0 - 2.0 * (y * y + z * z)
            yaw = math.atan2(t3, t4)
            return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]
        except Exception:
            return None

    def _apply_asset_transform(self, asset: Dict[str, Any], transform_node_name: Optional[str]) -> bool:
        try:
            if not transform_node_name:
                geometry_metadata = asset.get("geometry_metadata", {}) or {}
                transform_node_name = geometry_metadata.get("transform_node_name")
            if not transform_node_name:
                return False

            transform_metadata = asset.get("transform_metadata") or {}
            if not transform_metadata:
                return False

            matrix = transform_metadata.get("matrix")
            if isinstance(matrix, list) and len(matrix) == 16:
                lf.set_node_transform(transform_node_name, matrix)
                _logger.info("Applied saved matrix transform to '%s'", transform_node_name)
                return True

            translation = transform_metadata.get("translation", [0.0, 0.0, 0.0])
            scale = transform_metadata.get("scale", [1.0, 1.0, 1.0])
            euler_deg = transform_metadata.get("rotation_euler_deg")
            if not euler_deg:
                euler_deg = self._quat_to_euler_deg(
                    transform_metadata.get("rotation_quat", [0.0, 0.0, 0.0, 1.0])
                )
            if not euler_deg:
                euler_deg = [0.0, 0.0, 0.0]

            matrix = lf.compose_transform(translation, euler_deg, scale)
            lf.set_node_transform(transform_node_name, matrix)
            _logger.info("Applied saved transform to '%s'", transform_node_name)
            return True
        except Exception as e:
            _logger.warning(f"Failed to apply asset transform: {e}")
            return False

    def _queue_pending_transform_application(self, asset: Dict[str, Any]) -> None:
        transform_metadata = asset.get("transform_metadata") or {}
        geometry_metadata = asset.get("geometry_metadata", {}) or {}
        subtree_transforms = geometry_metadata.get("subtree_transforms")
        has_matrix = isinstance(transform_metadata.get("matrix"), list) and len(transform_metadata.get("matrix")) == 16
        has_subtree = isinstance(subtree_transforms, dict) and bool(subtree_transforms)
        if not has_matrix and not has_subtree:
            return

        self._pending_transform_applications.append(
            {
                "asset_id": str(asset.get("id", "")),
                "asset_name": str(asset.get("name", "")),
                "transform_metadata": transform_metadata,
                "geometry_metadata": geometry_metadata,
                "queued_at": time.time(),
                "attempts": 0,
                "root_applied": False,
                "pending_subtree_nodes": set(),
            }
        )

    def _resolve_loaded_asset_root_name(self, scene, pending: Dict[str, Any]) -> Optional[str]:
        geometry_metadata = pending.get("geometry_metadata", {}) or {}
        candidate_names: List[str] = []
        for key in ("scene_node_name", "transform_node_name"):
            value = geometry_metadata.get(key)
            if isinstance(value, str) and value:
                candidate_names.append(value)

        asset_name = pending.get("asset_name")
        if isinstance(asset_name, str) and asset_name:
            candidate_names.append(asset_name)

        seen: Set[str] = set()
        deduped = []
        for name in candidate_names:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)

        for name in deduped:
            try:
                if scene.get_node(name) is not None:
                    return name
            except Exception:
                continue
        return None

    def _apply_pending_transform(self, scene, pending: Dict[str, Any]) -> bool:
        transform_metadata = pending.get("transform_metadata", {}) or {}
        geometry_metadata = pending.get("geometry_metadata", {}) or {}
        root_name = self._resolve_loaded_asset_root_name(scene, pending)
        if not root_name:
            return False

        if not pending.get("root_applied"):
            matrix = transform_metadata.get("matrix")
            if isinstance(matrix, list) and len(matrix) == 16:
                lf.set_node_transform(root_name, matrix)
            else:
                translation = transform_metadata.get("translation", [0.0, 0.0, 0.0])
                scale = transform_metadata.get("scale", [1.0, 1.0, 1.0])
                euler_deg = transform_metadata.get("rotation_euler_deg")
                if not euler_deg:
                    euler_deg = self._quat_to_euler_deg(
                        transform_metadata.get("rotation_quat", [0.0, 0.0, 0.0, 1.0])
                    )
                if not euler_deg:
                    euler_deg = [0.0, 0.0, 0.0]
                composed = lf.compose_transform(translation, euler_deg, scale)
                lf.set_node_transform(root_name, composed)
            pending["root_applied"] = True

        subtree_transforms = geometry_metadata.get("subtree_transforms")
        if not isinstance(subtree_transforms, dict) or not subtree_transforms:
            return True

        if not pending.get("pending_subtree_nodes"):
            pending["pending_subtree_nodes"] = {
                str(name)
                for name in subtree_transforms.keys()
                if str(name) and str(name) != root_name
            }

        unresolved = set(pending.get("pending_subtree_nodes", set()))
        for node_name in list(unresolved):
            node = scene.get_node(node_name)
            if node is None:
                continue
            meta = subtree_transforms.get(node_name)
            if not isinstance(meta, dict):
                unresolved.discard(node_name)
                continue
            local_matrix = meta.get("local_matrix")
            if isinstance(local_matrix, list) and len(local_matrix) == 16:
                lf.set_node_transform(node_name, local_matrix)
            unresolved.discard(node_name)

        pending["pending_subtree_nodes"] = unresolved
        return not unresolved

    def _flush_pending_transform_applications(self) -> None:
        if not self._pending_transform_applications:
            return
        scene = lf.get_scene()
        if scene is None:
            return

        now = time.time()
        next_pending: List[Dict[str, Any]] = []
        for pending in self._pending_transform_applications:
            try:
                done = self._apply_pending_transform(scene, pending)
                if done:
                    continue
                pending["attempts"] = int(pending.get("attempts", 0)) + 1
                age = now - float(pending.get("queued_at", now))
                if age > 15.0 or pending["attempts"] > 40:
                    _logger.warning(
                        "Timed out applying deferred transform for asset '%s' (id=%s)",
                        pending.get("asset_name", ""),
                        pending.get("asset_id", ""),
                    )
                    continue
                next_pending.append(pending)
            except Exception as e:
                _logger.warning(
                    "Deferred transform apply failed for asset '%s': %s",
                    pending.get("asset_name", ""),
                    e,
                )
        self._pending_transform_applications = next_pending

    def on_remove_asset(self, _handle, _ev, args):
        """Remove a specific asset from the catalog by ID."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        if not asset_id:
            return

        if not self._asset_index:
            self._log_warn("Asset index not initialized")
            return

        try:
            # Remove asset from catalog
            if not self._delete_asset_from_catalog(asset_id):
                self._log_warn("Asset index does not support asset deletion")
                return

            # Remove from selection if selected
            if asset_id in self._selected_asset_ids:
                self._selected_asset_ids.discard(asset_id)
                self._update_selection_type()

            # Refresh UI
            self.refresh_catalog()

            self._log_info("Removed asset from catalog: %s", asset_id)
        except Exception as e:
            self._log_error("Failed to remove asset %s: %s", asset_id, e)

    def _get_available_projects_for_asset(self, asset: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get list of projects this asset can be moved to."""
        if not self._asset_index or not hasattr(self._asset_index, "projects"):
            return []

        current_project_id = asset.get("project_id", "")
        projects = []

        for proj_id, proj in self._asset_index.projects.items():
            if proj_id != current_project_id:
                projects.append({
                    "id": proj_id,
                    "name": proj.get("name", tr("asset_manager.unnamed_project")),
                })

        # Sort by name
        return sorted(projects, key=lambda p: p["name"].lower())

    def on_toggle_asset_menu(self, _handle, _ev, args):
        """Toggle dropdown menu for an asset."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        if not asset_id:
            return

        # Stop event propagation to prevent card selection
        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        # Toggle: if already open for this asset, close it; otherwise open for this asset
        if self._open_menu_asset_id == asset_id:
            self._open_menu_asset_id = None
        else:
            self._open_menu_asset_id = asset_id

        # Always reload projects when menu opens to ensure fresh data
        if self._handle:
            if self._open_menu_asset_id:
                projects = self.get_move_menu_projects()
                self._log_info("Loading %d projects for move menu", len(projects))
                self._handle.update_record_list("move_menu_projects", projects)
            else:
                self._handle.update_record_list("move_menu_projects", [])

        self._dirty_model("assets")

    def on_rename_asset(self, _handle, _ev, args):
        """Open rename dialog for an asset."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        if not asset_id:
            return

        # Stop event propagation
        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return

        asset = self._asset_index.assets.get(asset_id)
        if not asset:
            return

        # Close the menu
        self._open_menu_asset_id = None
        self._dirty_model("assets")

        # Prompt for rename using input dialog
        current_name = asset.get("name", "Unnamed")

        def _on_rename_result(new_name):
            if new_name and new_name.strip() and new_name.strip() != current_name:
                try:
                    self._asset_index.update_asset(asset_id, name=new_name.strip())
                    self._asset_index.save()
                    self.refresh_catalog()
                    self._log_info("Renamed asset to: %s", new_name.strip())
                except Exception as e:
                    self._log_error("Failed to rename asset: %s", e)

        lf.ui.input_dialog(
            "Rename Asset",
            f"Enter new name for: {current_name}",
            current_name,
            _on_rename_result
        )

    def on_show_in_folder(self, _handle, _ev, args):
        """Open file manager to show asset location."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        if not asset_id:
            return

        # Stop event propagation
        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return

        asset = self._asset_index.assets.get(asset_id)
        if not asset:
            return

        # Close the menu
        self._open_menu_asset_id = None
        self._dirty_model("assets")

        file_path = asset.get("absolute_path") or asset.get("path")
        if not file_path:
            self._log_warn("Asset has no file path: %s", asset_id)
            return

        try:
            import subprocess
            import platform

            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["open", "-R", file_path])
            elif system == "Windows":
                subprocess.run(["explorer", "/select,", file_path])
            else:  # Linux
                subprocess.run(["xdg-open", str(Path(file_path).parent)])

            self._log_info("Opened file location: %s", file_path)
        except Exception as e:
            self._log_error("Failed to open file location: %s", e)

    def on_move_to_project(self, _handle, _ev, args):
        """Move asset to a different project."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        if not asset_id:
            return

        # Stop event propagation
        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return

        asset = self._asset_index.assets.get(asset_id)
        if not asset:
            return

        # Close the menu
        self._open_menu_asset_id = None
        self._dirty_model("assets")

        # Get list of available projects
        if not hasattr(self._asset_index, "projects"):
            self._log_warn("No projects available")
            return

        projects = []
        for proj_id, proj in self._asset_index.projects.items():
            if proj_id != asset.get("project_id"):  # Exclude current project
                projects.append((proj_id, proj.get("name", "Unnamed")))

        if not projects:
            self._log_info("No other projects available to move to")
            return

        # Build project list string
        project_names = [f"{i+1}. {name}" for i, (_, name) in enumerate(projects)]
        project_list = "\n".join(project_names)
        current_project = self._asset_index.projects.get(asset.get("project_id", ""), {}).get("name", "Unknown")

        def _on_project_selected(result):
            if not result or not result.strip():
                return

            try:
                # Parse selection (number or name)
                selection = result.strip()
                selected_project_id = None
                selected_project_name = None

                # Try to parse as number first
                try:
                    idx = int(selection.split(".")[0]) - 1
                    if 0 <= idx < len(projects):
                        selected_project_id, selected_project_name = projects[idx]
                except (ValueError, IndexError):
                    # Try to match by name
                    for proj_id, proj_name in projects:
                        if selection.lower() in proj_name.lower():
                            selected_project_id = proj_id
                            selected_project_name = proj_name
                            break

                if not selected_project_id:
                    self._log_warn("Invalid project selection: %s", selection)
                    return

                # Update asset's project
                self._asset_index.update_asset(
                    asset_id,
                    project_id=selected_project_id,
                    scene_id=None  # Clear scene since scenes are project-specific
                )
                self._asset_index.save()
                self.refresh_catalog()
                self._log_info("Moved asset to project: %s", selected_project_name)

            except Exception as e:
                self._log_error("Failed to move asset: %s", e)

        lf.ui.input_dialog(
            "Move to Project",
            f"Current: {current_project}\n\nAvailable projects:\n{project_list}\n\nEnter number or name:",
            "",
            _on_project_selected
        )

    def _move_asset_to_project(self, asset_id: str, project_id: str) -> None:
        """Move asset to a specific project."""
        self._log_info("Attempting to move asset %s to project %s", asset_id, project_id)

        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            self._log_warn("Asset index not available")
            return

        asset = self._asset_index.assets.get(asset_id)
        if not asset:
            self._log_warn("Asset not found: %s", asset_id)
            return

        project = self._asset_index.projects.get(project_id)
        if not project:
            self._log_warn("Project not found: %s", project_id)
            return

        try:
            self._asset_index.update_asset(
                asset_id,
                project_id=project_id,
                scene_id=None  # Clear scene since scenes are project-specific
            )
            self._asset_index.save()
            self.refresh_catalog()
            self._log_info("Moved asset to project: %s", project.get("name", "Unnamed"))
        except Exception as e:
            self._log_error("Failed to move asset: %s", e)

    def on_create_project_and_move(self, _handle, _ev, args):
        """Create a new project and move asset to it."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        if not asset_id:
            return

        # Stop event propagation
        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return

        asset = self._asset_index.assets.get(asset_id)
        if not asset:
            return

        # Close menu
        self._open_menu_asset_id = None
        self._dirty_model("assets", "move_menu_projects")
        if self._handle:
            self._handle.update_record_list("move_menu_projects", [])

        def _on_project_name_entered(name):
            if not name or not name.strip():
                return

            try:
                # Create new project
                project = self._asset_index.create_project(name=name.strip())
                if not project:
                    self._log_error("Failed to create project")
                    return

                # Move asset to new project
                self._asset_index.update_asset(
                    asset_id,
                    project_id=project.id,
                    scene_id=None
                )
                self._asset_index.save()
                self.refresh_catalog()
                self._log_info("Created project '%s' and moved asset to it", name.strip())

            except Exception as e:
                self._log_error("Failed to create project and move asset: %s", e)

        lf.ui.input_dialog(
            "New Project",
            "Enter name for the new project:",
            "",
            _on_project_name_entered
        )

    def on_toggle_project_menu(self, _handle, _ev, args):
        """Toggle dropdown menu for a project."""
        project_id = self._resolve_event_value(args, _ev, "data-project-id")
        if not project_id:
            return

        # Stop event propagation to prevent row selection
        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        # Toggle: if already open for this project, close it; otherwise open for this project
        if self._open_menu_project_id == project_id:
            self._open_menu_project_id = None
        else:
            self._open_menu_project_id = project_id

        self._dirty_model("projects")

    def on_rename_project(self, _handle, _ev, args):
        """Open rename dialog for a project."""
        project_id = self._resolve_event_value(args, _ev, "data-project-id")
        if not project_id:
            return

        # Stop event propagation
        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        if not self._asset_index or not hasattr(self._asset_index, "projects"):
            return

        project = self._asset_index.projects.get(project_id)
        if not project:
            return

        # Close the menu
        self._open_menu_project_id = None
        self._dirty_model("projects")

        # Prompt for rename using input dialog
        current_name = project.get("name", "Unnamed Project")

        def _on_rename_result(new_name):
            if new_name and new_name.strip() and new_name.strip() != current_name:
                try:
                    self._asset_index.update_project(project_id, name=new_name.strip())
                    self._asset_index.save()
                    self.refresh_catalog()
                    self._log_info("Renamed project to: %s", new_name.strip())
                except Exception as e:
                    self._log_error("Failed to rename project: %s", e)

        lf.ui.input_dialog(
            "Rename Project",
            f"Enter new name for: {current_name}",
            current_name,
            _on_rename_result
        )

    def on_delete_project(self, _handle, _ev, args):
        """Delete a project after moving its assets to Default."""
        project_id = self._resolve_event_value(args, _ev, "data-project-id")
        if not project_id:
            return

        # Stop event propagation
        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        if not self._asset_index or not hasattr(self._asset_index, "projects"):
            return

        project = self._asset_index.projects.get(project_id)
        if not project:
            return

        # Close the menu
        self._open_menu_project_id = None
        self._dirty_model("projects")

        project_name = project.get("name", "Unnamed Project")

        # Find or create Default project
        default_project = None
        default_project_id = None
        for pid, proj in self._asset_index.projects.items():
            if proj.get("name", "").lower() == "default":
                default_project_id = pid
                default_project = proj
                break

        # Create Default project if it doesn't exist
        if not default_project_id:
            try:
                default_project = self._asset_index.create_project(name="Default")
                if default_project:
                    default_project_id = default_project.id
                    self._log_info("Created Default project for asset migration")
            except Exception as e:
                self._log_error("Failed to create Default project: %s", e)
                return

        if not default_project_id:
            self._log_error("Cannot delete project: Default project not available")
            return

        # Move all assets from this project to Default
        moved_count = 0
        if hasattr(self._asset_index, "assets"):
            for asset_id, asset in list(self._asset_index.assets.items()):
                if asset.get("project_id") == project_id:
                    try:
                        self._asset_index.update_asset(
                            asset_id,
                            project_id=default_project_id,
                            scene_id=None  # Clear scene since scenes are project-specific
                        )
                        moved_count += 1
                    except Exception as e:
                        self._log_warn("Failed to move asset %s to Default: %s", asset_id, e)

        # Delete the project
        try:
            if hasattr(self._asset_index, "delete_project"):
                self._asset_index.delete_project(project_id)
            elif hasattr(self._asset_index, "remove_project"):
                self._asset_index.remove_project(project_id)
            else:
                # Fallback: remove from projects dict directly
                if hasattr(self._asset_index, "projects"):
                    del self._asset_index.projects[project_id]

            self._asset_index.save()

            # Clear selection if the deleted project was selected
            if self._selected_project_id == project_id:
                self._selected_project_id = None
                self._selected_scene_id = None
                self._selected_asset_ids.clear()
                self._selection_type = "none"

            self.refresh_catalog()
            self._log_info(
                "Deleted project '%s' and moved %d assets to Default",
                project_name, moved_count
            )
        except Exception as e:
            self._log_error("Failed to delete project: %s", e)

    # ── Lifecycle ─────────────────────────────────────────────

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        set_active_asset_manager_panel(self)
        self._bind_dom_event_listeners(doc)

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
                if (
                    self._sync_existing_asset_metadata()
                    and self._asset_index.library_path.exists()
                ):
                    self._library_mtime = self._asset_index.library_path.stat().st_mtime
                if self._asset_index.library_path.exists():
                    self._library_mtime = self._asset_index.library_path.stat().st_mtime
            except Exception as e:
                _logger.warning(f"Failed to load asset index: {e}")

        # Sync the currently loaded runtime dataset into the catalog when possible.
        # Only auto-select the current scene asset on first mount, not on reopen,
        # to preserve user's previous selection and show all assets.
        has_existing_selection = bool(self._selected_asset_ids)
        self._sync_runtime_scene_catalog(select_current=not has_existing_selection)

        # Clear scene filter on reopen to show all assets in the project
        # (respecting active filters like Splat/PCL/Dataset/Checkpoint)
        if has_existing_selection:
            self._selected_scene_id = None

        # Initial refresh must dirty scalar bindings after catalog load.
        self.refresh_catalog()

    def on_scene_changed(self, doc):
        self._flush_pending_transform_applications()
        self._sync_runtime_scene_catalog(select_current=True)
        self.refresh_catalog()

    def on_update(self, doc):
        """Periodic update - check for missing files."""
        if not self._asset_index:
            return False
        self._flush_pending_transform_applications()

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
                    1
                    for asset in self._asset_index.assets.values()
                    if not asset.get("exists", True)
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
        clear_active_asset_manager_panel(self)
        if self._asset_index and hasattr(self._asset_index, "save"):
            try:
                self._asset_index.save()
            except Exception as e:
                _logger.warning(f"Failed to save asset index: {e}")

        doc.remove_data_model("asset_manager")
        self._handle = None
        self._doc = None

    def _bind_dom_event_listeners(self, doc) -> None:
        """Bind stable DOM listeners for dynamic Asset Manager rows.

        The generated asset/project/scene rows are replaced by data-for updates.
        Binding once to a stable parent mirrors the working popup panels and
        avoids relying on per-row data-event callbacks for card selection.
        """
        content = doc.get_element_by_id("asset-popup-content")
        if content:
            content.add_event_listener("click", self._on_asset_manager_click)

    def _on_asset_manager_click(self, event) -> None:
        container = event.current_target()
        target = event.target()
        if target is None:
            return

        tab_el = rml_widgets.find_ancestor_with_attribute(
            target, "data-asset-tab", container
        )
        if tab_el is not None:
            self._set_active_tab(tab_el.get_attribute("data-asset-tab", ""))
            self._stop_event(event)
            return

        action_el = rml_widgets.find_ancestor_with_attribute(
            target, "data-asset-action", container
        )
        if action_el is not None:
            action = action_el.get_attribute("data-asset-action", "")
            asset_id = action_el.get_attribute("data-asset-id", "")

            if action == "load":
                self.on_load_asset(None, event, [asset_id])
            elif action == "remove":
                self.on_remove_asset(None, event, [asset_id])
            elif action == "menu":
                self.on_toggle_asset_menu(None, event, [asset_id])
                self._stop_event(event)
                return
            elif action == "rename":
                self.on_rename_asset(None, event, [asset_id])
                self._stop_event(event)
                return
            elif action == "show_in_folder":
                self.on_show_in_folder(None, event, [asset_id])
                self._stop_event(event)
                return
            elif action == "move_to_project":
                self.on_move_to_project(None, event, [asset_id])
                self._stop_event(event)
                return
            elif action == "remove_from_menu":
                self.on_remove_asset(None, event, [asset_id])
                self._stop_event(event)
                return
            elif action == "create_project":
                self.on_create_project_and_move(None, event, [asset_id])
                # Close menu after creating project
                self._open_menu_asset_id = None
                self._dirty_model("assets", "move_menu_projects")
                if self._handle:
                    self._handle.update_record_list("move_menu_projects", [])
                self._stop_event(event)
                return
            elif action == "move_to_existing_project":
                project_id = action_el.get_attribute("data-project-id", "")
                self._log_info("Move to existing project clicked: asset=%s, project=%s", asset_id, project_id)
                if project_id:
                    self._move_asset_to_project(asset_id, project_id)
                    # Close menu after move
                    self._open_menu_asset_id = None
                    self._dirty_model("assets", "move_menu_projects")
                    if self._handle:
                        self._handle.update_record_list("move_menu_projects", [])
                else:
                    self._log_warn("No project_id found on action element")
                self._stop_event(event)
                return
            elif action in ("select", "artifact", "scene_asset"):
                # Close any open menu when selecting an asset
                if self._open_menu_asset_id:
                    self._open_menu_asset_id = None
                    self._dirty_model("assets", "move_menu_projects")
                    if self._handle:
                        self._handle.update_record_list("move_menu_projects", [])
                self._select_asset_id(
                    asset_id,
                    toggle=False,
                    multi_select=self._event_multi_select(event),
                )
            self._stop_event(event)
            return

        run_el = rml_widgets.find_ancestor_with_attribute(
            target, "data-run-id", container
        )
        if run_el is not None:
            run_id = run_el.get_attribute("data-run-id", "")
            if self._select_run_id(run_id):
                self._stop_event(event)
            return

        project_el = rml_widgets.find_ancestor_with_attribute(
            target, "data-project-id", container
        )
        if project_el is not None:
            # Check if this is a project action (menu, rename, delete)
            project_action_el = rml_widgets.find_ancestor_with_attribute(
                target, "data-project-action", container
            )
            if project_action_el is not None:
                action = project_action_el.get_attribute("data-project-action", "")
                project_id = project_action_el.get_attribute("data-project-id", "")

                if action == "menu":
                    self.on_toggle_project_menu(None, event, [project_id])
                    self._stop_event(event)
                    return
                elif action == "rename":
                    self.on_rename_project(None, event, [project_id])
                    self._stop_event(event)
                    return
                elif action == "delete":
                    self.on_delete_project(None, event, [project_id])
                    self._stop_event(event)
                    return

            # Regular project selection (not an action button)
            project_id = project_el.get_attribute("data-project-id", "")
            # Close any open project menu when selecting a project
            if self._open_menu_project_id:
                self._open_menu_project_id = None
                self._dirty_model("projects")
            if self._select_project_id(project_id):
                self._stop_event(event)
            return

        scene_el = rml_widgets.find_ancestor_with_attribute(
            target, "data-scene-id", container
        )
        if scene_el is not None:
            scene_id = scene_el.get_attribute("data-scene-id", "")
            if self._select_scene_id(scene_id):
                self._stop_event(event)
            return

        # Close open asset menu when clicking elsewhere
        if self._open_menu_asset_id:
            self._open_menu_asset_id = None
            self._dirty_model("assets", "move_menu_projects")
            if self._handle:
                self._handle.update_record_list("move_menu_projects", [])

        # Close open project menu when clicking elsewhere
        if self._open_menu_project_id:
            self._open_menu_project_id = None
            self._dirty_model("projects")

    def _event_multi_select(self, event) -> bool:
        for key in ("ctrl_key", "meta_key", "command_key"):
            try:
                if event.get_bool_parameter(key, False):
                    return True
            except Exception:
                pass
        return False

    def _stop_event(self, event) -> None:
        try:
            event.stop_propagation()
        except Exception:
            pass

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
                "has_selection",
                "has_multi_selection",
                "has_loadable_selection",
                "selection_type",
                "show_selection_none",
                "show_selection_asset",
                "show_selection_run",
                "show_selection_scene",
                "show_selection_project",
                "show_selection_multiple",
                "show_info_tab",
                "show_parameters_tab",
                "sort_label",
            ):
                self._handle.dirty(field)
            self._handle.dirty_all()

    def _sync_runtime_scene_catalog(self, select_current: bool = False) -> None:
        if not self._asset_index:
            return
        try:
            params = lf.dataset_params()
        except Exception:
            params = None

        if not params or not params.has_params() or not params.data_path:
            return

        try:
            context = ensure_dataset_catalog_context(
                params.data_path,
                asset_index=self._asset_index,
                scanner=self._asset_scanner,
                thumbnails=self._asset_thumbnails,
            )
            backfill_scene_provenance(self._asset_index, context.get("scene_id"))
            if select_current and context.get("asset_id"):
                self._selected_asset_ids = {context["asset_id"]}
                # Preserve user's existing project/scene filters - don't change them
                self._selected_run_id = None
                self._update_selection_type()
        except Exception as exc:
            _logger.debug("Failed to sync runtime scene catalog: %s", exc)

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
                self._handle.update_record_list(
                    "selected_run_parameters", self._format_run_parameters_for_ui(run)
                )

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
                self._handle.update_record_list(
                    "selected_run_artifacts", artifacts_list
                )
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
                        if not scene_data:
                            continue

                        scene_asset_count = 0
                        if hasattr(self._asset_index, "assets"):
                            for asset in self._asset_index.assets.values():
                                if asset.get("scene_id") == scene_id:
                                    scene_asset_count += 1
                        project_scenes.append(
                            {
                                "id": scene_id,
                                "name": scene_data.get(
                                    "name", tr("asset_manager.unnamed_scene")
                                ),
                                "asset_count": scene_asset_count,
                            }
                        )
                self._handle.update_record_list(
                    "selected_project_scenes", project_scenes
                )
            else:
                self._handle.update_record_list("selected_project_scenes", [])

            selected_asset = self._get_selected_asset()
            if selected_asset:
                self._handle.update_record_list(
                    "selected_asset_tags",
                    [{"value": tag} for tag in selected_asset.get("tags", [])],
                )
                self._handle.update_record_list(
                    "selected_asset_runs",
                    [
                        self._format_run_for_ui(run)
                        for run in self._runs_for_asset_context(selected_asset)
                    ],
                )
            else:
                self._handle.update_record_list("selected_asset_tags", [])
                self._handle.update_record_list("selected_asset_runs", [])

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
                    "selected_asset_gaussian_count",
                    "selected_asset_image_count",
                    "selected_asset_strategy",
                    "selected_asset_steps_scaler",
                    "selected_asset_optimizer",
                    "selected_asset_learning_rate",
                    "selected_asset_batch_size",
                    "selected_asset_train_time",
                    "selected_asset_has_training_provenance",
                    "selected_asset_has_runs",
                    "selected_asset_show_no_training",
                    "selected_asset_has_geometry_metadata",
                    "selected_asset_has_dataset_metadata",
                    "selected_asset_dataset_image_count",
                    "selected_asset_dataset_image_root",
                    "selected_asset_dataset_masks",
                    "selected_asset_dataset_sparse_model",
                    "selected_asset_dataset_camera_count",
                    "selected_asset_dataset_database",
                    "selected_asset_dataset_initial_points",
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
            self._handle.dirty_all()
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
            "selected_asset_runs",
            "show_selection_none",
            "show_selection_asset",
            "show_selection_run",
            "show_selection_scene",
            "show_selection_project",
            "show_selection_multiple",
            "show_info_tab",
            "show_parameters_tab",
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
                "selected_asset_runs",
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
                    "selected_asset_runs": lambda: [
                        self._format_run_for_ui(run)
                        for run in self._runs_for_asset_context(
                            self._get_selected_asset()
                        )
                    ],
                }
                if field in list_map:
                    self._handle.update_record_list(field, list_map[field]())

        # Update selection-specific record lists if needed
        if needs_selection_update and not self._updating_selection_details:
            self._update_selection_details()

    def _resolve_event_value(self, args, event, attr_name: str) -> str:
        if args:
            value = args[0]
            if value not in (None, ""):
                return str(value)

        if event is None:
            return ""

        for getter_name in ("current_target", "target"):
            getter = getattr(event, getter_name, None)
            if getter is None:
                continue
            try:
                element = getter()
            except Exception:
                element = None

            while element is not None:
                try:
                    value = element.get_attribute(attr_name, "")
                except Exception:
                    value = ""
                if value:
                    return str(value)
                try:
                    element = element.parent()
                except Exception:
                    element = None

        return ""
