# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager panel for browsing and managing Gaussian Splatting assets."""

import logging
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from urllib.parse import quote

import lichtfeld as lf

from . import rml_widgets
from .asset_manager_integration import (
    clear_active_asset_manager_panel,
    ensure_dataset_catalog_context,
    set_active_asset_manager_panel,
)
from .import_panels import open_url_import_panel, open_watch_dirs_dialog
from .types import Panel
from .ui import RuntimeState

_logger = logging.getLogger(__name__)

PRECISE_SCROLL_STEP = 32.0
RML_PATH_SAFE_CHARS = "/:._-~"

# Import backend components (to be implemented)
try:
    from .asset_index import (
        AssetIndex,
        Project,
        Scene,
        Asset,
        resolve_asset_manager_storage_path,
    )
    from .asset_scanner import AssetScanner
    from .asset_thumbnails import AssetThumbnails

    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    AssetIndex = None
    AssetScanner = None
    AssetThumbnails = None

def tr(key, **kwargs):
    tr_func = getattr(getattr(lf, "ui", None), "tr", None)
    try:
        result = tr_func(key) if callable(tr_func) else key
    except Exception:
        result = key
    if kwargs:
        try:
            return result.format(**kwargs)
        except Exception:
            return result
    return result


def _encode_rml_image_path(path: str) -> str:
    return quote(path, safe=RML_PATH_SAFE_CHARS)

__lfs_panel_classes__ = ["AssetManagerPanel"]
__lfs_panel_ids__ = ["lfs.asset_manager"]


class AssetManagerPanel(Panel):
    """Floating Asset Manager window for browsing splats and exports."""

    SORT_MODES = ("name", "size", "type")
    LOADABLE_TYPES = {"ply_3dgs", "ply_pcl", "rad", "sog", "spz", "checkpoint", "dataset", "mesh", "usd"}

    id = "lfs.asset_manager"
    label = "Asset Manager"
    space = lf.ui.PanelSpace.FLOATING
    order = 20
    template = "rmlui/asset_manager.rml"
    height_mode = lf.ui.PanelHeightMode.FILL
    size = (980, 620)
    update_policy = "dirty"

    # Storage path for asset manager data
    STORAGE_PATH = resolve_asset_manager_storage_path()

    @staticmethod
    def _dedupe_paths(paths: List[Path]) -> List[Path]:
        seen: Set[str] = set()
        result: List[Path] = []
        for path in paths:
            try:
                key = str(path.expanduser().resolve())
            except Exception:
                key = str(path.expanduser())
            if key in seen:
                continue
            seen.add(key)
            result.append(path.expanduser())
        return result

    @classmethod
    def _storage_candidates(cls) -> List[Path]:
        candidates: List[Path] = []

        for env_name in (
            "LICHTFELD_ASSET_MANAGER_DIR",
            "LFS_ASSET_MANAGER_DIR",
        ):
            env_value = os.environ.get(env_name, "").strip()
            if env_value:
                candidates.append(Path(env_value))

        candidates.append(resolve_asset_manager_storage_path())

        xdg_data_home = os.environ.get("XDG_DATA_HOME", "").strip()
        if xdg_data_home:
            candidates.append(Path(xdg_data_home) / "LichtFeldStudio" / "asset_manager")

        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            candidates.append(Path(appdata) / "LichtFeldStudio" / "asset_manager")

        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        if local_appdata:
            candidates.append(Path(local_appdata) / "LichtFeldStudio" / "asset_manager")

        home = Path.home()
        candidates.append(home / ".local" / "share" / "LichtFeldStudio" / "asset_manager")

        # Last resort keeps the panel usable in packaged environments where HOME
        # resolves inside a read-only mount. It is less durable, so only use it
        # when every platform data directory rejects writes.
        candidates.append(Path(tempfile.gettempdir()) / "LichtFeldStudio" / "asset_manager")

        return cls._dedupe_paths(candidates)

    @staticmethod
    def _path_accepts_writes(path: Path) -> bool:
        probe_path: Optional[Path] = None
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=".lfs-write-test-",
                dir=path,
                delete=False,
            ) as probe:
                probe.write(b"ok")
                probe_path = Path(probe.name)
            probe_path.unlink(missing_ok=True)
            return True
        except OSError as exc:
            _logger.debug("Asset Manager storage path is not writable: %s (%s)", path, exc)
            if probe_path is not None:
                try:
                    probe_path.unlink(missing_ok=True)
                except Exception:
                    pass
            return False
        except Exception as exc:
            _logger.debug("Asset Manager storage path probe failed: %s (%s)", path, exc)
            return False

    @classmethod
    def _resolve_writable_storage_path(cls) -> Path:
        for candidate in cls._storage_candidates():
            if cls._path_accepts_writes(candidate):
                return candidate

        # Let backend initialization report the concrete failure if even /tmp is
        # unavailable.
        return resolve_asset_manager_storage_path()

    @staticmethod
    def _copy_existing_catalog(source_dir: Path, target_dir: Path) -> None:
        if source_dir == target_dir:
            return

        source_library = source_dir / "library.json"
        target_library = target_dir / "library.json"
        try:
            if source_library.exists() and not target_library.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_library, target_library)
                _logger.info(
                    "Copied Asset Manager catalog from %s to writable storage %s",
                    source_library,
                    target_library,
                )
        except Exception as exc:
            _logger.warning(
                "Failed to copy Asset Manager catalog from %s to %s: %s",
                source_library,
                target_library,
                exc,
            )

        source_thumbnails = source_dir / "thumbnails"
        target_thumbnails = target_dir / "thumbnails"
        try:
            if source_thumbnails.exists() and not target_thumbnails.exists():
                shutil.copytree(source_thumbnails, target_thumbnails)
        except Exception as exc:
            _logger.debug(
                "Failed to copy Asset Manager thumbnails from %s to %s: %s",
                source_thumbnails,
                target_thumbnails,
                exc,
            )

    def _publish_storage_path(self) -> None:
        """Keep dialog/import helpers on the same writable catalog path."""
        storage_path = self.STORAGE_PATH

        try:
            from . import asset_manager_integration as integration

            integration.resolve_asset_manager_storage_path = lambda: storage_path
        except Exception as exc:
            _logger.debug("Failed to publish Asset Manager storage path: %s", exc)

        try:
            from . import import_panels

            import_panels.resolve_asset_manager_storage_path = lambda: storage_path
            if hasattr(import_panels, "URLImportPanel"):
                import_panels.URLImportPanel.STORAGE_PATH = storage_path
        except Exception as exc:
            _logger.debug("Failed to publish Asset Manager dialog storage path: %s", exc)

    def _configure_storage_path(self) -> None:
        requested_path = resolve_asset_manager_storage_path()
        writable_path = self._resolve_writable_storage_path()
        self.STORAGE_PATH = writable_path
        self.__class__.STORAGE_PATH = writable_path

        if writable_path != requested_path:
            self._copy_existing_catalog(requested_path, writable_path)
            _logger.warning(
                "Asset Manager catalog path %s is not writable; using %s",
                requested_path,
                writable_path,
            )

        self._publish_storage_path()

    def __init__(self):
        self._handle = None

        # Backend components
        self._asset_index: Optional[Any] = None
        self._asset_scanner: Optional[Any] = None
        self._asset_thumbnails: Optional[Any] = None

        # UI state
        self._selected_asset_ids: Set[str] = set()
        self._selected_project_id: Optional[str] = None
        self._selected_scene_id: Optional[str] = None
        self._active_filters: Set[str] = set()  # Multi-select: empty = show all
        self._view_mode: str = "list"  # gallery, list
        self._sort_mode: str = "type"  # name, size, type
        self._search_query: str = ""
        self._pending_tag_name: str = ""

        # Track which asset has its dropdown menu open
        self._open_menu_asset_id: Optional[str] = None
        self._load_menu_asset_id: Optional[str] = None

        # Track which project has its dropdown menu open
        self._open_menu_project_id: Optional[str] = None

        # Selection type for info panel display
        self._selection_type: str = "none"  # none, asset, scene, project, multiple

        # Import menu state
        self._import_menu_open: bool = False

        self._library_mtime: float = 0.0
        self._updating_selection_details: bool = False
        self._pending_transform_applications: List[Dict[str, Any]] = []
        self._reactive_unsubscribers = []
        self._last_scene_generation: Optional[int] = None
        self._last_language_generation: Optional[int] = None

        # New project menu state
        self._new_project_menu_open: bool = False

        # Collapse state for sidebar sections
        self._projects_collapsed: bool = True
        self._filters_collapsed: bool = True

        # Panel resize drag state
        self._sidebar_dragging: bool = False
        self._sidebar_drag_start_x: float = 0.0
        self._sidebar_start_width: float = 176.0
        self._sidebar_width: float = 176.0
        self._right_panel_dragging: bool = False
        self._right_panel_drag_start_x: float = 0.0
        self._right_panel_start_width: float = 300.0
        self._right_panel_width: float = 300.0

    # ── Initialization ────────────────────────────────────────

    def _initialize_backend(self):
        """Initialize backend components."""
        if not BACKEND_AVAILABLE:
            return False

        try:
            self._configure_storage_path()

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
        model.bind("search_query", self.get_search_query, self.set_search_query)

        # View state
        model.bind_func("is_gallery_view", lambda: self._view_mode == "gallery")
        model.bind_func("is_list_view", lambda: self._view_mode == "list")
        model.bind_func("sort_label", self.get_sort_label)

        # Panel widths for resizable sidebar and info panel
        model.bind_func("sidebar_width", lambda: f"{self._sidebar_width}dp")
        model.bind_func("right_panel_width", lambda: f"{self._right_panel_width}dp")

        # Active states
        model.bind_func("active_filters", self.get_active_filters)
        model.bind_func("selection_type", self.get_selection_type)
        model.bind_func("show_selection_none", lambda: self._selection_type == "none")
        model.bind_func("show_selection_asset", lambda: self._selection_type == "asset")
        model.bind_func("show_selection_scene", lambda: self._selection_type == "scene")
        model.bind_func(
            "show_selection_project", lambda: self._selection_type == "project"
        )
        model.bind_func(
            "show_selection_multiple", lambda: self._selection_type == "multiple"
        )

        # Panel label for floating window template
        model.bind_func("panel_label", lambda: tr("asset_manager.panel_title"))

        # Import menu state
        model.bind_func("import_menu_open", self.get_import_menu_open)

        # Import from URL action
        model.bind_func("import_from_url_label", lambda: tr("asset_manager.import_from_url"))
        model.bind_event("on_import_from_url", self.on_import_from_url)


        # New project menu state
        model.bind_func("new_project_menu_open", self.get_new_project_menu_open)
        model.bind_func("create_new_project_label", lambda: tr("asset_manager.action.create_new_project"))

        # Move menu projects list (for hover submenu)
        model.bind_record_list("move_menu_projects")

        # Selected IDs for UI conditionals
        model.bind_func("selected_project_id", self.get_selected_project_id)
        model.bind_func("selected_scene_id", self.get_selected_scene_id)

        # Selection count and state
        model.bind_func("selected_count", self.get_selected_count)
        model.bind_func("selected_count_text", self.get_selected_count_text)
        model.bind_func("has_selection", self.get_has_selection)
        model.bind_func("has_multi_selection", self.get_has_multi_selection)

        # Selected asset properties (flattened bind_func pattern)
        model.bind_func("selected_asset_name", self.get_selected_asset_name)
        model.bind_func("selected_asset_type", self.get_selected_asset_type)
        model.bind_func(
            "selected_asset_project_name", self.get_selected_asset_project_name
        )
        model.bind_func("selected_asset_scene_name", self.get_selected_asset_scene_name)
        model.bind_func("selected_asset_path", self.get_selected_asset_path)
        model.bind_func("selected_asset_size", self.get_selected_asset_size)
        model.bind_func("selected_asset_role", self.get_selected_asset_role)
        model.bind_func("selected_asset_points", self.get_selected_asset_points)
        model.bind_func("selected_asset_resolution", self.get_selected_asset_resolution)
        model.bind_func("selected_asset_duration", self.get_selected_asset_duration)
        model.bind_func("selected_asset_created", self.get_selected_asset_created)
        model.bind_func("selected_asset_modified", self.get_selected_asset_modified)
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
            "selected_asset_dataset_camera_count",
            self.get_selected_asset_dataset_camera_count,
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
        model.bind_func("selected_asset_pill_class", self.get_selected_asset_pill_class)
        model.bind_func("selected_asset_type_label", self.get_selected_asset_type_label)

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
        model.bind_func("selected_project_created", self.get_selected_project_created)
        model.bind_func("selected_project_modified", self.get_selected_project_modified)

        # UI Labels (for i18n)
        model.bind_func("search_icon_label", lambda: tr("asset_manager.toolbar.search_icon"))
        model.bind_func("search_placeholder", lambda: tr("asset_manager.toolbar.search_placeholder"))
        model.bind_func("gallery_label", lambda: tr("asset_manager.toolbar.view_gallery"))
        model.bind_func("list_label", lambda: tr("asset_manager.toolbar.view_list"))
        model.bind_func("import_label", lambda: tr("asset_manager.toolbar.import"))
        model.bind_func("import_splat_label", lambda: tr("asset_manager.import_menu.import_splat"))
        model.bind_func("import_mesh_label", lambda: tr("asset_manager.import_menu.import_mesh"))
        model.bind_func("import_dataset_label", lambda: tr("asset_manager.import_menu.import_dataset"))
        model.bind_func("import_checkpoint_label", lambda: tr("asset_manager.import_menu.import_checkpoint"))

        model.bind_func("projects_title", lambda: tr("asset_manager.sidebar.projects"))
        model.bind_func("scenes_title", lambda: tr("asset_manager.sidebar.scenes"))
        model.bind_func("filters_title", lambda: tr("asset_manager.sidebar.filters"))
        model.bind_func("gallery_title", lambda: tr("asset_manager.toolbar.view_gallery"))
        model.bind_func("list_title", lambda: tr("asset_manager.toolbar.view_list"))
        model.bind_func("edit_watch_dirs_label", lambda: tr("asset_manager.action.edit_watch_dirs"))
        model.bind_func("rename_project_label", lambda: tr("asset_manager.action.rename_project"))
        model.bind_func("delete_project_label", lambda: tr("asset_manager.action.delete_project"))
        model.bind_func("load_button_label", lambda: tr("asset_manager.action.load"))
        model.bind_func("load_new_label", lambda: tr("asset_manager.action.load_new"))
        model.bind_func("add_to_scene_label", lambda: tr("asset_manager.action.add_to_scene"))
        model.bind_func("rename_label", lambda: tr("asset_manager.action.rename"))
        model.bind_func("move_to_project_label", lambda: tr("asset_manager.action.move_to_project"))
        model.bind_func("new_project_label", lambda: tr("asset_manager.action.new_project"))
        model.bind_func("show_in_folder_label", lambda: tr("asset_manager.action.show_in_folder"))
        model.bind_func("update_thumbnail_label", lambda: tr("asset_manager.action.update_thumbnail"))
        model.bind_func("remove_label", lambda: tr("asset_manager.action.remove"))
        model.bind_func("refresh_label", lambda: tr("asset_manager.action.refresh"))
        model.bind_func("clean_missing_label", lambda: tr("asset_manager.action.clean_missing"))
        model.bind_func("refresh_tooltip", lambda: tr("asset_manager.tooltip.refresh"))
        model.bind_func("clean_missing_tooltip", lambda: tr("asset_manager.tooltip.clean_missing"))
        model.bind_func("col_name_label", lambda: tr("asset_manager.property.name"))
        model.bind_func("col_type_label", lambda: tr("asset_manager.property.type"))
        model.bind_func("col_project_label", lambda: tr("asset_manager.property.project"))
        model.bind_func("col_size_label", lambda: tr("asset_manager.property.size"))
        model.bind_func("col_modified_label", lambda: tr("asset_manager.property.modified"))
        model.bind_func("info_tab_label", lambda: tr("asset_manager.info_panel.info"))
        model.bind_func("select_item_hint", lambda: tr("asset_manager.status.select_item"))
        model.bind_func("asset_details_title", lambda: tr("asset_manager.info_panel.asset_details"))
        model.bind_func("prop_project_label", lambda: tr("asset_manager.property.project"))
        model.bind_func("prop_scene_label", lambda: tr("asset_manager.property.scene"))
        model.bind_func("prop_role_label", lambda: tr("asset_manager.property.role"))
        model.bind_func("prop_points_label", lambda: tr("asset_manager.property.points"))

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
        model.bind_func("scene_pill_label", lambda: tr("asset_manager.type.scene"))
        model.bind_func("scene_details_title", lambda: tr("asset_manager.info_panel.scene_details"))
        model.bind_func("prop_assets_label", lambda: tr("asset_manager.property.assets"))
        model.bind_func("scene_assets_title", lambda: tr("asset_manager.info_panel.scenes"))
        model.bind_func("project_pill_label", lambda: tr("asset_manager.type.project"))
        model.bind_func("project_details_title", lambda: tr("asset_manager.info_panel.project_details"))
        model.bind_func("prop_scenes_label", lambda: tr("asset_manager.property.scenes"))
        model.bind_func("scenes_list_title", lambda: tr("asset_manager.sidebar.scenes"))

        # Record lists for data-for loops (main lists)
        model.bind_record_list("projects")
        model.bind_record_list("scenes")
        model.bind_record_list("filters")
        model.bind_record_list("assets")
        model.bind_record_list("selected_asset_tags")

        # Record lists for nested struct lists
        model.bind_record_list("selected_scene_assets")

        self._handle = model.get_handle()

        # Initialize record lists
        self._update_all_record_lists()

        # Event handlers
        model.bind_event("toggle_filter", self.toggle_filter)
        model.bind_event("set_view_mode", self.set_view_mode)
        model.bind_event("cycle_sort_mode", self.cycle_sort_mode)
        model.bind_event("toggle_asset_selection", self.toggle_asset_selection)
        model.bind_event("on_search", self.on_search)
        model.bind_event("on_import_splat", self.on_import_splat)
        model.bind_event("on_import_mesh", self.on_import_mesh)
        model.bind_event("on_import_dataset", self.on_import_dataset)
        model.bind_event("on_load_selected", self.on_load_selected)
        model.bind_event("on_remove_from_catalog", self.on_remove_from_catalog)
        model.bind_event("select_project", self.select_project)
        model.bind_event("select_scene", self.select_scene)
        model.bind_event("toggle_import_menu", self.toggle_import_menu)
        model.bind_event("on_import_checkpoint", self.on_import_checkpoint)
        model.bind_event("on_locate_file", self.on_locate_file)
        model.bind_event("select_asset", self.select_asset_by_id)
        model.bind_event("on_load_asset", self.on_load_asset)
        model.bind_event("on_remove_asset", self.on_remove_asset)
        model.bind_event("on_update_thumbnail", self.on_update_thumbnail)
        model.bind_event("on_pending_tag_change", self.on_pending_tag_change)
        model.bind_event("on_add_tag", self.on_add_tag)
        model.bind_event("on_remove_tag", self.on_remove_tag)

        # Panel resize event handlers
        model.bind_event("on_sidebar_resize_start", self.on_sidebar_resize_start)
        model.bind_event("on_right_panel_resize_start", self.on_right_panel_resize_start)

        # New project event handlers
        model.bind_event("toggle_new_project_menu", self.toggle_new_project_menu)
        model.bind_event("on_create_project_dialog", self.on_create_project_dialog)
        model.bind_event("refresh_catalog", self.refresh_catalog_scan)
        model.bind_event("clean_missing", self.clean_missing)

        # Collapse state bindings
        model.bind_func("projects_collapsed", self.get_projects_collapsed)
        model.bind_func("filters_collapsed", self.get_filters_collapsed)
        model.bind_func("projects_expanded", self.get_projects_expanded)
        model.bind_func("filters_expanded", self.get_filters_expanded)
        model.bind_event("toggle_projects_collapsed", self.toggle_projects_collapsed)
        model.bind_event("toggle_filters_collapsed", self.toggle_filters_collapsed)

    # ── Data Retrieval Methods ─────────────────────────────────

    def get_search_query(self) -> str:
        return self._search_query

    def set_search_query(self, value: str) -> None:
        self._search_query = value
        # Trigger asset list refresh when search query changes
        self._dirty_model("search_query", "assets")

    def get_sort_label(self) -> str:
        labels = {
            "name": tr("asset_manager.toolbar.sort_by_name"),
            "size": tr("asset_manager.toolbar.sort_by_size"),
            "type": tr("asset_manager.toolbar.sort_by_type"),
        }
        return labels.get(self._sort_mode, tr("asset_manager.toolbar.sort_by_name"))

    def get_active_filters(self) -> Set[str]:
        return self._active_filters

    def get_selection_type(self) -> str:
        return self._selection_type

    def get_import_menu_open(self) -> bool:
        return self._import_menu_open

    def get_new_project_menu_open(self) -> bool:
        return self._new_project_menu_open

    def get_projects_collapsed(self) -> bool:
        return self._projects_collapsed

    def get_filters_collapsed(self) -> bool:
        return self._filters_collapsed

    def get_projects_expanded(self) -> bool:
        return not self._projects_collapsed

    def get_filters_expanded(self) -> bool:
        return not self._filters_collapsed

    def toggle_projects_collapsed(self, _handle=None, _ev=None, _args=None):
        self._projects_collapsed = not self._projects_collapsed
        self._dirty_model("projects_collapsed")
        self._dirty_model("projects_expanded")

    def toggle_filters_collapsed(self, _handle=None, _ev=None, _args=None):
        self._filters_collapsed = not self._filters_collapsed
        self._dirty_model("filters_collapsed")
        self._dirty_model("filters_expanded")

    def get_move_menu_projects(self) -> List[Dict[str, str]]:
        """Get projects for the currently open move menu."""
        if not self._open_menu_asset_id or not self._asset_index:
            return []

        asset = self._asset_index.assets.get(self._open_menu_asset_id)
        if not asset:
            return []

        return self._get_available_projects_for_asset(asset)

    def get_selected_project_id(self) -> Optional[str]:
        return self._selected_project_id

    def get_selected_scene_id(self) -> Optional[str]:
        return self._selected_scene_id

    def get_selected_count(self) -> int:
        """Return the number of selected assets."""
        return len(self._selected_asset_ids)

    def get_selected_count_text(self) -> str:
        """Return formatted text showing selected count."""
        count = len(self._selected_asset_ids)
        if count == 0:
            return tr("asset_manager.status.select_item")
        if count == 1:
            return tr("asset_manager.status.one_item_selected")
        return tr("asset_manager.status.multi_items_selected", count=count)

    def get_has_selection(self) -> bool:
        """Return True if any assets are selected."""
        return len(self._selected_asset_ids) > 0

    def get_has_multi_selection(self) -> bool:
        """Return True if multiple assets are selected."""
        return len(self._selected_asset_ids) > 1

    def _coerce_nonnegative_int(self, value: Any, default: int = 0) -> int:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(number):
            return default
        return max(0, int(number))

    def _coerce_optional_nonnegative_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return max(0, int(number))

    def _format_size(self, file_size_bytes: Any) -> str:
        file_size_bytes = self._coerce_nonnegative_int(file_size_bytes)
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
        valid_ids = set(self._asset_index.assets.keys())
        if not self._selected_asset_ids.issubset(valid_ids):
            self._selected_asset_ids.intersection_update(valid_ids)
            self._update_selection_type()
        if not self._selected_asset_ids:
            if self._selection_type == "scene" and not self._selected_scene_id:
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

    def _project_asset_count(self, project_id: str) -> int:
        """Count total assets in a project."""
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return 0
        return sum(
            1
            for asset in self._asset_index.assets.values()
            if asset.get("project_id") == project_id
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
            if not hasattr(self._asset_index, "create_project"):
                return
            try:
                self._asset_index.create_project(name=default_project_name)
                self._log_info(tr("asset_manager.msg.created_default"))
            except Exception as e:
                self._log_error(
                    tr("asset_manager.msg.failed_create_default", error=e)
                )

    def _format_display_name(self, name: str, max_length: int = 15) -> str:
        """Format a name for display, truncating with ... if too long."""
        if not name:
            return name
        if len(name) > max_length:
            return name[:max_length] + "..."
        return name

    def _get_asset_relationship_names(self, asset: Dict[str, Any]):
        project_name = ""
        scene_name = ""

        if self._asset_index and hasattr(self._asset_index, "projects"):
            project_name = self._asset_index.projects.get(
                asset.get("project_id"), {}
            ).get("name", "")
        if self._asset_index and hasattr(self._asset_index, "scenes"):
            scene_name = self._asset_index.scenes.get(asset.get("scene_id"), {}).get(
                "name", ""
            )

        return project_name, scene_name

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
    ) -> Dict[str, str]:
        asset_name = asset.get("name", "Unnamed")
        role_label = asset.get("role", "").replace("_", " ").title()
        display_name = self._asset_display_title(asset)

        if scene_name and scene_name != display_name:
            display_subtitle = scene_name
        elif project_name:
            display_subtitle = project_name
        elif asset_name and asset_name != display_name:
            display_subtitle = asset_name
        else:
            display_subtitle = role_label

        context_parts = []
        if project_name and project_name != display_subtitle:
            context_parts.append(project_name)

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

    # Missing-asset entries are pruned from the index once their catalog metadata
    # has had time to be touched without the file reappearing — distinguishes a
    # transient unmount from genuine cleanup.
    _STALE_ASSET_GRACE_DAYS = 7

    def get_filtered_assets(self) -> List[Dict[str, Any]]:
        """Return assets filtered by search query, active filter, and selections."""
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return []

        from datetime import datetime, timedelta
        now = datetime.now()
        stale_cutoff = timedelta(days=self._STALE_ASSET_GRACE_DAYS)
        prune_ids: List[str] = []

        assets = []
        for asset_id, asset in self._asset_index.assets.items():
            file_path = asset.get("absolute_path") or asset.get("path")
            if not file_path or not os.path.exists(file_path):
                modified_at = asset.get("modified_at", "")
                try:
                    ts = datetime.fromisoformat(modified_at.replace("Z", "+00:00")).replace(tzinfo=None)
                    if now - ts > stale_cutoff:
                        prune_ids.append(asset_id)
                except (ValueError, AttributeError):
                    prune_ids.append(asset_id)
                continue

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

            # Check search query - simple string match
            if self._search_query and not self._asset_matches_query(
                asset, self._search_query
            ):
                continue

            assets.append(self._format_asset_for_ui(asset))

        if prune_ids and hasattr(self._asset_index, "delete_asset"):
            for pid in prune_ids:
                self._asset_index.delete_asset(pid)
            if hasattr(self._asset_index, "save"):
                self._asset_index.save()

        return self._sort_assets(assets)

    def _asset_matches_query(self, asset: Dict[str, Any], query: str) -> bool:
        """Fuzzy search by asset name only.

        Matches if all characters in query appear in the asset name in order.
        Example: 'pt' matches 'points3D', 'tester', 'point_cloud'
        """
        query_l = query.strip().lower()
        if not query_l:
            return True

        asset_name = asset.get("name", "").lower()
        if not asset_name:
            return False

        # Fuzzy match: each query char must appear in name in order
        query_idx = 0
        name_idx = 0
        query_len = len(query_l)
        name_len = len(asset_name)

        while query_idx < query_len and name_idx < name_len:
            if query_l[query_idx] == asset_name[name_idx]:
                query_idx += 1
            name_idx += 1

        # Match if we found all query characters in order
        return query_idx == query_len

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
        return sorted(assets, key=lambda a: a.get("name", "").lower())

    def _thumbnail_decorator(self, asset: Dict[str, Any]) -> str:
        thumbnail_path = asset.get("thumbnail_path") or ""
        if not thumbnail_path:
            return "none"
        try:
            path = Path(str(thumbnail_path)).expanduser()
            if not path.exists():
                return "none"
            return f"image({_encode_rml_image_path(path.as_posix())})"
        except Exception:
            return "none"

    def _format_asset_for_ui(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        """Format asset data for UI display."""
        asset_id = asset.get("id", "")
        asset_type = asset.get("type", "")
        file_size_bytes = self._coerce_nonnegative_int(
            asset.get("file_size_bytes", 0)
        )

        # Format size string
        size_str = self._format_size(file_size_bytes)

        # Get geometry metadata
        geom = asset.get("geometry_metadata", {}) or {}
        gaussian_count = self._coerce_nonnegative_int(
            geom.get("gaussian_count", 0)
        )
        dataset_meta = asset.get("dataset_metadata", {}) or {}

        # Format gaussian count
        if asset_type == "dataset":
            image_count = self._coerce_nonnegative_int(
                dataset_meta.get("image_count", 0)
            )
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
            "dataset": "asset-thumb-dataset",
        }
        thumb_class = thumb_classes.get(asset_type, "asset-thumb-default")

        project_name, scene_name = self._get_asset_relationship_names(asset)
        display_fields = self._get_asset_display_fields(
            asset, project_name, scene_name
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
            "thumbnail_decorator": self._thumbnail_decorator(asset),
            "pill_class": f"asset-pill-{asset_type}" if asset_type else "",
            "is_selected": asset_id in self._selected_asset_ids,
            "exists": asset.get("exists", True),
            "status_label": tr("asset_manager.status.missing") if not asset.get("exists", True) else tr("asset_manager.status.available"),
            "can_load": asset_type in self.LOADABLE_TYPES and asset.get("exists", True),
            "project_id": asset.get("project_id"),
            "scene_id": asset.get("scene_id"),
            "project_name": project_name,
            "scene_name": scene_name,
            "modified_at": asset.get("modified_at", ""),
            "modified_label": self._format_timestamp(asset.get("modified_at", "")),
            "thumbnail_path": asset.get("thumbnail_path"),
            "menu_open": asset_id == self._open_menu_asset_id,
            "load_menu_open": asset_id == self._load_menu_asset_id,
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
            display_name = self._format_display_name(project.get("name", tr("asset_manager.unnamed_project")))
            projects.append(
                {
                    "id": project_id,
                    "name": display_name,
                    "full_name": project.get("name", tr("asset_manager.unnamed_project")),
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
            asset_count = self._scene_asset_count(scene_id)
            scenes.append(
                {
                    "id": scene_id,
                    "name": scene.get("name", tr("asset_manager.unnamed_scene")),
                    "description": scene.get("description", ""),
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
                "label": tr("asset_manager.filter.splat"),
                "count": splat_count,
                "is_selected": "splat" in self._active_filters,
            },
            {
                "id": "pcl",
                "label": tr("asset_manager.filter.pcl"),
                "count": pcl_count,
                "is_selected": "pcl" in self._active_filters,
            },
            {
                "id": "dataset",
                "label": tr("asset_manager.filter.dataset"),
                "count": dataset_count,
                "is_selected": "dataset" in self._active_filters,
            },
            {
                "id": "checkpoint",
                "label": tr("asset_manager.filter.checkpoints"),
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
            {"id": "checkpoint", "label": tr("asset_manager.filter.checkpoints"), "count": 0, "is_selected": False},
        ]

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

    def get_selected_asset_project_name(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        project_name, _scene_name = self._get_asset_relationship_names(asset)
        return self._format_display_name(project_name)

    def get_selected_asset_scene_name(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        _project_name, scene_name = self._get_asset_relationship_names(asset)
        return scene_name

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

    def get_selected_asset_role(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        role = asset.get("role", "")
        return role.replace("_", " ").title() if role else ""

    def get_selected_asset_resolution(self) -> str:
        return ""

    def get_selected_asset_duration(self) -> str:
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
        image_count = self._coerce_nonnegative_int(
            dataset_meta.get("image_count", 0)
        )
        if image_count or asset.get("type") == "dataset":
            return str(image_count)
        return ""

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
        mask_count = self._coerce_nonnegative_int(
            dataset_meta.get("mask_count", 0)
        )
        return str(mask_count)

    def get_selected_asset_dataset_camera_count(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        dataset_meta = asset.get("dataset_metadata", {}) or {}
        camera_count = dataset_meta.get("camera_count")
        if camera_count is None:
            return "--"
        return str(camera_count)

    def get_selected_asset_dataset_initial_points(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        dataset_meta = asset.get("dataset_metadata", {}) or {}
        initial_points = self._coerce_optional_nonnegative_int(
            dataset_meta.get("initial_points")
        )
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
            points = self._coerce_optional_nonnegative_int(
                dataset_meta.get("initial_points")
            )
            if points is None:
                return ""
            if points >= 1_000_000:
                return f"{points / 1_000_000:.2f}M"
            elif points >= 1_000:
                return f"{points / 1_000:.1f}K"
            return str(points)
        # For geometry files (PLY, SOG, etc.), show gaussian count
        geom = asset.get("geometry_metadata", {}) or {}
        gaussian_count = self._coerce_nonnegative_int(geom.get("gaussian_count"))
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
        }
        return type_labels.get(asset_type, asset_type.upper() if asset_type else "")

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
        name = project.get("name", "") if project else ""
        return self._format_display_name(name)

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
        name = project.get("name", "") if project else ""
        return self._format_display_name(name)

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

    def _ensure_import_project(
        self, default_name: str = "Default"
    ) -> Optional[str]:
        # Import to currently selected project if one is selected, otherwise use Default
        if not self._asset_index:
            return None
        
        # If a project is currently selected, use that
        if self._selected_project_id:
            project = self._asset_index.projects.get(self._selected_project_id)
            if project:
                return self._selected_project_id
        
        # Fall back to Default project
        self._ensure_default_project()
        for pid, proj in self._asset_index.projects.items():
            if proj.get("name") == "Default":
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

        if asset_type in ("ply_3dgs", "ply_pcl", "ply", "rad", "sog", "spz", "mesh"):
            kwargs["geometry_metadata"] = format_specific
        elif asset_type == "dataset":
            kwargs["dataset_metadata"] = format_specific

        return kwargs

    def _generate_asset_thumbnail_for_values(
        self,
        asset_id: str,
        asset_type: str,
        asset_path: str,
        dataset_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._asset_thumbnails or not self._asset_index or not asset_id:
            return
        try:
            thumb_path = None
            if asset_type == "dataset":
                generate_dataset_preview = getattr(
                    self._asset_thumbnails,
                    "generate_dataset_preview",
                    None,
                )
                if callable(generate_dataset_preview):
                    thumb_path = generate_dataset_preview(
                        asset_type,
                        asset_id,
                        asset_path,
                        dataset_metadata or {},
                    )
            else:
                generate_rendered_preview = getattr(
                    self._asset_thumbnails,
                    "generate_rendered_preview",
                    None,
                )
                if callable(generate_rendered_preview):
                    thumb_path = generate_rendered_preview(
                        asset_type,
                        asset_id,
                        asset_path,
                    )

            if thumb_path is None:
                thumb_path = self._asset_thumbnails.generate_placeholder(
                    asset_type,
                    asset_id,
                )
            self._asset_index.update_asset(asset_id, thumbnail_path=str(thumb_path))
        except Exception as exc:
            _logger.debug(f"Failed to generate thumbnail for {asset_id}: {exc}")

    def _generate_asset_thumbnail(self, asset: Any) -> None:
        if not asset:
            return
        asset_id = getattr(asset, "id", "")
        asset_type = getattr(asset, "type", "")
        asset_path = getattr(asset, "absolute_path", "") or getattr(asset, "path", "")
        dataset_metadata = getattr(asset, "dataset_metadata", {}) or {}
        self._generate_asset_thumbnail_for_values(
            asset_id,
            asset_type,
            asset_path,
            dataset_metadata,
        )

    def _is_managed_thumbnail_path(self, thumbnail_path: str) -> bool:
        if not self._asset_thumbnails or not thumbnail_path:
            return False

        try:
            thumbs_dir = (
                Path(self._asset_thumbnails.thumbnails_dir)
                .expanduser()
                .resolve()
            )
            path = Path(str(thumbnail_path)).expanduser().resolve()
            try:
                return path.is_relative_to(thumbs_dir)
            except AttributeError:
                return path == thumbs_dir or thumbs_dir in path.parents
        except Exception:
            return False

    def _asset_needs_thumbnail_refresh(self, asset: Dict[str, Any]) -> bool:
        if not self._asset_thumbnails:
            return False

        asset_id = asset.get("id", "")
        asset_type = asset.get("type", "")
        if not asset_id:
            return False

        thumbnail_path = asset.get("thumbnail_path") or ""
        thumbnail_exists = False
        if thumbnail_path:
            try:
                thumbnail_exists = Path(str(thumbnail_path)).expanduser().exists()
            except Exception:
                thumbnail_exists = False

        thumbnail_size_ok = True
        matches_expected_size = getattr(
            self._asset_thumbnails,
            "thumbnail_matches_expected_size",
            None,
        )
        if (
            thumbnail_exists
            and callable(matches_expected_size)
            and self._is_managed_thumbnail_path(str(thumbnail_path))
        ):
            thumbnail_size_ok = matches_expected_size(thumbnail_path)

        if asset_type == "dataset":
            expected_path = getattr(
                self._asset_thumbnails,
                "get_dataset_thumbnail_path",
                lambda _asset_id: None,
            )(asset_id)
            if (
                expected_path
                and str(thumbnail_path) == str(expected_path)
                and thumbnail_exists
                and thumbnail_size_ok
            ):
                return False
            placeholder_path = self._asset_thumbnails.get_thumbnail_path(asset_id)
            return (
                (not thumbnail_exists)
                or (not thumbnail_size_ok)
                or str(thumbnail_path) == str(placeholder_path)
            )

        if asset_type in {
            "checkpoint",
            "mesh",
            "ply_3dgs",
            "ply_pcl",
            "ply",
            "rad",
            "sog",
            "spz",
        }:
            has_rendered = getattr(
                self._asset_thumbnails,
                "has_rendered_thumbnail",
                lambda _aid: False,
            )(asset_id)
            return (
                not has_rendered
                or not thumbnail_exists
                or not thumbnail_size_ok
            )

        return (not thumbnail_exists) or (not thumbnail_size_ok)

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

        if asset_type in ("ply_3dgs", "ply_pcl", "ply", "rad", "sog", "spz", "mesh"):
            geom_meta = asset.get("geometry_metadata", {}) or {}
            # Need sync if empty or if gaussian_count is not present
            return not geom_meta or geom_meta.get("gaussian_count") is None
        return asset.get("file_size_bytes", 0) <= 0

    def _sync_existing_asset_metadata(self) -> bool:
        if not self._asset_index or not self._asset_scanner:
            return False

        updated_any = False
        for asset_id, asset in list(self._asset_index.assets.items()):
            if self._asset_needs_metadata_sync(asset):
                file_path = asset.get("absolute_path") or asset.get("path", "")
                try:
                    metadata = self._asset_scanner.scan_file(file_path)
                except Exception as exc:
                    _logger.debug(f"Failed to rescan asset metadata for {file_path}: {exc}")
                    metadata = None

                if metadata:
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

            asset = self._asset_index.assets.get(asset_id, asset)
            if self._asset_needs_thumbnail_refresh(asset):
                self._generate_asset_thumbnail_for_values(
                    asset_id,
                    asset.get("type", ""),
                    asset.get("absolute_path") or asset.get("path", ""),
                    asset.get("dataset_metadata", {}) or {},
                )
                updated_any = True

        return updated_any

    def _scan_and_register_asset(
        self,
        path: str,
        *,
        project_id: Optional[str],
        scene_id: Optional[str],
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
            "active_filters", "filters", "assets"
        )

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
            "show_selection_scene",
            "show_selection_project",
            "show_selection_multiple",
            "has_selection",
            "has_multi_selection",
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
        """Handle search input changes (real-time)."""
        if args and len(args) > 0:
            self._search_query = str(args[0])
        self._dirty_model("search_query", "assets")

    def on_pending_tag_change(self, _handle, _ev, args):
        """Update the pending tag input buffer."""
        self._pending_tag_name = str(args[0]) if args else ""

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
        self._dirty_model("assets", "selected_asset_tags")

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
        self._dirty_model("assets", "selected_asset_tags")

    # ── New Project Handlers ──────────────────────────────────

    def toggle_new_project_menu(self, _handle, _ev, _args):
        """Toggle the new project dropdown menu visibility."""
        self._new_project_menu_open = not self._new_project_menu_open
        self._dirty_model("new_project_menu_open")

    def on_create_project_dialog(self, _handle, _ev, _args):
        """Open system dialog to create a new project."""
        # Close the dropdown menu
        self._new_project_menu_open = False
        self._dirty_model("new_project_menu_open")

        def _on_project_name_entered(name):
            if not name or not name.strip():
                return

            name = name.strip()

            try:
                # Create new project
                project = self._asset_index.create_project(name=name)
                if not project:
                    self._log_error("Failed to create project")
                    return

                # Refresh the catalog to show the new project
                self.refresh_catalog()
                self._log_info("Created new project: %s", name)

            except Exception as e:
                self._log_error("Failed to create new project: %s", e)

        lf.ui.input_dialog(
            tr("asset_manager.dialog.create_new_project"),
            tr("asset_manager.dialog.enter_project_name"),
            "",
            _on_project_name_entered
        )

    # ── Panel Resize Handlers ─────────────────────────────────

    def on_sidebar_resize_start(self, _handle, event, _args):
        """Start dragging the sidebar resize handle."""
        self._sidebar_dragging = True
        self._sidebar_drag_start_x = float(event.get_parameter("mouse_x", "0"))
        # Use the current width from instance variable
        self._sidebar_start_width = self._sidebar_width
        event.stop_propagation()

    def on_sidebar_resize_delta(self, mouse_x: float) -> None:
        """Update sidebar width during drag."""
        if not self._sidebar_dragging:
            return
        delta_x = mouse_x - self._sidebar_drag_start_x
        new_width = self._sidebar_start_width + delta_x
        # Enforce minimum width of 160dp
        new_width = max(160.0, new_width)
        self._sidebar_width = new_width
        # The width is bound via data-style-width, so just dirty the model
        self._dirty_model("sidebar_width")

    def on_sidebar_resize_end(self) -> None:
        """End sidebar resize drag."""
        self._sidebar_dragging = False

    def on_right_panel_resize_start(self, _handle, event, _args):
        """Start dragging the right panel resize handle."""
        self._right_panel_dragging = True
        self._right_panel_drag_start_x = float(event.get_parameter("mouse_x", "0"))
        # Use the current width from instance variable
        self._right_panel_start_width = self._right_panel_width
        event.stop_propagation()

    def on_right_panel_resize_delta(self, mouse_x: float) -> None:
        """Update right panel width during drag."""
        if not self._right_panel_dragging:
            return
        delta_x = self._right_panel_drag_start_x - mouse_x
        new_width = self._right_panel_start_width + delta_x
        # Enforce minimum width of 200dp
        new_width = max(200.0, new_width)
        self._right_panel_width = new_width
        # The width is bound via data-style-width, so just dirty the model
        self._dirty_model("right_panel_width")

    def on_right_panel_resize_end(self) -> None:
        """End right panel resize drag."""
        self._right_panel_dragging = False

    def on_import_splat(self, _handle, _ev, args):
        """Import a splat/point-cloud file (PLY, SOG, SPZ, USD formats)."""
        if not self._asset_index:
            _logger.warning("Asset index not initialized")
            return

        file_path = lf.ui.open_ply_file_dialog("")
        if not file_path:
            return

        try:
            project_id = self._ensure_import_project()

            path_lower = file_path.lower()
            if path_lower.endswith('.ply'):
                asset_type = None  # Let scanner detect ply_3dgs vs ply_pcl
                fallback_role = (
                    "initial_point_cloud"
                    if 'point_cloud' in path_lower or 'initial' in path_lower
                    else "trained_output"
                )
            elif path_lower.endswith(('.sog', '.spz')):
                asset_type = path_lower.split('.')[-1]
                fallback_role = (
                    "initial_point_cloud"
                    if 'point_cloud' in path_lower or 'initial' in path_lower
                    else "trained_output"
                )
            elif path_lower.endswith(('.usd', '.usda', '.usdc', '.usdz')):
                asset_type = "usd"
                fallback_role = "reference"
            else:
                asset_type = None
                fallback_role = "reference"

            asset = self._scan_and_register_asset(
                file_path,
                project_id=project_id,
                scene_id=self._selected_scene_id,
                fallback_role=fallback_role,
                override_type=asset_type,
            )
            self._import_menu_open = False

            if asset:
                self._selected_asset_ids.add(asset.id)
                self._update_selection_type()

            self.refresh_catalog()
            self._dirty_model("import_menu_open")

            if asset:
                _logger.info(f"Imported asset: {asset.name}")

        except Exception as e:
            _logger.error(f"Failed to import splat: {e}")

    def on_import_mesh(self, _handle, _ev, args):
        """Import a mesh file (OBJ, FBX, GLTF, etc.)."""
        if not self._asset_index:
            _logger.warning("Asset index not initialized")
            return

        file_path = lf.ui.open_mesh_file_dialog("")
        if not file_path:
            return

        try:
            project_id = self._ensure_import_project()

            asset = self._scan_and_register_asset(
                file_path,
                project_id=project_id,
                scene_id=self._selected_scene_id,
                fallback_role="reference",
                override_type="mesh",
            )
            self._import_menu_open = False

            if asset:
                self._selected_asset_ids.add(asset.id)
                self._update_selection_type()

            self.refresh_catalog()
            self._dirty_model("import_menu_open")

            if asset:
                _logger.info(f"Imported asset: {asset.name}")

        except Exception as e:
            _logger.error(f"Failed to import mesh: {e}")

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
                self._asset_index.delete_asset(asset_id)
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

    def select_project(self, _handle, _ev, args):
        """Select a project to filter scenes and assets."""
        project_id = self._resolve_event_value(args, _ev, "data-project-id")
        self._select_project_id(project_id)

    def _select_project_id(self, project_id: str) -> bool:
        if not project_id:
            return False
        self._selected_project_id = project_id if project_id != "all" else None
        self._selected_scene_id = None  # Clear scene selection when project changes
        self._selected_asset_ids.clear()
        self._selection_type = "project" if self._selected_project_id else "none"

        self._dirty_model(
            "projects",
            "scenes",
            "assets",
            "selected_count",
            "selected_total_size",
            "selection_type",
            "selected_project_name",
            "selected_project_created",
            "selected_project_modified",
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

    def select_asset_by_id(self, _handle, _ev, args):
        """Select an asset by ID."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        self._select_asset_id(asset_id)

    def on_load_asset(self, _handle, _ev, args):
        """Load a specific asset by ID into the viewer."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        self._load_asset(asset_id, replace_scene=False)

    def on_load_asset_new(self, _handle, _ev, args):
        """Clear the scene and load a specific asset by ID into the viewer."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        self._load_asset(asset_id, replace_scene=True)

    def on_add_asset_to_scene(self, _handle, _ev, args):
        """Load a specific asset by ID into the current scene."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        self._load_asset(asset_id, replace_scene=False)

    def _load_asset(self, asset_id: str, *, replace_scene: bool) -> None:
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
            self._delete_asset_from_catalog(asset_id)
            return

        try:
            if replace_scene:
                lf.clear_scene()

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
        self._request_model_update()

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
            removed_asset_ids = self._delete_asset_and_managed_storage(asset_id)
            if not removed_asset_ids:
                self._log_warn("Asset index does not support asset deletion")
                return

            # Remove from selection if selected
            for removed_asset_id in removed_asset_ids:
                self._selected_asset_ids.discard(removed_asset_id)
            self._update_selection_type()

            # Refresh UI
            self.refresh_catalog()

            self._log_info("Removed asset from catalog: %s", ", ".join(removed_asset_ids))
        except Exception as e:
            self._log_error("Failed to remove asset %s: %s", asset_id, e)

    def _get_url_import_managed_root(self, asset: Dict[str, Any]) -> Optional[Path]:
        return None

    def _delete_asset_and_managed_storage(self, asset_id: str) -> List[str]:
        """Delete an asset and any URL-managed storage owned by it."""
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return []

        asset = self._asset_index.assets.get(asset_id)
        if not asset:
            return []

        managed_root = self._get_url_import_managed_root(asset)
        related_asset_ids = [asset_id]

        if managed_root is not None:
            related_asset_ids = []
            for candidate_id, candidate in self._asset_index.assets.items():
                candidate_root = self._get_url_import_managed_root(candidate)
                if candidate_root == managed_root:
                    related_asset_ids.append(candidate_id)

            if managed_root.exists():
                shutil.rmtree(managed_root)

        removed_asset_ids: List[str] = []
        for related_asset_id in related_asset_ids:
            if self._delete_asset_from_catalog(related_asset_id):
                removed_asset_ids.append(related_asset_id)

        return removed_asset_ids

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
            tr("asset_manager.dialog.rename_asset"),
            tr("asset_manager.dialog.enter_new_name", name=current_name),
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

    def on_update_thumbnail(self, _handle, _ev, args):
        """Update asset thumbnail from current camera pose."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        if not asset_id:
            return

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

        asset_path = asset.get("absolute_path") or asset.get("path")
        if not asset_path:
            self._log_warn("Asset has no file path: %s", asset_id)
            return

        asset_type = asset.get("type", "")
        if asset_type.lower() not in self.LOADABLE_TYPES:
            self._log_warn("Asset type not renderable: %s", asset_type)
            return

        try:
            camera = lf.get_camera("main")
            if camera is None:
                self._log_warn("No camera available for thumbnail update")
                return

            if not self._asset_thumbnails or not hasattr(self._asset_thumbnails, "generate_rendered_preview_from_camera"):
                self._log_warn("Thumbnail generator not available")
                return

            thumb_path = self._asset_thumbnails.generate_rendered_preview_from_camera(
                asset_type,
                asset_id,
                asset_path,
                eye=camera.eye,
                target=camera.target,
                up=camera.up,
            )
            if thumb_path is not None:
                self._asset_index.update_asset(asset_id, thumbnail_path=str(thumb_path))
                self._asset_index.save()
                self._log_info("Updated thumbnail for %s from current camera", asset_id)
                self.refresh_catalog()
            else:
                self._log_warn("Failed to render thumbnail from camera for %s", asset_id)
        except Exception as e:
            self._log_error("Failed to update thumbnail: %s", e)

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

        prompt = tr("asset_manager.dialog.current_project", name=current_project) + "\n\n"
        prompt += tr("asset_manager.dialog.available_projects") + "\n"
        prompt += project_list + "\n\n"
        prompt += tr("asset_manager.dialog.enter_number_or_name")
        lf.ui.input_dialog(
            tr("asset_manager.dialog.move_to_project"),
            prompt,
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

            name = name.strip()

            try:
                # Create new project
                project = self._asset_index.create_project(name=name)
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
                self._log_info("Created project '%s' and moved asset to it", name)

            except Exception as e:
                self._log_error("Failed to create project and move asset: %s", e)

        lf.ui.input_dialog(
            tr("asset_manager.dialog.new_project"),
            tr("asset_manager.dialog.enter_project_name"),
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

    def on_edit_watch_dirs(self, _handle, _ev, args):
        """Open the watched directories dialog for a project."""
        project_id = self._resolve_event_value(args, _ev, "data-project-id")
        if not project_id:
            return

        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        # Close the menu. Editing watch directories must not depend on or mutate
        # the current project selection; the clicked row is the source of truth.
        self._open_menu_project_id = None
        self._dirty_model("projects")

        ok = open_watch_dirs_dialog(project_id)
        if not ok:
            self._log_warn("Failed to open watch dirs dialog for project %s", project_id)

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
                new_name = new_name.strip()
                try:
                    self._asset_index.update_project(project_id, name=new_name)
                    self._asset_index.save()
                    self.refresh_catalog()
                    self._log_info("Renamed project to: %s", new_name)
                except Exception as e:
                    self._log_error("Failed to rename project: %s", e)

        lf.ui.input_dialog(
            tr("asset_manager.dialog.rename_project"),
            tr("asset_manager.dialog.enter_new_name", name=current_name),
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

        # Prevent deletion of the Default project
        project_name = project.get("name", "")
        if project_name.lower() == "default":
            self._log_warn("Cannot delete the Default project")
            return

        # Prevent deletion of the Default project
        project_name = project.get("name", "")
        if project_name.lower() == "default":
            self._log_warn("Cannot delete the Default project")
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
        self._last_scene_generation = RuntimeState.scene_generation.value
        self._last_language_generation = RuntimeState.language_generation.value
        self._subscribe_reactive_state()

    def on_scene_changed(self, doc):
        self._flush_pending_transform_applications()
        self._sync_runtime_scene_catalog(select_current=True)
        self._last_scene_generation = RuntimeState.scene_generation.value
        self.refresh_catalog()

    def on_update(self, doc):
        """Dirty-policy update for catalog and deferred scene work."""
        changed = False

        language_generation = RuntimeState.language_generation.value
        if language_generation != self._last_language_generation:
            self._last_language_generation = language_generation
            if self._handle:
                self._handle.dirty_all()
            changed = True

        if not self._asset_index:
            return changed

        self._flush_pending_transform_applications()
        if self._pending_transform_applications:
            self._request_model_update()

        scene_generation = RuntimeState.scene_generation.value
        if scene_generation != self._last_scene_generation:
            self._last_scene_generation = scene_generation
            self._sync_runtime_scene_catalog(select_current=True)
            self.refresh_catalog(request_update=False)
            changed = True

        try:
            library_path = self._asset_index.library_path
            if library_path.exists():
                current_mtime = library_path.stat().st_mtime
                if current_mtime > self._library_mtime:
                    self._asset_index.load()
                    if self._sync_existing_asset_metadata() and library_path.exists():
                        current_mtime = library_path.stat().st_mtime
                    self._library_mtime = current_mtime
                    self.refresh_catalog(request_update=False)
                    changed = True

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
                    self.refresh_catalog(request_update=False)
                    changed = True
        except Exception:
            pass

        return changed

    def on_unmount(self, doc):
        """Save index on unmount."""
        self._unsubscribe_reactive_state()
        clear_active_asset_manager_panel(self)
        if self._asset_index and hasattr(self._asset_index, "save"):
            try:
                self._asset_index.save()
            except Exception as e:
                _logger.warning(f"Failed to save asset index: {e}")

        doc.remove_data_model("asset_manager")
        self._handle = None
        self._doc = None

    def _subscribe_reactive_state(self):
        if self._reactive_unsubscribers:
            return

        native_signals = (
            RuntimeState.scene_generation,
            RuntimeState.language_generation,
        )
        self._reactive_unsubscribers = [
            signal.subscribe(lambda _value: self._request_model_update())
            for signal in native_signals
        ]

    def _unsubscribe_reactive_state(self):
        for unsubscribe in self._reactive_unsubscribers:
            try:
                unsubscribe()
            except Exception:
                pass
        self._reactive_unsubscribers = []

    def _request_model_update(self):
        if self._handle:
            rml_widgets.request_model_update(self._handle)

    def _bind_dom_event_listeners(self, doc) -> None:
        """Bind stable DOM listeners for dynamic Asset Manager rows.

        The generated asset/project/scene rows are replaced by data-for updates.
        Binding once to a stable parent mirrors the working popup panels and
        avoids relying on per-row data-event callbacks for card selection.
        """
        content = doc.get_element_by_id("asset-popup-content")
        if content:
            content.add_event_listener("mousedown", self._on_asset_manager_mousedown)
            content.add_event_listener("click", self._on_asset_manager_click)
            content.add_event_listener(
                "dblclick", self._on_asset_manager_double_click
            )
        gallery_scroll = doc.get_element_by_id("asset-gallery-scroll")
        if gallery_scroll:
            gallery_scroll.add_event_listener(
                "mousescroll", self._on_gallery_precise_scroll
            )

        # Resize-start is bound declaratively in RML via data-event-mousedown.
        # Only keep document-level listeners here for active drag tracking.
        doc.add_event_listener("mousemove", self._on_resize_mousemove)
        doc.add_event_listener("mouseup", self._on_resize_mouseup)

    def _on_gallery_precise_scroll(self, event) -> None:
        scroll_el = event.current_target()
        if not scroll_el:
            return

        try:
            wheel_delta = float(event.get_parameter("wheel_delta_y", "0"))
        except (TypeError, ValueError):
            return

        max_scroll = max(0.0, scroll_el.scroll_height - scroll_el.client_height)
        if max_scroll <= 0.0:
            event.stop_propagation()
            return

        new_scroll = min(
            max(scroll_el.scroll_top + wheel_delta * PRECISE_SCROLL_STEP, 0.0),
            max_scroll,
        )
        if abs(new_scroll - scroll_el.scroll_top) > 0.01:
            scroll_el.scroll_top = new_scroll

        event.stop_propagation()

    def _on_asset_manager_click(self, event) -> None:
        if self._input_capture_active():
            return

        container = event.current_target()
        target = event.target()
        if target is None:
            return

        action_el = rml_widgets.find_ancestor_with_attribute(
            target, "data-asset-action", container
        )
        if action_el is not None:
            action = action_el.get_attribute("data-asset-action", "")
            asset_id = action_el.get_attribute("data-asset-id", "")

            if action == "load":
                self.on_load_asset(None, event, [asset_id])
            elif action == "load_new":
                self._load_menu_asset_id = None
                self._dirty_model("assets")
                self.on_load_asset_new(None, event, [asset_id])
                self._stop_event(event)
                return
            elif action == "add_to_scene":
                self._load_menu_asset_id = None
                self._dirty_model("assets")
                self.on_add_asset_to_scene(None, event, [asset_id])
                self._stop_event(event)
                return
            elif action == "remove":
                self.on_remove_asset(None, event, [asset_id])
            elif action == "menu":
                self._load_menu_asset_id = None
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
            elif action == "update_thumbnail":
                self.on_update_thumbnail(None, event, [asset_id])
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
            elif action in ("select", "scene_asset"):
                # Close any open menu when selecting an asset
                if self._open_menu_asset_id:
                    self._open_menu_asset_id = None
                    self._dirty_model("assets", "move_menu_projects")
                    if self._handle:
                        self._handle.update_record_list("move_menu_projects", [])
                if self._load_menu_asset_id:
                    self._load_menu_asset_id = None
                    self._dirty_model("assets")
                self._select_asset_id(
                    asset_id,
                    toggle=False,
                    multi_select=self._event_multi_select(event),
                )
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
                elif action == "watch_dirs":
                    self.on_edit_watch_dirs(None, event, [project_id])
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

        if self._load_menu_asset_id:
            self._load_menu_asset_id = None
            self._dirty_model("assets")

        # Close open project menu when clicking elsewhere
        if self._open_menu_project_id:
            self._open_menu_project_id = None
            self._dirty_model("projects")

    def _on_asset_manager_mousedown(self, event) -> None:
        if self._input_capture_active():
            return

        try:
            button = int(event.get_parameter("button", "0"))
        except (AttributeError, TypeError, ValueError):
            return
        if button != 1:
            return

        container = event.current_target()
        target = event.target()
        if target is None:
            return

        action_el = rml_widgets.find_ancestor_with_attribute(
            target, "data-asset-action", container
        )
        if action_el is None:
            return

        action = action_el.get_attribute("data-asset-action", "")
        if action not in ("select", "scene_asset"):
            return

        asset_id = action_el.get_attribute("data-asset-id", "")
        if not asset_id:
            return

        if self._select_asset_id(asset_id):
            self._load_menu_asset_id = asset_id
            self._open_menu_asset_id = None
            self._open_menu_project_id = None
            self._dirty_model("assets", "projects")
        self._stop_event(event)

    def _on_asset_manager_double_click(self, event) -> None:
        if self._input_capture_active():
            return

        container = event.current_target()
        target = event.target()
        if target is None:
            return

        action_el = rml_widgets.find_ancestor_with_attribute(
            target, "data-asset-action", container
        )
        if action_el is None:
            return

        action = action_el.get_attribute("data-asset-action", "")
        if action not in ("select", "scene_asset"):
            return

        asset_id = action_el.get_attribute("data-asset-id", "")
        if not asset_id:
            return

        self._load_menu_asset_id = None
        self.on_load_asset(None, event, [asset_id])
        self._stop_event(event)

    def _input_capture_active(self) -> bool:
        keymap = getattr(lf, "keymap", None)
        is_capturing = getattr(keymap, "is_capturing", None)
        if not callable(is_capturing):
            return False
        try:
            return bool(is_capturing())
        except Exception:
            return False

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

    def _on_resize_mousemove(self, event) -> None:
        """Handle mousemove for panel resizing."""
        try:
            mouse_x = float(event.get_parameter("mouse_x", "0"))
        except (TypeError, ValueError):
            return
        if self._sidebar_dragging:
            self.on_sidebar_resize_delta(mouse_x)
            event.stop_propagation()
        elif self._right_panel_dragging:
            self.on_right_panel_resize_delta(mouse_x)
            event.stop_propagation()

    def _on_resize_mouseup(self, _event) -> None:
        """Handle mouseup to end panel resizing."""
        self.on_sidebar_resize_end()
        self.on_right_panel_resize_end()

    # ── Integration Hooks (Stubs) ─────────────────────────────

    def on_training_started(
        self, project_name: str, scene_name: str, parameters: Dict[str, Any]
    ) -> Optional[str]:
        """Called when training starts - create project/scene context.

        Returns:
            Scene ID if created, None otherwise.
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

            self._asset_index.save()

            # Update UI if panel is open
            self._selected_project_id = project_id
            self._selected_scene_id = scene_id
            self.refresh_catalog()

            return scene_id

        except Exception as e:
            _logger.error(f"Failed to create training context: {e}")
            return None

    def on_checkpoint_saved(
        self, scene_id: str, checkpoint_path: str, iteration: int
    ) -> Optional[str]:
        """Called when checkpoint is saved - add checkpoint asset.

        Returns:
            Asset ID if created, None otherwise.
        """
        if not self._asset_index:
            return None

        try:
            scene = (
                self._asset_index.scenes.get(scene_id)
                if hasattr(self._asset_index, "scenes")
                else None
            )
            if not scene:
                return None

            asset = self._scan_and_register_asset(
                checkpoint_path,
                project_id=scene.get("project_id"),
                scene_id=scene_id,
                fallback_role="training_checkpoint",
                override_type="checkpoint",
                override_role="training_checkpoint",
            )

            if asset:
                self._asset_index.save()

                # Refresh UI
                self.refresh_catalog()

                return asset.id
            return None

        except Exception as e:
            _logger.error(f"Failed to register checkpoint: {e}")
            return None

    def on_training_completed(
        self, scene_id: str, metrics: Optional[Dict[str, Any]] = None
    ):
        """Called when training completes."""
        if not self._asset_index:
            return

        try:
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
    ) -> Optional[str]:
        """Called when export is generated - register export asset.

        Args:
            file_path: Path to exported file
            export_type: Type of export (ply, rad, sog, spz, mp4, etc.)
            project_id: Optional associated project
            scene_id: Optional associated scene

        Returns:
            Asset ID if created, None otherwise.
        """
        if not self._asset_index:
            return None

        try:
            asset = self._scan_and_register_asset(
                project_id=project_id,
                path=file_path,
                scene_id=scene_id,
                fallback_role="export",
                override_type=export_type,
                override_role="export",
            )

            self._asset_index.save()

            # Refresh UI if panel is open
            self.refresh_catalog()

            return asset.id if asset else None

        except Exception as e:
            _logger.error(f"Failed to register export: {e}")
            return None

    # ── Helper Methods ─────────────────────────────────────────

    def refresh_catalog(self, *, request_update: bool = True):
        """Refresh all catalog data in the UI."""
        self._reconcile_selection()
        self._update_all_record_lists()
        if self._handle:
            self._handle.dirty_all()
            if request_update:
                self._request_model_update()

    def refresh_catalog_scan(self, _handle=None, _ev=None, _args=None):
        """Rescan all known asset directories and refresh the catalog."""
        if not self._asset_index:
            return
        try:
            # Rescan all asset paths in the index
            for asset in list(self._asset_index.assets.values()):
                path = asset.get("absolute_path") or asset.get("path")
                if path and os.path.exists(path):
                    try:
                        if self._asset_scanner:
                            metadata = self._asset_scanner.scan_file(path)
                            if metadata:
                                self._asset_index.update_asset(asset["id"], **metadata)
                    except Exception as exc:
                        _logger.debug(f"Failed to rescan {path}: {exc}")
            self._asset_index.save()
            self.refresh_catalog()
            self._log_info("Refreshed asset catalog")
        except Exception as exc:
            self._log_error(f"Failed to refresh catalog: {exc}")

    def clean_missing(self, _handle=None, _ev=None, _args=None):
        """Prune every catalog entry whose backing file is no longer on disk."""
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return
        prune_ids = [
            asset_id
            for asset_id, asset in self._asset_index.assets.items()
            if not (asset.get("absolute_path") or asset.get("path"))
            or not os.path.exists(asset.get("absolute_path") or asset.get("path"))
        ]
        if not prune_ids:
            return
        for pid in prune_ids:
            self._asset_index.delete_asset(pid)
        self._asset_index.save()
        self._log_info("Pruned %d missing asset(s) from catalog", len(prune_ids))
        self.refresh_catalog()

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
            if select_current and context.get("asset_id"):
                self._selected_asset_ids = {context["asset_id"]}
                # Preserve user's existing project/scene filters - don't change them
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
        # Note: "tags" record list removed - not bound in on_bind_model
        self._handle.update_record_list("assets", self.get_filtered_assets())

        # Update selection-specific record lists
        self._update_selection_details()

    def _update_selection_details(self):
        """Update record lists for selected scene and project."""
        if not self._handle or self._updating_selection_details:
            return
        self._updating_selection_details = True
        try:
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
                # Note: selected_project_scenes record list removed - not used in UI
            # Note: selected_project_scenes record list removed - not used in UI

            selected_asset = self._get_selected_asset()
            if selected_asset:
                self._handle.update_record_list(
                    "selected_asset_tags",
                    [{"value": tag} for tag in selected_asset.get("tags", [])],
                )
            else:
                self._handle.update_record_list("selected_asset_tags", [])

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
            "selected_asset_path",
            "selected_scene",
            "selected_scene_name",
            "selected_scene_project_name",
            "selected_scene_asset_count",
            "selected_scene_assets",
            "selected_project",
            "selected_project_name",
            "selected_asset_tags",
            "show_selection_none",
            "show_selection_asset",
            "show_selection_scene",
            "show_selection_project",
            "show_selection_multiple",
        }
        needs_selection_update = any(f in selection_fields for f in fields)

        for field in fields:
            self._handle.dirty(field)
            # Update record lists when they change
            if field in (
                "projects",
                "scenes",
                "filters",
                "assets",
                "selected_asset_tags",
            ):
                list_map = {
                    "projects": self.get_project_list,
                    "scenes": self.get_scene_list,
                    "filters": self.get_filter_list,
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

    # ── Import from URL handlers ───────────────────────────────

    def on_import_from_url(self, _handle, _ev, _args):
        """Open the retained URL import panel."""
        self._import_menu_open = False
        self._dirty_model("import_menu_open")
        open_url_import_panel()
