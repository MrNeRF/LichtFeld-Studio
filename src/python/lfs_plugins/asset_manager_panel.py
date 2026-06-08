# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager panel for browsing and managing Gaussian Splatting assets."""

import asyncio
import atexit
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Any
from urllib.parse import quote

import lichtfeld as lf

from . import rml_widgets
from .asset_manager_integration import (
    clear_active_asset_manager_panel,
    ensure_dataset_catalog_context,
    set_active_asset_manager_panel,
)
from .import_panels import (
    open_dataset_import_panel,
    open_url_import_panel,
    open_watch_dirs_dialog,
)
from .types import Panel
from .ui import RuntimeState

_logger = logging.getLogger(__name__)

PRECISE_SCROLL_STEP = 32.0
RML_PATH_SAFE_CHARS = "/:._-~"

# Import backend components (to be implemented)
try:
    from .asset_index import (
        AssetIndex,
        Folder,
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
        self._selected_folder_id: Optional[str] = None
        self._selected_scene_id: Optional[str] = None
        self._active_filters: Set[str] = set()  # Multi-select: empty = show all
        self._view_mode: str = "list"  # gallery, list
        self._sort_mode: str = "type"  # name, size, type
        self._search_query: str = ""
        self._pending_tag_name: str = ""

        # Track which asset has its dropdown menu open
        self._open_menu_asset_id: Optional[str] = None
        self._load_menu_asset_id: Optional[str] = None

        # Track which folder has its dropdown menu open
        self._open_menu_folder_id: Optional[str] = None

        # Selection type for info panel display
        self._selection_type: str = "none"  # none, asset, scene, folder, multiple

        # Import menu state
        self._import_menu_open: bool = False

        self._library_mtime: float = 0.0
        self._updating_selection_details: bool = False
        self._pending_transform_applications: List[Dict[str, Any]] = []
        self._reactive_unsubscribers = []
        self._last_scene_generation: Optional[int] = None
        self._last_language_generation: Optional[int] = None

        # Track background thumbnail generation threads for clean shutdown
        self._pending_thumbnail_threads: Set[threading.Thread] = set()
        self._pending_thumbnail_lock = threading.Lock()

        # Auto-save state
        self._auto_save_interval_sec: float = 30.0
        self._last_auto_save_time: float = 0.0

        # Deduplicate thumbnail-failure logs per asset
        self._thumbnail_warned_once: Set[str] = set()

        # Prevent spawning multiple concurrent thumbnail threads for the same
        # asset (e.g. when on_update() fires repeatedly while a thread is still
        # running).
        self._thumbnail_in_flight: Set[str] = set()

        # Track assets whose rendered thumbnail generation already failed so we
        # do not keep retrying on every on_update() cycle.
        self._thumbnail_render_failed: Set[str] = set()

        # Background asset scanning (metadata sync / refresh)
        self._scan_thread: Optional[threading.Thread] = None
        self._scan_thread_lock = threading.Lock()
        self._scan_ui_refresh_needed = False
        self._scan_status = "idle"
        self._scan_last_refresh_time = 0.0
        self._scan_requeue = False
        self._scan_queued_asset_ids: List[str] = []

        # New folder menu state
        self._new_folder_menu_open: bool = False

        # Collapse state for sidebar sections
        self._folders_collapsed: bool = True
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
            "show_selection_folder", lambda: self._selection_type == "folder"
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


        # New folder menu state
        model.bind_func("new_folder_menu_open", self.get_new_folder_menu_open)
        model.bind_func("create_new_folder_label", lambda: tr("asset_manager.action.create_new_folder"))

        # Move menu folders list (for hover submenu)
        model.bind_record_list("move_menu_folders")

        # Selected IDs for UI conditionals
        model.bind_func("selected_folder_id", self.get_selected_folder_id)
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
            "selected_asset_folder_name", self.get_selected_asset_folder_name
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
            "selected_asset_has_sh_degree", self.get_selected_asset_has_sh_degree
        )
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
            "selected_scene_folder_name", self.get_selected_scene_folder_name
        )
        model.bind_func(
            "selected_scene_asset_count", self.get_selected_scene_asset_count
        )
        model.bind_func("selected_scene_created", self.get_selected_scene_created)
        model.bind_func("selected_scene_modified", self.get_selected_scene_modified)

        # Selected folder properties (flattened)
        model.bind_func("selected_folder_name", self.get_selected_folder_name)
        model.bind_func("selected_folder_created", self.get_selected_folder_created)
        model.bind_func("selected_folder_modified", self.get_selected_folder_modified)

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

        model.bind_func("folders_title", lambda: tr("asset_manager.sidebar.folders"))
        model.bind_func("scenes_title", lambda: tr("asset_manager.sidebar.scenes"))
        model.bind_func("filters_title", lambda: tr("asset_manager.sidebar.filters"))
        model.bind_func("gallery_title", lambda: tr("asset_manager.toolbar.view_gallery"))
        model.bind_func("list_title", lambda: tr("asset_manager.toolbar.view_list"))
        model.bind_func("edit_watch_dirs_label", lambda: tr("asset_manager.action.edit_watch_dirs"))
        model.bind_func("rename_folder_label", lambda: tr("asset_manager.action.rename_folder"))
        model.bind_func("delete_folder_label", lambda: tr("asset_manager.action.delete_folder"))
        model.bind_func("load_button_label", lambda: tr("asset_manager.action.load"))
        model.bind_func("load_new_label", lambda: tr("asset_manager.action.load_new"))
        model.bind_func("add_to_scene_label", lambda: tr("asset_manager.action.add_to_scene"))
        model.bind_func("rename_label", lambda: tr("asset_manager.action.rename"))
        model.bind_func("move_to_folder_label", lambda: tr("asset_manager.action.move_to_folder"))
        model.bind_func("new_folder_label", lambda: tr("asset_manager.action.new_folder"))
        model.bind_func("show_in_folder_label", lambda: tr("asset_manager.action.show_in_folder"))
        model.bind_func("update_thumbnail_label", lambda: tr("asset_manager.action.update_thumbnail"))
        model.bind_func("remove_label", lambda: tr("asset_manager.action.remove"))
        model.bind_func("remove_from_catalog_label", lambda: tr("asset_manager.action.remove_from_catalog"))
        model.bind_func("refresh_label", lambda: tr("asset_manager.action.refresh"))
        model.bind_func("clean_missing_label", lambda: tr("asset_manager.action.clean_missing"))
        model.bind_func("refresh_tooltip", lambda: tr("asset_manager.tooltip.refresh"))
        model.bind_func("clean_missing_tooltip", lambda: tr("asset_manager.tooltip.clean_missing"))
        model.bind_func("col_name_label", lambda: tr("asset_manager.property.name"))
        model.bind_func("col_type_label", lambda: tr("asset_manager.property.type"))
        model.bind_func("col_folder_label", lambda: tr("asset_manager.property.folder"))
        model.bind_func("col_size_label", lambda: tr("asset_manager.property.size"))
        model.bind_func("col_modified_label", lambda: tr("asset_manager.property.modified"))
        model.bind_func("info_tab_label", lambda: tr("asset_manager.info_panel.info"))
        model.bind_func("select_item_hint", lambda: tr("asset_manager.status.select_item"))
        model.bind_func("asset_details_title", lambda: tr("asset_manager.info_panel.asset_details"))
        model.bind_func("prop_folder_label", lambda: tr("asset_manager.property.folder"))
        model.bind_func("prop_scene_label", lambda: tr("asset_manager.property.scene"))
        model.bind_func("prop_role_label", lambda: tr("asset_manager.property.role"))
        model.bind_func("prop_points_label", lambda: tr("asset_manager.property.points"))
        model.bind_func("selected_asset_sh_degree", self.get_selected_asset_sh_degree)
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
        model.bind_func("scene_pill_label", lambda: tr("asset_manager.type.scene"))
        model.bind_func("scene_details_title", lambda: tr("asset_manager.info_panel.scene_details"))
        model.bind_func("prop_assets_label", lambda: tr("asset_manager.property.assets"))
        model.bind_func("scene_assets_title", lambda: tr("asset_manager.info_panel.scenes"))
        model.bind_func("folder_pill_label", lambda: tr("asset_manager.type.folder"))
        model.bind_func("folder_details_title", lambda: tr("asset_manager.info_panel.folder_details"))
        model.bind_func("prop_scenes_label", lambda: tr("asset_manager.property.scenes"))
        model.bind_func("scenes_list_title", lambda: tr("asset_manager.sidebar.scenes"))
        model.bind_func("scan_status_label", self.get_scan_status_label)
        model.bind_func("scan_status_class", self.get_scan_status_class)
        model.bind_func("scan_status_visible", self.get_scan_status_visible)

        # Record lists for data-for loops (main lists)
        model.bind_record_list("folders")
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
        model.bind_event("select_folder", self.select_folder)
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

        # New folder event handlers
        model.bind_event("toggle_new_folder_menu", self.toggle_new_folder_menu)
        model.bind_event("on_create_folder_dialog", self.on_create_folder_dialog)
        model.bind_event("refresh_catalog", self.refresh_catalog_scan)
        model.bind_event("clean_missing", self.clean_missing)

        # Collapse state bindings
        model.bind_func("folders_collapsed", self.get_folders_collapsed)
        model.bind_func("filters_collapsed", self.get_filters_collapsed)
        model.bind_func("folders_expanded", self.get_folders_expanded)
        model.bind_func("filters_expanded", self.get_filters_expanded)
        model.bind_event("toggle_folders_collapsed", self.toggle_folders_collapsed)
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

    def get_new_folder_menu_open(self) -> bool:
        return self._new_folder_menu_open

    def get_folders_collapsed(self) -> bool:
        return self._folders_collapsed

    def get_filters_collapsed(self) -> bool:
        return self._filters_collapsed

    def get_folders_expanded(self) -> bool:
        return not self._folders_collapsed

    def get_filters_expanded(self) -> bool:
        return not self._filters_collapsed

    def toggle_folders_collapsed(self, _handle=None, _ev=None, _args=None):
        self._folders_collapsed = not self._folders_collapsed
        self._dirty_model("folders_collapsed")
        self._dirty_model("folders_expanded")

    def toggle_filters_collapsed(self, _handle=None, _ev=None, _args=None):
        self._filters_collapsed = not self._filters_collapsed
        self._dirty_model("filters_collapsed")
        self._dirty_model("filters_expanded")

    def get_move_menu_folders(self) -> List[Dict[str, str]]:
        """Get folders for the currently open move menu."""
        if not self._open_menu_asset_id or not self._asset_index:
            return []

        asset = self._asset_index.assets.get(self._open_menu_asset_id)
        if not asset:
            return []

        return self._get_available_folders_for_asset(asset)

    def get_selected_folder_id(self) -> Optional[str]:
        return self._selected_folder_id

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

    def _ellipsize_path(self, path: Any, max_chars: int = 56) -> str:
        path = str(path or "")
        if not path or len(path) <= max_chars:
            return path
        keep = max(8, (max_chars - 3) // 2)
        return f"{path[:keep]}...{path[-keep:]}"

    def _reconcile_selection(self) -> None:
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            self._selected_asset_ids.clear()
            self._selected_folder_id = None
            self._selected_scene_id = None
            self._update_selection_type()
            return
        if (
            self._selected_folder_id
            and self._selected_folder_id
            not in getattr(self._asset_index, "folders", {})
        ):
            self._selected_folder_id = None
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
            elif self._selection_type == "folder" and not self._selected_folder_id:
                self._selection_type = "none"

    @staticmethod
    def _asset_file_exists(asset: Dict[str, Any]) -> bool:
        """Single source of truth for asset presence, shared by the list and
        every count surface so sidebar badges never disagree with what renders."""
        file_path = asset.get("absolute_path") or asset.get("path")
        return bool(file_path) and os.path.exists(file_path)

    def _scene_asset_count(self, scene_id: str) -> int:
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return 0
        return sum(
            1
            for asset in self._asset_index.assets.values()
            if asset.get("scene_id") == scene_id and self._asset_file_exists(asset)
        )

    def _folder_asset_count(self, folder_id: str) -> int:
        """Count assets in a folder whose backing file is present on disk."""
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return 0
        return sum(
            1
            for asset in self._asset_index.assets.values()
            if asset.get("folder_id") == folder_id and self._asset_file_exists(asset)
        )

    def _folder_sort_key(self, folder_id: str) -> str:
        if not self._asset_index or not hasattr(self._asset_index, "folders"):
            return folder_id
        folder = self._asset_index.folders.get(folder_id, {})
        return self._sort_text(folder.get("name") or folder_id)

    def _default_folder_id(self) -> Optional[str]:
        if not self._asset_index or not hasattr(self._asset_index, "folders"):
            return None
        for folder_id, folder in self._asset_index.folders.items():
            if self._sort_text(folder.get("name")).strip() == "default":
                return folder_id
        return None

    @staticmethod
    def _sort_text(value: Any) -> str:
        return str(value or "").lower()

    def _repair_selected_folder(self) -> Optional[str]:
        if not self._asset_index or not hasattr(self._asset_index, "folders"):
            self._selected_folder_id = None
            self._selected_scene_id = None
            return None

        folders = self._asset_index.folders
        candidate_id: Optional[str] = None
        if self._selected_folder_id in folders:
            candidate_id = self._selected_folder_id

        if not candidate_id and self._selected_scene_id and hasattr(self._asset_index, "scenes"):
            scene = self._asset_index.scenes.get(self._selected_scene_id)
            scene_folder_id = scene.get("folder_id") if scene else None
            if scene_folder_id in folders:
                candidate_id = scene_folder_id

        if not candidate_id and hasattr(self._asset_index, "assets"):
            for asset_id in self._selected_asset_ids:
                asset = self._asset_index.assets.get(asset_id)
                asset_folder_id = asset.get("folder_id") if asset else None
                if asset_folder_id in folders:
                    candidate_id = asset_folder_id
                    break

        if not candidate_id:
            candidate_id = self._default_folder_id()

        if not candidate_id and folders:
            candidate_id = sorted(folders.keys(), key=self._folder_sort_key)[0]

        self._selected_folder_id = candidate_id
        if not candidate_id:
            self._selected_scene_id = None
            self._selected_asset_ids.clear()
            if self._selection_type == "folder":
                self._selection_type = "none"
            return None

        if self._selected_scene_id and hasattr(self._asset_index, "scenes"):
            scene = self._asset_index.scenes.get(self._selected_scene_id)
            if not scene or scene.get("folder_id") != candidate_id:
                self._selected_scene_id = None
                if self._selection_type == "scene":
                    self._selection_type = "folder"
        if self._selected_asset_ids and hasattr(self._asset_index, "assets"):
            visible_asset_ids = {
                aid
                for aid in self._selected_asset_ids
                if self._asset_index.assets.get(aid, {}).get("folder_id") == candidate_id
            }
            if visible_asset_ids != self._selected_asset_ids:
                self._selected_asset_ids = visible_asset_ids
                if not visible_asset_ids and self._selection_type == "asset":
                    self._selection_type = "folder"
        if self._selection_type == "none":
            self._selection_type = "folder"
        return candidate_id

    def _format_display_name(self, name: str, max_length: int = 15) -> str:
        """Format a name for display, truncating with ... if too long."""
        if not name:
            return name
        if len(name) > max_length:
            return name[:max_length] + "..."
        return name

    def _get_asset_relationship_names(self, asset: Dict[str, Any]):
        folder_name = ""
        scene_name = ""

        if self._asset_index and hasattr(self._asset_index, "folders"):
            folder_name = self._asset_index.folders.get(
                asset.get("folder_id"), {}
            ).get("name", "")
        if self._asset_index and hasattr(self._asset_index, "scenes"):
            scene_name = self._asset_index.scenes.get(asset.get("scene_id"), {}).get(
                "name", ""
            )

        return str(folder_name or ""), str(scene_name or "")

    def _asset_display_title(self, asset: Dict[str, Any]) -> str:
        # Prioritize custom name if set by user
        custom_name = str(asset.get("name") or "").strip()
        if custom_name:
            return custom_name

        # Fall back to filename from path
        file_path = asset.get("absolute_path") or asset.get("path") or ""
        if file_path:
            try:
                leaf = Path(os.path.normpath(str(file_path))).name
                if leaf:
                    return leaf
            except Exception:
                pass

        return tr("asset_manager.unnamed")

    def _get_asset_display_fields(
        self,
        asset: Dict[str, Any],
        folder_name: str,
        scene_name: str,
    ) -> Dict[str, str]:
        asset_name = str(asset.get("name") or "")
        role = str(asset.get("role") or "")
        role_label = role.replace("_", " ").title()
        display_name = self._asset_display_title(asset)

        if scene_name and scene_name != display_name:
            display_subtitle = scene_name
        elif folder_name:
            display_subtitle = folder_name
        elif asset_name and asset_name != display_name:
            display_subtitle = asset_name
        else:
            display_subtitle = role_label

        context_parts = []
        if folder_name and folder_name != display_subtitle:
            context_parts.append(folder_name)

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

        folder_id = self._repair_selected_folder()
        if not folder_id:
            return []

        assets = []
        for asset_id, asset in self._asset_index.assets.items():
            if not self._asset_file_exists(asset):
                modified_at = asset.get("modified_at", "")
                try:
                    ts = datetime.fromisoformat(modified_at.replace("Z", "+00:00")).replace(tzinfo=None)
                    if now - ts > stale_cutoff:
                        prune_ids.append(asset_id)
                except (ValueError, AttributeError):
                    prune_ids.append(asset_id)
                continue

            if asset.get("folder_id") != folder_id:
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
        query_l = str(query or "").strip().lower()
        if not query_l:
            return True

        asset_name = self._sort_text(asset.get("name"))
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
            return sorted(assets, key=lambda a: self._sort_text(a.get("name")))
        if self._sort_mode == "size":
            return sorted(
                assets, key=lambda a: a.get("file_size_bytes", 0), reverse=True
            )
        if self._sort_mode == "type":
            return sorted(assets, key=lambda a: self._sort_text(a.get("type")))
        return sorted(assets, key=lambda a: self._sort_text(a.get("name")))

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
        asset_id = str(asset.get("id") or "")
        asset_type = str(asset.get("type") or "")
        asset_name = str(asset.get("name") or tr("asset_manager.unnamed"))
        role = str(asset.get("role") or "")
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

        folder_name, scene_name = self._get_asset_relationship_names(asset)
        display_fields = self._get_asset_display_fields(
            asset, folder_name, scene_name
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
        tags = asset.get("tags") or []
        if isinstance(tags, (list, tuple, set)):
            tags_label = ", ".join(str(tag) for tag in tags if tag is not None)
        else:
            tags_label = str(tags)

        return {
            "id": asset_id,
            "name": asset_name,
            "display_name": display_fields["display_name"],
            "display_subtitle": display_fields["display_subtitle"],
            "context_label": display_fields["context_label"],
            "type": asset_type,
            "role": role,
            "type_label": type_label,
            "role_label": role.replace("_", " ").title(),
            "size_label": size_str,
            "file_size_bytes": file_size_bytes,
            "points_label": points_str,
            "gaussian_count": gaussian_count,
            # Record-list rows only support scalar fields in the current RML bridge.
            "tags_label": tags_label,
            "thumb_class": thumb_class,
            "thumb_label": asset_type.upper() if asset_type else tr("asset_manager.type.asset"),
            "thumbnail_decorator": self._thumbnail_decorator(asset),
            "pill_class": f"asset-pill-{asset_type.replace('_', '-')}" if asset_type else "",
            "is_selected": asset_id in self._selected_asset_ids,
            "exists": asset.get("exists", True),
            "status_label": tr("asset_manager.status.missing") if not asset.get("exists", True) else tr("asset_manager.status.available"),
            "can_load": asset_type in self.LOADABLE_TYPES and asset.get("exists", True),
            "folder_id": asset.get("folder_id"),
            "scene_id": asset.get("scene_id"),
            "folder_name": folder_name,
            "scene_name": scene_name,
            "modified_at": str(asset.get("modified_at") or ""),
            "modified_label": self._format_timestamp(asset.get("modified_at", "")),
            "thumbnail_path": asset.get("thumbnail_path"),
            "menu_open": asset_id == self._open_menu_asset_id,
            "load_menu_open": asset_id == self._load_menu_asset_id,
        }

    def get_folder_list(self) -> List[Dict[str, Any]]:
        """Return list of folders with asset counts for UI."""
        if not self._asset_index or not hasattr(self._asset_index, "folders"):
            return []

        self._repair_selected_folder()

        folders = []
        for folder_id, folder in self._asset_index.folders.items():
            # Show all folders, even empty ones (user must manually delete)
            asset_count = self._folder_asset_count(folder_id)
            display_name = self._format_display_name(folder.get("name", tr("asset_manager.unnamed_folder")))
            folders.append(
                {
                    "id": folder_id,
                    "name": display_name,
                    "full_name": folder.get("name", tr("asset_manager.unnamed_folder")),
                    "description": folder.get("description", ""),
                    "scene_count": asset_count,  # Now shows asset count instead of scene count
                    "is_selected": folder_id == self._selected_folder_id,
                    "thumbnail_asset_id": folder.get("thumbnail_asset_id"),
                    "menu_open": folder_id == self._open_menu_folder_id,
                }
            )

        return sorted(folders, key=lambda f: self._sort_text(f.get("name")))

    def get_scene_list(self) -> List[Dict[str, Any]]:
        """Return list of scenes for selected folder."""
        if not self._asset_index or not hasattr(self._asset_index, "scenes"):
            return []

        if not self._selected_folder_id:
            return []

        scenes = []
        for scene_id, scene in self._asset_index.scenes.items():
            if scene.get("folder_id") != self._selected_folder_id:
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

        return sorted(scenes, key=lambda s: self._sort_text(s.get("name")))

    def get_filter_list(self) -> List[Dict[str, Any]]:
        """Return list of filter categories with counts (multi-select checkboxes)."""
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return self._get_default_filters()
        folder_id = self._repair_selected_folder()
        if not folder_id:
            return self._get_default_filters()

        assets = [
            a
            for a in self._asset_index.assets.values()
            if self._asset_file_exists(a) and a.get("folder_id") == folder_id
        ]

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
            {
                "id": "splat",
                "label": tr("asset_manager.filter.splat"),
                "count": 0,
                "is_selected": "splat" in self._active_filters,
            },
            {
                "id": "pcl",
                "label": tr("asset_manager.filter.pcl"),
                "count": 0,
                "is_selected": "pcl" in self._active_filters,
            },
            {
                "id": "dataset",
                "label": tr("asset_manager.filter.dataset"),
                "count": 0,
                "is_selected": "dataset" in self._active_filters,
            },
            {
                "id": "checkpoint",
                "label": tr("asset_manager.filter.checkpoints"),
                "count": 0,
                "is_selected": "checkpoint" in self._active_filters,
            },
        ]

    # ── Scan Status Getters ───────────────────────────────────

    def get_scan_status_label(self) -> str:
        return tr("asset_manager.status.scanning")

    def get_scan_status_class(self) -> str:
        if self._scan_status == "scanning":
            return "scan-status-scanning"
        return "scan-status-idle"

    def get_scan_status_visible(self) -> bool:
        return True

    def _set_scan_status(self, status: str, *, refresh_needed: bool = False) -> None:
        with self._scan_thread_lock:
            self._scan_status = status
            if refresh_needed:
                self._scan_ui_refresh_needed = True
        self._request_model_update()

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
        asset_type = str(asset.get("type") or "") if asset else ""
        return asset_type.upper() if asset_type else ""

    def get_selected_asset_folder_name(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        folder_name, _scene_name = self._get_asset_relationship_names(asset)
        return self._format_display_name(folder_name)

    def get_selected_asset_scene_name(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        _folder_name, scene_name = self._get_asset_relationship_names(asset)
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
        role = str(asset.get("role") or "")
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

    def get_selected_asset_has_sh_degree(self) -> bool:
        asset = self._get_selected_asset()
        if not asset:
            return False
        geom = asset.get("geometry_metadata", {}) or {}
        return geom.get("sh_degree") is not None

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
        asset_type = str(asset.get("type") or "")
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

    def get_selected_asset_sh_degree(self) -> str:
        """Return SH degree when saved geometry metadata includes it."""
        asset = self._get_selected_asset()
        if not asset:
            return ""
        geom = asset.get("geometry_metadata", {}) or {}
        sh_degree = geom.get("sh_degree")
        _logger.info(
            "get_selected_asset_sh_degree: type=%s geom=%s sh_degree=%r",
            str(asset.get("type") or ""),
            geom,
            sh_degree,
        )
        if sh_degree is None:
            return "--"
        return str(int(sh_degree))

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
        asset_type = str(asset.get("type") or "")
        return f"asset-pill-{asset_type.replace('_', '-')}" if asset_type else ""

    def get_selected_asset_type_label(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        asset_type = str(asset.get("type") or "")
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

    def get_selected_scene_folder_name(self) -> str:
        scene = self._get_selected_scene()
        if not scene:
            return ""
        folder_id = scene.get("folder_id", "")
        if not folder_id or not self._asset_index:
            return ""
        folder = getattr(self._asset_index, "folders", {}).get(folder_id)
        name = folder.get("name", "") if folder else ""
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
            if asset.get("scene_id") == scene_id and self._asset_file_exists(asset)
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

    # ── Flattened Selected Folder Getters ─────────────────────

    def _get_selected_folder(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected folder, if any."""
        if not self._selected_folder_id:
            return None
        if not self._asset_index or not hasattr(self._asset_index, "folders"):
            return None
        return self._asset_index.folders.get(self._selected_folder_id)

    def get_selected_folder_name(self) -> str:
        folder = self._get_selected_folder()
        name = folder.get("name", "") if folder else ""
        return self._format_display_name(name)

    def get_selected_folder_created(self) -> str:
        folder = self._get_selected_folder()
        if not folder:
            return ""
        created_at = folder.get("created_at", "")
        return self._format_timestamp(created_at) if created_at else ""

    def get_selected_folder_modified(self) -> str:
        folder = self._get_selected_folder()
        if not folder:
            return ""
        modified_at = folder.get("modified_at", "")
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

    def _create_folder_from_name(self, name: str) -> Optional[str]:
        if not self._asset_index or not name or not name.strip():
            return None
        name = name.strip()
        try:
            folder = self._asset_index.create_folder(name=name)
            if not folder:
                self._log_error("Failed to create folder")
                return None

            self._selected_folder_id = folder.id
            self._selected_scene_id = None
            self._selected_asset_ids.clear()
            self._selection_type = "folder"
            self.refresh_catalog()
            self._log_info("Created new folder: %s", name)
            return folder.id
        except Exception as e:
            self._log_error("Failed to create new folder: %s", e)
            return None

    def _prompt_for_import_folder(
        self, continuation: Callable[[str], None]
    ) -> None:
        def _on_folder_name_entered(name):
            folder_id = self._create_folder_from_name(name)
            if folder_id:
                continuation(folder_id)

        lf.ui.input_dialog(
            tr("asset_manager.dialog.create_new_folder"),
            tr("asset_manager.dialog.enter_folder_name"),
            "",
            _on_folder_name_entered,
        )

    def _with_import_folder(self, continuation: Callable[[str], None]) -> None:
        if not self._asset_index:
            self._log_warn("Asset index not initialized")
            return
        folder_id = self._ensure_import_folder()
        if folder_id:
            continuation(folder_id)
            return
        self._prompt_for_import_folder(continuation)

    def _ensure_import_folder(self) -> Optional[str]:
        # Import to the selected folder, repairing selection to an existing folder.
        if not self._asset_index:
            return None
        return self._repair_selected_folder()

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

    @staticmethod
    def _maybe_await(coro_or_result):
        """Await if the value is a coroutine, otherwise return it directly.

        This lets the panel work with both async and sync thumbnail generators.
        """
        if asyncio.iscoroutine(coro_or_result):
            return asyncio.run(coro_or_result)
        return coro_or_result

    def _generate_asset_thumbnail_for_values(
        self,
        asset_id: str,
        asset_type: str,
        asset_path: str,
        dataset_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._asset_thumbnails:
            _logger.error(
                "Thumbnail generation skipped for %s: _asset_thumbnails is not initialized",
                asset_id,
            )
            return
        if not self._asset_index:
            _logger.error(
                "Thumbnail generation skipped for %s: _asset_index is not initialized",
                asset_id,
            )
            return
        if not asset_id:
            _logger.error("Thumbnail generation skipped: asset_id is empty")
            return

        def _warn_once(msg: str, *args) -> None:
            if asset_id not in self._thumbnail_warned_once:
                self._thumbnail_warned_once.add(asset_id)
                _logger.error(msg, *args)

        def _do_generate() -> None:
            try:
                thumb_path = None
                if asset_type == "dataset":
                    generate_dataset_preview = getattr(
                        self._asset_thumbnails,
                        "generate_dataset_preview",
                        None,
                    )
                    if callable(generate_dataset_preview):
                        thumb_path = self._maybe_await(
                            generate_dataset_preview(
                                asset_type,
                                asset_id,
                                asset_path,
                                dataset_metadata or {},
                            )
                        )
                        if thumb_path is None:
                            _warn_once(
                                "Dataset thumbnail generation returned None for %s (path=%s)",
                                asset_id,
                                asset_path,
                            )
                    else:
                        _warn_once(
                            "Dataset thumbnail generation unavailable for %s: generate_dataset_preview is not callable",
                            asset_id,
                        )
                else:
                    generate_rendered_preview = getattr(
                        self._asset_thumbnails,
                        "generate_rendered_preview",
                        None,
                    )
                    if callable(generate_rendered_preview):
                        thumb_path = self._maybe_await(
                            generate_rendered_preview(
                                asset_type,
                                asset_id,
                                asset_path,
                            )
                        )
                        if thumb_path is None:
                            _warn_once(
                                "Rendered thumbnail generation returned None for %s (type=%s, path=%s). "
                                "This usually means the renderer (lichtfeld.render_asset_preview) is missing or could not render the file.",
                                asset_id,
                                asset_type,
                                asset_path,
                            )
                    else:
                        _warn_once(
                            "Rendered thumbnail generation unavailable for %s: generate_rendered_preview is not callable",
                            asset_id,
                        )

                if thumb_path is None:
                    # Remember that rendered preview failed so we don't retry
                    # automatically on every on_update() cycle.
                    self._thumbnail_render_failed.add(asset_id)
                    _warn_once(
                        "Falling back to placeholder thumbnail for %s (type=%s, path=%s)",
                        asset_id,
                        asset_type,
                        asset_path,
                    )
                    thumb_path = self._maybe_await(
                        self._asset_thumbnails.generate_placeholder(
                            asset_type,
                            asset_id,
                        )
                    )
                    if thumb_path is None:
                        _warn_once(
                            "Placeholder thumbnail generation also failed for %s (type=%s)",
                            asset_id,
                            asset_type,
                        )
                        return

                # Success — clear the warning so future legitimate failures are reported
                self._thumbnail_warned_once.discard(asset_id)
                self._asset_index.update_asset(asset_id, thumbnail_path=str(thumb_path))
            except Exception as exc:
                _warn_once(
                    "Thumbnail generation failed for %s (type=%s, path=%s): %s: %s",
                    asset_id,
                    asset_type,
                    asset_path,
                    type(exc).__name__,
                    exc,
                )

        # Skip if a thumbnail thread for this asset is already running.
        with self._pending_thumbnail_lock:
            if asset_id in self._thumbnail_in_flight:
                return
            self._thumbnail_in_flight.add(asset_id)

        def _tracked_generate() -> None:
            with self._pending_thumbnail_lock:
                self._pending_thumbnail_threads.add(threading.current_thread())
            try:
                _do_generate()
            finally:
                with self._pending_thumbnail_lock:
                    self._pending_thumbnail_threads.discard(threading.current_thread())
                    self._thumbnail_in_flight.discard(asset_id)

        thread = threading.Thread(target=_tracked_generate, daemon=True)
        with self._pending_thumbnail_lock:
            self._pending_thumbnail_threads.add(thread)
        thread.start()

    def _join_pending_thumbnail_threads(self, timeout: float = 2.0) -> None:
        """Wait for background thumbnail generation threads to finish."""
        with self._pending_thumbnail_lock:
            threads = list(self._pending_thumbnail_threads)
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=timeout / max(len(threads), 1))
        with self._pending_thumbnail_lock:
            self._pending_thumbnail_threads = {
                thread for thread in self._pending_thumbnail_threads if thread.is_alive()
            }
            if not self._pending_thumbnail_threads:
                self._thumbnail_in_flight.clear()

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
        asset_type = str(asset.get("type") or "")
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
            if asset_id in self._thumbnail_render_failed:
                return False
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
        asset_type = str(asset.get("type") or "")
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
            needs_sh_degree = asset_type in ("ply_3dgs", "ply", "rad", "sog", "spz")
            return (
                not geom_meta
                or geom_meta.get("gaussian_count") is None
                or (needs_sh_degree and "sh_degree" not in geom_meta)
            )
        return asset.get("file_size_bytes", 0) <= 0

    # ── Background Asset Scanning ───────────────────────────────

    def _start_scan_worker(self, asset_ids: List[str], scan_type: str) -> None:
        """Start a background thread to scan assets and update the catalog.

        If a scan is already running, the request is requeued so it runs
        automatically after the current scan finishes.
        """
        asset_ids = list(dict.fromkeys(asset_ids))
        if not asset_ids:
            return

        thread_to_start = None
        with self._scan_thread_lock:
            if self._scan_thread is not None and self._scan_thread.is_alive():
                self._scan_queued_asset_ids = list(
                    dict.fromkeys(self._scan_queued_asset_ids + asset_ids)
                )
                self._scan_requeue = True
                self._scan_ui_refresh_needed = True
                return

            self._scan_requeue = False
            self._scan_queued_asset_ids = []
            self._scan_ui_refresh_needed = False
            self._scan_status = "scanning"
            self._scan_thread = threading.Thread(
                target=self._scan_worker,
                args=(asset_ids, scan_type),
                daemon=True,
            )
            thread_to_start = self._scan_thread

        thread_to_start.start()
        self._request_model_update()

    def _scan_worker(self, asset_ids: List[str], scan_type: str) -> None:
        """Run in a background thread: scan assets and update the catalog incrementally."""
        try:
            asset_ids = list(dict.fromkeys(asset_ids))
            while True:
                updated_any = False
                remaining_asset_ids: List[str] = []
                for asset_index, asset_id in enumerate(asset_ids):
                    with self._scan_thread_lock:
                        if self._scan_requeue:
                            # Another scan was requested; switch to the queued batch.
                            remaining_asset_ids = asset_ids[asset_index:]
                            break

                    asset = self._asset_index.assets.get(asset_id)
                    if asset is None:
                        continue

                    file_path = asset.get("absolute_path") or asset.get("path", "")
                    if not file_path or not os.path.exists(file_path):
                        continue

                    try:
                        metadata = self._asset_scanner.scan_file(file_path)
                    except Exception as exc:
                        _logger.debug(f"Failed to rescan asset metadata for {file_path}: {exc}")
                        metadata = None

                    if metadata:
                        # Merge new metadata into existing asset instead of overwriting
                        # so previously detected fields (like sh_degree) are not lost.
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
                        # Merge geometry/dataset metadata instead of replacing
                        for meta_key in ("geometry_metadata", "dataset_metadata", "transform_metadata"):
                            if meta_key in update_kwargs:
                                existing = asset.get(meta_key, {}) or {}
                                merged = dict(existing)
                                merged.update(update_kwargs[meta_key])
                                update_kwargs[meta_key] = merged
                        if update_kwargs:
                            self._asset_index.update_asset(asset_id, **update_kwargs)
                            updated_any = True

                    asset = self._asset_index.assets.get(asset_id, asset)
                    if self._asset_needs_thumbnail_refresh(asset):
                        self._generate_asset_thumbnail_for_values(
                            asset_id,
                            str(asset.get("type") or ""),
                            asset.get("absolute_path") or asset.get("path", ""),
                            asset.get("dataset_metadata", {}) or {},
                        )
                        updated_any = True

                    with self._scan_thread_lock:
                        self._scan_ui_refresh_needed = True

                if updated_any:
                    try:
                        self._asset_index.save()
                    except Exception as exc:
                        _logger.debug(f"Failed to save catalog after background scan: {exc}")

                next_asset_ids = None
                with self._scan_thread_lock:
                    if self._scan_requeue:
                        next_asset_ids = list(
                            dict.fromkeys(remaining_asset_ids + self._scan_queued_asset_ids)
                        )
                        self._scan_queued_asset_ids = []
                        self._scan_requeue = False
                        self._scan_ui_refresh_needed = True
                    else:
                        self._scan_ui_refresh_needed = True
                        self._scan_status = "idle"
                        if self._scan_thread is threading.current_thread():
                            self._scan_thread = None
                if next_asset_ids is not None:
                    asset_ids = next_asset_ids
                    continue
                self._request_model_update()
                break
        except Exception as exc:
            _logger.error(f"Background scan worker failed: {exc}")
            with self._scan_thread_lock:
                self._scan_status = "idle"
                self._scan_ui_refresh_needed = True
                self._scan_requeue = False
                self._scan_queued_asset_ids = []
                if self._scan_thread is threading.current_thread():
                    self._scan_thread = None
            self._request_model_update()

    def _sync_existing_asset_metadata(self) -> bool:
        """Non-blocking launcher: queue assets that need metadata sync for background scanning."""
        if not self._asset_index or not self._asset_scanner:
            return False
        asset_ids = [
            asset_id
            for asset_id, asset in list(self._asset_index.assets.items())
            if self._asset_needs_metadata_sync(asset)
        ]
        if asset_ids:
            self._start_scan_worker(asset_ids, "sync")
        return False

    def _scan_and_register_asset(
        self,
        path: str,
        *,
        folder_id: Optional[str],
        scene_id: Optional[str],
        fallback_role: str = "reference",
        override_type: Optional[str] = None,
        override_role: Optional[str] = None,
    ):
        if not folder_id:
            self._log_warn("Cannot register asset without a folder: %s", path)
            return None

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
            folder_id=folder_id,
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

    def _find_dataset_import_paths(self, path: str) -> List[str]:
        if not self._asset_scanner:
            return []
        detected_type = self._asset_scanner.detect_type(path)
        if detected_type == "dataset":
            return [str(Path(path).resolve())]

        datasets: List[str] = []
        seen: Set[str] = set()
        for metadata in self._asset_scanner.scan_directory(path, recursive=True):
            if metadata.get("type") != "dataset":
                continue
            metadata_path = metadata.get("path")
            if not metadata_path:
                continue
            resolved = str(Path(metadata_path).resolve())
            if resolved in seen:
                continue
            datasets.append(resolved)
            seen.add(resolved)
        return datasets

    def _drop_unknown_container_asset(self, path: str, folder_id: str) -> None:
        if not self._asset_index:
            return
        existing = self._asset_index.find_asset_by_path(
            str(Path(path).resolve()),
            folder_id=folder_id,
        )
        if existing is None or existing.type == "dataset":
            return
        if existing.type not in (None, "", "unknown"):
            return
        self._asset_index.delete_asset(existing.id)

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
            "show_selection_folder",
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

    # ── New Folder Handlers ──────────────────────────────────

    def toggle_new_folder_menu(self, _handle, _ev, _args):
        """Toggle the new folder dropdown menu visibility."""
        self._new_folder_menu_open = not self._new_folder_menu_open
        self._dirty_model("new_folder_menu_open")

    def on_create_folder_dialog(self, _handle, _ev, _args):
        """Open system dialog to create a new folder."""
        # Close the dropdown menu
        self._new_folder_menu_open = False
        self._dirty_model("new_folder_menu_open")

        def _on_folder_name_entered(name):
            self._create_folder_from_name(name)

        lf.ui.input_dialog(
            tr("asset_manager.dialog.create_new_folder"),
            tr("asset_manager.dialog.enter_folder_name"),
            "",
            _on_folder_name_entered
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

        def _continue_import(folder_id: str) -> None:
            file_path = lf.ui.open_ply_file_dialog("")
            if not file_path:
                return

            try:
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
                    folder_id=folder_id,
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

        self._with_import_folder(_continue_import)

    def on_import_mesh(self, _handle, _ev, args):
        """Import a mesh file (OBJ, FBX, GLTF, etc.)."""
        if not self._asset_index:
            _logger.warning("Asset index not initialized")
            return

        def _continue_import(folder_id: str) -> None:
            file_path = lf.ui.open_mesh_file_dialog("")
            if not file_path:
                return

            try:
                asset = self._scan_and_register_asset(
                    file_path,
                    folder_id=folder_id,
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

        self._with_import_folder(_continue_import)

    def on_import_dataset(self, _handle, _ev, args):
        """Import a dataset folder."""
        if not self._asset_index:
            _logger.warning("Asset index not initialized")
            return

        def _continue_import(folder_id: str) -> None:
            # Open folder dialog for datasets
            folder_path = lf.ui.open_dataset_folder_dialog()

            if not folder_path:
                return

            try:
                dataset_paths = self._find_dataset_import_paths(folder_path)
                if not dataset_paths:
                    _logger.warning(
                        "No importable dataset folders found under: %s",
                        folder_path,
                    )
                    return

                imported_assets = []
                for dataset_path in dataset_paths:
                    context = ensure_dataset_catalog_context(
                        dataset_path,
                        asset_index=self._asset_index,
                        scanner=self._asset_scanner,
                        thumbnails=self._asset_thumbnails,
                        folder_id=folder_id,
                    )
                    asset_id = context.get("asset_id")
                    asset = self._asset_index.get_asset(asset_id) if asset_id else None
                    if asset:
                        imported_assets.append(asset)

                if imported_assets and str(Path(folder_path).resolve()) not in dataset_paths:
                    self._drop_unknown_container_asset(folder_path, folder_id)

                # Link dataset to scene
                if imported_assets:
                    # Auto-select the newly imported dataset to show its info
                    # Add to selection instead of replacing (allow multiple imports)
                    self._selected_asset_ids.update(asset.id for asset in imported_assets)
                    # Preserve user's existing folder/scene filters - don't change them
                    # The dataset will appear in the catalog based on current filters
                    self._update_selection_type()
                self._import_menu_open = False

                # Refresh UI
                self.refresh_catalog()
                self._dirty_model("import_menu_open")
                self._update_selection_details()

                if imported_assets:
                    _logger.info(
                        "Imported %d dataset asset(s) from: %s",
                        len(imported_assets),
                        folder_path,
                    )

            except Exception as e:
                _logger.error(f"Failed to import dataset: {e}")

        self._with_import_folder(_continue_import)

    def on_load_selected(self, _handle, _ev, args):
        """Load selected asset(s) into the viewer."""
        if not self._selected_asset_ids:
            return

        for asset_id in sorted(self._selected_asset_ids):
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
                asset_type = str(asset.get("type") or "")
                if asset_type == "dataset":
                    if self._open_dataset_asset_import_panel(file_path, asset):
                        return
                    continue
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

    def select_folder(self, _handle, _ev, args):
        """Select a folder to filter scenes and assets."""
        folder_id = self._resolve_event_value(args, _ev, "data-folder-id")
        self._select_folder_id(folder_id)

    def _select_folder_id(self, folder_id: str) -> bool:
        if not folder_id:
            return False
        self._selected_folder_id = folder_id if folder_id != "all" else None
        self._selected_scene_id = None  # Clear scene selection when folder changes
        self._selected_asset_ids.clear()
        self._selection_type = "folder" if self._selected_folder_id else "none"

        self._dirty_model(
            "folders",
            "scenes",
            "assets",
            "selected_count",
            "selected_total_size",
            "selection_type",
            "selected_folder_name",
            "selected_folder_created",
            "selected_folder_modified",
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
            "selected_scene_folder_name",
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

        def _continue_import(folder_id: str) -> None:
            # Open file dialog for checkpoint
            file_path = lf.ui.open_checkpoint_file_dialog()

            if not file_path:
                return

            try:
                asset = self._scan_and_register_asset(
                    file_path,
                    folder_id=folder_id,
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

        self._with_import_folder(_continue_import)

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

    def _open_dataset_asset_import_panel(
        self,
        file_path: str,
        asset: Dict[str, Any],
        *,
        clear_scene_on_load: bool = False,
    ) -> bool:
        opened = open_dataset_import_panel(
            file_path,
            clear_scene_on_load=clear_scene_on_load,
        )
        if not opened:
            self._log_warn("Dataset import panel unavailable for: %s", file_path)
            return False
        self._log_info(
            "Opened dataset import panel for asset: %s",
            asset.get("name", "unknown"),
        )
        return True

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
            # Load based on asset type
            asset_type = str(asset.get("type") or "")
            if asset_type == "dataset":
                if not self._open_dataset_asset_import_panel(
                    file_path,
                    asset,
                    clear_scene_on_load=replace_scene,
                ):
                    return
            else:
                if replace_scene:
                    lf.clear_scene()
                # Regular mesh/splat file loading
                transform_node_name = self._load_asset_with_hierarchy(file_path)
                self._apply_asset_transform(asset, transform_node_name)
                self._log_info("Loaded asset: %s", asset.get("name", "unknown"))

            # Keep the requested asset selected; dataset assets finish loading
            # from the import panel after the user confirms its options.
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

    def _get_available_folders_for_asset(self, asset: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get list of folders this asset can be moved to."""
        if not self._asset_index or not hasattr(self._asset_index, "folders"):
            return []

        current_folder_id = asset.get("folder_id", "")
        folders = []

        for fld_id, fld in self._asset_index.folders.items():
            if fld_id != current_folder_id:
                folders.append({
                    "id": fld_id,
                    "name": fld.get("name", tr("asset_manager.unnamed_folder")),
                })

        # Sort by name
        return sorted(folders, key=lambda f: self._sort_text(f.get("name")))

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

        # Always reload folders when menu opens to ensure fresh data
        if self._handle:
            if self._open_menu_asset_id:
                folders = self.get_move_menu_folders()
                self._log_info("Loading %d folders for move menu", len(folders))
                self._handle.update_record_list("move_menu_folders", folders)
            else:
                self._handle.update_record_list("move_menu_folders", [])

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
        current_name = str(asset.get("name") or tr("asset_manager.unnamed"))

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

        # Allow re-logging and re-attempt on explicit user refresh
        self._thumbnail_warned_once.discard(asset_id)
        self._thumbnail_render_failed.discard(asset_id)

        # Close the menu
        self._open_menu_asset_id = None
        self._dirty_model("assets")

        asset_path = asset.get("absolute_path") or asset.get("path")
        if not asset_path:
            self._log_warn("Asset has no file path: %s", asset_id)
            return

        asset_type = str(asset.get("type") or "")
        if self._sort_text(asset_type) not in self.LOADABLE_TYPES:
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

            def _do_update() -> None:
                try:
                    thumb_path = self._maybe_await(
                        self._asset_thumbnails.generate_rendered_preview_from_camera(
                            asset_type,
                            asset_id,
                            asset_path,
                            eye=camera.eye,
                            target=camera.target,
                            up=camera.up,
                        )
                    )
                    if thumb_path is not None:
                        self._asset_index.update_asset(asset_id, thumbnail_path=str(thumb_path))
                        self._asset_index.save()
                        self._log_info("Updated thumbnail for %s from current camera", asset_id)
                    else:
                        self._log_warn("Failed to render thumbnail from camera for %s", asset_id)
                except Exception as exc:
                    self._log_error("Failed to update thumbnail: %s", exc)

            # Skip if a thumbnail thread for this asset is already running.
            with self._pending_thumbnail_lock:
                if asset_id in self._thumbnail_in_flight:
                    return
                self._thumbnail_in_flight.add(asset_id)

            def _tracked_update() -> None:
                with self._pending_thumbnail_lock:
                    self._pending_thumbnail_threads.add(threading.current_thread())
                try:
                    _do_update()
                finally:
                    with self._pending_thumbnail_lock:
                        self._pending_thumbnail_threads.discard(threading.current_thread())
                        self._thumbnail_in_flight.discard(asset_id)

            thread = threading.Thread(target=_tracked_update, daemon=True)
            with self._pending_thumbnail_lock:
                self._pending_thumbnail_threads.add(thread)
            thread.start()
        except Exception as e:
            self._log_error("Failed to update thumbnail: %s", e)

    def on_move_to_folder(self, _handle, _ev, args):
        """Move asset to a different folder."""
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

        # Get list of available folders
        if not hasattr(self._asset_index, "folders"):
            self._log_warn("No folders available")
            return

        folders = []
        for fld_id, fld in self._asset_index.folders.items():
            if fld_id != asset.get("folder_id"):  # Exclude current folder
                folders.append((fld_id, fld.get("name", "Unnamed")))

        if not folders:
            self._log_info("No other folders available to move to")
            return

        # Build folder list string
        folder_names = [f"{i+1}. {name}" for i, (_, name) in enumerate(folders)]
        folder_list = "\n".join(folder_names)
        current_folder = self._asset_index.folders.get(asset.get("folder_id", ""), {}).get("name", "Unknown")

        def _on_folder_selected(result):
            if not result or not result.strip():
                return

            try:
                # Parse selection (number or name)
                selection = result.strip()
                selected_folder_id = None
                selected_folder_name = None

                # Try to parse as number first
                try:
                    idx = int(selection.split(".")[0]) - 1
                    if 0 <= idx < len(folders):
                        selected_folder_id, selected_folder_name = folders[idx]
                except (ValueError, IndexError):
                    # Try to match by name
                    for fld_id, fld_name in folders:
                        if selection.lower() in self._sort_text(fld_name):
                            selected_folder_id = fld_id
                            selected_folder_name = fld_name
                            break

                if not selected_folder_id:
                    self._log_warn("Invalid folder selection: %s", selection)
                    return

                # Update asset's folder
                self._asset_index.update_asset(
                    asset_id,
                    folder_id=selected_folder_id,
                    scene_id=None  # Clear scene since scenes are folder-specific
                )
                self._asset_index.save()
                self.refresh_catalog()
                self._log_info("Moved asset to folder: %s", selected_folder_name)

            except Exception as e:
                self._log_error("Failed to move asset: %s", e)

        prompt = tr("asset_manager.dialog.current_folder", name=current_folder) + "\n\n"
        prompt += tr("asset_manager.dialog.available_folders") + "\n"
        prompt += folder_list + "\n\n"
        prompt += tr("asset_manager.dialog.enter_number_or_name")
        lf.ui.input_dialog(
            tr("asset_manager.dialog.move_to_folder"),
            prompt,
            "",
            _on_folder_selected
        )

    def _move_asset_to_folder(self, asset_id: str, folder_id: str) -> None:
        """Move asset to a specific folder."""
        self._log_info("Attempting to move asset %s to folder %s", asset_id, folder_id)

        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            self._log_warn("Asset index not available")
            return

        asset = self._asset_index.assets.get(asset_id)
        if not asset:
            self._log_warn("Asset not found: %s", asset_id)
            return

        folder = self._asset_index.folders.get(folder_id)
        if not folder:
            self._log_warn("Folder not found: %s", folder_id)
            return

        try:
            self._asset_index.update_asset(
                asset_id,
                folder_id=folder_id,
                scene_id=None  # Clear scene since scenes are folder-specific
            )
            self._asset_index.save()
            self.refresh_catalog()
            self._log_info("Moved asset to folder: %s", folder.get("name", "Unnamed"))
        except Exception as e:
            self._log_error("Failed to move asset: %s", e)

    def on_create_folder_and_move(self, _handle, _ev, args):
        """Create a new folder and move asset to it."""
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
        self._dirty_model("assets", "move_menu_folders")
        if self._handle:
            self._handle.update_record_list("move_menu_folders", [])

        def _on_folder_name_entered(name):
            if not name or not name.strip():
                return

            name = name.strip()

            try:
                # Create new folder
                folder = self._asset_index.create_folder(name=name)
                if not folder:
                    self._log_error("Failed to create folder")
                    return

                # Move asset to new folder
                self._asset_index.update_asset(
                    asset_id,
                    folder_id=folder.id,
                    scene_id=None
                )
                self._asset_index.save()
                self.refresh_catalog()
                self._log_info("Created folder '%s' and moved asset to it", name)

            except Exception as e:
                self._log_error("Failed to create folder and move asset: %s", e)

        lf.ui.input_dialog(
            tr("asset_manager.dialog.new_folder"),
            tr("asset_manager.dialog.enter_folder_name"),
            "",
            _on_folder_name_entered
        )

    def on_toggle_folder_menu(self, _handle, _ev, args):
        """Toggle dropdown menu for a folder."""
        folder_id = self._resolve_event_value(args, _ev, "data-folder-id")
        if not folder_id:
            return

        # Stop event propagation to prevent row selection
        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        # Toggle: if already open for this folder, close it; otherwise open for this folder
        if self._open_menu_folder_id == folder_id:
            self._open_menu_folder_id = None
        else:
            self._open_menu_folder_id = folder_id

        self._dirty_model("folders")

    def on_edit_watch_dirs(self, _handle, _ev, args):
        """Open the watched directories dialog for a folder."""
        folder_id = self._resolve_event_value(args, _ev, "data-folder-id")
        if not folder_id:
            return

        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        # Close the menu. Editing watch directories must not depend on or mutate
        # the current folder selection; the clicked row is the source of truth.
        self._open_menu_folder_id = None
        self._dirty_model("folders")

        ok = open_watch_dirs_dialog(folder_id)
        if not ok:
            self._log_warn("Failed to open watch dirs dialog for folder %s", folder_id)

    def on_rename_folder(self, _handle, _ev, args):
        """Open rename dialog for a folder."""
        folder_id = self._resolve_event_value(args, _ev, "data-folder-id")
        if not folder_id:
            return

        # Stop event propagation
        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        if not self._asset_index or not hasattr(self._asset_index, "folders"):
            return

        folder = self._asset_index.folders.get(folder_id)
        if not folder:
            return

        # Close the menu
        self._open_menu_folder_id = None
        self._dirty_model("folders")

        # Prompt for rename using input dialog
        current_name = folder.get("name", "Unnamed Folder")

        def _on_rename_result(new_name):
            if new_name and new_name.strip() and new_name.strip() != current_name:
                new_name = new_name.strip()
                try:
                    self._asset_index.update_folder(folder_id, name=new_name)
                    self._asset_index.save()
                    self.refresh_catalog()
                    self._log_info("Renamed folder to: %s", new_name)
                except Exception as e:
                    self._log_error("Failed to rename folder: %s", e)

        lf.ui.input_dialog(
            tr("asset_manager.dialog.rename_folder"),
            tr("asset_manager.dialog.enter_new_name", name=current_name),
            current_name,
            _on_rename_result
        )

    def on_delete_folder(self, _handle, _ev, args):
        """Delete a folder without creating an implicit fallback folder."""
        folder_id = self._resolve_event_value(args, _ev, "data-folder-id")
        if not folder_id:
            return

        # Stop event propagation
        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        if not self._asset_index or not hasattr(self._asset_index, "folders"):
            return

        folder = self._asset_index.folders.get(folder_id)
        if not folder:
            return

        # Close the menu
        self._open_menu_folder_id = None
        self._dirty_model("folders")

        folder_name = folder.get("name", "Unnamed Folder")

        assets_to_move = [
            asset_id
            for asset_id, asset in getattr(self._asset_index, "assets", {}).items()
            if asset.get("folder_id") == folder_id
        ]
        target_folder_id = None
        for fid in sorted(self._asset_index.folders.keys(), key=self._folder_sort_key):
            if fid != folder_id:
                target_folder_id = fid
                break

        if assets_to_move and not target_folder_id:
            self._log_warn(
                "Cannot delete folder '%s': create another folder or remove its assets first",
                folder_name,
            )
            return

        # Move all assets from this folder to another explicit folder before deleting.
        moved_count = 0
        target_folder_name = ""
        if target_folder_id:
            target_folder_name = self._asset_index.folders.get(
                target_folder_id, {}
            ).get("name", "another folder")
            for asset_id in assets_to_move:
                try:
                    self._asset_index.update_asset(
                        asset_id,
                        folder_id=target_folder_id,
                        scene_id=None  # Clear scene since scenes are folder-specific
                    )
                    moved_count += 1
                except Exception as e:
                    self._log_warn(
                        "Failed to move asset %s to folder '%s': %s",
                        asset_id,
                        target_folder_name,
                        e,
                    )
                    return

        # Delete the folder
        try:
            if hasattr(self._asset_index, "delete_folder"):
                self._asset_index.delete_folder(folder_id)
            elif hasattr(self._asset_index, "remove_folder"):
                self._asset_index.remove_folder(folder_id)
            else:
                # Fallback: remove from folders dict directly
                if hasattr(self._asset_index, "folders"):
                    del self._asset_index.folders[folder_id]

            self._asset_index.save()

            # Clear selection if the deleted folder was selected
            if self._selected_folder_id == folder_id:
                self._selected_folder_id = target_folder_id
                self._selected_scene_id = None
                self._selected_asset_ids.clear()
                self._selection_type = "folder" if target_folder_id else "none"
            self._repair_selected_folder()

            self.refresh_catalog()
            if moved_count:
                self._log_info(
                    "Deleted folder '%s' and moved %d assets to '%s'",
                    folder_name,
                    moved_count,
                    target_folder_name,
                )
            else:
                self._log_info("Deleted folder '%s'", folder_name)
        except Exception as e:
            self._log_error("Failed to delete folder: %s", e)

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
                self._sync_existing_asset_metadata()
                if self._asset_index.library_path.exists():
                    self._library_mtime = self._asset_index.library_path.stat().st_mtime
            except Exception as e:
                _logger.warning(f"Failed to load asset index: {e}")

        # Sync the currently loaded runtime dataset into the catalog when possible.
        # Only auto-select the current scene asset on first mount, not on reopen,
        # to preserve user's previous selection and show all assets.
        has_existing_selection = bool(self._selected_asset_ids)
        self._sync_runtime_scene_catalog(select_current=not has_existing_selection)

        # Clear scene filter on reopen to show all assets in the folder
        # (respecting active filters like Splat/PCL/Dataset/Checkpoint)
        if has_existing_selection:
            self._selected_scene_id = None

        # Initial refresh must dirty scalar bindings after catalog load.
        self.refresh_catalog()
        self._last_scene_generation = RuntimeState.scene_generation.value
        self._last_language_generation = RuntimeState.language_generation.value
        self._subscribe_reactive_state()
        _ensure_atexit_registered()

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
                    self._sync_existing_asset_metadata()
                    self._library_mtime = current_mtime
                    self.refresh_catalog(request_update=False)
                    changed = True

            # If a background scan just finished, refresh the UI (throttled).
            with self._scan_thread_lock:
                if self._scan_ui_refresh_needed:
                    now = time.time()
                    if now - self._scan_last_refresh_time > 0.2:
                        self._scan_ui_refresh_needed = False
                        self._scan_last_refresh_time = now
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

        # Auto-save: periodically persist catalog to disk so data survives
        # crashes or force-quits where on_unmount() is not called.
        try:
            now = time.time()
            if now - self._last_auto_save_time > self._auto_save_interval_sec:
                if self._asset_index and hasattr(self._asset_index, "save"):
                    saved = self._asset_index.save()
                    if saved and self._asset_index.library_path.exists():
                        self._library_mtime = self._asset_index.library_path.stat().st_mtime
                    self._last_auto_save_time = now
        except Exception:
            pass

        return changed

    def on_unmount(self, doc):
        """Save index on unmount."""
        self._unsubscribe_reactive_state()
        clear_active_asset_manager_panel(self)

        # Wait for any pending thumbnail generation threads to finish
        self._join_pending_thumbnail_threads(timeout=2.0)

        # Wait for any pending background scan to finish
        with self._scan_thread_lock:
            if self._scan_thread is not None and self._scan_thread.is_alive():
                self._scan_thread.join(timeout=2.0)

        if self._asset_index and hasattr(self._asset_index, "save"):
            try:
                saved = self._asset_index.save()
                if not saved:
                    _logger.error(
                        "Asset index save returned False during unmount (path=%s)",
                        getattr(self._asset_index, "library_path", "unknown"),
                    )
            except Exception as e:
                _logger.error(
                    "Failed to save asset index during unmount (path=%s): %s",
                    getattr(self._asset_index, "library_path", "unknown"),
                    e,
                    exc_info=True,
                )

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

        The generated asset/folder/scene rows are replaced by data-for updates.
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
            elif action == "move_to_folder":
                self.on_move_to_folder(None, event, [asset_id])
                self._stop_event(event)
                return
            elif action == "remove_from_menu":
                self._load_menu_asset_id = None
                self._open_menu_asset_id = None
                self.on_remove_asset(None, event, [asset_id])
                self._stop_event(event)
                return
            elif action == "create_folder":
                self.on_create_folder_and_move(None, event, [asset_id])
                # Close menu after creating folder
                self._open_menu_asset_id = None
                self._dirty_model("assets", "move_menu_folders")
                if self._handle:
                    self._handle.update_record_list("move_menu_folders", [])
                self._stop_event(event)
                return
            elif action == "move_to_existing_folder":
                folder_id = action_el.get_attribute("data-folder-id", "")
                self._log_info("Move to existing folder clicked: asset=%s, folder=%s", asset_id, folder_id)
                if folder_id:
                    self._move_asset_to_folder(asset_id, folder_id)
                    # Close menu after move
                    self._open_menu_asset_id = None
                    self._dirty_model("assets", "move_menu_folders")
                    if self._handle:
                        self._handle.update_record_list("move_menu_folders", [])
                else:
                    self._log_warn("No folder_id found on action element")
                self._stop_event(event)
                return
            elif action in ("select", "scene_asset"):
                # Close any open menu when selecting an asset
                if self._open_menu_asset_id:
                    self._open_menu_asset_id = None
                    self._dirty_model("assets", "move_menu_folders")
                    if self._handle:
                        self._handle.update_record_list("move_menu_folders", [])
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

        folder_el = rml_widgets.find_ancestor_with_attribute(
            target, "data-folder-id", container
        )
        if folder_el is not None:
            # Check if this is a folder action (menu, rename, delete)
            folder_action_el = rml_widgets.find_ancestor_with_attribute(
                target, "data-folder-action", container
            )
            if folder_action_el is not None:
                action = folder_action_el.get_attribute("data-folder-action", "")
                folder_id = folder_action_el.get_attribute("data-folder-id", "")

                if action == "menu":
                    self.on_toggle_folder_menu(None, event, [folder_id])
                    self._stop_event(event)
                    return
                elif action == "watch_dirs":
                    self.on_edit_watch_dirs(None, event, [folder_id])
                    self._stop_event(event)
                    return
                elif action == "rename":
                    self.on_rename_folder(None, event, [folder_id])
                    self._stop_event(event)
                    return
                elif action == "delete":
                    self.on_delete_folder(None, event, [folder_id])
                    self._stop_event(event)
                    return

            # Regular folder selection (not an action button)
            folder_id = folder_el.get_attribute("data-folder-id", "")
            # Close any open folder menu when selecting a folder
            if self._open_menu_folder_id:
                self._open_menu_folder_id = None
                self._dirty_model("folders")
            if self._select_folder_id(folder_id):
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
            self._dirty_model("assets", "move_menu_folders")
            if self._handle:
                self._handle.update_record_list("move_menu_folders", [])

        if self._load_menu_asset_id:
            self._load_menu_asset_id = None
            self._dirty_model("assets")

        # Close open folder menu when clicking elsewhere
        if self._open_menu_folder_id:
            self._open_menu_folder_id = None
            self._dirty_model("folders")

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
            self._open_menu_folder_id = None
            self._dirty_model("assets", "folders")
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
        self, folder_name: str, scene_name: str, parameters: Dict[str, Any]
    ) -> Optional[str]:
        """Called when training starts - create folder/scene context.

        Returns:
            Scene ID if created, None otherwise.
        """
        if not self._asset_index:
            return None

        try:
            # Create or get folder
            folder = self._asset_index.find_or_create_folder(folder_name)
            folder_id = folder.id

            # Create or get scene
            scene = self._asset_index.find_or_create_scene(folder_id, scene_name)
            scene_id = scene.id

            self._asset_index.save()

            # Update UI if panel is open
            self._selected_folder_id = folder_id
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
                folder_id=scene.get("folder_id"),
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
        folder_id: Optional[str] = None,
        scene_id: Optional[str] = None,
    ) -> Optional[str]:
        """Called when export is generated - register export asset.

        Args:
            file_path: Path to exported file
            export_type: Type of export (ply, rad, sog, spz, mp4, etc.)
            folder_id: Optional associated folder
            scene_id: Optional associated scene

        Returns:
            Asset ID if created, None otherwise.
        """
        if not self._asset_index:
            return None

        try:
            asset = self._scan_and_register_asset(
                folder_id=folder_id,
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
        """Non-blocking launcher: rescan all known assets in the background."""
        if not self._asset_index:
            return
        asset_ids = list(self._asset_index.assets.keys())
        if asset_ids:
            self._start_scan_worker(asset_ids, "refresh")
        self._log_info("Queued background catalog refresh (%d assets)", len(asset_ids))

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
            folder_id = self._repair_selected_folder()
            if not folder_id:
                return
            context = ensure_dataset_catalog_context(
                params.data_path,
                asset_index=self._asset_index,
                scanner=self._asset_scanner,
                thumbnails=self._asset_thumbnails,
                folder_id=folder_id,
            )
            if select_current and context.get("asset_id"):
                self._selected_asset_ids = {context["asset_id"]}
                # Preserve user's existing folder/scene filters - don't change them
                self._update_selection_type()
        except Exception as exc:
            _logger.debug("Failed to sync runtime scene catalog: %s", exc)

    def _update_all_record_lists(self):
        """Update all record lists in the data model."""
        if not self._handle:
            return

        self._handle.update_record_list("folders", self.get_folder_list())
        self._handle.update_record_list("scenes", self.get_scene_list())
        self._handle.update_record_list("filters", self.get_filter_list())
        # Note: "tags" record list removed - not bound in on_bind_model
        self._handle.update_record_list("assets", self.get_filtered_assets())

        # Update selection-specific record lists
        self._update_selection_details()

    def _update_selection_details(self):
        """Update record lists for selected scene and folder."""
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
                                    "name": str(
                                        asset.get("name")
                                        or tr("asset_manager.unnamed")
                                    ),
                                    "type": str(asset.get("type") or "").upper(),
                                }
                            )
                self._handle.update_record_list("selected_scene_assets", scene_assets)
            else:
                self._handle.update_record_list("selected_scene_assets", [])

            folder = self._get_selected_folder()
            if folder:
                scene_ids = folder.get("scene_ids", [])
                folder_scenes = []
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
                        folder_scenes.append(
                            {
                                "id": scene_id,
                                "name": scene_data.get(
                                    "name", tr("asset_manager.unnamed_scene")
                                ),
                                "asset_count": scene_asset_count,
                            }
                        )
                # Note: selected_folder_scenes record list removed - not used in UI
            # Note: selected_folder_scenes record list removed - not used in UI

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
            "selected_scene_folder_name",
            "selected_scene_asset_count",
            "selected_scene_assets",
            "selected_folder",
            "selected_folder_name",
            "selected_asset_tags",
            "show_selection_none",
            "show_selection_asset",
            "show_selection_scene",
            "show_selection_folder",
            "show_selection_multiple",
        }
        needs_selection_update = any(f in selection_fields for f in fields)

        for field in fields:
            self._handle.dirty(field)
            # Update record lists when they change
            if field in (
                "folders",
                "scenes",
                "filters",
                "assets",
                "selected_asset_tags",
            ):
                list_map = {
                    "folders": self.get_folder_list,
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
        self._with_import_folder(lambda _folder_id: open_url_import_panel())


# ── atexit backup ─────────────────────────────────────────

_atexit_registered = False


def _atexit_save_asset_manager() -> None:
    """Last-resort save when the process exits without on_unmount()."""
    try:
        from .asset_manager_integration import get_asset_manager_panel

        panel = get_asset_manager_panel()
        if panel is None:
            return
        index = getattr(panel, "_asset_index", None)
        if index is not None and hasattr(index, "save"):
            _logger.info("atexit: saving asset manager catalog to %s", index.library_path)
            saved = index.save()
            if not saved:
                _logger.error("atexit: asset manager save failed")
    except Exception:
        pass


def _ensure_atexit_registered() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_atexit_save_asset_manager)
        _atexit_registered = True
