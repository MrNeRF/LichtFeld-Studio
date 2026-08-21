# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager panel for browsing and managing LichtFeld projects."""

import atexit
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Any

import lichtfeld as lf

from . import rml_widgets
from .localization import localized_count
from .types import Panel
from .ui import RuntimeState
from .watch_dirs_panel import open_watch_dirs_dialog

_logger = logging.getLogger(__name__)
_active_asset_manager_panel = None

PRECISE_SCROLL_STEP = 32.0
ASSET_LIST_ROW_HEIGHT_DP = 44.0
ASSET_LIST_ROW_GAP_DP = 4.0
ASSET_LIST_WINDOW_FALLBACK_ROWS = 24
ASSET_LIST_WINDOW_OVERSCAN_ROWS = 6
ASSET_GALLERY_ROW_HEIGHT_DP = 220.0
ASSET_GALLERY_ROW_GAP_DP = 10.0
ASSET_GALLERY_WINDOW_FALLBACK_ROWS = 10
ASSET_GALLERY_WINDOW_OVERSCAN_ROWS = 2
ASSET_LIST_BOTTOM_SPACER_EXTRA_ROWS = 3
ASSET_GALLERY_BOTTOM_SPACER_EXTRA_ROWS = 1
SELECTION_DETAIL_DEFER_SECONDS = 0.035
ASSET_CARD_PREFERRED_WIDTH_DP = 208.0
ASSET_CARD_MIN_WIDTH_DP = 1.0
ASSET_CARD_GRID_HORIZONTAL_CHROME_DP = 48.0

ASSET_MANAGER_PERF_LOG_THRESHOLD_MS = 50.0

try:
    from .asset_index import (
        AssetIndex,
        is_supported_asset_path,
        resolve_asset_manager_storage_path,
    )
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    AssetIndex = None


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


__lfs_panel_classes__ = ["AssetManagerPanel"]
__lfs_panel_ids__ = ["lfs.asset_manager"]


class AssetManagerPanel(Panel):
    """Floating Asset Manager window for browsing LichtFeld projects."""

    SORT_MODES = ("name", "size")
    id = "lfs.asset_manager"
    label = "Asset Manager"
    space = lf.ui.PanelSpace.LEFT_DOCK
    order = 20
    template = "rmlui/asset_manager.rml"
    height_mode = lf.ui.PanelHeightMode.FILL
    size = (980, 620)
    options = {lf.ui.PanelOption.DEFAULT_CLOSED}
    update_policy = "dirty"

    STORAGE_PATH: Optional[Path] = None

    def _configure_storage_path(self) -> None:
        storage_path = resolve_asset_manager_storage_path()
        self.STORAGE_PATH = storage_path
        self.__class__.STORAGE_PATH = storage_path

    def __init__(self):
        self._handle = None
        self._doc = None

        # Backend components
        self._asset_index: Optional[Any] = None

        # UI state
        self._selected_asset_ids: Set[str] = set()
        self._selected_folder_id: Optional[str] = None
        self._selected_scene_id: Optional[str] = None
        self._view_mode: str = "list"  # gallery, list
        self._sort_mode: str = "name"
        self._search_query: str = ""
        self._last_asset_match_count = 0
        self._last_asset_visible_count = 0
        self._last_dirty_model_timing: Dict[str, Any] = {}
        self._last_asset_rows_update_count = 0
        self._last_asset_rows_update_ms = 0.0
        self._asset_filtered_cache_key: Optional[tuple] = None
        self._asset_filtered_cache: List[Dict[str, Any]] = []
        self._asset_window_scroll_top: float = 0.0
        self._asset_window_client_height: float = 0.0
        self._asset_window_client_width: float = 0.0
        self._asset_window_start_index: int = 0
        self._asset_window_end_index: int = 0
        self._asset_list_top_spacer_height: float = 0.0
        self._asset_list_bottom_spacer_height: float = 0.0
        self._asset_gallery_top_spacer_height: float = 0.0
        self._asset_gallery_bottom_spacer_height: float = 0.0
        self._asset_window_refresh_pending: bool = False
        self._asset_window_update_requested: bool = False
        self._asset_scroll_event_suppressed: bool = False
        self._asset_scroll_suppressed_top: float = -1.0
        self._catalog_assets_snapshot: Optional[Dict[str, Dict[str, Any]]] = None
        self._catalog_folders_snapshot: Optional[Dict[str, Dict[str, Any]]] = None
        self._catalog_scenes_snapshot: Optional[Dict[str, Dict[str, Any]]] = None
        self._catalog_stats_snapshot: Optional[Dict[str, Any]] = None
        self._selected_scene_assets_key: Optional[str] = None
        self._selection_detail_timer: Optional[threading.Timer] = None
        self._selection_detail_generation = 0
        self._selection_detail_lock = threading.Lock()
        self._pending_selection_detail_fields: tuple[str, ...] = ()
        self._pending_selection_detail_asset_id = ""
        self._pending_selection_detail_requested_at = 0.0

        # Track which asset has its dropdown menu open
        self._open_menu_asset_id: Optional[str] = None

        # Track which folder has its dropdown menu open
        self._open_menu_folder_id: Optional[str] = None

        # Selection type for info panel display
        self._selection_type: str = "none"  # none, asset, scene, folder, multiple

        self._library_mtime: float = 0.0
        self._updating_selection_details: bool = False
        self._reactive_unsubscribers = []
        self._last_scene_generation: Optional[int] = None
        self._last_language_generation: Optional[int] = None

        # Auto-save state
        self._auto_save_interval_sec: float = 30.0
        self._last_auto_save_time: float = 0.0

        # New folder menu state
        self._new_folder_menu_open: bool = False

        # Collapse state for sidebar sections
        self._folders_collapsed: bool = True

        # Panel resize drag state
        self._sidebar_dragging: bool = False
        self._sidebar_drag_start_y: float = 0.0
        self._sidebar_start_height: float = 176.0
        self._sidebar_resize_handle = None
        self._sidebar_height: float = 176.0
        self._right_panel_dragging: bool = False
        self._right_panel_drag_start_x: float = 0.0
        self._right_panel_start_width: float = 300.0
        self._right_panel_resize_handle = None
        self._right_panel_width: float = 300.0

        self._bottom_panel_dragging: bool = False
        self._bottom_panel_drag_start_y: float = 0.0
        self._bottom_panel_start_height: float = 220.0
        self._bottom_panel_resize_handle = None
        self._bottom_panel_height: float = 220.0
        self._asset_card_slot_width: float = ASSET_CARD_PREFERRED_WIDTH_DP

        # Dock state tracking (mirror histogram_panel pattern)
        self._panel_space = lf.ui.PanelSpace.LEFT_DOCK
        self._is_floating = False

    def capture_chrome(self):
        return {
            "folders_collapsed": bool(self._folders_collapsed),
            "sidebar_height": float(self._sidebar_height),
            "right_panel_width": float(self._right_panel_width),
            "bottom_panel_height": float(self._bottom_panel_height),
        }

    def apply_chrome(self, payload):
        self._folders_collapsed = True
        self._sidebar_height = 176.0
        self._right_panel_width = 300.0
        self._bottom_panel_height = 220.0
        if isinstance(payload, dict):
            if "folders_collapsed" in payload:
                self._folders_collapsed = bool(payload.get("folders_collapsed"))
            def _positive_float(key, current):
                value = payload.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    return float(value)
                return current

            self._sidebar_height = _positive_float("sidebar_height", self._sidebar_height)
            self._right_panel_width = _positive_float("right_panel_width", self._right_panel_width)
            self._bottom_panel_height = _positive_float("bottom_panel_height", self._bottom_panel_height)
        if self._handle:
            self._handle.dirty_all()

    # ── Initialization ────────────────────────────────────────

    def _initialize_backend(self):
        """Initialize backend components."""
        if not BACKEND_AVAILABLE:
            return False

        try:
            self._configure_storage_path()

            # Ensure storage directory exists
            self.STORAGE_PATH.mkdir(parents=True, exist_ok=True)

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

        # Panel dimensions for resizable sidebar and info panel
        model.bind_func("sidebar_height", lambda: f"{self._sidebar_height}dp")
        model.bind_func("right_panel_width", lambda: f"{self._right_panel_width}dp")
        model.bind_func("bottom_panel_height", lambda: f"{self._bottom_panel_height}dp")
        model.bind_func("sidebar_resize_dragging", lambda: self._sidebar_dragging)
        model.bind_func("right_panel_resize_dragging", lambda: self._right_panel_dragging)
        model.bind_func(
            "bottom_panel_resize_dragging", lambda: self._bottom_panel_dragging
        )
        model.bind_func(
            "asset_card_slot_width",
            lambda: f"{self._asset_card_slot_width:.1f}dp",
        )
        model.bind_func(
            "asset_list_top_spacer_height",
            lambda: f"{self._asset_list_top_spacer_height:.1f}dp",
        )
        model.bind_func(
            "asset_list_bottom_spacer_height",
            lambda: f"{self._asset_list_bottom_spacer_height:.1f}dp",
        )
        model.bind_func(
            "asset_gallery_top_spacer_height",
            lambda: f"{self._asset_gallery_top_spacer_height:.1f}dp",
        )
        model.bind_func(
            "asset_gallery_bottom_spacer_height",
            lambda: f"{self._asset_gallery_bottom_spacer_height:.1f}dp",
        )

        # Active states
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

        # Dock state (mirror histogram_panel pattern)
        model.bind_func("is_floating", lambda: self._is_floating)
        model.bind_func("close_label", lambda: tr("common.close"))

        model.bind_func("import_project_label", lambda: tr("menu.file.open_project"))


        # New folder menu state
        model.bind_func("new_folder_menu_open", self.get_new_folder_menu_open)
        model.bind_func("create_new_folder_label", lambda: tr("asset_manager.action.create_new_folder"))

        # Move menu folders list (for hover submenu)
        model.bind_record_list("move_menu_folders")

        # Selected IDs for UI conditionals
        model.bind_func("selected_folder_id", self.get_selected_folder_id)
        model.bind_func("selected_scene_id", self.get_selected_scene_id)
        model.bind_func("selected_asset_id", self.get_selected_asset_id)

        # Selection count and state
        model.bind_func("selected_count", self.get_selected_count)
        model.bind_func("selected_count_text", self.get_selected_count_text)
        model.bind_func("has_selection", self.get_has_selection)
        model.bind_func("has_multi_selection", self.get_has_multi_selection)

        # Selected asset properties (flattened bind_func pattern)
        model.bind_func("selected_asset_name", self.get_selected_asset_name)
        model.bind_func(
            "selected_asset_folder_name", self.get_selected_asset_folder_name
        )
        model.bind_func("selected_asset_scene_name", self.get_selected_asset_scene_name)
        model.bind_func("selected_asset_path", self.get_selected_asset_path)
        model.bind_func("selected_asset_size", self.get_selected_asset_size)
        model.bind_func("selected_asset_created", self.get_selected_asset_created)
        model.bind_func("selected_asset_modified", self.get_selected_asset_modified)
        model.bind_func(
            "selected_asset_file_missing", self.get_selected_asset_file_missing
        )
        model.bind_func(
            "selected_asset_expected_path", self.get_selected_asset_expected_path
        )

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
        model.bind_func("folders_title", lambda: tr("asset_manager.sidebar.folders"))
        model.bind_func("scenes_title", lambda: tr("asset_manager.sidebar.scenes"))
        model.bind_func("gallery_title", lambda: tr("asset_manager.toolbar.view_gallery"))
        model.bind_func("list_title", lambda: tr("asset_manager.toolbar.view_list"))
        model.bind_func("asset_results_summary", self.get_asset_results_summary)
        model.bind_func(
            "asset_results_summary_visible",
            self.get_asset_results_summary_visible,
        )
        model.bind_func(
            "edit_watch_dirs_label",
            lambda: tr("asset_manager.action.edit_watch_dirs"),
        )
        model.bind_func("rename_folder_label", lambda: tr("asset_manager.action.rename_folder"))
        model.bind_func("delete_folder_label", lambda: tr("asset_manager.action.delete_folder"))
        model.bind_func("load_button_label", lambda: tr("menu.file.open_project"))
        model.bind_func("rename_label", lambda: tr("asset_manager.action.rename"))
        model.bind_func("move_to_folder_label", lambda: tr("asset_manager.action.move_to_folder"))
        model.bind_func(
            "new_folder_label",
            lambda: f"{tr('asset_manager.action.new_folder')} and move here",
        )
        model.bind_func("show_in_folder_label", lambda: tr("asset_manager.action.show_in_folder"))
        model.bind_func("remove_label", lambda: tr("asset_manager.action.remove"))
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
        model.bind_func("prop_size_label", lambda: tr("asset_manager.property.size"))
        model.bind_func("prop_path_label", lambda: tr("asset_manager.property.path"))
        model.bind_func("prop_created_label", lambda: tr("asset_manager.property.created"))
        model.bind_func("prop_modified_label", lambda: tr("asset_manager.property.modified"))
        model.bind_func("file_not_found_title", self.get_selected_asset_verification_status)
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

        # Record lists for data-for loops (main lists)
        model.bind_record_list("folders")
        model.bind_record_list("scenes")
        model.bind_record_list("assets")

        # Record lists for nested struct lists
        model.bind_record_list("selected_scene_assets")

        self._handle = model.get_handle()

        # Initialize record lists
        self._update_all_record_lists()

        # Event handlers
        model.bind_event("set_view_mode", self.set_view_mode)
        model.bind_event("cycle_sort_mode", self.cycle_sort_mode)
        model.bind_event("toggle_asset_selection", self.toggle_asset_selection)
        model.bind_event("on_search", self.on_search)
        model.bind_event("on_import_project", self.on_import_project)
        model.bind_event("select_folder", self.select_folder)
        model.bind_event("select_scene", self.select_scene)
        model.bind_event("on_locate_file", self.on_locate_file)
        model.bind_event("select_asset", self.select_asset_by_id)
        model.bind_event("on_load_asset", self.on_load_asset)
        model.bind_event("on_remove_asset", self.on_remove_asset)

        # Panel resize event handlers
        model.bind_event("on_sidebar_resize_start", self.on_sidebar_resize_start)
        model.bind_event("on_right_panel_resize_start", self.on_right_panel_resize_start)
        model.bind_event("on_bottom_panel_resize_start", self.on_bottom_panel_resize_start)

        # New folder event handlers
        model.bind_event("toggle_new_folder_menu", self.toggle_new_folder_menu)
        model.bind_event("on_create_folder_dialog", self.on_create_folder_dialog)
        model.bind_event("refresh_catalog", self.verify_catalog)
        model.bind_event("clean_missing", self.clean_missing)

        # Collapse state bindings
        model.bind_func("folders_collapsed", self.get_folders_collapsed)
        model.bind_func("folders_expanded", self.get_folders_expanded)
        model.bind_event("toggle_folders_collapsed", self.toggle_folders_collapsed)

        # Close event
        model.bind_event("close_panel", self._on_close_panel)

    # ── Data Retrieval Methods ─────────────────────────────────

    def get_search_query(self) -> str:
        return self._search_query

    def set_search_query(self, value: str) -> None:
        self._search_query = value
        self._reset_asset_window_to_top()
        # Trigger asset list refresh when search query changes
        self._dirty_model("search_query", *self._asset_result_dirty_fields())

    def get_sort_label(self) -> str:
        labels = {
            "name": tr("asset_manager.toolbar.sort_by_name"),
            "size": tr("asset_manager.toolbar.sort_by_size"),
        }
        return labels.get(self._sort_mode, tr("asset_manager.toolbar.sort_by_name"))

    def get_selection_type(self) -> str:
        return self._selection_type

    def get_new_folder_menu_open(self) -> bool:
        return self._new_folder_menu_open

    def get_folders_collapsed(self) -> bool:
        return self._folders_collapsed

    def get_folders_expanded(self) -> bool:
        return not self._folders_collapsed

    def toggle_folders_collapsed(self, _handle=None, _ev=None, _args=None):
        self._folders_collapsed = not self._folders_collapsed
        self._dirty_model("folders_collapsed")
        self._dirty_model("folders_expanded")

    def get_move_menu_folders(self) -> List[Dict[str, str]]:
        """Get folders for the currently open move menu."""
        if not self._open_menu_asset_id or not self._asset_index:
            return []

        asset = self._asset_index_assets().get(self._open_menu_asset_id)
        if not asset:
            return []

        return self._get_available_folders_for_asset(asset)

    def get_selected_folder_id(self) -> Optional[str]:
        return self._selected_folder_id

    def get_selected_scene_id(self) -> Optional[str]:
        return self._selected_scene_id

    def get_selected_asset_id(self) -> str:
        if len(self._selected_asset_ids) != 1:
            return ""
        return next(iter(self._selected_asset_ids))

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
        return tr("asset_manager.status.multi_items_selected").format(count=count)

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
        assets = self._asset_index_assets()
        folders = self._asset_index_folders()
        scenes = self._asset_index_scenes()
        if not assets:
            self._selected_asset_ids.clear()
            if not folders:
                self._selected_folder_id = None
            if not scenes:
                self._selected_scene_id = None
            self._update_selection_type()
            if not folders and not scenes:
                return
        if (
            self._selected_folder_id
            and self._selected_folder_id
            not in folders
        ):
            self._selected_folder_id = None
        if (
            self._selected_scene_id
            and self._selected_scene_id not in scenes
        ):
            self._selected_scene_id = None
        valid_ids = set(assets.keys())
        if not self._selected_asset_ids.issubset(valid_ids):
            self._selected_asset_ids.intersection_update(valid_ids)
            self._update_selection_type()
        if not self._selected_asset_ids:
            if self._selection_type == "scene" and not self._selected_scene_id:
                self._selection_type = "none"
            elif self._selection_type == "folder" and not self._selected_folder_id:
                self._selection_type = "none"

    def _invalidate_catalog_cache(self) -> None:
        self._catalog_assets_snapshot = None
        self._catalog_folders_snapshot = None
        self._catalog_scenes_snapshot = None
        self._catalog_stats_snapshot = None
        self._asset_filtered_cache_key = None
        self._asset_filtered_cache = []

    def _asset_index_assets(self) -> Dict[str, Dict[str, Any]]:
        if not self._asset_index:
            return {}
        if self._catalog_assets_snapshot is None:
            list_projects = getattr(self._asset_index, "list_projects", None)
            if callable(list_projects):
                self._catalog_assets_snapshot = {
                    asset.id: getattr(asset, "__dict__", asset)
                    for asset in list_projects()
                }
                return self._catalog_assets_snapshot
            private_assets = getattr(self._asset_index, "_assets", None)
            if isinstance(private_assets, dict):
                # The UI hot path only reads values, so use a shallow dataclass
                # view instead of rebuilding serialized catalog rows.
                self._catalog_assets_snapshot = {
                    asset_id: getattr(asset, "__dict__", asset)
                    for asset_id, asset in private_assets.items()
                }
            else:
                try:
                    self._catalog_assets_snapshot = getattr(self._asset_index, "assets", {}) or {}
                except Exception:
                    self._catalog_assets_snapshot = {}
        return self._catalog_assets_snapshot

    def _asset_index_folders(self) -> Dict[str, Dict[str, Any]]:
        if not self._asset_index:
            return {}
        if self._catalog_folders_snapshot is None:
            private_folders = getattr(self._asset_index, "_folders", None)
            if isinstance(private_folders, dict):
                self._catalog_folders_snapshot = {
                    folder_id: getattr(folder, "__dict__", folder)
                    for folder_id, folder in private_folders.items()
                }
            else:
                try:
                    self._catalog_folders_snapshot = getattr(self._asset_index, "folders", {}) or {}
                except Exception:
                    self._catalog_folders_snapshot = {}
        return self._catalog_folders_snapshot

    def _asset_index_scenes(self) -> Dict[str, Dict[str, Any]]:
        if not self._asset_index:
            return {}
        if self._catalog_scenes_snapshot is None:
            private_scenes = getattr(self._asset_index, "_scenes", None)
            if isinstance(private_scenes, dict):
                self._catalog_scenes_snapshot = {
                    scene_id: getattr(scene, "__dict__", scene)
                    for scene_id, scene in private_scenes.items()
                }
            else:
                try:
                    self._catalog_scenes_snapshot = getattr(self._asset_index, "scenes", {}) or {}
                except Exception:
                    self._catalog_scenes_snapshot = {}
        return self._catalog_scenes_snapshot

    def _catalog_stats(self) -> Dict[str, Any]:
        if self._catalog_stats_snapshot is not None:
            return self._catalog_stats_snapshot

        folder_asset_counts: Dict[str, int] = {}
        scene_asset_counts: Dict[str, int] = {}
        assets_by_folder: Dict[str, List[Dict[str, Any]]] = {}

        for asset in self._asset_index_assets().values():
            folder_id = asset.get("folder_id")
            if folder_id:
                folder_asset_counts[folder_id] = folder_asset_counts.get(folder_id, 0) + 1
                assets_by_folder.setdefault(folder_id, []).append(asset)

            scene_id = asset.get("scene_id")
            if scene_id:
                scene_asset_counts[scene_id] = scene_asset_counts.get(scene_id, 0) + 1

        self._catalog_stats_snapshot = {
            "folder_asset_counts": folder_asset_counts,
            "scene_asset_counts": scene_asset_counts,
            "assets_by_folder": assets_by_folder,
        }
        return self._catalog_stats_snapshot

    def _folder_asset_counts(self) -> Dict[str, int]:
        return self._catalog_stats()["folder_asset_counts"]

    def _scene_asset_counts(self) -> Dict[str, int]:
        return self._catalog_stats()["scene_asset_counts"]

    def _folder_sort_key(self, folder_id: str) -> str:
        folder = self._asset_index_folders().get(folder_id, {})
        return self._sort_text(folder.get("name") or folder_id)

    def _default_folder_id(self) -> Optional[str]:
        for folder_id, folder in self._asset_index_folders().items():
            if self._sort_text(folder.get("name")).strip() == "default":
                return folder_id
        return None

    @staticmethod
    def _sort_text(value: Any) -> str:
        return str(value or "").lower()

    def _repair_selected_folder(self) -> Optional[str]:
        folders = self._asset_index_folders()
        if not folders:
            self._selected_folder_id = None
            self._selected_scene_id = None
            return None

        candidate_id: Optional[str] = None
        if self._selected_folder_id in folders:
            candidate_id = self._selected_folder_id

        scenes = self._asset_index_scenes()
        if not candidate_id and self._selected_scene_id:
            scene = scenes.get(self._selected_scene_id)
            scene_folder_id = scene.get("folder_id") if scene else None
            if scene_folder_id in folders:
                candidate_id = scene_folder_id

        assets = self._asset_index_assets()
        if not candidate_id:
            for asset_id in self._selected_asset_ids:
                asset = assets.get(asset_id)
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

        if self._selected_scene_id:
            scene = scenes.get(self._selected_scene_id)
            if not scene or scene.get("folder_id") != candidate_id:
                self._selected_scene_id = None
                if self._selection_type == "scene":
                    self._selection_type = "folder"
        if self._selected_asset_ids:
            visible_asset_ids = {
                aid
                for aid in self._selected_asset_ids
                if assets.get(aid, {}).get("folder_id") == candidate_id
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

        folder_name = self._asset_index_folders().get(asset.get("folder_id"), {}).get(
            "name", ""
        )
        scene_name = self._asset_index_scenes().get(asset.get("scene_id"), {}).get(
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
        display_name = self._asset_display_title(asset)

        if scene_name and scene_name != display_name:
            display_subtitle = scene_name
        elif folder_name:
            display_subtitle = folder_name
        else:
            display_subtitle = ""

        return {
            "display_name": display_name,
            "display_subtitle": display_subtitle,
        }

    def _reset_asset_window_to_top(self) -> None:
        scroll_el = self._asset_scroll_container()
        if scroll_el:
            try:
                scroll_el.scroll_top = 0.0
            except Exception:
                pass
        self._asset_window_scroll_top = 0.0
        self._asset_window_start_index = 0
        self._asset_window_end_index = 0
        self._asset_list_top_spacer_height = 0.0
        self._asset_list_bottom_spacer_height = 0.0
        self._asset_gallery_top_spacer_height = 0.0
        self._asset_gallery_bottom_spacer_height = 0.0
        self._asset_window_refresh_pending = False
        self._asset_window_update_requested = False

    def _request_asset_window_refresh(self) -> None:
        self._asset_window_refresh_pending = True
        if self._asset_window_update_requested:
            return
        self._asset_window_update_requested = True
        self._request_model_update()

    def _apply_asset_window_refresh(self, *, card_width_changed: bool = False) -> None:
        """Update the visible asset window without scheduling another panel refresh."""
        if not self._handle:
            return

        if card_width_changed:
            self._handle.dirty("asset_card_slot_width")

        for field in self._asset_window_dirty_fields():
            self._handle.dirty(field)

        records_start = time.perf_counter()
        rows = self.get_filtered_assets()
        self._handle.update_record_list("assets", rows)
        self._last_asset_rows_update_count = len(rows)
        self._last_asset_rows_update_ms = self._elapsed_ms(records_start)

    def _asset_result_dirty_fields(self) -> tuple[str, ...]:
        return (
            "assets",
            "asset_results_summary",
            "asset_results_summary_visible",
            "asset_list_top_spacer_height",
            "asset_list_bottom_spacer_height",
            "asset_gallery_top_spacer_height",
            "asset_gallery_bottom_spacer_height",
        )

    def _asset_window_dirty_fields(self) -> tuple[str, ...]:
        return (
            "assets",
            "asset_list_top_spacer_height",
            "asset_list_bottom_spacer_height",
            "asset_gallery_top_spacer_height",
            "asset_gallery_bottom_spacer_height",
        )

    def get_asset_results_summary_visible(self) -> bool:
        return self._last_asset_match_count > 0

    def get_asset_results_summary(self) -> str:
        total = self._last_asset_match_count
        if total <= 0:
            return ""
        return localized_count("asset_manager.status.showing_projects", total)

    def _asset_scroll_container(self, doc=None):
        root = doc or self._doc
        if not root:
            return None
        try:
            return root.get_element_by_id("asset-gallery-scroll")
        except Exception:
            return None

    def _sync_asset_window_viewport(self, doc=None) -> bool:
        scroll_el = self._asset_scroll_container(doc)
        if not scroll_el:
            return False

        try:
            next_scroll_top = max(0.0, float(scroll_el.scroll_top or 0.0))
            next_client_height = max(0.0, float(scroll_el.client_height or 0.0))
            next_client_width = max(0.0, float(scroll_el.client_width or 0.0))
        except Exception:
            return False

        if (
            abs(next_scroll_top - self._asset_window_scroll_top) <= 0.5
            and abs(next_client_height - self._asset_window_client_height) <= 0.5
            and abs(next_client_width - self._asset_window_client_width) <= 0.5
        ):
            return False

        self._asset_window_scroll_top = next_scroll_top
        self._asset_window_client_height = next_client_height
        self._asset_window_client_width = next_client_width

        folder_id = self._repair_selected_folder()
        if not folder_id:
            changed = (
                self._asset_window_start_index != 0
                or self._asset_window_end_index != 0
                or self._asset_list_top_spacer_height != 0.0
                or self._asset_list_bottom_spacer_height != 0.0
                or self._asset_gallery_top_spacer_height != 0.0
                or self._asset_gallery_bottom_spacer_height != 0.0
            )
            self._asset_window_start_index = 0
            self._asset_window_end_index = 0
            self._asset_list_top_spacer_height = 0.0
            self._asset_list_bottom_spacer_height = 0.0
            self._asset_gallery_top_spacer_height = 0.0
            self._asset_gallery_bottom_spacer_height = 0.0
            return changed

        total_count = len(self._get_filtered_assets_cache(folder_id))
        prev_state = (
            self._asset_window_start_index,
            self._asset_window_end_index,
            self._asset_list_top_spacer_height,
            self._asset_list_bottom_spacer_height,
            self._asset_gallery_top_spacer_height,
            self._asset_gallery_bottom_spacer_height,
        )
        self._compute_asset_window(total_count)
        next_state = (
            self._asset_window_start_index,
            self._asset_window_end_index,
            self._asset_list_top_spacer_height,
            self._asset_list_bottom_spacer_height,
            self._asset_gallery_top_spacer_height,
            self._asset_gallery_bottom_spacer_height,
        )
        return prev_state != next_state

    def _asset_filtered_cache_signature(self, folder_id: Optional[str]) -> tuple:
        return (
            folder_id,
            self._selected_scene_id,
            self._sort_mode,
            self._search_query,
        )

    def _get_filtered_assets_cache(self, folder_id: Optional[str]) -> List[Dict[str, Any]]:
        signature = self._asset_filtered_cache_signature(folder_id)
        if signature == self._asset_filtered_cache_key:
            return self._asset_filtered_cache

        assets_by_folder = self._catalog_stats()["assets_by_folder"]
        raw_assets = list(assets_by_folder.get(folder_id or "", []))
        matching_assets: List[Dict[str, Any]] = []
        search_query = self._search_query
        selected_scene_id = self._selected_scene_id
        for asset in raw_assets:
            if selected_scene_id and asset.get("scene_id") != selected_scene_id:
                continue
            if search_query and not self._asset_matches_query(asset, search_query):
                continue
            matching_assets.append(asset)

        self._asset_filtered_cache_key = signature
        self._asset_filtered_cache = self._sort_assets(matching_assets)
        return self._asset_filtered_cache

    def _compute_asset_window(
        self,
        total_count: int,
    ) -> tuple[int, int, float, float]:
        if total_count <= 0:
            self._asset_window_start_index = 0
            self._asset_window_end_index = 0
            self._asset_list_top_spacer_height = 0.0
            self._asset_list_bottom_spacer_height = 0.0
            self._asset_gallery_top_spacer_height = 0.0
            self._asset_gallery_bottom_spacer_height = 0.0
            return 0, 0, 0.0, 0.0

        if self._view_mode == "gallery":
            row_height = ASSET_GALLERY_ROW_HEIGHT_DP
            row_gap = ASSET_GALLERY_ROW_GAP_DP
            overscan_rows = ASSET_GALLERY_WINDOW_OVERSCAN_ROWS
            fallback_rows = ASSET_GALLERY_WINDOW_FALLBACK_ROWS
            available_width = max(
                ASSET_CARD_MIN_WIDTH_DP,
                self._asset_window_client_width - ASSET_CARD_GRID_HORIZONTAL_CHROME_DP,
            )
            slot_width = max(ASSET_CARD_MIN_WIDTH_DP, self._asset_card_slot_width)
            columns = max(
                1,
                int((available_width + row_gap) // (slot_width + row_gap)),
            )
            total_rows = int(math.ceil(total_count / float(columns)))
            scroll_row = int(self._asset_window_scroll_top // (row_height + row_gap))
            visible_rows = max(
                1,
                int(math.ceil(self._asset_window_client_height / (row_height + row_gap)))
                + overscan_rows * 2
                if self._asset_window_client_height > 0.0
                else fallback_rows,
            )
            start_row = max(0, scroll_row - overscan_rows)
            end_row = min(total_rows, start_row + visible_rows)
            start_index = min(total_count, start_row * columns)
            end_index = min(total_count, end_row * columns)
            top_spacer = float(start_row * (row_height + row_gap))
            bottom_spacer = float(
                max(0, total_rows - end_row + ASSET_GALLERY_BOTTOM_SPACER_EXTRA_ROWS)
                * (row_height + row_gap)
            )
            self._asset_window_start_index = start_index
            self._asset_window_end_index = end_index
            self._asset_gallery_top_spacer_height = top_spacer
            self._asset_gallery_bottom_spacer_height = bottom_spacer
            self._asset_list_top_spacer_height = 0.0
            self._asset_list_bottom_spacer_height = 0.0
            return start_index, end_index, top_spacer, bottom_spacer

        row_height = ASSET_LIST_ROW_HEIGHT_DP
        row_gap = ASSET_LIST_ROW_GAP_DP
        overscan_rows = ASSET_LIST_WINDOW_OVERSCAN_ROWS
        fallback_rows = ASSET_LIST_WINDOW_FALLBACK_ROWS
        row_pitch = row_height + row_gap
        scroll_row = int(self._asset_window_scroll_top // row_pitch)
        visible_rows = max(
            1,
            int(math.ceil(self._asset_window_client_height / row_pitch))
            + overscan_rows * 2
            if self._asset_window_client_height > 0.0
            else fallback_rows,
        )
        start_row = max(0, scroll_row - overscan_rows)
        end_row = min(total_count, start_row + visible_rows)
        top_spacer = float(start_row * row_pitch)
        bottom_spacer = float(
            max(0, total_count - end_row + ASSET_LIST_BOTTOM_SPACER_EXTRA_ROWS)
            * row_pitch
        )
        self._asset_window_start_index = start_row
        self._asset_window_end_index = end_row
        self._asset_list_top_spacer_height = top_spacer
        self._asset_list_bottom_spacer_height = bottom_spacer
        self._asset_gallery_top_spacer_height = 0.0
        self._asset_gallery_bottom_spacer_height = 0.0
        return start_row, end_row, top_spacer, bottom_spacer

    def get_filtered_assets(self) -> List[Dict[str, Any]]:
        """Return projects filtered by folder, scene, and search query."""
        assets_by_folder = self._catalog_stats()["assets_by_folder"]
        if not assets_by_folder:
            self._last_asset_match_count = 0
            self._last_asset_visible_count = 0
            self._asset_window_start_index = 0
            self._asset_window_end_index = 0
            self._asset_list_top_spacer_height = 0.0
            self._asset_list_bottom_spacer_height = 0.0
            self._asset_gallery_top_spacer_height = 0.0
            self._asset_gallery_bottom_spacer_height = 0.0
            return []

        folder_id = self._repair_selected_folder()
        if not folder_id:
            self._last_asset_match_count = 0
            self._last_asset_visible_count = 0
            self._asset_window_start_index = 0
            self._asset_window_end_index = 0
            self._asset_list_top_spacer_height = 0.0
            self._asset_list_bottom_spacer_height = 0.0
            self._asset_gallery_top_spacer_height = 0.0
            self._asset_gallery_bottom_spacer_height = 0.0
            return []

        sorted_assets = self._get_filtered_assets_cache(folder_id)
        total_count = len(sorted_assets)
        start_index, end_index, top_spacer, bottom_spacer = self._compute_asset_window(
            total_count
        )
        visible_assets = sorted_assets[start_index:end_index]
        self._last_asset_match_count = total_count
        self._last_asset_visible_count = len(visible_assets)
        return [
            self._format_asset_for_ui(asset)
            for asset in visible_assets
        ]

    def _asset_matches_query(self, asset: Dict[str, Any], query: str) -> bool:
        """Fuzzy search by asset name only.

        Matches if all characters in query appear in the asset name in order.
        Example: 'cty' matches 'city-project'.
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
        return sorted(assets, key=lambda a: self._sort_text(a.get("name")))

    @staticmethod
    def _verification_status_label(asset: Dict[str, Any]) -> str:
        disposition = asset.get("verification_disposition")
        if disposition == "CONTENT_MISMATCH":
            return tr("asset_manager.status.content_changed")
        if disposition == "TYPE_MISMATCH":
            return tr("asset_manager.status.type_changed")
        if disposition == "MISSING" or not asset.get("exists", True):
            return tr("asset_manager.status.missing")
        if not disposition:
            return tr("asset_manager.status.unverified")
        return tr("asset_manager.status.available")

    def _format_asset_for_ui(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        """Format asset data for UI display."""
        asset_id = str(asset.get("id") or "")
        asset_name = str(asset.get("name") or tr("asset_manager.unnamed"))
        file_size_bytes = self._coerce_nonnegative_int(
            asset.get("file_size_bytes", 0)
        )
        size_str = self._format_size(file_size_bytes)

        folder_name, scene_name = self._get_asset_relationship_names(asset)
        display_fields = self._get_asset_display_fields(
            asset, folder_name, scene_name
        )

        return {
            "id": asset_id,
            "name": asset_name,
            "display_name": display_fields["display_name"],
            "display_subtitle": display_fields["display_subtitle"],
            "size_label": size_str,
            "file_size_bytes": file_size_bytes,
            "is_selected": asset_id in self._selected_asset_ids,
            "exists": asset.get("exists", True),
            "status_label": self._verification_status_label(asset),
            "can_load": asset.get("exists", True),
            "folder_id": asset.get("folder_id"),
            "scene_id": asset.get("scene_id"),
            "folder_name": folder_name,
            "scene_name": scene_name,
            "modified_at": str(asset.get("modified_at") or ""),
            "modified_label": self._format_timestamp(asset.get("modified_at", "")),
            "menu_open": asset_id == self._open_menu_asset_id,
        }

    def get_folder_list(self) -> List[Dict[str, Any]]:
        """Return list of folders with asset counts for UI."""
        folders_index = self._asset_index_folders()
        if not folders_index:
            return []

        self._repair_selected_folder()

        folders = []
        asset_counts = self._folder_asset_counts()
        for folder_id, folder in folders_index.items():
            # Show all folders, even empty ones (user must manually delete)
            asset_count = asset_counts.get(folder_id, 0)
            display_name = self._format_display_name(folder.get("name", tr("asset_manager.unnamed_folder")))
            folders.append(
                {
                    "id": folder_id,
                    "name": display_name,
                    "full_name": folder.get("name", tr("asset_manager.unnamed_folder")),
                    "description": folder.get("description", ""),
                    "scene_count": asset_count,
                    "is_selected": folder_id == self._selected_folder_id,
                    "menu_open": folder_id == self._open_menu_folder_id,
                }
            )

        return sorted(folders, key=lambda f: self._sort_text(f.get("name")))

    def get_scene_list(self) -> List[Dict[str, Any]]:
        """Return list of scenes for selected folder."""
        scenes_index = self._asset_index_scenes()
        if not scenes_index:
            return []

        if not self._selected_folder_id:
            return []

        scenes = []
        asset_counts = self._scene_asset_counts()
        for scene_id, scene in scenes_index.items():
            if scene.get("folder_id") != self._selected_folder_id:
                continue
            # Show all scenes, even empty ones (user must manually delete)
            asset_count = asset_counts.get(scene_id, 0)
            scenes.append(
                {
                    "id": scene_id,
                    "name": scene.get("name", tr("asset_manager.unnamed_scene")),
                    "description": scene.get("description", ""),
                    "asset_count": asset_count,
                    "is_selected": scene_id == self._selected_scene_id,
                }
            )

        return sorted(scenes, key=lambda s: self._sort_text(s.get("name")))

    # ── Flattened Selected Asset Getters ─────────────────────

    def _get_selected_asset(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected single asset, if any."""
        if not self._selected_asset_ids or len(self._selected_asset_ids) != 1:
            return None
        asset_id = next(iter(self._selected_asset_ids))
        return self._asset_index_assets().get(asset_id)

    def get_selected_asset_name(self) -> str:
        asset = self._get_selected_asset()
        return self._asset_display_title(asset) if asset else ""

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


    def get_selected_asset_file_missing(self) -> bool:
        asset = self._get_selected_asset()
        if not asset:
            return False
        return not asset.get("exists", True)

    def get_selected_asset_verification_status(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        disposition = asset.get("verification_disposition")
        if disposition == "CONTENT_MISMATCH":
            return tr("asset_manager.status.content_changed")
        if disposition == "TYPE_MISMATCH":
            return tr("asset_manager.status.type_changed")
        if disposition == "MISSING":
            return tr("asset_manager.info_panel.file_not_found")
        return self._verification_status_label(asset)

    def get_selected_asset_expected_path(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        file_exists = asset.get("exists", True)
        if file_exists:
            return ""
        return asset.get("absolute_path") or asset.get("path", "")

    # ── Flattened Selected Scene Getters ───────────────────────

    def _get_selected_scene(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected scene, if any."""
        if not self._selected_scene_id:
            return None
        return self._asset_index_scenes().get(self._selected_scene_id)

    def get_selected_scene_name(self) -> str:
        scene = self._get_selected_scene()
        return scene.get("name", "") if scene else ""

    def get_selected_scene_folder_name(self) -> str:
        scene = self._get_selected_scene()
        if not scene:
            return ""
        folder_id = scene.get("folder_id", "")
        if not folder_id:
            return ""
        folder = self._asset_index_folders().get(folder_id)
        name = folder.get("name", "") if folder else ""
        return self._format_display_name(name)

    def get_selected_scene_asset_count(self) -> int:
        scene = self._get_selected_scene()
        if not scene or not self._asset_index:
            return 0
        scene_id = scene.get("id", "")
        if not scene_id:
            return 0
        return self._scene_asset_counts().get(scene_id, 0)

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
        return self._asset_index_folders().get(self._selected_folder_id)

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

    def on_import_project(self, _handle, _ev, _args):
        """Add one .licht project to the catalog by content identity."""
        if not self._asset_index:
            self._log_warn("Asset index not initialized")
            return

        def _continue_import(folder_id: str) -> None:
            file_path = lf.ui.open_project_file_dialog("")
            if not file_path:
                return
            if not is_supported_asset_path(file_path):
                self._log_warn("Asset Manager only supports .licht projects: %s", file_path)
                return

            try:
                asset, created = self._asset_index.register_licht_asset(
                    file_path,
                    folder_id=folder_id,
                )
                if asset is None:
                    self._log_error("Failed to register project: %s", file_path)
                    return
                self._selected_asset_ids = {asset.id}
                self._selection_type = "asset"
                self.refresh_catalog()
                if created:
                    self._log_info("Imported project: %s", asset.name)
                else:
                    self._log_info("Project content is already in the catalog: %s", asset.name)
            except Exception as exc:
                self._log_error("Failed to import project: %s", exc)

        self._with_import_folder(_continue_import)

    def set_view_mode(self, _handle, _ev, args):
        """Set the view mode (gallery or list)."""
        if not args:
            return
        mode = str(args[0])
        if mode not in ("gallery", "list"):
            return
        self._view_mode = mode
        self._reset_asset_window_to_top()
        self._dirty_model(
            "view_mode",
            "is_gallery_view",
            "is_list_view",
            *self._asset_result_dirty_fields(),
        )

    def cycle_sort_mode(self, _handle, _ev, args):
        """Cycle through supported sort modes."""
        try:
            current_index = self.SORT_MODES.index(self._sort_mode)
        except ValueError:
            current_index = 0
        self._sort_mode = self.SORT_MODES[(current_index + 1) % len(self.SORT_MODES)]
        self._reset_asset_window_to_top()
        self._dirty_model(
            "sort_mode",
            "sort_label",
            *self._asset_result_dirty_fields(),
        )

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

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000.0

    def _log_perf(self, message: str, *args: Any, elapsed_ms: Optional[float] = None) -> None:
        if elapsed_ms is not None and elapsed_ms < ASSET_MANAGER_PERF_LOG_THRESHOLD_MS:
            return
        if args:
            try:
                message = message % args
            except Exception:
                pass
        prefixed = "[AssetManagerPerf] " + message
        try:
            lf.log.info(prefixed)
        except Exception:
            _logger.info(prefixed)

    def _selection_count_fields(self) -> tuple[str, ...]:
        return (
            "selected_count",
            "selected_count_text",
            "has_selection",
            "has_multi_selection",
        )

    def _selected_asset_detail_fields(self) -> tuple[str, ...]:
        return (
            "selected_asset_name",
            "selected_asset_folder_name",
            "selected_asset_scene_name",
            "selected_asset_path",
            "selected_asset_size",
            "selected_asset_created",
            "selected_asset_modified",
            "selected_asset_file_missing",
            "selected_asset_expected_path",
        )

    def _selected_scene_detail_fields(self) -> tuple[str, ...]:
        return (
            "selected_scene_name",
            "selected_scene_folder_name",
            "selected_scene_asset_count",
            "selected_scene_created",
            "selected_scene_modified",
            "selected_scene_assets",
        )

    def _selected_folder_detail_fields(self) -> tuple[str, ...]:
        return (
            "selected_folder_name",
            "selected_folder_created",
            "selected_folder_modified",
        )

    def _selected_asset_dirty_fields(
        self,
        previous_selection: Set[str],
        current_selection: Set[str],
    ) -> tuple[str, ...]:
        fields = [
            "selected_asset_id",
            *self._selection_count_fields(),
            *self._selection_visibility_fields(),
            *self._selected_asset_detail_fields(),
        ]
        if len(previous_selection) > 1 or len(current_selection) > 1:
            fields.insert(0, "assets")
        return tuple(fields)

    @staticmethod
    def _ui_thread_scheduler():
        scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
        if scheduler is None:
            scheduler = getattr(lf.ui, "_run_on_ui_thread", None)
        return scheduler if callable(scheduler) else None

    def _cancel_selection_detail_timer(self) -> None:
        timer = self._selection_detail_timer
        self._selection_detail_timer = None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

    def _schedule_selection_detail_update(
        self,
        fields: tuple[str, ...],
        *,
        asset_id: str,
        requested_at: float,
    ) -> bool:
        scheduler = self._ui_thread_scheduler()
        if scheduler is None:
            return False

        with self._selection_detail_lock:
            self._selection_detail_generation += 1
            generation = self._selection_detail_generation
            self._pending_selection_detail_fields = fields
            self._pending_selection_detail_asset_id = asset_id
            self._pending_selection_detail_requested_at = requested_at
            self._cancel_selection_detail_timer()

            def fire() -> None:
                def flush() -> None:
                    self._flush_selection_detail_update(generation)

                try:
                    scheduler(flush)
                except Exception:
                    pass

            timer = threading.Timer(SELECTION_DETAIL_DEFER_SECONDS, fire)
            timer.daemon = True
            self._selection_detail_timer = timer
            timer.start()
        return True

    def _flush_selection_detail_update(self, generation: Optional[int] = None) -> bool:
        with self._selection_detail_lock:
            if generation is not None and generation != self._selection_detail_generation:
                return False
            fields = self._pending_selection_detail_fields
            asset_id = self._pending_selection_detail_asset_id
            requested_at = self._pending_selection_detail_requested_at
            self._pending_selection_detail_fields = ()
            self._pending_selection_detail_asset_id = ""
            self._pending_selection_detail_requested_at = 0.0
            self._selection_detail_timer = None

        if not fields:
            return False

        start = time.perf_counter()
        self._dirty_model(*fields)
        dirty_ms = self._elapsed_ms(start)
        wait_ms = (start - requested_at) * 1000.0 if requested_at else 0.0
        dirty_timing = self._last_dirty_model_timing or {}
        self._log_perf(
            (
                "select_details asset=%s wait=%.3fms dirty=%.3fms "
                "fields=%d records=%.3fms/%s request=%.3fms total=%.3fms"
            ),
            asset_id,
            wait_ms,
            dirty_ms,
            dirty_timing.get("field_count", len(fields)),
            dirty_timing.get("record_update_ms", 0.0),
            dirty_timing.get("record_updates", {}),
            dirty_timing.get("request_update_ms", 0.0),
            dirty_timing.get("total_ms", dirty_ms),
            elapsed_ms=dirty_ms,
        )
        return True

    def _select_asset_id(
        self,
        asset_id: str,
        *,
        toggle: bool = False,
        multi_select: bool = False,
        row_element: Any = None,
        container: Any = None,
    ) -> bool:
        total_start = time.perf_counter()
        if not asset_id:
            self._log_warn(
                "Asset Manager click ignored: no asset id resolved from event/DOM"
            )
            return False

        assets = self._asset_index_assets()
        asset = assets.get(asset_id)
        if asset is None:
            available = list(assets.keys())[:10]
            self._log_warn(
                "Asset Manager click resolved asset_id=%s but asset is missing "
                "from index. sample_ids=%s",
                asset_id,
                available,
            )
            return False

        previous_selection = set(self._selected_asset_ids)
        previous_type = self._selection_type

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
        if (
            self._selected_asset_ids == previous_selection
            and self._selection_type == previous_type
        ):
            self._log_perf(
                "select noop asset=%s total=%.3fms",
                asset_id,
                self._elapsed_ms(total_start),
                elapsed_ms=self._elapsed_ms(total_start),
            )
            return False
        dom_start = time.perf_counter()
        dom_rows = self._sync_asset_selection_dom(
            previous_selection,
            self._selected_asset_ids,
            row_element=row_element,
            container=container,
        )
        dom_ms = self._elapsed_ms(dom_start)

        detail_fields = self._selected_asset_dirty_fields(
            previous_selection,
            self._selected_asset_ids,
        )
        dirty_start = time.perf_counter()
        deferred = self._schedule_selection_detail_update(
            detail_fields,
            asset_id=asset_id,
            requested_at=total_start,
        )
        if not deferred:
            self._dirty_model(*detail_fields)
        dirty_ms = self._elapsed_ms(dirty_start)
        total_ms = self._elapsed_ms(total_start)
        dirty_timing = self._last_dirty_model_timing or {}
        self._log_perf(
            (
                "select asset=%s multi=%s previous=%d current=%d "
                "dom=%.3fms/%drows deferred=%s dirty=%.3fms fields=%d "
                "records=%.3fms/%s request=%.3fms total=%.3fms"
            ),
            asset_id,
            multi_select,
            len(previous_selection),
            len(self._selected_asset_ids),
            dom_ms,
            dom_rows,
            deferred,
            dirty_ms,
            0 if deferred else dirty_timing.get("field_count", len(detail_fields)),
            0.0 if deferred else dirty_timing.get("record_update_ms", 0.0),
            {} if deferred else dirty_timing.get("record_updates", {}),
            0.0 if deferred else dirty_timing.get("request_update_ms", 0.0),
            total_ms,
            elapsed_ms=total_ms,
        )
        return True

    def _update_selection_type(self):
        """Update selection type based on current selection."""
        if not self._selected_asset_ids:
            self._selection_type = "none"
        elif len(self._selected_asset_ids) == 1:
            self._selection_type = "asset"
        else:
            self._selection_type = "multiple"

    def _query_visible_asset_rows(self, root: Any) -> List[Any]:
        if root is None or not hasattr(root, "query_selector_all"):
            return []
        rows: List[Any] = []
        for selector in (".asset-card", ".asset-list-row", ".scene-asset-row"):
            try:
                rows.extend(list(root.query_selector_all(selector)))
            except Exception:
                continue
        return rows

    def _sync_asset_selection_dom(
        self,
        previous_selection: Set[str],
        current_selection: Set[str],
        *,
        row_element: Any = None,
        container: Any = None,
    ) -> int:
        root = container or self._doc
        rows = self._query_visible_asset_rows(root)
        if row_element is not None and row_element not in rows:
            rows.append(row_element)
        if not rows:
            return 0

        current = {str(asset_id) for asset_id in current_selection}
        selected_class = "is-multi-selected" if len(current) > 1 else "is-selected"
        changed = 0
        for row in rows:
            try:
                asset_id = row.get_attribute("data-asset-id", "")
            except Exception:
                asset_id = ""
            is_selected = asset_id in current
            for class_name in ("is-selected", "is-multi-selected"):
                try:
                    should_set = is_selected and class_name == selected_class
                    if row.is_class_set(class_name) != should_set:
                        row.set_class(class_name, should_set)
                        changed += 1
                except Exception:
                    continue
        return changed

    def on_search(self, _handle, _ev, args):
        """Handle search input changes (real-time)."""
        if args and len(args) > 0:
            self._search_query = str(args[0])
        self._reset_asset_window_to_top()
        self._dirty_model("search_query", *self._asset_result_dirty_fields())

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
        self._sidebar_drag_start_y = float(event.get_parameter("mouse_y", "0"))
        self._sidebar_start_height = self._sidebar_height
        self._sidebar_resize_handle = _handle
        if _handle is not None:
            try:
                _handle.set_class("dragging", True)
            except Exception:
                pass
        event.stop_propagation()

    def on_sidebar_resize_delta(self, mouse_y: float) -> None:
        """Update sidebar height during drag."""
        if not self._sidebar_dragging:
            return
        delta_y = mouse_y - self._sidebar_drag_start_y
        new_height = self._sidebar_start_height + delta_y
        # Enforce minimum height of 120dp and maximum of 400dp
        new_height = max(120.0, min(400.0, new_height))
        self._sidebar_height = new_height
        # The height is bound via data-style-height, so just dirty the model
        self._dirty_model("sidebar_height")

    def on_sidebar_resize_end(self, handle=None) -> None:
        """End sidebar resize drag."""
        self._sidebar_dragging = False
        handle = handle or self._sidebar_resize_handle
        if handle is not None:
            try:
                handle.set_class("dragging", False)
            except Exception:
                pass
        self._sidebar_resize_handle = None

    def on_right_panel_resize_start(self, _handle, event, _args):
        """Start dragging the right panel resize handle."""
        self._right_panel_dragging = True
        self._right_panel_drag_start_x = float(event.get_parameter("mouse_x", "0"))
        # Use the current width from instance variable
        self._right_panel_start_width = self._right_panel_width
        self._right_panel_resize_handle = _handle
        if _handle is not None:
            try:
                _handle.set_class("dragging", True)
            except Exception:
                pass
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
        handle = self._right_panel_resize_handle
        if handle is not None:
            try:
                handle.set_class("dragging", False)
            except Exception:
                pass
        self._right_panel_resize_handle = None

    def on_bottom_panel_resize_start(self, _handle, event, _args):
        """Start dragging the bottom panel resize handle."""
        self._bottom_panel_dragging = True
        self._bottom_panel_drag_start_y = float(event.get_parameter("mouse_y", "0"))
        self._bottom_panel_start_height = self._bottom_panel_height
        self._bottom_panel_resize_handle = _handle
        if _handle is not None:
            try:
                _handle.set_class("dragging", True)
            except Exception:
                pass
        event.stop_propagation()

    def on_bottom_panel_resize_delta(self, mouse_y: float) -> None:
        """Update bottom panel height during drag."""
        if not self._bottom_panel_dragging:
            return
        delta_y = self._bottom_panel_drag_start_y - mouse_y
        new_height = self._bottom_panel_start_height + delta_y
        # Enforce min/max height
        new_height = max(120.0, min(400.0, new_height))
        self._bottom_panel_height = new_height
        self._dirty_model("bottom_panel_height")

    def on_bottom_panel_resize_end(self, handle=None) -> None:
        """End bottom panel resize drag."""
        self._bottom_panel_dragging = False
        handle = handle or self._bottom_panel_resize_handle
        if handle is not None:
            try:
                handle.set_class("dragging", False)
            except Exception:
                pass
        self._bottom_panel_resize_handle = None

    def _delete_asset_from_catalog(self, asset_id: str) -> bool:
        if not self._asset_index:
            return False
        return bool(self._asset_index.delete_asset(asset_id))

    def select_folder(self, _handle, _ev, args):
        """Select a folder to filter scenes and assets."""
        folder_id = self._resolve_event_value(args, _ev, "data-folder-id")
        self._select_folder_id(folder_id)

    def _select_folder_id(self, folder_id: str) -> bool:
        total_start = time.perf_counter()
        if not folder_id:
            return False
        next_folder_id = folder_id if folder_id != "all" else None
        next_selection_type = "folder" if next_folder_id else "none"
        if (
            self._selected_folder_id == next_folder_id
            and self._selected_scene_id is None
            and not self._selected_asset_ids
            and self._selection_type == next_selection_type
        ):
            self._log_perf(
                "folder noop folder=%s total=%.3fms",
                folder_id,
                self._elapsed_ms(total_start),
                elapsed_ms=self._elapsed_ms(total_start),
            )
            return False

        self._selected_folder_id = next_folder_id
        self._selected_scene_id = None  # Clear scene selection when folder changes
        self._selected_asset_ids.clear()
        self._selection_type = next_selection_type
        self._reset_asset_window_to_top()

        self._dirty_model(
            "folders",
            "scenes",
            *self._asset_result_dirty_fields(),
            "selected_folder_id",
            "selected_scene_id",
            "selected_asset_id",
            *self._selection_count_fields(),
            *self._selection_visibility_fields(),
            *self._selected_asset_detail_fields(),
            *self._selected_scene_detail_fields(),
            *self._selected_folder_detail_fields(),
        )
        dirty_timing = self._last_dirty_model_timing or {}
        total_ms = self._elapsed_ms(total_start)
        self._log_perf(
            (
                "folder folder=%s rows=%d rows_ms=%.3f dirty_total=%.3fms "
                "records=%.3fms/%s request=%.3fms total=%.3fms"
            ),
            folder_id,
            self._last_asset_rows_update_count,
            self._last_asset_rows_update_ms,
            dirty_timing.get("total_ms", 0.0),
            dirty_timing.get("record_update_ms", 0.0),
            dirty_timing.get("record_updates", {}),
            dirty_timing.get("request_update_ms", 0.0),
            total_ms,
            elapsed_ms=total_ms,
        )
        return True

    def select_scene(self, _handle, _ev, args):
        """Select a scene to filter assets."""
        scene_id = self._resolve_event_value(args, _ev, "data-scene-id")
        self._select_scene_id(scene_id)

    def _select_scene_id(self, scene_id: str) -> bool:
        total_start = time.perf_counter()
        if not scene_id:
            return False
        next_scene_id = scene_id if scene_id != "all" else None
        next_selection_type = "scene" if next_scene_id else "none"
        if (
            self._selected_scene_id == next_scene_id
            and not self._selected_asset_ids
            and self._selection_type == next_selection_type
        ):
            self._log_perf(
                "scene noop scene=%s total=%.3fms",
                scene_id,
                self._elapsed_ms(total_start),
                elapsed_ms=self._elapsed_ms(total_start),
            )
            return False

        self._selected_scene_id = next_scene_id
        self._selected_asset_ids.clear()
        self._selection_type = next_selection_type
        self._reset_asset_window_to_top()

        self._dirty_model(
            "scenes",
            *self._asset_result_dirty_fields(),
            "selected_scene_id",
            "selected_asset_id",
            *self._selection_count_fields(),
            *self._selection_visibility_fields(),
            *self._selected_asset_detail_fields(),
            *self._selected_scene_detail_fields(),
            *self._selected_folder_detail_fields(),
        )
        dirty_timing = self._last_dirty_model_timing or {}
        total_ms = self._elapsed_ms(total_start)
        self._log_perf(
            (
                "scene scene=%s rows=%d rows_ms=%.3f dirty_total=%.3fms "
                "records=%.3fms/%s request=%.3fms total=%.3fms"
            ),
            scene_id,
            self._last_asset_rows_update_count,
            self._last_asset_rows_update_ms,
            dirty_timing.get("total_ms", 0.0),
            dirty_timing.get("record_update_ms", 0.0),
            dirty_timing.get("record_updates", {}),
            dirty_timing.get("request_update_ms", 0.0),
            total_ms,
            elapsed_ms=total_ms,
        )
        return True

    def on_locate_file(self, _handle, _ev, args):
        """Relink a missing project after verifying its content fingerprint."""
        if not self._selected_asset_ids or len(self._selected_asset_ids) != 1:
            return

        asset_id = list(self._selected_asset_ids)[0]
        if not self._asset_index or not hasattr(self._asset_index, "assets"):
            return

        asset = self._asset_index.assets.get(asset_id)
        if not asset:
            return

        file_path = lf.ui.open_project_file_dialog("")

        if not file_path:
            return

        try:
            if not self._asset_index.relink_asset(asset_id, file_path):
                self._log_warn("Selected project does not match the cataloged content")
                return
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
        self._load_asset(asset_id)

    def _load_asset(self, asset_id: str) -> None:
        if not asset_id:
            return

        if not self._asset_index:
            self._log_warn("Asset index not initialized")
            return

        asset_record = self._asset_index.verify_asset(asset_id)
        if not asset_record:
            self._log_warn("Asset not found: %s", asset_id)
            return
        asset = asset_record.to_dict()
        file_path = asset.get("absolute_path") or asset.get("path")
        if not file_path or not asset.get("exists", False):
            self._log_warn(
                "Project is unavailable (%s): %s",
                asset.get("verification_disposition") or "UNVERIFIED",
                file_path,
            )
            self.refresh_catalog()
            return

        try:
            # Projects replace the active document. discard_changes=False lets
            # the lifecycle layer reject an unsafe open when work is unsaved.
            lf.project_open(file_path, False)
            self._log_info("Opened project: %s", asset.get("name", "unknown"))

            self._selected_asset_ids = {asset_id}
            self._selection_type = "asset"
            self.refresh_catalog()
        except Exception as e:
            self._log_error("Failed to load asset %s: %s", asset_id, e)

    def on_remove_asset(self, _handle, _ev, args):
        """Remove a specific asset from the catalog by ID."""
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        if not asset_id:
            return

        if not self._asset_index:
            self._log_warn("Asset index not initialized")
            return

        try:
            if not self._delete_asset_from_catalog(asset_id):
                self._log_warn("Asset index does not support asset deletion")
                return

            self._selected_asset_ids.discard(asset_id)
            self._update_selection_type()
            self.refresh_catalog()
            self._log_info("Removed project from catalog: %s", asset_id)
        except Exception as e:
            self._log_error("Failed to remove project %s: %s", asset_id, e)

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

    def _open_asset_menu(self, asset_id: str) -> None:
        if not asset_id:
            return
        self._open_menu_folder_id = None
        self._open_menu_asset_id = asset_id
        if self._handle:
            folders = self.get_move_menu_folders()
            self._log_info("Loading %d folders for move menu", len(folders))
            self._handle.update_record_list("move_menu_folders", folders)
        self._dirty_model("assets", "folders")

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
            self._dirty_model("assets")
            if self._handle:
                self._handle.update_record_list("move_menu_folders", [])
        else:
            self._open_asset_menu(asset_id)

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
        """Open .licht-only watched-directory settings for a folder."""
        folder_id = self._resolve_event_value(args, _ev, "data-folder-id")
        if not folder_id or self._asset_index is None:
            return

        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        self._open_menu_folder_id = None
        self._dirty_model("folders")
        if not open_watch_dirs_dialog(
            self._asset_index,
            folder_id,
            self.refresh_catalog,
        ):
            self._log_warn(
                "Failed to open watched directories for folder %s", folder_id
            )

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
        total_start = time.perf_counter()
        folder_id = self._resolve_event_value(args, _ev, "data-folder-id")
        if not folder_id:
            return

        # Stop event propagation
        if _ev:
            try:
                _ev.stop_propagation()
            except Exception:
                pass

        if not self._asset_index:
            return

        folders = self._asset_index_folders()
        folder = folders.get(folder_id)
        if not folder:
            return

        # Close the menu
        self._open_menu_folder_id = None
        self._dirty_model("folders")

        folder_name = folder.get("name", "Unnamed Folder")

        scenes = self._asset_index_scenes()
        scene_ids_to_delete = {
            scene_id
            for scene_id, scene in scenes.items()
            if scene.get("folder_id") == folder_id
        }
        assets = self._asset_index_assets()
        assets_to_delete = [
            asset_id
            for asset_id, asset in assets.items()
            if asset.get("folder_id") == folder_id
            or asset.get("scene_id") in scene_ids_to_delete
        ]
        scene_count = len(scene_ids_to_delete)
        # Delete the folder
        try:
            delete_start = time.perf_counter()
            deleted = self._asset_index.delete_folder(folder_id)
            delete_ms = self._elapsed_ms(delete_start)
            if not deleted:
                self._log_warn("Failed to delete folder '%s'", folder_name)
                return
            self._invalidate_catalog_cache()

            # Clear selection if the deleted folder was selected
            if self._selected_folder_id == folder_id:
                self._selected_scene_id = None
                self._selected_asset_ids.clear()
                self._selection_type = "folder"
            self._repair_selected_folder()

            refresh_start = time.perf_counter()
            self.refresh_catalog()
            refresh_ms = self._elapsed_ms(refresh_start)
            self._log_perf(
                (
                    "delete_folder folder=%s assets=%d scenes=%d "
                    "delete=%.3fms refresh=%.3fms total=%.3fms"
                ),
                folder_id,
                len(assets_to_delete),
                scene_count,
                delete_ms,
                refresh_ms,
                self._elapsed_ms(total_start),
                elapsed_ms=self._elapsed_ms(total_start),
            )
            if assets_to_delete:
                self._log_info(
                    "Deleted folder '%s' and removed %d assets from the catalog",
                    folder_name,
                    len(assets_to_delete),
                )
            else:
                self._log_info("Deleted folder '%s'", folder_name)
        except Exception as e:
            self._log_error("Failed to delete folder: %s", e)

    # ── Lifecycle ─────────────────────────────────────────────

    def on_mount(self, doc):
        global _active_asset_manager_panel
        super().on_mount(doc)
        self._doc = doc
        _active_asset_manager_panel = self
        self._bind_dom_event_listeners(doc)
        self._sync_panel_space_state()

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
                self._asset_index.verify_projects()
                if self._asset_index.library_path.exists():
                    self._library_mtime = self._asset_index.library_path.stat().st_mtime
            except Exception as e:
                _logger.warning(f"Failed to load asset index: {e}")

        has_existing_selection = bool(self._selected_asset_ids)

        # Clear scene filter on reopen to show all assets in the folder
        if has_existing_selection:
            self._selected_scene_id = None

        # Initial refresh must dirty scalar bindings after catalog load.
        self.refresh_catalog()
        self._last_auto_save_time = time.time()
        self._last_scene_generation = RuntimeState.scene_generation.value
        self._last_language_generation = RuntimeState.language_generation.value
        self._subscribe_reactive_state()
        _ensure_atexit_registered()

    def on_scene_changed(self, doc):
        self._last_scene_generation = RuntimeState.scene_generation.value

    def on_update(self, doc):
        """Dirty-policy update for catalog and panel layout changes."""
        self._asset_window_update_requested = False
        pending_window_refresh = self._asset_window_refresh_pending
        window_changed = self._sync_asset_window_viewport(doc)
        card_width_changed = self._sync_gallery_card_width(doc)
        should_apply_window_refresh = (
            pending_window_refresh or window_changed or card_width_changed
        )
        if should_apply_window_refresh:
            self._apply_asset_window_refresh(
                card_width_changed=card_width_changed and self._view_mode == "gallery",
            )
        self._asset_window_refresh_pending = False

        changed = should_apply_window_refresh

        space_changed = self._sync_panel_space_state()
        if space_changed and self._handle:
            self._handle.dirty_all()
            changed = True

        language_generation = RuntimeState.language_generation.value
        if language_generation != self._last_language_generation:
            self._last_language_generation = language_generation
            if self._handle:
                self._handle.dirty_all()
            changed = True

        if not self._asset_index:
            return changed

        try:
            library_path = self._asset_index.library_path
            if library_path.exists():
                current_mtime = library_path.stat().st_mtime
                if current_mtime > self._library_mtime:
                    self._asset_index.load()
                    self._library_mtime = current_mtime
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

    def _sync_gallery_card_width(self, doc) -> bool:
        grid_el = doc.get_element_by_id("asset-card-grid") if doc else None
        if not grid_el:
            return False

        try:
            dp_ratio = max(1.0, float(lf.ui.get_ui_scale()))
            viewport_width_dp = float(grid_el.client_width or 0.0) / dp_ratio
        except Exception:
            return False

        available_width = max(
            ASSET_CARD_MIN_WIDTH_DP,
            viewport_width_dp - ASSET_CARD_GRID_HORIZONTAL_CHROME_DP,
        )
        next_width = min(ASSET_CARD_PREFERRED_WIDTH_DP, available_width)
        if abs(next_width - self._asset_card_slot_width) <= 0.5:
            return False

        self._asset_card_slot_width = next_width
        if self._handle:
            self._handle.dirty("asset_card_slot_width")
        return True

    def on_unmount(self, doc):
        """Save index on unmount."""
        global _active_asset_manager_panel
        self._cancel_selection_detail_timer()
        self._unsubscribe_reactive_state()
        if _active_asset_manager_panel is self:
            _active_asset_manager_panel = None

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
        content = doc.get_element_by_id("asset-main-row")
        if content:
            content.add_event_listener("mousedown", self._on_asset_manager_mousedown)
            content.add_event_listener("click", self._on_asset_manager_click)
            content.add_event_listener(
                "dblclick", self._on_asset_manager_double_click
            )
        scroll_el = doc.get_element_by_id("asset-gallery-scroll")
        if scroll_el:
            scroll_el.add_event_listener("scroll", self._on_asset_scroll)
            scroll_el.add_event_listener(
                "mousescroll", self._on_gallery_precise_scroll
            )

        # Resize-start is bound declaratively in RML via data-event-mousedown.
        # Only keep document-level listeners here for active drag tracking.
        doc.add_event_listener("mousemove", self._on_resize_mousemove)
        doc.add_event_listener("mouseup", self._on_resize_mouseup)

    def _on_asset_scroll(self, event) -> None:
        scroll_el = event.current_target()
        if not scroll_el:
            return
        if self._asset_scroll_event_suppressed:
            try:
                current_scroll_top = max(0.0, float(scroll_el.scroll_top or 0.0))
            except Exception:
                current_scroll_top = -1.0
            self._asset_scroll_event_suppressed = False
            if abs(current_scroll_top - self._asset_scroll_suppressed_top) <= 0.01:
                self._asset_scroll_suppressed_top = -1.0
                return
            self._asset_scroll_suppressed_top = -1.0
        self._request_asset_window_refresh()

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
            self._asset_scroll_event_suppressed = True
            self._asset_scroll_suppressed_top = new_scroll

        self._request_asset_window_refresh()

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
            elif action == "move_to_folder":
                self.on_move_to_folder(None, event, [asset_id])
                self._stop_event(event)
                return
            elif action == "remove_from_menu":
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
                self._select_asset_id(
                    asset_id,
                    toggle=False,
                    multi_select=self._event_multi_select(event),
                    row_element=action_el,
                    container=container,
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

        if button != 1:
            return

        self._select_asset_id(
            asset_id,
            toggle=False,
            multi_select=False,
            row_element=action_el,
            container=container,
        )
        self._open_asset_menu(asset_id)
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
            mouse_y = float(event.get_parameter("mouse_y", "0"))
        except (TypeError, ValueError):
            return
        if self._sidebar_dragging:
            self.on_sidebar_resize_delta(mouse_y)
            event.stop_propagation()
        elif self._right_panel_dragging:
            self.on_right_panel_resize_delta(mouse_x)
            event.stop_propagation()
        elif self._bottom_panel_dragging:
            self.on_bottom_panel_resize_delta(mouse_y)
            event.stop_propagation()

    def _on_resize_mouseup(self, _event) -> None:
        """Handle mouseup to end panel resizing."""
        self.on_sidebar_resize_end()
        self.on_right_panel_resize_end()
        self.on_bottom_panel_resize_end()

    def refresh_catalog(self, *, request_update: bool = True):
        """Refresh all catalog data in the UI."""
        total_start = time.perf_counter()
        self._invalidate_catalog_cache()
        reconcile_start = time.perf_counter()
        self._reconcile_selection()
        reconcile_ms = self._elapsed_ms(reconcile_start)
        records_start = time.perf_counter()
        record_summary = self._update_all_record_lists()
        records_ms = self._elapsed_ms(records_start)
        dirty_ms = 0.0
        request_ms = 0.0
        if self._handle:
            dirty_start = time.perf_counter()
            self._handle.dirty_all()
            dirty_ms = self._elapsed_ms(dirty_start)
            if request_update:
                request_start = time.perf_counter()
                self._request_model_update()
                request_ms = self._elapsed_ms(request_start)
        self._log_perf(
            (
                "refresh request=%s reconcile=%.3fms records=%.3fms/%s "
                "record_parts=%s dirty_all=%.3fms request_update=%.3fms total=%.3fms"
            ),
            request_update,
            reconcile_ms,
            records_ms,
            record_summary.get("counts", {}) if record_summary else {},
            record_summary.get("timings_ms", {}) if record_summary else {},
            dirty_ms,
            request_ms,
            self._elapsed_ms(total_start),
            elapsed_ms=self._elapsed_ms(total_start),
        )

    def verify_catalog(self, _handle=None, _ev=None, _args=None):
        """Non-blocking launcher: verify all project fingerprints."""
        if not self._asset_index:
            return

        def _verify() -> None:
            try:
                unavailable, total = self._asset_index.verify_projects()
                self._log_info(
                    "Verified %d project(s); %d unavailable",
                    total,
                    unavailable,
                )
                self.refresh_catalog()
            except Exception as exc:
                self._log_error("Failed to verify project catalog: %s", exc)

        threading.Thread(target=_verify, daemon=True).start()

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

    def _update_all_record_lists(self):
        """Update all record lists in the data model."""
        if not self._handle:
            return {"counts": {}, "timings_ms": {}}

        counts: Dict[str, int] = {}
        timings_ms: Dict[str, Dict[str, float]] = {}

        def update_record_list(name: str, builder) -> None:
            build_start = time.perf_counter()
            rows = builder()
            build_ms = self._elapsed_ms(build_start)
            update_start = time.perf_counter()
            self._handle.update_record_list(name, rows)
            update_ms = self._elapsed_ms(update_start)
            timings_ms[name] = {
                "build": build_ms,
                "update": update_ms,
                "total": build_ms + update_ms,
            }
            counts[name] = len(rows)
            if name == "assets":
                self._last_asset_rows_update_count = len(rows)
                self._last_asset_rows_update_ms = build_ms + update_ms

        update_record_list("folders", self.get_folder_list)
        update_record_list("scenes", self.get_scene_list)
        update_record_list("assets", self.get_filtered_assets)

        # Update selection-specific record lists
        selection_summary = self._update_selection_details()
        if selection_summary:
            counts.update(selection_summary.get("counts", {}))
            timings_ms.update(selection_summary.get("timings_ms", {}))
        return {"counts": counts, "timings_ms": timings_ms}

    def _update_selection_details(
        self,
        *,
        update_scene_assets: bool = True,
    ) -> Dict[str, Any]:
        """Update record lists for selected scene and folder."""
        if not self._handle or self._updating_selection_details:
            return {"counts": {}, "timings_ms": {}}
        self._updating_selection_details = True
        counts: Dict[str, int] = {}
        timings_ms: Dict[str, Dict[str, float]] = {}
        try:
            if update_scene_assets:
                scene_key = (
                    str(self._selected_scene_id or "")
                    if self._selection_type == "scene"
                    else ""
                )
                if scene_key != self._selected_scene_assets_key:
                    build_start = time.perf_counter()
                    rows = self._get_selected_scene_asset_rows() if scene_key else []
                    build_ms = self._elapsed_ms(build_start)
                    update_start = time.perf_counter()
                    self._handle.update_record_list("selected_scene_assets", rows)
                    update_ms = self._elapsed_ms(update_start)
                    self._selected_scene_assets_key = scene_key
                    counts["selected_scene_assets"] = len(rows)
                    timings_ms["selected_scene_assets"] = {
                        "build": build_ms,
                        "update": update_ms,
                        "total": build_ms + update_ms,
                    }
                    self._handle.dirty("selected_scene_assets")

            return {"counts": counts, "timings_ms": timings_ms}
        finally:
            self._updating_selection_details = False

    def _get_selected_scene_asset_rows(self) -> List[Dict[str, str]]:
        scene = self._get_selected_scene()
        assets = self._asset_index_assets()
        if not scene or not assets:
            return []
        scene_id = scene.get("id", "")
        if not scene_id:
            return []
        return [
            {
                "id": asset_id,
                "name": str(asset.get("name") or tr("asset_manager.unnamed")),
            }
            for asset_id, asset in assets.items()
            if asset.get("scene_id") == scene_id
        ]

    def _dirty_model(self, *fields):
        """Mark fields as dirty to trigger UI refresh."""
        if not self._handle:
            return

        total_start = time.perf_counter()
        record_update_ms = 0.0
        record_updates: Dict[str, int] = {}
        request_update_ms = 0.0
        if not fields:
            self._invalidate_catalog_cache()
            self._handle.dirty_all()
            records_start = time.perf_counter()
            record_summary = self._update_all_record_lists()
            record_update_ms = self._elapsed_ms(records_start)
            request_start = time.perf_counter()
            self._request_model_update()
            request_update_ms = self._elapsed_ms(request_start)
            self._last_dirty_model_timing = {
                "field_count": 0,
                "record_update_ms": record_update_ms,
                "record_updates": record_summary.get("counts", {})
                if record_summary
                else {"all": -1},
                "record_parts": record_summary.get("timings_ms", {})
                if record_summary
                else {},
                "request_update_ms": request_update_ms,
                "total_ms": self._elapsed_ms(total_start),
            }
            self._log_perf(
                "dirty_all records=%.3fms/%s record_parts=%s request=%.3fms total=%.3fms",
                record_update_ms,
                self._last_dirty_model_timing["record_updates"],
                self._last_dirty_model_timing["record_parts"],
                request_update_ms,
                self._last_dirty_model_timing["total_ms"],
                elapsed_ms=self._last_dirty_model_timing["total_ms"],
            )
            return

        fields_set = set(fields)
        # "assets" is also used for viewport/menu refreshes, so invalidating the
        # catalog cache here defeats virtualization by forcing a full regroup/filter
        # rebuild on scroll. Real catalog mutations already invalidate explicitly.
        if fields_set.intersection({"folders", "scenes"}):
            self._invalidate_catalog_cache()

        # Check if any selection-related fields are being dirtied.
        selection_fields = set(self._selection_count_fields())
        selection_fields.update(self._selection_visibility_fields())
        selection_fields.update(self._selected_asset_detail_fields())
        selection_fields.update(self._selected_scene_detail_fields())
        selection_fields.update(self._selected_folder_detail_fields())
        selection_fields.update(
            {
                "selected_asset",
                "selected_asset_id",
                "selected_folder",
                "selected_folder_id",
                "selected_scene",
                "selected_scene_id",
            }
        )
        needs_selection_update = any(f in selection_fields for f in fields)
        update_scene_assets = bool(
            fields_set.intersection(
                set(self._selected_scene_detail_fields())
                | {"selected_scene", "selected_scene_id", "selected_scene_assets"}
            )
        )
        for field in fields:
            self._handle.dirty(field)
            # Update record lists when they change
            if field in (
                "folders",
                "scenes",
                "assets",
            ):
                list_map = {
                    "folders": self.get_folder_list,
                    "scenes": self.get_scene_list,
                    "assets": self.get_filtered_assets,
                }
                if field in list_map:
                    records_start = time.perf_counter()
                    rows = list_map[field]()
                    self._handle.update_record_list(field, rows)
                    elapsed = self._elapsed_ms(records_start)
                    record_update_ms += elapsed
                    record_updates[field] = len(rows)
                    if field == "assets":
                        self._last_asset_rows_update_count = len(rows)
                        self._last_asset_rows_update_ms = elapsed

        # Update selection-specific record lists if needed
        if needs_selection_update and not self._updating_selection_details:
            records_start = time.perf_counter()
            selection_summary = self._update_selection_details(
                update_scene_assets=update_scene_assets,
            )
            record_update_ms += self._elapsed_ms(records_start)
            record_updates.update(selection_summary.get("counts", {}))

        request_start = time.perf_counter()
        self._request_model_update()
        request_update_ms = self._elapsed_ms(request_start)
        self._last_dirty_model_timing = {
            "field_count": len(fields),
            "record_update_ms": record_update_ms,
            "record_updates": record_updates,
            "request_update_ms": request_update_ms,
            "total_ms": self._elapsed_ms(total_start),
        }
        self._log_perf(
            "dirty fields=%d records=%.3fms/%s request=%.3fms total=%.3fms",
            len(fields),
            record_update_ms,
            record_updates,
            request_update_ms,
            self._last_dirty_model_timing["total_ms"],
            elapsed_ms=self._last_dirty_model_timing["total_ms"],
        )

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

    def _sync_panel_space_state(self) -> bool:
        info = None
        try:
            info = lf.ui.get_panel(self.id)
        except Exception:
            info = None
        panel_space = getattr(info, "space", self._panel_space)
        is_floating = panel_space == lf.ui.PanelSpace.FLOATING
        changed = panel_space != self._panel_space or is_floating != self._is_floating
        self._panel_space = panel_space
        self._is_floating = is_floating
        return changed

    def _on_close_panel(self, _handle, _event, _args):
        lf.ui.set_panel_enabled(self.id, False)


# ── atexit backup ─────────────────────────────────────────

_atexit_registered = False


def _atexit_save_asset_manager() -> None:
    """Last-resort save when the process exits without on_unmount()."""
    try:
        panel = _active_asset_manager_panel
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
