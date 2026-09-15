# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager panel for browsing UUID-identified .licht projects."""

from __future__ import annotations

import json
import logging
import math
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import quote

import lichtfeld as lf

from .asset_gallery_ui import GalleryAssetMixin, GALLERY_SCOPES, SCOPE_PUBLISHED, SCOPE_ATTENTION, SCOPE_TRANSFERS
from . import rml_widgets
from .asset_layout import (
    breakpoint_metrics,
    breakpoint_for_width,
    card_geometry,
    gallery_columns,
    gallery_slot_width,
    grid_columns,
    grid_slot_width,
    list_row_height,
    native_to_dp,
    panel_layout,
    list_columns,
)
from .asset_format import format_size
from .gallery_transfer_ui import transfer_rows
from .project_inspector import (
    InspectionFactsPipeline,
    dialog_model,
    details_rows,
    inspection_cache_key,
    operation_actions,
)
from .asset_watch import (
    AssetFolderScanProgress,
    scan_all_asset_folders,
    scan_asset_folder,
    verify_catalog_projects,
)
from .localization import localized_count
from .rml_keys import KI_DELETE, KI_DOWN, KI_ESCAPE, KI_LEFT, KI_RETURN, KI_RIGHT, KI_SPACE, KI_UP
from .types import Panel
from .panels import panel_class
from .ui import RuntimeState

_log = logging.getLogger(__name__)

PRECISE_SCROLL_STEP = 32.0
ASSET_LIST_ROW_HEIGHT_DP = 48.0
ASSET_GALLERY_ROW_HEIGHT_DP = 230.0
ASSET_CARD_PREFERRED_WIDTH_DP = 208.0
ASSET_CARD_GRID_HORIZONTAL_CHROME_DP = 48.0
ASSET_WINDOW_OVERSCAN_ROWS = 2
ASSET_LIST_FALLBACK_ROWS = 24
ASSET_GALLERY_FALLBACK_ROWS = 8
_RML_PATH_SAFE_CHARS = "/:._-~"
_THUMBNAIL_FIT_ALIGN = "cover center"
SCOPE_ALL = "__all__"
SCOPE_RECENT = "__recent__"
PROJECT_DRAG_PAYLOAD_TYPE = "application/x-lichtfeld-project"
_folder_scan_completed_in_process = False

try:
    from .asset_index import (
        AssetIndex,
        display_name,
        fix_action_for_health,
        is_supported_asset_path,
        resolve_default_asset_directory,
        resolve_asset_manager_storage_path,
    )
    from .asset_service import LibraryService

    BACKEND_AVAILABLE = True
except ImportError:
    AssetIndex = None
    LibraryService = None
    BACKEND_AVAILABLE = False


def tr(key: str, **kwargs: Any) -> str:
    translate = getattr(getattr(lf, "ui", None), "tr", None)
    try:
        result = translate(key) if callable(translate) else key
    except Exception:
        result = key
    if kwargs:
        try:
            return result.format(**kwargs)
        except Exception:
            pass
    return result


__lfs_panel_classes__ = ["AssetManagerPanel"]
__lfs_panel_ids__ = ["lfs.asset_manager"]


@panel_class("asset_manager")
class AssetManagerPanel(GalleryAssetMixin, Panel):
    """Dockable `.licht` project catalog."""

    SORT_MODES = ("name", "size", "iteration", "saved", "opened", "published")
    STORAGE_PATH: Optional[Path] = None

    def __init__(self):
        super().__init__()
        self._handle = None
        self._doc = None
        self._asset_index: Optional[Any] = None
        self._library_service: Optional[Any] = None

        self._selected_asset_ids: Set[str] = set()
        self._selection_cursor_id: Optional[str] = None
        self._selection_anchor_id: Optional[str] = None
        self._selected_folder_id: Optional[str] = SCOPE_ALL
        self._selection_type = "none"
        self._view_mode = "list"
        self._sort_mode = "name"
        self._sort_descending = False
        self._search_query = ""
        self._active_filter = "all"

        self._folders_collapsed = False
        self._sidebar_height = 280.0
        self._bottom_panel_height = 220.0
        self._info_preferred_height = 220.0
        self._navigator_width = 200.0
        self._inspector_width = 280.0
        self._inspector_preferred_height = 200.0
        self._tray_height = 120.0
        self._inspector_expanded = False
        self._quick_look_visible = False
        self._thumbnail_sizes = {
            "compact": 112.0,
            "narrow": 136.0,
            "medium": 168.0,
            "wide": 168.0,
        }
        self._layout_class = ""
        self._content_width = 0.0
        self._host_geometry = None
        self._last_ui_scale = 0.0
        self._list_column_overrides: Dict[str, float] = {}
        self._layout_signature = None
        self._main_min_height = 0.0
        self._folder_layout_initialized = False
        self._bottom_panel_dragging = False
        self._resize_region = ""
        self._resize_start_x = 0.0
        self._resize_start_y = 0.0
        self._bottom_panel_drag_start_y = 0.0
        self._bottom_panel_start_height = self._bottom_panel_height

        self._asset_card_slot_width = ASSET_CARD_PREFERRED_WIDTH_DP
        self._asset_window_scroll_top = 0.0
        self._asset_window_client_height = 0.0
        self._asset_window_client_width = 0.0
        self._asset_list_top_spacer_height = 0.0
        self._asset_list_bottom_spacer_height = 0.0
        self._asset_gallery_top_spacer_height = 0.0
        self._asset_gallery_bottom_spacer_height = 0.0
        self._asset_window_refresh_pending = False
        self._asset_scroll_event_suppressed = False
        self._asset_scroll_suppressed_top = -1.0
        self._last_asset_match_count = 0

        self._panel_space = lf.ui.PanelSpace.LEFT_DOCK
        self._is_floating = False
        self._reactive_unsubscribers: list[Callable[[], None]] = []

        self._folder_scan_lock = threading.Lock()
        self._folder_scan_active = False
        self._folder_scan_refresh_pending = False
        self._folder_scan_rerun_pending = False
        self._folder_scan_rerun_target: Optional[tuple[str, str, bool]] = None
        self._folder_scan_cancel: Optional[threading.Event] = None
        self._folder_scan_thread: Optional[threading.Thread] = None
        self._catalog_verify_active = False
        self._catalog_verify_refresh_pending = False
        self._catalog_verify_cancel: Optional[threading.Event] = None
        self._catalog_verify_thread: Optional[threading.Thread] = None
        self._catalog_epoch_seen: Optional[int] = None
        self._catalog_unsubscribe: Optional[Callable[[], None]] = None
        self._worker_notification_lock = threading.Lock()
        self._worker_notification_pending = False
        self._scan_progress = AssetFolderScanProgress()
        self._scan_stop_requested = False
        self._scan_stopped_visible = False
        self._published_scan_active = False
        self._published_scan_status = ""
        # Keep direct programmatic refreshes usable before the first DOM mount;
        # on_unmount flips this false and mount generations guard callbacks.
        self._panel_mounted = True
        self._mount_generation = 0
        self._backend_load_active = False
        self._catalog_load_failed = False
        self._catalog_notice = ""
        self._folder_scan_error = False
        self._folder_scan_unavailable = False
        self._drag_payload_token: Optional[int] = None
        self._gallery_drag = None
        self._gallery_drop_element = None
        self._last_project_write_generation: Optional[int] = None
        self._project_write_was_running = False
        self._last_project_write_path = ""
        self._thumbnail_sources_by_asset: Dict[str, str] = {}
        self._info_thumbnail_source = ""
        self._last_default_folder_path = ""
        self._inspection_pipeline: Optional[InspectionFactsPipeline] = None
        self._inspection_by_asset: Dict[str, Dict[str, Any]] = {}
        self._inspection_errors: Dict[str, str] = {}
        self._project_operations: Dict[str, Dict[str, Any]] = {}
        self._operation_counter = 0
        self._dialog_kind = ""
        self._dialog_asset_id = ""
        self._dialog_data: Dict[str, Any] = {}
        self._dialog_plan = None
        self._dialog_busy = False
        self._dialog_drop_checkpoints = True
        self._dialog_drop_dataset = False
        self._operations_expanded = None
        self._inspector_sections = {"project": True, "file": True, "gallery": True}
        self._info_thumbnail_geometry = None
        self._verify_results: Dict[str, str] = {}
        self._init_gallery()

    def capture_chrome(self) -> Dict[str, Any]:
        folder_id = self._selected_folder_id if self._selected_folder_id in self._asset_index_folders() else SCOPE_ALL
        return {
            "folders_collapsed": self._folders_collapsed,
            "sidebar_height": self._sidebar_height,
            "bottom_panel_height": self._info_preferred_height,
            "navigator_width": self._navigator_width,
            "inspector_width": self._inspector_width,
            "inspector_height": self._inspector_preferred_height,
            "tray_height": self._tray_height,
            "thumbnail_sizes": dict(self._thumbnail_sizes),
            "list_column_overrides": dict(self._list_column_overrides),
            "inspector_sections": dict(self._inspector_sections),
            "operations_expanded": self._operations_expanded,
            "sort_mode": self._sort_mode,
            "sort_descending": self._sort_descending,
            "selected_folder_id": folder_id,
        }

    def apply_chrome(self, payload: Any) -> None:
        if isinstance(payload, dict):
            if payload.get("sort_mode") in self.SORT_MODES:
                self._sort_mode = payload["sort_mode"]
                self._sort_descending = bool(payload.get("sort_descending", self._sort_mode != "name"))
            sections = payload.get("inspector_sections", {})
            if isinstance(sections, dict):
                for section in self._inspector_sections:
                    if isinstance(sections.get(section), bool):
                        self._inspector_sections[section] = sections[section]
            if isinstance(payload.get("operations_expanded"), bool):
                self._operations_expanded = payload["operations_expanded"]
            self._folders_collapsed = bool(
                payload.get("folders_collapsed", self._folders_collapsed)
            )
            self._folder_layout_initialized = "folders_collapsed" in payload
            value = payload.get("bottom_panel_height")
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                self._info_preferred_height = min(500.0, float(value))
                self._inspector_preferred_height = self._info_preferred_height
            for key, low, high, default in (
                ("navigator_width", 120.0, 240.0, 200.0),
                ("inspector_width", 240.0, 420.0, 280.0),
                ("inspector_height", 120.0, 450.0, 200.0),
                ("tray_height", 32.0, 450.0, 32.0),
            ):
                number = payload.get(key)
                if isinstance(number, (int, float)) and math.isfinite(number):
                    setattr(self, "_" + key, min(high, max(low, float(number))))
                elif not hasattr(self, "_" + key):
                    setattr(self, "_" + key, default)
            sizes = payload.get("thumbnail_sizes")
            if isinstance(sizes, dict):
                for name in ("compact", "narrow", "medium", "wide"):
                    number = sizes.get(name)
                    if isinstance(number, (int, float)) and math.isfinite(number):
                        self._thumbnail_sizes[name] = min(320.0, max(112.0, float(number)))
            overrides = payload.get("list_column_overrides")
            if isinstance(overrides, dict):
                self._list_column_overrides = {
                    name: min(280.0, max(64.0, float(value)))
                    for name, value in overrides.items()
                    if name in {"name", "gallery", "size", "modified", "folder"}
                    and isinstance(value, (int, float)) and math.isfinite(value)
                }
            folder_id = payload.get("selected_folder_id")
            self._selected_folder_id = str(folder_id) if folder_id in self._asset_index_folders() else SCOPE_ALL
            # Old sidebar heights are superseded by content/viewport sizing.
            self._layout_signature = None
            self._sync_panel_layout()
        if self._handle:
            self._handle.dirty_all()

    def _initialize_backend(self) -> bool:
        self._catalog_load_failed = False
        if not BACKEND_AVAILABLE:
            self._catalog_load_failed = True
            return False
        try:
            storage_path = resolve_asset_manager_storage_path()
            storage_path.mkdir(parents=True, exist_ok=True)
            self.STORAGE_PATH = storage_path
            self.__class__.STORAGE_PATH = storage_path
            try:
                index = AssetIndex(
                    library_path=storage_path / "library.json",
                    default_folder_path=resolve_default_asset_directory(),
                )
            except TypeError:
                index = AssetIndex()
            self._library_service = LibraryService(index)
            self._asset_index = self._library_service.index
            loaded = self._library_service._call("load")
            self._last_default_folder_path = str(resolve_default_asset_directory())
            if not loaded:
                self._catalog_load_failed = True
            return loaded
        except Exception as exc:
            self._log_error("Failed to initialize Asset Manager: %s", exc)
            self._catalog_load_failed = True
            return False

    def _start_backend_initialization(self) -> None:
        if self._backend_load_active or not BACKEND_AVAILABLE:
            if not BACKEND_AVAILABLE:
                self._catalog_load_failed = True
            return
        self._backend_load_active = True
        generation = self._mount_generation

        def worker() -> None:
            index = None
            storage_path = None
            default_path = ""
            loaded = False
            try:
                storage_path = resolve_asset_manager_storage_path()
                storage_path.mkdir(parents=True, exist_ok=True)
                try:
                    index = AssetIndex(
                        library_path=storage_path / "library.json",
                        default_folder_path=resolve_default_asset_directory(),
                    )
                except TypeError:
                    index = AssetIndex()
                service = LibraryService(index)
                index = service.index
                loaded = service._call("load")
                default_path = str(resolve_default_asset_directory())
            except Exception as exc:
                self._log_error("Failed to initialize Asset Manager: %s", exc)

            def complete() -> None:
                if generation != self._mount_generation or not self._panel_mounted:
                    self._backend_load_active = False
                    return
                self._backend_load_active = False
                self._catalog_load_failed = not loaded
                if index is not None:
                    self._asset_index = index
                    self._library_service = service
                    self.STORAGE_PATH = storage_path
                    self.__class__.STORAGE_PATH = storage_path
                    self._last_default_folder_path = default_path
                    self._catalog_epoch_seen = self._catalog_epoch()
                    self._subscribe_catalog()
                    self._repair_selection()
                    if self._gallery_focus_path:
                        self.focus_gallery(self._gallery_focus_path)
                    self._refresh_records(assets=True, folders=True)
                    if self._handle:
                        self._handle.dirty_all()
                    self._start_catalog_verify()
                    self._scan_asset_folders()
                self._request_model_update()

            scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
            if callable(scheduler):
                scheduler(complete)
            else:
                complete()

        threading.Thread(
            target=worker, daemon=True, name="AssetManagerCatalogLoad"
        ).start()

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("asset_manager")
        if model is None:
            return

        self._bind_gallery_model(model)
        model.bind("search_query", self.get_search_query, self.set_search_query)
        model.bind_func("search_is_empty", lambda: not self._search_query)
        model.bind("selected_folder_id", lambda: self._selected_folder_id or SCOPE_ALL, self._set_scope_value)
        model.bind("thumbnail_size", self.get_thumbnail_size, self.set_thumbnail_size)
        model.bind_func("is_gallery_view", lambda: self._view_mode == "gallery")
        model.bind_func("is_list_view", lambda: self._view_mode == "list")
        model.bind_func("sort_label", self.get_sort_label)
        model.bind_func("sort_tooltip", self.get_sort_tooltip)
        model.bind_func("filter_menu_label", lambda: tr("projects.toolbar.filter"))
        model.bind_func("active_filter_label", self.get_filter_label)
        model.bind_func("folders_collapsed", lambda: self._folders_collapsed)
        model.bind_func("folders_expanded", lambda: not self._folders_collapsed)
        model.bind_func("all_assets_selected", lambda: self._selected_folder_id == SCOPE_ALL)
        model.bind_func("all_assets_count", self.get_all_assets_count)
        model.bind_func("selected_asset_id", self.get_selected_asset_id)
        model.bind_func("selected_count", self.get_selected_count)
        model.bind_func("selected_count_text", self.get_selected_count_text)
        model.bind_func("show_selection_none", lambda: self._selection_type == "none")
        model.bind_func("show_selection_asset", lambda: self._selection_type == "asset")
        model.bind_func("show_selection_folder", lambda: self._selection_type == "folder")
        model.bind_func(
            "show_selection_multiple", lambda: self._selection_type == "multiple"
        )

        model.bind_func("asset_list_wide", lambda: list_columns(self._asset_window_client_width / self._ui_scale())["modified"])
        model.bind_func("asset_list_show_folder", lambda: list_columns(self._asset_window_client_width / self._ui_scale())["folder"])
        for column in ("name", "gallery", "size", "modified", "folder"):
            model.bind_func(
                f"asset_list_{column}_width",
                lambda column=column: f"{self._list_column_width(column):.1f}dp",
            )
        model.bind_func("asset_list_gallery_compact", lambda: list_columns(self._asset_window_client_width / self._ui_scale())["gallery"] < 140)
        model.bind_func("col_gallery_label", lambda: tr("projects.gallery.sidebar.title"))
        model.bind_func(
            "check_gallery_tooltip",
            lambda: f"{tr('projects.action.check_gallery')} · {self._gallery_checked_label()}",
        )
        model.bind_func("is_compact", lambda: self._layout_class == "compact")
        model.bind_func("is_narrow", lambda: self._layout_class == "narrow")
        model.bind_func("is_medium", lambda: self._layout_class == "medium")
        model.bind_func("is_wide", lambda: self._layout_class == "wide")
        model.bind_func("navigator_width", lambda: f"{self._navigator_width:.1f}dp")
        model.bind_func("navigator_style_width", self.get_navigator_style_width)
        model.bind_func("inspector_width", lambda: f"{self._inspector_width:.1f}dp")
        model.bind_func("inspector_style_width", self.get_inspector_style_width)
        for section in self._inspector_sections:
            model.bind_func("inspector_" + section + "_expanded",
                            lambda section=section: self._inspector_sections[section])
        model.bind_func("inspector_gallery_action_label", lambda: (
            self._gallery_badge(self._get_selected_asset())["gallery_action_label"]
            if self._get_selected_asset() else ""))
        model.bind_func("inspector_has_gallery_action", lambda: (
            not self._selected_details_rows().get("resumable") and bool(self._selected_gallery_action())))
        model.bind_func("inspector_gallery_action_tooltip", lambda: self._gallery_account_reason() or (
            self._gallery_badge(self._get_selected_asset())["gallery_action_label"]
            if self._get_selected_asset() else ""))
        model.bind_func("inspector_more_label", lambda: tr("common.more"))
        model.bind_func("inspector_training_tooltip", lambda: " · ".join(filter(None, (
            self._selected_details_rows().get("iteration", ""), self._selected_details_rows().get("strategy", "")))))
        model.bind_func("inspector_model_tooltip", lambda: " · ".join(filter(None, (
            self._selected_details_rows().get("gaussians", ""),
            "SH " + self._selected_details_rows()["sh_degree"] if self._selected_details_rows().get("sh_degree") else ""))))
        model.bind_func("inspector_reclaimable_tooltip", lambda: "{} ({})".format(
            self._selected_details_rows().get("dead_bytes", ""), self._selected_details_rows().get("reclaimable_percent", "")))
        model.bind_func("inspector_height", lambda: f"{self._inspector_preferred_height:.1f}dp")
        model.bind_func("inspector_style_height", self.get_inspector_style_height)
        model.bind_func("inspector_reserved_height", lambda: (
            f"{self._inspector_band_height() + 8.0:.1f}dp" if self._layout_class == "medium" else "0dp"
        ))
        model.bind_func("tray_height", lambda: f"{self._tray_height:.1f}dp")
        model.bind_func("sidebar_height", lambda: f"{self._sidebar_height:.1f}dp")
        model.bind_func("main_min_height", lambda: f"{self._main_min_height:.1f}dp")
        model.bind_func(
            "bottom_panel_height", lambda: f"{self._bottom_panel_height:.1f}dp"
        )
        model.bind_func(
            "bottom_panel_resize_dragging", lambda: self._bottom_panel_dragging
        )
        model.bind_func(
            "asset_card_slot_width", lambda: f"{self._asset_card_slot_width:.1f}dp"
        )
        model.bind_func(
            "asset_card_thumbnail_height",
            lambda: f"{card_geometry(max(1.0, self._asset_card_slot_width - 2.0))['thumbnail_height']:.1f}dp",
        )
        for field in (
            "asset_list_top_spacer_height",
            "asset_list_bottom_spacer_height",
            "asset_gallery_top_spacer_height",
            "asset_gallery_bottom_spacer_height",
        ):
            model.bind_func(
                field,
                lambda field=field: f"{getattr(self, '_' + field):.1f}dp",
            )

        model.bind_func("is_floating", lambda: self._is_floating)
        model.bind_func("inspector_expanded", lambda: self._inspector_expanded)
        model.bind_func(
            "quick_look_visible",
            lambda: self._quick_look_visible and bool(self.get_selected_asset_id()),
        )
        model.bind_func("quick_look_thumbnail", self.get_selected_asset_thumbnail_decorator)
        model.bind_func("quick_look_placeholder", self.get_selected_asset_placeholder)
        model.bind_func(
            "quick_look_has_thumbnail",
            lambda: self.get_selected_asset_thumbnail_decorator() != "none",
        )
        model.bind_func(
            "has_gallery_transfers",
            lambda: bool(self._all_transfer_rows()),
        )
        model.bind_func("asset_results_summary_visible", lambda: True)
        model.bind_func("asset_results_summary", self.get_asset_results_summary)
        model.bind_func("asset_search_empty", self.get_asset_search_empty)
        model.bind_func("catalog_notice", self.get_catalog_notice)
        model.bind_func("has_catalog_notice", self.get_has_catalog_notice)
        model.bind_func("scan_active", self.get_scan_active)
        model.bind_func("scan_status", self.get_scan_status)
        model.bind_func("has_scan_status", self.get_has_scan_status)
        model.bind_func("no_folders", lambda: not self._asset_index_folders())
        model.bind_func(
            "empty_folder",
            lambda: bool(self._asset_index_folders())
            and self._selected_folder_id in self._asset_index_folders()
            and not self._filtered_assets(),
        )
        model.bind_func("refresh_action_tooltip", self.get_refresh_action_tooltip)
        model.bind_func(
            "stop_scan_label", lambda: tr("projects.action.stop_scan")
        )

        model.bind_func("selected_asset_name", self.get_selected_asset_name)
        model.bind_func(
            "selected_asset_folder_name", self.get_selected_asset_folder_name
        )
        model.bind_func(
            "selected_asset_has_folder", self.get_selected_asset_has_folder
        )
        model.bind_func("selected_asset_path", self.get_selected_asset_path)
        model.bind_func("selected_asset_size", self.get_selected_asset_size)
        model.bind_func("selected_asset_created", self.get_selected_asset_created)
        model.bind_func("selected_asset_modified", self.get_selected_asset_modified)
        for field, getter in {
            "inspector_saved": lambda: self._selected_details_rows().get("saved", ""),
            "inspector_saved_at": lambda: self._selected_details_rows().get("saved_at", ""),
            "inspector_opened": lambda: self._selected_details_rows().get("opened", ""),
            "inspector_iteration": lambda: self._selected_details_rows().get("iteration", ""),
            "inspector_strategy": lambda: self._selected_details_rows().get("strategy", ""),
            "inspector_resumable": lambda: bool(self._selected_details_rows().get("resumable")),
            "inspector_gaussians": lambda: self._selected_details_rows().get("gaussians", ""),
            "inspector_sh_degree": lambda: self._selected_details_rows().get("sh_degree", ""),
            "inspector_dataset": lambda: self._selected_details_rows().get("dataset", ""),
            "inspector_dataset_path": lambda: self._selected_details_rows().get("dataset_path", ""),
            "inspector_dataset_reachable": lambda: bool(self._selected_details_rows().get("dataset_reachable")),
            "inspector_embedded": lambda: bool(self._selected_details_rows().get("embedded")),
            "inspector_has_metrics": lambda: bool(self._selected_details_rows().get("has_metrics")),
            "inspector_metrics": lambda: self._selected_details_rows().get("metrics", ""),
            "inspector_license": lambda: self._selected_details_rows().get("license_identifier", ""),
            "inspector_license_notice": lambda: self._selected_details_rows().get("license_notice", ""),
            "selected_project_title": lambda: self._selected_details_rows().get("title", ""),
            "inspector_physical_size": lambda: self._selected_details_rows().get("physical_size", ""),
            "inspector_dead_bytes": lambda: self._selected_details_rows().get("dead_bytes", ""),
            "inspector_reclaimable": lambda: self._selected_details_rows().get("reclaimable_percent", ""),
            "inspector_saves": lambda: self._selected_details_rows().get("saves", ""),
            "inspector_autosave_newer": lambda: bool(self._selected_details_rows().get("autosave_newer")),
            "inspector_has_details": lambda: bool(self._selected_inspection().get("details")),
            "inspector_card_diagnostic": lambda: str(getattr(self._selected_inspection().get("card"), "diagnostic", "") or ""),
            "inspector_can_resume": lambda: bool(self._selected_details_rows().get("resumable")),
            "inspector_operations_expanded": self.get_operations_expanded,
            "inspector_has_saved": lambda: bool(self._selected_details_rows().get("saved")),
            "inspector_has_saved_at": lambda: bool(self._selected_details_rows().get("saved_at")),
            "inspector_has_opened": lambda: bool(self._selected_details_rows().get("opened")),
            "inspector_has_iteration": lambda: bool(self._selected_details_rows().get("iteration")),
            "inspector_has_strategy": lambda: bool(self._selected_details_rows().get("strategy")),
            "inspector_has_gaussians": lambda: bool(self._selected_details_rows().get("gaussians")),
            "inspector_has_sh_degree": lambda: bool(self._selected_details_rows().get("sh_degree")),
            "inspector_has_dataset": lambda: bool(self._selected_details_rows().get("dataset")),
            "inspector_has_license": lambda: bool(self._selected_details_rows().get("license_identifier")),
            "inspector_has_title": lambda: bool(self._selected_details_rows().get("title")),
            "inspector_has_physical_size": lambda: bool(self._selected_details_rows().get("physical_size")),
            "inspector_has_dead_bytes": lambda: bool(self._selected_details_rows().get("dead_bytes")),
            "inspector_has_saves": lambda: bool(self._selected_details_rows().get("saves")),
            "inspector_autosave_newer_label": lambda: tr("projects.status.autosave_newer"),
            "inspector_verify_result": lambda: self._verify_results.get(self.get_selected_asset_id(), ""),
            "has_inspector_verify_result": lambda: bool(self._verify_results.get(self.get_selected_asset_id(), "")),
            "inspector_verify_label": lambda: tr("projects.property.verify"),
        }.items():
            model.bind_func(field, getter)
        model.bind_func("selected_health_state", self.get_selected_health_state)
        model.bind_func("selected_health_label", self.get_selected_health_label)
        model.bind_func("selected_has_problem", self.selected_has_problem)
        model.bind_func("selected_fix_label", self.get_selected_fix_label)
        model.bind_func("selected_fix_action", self.get_selected_fix_action)
        model.bind_func(
            "selected_asset_file_missing", self.get_selected_asset_file_missing
        )
        model.bind_func(
            "selected_asset_can_locate", self.get_selected_asset_can_locate
        )
        model.bind_func(
            "selected_fix_requires_action",
            lambda: self.selected_has_problem() and not self.get_selected_asset_can_locate(),
        )
        model.bind_func("locate_section_title", self.get_locate_section_title)
        model.bind_func(
            "selected_asset_relocation_candidate",
            self.get_selected_asset_relocation_candidate,
        )
        model.bind_func(
            "selected_asset_has_relocation_candidate",
            self.get_selected_asset_has_relocation_candidate,
        )
        model.bind_func(
            "selected_asset_expected_path", self.get_selected_asset_path
        )
        model.bind_func("selected_folder_name", self.get_selected_folder_name)
        model.bind_func("selected_folder_path", self.get_selected_folder_path)
        model.bind_func(
            "selected_folder_asset_count", self.get_selected_folder_asset_count
        )

        model.bind_func("panel_label", lambda: tr("projects.panel_title"))
        labels = {
            "close_label": "common.close",
            "import_project_label": "projects.action.add_existing",
            "import_project_tooltip": "projects.tooltip.add_existing",
            "no_search_results_label": "projects.status.no_search_results",
            "clear_search_label": "projects.action.clear_search",
            "search_placeholder": "projects.toolbar.search_icon",
            "search_icon_label": "projects.toolbar.search_icon",
            "all_assets_label": "projects.sidebar.all_assets",
            "folders_title": "projects.sidebar.folders",
            "col_name_label": "projects.property.name",
            "col_folder_label": "projects.property.folder",
            "col_size_label": "projects.property.size",
            "col_modified_label": "projects.property.modified",
            "info_tab_label": "projects.info_panel.info",
            "select_item_hint": "projects.status.select_item",
            "asset_details_title": "projects.info_panel.asset_details",
            "folder_details_title": "projects.info_panel.folder_details",
            "file_not_found_title": "projects.info_panel.file_not_found",
            "found_at_label": "projects.info_panel.found_at",
            "use_found_location_label": "projects.action.use_found_location",
            "prop_folder_label": "projects.property.folder",
            "prop_size_label": "projects.property.size",
            "prop_path_label": "projects.property.path",
            "prop_created_label": "projects.property.created",
            "prop_modified_label": "projects.property.modified",
            "prop_expected_path_label": "projects.property.expected_path",
            "prop_assets_label": "projects.property.assets",
            "locate_file_button_label": "projects.action.locate_file",
            "load_button_label": "menu.file.open_project",
            "inspector_title": "projects.inspector.title",
            "problem_title": "projects.inspector.problem",
            "project_section_title": "projects.inspector.project",
            "gallery_section_title": "projects.inspector.gallery",
            "file_section_title": "projects.inspector.file",
            "operations_section_title": "projects.inspector.operations",
            "open_button_label": "projects.action.open",
            "resume_button_label": "projects.action.resume_training",
            "scope_all_label": "projects.sidebar.all_projects",
            "scope_recent_label": "projects.sidebar.recent",
            "scope_published_label": "projects.gallery.sidebar.published",
            "view_menu_label": "projects.toolbar.view",
            "filter_label": "projects.toolbar.filter",
            "check_gallery_label": "projects.action.check_gallery",
            "no_folders_label": "projects.status.no_folders",
            "empty_folder_label": "projects.status.empty_folder",
            "thumbnail_size_label": "projects.toolbar.thumbnail_size",
            "resize_navigator_label": "projects.accessibility.resize_navigator",
            "resize_inspector_label": "projects.accessibility.resize_inspector",
            "resize_inspector_height_label": "projects.accessibility.resize_inspector_height",
            "resize_transfers_tray_label": "projects.accessibility.resize_transfers_tray",
            "inspector_saved_label": "projects.property.saved",
            "inspector_date_label": "projects.property.date",
            "inspector_opened_label": "projects.property.opened",
            "inspector_training_label": "projects.property.training",
            "inspector_model_label": "projects.property.model",
            "inspector_dataset_label": "projects.property.dataset",
            "inspector_metrics_label": "projects.property.metrics",
            "inspector_license_label": "projects.property.license",
            "inspector_title_label": "projects.property.title",
            "inspector_reclaimable_label": "projects.property.reclaimable",
            "inspector_saves_label": "projects.property.saves",
            "inspector_autosave_label": "projects.property.autosave",
        }
        for field, key in labels.items():
            model.bind_func(field, lambda key=key: tr(key))

        model.bind_func("dialog_visible", lambda: bool(self._dialog_kind))
        model.bind_func("dialog_kind", lambda: self._dialog_kind)
        model.bind_func("dialog_title", self.get_dialog_title)
        model.bind_func("dialog_busy", lambda: self._dialog_busy)
        model.bind_func("dialog_message", lambda: str(self._dialog_data.get("message", "")))
        for field in ("name", "path", "identifier", "notice", "destination", "format", "source", "recovery_note"):
            model.bind("dialog_" + field, lambda field=field: str(self._dialog_data.get(field, "")), lambda value, field=field: self._set_dialog_value(field, value))
        model.bind_func("dialog_confirm_label", self.get_dialog_confirm_label)
        model.bind_func("dialog_open_new_label", lambda: tr("projects.action.open_as_new_project"))
        model.bind_func("dialog_resume_label", lambda: tr("projects.action.resume_from_here"))
        model.bind_func("dialog_destination_placeholder", lambda: tr("projects.dialog.choose_destination"))
        model.bind_func("dialog_identifier_placeholder", lambda: tr("projects.property.identifier"))
        model.bind_func("dialog_notice_placeholder", lambda: tr("projects.property.notice"))
        model.bind_func("dialog_name_placeholder", lambda: tr("projects.property.display_name"))
        model.bind_func("dialog_projected_label", lambda: tr("projects.dialog.projected_size"))
        model.bind_func("dialog_path_label", lambda: tr("projects.property.path"))
        model.bind_func("dialog_reclaimable_label", lambda: tr("projects.property.reclaimable"))
        model.bind_func("dialog_keep_latest_label", lambda: tr("projects.dialog.keep_latest_checkpoint"))
        model.bind_func("dialog_drop_embedded_label", lambda: tr("projects.dialog.drop_embedded_dataset"))
        model.bind_func("dialog_choose_destination_label", lambda: tr("projects.dialog.choose_destination"))
        model.bind_func("dialog_current_viewport_label", lambda: tr("projects.dialog.current_viewport"))
        model.bind_func("dialog_first_dataset_label", lambda: tr("projects.dialog.first_dataset_image"))
        model.bind_func("dialog_first_embedded_label", lambda: tr("projects.dialog.first_embedded_image"))
        model.bind_func("dialog_image_file_label", lambda: tr("projects.dialog.image_file"))
        model.bind_func("dialog_cancel_label", lambda: tr("common.cancel"))
        model.bind_func("dialog_has_rows", lambda: bool(self._dialog_data.get("rows")))
        for field in ("path", "message", "reclaimable_bytes", "projected_compact_size"):
            model.bind_func("dialog_has_" + field, lambda field=field: bool(self._dialog_data.get(field)))
        for kind in ("save_history", "reduce_size", "export_as", "update_thumbnail", "set_license", "rename", "repair", "locate_dataset"):
            model.bind_func("dialog_is_" + kind, lambda kind=kind: self._dialog_kind == kind)
        model.bind_func("dialog_save_ready", lambda: self._dialog_kind == "save_history" and not self._dialog_busy)
        model.bind_func("dialog_other_ready", lambda: bool(self._dialog_kind) and self._dialog_kind != "save_history" and not self._dialog_busy)
        model.bind_func("dialog_reclaimable_bytes", lambda: str(self._dialog_data.get("reclaimable_bytes", "")))
        model.bind_func("dialog_projected_size", lambda: str(self._dialog_data.get("projected_compact_size", "")))
        model.bind_func("dialog_drop_checkpoints", lambda: self._dialog_drop_checkpoints)
        model.bind_func("dialog_drop_dataset", lambda: self._dialog_drop_dataset)
        model.bind_func("inspector_operation_actions", self.get_selected_operation_actions)

        model.bind_record_list("folders")
        model.bind_record_list("assets")
        model.bind_record_list("inspector_operation_rows")
        model.bind_record_list("dialog_rows")
        for event, handler in (
            ("open_gallery", self.on_open_gallery),
            ("toggle_folders_collapsed", self.toggle_folders_collapsed),
            ("add_asset_folder", self.add_asset_folder),
            ("on_import_project", self.on_import_project),
            ("on_load_asset", self.on_load_asset),
            ("set_view_mode", self.set_view_mode),
            ("cycle_sort_mode", self.cycle_sort_mode),
            ("open_sort_menu", self.open_sort_menu),
            ("close_quick_look", self.close_quick_look),
            ("open_view_menu", self.open_view_menu),
            ("open_filter_menu", self.open_filter_menu),
            ("toggle_inspector", self.toggle_inspector),
            ("toggle_inspector_section", self.toggle_inspector_section),
            ("open_inspector_menu", self.open_inspector_menu),
            ("refresh_catalog", self.refresh_catalog),
            ("on_locate_file", self.on_locate_file),
            ("on_use_found_location", self.on_use_found_location),
            ("on_selected_fix", self.on_selected_fix),
            ("open_project_operation", self.open_project_operation),
            ("dialog_cancel", self.close_project_dialog),
            ("dialog_confirm", self.confirm_project_dialog),
            ("dialog_restore_new", self.dialog_restore_new),
            ("dialog_resume_here", self.dialog_resume_here),
            ("dialog_choose_destination", self.dialog_choose_destination),
            ("dialog_set_format", self.dialog_set_format),
            ("dialog_set_source", self.dialog_set_source),
            ("dialog_set_license", self.dialog_set_license),
            ("dialog_set_name", self.dialog_set_name),
            ("dialog_select_generation", self.dialog_select_generation),
            ("dialog_toggle_checkpoints", self.dialog_toggle_checkpoints),
            ("dialog_toggle_dataset", self.dialog_toggle_dataset),
            ("toggle_operations", self.toggle_operations),
            ("on_bottom_panel_resize_start", self.on_bottom_panel_resize_start),
            ("close_panel", self._on_close_panel),
            ("clear_search", lambda *_args: self.set_search_query("")),
        ):
            model.bind_event(event, handler)
        self._handle = model.get_handle()
        self._handle.update_record_list("transfer_rows", self._all_transfer_rows())
        self._handle.update_record_list("inspector_operation_rows", self.get_selected_operation_actions())
        self._handle.update_record_list("dialog_rows", self._dialog_data.get("rows", []))

    def get_search_query(self) -> str:
        return self._search_query

    def _set_scope_value(self, value: Any) -> None:
        self._select_folder_id(str(value or SCOPE_ALL))

    def get_thumbnail_size(self) -> float:
        name = self._layout_class or breakpoint_for_width(self._content_width or 1100.0)
        return self._thumbnail_sizes.get(name, 168.0)

    def get_navigator_style_width(self) -> str:
        if self._layout_class in ("compact", "narrow"):
            return "auto"
        return f"{self._navigator_width:.1f}dp"

    def get_inspector_style_width(self) -> str:
        if self._layout_class != "wide":
            return "auto"
        return f"{self._inspector_width:.1f}dp"

    def get_inspector_style_height(self) -> str:
        if self._layout_class == "wide":
            return "auto"
        if self._layout_class in ("compact", "narrow"):
            return "32dp"
        return f"{self._inspector_band_height():.1f}dp"

    def _inspector_band_height(self) -> float:
        height = self._host_geometry[1] if self._host_geometry else 700.0
        return min(self._inspector_preferred_height, max(120.0, height / 2.0))

    def _dirty_layout_fields(self) -> None:
        self._dirty_fields(
            "is_compact", "is_narrow", "is_medium", "is_wide",
            "is_floating", "navigator_width", "navigator_style_width",
            "inspector_width", "inspector_style_width", "inspector_height",
            "inspector_style_height", "inspector_reserved_height", "thumbnail_size", "asset_card_slot_width",
            "asset_card_thumbnail_height", "tray_height", "bottom_panel_height",
            "sidebar_height", "main_min_height",
        )

    def set_thumbnail_size(self, value: Any) -> None:
        try:
            number = min(320.0, max(112.0, float(value)))
        except (TypeError, ValueError):
            return
        name = self._layout_class or breakpoint_for_width(self._content_width or 1100.0)
        if abs(self._thumbnail_sizes.get(name, number) - number) < 0.1:
            return
        self._thumbnail_sizes[name] = number
        self._reset_scroll()
        self._refresh_records(assets=True)
        self._dirty_fields("thumbnail_size")

    def set_search_query(self, value: str) -> None:
        self._search_query = str(value or "")
        self._dirty_fields("search_is_empty")
        visible_ids = {
            str(asset.get("id") or asset.get("project_uuid") or "")
            for asset in self._filtered_assets()
        }
        self._selected_asset_ids.intersection_update(visible_ids)
        if self._selection_cursor_id not in visible_ids:
            self._selection_cursor_id = next(iter(self._selected_asset_ids), None)
        self._update_selection_type()
        self._reset_scroll()
        self._refresh_records(assets=True, folders=True)
        self._dirty_selection()

    def get_sort_label(self) -> str:
        return tr("projects.toolbar.sort")

    def _sort_field_label(self, field: str) -> str:
        return tr({
            "name": "projects.property.name", "saved": "projects.property.saved",
            "opened": "projects.property.opened", "size": "projects.property.size",
            "iteration": "projects.sort.iteration", "published": "projects.gallery.sidebar.published",
        }[field])

    def get_sort_tooltip(self) -> str:
        direction = tr("projects.sort.descending" if self._sort_descending else "projects.sort.ascending")
        return f"{self._sort_field_label(self._sort_mode)} · {direction}"

    def _sort_menu_items(self) -> List[Dict[str, Any]]:
        fields = ["name", "saved", "opened", "size"]
        if any(self._cached_iteration(asset) is not None for asset in self._asset_index_assets().values()):
            fields.append("iteration")
        fields.append("published")
        items = [{"label": self._sort_field_label(field), "action": "sort:" + field,
                  "is_active": self._sort_mode == field} for field in fields]
        items.extend([
            {"label": tr("projects.sort.ascending"), "action": "order:ascending", "separator_before": True,
             "is_active": not self._sort_descending},
            {"label": tr("projects.sort.descending"), "action": "order:descending", "is_active": self._sort_descending},
        ])
        return items

    def _choose_sort(self, action: str) -> None:
        kind, _, value = action.partition(":")
        if kind == "sort" and value in self.SORT_MODES:
            self._sort_mode = value
            self._sort_descending = value != "name"
        elif kind == "order" and value in ("ascending", "descending"):
            self._sort_descending = value == "descending"
        else:
            return
        self._reset_scroll()
        self._refresh_records(assets=True)
        self._dirty_fields("sort_label", "sort_tooltip")

    def open_sort_menu(self, _handle=None, _event=None, _args=None) -> None:
        self._show_shared_context_menu(self._sort_menu_items(), self._choose_sort)

    def get_filter_label(self) -> str:
        return tr({
            "all": "projects.filter.all",
            "attention": "projects.filter.attention",
            "not_published": "projects.filter.not_published",
            "published": "projects.filter.published",
            "missing": "projects.filter.missing",
            "checkpoint": "projects.filter.checkpoint",
            "dataset": "projects.filter.dataset",
            "gallery": "projects.filter.gallery",
        }.get(self._active_filter, "projects.toolbar.filter"))

    def open_filter_menu(self, _handle=None, _ev=None, _args=None):
        filters = [
            ("projects.filter.all", "all"),
            ("projects.filter.attention", "attention"),
            ("projects.filter.not_published", "not_published"),
            ("projects.filter.published", "published"),
            ("projects.filter.missing", "missing"),
            ("projects.filter.checkpoint", "checkpoint"),
            ("projects.filter.dataset", "dataset"),
            ("projects.filter.gallery", "gallery"),
        ]

        def choose(action: str) -> None:
            self._set_filter(action)

        self._show_shared_context_menu(
            [{"label": tr(label), "action": action} for label, action in filters], choose
        )

    def _set_filter(self, value: str) -> None:
        if value not in {"all", "attention", "not_published", "published", "missing", "checkpoint", "dataset", "gallery"}:
            return
        self._active_filter = value
        self._reset_scroll()
        self._refresh_records(assets=True, folders=True)
        self._dirty_selection()
        self._dirty_fields("active_filter_label")

    def get_selected_asset_id(self) -> str:
        return next(iter(self._selected_asset_ids)) if len(self._selected_asset_ids) == 1 else ""

    def get_selected_count(self) -> int:
        return len(self._selected_asset_ids)

    def get_selection_type(self) -> str:
        return self._selection_type

    def get_selected_count_text(self) -> str:
        count = len(self._selected_asset_ids)
        if count == 0:
            return tr("projects.status.select_item")
        if count == 1:
            return tr("projects.status.one_item_selected")
        return tr("projects.status.multi_items_selected", count=count)

    @staticmethod
    def _sort_text(value: Any) -> str:
        return str(value or "").casefold()

    @staticmethod
    def _format_size(value: Any) -> str:
        return format_size(value)

    @staticmethod
    def _format_unix_ns(value: Any) -> str:
        try:
            nanoseconds = int(value)
            if nanoseconds <= 0:
                return ""
            return datetime.fromtimestamp(nanoseconds / 1_000_000_000).strftime(
                "%Y-%m-%d %H:%M"
            )
        except (OSError, OverflowError, TypeError, ValueError):
            return ""

    def _asset_index_assets(self) -> Dict[str, Dict[str, Any]]:
        if self._library_service is not None:
            assets = self._library_service.snapshot().get("projects", {})
        else:
            assets = getattr(self._asset_index, "assets", {}) if self._asset_index else {}
        return assets if isinstance(assets, dict) else {}

    def _asset_dict(self, asset_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if asset_id and asset_id.startswith("remote:"):
            return self._gallery_remote_assets().get(asset_id)
        if not asset_id or not self._asset_index:
            return None
        if self._library_service is not None:
            return self._library_service.snapshot().get("projects", {}).get(asset_id)
        getter = getattr(self._asset_index, "get_asset_dict", None)
        if callable(getter):
            return getter(asset_id)
        return self._asset_index_assets().get(asset_id)

    def _asset_index_folders(self) -> Dict[str, Dict[str, Any]]:
        if self._library_service is not None:
            folders = self._library_service.snapshot().get("folders", {})
        else:
            folders = getattr(self._asset_index, "folders", {}) if self._asset_index else {}
        return folders if isinstance(folders, dict) else {}

    def _library_command(self, command: str, *args: Any, **kwargs: Any) -> Any:
        if self._library_service is not None:
            return self._library_service._call(command, *args, **kwargs)
        return getattr(self._asset_index, command)(*args, **kwargs)

    @staticmethod
    def _native_io_call(name: str, *args: Any, **kwargs: Any) -> Any:
        io = getattr(lf, "io", None)
        function = getattr(io, name, None) if io is not None else None
        if not callable(function):
            raise RuntimeError(f"Project operation is unavailable: {name}")
        return function(*args, **kwargs)

    def _ensure_inspection_pipeline(self) -> InspectionFactsPipeline:
        if self._inspection_pipeline is None:
            scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
            self._inspection_pipeline = InspectionFactsPipeline(
                lambda path: self._native_io_call("inspect_project_card", path),
                lambda path: self._native_io_call("inspect_project_details", path),
                self._on_inspection_result,
                scheduler=scheduler if callable(scheduler) else None,
            )
        return self._inspection_pipeline

    def _start_inspection_refresh(self) -> None:
        if not self._asset_index or not self._panel_mounted or not self._handle:
            return
        entries = self._window_assets(self._filtered_assets())
        selected = self.get_selected_asset_id()
        self._ensure_inspection_pipeline().refresh(entries, selected)

    def _on_inspection_result(self, asset_id: str, kind: str, result: Any, error: Optional[Exception]) -> None:
        if not self._panel_mounted:
            return
        if error is not None:
            self._inspection_errors[asset_id] = str(error)
            if kind == "card":
                self._dirty_fields("assets", "selected_has_problem", "selected_health_label")
            return
        if asset_id != self.get_selected_asset_id() and kind == "details":
            # Background details are useful for later selection but must not
            # cause a hidden Inspector to redraw as if it were selected.
            self._inspection_by_asset.setdefault(asset_id, {})[kind] = result
        else:
            self._inspection_by_asset.setdefault(asset_id, {})[kind] = result
        if kind == "details":
            iteration = self._details_iteration(result)
            if iteration is not None:
                self._cache_iteration(asset_id, iteration)
            if self._dialog_kind == "save_history" and self._dialog_busy and asset_id == self._dialog_asset_id:
                self._dialog_busy = False
                self._dialog_data = dialog_model(
                    "save_history",
                    entry=self._dialog_entry(),
                    details=result,
                    format_size=self._format_size,
                    format_time=self._format_unix_ns,
                )
                if self._handle:
                    self._handle.update_record_list("dialog_rows", self._dialog_data.get("rows", []))
        self._refresh_records(assets=True)
        self._dirty_selection()

    @staticmethod
    def _details_iteration(details: Any) -> Optional[int]:
        card = getattr(details, "card", None)
        iteration = getattr(card, "iteration", None)
        if iteration is not None:
            try:
                return int(iteration)
            except (TypeError, ValueError):
                pass
        checkpoints = list(getattr(details, "retained_checkpoints", []) or [])
        values = [getattr(item, "iteration", None) for item in checkpoints]
        values = [int(item) for item in values if item is not None]
        if values:
            return max(values)
        saves = list(getattr(details, "save_history", []) or [])
        values = [getattr(item, "checkpoint_iteration", None) for item in saves]
        values = [int(item) for item in values if item is not None]
        return max(values) if values else None

    def _cache_iteration(self, asset_id: str, iteration: int) -> None:
        self._inspection_by_asset.setdefault(asset_id, {})["iteration"] = int(iteration)

    def _cached_iteration(self, asset: Dict[str, Any]) -> Optional[int]:
        cached = self._inspection_by_asset.get(asset.get("id"), {})
        value = cached.get("iteration", asset.get("iteration"))
        return int(value) if value is not None else None

    def _selected_inspection(self) -> Dict[str, Any]:
        return self._inspection_by_asset.get(self.get_selected_asset_id(), {})

    def _selected_details_rows(self) -> Dict[str, Any]:
        details = self._selected_inspection().get("details")
        if details is None:
            return {}
        return details_rows(
            self._get_selected_asset() or {},
            details,
            format_size=self._format_size,
            format_time=self._format_unix_ns,
        )

    def get_selected_operation_actions(self) -> List[Dict[str, Any]]:
        asset = self._get_selected_asset()
        if not asset:
            return []
        details = self._selected_inspection().get("details")
        rows = operation_actions(
            asset,
            details,
            has_operation=bool(self._project_operations),
        )
        keys = {
            "save_history": "projects.action.save_history",
            "reduce_size": "projects.action.reduce_size",
            "embed_dataset": "projects.action.embed_dataset",
            "locate_dataset": "projects.action.locate_dataset",
            "export_as": "projects.action.export_as",
            "update_thumbnail": "projects.action.update_thumbnail",
            "set_license": "projects.action.set_license",
            "rename": "projects.action.rename",
        }
        return [{**row, "label": tr(keys.get(row["action"], row["label"]))} for row in rows]

    def get_operations_expanded(self) -> bool:
        if self._operations_expanded is not None:
            return self._operations_expanded
        if self.selected_has_problem():
            return True
        rows = self._selected_details_rows()
        try:
            reclaimable = float(str(rows.get("reclaimable_percent", "0")).rstrip("%"))
        except (TypeError, ValueError):
            reclaimable = 0.0
        return reclaimable > 10.0

    def toggle_operations(self, _handle=None, _ev=None, _args=None) -> None:
        self._operations_expanded = not self.get_operations_expanded()
        self._dirty_fields("inspector_operations_expanded")

    def toggle_inspector_section(self, _handle=None, _ev=None, args=None) -> None:
        section = str(args[0]) if args else ""
        if section in self._inspector_sections:
            self._inspector_sections[section] = not self._inspector_sections[section]
            self._dirty_fields("inspector_" + section + "_expanded")

    def open_inspector_menu(self, _handle=None, _ev=None, _args=None) -> None:
        asset_id = self.get_selected_asset_id()
        if asset_id:
            self._show_asset_context_menu(asset_id)

    def _all_transfer_rows(self) -> List[Dict[str, Any]]:
        rows = list(transfer_rows(self._gallery_state))
        for operation in self._project_operations.values():
            progress = float(operation.get("progress", 0.0) or 0.0)
            status = str(operation.get("status", "running"))
            rows.append({
                "id": operation["id"],
                "title": operation.get("title", "Project operation"),
                "direction": "→",
                "status": status,
                "bytes": "",
                "phase": operation.get("phase", "Preparing"),
                "reason": operation.get("reason", "") if status == "failed" else "",
                "detail": "",
                "progress": progress,
                "progress_width": f"{progress:.1f}%",
                "indeterminate": status == "running" and progress <= 0,
                "can_pause": False,
                "can_resume": False,
                "can_cancel": False,
            })
        return rows

    def _refresh_transfer_rows(self) -> None:
        if self._handle:
            self._handle.update_record_list("transfer_rows", self._all_transfer_rows())
            self._handle.dirty("transfer_rows")
            self._handle.dirty("has_gallery_transfers")
        self._request_model_update()

    def _schedule_ui(self, callback: Callable[[], None]) -> None:
        scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
        if callable(scheduler):
            scheduler(callback)
        else:
            callback()

    def _default_folder_id(self) -> Optional[str]:
        folders = self._asset_index_folders()
        if "default" in folders:
            return "default"
        return min(folders, key=lambda folder_id: self._sort_text(folders[folder_id].get("name"))) if folders else None

    def _asset_matches_query(self, asset: Dict[str, Any], query: str) -> bool:
        if not query:
            return True
        haystack = " ".join(
            str(value)
            for value in (
                asset.get("name"),
                asset.get("path"),
                asset.get("type"),
                asset.get("project_uuid"),
                self._folder_name(asset.get("folder_id")),
                "licht project",
            )
        ).casefold()
        return query in haystack

    def _asset_matches_filter(self, asset: Dict[str, Any]) -> bool:
        active = self._active_filter
        if active == "all":
            return True
        facts = self._gallery_facts(asset)
        scene = self._gallery_scene(asset)
        if active == "attention":
            return bool(str(asset.get("status") or "") not in ("", "AVAILABLE")) or bool(facts.get("action"))
        if active == "not_published":
            return scene is None and not asset.get("remote_only")
        if active == "published":
            return scene is not None
        if active == "missing":
            return not bool(asset.get("exists", True)) or str(asset.get("status") or "") == "MISSING"
        if active == "checkpoint":
            return bool(asset.get("has_checkpoint") or asset.get("checkpoint_iteration"))
        if active == "dataset":
            return bool(asset.get("has_dataset") or asset.get("dataset_path"))
        if active == "gallery":
            return bool(scene or asset.get("remote_only") or facts.get("relationship") not in (None, "unlinked"))
        return True

    def _repair_selection(self) -> None:
        assets = self._all_display_assets()
        folders = self._asset_index_folders()
        self._selected_asset_ids.intersection_update(assets)
        if self._selection_cursor_id not in assets:
            self._selection_cursor_id = next(iter(self._selected_asset_ids), None)
        if self._selected_folder_id not in {*folders, SCOPE_ALL, SCOPE_RECENT, *GALLERY_SCOPES}:
            self._selected_folder_id = SCOPE_ALL
        self._update_selection_type()

    def _repair_selected_folder(self) -> Optional[str]:
        self._repair_selection()
        return self._selected_folder_id

    def _update_selection_type(self) -> None:
        if len(self._selected_asset_ids) > 1:
            self._selection_type = "multiple"
        elif len(self._selected_asset_ids) == 1:
            self._selection_type = "asset"
        elif self._selected_folder_id in self._asset_index_folders():
            self._selection_type = "folder"
        else:
            self._selection_type = "none"

    def _folder_name(self, folder_id: Any) -> str:
        folder = self._asset_index_folders().get(str(folder_id), {})
        return str(folder.get("name") or "")

    def _project_available(self, asset: Dict[str, Any]) -> bool:
        if "available" in asset:
            return bool(asset.get("available"))
        return bool(asset.get("exists", True))

    def _project_status_label(self, asset: Dict[str, Any]) -> str:
        status = str(asset.get("status") or "UNVERIFIED")
        key = {
            "AVAILABLE": "projects.status.available",
            "MISSING": "projects.status.missing",
            "IDENTITY_MISMATCH": "projects.status.identity_mismatch",
            "REPAIR_ONLY": "projects.status.needs_repair",
            "UNSUPPORTED_NEWER": "projects.status.newer_version",
        }.get(status, "projects.status.unverified")
        return tr(key)

    @staticmethod
    def _placeholder_label(display_name: str) -> str:
        return " ".join(str(display_name or "").split()[:2])[:24]

    def _get_asset_display_name(self, asset: Dict[str, Any]) -> str:
        asset_id = str(asset.get("id") or asset.get("project_uuid") or "")
        details = self._inspection_by_asset.get(asset_id, {}).get("details")
        title = str(getattr(getattr(details, "card", None), "title", "") or "")
        if title.strip():
            return title.strip()
        if "display_name" in asset and asset.get("display_name"):
            return str(asset["display_name"])
        if callable(globals().get("display_name")):
            value = display_name(asset)
            if value:
                return value
        path = str(asset.get("path") or "")
        path_stem = Path(path).stem if path else ""
        return str(asset.get("name") or path_stem or tr("projects.unnamed"))

    @staticmethod
    def _thumbnail_image_decorator(source: str) -> str:
        assert " " not in source
        return f"image({source} {_THUMBNAIL_FIT_ALIGN})"

    @staticmethod
    def _thumbnail_source_from_decorator(decorator: str) -> str:
        if not decorator.startswith("image(") or not decorator.endswith(")"):
            return ""
        inner = decorator[len("image(") : -1]
        suffix = f" {_THUMBNAIL_FIT_ALIGN}"
        if inner.endswith(suffix):
            inner = inner[: -len(suffix)]
        return inner

    @staticmethod
    def _thumbnail_decorator(asset: Dict[str, Any]) -> str:
        poster = asset.get("poster_path")
        if poster and (asset.get("prefer_poster") or not ((asset.get("has_preview") and asset.get("exists")) or asset.get("fallback_preview_path"))):
            return AssetManagerPanel._thumbnail_decorator({"fallback_preview_path": poster})
        if asset.get("has_preview") and asset.get("exists"):
            path = quote(str(asset.get("path") or ""), safe=_RML_PATH_SAFE_CHARS)
            revision_value = asset.get("commit_uuid") or "-".join(
                str(asset.get(field) or 0)
                for field in ("generation", "saved_at_unix_ns", "file_size_bytes")
            )
            revision = quote(str(revision_value), safe="-._~")
            return AssetManagerPanel._thumbnail_image_decorator(
                f"preview://kind=licht&thumb=256&rev={revision}&path={path}"
            )
        fallback = str(asset.get("fallback_preview_path") or "")
        if not fallback:
            return "none"
        fallback_path = Path(fallback)
        try:
            if not fallback_path.is_file():
                return "none"
            stat = fallback_path.stat()
        except OSError:
            return "none"
        revision = quote(f"{stat.st_size}-{stat.st_mtime_ns}", safe="-._~")
        encoded = quote(fallback, safe=_RML_PATH_SAFE_CHARS)
        return AssetManagerPanel._thumbnail_image_decorator(
            f"preview://kind=image&thumb=256&rev={revision}&path={encoded}"
        )

    def _sync_info_thumbnail(self, doc):
        query = getattr(doc, "query_selector", None)
        header = query(".asset-info-header") if callable(query) else None
        if header is None:
            return False
        element = doc.get_element_by_id("asset-info-thumbnail")
        asset = self._get_selected_asset() or {}
        decorator = self._thumbnail_decorator(self._asset_with_poster(asset)) if asset else "none"
        source = self._thumbnail_source_from_decorator(decorator)
        created = element is None
        if element is None:
            layout = query(".asset-info-asset-layout") if callable(query) else None
            details = query(".asset-info-details") if callable(query) else None
            if layout is not None and layout is not header and details is not None and details is not header:
                element = layout.insert_before("div", details)
            else:
                element = header.parent().insert_before("div", header)
            element.set_id("asset-info-thumbnail")
        # The Inspector owns a 12 dp scroll gutter. The band alone uses the
        # small landscape preview; the column fills its own content width.
        width = (max(0.0, self._inspector_width - 12.0) if self._layout_class == "wide"
                 else max(0.0, self._content_width - 12.0)
                 if self._layout_class in ("compact", "narrow") else 160.0)
        geometry = (width, width * 10.0 / 16.0)
        geometry_changed = geometry != self._info_thumbnail_geometry
        if created or geometry_changed:
            self._info_thumbnail_geometry = geometry
            element.set_property("width", f"{geometry[0]:.2f}dp")
            element.set_property("height", f"{geometry[1]:.2f}dp")
            element.set_property("flex-basis", f"{geometry[1] if self._layout_class != 'medium' else geometry[0]:.2f}dp")
        changed = source != self._info_thumbnail_source
        if changed:
            release = getattr(lf.ui, "release_rml_texture", None)
            if self._info_thumbnail_source and callable(release):
                release(self._info_thumbnail_source)
            self._info_thumbnail_source = source
            element.set_property("decorator", decorator)
        if changed or created:
            element.set_property("display", "block" if source else "none")
        return changed or created or geometry_changed

    def _format_asset_for_ui(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        asset_id = str(asset.get("id") or asset.get("project_uuid") or "")
        inspected = self._inspection_by_asset.get(asset_id, {})
        card = inspected.get("card")
        if card is not None:
            # Native card values are provisional but are fresher than the
            # catalog snapshot and keep cards useful while details load.
            asset = {
                **asset,
                "has_preview": bool(getattr(card, "has_preview", asset.get("has_preview"))),
                "file_size_bytes": int(getattr(card, "physical_file_size", asset.get("file_size_bytes", 0)) or 0),
                "saved_at_unix_ns": int(getattr(card, "saved_at_unix_ns", asset.get("saved_at_unix_ns", 0)) or 0),
                "commit_uuid": str(getattr(card, "commit_uuid", asset.get("commit_uuid", "")) or asset.get("commit_uuid", "")),
            }
        folder_name = self._folder_name(asset.get("folder_id"))
        thumbnail_decorator = self._thumbnail_decorator(self._asset_with_poster(asset))
        thumbnail_source = self._thumbnail_source_from_decorator(thumbnail_decorator)
        previous_source = self._thumbnail_sources_by_asset.get(asset_id, "")
        if previous_source and previous_source != thumbnail_source:
            release_texture = getattr(lf.ui, "release_rml_texture", None)
            if callable(release_texture):
                release_texture(previous_source)
        if thumbnail_source:
            self._thumbnail_sources_by_asset[asset_id] = thumbnail_source
        else:
            self._thumbnail_sources_by_asset.pop(asset_id, None)
        display_name = self._get_asset_display_name(asset)
        return {
            **asset,
            **self._gallery_badge(asset),
            "display_name": display_name,
            "placeholder_label": self._placeholder_label(display_name),
            "id": asset_id,
            "folder_name": folder_name,
            "size_label": self._format_size(asset.get("file_size_bytes", 0)),
            "saved_label": self._format_unix_ns(asset.get("saved_at_unix_ns", 0)),
            "status_label": self._project_status_label(asset),
            "health_label": self._project_status_label(asset),
            "has_problem": str(asset.get("status") or "") not in ("", "AVAILABLE", "READING"),
            "is_selected": str(asset.get("id") or asset.get("project_uuid"))
            in self._selected_asset_ids,
            "has_preview": bool(asset.get("has_preview")),
            "shows_placeholder": thumbnail_decorator == "none",
            "can_load": self._project_available(asset),
            "thumbnail_decorator": thumbnail_decorator,
        }

    def _release_obsolete_thumbnail_sources(self) -> None:
        ids = getattr(self._asset_index, "iter_project_ids", None)
        live_ids = set(ids() if callable(ids) else self._asset_index_assets())
        live_ids.update(self._gallery_remote_assets())
        stale_ids = set(self._thumbnail_sources_by_asset).difference(live_ids)
        release_texture = getattr(lf.ui, "release_rml_texture", None)
        for asset_id in stale_ids:
            source = self._thumbnail_sources_by_asset.pop(asset_id)
            if callable(release_texture):
                release_texture(source)

    def _release_thumbnails_outside_window(self) -> None:
        visible = {
            str(asset.get("id") or asset.get("project_uuid") or "")
            for asset in self._window_assets(self._filtered_assets())
        }
        release_texture = getattr(lf.ui, "release_rml_texture", None)
        for asset_id in set(self._thumbnail_sources_by_asset).difference(visible):
            source = self._thumbnail_sources_by_asset.pop(asset_id)
            if callable(release_texture):
                release_texture(source)

    def _filtered_assets(self, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
        folder_id = self._selected_folder_id if folder_id is None else folder_id
        query = self._search_query.strip().casefold()
        rows: List[Dict[str, Any]] = []
        source = self._gallery_rows(folder_id == SCOPE_ATTENTION) if folder_id in GALLERY_SCOPES else self._asset_index_assets().values()
        for asset in source:
            if folder_id not in (None, SCOPE_ALL, SCOPE_RECENT, *GALLERY_SCOPES) and asset.get("folder_id") != folder_id:
                continue
            if not self._asset_matches_query(asset, query):
                continue
            if not self._asset_matches_filter(asset):
                continue
            rows.append(asset)
        if folder_id == SCOPE_RECENT:
            recent_files = getattr(lf, "project_recent_files", lambda: [])()
            order = {Path(path): rank for rank, path in enumerate(recent_files)}
            rows = [asset for asset in rows if Path(asset.get("path") or "") in order]
            rows.sort(key=lambda asset: order[Path(asset["path"])])
            rows = rows[:10]
        else:
            recent = {str(Path(path)): -rank for rank, path in enumerate(
                getattr(lf, "project_recent_files", lambda: [])())} if self._sort_mode == "opened" else {}
            links = self._gallery_state.get("links", {})
            def sort_value(asset):
                name = self._sort_text(self._get_asset_display_name(asset))
                value = {
                    "name": name,
                    "saved": int(asset.get("saved_at_unix_ns") or asset.get("mtime_ns") or 0),
                    "size": int(asset.get("file_size_bytes") or 0),
                    "iteration": self._cached_iteration(asset) or 0,
                    "opened": recent.get(str(Path(asset.get("path") or "")), -len(recent) - 1),
                    "published": float(links.get(asset.get("id"), {}).get("exchangedAt") or 0),
                }[self._sort_mode]
                return value, name
            rows.sort(key=sort_value, reverse=self._sort_descending)
        self._last_asset_match_count = len(rows)
        return rows

    def _window_assets(self, assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        total = len(assets)
        scroll_top = self._asset_window_scroll_top
        client_height = self._asset_window_client_height
        if self._view_mode == "gallery":
            if self._layout_class:
                columns = grid_columns(self._asset_window_client_width, self.get_thumbnail_size())
                self._asset_card_slot_width = grid_slot_width(
                    self._asset_window_client_width, self.get_thumbnail_size()
                )
                card_height = card_geometry(max(1.0, self._asset_card_slot_width - 2.0))["height"] + 2.0
                row_height = card_height + 12.0
            else:
                columns = gallery_columns(self._asset_window_client_width)
                self._asset_card_slot_width = gallery_slot_width(self._asset_window_client_width)
                row_height = ASSET_GALLERY_ROW_HEIGHT_DP
            start_row = max(0, int(scroll_top // row_height) - ASSET_WINDOW_OVERSCAN_ROWS)
            visible_rows = (
                math.ceil(client_height / row_height)
                if client_height > 0
                else ASSET_GALLERY_FALLBACK_ROWS
            ) + ASSET_WINDOW_OVERSCAN_ROWS * 2
            start = min(total, start_row * columns)
            end = min(total, (start_row + visible_rows) * columns)
            total_rows = math.ceil(total / columns) if total else 0
            end_row = math.ceil(end / columns) if end else 0
            self._asset_gallery_top_spacer_height = start_row * row_height
            self._asset_gallery_bottom_spacer_height = max(0, total_rows - end_row) * row_height
            self._asset_list_top_spacer_height = 0.0
            self._asset_list_bottom_spacer_height = 0.0
        else:
            row_height = list_row_height(gallery_column_visible=True) if self._layout_class else ASSET_LIST_ROW_HEIGHT_DP
            start = max(0, int(scroll_top // row_height) - ASSET_WINDOW_OVERSCAN_ROWS)
            visible = (
                math.ceil(client_height / row_height)
                if client_height > 0
                else ASSET_LIST_FALLBACK_ROWS
            ) + ASSET_WINDOW_OVERSCAN_ROWS * 2
            end = min(total, start + visible)
            self._asset_list_top_spacer_height = start * row_height
            self._asset_list_bottom_spacer_height = max(0, total - end) * row_height
            self._asset_gallery_top_spacer_height = 0.0
            self._asset_gallery_bottom_spacer_height = 0.0
        return assets[start:end]

    def get_filtered_assets(self) -> List[Dict[str, Any]]:
        return [self._format_asset_for_ui(asset) for asset in self._window_assets(self._filtered_assets())]

    def get_folder_list(self) -> List[Dict[str, Any]]:
        counts: Dict[str, int] = {}
        query = self._search_query.strip().casefold()
        matching_assets = [
            asset
            for asset in self._asset_index_assets().values()
            if self._asset_matches_query(asset, query)
        ]
        for asset in matching_assets:
            folder_id = str(asset.get("folder_id") or "default")
            counts[folder_id] = counts.get(folder_id, 0) + 1
        folder_rows = [
            {
                "id": folder_id,
                "name": str(folder.get("name") or tr("projects.unnamed_folder")),
                "project_count": counts.get(folder_id, 0),
                "can_manage": True,
            }
            for folder_id, folder in self._asset_index_folders().items()
        ]
        return sorted(folder_rows, key=lambda row: self._sort_text(row["name"]))

    def get_all_assets_count(self) -> int:
        query = self._search_query.strip().casefold()
        if not query:
            count = getattr(self._asset_index, "count", None)
            if callable(count):
                return int(count())
        return sum(
            self._asset_matches_query(asset, query)
            for asset in self._asset_index_assets().values()
        )

    def get_asset_results_summary(self) -> str:
        try:
            return localized_count(
                "projects.status.showing_projects", self._last_asset_match_count
            )
        except Exception:
            return str(self._last_asset_match_count)

    def get_asset_search_empty(self) -> bool:
        return bool(self._search_query.strip()) and not self._filtered_assets()

    def get_catalog_notice(self) -> str:
        if self._catalog_notice:
            return self._catalog_notice
        if self._catalog_load_failed:
            return tr("projects.status.load_failed")
        issues = getattr(self._asset_index, "load_issues", None) if self._asset_index else None
        if issues:
            return tr("projects.status.skipped_entries", count=len(issues))
        return ""

    def _set_catalog_notice(self, message: str) -> None:
        self._catalog_notice = str(message or "")
        self._dirty_fields("catalog_notice", "has_catalog_notice")

    def get_has_catalog_notice(self) -> bool:
        return bool(self.get_catalog_notice())

    def get_scan_active(self) -> bool:
        with self._folder_scan_lock:
            return bool(self._folder_scan_active or self._catalog_verify_active)

    def get_scan_status(self) -> str:
        with self._folder_scan_lock:
            active = self._folder_scan_active
            stopped = self._scan_stopped_visible
            progress = self._scan_progress
        if active:
            directories, projects, root = progress.snapshot()
            name = Path(root).name or root
            return tr(
                "projects.status.scanning",
                name=name,
                folders=directories,
                projects=projects,
            )
        if self._catalog_verify_active:
            return tr(
                "projects.status.verifying",
                count=len(self._window_assets(self._filtered_assets())),
            )
        if stopped:
            return tr("projects.status.scan_stopped")
        return ""

    def get_has_scan_status(self) -> bool:
        return bool(self.get_scan_status())

    def get_refresh_action_tooltip(self) -> str:
        if self.get_scan_active():
            return "projects.action.stop_scan"
        return "projects.tooltip.refresh"

    def _get_selected_asset(self) -> Optional[Dict[str, Any]]:
        asset_id = self.get_selected_asset_id()
        return self._asset_dict(asset_id)

    def get_selected_asset_name(self) -> str:
        asset = self._get_selected_asset()
        return self._get_asset_display_name(asset) if asset else ""

    def get_selected_asset_folder_name(self) -> str:
        asset = self._get_selected_asset()
        return self._folder_name(asset.get("folder_id")) if asset else ""

    def get_selected_asset_has_folder(self) -> bool:
        return bool(self.get_selected_asset_folder_name())

    def get_selected_asset_path(self) -> str:
        asset = self._get_selected_asset()
        return str(asset.get("path") or "") if asset else ""

    def get_selected_asset_size(self) -> str:
        asset = self._get_selected_asset()
        return self._format_size(asset.get("file_size_bytes", 0)) if asset else ""

    def get_selected_asset_created(self) -> str:
        asset = self._get_selected_asset()
        return self._format_unix_ns(asset.get("created_at_unix_ns", 0)) if asset else ""

    def get_selected_asset_modified(self) -> str:
        asset = self._get_selected_asset()
        return self._format_unix_ns(asset.get("saved_at_unix_ns", 0)) if asset else ""

    def get_selected_health_state(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        return str(asset.get("status") or "READING")

    def get_selected_health_label(self) -> str:
        asset = self._get_selected_asset()
        return self._project_status_label(asset) if asset else ""

    def selected_has_problem(self) -> bool:
        return self.get_selected_health_state() not in ("", "AVAILABLE", "READING")

    def get_selected_fix_action(self) -> str:
        return fix_action_for_health(self.get_selected_health_state()) if self.get_selected_health_state() else ""

    def get_selected_fix_label(self) -> str:
        action = self.get_selected_fix_action()
        return tr({
            "locate": "projects.action.locate",
            "verify": "projects.action.verify",
            "repair": "projects.action.repair",
            "update": "projects.action.update_version",
        }.get(action, "")) if action else ""

    def get_selected_asset_file_missing(self) -> bool:
        asset = self._get_selected_asset()
        return bool(asset) and not bool(asset.get("exists", False))

    def get_selected_asset_can_locate(self) -> bool:
        asset = self._get_selected_asset()
        return bool(asset) and str(asset.get("status") or "") in {
            "MISSING",
            "IDENTITY_MISMATCH",
        }

    def get_locate_section_title(self) -> str:
        asset = self._get_selected_asset()
        if not asset:
            return ""
        if str(asset.get("status") or "") == "IDENTITY_MISMATCH":
            return tr("projects.status.identity_mismatch")
        return tr("projects.info_panel.file_not_found")

    def get_selected_asset_relocation_candidate(self) -> str:
        asset = self._get_selected_asset()
        return str(asset.get("relocation_candidate") or "") if asset else ""

    def get_selected_asset_has_relocation_candidate(self) -> bool:
        return bool(self.get_selected_asset_relocation_candidate())

    def _get_selected_folder(self) -> Optional[Dict[str, Any]]:
        return self._asset_index_folders().get(self._selected_folder_id or "")

    def get_selected_folder_name(self) -> str:
        folder = self._get_selected_folder()
        return str(folder.get("name") or "") if folder else ""

    def get_selected_folder_path(self) -> str:
        folder = self._get_selected_folder()
        return str(folder.get("path") or "") if folder else ""

    def get_selected_folder_asset_count(self) -> int:
        if not self._selected_folder_id:
            return 0
        return sum(
            asset.get("folder_id") == self._selected_folder_id
            for asset in self._asset_index_assets().values()
        )

    def toggle_folders_collapsed(self, _handle=None, _ev=None, _args=None):
        self._folders_collapsed = not self._folders_collapsed
        self._folder_layout_initialized = True
        self._layout_signature = None
        self._dirty_fields("folders_collapsed", "folders_expanded")

    def set_view_mode(self, _handle, _ev, args):
        mode = str(args[0]) if args else ""
        if mode not in ("gallery", "list") or mode == self._view_mode:
            return
        self._view_mode = mode
        self._reset_scroll()
        self._refresh_records(assets=True)
        self._dirty_fields("is_gallery_view", "is_list_view")

    def toggle_inspector(self, _handle=None, _ev=None, _args=None):
        if self._layout_class not in ("compact", "narrow"):
            return
        self._inspector_expanded = not self._inspector_expanded
        self._dirty_fields("inspector_expanded")

    def get_selected_asset_thumbnail_decorator(self) -> str:
        asset = self._get_selected_asset()
        return self._thumbnail_decorator(self._asset_with_poster(asset)) if asset else "none"

    def get_selected_asset_placeholder(self) -> str:
        asset = self._get_selected_asset()
        return self._project_status_label(asset) if asset else ""

    def open_quick_look(self, _handle=None, _ev=None, _args=None) -> None:
        if self.get_selected_asset_id():
            self._quick_look_visible = True
            self._dirty_fields(
                "quick_look_visible", "quick_look_thumbnail", "quick_look_placeholder"
            )

    def close_quick_look(self, _handle=None, _ev=None, _args=None) -> None:
        if self._quick_look_visible:
            self._quick_look_visible = False
            self._dirty_fields("quick_look_visible")

    def cycle_sort_mode(self, _handle=None, _ev=None, _args=None):
        index = (self.SORT_MODES.index(self._sort_mode) + 1) % len(self.SORT_MODES)
        self._choose_sort("sort:" + self.SORT_MODES[index])

    def open_view_menu(self, _handle=None, _ev=None, _args=None):
        items = [
            {"label": tr("projects.filter.all"), "action": "filter:all"},
            {"label": tr("projects.filter.attention"), "action": "filter:attention"},
            {"label": tr("projects.filter.not_published"), "action": "filter:not_published"},
            {"label": tr("projects.filter.published"), "action": "filter:published"},
            {"label": tr("projects.filter.missing"), "action": "filter:missing"},
            {"label": tr("projects.filter.checkpoint"), "action": "filter:checkpoint"},
            {"label": tr("projects.filter.dataset"), "action": "filter:dataset"},
            {"label": tr("projects.filter.gallery"), "action": "filter:gallery"},
            {"label": tr("projects.gallery.action.grid"), "action": "gallery"},
            {"label": tr("projects.gallery.action.list"), "action": "list"},
            *self._sort_menu_items(),
            {"label": f"{tr('projects.toolbar.thumbnail_size')} 112", "action": "thumbnail:112", "separator_before": True},
            {"label": f"{tr('projects.toolbar.thumbnail_size')} 208", "action": "thumbnail:208"},
            {"label": f"{tr('projects.toolbar.thumbnail_size')} 320", "action": "thumbnail:320"},
            {"label": tr("projects.action.check_gallery"), "action": "check_gallery", "separator_before": True},
            {"label": tr("projects.action.rescan_folders"), "action": "rescan_folders"},
        ]
        def choose(action: str) -> None:
            if action.startswith("filter:"):
                self._set_filter(action.partition(":")[2])
            elif action in ("gallery", "list"):
                self.set_view_mode(None, None, [action])
            elif action.startswith(("sort:", "order:")):
                self._choose_sort(action)
            elif action.startswith("thumbnail:"):
                self.set_thumbnail_size(action.partition(":")[2])
            elif action == "check_gallery":
                self._gallery_command("refresh")
            elif action == "rescan_folders":
                self.refresh_catalog(scan_folders=True)

        self._show_shared_context_menu(items, choose)

    def _add_folder_from_path(self, directory: str, *, recursive: bool = True) -> Optional[str]:
        if not self._asset_index or not directory.strip():
            return None
        folder = self._library_command("add_folder", directory.strip())
        if folder is None:
            return None
        self._selected_folder_id = folder.id
        self._selected_asset_ids.clear()
        self._selection_cursor_id = None
        self._selection_anchor_id = None
        self._update_selection_type()
        self.refresh_catalog(scan_folders=False)
        folder_path = str(getattr(folder, "path", "") or directory).strip()
        if recursive:
            self._scan_asset_folders(folder_id=folder.id, directory=folder_path)
        else:
            self._scan_asset_folders(folder_id=folder.id, directory=folder_path, recursive=False)
        return folder.id

    def add_asset_folder(self, _handle=None, _ev=None, _args=None):
        self.on_add_folder(None, None, None)

    def on_add_folder(self, _handle=None, _ev=None, _args=None):
        start = str(resolve_default_asset_directory())
        directory = lf.ui.open_folder_dialog(
            tr("projects.dialog.select_folder"), start
        )
        if directory:
            folder_only = tr("projects.action.folder_only")
            include_subfolders = tr("projects.action.include_subfolders")

            def choose_scan(button: str) -> None:
                if button == folder_only:
                    self._add_folder_from_path(str(directory), recursive=False)
                elif button == include_subfolders:
                    self._add_folder_from_path(str(directory), recursive=True)

            lf.ui.confirm_dialog(
                tr("projects.dialog.scan_depth"),
                tr("projects.dialog.scan_depth_message"),
                [folder_only, include_subfolders, tr("common.cancel")],
                choose_scan,
            )

    def on_import_project(self, _handle=None, _ev=None, _args=None):
        if not self._asset_index:
            return
        try:
            path = lf.ui.open_project_file_dialog(
                "", tr("projects.dialog.choose_existing")
            )
        except TypeError:
            path = lf.ui.open_project_file_dialog("")
        if not path:
            return
        if not is_supported_asset_path(path):
            self._log_warn("Asset Manager only supports .licht projects: %s", path)
            self._set_catalog_notice(tr("projects.status.import_failed"))
            return
        try:
            project, _created = self._library_command(
                "register_licht_asset",
                path,
            )
            if project is not None:
                self._selected_asset_ids = {project.id}
                self._selection_cursor_id = project.id
                self._update_selection_type()
                self.refresh_catalog(scan_folders=False)
            else:
                self._set_catalog_notice(tr("projects.status.import_failed"))
        except Exception:
            self._set_catalog_notice(tr("projects.status.import_failed"))

    def _select_folder_id(self, folder_id: str) -> bool:
        if folder_id == SCOPE_TRANSFERS:
            self.on_open_gallery()
            return True
        if folder_id not in {*self._asset_index_folders(), SCOPE_ALL, SCOPE_RECENT, *GALLERY_SCOPES}:
            return False
        if folder_id in self._asset_index_folders():
            self._gallery_last_folder = folder_id
        entering_gallery = folder_id in GALLERY_SCOPES and self._selected_folder_id != folder_id
        if self._inspection_pipeline is not None:
            self._inspection_pipeline.cancel()
        self._selected_folder_id = folder_id
        if entering_gallery:
            self._controller().refresh()
        self._selected_asset_ids.clear()
        self._selection_cursor_id = None
        self._selection_anchor_id = None
        self._update_selection_type()
        self._reset_scroll()
        self._refresh_records(assets=True, folders=True)
        self._dirty_fields(
            "selected_folder_id",
            "all_assets_selected",
            "show_selection_none",
            "show_selection_asset",
            "show_selection_folder",
            "show_selection_multiple",
            "selected_folder_name",
            "selected_folder_path",
            "selected_folder_asset_count",
        )
        self._start_inspection_refresh()
        return True

    def select_folder(self, _handle, _ev, args):
        self._select_folder_id(self._resolve_event_value(args, _ev, "data-folder-id"))

    def _select_asset_id(
        self,
        asset_id: str,
        *,
        multi_select: bool = False,
        range_select: bool = False,
        row_element=None,
        container=None,
    ) -> bool:
        if asset_id not in self._all_display_assets():
            return False
        visible_ids = [
            str(asset.get("id") or asset.get("project_uuid") or "")
            for asset in self._filtered_assets()
        ]
        if range_select and self._selection_anchor_id in visible_ids:
            start = visible_ids.index(self._selection_anchor_id)
            end = visible_ids.index(asset_id)
            lo, hi = sorted((start, end))
            self._selected_asset_ids = set(visible_ids[lo : hi + 1])
        elif multi_select:
            if asset_id in self._selected_asset_ids:
                self._selected_asset_ids.remove(asset_id)
            else:
                self._selected_asset_ids.add(asset_id)
        else:
            self._selected_asset_ids = {asset_id}
            self._selection_anchor_id = asset_id
        self._selection_cursor_id = (
            asset_id if asset_id in self._selected_asset_ids else next(iter(self._selected_asset_ids), None)
        )
        self._update_selection_type()
        self._sync_asset_selection_dom(container, row_element)
        self._dirty_selection()
        self._start_inspection_refresh()
        return True

    def toggle_asset_selection(self, _handle, _ev, args):
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        self._select_asset_id(
            asset_id,
            multi_select=self._event_multi_select(_ev),
            range_select=self._event_range_select(_ev),
        )

    def _dirty_selection(self) -> None:
        if self._handle:
            self._handle.dirty_all()
        self._dirty_fields(
            "selected_asset_id",
            "selected_count",
            "selected_count_text",
            "show_selection_none",
            "show_selection_asset",
            "show_selection_folder",
            "show_selection_multiple",
            "selected_asset_name",
            "selected_asset_folder_name",
            "selected_asset_has_folder",
            "selected_asset_path",
            "selected_asset_size",
            "selected_asset_created",
            "selected_asset_modified",
            "selected_health_state",
            "selected_health_label",
            "selected_has_problem",
            "selected_fix_label",
            "selected_fix_action",
            "selected_asset_file_missing",
            "selected_asset_can_locate",
            "selected_fix_requires_action",
            "locate_section_title",
            "selected_asset_relocation_candidate",
            "selected_asset_has_relocation_candidate",
            "selected_asset_expected_path",
            "quick_look_visible",
            "quick_look_thumbnail",
            "quick_look_placeholder",
            "quick_look_has_thumbnail",
            "inspector_saved", "inspector_saved_at", "inspector_opened",
            "inspector_iteration", "inspector_strategy", "inspector_resumable",
            "inspector_gaussians", "inspector_sh_degree", "inspector_dataset",
            "inspector_dataset_path", "inspector_dataset_reachable",
            "inspector_embedded", "inspector_has_metrics", "inspector_metrics",
            "inspector_license", "inspector_license_notice", "selected_project_title",
            "inspector_physical_size", "inspector_dead_bytes", "inspector_reclaimable",
            "inspector_saves", "inspector_autosave_newer", "inspector_has_details",
            "inspector_card_diagnostic", "inspector_operation_actions", "inspector_can_resume",
            "inspector_operations_expanded",
            "inspector_gallery_action_label", "inspector_has_gallery_action",
            "gallery_account_reason", "gallery_has_account_reason", "inspector_gallery_action_tooltip",
            "inspector_training_tooltip", "inspector_model_tooltip", "inspector_reclaimable_tooltip",
            "inspector_verify_result",
            "catalog_notice",
            "has_catalog_notice",
        )
        if self._handle:
            self._handle.update_record_list("inspector_operation_rows", self.get_selected_operation_actions())

    def on_locate_file(self, _handle=None, _ev=None, args=None):
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id") or self.get_selected_asset_id()
        if not asset_id or not self._asset_index:
            return
        path = lf.ui.open_project_file_dialog("")
        if not path:
            return
        try:
            if self._library_command("relink_asset", asset_id, path):
                self.refresh_catalog(scan_folders=False)
            else:
                self._set_catalog_notice(tr("projects.status.locate_id_mismatch"))
        except Exception:
            self._set_catalog_notice(tr("projects.status.locate_id_mismatch"))

    def on_selected_fix(self, _handle=None, _ev=None, _args=None):
        asset = self._get_selected_asset()
        if not asset:
            return
        action = self.get_selected_fix_action()
        if action == "locate":
            self.on_locate_file()
        elif action == "verify":
            self._start_project_operation(
                asset["id"], "Verify project",
                lambda progress, cancel: self._verify_project(asset["id"], asset["path"], progress, cancel),
            )
        elif action == "repair":
            self.open_project_operation(None, None, ["repair"])

    def _verify_project(self, asset_id: str, path: str, progress: Callable[..., None], cancel: Callable[[], bool]) -> Any:
        result = self._native_io_call("verify_project_file", path, progress, cancel)
        status = str(getattr(getattr(result, "status", None), "name", getattr(result, "status", "")) or "").lower()
        self._verify_results[asset_id] = status or tr("projects.status.verified")
        return result

    def on_use_found_location(self, _handle=None, _ev=None, args=None):
        asset_id = self._resolve_event_value((), _ev, "data-asset-id") or self.get_selected_asset_id()
        if not asset_id or not self._asset_index:
            return
        asset = self._asset_dict(asset_id)
        candidate = str((asset or {}).get("relocation_candidate") or "")
        if not candidate:
            return
        try:
            if self._library_command("relink_asset", asset_id, candidate):
                self.refresh_catalog(scan_folders=False)
            else:
                self._log_warn("Could not use the found location for this project")
        except Exception as exc:
            self._log_error("Failed to relink .licht project: %s", exc)

    def on_load_asset(self, _handle, _ev, args):
        self._load_asset(self._resolve_event_value(args, _ev, "data-asset-id"))

    def _dialog_entry(self) -> Optional[Dict[str, Any]]:
        return self._asset_dict(self._dialog_asset_id or self.get_selected_asset_id())

    def get_dialog_title(self) -> str:
        return tr({
            "save_history": "projects.dialog.save_history",
            "reduce_size": "projects.dialog.reduce_size",
            "export_as": "projects.dialog.export_as",
            "update_thumbnail": "projects.dialog.update_thumbnail",
            "set_license": "projects.dialog.set_license",
            "rename": "projects.dialog.rename_project",
            "repair": "projects.dialog.repair",
            "locate_dataset": "projects.dialog.locate_dataset",
        }.get(self._dialog_kind, "projects.inspector.operations"))

    def _set_dialog_value(self, field: str, item: Any) -> None:
        self._dialog_data[str(field)] = str(item or "")
        self._dirty_fields("dialog_" + str(field))

    def get_dialog_confirm_label(self) -> str:
        return tr({
            "save_history": "projects.action.resume_from_here",
            "reduce_size": "projects.action.reduce_size",
            "export_as": "projects.action.export",
            "update_thumbnail": "projects.action.update_thumbnail",
            "set_license": "common.save",
            "rename": "common.save",
            "repair": "projects.action.repair",
            "locate_dataset": "projects.action.locate_dataset",
        }.get(self._dialog_kind, "common.ok"))

    def _set_dialog(self, kind: str, data: Optional[Dict[str, Any]] = None) -> None:
        self._dialog_kind = str(kind or "")
        self._dialog_data = dict(data or {})
        self._dialog_busy = False
        if self._handle:
            self._handle.update_record_list("dialog_rows", self._dialog_data.get("rows", []))
            self._handle.dirty_all()
        self._request_model_update()

    def close_project_dialog(self, _handle=None, _ev=None, _args=None) -> None:
        self._dialog_kind = ""
        self._dialog_data = {}
        self._dialog_plan = None
        self._dialog_busy = False
        if self._handle:
            self._handle.update_record_list("dialog_rows", [])
            self._handle.dirty_all()
        self._request_model_update()

    def open_project_operation(self, _handle=None, _ev=None, args=None) -> None:
        action = self._resolve_event_value(args, _ev, "data-project-operation")
        if not action and args:
            action = str(args[0])
        if not action:
            return
        asset_id = self._resolve_event_value((), _ev, "data-asset-id") or self.get_selected_asset_id()
        asset = self._asset_dict(asset_id)
        if not asset or asset.get("remote_only"):
            return
        self._dialog_asset_id = asset_id
        details = self._inspection_by_asset.get(asset_id, {}).get("details")
        if action == "reduce_size":
            self._set_dialog("reduce_size", {"name": self._get_asset_display_name(asset), "path": asset.get("path", ""), "busy": True})
            self._dialog_busy = True
            self._dialog_plan = None
            self._run_dialog_worker(lambda: self._native_io_call("plan_reduce_size", asset["path"]), self._on_plan_ready)
            return
        if action == "save_history" and details is None:
            self._set_dialog("save_history", {"name": self._get_asset_display_name(asset), "path": asset.get("path", ""), "message": tr("projects.status.reading")})
            self._dialog_busy = True
            self._start_inspection_refresh()
            return
        data = dialog_model(
            action,
            entry=asset,
            details=details,
            plan=self._dialog_plan,
            format_size=self._format_size,
            format_time=self._format_unix_ns,
        )
        self._set_dialog(action, data)

    def _run_dialog_worker(self, function: Callable[[], Any], complete: Callable[[Any, Optional[Exception]], None]) -> None:
        generation = self._mount_generation

        def worker() -> None:
            try:
                result, error = function(), None
            except Exception as exc:
                result, error = None, exc
            self._schedule_ui(lambda: complete(result, error) if generation == self._mount_generation else None)

        threading.Thread(target=worker, daemon=True, name="ProjectsDialogWorker").start()

    def _on_plan_ready(self, plan: Any, error: Optional[Exception]) -> None:
        self._dialog_busy = False
        if error is not None:
            self._dialog_data["message"] = str(error)
        else:
            self._dialog_plan = plan
            self._dialog_data.update(dialog_model(
                "reduce_size",
                entry=self._dialog_entry(),
                plan=plan,
                format_size=self._format_size,
                format_time=self._format_unix_ns,
            ))
        if self._handle:
            self._handle.update_record_list("dialog_rows", self._dialog_data.get("rows", []))
            self._handle.dirty_all()
        self._request_model_update()

    def dialog_select_generation(self, _handle=None, _ev=None, args=None) -> None:
        value = self._resolve_event_value(args, _ev, "data-generation") or (str(args[0]) if args else "")
        try:
            self._dialog_data["generation"] = int(value)
        except (TypeError, ValueError):
            self._dialog_data["generation"] = 0
        self._dirty_fields("dialog_rows")

    def dialog_choose_destination(self, _handle=None, _ev=None, _args=None) -> None:
        path = lf.ui.open_project_file_dialog("")
        if path:
            self._dialog_data["destination"] = str(path)
            self._dirty_fields("dialog_destination")

    def dialog_set_format(self, _handle=None, _ev=None, args=None) -> None:
        if args:
            self._dialog_data["format"] = str(args[0]).lower()
            self._dirty_fields("dialog_format")

    def dialog_set_source(self, _handle=None, _ev=None, args=None) -> None:
        if args:
            self._dialog_data["source"] = str(args[0])
            self._dirty_fields("dialog_source")

    def dialog_set_license(self, _handle=None, _ev=None, args=None) -> None:
        if args:
            self._dialog_data["identifier"] = str(args[0])
            self._dirty_fields("dialog_identifier")

    def dialog_set_name(self, _handle=None, _ev=None, args=None) -> None:
        if args:
            self._dialog_data["name"] = str(args[0])
            self._dirty_fields("dialog_name")

    def dialog_toggle_checkpoints(self, _handle=None, _ev=None, _args=None) -> None:
        self._dialog_drop_checkpoints = not self._dialog_drop_checkpoints
        self._dirty_fields("dialog_drop_checkpoints")

    def dialog_toggle_dataset(self, _handle=None, _ev=None, _args=None) -> None:
        self._dialog_drop_dataset = not self._dialog_drop_dataset
        self._dirty_fields("dialog_drop_dataset")

    def _selected_save_generation(self) -> int:
        try:
            return int(self._dialog_data.get("generation") or 0)
        except (TypeError, ValueError):
            return 0

    def _selected_checkpoint_uuid(self) -> str:
        details = self._inspection_by_asset.get(self._dialog_asset_id, {}).get("details")
        checkpoints = list(getattr(details, "retained_checkpoints", []) or []) if details is not None else []
        generation = self._selected_save_generation()
        matching = [item for item in checkpoints if int(getattr(item, "source_generation", 0) or 0) == generation]
        item = max(matching or checkpoints, key=lambda value: int(getattr(value, "iteration", 0) or 0), default=None)
        return str(getattr(item, "instance_uuid", "") or "") if item is not None else ""

    def dialog_restore_new(self, _handle=None, _ev=None, _args=None) -> None:
        asset = self._dialog_entry()
        if not asset:
            return
        destination = str(self._dialog_data.get("destination") or "")
        if not destination:
            self.dialog_choose_destination()
            destination = str(self._dialog_data.get("destination") or "")
        if destination:
            self._start_project_operation(
                asset["id"], "Open save as new project",
                lambda _progress, _cancel: self._native_io_call("restore_save", asset["path"], self._selected_save_generation(), destination),
            )
            self.close_project_dialog()

    def dialog_resume_here(self, _handle=None, _ev=None, _args=None) -> None:
        asset = self._dialog_entry()
        checkpoint_uuid = self._selected_checkpoint_uuid()
        if not asset or not checkpoint_uuid:
            return
        self._start_project_operation(
            asset["id"], "Resume from here",
            lambda _progress, _cancel: self._native_io_call("rebind_checkpoint", asset["path"], checkpoint_uuid),
        )
        self.close_project_dialog()

    def confirm_project_dialog(self, _handle=None, _ev=None, _args=None) -> None:
        action = self._dialog_kind
        asset = self._dialog_entry()
        if not action or not asset or not asset.get("path"):
            return
        path = str(asset["path"])
        data = self._dialog_data
        if action == "save_history":
            generation = int(data.get("generation") or 0)
            destination = str(data.get("destination") or "")
            if not destination:
                self.dialog_choose_destination()
                destination = str(data.get("destination") or "")
            if not destination:
                return
            self._start_project_operation(asset["id"], "Restore save", lambda progress, cancel: self._native_io_call("restore_save", path, generation, destination))
        elif action == "reduce_size":
            options = {"drop_unbound_checkpoints": self._dialog_drop_checkpoints, "drop_embedded_dataset": self._dialog_drop_dataset}
            self._start_project_operation(asset["id"], "Reduce project size", lambda progress, cancel: self._native_io_call("reduce_size", path, options, progress, cancel))
        elif action == "export_as":
            destination = str(data.get("destination") or "")
            if not destination:
                destination = self._choose_export_destination(str(data.get("format") or "sog"))
                data["destination"] = destination
            if not destination:
                return
            self._start_project_operation(asset["id"], "Export project", lambda progress, cancel: self._native_io_call("export_project_as", path, data.get("format", "sog"), destination, progress, cancel))
        elif action == "update_thumbnail":
            self._start_thumbnail_operation(asset)
        elif action == "set_license":
            identifier = str(data.get("identifier") or "").strip()
            notice = str(data.get("notice") or "")
            self._start_project_operation(asset["id"], "Set project license", lambda _progress, _cancel: self._native_io_call("clear_project_license", path) if not identifier else self._native_io_call("set_project_license", path, identifier, notice))
        elif action == "rename":
            name = str(data.get("name") or "").strip()
            if name:
                self._start_project_operation(asset["id"], "Rename project", lambda _progress, _cancel: self._native_io_call("set_project_title", path, name), after=lambda: self._rename_catalog_entry(asset["id"], name))
        elif action == "repair":
            destination = str(data.get("destination") or "")
            if not destination:
                self.dialog_choose_destination()
                destination = str(data.get("destination") or "")
            if destination:
                self._start_project_operation(asset["id"], "Repair project", lambda _progress, _cancel: self._native_io_call("repair_project", path, destination))
        elif action == "locate_dataset":
            directory = lf.ui.open_folder_dialog(tr("projects.dialog.select_dataset"), str(Path(path).parent))
            if directory:
                self._start_project_operation(asset["id"], "Locate dataset", lambda _progress, _cancel: self._native_io_call("set_dataset_reference", path, directory))
        if action not in {"save_history", "reduce_size", "export_as", "update_thumbnail", "set_license", "rename", "repair", "locate_dataset"}:
            return
        self.close_project_dialog()

    def _choose_export_destination(self, format_name: str) -> str:
        chooser = getattr(lf.ui, "save_" + format_name + "_file_dialog", None)
        if callable(chooser):
            return str(chooser("export"))
        return str(getattr(lf.ui, "open_project_file_dialog", lambda *_args: "")(""))

    def _rename_catalog_entry(self, asset_id: str, name: str) -> None:
        try:
            self._library_command("update_asset", asset_id, name=name)
        except Exception:
            pass

    def _start_thumbnail_operation(self, asset: Dict[str, Any]) -> None:
        source = str(self._dialog_data.get("source") or "first_dataset")
        path = str(asset["path"])
        if source == "image_file":
            image_path = str(getattr(lf.ui, "open_image_dialog", lambda *_args: "")(""))
            if not image_path:
                return
            self._start_project_operation(asset["id"], "Update thumbnail", lambda _progress, _cancel: self._native_io_call("set_project_preview", path, Path(image_path).read_bytes()))
        elif source == "viewport":
            self._start_project_operation(asset["id"], "Update thumbnail", lambda _progress, _cancel: self._capture_viewport_preview(path))
        else:
            native_name = "preview_from_first_embedded_image" if source == "first_embedded" else "preview_from_first_dataset_image"
            self._start_project_operation(asset["id"], "Update thumbnail", lambda _progress, _cancel: self._native_io_call(native_name, path))

    @staticmethod
    def _capture_viewport_preview(path: str) -> Any:
        import os
        import tempfile
        fd, target_name = tempfile.mkstemp(prefix="lfs-project-preview-", suffix=".png")
        os.close(fd)
        target = Path(target_name)
        try:
            exporter = getattr(lf, "export_viewport_image", None)
            if not callable(exporter):
                raise RuntimeError("The current viewport has no captured image")
            exporter(str(target), "png")
            return AssetManagerPanel._native_io_call("set_project_preview", path, target.read_bytes())
        finally:
            target.unlink(missing_ok=True)

    def _start_project_operation(
        self,
        asset_id: str,
        title: str,
        operation: Callable[[Callable[..., None], Callable[[], bool]], Any],
        *,
        after: Optional[Callable[[], None]] = None,
    ) -> None:
        self._operation_counter += 1
        operation_id = f"project-{self._operation_counter}"
        cancel = threading.Event()
        self._project_operations[operation_id] = {
            "id": operation_id,
            "asset_id": asset_id,
            "title": title,
            "status": "running",
            "phase": tr("projects.transfer.preparing"),
            "progress": 0.0,
            "cancel": cancel,
        }
        self._refresh_transfer_rows()

        def progress(value: Any = 0.0, stage: str = "") -> None:
            try:
                percent = max(0.0, min(100.0, float(value) * 100.0 if float(value) <= 1.0 else float(value)))
            except (TypeError, ValueError):
                percent = 0.0
            self._schedule_ui(lambda: self._update_project_operation(operation_id, percent, stage))

        def worker() -> None:
            try:
                result = operation(progress, cancel.is_set)
                error = None
            except Exception as exc:
                result, error = None, exc

            def complete() -> None:
                row = self._project_operations.get(operation_id)
                if row is None:
                    return
                if error is None:
                    row.update(status="completed", progress=100.0, phase=tr("projects.transfer.done"), result=result)
                    self._inspection_by_asset.pop(asset_id, None)
                    self._inspection_errors.pop(asset_id, None)
                    if after is not None:
                        after()
                    self._start_inspection_refresh()
                    self.refresh_catalog(scan_folders=False)
                else:
                    row.update(status="failed", phase=tr("projects.transfer.failed"), reason=str(error))
                    self._set_catalog_notice(str(error))
                self._refresh_transfer_rows()
                self._dirty_selection()

            self._schedule_ui(complete)

        threading.Thread(target=worker, daemon=True, name="ProjectsOperation").start()

    def _update_project_operation(self, operation_id: str, progress: float, stage: str) -> None:
        row = self._project_operations.get(operation_id)
        if row is None:
            return
        row["progress"] = progress
        row["phase"] = str(stage or tr("projects.transfer.preparing"))
        self._refresh_transfer_rows()

    def native_file_drop(self, path: str) -> bool:
        """Register a native .licht drop when Projects owns the drop target."""
        if not self._asset_index or not is_supported_asset_path(path):
            return False
        try:
            project, _created = self._library_command("register_licht_asset", path)
        except Exception as exc:
            self._log_error("Failed to add dropped .licht project %s: %s", path, exc)
            self._set_catalog_notice(tr("projects.status.import_failed"))
            return True
        if project is None:
            self._set_catalog_notice(tr("projects.status.import_failed"))
            return True
        self._selected_asset_ids = {project.id}
        self._selection_cursor_id = project.id
        self._update_selection_type()
        self.refresh_catalog(scan_folders=False)
        return True

    def on_open_gallery(self, _handle=None, _event=None, _args=None):
        # Transfers live in the footer tray; keep this legacy callback as a
        # harmless focus hook for saved layouts and older menu commands.
        self._request_model_update()

    def _load_asset(self, asset_id: str) -> None:
        if not asset_id or not self._asset_index:
            return
        if asset_id.startswith("remote:"):
            self._select_asset_id(asset_id)
            self._gallery_command("pull_open")
            return
        project = self._library_command("verify_asset", asset_id)
        if project is None:
            return
        asset = project.to_dict() if hasattr(project, "to_dict") else (self._asset_dict(asset_id) or {})
        if not self._project_available(asset):
            self.refresh_catalog(scan_folders=False)
            return
        self._selected_asset_ids = {asset_id}
        self._selection_cursor_id = asset_id
        self._update_selection_type()
        self._dirty_selection()
        from .file_menu import open_project_with_confirmation

        open_project_with_confirmation(
            str(asset.get("path") or ""),
            keep_asset_manager_open=True,
        )

    def on_remove_asset(self, _handle, _ev, args):
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        if asset_id and self._asset_index and self._library_command("delete_asset", asset_id):
            self._selected_asset_ids.discard(asset_id)
            if self._selection_cursor_id == asset_id:
                self._selection_cursor_id = None
            self.refresh_catalog(scan_folders=False)

    def _show_shared_context_menu(
        self,
        items: List[Dict[str, Any]],
        on_action: Callable[[str], None],
    ) -> bool:
        show = getattr(lf.ui, "show_context_menu", None)
        mouse_position = getattr(lf.ui, "get_mouse_screen_pos", None)
        if not callable(show) or not callable(mouse_position):
            return False
        try:
            x, y = mouse_position()
            show(items, float(x), float(y), on_action)
            return True
        except Exception as exc:
            self._log_error("Failed to show context menu: %s", exc)
            return False

    def _asset_context_menu_items(self, asset: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not asset.get("remote_only"):
            items.append({"label": tr("projects.action.open"), "action": "load"})
        items.extend(self._gallery_context_items(asset))
        if asset.get("remote_only"):
            return items
        if str(asset.get("relocation_candidate") or ""):
            items.append(
                {
                    "label": tr("projects.action.use_found_location"),
                    "action": "use_found_location",
                }
            )
        items.extend(
            [
                {"label": tr("projects.action.rename"), "action": "rename"},
                {
                    "label": tr("projects.action.show_in_folder"),
                    "action": "show_in_folder",
                    "separator_before": True,
                },
                {"label": tr("projects.action.remove_from_library"), "action": "remove"},
                {
                    "label": tr("projects.action.move_to_trash"),
                    "action": "trash",
                    "separator_before": True,
                },
            ]
        )
        details = self._inspection_by_asset.get(str(asset.get("id") or asset.get("project_uuid") or ""), {}).get("details")
        if details is not None:
            labels = {
                "save_history": "projects.action.save_history",
                "reduce_size": "projects.action.reduce_size",
                "embed_dataset": "projects.action.embed_dataset",
                "locate_dataset": "projects.action.locate_dataset",
                "export_as": "projects.action.export_as",
                "update_thumbnail": "projects.action.update_thumbnail",
                "set_license": "projects.action.set_license",
            }
            for operation in operation_actions(asset, details):
                action = str(operation.get("action") or "")
                if action == "rename" or action not in labels:
                    continue
                items.append({
                    "label": tr(labels[action]),
                    "action": "project:" + action,
                    "separator_before": action == "save_history",
                })
        return items

    def _handle_asset_context_action(self, action: str, asset_id: str) -> None:
        if action.startswith("gallery:"):
            self._select_asset_id(asset_id)
            self._gallery_command(action.split(":", 1)[1])
        elif action == "load":
            self._load_asset(asset_id)
        elif action == "use_found_location":
            self.on_use_found_location(None, None, [asset_id])
        elif action == "rename":
            self.on_rename_asset(None, None, [asset_id])
        elif action == "show_in_folder":
            self.on_show_in_folder(None, None, [asset_id])
        elif action == "remove":
            self.on_remove_asset(None, None, [asset_id])
        elif action == "trash":
            self.on_move_asset_to_trash(None, None, [asset_id])
        elif action.startswith("project:"):
            self._select_asset_id(asset_id)
            self.open_project_operation(None, None, [action.partition(":")[2]])

    def _show_asset_context_menu(self, asset_id: str) -> bool:
        asset = self._asset_dict(asset_id)
        return bool(asset) and self._show_shared_context_menu(
            self._asset_context_menu_items(asset),
            lambda action: self._handle_asset_context_action(action, asset_id),
        )

    def on_rename_asset(self, _handle, _ev, args):
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        asset = self._asset_dict(asset_id)
        if not asset or not self._asset_index:
            return
        current_name = str(asset.get("name") or Path(str(asset.get("path") or "")).stem)

        def rename(name: Any) -> None:
            value = str(name or "").strip()
            if value and value != current_name:
                self._library_command("update_asset", asset_id, name=value)
                self.refresh_catalog(scan_folders=False)

        lf.ui.input_dialog(
            tr("projects.dialog.rename_asset"),
            tr("projects.dialog.enter_new_name", name=current_name),
            current_name,
            rename,
        )

    def on_show_in_folder(self, _handle, _ev, args):
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        asset = self._asset_dict(asset_id)
        if asset:
            reveal = getattr(lf.ui, "reveal_in_file_manager", None)
            if callable(reveal):
                reveal(str(asset.get("path") or ""))

    def on_move_asset_to_trash(self, _handle, _ev, args):
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        asset = self._asset_dict(asset_id)
        path = str(asset.get("path") or "") if asset else ""
        if not asset_id or not path or not self._asset_index:
            return
        label = tr("projects.action.move_to_trash")

        def confirmed(button: str) -> None:
            if button != label:
                return
            try:
                subprocess.run(["gio", "trash", path], check=True, capture_output=True)
                self._library_command("delete_asset", asset_id)
                self._selected_asset_ids.discard(asset_id)
                if self._selection_cursor_id == asset_id:
                    self._selection_cursor_id = None
                self.refresh_catalog(scan_folders=False)
            except (OSError, subprocess.CalledProcessError) as exc:
                self._set_catalog_notice(tr("projects.status.trash_failed"))
                self._log_error("Failed to move project to trash %s: %s", path, exc)
                self._request_model_update()

        lf.ui.confirm_dialog(
            label,
            f'{label}\n\n{path}',
            [tr("common.cancel"), label],
            confirmed,
            "error",
        )

    def _folder_context_menu_items(self, folder_id: str) -> List[Dict[str, Any]]:
        items = [
            {"label": tr("projects.action.show_in_folder"), "action": "show"},
            {"label": tr("projects.action.rescan_folders"), "action": "rescan"},
        ]
        if folder_id == "default":
            items.append(
                {
                    "label": tr("projects.action.settings"),
                    "action": "settings",
                    "separator_before": True,
                }
            )
        else:
            items.append(
                {
                    "label": tr("projects.action.remove_folder"),
                    "action": "remove",
                    "separator_before": True,
                }
            )
        if any(
            asset.get("folder_id") == folder_id
            and (not asset.get("exists", True) or asset.get("status") == "MISSING")
            for asset in self._asset_index_assets().values()
        ):
            items.append(
                {
                    "label": tr("projects.action.clean_missing"),
                    "action": "clean_missing",
                }
            )
        return items

    def _show_folder_context_menu(self, folder_id: str) -> bool:
        if folder_id not in self._asset_index_folders():
            return False
        return self._show_shared_context_menu(
            self._folder_context_menu_items(folder_id),
            lambda action: self._handle_folder_context_action(action, folder_id),
        )

    def _handle_folder_context_action(self, action: str, folder_id: str) -> None:
        if action == "show":
            folder = self._asset_index_folders().get(folder_id, {})
            reveal = getattr(lf.ui, "reveal_in_file_manager", None)
            if callable(reveal) and folder.get("path"):
                reveal(str(folder["path"]))
        elif action == "rescan":
            folder = self._asset_index_folders().get(folder_id, {})
            self.refresh_catalog(scan_folders=False)
            self._scan_asset_folders(folder_id=folder_id, directory=str(folder.get("path") or ""))
        elif action == "settings":
            lf.ui.set_panel_enabled("lfs.preferences", True)
        elif action == "remove":
            self.on_delete_folder(None, None, [folder_id])
        elif action == "clean_missing":
            removed = self._library_command("clean_missing_entries", folder_id)
            self._catalog_notice = tr(
                "projects.status.cleaned_missing", count=int(removed or 0)
            )
            self.refresh_catalog(scan_folders=False)

    def on_delete_folder(self, _handle, _ev, args):
        folder_id = self._resolve_event_value(args, _ev, "data-folder-id")
        folder = self._asset_index_folders().get(folder_id)
        if not folder_id or folder_id == "default" or not self._asset_index or not folder:
            return
        project_count = sum(
            asset.get("folder_id") == folder_id
            for asset in self._asset_index_assets().values()
        )
        delete_label = tr("projects.action.remove_folder")

        def delete_confirmed(button: str) -> None:
            if button != delete_label:
                return
            if self._asset_index and self._library_command("delete_folder", folder_id):
                self._selected_folder_id = SCOPE_ALL
                self._selected_asset_ids.clear()
                self._selection_cursor_id = None
                self.refresh_catalog(scan_folders=False)

        lf.ui.confirm_dialog(
            tr("projects.dialog.remove_folder"),
            tr(
                "projects.dialog.remove_folder_message",
                name=str(folder.get("name") or ""),
                count=project_count,
            ),
            [tr("common.cancel"), delete_label],
            delete_confirmed,
        )

    def refresh_catalog(
        self,
        _handle=None,
        _ev=None,
        _args=None,
        *,
        request_update: bool = True,
        scan_folders: bool = True,
    ):
        if scan_folders:
            cancel = None
            verify_cancel = None
            with self._folder_scan_lock:
                if self._folder_scan_active or self._catalog_verify_active:
                    self._folder_scan_rerun_pending = False
                    self._folder_scan_rerun_target = None
                    self._scan_stop_requested = True
                    cancel = self._folder_scan_cancel
                    verify_cancel = self._catalog_verify_cancel
            if cancel is not None:
                cancel.set()
            if verify_cancel is not None:
                verify_cancel.set()
                return
            if cancel is not None:
                return
        self._sync_default_folder_path()
        if self._catalog_notice:
            self._set_catalog_notice("")
        self._repair_selection()
        self._refresh_records(assets=True, folders=True)
        if self._handle:
            self._handle.dirty_all()
        if request_update:
            self._request_model_update()
        if scan_folders:
            with self._folder_scan_lock:
                self._scan_stopped_visible = False
            self._start_catalog_verify()
            self._scan_asset_folders()

    def _scan_asset_folders(
        self,
        folder_id: Optional[str] = None,
        directory: Optional[str] = None,
        *,
        recursive: bool = True,
    ) -> None:
        if not self._asset_index:
            return
        target: Optional[tuple[str, str, bool]]
        if folder_id and directory:
            target = (str(folder_id), str(directory), bool(recursive))
        else:
            target = None
        with self._folder_scan_lock:
            if self._folder_scan_active:
                if self._folder_scan_rerun_pending:
                    if self._folder_scan_rerun_target != target:
                        self._folder_scan_rerun_target = None
                else:
                    self._folder_scan_rerun_pending = True
                    self._folder_scan_rerun_target = target
                return
            if not self._panel_mounted:
                return
            if target is None and not any(
                folder.get("path")
                for folder in self._asset_index_folders().values()
            ):
                return
            self._folder_scan_active = True
            self._folder_scan_rerun_pending = False
            self._folder_scan_rerun_target = None
            self._scan_stop_requested = False
            self._scan_stopped_visible = False
            progress = AssetFolderScanProgress(self._queue_worker_update)
            if target is not None:
                progress.report(current_root=target[1])
            else:
                for folder in self._asset_index_folders().values():
                    path = str(folder.get("path") or "").strip()
                    if path:
                        progress.report(current_root=path)
                        break
            self._scan_progress = progress
            cancel_event = threading.Event()
            self._folder_scan_cancel = cancel_event
            scan_folder_id = target[0] if target else None
            scan_directory = target[1] if target else None
            scan_recursive = target[2] if target else True
            thread = threading.Thread(
                target=self._folder_scan_worker,
                args=(
                    self._asset_index,
                    cancel_event,
                    scan_folder_id,
                    scan_directory,
                    scan_recursive,
                    progress,
                    self._mount_generation,
                ),
                daemon=True,
                name="AssetManagerFolderScan",
            )
            self._folder_scan_thread = thread
        thread.start()
        self._publish_scan_progress()

    def _folder_scan_worker(
        self,
        index: Any,
        cancel_event: threading.Event,
        folder_id: Optional[str],
        directory: Optional[str],
        recursive: bool,
        progress: AssetFolderScanProgress,
        generation: int,
    ) -> None:
        global _folder_scan_completed_in_process
        try:
            if self._library_service is not None and not (folder_id and directory):
                result = self._library_service.scan(cancel_event, progress=progress)
            elif folder_id and directory:
                scan_args = (index, folder_id, directory, cancel_event)
                if recursive:
                    result = scan_asset_folder(*scan_args, progress=progress)
                else:
                    result = scan_asset_folder(*scan_args, progress=progress, recursive=False)
            else:
                result = scan_all_asset_folders(
                    index, cancel_event, progress=progress
                )
            with self._folder_scan_lock:
                self._folder_scan_error = bool(result.failed)
                self._folder_scan_unavailable = bool(getattr(result, "unavailable", False))
            _log.info(
                "Asset folder scan: discovered=%d added=%d existing=%d failed=%d cancelled=%s",
                result.discovered,
                result.added,
                result.already_cataloged,
                result.failed,
                result.cancelled,
            )
        except Exception:
            with self._folder_scan_lock:
                self._folder_scan_error = True
            _log.exception("Asset Manager folder scan failed")
        finally:
            with self._folder_scan_lock:
                self._folder_scan_active = False
                self._folder_scan_refresh_pending = True
                if self._scan_stop_requested:
                    self._scan_stopped_visible = True
                self._scan_stop_requested = False
                if self._folder_scan_thread is threading.current_thread():
                    self._folder_scan_thread = None
                _folder_scan_completed_in_process = True
            scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
            if callable(scheduler):
                scheduler(lambda: self._complete_folder_scan(generation))

    def _finish_folder_scan(self) -> None:
        with self._folder_scan_lock:
            if not self._folder_scan_refresh_pending:
                return
            self._folder_scan_refresh_pending = False
        if not self._panel_mounted:
            return
        self.refresh_catalog(scan_folders=False)

    def _complete_folder_scan(self, generation: Optional[int] = None) -> None:
        if generation is not None and generation != self._mount_generation:
            return
        self._finish_folder_scan()
        with self._folder_scan_lock:
            scan_error = self._folder_scan_error
            scan_unavailable = self._folder_scan_unavailable
            self._folder_scan_error = False
            self._folder_scan_unavailable = False
        if scan_unavailable:
            self._set_catalog_notice(tr("projects.status.folder_unavailable"))
        elif scan_error:
            self._set_catalog_notice(tr("projects.status.scan_errors"))
        with self._folder_scan_lock:
            rerun = self._folder_scan_rerun_pending
            target = self._folder_scan_rerun_target
            self._folder_scan_rerun_pending = False
            self._folder_scan_rerun_target = None
        self._publish_scan_progress()
        if rerun and self._panel_mounted:
            if target is not None:
                self._scan_asset_folders(folder_id=target[0], directory=target[1], recursive=target[2])
            else:
                self._scan_asset_folders()

    def _catalog_epoch(self) -> Optional[int]:
        if not self._asset_index:
            return None
        getter = getattr(self._asset_index, "catalog_epoch", None)
        if callable(getter):
            return int(getter())
        if isinstance(getter, int):
            return getter
        return None

    def _publish_catalog_if_changed(self) -> bool:
        epoch = self._catalog_epoch()
        if epoch is None or epoch == self._catalog_epoch_seen:
            return False
        self._catalog_epoch_seen = epoch
        self._refresh_records(assets=True, folders=True)
        self._dirty_selection()
        self._start_inspection_refresh()
        return True

    def _subscribe_catalog(self) -> None:
        subscribe = getattr(self._asset_index, "subscribe", None)
        if self._catalog_unsubscribe is None and callable(subscribe):
            self._catalog_unsubscribe = subscribe(self._queue_worker_update)

    def _queue_worker_update(self) -> None:
        scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
        if not callable(scheduler):
            return
        with self._worker_notification_lock:
            if self._worker_notification_pending or not self._panel_mounted:
                return
            self._worker_notification_pending = True
            generation = self._mount_generation

        def complete() -> None:
            with self._worker_notification_lock:
                self._worker_notification_pending = False
            if generation == self._mount_generation and self._panel_mounted:
                self._request_model_update()

        scheduler(complete)

    def _start_catalog_verify(self) -> None:
        if not self._asset_index or not self._panel_mounted:
            return
        if not callable(getattr(self._asset_index, "verify_asset", None)):
            return
        if not callable(getattr(self._asset_index, "list_projects", None)):
            return
        with self._folder_scan_lock:
            if self._catalog_verify_active:
                return
            if not self._panel_mounted:
                return
            self._catalog_verify_active = True
            self._catalog_verify_succeeded = False
            self._catalog_verify_refresh_pending = False
            cancel_event = threading.Event()
            self._catalog_verify_cancel = cancel_event
            visible_ids = [
                str(asset.get("id") or asset.get("project_uuid") or "")
                for asset in self._window_assets(self._filtered_assets())
            ]
            thread = threading.Thread(
                target=self._catalog_verify_worker,
                args=(self._asset_index, cancel_event, visible_ids, self._mount_generation),
                daemon=True,
                name="AssetManagerCatalogVerify",
            )
            self._catalog_verify_thread = thread
        thread.start()

    def _catalog_verify_worker(
        self, index: Any, cancel_event: threading.Event,
        visible_ids: List[str], generation: int,
    ) -> None:
        try:
            if self._library_service is not None:
                verified = self._library_service._call(
                    "verify_projects_batch", visible_ids
                )
            else:
                verified = verify_catalog_projects(
                    index, cancel_event, visible_asset_ids=visible_ids
                )
            self._catalog_verify_succeeded = not cancel_event.is_set()
            _log.info("Asset catalog verify: verified=%d cancelled=%s", verified, cancel_event.is_set())
        except Exception:
            _log.exception("Asset Manager catalog verify failed")
        finally:
            with self._folder_scan_lock:
                self._catalog_verify_active = False
                self._catalog_verify_refresh_pending = True
                if self._catalog_verify_thread is threading.current_thread():
                    self._catalog_verify_thread = None
            scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
            if callable(scheduler):
                scheduler(lambda: self._complete_catalog_verify(generation))

    def _complete_catalog_verify(self, generation: Optional[int] = None) -> None:
        if generation is not None and generation != self._mount_generation:
            return
        with self._folder_scan_lock:
            if not self._catalog_verify_refresh_pending:
                return
            self._catalog_verify_refresh_pending = False
        if not self._panel_mounted:
            return
        self._publish_catalog_if_changed()
        self._refresh_records(assets=True, folders=True)
        if self._handle:
            self._handle.dirty_all()

    def _publish_scan_progress(self) -> bool:
        active = self.get_scan_active()
        status = self.get_scan_status()
        if (
            active == self._published_scan_active
            and status == self._published_scan_status
        ):
            return False
        self._published_scan_active = active
        self._published_scan_status = status
        self._dirty_fields(
            "scan_active",
            "scan_status",
            "has_scan_status",
            "refresh_action_tooltip",
            "stop_scan_label",
        )
        return True

    def _sync_default_folder_path(self) -> bool:
        if not self._asset_index:
            return False
        current = str(resolve_default_asset_directory())
        if current == self._last_default_folder_path:
            return False
        setter = getattr(self._library_service, "_call", None) if self._library_service else getattr(self._asset_index, "set_default_folder_path", None)
        if not callable(setter):
            self._last_default_folder_path = current
            return False
        result = setter("set_default_folder_path", current) if self._library_service else setter(current)
        if not result:
            return False
        self._last_default_folder_path = current
        self._repair_selection()
        self._refresh_records(assets=True, folders=True)
        if self._handle:
            self._handle.dirty_all()
        self._scan_asset_folders()
        return True

    def _refresh_records(self, *, assets: bool = False, folders: bool = False) -> None:
        if not self._handle:
            return
        if folders:
            self._handle.update_record_list("folders", self.get_folder_list())
            self._handle.dirty("folders")
            self._handle.dirty("all_assets_count")
        if assets:
            self._release_obsolete_thumbnail_sources()
            rows = self.get_filtered_assets()
            self._release_thumbnails_outside_window()
            self._handle.update_record_list("assets", rows)
            self._handle.dirty("assets")
            for field in (
                "asset_results_summary",
                "asset_list_top_spacer_height",
                "asset_list_bottom_spacer_height",
                "asset_gallery_top_spacer_height",
                "asset_gallery_bottom_spacer_height",
                "asset_card_slot_width",
                "asset_list_wide",
                "asset_list_show_folder",
                "asset_list_gallery_compact",
                "asset_list_name_width", "asset_list_gallery_width", "asset_list_size_width",
                "asset_list_modified_width", "asset_list_folder_width",
            ):
                self._handle.dirty(field)
        self._request_model_update()

    def _update_all_record_lists(self):
        self._refresh_records(assets=True, folders=True)
        return {"counts": {"folders": len(self.get_folder_list()), "assets": len(self.get_filtered_assets())}}

    def _dirty_model(self, *fields):
        field_set = set(fields)
        self._refresh_records(
            assets="assets" in field_set,
            folders="folders" in field_set,
        )
        self._dirty_fields(*(field for field in fields if field not in ("assets", "folders")))

    def _dirty_fields(self, *fields: str) -> None:
        if not self._handle:
            return
        for field in fields:
            self._handle.dirty(field)
        self._request_model_update()

    def _request_model_update(self) -> None:
        if self._handle:
            rml_widgets.request_model_update(self._handle)

    def _reset_scroll(self) -> None:
        self._asset_window_scroll_top = 0.0
        scroll = self._asset_scroll_container()
        if scroll:
            scroll.scroll_top = 0.0

    def _asset_scroll_container(self, doc=None):
        document = doc or self._doc
        return document.get_element_by_id("asset-gallery-scroll") if document else None

    @staticmethod
    def _ui_scale():
        return max(0.1, float(getattr(lf.ui, "get_ui_scale", lambda: 1.0)() or 1.0))

    def _sync_panel_layout(self, doc=None):
        document = doc or self._doc
        popup = document.get_element_by_id("asset-popup") if document else None
        if not popup:
            return False
        scale = self._ui_scale()
        scale_changed = abs(scale - self._last_ui_scale) > 0.001
        self._last_ui_scale = scale
        height = float(popup.client_height or 0) / scale
        if self._host_geometry:
            height = self._host_geometry[1]
        if height <= 0:
            return False
        widths = []
        for identifier in ("asset-shell", "asset-popup"):
            element = document.get_element_by_id(identifier) if document else None
            value = float(getattr(element, "client_width", 0) or 0) if element else 0.0
            if value > 0:
                widths.append(value / scale)
        width = max(widths, default=0.0)
        if self._host_geometry:
            width = self._host_geometry[0]
        if width <= 0:
            width = self._content_width
        if width > 0:
            layout_metrics = breakpoint_metrics(width)
            layout_changed = (
                layout_metrics["breakpoint"] != self._layout_class
                or abs(width - self._content_width) > 0.5
                or scale_changed
            )
            if layout_changed:
                self._layout_class = layout_metrics["breakpoint"]
                self._content_width = width
                self._navigator_width = min(
                    layout_metrics["navigator_max"],
                    max(layout_metrics["navigator_min"], self._navigator_width),
                ) if layout_metrics["navigator_mode"] == "column" else 0.0
                if self._layout_class == "wide":
                    self._inspector_width = min(420.0, max(240.0, self._inspector_width))
                self._dirty_layout_fields()
            elif abs(width - self._content_width) > 0.5:
                self._content_width = width
        if not self._folder_layout_initialized:
            self._folder_layout_initialized = True
            self._folders_collapsed = height < 640
            self._dirty_fields("folders_collapsed", "folders_expanded")
        def measured(identifier, fallback, *, content=False):
            element = document.get_element_by_id(identifier)
            value = getattr(element, "scroll_height" if content else "client_height", 0) if element else 0
            return float(value) / scale if value else fallback
        folder_count = len(self._asset_index_folders())
        local = 77.0 + (0 if self._folders_collapsed else 34.0 * folder_count)
        content = 16.0 + measured("asset-sidebar-local-content", local, content=True) + measured("asset-sidebar-gallery", 132.0) + 8.0
        toolbar = measured("asset-popup-toolbar", 114.0) + 1.0
        header = measured("asset-results-header", 48.0) + 1.0
        signature = (scale, width, self._layout_class, self._is_floating, height, content, toolbar, header,
                     self._info_preferred_height, self._folders_collapsed)
        if signature == self._layout_signature:
            return False
        self._layout_signature = signature
        layout = panel_layout(height, info_height=self._info_preferred_height, toolbar_height=toolbar,
                              results_header_height=header, sidebar_content_height=content)
        self._sidebar_height = layout["sidebar"]
        self._bottom_panel_height = layout["info"]
        # The navigator is beside the results. Its old stacked minimum must
        # not force the browser and Inspector beyond the native host bounds.
        self._main_min_height = 0.0
        self._dirty_fields("sidebar_height", "bottom_panel_height", "main_min_height")
        self._dirty_fields("inspector_style_height", "inspector_reserved_height")
        if scale_changed:
            self._dirty_layout_fields()
        return True

    def on_host_geometry_changed(self, width: float, height: float, scale: float) -> None:
        """Use native host bounds, which cannot grow with overflowing children."""
        self._host_geometry = (width / scale, height / scale)
        self._layout_signature = None
        self._request_model_update()

    def _sync_asset_window_viewport(self, doc=None) -> bool:
        scroll = self._asset_scroll_container(doc)
        if not scroll:
            return False
        try:
            scale = self._ui_scale()
            values = tuple(native_to_dp(value, scale) for value in (
                scroll.scroll_top,
                scroll.client_height,
                getattr(scroll, "client_width", 0.0),
            ))
        except (TypeError, ValueError):
            return False
        old = (
            self._asset_window_scroll_top,
            self._asset_window_client_height,
            self._asset_window_client_width,
        )
        self._asset_window_scroll_top, self._asset_window_client_height, self._asset_window_client_width = values
        return any(abs(before - after) > 0.5 for before, after in zip(old, values))

    def _sync_gallery_card_width(self, doc=None) -> bool:
        old = self._asset_card_slot_width
        self._sync_asset_window_viewport(doc)
        if self._layout_class:
            self._asset_card_slot_width = grid_slot_width(
                self._asset_window_client_width, self.get_thumbnail_size()
            )
        else:
            self._asset_card_slot_width = gallery_slot_width(self._asset_window_client_width)
        return abs(old - self._asset_card_slot_width) > 0.5

    def _bind_dom_event_listeners(self, doc) -> None:
        shell = doc.get_element_by_id("asset-shell")
        if shell:
            shell.add_event_listener("keydown", self._on_asset_manager_keydown)
            shell.add_event_listener("mousedown", self._on_asset_manager_mousedown)
            shell.add_event_listener("click", self._on_asset_manager_click)
            shell.add_event_listener("dblclick", self._on_asset_manager_double_click)
            shell.add_event_listener("dragstart", self._on_asset_drag_start)
            shell.add_event_listener("dragend", self._on_asset_drag_end)
            shell.add_event_listener("dragover", self._on_gallery_drag_over)
            shell.add_event_listener("dragout", self._on_gallery_drag_out)
            shell.add_event_listener("dragdrop", self._on_gallery_drop)
        scroll = doc.get_element_by_id("asset-gallery-scroll")
        if scroll:
            scroll.add_event_listener("scroll", self._on_asset_scroll)
            scroll.add_event_listener("mousescroll", self._on_gallery_precise_scroll)
            scroll.add_event_listener("keydown", self._on_asset_results_keydown)
        doc.add_event_listener("mousemove", self._on_resize_mousemove)
        doc.add_event_listener("mouseup", self._on_resize_mouseup)

    def _on_asset_scroll(self, event) -> None:
        scroll = event.current_target()
        if self._asset_scroll_event_suppressed:
            current = float(scroll.scroll_top or 0.0)
            self._asset_scroll_event_suppressed = False
            if abs(current - self._asset_scroll_suppressed_top) <= 0.01:
                return
        self._asset_window_refresh_pending = True
        self._request_model_update()

    def _on_gallery_precise_scroll(self, event) -> None:
        scroll = event.current_target()
        if not scroll:
            return
        try:
            delta = float(event.get_parameter("wheel_delta_y", "0"))
        except (TypeError, ValueError):
            return
        maximum = max(0.0, float(scroll.scroll_height) - float(scroll.client_height))
        new_top = min(max(float(scroll.scroll_top) + delta * PRECISE_SCROLL_STEP * self._ui_scale(), 0.0), maximum)
        if abs(new_top - float(scroll.scroll_top)) > 0.01:
            scroll.scroll_top = new_top
            self._asset_scroll_event_suppressed = True
            self._asset_scroll_suppressed_top = new_top
        self._asset_window_refresh_pending = True
        self._request_model_update()
        self._stop_event(event)

    def _on_asset_manager_click(self, event) -> None:
        if self._input_capture_active():
            return
        container = event.current_target()
        target = event.target()
        action_element = rml_widgets.find_ancestor_with_attribute(target, "data-asset-action", container)
        if action_element is not None:
            action = action_element.get_attribute("data-asset-action", "")
            asset_id = action_element.get_attribute("data-asset-id", "")
            if action == "gallery" or action.startswith("gallery:"):
                self._select_asset_id(asset_id)
                self._gallery_command(action.partition(":")[2] or "primary")
            elif action == "load":
                self._load_asset(asset_id)
            elif action == "menu":
                self._show_asset_context_menu(asset_id)
            elif action == "select":
                self._select_asset_id(
                    asset_id,
                    multi_select=self._event_multi_select(event),
                    range_select=self._event_range_select(event),
                    row_element=action_element,
                    container=container,
                )
                self._focus_asset_results()
            self._stop_event(event)
            return
        folder_element = rml_widgets.find_ancestor_with_attribute(target, "data-folder-id", container)
        if folder_element is None:
            return
        menu_element = rml_widgets.find_ancestor_with_attribute(target, "data-folder-action", container)
        if menu_element is not None:
            self._show_folder_context_menu(menu_element.get_attribute("data-folder-id", ""))
        else:
            self._select_folder_id(folder_element.get_attribute("data-folder-id", ""))
        self._stop_event(event)

    def _on_asset_manager_mousedown(self, event) -> None:
        if self._input_capture_active():
            return
        try:
            button = int(event.get_parameter("button", "0"))
        except (TypeError, ValueError):
            return
        container = event.current_target()
        resize_element = rml_widgets.find_ancestor_with_attribute(
            event.target(), "data-resize", container
        )
        if resize_element is not None:
            if button == 0:
                self._start_resize(resize_element.get_attribute("data-resize", ""), event)
                # RmlUi detects double clicks only after mousedown propagates.
            return
        if button != 1:
            return
        element = rml_widgets.find_ancestor_with_attribute(event.target(), "data-asset-action", container)
        if element is None or element.get_attribute("data-asset-action", "") != "select":
            return
        asset_id = element.get_attribute("data-asset-id", "")
        if self._select_asset_id(asset_id, row_element=element, container=container):
            self._show_asset_context_menu(asset_id)
            self._stop_event(event)

    def _on_asset_manager_double_click(self, event) -> None:
        if self._input_capture_active():
            return
        container = event.current_target()
        target = event.target()
        if rml_widgets.find_ancestor_with_attribute(target, "data-thumbnail-size", container) is not None:
            self.set_thumbnail_size({
                "compact": 112.0, "narrow": 136.0,
                "medium": 168.0, "wide": 168.0,
            }.get(self._layout_class, 168.0))
            self._stop_event(event)
            return
        resize_element = rml_widgets.find_ancestor_with_attribute(
            target, "data-resize", container
        )
        if resize_element is not None:
            self._reset_resize(resize_element.get_attribute("data-resize", ""))
            self._stop_event(event)
            return
        element = rml_widgets.find_ancestor_with_attribute(event.target(), "data-asset-action", container)
        if element is None or element.get_attribute("data-asset-action", "") != "select":
            return
        asset_id = element.get_attribute("data-asset-id", "")
        if asset_id:
            self._load_asset(asset_id)
            self._stop_event(event)

    def _on_asset_drag_start(self, event) -> None:
        container = event.current_target()
        element = rml_widgets.find_ancestor_with_attribute(
            event.target(), "data-asset-action", container
        )
        if element is None or element.get_attribute("data-asset-action", "") != "select":
            return
        asset_id = element.get_attribute("data-asset-id", "")
        if not asset_id or not self._asset_index:
            return
        remote = self._asset_dict(asset_id) or {}
        if remote.get("remote_only"):
            self._begin_remote_gallery_drag(remote, event)
            return
        project = self._library_command("verify_asset", asset_id)
        if project is None:
            self.refresh_catalog(scan_folders=False)
            return
        asset = (
            project.to_dict()
            if project is not None and hasattr(project, "to_dict")
            else (self._asset_dict(asset_id) or {})
        )
        if not self._project_available(asset):
            self.refresh_catalog(scan_folders=False)
            return
        begin_drag = getattr(lf.ui, "begin_drag_payload", None)
        if not callable(begin_drag):
            return
        if self._drag_payload_token is not None:
            cancel_drag = getattr(lf.ui, "cancel_drag_payload", None)
            if callable(cancel_drag):
                cancel_drag(self._drag_payload_token)
        token = begin_drag(
            PROJECT_DRAG_PAYLOAD_TYPE,
            str(asset.get("path") or ""),
            self._get_asset_display_name(asset),
        )
        self._drag_payload_token = int(token)
        self._gallery_drag = (asset_id, self._gallery_state.get("identity"))
        self._selected_asset_ids = {asset_id}
        self._selection_cursor_id = asset_id
        self._update_selection_type()
        self._sync_asset_selection_dom(container, element)
        self._dirty_selection()
        self._stop_event(event)

    def _begin_remote_gallery_drag(self, asset, event):
        from .asset_gallery_ui import GALLERY_DRAG_PAYLOAD_TYPE
        payload = self._gallery_drag_payload(asset)
        if payload is None:
            return
        if self._drag_payload_token is not None:
            lf.ui.cancel_drag_payload(self._drag_payload_token)
        self._drag_payload_token = int(lf.ui.begin_drag_payload(
            GALLERY_DRAG_PAYLOAD_TYPE, json.dumps(payload), self._get_asset_display_name(asset)))
        self._gallery_drag = (asset["id"], self._gallery_state.get("identity"))
        self._select_asset_id(asset["id"])
        self._stop_event(event)

    def _gallery_drop_target(self, event):
        if not self._gallery_drag or self._gallery_drag[1] != self._gallery_state.get("identity"):
            return None
        element = rml_widgets.find_ancestor_with_attribute(event.target(), "data-folder-id", event.current_target())
        if element is None:
            return None
        asset = self._asset_dict(self._gallery_drag[0]) or {}
        folder = element.get_attribute("data-folder-id", "")
        if (folder == SCOPE_PUBLISHED
                or asset.get("remote_only") and folder in self._asset_index_folders()):
            return element
        return None

    def _on_gallery_drag_over(self, event):
        element = self._gallery_drop_target(event)
        if element is not self._gallery_drop_element:
            self._on_gallery_drag_out(event)
            self._gallery_drop_element = element
            if element:
                element.set_class("is-drag-over", True)

    def _on_gallery_drag_out(self, event):
        if self._gallery_drop_element:
            self._gallery_drop_element.set_class("is-drag-over", False)
            self._gallery_drop_element = None

    def _on_gallery_drop(self, event):
        element = self._gallery_drop_target(event)
        if element is None:
            return
        identifier, identity = self._gallery_drag
        folder = element.get_attribute("data-folder-id", "")
        self._on_gallery_drag_out(event)
        token, self._drag_payload_token = self._drag_payload_token, None
        self._gallery_drag = None
        if token is not None:
            lf.ui.cancel_drag_payload(token)
        self._gallery_drop_asset(identifier, folder, identity)
        self._stop_event(event)

    def _on_asset_drag_end(self, event) -> None:
        self._on_gallery_drag_out(event)
        self._gallery_drag = None
        token = self._drag_payload_token
        self._drag_payload_token = None
        end_drag = getattr(lf.ui, "end_drag_payload", None)
        if token is not None and callable(end_drag):
            end_drag(token)
            self._stop_event(event)

    def _focus_asset_results(self) -> None:
        scroll = self._asset_scroll_container()
        focus = getattr(scroll, "focus", None)
        if callable(focus):
            focus()

    def _gallery_columns(self) -> int:
        if self._layout_class:
            return grid_columns(self._asset_window_client_width, self.get_thumbnail_size())
        return gallery_columns(self._asset_window_client_width)

    def _scroll_cursor_into_view(self, index: int) -> None:
        scroll = self._asset_scroll_container()
        if self._view_mode == "gallery":
            row = index // self._gallery_columns()
            row_height = ASSET_GALLERY_ROW_HEIGHT_DP
            if self._layout_class:
                row_height = grid_slot_width(
                    self._asset_window_client_width, self.get_thumbnail_size()
                ) * 10.0 / 16.0 + 52.0
            start = row * row_height
            end = start + row_height
        else:
            row_height = list_row_height(gallery_column_visible=True) if self._layout_class else ASSET_LIST_ROW_HEIGHT_DP
            start = index * row_height
            end = start + row_height
        top = self._asset_window_scroll_top
        height = self._asset_window_client_height
        if start < top:
            top = start
        elif height > 0 and end > top + height:
            top = max(0.0, end - height)
        self._asset_window_scroll_top = top
        if scroll is not None:
            scroll.scroll_top = top * self._ui_scale()

    def _navigate_selection(self, key: int) -> bool:
        rows = self._filtered_assets()
        if not rows:
            return False
        ids = [str(asset.get("id") or asset.get("project_uuid") or "") for asset in rows]
        if self._view_mode == "list":
            offsets = {KI_UP: -1, KI_DOWN: 1}
        else:
            columns = self._gallery_columns()
            offsets = {KI_LEFT: -1, KI_RIGHT: 1, KI_UP: -columns, KI_DOWN: columns}
        offset = offsets.get(key)
        if offset is None:
            return False
        if self._selection_cursor_id in ids:
            index = ids.index(self._selection_cursor_id)
            index = max(0, min(len(ids) - 1, index + offset))
        else:
            index = len(ids) - 1 if offset < 0 else 0
        asset_id = ids[index]
        self._selected_asset_ids = {asset_id}
        self._selection_cursor_id = asset_id
        self._update_selection_type()
        self._scroll_cursor_into_view(index)
        self._refresh_records(assets=True)
        self._dirty_selection()
        return True

    def _delete_selected_assets(self) -> bool:
        if not self._asset_index or not self._selected_asset_ids:
            return False
        rows = self._filtered_assets()
        ids = [str(asset.get("id") or asset.get("project_uuid") or "") for asset in rows]
        cursor_index = ids.index(self._selection_cursor_id) if self._selection_cursor_id in ids else 0
        selected = self._selected_asset_ids.intersection(ids)
        if not selected:
            return False
        delete_label = tr("common.delete")

        def confirmed(button: str) -> None:
            if button != delete_label or self._library_command("delete_assets", list(selected)) <= 0:
                return
            self._selected_asset_ids.clear()
            self._selection_cursor_id = None
            remaining = self._filtered_assets()
            if remaining:
                next_index = min(cursor_index, len(remaining) - 1)
                next_id = str(
                    remaining[next_index].get("id")
                    or remaining[next_index].get("project_uuid")
                    or ""
                )
                self._selected_asset_ids = {next_id}
                self._selection_cursor_id = next_id
                self._scroll_cursor_into_view(next_index)
            self._update_selection_type()
            self._refresh_records(assets=True, folders=True)
            self._dirty_selection()

        lf.ui.confirm_dialog(
            delete_label,
            tr("projects.dialog.delete_projects"),
            [tr("common.cancel"), delete_label],
            confirmed,
            "error",
        )
        return True

    def _on_gallery_shortcut(self, event):
        if self._input_capture_active():
            return False
        target = event.target()
        tag = getattr(target, "tag_name", "")
        if callable(tag):
            tag = tag()
        if tag in ("input", "textarea", "select"):
            return False
        from .gallery_shortcuts import shortcut_command
        try:
            key = int(event.get_parameter("key_identifier", "0"))
        except (TypeError, ValueError):
            return False
        command = shortcut_command(getattr(lf, "keymap", None), key,
            **{name: event.get_bool_parameter(name + "_key", False) for name in ("ctrl", "shift", "alt", "meta")})
        if command == "refresh_scope":
            if self._selected_folder_id in GALLERY_SCOPES:
                self._gallery_command("refresh")
            else:
                self.refresh_catalog()
        elif command:
            self._gallery_command(command)
        else:
            return False
        self._stop_event(event)
        return True

    def _on_asset_manager_keydown(self, event):
        try:
            key = int(event.get_parameter("key_identifier", "0"))
        except (TypeError, ValueError):
            key = 0
        if key == KI_ESCAPE and self._quick_look_visible:
            self.close_quick_look()
            self._stop_event(event)
            return True
        if key == KI_ESCAPE and self._inspector_expanded:
            self._inspector_expanded = False
            self._dirty_fields("inspector_expanded")
            self._stop_event(event)
            return True
        target = event.target()
        container = event.current_target()
        tag = getattr(target, "tag_name", "")
        if callable(tag):
            tag = tag()
        if key == KI_SPACE and tag not in ("input", "textarea", "select"):
            self.open_quick_look()
            if self._quick_look_visible:
                self._stop_event(event)
                return True
        element = rml_widgets.find_ancestor_with_attribute(target, "data-folder-id", container)
        action = rml_widgets.find_ancestor_with_attribute(target, "data-sidebar-action", container)
        if key in (KI_RETURN, 32) and (element is not None or action is not None):
            if action is not None and action.get_attribute("data-sidebar-action", "") == "toggle_folders":
                self.toggle_folders_collapsed()
            elif element is not None:
                self._select_folder_id(element.get_attribute("data-folder-id", ""))
            self._stop_event(event)
            return True
        return self._on_gallery_shortcut(event)

    def _on_asset_results_keydown(self, event) -> None:
        if self._on_gallery_shortcut(event):
            return
        try:
            key = int(event.get_parameter("key_identifier", "0"))
        except (TypeError, ValueError):
            return
        if key == KI_SPACE:
            self.open_quick_look()
            if self._quick_look_visible:
                self._stop_event(event)
            return
        if self._navigate_selection(key):
            self._stop_event(event)
            return
        if key == KI_RETURN:
            if any(event.get_bool_parameter(name + "_key", False) for name in ("ctrl", "shift", "alt", "meta")):
                return
            asset_id = self._selection_cursor_id or self.get_selected_asset_id()
            visible_ids = {
                str(asset.get("id") or asset.get("project_uuid") or "")
                for asset in self._filtered_assets()
            }
            if asset_id in visible_ids:
                self._load_asset(asset_id)
                self._stop_event(event)
            return
        if key == KI_DELETE:
            if self._delete_selected_assets():
                self._stop_event(event)
            return
        if 2 <= key <= 37 and not self._event_multi_select(event):
            character = str((key - 2) % 10) if key <= 11 else chr(ord("a") + key - 12)
            self.set_search_query(self._search_query + character)
            search = self._doc.get_element_by_id("asset-search-input") if self._doc else None
            focus = getattr(search, "focus", None)
            if callable(focus):
                focus()
            set_selection = getattr(search, "set_selection_range", None)
            if callable(set_selection):
                set_selection(len(self._search_query), len(self._search_query))
            self._stop_event(event)

    def _sync_asset_selection_dom(self, container=None, selected_element=None) -> None:
        root = container or self._doc
        if root is None:
            return
        try:
            rows = root.query_selector_all(".asset-card, .asset-list-row")
        except Exception:
            rows = []
        for row in rows:
            asset_id = row.get_attribute("data-asset-id", "")
            row.set_class("is-selected", asset_id in self._selected_asset_ids)
        if selected_element is not None:
            selected_element.set_class(
                "is-selected",
                selected_element.get_attribute("data-asset-id", "") in self._selected_asset_ids,
            )

    @staticmethod
    def _event_multi_select(event) -> bool:
        if event is None:
            return False
        return any(
            event.get_bool_parameter(key, False)
            for key in ("ctrl_key", "meta_key", "command_key")
        )

    @staticmethod
    def _event_range_select(event) -> bool:
        return bool(event and event.get_bool_parameter("shift_key", False))

    @staticmethod
    def _stop_event(event) -> None:
        try:
            event.stop_propagation()
        except Exception:
            pass

    @staticmethod
    def _input_capture_active() -> bool:
        is_capturing = getattr(getattr(lf, "keymap", None), "is_capturing", None)
        try:
            return bool(is_capturing()) if callable(is_capturing) else False
        except Exception:
            return False

    def on_bottom_panel_resize_start(self, _handle, event, _args):
        self._start_resize("inspector-height", event)

    def _list_column_width(self, column: str) -> float:
        width = self._asset_window_client_width / self._ui_scale()
        columns = list_columns(width)
        widths = {"size": 48.0, "modified": 72.0, "folder": 58.0,
                  "gallery": columns["gallery"]}
        widths.update(self._list_column_overrides)
        fixed = widths["size"] + columns["modified"] * widths["modified"] + columns["folder"] * widths["folder"]
        gaps = 8.0 * (3 + int(columns["modified"]) + int(columns["folder"]))
        remaining = max(0.0, width - 24.0 - 32.0 - gaps - fixed)
        if "gallery" not in self._list_column_overrides:
            widths["gallery"] = min(widths["gallery"], max(96.0, remaining - 64.0))
        widths["name"] = self._list_column_overrides.get("name", max(64.0, remaining - widths["gallery"]))
        return float(widths[column])

    def _start_resize(self, region: str, event) -> None:
        self._resize_region = region
        self._resize_start_x = float(event.get_parameter("mouse_x", "0"))
        self._resize_start_y = float(event.get_parameter("mouse_y", "0"))
        self._resize_start_navigator = self._navigator_width
        self._resize_start_inspector = self._inspector_width
        self._resize_start_height = self._inspector_preferred_height
        self._resize_start_tray = self._tray_height
        self._bottom_panel_dragging = region == "inspector-height"
        if region.startswith("list-column:"):
            self._resize_start_column = region.partition(":")[2]
            self._resize_start_column_width = self._list_column_width(self._resize_start_column)
        self._dirty_fields("bottom_panel_resize_dragging")

    def _reset_resize(self, region: str) -> None:
        self._resize_region = ""
        self._bottom_panel_dragging = False
        defaults = breakpoint_metrics(self._content_width or 1100.0)
        if region == "navigator":
            self._navigator_width = defaults["navigator_default"]
            self._dirty_fields("navigator_width")
        elif region == "inspector":
            self._inspector_width = defaults["inspector_default"]
            self._dirty_layout_fields()
        elif region == "inspector-height":
            self._inspector_preferred_height = defaults["inspector_default"]
            self._info_preferred_height = self._inspector_preferred_height
            self._sync_panel_layout()
            self._dirty_fields("inspector_height", "bottom_panel_height")
        elif region == "tray":
            self._tray_height = 120.0
            self._dirty_fields("tray_height")
        elif region.startswith("list-column:"):
            column = region.partition(":")[2]
            self._list_column_overrides.pop(column, None)
            self._dirty_fields(
                *(f"asset_list_{name}_width" for name in ("name", "gallery", "size", "modified", "folder"))
            )

    def _on_resize_mousemove(self, event) -> None:
        try:
            mouse_y = float(event.get_parameter("mouse_y", "0"))
        except (TypeError, ValueError):
            return
        region = getattr(self, "_resize_region", "")
        delta_x = (float(event.get_parameter("mouse_x", "0")) - self._resize_start_x) / self._ui_scale()
        delta_y = (mouse_y - self._resize_start_y) / self._ui_scale()
        if region == "navigator":
            self._navigator_width = min(240.0, max(120.0, self._resize_start_navigator + delta_x))
            self._dirty_fields("navigator_width")
        elif region == "inspector":
            self._inspector_width = min(420.0, max(240.0, self._resize_start_inspector - delta_x))
            self._dirty_layout_fields()
        elif region == "tray":
            popup = self._doc.get_element_by_id("asset-popup") if self._doc else None
            panel_height = native_to_dp(
                getattr(popup, "client_height", 0), self._ui_scale()
            ) if popup else 0.0
            maximum = max(120.0, min(450.0, panel_height * 0.5))
            self._tray_height = min(maximum, max(120.0, self._resize_start_tray - delta_y))
            self._dirty_fields("tray_height")
        elif region == "inspector-height" or self._bottom_panel_dragging:
            popup = self._doc.get_element_by_id("asset-popup") if self._doc else None
            panel_height = native_to_dp(
                getattr(popup, "client_height", 0), self._ui_scale()
            ) if popup else 0.0
            maximum = max(120.0, min(450.0, panel_height * 0.5))
            self._inspector_preferred_height = min(
                maximum, max(120.0, self._resize_start_height - delta_y)
            )
            self._info_preferred_height = self._inspector_preferred_height
            self._sync_panel_layout()
            self._dirty_fields("bottom_panel_height", "inspector_height")
            self._stop_event(event)
        elif region.startswith("list-column:"):
            column = region.partition(":")[2]
            self._list_column_overrides[column] = min(
                280.0, max(64.0, self._resize_start_column_width + delta_x)
            )
            self._dirty_fields(
                *(f"asset_list_{name}_width" for name in ("name", "gallery", "size", "modified", "folder"))
            )
            self._stop_event(event)

    def _on_resize_mouseup(self, _event) -> None:
        if getattr(self, "_resize_region", ""):
            self._bottom_panel_dragging = False
            self._resize_region = ""
            self._dirty_fields("bottom_panel_resize_dragging")

    def _resolve_event_value(self, args, event, attribute: str) -> str:
        if args and args[0] not in (None, ""):
            return str(args[0])
        if event is None:
            return ""
        for getter_name in ("current_target", "target"):
            getter = getattr(event, getter_name, None)
            element = getter() if callable(getter) else None
            while element is not None:
                value = element.get_attribute(attribute, "")
                if value:
                    return str(value)
                element = element.parent()
        return ""

    def _subscribe_reactive_state(self) -> None:
        if self._reactive_unsubscribers:
            return
        signal = getattr(RuntimeState, "language_generation", None)
        subscribe = getattr(signal, "subscribe", None)
        if callable(subscribe):
            self._reactive_unsubscribers.append(subscribe(lambda _value: self._language_changed()))

    def _language_changed(self) -> None:
        self._refresh_records(assets=True, folders=True)
        if self._handle:
            self._handle.dirty_all()
        self._request_model_update()

    def _unsubscribe_reactive_state(self) -> None:
        for unsubscribe in self._reactive_unsubscribers:
            try:
                unsubscribe()
            except Exception:
                pass
        self._reactive_unsubscribers = []

    def _sync_panel_space_state(self) -> bool:
        get_panel = getattr(lf.ui, "get_panel", None)
        try:
            info = get_panel(self.id) if callable(get_panel) else None
        except Exception:
            info = None
        panel_space = getattr(info, "space", self._panel_space)
        is_floating = panel_space == lf.ui.PanelSpace.FLOATING
        changed = panel_space != self._panel_space or is_floating != self._is_floating
        self._panel_space = panel_space
        self._is_floating = is_floating
        if changed:
            self._layout_signature = None
            self._dirty_layout_fields()
        return changed

    def _refresh_after_project_write(self) -> bool:
        poll_write = getattr(lf, "project_poll_write", None)
        if not callable(poll_write) or not self._asset_index:
            return False
        try:
            poll = poll_write()
            if not isinstance(poll, dict) or "generation" not in poll:
                return False
            generation = int(poll.get("generation") or 0)
            running = bool(poll.get("running"))
            path = str(poll.get("path") or "")
            error = str(poll.get("error") or "")
        except Exception:
            self._log_warn("Failed to poll .licht project save state")
            return False

        previous_generation = self._last_project_write_generation
        completed = (
            previous_generation is not None
            and not running
            and not error
            and (
                self._project_write_was_running
                or generation != previous_generation
                or path != self._last_project_write_path
            )
        )
        self._last_project_write_generation = generation
        self._project_write_was_running = running
        self._last_project_write_path = path
        if not completed or not path:
            return False

        find_by_path = getattr(self._asset_index, "find_asset_by_path", None)
        project = find_by_path(path) if callable(find_by_path) else None
        if project is None:
            folder_id_for_path = getattr(self._asset_index, "folder_id_for_path", None)
            if not callable(folder_id_for_path) or folder_id_for_path(path) is None:
                return False
            try:
                self._library_command("register_licht_asset", path)
            except Exception as exc:
                self._log_error("Failed to register saved project %s: %s", path, exc)
                return False
            self._refresh_records(assets=True, folders=True)
            if self._handle:
                self._handle.dirty_all()
            return True
        verify_asset = getattr(self._asset_index, "verify_asset", None)
        if not callable(verify_asset) or self._library_command("verify_asset", project.id) is None:
            return False
        self._refresh_records(assets=True, folders=True)
        if self._handle:
            self._handle.dirty_all()
        return True

    def on_mount(self, doc):
        super().on_mount(doc)
        self._panel_mounted = True
        self._mount_generation += 1
        self._doc = doc
        self._subscribe_gallery()
        if self._asset_index is None:
            self._start_backend_initialization()
        self._repair_selection()
        self._bind_dom_event_listeners(doc)
        self._subscribe_reactive_state()
        self._sync_panel_space_state()
        self._sync_panel_layout(doc)
        self._sync_asset_window_viewport(doc)
        self._refresh_records(assets=True, folders=True)
        if self._handle:
            self._handle.dirty_all()
        self._catalog_epoch_seen = self._catalog_epoch()
        self._subscribe_catalog()
        self._sync_default_folder_path()
        self._refresh_after_project_write()
        if self._asset_index is not None:
            self._start_catalog_verify()
            self._start_inspection_refresh()
        if self._asset_index is not None and not _folder_scan_completed_in_process:
            self._scan_asset_folders()

    def on_update(self, doc):
        changed = self._sync_panel_space_state()
        changed = self._sync_default_folder_path() or changed
        changed = self._refresh_after_project_write() or changed
        changed = self._sync_panel_layout(doc) or changed
        changed = self._sync_info_thumbnail(doc) or changed
        if self._publish_catalog_if_changed():
            changed = True
        if self._publish_scan_progress():
            changed = True
        if self._asset_window_refresh_pending or self._sync_asset_window_viewport(doc):
            self._asset_window_refresh_pending = False
            self._refresh_records(assets=True)
            changed = True
        return changed

    def on_unmount(self, doc):
        if self._gallery_toast_timer:
            self._gallery_toast_timer.cancel()
            self._gallery_toast_timer = None
        if self._gallery_undo_timer:
            self._gallery_undo_timer.cancel()
            self._gallery_undo_timer = None
        if self._gallery_unsubscribe:
            self._gallery_unsubscribe()
            self._gallery_unsubscribe = None
        with self._folder_scan_lock:
            self._panel_mounted = False
            self._mount_generation += 1
            self._folder_scan_rerun_pending = False
            self._folder_scan_rerun_target = None
            cancel = self._folder_scan_cancel
            verify_cancel = self._catalog_verify_cancel
        if cancel is not None:
            cancel.set()
        if verify_cancel is not None:
            verify_cancel.set()
        if self._inspection_pipeline is not None:
            self._inspection_pipeline.close()
        if self._catalog_unsubscribe:
            self._catalog_unsubscribe()
            self._catalog_unsubscribe = None
        if self._drag_payload_token is not None:
            cancel_drag = getattr(lf.ui, "cancel_drag_payload", None)
            if callable(cancel_drag):
                cancel_drag(self._drag_payload_token)
            self._drag_payload_token = None
        release_texture = getattr(lf.ui, "release_rml_texture", None)
        if callable(release_texture):
            for source in self._thumbnail_sources_by_asset.values():
                release_texture(source)
        self._thumbnail_sources_by_asset.clear()
        if self._info_thumbnail_source:
            if callable(release_texture):
                release_texture(self._info_thumbnail_source)
            self._info_thumbnail_source = ""
        self._unsubscribe_reactive_state()
        try:
            doc.remove_data_model("asset_manager")
        except Exception:
            pass
        self._handle = None
        self._doc = None

    def _on_close_panel(self, _handle=None, _event=None, _args=None):
        self._dismiss_gallery_undo()
        lf.ui.set_panel_enabled(self.id, False)

    @staticmethod
    def _log_info(message: str, *args: Any) -> None:
        text = message % args if args else message
        logger = getattr(lf, "log", None)
        log = getattr(logger, "info", None)
        (log if callable(log) else _log.info)(text)

    @staticmethod
    def _log_warn(message: str, *args: Any) -> None:
        text = message % args if args else message
        logger = getattr(lf, "log", None)
        log = getattr(logger, "warn", None)
        (log if callable(log) else _log.warning)(text)

    @staticmethod
    def _log_error(message: str, *args: Any) -> None:
        text = message % args if args else message
        logger = getattr(lf, "log", None)
        log = getattr(logger, "error", None)
        (log if callable(log) else _log.error)(text)
