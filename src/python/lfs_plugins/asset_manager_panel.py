# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager panel for browsing UUID-identified .licht projects."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import subprocess
import threading
import time
import uuid
import queue
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import quote

import lichtfeld as lf

from .asset_gallery_ui import GalleryAssetMixin, GALLERY_SCOPES, SCOPE_PUBLISHED, SCOPE_ATTENTION
from . import rml_widgets
from .asset_layout import (
    INSPECTOR_COLUMN_MIN,
    breakpoint_metrics,
    breakpoint_for_width,
    card_geometry,
    gallery_columns,
    gallery_slot_width,
    grid_columns,
    grid_slot_width,
    native_to_dp,
    panel_layout,
    list_columns,
    list_column_widths,
)
from .asset_format import format_size
from .project_inspector import (
    InspectionFactsPipeline,
    dialog_model,
    contents_rows,
    pending_removals,
    license_name,
    license_value,
    details_rows,
    operation_actions,
    thumbnail_source_options,
)
from .project_dialog import form_content
from .project_thumbnail import active_project_path, has_renderable_project_viewport, is_active_project_path
from .project_manager_preferences import (
    read_preferences as read_project_manager_preferences,
    read_state as read_project_manager_state,
    set_state as set_project_manager_state,
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
ASSET_LIST_ROW_HEIGHT_DP = 40.0
ASSET_GALLERY_ROW_HEIGHT_DP = 230.0
ASSET_CARD_PREFERRED_WIDTH_DP = 208.0
ASSET_WINDOW_OVERSCAN_ROWS = 2
ASSET_WINDOW_BATCH_ROWS = 4
ASSET_LIST_FALLBACK_ROWS = 24
ASSET_GALLERY_FALLBACK_ROWS = 8
_RML_PATH_SAFE_CHARS = "/:._-~"
_THUMBNAIL_FIT_ALIGN = "cover center"
_SELECTION_UNCHANGED = object()
SCOPE_ALL = "__all__"
SCOPE_RECENT = "__recent__"
PROJECT_DRAG_PAYLOAD_TYPE = "application/x-lichtfeld-project"
_folder_scan_completed_in_process = False


def _move_to_trash(path: str, *, platform: str = os.name, shell32=None) -> None:
    """Move one project to the platform recycle bin without deleting it."""
    if platform != "nt":
        subprocess.run(["gio", "trash", path], check=True, capture_output=True)
        return

    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR),
            ("fFlags", wintypes.WORD),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    FO_DELETE = 0x0003
    FOF_SILENT = 0x0004
    FOF_ALLOWUNDO = 0x0040
    FOF_NOERRORUI = 0x0400
    FOF_WANTNUKEWARNING = 0x4000

    operation = SHFILEOPSTRUCTW()
    operation.wFunc = FO_DELETE
    operation.pFrom = str(Path(path).resolve()) + "\0\0"
    # Keep the Shell's permanent-delete confirmation. If this path cannot be
    # recycled, Windows must let the user cancel instead of deleting silently.
    operation.fFlags = FOF_SILENT | FOF_ALLOWUNDO | FOF_NOERRORUI | FOF_WANTNUKEWARNING
    shell32 = shell32 or ctypes.windll.shell32
    result = shell32.SHFileOperationW(ctypes.byref(operation))
    if result or operation.fAnyOperationsAborted:
        raise OSError(result or 1, "The project was not moved to the Recycle Bin", path)

try:
    from .asset_index import (
        AssetIndex,
        display_name,
        fix_action_for_health,
        is_supported_asset_path,
        resolve_default_asset_directory,
        resolve_asset_manager_storage_path,
    )
    from .asset_index import LibraryService

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

    SORT_MODES = ("name", "size", "iteration", "saved", "opened", "published", "gallery", "folder")
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
        self._navigator_widths = {"wide": 200.0}
        self._text_column_metrics = None
        self._text_measure_key = None
        self._text_locales = None
        self._inspector_label_width = 168.0
        self._inspector_width = INSPECTOR_COLUMN_MIN
        self._inspector_preferred_height = 1000.0
        self._inspector_expanded = False
        self._quick_look_visible = False
        self._thumbnail_menu_visible = False
        self._thumbnail_size = 168.0
        self._layout_class = ""
        self._content_width = 0.0
        self._host_geometry = None
        self._layout_recheck_pending = False
        self._last_ui_scale = 0.0
        self._list_column_overrides: Dict[str, float] = {}
        self._layout_signature = None
        self._main_min_height = 0.0
        self._folder_layout_initialized = False
        self._bottom_panel_dragging = False
        self._resize_region = ""
        self._resize_start_x = 0.0
        self._resize_start_y = 0.0
        self._resize_scale = 1.0
        self._resize_inspector_height_max = 1000.0
        self._resize_last_height = self._inspector_preferred_height
        self._bottom_panel_drag_start_y = 0.0
        self._bottom_panel_start_height = self._bottom_panel_height

        self._asset_card_slot_width = ASSET_CARD_PREFERRED_WIDTH_DP
        self._asset_window_scroll_top = 0.0
        self._asset_window_client_height = 0.0
        self._asset_window_client_width = 0.0
        self._responsive_width_signature = None
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
        self._recent_scope_cache_signature: Optional[tuple[Any, ...]] = None
        self._recent_scope_cache_rows: List[Dict[str, Any]] = []
        self._recent_scope_cache_only_by_id: Dict[str, Dict[str, Any]] = {}
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
        self._contents_feedback: Dict[str, Dict[str, Any]] = {}
        self._ui_callbacks = queue.SimpleQueue()
        self._dialog_kind = ""
        self._dialog_asset_id = ""
        self._dialog_data: Dict[str, Any] = {}
        self._dialog_busy = False
        self._dialog_serial = 0
        self._dialog_key = ""
        self._operations_expanded = None
        self._inspector_sections = {"project": True, "file": True, "gallery": True}
        self._info_thumbnail_geometry = None
        self._verify_results: Dict[str, str] = {}
        self._observed_outer_panel_width = None
        self._outer_panel_width_save_deadline = 0.0
        self._remembered_left_dock_width: Optional[float] = None
        self._init_gallery()
        self._restore_project_manager_preferences()

    def capture_chrome(self) -> Dict[str, Any]:
        folder_id = self._selected_folder_id if self._selected_folder_id in self._asset_index_folders() else SCOPE_ALL
        return {
            "view_mode": self._view_mode,
            "folders_collapsed": self._folders_collapsed,
            "sidebar_height": self._sidebar_height,
            "bottom_panel_height": self._info_preferred_height,
            "navigator_width": self._navigator_width,
            "navigator_widths": dict(self._navigator_widths),
            "inspector_width": self._inspector_width,
            "inspector_height": self._inspector_preferred_height,
            "inspector_height_version": 3,
            "thumbnail_size": self._thumbnail_size,
            "list_column_overrides": dict(self._list_column_overrides),
            "inspector_sections": dict(self._inspector_sections),
            "operations_expanded": self._operations_expanded,
            "sort_mode": self._sort_mode,
            "sort_descending": self._sort_descending,
            "selected_folder_id": folder_id,
        }

    def apply_chrome(self, payload: Any) -> None:
        preferences = read_project_manager_preferences()
        remembered = {}
        if preferences["rememberState"]:
            remembered = read_project_manager_state()
            if remembered:
                payload = remembered
        self._apply_chrome_payload(payload, preferences, device_state=bool(remembered))

    def _apply_chrome_payload(
        self,
        payload: Any,
        preferences: Optional[Dict[str, Any]] = None,
        *,
        device_state: bool = False,
    ) -> None:
        preferences = preferences or read_project_manager_preferences()
        if isinstance(payload, dict):
            if device_state:
                panel_width = payload.get("panel_width")
                if (
                    isinstance(panel_width, (int, float))
                    and not isinstance(panel_width, bool)
                    and math.isfinite(panel_width)
                    and panel_width > 0.0
                ):
                    self._remembered_left_dock_width = float(panel_width)
                    self._restore_remembered_left_dock_width()
            view_mode = payload.get("view_mode")
            if preferences["defaultView"] == "remember" and view_mode in {"gallery", "list"}:
                self._view_mode = view_mode
            widths = payload.get("navigator_widths")
            if not isinstance(widths, dict):
                legacy_width = payload.get("navigator_width")
                widths = {layout: legacy_width for layout in ("medium", "wide")} if (
                    isinstance(legacy_width, (int, float)) and math.isfinite(legacy_width) and legacy_width > 0
                ) else {}
            for layout, default in (("medium", 160.0), ("wide", 200.0)):
                value = widths.get(layout, default)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    self._navigator_widths[layout] = min(240.0, max(160.0 if layout == "wide" else 120.0, float(value)))
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
                self._info_preferred_height = min(500.0, max(180.0, float(value)))
                self._inspector_preferred_height = self._info_preferred_height
            for key, low, high, default in (
                ("navigator_width", 120.0, 240.0, 200.0),
                ("inspector_width", INSPECTOR_COLUMN_MIN, 420.0, INSPECTOR_COLUMN_MIN),
                ("inspector_height", 180.0, 1000.0, 1000.0),
            ):
                number = payload.get(key)
                attribute = "_inspector_preferred_height" if key == "inspector_height" else "_" + key
                if isinstance(number, (int, float)) and math.isfinite(number):
                    setattr(self, attribute, min(high, max(low, float(number))))
                elif not hasattr(self, attribute):
                    setattr(self, attribute, default)
            if payload.get("inspector_height_version") != 3:
                # Version 3 makes the stacked Inspector fill up to half of the
                # panel. Do not retain the former fixed-height preference.
                self._inspector_preferred_height = 1000.0
            number = payload.get("thumbnail_size")
            if not isinstance(number, (int, float)):
                # Migrate the former per-breakpoint preference deterministically.
                sizes = payload.get("thumbnail_sizes")
                number = sizes.get("wide") if isinstance(sizes, dict) else None
            if isinstance(number, (int, float)) and math.isfinite(number):
                self._thumbnail_size = min(320.0, max(112.0, float(number)))
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
        if preferences["defaultView"] in {"gallery", "list"}:
            self._view_mode = preferences["defaultView"]
        if self._handle:
            self._handle.dirty_all()

    def _restore_project_manager_preferences(self) -> None:
        self._remembered_left_dock_width = None
        preferences = read_project_manager_preferences()
        payload = read_project_manager_state() if preferences["rememberState"] else {}
        self._apply_chrome_payload(payload, preferences, device_state=bool(payload))

    def _restore_remembered_left_dock_width(self, panel_space=None) -> None:
        width = self._remembered_left_dock_width
        if width is None:
            return
        if panel_space is None:
            get_panel = getattr(lf.ui, "get_panel", None)
            try:
                info = get_panel(self.id) if callable(get_panel) else None
            except Exception:
                info = None
            if info is None:
                return
            panel_space = getattr(info, "space", None)
        if panel_space != lf.ui.PanelSpace.LEFT_DOCK:
            return
        set_left_dock_width = getattr(lf.ui, "set_left_dock_width", None)
        if not callable(set_left_dock_width):
            return
        set_left_dock_width(width)
        self._remembered_left_dock_width = None

    def reload_project_manager_preferences(self) -> None:
        self._restore_project_manager_preferences()
        self._reset_scroll()
        self._refresh_records(assets=True)
        self._dirty_fields("is_gallery_view", "is_list_view", "thumbnail_size")
        self._dirty_layout_fields()
        self._request_model_update()

    def _persist_project_manager_state(self) -> None:
        try:
            if not read_project_manager_preferences()["rememberState"]:
                return
            # Merge with the previous device state so a temporarily unavailable
            # native geometry query cannot erase the last valid outer width.
            state = read_project_manager_state()
            state.update(self.capture_chrome())
            # Selection belongs to the current catalog, not to device chrome.
            state.pop("selected_folder_id", None)
            # Startup visibility is an explicit preference, not transient panel
            # registry state observed during application shutdown.
            state.pop("panel_open", None)
            if self._panel_space == lf.ui.PanelSpace.LEFT_DOCK:
                panel_width = self._observed_outer_panel_width
                get_left_dock_width = getattr(lf.ui, "get_left_dock_width", None)
                if callable(get_left_dock_width):
                    current_width = float(get_left_dock_width())
                    if math.isfinite(current_width) and current_width > 0.0:
                        panel_width = current_width
                if (isinstance(panel_width, (int, float))
                        and math.isfinite(panel_width) and panel_width > 0.0):
                    state["panel_width"] = float(panel_width)
            set_project_manager_state(state)
        except (OSError, TypeError, ValueError, AttributeError) as exc:
            self._log_warn("Failed to save Project Manager preferences: %s", exc)

    def _start_backend_initialization(self) -> None:
        if self._backend_load_active or not BACKEND_AVAILABLE:
            if not BACKEND_AVAILABLE:
                self._catalog_load_failed = True
            return
        self._backend_load_active = True
        generation = self._mount_generation

        def worker() -> None:
            index = None
            service = None
            storage_path = None
            default_path = ""
            loaded = False
            recovered_operations = {}
            backend_error = None
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
                from .project_operations import ProjectOperations
                recovered_operations = ProjectOperations(lf.io).recover()
            except Exception as exc:
                backend_error = exc
                self._log_error("Failed to initialize Projects path=%s: %s", storage_path, exc)

            def complete() -> None:
                if generation != self._mount_generation or not self._panel_mounted:
                    self._backend_load_active = False
                    return
                self._backend_load_active = False
                self._catalog_load_failed = not loaded
                self._project_operations.update(recovered_operations)
                if backend_error is not None:
                    self._set_catalog_notice(str(backend_error))
                if index is not None and service is not None:
                    self._asset_index = index
                    self._library_service = service
                    self.STORAGE_PATH = storage_path
                    self.__class__.STORAGE_PATH = storage_path
                    self._last_default_folder_path = default_path
                    self._invalidate_recent_scope_cache()
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

            self._schedule_ui(complete)

        try:
            threading.Thread(target=worker, daemon=True, name="AssetManagerCatalogLoad").start()
        except Exception as exc:
            self._backend_load_active = False
            self._catalog_load_failed = True
            _log.exception("Start Projects catalog worker failed path=%s", self.STORAGE_PATH)
            self._set_catalog_notice(str(exc))

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("asset_manager")
        if model is None:
            return

        self._bind_gallery_model(model)
        model.bind("search_query", self.get_search_query, self.set_search_query)
        model.bind_func("search_is_empty", lambda: not self._search_query)
        model.bind("selected_folder_id", lambda: self._selected_folder_id or SCOPE_ALL, self._set_scope_value)
        model.bind("thumbnail_size", self.get_thumbnail_size, self.set_thumbnail_size)
        model.bind_func("thumbnail_menu_visible", lambda: self._thumbnail_menu_visible)
        model.bind_func("thumbnail_reset_label", lambda: tr("common.reset"))
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

        model.bind_func("asset_list_wide", lambda: self._list_columns()["modified"])
        model.bind_func("asset_list_show_folder", lambda: self._list_columns()["folder"])
        model.bind_func("asset_list_show_size", lambda: self._list_columns()["size"])
        for column in ("name", "gallery", "size", "modified", "folder"):
            model.bind_func(
                f"asset_list_{column}_width",
                lambda column=column: f"{self._list_column_width(column):.1f}dp",
            )
            label_binding = "col_" + column + "_label"
            model.bind_func(label_binding, lambda column=column: self._list_header_label(column))
        model.bind_func("asset_list_gallery_compact", lambda: self._list_columns()["gallery"] == 32)
        model.bind_func(
            "check_gallery_tooltip",
            lambda: f"{tr('projects.action.check_gallery')} · {self._gallery_checked_label()}",
        )
        model.bind_func("is_compact", lambda: self._layout_class == "compact")
        model.bind_func("is_narrow", lambda: self._layout_class == "narrow")
        model.bind_func("is_medium", lambda: self._layout_class == "medium")
        model.bind_func("is_wide", lambda: self._layout_class == "wide")
        model.bind_func("gallery_review_open", self._gallery_review_open)
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
        model.bind_func("inspector_gallery_action_enabled", lambda: (
            self._gallery_badge(self._get_selected_asset())["gallery_action_enabled"] if self._get_selected_asset() else False))
        model.bind_func("inspector_has_gallery_action", lambda: (
            not self.get_selected_asset_can_locate()
            and bool(self._selected_gallery_action())))
        model.bind_func("open_button_label", lambda: tr(
            "projects.action.locate" if self.get_selected_asset_can_locate() else
            "projects.action.repair" if (self._get_selected_asset() or {}).get("status") == "REPAIR_ONLY" else
            "projects.action.open"))
        model.bind_func("inspector_gallery_action_tooltip", lambda: (
            self._gallery_badge(self._get_selected_asset())["gallery_action_label"]
            if self._get_selected_asset() else ""))
        model.bind_func("inspector_gallery_title", lambda: self._gallery_details()["title"])
        model.bind_func("inspector_gallery_description", lambda: self._gallery_details()["description"])
        model.bind_func("inspector_can_edit_gallery_details", self._can_edit_gallery_details)
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
        model.bind_func("contents_undo_label", lambda: tr("projects.contents.undo"))
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
        model.bind_func(
            "quick_look_has_thumbnail",
            lambda: self.get_selected_asset_thumbnail_decorator() != "none",
        )
        model.bind_func("asset_results_summary_visible", lambda: True)
        model.bind_func("asset_results_summary", self.get_asset_results_summary)
        model.bind_func("asset_search_empty", self.get_asset_search_empty)
        model.bind_func("catalog_notice", self.get_catalog_notice)
        model.bind_func("has_catalog_notice", self.get_has_catalog_notice)
        model.bind_func("catalog_loading", lambda: self._backend_load_active)
        model.bind_func("scan_active", self.get_scan_active)
        model.bind_func("scan_status", self.get_scan_status)
        model.bind_func("has_scan_status", self.get_has_scan_status)
        model.bind_func(
            "no_folders",
            lambda: not self._backend_load_active and not self._asset_index_folders(),
        )
        model.bind_func(
            "empty_folder",
            lambda: not self._backend_load_active
            and bool(self._asset_index_folders())
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
            "inspector_training_summary": lambda: self._inspector_fact_summary("iteration", "strategy"),
            "inspector_model_summary": lambda: self._inspector_fact_summary("gaussians", "sh_degree", "SH "),
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
            "inspector_license": lambda: license_name(self._selected_details_rows().get("license_identifier", ""), tr),
            "inspector_license_notice": lambda: self._selected_details_rows().get("license_notice", ""),
            "selected_project_title": lambda: self._selected_details_rows().get("title", ""),
            "inspector_physical_size": lambda: self._selected_details_rows().get("physical_size", ""),
            "inspector_dead_bytes": lambda: self._selected_details_rows().get("dead_bytes", ""),
            "inspector_reclaimable": lambda: self._selected_details_rows().get("reclaimable_percent", ""),
            "inspector_saves": lambda: self._selected_details_rows().get("saves", ""),
            "inspector_autosave_newer": lambda: bool(self._selected_details_rows().get("autosave_newer")),
            "inspector_has_details": lambda: bool(self._selected_inspection().get("details")),
            "inspector_card_diagnostic": lambda: str(getattr(self._selected_inspection().get("card"), "diagnostic", "") or ""),
            "inspector_can_resume": lambda: self._project_available(self._get_selected_asset() or {}) and not self._selected_transfer_recovery() and bool(self._selected_details_rows().get("resumable")),
            "inspector_operations_expanded": self.get_operations_expanded,
            "contents_pending": self.get_contents_pending,
            "contents_has_pending": lambda: bool(self.get_contents_pending()),
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
            "operations_section_title": "projects.contents.title",
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

        model.bind_record_list("folders")
        model.bind_record_list("assets")
        model.bind_record_list("contents_rows")
        for event, handler in (
            ("toggle_folders_collapsed", self.toggle_folders_collapsed),
            ("add_asset_folder", self.add_asset_folder),
            ("on_import_project", self.on_import_project),
            ("on_load_asset", self.on_load_asset),
            ("set_view_mode", self.set_view_mode),
            ("cycle_sort_mode", self.cycle_sort_mode),
            ("open_sort_menu", self.open_sort_menu),
            ("sort_list_column", self.sort_list_column),
            ("close_quick_look", self.close_quick_look),
            ("open_view_menu", self.open_view_menu),
            ("close_thumbnail_menu", self.close_thumbnail_menu),
            ("reset_thumbnail_size", self.reset_thumbnail_size),
            ("open_filter_menu", self.open_filter_menu),
            ("toggle_inspector", self.toggle_inspector),
            ("toggle_inspector_section", self.toggle_inspector_section),
            ("open_inspector_menu", self.open_inspector_menu),
            ("refresh_catalog", self.refresh_catalog),
            ("on_locate_file", self.on_locate_file),
            ("on_use_found_location", self.on_use_found_location),
            ("on_selected_fix", self.on_selected_fix),
            ("open_project_operation", self.open_project_operation),
            ("open_gallery_details", self.open_gallery_details),
            ("contents_action", self.on_contents_action),
            ("toggle_operations", self.toggle_operations),
            ("on_bottom_panel_resize_start", self.on_bottom_panel_resize_start),
            ("close_panel", self._on_close_panel),
            ("clear_search", lambda *_args: self.set_search_query("")),
        ):
            model.bind_event(event, handler)
        self._handle = model.get_handle()
        self._handle.update_record_list("contents_rows", self.get_contents_rows())

    def get_search_query(self) -> str:
        return self._search_query

    def _set_scope_value(self, value: Any) -> None:
        self._select_folder_id(str(value or SCOPE_ALL))

    def select_projects_scope(self) -> None:
        if self._selected_folder_id in GALLERY_SCOPES:
            self._select_folder_id(SCOPE_ALL)

    def get_thumbnail_size(self) -> float:
        return self._thumbnail_size

    def get_navigator_style_width(self) -> str:
        if self._layout_class in ("compact", "narrow", "medium"):
            return "auto"
        return f"{self._navigator_width:.1f}dp"

    def get_inspector_style_width(self) -> str:
        if self._layout_class in ("compact", "narrow"):
            return "auto"
        return f"{self._inspector_width:.1f}dp"

    def get_inspector_style_height(self) -> str:
        if self._layout_class in ("medium", "wide"):
            return "auto"
        return f"{self._inspector_band_height():.1f}dp"

    def _inspector_band_height(self) -> float:
        height = self._host_geometry[1] if self._host_geometry else 700.0
        metrics = breakpoint_metrics(self._content_width or 600.0)
        maximum = min(metrics["inspector_max"], max(metrics["inspector_min"], height * 0.5))
        return min(self._inspector_preferred_height, maximum)

    def _dirty_layout_fields(self) -> None:
        self._dirty_fields(
            "is_compact", "is_narrow", "is_medium", "is_wide",
            "is_floating", "navigator_width", "navigator_style_width",
            "inspector_width", "inspector_style_width", "inspector_height",
            "inspector_style_height", "thumbnail_size", "asset_card_slot_width",
            "asset_card_thumbnail_height", "bottom_panel_height",
            "sidebar_height", "main_min_height",
        )

    def _dirty_list_layout_fields(self) -> None:
        self._responsive_width_signature = None
        self._dirty_fields(
            "asset_list_wide", "asset_list_show_size", "asset_list_show_folder",
            "asset_list_gallery_compact",
            *(f"asset_list_{name}_width" for name in ("name", "gallery", "size", "modified", "folder")),
        )

    def _dirty_inspector_layout(self) -> None:
        self._asset_window_refresh_pending = True
        self._dirty_fields("inspector_expanded")
        self._dirty_list_layout_fields()
        self._request_model_update()

    def set_thumbnail_size(self, value: Any) -> None:
        try:
            number = min(320.0, max(112.0, float(value)))
        except (TypeError, ValueError):
            return
        if abs(self._thumbnail_size - number) < 0.1:
            return
        self._thumbnail_size = number
        self._reset_scroll()
        self._refresh_records(assets=True)
        self._dirty_fields("thumbnail_size")
        self._persist_project_manager_state()

    def set_search_query(self, value: str) -> None:
        self._search_query = str(value or "")
        self._dirty_fields("search_is_empty")
        visible_ids = {
            str(asset.get("id") or asset.get("project_uuid") or "")
            for asset in self._filtered_assets()
        }
        selected = self._selected_asset_ids.intersection(visible_ids)
        cursor = self._selection_cursor_id if self._selection_cursor_id in visible_ids else next(iter(selected), None)
        self._set_asset_selection(selected, cursor=cursor)
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
            "gallery": "projects.gallery.sidebar.title", "folder": "projects.property.folder",
        }[field])

    def get_sort_tooltip(self) -> str:
        direction = tr("projects.sort.descending" if self._sort_descending else "projects.sort.ascending")
        return f"{self._sort_field_label(self._sort_mode)} · {direction}"

    def _sort_menu_items(self) -> List[Dict[str, Any]]:
        fields = ["name", "saved", "opened", "size"]
        if any(self._cached_iteration(asset) is not None for asset in self._asset_index_assets().values()):
            fields.append("iteration")
        fields.extend(("published", "gallery", "folder"))
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
            self._sort_descending = value not in ("name", "gallery", "folder")
        elif kind == "order" and value in ("ascending", "descending"):
            self._sort_descending = value == "descending"
        else:
            return
        self._reset_scroll()
        self._refresh_records(assets=True)
        self._dirty_fields("sort_label", "sort_tooltip")
        self._persist_project_manager_state()

    def open_sort_menu(self, _handle=None, _event=None, _args=None) -> None:
        self._show_shared_context_menu(self._sort_menu_items(), self._choose_sort)

    def _list_header_label(self, column: str) -> str:
        field = "saved" if column == "modified" else column
        label = self._sort_field_label(field)
        if self._sort_mode == field:
            arrow = "↓" if self._sort_descending else "↑"
            return f"{arrow} {label}" if column == "size" else f"{label} {arrow}"
        return label

    def sort_list_column(self, _handle=None, _event=None, args=None) -> None:
        column = str((args or [""])[0])
        field = "saved" if column == "modified" else column
        if field == self._sort_mode:
            self._choose_sort("order:" + ("ascending" if self._sort_descending else "descending"))
        else:
            self._choose_sort("sort:" + field)

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
        if asset_id and asset_id.startswith("recent:"):
            if self._selected_folder_id != SCOPE_RECENT:
                return None
            return self._recent_only_assets().get(asset_id)
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

    @staticmethod
    def _project_path_key(path: Any) -> str:
        text = str(path or "").strip()
        if not text:
            return ""
        try:
            expanded = os.path.expanduser(text)
            normalized = os.path.realpath(os.path.abspath(expanded))
            return os.path.normcase(normalized)
        except (OSError, RuntimeError, TypeError, ValueError):
            return os.path.normcase(os.path.abspath(text))

    def _recent_paths(self) -> List[str]:
        recent_files = getattr(lf, "project_recent_files", None)
        if not callable(recent_files):
            return []
        try:
            paths = list(recent_files() or [])
        except Exception:
            return []
        return [str(path).strip() for path in paths if path is not None and str(path).strip()]

    def _invalidate_recent_scope_cache(self) -> None:
        self._recent_scope_cache_signature = None
        self._recent_scope_cache_rows = []
        self._recent_scope_cache_only_by_id = {}

    def _recent_scope_assets(self) -> List[Dict[str, Any]]:
        """Project the MRU list into catalog assets or temporary, open-only rows."""
        paths = self._recent_paths()
        observations = []
        for path in paths:
            try:
                candidate = Path(path)
                stat = candidate.stat()
                exists = candidate.is_file()
                observations.append((
                    path,
                    exists,
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                    int(stat.st_dev),
                    int(stat.st_ino),
                ))
            except OSError:
                observations.append((path, False, 0, 0, 0, 0))
        signature = (self._catalog_epoch(), tuple(observations))
        if signature == self._recent_scope_cache_signature:
            return self._recent_scope_cache_rows

        assets_by_path = {}
        for asset in self._asset_index_assets().values():
            key = self._project_path_key(asset.get("path"))
            if key:
                assets_by_path.setdefault(key, asset)

        rows = []
        seen_paths = set()
        for path, exists, size, mtime_ns, st_dev, st_ino in observations:
            path_key = self._project_path_key(path)
            if not path_key or path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            asset = assets_by_path.get(path_key)
            if asset is None:
                asset_id = "recent:" + hashlib.sha256(path_key.encode("utf-8")).hexdigest()
                asset = {
                    "id": asset_id,
                    "name": Path(path).stem,
                    "name_origin": "stem",
                    "display_name": Path(path).stem,
                    "path": path,
                    "folder_id": "",
                    "exists": exists,
                    "available": exists,
                    "status": "",
                    "has_preview": False,
                    "file_size_bytes": size,
                    "saved_at_unix_ns": mtime_ns,
                    "stat_identity": {
                        "size": size,
                        "mtime_ns": mtime_ns,
                        "st_dev": st_dev,
                        "st_ino": st_ino,
                    },
                    "recent_only": True,
                }
            rows.append(asset)
            if len(rows) >= 10:
                break
        self._recent_scope_cache_signature = signature
        self._recent_scope_cache_rows = rows
        self._recent_scope_cache_only_by_id = {
            str(asset["id"]): asset for asset in rows if asset.get("recent_only")
        }
        return self._recent_scope_cache_rows

    def _recent_only_assets(self) -> Dict[str, Dict[str, Any]]:
        self._recent_scope_assets()
        return self._recent_scope_cache_only_by_id

    def _asset_index_folders(self) -> Dict[str, Dict[str, Any]]:
        if self._library_service is not None:
            folders = self._library_service.snapshot().get("folders", {})
        else:
            folders = getattr(self._asset_index, "folders", {}) if self._asset_index else {}
        return folders if isinstance(folders, dict) else {}

    def _library_command(self, command: str, *args: Any, **kwargs: Any) -> Any:
        if self._library_service is not None:
            result = self._library_service._call(command, *args, **kwargs)
        else:
            result = getattr(self._asset_index, command)(*args, **kwargs)
        reason = getattr(self._asset_index, "last_error", "")
        if reason:
            self._set_catalog_notice(reason)
        return result

    @staticmethod
    def _native_io_call(name: str, *args: Any, **kwargs: Any) -> Any:
        io = getattr(lf, "io", None)
        function = getattr(io, name, None) if io is not None else None
        if not callable(function):
            raise RuntimeError(f"Project operation is unavailable: {name}")
        return function(*args, **kwargs)

    @staticmethod
    def _active_project_path() -> str:
        return active_project_path()

    @staticmethod
    def _thumbnail_source_availability(path: str) -> tuple[bool, bool]:
        """Decode-probe image sources only while opening or confirming the dialog."""
        try:
            available = AssetManagerPanel._native_io_call(
                "inspect_project_thumbnail_sources", path
            )
            return (
                bool(getattr(available, "first_dataset_image", False)),
                bool(getattr(available, "first_embedded_image", False)),
            )
        except Exception:
            _log.debug("Could not inspect project thumbnail sources path=%s", path, exc_info=True)
            return False, False

    @staticmethod
    def _is_active_project_path(path: str) -> bool:
        return is_active_project_path(path)

    @staticmethod
    def _has_renderable_project_viewport(path: str) -> bool:
        return has_renderable_project_viewport(path)

    def _ensure_inspection_pipeline(self) -> InspectionFactsPipeline:
        if self._inspection_pipeline is None:
            scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
            self._inspection_pipeline = InspectionFactsPipeline(
                lambda path: self._native_io_call("inspect_project_card", path),
                self._inspect_contents,
                self._on_inspection_result,
                scheduler=scheduler if callable(scheduler) else None,
            )
        return self._inspection_pipeline

    def _inspect_contents(self, path):
        details = self._native_io_call("inspect_project_details", path)
        plan = self._native_io_call("plan_reduce_size", path)
        return {"details": details, "plan": plan, "card": getattr(details, "card", None)}

    def _start_inspection_refresh(self) -> None:
        if not self._asset_index or not self._panel_mounted or not self._handle:
            return
        entries = self._window_assets(self._filtered_assets())
        selected = self.get_selected_asset_id()
        self._ensure_inspection_pipeline().refresh(entries, selected)

    def _on_inspection_result(self, asset_id: str, kind: str, result: Any, error: Optional[Exception]) -> None:
        if not self._panel_mounted:
            return
        if self._contents_busy(asset_id):
            return
        if error is not None:
            self._inspection_errors[asset_id] = str(error)
            self._dirty_fields(
                "assets", "selected_has_problem", "selected_health_label",
                "catalog_notice", "has_catalog_notice",
            )
            return
        cleared_error = self._inspection_errors.pop(asset_id, None)
        if cleared_error is not None:
            self._dirty_fields(
                "selected_has_problem", "selected_health_label",
                "catalog_notice", "has_catalog_notice",
            )
        if kind == "details" and isinstance(result, dict) and "details" in result:
            self._inspection_by_asset.setdefault(asset_id, {})["plan"] = result["plan"]
            result = result["details"]
        self._inspection_by_asset.setdefault(asset_id, {})[kind] = result
        if kind == "details":
            iteration = self._details_iteration(result)
            if iteration is not None:
                self._cache_iteration(asset_id, iteration)
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

    def get_operations_expanded(self) -> bool:
        return self._operations_expanded is not False

    def _contents_busy(self, asset_id):
        return any(row.get("asset_id") == asset_id and row.get("status") == "running"
                   for row in self._project_operations.values())

    def get_contents_rows(self):
        facts = self._selected_inspection()
        rows = contents_rows(self._get_selected_asset() or {}, facts.get("details"), facts.get("plan"),
                             tr=tr, format_size=self._format_size, format_time=self._format_contents_time,
                             busy=self._contents_busy(self.get_selected_asset_id()))
        feedback = self._contents_feedback.get(self.get_selected_asset_id(), {})
        for row in rows:
            if row["id"] == feedback.get("row_id"):
                if feedback.get("status") == "running":
                    row["pending"] = True
                    if feedback.get("operation_kind") == "remove":
                        row["detail"] = tr("projects.contents.removing").format(part=row["label"])
                elif feedback.get("status") == "failed":
                    row["error"] = feedback.get("reason", "")
            row["has_detail"] = bool(row["detail"])
            row["has_error"] = bool(row["error"])
        return rows

    @staticmethod
    def _format_contents_time(value):
        try:
            if int(value) <= 0:
                return ""
            date = datetime.fromtimestamp(int(value) / 1_000_000_000)
            return tr("projects.contents.date").format(day=date.day, month=tr(f"projects.contents.month.{date.month}"),
                                                       year=date.year, time=date.strftime("%H:%M"))
        except (OSError, OverflowError, TypeError, ValueError):
            return ""

    def get_contents_pending(self):
        rows = pending_removals(self._selected_inspection().get("details"))
        if not rows:
            return ""
        return tr("projects.contents.pending_total").format(size=self._format_size(sum(int(row.get("bytes", 0)) for row in rows)))

    def _embed_contents_dataset(self, path, progress, cancel):
        details = self._native_io_call("inspect_project_details", path)
        legacy = next((ref for ref in details.references if ref.key == "training_parameters"), None)
        if legacy is not None:
            self._native_io_call("set_dataset_reference", path, str(legacy.path))
        return self._native_io_call("embed_dataset_file", path, progress, cancel)

    def on_contents_action(self, _handle=None, _event=None, args=None):
        if not args or len(args) < 2:
            return
        row_id, action = str(args[0]), str(args[1])
        row = next((row for row in self.get_contents_rows() if row["id"] == row_id), None)
        asset = self._get_selected_asset()
        if not asset or not row or row["disabled"]:
            return
        self._dialog_asset_id = asset["id"]
        path = str(asset["path"])
        if action == "remove":
            if not row["removable"] or row["remove_disabled"]:
                return
            subject = row["label"]
            message = tr("projects.contents.confirm_remove").format(part=subject, size=row["size"] or self._format_size(0))
            if row.get("bound"):
                message += "\n" + tr("projects.contents.keep_model")
            self._set_dialog("remove_content", {"row": row, "message": message})
        elif action == "compact":
            self._set_dialog("compact_content", {"message": tr("projects.contents.confirm_compact")})
        elif action == "restore":
            self._start_project_operation(asset["id"], tr("projects.contents.restore"),
                lambda _progress, _cancel: self._native_io_call("restore_save", path, row["generation"], path),
                content_row=row, operation_kind="restore")
        elif action == "undo_remove":
            self._start_project_operation(asset["id"], tr("projects.contents.undo"),
                lambda _progress, _cancel: self._native_io_call("undo_contents_removal", path, row["removal_id"]),
                content_row=row, operation_kind="undo")
        elif action == "resume":
            if row.get("bound"):
                self._load_asset(asset["id"])
            else:
                self._start_project_operation(asset["id"], tr("projects.action.resume_training"),
                    lambda _progress, _cancel: self._native_io_call("rebind_checkpoint", path, row["checkpoint_uuid"]),
                    after=lambda: self._load_asset(asset["id"]))
        elif action == "embed" and not row["action_disabled"]:
            self._start_project_operation(asset["id"], tr("projects.action.embed_dataset"),
                lambda progress, cancel: self._embed_contents_dataset(path, progress, cancel))
        elif action == "locate":
            directory = lf.ui.open_folder_dialog(tr("projects.dialog.select_dataset"), str(Path(path).parent))
            if directory:
                self._start_project_operation(asset["id"], tr("projects.action.locate_dataset"),
                    lambda _progress, _cancel: self._native_io_call("set_dataset_reference", path, directory))
        elif action in {"thumbnail", "license"}:
            self.open_project_operation(None, None, ["update_thumbnail" if action == "thumbnail" else "license"])

    def toggle_operations(self, _handle=None, _ev=None, _args=None) -> None:
        self._operations_expanded = not self.get_operations_expanded()
        self._dirty_fields("inspector_operations_expanded")
        self._persist_project_manager_state()

    def toggle_inspector_section(self, _handle=None, _ev=None, args=None) -> None:
        section = str(args[0]) if args else ""
        if section in self._inspector_sections:
            self._inspector_sections[section] = not self._inspector_sections[section]
            self._dirty_fields("inspector_" + section + "_expanded")
            self._persist_project_manager_state()

    def open_inspector_menu(self, _handle=None, _ev=None, _args=None) -> None:
        asset_id = self.get_selected_asset_id()
        if asset_id:
            self._show_asset_context_menu(asset_id)

    def _inspector_fact_summary(self, primary, secondary, prefix="") -> str:
        facts = self._selected_details_rows()
        parts = [str(facts.get(primary, ""))]
        if facts.get(secondary) not in (None, ""):
            parts.append(prefix + str(facts[secondary]))
        return " · ".join(part for part in parts if part)

    def _schedule_ui(self, callback: Callable[[], None]) -> None:
        scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
        if callable(scheduler):
            try:
                scheduler(callback)
            except Exception:
                _log.exception("Schedule Projects callback failed path=%s", self.STORAGE_PATH)
                self._ui_callbacks.put(callback)
                try:
                    self._request_model_update()
                except Exception:
                    _log.exception("Wake Projects UI failed path=%s", self.STORAGE_PATH)
        else:
            callback()

    def _drain_ui_callbacks(self):
        while not self._ui_callbacks.empty():
            callback = self._ui_callbacks.get_nowait()
            try:
                callback()
            except Exception as exc:
                _log.exception("Projects callback failed path=%s", self.STORAGE_PATH)
                self._set_catalog_notice(str(exc))

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
            return bool((asset.get("inspection") or {}).get("has_checkpoint"))
        if active == "dataset":
            return bool((asset.get("inspection") or {}).get("has_dataset"))
        if active == "gallery":
            return bool(scene or asset.get("remote_only") or facts.get("relationship") not in (None, "unlinked"))
        return True

    def _repair_selection(self) -> None:
        assets = self._all_display_assets()
        if self._selected_folder_id == SCOPE_RECENT:
            assets.update(self._recent_only_assets())
        folders = self._asset_index_folders()
        selected = self._selected_asset_ids.intersection(assets)
        cursor = self._selection_cursor_id
        if cursor not in assets:
            cursor = next(iter(selected), None)
        self._set_asset_selection(selected, cursor=cursor)
        if self._selected_folder_id not in {*folders, SCOPE_ALL, SCOPE_RECENT, *GALLERY_SCOPES}:
            self._selected_folder_id = SCOPE_ALL
        self._update_selection_type()

    def _set_asset_selection(
        self,
        asset_ids,
        *,
        cursor=_SELECTION_UNCHANGED,
        anchor=_SELECTION_UNCHANGED,
    ) -> bool:
        """Apply selection changes while keeping open previews attached to the cursor."""
        selected = set(asset_ids)
        changed = selected != self._selected_asset_ids
        self._selected_asset_ids = selected
        if cursor is not _SELECTION_UNCHANGED:
            self._selection_cursor_id = cursor
        elif self._selection_cursor_id not in selected:
            self._selection_cursor_id = next(iter(selected), None)
        if anchor is not _SELECTION_UNCHANGED:
            self._selection_anchor_id = anchor
        elif self._selection_anchor_id not in selected:
            self._selection_anchor_id = self._selection_cursor_id
        if changed and self._inspector_expanded and not selected:
            self._inspector_expanded = False
            self._dirty_inspector_layout()
        self._update_selection_type()
        return changed

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
            "UNREADABLE": "projects.status.unreadable",
            "UNSUPPORTED": "projects.status.unreadable",
            "IDENTITY_MISMATCH": "projects.status.identity_mismatch",
            "REPAIR_ONLY": "projects.status.needs_repair",
            "UNSUPPORTED_NEWER": "projects.status.newer_version",
        }.get(status, "projects.status.unverified")
        return tr(key)

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
            dimensions = ""
            preview_width = int(asset.get("preview_width") or 0)
            preview_height = int(asset.get("preview_height") or 0)
            if preview_width > 0 and preview_height > 0:
                dimensions = f"&w={preview_width}&h={preview_height}"
            return AssetManagerPanel._thumbnail_image_decorator(
                f"preview://kind=licht&thumb=256&rev={revision}{dimensions}&path={path}"
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
        dimensions = ""
        preview_width = int(asset.get("preview_width") or 0)
        preview_height = int(asset.get("preview_height") or 0)
        if preview_width > 0 and preview_height > 0:
            dimensions = f"&w={preview_width}&h={preview_height}"
        return AssetManagerPanel._thumbnail_image_decorator(
            f"preview://kind=image&thumb=256&rev={revision}{dimensions}&path={encoded}"
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
        placeholder = element.query_selector(".asset-thumbnail-placeholder")
        if placeholder is None:
            placeholder = element.append_child("span")
            placeholder.set_class_names("asset-thumbnail-placeholder asset-info-thumbnail-placeholder")
            placeholder.set_attribute("aria-hidden", "true")
            image = placeholder.append_child("img")
            image.set_attribute("src", "../icon/scene/splat.png")
            image.set_attribute("alt", "")
        placeholder_title = self._get_asset_display_name(asset) if asset else ""
        placeholder_title_changed = placeholder.get_attribute("title", "") != placeholder_title
        if placeholder_title_changed:
            placeholder.set_attribute("title", placeholder_title)
        # Side inspectors use their content width. The stacked inspector caps
        # its poster so opening it never consumes the whole results viewport.
        width = (max(0.0, self._inspector_width - 12.0) if self._layout_class in ("wide", "medium")
                 else min(240.0, max(0.0, self._content_width - 24.0))
                 if self._layout_class in ("compact", "narrow") else 160.0)
        geometry = (width, width * 10.0 / 16.0)
        geometry_changed = geometry != self._info_thumbnail_geometry
        if created or geometry_changed:
            self._info_thumbnail_geometry = geometry
            element.set_property("width", f"{geometry[0]:.2f}dp")
            element.set_property("height", f"{geometry[1]:.2f}dp")
            # Every responsive layout stacks the preview above the details, so
            # the flex main axis is vertical. Keep the basis equal to the 16:10
            # preview height; using its width in the medium breakpoint expands
            # the element to a square while the Contents inspector is visible.
            element.set_property("flex-basis", f"{geometry[1]:.2f}dp")
        changed = source != self._info_thumbnail_source
        if changed:
            release = getattr(lf.ui, "release_rml_texture", None)
            if self._info_thumbnail_source and callable(release):
                release(self._info_thumbnail_source)
            self._info_thumbnail_source = source
            element.set_property("decorator", decorator)
        placeholder_display = "none" if source else "flex"
        placeholder_display_changed = placeholder.get_property("display") != placeholder_display
        if placeholder_display_changed:
            placeholder.set_property("display", placeholder_display)
        element_display = "flex" if asset else "none"
        visibility_changed = element.get_property("display") != element_display
        if visibility_changed:
            element.set_property("display", element_display)
        return (changed or created or geometry_changed or placeholder_title_changed or
                placeholder_display_changed or visibility_changed)

    def _asset_with_inspection(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        asset_id = str(asset.get("id") or asset.get("project_uuid") or "")
        inspected = self._inspection_by_asset.get(asset_id, {})
        details = inspected.get("details")
        card = getattr(details, "card", None) or inspected.get("card")
        result = dict(asset)
        if card is not None:
            # Native values are fresher than catalog or temporary Recent rows.
            result.update({
                "has_preview": bool(getattr(card, "has_preview", result.get("has_preview"))),
                "preview_width": int(getattr(card, "preview_width", result.get("preview_width", 0)) or 0),
                "preview_height": int(getattr(card, "preview_height", result.get("preview_height", 0)) or 0),
                "file_size_bytes": int(getattr(card, "physical_file_size", result.get("file_size_bytes", 0)) or 0),
                "saved_at_unix_ns": int(getattr(card, "saved_at_unix_ns", result.get("saved_at_unix_ns", 0)) or 0),
                "commit_uuid": str(getattr(card, "commit_uuid", result.get("commit_uuid", "")) or result.get("commit_uuid", "")),
                "native_project_uuid": str(getattr(card, "project_uuid", "") or ""),
            })
        if result.get("recent_only") and asset_id in self._inspection_errors:
            result["status"] = "UNREADABLE"
            result["inspection_error"] = self._inspection_errors[asset_id]
        return result

    def _format_asset_for_ui(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        asset = self._asset_with_inspection(asset)
        asset_id = str(asset.get("id") or asset.get("project_uuid") or "")
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
            "can_load": self._project_available(asset) and not asset.get("recent_only"),
            "thumbnail_decorator": thumbnail_decorator,
        }

    def _release_obsolete_thumbnail_sources(self) -> None:
        ids = getattr(self._asset_index, "iter_project_ids", None)
        live_ids = set(ids() if callable(ids) else self._asset_index_assets())
        live_ids.update(self._gallery_remote_assets())
        live_ids.update(self._recent_only_assets())
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
        if folder_id == SCOPE_RECENT:
            source = self._recent_scope_assets()
        elif folder_id in GALLERY_SCOPES:
            source = self._gallery_rows(folder_id == SCOPE_ATTENTION)
        else:
            source = self._asset_index_assets().values()
        for asset in source:
            if folder_id not in (None, SCOPE_ALL, SCOPE_RECENT, *GALLERY_SCOPES) and asset.get("folder_id") != folder_id:
                continue
            if not self._asset_matches_query(asset, query):
                continue
            if not self._asset_matches_filter(asset):
                continue
            rows.append(asset)
        if folder_id != SCOPE_RECENT:
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
                    "gallery": self._sort_text(self._gallery_badge(asset)["gallery_label"]) if self._sort_mode == "gallery" else "",
                    "folder": self._sort_text(self._folder_name(asset.get("folder_id"))),
                }[self._sort_mode]
                return value, name
            rows.sort(key=sort_value, reverse=self._sort_descending)
        self._last_asset_match_count = len(rows)
        return rows

    def _gallery_window_metrics(self, client_width: float) -> tuple[int, float, float]:
        if self._layout_class:
            columns = grid_columns(client_width, self.get_thumbnail_size())
            slot_width = grid_slot_width(client_width, self.get_thumbnail_size())
            row_height = card_geometry(max(1.0, slot_width - 2.0))["height"] + 14.0
        else:
            columns = gallery_columns(client_width)
            slot_width = gallery_slot_width(client_width)
            row_height = ASSET_GALLERY_ROW_HEIGHT_DP
        return columns, slot_width, row_height

    def _update_gallery_window_geometry(self, total: int) -> tuple[int, int]:
        columns, slot_width, row_height = self._gallery_window_metrics(
            self._asset_window_client_width
        )
        self._asset_card_slot_width = slot_width
        scroll_top = self._asset_window_scroll_top
        client_height = self._asset_window_client_height
        first_row = max(0, int(scroll_top // row_height) - ASSET_WINDOW_OVERSCAN_ROWS)
        start_row = first_row // ASSET_WINDOW_BATCH_ROWS * ASSET_WINDOW_BATCH_ROWS
        visible_rows = (
            math.ceil(client_height / row_height)
            if client_height > 0
            else ASSET_GALLERY_FALLBACK_ROWS
        ) + ASSET_WINDOW_OVERSCAN_ROWS * 2 + ASSET_WINDOW_BATCH_ROWS - 1
        start = min(total, start_row * columns)
        end = min(total, (start_row + visible_rows) * columns)
        total_rows = math.ceil(total / columns) if total else 0
        end_row = math.ceil(end / columns) if end else 0
        self._asset_gallery_top_spacer_height = start_row * row_height
        self._asset_gallery_bottom_spacer_height = max(0, total_rows - end_row) * row_height
        self._asset_list_top_spacer_height = 0.0
        self._asset_list_bottom_spacer_height = 0.0
        return start, end

    def _window_assets(self, assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        total = len(assets)
        scroll_top = self._asset_window_scroll_top
        client_height = self._asset_window_client_height
        if self._view_mode == "gallery":
            start, end = self._update_gallery_window_geometry(total)
        else:
            row_height = ASSET_LIST_ROW_HEIGHT_DP
            first = max(0, int(scroll_top // row_height) - ASSET_WINDOW_OVERSCAN_ROWS)
            start = first // ASSET_WINDOW_BATCH_ROWS * ASSET_WINDOW_BATCH_ROWS
            visible = (
                math.ceil(client_height / row_height)
                if client_height > 0
                else ASSET_LIST_FALLBACK_ROWS
            ) + ASSET_WINDOW_OVERSCAN_ROWS * 2 + ASSET_WINDOW_BATCH_ROWS - 1
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
        if self._asset_index and getattr(self._asset_index, "last_error", ""):
            return self._asset_index.last_error
        selected_id = self.get_selected_asset_id()
        if selected_id.startswith("recent:") and selected_id in self._inspection_errors:
            return self._inspection_errors[selected_id]
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
        asset = self._asset_dict(asset_id)
        return self._asset_with_inspection(asset) if asset else None

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
        if size := self._selected_details_rows().get("physical_size"):
            return size
        asset = self._get_selected_asset()
        return self._format_size(asset.get("file_size_bytes", 0)) if asset else ""

    def get_selected_asset_created(self) -> str:
        asset = self._get_selected_asset()
        return self._format_unix_ns(asset.get("created_at_unix_ns", 0)) if asset else ""

    def get_selected_asset_modified(self) -> str:
        if saved := self._selected_details_rows().get("saved_at"):
            return saved
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
        self._persist_project_manager_state()

    def set_view_mode(self, _handle, _ev, args):
        mode = str(args[0]) if args else ""
        if mode not in ("gallery", "list") or mode == self._view_mode:
            return
        self._view_mode = mode
        self._reset_scroll()
        self._refresh_records(assets=True)
        self._dirty_fields("is_gallery_view", "is_list_view")
        self._persist_project_manager_state()

    def toggle_inspector(self, _handle=None, _ev=None, _args=None):
        self._inspector_expanded = not self._inspector_expanded
        self._dirty_inspector_layout()

    def get_selected_asset_thumbnail_decorator(self) -> str:
        asset = self._get_selected_asset()
        return self._thumbnail_decorator(self._asset_with_poster(asset)) if asset else "none"

    def open_quick_look(self, _handle=None, _ev=None, _args=None) -> None:
        if self.get_selected_asset_id():
            self._quick_look_visible = True
            self._dirty_fields(
                "quick_look_visible", "quick_look_thumbnail"
            )

    def close_quick_look(self, _handle=None, _ev=None, _args=None) -> None:
        if self._quick_look_visible:
            self._quick_look_visible = False
            self._dirty_fields("quick_look_visible")

    def toggle_quick_look(self) -> None:
        if self._quick_look_visible:
            self.close_quick_look()
        else:
            self.open_quick_look()

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
            {"label": tr("projects.property.size") + " ›", "action": "thumbnail", "separator_before": True},
            {"label": tr("projects.action.check_gallery"), "action": "check_gallery", "separator_before": True},
            {"label": tr("projects.action.rescan_folders"), "action": "rescan_folders"},
        ]
        icons = {
            "filter:all": "archive", "filter:attention": "gallery-cloud-bang",
            "filter:not_published": "gallery-cloud-dotted", "filter:published": "gallery-cloud-check",
            "filter:missing": "gallery-cloud-strike", "filter:checkpoint": "gpu",
            "filter:dataset": "scene/dataset", "filter:gallery": "gallery-cloud",
            "gallery": "layout-grid", "list": "layout-list", "thumbnail": "arrows-maximize",
            "check_gallery": "gallery-cloud", "rescan_folders": "sequencer/rotate-cw",
        }
        for item in items:
            item["icon"] = "../icon/" + icons.get(item["action"], "adjustments") + ".png"
        def choose(action: str) -> None:
            if action.startswith("filter:"):
                self._set_filter(action.partition(":")[2])
            elif action in ("gallery", "list"):
                self.set_view_mode(None, None, [action])
            elif action.startswith(("sort:", "order:")):
                self._choose_sort(action)
            elif action == "thumbnail":
                self._thumbnail_menu_visible = True
                self._dirty_fields("thumbnail_menu_visible")
            elif action == "check_gallery":
                self._gallery_command("refresh")
            elif action == "rescan_folders":
                self.refresh_catalog(scan_folders=True)

        self._show_shared_context_menu(items, choose)

    def close_thumbnail_menu(self, _handle=None, _ev=None, _args=None) -> None:
        if self._thumbnail_menu_visible:
            self._thumbnail_menu_visible = False
            self._dirty_fields("thumbnail_menu_visible")

    def reset_thumbnail_size(self, _handle=None, _ev=None, _args=None) -> None:
        self.set_thumbnail_size(168.0)

    def _add_folder_from_path(self, directory: str, *, recursive: bool = True) -> Optional[str]:
        if not self._asset_index or not directory.strip():
            return None
        folder = self._library_command(
            "add_folder", directory.strip(), recursive=recursive
        )
        if folder is None:
            return None
        self._selected_folder_id = folder.id
        self._set_asset_selection(set(), cursor=None, anchor=None)
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
                self._set_asset_selection({project.id}, cursor=project.id, anchor=project.id)
                self.refresh_catalog(scan_folders=False)
            else:
                self._set_catalog_notice(tr("projects.status.import_failed"))
        except Exception:
            self._set_catalog_notice(tr("projects.status.import_failed"))

    def _select_folder_id(self, folder_id: str) -> bool:
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
        self._set_asset_selection(set(), cursor=None, anchor=None)
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
        if asset_id.startswith("recent:"):
            if (
                self._selected_folder_id != SCOPE_RECENT
                or asset_id not in self._recent_only_assets()
            ):
                return False
        elif asset_id not in self._all_display_assets():
            return False
        visible_ids = [
            str(asset.get("id") or asset.get("project_uuid") or "")
            for asset in self._filtered_assets()
        ]
        if range_select and self._selection_anchor_id in visible_ids:
            start = visible_ids.index(self._selection_anchor_id)
            end = visible_ids.index(asset_id)
            lo, hi = sorted((start, end))
            selected = set(visible_ids[lo : hi + 1])
        elif multi_select:
            selected = set(self._selected_asset_ids)
            if asset_id in self._selected_asset_ids:
                selected.remove(asset_id)
            else:
                selected.add(asset_id)
        else:
            selected = {asset_id}
        cursor = asset_id if asset_id in selected else next(iter(selected), None)
        anchor = asset_id if not multi_select and not range_select else _SELECTION_UNCHANGED
        self._set_asset_selection(selected, cursor=cursor, anchor=anchor)
        self._sync_asset_selection_dom(container, row_element)
        self._dirty_selection()
        self._start_inspection_refresh()
        return True

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
            "open_button_label",
            "selected_fix_requires_action",
            "locate_section_title",
            "selected_asset_relocation_candidate",
            "selected_asset_has_relocation_candidate",
            "selected_asset_expected_path",
            "quick_look_visible",
            "quick_look_thumbnail",
            "quick_look_has_thumbnail",
            "inspector_saved", "inspector_saved_at", "inspector_opened",
            "inspector_iteration", "inspector_strategy", "inspector_resumable",
            "inspector_training_summary", "inspector_model_summary",
            "inspector_gaussians", "inspector_sh_degree", "inspector_dataset",
            "inspector_dataset_path", "inspector_dataset_reachable",
            "inspector_embedded", "inspector_has_metrics", "inspector_metrics",
            "inspector_license", "inspector_license_notice", "selected_project_title",
            "inspector_physical_size", "inspector_dead_bytes", "inspector_reclaimable",
            "inspector_saves", "inspector_autosave_newer", "inspector_has_details",
            "inspector_card_diagnostic", "inspector_can_resume",
            "inspector_operations_expanded", "contents_pending", "contents_has_pending",
            "inspector_gallery_action_label", "inspector_has_gallery_action", "inspector_gallery_action_enabled",
            "inspector_gallery_action_tooltip",
            "inspector_gallery_title", "inspector_gallery_description", "inspector_can_edit_gallery_details",
            "inspector_training_tooltip", "inspector_model_tooltip", "inspector_reclaimable_tooltip",
            "inspector_verify_result",
            "catalog_notice",
            "has_catalog_notice",
        )
        if self._handle:
            self._handle.update_record_list("contents_rows", self.get_contents_rows())

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

    def _selected_transfer_recovery(self, *, label=False):
        asset = self._get_selected_asset()
        if not asset:
            return ""
        badge = self._gallery_badge(asset)
        if not badge["health_badge"] and badge["gallery_action"] in ("resume", "retry"):
            return badge["gallery_action_label"] if label else badge["gallery_action"]
        return ""

    def on_load_asset(self, _handle, _ev, args):
        # The action button must not also toggle its containing compact strip.
        if _ev is not None:
            self._stop_event(_ev)
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id") or self.get_selected_asset_id()
        asset = self._asset_dict(asset_id) or {}
        if str(asset.get("status") or "") in ("MISSING", "IDENTITY_MISMATCH"):
            self.on_locate_file(None, None, [asset_id])
        elif asset.get("status") == "REPAIR_ONLY":
            self.open_project_operation(None, None, ["repair"])
        else:
            self._load_asset(asset_id)

    def _dialog_entry(self) -> Optional[Dict[str, Any]]:
        return self._asset_dict(self._dialog_asset_id or self.get_selected_asset_id())

    def get_dialog_title(self) -> str:
        return tr({
            "export_as": "projects.dialog.export_as",
            "update_thumbnail": "projects.dialog.update_thumbnail",
            "license": "projects.contents.license_chooser",
            "remove_content": "projects.contents.remove",
            "compact_content": "projects.contents.compact",
            "rename": "projects.dialog.rename_project",
            "gallery_details": "projects.gallery.details.dialog",
            "repair": "projects.dialog.repair",
            "locate_dataset": "projects.dialog.locate_dataset",
        }.get(self._dialog_kind, "projects.inspector.operations"))

    def get_dialog_confirm_label(self) -> str:
        return tr({
            "export_as": "projects.action.export",
            "update_thumbnail": "projects.action.update_thumbnail",
            "license": "common.save",
            "remove_content": "projects.contents.remove",
            "compact_content": "projects.contents.compact",
            "rename": "common.save",
            "gallery_details": "common.save",
            "repair": "projects.action.repair",
            "locate_dataset": "projects.action.locate_dataset",
        }.get(self._dialog_kind, "common.ok"))

    def _set_dialog(self, kind: str, data: Optional[Dict[str, Any]] = None) -> None:
        self._dialog_serial += 1
        self._dialog_key = f"projects:{id(self)}:{self._dialog_serial}"
        self._dialog_kind = str(kind or "")
        self._dialog_data = dict(data or {})
        self._dialog_busy = bool(self._dialog_data.get("busy"))
        if self._handle:
            self._handle.dirty_all()
        self._request_model_update()
        self._show_project_form()

    def _project_form(self):
        return form_content(self._dialog_kind, self._dialog_data, tr=tr,
                            confirm_label=self.get_dialog_confirm_label(), busy=self._dialog_busy)

    def _show_project_form(self) -> None:
        if not self._dialog_kind:
            return
        body, buttons = self._project_form()
        key = self._dialog_key
        lf.ui.form_dialog(key, self.get_dialog_title(), body, buttons,
                          lambda label, values: self._project_form_result(key, label, values),
                          lambda values: self._project_form_changed(key, values),
                          width=560 if self._dialog_kind in {"remove_content", "compact_content", "license", "gallery_details"} else 720)

    def _refresh_project_form(self, *, body: bool = True) -> None:
        if not self._dialog_kind:
            return
        content, buttons = self._project_form()
        lf.ui.form_dialog_update(self._dialog_key, buttons, content if body else None)

    def _read_project_form(self, values) -> None:
        for key in ("destination", "format", "source", "name", "gallery_title", "gallery_description", "license_choice", "license_name", "license_text", "attribution"):
            if key in values:
                self._dialog_data[key] = str(values[key])

    def _project_form_changed(self, key, values) -> None:
        if key != self._dialog_key or not self._dialog_kind:
            return
        previous = self._dialog_data.get("license_choice")
        self._read_project_form(values)
        current = self._dialog_data.get("license_choice")
        if previous != current:
            # Replace form content after the native control finishes its event.
            self._schedule_ui(lambda: self._refresh_project_form() if key == self._dialog_key else None)

    def _project_form_result(self, key, label, values) -> None:
        if key != self._dialog_key or not self._dialog_kind:
            return
        self._read_project_form(values)
        if not label or label == tr("common.cancel"):
            self.close_project_dialog()
            return
        if label == tr("projects.dialog.choose_destination"):
            self.dialog_choose_destination()
        else:
            self.confirm_project_dialog()
        # A canceled native picker returns to the same form and its values.
        if self._dialog_kind:
            self._show_project_form()

    def close_project_dialog(self, _handle=None, _ev=None, _args=None) -> None:
        self._dialog_kind = ""
        self._dialog_key = ""
        self._dialog_data = {}
        self._dialog_busy = False
        if self._handle:
            self._handle.dirty_all()
        self._request_model_update()

    def _can_edit_gallery_details(self) -> bool:
        asset = self._get_selected_asset()
        if not asset or asset.get("remote_only"):
            return False
        linked = self._gallery_project_id(asset) in self._gallery_state.get("links", {})
        return bool(linked or not asset.get("recent_only") and asset["id"] in self._asset_index_assets())

    def open_gallery_details(self, _handle=None, _ev=None, _args=None) -> None:
        if not self._can_edit_gallery_details():
            return
        asset = self._get_selected_asset()
        details = self._gallery_details(asset)
        self._dialog_asset_id = asset["id"]
        self._set_dialog("gallery_details", {
            "path": asset.get("path", ""), "gallery_title": details["title"],
            "gallery_description": details["description"],
            "gallery_project_id": self._gallery_project_id(asset),
            "gallery_identity": self._gallery_state.get("identity"),
        })

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
        if action == "update_thumbnail":
            dataset_available, embedded_available = self._thumbnail_source_availability(
                str(asset.get("path") or "")
            )
            data = dialog_model(
                action,
                entry=asset,
                details=details,
                dataset_available=dataset_available,
                embedded_available=embedded_available,
                viewport_available=self._has_renderable_project_viewport(
                    str(asset.get("path") or "")
                ),
            )
        else:
            data = dialog_model(action, entry=asset, details=details)
        data["name"] = self._get_asset_display_name(asset)
        self._set_dialog(action, data)

    def dialog_choose_destination(self, _handle=None, _ev=None, _args=None) -> None:
        if self._dialog_kind == "export_as":
            path = self._choose_export_destination(str(self._dialog_data.get("format") or "sog"))
        else:
            source = Path(str(self._dialog_data.get("path") or "project.licht"))
            path = lf.ui.save_project_file_dialog(source.stem + "-copy.licht", str(source.parent))
        if path:
            self._dialog_data["destination"] = str(path)

    def confirm_project_dialog(self, _handle=None, _ev=None, _args=None) -> None:
        action = self._dialog_kind
        asset = self._dialog_entry()
        if not action or not asset or not asset.get("path"):
            return
        path = str(asset["path"])
        data = self._dialog_data
        if action == "gallery_details":
            project_id = data["gallery_project_id"]
            title = str(data.get("gallery_title", "")).strip()
            description = str(data.get("gallery_description", ""))
            if not title:
                data["message"] = tr("projects.gallery.details.title_required")
                self._refresh_project_form()
                return
            try:
                if project_id in self._gallery_state.get("links", {}):
                    controller = self._controller()
                    if controller.service.identity() != data["gallery_identity"]:
                        raise ValueError(tr("projects.gallery.error.account_changed"))
                    controller.service.set_local_details(project_id, title, description)
                    controller._schedule_poll()
                elif not asset.get("recent_only") and self._library_command(
                    "update_asset", asset["id"], gallery_details_draft={"title": title, "description": description}
                ) is not None:
                    self.refresh_catalog(scan_folders=False)
                else:
                    raise ValueError(tr("projects.gallery.error.storage"))
            except Exception as exc:
                data["message"] = str(exc)
                self._refresh_project_form()
                return
            self.close_project_dialog()
            return
        if action == "export_as":
            destination = str(data.get("destination") or "")
            if not destination:
                destination = self._choose_export_destination(str(data.get("format") or "sog"))
                data["destination"] = destination
            if not destination:
                return
            self._start_project_operation(asset["id"], "Export project", lambda progress, cancel: self._native_io_call("export_project_as", path, data.get("format", "sog"), destination, progress, cancel), backup=False)
        elif action == "update_thumbnail":
            dataset_available, embedded_available = self._thumbnail_source_availability(path)
            sources = thumbnail_source_options(
                asset,
                dataset_available=dataset_available,
                embedded_available=embedded_available,
                viewport_available=self._has_renderable_project_viewport(path),
            )
            if data.get("source") not in sources:
                data["sources"] = sources
                data["source"] = next(
                    (
                        source
                        for source in ("first_dataset", "first_embedded", "viewport")
                        if source in sources
                    ),
                    "image_file",
                )
                self._refresh_project_form()
                return
            if not self._start_thumbnail_operation(asset):
                return
        elif action == "license":
            try:
                identifier, notice = license_value(data)
            except ValueError as error:
                data["message"] = tr(str(error))
                return
            self._start_project_operation(asset["id"], tr("projects.contents.license_chooser"), lambda _progress, _cancel: self._native_io_call("set_project_license", path, identifier, notice))
        elif action == "remove_content":
            row = data["row"]
            options = {"compact": False, "drop_unbound_checkpoints": False}
            if row["kind"] == "license":
                function = lambda _progress, _cancel: self._native_io_call("clear_project_license", path)
            else:
                if row["kind"] == "save": options["save_generation"] = row["generation"]
                elif row["kind"] == "checkpoint": options["checkpoint_uuid"] = row["checkpoint_uuid"]
                elif row["kind"] == "dataset": options["drop_embedded_dataset"] = True
                elif row["kind"] == "thumbnail": options["drop_thumbnail"] = True
                elif row["kind"] == "metrics": options["drop_metrics"] = True
                function = lambda progress, cancel: self._native_io_call("reduce_size", path, options, progress, cancel)
            self._start_project_operation(asset["id"], tr("projects.contents.removing").format(part=row["label"]), function,
                                          content_row=row, operation_kind="remove")
        elif action == "compact_content":
            self._start_project_operation(asset["id"], tr("projects.contents.compact"),
                lambda progress, cancel: self._native_io_call("compact_project_file", path, progress, cancel),
                content_row={"id": "compact", "kind": "compact"}, operation_kind="compact")
        elif action == "rename":
            name = str(data.get("name") or "").strip()
            if name:
                after = None if asset.get("recent_only") else lambda: self._rename_catalog_entry(asset["id"], name)
                self._start_project_operation(asset["id"], "Rename project", lambda _progress, _cancel: self._native_io_call("set_project_title", path, name), after=after)
        elif action == "repair":
            destination = str(data.get("destination") or "")
            if not destination:
                self.dialog_choose_destination()
                destination = str(data.get("destination") or "")
            if destination:
                self._start_project_operation(asset["id"], "Repair project", lambda _progress, _cancel: self._native_io_call("repair_project", path, destination, asset["id"]), backup=False)
        elif action == "locate_dataset":
            directory = lf.ui.open_folder_dialog(tr("projects.dialog.select_dataset"), str(Path(path).parent))
            if directory:
                self._start_project_operation(asset["id"], "Locate dataset", lambda _progress, _cancel: self._native_io_call("set_dataset_reference", path, directory))
        if action not in {"export_as", "update_thumbnail", "license", "rename", "repair", "locate_dataset", "remove_content", "compact_content"}:
            return
        self.close_project_dialog()

    def _choose_export_destination(self, format_name: str) -> str:
        chooser = getattr(lf.ui, "save_" + format_name + "_file_dialog", None)
        if callable(chooser):
            return str(chooser("export"))
        return str(getattr(lf.ui, "open_project_file_dialog", lambda *_args: "")(""))

    def _rename_catalog_entry(self, asset_id: str, name: str) -> None:
        self._library_command("update_asset", asset_id, name=name)

    def _start_thumbnail_operation(self, asset: Dict[str, Any]) -> bool:
        source = str(self._dialog_data.get("source") or "first_dataset")
        path = str(asset["path"])
        active = self._is_active_project_path(path)
        project_id = str(asset.get("native_project_uuid") or asset.get("project_uuid") or asset["id"])
        if active and project_id.startswith("recent:"):
            card = self._native_io_call("inspect_project_card", path)
            project_id = str(card.project_uuid)
        if source == "image_file":
            image_path = str(getattr(lf.ui, "open_image_dialog", lambda *_args: "")(""))
            if not image_path:
                return False
            if active:
                self._start_project_operation(
                    asset["id"],
                    tr("projects.action.update_thumbnail"),
                    lambda _progress, _cancel: self._apply_encoded_active_preview(
                        path, project_id, "encode_preview_from_image_file", image_path
                    ),
                    reverify_asset=True,
                    closed_file=False,
                )
            else:
                self._start_project_operation(asset["id"], tr("projects.action.update_thumbnail"), lambda _progress, _cancel: self._native_io_call("preview_from_image_file", path, image_path), reverify_asset=True)
        elif source == "viewport":
            self._start_project_operation(
                asset["id"],
                tr("projects.action.update_thumbnail"),
                lambda _progress, _cancel: self._capture_viewport_preview(path, project_id),
                reverify_asset=True,
                closed_file=False,
            )
        else:
            native_name = "preview_from_first_embedded_image" if source == "first_embedded" else "preview_from_first_dataset_image"
            encode_name = (
                "encode_preview_from_first_embedded_image"
                if source == "first_embedded"
                else "encode_preview_from_first_dataset_image"
            )
            if active:
                self._start_project_operation(
                    asset["id"],
                    tr("projects.action.update_thumbnail"),
                    lambda _progress, _cancel: self._apply_encoded_active_preview(
                        path, project_id, encode_name, path
                    ),
                    reverify_asset=True,
                    closed_file=False,
                )
            else:
                self._start_project_operation(asset["id"], tr("projects.action.update_thumbnail"), lambda _progress, _cancel: self._native_io_call(native_name, path), reverify_asset=True)
        return True

    @staticmethod
    def _capture_viewport_preview(path: str, project_id: str) -> Any:
        from .asset_storage import preview_capture

        active_path = AssetManagerPanel._active_project_path()
        renderable_viewport = False
        try:
            scene_getter = getattr(lf, "get_render_scene", None)
            scene = scene_getter() if callable(scene_getter) else None
            exporter = getattr(lf, "export_viewport_image", None)
            renderable_viewport = (
                scene is not None
                and int(getattr(scene, "total_gaussian_count", 0) or 0) > 0
                and callable(exporter)
            )
        except (RuntimeError, TypeError, ValueError):
            pass
        if "viewport" not in thumbnail_source_options(
            {"path": path},
            viewport_available=(
                renderable_viewport
                and bool(active_path)
                and Path(path).resolve() == Path(active_path).resolve()
            ),
        ):
            raise RuntimeError("The current viewport no longer belongs to this project")

        with preview_capture(project_id) as target:
            exporter = getattr(lf, "export_viewport_image", None)
            if not callable(exporter):
                raise RuntimeError("The current viewport has no captured image")
            exporter(str(target), "png")
            png = target.read_bytes()
        if not AssetManagerPanel._is_active_project_path(path):
            raise RuntimeError("The current viewport no longer belongs to this project")
        AssetManagerPanel._apply_active_project_preview(path, project_id, png)

    @staticmethod
    def _apply_encoded_active_preview(
        path: str, project_id: str, native_name: str, *args: Any
    ) -> None:
        png = AssetManagerPanel._native_io_call(native_name, *args)
        AssetManagerPanel._apply_active_project_preview(path, project_id, png)

    @staticmethod
    def _thumbnail_error_message(error: Any) -> str:
        message = str(error)
        for line in message.splitlines():
            if line.strip().startswith("user_message:"):
                return line.split("user_message:", 1)[1].strip()
        return message

    @staticmethod
    def _apply_active_project_preview(path: str, project_id: str, png: Any) -> None:
        apply = getattr(lf, "project_set_preview", None)
        if not callable(apply):
            raise RuntimeError("The active project cannot accept this thumbnail")
        try:
            apply(png, path=path, project_uuid=str(project_id or ""))
        except RuntimeError as exc:
            raise RuntimeError(AssetManagerPanel._thumbnail_error_message(exc)) from exc
        poll = getattr(lf, "project_poll_write", None)
        if not callable(poll):
            return
        deadline = time.monotonic() + 600.0
        while time.monotonic() < deadline:
            try:
                state = poll()
            except Exception as exc:
                raise RuntimeError(AssetManagerPanel._thumbnail_error_message(exc)) from exc
            if not isinstance(state, dict):
                return
            error = state.get("error") or ""
            if error:
                raise RuntimeError(AssetManagerPanel._thumbnail_error_message(error))
            if not state.get("running"):
                return
            time.sleep(0.05)
        raise RuntimeError("The project thumbnail write did not finish")

    def _start_project_operation(
        self,
        asset_id: str,
        title: str,
        operation: Callable[[Callable[..., None], Callable[[], bool]], Any],
        *,
        after: Optional[Callable[[], None]] = None,
        backup: bool = True,
        content_row: Optional[Dict[str, Any]] = None,
        operation_kind: str = "",
        reverify_asset: bool = False,
        closed_file: bool = True,
    ) -> None:
        if self._contents_busy(asset_id):
            return
        asset = dict(self._asset_dict(asset_id) or {})
        if not asset.get("path"):
            self._set_catalog_notice(tr("projects.status.locate_id_mismatch"))
            return
        project_name = self._get_asset_display_name(asset)
        asset["operation_path"] = str(Path(asset["path"]).resolve())
        inspected = self._inspection_by_asset.get(asset_id, {})
        card = getattr(inspected.get("details"), "card", None) or inspected.get("card")
        native_project_id = str(getattr(card, "project_uuid", "") or "") if card is not None else ""
        if asset.get("recent_only"):
            if not native_project_id:
                self._set_catalog_notice(
                    self._inspection_errors.get(asset_id) or tr("projects.status.unreadable")
                )
                return
            asset["id"] = native_project_id
            asset["commit_uuid"] = str(card.commit_uuid)
        elif card is not None and native_project_id == asset_id:
            asset["commit_uuid"] = str(card.commit_uuid)
        operation_id = "project-" + str(uuid.uuid4())
        cancel = threading.Event()
        metadata = dict(project_name=project_name, operation_kind=operation_kind,
                        part={key: (content_row or {}).get(key, "") for key in ("kind", "label", "number", "total", "images")})
        self._project_operations[operation_id] = {
            "asset_id": asset_id,
            "status": "running",
        }
        self._contents_feedback[asset_id] = dict(row_id=(content_row or {}).get("id", ""), status="running", operation_kind=operation_kind)
        if self._inspection_pipeline is not None:
            self._inspection_pipeline.cancel()
        self._request_model_update()
        self._dirty_selection()

        def complete(error=None, facts=None, inspection_error="") -> None:
            row = self._project_operations.get(operation_id)
            if row is None:
                return
            try:
                if error is not None:
                    raise error
                row["status"] = "completed"
                if self._inspection_pipeline is not None:
                    self._inspection_pipeline.invalidate(asset_id)
                if reverify_asset:
                    self._inspection_by_asset.pop(asset_id, None)
                if facts:
                    self._inspection_by_asset[asset_id] = facts
                self._inspection_errors.pop(asset_id, None)
                if reverify_asset and not asset.get("recent_only"):
                    verify_asset = getattr(self._asset_index, "verify_asset", None)
                    if callable(verify_asset):
                        self._library_command("verify_asset", asset_id)
                if after is not None:
                    after()
                self._contents_feedback.pop(asset_id, None)
                if inspection_error:
                    self._contents_feedback[asset_id] = dict(row_id=(content_row or {}).get("id", ""), status="failed",
                        reason=tr("projects.contents.refresh_failed").format(reason=inspection_error))
                self.refresh_catalog(scan_folders=False)
            except Exception as exc:
                _log.exception("Complete project operation failed operation=%s path=%s", title, asset["path"])
                row["status"] = "failed"
                self._contents_feedback[asset_id] = dict(row_id=(content_row or {}).get("id", ""), status="failed", reason=str(exc))
                if content_row is None:
                    self._set_catalog_notice(str(exc))
            finally:
                self._request_model_update()
                self._dirty_selection()

        def worker() -> None:
            facts = None
            inspection_error = ""

            try:
                if closed_file:
                    from .project_operations import ProjectOperations
                    store = ProjectOperations(lf.io)
                    store.run(
                        operation_id,
                        asset,
                        title,
                        lambda: operation(lambda *_args: None, cancel.is_set),
                        backup=backup,
                        metadata=metadata,
                    )
                else:
                    operation(lambda *_args: None, cancel.is_set)
                error = None
            except Exception as exc:
                _log.exception("Project worker failed operation=%s path=%s", title, asset["path"])
                error = exc
            if error is None and (content_row or operation_kind or asset.get("recent_only")):
                try:
                    facts = self._inspect_contents(str(asset["path"]))
                except Exception as exc:
                    _log.exception("Inspect completed project operation failed path=%s", asset["path"])
                    inspection_error = str(exc)
            self._schedule_ui(lambda: complete(error, facts, inspection_error))

        try:
            threading.Thread(target=worker, daemon=True, name="ProjectsOperation").start()
        except Exception as exc:
            complete(error=exc)

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
        self._set_asset_selection({project.id}, cursor=project.id, anchor=project.id)
        self.refresh_catalog(scan_folders=False)
        return True

    def _load_asset(self, asset_id: str) -> None:
        if not asset_id:
            return
        if asset_id.startswith("recent:"):
            asset = self._asset_dict(asset_id)
            if not asset or not asset.get("recent_only"):
                return
            self._set_asset_selection({asset_id}, cursor=asset_id, anchor=asset_id)
            self._dirty_selection()
            from .file_menu import open_recent_project_with_confirmation

            open_recent_project_with_confirmation(
                str(asset.get("path") or ""),
                keep_asset_manager_open=True,
            )
            return
        if not self._asset_index:
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
        self._set_asset_selection({asset_id}, cursor=asset_id, anchor=asset_id)
        self._dirty_selection()
        from .file_menu import open_project_with_confirmation

        open_project_with_confirmation(
            str(asset.get("path") or ""),
            keep_asset_manager_open=True,
        )

    def on_remove_asset(self, _handle, _ev, args):
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        if asset_id and self._asset_index and self._library_command("delete_asset", asset_id):
            selected = self._selected_asset_ids - {asset_id}
            cursor = None if self._selection_cursor_id == asset_id else self._selection_cursor_id
            self._set_asset_selection(selected, cursor=cursor)
            self.refresh_catalog(scan_folders=False)

    def _show_shared_context_menu(
        self,
        items: List[Dict[str, Any]],
        on_action: Callable[[str], None],
        anchor=None,
    ) -> bool:
        show = getattr(lf.ui, "show_context_menu", None)
        mouse_position = getattr(lf.ui, "get_mouse_screen_pos", None)
        if not callable(show) or (anchor is None and not callable(mouse_position)):
            return False
        try:
            if anchor is not None:
                x = float(anchor.absolute_left)
                y = float(anchor.absolute_top) + float(anchor.absolute_height)
            else:
                x, y = mouse_position()
            show(items, float(x), float(y), on_action)
            return True
        except Exception as exc:
            self._log_error("Failed to show context menu: %s", exc)
            return False

    def _asset_context_menu_items(self, asset: Dict[str, Any]) -> List[Dict[str, Any]]:
        if asset.get("recent_only"):
            asset = self._asset_with_inspection(asset)
            items = [{"label": tr("projects.action.open"), "action": "load"}]
            if self._project_available(asset):
                items.append({
                    "label": tr("projects.action.show_in_folder"),
                    "action": "show_in_folder",
                    "separator_before": True,
                })
            details = self._inspection_by_asset.get(str(asset.get("id") or ""), {}).get("details")
            if details is not None:
                labels = {
                    "inspector": "projects.inspector.title",
                    "export_as": "projects.action.export_as",
                    "update_thumbnail": "projects.action.update_thumbnail",
                    "rename": "projects.action.rename",
                }
                for operation in operation_actions(asset):
                    action = str(operation.get("action") or "")
                    if action in labels:
                        items.append({
                            "label": tr(labels[action]),
                            "action": action if action == "inspector" else "project:" + action,
                            "separator_before": action == "inspector",
                        })
            return items
        items: List[Dict[str, Any]] = []
        if not asset.get("remote_only") and self._project_available(asset):
            items.append({"label": tr("projects.action.open"), "action": "load"})
        if not asset.get("remote_only"):
            items.append({"label": tr("projects.inspector.title"), "action": "inspector"})
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
        if any(operation.get("action") == "rename" for operation in operation_actions(asset)):
            items.append({"label": tr("projects.action.rename"), "action": "project:rename"})
        items.extend(
            [
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
        if details is not None or asset.get("status") == "REPAIR_ONLY":
            labels = {
                "repair": "projects.action.repair",
                "embed_dataset": "projects.action.embed_dataset",
                "locate_dataset": "projects.action.locate_dataset",
                "export_as": "projects.action.export_as",
                "update_thumbnail": "projects.action.update_thumbnail",
            }
            for operation in operation_actions(asset):
                action = str(operation.get("action") or "")
                if action in ("rename", "inspector") or action not in labels:
                    continue
                items.append({
                    "label": tr(labels[action]),
                    "action": "project:" + action,
                    "separator_before": action == "export_as",
                })
        return items

    def _handle_asset_context_action(self, action: str, asset_id: str) -> None:
        if action.startswith("gallery:"):
            if not self._select_asset_id(asset_id):
                return
            self._gallery_command(action.split(":", 1)[1])
        elif action == "load":
            self._load_asset(asset_id)
        elif action == "inspector":
            if self._select_asset_id(asset_id):
                self._inspector_expanded = True
                self._operations_expanded = True
                self._dirty_inspector_layout()
        elif action == "use_found_location":
            self.on_use_found_location(None, None, [asset_id])
        elif action == "show_in_folder":
            self.on_show_in_folder(None, None, [asset_id])
        elif action == "remove":
            self.on_remove_asset(None, None, [asset_id])
        elif action == "trash":
            self.on_move_asset_to_trash(None, None, [asset_id])
        elif action.startswith("project:"):
            if not self._select_asset_id(asset_id):
                return
            self.open_project_operation(None, None, [action.partition(":")[2]])

    def _show_asset_context_menu(self, asset_id: str, anchor=None) -> bool:
        asset = self._asset_dict(asset_id)
        return bool(asset) and self._show_shared_context_menu(
            self._asset_context_menu_items(asset),
            lambda action: self._handle_asset_context_action(action, asset_id),
            anchor,
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
                _move_to_trash(path)
                self._library_command("delete_asset", asset_id)
                selected = self._selected_asset_ids - {asset_id}
                cursor = None if self._selection_cursor_id == asset_id else self._selection_cursor_id
                self._set_asset_selection(selected, cursor=cursor)
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
            self._scan_asset_folders(
                folder_id=folder_id,
                directory=str(folder.get("path") or ""),
                recursive=folder.get("recursive", True) is not False,
            )
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
                self._set_asset_selection(set(), cursor=None, anchor=None)
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
        self._invalidate_recent_scope_cache()
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
        self._invalidate_recent_scope_cache()
        self._repair_selection()
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
                self._invalidate_recent_scope_cache()
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
            verified = verify_catalog_projects(
                self._library_service or index, cancel_event, visible_asset_ids=visible_ids
            )
            self._catalog_verify_succeeded = not cancel_event.is_set()
            _log.info("Asset catalog verify: verified=%d cancelled=%s", verified, cancel_event.is_set())
        except Exception as exc:
            _log.exception("Projects catalog verification failed path=%s", self.STORAGE_PATH)
            reason = str(exc)
            self._schedule_ui(lambda: self._set_catalog_notice(reason))
        finally:
            with self._folder_scan_lock:
                self._catalog_verify_active = False
                self._catalog_verify_refresh_pending = True
                if self._catalog_verify_thread is threading.current_thread():
                    self._catalog_verify_thread = None
            self._schedule_ui(lambda: self._complete_catalog_verify(generation))

    def _complete_catalog_verify(self, generation: Optional[int] = None) -> None:
        if generation is not None and generation != self._mount_generation:
            return
        with self._folder_scan_lock:
            if not self._catalog_verify_refresh_pending:
                return
            self._catalog_verify_refresh_pending = False
        if not self._panel_mounted:
            return
        self._invalidate_recent_scope_cache()
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
            self._measure_text_columns()
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
                "asset_card_thumbnail_height",
                "asset_list_wide",
                "asset_list_show_folder",
                "asset_list_show_size",
                "asset_list_gallery_compact",
                "col_name_label", "col_gallery_label", "col_size_label", "col_modified_label", "col_folder_label",
                "asset_list_name_width", "asset_list_gallery_width", "asset_list_size_width",
                "asset_list_modified_width", "asset_list_folder_width",
            ):
                self._handle.dirty(field)
        self._request_model_update()

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
            breakpoint_changed = layout_metrics["breakpoint"] != self._layout_class
            width_changed = abs(width - self._content_width) > 0.5
            if breakpoint_changed or scale_changed:
                self._layout_class = layout_metrics["breakpoint"]
                self._content_width = width
                if layout_metrics["navigator_mode"] == "column":
                    self._navigator_width = self._navigator_widths[self._layout_class]
                if self._layout_class == "wide":
                    self._inspector_width = min(420.0, max(INSPECTOR_COLUMN_MIN, self._inspector_width))
                self._dirty_layout_fields()
            elif width_changed:
                self._content_width = width
                # The results viewport below owns continuous-width bindings.
                # Waiting for its measured width avoids a stale first update
                # followed by a second update on every host-resize frame.
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
        # Width-specific bindings are updated above. The vertical composition
        # only changes at a breakpoint, not for every pixel inside one.
        signature = (scale, self._layout_class, self._is_floating, height, content, toolbar, header,
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
        self._dirty_fields("inspector_style_height")
        if scale_changed:
            self._dirty_layout_fields()
        self._request_layout_recheck()
        return True

    def _request_layout_recheck(self) -> None:
        # Rml applies the responsive styles after on_update. Read the resulting
        # browser width once on the next UI turn, including float/dock changes.
        schedule = getattr(lf.ui, "schedule", None)
        if not self._panel_mounted or self._layout_recheck_pending or not callable(schedule):
            return
        self._layout_recheck_pending = True
        generation = self._mount_generation

        def recheck() -> None:
            if generation != self._mount_generation:
                return
            self._layout_recheck_pending = False
            if self._panel_mounted and self._sync_asset_window_viewport():
                self._refresh_records(assets=True)
                # The new row heights can change scrollbar space once more.
                self._request_layout_recheck()

        schedule(recheck)

    def on_host_geometry_changed(self, width: float, height: float, scale: float) -> None:
        """Use native host bounds, which cannot grow with overflowing children."""
        geometry = (width / scale, height / scale)
        if self._host_geometry and all(
                abs(before - after) <= 0.5
                for before, after in zip(self._host_geometry, geometry)):
            return
        previous = self._host_geometry
        self._host_geometry = geometry
        if not self._is_floating and math.isfinite(geometry[0]) and geometry[0] > 0.0:
            previous_width = self._observed_outer_panel_width
            self._observed_outer_panel_width = geometry[0]
            if previous_width is not None and abs(previous_width - geometry[0]) > 0.5:
                self._outer_panel_width_save_deadline = time.monotonic() + 0.25
        if (previous is None or abs(previous[1] - geometry[1]) > 0.5
                or breakpoint_for_width(previous[0]) != breakpoint_for_width(geometry[0])):
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
        old_window = self._asset_window_viewport_signature(*old)
        self._asset_window_scroll_top, self._asset_window_client_height, self._asset_window_client_width = values
        width_changed = abs(old[2] - values[2]) > 0.5
        if width_changed:
            # Updating every width binding for every native pixel is expensive:
            # each binding is applied to the header and every visible row. Keep
            # structural changes exact, but coalesce continuous geometry to an
            # 8 dp bucket while the host is being dragged.
            effective_width = self._effective_list_layout_width(values[2])
            columns = list_column_widths(
                effective_width, self._list_column_overrides, self._text_column_metrics
            )
            signature = (
                self._view_mode,
                round(values[2], 1) if self._view_mode == "gallery" else None,
                columns["gallery"], columns["size"],
                columns["modified"], columns["folder"],
            )
            if signature != self._responsive_width_signature:
                self._responsive_width_signature = signature
                if self._view_mode == "gallery":
                    self._update_gallery_window_geometry(self._last_asset_match_count)
                    self._dirty_fields(
                        "asset_card_slot_width", "asset_card_thumbnail_height",
                        "asset_gallery_top_spacer_height", "asset_gallery_bottom_spacer_height",
                    )
                else:
                    self._dirty_fields(
                        "asset_list_wide", "asset_list_show_size", "asset_list_show_folder",
                        "asset_list_gallery_compact", "asset_list_gallery_width",
                        "asset_list_size_width", "asset_list_modified_width",
                        "asset_list_folder_width",
                    )
        return old_window != self._asset_window_viewport_signature(*values)

    def _asset_window_viewport_signature(
        self, scroll_top: float, client_height: float, client_width: float
    ) -> tuple:
        """Geometry that actually changes the virtualized record window."""
        if self._view_mode == "gallery":
            columns, _slot_width, row_height = self._gallery_window_metrics(client_width)
            first = max(0, int(scroll_top // row_height) - ASSET_WINDOW_OVERSCAN_ROWS)
            start = first // ASSET_WINDOW_BATCH_ROWS * ASSET_WINDOW_BATCH_ROWS
            visible = (
                math.ceil(client_height / row_height)
                if client_height > 0
                else ASSET_GALLERY_FALLBACK_ROWS
            ) + ASSET_WINDOW_OVERSCAN_ROWS * 2 + ASSET_WINDOW_BATCH_ROWS - 1
            return "gallery", columns, start, visible
        row_height = ASSET_LIST_ROW_HEIGHT_DP
        first = max(0, int(scroll_top // row_height) - ASSET_WINDOW_OVERSCAN_ROWS)
        start = first // ASSET_WINDOW_BATCH_ROWS * ASSET_WINDOW_BATCH_ROWS
        visible = (
            math.ceil(client_height / row_height)
            if client_height > 0
            else ASSET_LIST_FALLBACK_ROWS
        ) + ASSET_WINDOW_OVERSCAN_ROWS * 2 + ASSET_WINDOW_BATCH_ROWS - 1
        return "list", row_height, start, visible

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
        if self._sync_asset_window_viewport():
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
        if self._sync_asset_window_viewport():
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
                self._show_asset_context_menu(asset_id, action_element)
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
            self.reset_thumbnail_size()
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
        if asset_id.startswith("recent:"):
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
        self._set_asset_selection({asset_id}, cursor=asset_id, anchor=asset_id)
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
            row_height = ASSET_LIST_ROW_HEIGHT_DP
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
        self._set_asset_selection({asset_id}, cursor=asset_id, anchor=asset_id)
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
        selected.intersection_update(self._asset_index_assets())
        if not selected:
            return False
        delete_label = tr("common.delete")

        def confirmed(button: str) -> None:
            if button != delete_label or self._library_command("delete_assets", list(selected)) <= 0:
                return
            self._set_asset_selection(set(), cursor=None, anchor=None)
            remaining = self._filtered_assets()
            if remaining:
                next_index = min(cursor_index, len(remaining) - 1)
                next_id = str(
                    remaining[next_index].get("id")
                    or remaining[next_index].get("project_uuid")
                    or ""
                )
                self._set_asset_selection({next_id}, cursor=next_id, anchor=next_id)
                self._scroll_cursor_into_view(next_index)
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
        if self._input_capture_active():
            return False
        try:
            key = int(event.get_parameter("key_identifier", "0"))
        except (TypeError, ValueError):
            key = 0
        if key == KI_ESCAPE and self._thumbnail_menu_visible:
            self.close_thumbnail_menu()
            self._stop_event(event)
            return True
        if key == KI_ESCAPE and self._quick_look_visible:
            self.close_quick_look()
            self._stop_event(event)
            return True
        if key == KI_ESCAPE and self._inspector_expanded:
            self._inspector_expanded = False
            self._dirty_inspector_layout()
            self._stop_event(event)
            return True
        target = event.target()
        container = event.current_target()
        asset_action = rml_widgets.find_ancestor_with_attribute(
            target, "data-asset-action", container
        )
        if (
            asset_action is not None
            and asset_action.get_attribute("data-asset-action", "") != "select"
        ):
            return False
        tag = getattr(target, "tag_name", "")
        if callable(tag):
            tag = tag()
        if key == KI_SPACE and tag not in ("input", "textarea", "select"):
            self.toggle_quick_look()
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
        if self._input_capture_active():
            return
        container = event.current_target()
        action_element = rml_widgets.find_ancestor_with_attribute(
            event.target(), "data-asset-action", container
        )
        if (
            action_element is not None
            and action_element.get_attribute("data-asset-action", "") != "select"
        ):
            return
        if self._on_gallery_shortcut(event):
            return
        try:
            key = int(event.get_parameter("key_identifier", "0"))
        except (TypeError, ValueError):
            return
        if key == KI_SPACE:
            self.toggle_quick_look()
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
    def _gallery_review_open() -> bool:
        get_panel = getattr(lf.ui, "get_panel_object", None)
        panel = get_panel("lfs.gallery_file") if callable(get_panel) else None
        return bool(panel and getattr(panel, "_review", None))

    @staticmethod
    def _input_capture_active() -> bool:
        if AssetManagerPanel._gallery_review_open():
            return True
        is_capturing = getattr(getattr(lf, "keymap", None), "is_capturing", None)
        try:
            return bool(is_capturing()) if callable(is_capturing) else False
        except Exception:
            return False

    def on_bottom_panel_resize_start(self, _handle, event, _args):
        self._start_resize("inspector-height", event)

    def _list_columns(self):
        return list_columns(
            self._effective_list_layout_width(),
            self._text_column_metrics,
            self._list_column_overrides,
        )

    def _effective_list_layout_width(self, client_width: Optional[float] = None) -> float:
        """Keep list columns stable when the stacked Inspector docks beside them."""
        width = max(0.0, float(
            self._asset_window_client_width if client_width is None else client_width
        ))
        if not self._inspector_expanded or self._layout_class not in ("compact", "narrow"):
            return width
        host_width = max(width, float(self._content_width or 0.0))
        future_side_width = max(260.0, host_width - INSPECTOR_COLUMN_MIN)
        return min(width, future_side_width)

    def _measure_text_columns(self):
        if not self._doc:
            return
        prose = self._doc.get_element_by_id("asset-measure-prose")
        mono = self._doc.get_element_by_id("asset-measure-mono")
        if not prose or not mono or not hasattr(prose, "measure_text"):
            return
        scale = self._ui_scale()
        folders = tuple(sorted({self._folder_name(a.get("folder_id")) for a in self._asset_index_assets().values()}))
        key = (scale, lf.ui.get_current_language(), folders)
        if key == self._text_measure_key:
            return
        if self._text_locales is None:
            import json
            lf.ui.get_languages()  # Load fallback glyphs before measuring every locale.
            directory = Path(lf.ui.resource_directory()) / "locales"
            def flatten(data, prefix=""):
                result = {}
                for name, value in data.items():
                    full = prefix + name
                    if isinstance(value, dict):
                        result.update(flatten(value, full + "."))
                    else:
                        result[full] = value
                return result
            self._text_locales = [flatten(json.loads(path.read_text())) for path in sorted(directory.glob("*.json"))]
        def widest(element, texts):
            return max((element.measure_text(text) / scale for text in texts), default=0.0)
        gallery = [value.format(percent=100) for locale in self._text_locales for name, value in locale.items()
                   if name.startswith("projects.gallery.state.") and "{" not in value.replace("{percent}", "")]
        labels = [value for locale in self._text_locales for name, value in locale.items()
                  if name.startswith("projects.property.")]
        self._inspector_label_width = math.ceil(widest(prose, labels))
        self._text_column_metrics = dict(
            gallery=math.ceil(widest(prose, gallery)) + 16.0 + 24.0,
            size=math.ceil(widest(mono, ["1023.9 " + unit for unit in ("B", "KB", "MB", "GB", "TB")])) + 16.0,
            modified=math.ceil(widest(mono, ["2000-12-30 23:59"])) + 16.0,
            folder=min(240.0, math.ceil(widest(prose, folders)) + 16.0))
        for column in self._text_column_metrics:
            self._text_column_metrics[column] = max(self._text_column_metrics[column],
                math.ceil(prose.measure_text(self._list_header_label(column)) / scale) + 16.0)
        for label in self._doc.query_selector_all(".parameter-label"):
            label.set_property("width", f"{self._inspector_label_width}dp")
            label.set_property("min-width", f"{self._inspector_label_width}dp")
            label.set_property("flex-basis", f"{self._inspector_label_width}dp")
        self._text_measure_key = key

    def _list_column_width(self, column: str) -> float:
        return list_column_widths(
            self._effective_list_layout_width(),
            self._list_column_overrides,
            self._text_column_metrics,
        )[column]

    def _start_resize(self, region: str, event) -> None:
        self._resize_region = region
        self._resize_start_x = float(event.get_parameter("mouse_x", "0"))
        self._resize_start_y = float(event.get_parameter("mouse_y", "0"))
        self._resize_start_navigator = self._navigator_width
        self._resize_start_inspector = self._inspector_width
        self._resize_start_height = (
            self._inspector_band_height()
            if region == "inspector-height"
            else self._inspector_preferred_height
        )
        self._resize_scale = self._ui_scale()
        self._resize_last_height = self._resize_start_height
        self._bottom_panel_dragging = region == "inspector-height"
        if self._bottom_panel_dragging:
            metrics = breakpoint_metrics(self._content_width or 600.0)
            panel_height = self._host_geometry[1] if self._host_geometry else 700.0
            self._resize_inspector_height_max = min(
                metrics["inspector_max"],
                max(metrics["inspector_min"], panel_height * 0.5),
            )
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
            self._navigator_widths[self._layout_class] = self._navigator_width
            self._dirty_fields("navigator_width", "navigator_style_width")
        elif region == "inspector":
            self._inspector_width = defaults["inspector_default"]
            self._dirty_layout_fields()
        elif region == "inspector-height":
            self._inspector_preferred_height = defaults["inspector_default"]
            self._info_preferred_height = self._inspector_preferred_height
            self._sync_panel_layout()
            self._dirty_fields("inspector_height", "inspector_style_height", "bottom_panel_height")
        elif region.startswith("list-column:"):
            column = region.partition(":")[2]
            self._list_column_overrides.pop(column, None)
            self._dirty_fields(
                "asset_list_wide", "asset_list_show_size", "asset_list_show_folder", "asset_list_gallery_compact",
                *(f"asset_list_{name}_width" for name in ("name", "gallery", "size", "modified", "folder"))
            )
        self._persist_project_manager_state()

    def _on_resize_mousemove(self, event) -> None:
        try:
            mouse_y = float(event.get_parameter("mouse_y", "0"))
        except (TypeError, ValueError):
            return
        region = getattr(self, "_resize_region", "")
        scale = max(0.001, getattr(self, "_resize_scale", self._ui_scale()))
        delta_x = (float(event.get_parameter("mouse_x", "0")) - self._resize_start_x) / scale
        delta_y = (mouse_y - self._resize_start_y) / scale
        if region == "navigator":
            metrics = breakpoint_metrics(self._content_width or 1100.0)
            target = min(metrics["navigator_max"], max(
                metrics["navigator_min"], self._resize_start_navigator + delta_x))
            self._navigator_width = target
            self._navigator_widths[self._layout_class] = self._navigator_width
            self._dirty_fields("navigator_width", "navigator_style_width")
        elif region == "inspector":
            self._inspector_width = min(420.0, max(INSPECTOR_COLUMN_MIN, self._resize_start_inspector - delta_x))
            self._dirty_layout_fields()
        elif region == "inspector-height" or self._bottom_panel_dragging:
            metrics = breakpoint_metrics(self._content_width or 600.0)
            target = min(
                self._resize_inspector_height_max,
                max(metrics["inspector_min"], self._resize_start_height - delta_y),
            )
            # Avoid a full panel measurement/layout pass for every mouse pixel.
            # The Inspector height binding is the only live geometry dependency.
            if abs(target - self._resize_last_height) >= 0.5:
                self._inspector_preferred_height = target
                self._resize_last_height = target
                # RmlPanelHost owns the live native-pixel resize. Keeping this
                # callback state-only avoids a second layout invalidation from
                # Python during the same pointer frame; bindings reconcile on Up.
            self._stop_event(event)
        elif region.startswith("list-column:"):
            column = region.partition(":")[2]
            minimum_name = 80.0
            if column == "name":
                minimum_gallery = 32.0 if self._list_columns()["gallery"] == 32 else (self._text_column_metrics or {}).get("gallery", 32.0)
                maximum = self._list_column_width("name") + max(0.0, self._list_column_width("gallery") - minimum_gallery)
                minimum = minimum_name
                self._list_column_overrides.pop("gallery", None)
            else:
                maximum = min(280.0, self._list_column_width(column) + max(0.0, self._list_column_width("name") - minimum_name))
                minimum = (self._text_column_metrics or {}).get(column, 32.0)
                self._list_column_overrides.pop("name", None)
            target = min(maximum, max(minimum, self._resize_start_column_width + delta_x))
            self._list_column_overrides[column] = target
            self._dirty_fields(
                "asset_list_wide", "asset_list_show_size", "asset_list_show_folder", "asset_list_gallery_compact",
                *(f"asset_list_{name}_width" for name in ("name", "gallery", "size", "modified", "folder"))
            )
            self._stop_event(event)

    def _on_resize_mouseup(self, event) -> None:
        region = getattr(self, "_resize_region", "")
        if region:
            if region == "inspector-height":
                self._info_preferred_height = self._inspector_preferred_height
            self._bottom_panel_dragging = False
            self._resize_region = ""
            if region == "inspector-height":
                self._dirty_fields(
                    "bottom_panel_resize_dragging", "inspector_height", "inspector_style_height"
                )
            else:
                self._dirty_fields("bottom_panel_resize_dragging")
            self._persist_project_manager_state()
            self._stop_event(event)

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
        self._dirty_selection()
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
        if info is not None:
            self._restore_remembered_left_dock_width(panel_space)
        if changed:
            self._layout_signature = None
            self._dirty_layout_fields()
        return changed

    def _sync_outer_panel_width_preference(self) -> None:
        if self._is_floating or not read_project_manager_preferences()["rememberState"]:
            self._outer_panel_width_save_deadline = 0.0
            return
        getter = getattr(lf.ui, "get_left_dock_width", None)
        if not callable(getter):
            return
        width = float(getter())
        if not math.isfinite(width) or width <= 0.0:
            return
        now = time.monotonic()
        if self._observed_outer_panel_width is None:
            self._observed_outer_panel_width = width
            return
        if abs(width - self._observed_outer_panel_width) > 0.5:
            self._observed_outer_panel_width = width
            self._outer_panel_width_save_deadline = now + 0.25
            return
        if self._outer_panel_width_save_deadline and now >= self._outer_panel_width_save_deadline:
            self._outer_panel_width_save_deadline = 0.0
            self._persist_project_manager_state()

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

    def refresh_after_thumbnail_write(self, path: str) -> None:
        if not self._panel_mounted or not self._asset_index:
            return
        key = self._project_path_key(path)
        assets = list(self._asset_index_assets().values()) + list(self._recent_only_assets().values())
        for asset in assets:
            if self._project_path_key(asset.get("path")) != key:
                continue
            asset_id = str(asset["id"])
            if not asset.get("recent_only"):
                self._library_command("verify_asset", asset_id)
            if self._inspection_pipeline is not None:
                self._inspection_pipeline.invalidate(asset_id)
            self._inspection_by_asset.pop(asset_id, None)
        self._invalidate_recent_scope_cache()
        self._refresh_records(assets=True)
        self._dirty_selection()
        self._start_inspection_refresh()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._invalidate_recent_scope_cache()
        RuntimeState.projects_panel_visible.value = True
        self._panel_mounted = True
        self._mount_generation += 1
        self._doc = doc
        self._text_measure_key = None
        self._subscribe_gallery()
        if self._asset_index is None:
            self._start_backend_initialization()
        self._repair_selection()
        self._bind_dom_event_listeners(doc)
        self._subscribe_reactive_state()
        self._sync_panel_space_state()
        self._sync_panel_layout(doc)
        self._sync_asset_window_viewport(doc)
        self._request_layout_recheck()
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
        self._persist_project_manager_state()

    def on_update(self, doc):
        self._drain_ui_callbacks()
        changed = self._sync_panel_space_state()
        self._sync_outer_panel_width_preference()
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
        self._persist_project_manager_state()
        RuntimeState.projects_panel_visible.value = False
        self._layout_recheck_pending = False
        self._thumbnail_menu_visible = False
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
        self._persist_project_manager_state()
        lf.ui.set_panel_enabled(self.id, False)

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
