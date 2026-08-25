# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager panel for browsing UUID-identified .licht projects."""

from __future__ import annotations

import logging
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import quote

import lichtfeld as lf

from . import rml_widgets
from .asset_watch import scan_all_asset_folders
from .localization import localized_count
from .rml_keys import KI_DELETE, KI_DOWN, KI_LEFT, KI_RETURN, KI_RIGHT, KI_UP
from .types import Panel
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
SCOPE_ALL = "__all__"
PROJECT_DRAG_PAYLOAD_TYPE = "application/x-lichtfeld-project"

try:
    from .asset_index import (
        AssetIndex,
        is_supported_asset_path,
        resolve_default_asset_directory,
        resolve_asset_manager_storage_path,
    )

    BACKEND_AVAILABLE = True
except ImportError:
    AssetIndex = None
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


class AssetManagerPanel(Panel):
    """Dockable `.licht` project catalog."""

    SORT_MODES = ("name", "size")
    id = "lfs.asset_manager"
    label = "Asset Manager"
    space = lf.ui.PanelSpace.LEFT_DOCK
    order = 20
    template = "rmlui/asset_manager.rml"
    height_mode = lf.ui.PanelHeightMode.FILL
    size = (980, 620)
    options = {lf.ui.PanelOption.DEFAULT_CLOSED}
    update_policy = "interval"
    update_interval_ms = 250

    STORAGE_PATH: Optional[Path] = None

    def __init__(self):
        super().__init__()
        self._handle = None
        self._doc = None
        self._asset_index: Optional[Any] = None

        self._selected_asset_ids: Set[str] = set()
        self._selection_cursor_id: Optional[str] = None
        self._selected_folder_id: Optional[str] = SCOPE_ALL
        self._selection_type = "none"
        self._view_mode = "list"
        self._sort_mode = "name"
        self._search_query = ""

        self._folders_collapsed = False
        self._sidebar_height = 176.0
        self._bottom_panel_height = 220.0
        self._sidebar_dragging = False
        self._sidebar_drag_start_y = 0.0
        self._sidebar_start_height = self._sidebar_height
        self._bottom_panel_dragging = False
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
        self._drag_payload_token: Optional[int] = None
        self._last_project_write_generation: Optional[int] = None
        self._project_write_was_running = False
        self._last_project_write_path = ""
        self._thumbnail_sources_by_asset: Dict[str, str] = {}
        self._last_default_folder_path = ""

    def capture_chrome(self) -> Dict[str, Any]:
        return {
            "folders_collapsed": self._folders_collapsed,
            "sidebar_height": self._sidebar_height,
            "bottom_panel_height": self._bottom_panel_height,
        }

    def apply_chrome(self, payload: Any) -> None:
        if isinstance(payload, dict):
            self._folders_collapsed = bool(
                payload.get("folders_collapsed", self._folders_collapsed)
            )
            for key, attr in (
                ("sidebar_height", "_sidebar_height"),
                ("bottom_panel_height", "_bottom_panel_height"),
            ):
                value = payload.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    setattr(self, attr, float(value))
        if self._handle:
            self._handle.dirty_all()

    def _initialize_backend(self) -> bool:
        if not BACKEND_AVAILABLE:
            return False
        try:
            storage_path = resolve_asset_manager_storage_path()
            storage_path.mkdir(parents=True, exist_ok=True)
            self.STORAGE_PATH = storage_path
            self.__class__.STORAGE_PATH = storage_path
            self._asset_index = AssetIndex()
            loaded = self._asset_index.load()
            self._last_default_folder_path = str(resolve_default_asset_directory())
            return loaded
        except Exception as exc:
            self._log_error("Failed to initialize Asset Manager: %s", exc)
            return False

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("asset_manager")
        if model is None:
            return

        model.bind("search_query", self.get_search_query, self.set_search_query)
        model.bind_func("is_gallery_view", lambda: self._view_mode == "gallery")
        model.bind_func("is_list_view", lambda: self._view_mode == "list")
        model.bind_func("sort_label", self.get_sort_label)
        model.bind_func("folders_collapsed", lambda: self._folders_collapsed)
        model.bind_func("folders_expanded", lambda: not self._folders_collapsed)
        model.bind_func("selected_folder_id", lambda: self._selected_folder_id)
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

        model.bind_func("sidebar_height", lambda: f"{self._sidebar_height:.1f}dp")
        model.bind_func(
            "bottom_panel_height", lambda: f"{self._bottom_panel_height:.1f}dp"
        )
        model.bind_func("sidebar_resize_dragging", lambda: self._sidebar_dragging)
        model.bind_func(
            "bottom_panel_resize_dragging", lambda: self._bottom_panel_dragging
        )
        model.bind_func(
            "asset_card_slot_width", lambda: f"{self._asset_card_slot_width:.1f}dp"
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
        model.bind_func("asset_results_summary_visible", lambda: True)
        model.bind_func("asset_results_summary", self.get_asset_results_summary)

        model.bind_func("selected_asset_name", self.get_selected_asset_name)
        model.bind_func(
            "selected_asset_folder_name", self.get_selected_asset_folder_name
        )
        model.bind_func("selected_asset_path", self.get_selected_asset_path)
        model.bind_func("selected_asset_size", self.get_selected_asset_size)
        model.bind_func("selected_asset_created", self.get_selected_asset_created)
        model.bind_func("selected_asset_modified", self.get_selected_asset_modified)
        model.bind_func(
            "selected_asset_file_missing", self.get_selected_asset_file_missing
        )
        model.bind_func(
            "selected_asset_expected_path", self.get_selected_asset_path
        )
        model.bind_func("selected_folder_name", self.get_selected_folder_name)
        model.bind_func("selected_folder_path", self.get_selected_folder_path)
        model.bind_func(
            "selected_folder_asset_count", self.get_selected_folder_asset_count
        )

        model.bind_func("panel_label", lambda: tr("asset_manager.panel_title"))
        labels = {
            "close_label": "common.close",
            "import_project_label": "menu.file.open_project",
            "search_placeholder": "asset_manager.toolbar.search_placeholder",
            "search_icon_label": "asset_manager.toolbar.search_icon",
            "all_assets_label": "asset_manager.sidebar.all_assets",
            "folders_title": "asset_manager.sidebar.folders",
            "col_name_label": "asset_manager.property.name",
            "col_folder_label": "asset_manager.property.folder",
            "col_size_label": "asset_manager.property.size",
            "col_modified_label": "asset_manager.property.modified",
            "info_tab_label": "asset_manager.info_panel.info",
            "select_item_hint": "asset_manager.status.select_item",
            "asset_details_title": "asset_manager.info_panel.asset_details",
            "folder_details_title": "asset_manager.info_panel.folder_details",
            "file_not_found_title": "asset_manager.info_panel.file_not_found",
            "prop_folder_label": "asset_manager.property.folder",
            "prop_size_label": "asset_manager.property.size",
            "prop_path_label": "asset_manager.property.path",
            "prop_created_label": "asset_manager.property.created",
            "prop_modified_label": "asset_manager.property.modified",
            "prop_expected_path_label": "asset_manager.property.expected_path",
            "prop_assets_label": "asset_manager.property.assets",
            "locate_file_button_label": "asset_manager.action.locate_file",
            "load_button_label": "menu.file.open_project",
        }
        for field, key in labels.items():
            model.bind_func(field, lambda key=key: tr(key))

        model.bind_record_list("folders")
        model.bind_record_list("assets")
        for event, handler in (
            ("toggle_folders_collapsed", self.toggle_folders_collapsed),
            ("add_asset_folder", self.add_asset_folder),
            ("on_import_project", self.on_import_project),
            ("set_view_mode", self.set_view_mode),
            ("cycle_sort_mode", self.cycle_sort_mode),
            ("refresh_catalog", self.refresh_catalog),
            ("on_locate_file", self.on_locate_file),
            ("on_sidebar_resize_start", self.on_sidebar_resize_start),
            ("on_bottom_panel_resize_start", self.on_bottom_panel_resize_start),
            ("close_panel", self._on_close_panel),
        ):
            model.bind_event(event, handler)
        self._handle = model.get_handle()

    def get_search_query(self) -> str:
        return self._search_query

    def set_search_query(self, value: str) -> None:
        self._search_query = str(value or "")
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
        key = (
            "asset_manager.toolbar.sort_by_size"
            if self._sort_mode == "size"
            else "asset_manager.toolbar.sort_by_name"
        )
        return tr(key)

    def get_selected_asset_id(self) -> str:
        return next(iter(self._selected_asset_ids)) if len(self._selected_asset_ids) == 1 else ""

    def get_selected_count(self) -> int:
        return len(self._selected_asset_ids)

    def get_selection_type(self) -> str:
        return self._selection_type

    def get_selected_count_text(self) -> str:
        count = len(self._selected_asset_ids)
        if count == 0:
            return tr("asset_manager.status.select_item")
        if count == 1:
            return tr("asset_manager.status.one_item_selected")
        return tr("asset_manager.status.multi_items_selected", count=count)

    @staticmethod
    def _sort_text(value: Any) -> str:
        return str(value or "").casefold()

    @staticmethod
    def _format_size(value: Any) -> str:
        try:
            size = max(0, int(value))
        except (TypeError, ValueError):
            size = 0
        for divisor, key in (
            (1024**3, "asset_manager.unit.gb"),
            (1024**2, "asset_manager.unit.mb"),
            (1024, "asset_manager.unit.kb"),
        ):
            if size >= divisor:
                return f"{size / divisor:.1f} {tr(key)}"
        return f"{size} {tr('asset_manager.unit.b')}"

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
        assets = getattr(self._asset_index, "assets", {}) if self._asset_index else {}
        return assets if isinstance(assets, dict) else {}

    def _asset_index_folders(self) -> Dict[str, Dict[str, Any]]:
        folders = getattr(self._asset_index, "folders", {}) if self._asset_index else {}
        return folders if isinstance(folders, dict) else {}

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

    def _repair_selection(self) -> None:
        assets = self._asset_index_assets()
        folders = self._asset_index_folders()
        self._selected_asset_ids.intersection_update(assets)
        if self._selection_cursor_id not in assets:
            self._selection_cursor_id = next(iter(self._selected_asset_ids), None)
        if self._selected_folder_id not in {*folders, SCOPE_ALL}:
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
            "AVAILABLE": "asset_manager.status.available",
            "MISSING": "asset_manager.status.missing",
            "IDENTITY_MISMATCH": "asset_manager.status.identity_mismatch",
        }.get(status, "asset_manager.status.unverified")
        return tr(key)

    def _get_asset_display_name(self, asset: Dict[str, Any]) -> str:
        path = str(asset.get("path") or "")
        path_stem = Path(path).stem if path else ""
        return str(asset.get("name") or path_stem or tr("asset_manager.unnamed"))

    @staticmethod
    def _thumbnail_decorator(asset: Dict[str, Any]) -> str:
        if not asset.get("has_preview") or not asset.get("exists"):
            return "none"
        path = quote(str(asset.get("path") or ""), safe=_RML_PATH_SAFE_CHARS)
        revision_value = asset.get("commit_uuid") or "-".join(
            str(asset.get(field) or 0)
            for field in ("generation", "saved_at_unix_ns", "file_size_bytes")
        )
        revision = quote(str(revision_value), safe="-._~")
        return f"image(preview://kind=licht&thumb=256&rev={revision}&path={path})"

    def _format_asset_for_ui(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        folder_name = self._folder_name(asset.get("folder_id"))
        asset_id = str(asset.get("id") or asset.get("project_uuid") or "")
        thumbnail_decorator = self._thumbnail_decorator(asset)
        thumbnail_source = (
            thumbnail_decorator[6:-1]
            if thumbnail_decorator.startswith("image(")
            else ""
        )
        previous_source = self._thumbnail_sources_by_asset.get(asset_id, "")
        if previous_source and previous_source != thumbnail_source:
            release_texture = getattr(lf.ui, "release_rml_texture", None)
            if callable(release_texture):
                release_texture(previous_source)
        if thumbnail_source:
            self._thumbnail_sources_by_asset[asset_id] = thumbnail_source
        else:
            self._thumbnail_sources_by_asset.pop(asset_id, None)
        return {
            **asset,
            "display_name": self._get_asset_display_name(asset),
            "id": asset_id,
            "folder_name": folder_name,
            "size_label": self._format_size(asset.get("file_size_bytes", 0)),
            "saved_label": self._format_unix_ns(asset.get("saved_at_unix_ns", 0)),
            "status_label": self._project_status_label(asset),
            "is_selected": str(asset.get("id") or asset.get("project_uuid"))
            in self._selected_asset_ids,
            "can_load": self._project_available(asset),
            "thumbnail_decorator": thumbnail_decorator,
        }

    def _release_obsolete_thumbnail_sources(self) -> None:
        live_ids = set(self._asset_index_assets())
        stale_ids = set(self._thumbnail_sources_by_asset).difference(live_ids)
        release_texture = getattr(lf.ui, "release_rml_texture", None)
        for asset_id in stale_ids:
            source = self._thumbnail_sources_by_asset.pop(asset_id)
            if callable(release_texture):
                release_texture(source)

    def _filtered_assets(self, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
        folder_id = self._selected_folder_id if folder_id is None else folder_id
        query = self._search_query.strip().casefold()
        rows: List[Dict[str, Any]] = []
        for asset in self._asset_index_assets().values():
            if folder_id not in (None, SCOPE_ALL) and asset.get("folder_id") != folder_id:
                continue
            if not self._asset_matches_query(asset, query):
                continue
            rows.append(asset)
        if self._sort_mode == "size":
            rows.sort(
                key=lambda asset: (
                    -int(asset.get("file_size_bytes") or 0),
                    self._sort_text(asset.get("name")),
                )
            )
        else:
            rows.sort(key=lambda asset: self._sort_text(asset.get("name") or asset.get("path")))
        self._last_asset_match_count = len(rows)
        return rows

    def _window_assets(self, assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        total = len(assets)
        scroll_top = self._asset_window_scroll_top
        client_height = self._asset_window_client_height
        if self._view_mode == "gallery":
            available_width = max(
                ASSET_CARD_PREFERRED_WIDTH_DP,
                self._asset_window_client_width - ASSET_CARD_GRID_HORIZONTAL_CHROME_DP,
            )
            columns = max(1, int(available_width // ASSET_CARD_PREFERRED_WIDTH_DP))
            self._asset_card_slot_width = max(1.0, available_width / columns)
            start_row = max(0, int(scroll_top // ASSET_GALLERY_ROW_HEIGHT_DP) - ASSET_WINDOW_OVERSCAN_ROWS)
            visible_rows = (
                math.ceil(client_height / ASSET_GALLERY_ROW_HEIGHT_DP)
                if client_height > 0
                else ASSET_GALLERY_FALLBACK_ROWS
            ) + ASSET_WINDOW_OVERSCAN_ROWS * 2
            start = min(total, start_row * columns)
            end = min(total, (start_row + visible_rows) * columns)
            total_rows = math.ceil(total / columns) if total else 0
            end_row = math.ceil(end / columns) if end else 0
            self._asset_gallery_top_spacer_height = start_row * ASSET_GALLERY_ROW_HEIGHT_DP
            self._asset_gallery_bottom_spacer_height = max(0, total_rows - end_row) * ASSET_GALLERY_ROW_HEIGHT_DP
            self._asset_list_top_spacer_height = 0.0
            self._asset_list_bottom_spacer_height = 0.0
        else:
            start = max(0, int(scroll_top // ASSET_LIST_ROW_HEIGHT_DP) - ASSET_WINDOW_OVERSCAN_ROWS)
            visible = (
                math.ceil(client_height / ASSET_LIST_ROW_HEIGHT_DP)
                if client_height > 0
                else ASSET_LIST_FALLBACK_ROWS
            ) + ASSET_WINDOW_OVERSCAN_ROWS * 2
            end = min(total, start + visible)
            self._asset_list_top_spacer_height = start * ASSET_LIST_ROW_HEIGHT_DP
            self._asset_list_bottom_spacer_height = max(0, total - end) * ASSET_LIST_ROW_HEIGHT_DP
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
                "name": str(folder.get("name") or tr("asset_manager.unnamed_folder")),
                "project_count": counts.get(folder_id, 0),
                "can_manage": True,
            }
            for folder_id, folder in self._asset_index_folders().items()
        ]
        return sorted(folder_rows, key=lambda row: self._sort_text(row["name"]))

    def get_all_assets_count(self) -> int:
        query = self._search_query.strip().casefold()
        return sum(
            self._asset_matches_query(asset, query)
            for asset in self._asset_index_assets().values()
        )

    def get_asset_results_summary(self) -> str:
        try:
            return localized_count(
                "asset_manager.status.showing_projects", self._last_asset_match_count
            )
        except Exception:
            return str(self._last_asset_match_count)

    def _get_selected_asset(self) -> Optional[Dict[str, Any]]:
        asset_id = self.get_selected_asset_id()
        return self._asset_index_assets().get(asset_id) if asset_id else None

    def get_selected_asset_name(self) -> str:
        asset = self._get_selected_asset()
        return self._get_asset_display_name(asset) if asset else ""

    def get_selected_asset_folder_name(self) -> str:
        asset = self._get_selected_asset()
        return self._folder_name(asset.get("folder_id")) if asset else ""

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

    def get_selected_asset_file_missing(self) -> bool:
        asset = self._get_selected_asset()
        return bool(asset) and not bool(asset.get("exists", False))

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
        self._dirty_fields("folders_collapsed", "folders_expanded")

    def set_view_mode(self, _handle, _ev, args):
        mode = str(args[0]) if args else ""
        if mode not in ("gallery", "list") or mode == self._view_mode:
            return
        self._view_mode = mode
        self._reset_scroll()
        self._refresh_records(assets=True)
        self._dirty_fields("is_gallery_view", "is_list_view")

    def cycle_sort_mode(self, _handle=None, _ev=None, _args=None):
        index = (self.SORT_MODES.index(self._sort_mode) + 1) % len(self.SORT_MODES)
        self._sort_mode = self.SORT_MODES[index]
        self._reset_scroll()
        self._refresh_records(assets=True)
        self._dirty_fields("sort_label")

    def _add_folder_from_path(self, directory: str) -> Optional[str]:
        if not self._asset_index or not directory.strip():
            return None
        folder = self._asset_index.add_folder(directory.strip())
        if folder is None:
            return None
        self._selected_folder_id = folder.id
        self._selected_asset_ids.clear()
        self._selection_cursor_id = None
        self._update_selection_type()
        self.refresh_catalog(scan_folders=False)
        return folder.id

    def add_asset_folder(self, _handle=None, _ev=None, _args=None):
        self.on_add_folder(None, None, None)

    def on_add_folder(self, _handle=None, _ev=None, _args=None):
        start = str(resolve_default_asset_directory())
        directory = lf.ui.open_folder_dialog(
            tr("asset_manager.dialog.select_folder"), start
        )
        if directory:
            self._add_folder_from_path(str(directory))

    def on_import_project(self, _handle=None, _ev=None, _args=None):
        if not self._asset_index:
            return
        path = lf.ui.open_project_file_dialog("")
        if not path:
            return
        if not is_supported_asset_path(path):
            self._log_warn("Asset Manager only supports .licht projects: %s", path)
            return
        try:
            project, _created = self._asset_index.register_licht_asset(
                path,
            )
            if project is not None:
                self._selected_asset_ids = {project.id}
                self._selection_cursor_id = project.id
                self._update_selection_type()
                self.refresh_catalog(scan_folders=False)
        except Exception as exc:
            self._log_error("Failed to import .licht project %s: %s", path, exc)

    def _select_folder_id(self, folder_id: str) -> bool:
        if folder_id not in {*self._asset_index_folders(), SCOPE_ALL}:
            return False
        self._selected_folder_id = folder_id
        self._selected_asset_ids.clear()
        self._selection_cursor_id = None
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
        return True

    def select_folder(self, _handle, _ev, args):
        self._select_folder_id(self._resolve_event_value(args, _ev, "data-folder-id"))

    def _select_asset_id(
        self,
        asset_id: str,
        *,
        multi_select: bool = False,
        row_element=None,
        container=None,
    ) -> bool:
        if asset_id not in self._asset_index_assets():
            return False
        if multi_select:
            if asset_id in self._selected_asset_ids:
                self._selected_asset_ids.remove(asset_id)
            else:
                self._selected_asset_ids.add(asset_id)
        else:
            self._selected_asset_ids = {asset_id}
        self._selection_cursor_id = (
            asset_id if asset_id in self._selected_asset_ids else next(iter(self._selected_asset_ids), None)
        )
        self._update_selection_type()
        self._sync_asset_selection_dom(container, row_element)
        self._dirty_selection()
        return True

    def toggle_asset_selection(self, _handle, _ev, args):
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        self._select_asset_id(asset_id, multi_select=self._event_multi_select(_ev))

    def _dirty_selection(self) -> None:
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
            "selected_asset_path",
            "selected_asset_size",
            "selected_asset_created",
            "selected_asset_modified",
            "selected_asset_file_missing",
            "selected_asset_expected_path",
        )

    def on_locate_file(self, _handle=None, _ev=None, args=None):
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id") or self.get_selected_asset_id()
        if not asset_id or not self._asset_index:
            return
        path = lf.ui.open_project_file_dialog("")
        if not path:
            return
        try:
            if self._asset_index.relink_asset(asset_id, path):
                self.refresh_catalog(scan_folders=False)
            else:
                self._log_warn("Selected file belongs to a different .licht project")
        except Exception as exc:
            self._log_error("Failed to relink .licht project: %s", exc)

    def on_load_asset(self, _handle, _ev, args):
        self._load_asset(self._resolve_event_value(args, _ev, "data-asset-id"))

    def _load_asset(self, asset_id: str) -> None:
        if not asset_id or not self._asset_index:
            return
        project = self._asset_index.verify_asset(asset_id)
        if project is None:
            return
        asset = project.to_dict() if hasattr(project, "to_dict") else self._asset_index_assets().get(asset_id, {})
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
        if asset_id and self._asset_index and self._asset_index.delete_asset(asset_id):
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
        if self._project_available(asset):
            items.append({"label": tr("menu.file.open_project"), "action": "load"})
        items.extend(
            [
                {"label": tr("asset_manager.action.rename"), "action": "rename"},
                {
                    "label": tr("asset_manager.action.show_in_folder"),
                    "action": "show_in_folder",
                    "separator_before": True,
                },
                {"label": tr("asset_manager.action.remove"), "action": "remove"},
            ]
        )
        return items

    def _handle_asset_context_action(self, action: str, asset_id: str) -> None:
        if action == "load":
            self._load_asset(asset_id)
        elif action == "rename":
            self.on_rename_asset(None, None, [asset_id])
        elif action == "show_in_folder":
            self.on_show_in_folder(None, None, [asset_id])
        elif action == "remove":
            self.on_remove_asset(None, None, [asset_id])

    def _show_asset_context_menu(self, asset_id: str) -> bool:
        asset = self._asset_index_assets().get(asset_id)
        return bool(asset) and self._show_shared_context_menu(
            self._asset_context_menu_items(asset),
            lambda action: self._handle_asset_context_action(action, asset_id),
        )

    def on_rename_asset(self, _handle, _ev, args):
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        asset = self._asset_index_assets().get(asset_id)
        if not asset or not self._asset_index:
            return
        current_name = str(asset.get("name") or Path(str(asset.get("path") or "")).stem)

        def rename(name: Any) -> None:
            value = str(name or "").strip()
            if value and value != current_name:
                self._asset_index.update_asset(asset_id, name=value)
                self.refresh_catalog(scan_folders=False)

        lf.ui.input_dialog(
            tr("asset_manager.dialog.rename_asset"),
            tr("asset_manager.dialog.enter_new_name", name=current_name),
            current_name,
            rename,
        )

    def on_show_in_folder(self, _handle, _ev, args):
        asset_id = self._resolve_event_value(args, _ev, "data-asset-id")
        asset = self._asset_index_assets().get(asset_id)
        if asset:
            reveal = getattr(lf.ui, "reveal_in_file_manager", None)
            if callable(reveal):
                reveal(str(asset.get("path") or ""))

    def _folder_context_menu_items(self, folder_id: str) -> List[Dict[str, Any]]:
        items = [{"label": tr("asset_manager.action.show_in_folder"), "action": "show"}]
        if folder_id == "default":
            items.append(
                {
                    "label": tr("asset_manager.action.settings"),
                    "action": "settings",
                    "separator_before": True,
                }
            )
        else:
            items.append(
                {
                    "label": tr("asset_manager.action.remove_folder"),
                    "action": "remove",
                    "separator_before": True,
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
        elif action == "settings":
            lf.ui.set_panel_enabled("lfs.preferences", True)
        elif action == "remove":
            self.on_delete_folder(None, None, [folder_id])

    def on_delete_folder(self, _handle, _ev, args):
        folder_id = self._resolve_event_value(args, _ev, "data-folder-id")
        folder = self._asset_index_folders().get(folder_id)
        if not folder_id or folder_id == "default" or not self._asset_index or not folder:
            return
        project_count = sum(
            asset.get("folder_id") == folder_id
            for asset in self._asset_index_assets().values()
        )
        delete_label = tr("asset_manager.action.remove_folder")

        def delete_confirmed(button: str) -> None:
            if button != delete_label:
                return
            if self._asset_index and self._asset_index.delete_folder(folder_id):
                self._selected_folder_id = SCOPE_ALL
                self._selected_asset_ids.clear()
                self._selection_cursor_id = None
                self.refresh_catalog(scan_folders=False)

        lf.ui.confirm_dialog(
            tr("asset_manager.dialog.remove_folder"),
            tr(
                "asset_manager.dialog.remove_folder_message",
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
        self._sync_default_folder_path()
        if self._asset_index:
            self._asset_index.verify_projects()
        self._repair_selection()
        self._refresh_records(assets=True, folders=True)
        if self._handle:
            self._handle.dirty_all()
        if request_update:
            self._request_model_update()
        if scan_folders:
            self._scan_asset_folders()

    def _scan_asset_folders(self) -> None:
        if not self._asset_index:
            return
        with self._folder_scan_lock:
            if self._folder_scan_active:
                return
            if not any(
                folder.get("path")
                for folder in self._asset_index_folders().values()
            ):
                return
            self._folder_scan_active = True
        index = self._asset_index

        def worker() -> None:
            try:
                result = scan_all_asset_folders(index)
                _log.info(
                    "Asset folder scan: discovered=%d added=%d existing=%d failed=%d",
                    result.discovered,
                    result.added,
                    result.already_cataloged,
                    result.failed,
                )
            except Exception:
                _log.exception("Asset Manager folder scan failed")
            finally:
                with self._folder_scan_lock:
                    self._folder_scan_active = False
                    self._folder_scan_refresh_pending = True
                scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
                if callable(scheduler):
                    scheduler(self._finish_folder_scan)

        threading.Thread(
            target=worker,
            daemon=True,
            name="AssetManagerFolderScan",
        ).start()

    def _finish_folder_scan(self) -> None:
        with self._folder_scan_lock:
            if not self._folder_scan_refresh_pending:
                return
            self._folder_scan_refresh_pending = False
        self.refresh_catalog(scan_folders=False)

    def _sync_default_folder_path(self) -> bool:
        if not self._asset_index:
            return False
        current = str(resolve_default_asset_directory())
        if current == self._last_default_folder_path:
            return False
        setter = getattr(self._asset_index, "set_default_folder_path", None)
        if not callable(setter):
            self._last_default_folder_path = current
            return False
        if not setter(current):
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
            self._handle.update_record_list("assets", self.get_filtered_assets())
            self._handle.dirty("assets")
            for field in (
                "asset_results_summary",
                "asset_list_top_spacer_height",
                "asset_list_bottom_spacer_height",
                "asset_gallery_top_spacer_height",
                "asset_gallery_bottom_spacer_height",
                "asset_card_slot_width",
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

    def _sync_asset_window_viewport(self, doc=None) -> bool:
        scroll = self._asset_scroll_container(doc)
        if not scroll:
            return False
        try:
            values = (
                max(0.0, float(scroll.scroll_top or 0.0)),
                max(0.0, float(scroll.client_height or 0.0)),
                max(0.0, float(getattr(scroll, "client_width", 0.0) or 0.0)),
            )
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
        available = max(
            ASSET_CARD_PREFERRED_WIDTH_DP,
            self._asset_window_client_width - ASSET_CARD_GRID_HORIZONTAL_CHROME_DP,
        )
        columns = max(1, int(available // ASSET_CARD_PREFERRED_WIDTH_DP))
        self._asset_card_slot_width = max(1.0, available / columns)
        return abs(old - self._asset_card_slot_width) > 0.5

    def _bind_dom_event_listeners(self, doc) -> None:
        shell = doc.get_element_by_id("asset-shell")
        if shell:
            shell.add_event_listener("mousedown", self._on_asset_manager_mousedown)
            shell.add_event_listener("click", self._on_asset_manager_click)
            shell.add_event_listener("dblclick", self._on_asset_manager_double_click)
            shell.add_event_listener("dragstart", self._on_asset_drag_start)
            shell.add_event_listener("dragend", self._on_asset_drag_end)
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
        new_top = min(max(float(scroll.scroll_top) + delta * PRECISE_SCROLL_STEP, 0.0), maximum)
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
            if action == "load":
                self._load_asset(asset_id)
            elif action == "menu":
                self._show_asset_context_menu(asset_id)
            elif action == "select":
                self._select_asset_id(
                    asset_id,
                    multi_select=self._event_multi_select(event),
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
        if button != 1:
            return
        container = event.current_target()
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
        verify_asset = getattr(self._asset_index, "verify_asset", None)
        project = verify_asset(asset_id) if callable(verify_asset) else None
        if callable(verify_asset) and project is None:
            self.refresh_catalog(scan_folders=False)
            return
        asset = (
            project.to_dict()
            if project is not None and hasattr(project, "to_dict")
            else self._asset_index_assets().get(asset_id, {})
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
        self._selected_asset_ids = {asset_id}
        self._selection_cursor_id = asset_id
        self._update_selection_type()
        self._sync_asset_selection_dom(container, element)
        self._dirty_selection()
        self._stop_event(event)

    def _on_asset_drag_end(self, event) -> None:
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
        available_width = max(
            ASSET_CARD_PREFERRED_WIDTH_DP,
            self._asset_window_client_width - ASSET_CARD_GRID_HORIZONTAL_CHROME_DP,
        )
        return max(1, int(available_width // ASSET_CARD_PREFERRED_WIDTH_DP))

    def _scroll_cursor_into_view(self, index: int) -> None:
        scroll = self._asset_scroll_container()
        if self._view_mode == "gallery":
            row = index // self._gallery_columns()
            start = row * ASSET_GALLERY_ROW_HEIGHT_DP
            end = start + ASSET_GALLERY_ROW_HEIGHT_DP
        else:
            start = index * ASSET_LIST_ROW_HEIGHT_DP
            end = start + ASSET_LIST_ROW_HEIGHT_DP
        top = self._asset_window_scroll_top
        height = self._asset_window_client_height
        if start < top:
            top = start
        elif height > 0 and end > top + height:
            top = max(0.0, end - height)
        self._asset_window_scroll_top = top
        if scroll is not None:
            scroll.scroll_top = top

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
        if self._asset_index.delete_assets(list(selected)) <= 0:
            return False
        self._selected_asset_ids.clear()
        self._selection_cursor_id = None
        remaining = self._filtered_assets()
        if remaining:
            cursor_index = min(cursor_index, len(remaining) - 1)
            asset_id = str(
                remaining[cursor_index].get("id")
                or remaining[cursor_index].get("project_uuid")
                or ""
            )
            self._selected_asset_ids = {asset_id}
            self._selection_cursor_id = asset_id
            self._scroll_cursor_into_view(cursor_index)
        self._update_selection_type()
        self._refresh_records(assets=True, folders=True)
        self._dirty_selection()
        return True

    def _on_asset_results_keydown(self, event) -> None:
        try:
            key = int(event.get_parameter("key_identifier", "0"))
        except (TypeError, ValueError):
            return
        if self._navigate_selection(key):
            self._stop_event(event)
            return
        if key == KI_RETURN:
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

    def on_sidebar_resize_start(self, _handle, event, _args):
        self._sidebar_dragging = True
        self._sidebar_drag_start_y = float(event.get_parameter("mouse_y", "0"))
        self._sidebar_start_height = self._sidebar_height
        self._dirty_fields("sidebar_resize_dragging")

    def on_bottom_panel_resize_start(self, _handle, event, _args):
        self._bottom_panel_dragging = True
        self._bottom_panel_drag_start_y = float(event.get_parameter("mouse_y", "0"))
        self._bottom_panel_start_height = self._bottom_panel_height
        self._dirty_fields("bottom_panel_resize_dragging")

    def _on_resize_mousemove(self, event) -> None:
        try:
            mouse_y = float(event.get_parameter("mouse_y", "0"))
        except (TypeError, ValueError):
            return
        if self._sidebar_dragging:
            self._sidebar_height = max(80.0, min(420.0, self._sidebar_start_height + mouse_y - self._sidebar_drag_start_y))
            self._dirty_fields("sidebar_height")
            self._stop_event(event)
        elif self._bottom_panel_dragging:
            self._bottom_panel_height = max(120.0, min(500.0, self._bottom_panel_start_height - mouse_y + self._bottom_panel_drag_start_y))
            self._dirty_fields("bottom_panel_height")
            self._stop_event(event)

    def _on_resize_mouseup(self, _event) -> None:
        if self._sidebar_dragging:
            self._sidebar_dragging = False
            self._dirty_fields("sidebar_resize_dragging")
        if self._bottom_panel_dragging:
            self._bottom_panel_dragging = False
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
            return False
        verify_asset = getattr(self._asset_index, "verify_asset", None)
        if not callable(verify_asset) or verify_asset(project.id) is None:
            return False
        self._refresh_records(assets=True, folders=True)
        if self._handle:
            self._handle.dirty_all()
        return True

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        if self._asset_index is None:
            self._initialize_backend()
        self._repair_selection()
        self._bind_dom_event_listeners(doc)
        self._subscribe_reactive_state()
        self._sync_panel_space_state()
        self._sync_asset_window_viewport(doc)
        self._refresh_records(assets=True, folders=True)
        if self._handle:
            self._handle.dirty_all()
        self._refresh_after_project_write()
        self._scan_asset_folders()

    def on_update(self, doc):
        changed = self._sync_default_folder_path()
        changed = self._refresh_after_project_write() or changed
        if self._sync_panel_space_state():
            self._dirty_fields("is_floating")
            changed = True
        if self._folder_scan_refresh_pending:
            self._finish_folder_scan()
            changed = True
        if self._asset_window_refresh_pending or self._sync_asset_window_viewport(doc):
            self._asset_window_refresh_pending = False
            self._refresh_records(assets=True)
            changed = True
        return changed

    def on_unmount(self, doc):
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
        self._unsubscribe_reactive_state()
        try:
            doc.remove_data_model("asset_manager")
        except Exception:
            pass
        self._handle = None
        self._doc = None

    def _on_close_panel(self, _handle=None, _event=None, _args=None):
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
