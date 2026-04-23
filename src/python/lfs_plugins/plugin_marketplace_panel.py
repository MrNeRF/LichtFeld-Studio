# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unified plugin marketplace floating panel."""

from html import escape
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import lichtfeld as lf

from . import rml_widgets as w
from .marketplace import (
    MarketplacePluginEntry,
    PluginMarketplaceCatalog,
)
from .plugin import PluginInfo, PluginState
from .types import Panel

__lfs_panel_classes__ = ["PluginMarketplacePanel"]
__lfs_panel_ids__ = ["lfs.plugin_marketplace"]

MAX_OUTPUT_LINES = 100
SUCCESS_DISMISS_SEC = 3.0
ERROR_DISMISS_SEC = 5.0
_CARD_GAP_DP = 12
_CARD_MIN_WIDTH_DP = 220
_GRID_SIDE_MARGIN_DP = 20
_SCROLLBAR_GUTTER_DP = 16

_PHASE_MILESTONES: List[Tuple[str, float]] = [
    ("cloning", 0.05),
    ("cloned", 0.30),
    ("downloading", 0.05),
    ("extracting", 0.35),
    ("syncing dependencies", 0.40),
    ("updating", 0.05),
    ("updated", 0.50),
    ("unloading", 0.20),
    ("uninstalling", 0.20),
]
_NUDGE_FRACTION = 0.08
_PROGRESS_CEILING = 0.95
_MAX_REPO_LABEL_CHARS = 30
_VIEW_MODE_CARD = "card"
_VIEW_MODE_LIST = "list"
_VALID_VIEW_MODES = frozenset({_VIEW_MODE_CARD, _VIEW_MODE_LIST})


class CardOpPhase(Enum):
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class CardOpState:
    phase: CardOpPhase = CardOpPhase.IDLE
    message: str = ""
    progress: float = 0.0
    output_lines: List[str] = field(default_factory=list)
    finished_at: float = 0.0


class PluginMarketplacePanel(Panel):
    """Floating plugin window for browsing, installing, and managing plugins."""

    id = "lfs.plugin_marketplace"
    label = "Plugin Marketplace"
    space = lf.ui.PanelSpace.FLOATING
    order = 91
    template = "rmlui/plugin_marketplace.rml"
    height_mode = lf.ui.PanelHeightMode.FILL
    size = (770, 560)
    update_interval_ms = 100

    def __init__(self):
        self._catalog = PluginMarketplaceCatalog()
        self._url_plugin_names: Dict[str, str] = {}
        self._manual_url = ""
        self._install_filter_idx = 0
        self._sort_idx = 2
        self._git_available = shutil.which("git") is not None
        self._git_checkout_selected: Dict[str, bool] = {}

        self._card_ops: Dict[str, CardOpState] = {}
        self._lock = threading.RLock()
        self._pending_uninstall_name = ""
        self._pending_uninstall_card_id = ""

        self._discover_cache: Optional[List[PluginInfo]] = None

        self._doc = None
        self._handle = None
        self._last_card_phases: Dict[str, Tuple] = {}
        self._entries_dirty = True
        self._needs_resort = True
        self._prev_snapshot_key: Optional[Tuple] = None
        self._cached_entries: List[MarketplacePluginEntry] = []
        self._cached_card_records: List[Dict[str, object]] = []
        self._cached_card_ids: List[str] = []
        self._cached_installed_lookup: Dict[str, str] = {}
        self._cached_installed_versions: Dict[str, str] = {}
        self._cached_installed_names: Set[str] = set()
        self._formats_open = False
        self._expanded_list_rows: Set[str] = set()
        self._collapsed_auto_list_rows: Set[str] = set()
        self._view_mode = _VIEW_MODE_LIST
        self._last_lang = ""
        self._last_grid_signature: Optional[Tuple] = None
        self._escape_revert = w.EscapeRevertController()

    # ── Data model ────────────────────────────────────────────

    def on_bind_model(self, ctx):
        import lichtfeld as lf

        model = ctx.create_data_model("plugin_marketplace")
        if model is None:
            return

        model.bind_func("panel_label", lambda: lf.ui.tr("menu.tools.plugin_marketplace"))

        model.bind(
            "manual_url",
            lambda: self._manual_url,
            lambda v: setattr(self, "_manual_url", v),
        )
        model.bind(
            "filter_idx",
            lambda: str(self._install_filter_idx),
            self._set_filter_idx,
        )
        model.bind(
            "sort_idx",
            lambda: str(self._sort_idx),
            self._set_sort_idx,
        )

        model.bind_event("do_install_url", self._on_manual_form_submit)
        model.bind_event("confirm_yes", self._on_confirm_yes)
        model.bind_event("confirm_no", self._on_confirm_no)
        model.bind_event("view_cards", self._on_view_cards)
        model.bind_event("view_list", self._on_view_list)

        self._handle = model.get_handle()

    def _set_filter_idx(self, v):
        try:
            idx = int(v)
        except (ValueError, TypeError):
            return
        if idx != self._install_filter_idx:
            self._install_filter_idx = idx
            self._entries_dirty = True
            self._needs_resort = True

    def _set_sort_idx(self, v):
        try:
            idx = int(v)
        except (ValueError, TypeError):
            return
        if idx != self._sort_idx:
            self._sort_idx = idx
            self._entries_dirty = True
            self._needs_resort = True

    # ── Lifecycle ─────────────────────────────────────────────

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        self._last_lang = lf.ui.get_current_language()
        self._entries_dirty = True
        self._last_card_phases.clear()
        self._last_grid_signature = None
        self._stable_layout_width = None
        self._escape_revert.clear()

        formats_header = doc.get_element_by_id("formats-header")
        if formats_header:
            formats_header.add_event_listener("click", self._on_toggle_formats)
            formats_content = doc.get_element_by_id("formats-content")
            formats_arrow = doc.get_element_by_id("formats-arrow")
            if formats_content:
                w.sync_section_state(formats_content, self._formats_open,
                                     formats_header, formats_arrow)

        grid_el = doc.get_element_by_id("card-grid")
        if grid_el:
            grid_el.add_event_listener("click", self._on_card_click)
            grid_el.add_event_listener("change", self._on_card_change)

        manual_form = doc.get_element_by_id("manual-install-form")
        if manual_form:
            manual_form.add_event_listener("submit", self._on_manual_form_submit)
            manual_form.add_event_listener("change", self._on_manual_form_change)

        manual_url_input = doc.get_element_by_id("manual-url-input")
        if manual_url_input:
            w.bind_select_all_on_focus(manual_url_input)
            self._escape_revert.bind(
                manual_url_input,
                "manual_url",
                lambda: str(self._manual_url or ""),
                self._restore_manual_url,
            )
        self._sync_view_mode_controls(doc)

    def on_update(self, doc):
        from .manager import PluginManager

        mgr = PluginManager.instance()
        self._ensure_loaded()

        current_lang = lf.ui.get_current_language()
        if current_lang != self._last_lang:
            self._last_lang = current_lang
            self._entries_dirty = True
            self._last_card_phases.clear()

        entries_raw, is_loading, registry_loaded = self._catalog.snapshot()
        self._update_catalog_status(doc, len(entries_raw), is_loading, registry_loaded)
        self._sync_view_mode_controls(doc)

        snapshot_key = (tuple(entries_raw), is_loading, registry_loaded)
        if snapshot_key != self._prev_snapshot_key:
            self._prev_snapshot_key = snapshot_key
            self._entries_dirty = True
            self._needs_resort = True

        if self._entries_dirty:
            self._entries_dirty = False
            entries = self._with_local_plugins(entries_raw, mgr)
            installed_lookup = self._get_installed_plugin_lookup(mgr)
            installed_versions = self._get_installed_plugin_versions(mgr)
            installed_names = set(installed_lookup.values())
            needs_resort = self._needs_resort
            self._needs_resort = False
            preserve = not needs_resort and bool(self._cached_card_ids)
            entries = self._filter_and_sort_entries(
                entries, set(installed_lookup.keys()), installed_names,
                preserve_order=preserve,
            )
            card_ids = [
                e.registry_id or e.name or str(i)
                for i, e in enumerate(entries)
            ]

            self._cached_entries = entries
            self._cached_card_records = records = [
                self._build_card_record(
                    entry, card_ids[i], mgr,
                    installed_lookup, installed_versions, installed_names,
                )
                for i, entry in enumerate(entries)
            ]
            self._cached_card_ids = card_ids
            self._cached_installed_lookup = installed_lookup
            self._cached_installed_versions = installed_versions
            self._cached_installed_names = installed_names
            self._last_card_phases.clear()
            self._render_entry_layout(doc, records, force=True)

            empty_el = doc.get_element_by_id("empty-state")
            grid_el = doc.get_element_by_id("card-grid")
            if empty_el:
                empty_el.set_class("hidden", len(entries) > 0)
            if grid_el:
                grid_el.set_class("hidden", len(entries) == 0)
        else:
            self._render_entry_layout(doc, self._cached_card_records)

        self._update_card_states(
            doc, self._cached_entries, self._cached_card_ids, mgr,
            self._cached_installed_lookup, self._cached_installed_versions,
            self._cached_installed_names,
        )
        self._update_manual_feedback(doc)

    # ── Card record building ──────────────────────────────────

    def _build_card_record(self, entry, card_id, mgr,
                           installed_lookup, installed_versions, installed_names):
        import lichtfeld as lf
        from .settings import SettingsManager

        tr = lf.ui.tr
        plugin_name = self._resolve_entry_plugin_name(entry, installed_lookup, installed_names)
        plugin_state = mgr.get_state(plugin_name) if plugin_name else None
        is_installed = plugin_name is not None
        is_local = self._is_local_entry(entry)
        is_local_only = self._is_local_only_entry(entry)
        has_github = bool(entry.github_url)
        card_state = self._get_card_state(card_id)
        buttons_busy = card_state.phase == CardOpPhase.IN_PROGRESS

        name = entry.name or entry.repo or tr("plugin_marketplace.unknown_plugin")

        repo_label = ""
        if entry.owner and entry.repo:
            repo_label = f"{entry.owner}/{entry.repo}"
        elif entry.repo:
            repo_label = entry.repo
        if repo_label:
            repo_label = self._truncate_text(repo_label, _MAX_REPO_LABEL_CHARS)

        desc = entry.description
        if not desc and plugin_name and self._discover_cache:
            for p in self._discover_cache:
                if p.name == plugin_name:
                    desc = p.description
                    break
        description = self._truncate_text(desc or tr("plugin_marketplace.no_description"), 90)

        version_label = ""
        has_version = False
        if plugin_name and plugin_state == PluginState.ACTIVE:
            version = installed_versions.get(plugin_name, "").strip()
            if version:
                version_label = version if version.lower().startswith("v") else f"v{version}"
                has_version = True

        metrics = []
        if not is_local_only:
            if entry.stars > 0:
                metrics.append(f"{tr('plugin_marketplace.stars')}: {entry.stars}")
            if entry.downloads > 0:
                metrics.append(f"{tr('plugin_marketplace.downloads')}: {entry.downloads}")
        metrics_text = "  |  ".join(metrics)

        tags = self._entry_type_tags(entry)
        tags_text = "  |  ".join(tags[:3])

        status_text = ""
        status_class = "status-muted"
        if is_installed:
            state_str = plugin_state.value if plugin_state else tr("plugin_manager.status_not_loaded")
            status_text = f"{tr('plugin_manager.status')}: {state_str}"
            if plugin_state == PluginState.ACTIVE:
                status_class = "status-success"

        is_remote_installed = is_installed and not is_local
        is_local_with_github = is_installed and is_local and has_github

        show_startup = (not buttons_busy) and is_installed and bool(plugin_name)
        startup_checked = False
        if show_startup:
            startup_checked = SettingsManager.instance().get(plugin_name).get("load_on_startup", False)

        return {
            "card_id": card_id,
            "name": name,
            "has_version": has_version,
            "version_label": version_label,
            "has_repo": bool(repo_label),
            "repo_label": repo_label,
            "has_metrics": bool(metrics_text),
            "metrics_text": metrics_text,
            "has_tags": bool(tags_text),
            "tags_text": tags_text,
            "is_local": is_local,
            "is_installed": is_installed,
            "status_text": status_text,
            "status_class": status_class,
            "has_error": bool(entry.error),
            "description": description,
            "info_action": "open-url" if has_github else "",
            "github_url": entry.github_url or "",
            "plugin_name": plugin_name or "",
            "show_install": (not buttons_busy) and not is_installed and not is_local_only and not entry.error,
            "show_git_checkout": (
                self._git_available
                and (not buttons_busy)
                and not is_installed
                and not is_local_only
                and not entry.error
            ),
            "git_checkout_selected": self._git_checkout_selected.get(card_id, False),
            "show_load": (not buttons_busy) and is_installed and plugin_state != PluginState.ACTIVE,
            "show_unload": (not buttons_busy) and is_installed and plugin_state == PluginState.ACTIVE,
            "show_reload": (not buttons_busy) and is_remote_installed and plugin_state == PluginState.ACTIVE,
            "show_update": (not buttons_busy) and (is_local_with_github or (is_remote_installed and plugin_state != PluginState.ACTIVE)),
            "show_uninstall": (not buttons_busy) and is_installed,
            "show_startup": show_startup,
            "startup_checked": startup_checked,
            "summary_status_text": status_text,
            "summary_marker": "●" if (is_local and not has_github) else "★",
            "summary_marker_class": "plugin-list-marker--local" if (is_local and not has_github) else "plugin-list-marker--remote",
        }

    def _render_entry_layout(self, doc, records: List[Dict[str, object]], force: bool = False):
        grid_el = doc.get_element_by_id("card-grid")
        if not grid_el:
            return

        grid_el.set_class("card-grid--list", self._view_mode == _VIEW_MODE_LIST)

        if self._view_mode == _VIEW_MODE_LIST:
            card_ids = tuple(str(r.get("card_id", "")) for r in records)
            signature = (self._view_mode, card_ids)
            if not force and signature == self._last_grid_signature:
                return
            self._last_grid_signature = signature
            grid_el.set_inner_rml("".join(self._build_list_markup(record) for record in records))
            return

        viewport_width = self._grid_viewport_width(doc, grid_el)
        layout_width = self._stabilize_layout_width(viewport_width)
        columns, row_width = self._compute_grid_layout(layout_width)
        card_ids = tuple(str(r.get("card_id", "")) for r in records)
        signature = (self._view_mode, card_ids, columns, row_width)
        if not force and signature == self._last_grid_signature:
            return
        self._last_grid_signature = signature

        if not records:
            grid_el.set_inner_rml("")
            return

        rows: List[str] = []
        for i in range(0, len(records), columns):
            chunk = list(records[i:i + columns])
            row_class = "card-row card-row--single" if columns == 1 else "card-row"
            row_parts = [
                f'<div class="{row_class}" style="width: {row_width}dp; margin-left: {_GRID_SIDE_MARGIN_DP}dp; margin-right: {_GRID_SIDE_MARGIN_DP}dp;">'
            ]
            for record in chunk:
                row_parts.append(self._build_card_markup(record))
            for _ in range(columns - len(chunk)):
                row_parts.append(
                    f'<div class="plugin-card plugin-card--placeholder" style="width: {_CARD_MIN_WIDTH_DP}dp;"></div>'
                )
            row_parts.append("</div>")
            rows.append("".join(row_parts))
        grid_el.set_inner_rml("".join(rows))

    def _grid_viewport_width(self, doc, grid_el) -> int:
        dp_ratio = max(1.0, lf.ui.get_ui_scale())

        main_area_el = doc.get_element_by_id("main-area")
        if main_area_el and getattr(main_area_el, "client_width", 0):
            return int(max(0.0, float(main_area_el.client_width or 0.0) / dp_ratio))

        content_el = doc.get_element_by_id("content")
        if content_el and getattr(content_el, "client_width", 0):
            return int(max(0.0, float(content_el.client_width or 0.0) / dp_ratio))

        return int(max(0.0, float(grid_el.client_width or 0.0) / dp_ratio))

    def _stabilize_layout_width(self, width: int) -> int:
        return max(0, width - _SCROLLBAR_GUTTER_DP)

    def _compute_grid_layout(self, width: int) -> Tuple[int, int]:
        if width <= 0:
            return 1, _CARD_MIN_WIDTH_DP

        usable_width = max(0, width - (2 * _GRID_SIDE_MARGIN_DP))
        if usable_width <= 0:
            return 1, _CARD_MIN_WIDTH_DP

        columns = max(
            1,
            (usable_width + _CARD_GAP_DP) // (_CARD_MIN_WIDTH_DP + _CARD_GAP_DP),
        )
        while columns > 1 and self._min_row_width(columns) > usable_width:
            columns -= 1

        row_width = max(self._min_row_width(columns), usable_width)
        return columns, row_width

    @staticmethod
    def _min_row_width(columns: int) -> int:
        return (columns * _CARD_MIN_WIDTH_DP) + ((columns - 1) * _CARD_GAP_DP)

    def _build_card_inner_markup(self, record: Dict[str, object]) -> str:
        tr = lf.ui.tr

        def esc(key: str) -> str:
            return escape(str(record.get(key, "")), quote=True)

        def text_span(condition_key: str, body: str) -> str:
            return body if record.get(condition_key) else ""

        info_attrs = []
        if record.get("info_action"):
            info_attrs.append(f'data-action="{esc("info_action")}"')
        if record.get("github_url"):
            info_attrs.append(f'data-url="{esc("github_url")}"')
        info_attr_text = (" " + " ".join(info_attrs)) if info_attrs else ""

        status_span = ""
        if record.get("is_installed"):
            status_span = (
                f'<span class="card-status {esc("status_class")}">{esc("status_text")}</span>'
            )

        invalid_link = (
            f'<span class="card-error status-error">{escape(tr("plugin_marketplace.invalid_link"))}</span>'
            if record.get("has_error")
            else ""
        )
        description = (
            ""
            if record.get("has_error")
            else f'<span class="card-description text-disabled">{esc("description")}</span>'
        )
        version_span = text_span(
            "has_version",
            f'<span class="card-version status-info">{esc("version_label")}</span>',
        )
        repo_span = text_span(
            "has_repo",
            f'<span class="card-repo text-disabled">{esc("repo_label")}</span>',
        )
        metrics_span = text_span(
            "has_metrics",
            f'<span class="card-metrics mp-warning-text">{esc("metrics_text")}</span>',
        )
        tags_span = text_span(
            "has_tags",
            f'<span class="card-tags text-disabled">{esc("tags_text")}</span>',
        )
        local_span = (
            f'<span class="card-local status-info">{escape(tr("plugin_marketplace.local_install"))}</span>'
            if record.get("is_local")
            else ""
        )

        return (
            f'<div class="card-info"{info_attr_text}>'
            f'<span class="card-name">{esc("name")}</span>'
            f'{version_span}'
            f'{repo_span}'
            f'{metrics_span}'
            f'{tags_span}'
            f'{local_span}'
            f'{status_span}'
            f'{invalid_link}'
            f'{description}'
            '<div class="separator"></div>'
            '</div>'
            f'{self._build_feedback_markup(record)}'
            f'{self._build_git_row(record)}'
            f'{self._build_startup_row(record)}'
            f'<div class="card-buttons" id="btns-{esc("card_id")}">{self._build_action_buttons_markup(record)}</div>'
        )

    def _build_card_markup(self, record: Dict[str, object]) -> str:
        def esc(key: str) -> str:
            return escape(str(record.get(key, "")), quote=True)

        return (
            f'<div class="plugin-card" id="card-{esc("card_id")}" data-card-id="{esc("card_id")}" '
            f'style="width: {_CARD_MIN_WIDTH_DP}dp;">'
            f'{self._build_card_inner_markup(record)}'
            '</div>'
        )

    def _build_list_inner_markup(self, record: Dict[str, object]) -> str:
        tr = lf.ui.tr

        def esc(key: str) -> str:
            return escape(str(record.get(key, "")), quote=True)

        card_id = str(record.get("card_id", ""))
        expanded = self._is_list_row_expanded(card_id)
        disclosure = "▼" if expanded else "▶"
        expanded_class = "" if expanded else " hidden"

        summary_status = (
            f'<span class="plugin-list-summary-status {esc("status_class")}">{esc("summary_status_text")}</span>'
            if record.get("summary_status_text")
            else ""
        )
        version_span = (
            f'<span class="plugin-list-version status-info">{esc("version_label")}</span>'
            if record.get("has_version")
            else ""
        )

        detail_meta_parts: List[str] = []
        if record.get("has_repo"):
            if record.get("github_url"):
                detail_meta_parts.append(
                    f'<button type="button" class="plugin-list-link" data-action="open-url" data-url="{esc("github_url")}">{esc("repo_label")}</button>'
                )
            else:
                detail_meta_parts.append(f'<span class="plugin-list-meta-item text-disabled">{esc("repo_label")}</span>')
        if record.get("has_metrics"):
            detail_meta_parts.append(f'<span class="plugin-list-meta-item mp-warning-text">{esc("metrics_text")}</span>')
        if record.get("has_tags"):
            detail_meta_parts.append(f'<span class="plugin-list-meta-item text-disabled">{esc("tags_text")}</span>')
        if record.get("is_local"):
            detail_meta_parts.append(
                f'<span class="plugin-list-meta-item status-info">{escape(tr("plugin_marketplace.local_install"))}</span>'
            )
        detail_meta = (
            f'<div class="plugin-list-detail-meta">{"".join(detail_meta_parts)}</div>'
            if detail_meta_parts
            else ""
        )

        detail_text = (
            f'<span class="plugin-list-error status-error">{escape(tr("plugin_marketplace.invalid_link"))}</span>'
            if record.get("has_error")
            else f'<span class="plugin-list-description text-disabled">{esc("description")}</span>'
        )

        return (
            '<div class="plugin-list-summary">'
            f'<button type="button" class="plugin-list-disclosure" data-action="toggle-expand" data-card-id="{esc("card_id")}">{disclosure}</button>'
            f'<button type="button" class="plugin-list-name-btn" data-action="toggle-expand" data-card-id="{esc("card_id")}">'
            f'<span class="plugin-list-name">{esc("name")}</span>'
            f'{version_span}'
            '</button>'
            '<div class="plugin-list-summary-right">'
            f'{summary_status}'
            f'<span class="plugin-list-marker {esc("summary_marker_class")}">{esc("summary_marker")}</span>'
            '</div>'
            '</div>'
            f'<div class="plugin-list-details{expanded_class}">'
            f'{detail_meta}'
            f'{detail_text}'
            f'{self._build_feedback_markup(record, extra_class="plugin-list-feedback")}'
            f'<div class="plugin-list-buttons" id="btns-{esc("card_id")}">{self._build_action_buttons_markup(record)}</div>'
            '</div>'
        )

    def _build_list_markup(self, record: Dict[str, object]) -> str:
        def esc(key: str) -> str:
            return escape(str(record.get(key, "")), quote=True)

        return (
            f'<div class="plugin-list-row" id="card-{esc("card_id")}" data-card-id="{esc("card_id")}">'
            f'{self._build_list_inner_markup(record)}'
            '</div>'
        )

    def _build_action_buttons_markup(self, record: Dict[str, object]) -> str:
        tr = lf.ui.tr

        def esc(key: str) -> str:
            return escape(str(record.get(key, "")), quote=True)

        buttons: List[str] = []
        if record.get("show_install"):
            buttons.append(
                f'<button class="btn btn--success" data-action="install" data-card-id="{esc("card_id")}">'
                f'{escape(tr("plugin_marketplace.button.install"))}</button>'
            )
        if record.get("show_load"):
            buttons.append(
                f'<button class="btn btn--success" data-action="load" data-card-id="{esc("card_id")}" '
                f'data-plugin="{esc("plugin_name")}">{escape(tr("plugin_manager.button.load"))}</button>'
            )
        if record.get("show_unload"):
            buttons.append(
                f'<button class="btn btn--warning" data-action="unload" data-card-id="{esc("card_id")}" '
                f'data-plugin="{esc("plugin_name")}">{escape(tr("plugin_manager.button.unload"))}</button>'
            )
        if record.get("show_reload"):
            buttons.append(
                f'<button class="btn btn--primary" data-action="reload" data-card-id="{esc("card_id")}" '
                f'data-plugin="{esc("plugin_name")}">{escape(tr("plugin_manager.button.reload"))}</button>'
            )
        if record.get("show_update"):
            buttons.append(
                f'<button class="btn btn--primary" data-action="update" data-card-id="{esc("card_id")}" '
                f'data-plugin="{esc("plugin_name")}">{escape(tr("plugin_manager.button.update"))}</button>'
            )
        if record.get("show_uninstall"):
            buttons.append(
                f'<button class="btn btn--error" data-action="uninstall" data-card-id="{esc("card_id")}" '
                f'data-plugin="{esc("plugin_name")}">{escape(tr("plugin_manager.button.uninstall"))}</button>'
            )
        return "".join(buttons)

    def _build_feedback_markup(self, record: Dict[str, object], extra_class: str = "") -> str:
        card_id = escape(str(record.get("card_id", "")), quote=True)
        class_suffix = f" {extra_class}" if extra_class else ""
        return (
            f'<div class="card-feedback hidden{class_suffix}" id="feedback-{card_id}">'
            f'<progress class="card-progress hidden" id="feedback-{card_id}-progress" max="1" value="0"></progress>'
            f'<span class="card-progress-text hidden" id="feedback-{card_id}-progress-text"></span>'
            f'<span class="status-text status-success hidden" id="feedback-{card_id}-success"></span>'
            f'<span class="status-text status-error hidden" id="feedback-{card_id}-error"></span>'
            '</div>'
        )

    def _build_startup_row(
        self,
        record: Dict[str, object],
        row_class: str = "card-startup-row",
        label_class: str = "card-startup-label text-disabled",
    ) -> str:
        if not record.get("show_startup"):
            return ""

        tr = lf.ui.tr

        def esc(key: str) -> str:
            return escape(str(record.get(key, "")), quote=True)

        checked = ' checked="checked"' if record.get("startup_checked") else ""
        return (
            f'<div class="{row_class}"><label>'
            f'<input type="checkbox"{checked} data-action="startup" '
            f'data-card-id="{esc("card_id")}" data-plugin="{esc("plugin_name")}" />'
            f'<span class="{label_class}">{escape(tr("plugin_marketplace.load_on_startup"))}</span>'
            '</label></div>'
        )

    def _build_git_row(
        self,
        record: Dict[str, object],
        row_class: str = "card-git-row",
        label_class: str = "card-startup-label text-disabled",
    ) -> str:
        if not record.get("show_git_checkout"):
            return ""

        tr = lf.ui.tr
        checked = ' checked="checked"' if record.get("git_checkout_selected") else ""
        card_id = escape(str(record.get("card_id", "")), quote=True)
        return (
            f'<div class="{row_class}"><label>'
            f'<input type="checkbox"{checked} data-action="git-checkout" '
            f'data-card-id="{card_id}" />'
            f'<span class="{label_class}">{escape(tr("plugin_marketplace.install_as_git_checkout"))}</span>'
            '</label></div>'
        )

    # ── Card state updates (per-frame, minimal DOM touches) ───

    def _update_card_states(self, doc, entries, card_ids, mgr,
                            installed_lookup, installed_versions, installed_names):
        import lichtfeld as lf

        tr = lf.ui.tr

        for i, card_id in enumerate(card_ids):
            state = self._get_card_state(card_id)
            phase_key = (state.phase, state.message, round(state.progress, 2))
            prev_key = self._last_card_phases.get(card_id)
            if prev_key == phase_key:
                continue
            self._last_card_phases[card_id] = phase_key

            prev_phase = prev_key[0] if prev_key else CardOpPhase.IDLE
            if state.phase != prev_phase:
                self._refresh_card_record(doc, card_id, mgr)

            card_el = doc.get_element_by_id(f"card-{card_id}")
            if not card_el:
                continue

            card_el.set_class("card--in-progress", state.phase == CardOpPhase.IN_PROGRESS)
            card_el.set_class("card--success", state.phase == CardOpPhase.SUCCESS)
            card_el.set_class("card--error", state.phase == CardOpPhase.ERROR)
            self._sync_feedback_state(doc, f"feedback-{card_id}", state, tr("plugin_manager.working"))

    def _refresh_card_record(self, doc, card_id: str, mgr):
        try:
            idx = self._cached_card_ids.index(card_id)
        except ValueError:
            return

        entry = self._cached_entries[idx]
        installed_lookup = self._get_installed_plugin_lookup(mgr)
        installed_versions = self._get_installed_plugin_versions(mgr)
        installed_names = set(installed_lookup.values())

        self._cached_installed_lookup = installed_lookup
        self._cached_installed_versions = installed_versions
        self._cached_installed_names = installed_names

        record = self._build_card_record(
            entry,
            card_id,
            mgr,
            installed_lookup,
            installed_versions,
            installed_names,
        )
        self._cached_card_records[idx] = record

        card_el = doc.get_element_by_id(f"card-{card_id}")
        if not card_el:
            return

        if self._view_mode == _VIEW_MODE_LIST:
            card_el.set_inner_rml(self._build_list_inner_markup(record))
        else:
            card_el.set_inner_rml(self._build_card_inner_markup(record))

    def _update_manual_feedback(self, doc):
        card_id = "__manual_url__"
        state = self._get_card_state(card_id)
        feedback_el = doc.get_element_by_id("manual-feedback")
        if not feedback_el:
            return

        import lichtfeld as lf
        tr = lf.ui.tr

        phase_key = (state.phase, state.message, round(state.progress, 2))
        cache_key = "_manual_feedback_"
        if self._last_card_phases.get(cache_key) == phase_key:
            return
        self._last_card_phases[cache_key] = phase_key

        btn = doc.get_element_by_id("btn-install-url")

        self._sync_feedback_state(doc, "manual-feedback", state, tr("plugin_manager.working"))

        if btn:
            if state.phase == CardOpPhase.IN_PROGRESS:
                btn.set_attribute("disabled", "disabled")
            else:
                btn.remove_attribute("disabled")

        if state.phase == CardOpPhase.SUCCESS:
            self._manual_url = ""
            if self._handle:
                self._handle.dirty("manual_url")

    def _update_catalog_status(self, doc, entry_count: int, is_loading: bool, registry_loaded: bool):
        status_el = doc.get_element_by_id("catalog-status")
        if not status_el:
            return

        if is_loading and entry_count == 0:
            text = "Fetching plugin registry..."
            tone = "status-info"
        elif registry_loaded:
            noun = "plugin" if entry_count == 1 else "plugins"
            text = f"Registry loaded: {entry_count} {noun} in the marketplace catalog."
            tone = "status-success" if entry_count > 0 else "status-info"
        else:
            noun = "plugin" if entry_count == 1 else "plugins"
            text = f"Registry unavailable: showing {entry_count} fallback {noun}."
            tone = "status-warning"

        status_el.set_text(text)
        status_el.set_class("status-info", tone == "status-info")
        status_el.set_class("status-success", tone == "status-success")
        status_el.set_class("status-warning", tone == "status-warning")

    def _sync_feedback_state(self, doc, element_prefix: str, state: CardOpState, working_text: str):
        feedback_el = doc.get_element_by_id(element_prefix)
        if not feedback_el:
            return

        show_progress = state.phase == CardOpPhase.IN_PROGRESS
        show_success = state.phase == CardOpPhase.SUCCESS
        show_error = state.phase == CardOpPhase.ERROR

        progress_el = doc.get_element_by_id(f"{element_prefix}-progress")
        progress_text_el = doc.get_element_by_id(f"{element_prefix}-progress-text")
        success_el = doc.get_element_by_id(f"{element_prefix}-success")
        error_el = doc.get_element_by_id(f"{element_prefix}-error")

        feedback_el.set_class("hidden", not (show_progress or show_success or show_error))

        if progress_el:
            progress_el.set_class("hidden", not show_progress)
            if show_progress:
                progress_el.set_attribute("value", f"{state.progress:.2f}")
        if progress_text_el:
            progress_text_el.set_class("hidden", not show_progress)
            progress_text_el.set_text(state.message or working_text if show_progress else "")
        if success_el:
            success_el.set_class("hidden", not show_success)
            success_el.set_text(state.message if show_success else "")
        if error_el:
            error_el.set_class("hidden", not show_error)
            error_el.set_text(state.message if show_error else "")

    # ── Event handlers ────────────────────────────────────────

    def _on_toggle_formats(self, _ev):
        self._formats_open = not self._formats_open
        doc = self._doc
        header = doc.get_element_by_id("formats-header")
        content = doc.get_element_by_id("formats-content")
        arrow = doc.get_element_by_id("formats-arrow")
        if content:
            w.animate_section_toggle(content, self._formats_open, arrow,
                                     header_element=header)

    def _on_manual_form_submit(self, ev_or_handle=None, _ev=None, _args=None):
        from .manager import PluginManager
        mgr = PluginManager.instance()
        self._install_plugin_from_url(mgr, self._manual_url, "__manual_url__")
        if _ev is None and ev_or_handle is not None and hasattr(ev_or_handle, "stop_propagation"):
            ev_or_handle.stop_propagation()

    def _restore_manual_url(self, snapshot):
        self._manual_url = str(snapshot or "")
        if self._handle:
            self._handle.dirty("manual_url")

    def _on_manual_form_change(self, ev):
        target = ev.target()
        if target is None or not ev.get_bool_parameter("linebreak", False):
            return
        if target.get_attribute("id", "") != "manual-url-input":
            return

        form = ev.current_target()
        if form is not None:
            form.submit()
            ev.stop_propagation()

    def _on_confirm_yes(self, handle, event, args):
        from .manager import PluginManager
        mgr = PluginManager.instance()
        name = self._pending_uninstall_name
        card_id = self._pending_uninstall_card_id
        self._pending_uninstall_name = ""
        self._pending_uninstall_card_id = ""
        overlay = self._doc.get_element_by_id("confirm-overlay")
        if overlay:
            overlay.set_class("hidden", True)
        if name:
            self._uninstall_plugin(mgr, name, card_id)

    def _on_confirm_no(self, handle, event, args):
        self._pending_uninstall_name = ""
        self._pending_uninstall_card_id = ""
        overlay = self._doc.get_element_by_id("confirm-overlay")
        if overlay:
            overlay.set_class("hidden", True)

    def _on_view_cards(self, *_args):
        self._set_view_mode(_VIEW_MODE_CARD)

    def _on_view_list(self, *_args):
        self._set_view_mode(_VIEW_MODE_LIST)

    def _on_card_click(self, ev):
        import lichtfeld as lf
        from .manager import PluginManager

        target = ev.target()
        if target is None:
            return

        action, card_id, plugin_name = self._find_card_action(target)
        if not action:
            return

        mgr = PluginManager.instance()

        if action == "open-url":
            url = self._find_data_attr(target, "data-url")
            if url:
                lf.ui.open_url(url)
            return

        if action == "toggle-expand" and card_id:
            self._toggle_list_row(card_id)
            return

        if action == "startup":
            return

        if not card_id:
            return

        entry = None
        for i, e in enumerate(self._cached_entries):
            eid = e.registry_id or e.name or str(i)
            if eid == card_id:
                entry = e
                break

        if action == "install" and entry:
            self._install_plugin_from_marketplace(mgr, entry, card_id)
        elif action == "load" and plugin_name:
            self._load_plugin(mgr, plugin_name, card_id)
        elif action == "unload" and plugin_name:
            self._unload_plugin(mgr, plugin_name, card_id)
        elif action == "reload" and plugin_name:
            self._reload_plugin(mgr, plugin_name, card_id)
        elif action == "update" and plugin_name:
            self._update_plugin(mgr, plugin_name, card_id)
        elif action == "uninstall" and plugin_name:
            self._request_uninstall_confirmation(plugin_name, card_id, ev)

    def _on_card_change(self, ev):
        target = ev.target()
        if target is None:
            return

        action, card_id, plugin_name = self._find_card_action(target)
        if action == "startup" and plugin_name:
            self._set_startup_preference(target, plugin_name)
            return
        if action == "git-checkout" and card_id:
            self._set_git_checkout_preference(target, card_id)

    def _find_card_action(self, element):
        while element is not None:
            action = element.get_attribute("data-action")
            if action:
                card_id = element.get_attribute("data-card-id", "")
                plugin_name = element.get_attribute("data-plugin", "")
                return action, card_id, plugin_name or None
            element = element.parent()
        return None, None, None

    def _find_element_with_attr(self, element, attr, value):
        while element is not None:
            if element.get_attribute(attr, "") == value:
                return element
            element = element.parent()
        return None

    def _find_data_attr(self, element, attr):
        while element is not None:
            val = element.get_attribute(attr, "")
            if val:
                return val
            element = element.parent()
        return None

    def _set_startup_preference(self, element, plugin_name: str):
        from .settings import SettingsManager

        cb_el = self._find_element_with_attr(element, "type", "checkbox")
        checked = cb_el.has_attribute("checked") if cb_el else False
        prefs = SettingsManager.instance().get(plugin_name)
        if prefs.get("load_on_startup", False) == checked:
            return

        prefs.set("load_on_startup", checked)
        self._entries_dirty = True

    def _set_git_checkout_preference(self, element, card_id: str):
        cb_el = self._find_element_with_attr(element, "type", "checkbox")
        checked = cb_el.has_attribute("checked") if cb_el else False
        if self._git_checkout_selected.get(card_id, False) == checked:
            return
        self._git_checkout_selected[card_id] = checked

    def _set_view_mode(self, view_mode: str):
        if view_mode not in _VALID_VIEW_MODES or view_mode == self._view_mode:
            return
        self._view_mode = view_mode
        self._entries_dirty = True
        self._last_grid_signature = None
        if self._doc:
            self._sync_view_mode_controls(self._doc)

    def _sync_view_mode_controls(self, doc):
        tr = lf.ui.tr
        cards_btn = doc.get_element_by_id("view-cards-btn")
        list_btn = doc.get_element_by_id("view-list-btn")
        if cards_btn:
            cards_btn.set_class("selected", self._view_mode == _VIEW_MODE_CARD)
            cards_btn.set_attribute("title", tr("plugin_marketplace.view.grid"))
        if list_btn:
            list_btn.set_class("selected", self._view_mode == _VIEW_MODE_LIST)
            list_btn.set_attribute("title", tr("plugin_marketplace.view.list"))

    def _is_list_row_expanded(self, card_id: str) -> bool:
        if card_id in self._expanded_list_rows:
            return True
        state = self._get_card_state(card_id)
        auto_expanded = state.phase in (CardOpPhase.IN_PROGRESS, CardOpPhase.ERROR)
        if not auto_expanded:
            self._collapsed_auto_list_rows.discard(card_id)
            return False
        return card_id not in self._collapsed_auto_list_rows

    def _set_list_row_expanded(self, card_id: str, expanded: bool, rerender: bool = True):
        state = self._get_card_state(card_id)
        auto_expanded = state.phase in (CardOpPhase.IN_PROGRESS, CardOpPhase.ERROR)
        if expanded:
            already_expanded = (
                card_id in self._expanded_list_rows
                or (auto_expanded and card_id not in self._collapsed_auto_list_rows)
            )
            self._collapsed_auto_list_rows.discard(card_id)
            if already_expanded:
                return
            self._expanded_list_rows.add(card_id)
        else:
            changed = False
            if card_id in self._expanded_list_rows:
                self._expanded_list_rows.discard(card_id)
                changed = True
            if auto_expanded and card_id not in self._collapsed_auto_list_rows:
                self._collapsed_auto_list_rows.add(card_id)
                changed = True
            if not changed:
                return

        self._last_grid_signature = None
        if rerender and self._doc and self._view_mode == _VIEW_MODE_LIST:
            self._render_entry_layout(self._doc, self._cached_card_records, force=True)

    def _toggle_list_row(self, card_id: str):
        self._set_list_row_expanded(card_id, not self._is_list_row_expanded(card_id), rerender=True)

    def _selected_install_transport(self, card_id: str) -> str:
        if self._git_available and self._git_checkout_selected.get(card_id, False):
            return "git"
        return "archive"

    def _request_uninstall_confirmation(self, name, card_id, ev):
        import lichtfeld as lf

        if not name:
            return
        self._pending_uninstall_name = name
        self._pending_uninstall_card_id = card_id

        tr = lf.ui.tr
        doc = self._doc

        msg_el = doc.get_element_by_id("confirm-message")
        if msg_el:
            msg_el.set_text(tr("plugin_marketplace.confirm_uninstall_message").format(name=name))
        overlay = doc.get_element_by_id("confirm-overlay")
        if overlay:
            overlay.set_class("hidden", False)

    # ── Business logic (unchanged) ────────────────────────────

    def _ensure_loaded(self):
        # Alphabetical sorting only needs registry metadata. Popularity sorting depends on
        # GitHub enrichment for curated entries, so defer that extra fetch until requested.
        self._catalog.refresh_async(require_github_enrichment=self._sort_idx in (0, 1))

    def _invalidate_discover_cache(self):
        self._discover_cache = None
        self._entries_dirty = True

    def _get_discovered_plugins(self, mgr) -> List[PluginInfo]:
        cache = self._discover_cache
        if cache is None:
            cache = mgr.discover()
            self._discover_cache = cache
        return cache

    def _get_card_state(self, card_id: str) -> CardOpState:
        with self._lock:
            state = self._card_ops.get(card_id)
            if state is None:
                return CardOpState()
            if state.phase in (CardOpPhase.SUCCESS, CardOpPhase.ERROR) and state.finished_at > 0:
                dismiss_after = (
                    SUCCESS_DISMISS_SEC
                    if state.phase == CardOpPhase.SUCCESS
                    else ERROR_DISMISS_SEC
                )
                if time.monotonic() - state.finished_at >= dismiss_after:
                    state.phase = CardOpPhase.IDLE
                    state.message = ""
                    state.progress = 0.0
                    state.output_lines.clear()
                    state.finished_at = 0.0
            return CardOpState(
                phase=state.phase,
                message=state.message,
                progress=state.progress,
                output_lines=list(state.output_lines),
                finished_at=state.finished_at,
            )

    def _filter_and_sort_entries(
        self,
        entries: List[MarketplacePluginEntry],
        installed_keys: Set[str],
        installed_names: Set[str],
        preserve_order: bool = False,
    ) -> List[MarketplacePluginEntry]:
        filtered = []
        for entry in entries:
            is_installed = self._is_marketplace_entry_installed(entry, installed_keys, installed_names)
            if self._install_filter_idx == 1 and not is_installed:
                continue
            if self._install_filter_idx == 2 and is_installed:
                continue
            filtered.append(entry)

        if preserve_order:
            return self._stable_merge(filtered)
        return self._sort_entries(filtered)

    def _sort_entries(self, entries: List[MarketplacePluginEntry]) -> List[MarketplacePluginEntry]:
        def popularity(e):
            return (e.stars + e.downloads, e.name.lower())

        if self._sort_idx == 1:
            return sorted(entries, key=popularity)
        if self._sort_idx == 2:
            return sorted(entries, key=lambda e: e.name.lower())
        if self._sort_idx == 3:
            return sorted(entries, key=lambda e: e.name.lower(), reverse=True)
        return sorted(entries, key=popularity, reverse=True)

    @staticmethod
    def _entry_key(entry: MarketplacePluginEntry) -> str:
        if entry.owner and entry.repo:
            return entry.owner.lower() + "/" + entry.repo.lower()
        return entry.registry_id or entry.name or entry.source_url

    def _stable_merge(self, entries: List[MarketplacePluginEntry]) -> List[MarketplacePluginEntry]:
        """Keep existing card order, append new entries sorted at the end."""
        prev_order = {self._entry_key(e): i for i, e in enumerate(self._cached_entries)}
        existing = []
        new_entries = []
        for e in entries:
            idx = prev_order.get(self._entry_key(e))
            if idx is not None:
                existing.append((idx, e))
            else:
                new_entries.append(e)

        existing.sort(key=lambda t: t[0])
        return [e for _, e in existing] + self._sort_entries(new_entries)

    @staticmethod
    def _advance_progress(state: CardOpState, msg: str):
        lower = msg.lower()
        for keyword, milestone in _PHASE_MILESTONES:
            if keyword in lower:
                state.progress = max(state.progress, milestone)
                return
        remaining = _PROGRESS_CEILING - state.progress
        if remaining > 0.01:
            state.progress += remaining * _NUDGE_FRACTION

    def _run_async(self, card_id: str, operation, success_msg: str, error_prefix: str):
        with self._lock:
            existing = self._card_ops.get(card_id)
            if existing and existing.phase == CardOpPhase.IN_PROGRESS:
                return
            state = CardOpState(phase=CardOpPhase.IN_PROGRESS)
            self._card_ops[card_id] = state

        def on_progress(msg: str):
            with self._lock:
                self._advance_progress(state, msg)
                state.message = msg
                state.output_lines.append(msg)
                if len(state.output_lines) > MAX_OUTPUT_LINES:
                    state.output_lines = state.output_lines[-MAX_OUTPUT_LINES:]

        def worker():
            try:
                result = operation(on_progress)
                if result is False:
                    raise RuntimeError(error_prefix)
                with self._lock:
                    state.progress = 1.0
                    if isinstance(result, str):
                        state.message = success_msg.format(result)
                    else:
                        state.message = success_msg
                    state.phase = CardOpPhase.SUCCESS
                    state.finished_at = time.monotonic()
            except Exception as e:
                detail = str(e).strip()
                with self._lock:
                    if detail:
                        state.message = f"{error_prefix}: {detail}"
                    else:
                        state.message = error_prefix
                    state.phase = CardOpPhase.ERROR
                    state.finished_at = time.monotonic()

        threading.Thread(target=worker, daemon=True).start()

    def _install_plugin_from_marketplace(self, mgr, entry: MarketplacePluginEntry, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr
        transport = self._selected_install_transport(card_id)

        def do_install(on_progress):
            if entry.registry_id:
                name = mgr.install_from_registry(
                    entry.registry_id,
                    on_progress=on_progress,
                    transport=transport,
                )
            else:
                name = mgr.install(entry.source_url, on_progress=on_progress, transport=transport)
            if mgr.get_state(name) == PluginState.ERROR:
                err = mgr.get_error(name) or tr("plugin_manager.status.load_failed")
                raise RuntimeError(err)
            norm_url = self._normalize_url(entry.source_url)
            if norm_url:
                with self._lock:
                    self._url_plugin_names[norm_url] = name
            self._invalidate_discover_cache()
            return name

        self._run_async(
            card_id,
            do_install,
            tr("plugin_manager.status.installed"),
            tr("plugin_manager.status.install_failed"),
        )

    def _install_plugin_from_url(self, mgr, url: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr
        clean_url = url.strip()
        if not clean_url:
            with self._lock:
                self._card_ops[card_id] = CardOpState(
                    phase=CardOpPhase.ERROR,
                    message=tr("plugin_manager.error.enter_github_url"),
                    finished_at=time.monotonic(),
                )
            return

        def do_install(on_progress):
            name = mgr.install(clean_url, on_progress=on_progress)
            if mgr.get_state(name) == PluginState.ERROR:
                err = mgr.get_error(name) or tr("plugin_manager.status.load_failed")
                raise RuntimeError(err)
            with self._lock:
                self._url_plugin_names[self._normalize_url(clean_url)] = name
            self._invalidate_discover_cache()
            return name

        self._run_async(
            card_id,
            do_install,
            tr("plugin_manager.status.installed"),
            tr("plugin_manager.status.install_failed"),
        )

    def _load_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr
        if mgr.get_state(name) == PluginState.ACTIVE:
            return

        def do_load(on_progress):
            ok = mgr.load(name, on_progress=on_progress)
            if not ok:
                err = mgr.get_error(name) or tr("plugin_manager.status.load_failed")
                raise RuntimeError(err)
            return True

        self._run_async(
            card_id,
            do_load,
            tr("plugin_manager.status.loaded").format(name=name),
            tr("plugin_manager.status.load_failed"),
        )

    def _unload_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr
        if mgr.get_state(name) != PluginState.ACTIVE:
            return

        try:
            if not mgr.unload(name):
                with self._lock:
                    state = self._card_ops.setdefault(card_id, CardOpState())
                    state.phase = CardOpPhase.ERROR
                    state.message = tr("plugin_manager.status.unload_failed")
                    state.finished_at = time.monotonic()
                return
            with self._lock:
                state = self._card_ops.setdefault(card_id, CardOpState())
                state.phase = CardOpPhase.SUCCESS
                state.message = tr("plugin_manager.status.unloaded").format(name=name)
                state.finished_at = time.monotonic()
        except Exception as e:
            with self._lock:
                state = self._card_ops.setdefault(card_id, CardOpState())
                state.phase = CardOpPhase.ERROR
                state.message = f"{tr('plugin_manager.status.unload_failed')}: {e}"
                state.finished_at = time.monotonic()

    def _reload_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        if not mgr.unload(name):
            with self._lock:
                state = self._card_ops.setdefault(card_id, CardOpState())
                state.phase = CardOpPhase.ERROR
                state.message = tr("plugin_manager.status.unload_failed")
                state.finished_at = time.monotonic()
            return

        def do_load(on_progress):
            ok = mgr.load(name, on_progress=on_progress)
            if not ok:
                err = mgr.get_error(name) or tr("plugin_manager.status.reload_failed")
                raise RuntimeError(err)
            return True

        self._run_async(
            card_id,
            do_load,
            tr("plugin_manager.status.reloaded").format(name=name),
            tr("plugin_manager.status.reload_failed"),
        )

    def _update_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_update(on_progress):
            mgr.update(name, on_progress=on_progress)
            self._invalidate_discover_cache()
            return True

        self._run_async(
            card_id,
            do_update,
            tr("plugin_manager.status.updated").format(name=name),
            tr("plugin_manager.status.update_failed"),
        )

    def _uninstall_plugin(self, mgr, name: str, card_id: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_uninstall(on_progress):
            on_progress(tr("plugin_manager.status.uninstalling").format(name=name))
            if not mgr.uninstall(name):
                raise RuntimeError(tr("plugin_manager.status.uninstall_failed"))
            self._invalidate_discover_cache()

        self._run_async(
            card_id,
            do_uninstall,
            tr("plugin_manager.status.uninstalled").format(name=name),
            tr("plugin_manager.status.uninstall_failed"),
        )

    def _with_local_plugins(self, entries: List[MarketplacePluginEntry], mgr) -> List[MarketplacePluginEntry]:
        merged = list(entries)
        known_keys: Set[str] = set()
        catalog_urls: Set[str] = set()
        for entry in merged:
            known_keys.update(self._entry_keys(entry))
            norm = self._normalize_url(entry.source_url)
            if norm:
                catalog_urls.add(norm)

        for plugin in self._get_discovered_plugins(mgr):
            plugin_keys = self._plugin_keys(plugin.name, plugin.path.name)
            if any(k in known_keys for k in plugin_keys):
                continue

            remote_url = self._remote_source_url(plugin.path)
            if remote_url:
                norm_remote = self._normalize_url(remote_url)
                if norm_remote in catalog_urls:
                    with self._lock:
                        self._url_plugin_names[norm_remote] = plugin.name
                    known_keys.update(plugin_keys)
                    continue

            source_path = str(plugin.path)
            merged.append(
                MarketplacePluginEntry(
                    source_url=source_path,
                    github_url=remote_url or "",
                    owner="",
                    repo=plugin.path.name,
                    name=plugin.name,
                    description=plugin.description or "",
                )
            )
            with self._lock:
                self._url_plugin_names[self._normalize_url(source_path)] = plugin.name
                if remote_url:
                    self._url_plugin_names[self._normalize_url(remote_url)] = plugin.name
            known_keys.update(plugin_keys)

        return merged

    @staticmethod
    def _remote_source_url(plugin_path: Path) -> str:
        from .installer import read_plugin_source_metadata

        source_info = read_plugin_source_metadata(plugin_path)
        if source_info:
            if source_info.github_url:
                return source_info.github_url
            if source_info.origin:
                return source_info.origin
        return PluginMarketplacePanel._git_remote_url(plugin_path)

    @staticmethod
    def _git_remote_url(plugin_path: Path) -> str:
        import subprocess
        if not (plugin_path / ".git").exists():
            return ""
        try:
            result = subprocess.run(
                ["git", "-C", str(plugin_path), "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=3,
            )
            url = result.stdout.strip()
            if url.endswith(".git"):
                url = url[:-4]
            return url
        except Exception:
            return ""

    def _get_installed_plugin_lookup(self, mgr) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for plugin in self._get_discovered_plugins(mgr):
            for key in self._plugin_keys(plugin.name, plugin.path.name):
                lookup[key] = plugin.name
        return lookup

    def _get_installed_plugin_versions(self, mgr) -> Dict[str, str]:
        return {plugin.name: plugin.version for plugin in self._get_discovered_plugins(mgr)}

    def _resolve_entry_plugin_name(
        self,
        entry: MarketplacePluginEntry,
        installed_lookup: Dict[str, str],
        installed_names: Set[str],
    ):
        norm_url = self._normalize_url(entry.source_url)
        by_url = None
        if norm_url:
            with self._lock:
                by_url = self._url_plugin_names.get(norm_url)
        if by_url and by_url in installed_names:
            return by_url
        for key in self._entry_keys(entry):
            plugin_name = installed_lookup.get(key)
            if plugin_name:
                return plugin_name
        return None

    @staticmethod
    def _normalize_url(url: str) -> str:
        return str(url or "").strip().rstrip("/")

    def _is_marketplace_entry_installed(
        self,
        entry: MarketplacePluginEntry,
        installed_keys: Set[str],
        installed_names: Set[str],
    ) -> bool:
        if any(key in installed_keys for key in self._entry_keys(entry)):
            return True
        norm_url = self._normalize_url(entry.source_url)
        if not norm_url:
            return False
        with self._lock:
            by_url = self._url_plugin_names.get(norm_url)
        return by_url is not None and by_url in installed_names

    @staticmethod
    def _is_local_entry(entry: MarketplacePluginEntry) -> bool:
        source = str(entry.source_url or "").strip()
        if not source:
            return False
        if source.startswith(("http://", "https://", "github:")):
            return False
        return Path(source).is_absolute() or source.startswith("~")

    @staticmethod
    def _is_local_only_entry(entry: MarketplacePluginEntry) -> bool:
        return PluginMarketplacePanel._is_local_entry(entry) and not bool(entry.github_url)

    def _entry_keys(self, entry: MarketplacePluginEntry) -> Set[str]:
        from .installer import normalize_repo_name

        normalized_repo = normalize_repo_name(entry.repo) if entry.repo else ""
        return self._plugin_keys(
            entry.repo,
            entry.name,
            normalized_repo,
            f"{entry.owner}-{entry.repo}" if entry.owner and entry.repo else "",
            f"{entry.owner}_{entry.repo}" if entry.owner and entry.repo else "",
        )

    @staticmethod
    def _plugin_keys(*values: str) -> Set[str]:
        keys = set()
        for value in values:
            raw = str(value or "").strip()
            if not raw:
                continue
            lower = raw.lower()
            keys.add(lower)
            normalized = "".join(ch for ch in lower if ch.isalnum())
            if normalized:
                keys.add(normalized)
        return keys

    @staticmethod
    def _entry_type_tags(entry: MarketplacePluginEntry) -> List[str]:
        tags: List[str] = []
        for topic in entry.topics:
            clean = topic.replace("_", " ").replace("-", " ").strip()
            if not clean:
                continue
            pretty = " ".join(part.capitalize() for part in clean.split())
            if pretty and pretty not in tags:
                tags.append(pretty)
        if entry.language and entry.language not in tags and entry.language.lower() != "python":
            tags.append(entry.language)
        return tags

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3].rstrip() + "..."
