# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin Marketplace panel UI."""

from typing import List, Set
import threading

from .marketplace import (
    MarketplacePluginEntry,
    PluginMarketplaceCatalog,
    get_plugin_marketplace_urls,
)
from .types import Panel

MAX_OUTPUT_LINES = 100


class PluginMarketplacePanel(Panel):
    """Floating marketplace browser for plugin repositories."""

    idname = "lfs.plugin_marketplace"
    label = "Plugin Marketplace"
    space = "FLOATING"
    order = 91
    options = {"DEFAULT_CLOSED"}

    GRID_COLUMNS = 100
    CARD_WIDTH = 320
    CARD_HEIGHT = 150
    CARD_SPACING = 12
    FILTER_WIDTH = 140
    SORT_WIDTH = 170

    def __init__(self):
        self._catalog = PluginMarketplaceCatalog()
        self._configured_urls: List[str] = []
        self._install_filter_idx = 0
        self._sort_idx = 0

        self._status_message = ""
        self._status_is_error = False
        self._operation_in_progress = False
        self._output_lines: List[str] = []
        self._lock = threading.Lock()

    def draw(self, layout):
        import lichtfeld as lf
        from .manager import PluginManager

        tr = lf.ui.tr
        theme = lf.ui.theme()
        palette = theme.palette
        mgr = PluginManager.instance()
        self._sync_catalog_urls()

        scale = layout.get_dpi_scale()

        layout.text_colored(
            tr("plugin_marketplace.title_line"),
            palette.info,
        )
        layout.spacing()

        self._draw_marketplace_controls(layout, scale)
        layout.spacing()

        layout.text_colored(tr("plugin_marketplace.warning_body"), palette.warning)
        layout.spacing()

        entries, is_loading = self._catalog.snapshot()
        installed_keys = self._get_installed_plugin_keys(mgr)
        entries = self._filter_and_sort_entries(entries, installed_keys)

        if is_loading:
            layout.text_disabled(tr("plugin_marketplace.loading"))

        self._draw_status(layout)

        layout.spacing()
        layout.separator()
        layout.spacing()

        if not entries:
            layout.text_disabled(tr("plugin_marketplace.no_plugins"))
            layout.text_disabled(tr("plugin_marketplace.edit_list_hint"))
            return

        card_w = self.CARD_WIDTH * scale
        card_h = self.CARD_HEIGHT * scale
        spacing = self.CARD_SPACING * scale

        avail_w, avail_h = layout.get_content_region_avail()
        visible_columns = self._visible_columns(avail_w, card_w, spacing)
        card_w = self._fit_card_width(avail_w, card_w, spacing, visible_columns, scale)

        scroll_height = max(220 * scale, avail_h - 24 * scale)
        with self._lock:
            install_in_progress = self._operation_in_progress

        if layout.begin_child("##plugin_marketplace_scroll", (0, scroll_height), border=False):
            row_count = (len(entries) + visible_columns - 1) // visible_columns
            for row in range(row_count):
                base = row * visible_columns
                drawn = 0
                for col in range(visible_columns):
                    idx = base + col
                    if idx >= len(entries):
                        break

                    if drawn > 0:
                        layout.same_line(spacing=spacing)

                    self._draw_plugin_card(
                        layout,
                        mgr,
                        idx,
                        entries[idx],
                        installed_keys,
                        install_in_progress,
                        card_w,
                        card_h,
                        scale,
                    )
                    drawn += 1

                if drawn > 0:
                    layout.spacing()
        layout.end_child()

    def _draw_plugin_card(
        self,
        layout,
        mgr,
        idx: int,
        entry: MarketplacePluginEntry,
        installed_keys: Set[str],
        install_in_progress: bool,
        card_w: float,
        card_h: float,
        scale: float,
    ):
        import lichtfeld as lf

        tr = lf.ui.tr
        theme = lf.ui.theme()
        palette = theme.palette
        card_rounding = max(theme.sizes.frame_rounding, theme.sizes.popup_rounding)
        layout.push_style_var("ChildRounding", card_rounding * scale)
        layout.push_style_color("ChildBg", palette.surface)
        layout.push_style_color("Border", palette.border)

        if layout.begin_child(f"##plugin_card_{idx}", (card_w, card_h), border=True):
            is_installed = self._is_marketplace_entry_installed(entry, installed_keys)
            short_name = entry.name or entry.repo or tr("plugin_marketplace.unknown_plugin")
            repo_label = f"{entry.owner}/{entry.repo}" if entry.owner and entry.repo else entry.repo
            description = self._truncate_text(entry.description or tr("plugin_marketplace.no_description"), 100)

            layout.text_colored(short_name, palette.text)
            if repo_label:
                layout.text_disabled(repo_label)
            layout.text_colored(f"{tr('plugin_marketplace.stars')}: {entry.stars}", palette.warning)

            if entry.error:
                layout.text_colored(tr("plugin_marketplace.invalid_link"), palette.error)
            else:
                layout.text_wrapped(description)

            layout.spacing()
            layout.separator()
            layout.spacing()

            button_width = 106 * scale
            button_height = 24 * scale
            disable_install = install_in_progress or is_installed or bool(entry.error)
            install_label = (
                tr("plugin_marketplace.button.installed")
                if is_installed
                else tr("plugin_marketplace.button.install")
            )

            if disable_install:
                layout.begin_disabled()
            if layout.button_styled(f"{install_label}##install_{idx}", "success", (button_width, button_height)):
                if not disable_install:
                    self._install_plugin_from_marketplace(mgr, entry.source_url)
            if layout.is_item_hovered():
                layout.set_mouse_cursor_hand()
            if disable_install:
                layout.end_disabled()

            layout.same_line(spacing=8 * scale)
            if layout.button_styled(
                f"{tr('plugin_marketplace.button.github')}##github_{idx}",
                "primary",
                (button_width, button_height),
            ):
                lf.ui.open_url(entry.github_url)
            if layout.is_item_hovered():
                layout.set_mouse_cursor_hand()
        layout.end_child()

        layout.pop_style_color(2)
        layout.pop_style_var()

    def _draw_marketplace_controls(self, layout, scale: float):
        import lichtfeld as lf

        tr = lf.ui.tr
        avail_w, _ = layout.get_content_region_avail()
        filter_items = [
            tr("plugin_marketplace.filter.all"),
            tr("plugin_marketplace.filter.installed"),
            tr("plugin_marketplace.filter.not_installed"),
        ]
        sort_items = [
            tr("plugin_marketplace.sort.stars_desc"),
            tr("plugin_marketplace.sort.stars_asc"),
            tr("plugin_marketplace.sort.name_asc"),
            tr("plugin_marketplace.sort.name_desc"),
        ]

        filter_w = max(1.0, min(self.FILTER_WIDTH * scale, avail_w))
        remaining = max(1.0, avail_w - filter_w - 8 * scale)
        sort_w = max(1.0, min(self.SORT_WIDTH * scale, remaining))

        layout.text_disabled(tr("plugin_marketplace.filter_label"))
        layout.same_line(spacing=6 * scale)
        layout.set_next_item_width(filter_w)
        _, self._install_filter_idx = layout.combo(
            "##install_filter",
            self._install_filter_idx,
            filter_items,
        )
        layout.same_line(spacing=10 * scale)
        layout.text_disabled(tr("plugin_marketplace.sort_label"))
        layout.same_line(spacing=6 * scale)
        layout.set_next_item_width(sort_w)
        _, self._sort_idx = layout.combo("##sort_filter", self._sort_idx, sort_items)

    @staticmethod
    def _visible_columns(avail_w: float, card_w: float, spacing: float) -> int:
        if avail_w <= 0:
            return 1
        return max(1, min(PluginMarketplacePanel.GRID_COLUMNS, int((avail_w + spacing) // (card_w + spacing))))

    @staticmethod
    def _fit_card_width(
        avail_w: float,
        preferred_card_w: float,
        spacing: float,
        columns: int,
        scale: float,
    ) -> float:
        if columns <= 0:
            return preferred_card_w
        usable = max(1.0, avail_w - (columns - 1) * spacing - 2.0 * scale)
        return max(1.0, min(preferred_card_w, usable / columns))

    def _sync_catalog_urls(self):
        current_urls = get_plugin_marketplace_urls()
        if current_urls != self._configured_urls:
            self._configured_urls = current_urls
            self._catalog.set_urls(current_urls)
            self._catalog.refresh_async(force=True)

    def _filter_and_sort_entries(
        self,
        entries: List[MarketplacePluginEntry],
        installed_keys: Set[str],
    ) -> List[MarketplacePluginEntry]:
        filtered: List[MarketplacePluginEntry] = []
        for entry in entries:
            is_installed = self._is_marketplace_entry_installed(entry, installed_keys)
            if self._install_filter_idx == 1 and not is_installed:
                continue
            if self._install_filter_idx == 2 and is_installed:
                continue

            filtered.append(entry)

        if self._sort_idx == 1:
            return sorted(filtered, key=lambda e: (e.stars, e.name.lower()))
        if self._sort_idx == 2:
            return sorted(filtered, key=lambda e: e.name.lower())
        if self._sort_idx == 3:
            return sorted(filtered, key=lambda e: e.name.lower(), reverse=True)
        return sorted(filtered, key=lambda e: (e.stars, e.name.lower()), reverse=True)

    def _set_status(self, message: str, is_error: bool = False):
        self._status_message = message
        self._status_is_error = is_error

    def _add_output(self, line: str):
        with self._lock:
            self._output_lines.append(line)
            if len(self._output_lines) > MAX_OUTPUT_LINES:
                self._output_lines = self._output_lines[-MAX_OUTPUT_LINES:]

    def _clear_output(self):
        with self._lock:
            self._output_lines.clear()

    def _run_async(self, operation, success_msg: str, error_prefix: str):
        def on_progress(msg: str):
            self._set_status(msg)
            self._add_output(msg)

        def worker():
            with self._lock:
                self._operation_in_progress = True
            self._clear_output()
            try:
                result = operation(on_progress)
                self._set_status(success_msg.format(result) if result else success_msg)
            except Exception as e:
                self._set_status(f"{error_prefix}: {e}", True)
            finally:
                with self._lock:
                    self._operation_in_progress = False

        threading.Thread(target=worker, daemon=True).start()

    def _install_plugin_from_marketplace(self, mgr, url: str):
        self._run_async(
            lambda cb: mgr.install(url, on_progress=cb),
            "Installed: {}",
            "Install failed",
        )

    def _draw_status(self, layout):
        import lichtfeld as lf

        palette = lf.ui.theme().palette
        if not self._status_message:
            return
        color = palette.error if self._status_is_error else palette.success
        layout.text_colored(self._status_message, color)

    def _get_installed_plugin_keys(self, mgr) -> Set[str]:
        keys = set()
        for plugin in mgr.discover():
            keys.add(plugin.name.lower())
            keys.add(plugin.path.name.lower())
        return keys

    def _is_marketplace_entry_installed(
        self,
        entry: MarketplacePluginEntry,
        installed_keys: Set[str],
    ) -> bool:
        candidates = {
            entry.repo.lower(),
            entry.name.lower(),
        }
        return any(candidate in installed_keys for candidate in candidates if candidate)

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3].rstrip() + "..."
