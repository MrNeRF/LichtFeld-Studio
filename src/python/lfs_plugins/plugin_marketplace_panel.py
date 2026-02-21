# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unified plugin marketplace floating panel."""

import configparser
import json
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Set

from .installer import parse_github_url
from .marketplace import (
    MarketplacePluginEntry,
    PluginMarketplaceCatalog,
    get_plugin_marketplace_urls,
    set_plugin_marketplace_urls,
)
from .plugin import PluginState
from .types import Panel

MAX_OUTPUT_LINES = 100


class PluginMarketplacePanel(Panel):
    """Floating plugin window for browsing, installing, and managing plugins."""

    idname = "lfs.plugin_marketplace"
    label = "Plugin Marketplace"
    space = "FLOATING"
    order = 91
    options = {"DEFAULT_CLOSED"}

    GRID_COLUMNS = 100
    CARD_WIDTH = 330
    CARD_HEIGHT = 200
    CARD_SPACING = 12
    FILTER_WIDTH = 140
    SORT_WIDTH = 170

    def __init__(self):
        self._catalog = PluginMarketplaceCatalog()
        self._configured_urls: List[str] = []
        self._url_plugin_names: Dict[str, str] = {}
        self._local_origin_cache: Dict[str, Dict[str, object]] = {}
        self._manual_url = ""
        self._install_filter_idx = 0
        self._sort_idx = 0

        self._status_message = ""
        self._status_is_error = False
        self._operation_in_progress = False
        self._output_lines: List[str] = []
        self._lock = threading.Lock()
        self._pending_uninstall_name = ""
        self._pending_uninstall_open = False

    def draw(self, layout):
        import lichtfeld as lf
        from .manager import PluginManager

        tr = lf.ui.tr
        theme = lf.ui.theme()
        palette = theme.palette
        mgr = PluginManager.instance()
        self._sync_catalog_urls()

        scale = layout.get_dpi_scale()
        self._draw_uninstall_confirmation_modal(layout, mgr, scale)

        layout.text_colored(
            tr("plugin_marketplace.title_line"),
            palette.info,
        )
        layout.spacing()

        layout.text_colored(tr("plugin_marketplace.warning_body"), palette.warning)
        self._draw_warning_error(layout)
        layout.spacing()

        self._draw_marketplace_controls(layout, mgr, scale)
        layout.spacing()

        entries, is_loading = self._catalog.snapshot()
        entries = self._with_local_plugins(entries, mgr)
        installed_lookup = self._get_installed_plugin_lookup(mgr)
        installed_versions = self._get_installed_plugin_versions(mgr)
        installed_names = set(installed_lookup.values())
        entries = self._filter_and_sort_entries(entries, set(installed_lookup.keys()))

        if is_loading:
            layout.text_disabled(tr("plugin_marketplace.loading"))

        self._draw_status(layout)
        self._draw_output(layout)

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
                        installed_lookup,
                        installed_versions,
                        installed_names,
                        install_in_progress,
                        card_w,
                        card_h,
                        scale,
                    )
                    drawn += 1

                if drawn > 0:
                    layout.spacing()
        layout.end_child()

    def _draw_manual_install_controls(self, layout, mgr, scale: float):
        import lichtfeld as lf

        tr = lf.ui.tr
        with self._lock:
            in_progress = self._operation_in_progress

        layout.label(tr("plugin_manager.github_url_or_shorthand"))
        input_width = max(140.0, layout.get_content_region_avail()[0] - 104.0 * scale)
        layout.set_next_item_width(input_width)
        _, self._manual_url = layout.input_text("##marketplace_install_url", self._manual_url)

        layout.same_line(spacing=8.0 * scale)
        if in_progress:
            layout.begin_disabled()
        if layout.button_styled(tr("plugin_manager.button.install_plugin"), "success", (0, 28)):
            if not in_progress:
                self._install_plugin_from_url(mgr, self._manual_url)
        if in_progress:
            layout.end_disabled()

        if layout.tree_node(tr("plugin_manager.supported_formats")):
            layout.bullet_text("https://github.com/owner/repo")
            layout.bullet_text("github:owner/repo")
            layout.bullet_text("owner/repo")
            layout.tree_pop()

    def _draw_plugin_card(
        self,
        layout,
        mgr,
        idx: int,
        entry: MarketplacePluginEntry,
        installed_lookup: Dict[str, str],
        installed_versions: Dict[str, str],
        installed_names: Set[str],
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

        plugin_name = self._resolve_entry_plugin_name(entry, installed_lookup, installed_names)
        plugin_state = mgr.get_state(plugin_name) if plugin_name else None
        is_installed = plugin_name is not None
        is_local = self._is_local_entry(entry)
        has_github = bool(entry.github_url)
        is_local_only = self._is_local_only_entry(entry)

        if layout.begin_child(f"##plugin_card_{idx}", (card_w, card_h), border=True):
            short_name = entry.name or entry.repo or tr("plugin_marketplace.unknown_plugin")
            repo_label = f"{entry.owner}/{entry.repo}" if entry.owner and entry.repo else entry.repo
            description = self._truncate_text(entry.description or tr("plugin_marketplace.no_description"), 90)

            layout.text_colored(short_name, palette.text)
            if plugin_name and plugin_state == PluginState.ACTIVE:
                version = installed_versions.get(plugin_name, "").strip()
                if version:
                    version_label = version if version.lower().startswith("v") else f"v{version}"
                    layout.same_line(spacing=6 * scale)
                    layout.text_colored(version_label, palette.info)
            if repo_label:
                layout.text_disabled(repo_label)
            if not is_local_only:
                layout.text_colored(f"{tr('plugin_marketplace.stars')}: {entry.stars}", palette.warning)

            tags = self._entry_type_tags(entry)
            if tags:
                layout.text_disabled("  |  ".join(tags[:3]))
            if is_local:
                layout.text_colored(tr("plugin_marketplace.local_install"), palette.info)

            if is_installed:
                state_str = plugin_state.value if plugin_state else tr("plugin_manager.status_not_loaded")
                layout.text_colored(
                    f"{tr('plugin_manager.status')}: {state_str}",
                    palette.success if plugin_state == PluginState.ACTIVE else palette.text_dim,
                )

            if entry.error:
                layout.text_colored(tr("plugin_marketplace.invalid_link"), palette.error)
            else:
                layout.text_wrapped(description)

            layout.spacing()
            layout.separator()
            layout.spacing()

            button_spacing = 6 * scale
            button_height = 25 * scale
            avail_button_w, _ = layout.get_content_region_avail()
            if is_installed:
                disabled = install_in_progress
                if disabled:
                    layout.begin_disabled()

                if is_local_only:
                    button_width = max(40.0, (avail_button_w - button_spacing) * 0.5)
                    load_label = (
                        tr("plugin_manager.button.unload")
                        if plugin_state == PluginState.ACTIVE
                        else tr("plugin_manager.button.load")
                    )
                    load_style = "warning" if plugin_state == PluginState.ACTIVE else "success"
                    if layout.button_styled(
                        f"{load_label}##loadtoggle_{idx}",
                        load_style,
                        (button_width, button_height),
                    ):
                        if plugin_state == PluginState.ACTIVE:
                            self._unload_plugin(mgr, plugin_name)
                        else:
                            self._load_plugin(mgr, plugin_name)
                    layout.same_line(spacing=button_spacing)
                    if layout.button_styled(
                        f"{tr('plugin_manager.button.uninstall')}##uninstall_{idx}",
                        "error",
                        (button_width, button_height),
                    ):
                        self._request_uninstall_confirmation(plugin_name)
                else:
                    button_width = max(40.0, (avail_button_w - button_spacing * 2.0) / 3.0)
                    if is_local and has_github:
                        load_label = (
                            tr("plugin_manager.button.unload")
                            if plugin_state == PluginState.ACTIVE
                            else tr("plugin_manager.button.load")
                        )
                        load_style = "warning" if plugin_state == PluginState.ACTIVE else "success"
                        if layout.button_styled(
                            f"{load_label}##loadtoggle_{idx}",
                            load_style,
                            (button_width, button_height),
                        ):
                            if plugin_state == PluginState.ACTIVE:
                                self._unload_plugin(mgr, plugin_name)
                            else:
                                self._load_plugin(mgr, plugin_name)
                        layout.same_line(spacing=button_spacing)
                        if layout.button_styled(
                            f"{tr('plugin_manager.button.update')}##update_{idx}",
                            "primary",
                            (button_width, button_height),
                        ):
                            self._update_plugin(mgr, plugin_name)
                        layout.same_line(spacing=button_spacing)
                        if layout.button_styled(
                            f"{tr('plugin_manager.button.uninstall')}##uninstall_{idx}",
                            "error",
                            (button_width, button_height),
                        ):
                            self._request_uninstall_confirmation(plugin_name)
                    else:
                        if plugin_state == PluginState.ACTIVE:
                            if layout.button_styled(
                                f"{tr('plugin_manager.button.reload')}##reload_{idx}",
                                "primary",
                                (button_width, button_height),
                            ):
                                self._reload_plugin(mgr, plugin_name)
                            layout.same_line(spacing=button_spacing)
                            if layout.button_styled(
                                f"{tr('plugin_manager.button.unload')}##unload_{idx}",
                                "warning",
                                (button_width, button_height),
                            ):
                                self._unload_plugin(mgr, plugin_name)
                        else:
                            if layout.button_styled(
                                f"{tr('plugin_manager.button.load')}##load_{idx}",
                                "success",
                                (button_width, button_height),
                            ):
                                self._load_plugin(mgr, plugin_name)
                            layout.same_line(spacing=button_spacing)
                            if layout.button_styled(
                                f"{tr('plugin_manager.button.update')}##update_{idx}",
                                "primary",
                                (button_width, button_height),
                            ):
                                self._update_plugin(mgr, plugin_name)
                        layout.same_line(spacing=button_spacing)
                        if layout.button_styled(
                            f"{tr('plugin_manager.button.uninstall')}##uninstall_{idx}",
                            "error",
                            (button_width, button_height),
                        ):
                            self._request_uninstall_confirmation(plugin_name)

                layout.spacing()
                if has_github:
                    if layout.button_styled(
                        f"{tr('plugin_marketplace.button.github')}##github_{idx}",
                        "primary",
                        (avail_button_w, button_height),
                    ):
                        lf.ui.open_url(entry.github_url)

                if disabled:
                    layout.end_disabled()
            else:
                if is_local_only:
                    layout.spacing()
                    layout.end_child()
                    layout.pop_style_color(2)
                    layout.pop_style_var()
                    return

                button_width = max(40.0, (avail_button_w - button_spacing) * 0.5)
                disable_install = install_in_progress or bool(entry.error)

                if disable_install:
                    layout.begin_disabled()
                if layout.button_styled(
                    f"{tr('plugin_marketplace.button.install')}##install_{idx}",
                    "success",
                    (button_width, button_height),
                ):
                    if not disable_install:
                        self._install_plugin_from_marketplace(mgr, entry.source_url)
                if disable_install:
                    layout.end_disabled()

                layout.same_line(spacing=button_spacing)
                if has_github:
                    if layout.button_styled(
                        f"{tr('plugin_marketplace.button.github')}##github_{idx}",
                        "primary",
                        (button_width, button_height),
                    ):
                        lf.ui.open_url(entry.github_url)
        layout.end_child()

        layout.pop_style_color(2)
        layout.pop_style_var()

    def _draw_marketplace_controls(self, layout, mgr, scale: float):
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

        layout.spacing()
        self._draw_manual_install_controls(layout, mgr, scale)

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
        filtered = []
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
                if result is False:
                    raise RuntimeError("")
                if isinstance(result, str):
                    self._set_status(success_msg.format(result))
                else:
                    self._set_status(success_msg)
            except Exception as e:
                detail = str(e).strip()
                if detail:
                    self._set_status(f"{error_prefix}: {detail}", True)
                else:
                    self._set_status(error_prefix, True)
            finally:
                with self._lock:
                    self._operation_in_progress = False

        threading.Thread(target=worker, daemon=True).start()

    def _install_plugin_from_marketplace(self, mgr, url: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_install(on_progress):
            name = mgr.install(url, on_progress=on_progress)
            if mgr.get_state(name) == PluginState.ERROR:
                err = mgr.get_error(name) or tr("plugin_manager.status.load_failed")
                raise RuntimeError(err)
            self._url_plugin_names[self._normalize_url(url)] = name
            self._remember_marketplace_url(url)
            return name

        self._run_async(
            do_install,
            tr("plugin_manager.status.installed"),
            tr("plugin_manager.status.install_failed"),
        )

    def _install_plugin_from_url(self, mgr, url: str):
        import lichtfeld as lf

        tr = lf.ui.tr
        clean_url = url.strip()
        if not clean_url:
            self._set_status(tr("plugin_manager.error.enter_github_url"), True)
            return

        def do_install(on_progress):
            name = mgr.install(clean_url, on_progress=on_progress)
            if mgr.get_state(name) == PluginState.ERROR:
                err = mgr.get_error(name) or tr("plugin_manager.status.load_failed")
                raise RuntimeError(err)
            self._manual_url = ""
            self._url_plugin_names[self._normalize_url(clean_url)] = name
            self._remember_marketplace_url(clean_url)
            return name

        self._run_async(
            do_install,
            tr("plugin_manager.status.installed"),
            tr("plugin_manager.status.install_failed"),
        )

    def _load_plugin(self, mgr, name: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_load(on_progress):
            ok = mgr.load(name, on_progress=on_progress)
            if not ok:
                err = mgr.get_error(name) or tr("plugin_manager.status.load_failed")
                raise RuntimeError(err)
            return True

        self._run_async(
            do_load,
            tr("plugin_manager.status.loaded").format(name=name),
            tr("plugin_manager.status.load_failed"),
        )

    def _unload_plugin(self, mgr, name: str):
        import lichtfeld as lf

        tr = lf.ui.tr
        try:
            mgr.unload(name)
            self._set_status(tr("plugin_manager.status.unloaded").format(name=name))
        except Exception as e:
            self._set_status(f"{tr('plugin_manager.status.unload_failed')}: {e}", True)

    def _reload_plugin(self, mgr, name: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_reload(on_progress):
            mgr.unload(name)
            ok = mgr.load(name, on_progress=on_progress)
            if not ok:
                err = mgr.get_error(name) or tr("plugin_manager.status.reload_failed")
                raise RuntimeError(err)
            return True

        self._run_async(
            do_reload,
            tr("plugin_manager.status.reloaded").format(name=name),
            tr("plugin_manager.status.reload_failed"),
        )

    def _update_plugin(self, mgr, name: str):
        import lichtfeld as lf

        tr = lf.ui.tr
        self._run_async(
            lambda cb: mgr.update(name, on_progress=cb),
            tr("plugin_manager.status.updated").format(name=name),
            tr("plugin_manager.status.update_failed"),
        )

    def _uninstall_plugin(self, mgr, name: str):
        import lichtfeld as lf

        tr = lf.ui.tr
        try:
            mgr.uninstall(name)
            self._set_status(tr("plugin_manager.status.uninstalled").format(name=name))
        except Exception as e:
            self._set_status(f"{tr('plugin_manager.status.uninstall_failed')}: {e}", True)

    def _request_uninstall_confirmation(self, name: str):
        if not name:
            return
        self._pending_uninstall_name = name
        self._pending_uninstall_open = True

    def _draw_uninstall_confirmation_modal(self, layout, mgr, scale: float):
        import lichtfeld as lf

        tr = lf.ui.tr
        if not self._pending_uninstall_name and not self._pending_uninstall_open:
            return

        popup_title = tr("plugin_marketplace.confirm_uninstall_title")
        popup_id = f"{popup_title}##plugin_marketplace_uninstall_confirm"

        if self._pending_uninstall_open:
            layout.set_next_window_pos_viewport_center(always=True)
            layout.set_next_window_size((380 * scale, 0))
            layout.open_popup(popup_id)
            self._pending_uninstall_open = False

        layout.push_modal_style()
        if layout.begin_popup_modal(popup_id):
            avail_width = layout.get_content_region_avail()[0]
            text_width = layout.calc_text_size(
                tr("plugin_marketplace.confirm_uninstall_message").format(name=self._pending_uninstall_name)
            )[0]
            layout.set_cursor_pos_x(layout.get_cursor_pos()[0] + max(0.0, (avail_width - text_width) * 0.5))
            layout.text_wrapped(
                tr("plugin_marketplace.confirm_uninstall_message").format(name=self._pending_uninstall_name)
            )
            layout.spacing()
            layout.separator()
            layout.spacing()

            button_width = 92 * scale
            button_spacing = 8 * scale
            avail_width = layout.get_content_region_avail()[0]
            total_width = button_width * 2 + button_spacing
            layout.set_cursor_pos_x(layout.get_cursor_pos()[0] + max(0.0, (avail_width - total_width) * 0.5))

            if layout.button_styled(
                tr("plugin_marketplace.confirm_uninstall_no"),
                "secondary",
                (button_width, 0),
            ) or lf.ui.is_key_pressed(lf.ui.Key.ESCAPE):
                self._pending_uninstall_name = ""
                layout.close_current_popup()

            layout.same_line(0, button_spacing)
            if layout.button_styled(
                tr("plugin_marketplace.confirm_uninstall_yes"),
                "error",
                (button_width, 0),
            ):
                uninstall_name = self._pending_uninstall_name
                self._pending_uninstall_name = ""
                layout.close_current_popup()
                self._uninstall_plugin(mgr, uninstall_name)

            layout.end_popup_modal()
        layout.pop_modal_style()

    def _draw_status(self, layout):
        import lichtfeld as lf

        palette = lf.ui.theme().palette
        if not self._status_message or self._status_is_error:
            return
        layout.text_colored(self._status_message, palette.success)

    def _draw_warning_error(self, layout):
        import lichtfeld as lf

        if not self._status_message or not self._status_is_error:
            return
        layout.text_colored(self._status_message, lf.ui.theme().palette.warning)

    def _draw_output(self, layout):
        import lichtfeld as lf

        tr = lf.ui.tr
        with self._lock:
            lines = list(self._output_lines[-15:])
            in_progress = self._operation_in_progress

        if not lines and not in_progress:
            return

        if in_progress:
            layout.progress_bar(-1.0, self._status_message or tr("plugin_manager.working"))

        if lines and layout.tree_node(tr("plugin_manager.output")):
            for line in lines:
                layout.text_wrapped(line)
            layout.tree_pop()

    def _remember_marketplace_url(self, url: str):
        value = self._normalize_url(url)
        if not value:
            return
        urls = get_plugin_marketplace_urls()
        if any(self._normalize_url(existing) == value for existing in urls):
            return
        urls.append(value)
        set_plugin_marketplace_urls(urls)
        self._configured_urls = []

    def _with_local_plugins(self, entries: List[MarketplacePluginEntry], mgr) -> List[MarketplacePluginEntry]:
        merged = list(entries)
        known_keys: Set[str] = set()
        for entry in merged:
            known_keys.update(self._entry_keys(entry))

        for plugin in mgr.discover():
            plugin_keys = self._plugin_keys(plugin.name, plugin.path.name)
            if any(k in known_keys for k in plugin_keys):
                continue

            source_path = str(plugin.path)
            origin = self._get_local_plugin_origin(plugin.path)
            merged.append(
                MarketplacePluginEntry(
                    source_url=source_path,
                    github_url=str(origin.get("github_url", "")),
                    owner=str(origin.get("owner", "")),
                    repo=str(origin.get("repo", "")) or plugin.path.name,
                    name=plugin.name,
                    description=plugin.description or "",
                    stars=int(origin.get("stars", 0)),
                    language="",
                    topics=(),
                )
            )
            self._url_plugin_names[self._normalize_url(source_path)] = plugin.name
            known_keys.update(plugin_keys)

        return merged

    def _get_installed_plugin_lookup(self, mgr) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for plugin in mgr.discover():
            for key in self._plugin_keys(plugin.name, plugin.path.name):
                lookup[key] = plugin.name
        return lookup

    @staticmethod
    def _get_installed_plugin_versions(mgr) -> Dict[str, str]:
        return {plugin.name: plugin.version for plugin in mgr.discover()}

    def _resolve_entry_plugin_name(
        self,
        entry: MarketplacePluginEntry,
        installed_lookup: Dict[str, str],
        installed_names: Set[str],
    ):
        by_url = self._url_plugin_names.get(self._normalize_url(entry.source_url))
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
    ) -> bool:
        return any(key in installed_keys for key in self._entry_keys(entry))

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
        return self._plugin_keys(
            entry.repo,
            entry.name,
            f"{entry.owner}-{entry.repo}" if entry.owner and entry.repo else "",
            f"{entry.owner}_{entry.repo}" if entry.owner and entry.repo else "",
        )

    def _get_local_plugin_origin(self, plugin_path: Path) -> Dict[str, object]:
        key = str(plugin_path)
        cached = self._local_origin_cache.get(key)
        if cached is not None:
            return cached

        # Strict policy: only the plugin root git remote is considered authoritative.
        github_url = self._extract_github_url_from_git(plugin_path)
        info = {"github_url": "", "owner": "", "repo": "", "stars": 0}
        if github_url:
            info = self._probe_github_repo(github_url)

        self._local_origin_cache[key] = info
        return info

    @staticmethod
    def _extract_github_url_from_git(plugin_path: Path) -> str:
        def run_git(args: List[str], timeout: float = 2.0) -> str:
            try:
                result = subprocess.run(
                    ["git", "-C", str(plugin_path), *args],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                return ""
            return ""

        try:
            # Primary remote.
            origin_url = run_git(["config", "--get", "remote.origin.url"], timeout=3.0)
            normalized = PluginMarketplacePanel._normalize_github_repo_url(origin_url)
            if normalized:
                return normalized

            # Fallback: scan all remotes.
            remotes = run_git(["remote", "-v"], timeout=3.0)
            for line in remotes.splitlines():
                parts = line.split()
                if len(parts) < 2:
                    continue
                normalized = PluginMarketplacePanel._normalize_github_repo_url(parts[1])
                if normalized:
                    return normalized

            # Fallback: parse .git/config directly.
            git_dir = plugin_path / ".git"
            git_config_path = git_dir / "config"
            if git_dir.is_file():
                ref = git_dir.read_text(encoding="utf-8", errors="ignore").strip()
                if ref.startswith("gitdir:"):
                    rel = ref.split("gitdir:", 1)[1].strip()
                    git_config_path = (plugin_path / rel / "config").resolve()
            if git_config_path.exists():
                parser = configparser.ConfigParser()
                parser.read(git_config_path, encoding="utf-8")
                for section in parser.sections():
                    if not section.startswith("remote "):
                        continue
                    url = parser.get(section, "url", fallback="")
                    normalized = PluginMarketplacePanel._normalize_github_repo_url(url)
                    if normalized:
                        return normalized
        except Exception:
            return ""
        return ""

    @staticmethod
    def _normalize_github_repo_url(url: str) -> str:
        raw = str(url or "").strip()
        if not raw:
            return ""
        if raw.startswith("git@github.com:"):
            raw = "https://github.com/" + raw[len("git@github.com:"):]
        elif raw.startswith("ssh://git@github.com/"):
            raw = "https://github.com/" + raw[len("ssh://git@github.com/"):]
        if raw.endswith(".git"):
            raw = raw[:-4]
        try:
            owner, repo, _ = parse_github_url(raw)
            return f"https://github.com/{owner}/{repo}"
        except Exception:
            return ""

    @staticmethod
    def _probe_github_repo(github_url: str) -> Dict[str, object]:
        try:
            owner, repo, _ = parse_github_url(github_url)
        except Exception:
            return {"github_url": "", "owner": "", "repo": "", "stars": 0}

        fallback = {
            "github_url": github_url,
            "owner": owner,
            "repo": repo,
            "stars": 0,
        }

        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        req = urllib.request.Request(
            api_url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "LichtFeld-PluginMarketplace/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            stars = int(data.get("stargazers_count", 0))
            html_url = str(data.get("html_url") or github_url)
            return {
                "github_url": html_url,
                "owner": owner,
                "repo": repo,
                "stars": stars,
            }
        except urllib.error.HTTPError as e:
            if e.code in (401, 404):
                return {"github_url": "", "owner": "", "repo": "", "stars": 0}
            return fallback
        except Exception:
            return fallback

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
