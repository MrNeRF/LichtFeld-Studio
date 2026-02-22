# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unified plugin marketplace floating panel."""

import threading
from pathlib import Path
from typing import Dict, List, Optional, Set

from .marketplace import (
    MarketplacePluginEntry,
    PluginMarketplaceCatalog,
)
from .plugin import PluginInfo, PluginState
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
        self._url_plugin_names: Dict[str, str] = {}
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

        self._discover_cache: Optional[List[PluginInfo]] = None
        self._clear_manual_url = False

    def draw(self, layout):
        import lichtfeld as lf
        from .manager import PluginManager

        tr = lf.ui.tr
        theme = lf.ui.theme()
        palette = theme.palette
        mgr = PluginManager.instance()
        self._ensure_loaded()

        if self._clear_manual_url:
            self._manual_url = ""
            self._clear_manual_url = False

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
        entries = self._filter_and_sort_entries(entries, set(installed_lookup.keys()), installed_names)

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

    def _ensure_loaded(self):
        self._catalog.refresh_async()

    def _invalidate_discover_cache(self):
        self._discover_cache = None

    def _get_discovered_plugins(self, mgr) -> List[PluginInfo]:
        cache = self._discover_cache
        if cache is None:
            cache = mgr.discover()
            self._discover_cache = cache
        return cache

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

        card_id = entry.registry_id or entry.name or str(idx)
        if layout.begin_child(f"##plugin_card_{card_id}", (card_w, card_h), border=True):
            self._draw_card_info(
                layout, entry, plugin_name, plugin_state,
                is_installed, is_local, is_local_only, installed_versions, scale,
            )

            button_spacing = 6 * scale
            button_height = 25 * scale
            avail_button_w, _ = layout.get_content_region_avail()
            if is_installed:
                self._draw_card_buttons_installed(
                    layout, mgr, idx, entry, plugin_name, plugin_state,
                    is_local, has_github, is_local_only, install_in_progress,
                    avail_button_w, button_spacing, button_height, scale,
                )
            else:
                if is_local_only:
                    layout.spacing()
                    layout.end_child()
                    layout.pop_style_color(2)
                    layout.pop_style_var()
                    return

                self._draw_card_buttons_not_installed(
                    layout, mgr, idx, entry, install_in_progress,
                    has_github, avail_button_w, button_spacing, button_height, scale,
                )
        layout.end_child()

        layout.pop_style_color(2)
        layout.pop_style_var()

    def _draw_card_info(
        self, layout, entry, plugin_name, plugin_state,
        is_installed, is_local, is_local_only, installed_versions, scale,
    ):
        import lichtfeld as lf

        tr = lf.ui.tr
        palette = lf.ui.theme().palette

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
            metrics = []
            if entry.stars > 0:
                metrics.append(f"{tr('plugin_marketplace.stars')}: {entry.stars}")
            if entry.downloads > 0:
                metrics.append(f"{tr('plugin_marketplace.downloads')}: {entry.downloads}")
            if metrics:
                layout.text_colored("  |  ".join(metrics), palette.warning)

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

    def _draw_card_buttons_installed(
        self, layout, mgr, idx, entry, plugin_name, plugin_state,
        is_local, has_github, is_local_only, install_in_progress,
        avail_button_w, button_spacing, button_height, scale,
    ):
        import lichtfeld as lf

        tr = lf.ui.tr
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

    def _draw_card_buttons_not_installed(
        self, layout, mgr, idx, entry, install_in_progress,
        has_github, avail_button_w, button_spacing, button_height, scale,
    ):
        import lichtfeld as lf

        tr = lf.ui.tr
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
                self._install_plugin_from_marketplace(mgr, entry)
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
            tr("plugin_marketplace.sort.popularity_desc"),
            tr("plugin_marketplace.sort.popularity_asc"),
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

    def _filter_and_sort_entries(
        self,
        entries: List[MarketplacePluginEntry],
        installed_keys: Set[str],
        installed_names: Set[str],
    ) -> List[MarketplacePluginEntry]:
        filtered = []
        for entry in entries:
            is_installed = self._is_marketplace_entry_installed(entry, installed_keys, installed_names)
            if self._install_filter_idx == 1 and not is_installed:
                continue
            if self._install_filter_idx == 2 and is_installed:
                continue
            filtered.append(entry)

        def popularity(e):
            return (e.stars + e.downloads, e.name.lower())

        if self._sort_idx == 1:
            return sorted(filtered, key=popularity)
        if self._sort_idx == 2:
            return sorted(filtered, key=lambda e: e.name.lower())
        if self._sort_idx == 3:
            return sorted(filtered, key=lambda e: e.name.lower(), reverse=True)
        return sorted(filtered, key=popularity, reverse=True)

    def _set_status(self, message: str, is_error: bool = False):
        with self._lock:
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
                    raise RuntimeError(error_prefix)
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

    def _install_plugin_from_marketplace(self, mgr, entry: MarketplacePluginEntry):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_install(on_progress):
            if entry.registry_id:
                name = mgr.install_from_registry(entry.registry_id, on_progress=on_progress)
            else:
                name = mgr.install(entry.source_url, on_progress=on_progress)
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
            self._clear_manual_url = True
            with self._lock:
                self._url_plugin_names[self._normalize_url(clean_url)] = name
            self._invalidate_discover_cache()
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
            self._invalidate_discover_cache()
            return True

        self._run_async(
            do_load,
            tr("plugin_manager.status.loaded").format(name=name),
            tr("plugin_manager.status.load_failed"),
        )

    def _unload_plugin(self, mgr, name: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_unload(on_progress):
            on_progress(tr("plugin_manager.status.unloading").format(name=name))
            if not mgr.unload(name):
                raise RuntimeError(tr("plugin_manager.status.unload_failed"))
            self._invalidate_discover_cache()

        self._run_async(
            do_unload,
            tr("plugin_manager.status.unloaded").format(name=name),
            tr("plugin_manager.status.unload_failed"),
        )

    def _reload_plugin(self, mgr, name: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_reload(on_progress):
            mgr.unload(name)
            ok = mgr.load(name, on_progress=on_progress)
            if not ok:
                err = mgr.get_error(name) or tr("plugin_manager.status.reload_failed")
                raise RuntimeError(err)
            self._invalidate_discover_cache()
            return True

        self._run_async(
            do_reload,
            tr("plugin_manager.status.reloaded").format(name=name),
            tr("plugin_manager.status.reload_failed"),
        )

    def _update_plugin(self, mgr, name: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_update(on_progress):
            mgr.update(name, on_progress=on_progress)
            self._invalidate_discover_cache()
            return True

        self._run_async(
            do_update,
            tr("plugin_manager.status.updated").format(name=name),
            tr("plugin_manager.status.update_failed"),
        )

    def _uninstall_plugin(self, mgr, name: str):
        import lichtfeld as lf

        tr = lf.ui.tr

        def do_uninstall(on_progress):
            on_progress(tr("plugin_manager.status.uninstalling").format(name=name))
            if not mgr.uninstall(name):
                raise RuntimeError(tr("plugin_manager.status.uninstall_failed"))
            self._invalidate_discover_cache()

        self._run_async(
            do_uninstall,
            tr("plugin_manager.status.uninstalled").format(name=name),
            tr("plugin_manager.status.uninstall_failed"),
        )

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
        with self._lock:
            msg, is_err = self._status_message, self._status_is_error
        if not msg or is_err:
            return
        layout.text_colored(msg, palette.success)

    def _draw_warning_error(self, layout):
        import lichtfeld as lf

        with self._lock:
            msg, is_err = self._status_message, self._status_is_error
        if not msg or not is_err:
            return
        layout.text_colored(msg, lf.ui.theme().palette.warning)

    def _draw_output(self, layout):
        import lichtfeld as lf

        tr = lf.ui.tr
        with self._lock:
            lines = list(self._output_lines[-15:])
            in_progress = self._operation_in_progress
            status = self._status_message

        if not lines and not in_progress:
            return

        if in_progress:
            layout.progress_bar(-1.0, status or tr("plugin_manager.working"))

        if lines and layout.tree_node(tr("plugin_manager.output")):
            for line in lines:
                layout.text_wrapped(line)
            layout.tree_pop()

    def _with_local_plugins(self, entries: List[MarketplacePluginEntry], mgr) -> List[MarketplacePluginEntry]:
        merged = list(entries)
        known_keys: Set[str] = set()
        for entry in merged:
            known_keys.update(self._entry_keys(entry))

        for plugin in self._get_discovered_plugins(mgr):
            plugin_keys = self._plugin_keys(plugin.name, plugin.path.name)
            if any(k in known_keys for k in plugin_keys):
                continue

            source_path = str(plugin.path)
            merged.append(
                MarketplacePluginEntry(
                    source_url=source_path,
                    github_url="",
                    owner="",
                    repo=plugin.path.name,
                    name=plugin.name,
                    description=plugin.description or "",
                )
            )
            with self._lock:
                self._url_plugin_names[self._normalize_url(source_path)] = plugin.name
            known_keys.update(plugin_keys)

        return merged

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
