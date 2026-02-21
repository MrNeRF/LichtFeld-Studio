# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Built-in plugin system panels."""

from typing import List, Set
import threading

from .marketplace import (
    MarketplacePluginEntry,
    PluginMarketplaceCatalog,
    get_plugin_marketplace_urls,
)
from .types import Panel

MAX_OUTPUT_LINES = 100


class PluginManagerPanel(Panel):
    """GUI panel for managing plugins."""

    idname = "lfs.plugin_manager"
    label = "Plugin Manager"
    space = "MAIN_PANEL_TAB"
    order = 90
    MARKETPLACE_PANEL_ID = "lfs.plugin_marketplace"

    def __init__(self):
        self.github_url = ""
        self.status_message = ""
        self.status_is_error = False
        self.selected_plugin_idx = -1
        self._operation_in_progress = False
        self._output_lines: List[str] = []
        self._lock = threading.Lock()

    def draw(self, layout):
        import lichtfeld as lf
        from .manager import PluginManager
        from .plugin import PluginState

        mgr = PluginManager.instance()

        # Install from GitHub section
        if layout.collapsing_header("Install from GitHub", default_open=True):
            layout.label("GitHub URL or shorthand:")
            _, self.github_url = layout.input_text("##github_url", self.github_url)

            layout.spacing()

            # Show supported formats
            if layout.tree_node("Supported formats"):
                layout.bullet_text("https://github.com/owner/repo")
                layout.bullet_text("github:owner/repo")
                layout.bullet_text("owner/repo")
                layout.tree_pop()

            layout.spacing()

            with self._lock:
                in_progress = self._operation_in_progress

            if in_progress:
                layout.progress_bar(-1.0, self.status_message or "Working...")
                if self._output_lines:
                    if layout.tree_node("Output"):
                        for line in self._output_lines[-15:]:
                            layout.text_wrapped(line)
                        layout.tree_pop()
            else:
                if layout.button("Install Plugin", (0, 28)):
                    self._install_plugin(mgr)
                if layout.button("Explore Options", (0, 28)):
                    lf.ui.set_panel_enabled(self.MARKETPLACE_PANEL_ID, True)

        layout.separator()

        # Installed plugins section
        if layout.collapsing_header("Installed Plugins", default_open=True):
            plugins = mgr.discover()

            if not plugins:
                layout.text_colored("No plugins installed", (0.6, 0.6, 0.6, 1.0))
            else:
                plugin_names = [p.name for p in plugins]
                _, self.selected_plugin_idx = layout.listbox(
                    "##plugins", self.selected_plugin_idx, plugin_names, 5
                )

                if 0 <= self.selected_plugin_idx < len(plugins):
                    plugin = plugins[self.selected_plugin_idx]
                    state = mgr.get_state(plugin.name)

                    layout.spacing()
                    layout.label(f"Version: {plugin.version}")
                    if plugin.description:
                        layout.label(f"Description: {plugin.description}")

                    state_str = state.value if state else "not loaded"
                    if state == PluginState.ACTIVE:
                        layout.text_colored(f"Status: {state_str}", (0.3, 0.9, 0.3, 1.0))
                    elif state == PluginState.ERROR:
                        layout.text_colored(f"Status: {state_str}", (0.9, 0.3, 0.3, 1.0))
                        error = mgr.get_error(plugin.name)
                        if error:
                            layout.text_wrapped(error)
                        tb = mgr.get_traceback(plugin.name)
                        if tb and layout.tree_node("Traceback"):
                            for line in tb.strip().split("\n"):
                                layout.text_wrapped(line)
                            layout.tree_pop()
                    else:
                        layout.label(f"Status: {state_str}")

                    layout.spacing()

                    # Action buttons
                    with self._lock:
                        in_progress = self._operation_in_progress

                    if not in_progress:
                        if state == PluginState.ACTIVE:
                            if layout.button("Reload"):
                                self._reload_plugin(mgr, plugin.name)
                            layout.same_line()
                            if layout.button("Unload"):
                                self._unload_plugin(mgr, plugin.name)
                        else:
                            if layout.button("Load"):
                                self._load_plugin(mgr, plugin.name)

                        layout.same_line()
                        if layout.button("Update"):
                            self._update_plugin(mgr, plugin.name)

                        layout.same_line()
                        if layout.button("Uninstall"):
                            self._uninstall_plugin(mgr, plugin.name)

        # Status message
        if self.status_message:
            layout.separator()
            if self.status_is_error:
                layout.text_colored(self.status_message, (0.9, 0.3, 0.3, 1.0))
            else:
                layout.text_colored(self.status_message, (0.3, 0.9, 0.3, 1.0))

    def _set_status(self, message: str, is_error: bool = False):
        self.status_message = message
        self.status_is_error = is_error

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

    def _install_plugin(self, mgr):
        url = self.github_url.strip()
        if not url:
            self._set_status("Please enter a GitHub URL", True)
            return

        def do_install(on_progress):
            name = mgr.install(url, on_progress=on_progress)
            self.github_url = ""
            return name

        self._run_async(do_install, "Installed: {}", "Install failed")

    def _load_plugin(self, mgr, name: str):
        self._run_async(
            lambda cb: mgr.load(name, on_progress=cb),
            f"Loaded: {name}", "Load failed"
        )

    def _unload_plugin(self, mgr, name: str):
        try:
            mgr.unload(name)
            self._set_status(f"Unloaded: {name}")
        except Exception as e:
            self._set_status(f"Unload failed: {e}", True)

    def _reload_plugin(self, mgr, name: str):
        def do_reload(on_progress):
            mgr.unload(name)
            mgr.load(name, on_progress=on_progress)

        self._run_async(do_reload, f"Reloaded: {name}", "Reload failed")

    def _update_plugin(self, mgr, name: str):
        self._run_async(
            lambda cb: mgr.update(name, on_progress=cb),
            f"Updated: {name}", "Update failed"
        )

    def _uninstall_plugin(self, mgr, name: str):
        try:
            mgr.uninstall(name)
            self._set_status(f"Uninstalled: {name}")
            self.selected_plugin_idx = -1
        except Exception as e:
            self._set_status(f"Uninstall failed: {e}", True)


class PluginMarketplacePanel(Panel):
    """Floating marketplace browser for plugin repositories."""

    idname = "lfs.plugin_marketplace"
    label = "Plugin Marketplace"
    space = "FLOATING"
    order = 91
    options = {"DEFAULT_CLOSED"}

    FILTER_OPTIONS = [
        "All Plugins",
        "Not Installed",
        "Installed",
    ]

    GRID_COLUMNS = 3
    CARD_WIDTH = 320
    CARD_HEIGHT = 160
    CARD_SPACING = 12

    def __init__(self):
        self._catalog = PluginMarketplaceCatalog()
        self._configured_urls: List[str] = []
        self._filter_idx = 0

        self._status_message = ""
        self._status_is_error = False
        self._operation_in_progress = False
        self._output_lines: List[str] = []
        self._lock = threading.Lock()

    def draw(self, layout):
        from .manager import PluginManager

        mgr = PluginManager.instance()
        self._sync_catalog_urls()

        scale = layout.get_dpi_scale()

        layout.text_colored(
            "Browse installable plugins from GitHub repositories.",
            (0.72, 0.79, 0.9, 1.0),
        )
        layout.spacing()

        _, self._filter_idx = layout.combo("Filter", self._filter_idx, self.FILTER_OPTIONS)

        entries, is_loading = self._catalog.snapshot()
        installed_keys = self._get_installed_plugin_keys(mgr)
        entries = self._filter_entries(entries, installed_keys)

        if is_loading:
            layout.text_disabled("Fetching GitHub metadata...")

        self._draw_status(layout)

        layout.spacing()
        layout.separator()
        layout.spacing()

        if not entries:
            layout.text_disabled("No marketplace plugins configured.")
            layout.text_disabled("Edit PLUGIN_MARKETPLACE_URLS in src/python/lfs_plugins/marketplace.py.")
            return

        card_w = self.CARD_WIDTH * scale
        card_h = self.CARD_HEIGHT * scale
        spacing = self.CARD_SPACING * scale

        _, avail_h = layout.get_content_region_avail()

        scroll_height = max(220 * scale, avail_h - 24 * scale)
        with self._lock:
            install_in_progress = self._operation_in_progress

        if layout.begin_child("##plugin_marketplace_scroll", (0, scroll_height), border=False):
            row_count = (len(entries) + self.GRID_COLUMNS - 1) // self.GRID_COLUMNS
            for row in range(row_count):
                base = row * self.GRID_COLUMNS
                drawn = 0
                for col in range(self.GRID_COLUMNS):
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

        layout.push_style_var("ChildRounding", 11 * scale)
        layout.push_style_color("ChildBg", (0.12, 0.14, 0.17, 0.96))
        layout.push_style_color("Border", (0.28, 0.32, 0.39, 0.88))

        if layout.begin_child(f"##plugin_card_{idx}", (card_w, card_h), border=True):
            is_installed = self._is_marketplace_entry_installed(entry, installed_keys)
            short_name = entry.name or entry.repo or "Unknown Plugin"
            repo_label = f"{entry.owner}/{entry.repo}" if entry.owner and entry.repo else entry.repo
            description = self._truncate_text(entry.description or "No description provided.", 100)

            layout.text_colored(short_name, (0.82, 0.89, 0.99, 1.0))
            if repo_label:
                layout.text_disabled(repo_label)
            layout.text_colored(f"Stars: {entry.stars}", (0.95, 0.8, 0.35, 1.0))

            if entry.error:
                layout.text_colored("Invalid GitHub link", (0.95, 0.45, 0.45, 1.0))
            else:
                layout.text_wrapped(description)

            layout.spacing()
            layout.separator()
            layout.spacing()

            button_width = 106 * scale
            button_height = 24 * scale
            disable_install = install_in_progress or is_installed or bool(entry.error)
            install_label = "Installed" if is_installed else "Install"

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
            if layout.button_styled(f"GitHub##github_{idx}", "primary", (button_width, button_height)):
                lf.ui.open_url(entry.github_url)
            if layout.is_item_hovered():
                layout.set_mouse_cursor_hand()
        layout.end_child()

        layout.pop_style_color(2)
        layout.pop_style_var()

    def _sync_catalog_urls(self):
        current_urls = get_plugin_marketplace_urls()
        if current_urls != self._configured_urls:
            self._configured_urls = current_urls
            self._catalog.set_urls(current_urls)
            self._catalog.refresh_async(force=True)

    def _filter_entries(
        self,
        entries: List[MarketplacePluginEntry],
        installed_keys: Set[str],
    ) -> List[MarketplacePluginEntry]:
        sorted_entries = sorted(entries, key=lambda e: (e.stars, e.name.lower()), reverse=True)

        if self._filter_idx == 1:
            return [e for e in sorted_entries if not self._is_marketplace_entry_installed(e, installed_keys)]
        if self._filter_idx == 2:
            return [e for e in sorted_entries if self._is_marketplace_entry_installed(e, installed_keys)]
        return sorted_entries

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
        if not self._status_message:
            return
        color = (0.95, 0.45, 0.45, 1.0) if self._status_is_error else (0.35, 0.9, 0.45, 1.0)
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


def register_builtin_panels():
    """Initialize built-in plugin system panels."""
    try:
        import lichtfeld as lf

        # Main panel tabs (Rendering must be first)
        from .rendering_panel import RenderingPanel
        lf.register_class(RenderingPanel)

        from .training_panel import TrainingPanel
        lf.register_class(TrainingPanel)

        lf.register_class(PluginManagerPanel)

        from .scene_panel import ScenePanel
        lf.register_class(ScenePanel)

        from .toolbar import GizmoToolbar, UtilityToolbar
        lf.register_class(UtilityToolbar)
        lf.register_class(GizmoToolbar)

        from . import selection_groups
        selection_groups.register()

        from . import transform_controls
        transform_controls.register()

        from . import operators
        operators.register()

        from . import sequencer_ops
        sequencer_ops.register()

        from . import tools
        tools.register()

        from . import file_menu, edit_menu, view_menu, help_menu
        file_menu.register()
        edit_menu.register()
        view_menu.register()
        help_menu.register()

        # Floating panels
        from .export_panel import ExportPanel
        lf.register_class(ExportPanel)
        lf.ui.set_panel_enabled("lfs.export", False)

        from .about_panel import AboutPanel
        lf.register_class(AboutPanel)
        lf.ui.set_panel_enabled("lfs.about", False)

        from .getting_started_panel import GettingStartedPanel
        lf.register_class(GettingStartedPanel)
        lf.ui.set_panel_enabled("lfs.getting_started", False)

        from .image_preview_panel import ImagePreviewPanel
        lf.register_class(ImagePreviewPanel)
        lf.ui.set_panel_enabled("lfs.image_preview", False)

        from .scripts_panel import ScriptsPanel
        lf.register_class(ScriptsPanel)
        lf.ui.set_panel_enabled("lfs.scripts", False)

        from .input_settings_panel import InputSettingsPanel
        lf.register_class(InputSettingsPanel)
        lf.ui.set_panel_enabled("lfs.input_settings", False)

        lf.register_class(PluginMarketplacePanel)
        lf.ui.set_panel_enabled("lfs.plugin_marketplace", False)

        # Status bar (must be registered last, always visible)
        from .status_bar_panel import StatusBarPanel
        lf.register_class(StatusBarPanel)

        # Viewport overlays
        from .overlays import register as register_overlays
        register_overlays()
    except Exception as e:
        import traceback
        print(f"[ERROR] register_builtin_panels failed: {e}")
        traceback.print_exc()
