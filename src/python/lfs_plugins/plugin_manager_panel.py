# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin Manager panel UI."""

from typing import List
import threading

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

        tr = lf.ui.tr
        theme = lf.ui.theme()
        palette = theme.palette
        mgr = PluginManager.instance()

        # Install from GitHub section
        if layout.collapsing_header(tr("plugin_manager.section.install_from_github"), default_open=True):
            layout.label(tr("plugin_manager.github_url_or_shorthand"))
            _, self.github_url = layout.input_text("##github_url", self.github_url)

            layout.spacing()

            # Show supported formats
            if layout.tree_node(tr("plugin_manager.supported_formats")):
                layout.bullet_text("https://github.com/owner/repo")
                layout.bullet_text("github:owner/repo")
                layout.bullet_text("owner/repo")
                layout.tree_pop()

            layout.spacing()

            with self._lock:
                in_progress = self._operation_in_progress

            if in_progress:
                layout.progress_bar(-1.0, self.status_message or tr("plugin_manager.working"))
                if self._output_lines:
                    if layout.tree_node(tr("plugin_manager.output")):
                        for line in self._output_lines[-15:]:
                            layout.text_wrapped(line)
                        layout.tree_pop()
            else:
                avail_w, _ = layout.get_content_region_avail()
                top_button_w = max(120.0, (avail_w - 8.0) * 0.5)
                if layout.button_styled(tr("plugin_manager.button.install_plugin"), "success", (top_button_w, 28)):
                    self._install_plugin(mgr)
                layout.same_line(spacing=8.0)
                if layout.button_styled(tr("plugin_manager.button.explore_options"), "primary", (top_button_w, 28)):
                    lf.ui.set_panel_enabled(self.MARKETPLACE_PANEL_ID, True)

        layout.separator()

        # Installed plugins section
        if layout.collapsing_header(tr("plugin_manager.section.installed_plugins"), default_open=True):
            plugins = mgr.discover()

            if not plugins:
                layout.text_colored(tr("plugin_manager.no_plugins_installed"), palette.text_dim)
            else:
                plugin_names = [p.name for p in plugins]
                _, self.selected_plugin_idx = layout.listbox(
                    "##plugins", self.selected_plugin_idx, plugin_names, 5
                )

                if 0 <= self.selected_plugin_idx < len(plugins):
                    plugin = plugins[self.selected_plugin_idx]
                    state = mgr.get_state(plugin.name)

                    layout.spacing()
                    layout.label(f"{tr('plugin_manager.version')}: {plugin.version}")
                    if plugin.description:
                        layout.label(f"{tr('plugin_manager.description')}: {plugin.description}")

                    state_str = state.value if state else tr("plugin_manager.status_not_loaded")
                    if state == PluginState.ACTIVE:
                        layout.text_colored(f"{tr('plugin_manager.status')}: {state_str}", palette.success)
                    elif state == PluginState.ERROR:
                        layout.text_colored(f"{tr('plugin_manager.status')}: {state_str}", palette.error)
                        error = mgr.get_error(plugin.name)
                        if error:
                            layout.text_wrapped(error)
                        tb = mgr.get_traceback(plugin.name)
                        if tb and layout.tree_node("Traceback"):
                            for line in tb.strip().split("\n"):
                                layout.text_wrapped(line)
                            layout.tree_pop()
                    else:
                        layout.label(f"{tr('plugin_manager.status')}: {state_str}")

                    layout.spacing()

                    # Action buttons
                    with self._lock:
                        in_progress = self._operation_in_progress

                    if not in_progress:
                        if state == PluginState.ACTIVE:
                            if layout.button_styled(tr("plugin_manager.button.reload"), "primary"):
                                self._reload_plugin(mgr, plugin.name)
                            layout.same_line()
                            if layout.button_styled(tr("plugin_manager.button.unload"), "warning"):
                                self._unload_plugin(mgr, plugin.name)
                        else:
                            if layout.button_styled(tr("plugin_manager.button.load"), "success"):
                                self._load_plugin(mgr, plugin.name)

                        layout.same_line()
                        if layout.button_styled(tr("plugin_manager.button.update"), "primary"):
                            self._update_plugin(mgr, plugin.name)

                        layout.same_line()
                        if layout.button_styled(tr("plugin_manager.button.uninstall"), "error"):
                            self._uninstall_plugin(mgr, plugin.name)

        # Status message
        if self.status_message:
            layout.separator()
            if self.status_is_error:
                layout.text_colored(self.status_message, palette.error)
            else:
                layout.text_colored(self.status_message, palette.success)

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
        import lichtfeld as lf

        tr = lf.ui.tr
        url = self.github_url.strip()
        if not url:
            self._set_status(tr("plugin_manager.error.enter_github_url"), True)
            return

        def do_install(on_progress):
            name = mgr.install(url, on_progress=on_progress)
            self.github_url = ""
            return name

        self._run_async(
            do_install,
            tr("plugin_manager.status.installed"),
            tr("plugin_manager.status.install_failed"),
        )

    def _load_plugin(self, mgr, name: str):
        import lichtfeld as lf

        tr = lf.ui.tr
        self._run_async(
            lambda cb: mgr.load(name, on_progress=cb),
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
            mgr.load(name, on_progress=on_progress)

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
            self.selected_plugin_idx = -1
        except Exception as e:
            self._set_status(f"{tr('plugin_manager.status.uninstall_failed')}: {e}", True)
