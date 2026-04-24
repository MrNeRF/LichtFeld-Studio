# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hot reload file watcher."""

import hashlib
import importlib
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Set, TYPE_CHECKING

from .plugin import PluginState

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .manager import PluginManager


class PluginWatcher:
    """Watch plugin files for changes and trigger reloads."""

    def __init__(self, manager: "PluginManager", poll_interval: float = 1.0,
                 watch_builtins: bool = True):
        self.manager = manager
        self.poll_interval = poll_interval
        self.watch_builtins = watch_builtins
        self._running = False
        self._thread: threading.Thread = None
        self._pending_reloads: Set[str] = set()
        self._pending_builtin_reloads: Set[Path] = set()
        self._plugin_reload_scheduled = False
        self._builtin_reload_scheduled = False
        self._lock = threading.Lock()
        self._file_hashes: Dict[str, Dict[Path, str]] = {}
        self._builtin_path = Path(__file__).parent
        self._builtin_mtimes: Dict[Path, float] = {}

    def start(self):
        """Start the file watcher thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the file watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _watch_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                self._check_for_changes()
                if self.watch_builtins:
                    self._check_builtin_changes()
                self._process_pending_reloads()
            except Exception as e:
                _log.error("Watcher loop error: %s", e, exc_info=True)
            time.sleep(self.poll_interval)

    def _check_for_changes(self):
        """Check all loaded plugins for file changes."""
        for name, plugin in self.manager.get_active_plugins_snapshot():
            if not plugin.info.hot_reload:
                continue
            if self._has_changes(plugin):
                with self._lock:
                    self._pending_reloads.add(name)

    def _has_changes(self, plugin) -> bool:
        """Check if any plugin files were modified."""
        plugin_name = plugin.info.name

        for py_file in plugin.info.path.rglob("*.py"):
            if ".venv" in py_file.parts:
                continue

            try:
                current_mtime = py_file.stat().st_mtime
                prev_mtime = plugin.file_mtimes.get(py_file, 0)

                if current_mtime > prev_mtime:
                    return True

                if current_mtime == prev_mtime and prev_mtime > 0:
                    if self._content_changed(plugin_name, py_file):
                        return True

            except FileNotFoundError:
                if py_file in plugin.file_mtimes:
                    return True
            except PermissionError:
                _log.warning("Permission denied: %s", py_file)
            except OSError as e:
                _log.debug("OSError checking %s: %s", py_file, e)

        return False

    def _content_changed(self, plugin_name: str, py_file: Path) -> bool:
        """Check if file content changed via SHA256 hash."""
        try:
            content = py_file.read_bytes()
            current_hash = hashlib.sha256(content).hexdigest()

            if plugin_name not in self._file_hashes:
                self._file_hashes[plugin_name] = {}

            prev_hash = self._file_hashes[plugin_name].get(py_file)
            self._file_hashes[plugin_name][py_file] = current_hash

            return prev_hash is not None and current_hash != prev_hash
        except OSError:
            return False

    def _check_builtin_changes(self):
        """Check builtin lfs_plugins files for changes."""
        for py_file in self._builtin_path.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue

            try:
                mtime = py_file.stat().st_mtime
                prev_mtime = self._builtin_mtimes.get(py_file, 0)

                if prev_mtime > 0 and mtime > prev_mtime:
                    self._queue_builtin_reload(py_file)

                self._builtin_mtimes[py_file] = mtime
            except OSError:
                continue

    def _queue_builtin_reload(self, path: Path):
        """Queue a builtin reload to run on the UI thread."""
        should_schedule = False
        with self._lock:
            self._pending_builtin_reloads.add(path)
            if not self._builtin_reload_scheduled:
                self._builtin_reload_scheduled = True
                should_schedule = True

        if not should_schedule:
            return

        if not self._schedule_on_ui_thread(self._process_pending_builtin_reloads_on_ui):
            with self._lock:
                self._builtin_reload_scheduled = False
            self._process_pending_builtin_reloads_on_ui()

    def _process_pending_builtin_reloads_on_ui(self):
        """Process queued builtin reloads on the UI thread."""
        with self._lock:
            pending = sorted(self._pending_builtin_reloads, key=str)
            self._pending_builtin_reloads.clear()
            self._builtin_reload_scheduled = False

        for path in pending:
            self._reload_builtin(path)

    def _reload_builtin(self, path: Path):
        """Reload a builtin module with full state preservation."""
        try:
            rel_path = path.relative_to(self._builtin_path.parent)
            module_parts = list(rel_path.with_suffix("").parts)
            if module_parts and module_parts[-1] == "__init__":
                module_parts.pop()
            module_name = ".".join(module_parts)

            if module_name not in sys.modules:
                return

            self._save_property_group_values()

            import lichtfeld as lf
            old_module = sys.modules[module_name]
            has_panel_registration = bool(
                getattr(old_module, "__lfs_panel_ids__", [])
                or getattr(old_module, "__lfs_panel_classes__", [])
            )
            panel_states = self._capture_panel_states(module_name, lf)

            if hasattr(old_module, "unregister"):
                try:
                    old_module.unregister()
                except Exception:
                    _log.warning("Builtin unregister failed before reload: %s", module_name, exc_info=True)

            if has_panel_registration and hasattr(lf.ui, "unregister_panels_for_module"):
                try:
                    lf.ui.unregister_panels_for_module(module_name)
                except Exception:
                    _log.warning("Builtin panel unregister failed before reload: %s", module_name, exc_info=True)

            self._unregister_menu_classes_for_module(module_name)
            module = importlib.reload(old_module)
            self._refresh_builtin_module(module_name, module, lf, panel_states)
            self._request_redraw()
            _log.info("Hot-reloaded builtin Python module: %s", module_name)

        except Exception as e:
            _log.warning("Hot-reload failed for builtin module %s: %s", path.name, e, exc_info=True)
            try:
                import lichtfeld as lf
                if hasattr(lf, "LOG"):
                    lf.LOG.warning(f"Hot-reload failed for {path.name}: {e}")
            except Exception:
                pass

    def _builtin_panel_ids(self, module_name: str) -> list[str]:
        module = sys.modules.get(module_name)
        if module is None:
            return []
        return list(getattr(module, "__lfs_panel_ids__", []))

    def _capture_panel_states(self, module_name: str, lf) -> dict[str, bool]:
        states = {}
        if not hasattr(lf.ui, "is_panel_enabled"):
            return states
        for panel_id in self._builtin_panel_ids(module_name):
            try:
                states[panel_id] = bool(lf.ui.is_panel_enabled(panel_id))
            except Exception:
                pass
        return states

    def _restore_panel_states(self, lf, states: dict[str, bool]):
        if not hasattr(lf.ui, "set_panel_enabled"):
            return
        for panel_id, enabled in states.items():
            try:
                lf.ui.set_panel_enabled(panel_id, enabled)
            except Exception:
                _log.debug("Failed to restore panel state for %s", panel_id, exc_info=True)

    def _register_panel_classes(self, module, lf, class_names: list[str]):
        for class_name in class_names:
            panel_class = getattr(module, class_name, None)
            if panel_class is None:
                _log.warning("Hot-reload could not find panel class %s.%s", module.__name__, class_name)
                continue
            lf.register_class(panel_class)

    def _refresh_builtin_module(self, module_name: str, module, lf, panel_states: dict[str, bool]):
        after_reload = getattr(module, "__lfs_after_reload__", None)
        panel_classes = list(getattr(module, "__lfs_panel_classes__", []))
        if panel_classes:
            self._register_panel_classes(module, lf, panel_classes)
            if after_reload is not None:
                after_reload(lf)
            self._restore_panel_states(lf, panel_states)
            return

        if hasattr(module, "register"):
            module.register()

        self._refresh_builtin_menus(module, lf)
        if after_reload is not None:
            after_reload(lf)

    def _unregister_menu_classes_for_module(self, module_name: str):
        try:
            from .layouts import menus
            menus.unregister_module(module_name)
        except Exception:
            _log.debug("Failed to prune menu classes before reload: %s", module_name, exc_info=True)

    def _refresh_builtin_menus(self, module, lf):
        if not hasattr(lf.ui, "register_menu"):
            return

        for class_name in getattr(module, "__lfs_menu_classes__", []):
            menu_class = getattr(module, class_name, None)
            if menu_class is None:
                _log.warning("Hot-reload could not find menu class %s.%s", module.__name__, class_name)
                continue

            try:
                lf.ui.register_menu(menu_class)
            except Exception:
                _log.warning("Failed to refresh builtin menu after reload: %s", module.__name__, exc_info=True)

    def _save_property_group_values(self):
        try:
            from .props import PropertyGroup
            for cls_name, instance in list(PropertyGroup._instances.items()):
                if instance:
                    instance._save_values()
                    PropertyGroup._instances[cls_name] = None
        except Exception:
            _log.warning("Failed to preserve property group values before reload", exc_info=True)

    def _request_redraw(self):
        try:
            import lichtfeld as lf
            if hasattr(lf.ui, "request_redraw"):
                lf.ui.request_redraw()
        except Exception:
            _log.debug("Failed to request redraw after hot reload", exc_info=True)

    def _schedule_on_ui_thread(self, callback: Callable[[], None]) -> bool:
        """Schedule callback on the LichtFeld UI thread if the runtime supports it."""
        try:
            import lichtfeld as lf
            runner = getattr(lf.ui, "schedule_on_ui_thread", None)
            if runner is None:
                runner = getattr(lf.ui, "_run_on_ui_thread", None)
            if runner is None:
                return False
            runner(callback)
            self._request_redraw()
            return True
        except Exception:
            _log.warning("Failed to schedule hot reload on UI thread", exc_info=True)
            return False

    def _process_pending_reloads(self):
        """Schedule queued plugin reloads."""
        should_schedule = False
        with self._lock:
            if self._pending_reloads and not self._plugin_reload_scheduled:
                self._plugin_reload_scheduled = True
                should_schedule = True

        if not should_schedule:
            return

        if not self._schedule_on_ui_thread(self._process_pending_plugin_reloads_on_ui):
            with self._lock:
                self._plugin_reload_scheduled = False
            self._process_pending_plugin_reloads_on_ui()

    def _process_pending_plugin_reloads_on_ui(self):
        """Process queued plugin reloads on the UI thread."""
        with self._lock:
            pending = self._pending_reloads.copy()
            self._pending_reloads.clear()
            self._plugin_reload_scheduled = False

        for name in pending:
            try:
                success = self.manager.reload(name)
                if success:
                    _log.info("Hot-reloaded plugin: %s", name)
                else:
                    error = self.manager.get_error(name)
                    _log.error("Hot-reload failed for %s: %s", name, error)
            except Exception as e:
                _log.error("Hot-reload exception for %s: %s", name, e, exc_info=True)

    def clear_plugin_hashes(self, plugin_name: str):
        """Clear stored hashes for a plugin."""
        self._file_hashes.pop(plugin_name, None)
