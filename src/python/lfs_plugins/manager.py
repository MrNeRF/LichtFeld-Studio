# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin manager for discovery, loading, and lifecycle."""

import importlib.machinery
import importlib.util
import logging
import sys
import tarfile
import tempfile
import threading
import traceback
import types
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .capabilities import CapabilityRegistry
from .errors import PluginError, PluginVersionError
from .installer import PluginInstaller, clone_from_url, uninstall_plugin, update_plugin
from .plugin import PluginInfo, PluginInstance, PluginState
from .registry import RegistryClient, RegistryPluginInfo, RegistryVersionInfo
from .watcher import PluginWatcher

try:
    import tomllib
except ImportError:
    import tomli as tomllib

try:
    from packaging.version import Version
except ImportError:
    Version = None

_log = logging.getLogger(__name__)

try:
    import lichtfeld as _lf

    class _LfLogHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                _lf.log.error(msg)
            elif record.levelno >= logging.WARNING:
                _lf.log.warn(msg)
            else:
                _lf.log.info(msg)

    _log.addHandler(_LfLogHandler())
    _log.setLevel(logging.DEBUG)
except Exception:
    pass

LICHTFELD_VERSION = "1.0.0"
MODULE_PREFIX = "lfs_plugins"


class PluginManager:
    """Singleton managing plugin discovery, loading, and lifecycle."""

    _instance: Optional["PluginManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._plugins: Dict[str, PluginInstance] = {}
        self._plugins_lock = threading.RLock()
        self._plugins_dir = Path.home() / ".lichtfeld" / "plugins"
        self._watcher: Optional[PluginWatcher] = None
        self._on_plugin_loaded: List[Callable] = []
        self._on_plugin_unloaded: List[Callable] = []
        self._registry: Optional[RegistryClient] = None

    @classmethod
    def instance(cls) -> "PluginManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def plugins_dir(self) -> Path:
        return self._plugins_dir

    @property
    def registry(self) -> RegistryClient:
        """Lazy-initialized registry client."""
        if self._registry is None:
            self._registry = RegistryClient()
        return self._registry

    def get_active_plugins_snapshot(self) -> List[tuple]:
        """Return thread-safe snapshot of active plugins."""
        with self._plugins_lock:
            return [(name, plugin) for name, plugin in self._plugins.items()
                    if plugin.state == PluginState.ACTIVE]

    def discover(self) -> List[PluginInfo]:
        """Scan plugins directory for valid plugins."""
        if not self._plugins_dir.exists():
            self._plugins_dir.mkdir(parents=True, exist_ok=True)
            return []

        plugins = []
        for entry in self._plugins_dir.iterdir():
            if entry.is_dir() and (entry / "pyproject.toml").exists():
                try:
                    plugins.append(self._parse_manifest(entry))
                except Exception as e:
                    _log.warning("Skipping invalid plugin '%s': %s", entry.name, e)
        return plugins

    def _parse_manifest(self, plugin_dir: Path) -> PluginInfo:
        """Parse pyproject.toml manifest."""
        with open(plugin_dir / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)

        project = data.get("project", {})
        lf = data.get("tool", {}).get("lichtfeld", {})

        if "tool" not in data or "lichtfeld" not in data["tool"]:
            raise ValueError("Missing [tool.lichtfeld] section")

        for field in ("name", "version", "description"):
            if field not in project:
                raise ValueError(f"Missing project.{field}")

        for field in ("auto_start", "hot_reload"):
            if field not in lf:
                raise ValueError(f"Missing tool.lichtfeld.{field}")

        authors = project.get("authors", [])
        author = authors[0].get("name", "") if authors else lf.get("author", "")

        return PluginInfo(
            name=project["name"],
            version=project["version"],
            path=plugin_dir,
            description=project["description"],
            author=author,
            entry_point=lf.get("entry_point", "__init__"),
            dependencies=project.get("dependencies", []),
            auto_start=lf["auto_start"],
            hot_reload=lf["hot_reload"],
            min_lichtfeld_version=lf.get("min_lichtfeld_version", ""),
        )

    def load(self, name: str, on_progress: Optional[Callable] = None) -> bool:
        """Load a plugin by name."""
        with self._plugins_lock:
            plugin = self._plugins.get(name)
            if not plugin:
                for info in self.discover():
                    if info.name == name:
                        plugin = PluginInstance(info=info)
                        self._plugins[name] = plugin
                        break

        if not plugin:
            raise PluginError(f"Plugin '{name}' not found")

        self._check_version_compatibility(plugin, name)

        try:
            plugin.state = PluginState.INSTALLING
            installer = PluginInstaller(plugin)
            installer.ensure_venv()
            progress_fn = on_progress or (lambda msg: _log.info("  [%s] %s", name, msg))
            installer.install_dependencies(progress_fn)

            plugin.state = PluginState.LOADING
            self._load_module(plugin)

            if hasattr(plugin.module, "on_load"):
                plugin.module.on_load()

            plugin.state = PluginState.ACTIVE
            self._update_file_mtimes(plugin)

            for cb in self._on_plugin_loaded:
                try:
                    cb(plugin.info)
                except Exception as cb_err:
                    _log.warning("on_plugin_loaded callback failed: %s", cb_err)

            return True

        except Exception as e:
            plugin.state = PluginState.ERROR
            plugin.error = str(e)
            plugin.error_traceback = traceback.format_exc()
            _log.error("load(%s) failed: %s\n%s", name, e, plugin.error_traceback)
            return False

    def _check_version_compatibility(self, plugin: PluginInstance, name: str):
        """Raise PluginVersionError if plugin requires newer LichtFeld."""
        if not plugin.info.min_lichtfeld_version or Version is None:
            return
        required = Version(plugin.info.min_lichtfeld_version)
        current = Version(LICHTFELD_VERSION)
        if current < required:
            raise PluginVersionError(f"Plugin '{name}' requires LichtFeld >= {required}, but you have {current}")

    def _load_module(self, plugin: PluginInstance):
        """Import plugin module with persistent venv path."""
        paths_to_add = []
        venv_site = self._get_venv_site_packages(plugin)
        if venv_site and venv_site.exists():
            paths_to_add.append(str(venv_site))
        paths_to_add.append(str(plugin.info.path))

        # Persistently add paths so lazy imports work later
        plugin.sys_paths = []
        for p in paths_to_add:
            if p not in sys.path:
                sys.path.insert(0, p)
                plugin.sys_paths.append(p)

        module_name = f"{MODULE_PREFIX}.{plugin.info.name}"
        importlib.invalidate_caches()

        entry_file = plugin.info.path / f"{plugin.info.entry_point}.py"
        source_code = entry_file.read_text(encoding="utf-8")
        code = compile(source_code, str(entry_file), "exec")

        module = types.ModuleType(module_name)
        module.__file__ = str(entry_file)
        module.__loader__ = importlib.machinery.SourceFileLoader(module_name, str(entry_file))
        module.__package__ = module_name
        module.__path__ = [str(plugin.info.path)]
        module.__spec__ = importlib.util.spec_from_file_location(module_name, entry_file, loader=module.__loader__, submodule_search_locations=[str(plugin.info.path)])

        sys.modules[module_name] = module

        try:
            exec(code, module.__dict__)
        except Exception:
            sys.modules.pop(module_name, None)
            # Clean up paths on failure
            for p in plugin.sys_paths:
                if p in sys.path:
                    sys.path.remove(p)
            plugin.sys_paths = []
            raise
        plugin.module = module

    def _get_venv_site_packages(self, plugin: PluginInstance) -> Optional[Path]:
        """Get site-packages path for plugin venv."""
        venv = plugin.venv_path
        if not venv or not venv.exists():
            return None

        # Unix layout
        lib_dir = venv / "lib"
        if lib_dir.exists():
            for d in lib_dir.iterdir():
                if d.name.startswith("python"):
                    sp = d / "site-packages"
                    if sp.exists():
                        return sp

        # Windows layout
        sp = venv / "Lib" / "site-packages"
        return sp if sp.exists() else None

    # Sub-package discovery relies on __path__ and __spec__ (set in
    # _load_module) which Python's PathFinder uses to locate and load
    # sub-packages on demand — no pre-registration needed.

    def unload(self, name: str) -> bool:
        """Unload a plugin."""
        with self._plugins_lock:
            plugin = self._plugins.get(name)
            if not plugin or plugin.state != PluginState.ACTIVE:
                return False

        try:
            if plugin.module and hasattr(plugin.module, "on_unload"):
                plugin.module.on_unload()

            CapabilityRegistry.instance().unregister_all_for_plugin(name)

            try:
                import lichtfeld as lf
                lf.ui.free_plugin_icons(name)
                lf.ui.free_plugin_textures(name)
            except Exception:
                pass

            try:
                from .ui.subscription_registry import SubscriptionRegistry
                SubscriptionRegistry.instance().unsubscribe_all(name)
            except Exception:
                _log.exception("Failed to cleanup signal subscriptions for '%s'", name)

            module_prefix = f"{MODULE_PREFIX}.{plugin.info.name}"
            to_remove = [m for m in sys.modules if m == module_prefix or m.startswith(f"{module_prefix}.")]
            for m in to_remove:
                sys.modules.pop(m, None)

            # Clean up sys.path entries added during load
            for p in plugin.sys_paths:
                if p in sys.path:
                    sys.path.remove(p)
            plugin.sys_paths = []

            plugin.module = None
            with self._plugins_lock:
                plugin.state = PluginState.UNLOADED

            if self._watcher:
                self._watcher.clear_plugin_hashes(name)

            for cb in self._on_plugin_unloaded:
                try:
                    cb(plugin.info)
                except Exception as cb_err:
                    _log.warning("on_plugin_unloaded callback failed: %s", cb_err)

            return True

        except Exception as e:
            plugin.error = str(e)
            with self._plugins_lock:
                plugin.state = PluginState.UNLOADED
            return False

    def reload(self, name: str) -> bool:
        """Hot reload a plugin.

        Note: PyTorch models cannot be safely unloaded (corrupts shared CUDA context).
        This reload keeps old models in memory - will leak GPU memory on each reload.
        Restart the application to fully reclaim memory.
        """
        from .utils import get_gpu_memory

        plugin = self._plugins.get(name)
        if not plugin or plugin.state != PluginState.ACTIVE:
            return self.load(name)

        mem_before = get_gpu_memory()

        try:
            if plugin.module and hasattr(plugin.module, "on_unload"):
                plugin.module.on_unload()

            CapabilityRegistry.instance().unregister_all_for_plugin(name)

            try:
                from .ui.subscription_registry import SubscriptionRegistry
                SubscriptionRegistry.instance().unsubscribe_all(name)
            except Exception:
                _log.exception("Failed to cleanup signal subscriptions for '%s'", name)

            module_prefix = f"{MODULE_PREFIX}.{plugin.info.name}"
            to_remove = [m for m in sys.modules if m == module_prefix or m.startswith(f"{module_prefix}.")]
            for m in to_remove:
                sys.modules.pop(m, None)

            self._load_module(plugin)

            if hasattr(plugin.module, "on_load"):
                plugin.module.on_load()

            self._update_file_mtimes(plugin)

            for cb in self._on_plugin_loaded:
                try:
                    cb(plugin.info)
                except Exception as cb_err:
                    _log.warning("on_plugin_loaded callback failed: %s", cb_err)

            mem_after = get_gpu_memory()
            growth_mb = (mem_after - mem_before) / (1024 * 1024)
            if growth_mb > 10:
                _log.warning(
                    f"Plugin '{name}' reload: GPU +{growth_mb:.0f}MB "
                    "(PyTorch models leak on reload - restart app to reclaim)"
                )

            return True

        except Exception as e:
            plugin.state = PluginState.ERROR
            plugin.error = str(e)
            plugin.error_traceback = traceback.format_exc()
            _log.error("reload(%s) failed: %s", name, e)
            return False

    def load_all(self) -> Dict[str, bool]:
        """Load all discovered plugins with auto_start=True."""
        discovered = self.discover()
        _log.info("load_all: discovered %d plugins: %s", len(discovered), [p.name for p in discovered])
        results = {}
        for info in discovered:
            if info.auto_start:
                _log.info("load_all: loading %s (auto_start=True)", info.name)
                success = self.load(info.name)
                results[info.name] = success
                if not success:
                    plugin = self._plugins.get(info.name)
                    if plugin and plugin.error:
                        _log.error("load_all: %s failed: %s", info.name, plugin.error)
        return results

    def list_loaded(self) -> List[str]:
        """List names of loaded plugins."""
        return [name for name, p in self._plugins.items() if p.state == PluginState.ACTIVE]

    def get_info(self, name: str) -> Optional[PluginInfo]:
        plugin = self._plugins.get(name)
        return plugin.info if plugin else None

    def get_state(self, name: str) -> Optional[PluginState]:
        plugin = self._plugins.get(name)
        return plugin.state if plugin else None

    def get_error(self, name: str) -> Optional[str]:
        plugin = self._plugins.get(name)
        return plugin.error if plugin else None

    def get_traceback(self, name: str) -> Optional[str]:
        plugin = self._plugins.get(name)
        return plugin.error_traceback if plugin else None

    def _update_file_mtimes(self, plugin: PluginInstance):
        """Record file modification times for hot reload."""
        plugin.file_mtimes.clear()
        for py_file in plugin.info.path.rglob("*.py"):
            if ".venv" not in py_file.parts:
                plugin.file_mtimes[py_file] = py_file.stat().st_mtime

    def start_watcher(self, poll_interval: float = 1.0):
        """Start hot reload file watcher."""
        if self._watcher:
            return
        self._watcher = PluginWatcher(self, poll_interval)
        self._watcher.start()

    def stop_watcher(self):
        """Stop hot reload file watcher."""
        if self._watcher:
            self._watcher.stop()
            self._watcher = None

    def on_plugin_loaded(self, callback: Callable):
        self._on_plugin_loaded.append(callback)

    def on_plugin_unloaded(self, callback: Callable):
        self._on_plugin_unloaded.append(callback)

    def install(self, url: str, on_progress: Optional[Callable[[str], None]] = None, auto_load: bool = True) -> str:
        """Install a plugin from GitHub URL."""
        plugin_dir = clone_from_url(url, self._plugins_dir, on_progress)
        info = self._parse_manifest(plugin_dir)
        if auto_load:
            self.load(info.name, on_progress)
        return info.name

    def update(self, name: str, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Update a plugin by pulling latest from git."""
        plugin = self._plugins.get(name)
        plugin_dir = plugin.info.path if plugin else self._find_plugin_dir(name)

        was_loaded = plugin and plugin.state == PluginState.ACTIVE
        if was_loaded:
            self.unload(name)

        update_plugin(plugin_dir, on_progress)

        if was_loaded:
            self.load(name, on_progress)
        return True

    def uninstall(self, name: str) -> bool:
        """Uninstall a plugin by removing its directory."""
        with self._plugins_lock:
            plugin = self._plugins.get(name)
            if plugin:
                if plugin.state == PluginState.ACTIVE:
                    self.unload(name)
                plugin_dir = plugin.info.path
                del self._plugins[name]
            else:
                plugin_dir = self._find_plugin_dir(name)

        return uninstall_plugin(plugin_dir)

    def _find_plugin_dir(self, name: str) -> Path:
        """Find plugin directory by name."""
        for info in self.discover():
            if info.name == name:
                return info.path
        raise PluginError(f"Plugin '{name}' not found")

    def search(self, query: str, compatible_only: bool = True) -> List[RegistryPluginInfo]:
        """Search plugin registry."""
        return self.registry.search(query, compatible_only, LICHTFELD_VERSION)

    def install_from_registry(
        self,
        plugin_id: str,
        version: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        auto_load: bool = True,
    ) -> str:
        """Install plugin from registry."""
        version_info = self.registry.resolve_version(plugin_id, version, LICHTFELD_VERSION)

        if version_info.git_ref:
            plugin_data = self.registry.get_plugin(plugin_id)
            repo_url = plugin_data.get("repository", "")
            if repo_url:
                return self.install(f"{repo_url}@{version_info.git_ref}", on_progress, auto_load)

        if version_info.download_url:
            return self._install_from_tarball(plugin_id, version_info, on_progress, auto_load)

        raise PluginError(f"No download method available for {plugin_id}")

    def _install_from_tarball(
        self,
        plugin_id: str,
        version_info: RegistryVersionInfo,
        on_progress: Optional[Callable[[str], None]],
        auto_load: bool,
    ) -> str:
        """Install plugin from tarball URL."""
        _, name = self.registry._parse_id(plugin_id)

        if on_progress:
            on_progress(f"Downloading {name}...")

        req = urllib.request.Request(version_info.download_url, headers={"User-Agent": "LichtFeld-PluginManager/1.0"})

        with urllib.request.urlopen(req, timeout=60) as resp:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp.write(resp.read())
                tmp_path = Path(tmp.name)

        try:
            if version_info.checksum and not self.registry.verify_checksum(tmp_path, version_info.checksum):
                raise PluginError(f"Checksum verification failed for {name}")

            target_dir = self._plugins_dir / name
            if target_dir.exists():
                raise PluginError(f"Plugin directory already exists: {target_dir}")

            if on_progress:
                on_progress(f"Extracting {name}...")

            self._extract_tarball(tmp_path, target_dir)

            info = self._parse_manifest(target_dir)
            if auto_load:
                self.load(info.name, on_progress)

            return info.name

        finally:
            tmp_path.unlink(missing_ok=True)

    def _extract_tarball(self, src: Path, dest: Path):
        """Extract tarball, stripping top-level directory if present."""
        with tarfile.open(src, "r:gz") as tar:
            members = tar.getmembers()
            if not members:
                return

            # Check if all files are under a common prefix
            first_part = members[0].name.split("/")[0] if "/" in members[0].name else None
            strip_prefix = first_part and all(m.name.startswith(f"{first_part}/") for m in members if m.name)

            for member in members:
                if strip_prefix and member.name.startswith(f"{first_part}/"):
                    member.name = member.name[len(first_part) + 1 :]
                if member.name:
                    tar.extract(member, dest)

    def check_updates(self) -> Dict[str, tuple]:
        """Check for available updates. Returns {name: (current, available)}."""
        updates = {}
        for info in self.discover():
            try:
                registry_info = self.registry.get_plugin(info.name)
                latest = registry_info.get("latest_version", "0.0.0")
                if Version is not None and Version(latest) > Version(info.version):
                    updates[info.name] = (info.version, latest)
                elif Version is None and latest != info.version:
                    updates[info.name] = (info.version, latest)
            except Exception:
                pass
        return updates
