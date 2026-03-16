# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin manager for discovery, loading, and lifecycle."""

import builtins
import importlib.machinery
import importlib.util
import logging
import shutil
import sys
import threading
import time
import traceback
import types
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .capabilities import CapabilityRegistry
from .compat import (
    LICHTFELD_VERSION,
    PLUGIN_API_VERSION,
    SUPPORTED_PLUGIN_FEATURES,
    compatibility_errors,
    validate_manifest_compatibility_fields,
)
from .errors import PluginError, PluginVersionError
from .installer import (
    PluginInstaller,
    PluginSourceInfo,
    clone_from_url,
    prepare_archive_from_download_url,
    prepare_github_archive,
    read_plugin_source_metadata,
    uninstall_plugin,
    update_plugin,
    write_plugin_source_metadata,
)
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

    @staticmethod
    def _normalize_install_transport(transport: str) -> str:
        value = str(transport or "archive").strip().lower()
        if value in {"", "auto"}:
            return "archive"
        if value not in {"archive", "git"}:
            raise PluginError(f"Unsupported plugin install transport: {transport}")
        return value

    @staticmethod
    def _safe_write_source_metadata(plugin_dir: Path, source_info: PluginSourceInfo) -> None:
        try:
            write_plugin_source_metadata(plugin_dir, source_info)
        except Exception as exc:
            _log.warning("Failed to write plugin source metadata for '%s': %s", plugin_dir, exc)

    @staticmethod
    def _source_info_for_git_url(
        url: str,
        *,
        registry_id: str = "",
        version: str = "",
    ) -> PluginSourceInfo:
        from .installer import parse_github_url

        owner, repo, ref = parse_github_url(url)
        return PluginSourceInfo(
            transport="git",
            origin=url.strip(),
            github_url=f"https://github.com/{owner}/{repo}",
            owner=owner,
            repo=repo,
            requested_ref=ref or "",
            resolved_ref=ref or "",
            registry_id=registry_id,
            version=version,
            git_remote=f"https://github.com/{owner}/{repo}.git",
        )

    def _finalize_new_plugin_install(
        self,
        staging_dir: Path,
        source_info: PluginSourceInfo,
        on_progress: Optional[Callable[[str], None]],
        auto_load: bool,
    ) -> str:
        try:
            info = self._parse_manifest(staging_dir)
            target_dir = self._plugins_dir / info.name
            if target_dir.exists() and target_dir != staging_dir:
                raise PluginError(f"Plugin directory already exists: {target_dir}")
            if staging_dir != target_dir:
                staging_dir.replace(target_dir)
            self._safe_write_source_metadata(target_dir, source_info)
            if auto_load:
                self.load(info.name, on_progress)
            return info.name
        except Exception:
            if staging_dir.exists() and staging_dir.parent == self._plugins_dir and staging_dir.name.startswith("."):
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def _replace_plugin_install(
        self,
        plugin_dir: Path,
        staging_dir: Path,
        source_info: PluginSourceInfo,
    ) -> str:
        current_info = self._parse_manifest(plugin_dir)
        new_info = self._parse_manifest(staging_dir)
        if new_info.name != current_info.name:
            raise PluginError(
                f"Updated plugin manifest changed project.name from '{current_info.name}' to '{new_info.name}'"
            )

        backup_dir = self._plugins_dir / f".{plugin_dir.name}.backup-{uuid.uuid4().hex[:8]}"
        try:
            plugin_dir.replace(backup_dir)
            staging_dir.replace(plugin_dir)

            old_venv = backup_dir / ".venv"
            new_venv = plugin_dir / ".venv"
            if old_venv.exists() and not new_venv.exists():
                try:
                    old_venv.replace(new_venv)
                except Exception as exc:
                    _log.warning("Failed to preserve plugin venv during update for '%s': %s", current_info.name, exc)

            self._safe_write_source_metadata(plugin_dir, source_info)
            return current_info.name
        except Exception:
            if not plugin_dir.exists() and backup_dir.exists():
                backup_dir.replace(plugin_dir)
            raise
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)

    def get_active_plugins_snapshot(self) -> List[tuple]:
        """Return thread-safe snapshot of active plugins."""
        with self._plugins_lock:
            return [(name, plugin) for name, plugin in self._plugins.items()
                    if plugin.state == PluginState.ACTIVE]

    def discover(self) -> List[PluginInfo]:
        """Scan plugins directory for valid plugins."""
        if not self._plugins_dir.exists():
            self._plugins_dir.mkdir(parents=True, exist_ok=True)

        plugins = []
        for entry in self._plugins_dir.iterdir():
            if entry.is_dir() and (entry / "pyproject.toml").exists():
                try:
                    plugins.append(self._parse_manifest(entry))
                except Exception as e:
                    _log.warning("Skipping plugin '%s': invalid manifest. %s", entry.name, e)
        return plugins

    def pre_register(self, discovered: List[PluginInfo]) -> None:
        """Pre-register discovered plugins so load() skips re-discovery."""
        with self._plugins_lock:
            for info in discovered:
                if info.name not in self._plugins:
                    self._plugins[info.name] = PluginInstance(info=info)

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

        if "hot_reload" not in lf:
            raise ValueError("Missing tool.lichtfeld.hot_reload")
        compatibility_errors_in_manifest = validate_manifest_compatibility_fields(lf)
        if compatibility_errors_in_manifest:
            raise ValueError(compatibility_errors_in_manifest[0].removeprefix("pyproject.toml: "))

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
            auto_start=lf.get("auto_start", False),
            hot_reload=lf["hot_reload"],
            plugin_api=lf["plugin_api"].strip(),
            lichtfeld_version=lf["lichtfeld_version"].strip(),
            required_features=list(lf["required_features"]),
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
            t0 = time.monotonic()
            plugin.state = PluginState.INSTALLING
            installer = PluginInstaller(plugin)
            installer.ensure_venv()
            t_venv = time.monotonic()
            progress_fn = on_progress or (lambda msg: _log.info("  [%s] %s", name, msg))
            installer.install_dependencies(progress_fn)
            t_deps = time.monotonic()

            plugin.state = PluginState.LOADING
            self._load_module(plugin)
            t_module = time.monotonic()

            if hasattr(plugin.module, "on_load"):
                plugin.module.on_load()
            t_onload = time.monotonic()

            plugin.state = PluginState.ACTIVE
            self._update_file_mtimes(plugin)

            _log.info(
                "load(%s) timing: venv=%.0fms deps=%.0fms module=%.0fms on_load=%.0fms total=%.0fms",
                name,
                (t_venv - t0) * 1000,
                (t_deps - t_venv) * 1000,
                (t_module - t_deps) * 1000,
                (t_onload - t_module) * 1000,
                (t_onload - t0) * 1000,
            )

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
        """Raise PluginVersionError if plugin compatibility contract is not satisfied."""
        issues = compatibility_errors(
            plugin.info.plugin_api,
            plugin.info.lichtfeld_version,
            plugin.info.required_features,
            current_plugin_api=PLUGIN_API_VERSION,
            current_lichtfeld_version=LICHTFELD_VERSION,
            supported_features=SUPPORTED_PLUGIN_FEATURES,
        )
        if issues:
            raise PluginVersionError(f"Plugin '{name}' {'; '.join(issues)}")

    _SLOW_IMPORT_THRESHOLD_MS = 100
    _SLOW_TOTAL_THRESHOLD_MS = 500

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
            self._exec_with_import_audit(code, module, plugin.info.name)
        except Exception:
            sys.modules.pop(module_name, None)
            # Clean up paths on failure
            for p in plugin.sys_paths:
                if p in sys.path:
                    sys.path.remove(p)
            plugin.sys_paths = []
            raise
        plugin.module = module

    def _exec_with_import_audit(self, code, module, plugin_name: str):
        """Execute plugin code while tracking slow top-level imports."""
        slow_imports: list[tuple[str, float]] = []
        original_import = builtins.__import__
        depth = 0

        def _auditing_import(name, *args, **kwargs):
            nonlocal depth
            already_loaded = name in sys.modules
            if already_loaded or depth > 0:
                return original_import(name, *args, **kwargs)
            depth += 1
            t = time.monotonic()
            try:
                return original_import(name, *args, **kwargs)
            finally:
                elapsed_ms = (time.monotonic() - t) * 1000
                depth -= 1
                if elapsed_ms >= self._SLOW_IMPORT_THRESHOLD_MS:
                    slow_imports.append((name, elapsed_ms))

        t0 = time.monotonic()
        builtins.__import__ = _auditing_import
        try:
            exec(code, module.__dict__)
        finally:
            builtins.__import__ = original_import

        total_ms = (time.monotonic() - t0) * 1000
        if total_ms >= self._SLOW_TOTAL_THRESHOLD_MS:
            lines = [f"Plugin '{plugin_name}' module load took {total_ms:.0f}ms — defer heavy imports to speed up startup:"]
            for name, ms in sorted(slow_imports, key=lambda x: -x[1]):
                lines.append(f"  {name}: {ms:.0f}ms")
            _log.warning("\n".join(lines))

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

            try:
                import lichtfeld as lf
                lf.ui.unregister_panels_for_module(f"{MODULE_PREFIX}.{plugin.info.name}")
            except Exception:
                pass

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

            try:
                import lichtfeld as lf
                lf.ui.unregister_panels_for_module(f"{MODULE_PREFIX}.{plugin.info.name}")
            except Exception:
                pass

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
        """Load all discovered plugins where the user enabled load_on_startup."""
        from .settings import SettingsManager

        discovered = self.discover()
        self.pre_register(discovered)
        _log.info("load_all: discovered %d plugins: %s", len(discovered), [p.name for p in discovered])
        results = {}
        for info in discovered:
            prefs = SettingsManager.instance().get(info.name)
            if prefs.get("load_on_startup", False):
                _log.info("load_all: loading %s (user-enabled)", info.name)
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

    def install(
        self,
        url: str,
        on_progress: Optional[Callable[[str], None]] = None,
        auto_load: bool = True,
        transport: str = "archive",
        source_info: Optional[PluginSourceInfo] = None,
    ) -> str:
        """Install a plugin from GitHub using the selected transport."""
        mode = self._normalize_install_transport(transport)
        if mode == "git":
            plugin_dir = clone_from_url(url, self._plugins_dir, on_progress)
            info = self._parse_manifest(plugin_dir)
            self._safe_write_source_metadata(plugin_dir, source_info or self._source_info_for_git_url(url))
            if auto_load:
                self.load(info.name, on_progress)
            return info.name

        staging_dir, source_info = prepare_github_archive(url, self._plugins_dir, on_progress)
        return self._finalize_new_plugin_install(staging_dir, source_info, on_progress, auto_load)

    def update(self, name: str, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Update a plugin according to its recorded install transport."""
        plugin = self._plugins.get(name)
        plugin_dir = plugin.info.path if plugin else self._find_plugin_dir(name)

        was_loaded = plugin and plugin.state == PluginState.ACTIVE
        if was_loaded:
            self.unload(name)

        source_info = read_plugin_source_metadata(plugin_dir)
        if source_info and source_info.transport == "archive":
            if source_info.registry_id:
                self._update_archive_plugin_from_registry(plugin_dir, source_info, on_progress)
            else:
                self._update_archive_plugin_from_github(plugin_dir, source_info, on_progress)
        elif source_info and source_info.transport == "git":
            update_plugin(plugin_dir, on_progress)
        elif (plugin_dir / ".git").exists():
            update_plugin(plugin_dir, on_progress)
        else:
            raise PluginError(
                f"Plugin '{name}' was not installed from a known remote source and cannot be updated automatically"
            )

        refreshed_info = self._parse_manifest(plugin_dir)
        if plugin:
            plugin.info = refreshed_info
            plugin.error = None
            plugin.error_traceback = None

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
        return self.registry.search(
            query,
            compatible_only,
            LICHTFELD_VERSION,
            plugin_api=PLUGIN_API_VERSION,
            supported_features=SUPPORTED_PLUGIN_FEATURES,
        )

    def install_from_registry(
        self,
        plugin_id: str,
        version: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        auto_load: bool = True,
        transport: str = "archive",
    ) -> str:
        """Install plugin from registry."""
        mode = self._normalize_install_transport(transport)
        version_info = self.registry.resolve_version(
            plugin_id,
            version,
            LICHTFELD_VERSION,
            plugin_api=PLUGIN_API_VERSION,
            supported_features=SUPPORTED_PLUGIN_FEATURES,
        )
        plugin_data = self.registry.get_plugin(plugin_id)
        repo_url = plugin_data.get("repository", "")

        if mode == "git":
            if version_info.git_ref and repo_url:
                install_url = f"{repo_url}@{version_info.git_ref}"
                source_info = self._source_info_for_git_url(
                    install_url,
                    registry_id=plugin_id,
                    version=version_info.version,
                )
                return self.install(
                    install_url,
                    on_progress,
                    auto_load,
                    transport="git",
                    source_info=source_info,
                )
            raise PluginError(f"No git install method available for {plugin_id}")

        if version_info.download_url:
            source_info = PluginSourceInfo(
                transport="archive",
                origin=repo_url or version_info.download_url,
                registry_id=plugin_id,
                version=version_info.version,
                archive_url=version_info.download_url,
                checksum=version_info.checksum,
            )
            staging_dir = prepare_archive_from_download_url(
                version_info.download_url,
                self._plugins_dir,
                temp_prefix=f".{plugin_id.replace(':', '-')}-",
                on_progress=on_progress,
                archive_validator=(
                    (lambda path: self._verify_registry_archive_checksum(path, version_info, plugin_id))
                    if version_info.checksum
                    else None
                ),
            )
            return self._finalize_new_plugin_install(staging_dir, source_info, on_progress, auto_load)

        if repo_url:
            install_url = f"{repo_url}@{version_info.git_ref}" if version_info.git_ref else repo_url
            staging_dir, source_info = prepare_github_archive(install_url, self._plugins_dir, on_progress)
            source_info = PluginSourceInfo(
                transport=source_info.transport,
                origin=source_info.origin,
                github_url=source_info.github_url,
                owner=source_info.owner,
                repo=source_info.repo,
                requested_ref=source_info.requested_ref,
                resolved_ref=source_info.resolved_ref,
                registry_id=plugin_id,
                version=version_info.version,
                archive_url=source_info.archive_url,
                checksum=version_info.checksum,
            )
            return self._finalize_new_plugin_install(staging_dir, source_info, on_progress, auto_load)

        raise PluginError(f"No download method available for {plugin_id}")

    def _verify_registry_archive_checksum(
        self,
        archive_path: Path,
        version_info: RegistryVersionInfo,
        plugin_id: str,
    ) -> None:
        if version_info.checksum and not self.registry.verify_checksum(archive_path, version_info.checksum):
            raise PluginError(f"Checksum verification failed for {plugin_id}")

    def _update_archive_plugin_from_registry(
        self,
        plugin_dir: Path,
        source_info: PluginSourceInfo,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        plugin_id = source_info.registry_id
        if not plugin_id:
            raise PluginError(f"Missing registry id for archive-installed plugin: {plugin_dir.name}")

        version_info = self.registry.resolve_version(
            plugin_id,
            None,
            LICHTFELD_VERSION,
            plugin_api=PLUGIN_API_VERSION,
            supported_features=SUPPORTED_PLUGIN_FEATURES,
        )
        plugin_data = self.registry.get_plugin(plugin_id)
        repo_url = plugin_data.get("repository", "")

        if version_info.download_url:
            new_source_info = PluginSourceInfo(
                transport="archive",
                origin=repo_url or version_info.download_url,
                registry_id=plugin_id,
                version=version_info.version,
                archive_url=version_info.download_url,
                checksum=version_info.checksum,
            )
            staging_dir = prepare_archive_from_download_url(
                version_info.download_url,
                self._plugins_dir,
                temp_prefix=f".{plugin_dir.name}-",
                on_progress=on_progress,
                archive_validator=(
                    (lambda path: self._verify_registry_archive_checksum(path, version_info, plugin_id))
                    if version_info.checksum
                    else None
                ),
            )
            self._replace_plugin_install(plugin_dir, staging_dir, new_source_info)
            return

        if repo_url:
            install_url = f"{repo_url}@{version_info.git_ref}" if version_info.git_ref else repo_url
            staging_dir, github_source_info = prepare_github_archive(install_url, self._plugins_dir, on_progress)
            new_source_info = PluginSourceInfo(
                transport="archive",
                origin=github_source_info.origin,
                github_url=github_source_info.github_url,
                owner=github_source_info.owner,
                repo=github_source_info.repo,
                requested_ref=github_source_info.requested_ref,
                resolved_ref=github_source_info.resolved_ref,
                registry_id=plugin_id,
                version=version_info.version,
                archive_url=github_source_info.archive_url,
                checksum=version_info.checksum,
            )
            self._replace_plugin_install(plugin_dir, staging_dir, new_source_info)
            return

        raise PluginError(f"No archive update method available for {plugin_id}")

    def _update_archive_plugin_from_github(
        self,
        plugin_dir: Path,
        source_info: PluginSourceInfo,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        origin = source_info.origin or source_info.github_url
        if not origin:
            raise PluginError(f"Missing GitHub source for archive-installed plugin: {plugin_dir.name}")
        install_url = origin
        if source_info.requested_ref and "@" not in install_url and "/tree/" not in install_url:
            install_url = f"{install_url}@{source_info.requested_ref}"
        staging_dir, new_source_info = prepare_github_archive(install_url, self._plugins_dir, on_progress)
        merged_source_info = PluginSourceInfo(
            transport="archive",
            origin=new_source_info.origin,
            github_url=new_source_info.github_url,
            owner=new_source_info.owner,
            repo=new_source_info.repo,
            requested_ref=new_source_info.requested_ref,
            resolved_ref=new_source_info.resolved_ref,
            registry_id=source_info.registry_id,
            version=source_info.version,
            archive_url=new_source_info.archive_url,
            checksum=source_info.checksum,
        )
        self._replace_plugin_install(plugin_dir, staging_dir, merged_source_info)

    def check_updates(self) -> Dict[str, tuple]:
        """Check for available updates. Returns {name: (current, available)}."""
        updates = {}
        for info in self.discover():
            try:
                registry_plugin_id = self._resolve_registry_plugin_id(info.name)
                if not registry_plugin_id:
                    continue
                registry_info = self.registry.get_plugin(registry_plugin_id)
                latest = registry_info.get("latest_version", "0.0.0")
                if Version is not None and Version(latest) > Version(info.version):
                    updates[info.name] = (info.version, latest)
                elif Version is None and latest != info.version:
                    updates[info.name] = (info.version, latest)
            except Exception:
                pass
        return updates

    def _resolve_registry_plugin_id(self, plugin_name: str) -> Optional[str]:
        matches = [entry for entry in self.search(plugin_name, compatible_only=False) if entry.name == plugin_name]
        if len(matches) == 1:
            return matches[0].full_id
        return None
