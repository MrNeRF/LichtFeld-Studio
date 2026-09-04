# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin manager for discovery, loading, and lifecycle."""

import importlib.machinery
import importlib.util
import logging
import os
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
from .errors import PluginError, PluginLoadCancelled, PluginVersionError
from .installer import (
    PluginInstaller,
    PluginSourceInfo,
    clone_from_url,
    prepare_archive_from_download_url,
    prepare_github_archive,
    read_plugin_source_metadata,
    normalize_repo_name,
    parse_github_url,
    uninstall_plugin,
    update_plugin,
    write_plugin_source_metadata,
)
from .plugin import PluginInfo, PluginInstance, PluginState, iter_plugin_watch_files, validate_plugin_name
from .plugin_localization import PluginCatalogError, read_plugin_catalogs
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
        self._plugin_locks: Dict[str, threading.RLock] = {}
        self._plugin_locks_lock = threading.Lock()
        self._plugins_dir = Path(
            os.environ.get(
                "LFS_RESOLVED_PLUGIN_DIR",
                Path.home() / ".lichtfeld" / "plugins",
            )
        )
        self._watcher: Optional[PluginWatcher] = None
        self._on_plugin_loaded: List[Callable] = []
        self._on_plugin_unloaded: List[Callable] = []
        self._on_plugin_changed: List[Callable] = []
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
    def _safe_write_source_metadata(
        plugin_dir: Path,
        source_info: PluginSourceInfo,
        *,
        strict: bool = False,
    ) -> None:
        try:
            write_plugin_source_metadata(plugin_dir, source_info)
        except Exception as exc:
            _log.error("Failed to write plugin source metadata for '%s': %s", plugin_dir, exc)
            if strict:
                raise PluginError(
                    f"Failed to write plugin source metadata for '{plugin_dir}': {exc}"
                ) from exc

    @staticmethod
    def _validate_plugin_name(name: str) -> str:
        """Validate a name before using it as a plugin directory component."""
        try:
            return validate_plugin_name(name)
        except ValueError as exc:
            raise PluginError(str(exc)) from exc

    def _plugin_lifecycle(self, name: str):
        """Return a reentrant lock scoped to one plugin name."""
        name = self._validate_plugin_name(name)
        with self._plugin_locks_lock:
            lock = self._plugin_locks.get(name)
            if lock is None:
                lock = threading.RLock()
                self._plugin_locks[name] = lock
            return lock

    def _notify_plugin_changed(self, info: PluginInfo) -> None:
        for cb in list(self._on_plugin_changed):
            try:
                cb(info)
            except Exception as cb_err:
                _log.warning("on_plugin_changed callback failed: %s", cb_err)

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
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> str:
        plugin: Optional[PluginInstance] = None
        info: Optional[PluginInfo] = None
        target_dir: Optional[Path] = None
        published = False
        try:
            self._raise_if_cancelled(should_cancel)
            info = self._parse_manifest(staging_dir)
            name = self._validate_plugin_name(info.name)
            with self._plugin_lifecycle(name):
                target_dir = self._plugins_dir / name
                if target_dir.exists() and target_dir != staging_dir:
                    raise PluginError(f"Plugin directory already exists: {target_dir}")

                self._safe_write_source_metadata(staging_dir, source_info)
                if auto_load:
                    plugin = PluginInstance(info=info)
                    with self._plugins_lock:
                        if name in self._plugins:
                            raise PluginError(f"Plugin '{name}' is already managed")
                        self._plugins[name] = plugin
                    loaded = self.load(name, on_progress, should_cancel=should_cancel)
                    if not loaded:
                        detail = plugin.error or f"Failed to load plugin '{name}'"
                        raise PluginError(detail)
                    self._raise_if_cancelled(should_cancel)

                if staging_dir != target_dir:
                    staging_dir.replace(target_dir)
                    published = True
                if plugin is not None:
                    self._rebase_published_plugin(plugin, staging_dir, target_dir)
                    self._update_file_mtimes(plugin)
                info.path = target_dir
                self._notify_plugin_changed(info)
                return name
        except BaseException:
            if plugin is not None:
                if plugin.state == PluginState.ACTIVE:
                    self.unload(plugin.info.name)
                with self._plugins_lock:
                    if self._plugins.get(plugin.info.name) is plugin:
                        del self._plugins[plugin.info.name]
            if published and target_dir is not None and target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            if staging_dir.exists() and staging_dir.parent == self._plugins_dir and staging_dir.name.startswith("."):
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise
            if staging_dir.exists() and staging_dir.parent == self._plugins_dir and staging_dir.name.startswith("."):
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    @staticmethod
    def _rebase_published_plugin(
        plugin: PluginInstance,
        staging_dir: Path,
        target_dir: Path,
    ) -> None:
        """Move runtime paths from the private staging tree to its public path."""
        old_root = str(staging_dir)
        new_root = str(target_dir)

        def rebase(path: str) -> str:
            if path == old_root or path.startswith(old_root + os.sep):
                return new_root + path[len(old_root):]
            return path

        plugin.info.path = target_dir
        if plugin.venv_path is not None:
            plugin.venv_path = Path(rebase(str(plugin.venv_path)))
        old_sys_paths = list(plugin.sys_paths)
        plugin.sys_paths = [rebase(path) for path in old_sys_paths]
        for old_path, new_path in zip(old_sys_paths, plugin.sys_paths):
            if old_path == new_path:
                continue
            try:
                index = sys.path.index(old_path)
            except ValueError:
                continue
            if new_path not in sys.path:
                sys.path[index] = new_path
            else:
                sys.path.pop(index)
        module_prefix = f"{MODULE_PREFIX}.{plugin.info.name}"
        for module_name, module in list(sys.modules.items()):
            if module is None or (
                module_name != module_prefix
                and not module_name.startswith(f"{module_prefix}.")
            ):
                continue
            if hasattr(module, "__file__") and module.__file__:
                module.__file__ = rebase(module.__file__)
            if hasattr(module, "__path__"):
                module.__path__ = [rebase(path) for path in module.__path__]
            spec = getattr(module, "__spec__", None)
            if spec is not None:
                if spec.origin:
                    spec.origin = rebase(spec.origin)
                if spec.submodule_search_locations:
                    spec.submodule_search_locations[:] = [
                        rebase(path) for path in spec.submodule_search_locations
                    ]
            loader = getattr(module, "__loader__", None)
            if loader is not None and hasattr(loader, "path"):
                loader.path = rebase(loader.path)
        plugin.file_mtimes.clear()

    def _replace_plugin_install(
        self,
        plugin_dir: Path,
        staging_dir: Path,
        source_info: PluginSourceInfo,
        after_replace: Optional[Callable[[], object]] = None,
    ) -> str:
        current_info = self._parse_manifest(plugin_dir)
        new_info = self._parse_manifest(staging_dir)
        self._validate_plugin_name(current_info.name)
        self._validate_plugin_name(new_info.name)
        if new_info.name != current_info.name:
            raise PluginError(
                f"Updated plugin manifest changed project.name from '{current_info.name}' to '{new_info.name}'"
            )

        backup_dir = self._plugins_dir / f".{plugin_dir.name}.backup-{uuid.uuid4().hex[:8]}"
        moved_venv = False
        committed = False
        try:
            plugin_dir.replace(backup_dir)
            staging_dir.replace(plugin_dir)

            old_venv = backup_dir / ".venv"
            new_venv = plugin_dir / ".venv"
            if old_venv.exists() and not new_venv.exists():
                old_venv.replace(new_venv)
                moved_venv = True

            self._safe_write_source_metadata(plugin_dir, source_info, strict=True)
            if after_replace is not None:
                result = after_replace()
                if result is False:
                    raise PluginError(f"Failed to reload updated plugin '{current_info.name}'")

            # The old tree is the rollback point until every replacement step,
            # including metadata and optional reload, has completed.
            shutil.rmtree(backup_dir)
            committed = True
            return current_info.name
        except Exception:
            try:
                if moved_venv and new_venv.exists() and not old_venv.exists():
                    new_venv.replace(old_venv)
                if plugin_dir.exists():
                    shutil.rmtree(plugin_dir)
                if backup_dir.exists() and not plugin_dir.exists():
                    backup_dir.replace(plugin_dir)
            except Exception:
                _log.exception("Failed to restore plugin backup for '%s'", current_info.name)
            raise
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            if committed and backup_dir.exists():
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

        validate_plugin_name(project["name"])

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

    @staticmethod
    def _raise_if_cancelled(should_cancel: Optional[Callable[[], bool]]) -> None:
        if should_cancel and should_cancel():
            raise PluginLoadCancelled("Plugin loading cancelled")

    @staticmethod
    def _emit_stage(
        on_stage: Optional[Callable[[str, str], None]],
        phase: str,
        detail: str,
    ) -> None:
        if on_stage:
            on_stage(phase, detail)

    def load(
        self,
        name: str,
        on_progress: Optional[Callable[[str], None]] = None,
        on_stage: Optional[Callable[[str, str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """Load a plugin by name."""
        with self._plugin_lifecycle(name):
            return self._load_locked(name, on_progress, on_stage, should_cancel)

    def _load_locked(
        self,
        name: str,
        on_progress: Optional[Callable[[str], None]],
        on_stage: Optional[Callable[[str, str], None]],
        should_cancel: Optional[Callable[[], bool]],
    ) -> bool:
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

        if plugin.state == PluginState.ACTIVE:
            return True

        self._check_version_compatibility(plugin, name)

        try:
            t0 = time.monotonic()
            plugin.error = None
            plugin.error_traceback = None
            self._raise_if_cancelled(should_cancel)
            plugin.state = PluginState.INSTALLING
            installer = PluginInstaller(plugin)
            progress_fn = on_progress or (lambda msg: _log.info("  [%s] %s", name, msg))
            self._emit_stage(on_stage, "environment", f"Preparing environment for {name}")
            installer.ensure_venv(progress_fn, should_cancel)
            t_venv = time.monotonic()
            self._raise_if_cancelled(should_cancel)
            self._emit_stage(on_stage, "dependencies", f"Installing dependencies for {name}")
            installer.install_dependencies(progress_fn, should_cancel)
            t_deps = time.monotonic()

            self._raise_if_cancelled(should_cancel)
            plugin.state = PluginState.LOADING
            self._emit_stage(on_stage, "import", f"Importing {name}")
            self._register_localization_catalogs(plugin)
            self._load_module(plugin)
            t_module = time.monotonic()

            self._raise_if_cancelled(should_cancel)
            self._emit_stage(on_stage, "activation", f"Activating {name}")
            if hasattr(plugin.module, "on_load"):
                plugin.module.on_load()
            t_onload = time.monotonic()

            self._raise_if_cancelled(should_cancel)
            plugin.state = PluginState.ACTIVE
            self._update_file_mtimes(plugin)
            self._emit_stage(on_stage, "complete", f"Loaded {name}")

            _log.info(
                "load(%s) timing: venv=%.0fms deps=%.0fms module=%.0fms on_load=%.0fms total=%.0fms",
                name,
                (t_venv - t0) * 1000,
                (t_deps - t_venv) * 1000,
                (t_module - t_deps) * 1000,
                (t_onload - t_module) * 1000,
                (t_onload - t0) * 1000,
            )

            for cb in list(self._on_plugin_loaded):
                try:
                    cb(plugin.info)
                except Exception as cb_err:
                    _log.warning("on_plugin_loaded callback failed: %s", cb_err)

            return True

        except PluginLoadCancelled:
            self._rollback_failed_load(plugin)
            plugin.state = PluginState.UNLOADED
            plugin.error = None
            plugin.error_traceback = None
            raise
        except Exception as e:
            error_traceback = traceback.format_exc()
            self._rollback_failed_load(plugin)
            plugin.state = PluginState.ERROR
            plugin.error = str(e)
            plugin.error_traceback = error_traceback
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
        module.__lfs_plugin_name__ = plugin.info.name

        sys.modules[module_name] = module

        try:
            self._exec_module_timed(code, module, plugin.info.name)
        except Exception:
            sys.modules.pop(module_name, None)
            # Clean up paths on failure
            for p in plugin.sys_paths:
                if p in sys.path:
                    sys.path.remove(p)
            plugin.sys_paths = []
            raise
        plugin.module = module

    def _exec_module_timed(self, code, module, plugin_name: str):
        """Execute plugin code and report owner-scoped total import time."""
        t0 = time.monotonic()
        exec(code, module.__dict__)

        total_ms = (time.monotonic() - t0) * 1000
        if total_ms >= self._SLOW_TOTAL_THRESHOLD_MS:
            _log.warning("Plugin '%s' module load took %.0fms", plugin_name, total_ms)

    def _rollback_failed_load(self, plugin: PluginInstance) -> None:
        """Best-effort cleanup after import or activation fails."""
        name = plugin.info.name
        if plugin.module and hasattr(plugin.module, "on_unload"):
            try:
                plugin.module.on_unload()
            except Exception:
                _log.exception("Failed to run on_unload while rolling back '%s'", name)

        self._unregister_localization_catalogs(plugin)

        try:
            CapabilityRegistry.instance().unregister_all_for_plugin(name)
        except Exception:
            _log.exception("Failed to unregister capabilities while rolling back '%s'", name)

        try:
            from .ui.subscription_registry import SubscriptionRegistry

            SubscriptionRegistry.instance().unsubscribe_all(name)
        except Exception:
            _log.exception("Failed to remove subscriptions while rolling back '%s'", name)

        try:
            import lichtfeld as lf
        except Exception:
            _log.exception("Failed to access UI registrations while rolling back '%s'", name)
        else:
            cleanup_calls = (
                (
                    "panels",
                    lambda: lf.ui.unregister_panels_for_module(f"{MODULE_PREFIX}.{name}"),
                ),
                ("icons", lambda: lf.ui.free_plugin_icons(name)),
                ("textures", lambda: lf.ui.free_plugin_textures(name)),
            )
            for resource, cleanup in cleanup_calls:
                try:
                    cleanup()
                except Exception:
                    _log.exception(
                        "Failed to remove %s while rolling back '%s'", resource, name
                    )

        module_prefix = f"{MODULE_PREFIX}.{name}"
        for module_name in [
            item
            for item in sys.modules
            if item == module_prefix or item.startswith(f"{module_prefix}.")
        ]:
            sys.modules.pop(module_name, None)

        for path in plugin.sys_paths:
            if path in sys.path:
                sys.path.remove(path)
        plugin.sys_paths = []
        plugin.module = None

    def _register_localization_catalogs(self, plugin: PluginInstance) -> None:
        """Validate and register plugin catalogs before module activation."""
        if plugin.localization_tokens:
            self._unregister_localization_catalogs(plugin)
            if plugin.localization_tokens:
                raise PluginError(
                    f"Plugin '{plugin.info.name}' has localization catalogs that could not be cleared"
                )

        try:
            bundle = read_plugin_catalogs(plugin.info.path, plugin.info.name)
        except (OSError, PluginCatalogError) as exc:
            raise PluginError(
                f"Plugin '{plugin.info.name}' localization is invalid: {exc}"
            ) from exc

        if not bundle.catalogs:
            return

        try:
            import lichtfeld as lf

            register_catalog = lf.ui._loc_register_catalog
            for language, entries in sorted(bundle.catalogs.items()):
                token = int(register_catalog(bundle.owner_id, language, entries))
                if token <= 0:
                    raise RuntimeError("host returned an invalid localization ownership token")
                plugin.localization_tokens.append(token)
        except Exception as exc:
            self._unregister_localization_catalogs(plugin)
            raise PluginError(
                f"Plugin '{plugin.info.name}' localization registration failed: {exc}"
            ) from exc

    @staticmethod
    def _unregister_localization_catalogs(plugin: PluginInstance) -> None:
        """Best-effort removal of exactly the catalogs registered by a plugin."""
        if not plugin.localization_tokens:
            return

        try:
            import lichtfeld as lf

            unregister_catalog = lf.ui._loc_unregister_catalog
        except Exception:
            _log.exception(
                "Failed to access localization cleanup for '%s'", plugin.info.name
            )
            return

        remaining_tokens = []
        for token in reversed(plugin.localization_tokens):
            try:
                unregister_catalog(token)
            except Exception:
                remaining_tokens.append(token)
                _log.exception(
                    "Failed to unregister localization catalog for '%s'",
                    plugin.info.name,
                )
        plugin.localization_tokens = list(reversed(remaining_tokens))

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
        with self._plugin_lifecycle(name):
            return self._unload_locked(name)

    def _unload_locked(self, name: str) -> bool:
        with self._plugins_lock:
            plugin = self._plugins.get(name)
            if not plugin or plugin.state != PluginState.ACTIVE:
                return False

        module_prefix = f"{MODULE_PREFIX}.{plugin.info.name}"
        unload_ok = True

        try:
            if plugin.module and hasattr(plugin.module, "on_unload"):
                try:
                    plugin.module.on_unload()
                except Exception as e:
                    unload_ok = False
                    plugin.error = str(e)
                    _log.exception("Plugin '%s' on_unload failed", name)

            self._unregister_localization_catalogs(plugin)

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
                lf.ui.unregister_panels_for_module(module_prefix)
                if hasattr(lf.ui, "clear_hooks_for_module"):
                    lf.ui.clear_hooks_for_module(module_prefix)
            except Exception:
                pass

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

            for cb in list(self._on_plugin_unloaded):
                try:
                    cb(plugin.info)
                except Exception as cb_err:
                    _log.warning("on_plugin_unloaded callback failed: %s", cb_err)

            return unload_ok

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
        with self._plugin_lifecycle(name):
            return self._reload_locked(name)

    def _reload_locked(self, name: str) -> bool:
        from .utils import get_gpu_memory

        plugin = self._plugins.get(name)
        if not plugin or plugin.state != PluginState.ACTIVE:
            return self.load(name)

        mem_before = get_gpu_memory()

        module_prefix = f"{MODULE_PREFIX}.{plugin.info.name}"

        try:
            if plugin.module and hasattr(plugin.module, "on_unload"):
                try:
                    plugin.module.on_unload()
                except Exception as e:
                    plugin.error = str(e)
                    _log.exception("Plugin '%s' on_unload failed during reload", name)
            plugin.module = None

            self._unregister_localization_catalogs(plugin)

            CapabilityRegistry.instance().unregister_all_for_plugin(name)

            try:
                from .ui.subscription_registry import SubscriptionRegistry
                SubscriptionRegistry.instance().unsubscribe_all(name)
            except Exception:
                _log.exception("Failed to cleanup signal subscriptions for '%s'", name)

            try:
                import lichtfeld as lf
                lf.ui.unregister_panels_for_module(module_prefix)
                if hasattr(lf.ui, "clear_hooks_for_module"):
                    lf.ui.clear_hooks_for_module(module_prefix)
            except Exception:
                pass

            to_remove = [m for m in sys.modules if m == module_prefix or m.startswith(f"{module_prefix}.")]
            for m in to_remove:
                sys.modules.pop(m, None)

            for path in plugin.sys_paths:
                if path in sys.path:
                    sys.path.remove(path)
            plugin.sys_paths = []

            self._register_localization_catalogs(plugin)
            self._load_module(plugin)

            if hasattr(plugin.module, "on_load"):
                plugin.module.on_load()

            self._update_file_mtimes(plugin)

            for cb in list(self._on_plugin_loaded):
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
            self._rollback_failed_load(plugin)
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
        for source_file in iter_plugin_watch_files(plugin.info.path):
            plugin.file_mtimes[source_file] = source_file.stat().st_mtime
        if self._watcher:
            self._watcher.clear_plugin_hashes(plugin.info.name)

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

    def on_plugin_changed(self, callback: Callable):
        """Subscribe to successful install, update, and uninstall changes."""
        self._on_plugin_changed.append(callback)

    def remove_plugin_loaded_callback(self, callback: Callable):
        try:
            self._on_plugin_loaded.remove(callback)
        except ValueError:
            pass

    def remove_plugin_unloaded_callback(self, callback: Callable):
        try:
            self._on_plugin_unloaded.remove(callback)
        except ValueError:
            pass

    def remove_plugin_changed_callback(self, callback: Callable):
        try:
            self._on_plugin_changed.remove(callback)
        except ValueError:
            pass

    def install(
        self,
        url: str,
        on_progress: Optional[Callable[[str], None]] = None,
        auto_load: bool = True,
        transport: str = "archive",
        source_info: Optional[PluginSourceInfo] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> str:
        """Install a plugin from GitHub using the selected transport."""
        self._raise_if_cancelled(should_cancel)
        mode = self._normalize_install_transport(transport)
        if mode == "git":
            # Git cloning chooses its target from the repository name before
            # the manifest is available. Hold the predicted name lock for that
            # filesystem phase, then validate and lock the manifest name too.
            predicted_name = normalize_repo_name(parse_github_url(url)[1])
            with self._plugin_lifecycle(predicted_name):
                clone_args = (url, self._plugins_dir, on_progress)
                if should_cancel is None:
                    staging_dir = clone_from_url(*clone_args)
                else:
                    staging_dir = clone_from_url(*clone_args, should_cancel=should_cancel)
                return self._finalize_new_plugin_install(
                    staging_dir,
                    source_info or self._source_info_for_git_url(url),
                    on_progress,
                    auto_load,
                    should_cancel,
                )

        predicted_name = None
        try:
            predicted_name = normalize_repo_name(parse_github_url(url)[1])
            self._validate_plugin_name(predicted_name)
        except (PluginError, ValueError):
            predicted_name = None
        if predicted_name:
            with self._plugin_lifecycle(predicted_name):
                if should_cancel is None:
                    staging_dir, source_info = prepare_github_archive(
                        url, self._plugins_dir, on_progress
                    )
                else:
                    staging_dir, source_info = prepare_github_archive(
                        url, self._plugins_dir, on_progress, should_cancel=should_cancel
                    )
                return self._finalize_new_plugin_install(
                    staging_dir, source_info, on_progress, auto_load, should_cancel
                )
        else:
            if should_cancel is None:
                staging_dir, source_info = prepare_github_archive(
                    url, self._plugins_dir, on_progress
                )
            else:
                staging_dir, source_info = prepare_github_archive(
                    url, self._plugins_dir, on_progress, should_cancel=should_cancel
                )
        return self._finalize_new_plugin_install(
            staging_dir, source_info, on_progress, auto_load, should_cancel
        )

    def update(self, name: str, on_progress: Optional[Callable[[str], None]] = None) -> bool:
        """Update a plugin according to its recorded install transport."""
        with self._plugin_lifecycle(name):
            return self._update_locked(name, on_progress)

    def _update_locked(self, name: str, on_progress: Optional[Callable[[str], None]]) -> bool:
        plugin = self._plugins.get(name)
        plugin_dir = plugin.info.path if plugin else self._find_plugin_dir(name)
        current_info = self._parse_manifest(plugin_dir)

        was_loaded = plugin and plugin.state == PluginState.ACTIVE
        if was_loaded:
            if not self.unload(name):
                raise PluginError(
                    f"Cannot update plugin '{name}': unload failed"
                    + (f": {plugin.error}" if plugin.error else "")
                )

        source_info = read_plugin_source_metadata(plugin_dir)
        replacement_started = False

        def reload_replacement():
            if was_loaded and not self.load(name, on_progress):
                raise PluginError(
                    f"Failed to load updated plugin '{name}': "
                    f"{self.get_error(name) or 'unknown error'}"
                )
            return True

        try:
            if source_info and source_info.transport == "archive":
                replacement_started = True
                if source_info.registry_id:
                    self._update_archive_plugin_from_registry(
                        plugin_dir, source_info, on_progress,
                        after_replace=reload_replacement,
                    )
                else:
                    self._update_archive_plugin_from_github(
                        plugin_dir, source_info, on_progress,
                        after_replace=reload_replacement,
                    )
            elif source_info and source_info.transport == "git":
                if not update_plugin(plugin_dir, on_progress):
                    raise PluginError(f"Failed to update plugin '{name}'")
            elif (plugin_dir / ".git").exists():
                if not update_plugin(plugin_dir, on_progress):
                    raise PluginError(f"Failed to update plugin '{name}'")
            else:
                raise PluginError(
                    f"Plugin '{name}' was not installed from a known remote source and cannot be updated automatically"
                )

            refreshed_info = self._parse_manifest(plugin_dir)
            if plugin:
                plugin.info = refreshed_info
                plugin.error = None
                plugin.error_traceback = None

            if was_loaded and not replacement_started:
                if not self.load(name, on_progress):
                    raise PluginError(
                        f"Failed to load updated plugin '{name}': "
                        f"{self.get_error(name) or 'unknown error'}"
                    )
            self._notify_plugin_changed(refreshed_info)
            return True
        except Exception:
            if plugin and replacement_started:
                plugin.info = current_info
                if was_loaded:
                    if plugin.state == PluginState.ACTIVE:
                        self.unload(name)
                    if not self.load(name, on_progress):
                        _log.error("Failed to reload restored plugin '%s': %s", name, self.get_error(name))
            raise

    def uninstall(self, name: str) -> bool:
        """Uninstall a plugin by removing its directory."""
        with self._plugin_lifecycle(name):
            return self._uninstall_locked(name)

    def _uninstall_locked(self, name: str) -> bool:
        with self._plugins_lock:
            plugin = self._plugins.get(name)
            plugin_dir = plugin.info.path if plugin else None

        if plugin and plugin.state == PluginState.ACTIVE:
            if not self.unload(name):
                _log.error("Cannot uninstall plugin '%s': unload failed", name)
                return False
            if plugin.state == PluginState.ACTIVE:
                _log.error("Cannot uninstall plugin '%s': plugin was reactivated during unload", name)
                return False

        if plugin_dir is None:
            plugin_dir = self._find_plugin_dir(name)

        try:
            removed = uninstall_plugin(plugin_dir)
        except Exception as exc:
            _log.error("Failed to uninstall plugin '%s': %s", name, exc)
            raise
        if not removed:
            _log.error("Failed to uninstall plugin '%s': plugin directory was not removed", name)
            return False

        with self._plugins_lock:
            if self._plugins.get(name) is plugin:
                del self._plugins[name]
        if plugin:
            self._notify_plugin_changed(plugin.info)
        else:
            self._notify_plugin_changed(PluginInfo(name=name, version="", path=plugin_dir))
        return True

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
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> str:
        """Install plugin from registry."""
        self._raise_if_cancelled(should_cancel)
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
                    should_cancel=should_cancel,
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
            archive_kwargs = {
                "temp_prefix": f".{plugin_id.replace(':', '-')}-",
                "on_progress": on_progress,
                "archive_validator": (
                    (lambda path: self._verify_registry_archive_checksum(path, version_info, plugin_id))
                    if version_info.checksum
                    else None
                ),
            }
            if should_cancel is not None:
                archive_kwargs["should_cancel"] = should_cancel
            staging_dir = prepare_archive_from_download_url(
                version_info.download_url,
                self._plugins_dir,
                **archive_kwargs,
            )
            return self._finalize_new_plugin_install(
                staging_dir, source_info, on_progress, auto_load, should_cancel
            )

        if repo_url:
            install_url = f"{repo_url}@{version_info.git_ref}" if version_info.git_ref else repo_url
            if should_cancel is None:
                staging_dir, source_info = prepare_github_archive(
                    install_url, self._plugins_dir, on_progress
                )
            else:
                staging_dir, source_info = prepare_github_archive(
                    install_url,
                    self._plugins_dir,
                    on_progress,
                    should_cancel=should_cancel,
                )
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
            return self._finalize_new_plugin_install(
                staging_dir, source_info, on_progress, auto_load, should_cancel
            )

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
        after_replace: Optional[Callable[[], object]] = None,
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
            self._replace_plugin_install(
                plugin_dir, staging_dir, new_source_info, after_replace=after_replace
            )
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
            self._replace_plugin_install(
                plugin_dir, staging_dir, new_source_info, after_replace=after_replace
            )
            return

        raise PluginError(f"No archive update method available for {plugin_id}")

    def _update_archive_plugin_from_github(
        self,
        plugin_dir: Path,
        source_info: PluginSourceInfo,
        on_progress: Optional[Callable[[str], None]] = None,
        after_replace: Optional[Callable[[], object]] = None,
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
        self._replace_plugin_install(
            plugin_dir, staging_dir, merged_source_info, after_replace=after_replace
        )

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
