# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin settings storage - persistent key-value configuration per plugin."""

import logging
import copy
import os
import threading
from _lfs_path_io import Path
from typing import Any, Dict, Optional

from .plugin import validate_plugin_name
from .persisted_settings import load_object, write_object

_log = logging.getLogger(__name__)


class PluginSettings:
    """Settings object for a single plugin.

    Provides get/set interface with auto-persistence to JSON file.
    """

    def __init__(self, plugin_name: str, settings_dir: Path):
        self._plugin_name = validate_plugin_name(plugin_name)
        self._file = settings_dir / self._plugin_name / "settings.json"
        self._backup_root = Path(os.environ.get(
            "LFS_RESOLVED_DATA_DIR", settings_dir.parent / "data")) / "backups"
        self._data: Dict[str, Any] = {}
        self._loaded = False
        self._snapshot: Dict[str, Any] = {}
        self._clear_all = False
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._data, _ = load_object(self._file)
            self._snapshot = copy.deepcopy(self._data)
            self._loaded = True

    def _save(self) -> None:
        with self._lock:
            try:
                latest, damaged = load_object(self._file)
                if self._clear_all:
                    merged = {}
                else:
                    merged = latest
                    for key in self._snapshot:
                        if key not in self._data:
                            merged.pop(key, None)
                    for key, value in self._data.items():
                        if key not in self._snapshot or value != self._snapshot[key]:
                            merged[key] = value
                write_object(self._file, merged, damaged=damaged,
                             backup_root=self._backup_root,
                             backup_category="corrupt-plugin-settings")
                self._data = merged
                self._snapshot = copy.deepcopy(merged)
                self._clear_all = False
                _log.debug("Saved settings for %s to %s", self._plugin_name, self._file)
            except Exception as e:
                _log.error("Failed to save settings for %s: %s", self._plugin_name, e)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value with optional default."""
        self._ensure_loaded()
        value = self._data.get(key, default)
        if default is not None:
            expected = type(default)
            compatible = type(value) is expected
            if expected in (int, float) and type(value) in (int, float):
                compatible = True
            if expected in (bool, int, float, str, list, dict) and not compatible:
                _log.warning("Invalid type for plugin setting %s.%s; using its default",
                             self._plugin_name, key)
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a setting value (auto-saves)."""
        self._ensure_loaded()
        with self._lock:
            self._data[key] = value
        self._save()

    def update(self, values: Dict[str, Any]) -> None:
        """Update multiple settings at once (single save)."""
        self._ensure_loaded()
        with self._lock:
            self._data.update(values)
        self._save()

    def delete(self, key: str) -> bool:
        """Delete a setting. Returns True if key existed."""
        self._ensure_loaded()
        with self._lock:
            if key in self._data:
                del self._data[key]
                existed = True
            else:
                existed = False
        if existed:
            self._save()
            return True
        return False

    def clear(self) -> None:
        """Clear all settings."""
        self._ensure_loaded()
        with self._lock:
            self._data = {}
            self._loaded = True
            self._clear_all = True
        self._save()

    def all(self) -> Dict[str, Any]:
        """Get all settings as a dict (copy)."""
        self._ensure_loaded()
        with self._lock:
            return dict(self._data)

    def __contains__(self, key: str) -> bool:
        self._ensure_loaded()
        return key in self._data

    def __repr__(self) -> str:
        return f"PluginSettings({self._plugin_name!r})"


class SettingsManager:
    """Singleton managing settings for all plugins."""

    _instance: Optional["SettingsManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._settings_dir = Path(
            os.environ.get(
                "LFS_RESOLVED_PLUGIN_DIR",
                Path.home() / ".lichtfeld" / "plugins",
            )
        )
        self._cache: Dict[str, PluginSettings] = {}
        self._cache_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "SettingsManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get(self, plugin_name: str) -> PluginSettings:
        """Get settings object for a plugin."""
        validate_plugin_name(plugin_name)
        with self._cache_lock:
            if plugin_name not in self._cache:
                self._cache[plugin_name] = PluginSettings(plugin_name, self._settings_dir)
            return self._cache[plugin_name]
