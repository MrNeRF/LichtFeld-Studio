# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin settings storage - persistent key-value configuration per plugin."""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)


class PluginSettings:
    """Settings object for a single plugin.

    Provides get/set interface with auto-persistence to JSON file.
    """

    def __init__(self, plugin_name: str, settings_dir: Path):
        self._plugin_name = plugin_name
        self._file = settings_dir / plugin_name / "settings.json"
        self._data: Dict[str, Any] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if self._file.exists():
                try:
                    with open(self._file, "r", encoding="utf-8") as f:
                        self._data = json.load(f)
                    _log.debug("Loaded settings for %s from %s", self._plugin_name, self._file)
                except Exception as e:
                    _log.warning("Failed to load settings for %s: %s", self._plugin_name, e)
                    self._data = {}
            self._loaded = True

    def _save(self) -> None:
        with self._lock:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self._file, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2)
                _log.debug("Saved settings for %s to %s", self._plugin_name, self._file)
            except Exception as e:
                _log.error("Failed to save settings for %s: %s", self._plugin_name, e)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value with optional default."""
        self._ensure_loaded()
        return self._data.get(key, default)

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
                self._save()
                return True
        return False

    def clear(self) -> None:
        """Clear all settings."""
        with self._lock:
            self._data = {}
            self._loaded = True
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
        with self._cache_lock:
            if plugin_name not in self._cache:
                self._cache[plugin_name] = PluginSettings(plugin_name, self._settings_dir)
            return self._cache[plugin_name]
