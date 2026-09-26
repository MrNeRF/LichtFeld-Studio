# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Device-local gallery preferences; never part of account credentials or projects."""
import json
import threading
from pathlib import Path

DEFAULTS = dict(uploadFormat="sog", posterCacheMiB=64)
_lock = threading.RLock()


def _root(root):
    if root is None:
        from .asset_index import resolve_asset_manager_storage_path
        root = resolve_asset_manager_storage_path() / "gallery"
    return Path(root)


def read_preferences(root=None):
    result = DEFAULTS.copy()
    try:
        raw = json.loads((_root(root) / "preferences.json").read_text())
        for key in DEFAULTS:
            if key in raw:
                try:
                    result[key] = _validate(key, raw[key])
                except (ValueError, TypeError):
                    pass
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return result


def _validate(key, value):
    if key == "uploadFormat":
        if value not in ("studio", "sog", "ssog", "spz"):
            raise ValueError("Unsupported upload format")
        return value
    if key != "posterCacheMiB" or isinstance(value, bool):
        raise ValueError("Unknown gallery preference")
    number = int(value)
    if str(number) != str(value) or not 1 <= number <= 4096:
        raise ValueError("Gallery preference is outside its supported range")
    return number


def set_preference(key, value, root=None):
    value = _validate(key, value)
    with _lock:
        path = _root(root) / "preferences.json"
        values = read_preferences(path.parent)
        values[key] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values))
        temporary.replace(path)
