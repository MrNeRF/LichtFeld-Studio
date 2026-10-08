# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Device-local gallery preferences; never part of account credentials or projects."""
import json
import threading
from pathlib import Path

UPLOAD_FORMATS = ("auto", "studio", "sog", "ssog", "spz")
DEFAULTS = dict(uploadFormat="auto", posterCacheMiB=64)
# Earlier builds wrote their default ("sog", later "ssog") back with every
# change, so either format from an older file records no user choice and
# yields the current default.
VERSION = 3
# Phones hold at most 2M splats, so larger scenes need LOD levels.
AUTO_SSOG_ABOVE = 2_000_000
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
        if raw.get("version") != VERSION and result["uploadFormat"] in ("sog", "ssog"):
            result["uploadFormat"] = DEFAULTS["uploadFormat"]
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return result


def _validate(key, value):
    if key == "uploadFormat":
        if value not in UPLOAD_FORMATS:
            raise ValueError("Unsupported upload format")
        return value
    if key != "posterCacheMiB" or isinstance(value, bool):
        raise ValueError("Unknown gallery preference")
    number = int(value)
    if str(number) != str(value) or not 1 <= number <= 4096:
        raise ValueError("Gallery preference is outside its supported range")
    return number


def resolve_upload_format(upload_format, splat_count):
    """The format to publish: auto picks SSOG above the phone budget, and when the count is unknown."""
    if upload_format != "auto":
        return upload_format
    return "sog" if type(splat_count) is int and 0 < splat_count <= AUTO_SSOG_ABOVE else "ssog"


def set_preference(key, value, root=None):
    value = _validate(key, value)
    with _lock:
        path = _root(root) / "preferences.json"
        values = read_preferences(path.parent)
        if values[key] == value:
            return
        values[key] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(values, version=VERSION)))
        temporary.replace(path)
