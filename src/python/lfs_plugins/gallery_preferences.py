# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Device-local gallery preferences; never part of account credentials or projects."""
import logging
import threading
from _lfs_path_io import Path

from .persisted_settings import load_object, write_object

UPLOAD_FORMATS = ("auto", "studio", "sog", "ssog", "spz")
DEFAULTS = dict(uploadFormat="auto", posterCacheMiB=64)
# Earlier builds wrote their default ("sog", later "ssog") back with every
# change, so either format from an older file records no user choice and
# yields the current default.
VERSION = 3
# Phones hold at most 2M splats, so larger scenes need LOD levels.
AUTO_SSOG_ABOVE = 2_000_000
_lock = threading.RLock()
_log = logging.getLogger(__name__)


def _root(root):
    if root is None:
        from .asset_index import resolve_asset_manager_storage_path
        root = resolve_asset_manager_storage_path() / "gallery"
    return Path(root)


def read_preferences(root=None):
    result = DEFAULTS.copy()
    raw, _ = load_object(_root(root) / "preferences.json")
    for key in DEFAULTS:
        if key in raw:
            try:
                result[key] = _validate(key, raw[key])
            except (ValueError, TypeError) as error:
                _log.warning("Invalid gallery preference %s: %s", key, error)
    version = raw.get("version", 0)
    if type(version) is not int or version < 0:
        _log.warning("Invalid gallery preference version; using current defaults")
    if type(version) is int and version < VERSION and result["uploadFormat"] in ("sog", "ssog"):
        result["uploadFormat"] = DEFAULTS["uploadFormat"]
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
        raw, damaged = load_object(path)
        damaged = damaged or ("version" in raw and
                              (type(raw["version"]) is not int or raw["version"] < 0))
        if type(raw.get("version")) is int and raw["version"] > VERSION:
            raw[key] = value
        else:
            raw.update(values)
        if type(raw.get("version")) is not int or raw["version"] <= VERSION:
            raw["version"] = VERSION
        write_object(path, raw, damaged=damaged,
                     backup_category="corrupt-gallery-preferences")
