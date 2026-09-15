# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Disposable project previews inside the LichtFeld home."""

import hashlib
import logging
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

from .environment import value as environment_value

_log = logging.getLogger(__name__)
_preview_lock = threading.RLock()
_active_previews = set()
PREVIEW_CACHE_BYTES = 200 * 1024 * 1024


def lichtfeld_home():
    root = environment_value("LFS_HOME")
    if root:
        return Path(root).expanduser().resolve()
    storage = environment_value("LFS_RESOLVED_ASSET_LIBRARY_DIR")
    if storage:
        return Path(storage).expanduser().resolve().parent.parent
    return Path.home() / ".lichtfeld"


def _preview_prefix(project_id):
    return hashlib.sha256(str(project_id).encode("utf-8")).hexdigest() + "-"


def prune_previews(project_ids=()):
    directory = lichtfeld_home() / "cache" / "previews"
    prefixes = tuple(_preview_prefix(project_id) for project_id in project_ids)
    with _preview_lock:
        try:
            files = []
            for path in directory.glob("*.png"):
                if path in _active_previews or path.is_symlink():
                    continue
                if prefixes and path.name.startswith(prefixes):
                    path.unlink(missing_ok=True)
                else:
                    stat = path.stat()
                    files.append((stat.st_mtime_ns, stat.st_size, path))
            total = sum(size for _, size, _ in files)
            for _, size, path in sorted(files):
                if total <= PREVIEW_CACHE_BYTES:
                    break
                path.unlink(missing_ok=True)
                total -= size
        except OSError as exc:
            _log.warning("Could not prune project previews in %s: %s", directory, exc)


@contextmanager
def preview_capture(project_id):
    directory = lichtfeld_home() / "cache" / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    with _preview_lock:
        prune_previews()
        fd, name = tempfile.mkstemp(prefix=_preview_prefix(project_id), suffix=".png", dir=directory)
        os.close(fd)
        target = Path(name)
        _active_previews.add(target)
    try:
        yield target
    finally:
        with _preview_lock:
            _active_previews.discard(target)
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                _log.warning("Could not remove project preview %s: %s", target, exc)
            prune_previews()
