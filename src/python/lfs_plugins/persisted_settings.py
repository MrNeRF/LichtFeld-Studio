# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared handling for user-editable JSON settings files."""
from _lfs_path_io import native_path

import json
import itertools
import logging
import os
from _lfs_path_io import Path
import shutil
import tempfile
import time

_log = logging.getLogger(__name__)
_backup_sequence = itertools.count()


def load_object(path: Path) -> tuple[dict, bool]:
    """Return an object and whether an existing file needs a backup before writing."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("settings root must be an object")
        return raw, False
    except FileNotFoundError:
        return {}, False
    except (OSError, ValueError, UnicodeError) as error:
        _log.warning("Could not load settings from %s: %s", path, error)
        return {}, path.exists()


def _backup_corrupt(path: Path, backup_root: Path, category: str) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    for _ in range(1000):
        directory = backup_root / (f"reset-{category}-{time.time_ns() // 1_000_000}-"
                                   f"{os.getpid()}-{next(_backup_sequence)}")
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        backup = directory / path.name
        try:
            shutil.copy2(native_path(path), native_path(backup))
        except Exception:
            shutil.rmtree(native_path(directory, force_extended=True))
            raise
        _log.warning("Damaged settings from %s were backed up to %s", path, backup)
        return backup
    raise OSError(f"No backup name available for {path}")


def write_object(path: Path, values: dict, *, damaged: bool = False,
                 backup_root: Path | None = None,
                 backup_category: str = "corrupt-settings") -> None:
    """Publish atomically; a damaged original must be copied first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if damaged and path.exists():
        resolved_backup_root = backup_root or Path(os.environ.get(
            "LFS_RESOLVED_DATA_DIR", path.parent)) / "backups"
        _backup_corrupt(path, resolved_backup_root, backup_category)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=native_path(path.parent),
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as output:
            temporary = Path(output.name)
            json.dump(values, output, indent=2, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(native_path(temporary), native_path(path))
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
