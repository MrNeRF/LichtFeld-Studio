# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Watched-directory discovery for Asset Manager .licht projects."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .asset_index import is_supported_asset_path

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatchScanResult:
    """Summary of one watched-directory scan."""

    discovered: int = 0
    added: int = 0
    already_cataloged: int = 0
    failed: int = 0


def discover_licht_projects(directory: str) -> list[str]:
    """Recursively list .licht files beneath one directory."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        _log.warning("Watched directory is unavailable: %s", root)
        return []

    projects: list[str] = []

    def _on_error(exc: OSError) -> None:
        _log.warning("Could not scan watched directory: %s", exc)

    for current_root, directory_names, filenames in os.walk(
        root,
        topdown=True,
        onerror=_on_error,
        followlinks=False,
    ):
        directory_names.sort()
        for filename in sorted(filenames):
            path = Path(current_root) / filename
            if is_supported_asset_path(str(path)) and path.is_file():
                projects.append(str(path.resolve()))
    return projects


def scan_watch_directories(
    index: Any,
    folder_id: str,
    directories: Iterable[str],
) -> WatchScanResult:
    """Discover and register .licht projects from configured directories."""
    project_paths: list[str] = []
    seen_paths = set()
    for directory in directories:
        for path in discover_licht_projects(directory):
            key = os.path.normcase(path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            project_paths.append(path)

    return _register_discovered(
        index,
        [(path, folder_id) for path in project_paths],
    )


def scan_all_watch_directories(index: Any) -> WatchScanResult:
    """Scan every folder, assigning new projects to the most-specific root."""
    roots: list[tuple[Path, str]] = []
    seen_roots = set()
    for folder_id, folder in (getattr(index, "folders", {}) or {}).items():
        for directory in folder.get("watch_directories", []):
            root = Path(directory).expanduser().resolve()
            key = os.path.normcase(str(root))
            if key in seen_roots:
                continue
            seen_roots.add(key)
            roots.append((root, folder_id))

    roots.sort(
        key=lambda item: (
            -len(item[0].parts),
            os.path.normcase(str(item[0])),
            item[1],
        )
    )
    discovered: list[tuple[str, str]] = []
    seen_paths = set()
    for root, folder_id in roots:
        for path in discover_licht_projects(str(root)):
            key = os.path.normcase(path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            discovered.append((path, folder_id))

    return _register_discovered(index, discovered)


def _register_discovered(
    index: Any,
    discovered: list[tuple[str, str]],
) -> WatchScanResult:
    added = 0
    already_cataloged = 0
    failed = 0
    for path, folder_id in discovered:
        try:
            project, created = index.register_licht_asset(
                path,
                folder_id=folder_id,
                adopt_existing=False,
                save=False,
            )
            if project is None:
                failed += 1
            elif created:
                added += 1
            else:
                already_cataloged += 1
        except Exception:
            failed += 1
            _log.warning("Failed to register watched .licht project: %s", path, exc_info=True)

    if discovered and not index.save():
        _log.error("Failed to persist watched .licht project scan")
        failed += 1
    return WatchScanResult(
        discovered=len(discovered),
        added=added,
        already_cataloged=already_cataloged,
        failed=failed,
    )
