# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Filesystem-folder discovery for Asset Manager .licht projects."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asset_index import is_supported_asset_path

_log = logging.getLogger(__name__)
_PRUNED_DIRECTORY_NAMES = frozenset(
    {
        "__pycache__",
        "dense",
        "depth",
        "depths",
        "images",
        "masks",
        "sparse",
        "stereo",
    }
)


@dataclass(frozen=True)
class AssetFolderScanResult:
    """Summary of one Asset Manager folder scan."""

    discovered: int = 0
    added: int = 0
    already_cataloged: int = 0
    failed: int = 0
    cancelled: bool = False


def discover_licht_projects(
    directory: str,
    cancel_event: threading.Event | None = None,
) -> list[str]:
    """Recursively list .licht files beneath one Asset Manager folder."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        _log.warning("Asset Manager folder is unavailable: %s", root)
        return []

    projects: list[str] = []

    def _on_error(exc: OSError) -> None:
        _log.warning("Could not scan Asset Manager folder: %s", exc)

    for current_root, directory_names, filenames in os.walk(
        root,
        topdown=True,
        onerror=_on_error,
        followlinks=False,
    ):
        if cancel_event is not None and cancel_event.is_set():
            break
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name.casefold() not in _PRUNED_DIRECTORY_NAMES
        )
        for filename in sorted(filenames):
            if cancel_event is not None and cancel_event.is_set():
                break
            path = Path(current_root) / filename
            if not is_supported_asset_path(str(path)):
                continue
            try:
                if path.is_file():
                    projects.append(str(path.resolve()))
            except OSError as exc:
                _log.warning("Could not inspect Asset Manager path %s: %s", path, exc)
    return projects


def scan_asset_folder(
    index: Any,
    folder_id: str,
    directory: str,
    cancel_event: threading.Event | None = None,
) -> AssetFolderScanResult:
    """Discover and register .licht projects from one real filesystem folder."""
    if cancel_event is not None and cancel_event.is_set():
        return AssetFolderScanResult(cancelled=True)
    project_paths = discover_licht_projects(directory, cancel_event)

    return _register_discovered(
        index,
        [(path, folder_id) for path in project_paths],
        cancel_event,
    )


def scan_all_asset_folders(
    index: Any,
    cancel_event: threading.Event | None = None,
) -> AssetFolderScanResult:
    """Scan every real folder, assigning projects to the most-specific root."""
    roots: list[tuple[Path, str]] = []
    seen_roots = set()
    for folder_id, folder in (getattr(index, "folders", {}) or {}).items():
        directory = str(folder.get("path") or "").strip()
        if not directory:
            continue
        try:
            root = Path(directory).expanduser().resolve()
        except OSError as exc:
            _log.warning("Could not resolve Asset Manager folder %s: %s", directory, exc)
            continue
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
        if cancel_event is not None and cancel_event.is_set():
            return AssetFolderScanResult(cancelled=True)
        for path in discover_licht_projects(str(root), cancel_event):
            key = os.path.normcase(path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            discovered.append((path, folder_id))

    return _register_discovered(index, discovered, cancel_event)


def _register_discovered(
    index: Any,
    discovered: list[tuple[str, str]],
    cancel_event: threading.Event | None = None,
) -> AssetFolderScanResult:
    lock = getattr(index, "_lock", None)
    if lock is None:
        return _register_discovered_locked(index, discovered, cancel_event)
    with lock:
        return _register_discovered_locked(index, discovered, cancel_event)


def _register_discovered_locked(
    index: Any,
    discovered: list[tuple[str, str]],
    cancel_event: threading.Event | None,
) -> AssetFolderScanResult:
    snapshot = getattr(index, "_snapshot_state", lambda: None)()
    added = 0
    already_cataloged = 0
    failed = 0
    for path, folder_id in discovered:
        if cancel_event is not None and cancel_event.is_set():
            restore = getattr(index, "_restore_state", None)
            if snapshot is not None and callable(restore):
                restore(snapshot)
            return AssetFolderScanResult(
                discovered=len(discovered),
                already_cataloged=already_cataloged,
                failed=failed,
                cancelled=True,
            )
        try:
            find_by_path = getattr(index, "find_asset_by_path", None)
            if callable(find_by_path):
                existing = find_by_path(path)
                if existing is not None and getattr(existing, "status", "AVAILABLE") not in {
                    "MISSING",
                    "UNREADABLE",
                    "UNVERIFIED",
                }:
                    already_cataloged += 1
                    continue
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
            _log.warning("Failed to register Asset Manager project: %s", path, exc_info=True)

    if cancel_event is not None and cancel_event.is_set():
        restore = getattr(index, "_restore_state", None)
        if snapshot is not None and callable(restore):
            restore(snapshot)
        return AssetFolderScanResult(
            discovered=len(discovered),
            already_cataloged=already_cataloged,
            failed=failed,
            cancelled=True,
        )
    if added and not index.save():
        _log.error("Failed to persist Asset Manager folder scan")
        restore = getattr(index, "_restore_state", None)
        if snapshot is not None and callable(restore):
            restore(snapshot)
        failed += added
        added = 0
    return AssetFolderScanResult(
        discovered=len(discovered),
        added=added,
        already_cataloged=already_cataloged,
        failed=failed,
    )
