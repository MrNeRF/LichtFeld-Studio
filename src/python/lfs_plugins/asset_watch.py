# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Filesystem-folder discovery for Asset Manager .licht projects."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .asset_index import is_supported_asset_path

SCAN_BATCH_SIZE = 25
SCAN_BATCH_INTERVAL_S = 0.25

_log = logging.getLogger(__name__)
_PRUNED_DIRECTORY_NAMES = frozenset(
    {
        "__pycache__",
        "cmakefiles",
        "dense",
        "depth",
        "depths",
        "images",
        "masks",
        "node_modules",
        "site-packages",
        "sparse",
        "stereo",
        "vcpkg_installed",
    }
)


class AssetFolderScanProgress:
    """Worker-side counters for one Asset Manager folder scan."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._directories_visited = 0
        self._projects_found = 0
        self._current_root = ""

    def add_directory(self) -> None:
        with self._lock:
            self._directories_visited += 1

    def add_project(self) -> None:
        with self._lock:
            self._projects_found += 1

    def report(
        self,
        *,
        directories: int | None = None,
        projects: int | None = None,
        current_root: str | None = None,
    ) -> None:
        with self._lock:
            if directories is not None:
                self._directories_visited = directories
            if projects is not None:
                self._projects_found = projects
            if current_root is not None:
                self._current_root = current_root

    def snapshot(self) -> tuple[int, int, str]:
        with self._lock:
            return (
                self._directories_visited,
                self._projects_found,
                self._current_root,
            )


@dataclass(frozen=True)
class AssetFolderScanResult:
    """Summary of one Asset Manager folder scan."""

    discovered: int = 0
    added: int = 0
    already_cataloged: int = 0
    failed: int = 0
    cancelled: bool = False


def _directory_is_pruned(name: str) -> bool:
    return name.startswith(".") or name.casefold() in _PRUNED_DIRECTORY_NAMES


def iter_licht_projects(
    directory: str,
    cancel_event: threading.Event | None = None,
    progress: AssetFolderScanProgress | None = None,
) -> Iterator[str]:
    """Yield .licht files beneath one Asset Manager folder as they are found."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        _log.warning("Asset Manager folder is unavailable: %s", root)
        return

    visited_directories = 0
    if progress is not None:
        progress.report(current_root=str(root))

    def _on_error(exc: OSError) -> None:
        _log.warning("Could not scan Asset Manager folder: %s", exc)

    for current_root, directory_names, filenames in os.walk(
        root,
        topdown=True,
        onerror=_on_error,
        followlinks=False,
    ):
        if cancel_event is not None and cancel_event.is_set():
            return
        visited_directories += 1
        if progress is not None:
            progress.add_directory()
        if visited_directories == 10001:
            _log.warning(
                "Asset folder %s is very large (>10000 directories); consider a smaller folder",
                root,
            )
        directory_names[:] = sorted(
            name for name in directory_names if not _directory_is_pruned(name)
        )
        for filename in sorted(filenames):
            if cancel_event is not None and cancel_event.is_set():
                return
            path = Path(current_root) / filename
            if not is_supported_asset_path(str(path)):
                continue
            try:
                if path.is_file():
                    resolved = str(path.resolve())
                    if progress is not None:
                        progress.add_project()
                    yield resolved
            except OSError as exc:
                _log.warning("Could not inspect Asset Manager path %s: %s", path, exc)


def discover_licht_projects(
    directory: str,
    cancel_event: threading.Event | None = None,
    progress: AssetFolderScanProgress | None = None,
) -> list[str]:
    """Recursively list .licht files beneath one Asset Manager folder."""
    return list(iter_licht_projects(directory, cancel_event, progress))


def scan_asset_folder(
    index: Any,
    folder_id: str,
    directory: str,
    cancel_event: threading.Event | None = None,
    progress: AssetFolderScanProgress | None = None,
) -> AssetFolderScanResult:
    """Discover and register .licht projects from one real filesystem folder."""
    if cancel_event is not None and cancel_event.is_set():
        return AssetFolderScanResult(cancelled=True)
    if progress is not None:
        progress.report(current_root=directory)
    return _register_discovered_streaming(
        index,
        (
            (path, folder_id)
            for path in iter_licht_projects(directory, cancel_event, progress)
        ),
        cancel_event,
        progress,
    )


def scan_all_asset_folders(
    index: Any,
    cancel_event: threading.Event | None = None,
    progress: AssetFolderScanProgress | None = None,
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

    def _iter_all() -> Iterator[tuple[str, str]]:
        seen_paths: set[str] = set()
        for root, assigned_folder_id in roots:
            if cancel_event is not None and cancel_event.is_set():
                return
            if progress is not None:
                progress.report(current_root=str(root))
            for path in iter_licht_projects(str(root), cancel_event, progress):
                path_key = os.path.normcase(path)
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                yield path, assigned_folder_id

    if cancel_event is not None and cancel_event.is_set():
        return AssetFolderScanResult(cancelled=True)
    return _register_discovered_streaming(index, _iter_all(), cancel_event, progress)


def verify_catalog_projects(
    index: Any,
    cancel_event: threading.Event | None = None,
    *,
    batch_size: int = SCAN_BATCH_SIZE,
    interval_s: float = SCAN_BATCH_INTERVAL_S,
) -> int:
    """Verify catalog rows in batches, releasing between batches."""
    list_projects = getattr(index, "list_projects", None)
    verify_batch = getattr(index, "verify_projects_batch", None)
    verify_asset = getattr(index, "verify_asset", None)
    if not callable(list_projects):
        return 0
    asset_ids = [
        str(getattr(project, "id", None) or getattr(project, "project_uuid", "") or "")
        for project in list_projects()
    ]
    asset_ids = [asset_id for asset_id in asset_ids if asset_id]
    verified = 0
    batch: list[str] = []
    last_flush = time.monotonic()

    def flush() -> bool:
        nonlocal batch, verified, last_flush
        if not batch:
            return False
        if cancel_event is not None and cancel_event.is_set():
            return True
        if callable(verify_batch):
            verified += verify_batch(batch)
        elif callable(verify_asset):
            for asset_id in batch:
                if cancel_event is not None and cancel_event.is_set():
                    return True
                if verify_asset(asset_id) is not None:
                    verified += 1
        else:
            return True
        batch = []
        last_flush = time.monotonic()
        return False

    for asset_id in asset_ids:
        if cancel_event is not None and cancel_event.is_set():
            return verified
        batch.append(asset_id)
        if len(batch) >= batch_size or (
            batch and (time.monotonic() - last_flush) >= interval_s
        ):
            if flush():
                return verified
    flush()
    return verified


def _should_flush_batch(batch: list[tuple[str, str]], last_flush: float) -> bool:
    if not batch:
        return False
    if len(batch) >= SCAN_BATCH_SIZE:
        return True
    return (time.monotonic() - last_flush) >= SCAN_BATCH_INTERVAL_S


def _register_discovered_streaming(
    index: Any,
    discovered: Iterable[tuple[str, str]],
    cancel_event: threading.Event | None = None,
    progress: AssetFolderScanProgress | None = None,
) -> AssetFolderScanResult:
    batch: list[tuple[str, str]] = []
    last_flush = time.monotonic()
    discovered_count = 0
    added = 0
    already_cataloged = 0
    failed = 0
    cancelled = False

    def flush() -> bool:
        nonlocal batch, added, already_cataloged, failed, last_flush
        if not batch:
            return False
        batch_added, batch_already, batch_failed, was_cancelled = (
            _commit_registration_batch(index, batch, cancel_event)
        )
        batch = []
        last_flush = time.monotonic()
        if was_cancelled:
            return True
        added += batch_added
        already_cataloged += batch_already
        failed += batch_failed
        if progress is not None:
            _directories, projects_found, current_root = progress.snapshot()
            progress.report(
                projects=projects_found,
                current_root=current_root,
            )
        return False

    for item in discovered:
        discovered_count += 1
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        batch.append(item)
        if _should_flush_batch(batch, last_flush):
            if flush():
                cancelled = True
                break
    if not cancelled:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
        elif flush():
            cancelled = True

    if added:
        save = getattr(index, "save", None)
        if callable(save) and not save():
            _log.error("Failed to persist Asset Manager folder scan")
            if not cancelled:
                failed += added
                added = 0
    return AssetFolderScanResult(
        discovered=discovered_count,
        added=added,
        already_cataloged=already_cataloged,
        failed=failed,
        cancelled=cancelled,
    )


def _existing_skips_inspection(existing: Any) -> bool:
    return getattr(existing, "status", "AVAILABLE") not in {
        "MISSING",
        "UNREADABLE",
        "UNVERIFIED",
    }


def _commit_registration_batch(
    index: Any,
    batch: list[tuple[str, str]],
    cancel_event: threading.Event | None,
) -> tuple[int, int, int, bool]:
    """Commit one discovered batch. Cancel drops this batch if it is not committed."""
    if not batch:
        return 0, 0, 0, False
    if cancel_event is not None and cancel_event.is_set():
        return 0, 0, 0, True

    inspect = getattr(index, "_inspect_path", None)
    if not callable(inspect):
        return _commit_batch_with_snapshot(index, batch, cancel_event)

    prepared: list[tuple[str, str, Any]] = []
    already_cataloged = 0
    failed = 0
    find_by_path = getattr(index, "find_asset_by_path", None)
    for path, folder_id in batch:
        if cancel_event is not None and cancel_event.is_set():
            return 0, 0, 0, True
        try:
            if callable(find_by_path):
                existing = find_by_path(path)
                if existing is not None and _existing_skips_inspection(existing):
                    already_cataloged += 1
                    continue
            prepared.append((path, folder_id, inspect(path)))
        except Exception:
            failed += 1
            _log.warning("Failed to register Asset Manager project: %s", path, exc_info=True)

    if cancel_event is not None and cancel_event.is_set():
        return 0, 0, 0, True

    added = 0
    lock = getattr(index, "_lock", None)

    def commit_prepared() -> None:
        nonlocal added, already_cataloged, failed
        for path, folder_id, inspection in prepared:
            try:
                project, created = index.register_licht_asset(
                    path,
                    folder_id=folder_id,
                    adopt_existing=False,
                    save=False,
                    inspection=inspection,
                )
                if project is None:
                    failed += 1
                elif created:
                    added += 1
                else:
                    already_cataloged += 1
            except Exception:
                failed += 1
                _log.warning(
                    "Failed to register Asset Manager project: %s", path, exc_info=True
                )

    if lock is None:
        commit_prepared()
    else:
        with lock:
            if cancel_event is not None and cancel_event.is_set():
                return 0, 0, 0, True
            commit_prepared()
    return added, already_cataloged, failed, False


def _commit_batch_with_snapshot(
    index: Any,
    batch: list[tuple[str, str]],
    cancel_event: threading.Event | None,
) -> tuple[int, int, int, bool]:
    snapshot_fn = getattr(index, "_snapshot_state", None)
    restore_fn = getattr(index, "_restore_state", None)
    snapshot = snapshot_fn() if callable(snapshot_fn) else None
    added = 0
    already_cataloged = 0
    failed = 0
    find_by_path = getattr(index, "find_asset_by_path", None)

    def restore() -> None:
        if snapshot is not None and callable(restore_fn):
            restore_fn(snapshot)

    for path, folder_id in batch:
        if cancel_event is not None and cancel_event.is_set():
            restore()
            return 0, 0, 0, True
        try:
            if callable(find_by_path):
                existing = find_by_path(path)
                if existing is not None and _existing_skips_inspection(existing):
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
        restore()
        return 0, 0, 0, True
    return added, already_cataloged, failed, False
