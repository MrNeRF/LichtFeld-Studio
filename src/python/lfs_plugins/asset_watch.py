# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Filesystem-folder discovery for Asset Manager .licht projects."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from types import SimpleNamespace

from .asset_index import (
    AssetObservation,
    is_supported_asset_path,
    resolve_asset_manager_storage_path,
)
from .project_identity import ProjectPathIdentity

SCAN_BATCH_SIZE = 25
SCAN_BATCH_INTERVAL_S = 0.25
SCAN_CACHE_FILENAME = "scan_cache.json"
SCAN_CACHE_MTIME_GUARD_S = 2.0

_log = logging.getLogger(__name__)
_REINSPECT_STATUSES = frozenset(
    {
        "MISSING",
        "UNREADABLE",
        "UNVERIFIED",
        "READING",
        "IDENTITY_MISMATCH",
        "UNSUPPORTED",
        "REPLACED_PUBLISHED",
        "DIVERGED_COPIES",
        "REPAIR_ONLY",
        "UNSUPPORTED_NEWER",
    }
)
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

    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._directories_visited = 0
        self._projects_found = 0
        self._current_root = ""
        self._on_change = on_change
        self._last_notification = 0.0

    def _notify(self) -> None:
        # Progress producers wake the UI at most four times a second. There is
        # no timer after the worker stops; completion publishes the final state.
        now = time.monotonic()
        if self._on_change and now - self._last_notification >= SCAN_BATCH_INTERVAL_S:
            self._last_notification = now
            self._on_change()

    def add_directory(self) -> None:
        with self._lock:
            self._directories_visited += 1
        self._notify()

    def add_project(self) -> None:
        with self._lock:
            self._projects_found += 1
        self._notify()

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
        self._notify()

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
    unavailable: bool = False
    cancelled: bool = False


def _directory_is_pruned(name: str) -> bool:
    return name.startswith(".") or name.casefold() in _PRUNED_DIRECTORY_NAMES


def _entry_is_dir(entry: Any) -> bool:
    """Support both real DirEntry objects and lightweight test doubles."""
    try:
        return entry.is_dir(follow_symlinks=False)
    except TypeError:
        return entry.is_dir()


def _directory_key(directory: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(directory)))


def _directory_mtime_is_recent(mtime_ns: int) -> bool:
    return abs(time.time() - (int(mtime_ns) / 1_000_000_000)) < SCAN_CACHE_MTIME_GUARD_S


class _DirectoryScanCache:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self._loaded = False
        self._changed = False
        self._path: Path | None = None

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            self._path = resolve_asset_manager_storage_path() / SCAN_CACHE_FILENAME
            with self._path.open("r", encoding="utf-8") as stream:
                raw_entries = json.load(stream)
            if not isinstance(raw_entries, dict):
                raise ValueError("scan cache root must be an object")
            entries: dict[str, dict[str, Any]] = {}
            for raw_directory, raw_entry in raw_entries.items():
                if not isinstance(raw_directory, str) or not isinstance(raw_entry, dict):
                    raise ValueError("invalid scan cache entry")
                mtime_ns = raw_entry.get("mtime_ns")
                dirs = raw_entry.get("dirs")
                licht = raw_entry.get("licht")
                recursive = raw_entry.get("recursive")
                if recursive is None:
                    # Entries written before scan depth became part of the
                    # cache contract cannot reveal how they were produced.
                    self._changed = True
                    continue
                if (
                    not isinstance(mtime_ns, int)
                    or not isinstance(dirs, list)
                    or not isinstance(licht, list)
                    or not isinstance(recursive, bool)
                    or not all(isinstance(name, str) and Path(name).name == name for name in dirs)
                    or not all(isinstance(name, str) and Path(name).name == name for name in licht)
                ):
                    raise ValueError("invalid scan cache entry")
                entries[_directory_key(raw_directory)] = {
                    "mtime_ns": int(mtime_ns),
                    "dirs": list(dirs),
                    "licht": list(licht),
                    "recursive": recursive,
                }
            self._entries = entries
        except FileNotFoundError:
            return
        except Exception as exc:
            self._entries = {}
            _log.warning("Ignoring invalid Asset Manager scan cache %s: %s", self._path, exc)

    def trusted(
        self,
        directory: str,
        mtime_ns: int,
        *,
        recursive: bool,
    ) -> dict[str, Any] | None:
        self._load()
        key = _directory_key(directory)
        if _directory_mtime_is_recent(mtime_ns):
            self.drop(key, recursive=False)
            return None
        entry = self._entries.get(key)
        if (
            entry is None
            or entry["mtime_ns"] != int(mtime_ns)
            or entry["recursive"] is not recursive
        ):
            return None
        return entry

    def drop(self, directory: str, *, recursive: bool = True) -> None:
        self._load()
        key = _directory_key(directory)
        if not recursive:
            if self._entries.pop(key, None) is not None:
                self._changed = True
            return
        prefix = key + os.sep
        removed = [
            cached_key
            for cached_key in self._entries
            if cached_key == key or cached_key.startswith(prefix)
        ]
        for cached_key in removed:
            del self._entries[cached_key]
        if removed:
            self._changed = True

    def replace(
        self,
        directory: str,
        mtime_ns: int,
        dirs: list[str],
        licht: list[str],
        *,
        recursive: bool,
    ) -> None:
        self._load()
        key = _directory_key(directory)
        previous = self._entries.get(key)
        if previous is not None:
            current_dirs = set(dirs)
            for name in previous["dirs"]:
                if name not in current_dirs:
                    self.drop(os.path.join(key, name))
        if _directory_mtime_is_recent(mtime_ns):
            self.drop(key, recursive=False)
            return
        entry = {
            "mtime_ns": int(mtime_ns),
            "dirs": list(dirs),
            "licht": list(licht),
            "recursive": recursive,
        }
        if previous != entry:
            self._entries[key] = entry
            self._changed = True

    def persist(self) -> None:
        self._load()
        if not self._changed or self._path is None:
            return
        temp_path: Path | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(
                prefix=f"{self._path.stem}.",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            temp_path = Path(temp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self._entries, stream, separators=(",", ":"), ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self._path)
            self._changed = False
        except Exception as exc:
            _log.error("Failed to save Asset Manager scan cache %s: %s", self._path, exc)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def iter_licht_projects(
    directory: str,
    cancel_event: threading.Event | None = None,
    progress: AssetFolderScanProgress | None = None,
    *,
    scan_cache: _DirectoryScanCache | None = None,
    recursive: bool = True,
) -> Iterator[str]:
    """Yield .licht files beneath one Asset Manager folder as they are found."""
    cache = scan_cache or _DirectoryScanCache()
    root = Path(directory).expanduser()
    if not root.is_dir():
        _log.warning("Asset Manager folder is unavailable: %s", root)
        return

    pruned_user_directories: set[str] = set()
    try:
        import lichtfeld as lf

        getter = getattr(getattr(lf, "ui", None), "get_default_project_location", None)
        if callable(getter):
            user_root = os.path.normcase(
                os.path.dirname(os.path.abspath(str(getter() or "")))
            )
            pruned_user_directories = {
                os.path.join(user_root, "tmp"),
                os.path.join(user_root, "recovery"),
            }
    except (OSError, RuntimeError, TypeError, ValueError):
        pass

    try:
        if os.path.normcase(os.path.abspath(str(root))) in pruned_user_directories:
            return
    except OSError:
        pass

    visited_directories = 0
    if progress is not None:
        progress.report(current_root=str(root))

    def _on_error(exc: OSError) -> None:
        _log.warning("Could not scan Asset Manager folder: %s", exc)

    pending = [root]
    seen_target_ids: set[tuple[Any, ...]] = set()

    def _target_key(path: Path) -> tuple[Any, ...]:
        try:
            stat = path.stat()
            return ("inode", int(stat.st_dev), int(stat.st_ino))
        except OSError:
            # Keep a broken symlink visible so an existing catalog row can be
            # reconciled as MISSING; two broken links are distinct locators.
            return ("broken", os.path.abspath(str(path)))

    def _yield_path(path: Path) -> str | None:
        if not is_supported_asset_path(str(path)):
            return None
        target_key = _target_key(path)
        if target_key in seen_target_ids:
            return None
        seen_target_ids.add(target_key)
        return os.path.abspath(str(path))

    while pending:
        current = pending.pop()
        if cancel_event is not None and cancel_event.is_set():
            return
        current_text = _directory_key(current)
        if current != root and current.is_symlink():
            cache.drop(current_text)
            continue
        try:
            stat = current.stat()
        except OSError as exc:
            cache.drop(current_text)
            _on_error(exc)
            continue
        visited_directories += 1
        if progress is not None:
            progress.add_directory()
        if visited_directories == 10001:
            _log.warning(
                "Asset folder %s is very large (>10000 directories); consider a smaller folder",
                root,
            )
        cached = cache.trusted(
            current_text,
            int(stat.st_mtime_ns),
            recursive=recursive,
        )
        if cached is not None:
            for name in cached["licht"]:
                if cancel_event is not None and cancel_event.is_set():
                    return
                path = current / name
                resolved = _yield_path(path)
                if resolved is None:
                    continue
                if progress is not None:
                    progress.add_project()
                yield resolved
            kept_directories = [current / name for name in cached["dirs"]]
            pending.extend(reversed(kept_directories))
            continue

        kept_directories: list[Path] = []
        licht_names: list[str] = []
        try:
            entries = os.scandir(current)
        except OSError as exc:
            cache.drop(current_text)
            _on_error(exc)
            continue
        with entries:
            for entry in entries:
                name = entry.name
                if _entry_is_dir(entry):
                    if _directory_is_pruned(name):
                        continue
                    child_text = os.path.normcase(
                        os.path.abspath(os.path.join(current_text, name))
                    )
                    if any(
                        child_text == prune or child_text.startswith(prune + os.sep)
                        for prune in pruned_user_directories
                    ):
                        continue
                    if recursive:
                        kept_directories.append(Path(entry.path))
                    continue
                path = Path(entry.path)
                # A symlink to a .licht file is a locator, while a symlinked
                # directory is never traversed.  Broken .licht links remain
                # discoverable for an existing catalog row to become MISSING.
                is_link = bool(getattr(entry, "is_symlink", lambda: False)())
                if is_link:
                    if not is_supported_asset_path(str(path)):
                        continue
                elif not entry.is_file(follow_symlinks=False):
                    continue
                licht_names.append(name)
                if cancel_event is not None and cancel_event.is_set():
                    return
                try:
                    resolved = _yield_path(path)
                    if resolved is None:
                        continue
                    if progress is not None:
                        progress.add_project()
                    yield resolved
                except OSError as exc:
                    _log.warning("Could not inspect Asset Manager path %s: %s", path, exc)
        cache.replace(
            current_text,
            int(stat.st_mtime_ns),
            [path.name for path in kept_directories],
            licht_names,
            recursive=recursive,
        )
        pending.extend(reversed(kept_directories))


def scan_asset_folder(
    index: Any,
    folder_id: str,
    directory: str,
    cancel_event: threading.Event | None = None,
    progress: AssetFolderScanProgress | None = None,
    *,
    recursive: bool = True,
) -> AssetFolderScanResult:
    """Discover and register .licht projects from one real filesystem folder."""
    if cancel_event is not None and cancel_event.is_set():
        return AssetFolderScanResult(cancelled=True)
    if not Path(directory).expanduser().is_dir():
        _log.warning("Asset Manager folder is unavailable: %s", directory)
        return AssetFolderScanResult(unavailable=True)
    if progress is not None:
        progress.report(current_root=directory)
    cache = _DirectoryScanCache()
    try:
        if callable(getattr(index, "reconcile_observations", None)):
            discovered = list(
                iter_licht_projects(
                    directory, cancel_event, progress,
                    scan_cache=cache, recursive=recursive,
                )
            )
            was_cancelled = cancel_event is not None and cancel_event.is_set()
            added, already, failed, _ = _commit_registration_batch(
                index, [(path, folder_id) for path in discovered], cancel_event,
                folder_ids={folder_id}, save=True,
            )
            was_cancelled = was_cancelled or (
                cancel_event is not None and cancel_event.is_set()
            )
            return AssetFolderScanResult(
                discovered=len(discovered), added=added, already_cataloged=already,
                failed=failed, cancelled=was_cancelled,
            )
        return _register_discovered_streaming(
            index,
            (
                (path, folder_id)
                for path in iter_licht_projects(
                    directory, cancel_event, progress,
                    scan_cache=cache, recursive=recursive,
                )
            ),
            cancel_event,
            progress,
        )
    finally:
        cache.persist()


def scan_all_asset_folders(
    index: Any,
    cancel_event: threading.Event | None = None,
    progress: AssetFolderScanProgress | None = None,
) -> AssetFolderScanResult:
    """Scan every real folder, assigning projects to the most-specific root."""
    roots: list[tuple[Path, str, bool]] = []
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
        roots.append((root, folder_id, folder.get("recursive", True) is not False))

    roots.sort(
        key=lambda item: (
            -len(item[0].parts),
            os.path.normcase(str(item[0])),
            item[1],
        )
    )

    cache = _DirectoryScanCache()

    def _iter_all() -> Iterator[tuple[str, str]]:
        seen_paths: set[str] = set()
        for root, assigned_folder_id, recursive in roots:
            if cancel_event is not None and cancel_event.is_set():
                return
            if progress is not None:
                progress.report(current_root=str(root))
            for path in iter_licht_projects(
                str(root), cancel_event, progress,
                scan_cache=cache, recursive=recursive,
            ):
                path_key = os.path.normcase(path)
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                yield path, assigned_folder_id

    if cancel_event is not None and cancel_event.is_set():
        return AssetFolderScanResult(cancelled=True)
    if not roots:
        return AssetFolderScanResult()
    try:
        if callable(getattr(index, "reconcile_observations", None)):
            discovered = list(_iter_all())
            was_cancelled = cancel_event is not None and cancel_event.is_set()
            added, already, failed, _ = _commit_registration_batch(
                index, discovered, cancel_event,
                folder_ids={folder_id for _root, folder_id, _recursive in roots},
                save=True,
            )
            was_cancelled = was_cancelled or (
                cancel_event is not None and cancel_event.is_set()
            )
            return AssetFolderScanResult(
                discovered=len(discovered), added=added, already_cataloged=already,
                failed=failed, cancelled=was_cancelled,
            )
        return _register_discovered_streaming(index, _iter_all(), cancel_event, progress)
    finally:
        cache.persist()


def verify_catalog_projects(
    index: Any,
    cancel_event: threading.Event | None = None,
    *,
    visible_asset_ids: Iterable[str] | None = None,
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
    if visible_asset_ids is not None:
        visible = [str(asset_id) for asset_id in visible_asset_ids]
        visible_set = set(visible)
        asset_ids = [
            *[asset_id for asset_id in visible if asset_id in asset_ids],
            *[asset_id for asset_id in asset_ids if asset_id not in visible_set],
        ]
    verified = 0
    batch: list[str] = []
    last_flush = time.monotonic()

    def flush(*, pause: bool = True) -> bool:
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
        if pause and interval_s > 0 and cancel_event is not None:
            cancel_event.wait(interval_s)
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
    flush(pause=False)
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
        # os.walk sorted filenames before yielding them. Keep that stable
        # locator choice while retaining streaming between registration batches.
        batch.sort(key=lambda item: os.path.normcase(item[0]))
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
    return getattr(existing, "status", "AVAILABLE") not in _REINSPECT_STATUSES


def _known_path_is_unchanged(index: Any, existing: Any, path: str) -> bool:
    if getattr(existing, "status", "AVAILABLE") in _REINSPECT_STATUSES:
        return False
    try:
        stat = os.stat(path)
    except OSError:
        return False
    identity = getattr(existing, "stat_identity", None) or {}
    if identity:
        for key in ("size", "mtime_ns", "st_dev", "st_ino", "st_ctime_ns"):
            stat_key = "st_" + key if not key.startswith("st_") else key
            if key in identity and int(identity[key]) != int(getattr(stat, stat_key, -1)):
                return False
    else:
        size = getattr(existing, "path_size_bytes", None)
        mtime_ns = getattr(existing, "path_mtime_ns", None)
        if size is not None and mtime_ns is not None and (
            int(stat.st_size), int(stat.st_mtime_ns)
        ) != (int(size), int(mtime_ns)):
            return False
    cheap_head = getattr(index, "_cheap_head_identity", None)
    if callable(cheap_head):
        observed = cheap_head(path)
        expected_uuid = str(getattr(existing, "project_uuid", getattr(existing, "id", "")))
        expected_commit = str(getattr(existing, "commit_uuid", "") or "")
        if observed is not None and observed != (expected_uuid, expected_commit):
            return False
    return True


def _commit_registration_batch(
    index: Any,
    batch: list[tuple[str, str]],
    cancel_event: threading.Event | None,
    *,
    folder_ids: set[str] | None = None,
    save: bool = False,
) -> tuple[int, int, int, bool]:
    """Commit one discovered batch. Cancel drops this batch if it is not committed."""
    if not batch and folder_ids is None:
        return 0, 0, 0, False
    if cancel_event is not None and cancel_event.is_set():
        return 0, 0, 0, True

    inspect = getattr(index, "_inspect_path", None)
    if not callable(inspect):
        return _commit_batch_with_snapshot(index, batch, cancel_event)

    prepared: list[tuple[str, str, Any, ProjectPathIdentity]] = []
    already_cataloged = 0
    repair_only = 0
    failed = 0
    find_by_path = getattr(index, "find_asset_by_path", None)
    for path, folder_id in batch:
        if cancel_event is not None and cancel_event.is_set():
            return 0, 0, 0, True
        existing = None
        try:
            path_identity = ProjectPathIdentity.capture(path)
            if callable(find_by_path):
                existing = find_by_path(path)
                if existing is not None and _known_path_is_unchanged(index, existing, path):
                    already_cataloged += 1
                    if callable(getattr(index, "reconcile_observations", None)):
                        prepared.append(
                            (
                                path,
                                folder_id,
                                SimpleNamespace(
                                    project_uuid=getattr(existing, "project_uuid", getattr(existing, "id", "")),
                                    file_uuid=getattr(existing, "file_uuid", ""),
                                    commit_uuid=getattr(existing, "commit_uuid", ""),
                                    generation=getattr(existing, "generation", 0),
                                    created_at_unix_ns=getattr(existing, "created_at_unix_ns", 0),
                                    saved_at_unix_ns=getattr(existing, "saved_at_unix_ns", 0),
                                    physical_file_size=getattr(existing, "file_size_bytes", 0),
                                    role=SimpleNamespace(name=getattr(existing, "role", "MASTER")),
                                    open_state=SimpleNamespace(name=getattr(existing, "open_state", "OPEN")),
                                    has_preview=getattr(existing, "has_preview", False),
                                    iteration=getattr(existing, "iteration", None),
                                ),
                                path_identity,
                            )
                        )
                    continue
            prepared.append((path, folder_id, inspect(path), path_identity))
        except Exception:
            verify = getattr(index, "verify_asset", None)
            if existing is not None and callable(verify):
                verified = verify(existing.id)
                if verified is not None and getattr(verified, "status", "") == "REPAIR_ONLY":
                    # Keep the known locator and its explicit Repair action.
                    # Missing heads are an expected catalog state, not a
                    # failed registration of a different project.
                    already_cataloged += 1
                    repair_only += 1
                    continue
            failed += 1
            _log.warning("Failed to register Asset Manager project: %s", path, exc_info=True)

    if cancel_event is not None and cancel_event.is_set():
        return 0, 0, 0, True

    reconcile = getattr(index, "reconcile_observations", None)
    if callable(reconcile):
        observations = [
            AssetObservation(
                path=path,
                folder_id=folder_id,
                inspection=inspection,
                stat_identity={
                    key: int(value)
                    for key, value in (getattr(index, "_path_identity", lambda _p: {}) (path) or {}).items()
                },
                path_identity=path_identity,
            )
            for path, folder_id, inspection, path_identity in prepared
        ]
        result = reconcile(
            observations,
            folder_ids=folder_ids if folder_ids is not None else {folder_id for _path, folder_id in batch},
            save=save,
        )
        return (
            int(result.get("added", 0)),
            int(result.get("already_cataloged", 0)) + repair_only,
            int(result.get("failed", 0)) + failed,
            False,
        )

    added = 0
    lock = getattr(index, "_lock", None)

    def commit_prepared() -> None:
        nonlocal added, already_cataloged, failed
        for path, folder_id, inspection, path_identity in prepared:
            try:
                path_identity.validate()
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
