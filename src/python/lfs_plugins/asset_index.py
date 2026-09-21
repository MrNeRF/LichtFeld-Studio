# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal, UUID-based persistence for Asset Manager .licht projects."""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from .asset_storage import prune_previews
from .project_identity import ProjectPathIdentity
from .environment import flag as environment_flag, value as environment_value

_log = logging.getLogger(__name__)
_T = TypeVar("_T")
_ASSET_INDEX_LOCK = threading.RLock()
_CASE_SENSITIVITY_BY_DEVICE: Dict[int, bool] = {}

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses the process lock only.
    fcntl = None

SCHEMA_VERSION = 6
SUPPORTED_ASSET_EXTENSION = ".licht"
DEFAULT_FOLDER_ID = "default"

HEALTH_FIX_ACTIONS = {
    "AVAILABLE": None,
    "READING": None,
    "MISSING": "locate",
    "REPLACED_PUBLISHED": "review_replacement",
    "UNREADABLE": "verify",
    "REPAIR_ONLY": "repair",
    "UNSUPPORTED_NEWER": "update",
    "DIVERGED_COPIES": "review_copies",
}

_PROJECT_STORAGE_FIELDS = frozenset(
    {
        "name",
        "path",
        "folder_id",
        "size",
        "mtime_ns",
        "fallback_preview_path",
        "file_uuid",
        "commit_uuid",
        "generation",
        "created_at_unix_ns",
        "saved_at_unix_ns",
        "file_size_bytes",
        "role",
        "open_state",
        "has_preview",
        "preview_width",
        "preview_height",
        "status",
        "iteration",
        "name_origin",
        "previous_project_uuid",
        "aliases",
        "stat_identity",
        "inspection",
    }
)
_INSPECTION_STORAGE_FIELDS = _PROJECT_STORAGE_FIELDS - {
    "name",
    "path",
    "folder_id",
    "size",
    "mtime_ns",
    "fallback_preview_path",
    "name_origin",
    "previous_project_uuid",
    "aliases",
    "stat_identity",
    "inspection",
}


def _normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _stat_identity_from_stat(stat: os.stat_result) -> Dict[str, int]:
    result = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    for name in ("st_dev", "st_ino", "st_ctime_ns"):
        value = getattr(stat, name, None)
        if value is not None:
            result[name] = int(value)
    return result


def _stat_identity(path: str) -> Optional[Dict[str, int]]:
    try:
        return _stat_identity_from_stat(os.stat(path))
    except OSError:
        return None


def _filesystem_is_case_sensitive(path: str) -> bool:
    """Detect the mounted filesystem rather than assuming the host OS."""
    if os.path.normcase("Aa") != "Aa":
        return False
    directory = Path(path).expanduser()
    if not directory.is_dir():
        directory = directory.parent
    try:
        device = int(directory.stat().st_dev)
    except OSError:
        return True
    cached = _CASE_SENSITIVITY_BY_DEVICE.get(device)
    if cached is not None:
        return cached
    result = True
    probe: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".lfsCaseProbe", dir=str(directory), delete=False
        ) as handle:
            probe = Path(handle.name)
        result = not probe.with_name(probe.name.swapcase()).exists()
    except OSError:
        pass
    finally:
        if probe is not None:
            probe.unlink(missing_ok=True)
    _CASE_SENSITIVITY_BY_DEVICE[device] = result
    return result


def _entry_value(entry: Any, name: str, default: Any = None) -> Any:
    if isinstance(entry, dict):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _links_map(links_snapshot: Any) -> Optional[Dict[str, Any]]:
    if links_snapshot is None:
        return None
    if isinstance(links_snapshot, dict):
        if links_snapshot.get("established") is False:
            return None
        if "signed_in" in links_snapshot and not links_snapshot.get("signed_in"):
            return None
        if links_snapshot.get("storage_issue"):
            return None
        links = links_snapshot.get("links")
        return links if isinstance(links, dict) else links_snapshot
    links = getattr(links_snapshot, "links", None)
    return links if isinstance(links, dict) else None


def previous_scene_for(entry: Any, links_snapshot: Any) -> Optional[Dict[str, Any]]:
    """Return the old account-scoped journal link referenced by an entry."""
    previous = _entry_value(entry, "previous_project_uuid", "")
    links = _links_map(links_snapshot)
    if not previous or links is None:
        return None
    value = links.get(str(previous))
    return dict(value) if isinstance(value, dict) else None


def last_known_gallery_label(entry: Any, links_snapshot: Any) -> Optional[str]:
    """Join an established journal snapshot without persisting a projection."""
    links = _links_map(links_snapshot)
    if links is None:
        return None
    project_uuid = str(_entry_value(entry, "project_uuid", _entry_value(entry, "id", "")))
    link = links.get(project_uuid)
    if not isinstance(link, dict):
        link = previous_scene_for(entry, links_snapshot)
    if not isinstance(link, dict):
        return "Not published"
    labels = {
        "equal": "Published",
        "up_to_date": "Published",
        "local": "Changes here",
        "local_changes": "Changes here",
        "remote": "Changes in gallery",
        "portal_changes": "Changes in gallery",
        "diverged": "Conflict",
        "conflict": "Conflict",
        "remote_only": "Gallery only",
        "remote_deleted": "Removed from gallery",
        "removed_on_portal": "Removed from gallery",
    }
    state = str(link.get("state") or link.get("asset_sync_state") or "")
    return labels.get(state, "Linked, comparison unavailable")


def display_name(entry: Any) -> str:
    """Return the one display name shared by the panel and project choosers."""
    name = str(_entry_value(entry, "name", "") or "").strip()
    origin = str(_entry_value(entry, "name_origin", "") or "")
    path = str(_entry_value(entry, "path", "") or "")
    if name and (origin == "user" or (origin == "" and name != Path(path).stem)):
        return name
    return Path(path).stem or name


def fix_action_for_health(state: str) -> Optional[str]:
    return HEALTH_FIX_ACTIONS.get(str(state), "verify")




def _path_is_within(path: str, directory: str) -> bool:
    try:
        return Path(AssetIndex._path_key(path)).is_relative_to(
            Path(AssetIndex._path_key(directory))
        )
    except (OSError, ValueError):
        return False


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    return str(value).rsplit(".", 1)[-1]


def _synchronized(method: Callable[..., _T]) -> Callable[..., _T]:
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    result: List[Path] = []
    seen = set()
    for path in paths:
        expanded = path.expanduser()
        try:
            key = os.path.normcase(str(expanded.resolve()))
        except OSError:
            key = os.path.normcase(str(expanded))
        if key not in seen:
            seen.add(key)
            result.append(expanded)
    return result


def _legacy_storage_paths(native_storage: Optional[Path] = None) -> List[Path]:
    paths: List[Path] = []
    if native_storage is not None:
        paths.append(native_storage.parent.parent / "asset_manager")
    paths.append(Path.home() / ".lichtfeld" / "asset_manager")
    for variable in ("APPDATA", "LOCALAPPDATA"):
        base = environment_value(variable)
        if base:
            paths.append(Path(base) / "LichtFeldStudio" / "asset_manager")
    return _dedupe_paths(paths)


def resolve_asset_manager_storage_path() -> Path:
    override = environment_value("LFS_ASSET_MANAGER_DIR")
    if override:
        return Path(override).expanduser()

    resolved = environment_value("LFS_RESOLVED_ASSET_LIBRARY_DIR")
    if resolved:
        native_storage = Path(resolved).expanduser()
    else:
        import lichtfeld as lf

        native_storage = Path(lf.io.asset_library_dir())

    return native_storage


def resolve_asset_manager_library_path() -> Path:
    return resolve_asset_manager_storage_path() / "library.json"


def resolve_default_asset_directory() -> Path:
    override = environment_value("LFS_ASSET_MANAGER_ASSETS_DIR")
    if override:
        return Path(override).expanduser()

    try:
        import lichtfeld as lf

        getter = getattr(getattr(lf, "ui", None), "get_project_location", None)
        if callable(getter):
            resolved = str(getter() or "").strip()
            if resolved:
                return Path(resolved).expanduser()
    except Exception as exc:
        _log.warning("Could not read the default project folder: %s", exc)

    from .asset_storage import lichtfeld_home
    return lichtfeld_home() / "projects"


def is_supported_asset_path(path: str) -> bool:
    return Path(path).suffix.lower() == SUPPORTED_ASSET_EXTENSION


@dataclass
class Folder:
    id: str
    path: str
    recursive: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        directory = Path(self.path)
        return directory.name or str(directory)

    def to_storage_dict(self) -> Dict[str, Any]:
        record = {**self.extra, "path": self.path}
        if not self.recursive:
            record["recursive"] = False
        return record

    def to_dict(self) -> Dict[str, Any]:
        record = {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "is_default": self.id == DEFAULT_FOLDER_ID,
        }
        if not self.recursive:
            record["recursive"] = False
        return record


@dataclass
class Project:
    """Persisted locator plus inspection data derived from the .licht file."""

    project_uuid: str
    name: str
    path: str
    folder_id: str
    file_uuid: str = ""
    commit_uuid: str = ""
    generation: int = 0
    created_at_unix_ns: int = 0
    saved_at_unix_ns: int = 0
    file_size_bytes: int = 0
    role: str = ""
    open_state: str = ""
    has_preview: bool = False
    preview_width: int = 0
    preview_height: int = 0
    iteration: Optional[int] = None
    exists: bool = False
    available: bool = False
    status: str = "READING"
    error: str = ""
    relocation_candidate: str = ""
    fallback_preview_path: str = ""
    path_size_bytes: int = 0
    path_mtime_ns: int = 0
    inspection_verified: bool = False
    inspection_restored: bool = False
    name_origin: str = "stem"
    previous_project_uuid: str = ""
    aliases: List[Dict[str, Any]] = field(default_factory=list)
    stat_identity: Dict[str, int] = field(default_factory=dict)
    inspection: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.project_uuid

    @property
    def other_paths(self) -> List[str]:
        return [str(alias.get("path", "")) for alias in self.aliases if alias.get("path")]

    def to_storage_dict(self) -> Dict[str, Any]:
        record = {
            **self.extra,
            "path": self.path,
            "folder_id": self.folder_id,
            "size": self.path_size_bytes,
            "mtime_ns": self.path_mtime_ns,
            "fallback_preview_path": self.fallback_preview_path,
            "name_origin": self.name_origin,
            "previous_project_uuid": self.previous_project_uuid,
            "aliases": [dict(alias) for alias in self.aliases],
            "stat_identity": dict(self.stat_identity),
            "inspection": dict(self.inspection),
        }
        if self.name_origin == "user":
            record["name"] = self.name
        if self.inspection_verified or self.inspection_restored:
            record.update(
                {
                    "file_uuid": self.file_uuid,
                    "commit_uuid": self.commit_uuid,
                    "generation": self.generation,
                    "created_at_unix_ns": self.created_at_unix_ns,
                    "saved_at_unix_ns": self.saved_at_unix_ns,
                    "file_size_bytes": self.file_size_bytes,
                    "role": self.role,
                    "open_state": self.open_state,
                    "has_preview": self.has_preview,
                    "preview_width": self.preview_width,
                    "preview_height": self.preview_height,
                    "status": self.status,
                    "iteration": self.iteration,
                }
            )
        record.pop("gallery", None)
        return record

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.project_uuid,
            "project_uuid": self.project_uuid,
            **self.to_storage_dict(),
            "file_uuid": self.file_uuid,
            "commit_uuid": self.commit_uuid,
            "generation": self.generation,
            "created_at_unix_ns": self.created_at_unix_ns,
            "saved_at_unix_ns": self.saved_at_unix_ns,
            "file_size_bytes": self.file_size_bytes,
            "role": self.role,
            "open_state": self.open_state,
            "has_preview": self.has_preview,
            "preview_width": self.preview_width,
            "preview_height": self.preview_height,
            "exists": self.exists,
            "available": self.available,
            "status": self.status,
            "error": self.error,
            "relocation_candidate": self.relocation_candidate,
            "name_origin": self.name_origin,
            "previous_project_uuid": self.previous_project_uuid,
            "aliases": [dict(alias) for alias in self.aliases],
            "other_paths": [str(alias.get("path", "")) for alias in self.aliases],
            "stat_identity": dict(self.stat_identity),
            "inspection": dict(self.inspection),
        }


@dataclass(frozen=True)
class AssetObservation:
    """A complete scan observation collected before reconciliation."""

    path: str
    folder_id: str
    inspection: Any = None
    error: str = ""
    stat_identity: Dict[str, int] = field(default_factory=dict)
    path_identity: Optional[ProjectPathIdentity] = None

    def __post_init__(self):
        if self.path and self.path_identity is None:
            object.__setattr__(self, "path_identity", ProjectPathIdentity.capture(self.path))

    @property
    def project_uuid(self) -> str:
        return str(getattr(self.inspection, "project_uuid", "") or "")

    @property
    def file_uuid(self) -> str:
        return str(getattr(self.inspection, "file_uuid", "") or "")

    @property
    def commit_uuid(self) -> str:
        return str(getattr(self.inspection, "commit_uuid", "") or "")


class AssetIndex:
    """Small JSON locator index; project metadata stays inside each .licht file."""

    def __init__(
        self,
        library_path: Optional[Path] = None,
        default_folder_path: Optional[Path] = None,
    ):
        self._library_locator = Path(library_path or resolve_asset_manager_library_path()).expanduser().absolute()
        self._library_path = self._library_locator.resolve()
        self._library_path.parent.mkdir(parents=True, exist_ok=True)
        self._uses_default_library_path = library_path is None
        if default_folder_path is None:
            default_folder_path = (
                resolve_default_asset_directory()
                if library_path is None
                else self._library_path.parent
            )
        self._default_folder_path = _normalize_path(str(default_folder_path))
        if not environment_flag("LFS_SAFE_MODE", False):
            try:
                Path(self._default_folder_path).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                _log.warning(
                    "Could not create the default Asset Manager folder %s: %s",
                    self._default_folder_path,
                    exc,
                )
        self._lock = _ASSET_INDEX_LOCK
        self._folders: Dict[str, Folder] = {}
        self._projects: Dict[str, Project] = {}
        self._project_by_path: Dict[str, str] = {}
        self._catalog_epoch = 0
        self._catalog_subscribers: list[Callable[[], None]] = []
        self._assets_snapshot_epoch: Optional[int] = None
        self._assets_snapshot: Optional[Dict[str, Dict[str, Any]]] = None
        self._catalog_extra: Dict[str, Any] = {}
        self.load_issues: List[str] = []
        self.last_error = ""
        self._write_checks: Dict[str, Tuple[ProjectPathIdentity, Optional[str]]] = {}
        self._library_canonical_path = self._library_path.resolve()

    @property
    def library_path(self) -> Path:
        return self._library_path

    @property
    @_synchronized
    def folders(self) -> Dict[str, Dict[str, Any]]:
        return {folder_id: folder.to_dict() for folder_id, folder in self._folders.items()}

    @property
    @_synchronized
    def assets(self) -> Dict[str, Dict[str, Any]]:
        if self._assets_snapshot_epoch != self._catalog_epoch:
            self._assets_snapshot = {
                project_uuid: project.to_dict()
                for project_uuid, project in self._projects.items()
            }
            self._assets_snapshot_epoch = self._catalog_epoch
        return self._assets_snapshot or {}

    @_synchronized
    def snapshot(self) -> Dict[str, Any]:
        return {
            "folders": {
                folder_id: folder.to_dict() for folder_id, folder in self._folders.items()
            },
            "projects": {
                project_uuid: project.to_dict()
                for project_uuid, project in self._projects.items()
            },
            "epoch": self._catalog_epoch,
        }

    @_synchronized
    def get_asset_dict(self, asset_id: str) -> Optional[Dict[str, Any]]:
        project = self._projects.get(str(asset_id))
        return project.to_dict() if project is not None else None

    @_synchronized
    def iter_project_ids(self) -> List[str]:
        return list(self._projects)

    @_synchronized
    def count(self) -> int:
        return len(self._projects)

    @staticmethod
    def _path_key(path: str) -> str:
        normalized = os.path.realpath(_normalize_path(path))
        return normalized if _filesystem_is_case_sensitive(normalized) else normalized.casefold()

    @staticmethod
    def _inspection_is_master(inspection: Any) -> bool:
        return _enum_name(inspection.role) == "MASTER"

    @staticmethod
    def _set_inspection_runtime_state(project: Project) -> None:
        project.exists = True
        project.available = (
            project.role == "MASTER" and project.open_state == "OPEN"
        )
        if project.available:
            project.status = "AVAILABLE"
        elif project.open_state == "REPAIR_ONLY":
            project.status = "REPAIR_ONLY"
        elif project.open_state == "UNSUPPORTED_NEWER":
            project.status = "UNSUPPORTED_NEWER"
        else:
            project.status = "UNSUPPORTED"
        project.error = ""

    @staticmethod
    def _inspection_name(path: str) -> str:
        return Path(path).stem

    @staticmethod
    def _inspect_path(path: str, resolve_fallback: bool = True) -> Any:
        import lichtfeld as lf

        if resolve_fallback:
            return lf.io.inspect_project(path)
        return lf.io.inspect_project(path, resolve_preview_fallback=False)

    @staticmethod
    def _path_stat(path: str) -> Optional[Tuple[int, int]]:
        identity = _stat_identity(path)
        if identity is None:
            return None
        return identity["size"], identity["mtime_ns"]

    @staticmethod
    def _path_identity(path: str) -> Optional[Dict[str, int]]:
        return _stat_identity(path)

    @staticmethod
    def _cheap_head_identity(path: str) -> Optional[Tuple[str, str]]:
        """Use a native head-only reader when a binding provides one."""
        try:
            import lichtfeld as lf

            io = getattr(lf, "io", None)
            reader = getattr(io, "inspect_project_head", None)
            if not callable(reader):
                reader = getattr(io, "read_project_head", None)
            if not callable(reader):
                return None
            value = reader(path)
            project_uuid = getattr(value, "project_uuid", None)
            commit_uuid = getattr(value, "commit_uuid", None)
            if isinstance(value, dict):
                project_uuid = value.get("project_uuid", project_uuid)
                commit_uuid = value.get("commit_uuid", commit_uuid)
            if project_uuid and commit_uuid:
                return str(project_uuid), str(commit_uuid)
        except Exception:
            _log.debug("Cheap Asset Manager head read failed", exc_info=True)
        return None

    def _touch_catalog(self) -> None:
        self._catalog_epoch += 1
        self._assets_snapshot_epoch = None
        self._assets_snapshot = None
        for callback in tuple(self._catalog_subscribers):
            callback()

    @_synchronized
    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Notify changes on the writer thread; callbacks only enqueue UI work."""
        self._catalog_subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._catalog_subscribers:
                    self._catalog_subscribers.remove(callback)

        return unsubscribe

    @_synchronized
    def catalog_epoch(self) -> int:
        return self._catalog_epoch

    def _apply_inspection(self, project: Project, inspection: Any) -> None:
        if self._projects.get(project.project_uuid) is project:
            self._remember_identity(project.path, project.project_uuid)
        project.file_uuid = str(inspection.file_uuid)
        project.commit_uuid = str(inspection.commit_uuid)
        project.generation = int(inspection.generation)
        project.created_at_unix_ns = int(inspection.created_at_unix_ns)
        project.saved_at_unix_ns = int(inspection.saved_at_unix_ns)
        project.file_size_bytes = int(inspection.physical_file_size)
        project.role = _enum_name(inspection.role)
        project.open_state = _enum_name(inspection.open_state)
        project.has_preview = bool(inspection.has_preview)
        project.preview_width = int(getattr(inspection, "preview_width", 0) or 0)
        project.preview_height = int(getattr(inspection, "preview_height", 0) or 0)
        iteration = getattr(inspection, "iteration", None)
        if iteration is not None:
            try:
                project.iteration = int(iteration)
            except (TypeError, ValueError):
                pass
        project.fallback_preview_path = str(
            getattr(inspection, "fallback_preview_path", "") or ""
        )
        self._set_inspection_runtime_state(project)
        project.inspection_verified = True
        project.inspection_restored = False
        metadata = self._path_stat(project.path)
        if metadata is not None:
            project.path_size_bytes, project.path_mtime_ns = metadata
        identity = self._path_identity(project.path)
        if identity is not None:
            project.stat_identity = identity
        facts = getattr(inspection, "inspection", None)
        if isinstance(facts, dict):
            project.inspection = {"version": 1, **facts}
        else:
            project.inspection = {"version": 1, "iteration": project.iteration}
        self._touch_catalog()

    @staticmethod
    def _has_persisted_inspection(value: Dict[str, Any]) -> bool:
        return {
            "size",
            "mtime_ns",
            "fallback_preview_path",
        }.issubset(value) and {
            "file_uuid",
            "commit_uuid",
            "generation",
            "created_at_unix_ns",
            "saved_at_unix_ns",
            "file_size_bytes",
            "role",
            "open_state",
            "has_preview",
        }.issubset(value)

    def _restore_inspection(self, project: Project, value: Dict[str, Any]) -> None:
        project.file_uuid = str(value["file_uuid"] or "")
        project.commit_uuid = str(value["commit_uuid"] or "")
        project.generation = int(value["generation"])
        project.created_at_unix_ns = int(value["created_at_unix_ns"])
        project.saved_at_unix_ns = int(value["saved_at_unix_ns"])
        project.file_size_bytes = int(value["file_size_bytes"])
        project.role = str(value["role"] or "")
        project.open_state = str(value["open_state"] or "")
        project.has_preview = bool(value["has_preview"])
        project.preview_width = int(value.get("preview_width", 0) or 0)
        project.preview_height = int(value.get("preview_height", 0) or 0)
        project.iteration = value.get("iteration")
        if project.iteration is not None:
            project.iteration = int(project.iteration)
        project.stat_identity = dict(value.get("stat_identity") or {})
        project.inspection = dict(value.get("inspection") or {})
        project.status = str(value.get("status") or "READING")
        if project.status == "UNVERIFIED":
            project.status = "READING"
        if project.status in {"MISSING", "UNREADABLE", "IDENTITY_MISMATCH", "UNSUPPORTED"}:
            project.exists = project.status != "MISSING"
            project.available = False
        else:
            self._set_inspection_runtime_state(project)
        project.inspection_verified = False
        project.inspection_restored = True

    def _clear_runtime(
        self,
        project: Project,
        status: str,
        error: str = "",
        *,
        file_size_bytes: int = 0,
    ) -> None:
        if status != "MISSING":
            project.file_uuid = ""
            project.commit_uuid = ""
            project.generation = 0
            project.created_at_unix_ns = 0
            project.saved_at_unix_ns = 0
            project.file_size_bytes = int(file_size_bytes)
            project.role = ""
            project.open_state = ""
            project.has_preview = False
            project.preview_width = 0
            project.preview_height = 0
        project.fallback_preview_path = ""
        project.exists = status != "MISSING"
        project.available = False
        project.status = status
        project.error = error
        if status != "MISSING":
            project.path_size_bytes = 0
            project.path_mtime_ns = 0
        project.inspection_verified = True
        project.inspection_restored = False
        self._touch_catalog()

    def _read_project_runtime(
        self,
        path: str,
        expected_uuid: str,
        *,
        resolve_fallback: bool = False,
        known_metadata: Optional[Any] = None,
    ) -> Tuple[str, Any]:
        metadata = self._path_stat(path)
        if metadata is None:
            return "MISSING", None
        if known_metadata is not None:
            if isinstance(known_metadata, dict):
                current_identity = self._path_identity(path) or {}
                unchanged = all(
                    current_identity.get(key) == int(value)
                    for key, value in known_metadata.items()
                    if key in {"size", "mtime_ns", "st_dev", "st_ino", "st_ctime_ns"}
                )
                expected_commit = str(known_metadata.get("commit_uuid") or "")
            else:
                unchanged = metadata == known_metadata
                expected_commit = ""
            if unchanged:
                head = self._cheap_head_identity(path)
                if head is None or (
                    head[0] == expected_uuid
                    and (not expected_commit or head[1] == expected_commit)
                ):
                    return "UNCHANGED", metadata
        try:
            if resolve_fallback:
                inspection = self._inspect_path(path)
            else:
                try:
                    inspection = self._inspect_path(path, False)
                except TypeError:
                    # Test doubles and older native bindings have the old
                    # one-argument shape; bulk verification remains correct.
                    inspection = self._inspect_path(path)
        except Exception as exc:
            # Full inspection needs a valid head. The native classifier can
            # still distinguish two damaged heads from an unreadable file.
            try:
                import lichtfeld as lf
                classification = lf.io.classify_project(path)
                if _enum_name(classification.state) == "REPAIR_ONLY":
                    return "REPAIR_ONLY", str(classification.diagnostic or exc)
            except Exception:
                pass
            return "UNREADABLE", str(exc)
        if str(inspection.project_uuid) != expected_uuid:
            return (
                "IDENTITY_MISMATCH",
                {
                    "error": "The file at this path belongs to a different project",
                    "file_size_bytes": int(inspection.physical_file_size),
                    "inspection": inspection,
                },
            )
        if not self._inspection_is_master(inspection):
            return "UNSUPPORTED", "Not a master project container"
        return "AVAILABLE", inspection

    def _apply_runtime_result(self, project: Project, kind: str, payload: Any) -> None:
        if kind == "AVAILABLE":
            project.relocation_candidate = ""
            self._apply_inspection(project, payload)
            return
        if kind == "IDENTITY_MISMATCH":
            self._clear_runtime(
                project,
                kind,
                payload["error"],
                file_size_bytes=payload["file_size_bytes"],
            )
            return
        self._clear_runtime(project, kind, str(payload or ""))
        if kind == "REPAIR_ONLY":
            project.open_state = kind

    def _refresh_project(self, project: Project) -> None:
        identity = ProjectPathIdentity.capture(project.path)
        kind, payload = self._read_project_runtime(
            project.path, project.project_uuid, resolve_fallback=True
        )
        if self._projects.get(project.project_uuid) is project:
            self._write_checks[project.path] = (identity, project.project_uuid if kind == "AVAILABLE" else None)
        self._apply_runtime_result(project, kind, payload)

    def _mutation_preflight(self, project: Project) -> bool:
        """Pin the current project and commit before a catalog mutation."""
        if _stat_identity(project.path) is None:
            return True
        try:
            inspection = self._inspect_path(project.path, False)
        except TypeError:
            inspection = self._inspect_path(project.path)
        except Exception as exc:
            self.last_error = f"Could not check project identity at {project.path}: {exc}"
            _log.warning(self.last_error)
            return False
        if str(getattr(inspection, "project_uuid", "")) != project.project_uuid:
            self.last_error = f"The project identity changed at {project.path}. Refresh Projects and try again."
            return False
        expected_commit = str(project.commit_uuid or "")
        if expected_commit and str(getattr(inspection, "commit_uuid", "")) != expected_commit:
            self.last_error = f"The project changed at {project.path}. Refresh Projects and try again."
            return False
        return True

    def _remember_identity(self, path: str, project_uuid: Optional[str], *, allow_missing: bool = False) -> None:
        identity = ProjectPathIdentity.capture(path)
        expected = None if allow_missing and identity.identity is None else project_uuid
        self._write_checks.setdefault(path, (identity, expected))

    def _check_write_identities(self) -> None:
        if self._library_locator.resolve() != self._library_canonical_path:
            raise ValueError("The library path changed. Reopen Projects and try again.")
        for path, (identity, expected) in self._write_checks.items():
            identity.validate()
            if expected is not None:
                inspection = self._inspect_path(path)
                if str(inspection.project_uuid) != expected:
                    raise ValueError(f"The project identity changed at {path}. Refresh Projects and try again.")
                identity.validate()

    def _rebuild_path_lookup(self) -> None:
        self._project_by_path = {
            self._path_key(project.path): project_uuid
            for project_uuid, project in self._projects.items()
        }

    def _observation_from(self, value: Any, folder_id: str = "") -> AssetObservation:
        if isinstance(value, AssetObservation):
            return AssetObservation(
                path=value.path,
                folder_id=self._folder_id_for_path(value.path) or value.folder_id or folder_id,
                inspection=value.inspection,
                error=value.error,
                stat_identity=dict(value.stat_identity),
                path_identity=value.path_identity,
            )
        if isinstance(value, dict):
            path = str(value.get("path") or "")
            inspection = value.get("inspection")
            effective_folder = self._folder_id_for_path(path) or str(
                value.get("folder_id") or folder_id
            )
            return AssetObservation(
                path=path,
                folder_id=effective_folder,
                inspection=inspection,
                error=str(value.get("error") or ""),
                stat_identity=dict(value.get("stat_identity") or _stat_identity(path) or {}),
                path_identity=value.get("path_identity"),
            )
        path = str(getattr(value, "path", "") or "")
        effective_folder = self._folder_id_for_path(path) or str(
            getattr(value, "folder_id", "") or folder_id
        )
        return AssetObservation(
            path=path,
            folder_id=effective_folder,
            inspection=getattr(value, "inspection", None),
            error=str(getattr(value, "error", "") or ""),
            stat_identity=dict(getattr(value, "stat_identity", None) or _stat_identity(path) or {}),
            path_identity=getattr(value, "path_identity", None),
        )

    def reconcile_observations(
        self,
        observations: Iterable[Any],
        *,
        folder_ids: Iterable[str] | None = None,
        save: bool = True,
    ) -> Dict[str, int]:
        """Resolve all scan observations together, independent of traversal order."""
        normalized = [self._observation_from(item) for item in observations]
        normalized = [item for item in normalized if item.path]
        with self._lock:
            previous_state = self._snapshot_state(project_ids=list(self._projects))
            before_ids = set(self._projects)
            observed_by_uuid: Dict[str, List[AssetObservation]] = {}
            observed_paths: Dict[str, AssetObservation] = {}
            for item in normalized:
                self._write_checks[item.path] = (item.path_identity, item.project_uuid or None)
                if not item.inspection or not self._inspection_is_master(item.inspection):
                    continue
                project_uuid = item.project_uuid
                if not project_uuid:
                    continue
                observed_by_uuid.setdefault(project_uuid, []).append(item)
                observed_paths[self._path_key(item.path)] = item

            changed = False
            added = 0
            replaced = 0
            aliases = 0
            for project_uuid, items in sorted(observed_by_uuid.items()):
                items = sorted(items, key=lambda item: self._path_key(item.path))
                project = self._projects.get(project_uuid)
                if project is None:
                    path_key = self._path_key(items[0].path)
                    old_uuid = self._project_by_path.get(path_key)
                    old_project = self._projects.get(old_uuid) if old_uuid else None
                    old_is_observed_elsewhere = bool(
                        old_uuid and len(observed_by_uuid.get(old_uuid, [])) > 0
                    )
                    if old_project is not None and not old_is_observed_elsewhere:
                        self._projects.pop(old_uuid, None)
                        self._project_by_path.pop(path_key, None)
                        project = Project(
                            project_uuid=project_uuid,
                            name=old_project.name,
                            path=items[0].path,
                            folder_id=items[0].folder_id or old_project.folder_id,
                            name_origin=old_project.name_origin,
                            previous_project_uuid=old_uuid,
                        )
                        added += 1
                        replaced += 1
                    else:
                        project = Project(
                            project_uuid=project_uuid,
                            name=self._inspection_name(items[0].path),
                            path=items[0].path,
                            folder_id=items[0].folder_id or DEFAULT_FOLDER_ID,
                            name_origin="stem",
                        )
                        added += 1
                    self._projects[project_uuid] = project
                    changed = True

                path_keys = {self._path_key(item.path) for item in items}
                current_key = self._path_key(project.path)
                current_observation = next(
                    (item for item in items if self._path_key(item.path) == current_key),
                    None,
                )
                if current_observation is None:
                    # A missing old locator or an overwrite at that locator is
                    # resolved only after the whole observation set is known.
                    chosen = items[0]
                    self._project_by_path.pop(current_key, None)
                    project.path = chosen.path
                    project.folder_id = chosen.folder_id or project.folder_id
                    current_observation = chosen
                    changed = True
                elif project.folder_id != current_observation.folder_id and current_observation.folder_id:
                    project.folder_id = current_observation.folder_id
                    changed = True

                self._project_by_path[self._path_key(project.path)] = project_uuid
                self._apply_inspection(project, current_observation.inspection)
                project.stat_identity = dict(
                    current_observation.stat_identity
                    or _stat_identity(project.path)
                    or project.stat_identity
                )
                prior_aliases = {
                    self._path_key(str(alias.get("path", ""))): alias
                    for alias in project.aliases
                    if alias.get("path")
                }
                for item in items:
                    item_key = self._path_key(item.path)
                    if item_key == self._path_key(project.path):
                        continue
                    alias = {
                        "path": _normalize_path(item.path),
                        "file_uuid": item.file_uuid,
                        "commit_uuid": item.commit_uuid,
                        "stat_identity": dict(item.stat_identity),
                    }
                    prior_aliases[item_key] = alias
                    self._project_by_path[item_key] = project_uuid
                project.aliases = list(prior_aliases.values())
                project.status = (
                    "DIVERGED_COPIES"
                    if len({str(item.commit_uuid) for item in items if item.commit_uuid}) > 1
                    else project.status
                )
                aliases += max(0, len(items) - 1)
                changed = True

            scope = set(str(folder_id) for folder_id in (folder_ids or []))
            for project in self._projects.values():
                if scope and project.folder_id not in scope:
                    continue
                if project.project_uuid in observed_by_uuid:
                    continue
                if _stat_identity(project.path) is None:
                    self._remember_identity(project.path, None)
                    project.status = "MISSING"
                    project.exists = False
                    project.available = False
                    changed = True
            self._rebuild_path_lookup()
            for item in normalized:
                if not item.error or item.inspection is not None:
                    continue
                project_uuid = self._project_by_path.get(self._path_key(item.path))
                project = self._projects.get(project_uuid) if project_uuid else None
                if project is not None:
                    self._clear_runtime(
                        project,
                        "MISSING" if _stat_identity(item.path) is None else "UNREADABLE",
                        item.error,
                    )
                    changed = True
            if changed:
                self._touch_catalog()
                if save and not self.save():
                    self._restore_state(previous_state)
                    return {"added": 0, "replaced": 0, "aliases": 0, "failed": 1}
            return {
                "added": added,
                "replaced": replaced,
                "aliases": aliases,
                "failed": 0,
                "already_cataloged": max(0, len(normalized) - added),
                "removed": len(before_ids - set(self._projects)),
            }

    def _folder_id_for_path(self, path: str) -> Optional[str]:
        candidates = [
            folder
            for folder in self._folders.values()
            if _path_is_within(path, folder.path)
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda folder: (len(Path(folder.path).parts), folder.id),
        ).id

    def _add_folder_record(
        self,
        directory: str,
        *,
        preferred_id: Optional[str] = None,
    ) -> Folder:
        normalized = _normalize_path(directory)
        key = self._path_key(normalized)
        for folder in self._folders.values():
            if self._path_key(folder.path) == key:
                return folder
        folder_id = preferred_id or str(uuid.uuid4())
        if folder_id == DEFAULT_FOLDER_ID or folder_id in self._folders:
            folder_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"lichtfeld-asset-folder:{key}")
            )
            if folder_id in self._folders:
                folder_id = str(uuid.uuid4())
        folder = Folder(id=folder_id, path=normalized)
        self._folders[folder.id] = folder
        return folder

    def _snapshot_state(
        self,
        project_ids: Optional[List[str]] = None,
        folder_ids: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Folder], Dict[str, Project], Dict[str, str], Dict[str, Tuple[ProjectPathIdentity, Optional[str]]]]:
        """Capture only records a mutation may edit for save rollback."""
        folders = self._folders.copy()
        for folder_id in folder_ids or []:
            if folder_id in folders:
                folders[folder_id] = copy(folders[folder_id])
        projects = self._projects.copy()
        for project_id in project_ids or []:
            if project_id in projects:
                projects[project_id] = copy(projects[project_id])
        return folders, projects, self._project_by_path.copy(), self._write_checks.copy()

    def _restore_state(
        self,
        state: Tuple[Dict[str, Folder], Dict[str, Project], Dict[str, str], Dict[str, Tuple[ProjectPathIdentity, Optional[str]]]],
    ) -> None:
        self._folders, self._projects, self._project_by_path, self._write_checks = state
        self._touch_catalog()

    def _ensure_default_folder(self) -> bool:
        folder = self._folders.get(DEFAULT_FOLDER_ID)
        if folder is None:
            self._folders[DEFAULT_FOLDER_ID] = Folder(
                id=DEFAULT_FOLDER_ID,
                path=self._default_folder_path,
            )
            return True
        if self._path_key(folder.path) != self._path_key(self._default_folder_path):
            folder.path = self._default_folder_path
            return True
        return False

    def _initialize_empty(self) -> None:
        self._folders = {}
        self._projects = {}
        self._project_by_path = {}
        self._catalog_extra = {}
        self._write_checks.clear()
        self._ensure_default_folder()
        self._touch_catalog()

    def _load_v3(self, data: Dict[str, Any]) -> bool:
        folders_data = data.get("folders")
        projects_data = data.get("projects")
        if not isinstance(folders_data, dict) or not isinstance(projects_data, dict):
            raise ValueError("Asset Manager schema v3 requires folders and projects objects")
        self._folders = {}
        self._projects = {}
        self._project_by_path = {}
        self._catalog_extra = {
            key: value
            for key, value in data.items()
            if key not in {"schema_version", "folders", "projects"}
        }
        normalized = False

        stored_default_path = ""
        stored_default_recursive = True
        stored_default_extra: Dict[str, Any] = {}
        for folder_id, value in folders_data.items():
            if not isinstance(folder_id, str) or not isinstance(value, dict):
                raise ValueError("Invalid Asset Manager folder record")
            folder_extra = {
                key: item
                for key, item in value.items()
                if key not in {"path", "recursive"}
            }
            raw_recursive = value.get("recursive", True)
            recursive = raw_recursive if isinstance(raw_recursive, bool) else True
            normalized = normalized or not isinstance(raw_recursive, bool)
            raw_path = str(value.get("path") or "").strip()
            if not raw_path:
                normalized = True
                continue
            path = _normalize_path(raw_path)
            normalized = normalized or path != raw_path
            if folder_id == DEFAULT_FOLDER_ID:
                stored_default_path = path
                stored_default_recursive = recursive
                stored_default_extra = folder_extra
                continue
            if self._path_key(path) == self._path_key(self._default_folder_path):
                normalized = True
                continue
            before = len(self._folders)
            added_folder = self._add_folder_record(path, preferred_id=folder_id)
            added_folder.recursive = recursive
            added_folder.extra.update(folder_extra)
            normalized = normalized or len(self._folders) == before or path != raw_path

        self._ensure_default_folder()
        self._folders[DEFAULT_FOLDER_ID].recursive = stored_default_recursive
        self._folders[DEFAULT_FOLDER_ID].extra.update(stored_default_extra)
        if (
            stored_default_path
            and self._path_key(stored_default_path)
            != self._path_key(self._default_folder_path)
        ):
            migrated_folder = self._add_folder_record(stored_default_path)
            migrated_folder.recursive = stored_default_recursive
            normalized = True

        seen_paths = set()
        for project_uuid, value in projects_data.items():
            if not isinstance(value, dict):
                self.load_issues.append(
                    f"Skipped catalog entry {project_uuid}: record is not an object"
                )
                continue
            try:
                canonical_uuid = str(uuid.UUID(str(project_uuid)))
            except ValueError:
                self.load_issues.append(
                    f"Skipped catalog entry {project_uuid}: invalid project UUID"
                )
                continue
            if canonical_uuid != project_uuid:
                self.load_issues.append(
                    f"Skipped catalog entry {project_uuid}: project UUID is not canonical"
                )
                continue

            stored_path = str(value.get("path") or "")
            if not stored_path.strip():
                self.load_issues.append(
                    f"Skipped catalog entry {project_uuid}: empty path"
                )
                continue
            path = _normalize_path(stored_path)
            if not is_supported_asset_path(path):
                self.load_issues.append(
                    f"Skipped catalog entry {project_uuid}: not a .licht file: {path}"
                )
                continue
            path_key = self._path_key(path)
            if path_key in seen_paths:
                self.load_issues.append(
                    f"Skipped catalog entry {project_uuid}: duplicate path: {path}"
                )
                continue
            seen_paths.add(path_key)

            normalized = normalized or path != stored_path
            stored_folder_id = str(value.get("folder_id") or DEFAULT_FOLDER_ID)
            folder_id = self._folder_id_for_path(path)
            if folder_id is None:
                folder_id = self._add_folder_record(str(Path(path).parent)).id
                normalized = True
            if folder_id != stored_folder_id:
                normalized = True
            raw_name = str(value.get("name") or self._inspection_name(path))
            if value.get("role") and str(value.get("role")) != "MASTER":
                self.load_issues.append(
                    f"Skipped catalog entry {project_uuid}: non-master container"
                )
                normalized = True
                continue
            name_origin = str(value.get("name_origin") or "")
            if name_origin not in {"user", "stem", "folder"}:
                name_origin = "stem" if raw_name == Path(path).stem else "user"
                normalized = True
            known_fields = _PROJECT_STORAGE_FIELDS | {"gallery"}
            project = Project(
                project_uuid=canonical_uuid,
                name=raw_name,
                path=path,
                folder_id=folder_id,
                name_origin=name_origin,
                previous_project_uuid=str(value.get("previous_project_uuid") or ""),
                aliases=[dict(alias) for alias in value.get("aliases", []) if isinstance(alias, dict)],
                stat_identity=dict(value.get("stat_identity") or _stat_identity(path) or {}),
                inspection=dict(value.get("inspection") or {}),
                extra={key: item for key, item in value.items() if key not in known_fields},
                exists=True,
                status="READING",
                path_size_bytes=int(value.get("size") or 0),
                path_mtime_ns=int(value.get("mtime_ns") or 0),
                fallback_preview_path=str(value.get("fallback_preview_path") or ""),
            )
            if "gallery" in value:
                normalized = True
            if self._has_persisted_inspection(value):
                self._restore_inspection(project, value)
            self._projects[canonical_uuid] = project
            self._project_by_path[path_key] = canonical_uuid
        if self._projects:
            self._touch_catalog()
        return normalized

    def _migrate_legacy(self, data: Dict[str, Any]) -> None:
        self._initialize_empty()

        legacy_folders = data.get("folders")
        legacy_assets = data.get("assets")
        # Before #1265 the object named "projects" held folder-like records;
        # the actual catalog entries were in "assets" and linked by project_id.
        if not isinstance(legacy_folders, dict) and isinstance(legacy_assets, dict):
            legacy_folders = data.get("projects", {})
        if not isinstance(legacy_folders, dict):
            legacy_folders = {}
        for folder_id, value in legacy_folders.items():
            if not isinstance(folder_id, str) or not isinstance(value, dict):
                continue
            raw_directories = value.get("watch_directories", [])
            if not isinstance(raw_directories, (list, tuple)):
                raw_directories = []
            for index, directory in enumerate(raw_directories):
                text = str(directory or "").strip()
                if not text:
                    continue
                preferred_id = folder_id if index == 0 and folder_id != DEFAULT_FOLDER_ID else None
                self._add_folder_record(text, preferred_id=preferred_id)

        legacy_projects = legacy_assets
        if not isinstance(legacy_projects, dict):
            legacy_projects = data.get("projects")
        if not isinstance(legacy_projects, dict):
            legacy_projects = {}

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        strict_v2 = data.get("schema_version") == 2
        for legacy_id, value in legacy_projects.items():
            if not isinstance(value, dict):
                continue
            if strict_v2:
                canonical_id = str(uuid.UUID(str(legacy_id)))
                if canonical_id != legacy_id:
                    raise ValueError(f"Project UUID is not canonical: {legacy_id}")
                value = {**value, "project_uuid": canonical_id}
            raw_path = value.get("absolute_path") or value.get("path")
            if not raw_path:
                continue
            path = _normalize_path(str(raw_path))
            if is_supported_asset_path(path):
                candidates.append((path, value))

        for path, value in sorted(candidates, key=lambda item: self._path_key(item[0])):
            if self._path_key(path) in self._project_by_path:
                continue
            inspection = None
            try:
                if Path(path).is_file():
                    inspection = self._inspect_path(path)
                    if not self._inspection_is_master(inspection):
                        continue
            except Exception as exc:
                _log.warning("Preserving unreadable legacy .licht project %s: %s", path, exc)

            if inspection is not None:
                project_uuid = str(inspection.project_uuid)
                uuid.UUID(project_uuid)
            else:
                project_uuid = ""
                for candidate_id in (value.get("project_uuid"), value.get("id")):
                    try:
                        project_uuid = str(uuid.UUID(str(candidate_id)))
                        break
                    except (ValueError, TypeError, AttributeError):
                        project_uuid = ""
                if not project_uuid:
                    project_uuid = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"lichtfeld-legacy-project:{self._path_key(path)}",
                        )
                    )
            if project_uuid in self._projects:
                if inspection is not None:
                    continue
                project_uuid = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"lichtfeld-legacy-project:{self._path_key(path)}",
                    )
                )
                if project_uuid in self._projects:
                    continue

            folder_id = self._folder_id_for_path(path)
            if folder_id is None:
                folder_id = self._add_folder_record(str(Path(path).parent)).id
            project = Project(
                project_uuid=project_uuid,
                name=str(value.get("name") or self._inspection_name(path)),
                path=path,
                folder_id=folder_id,
                name_origin=(
                    str(value.get("name_origin") or "")
                    if str(value.get("name_origin") or "") in {"user", "stem", "folder"}
                    else ("stem" if str(value.get("name") or self._inspection_name(path)) == Path(path).stem else "user")
                ),
                extra={key: item for key, item in value.items() if key not in _PROJECT_STORAGE_FIELDS},
            )
            if inspection is not None:
                self._apply_inspection(project, inspection)
            elif Path(path).is_file():
                self._clear_runtime(project, "UNREADABLE", "Legacy project needs inspection")
            else:
                self._clear_runtime(project, "MISSING")
            self._projects[project_uuid] = project
            self._project_by_path[self._path_key(path)] = project_uuid
        self._rebuild_path_lookup()

    def _canonical_cleanup_paths(self) -> Optional[Tuple[Path, Path]]:
        if not self._uses_default_library_path or environment_value("LFS_ASSET_MANAGER_DIR"):
            return None
        try:
            expected = resolve_asset_manager_library_path().resolve()
            actual = self._library_path.resolve()
        except OSError:
            return None
        if actual != expected:
            return None

        storage = actual.parent
        if storage.name != "asset_library" or storage.parent.name != "data":
            return None
        legacy = storage.parent.parent / "asset_manager"
        if legacy == storage or legacy.name != "asset_manager":
            return None
        return storage / "thumbnails", legacy

    def _legacy_library_path(self) -> Optional[Path]:
        if not self._uses_default_library_path or environment_value(
            "LFS_ASSET_MANAGER_DIR"
        ):
            return None
        paths = self._canonical_cleanup_paths()
        candidates: List[Path] = []
        if paths is not None:
            candidates.append(paths[1])
        candidates.extend(_legacy_storage_paths())
        target_key = os.path.normcase(str(self._library_path))
        for storage in _dedupe_paths(candidates):
            candidate = storage / "library.json"
            if os.path.normcase(str(candidate)) != target_key and candidate.is_file():
                return candidate
        return None

    def _cleanup_obsolete_storage(self) -> None:
        paths = self._canonical_cleanup_paths()
        if paths is None:
            return
        for obsolete in paths:
            if not obsolete.exists():
                continue
            try:
                shutil.rmtree(obsolete)
                _log.info("Removed obsolete Asset Manager storage: %s", obsolete)
            except OSError as exc:
                _log.warning("Could not remove obsolete Asset Manager storage %s: %s", obsolete, exc)

    @contextmanager
    def _catalog_write_lock(self):
        lock_path = self._library_path.with_name(self._library_path.name + ".lock")
        handle = None
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("a+", encoding="utf-8")
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if handle is not None:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()

    def _preserve_legacy_backup(self, source_path: Path) -> None:
        backup = self._library_path.with_name(self._library_path.name + ".legacy.bak")
        backup_temp = backup.with_suffix(backup.suffix + ".tmp")
        try:
            self._library_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, backup_temp)
            os.replace(backup_temp, backup)
        finally:
            backup_temp.unlink(missing_ok=True)

    def _preserve_v5_backup(self, source_path: Path) -> None:
        """Keep the exact v5 input once; the rolling .bak is not a migration backup."""
        backup = self._library_path.with_name(self._library_path.name + ".v5.bak")
        if backup.exists():
            return
        backup_temp = backup.with_suffix(backup.suffix + ".tmp")
        try:
            self._library_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, backup_temp)
            os.replace(backup_temp, backup)
        finally:
            backup_temp.unlink(missing_ok=True)

    @_synchronized
    def load(self) -> bool:
        self.load_issues = []
        previous_state = self._snapshot_state()
        self._write_checks = {}
        source_path = self._library_path
        migrating_legacy_location = False
        if not source_path.exists():
            legacy_path = self._legacy_library_path()
            if legacy_path is not None and legacy_path.is_file():
                source_path = legacy_path
                migrating_legacy_location = True
            else:
                self._initialize_empty()
                saved = self.save()
                if saved:
                    self._cleanup_obsolete_storage()
                else:
                    self._restore_state(previous_state)
                return saved

        try:
            with source_path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
            if not isinstance(data, dict):
                raise ValueError("Asset Manager catalog root must be an object")

            if isinstance(data.get("schema_version"), int) and data["schema_version"] > SCHEMA_VERSION:
                raise ValueError("Unsupported future Asset Manager catalog schema")
            is_current = data.get("schema_version") in (SCHEMA_VERSION, 3, 4, 5)
            if is_current and not migrating_legacy_location:
                migrating = data.get("schema_version") != SCHEMA_VERSION
                normalized = self._load_v3(data)
                if migrating:
                    if data.get("schema_version") == 5:
                        self._preserve_v5_backup(source_path)
                    else:
                        self._preserve_legacy_backup(source_path)
                if (normalized or migrating) and not self.save():
                    self._restore_state(previous_state)
                    return False
                self._cleanup_obsolete_storage()
            else:
                self._migrate_legacy(data)
                self._preserve_legacy_backup(source_path)
                if not self.save():
                    self._restore_state(previous_state)
                    return False
                self._cleanup_obsolete_storage()
                _log.info("Migrated Asset Manager catalog to schema v%d", SCHEMA_VERSION)
            _log.info(
                "Loaded Asset Manager library with %d folders and %d projects",
                len(self._folders),
                len(self._projects),
            )
            self._touch_catalog()
            return True
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            self._restore_state(previous_state)
            _log.error("Failed to load Asset Manager library %s: %s", source_path, exc)
            return False

    @_synchronized
    def save(self) -> bool:
        temp_path: Optional[Path] = None
        try:
            data = {
                **self._catalog_extra,
                "schema_version": SCHEMA_VERSION,
                "folders": {
                    folder_id: folder.to_storage_dict()
                    for folder_id, folder in self._folders.items()
                },
                "projects": {
                    project_uuid: project.to_storage_dict()
                    for project_uuid, project in self._projects.items()
                },
            }
            with self._catalog_write_lock():
                self._library_path.parent.mkdir(parents=True, exist_ok=True)
                fd, temp_name = tempfile.mkstemp(
                    prefix=f"{self._library_path.stem}.",
                    suffix=".tmp",
                    dir=str(self._library_path.parent),
                )
                temp_path = Path(temp_name)
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(data, stream, indent=2, ensure_ascii=False)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())

                if self._library_path.exists():
                    backup = self._library_path.with_suffix(".json.bak")
                    backup_temp = backup.with_suffix(backup.suffix + ".tmp")
                    try:
                        shutil.copy2(self._library_path, backup_temp)
                        os.replace(backup_temp, backup)
                    finally:
                        backup_temp.unlink(missing_ok=True)
                self._check_write_identities()
                os.replace(temp_path, self._library_path)
            self._write_checks.clear()
            self.last_error = ""
            self._touch_catalog()
            return True
        except Exception as exc:
            self.last_error = str(exc)
            _log.error("Failed to save Asset Manager library %s: %s", self._library_path, exc)
            return False
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @_synchronized
    def add_folder(
        self,
        directory: str,
        *,
        recursive: Optional[bool] = None,
    ) -> Optional[Folder]:
        path = Path(_normalize_path(directory))
        if not path.is_dir():
            return None
        existing = next(
            (
                folder
                for folder in self._folders.values()
                if self._path_key(folder.path) == self._path_key(str(path))
            ),
            None,
        )
        if existing is not None:
            if recursive is None or existing.recursive == bool(recursive):
                return existing
            previous_state = self._snapshot_state(folder_ids=[existing.id])
            existing.recursive = bool(recursive)
            if not self.save():
                self._restore_state(previous_state)
                return None
            return existing
        previous_state = self._snapshot_state()
        folder = self._add_folder_record(str(path))
        folder.recursive = True if recursive is None else bool(recursive)
        if not self.save():
            self._restore_state(previous_state)
            return None
        return folder

    @_synchronized
    def delete_folder(self, folder_id: str) -> int:
        folder = self._folders.get(folder_id)
        if folder is None or folder_id == DEFAULT_FOLDER_ID:
            return 0
        previous_state = self._snapshot_state(
            project_ids=list(self._projects), folder_ids=[DEFAULT_FOLDER_ID]
        )
        removed_ids = [project.project_uuid for project in self._projects.values() if project.folder_id == folder_id]
        for identifier in removed_ids:
            project = self._projects[identifier]
            self._remember_identity(project.path, identifier, allow_missing=True)
        del self._folders[folder_id]
        self._projects = {
            project_uuid: project
            for project_uuid, project in self._projects.items()
            if project.folder_id != folder_id
        }
        self._rebuild_path_lookup()
        if self.save():
            prune_previews(removed_ids)
            return len(removed_ids)
        self._restore_state(previous_state)
        return 0

    @_synchronized
    def clean_missing_entries(self, folder_id: str) -> int:
        """Forget missing catalog rows in a folder; files and journal links stay untouched."""
        candidates = [
            project.project_uuid
            for project in self._projects.values()
            if project.folder_id == str(folder_id)
            and (_stat_identity(project.path) is None or project.status == "MISSING")
        ]
        return self.delete_assets(candidates)

    @_synchronized
    def set_default_folder_path(self, directory: str) -> bool:
        normalized = _normalize_path(directory)
        if not environment_flag("LFS_SAFE_MODE", False):
            try:
                Path(normalized).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                _log.error(
                    "Could not create the default Asset Manager folder %s: %s",
                    normalized,
                    exc,
                )
                return False
            if not Path(normalized).is_dir():
                return False
        folder = self._folders.get(DEFAULT_FOLDER_ID)
        if folder is not None and self._path_key(folder.path) == self._path_key(normalized):
            return True
        previous_state = self._snapshot_state(
            project_ids=list(self._projects), folder_ids=[DEFAULT_FOLDER_ID]
        )
        for project in self._projects.values():
            self._remember_identity(project.path, project.project_uuid, allow_missing=True)
        previous_default_path = self._default_folder_path
        old_path = folder.path if folder is not None else ""
        new_path_key = self._path_key(normalized)
        duplicate_ids = [
            folder_id
            for folder_id, item in self._folders.items()
            if folder_id != DEFAULT_FOLDER_ID
            and self._path_key(item.path) == new_path_key
        ]
        for folder_id in duplicate_ids:
            del self._folders[folder_id]
        self._default_folder_path = normalized
        if folder is None:
            self._folders[DEFAULT_FOLDER_ID] = Folder(DEFAULT_FOLDER_ID, normalized)
        else:
            folder.path = normalized
        if (
            old_path
            and self._path_key(old_path) != new_path_key
            and any(
                _path_is_within(project.path, old_path)
                for project in self._projects.values()
            )
        ):
            self._add_folder_record(old_path)
        for project in self._projects.values():
            resolved_folder = self._folder_id_for_path(project.path)
            if resolved_folder is None:
                resolved_folder = self._add_folder_record(str(Path(project.path).parent)).id
            project.folder_id = resolved_folder
        if self.save():
            return True
        self._default_folder_path = previous_default_path
        self._restore_state(previous_state)
        return False

    @_synchronized
    def folder_id_for_path(self, path: str) -> Optional[str]:
        return self._folder_id_for_path(path)

    @_synchronized
    def update_asset(self, asset_id: str, *, save: bool = True, **kwargs) -> Optional[Project]:
        project = self._projects.get(asset_id)
        if project is None:
            return None
        if kwargs and not self._mutation_preflight(project):
            _log.warning("Asset Manager mutation preflight rejected %s", asset_id)
            return None
        previous_state = (
            self._snapshot_state(project_ids=[asset_id]) if save else None
        )
        self._remember_identity(project.path, asset_id, allow_missing=True)
        if "folder_id" in kwargs:
            target = self._folders.get(str(kwargs["folder_id"]))
            resolved_folder_id = self._folder_id_for_path(project.path)
            if target is None or target.id != resolved_folder_id:
                return None
            project.folder_id = target.id
        if "name" in kwargs:
            project.name = str(kwargs["name"])
            project.name_origin = "user"
        if "viewing_copy" in kwargs:
            project.extra = {**project.extra, "viewing_copy": bool(kwargs["viewing_copy"])}
        if not save:
            self._touch_catalog()
        if save and not self.save():
            assert previous_state is not None
            self._restore_state(previous_state)
            return None
        return project

    @_synchronized
    def delete_asset(self, asset_id: str) -> bool:
        return self.delete_assets([asset_id]) == 1

    @_synchronized
    def delete_assets(self, asset_ids: List[str]) -> int:
        previous_state = self._snapshot_state()
        for asset_id in dict.fromkeys(asset_ids):
            project = self._projects.get(asset_id)
            if project is not None:
                self._remember_identity(project.path, asset_id, allow_missing=True)
        removed: Dict[str, Project] = {}
        for asset_id in dict.fromkeys(asset_ids):
            project = self._projects.pop(asset_id, None)
            if project is None:
                continue
            self._project_by_path.pop(self._path_key(project.path), None)
            removed[asset_id] = project
        if not removed:
            return 0
        if not self.save():
            self._restore_state(previous_state)
            return 0
        self._touch_catalog()
        prune_previews(removed)
        return len(removed)

    @_synchronized
    def get_asset(self, asset_id: str) -> Optional[Project]:
        return self._projects.get(asset_id)

    def register_licht_asset(
        self,
        project_path: str,
        *,
        folder_id: Optional[str] = None,
        name: Optional[str] = None,
        adopt_existing: bool = True,
        save: bool = True,
        inspection: Any = None,
    ) -> Tuple[Optional[Project], bool]:
        path = _normalize_path(project_path)
        planned_path = ProjectPathIdentity.capture(path)
        if not is_supported_asset_path(path):
            _log.warning("Asset Manager only supports .licht projects: %s", path)
            return None, False
        if not Path(path).is_file():
            raise FileNotFoundError(path)

        if inspection is None:
            inspection = self._inspect_path(path)
        if not self._inspection_is_master(inspection):
            raise ValueError("Asset Manager only registers master .licht project files")
        project_uuid = str(inspection.project_uuid)
        uuid.UUID(project_uuid)

        with self._lock:
            previous_state = (
                self._snapshot_state(project_ids=[project_uuid]) if save else None
            )
            target_folder_id = self._folder_id_for_path(path)
            if target_folder_id is None:
                target_folder_id = self._add_folder_record(str(Path(path).parent)).id

            path_key = self._path_key(path)
            stale_uuid = self._project_by_path.get(path_key)
            stale_project = self._projects.get(stale_uuid) if stale_uuid else None
            if stale_uuid is not None and stale_uuid != project_uuid:
                self._projects.pop(stale_uuid, None)
                self._project_by_path.pop(path_key, None)

            project = self._projects.get(project_uuid)
            created = project is None
            persisted_changed = created or stale_uuid is not None
            if project is None:
                project = Project(
                    project_uuid=project_uuid,
                    name=name or self._inspection_name(path),
                    path=path,
                    folder_id=target_folder_id,
                    name_origin="user" if name is not None else "stem",
                    previous_project_uuid=(stale_uuid or ""),
                )
                if stale_project is not None and name is None:
                    project.name = stale_project.name
                    project.name_origin = stale_project.name_origin
                self._projects[project_uuid] = project
                self._project_by_path[path_key] = project_uuid
                self._apply_inspection(project, inspection)
            else:
                use_observed_path = adopt_existing or self._path_key(project.path) == path_key
                if use_observed_path:
                    old_path_key = self._path_key(project.path)
                    if old_path_key != path_key:
                        self._project_by_path.pop(old_path_key, None)
                        project.path = path
                        self._project_by_path[path_key] = project_uuid
                        persisted_changed = True
                    self._apply_inspection(project, inspection)
                else:
                    if not Path(project.path).is_file():
                        project.relocation_candidate = path
                    self._refresh_project(project)

                if name is not None:
                    if project.name != name or project.name_origin != "user":
                        project.name = name
                        project.name_origin = "user"
                        persisted_changed = True
                if adopt_existing and project.folder_id != target_folder_id:
                    project.folder_id = target_folder_id
                    persisted_changed = True

            self._write_checks[path] = (planned_path, project_uuid)
            if save and persisted_changed and not self.save():
                assert previous_state is not None
                self._restore_state(previous_state)
                return None, False
            return project, created

    def verify_asset(self, asset_id: str) -> Optional[Project]:
        with self._lock:
            project = self._projects.get(asset_id)
            if project is None:
                return None
            path = project.path
            expected_uuid = project.project_uuid
        path_identity = ProjectPathIdentity.capture(path)
        kind, payload = self._read_project_runtime(
            path, expected_uuid, resolve_fallback=True
        )
        with self._lock:
            project = self._projects.get(asset_id)
            if project is None:
                return None
            if project.path != path or project.project_uuid != expected_uuid:
                return project
            if kind == "UNCHANGED":
                project.inspection_verified = True
                project.inspection_restored = False
            else:
                previous_state = self._snapshot_state(project_ids=[asset_id])
                self._write_checks[path] = (path_identity, expected_uuid if kind == "AVAILABLE" else None)
                self._apply_runtime_result(project, kind, payload)
                if not self.save():
                    self._restore_state(previous_state)
            return self._projects[asset_id]

    @_synchronized
    def relink_asset(self, asset_id: str, new_path: str) -> bool:
        project = self._projects.get(asset_id)
        path = _normalize_path(new_path)
        planned_path = ProjectPathIdentity.capture(path)
        if project is None or not is_supported_asset_path(path) or not Path(path).is_file():
            return False
        inspection = self._inspect_path(path)
        if (
            not self._inspection_is_master(inspection)
            or str(inspection.project_uuid) != project.project_uuid
        ):
            self.last_error = f"The project identity changed at {path}. Choose the matching project."
            return False

        path_key = self._path_key(path)
        conflicting_uuid = self._project_by_path.get(path_key)
        if conflicting_uuid is not None and conflicting_uuid != asset_id:
            return False
        previous_state = self._snapshot_state(project_ids=[asset_id])
        folder_id = self._folder_id_for_path(path)
        if folder_id is None:
            folder_id = self._add_folder_record(str(Path(path).parent)).id
        self._project_by_path.pop(self._path_key(project.path), None)
        project.path = path
        project.folder_id = folder_id
        project.relocation_candidate = ""
        self._project_by_path[path_key] = asset_id
        self._apply_inspection(project, inspection)
        self._write_checks[path] = (planned_path, asset_id)
        if self.save():
            return True
        self._restore_state(previous_state)
        return False

    def verify_projects_batch(self, asset_ids: List[str]) -> int:
        with self._lock:
            work = []
            for asset_id in dict.fromkeys(asset_ids):
                project = self._projects.get(asset_id)
                if project is not None:
                    work.append(
                        (
                            asset_id,
                            project.path,
                            project.project_uuid,
                            ProjectPathIdentity.capture(project.path),
                        )
                    )
        results = []
        for asset_id, path, expected_uuid, path_identity in work:
            results.append(
                (
                    asset_id,
                    path,
                    expected_uuid,
                    path_identity,
                    self._read_project_runtime(
                        path, expected_uuid
                    ),
                )
            )
        with self._lock:
            previous_state = self._snapshot_state(project_ids=list(self._projects))
            verified = 0
            changed = False
            for asset_id, path, expected_uuid, path_identity, (kind, payload) in results:
                project = self._projects.get(asset_id)
                if (
                    project is None
                    or project.path != path
                    or project.project_uuid != expected_uuid
                ):
                    continue
                if kind == "UNCHANGED":
                    project.inspection_verified = True
                    project.inspection_restored = False
                else:
                    self._write_checks[path] = (path_identity, expected_uuid if kind == "AVAILABLE" else None)
                    self._apply_runtime_result(project, kind, payload)
                    changed = True
                verified += 1
            if changed and not self.save():
                self._restore_state(previous_state)
                return 0
            return verified

    @_synchronized
    def list_projects(self, folder_id: Optional[str] = None) -> List[Project]:
        projects = list(self._projects.values())
        if folder_id is not None:
            projects = [project for project in projects if project.folder_id == folder_id]
        return projects

    def reconcile_all(
        self,
        *,
        progress: Optional[Callable[..., Any]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, int]:
        """Force-read every catalog path and reconcile one complete observation set."""
        with self._lock:
            projects = list(self._projects.values())
        total = len(projects)
        observations: List[AssetObservation] = []
        for done, project in enumerate(projects):
            if cancel_event is not None and cancel_event.is_set():
                return {"cancelled": 1, "processed": done, "total": total}
            path_identity = ProjectPathIdentity.capture(project.path)
            kind, payload = self._read_project_runtime(
                project.path, project.project_uuid, resolve_fallback=True
            )
            inspection = (
                payload if kind == "AVAILABLE" else
                payload.get("inspection") if kind == "IDENTITY_MISMATCH" and isinstance(payload, dict) else None
            )
            observations.append(
                AssetObservation(
                    path=project.path,
                    folder_id=project.folder_id,
                    inspection=inspection,
                    error=str(payload or "") if inspection is None else "",
                    stat_identity=_stat_identity(project.path) or project.stat_identity,
                    path_identity=path_identity,
                )
            )
            if progress is not None:
                try:
                    progress(done + 1, total, project.path)
                except TypeError:
                    try:
                        progress({"done": done + 1, "total": total, "path": project.path})
                    except TypeError:
                        progress(done + 1)
        result = self.reconcile_observations(
            observations, folder_ids=self._folders.keys(), save=True
        )
        result.update({"processed": total, "total": total, "cancelled": 0})
        return result

    @_synchronized
    def find_asset_by_path(
        self,
        project_path: str,
        folder_id: Optional[str] = None,
    ) -> Optional[Project]:
        project_uuid = self._project_by_path.get(self._path_key(project_path))
        project = self._projects.get(project_uuid) if project_uuid else None
        if project is not None and (folder_id is None or project.folder_id == folder_id):
            return project
        return None


class LibraryService:
    """Application-lifetime worker facade for the local AssetIndex."""

    def __init__(self, index: Optional[AssetIndex] = None):
        self.index = index or AssetIndex()
        self._commands: queue.Queue[Any] = queue.Queue()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run, name="lichtfeld-library", daemon=True
        )
        self._worker.start()

    def _run(self) -> None:
        while True:
            command = self._commands.get()
            if command is None:
                return
            method, args, kwargs, result = command
            try:
                if method == "__scan__":
                    from .asset_watch import scan_all_asset_folders

                    value = scan_all_asset_folders(self.index, *args, **kwargs)
                else:
                    value = getattr(self.index, method)(*args, **kwargs)
                result.put((True, value))
            except BaseException as exc:  # return failures to the caller
                result.put((False, exc))

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("LibraryService is closed")
        if threading.current_thread() is self._worker:
            return getattr(self.index, method)(*args, **kwargs)
        result: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._commands.put((method, args, kwargs, result))
        ok, value = result.get()
        if not ok:
            raise value
        return value

    def snapshot(self) -> Dict[str, Any]:
        return self.index.snapshot()

    def register(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("register_licht_asset", *args, **kwargs)

    def verify(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("verify_asset", *args, **kwargs)

    def list_projects(self) -> List[Project]:
        return self._call("list_projects")

    def verify_projects_batch(self, asset_ids: List[str]) -> int:
        return self._call("verify_projects_batch", asset_ids)

    def relink(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("relink_asset", *args, **kwargs)

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("delete_asset", *args, **kwargs)

    def add_folder(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("add_folder", *args, **kwargs)

    def remove_folder(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("delete_folder", *args, **kwargs)

    def clean_missing(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("clean_missing_entries", *args, **kwargs)

    def reconcile(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("reconcile_all", *args, **kwargs)

    def scan(self, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("LibraryService is closed")
        if threading.current_thread() is self._worker:
            from .asset_watch import scan_all_asset_folders

            return scan_all_asset_folders(self.index, *args, **kwargs)
        result: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._commands.put(("__scan__", args, kwargs, result))
        ok, value = result.get()
        if not ok:
            raise value
        return value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._commands.put(None)
        self._worker.join(timeout=5.0)

    def __enter__(self) -> "LibraryService":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
