# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal, UUID-based persistence for Asset Manager .licht projects."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from .environment import flag as environment_flag, value as environment_value

_log = logging.getLogger(__name__)
_T = TypeVar("_T")
_ASSET_INDEX_LOCK = threading.RLock()

SCHEMA_VERSION = 2
SUPPORTED_ASSET_EXTENSION = ".licht"
DEFAULT_FOLDER_ID = "default"
DEFAULT_FOLDER_NAME = "Default"


def _normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _normalize_watch_directories(paths: Any) -> List[str]:
    if not isinstance(paths, (list, tuple)):
        return []

    normalized_paths: List[str] = []
    seen = set()
    for path in paths:
        text = str(path).strip()
        if not text:
            continue
        normalized = _normalize_path(text)
        key = os.path.normcase(normalized)
        if key not in seen:
            seen.add(key)
            normalized_paths.append(normalized)
    return normalized_paths


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


def _path_accepts_writes(path: Path) -> bool:
    probe_path: Optional[Path] = None
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".lfs-write-test-", dir=path, delete=False
        ) as probe:
            probe.write(b"ok")
            probe_path = Path(probe.name)
        probe_path.unlink(missing_ok=True)
        return True
    except OSError as exc:
        _log.debug("Asset Manager storage path is not writable: %s (%s)", path, exc)
        if probe_path is not None:
            try:
                probe_path.unlink(missing_ok=True)
            except OSError:
                pass
        return False


def _copy_existing_catalog(source: Path, target: Path) -> None:
    source_library = source / "library.json"
    target_library = target / "library.json"
    if source == target or not source_library.is_file() or target_library.exists():
        return
    try:
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_library, target_library)
        _log.info(
            "Copied Asset Manager catalog from %s to writable storage %s",
            source_library,
            target_library,
        )
    except OSError as exc:
        _log.warning(
            "Could not copy Asset Manager catalog from %s to %s: %s",
            source_library,
            target_library,
            exc,
        )


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

    if environment_flag("LFS_SAFE_MODE", False):
        return native_storage

    candidates = [native_storage]
    appdata = environment_value("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "LichtFeldStudio" / "asset_manager")
    local_appdata = environment_value("LOCALAPPDATA")
    if local_appdata:
        candidates.append(Path(local_appdata) / "LichtFeldStudio" / "asset_manager")
    candidates.append(Path(tempfile.gettempdir()) / "LichtFeldStudio" / "asset_manager")

    for candidate in _dedupe_paths(candidates):
        if _path_accepts_writes(candidate):
            if candidate != native_storage:
                _copy_existing_catalog(native_storage, candidate)
                _log.warning(
                    "Asset Manager catalog path %s is not writable; using %s",
                    native_storage,
                    candidate,
                )
            return candidate

    return native_storage


def resolve_asset_manager_library_path() -> Path:
    return resolve_asset_manager_storage_path() / "library.json"


def is_supported_asset_path(path: str) -> bool:
    return Path(path).suffix.lower() == SUPPORTED_ASSET_EXTENSION


@dataclass
class Folder:
    id: str
    name: str
    watch_directories: List[str]

    def to_storage_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "watch_directories": list(self.watch_directories),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, **self.to_storage_dict()}


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
    exists: bool = False
    available: bool = False
    status: str = "UNVERIFIED"
    error: str = ""

    @property
    def id(self) -> str:
        return self.project_uuid

    def to_storage_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "folder_id": self.folder_id,
        }

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
            "exists": self.exists,
            "available": self.available,
            "status": self.status,
            "error": self.error,
        }


class AssetIndex:
    """Small JSON locator index; project metadata stays inside each .licht file."""

    def __init__(self, library_path: Optional[Path] = None):
        self._library_path = library_path or resolve_asset_manager_library_path()
        self._library_path.parent.mkdir(parents=True, exist_ok=True)
        self._uses_default_library_path = library_path is None
        self._lock = _ASSET_INDEX_LOCK
        self._folders: Dict[str, Folder] = {}
        self._projects: Dict[str, Project] = {}
        self._project_by_path: Dict[str, str] = {}

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
        return {
            project_uuid: project.to_dict()
            for project_uuid, project in self._projects.items()
        }

    @staticmethod
    def _path_key(path: str) -> str:
        return os.path.normcase(_normalize_path(path))

    @staticmethod
    def _inspection_is_master(inspection: Any) -> bool:
        return _enum_name(inspection.role) == "MASTER"

    @staticmethod
    def _inspection_name(path: str) -> str:
        return Path(path).stem

    @staticmethod
    def _inspect_path(path: str) -> Any:
        import lichtfeld as lf

        return lf.io.inspect_project(path)

    def _apply_inspection(self, project: Project, inspection: Any) -> None:
        project.file_uuid = str(inspection.file_uuid)
        project.commit_uuid = str(inspection.commit_uuid)
        project.generation = int(inspection.generation)
        project.created_at_unix_ns = int(inspection.created_at_unix_ns)
        project.saved_at_unix_ns = int(inspection.saved_at_unix_ns)
        project.file_size_bytes = int(inspection.physical_file_size)
        project.role = _enum_name(inspection.role)
        project.open_state = _enum_name(inspection.open_state)
        project.has_preview = bool(inspection.has_preview)
        project.exists = True
        project.available = project.role == "MASTER" and project.open_state == "OPEN"
        project.status = "AVAILABLE" if project.available else "UNSUPPORTED"
        project.error = ""

    def _clear_runtime(self, project: Project, status: str, error: str = "") -> None:
        project.file_uuid = ""
        project.commit_uuid = ""
        project.generation = 0
        project.created_at_unix_ns = 0
        project.saved_at_unix_ns = 0
        project.file_size_bytes = 0
        project.role = ""
        project.open_state = ""
        project.has_preview = False
        project.exists = status != "MISSING"
        project.available = False
        project.status = status
        project.error = error

    def _refresh_project(self, project: Project) -> None:
        if not Path(project.path).is_file():
            self._clear_runtime(project, "MISSING")
            return
        try:
            inspection = self._inspect_path(project.path)
            if str(inspection.project_uuid) != project.project_uuid:
                self._clear_runtime(
                    project,
                    "IDENTITY_MISMATCH",
                    "The file at this path belongs to a different project",
                )
            elif not self._inspection_is_master(inspection):
                self._clear_runtime(project, "UNSUPPORTED", "Not a master project container")
            else:
                self._apply_inspection(project, inspection)
        except Exception as exc:
            self._clear_runtime(project, "UNREADABLE", str(exc))

    def _rebuild_path_lookup(self) -> None:
        self._project_by_path = {
            self._path_key(project.path): project_uuid
            for project_uuid, project in self._projects.items()
        }

    def _snapshot_state(
        self,
    ) -> Tuple[Dict[str, Folder], Dict[str, Project], Dict[str, str]]:
        return deepcopy((self._folders, self._projects, self._project_by_path))

    def _restore_state(
        self,
        state: Tuple[Dict[str, Folder], Dict[str, Project], Dict[str, str]],
    ) -> None:
        self._folders, self._projects, self._project_by_path = state

    def _ensure_default_folder(self) -> bool:
        folder = self._folders.get(DEFAULT_FOLDER_ID)
        if folder is None:
            self._folders[DEFAULT_FOLDER_ID] = Folder(
                id=DEFAULT_FOLDER_ID,
                name=DEFAULT_FOLDER_NAME,
                watch_directories=[],
            )
            return True
        if folder.name != DEFAULT_FOLDER_NAME:
            folder.name = DEFAULT_FOLDER_NAME
            return True
        return False

    def _initialize_empty(self) -> None:
        self._folders = {}
        self._projects = {}
        self._project_by_path = {}
        self._ensure_default_folder()

    def _load_v2(self, data: Dict[str, Any]) -> bool:
        folders_data = data.get("folders")
        projects_data = data.get("projects")
        if not isinstance(folders_data, dict) or not isinstance(projects_data, dict):
            raise ValueError("Asset Manager schema v2 requires folders and projects objects")
        normalized = set(data) != {"schema_version", "folders", "projects"}

        folders: Dict[str, Folder] = {}
        for folder_id, value in folders_data.items():
            if not isinstance(folder_id, str) or not isinstance(value, dict):
                raise ValueError("Invalid Asset Manager folder record")
            normalized = normalized or set(value) != {"name", "watch_directories"}
            watch_directories = _normalize_watch_directories(
                value.get("watch_directories", [])
            )
            normalized = normalized or watch_directories != value.get(
                "watch_directories", []
            )
            folders[folder_id] = Folder(
                id=folder_id,
                name=str(value.get("name") or DEFAULT_FOLDER_NAME),
                watch_directories=watch_directories,
            )

        if DEFAULT_FOLDER_ID not in folders:
            folders[DEFAULT_FOLDER_ID] = Folder(
                id=DEFAULT_FOLDER_ID,
                name=DEFAULT_FOLDER_NAME,
                watch_directories=[],
            )
            normalized = True
        elif folders[DEFAULT_FOLDER_ID].name != DEFAULT_FOLDER_NAME:
            folders[DEFAULT_FOLDER_ID].name = DEFAULT_FOLDER_NAME
            normalized = True

        projects: Dict[str, Project] = {}
        project_by_path: Dict[str, str] = {}
        seen_paths = set()
        for project_uuid, value in projects_data.items():
            if not isinstance(value, dict):
                raise ValueError("Invalid Asset Manager project record")
            normalized = normalized or set(value) != {"name", "path", "folder_id"}
            try:
                canonical_uuid = str(uuid.UUID(str(project_uuid)))
            except ValueError as exc:
                raise ValueError(f"Invalid project UUID: {project_uuid}") from exc
            if canonical_uuid != project_uuid:
                raise ValueError(f"Project UUID is not canonical: {project_uuid}")

            stored_path = str(value.get("path") or "")
            path = _normalize_path(stored_path)
            normalized = normalized or path != stored_path
            if not is_supported_asset_path(path):
                raise ValueError(f"Asset Manager project is not a .licht file: {path}")
            path_key = self._path_key(path)
            if path_key in seen_paths:
                raise ValueError(f"Duplicate Asset Manager project path: {path}")
            seen_paths.add(path_key)

            folder_id = str(value.get("folder_id") or DEFAULT_FOLDER_ID)
            if folder_id not in folders:
                folder_id = DEFAULT_FOLDER_ID
                normalized = True
            project = Project(
                project_uuid=canonical_uuid,
                name=str(value.get("name") or self._inspection_name(path)),
                path=path,
                folder_id=folder_id,
            )
            self._refresh_project(project)
            projects[canonical_uuid] = project
            project_by_path[path_key] = canonical_uuid

        # Do not expose a half-loaded catalog if a later row is malformed.
        self._folders = folders
        self._projects = projects
        self._project_by_path = project_by_path
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
        if isinstance(legacy_folders, dict):
            for folder_id, value in legacy_folders.items():
                if not isinstance(folder_id, str) or not isinstance(value, dict):
                    continue
                self._folders[folder_id] = Folder(
                    id=folder_id,
                    name=str(value.get("name") or DEFAULT_FOLDER_NAME),
                    watch_directories=_normalize_watch_directories(
                        value.get("watch_directories", [])
                    ),
                )
        self._ensure_default_folder()

        legacy_projects = legacy_assets
        if not isinstance(legacy_projects, dict):
            legacy_projects = data.get("projects")
        if not isinstance(legacy_projects, dict):
            legacy_projects = {}

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for value in legacy_projects.values():
            if not isinstance(value, dict):
                continue
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

            folder_id = str(
                value.get("folder_id") or value.get("project_id") or DEFAULT_FOLDER_ID
            )
            if folder_id not in self._folders:
                folder_id = DEFAULT_FOLDER_ID
            project = Project(
                project_uuid=project_uuid,
                name=str(value.get("name") or self._inspection_name(path)),
                path=path,
                folder_id=folder_id,
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

    def _preserve_legacy_backup(self, source_path: Path) -> None:
        if source_path == self._library_path:
            return
        backup = self._library_path.with_suffix(".json.bak")
        backup_temp = backup.with_suffix(backup.suffix + ".tmp")
        try:
            self._library_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, backup_temp)
            os.replace(backup_temp, backup)
        finally:
            backup_temp.unlink(missing_ok=True)

    @_synchronized
    def load(self) -> bool:
        previous_state = self._snapshot_state()
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

            is_v2 = data.get("schema_version") == SCHEMA_VERSION
            if is_v2 and not migrating_legacy_location:
                if self._load_v2(data) and not self.save():
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
            os.replace(temp_path, self._library_path)
            return True
        except Exception as exc:
            _log.error("Failed to save Asset Manager library %s: %s", self._library_path, exc)
            return False
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @_synchronized
    def ensure_default_catalog(self) -> None:
        self._initialize_empty()

    @_synchronized
    def create_folder(self, name: str) -> Optional[Folder]:
        previous_state = self._snapshot_state()
        folder = Folder(id=str(uuid.uuid4()), name=name, watch_directories=[])
        self._folders[folder.id] = folder
        if not self.save():
            self._restore_state(previous_state)
            return None
        return folder

    @_synchronized
    def update_folder(self, folder_id: str, **kwargs) -> Optional[Folder]:
        folder = self._folders.get(folder_id)
        if folder is None:
            return None
        previous_state = self._snapshot_state()
        if "name" in kwargs and folder_id != DEFAULT_FOLDER_ID:
            folder.name = str(kwargs["name"])
        if "watch_directories" in kwargs:
            folder.watch_directories = _normalize_watch_directories(
                kwargs["watch_directories"]
            )
        if not self.save():
            self._restore_state(previous_state)
            return None
        return folder

    @_synchronized
    def delete_folder(self, folder_id: str) -> bool:
        folder = self._folders.get(folder_id)
        if folder is None:
            return False
        previous_state = self._snapshot_state()
        if folder_id == DEFAULT_FOLDER_ID:
            folder.watch_directories = []
        else:
            del self._folders[folder_id]
        self._projects = {
            project_uuid: project
            for project_uuid, project in self._projects.items()
            if project.folder_id != folder_id
        }
        self._rebuild_path_lookup()
        if self.save():
            return True
        self._restore_state(previous_state)
        return False

    @_synchronized
    def get_watch_dirs(self, folder_id: str) -> List[str]:
        folder = self._folders.get(folder_id)
        return list(folder.watch_directories) if folder is not None else []

    @_synchronized
    def set_watch_dirs(self, folder_id: str, paths: List[str]) -> bool:
        folder = self._folders.get(folder_id)
        if folder is None:
            return False
        old_paths = folder.watch_directories
        folder.watch_directories = _normalize_watch_directories(paths)
        if self.save():
            return True
        folder.watch_directories = old_paths
        return False

    @_synchronized
    def update_asset(self, asset_id: str, *, save: bool = True, **kwargs) -> Optional[Project]:
        project = self._projects.get(asset_id)
        if project is None:
            return None
        previous_state = self._snapshot_state() if save else None
        if "name" in kwargs:
            project.name = str(kwargs["name"])
        if "folder_id" in kwargs and kwargs["folder_id"] in self._folders:
            project.folder_id = str(kwargs["folder_id"])
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
            self._projects.update(removed)
            self._rebuild_path_lookup()
            return 0
        return len(removed)

    @_synchronized
    def get_asset(self, asset_id: str) -> Optional[Project]:
        return self._projects.get(asset_id)

    @_synchronized
    def register_licht_asset(
        self,
        project_path: str,
        *,
        folder_id: Optional[str] = None,
        name: Optional[str] = None,
        adopt_existing: bool = True,
        save: bool = True,
    ) -> Tuple[Optional[Project], bool]:
        previous_state = self._snapshot_state() if save else None
        path = _normalize_path(project_path)
        if not is_supported_asset_path(path):
            _log.warning("Asset Manager only supports .licht projects: %s", path)
            return None, False
        if not Path(path).is_file():
            raise FileNotFoundError(path)

        target_folder_id = folder_id or DEFAULT_FOLDER_ID
        if target_folder_id not in self._folders:
            _log.error("Cannot register project: folder %s does not exist", target_folder_id)
            return None, False

        inspection = self._inspect_path(path)
        if not self._inspection_is_master(inspection):
            raise ValueError("Asset Manager only registers master .licht project files")
        project_uuid = str(inspection.project_uuid)
        uuid.UUID(project_uuid)

        path_key = self._path_key(path)
        stale_uuid = self._project_by_path.get(path_key)
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
            )
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
                self._refresh_project(project)

            if name is not None and project.name != name:
                project.name = name
                persisted_changed = True
            if adopt_existing and folder_id is not None and project.folder_id != folder_id:
                project.folder_id = folder_id
                persisted_changed = True

        if save and persisted_changed and not self.save():
            assert previous_state is not None
            self._restore_state(previous_state)
            return None, False
        return project, created

    @_synchronized
    def verify_asset(self, asset_id: str) -> Optional[Project]:
        project = self._projects.get(asset_id)
        if project is not None:
            self._refresh_project(project)
        return project

    @_synchronized
    def relink_asset(self, asset_id: str, new_path: str) -> bool:
        project = self._projects.get(asset_id)
        path = _normalize_path(new_path)
        if project is None or not is_supported_asset_path(path) or not Path(path).is_file():
            return False
        inspection = self._inspect_path(path)
        if (
            not self._inspection_is_master(inspection)
            or str(inspection.project_uuid) != project.project_uuid
        ):
            return False

        path_key = self._path_key(path)
        conflicting_uuid = self._project_by_path.get(path_key)
        if conflicting_uuid is not None and conflicting_uuid != asset_id:
            return False
        previous_state = self._snapshot_state()
        self._project_by_path.pop(self._path_key(project.path), None)
        project.path = path
        self._project_by_path[path_key] = asset_id
        self._apply_inspection(project, inspection)
        if self.save():
            return True
        self._restore_state(previous_state)
        return False

    @_synchronized
    def verify_projects(self) -> Tuple[int, int]:
        for project in self._projects.values():
            self._refresh_project(project)
        unavailable = sum(not project.available for project in self._projects.values())
        return unavailable, len(self._projects)

    @_synchronized
    def list_projects(self, folder_id: Optional[str] = None) -> List[Project]:
        projects = list(self._projects.values())
        if folder_id is not None:
            projects = [project for project in projects if project.folder_id == folder_id]
        return projects

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
