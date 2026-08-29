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

SCHEMA_VERSION = 3
SUPPORTED_ASSET_EXTENSION = ".licht"
DEFAULT_FOLDER_ID = "default"


def _normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _path_is_within(path: str, directory: str) -> bool:
    try:
        return Path(_normalize_path(path)).is_relative_to(
            Path(_normalize_path(directory))
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


def resolve_default_asset_directory() -> Path:
    override = environment_value("LFS_ASSET_MANAGER_ASSETS_DIR")
    if override:
        return Path(override).expanduser()

    try:
        import lichtfeld as lf

        getter = getattr(getattr(lf, "ui", None), "get_asset_manager_directory", None)
        if callable(getter):
            resolved = str(getter() or "").strip()
            if resolved:
                return Path(resolved).expanduser()
    except Exception:
        pass

    return Path.home() / ".lichtfeld" / "assets"


def is_supported_asset_path(path: str) -> bool:
    return Path(path).suffix.lower() == SUPPORTED_ASSET_EXTENSION


@dataclass
class Folder:
    id: str
    path: str

    @property
    def name(self) -> str:
        directory = Path(self.path)
        return directory.name or str(directory)

    def to_storage_dict(self) -> Dict[str, Any]:
        return {"path": self.path}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "is_default": self.id == DEFAULT_FOLDER_ID,
        }


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
    relocation_candidate: str = ""
    fallback_preview_path: str = ""

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
            "relocation_candidate": self.relocation_candidate,
            "fallback_preview_path": self.fallback_preview_path,
        }


class AssetIndex:
    """Small JSON locator index; project metadata stays inside each .licht file."""

    def __init__(
        self,
        library_path: Optional[Path] = None,
        default_folder_path: Optional[Path] = None,
    ):
        self._library_path = library_path or resolve_asset_manager_library_path()
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
        self.load_issues: List[str] = []

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

    def _touch_catalog(self) -> None:
        self._catalog_epoch += 1

    @_synchronized
    def catalog_epoch(self) -> int:
        return self._catalog_epoch

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
        project.fallback_preview_path = str(
            getattr(inspection, "fallback_preview_path", "") or ""
        )
        project.exists = True
        project.available = project.role == "MASTER" and project.open_state == "OPEN"
        if project.available:
            project.status = "AVAILABLE"
        elif project.open_state == "REPAIR_ONLY":
            project.status = "REPAIR_ONLY"
        elif project.open_state == "UNSUPPORTED_NEWER":
            project.status = "UNSUPPORTED_NEWER"
        else:
            project.status = "UNSUPPORTED"
        project.error = ""
        self._touch_catalog()

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
        project.fallback_preview_path = ""
        project.exists = status != "MISSING"
        project.available = False
        project.status = status
        project.error = error
        self._touch_catalog()

    def _read_project_runtime(self, path: str, expected_uuid: str) -> Tuple[str, Any]:
        if not Path(path).is_file():
            return "MISSING", None
        try:
            inspection = self._inspect_path(path)
        except Exception as exc:
            return "UNREADABLE", str(exc)
        if str(inspection.project_uuid) != expected_uuid:
            return (
                "IDENTITY_MISMATCH",
                "The file at this path belongs to a different project",
            )
        if not self._inspection_is_master(inspection):
            return "UNSUPPORTED", "Not a master project container"
        return "AVAILABLE", inspection

    def _apply_runtime_result(self, project: Project, kind: str, payload: Any) -> None:
        if kind == "AVAILABLE":
            project.relocation_candidate = ""
            self._apply_inspection(project, payload)
            return
        self._clear_runtime(project, kind, str(payload or ""))

    def _refresh_project(self, project: Project) -> None:
        kind, payload = self._read_project_runtime(project.path, project.project_uuid)
        self._apply_runtime_result(project, kind, payload)

    def _rebuild_path_lookup(self) -> None:
        self._project_by_path = {
            self._path_key(project.path): project_uuid
            for project_uuid, project in self._projects.items()
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
        self._ensure_default_folder()

    def _load_v3(self, data: Dict[str, Any]) -> bool:
        folders_data = data.get("folders")
        projects_data = data.get("projects")
        if not isinstance(folders_data, dict) or not isinstance(projects_data, dict):
            raise ValueError("Asset Manager schema v3 requires folders and projects objects")
        normalized = set(data) != {"schema_version", "folders", "projects"}

        self._folders = {}
        self._projects = {}
        self._project_by_path = {}
        stored_default_path = ""
        for folder_id, value in folders_data.items():
            if not isinstance(folder_id, str) or not isinstance(value, dict):
                raise ValueError("Invalid Asset Manager folder record")
            normalized = normalized or set(value) != {"path"}
            raw_path = str(value.get("path") or "").strip()
            if not raw_path:
                normalized = True
                continue
            path = _normalize_path(raw_path)
            normalized = normalized or path != raw_path
            if folder_id == DEFAULT_FOLDER_ID:
                stored_default_path = path
                continue
            if self._path_key(path) == self._path_key(self._default_folder_path):
                normalized = True
                continue
            before = len(self._folders)
            self._add_folder_record(path, preferred_id=folder_id)
            normalized = normalized or len(self._folders) == before

        self._ensure_default_folder()
        if (
            stored_default_path
            and self._path_key(stored_default_path)
            != self._path_key(self._default_folder_path)
        ):
            self._add_folder_record(stored_default_path)
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

            normalized = normalized or set(value) != {"name", "path", "folder_id"}
            normalized = normalized or path != stored_path
            stored_folder_id = str(value.get("folder_id") or DEFAULT_FOLDER_ID)
            folder_id = self._folder_id_for_path(path)
            if folder_id is None:
                folder_id = self._add_folder_record(str(Path(path).parent)).id
                normalized = True
            if folder_id != stored_folder_id:
                normalized = True
            project = Project(
                project_uuid=canonical_uuid,
                name=str(value.get("name") or self._inspection_name(path)),
                path=path,
                folder_id=folder_id,
                exists=True,
                status="UNVERIFIED",
            )
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
        backup = self._library_path.with_name(self._library_path.name + ".legacy.bak")
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

            is_current = data.get("schema_version") == SCHEMA_VERSION
            if is_current and not migrating_legacy_location:
                if self._load_v3(data) and not self.save():
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
    def add_folder(self, directory: str) -> Optional[Folder]:
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
            return existing
        previous_state = self._snapshot_state()
        folder = self._add_folder_record(str(path))
        if not self.save():
            self._restore_state(previous_state)
            return None
        return folder

    @_synchronized
    def delete_folder(self, folder_id: str) -> bool:
        folder = self._folders.get(folder_id)
        if folder is None or folder_id == DEFAULT_FOLDER_ID:
            return False
        previous_state = self._snapshot_state()
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
        previous_state = self._snapshot_state()
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
        previous_state = self._snapshot_state() if save else None
        if "folder_id" in kwargs:
            target = self._folders.get(str(kwargs["folder_id"]))
            resolved_folder_id = self._folder_id_for_path(project.path)
            if target is None or target.id != resolved_folder_id:
                return None
            project.folder_id = target.id
        if "name" in kwargs:
            project.name = str(kwargs["name"])
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
        self._touch_catalog()
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
            previous_state = self._snapshot_state() if save else None
            target_folder_id = self._folder_id_for_path(path)
            if target_folder_id is None:
                target_folder_id = self._add_folder_record(str(Path(path).parent)).id

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
                    if not Path(project.path).is_file():
                        project.relocation_candidate = path
                    self._refresh_project(project)

                if name is not None and project.name != name:
                    project.name = name
                    persisted_changed = True
                if adopt_existing and project.folder_id != target_folder_id:
                    project.folder_id = target_folder_id
                    persisted_changed = True

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
        kind, payload = self._read_project_runtime(path, expected_uuid)
        with self._lock:
            project = self._projects.get(asset_id)
            if project is None:
                return None
            if project.path != path or project.project_uuid != expected_uuid:
                return project
            self._apply_runtime_result(project, kind, payload)
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
        folder_id = self._folder_id_for_path(path)
        if folder_id is None:
            folder_id = self._add_folder_record(str(Path(path).parent)).id
        self._project_by_path.pop(self._path_key(project.path), None)
        project.path = path
        project.folder_id = folder_id
        project.relocation_candidate = ""
        self._project_by_path[path_key] = asset_id
        self._apply_inspection(project, inspection)
        if self.save():
            return True
        self._restore_state(previous_state)
        return False

    def verify_projects_batch(self, asset_ids: List[str]) -> int:
        verified = 0
        for asset_id in asset_ids:
            if self.verify_asset(asset_id) is not None:
                verified += 1
        return verified

    def verify_projects(self) -> Tuple[int, int]:
        with self._lock:
            asset_ids = list(self._projects)
        self.verify_projects_batch(asset_ids)
        with self._lock:
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
