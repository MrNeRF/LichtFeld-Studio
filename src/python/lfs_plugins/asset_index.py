# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Index module for JSON persistence of the Asset Manager catalog."""

import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

_log = logging.getLogger(__name__)
_T = TypeVar("_T")
_ASSET_INDEX_LOCK = threading.RLock()

LIBRARY_VERSION = "1.0.0"
LEGACY_STORAGE_PATH = Path.home() / ".lichtfeld" / "asset_manager"
DEFAULT_LIBRARY_PATH = LEGACY_STORAGE_PATH / "library.json"
LEGACY_LIBRARY_PATH = LEGACY_STORAGE_PATH / "library.json"
DEFAULT_FOLDER_ID = "default"
DEFAULT_FOLDER_NAME = "Default"


def _synchronized(method: Callable[..., _T]) -> Callable[..., _T]:
    """Serialize access to the in-memory catalog and backing JSON file."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _dedupe_paths(paths: List[Path]) -> List[Path]:
    seen: set[str] = set()
    result: List[Path] = []
    for path in paths:
        try:
            expanded = path.expanduser()
            key = str(expanded.resolve())
        except Exception:
            expanded = path.expanduser()
            key = str(expanded)
        if key in seen:
            continue
        seen.add(key)
        result.append(expanded)
    return result


def _storage_candidates() -> List[Path]:
    candidates: List[Path] = []

    for env_name in ("LICHTFELD_ASSET_MANAGER_DIR", "LFS_ASSET_MANAGER_DIR"):
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            candidates.append(Path(env_value))

    candidates.append(LEGACY_STORAGE_PATH)

    xdg_data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    if xdg_data_home:
        candidates.append(Path(xdg_data_home) / "LichtFeldStudio" / "asset_manager")

    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        candidates.append(Path(appdata) / "LichtFeldStudio" / "asset_manager")

    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        candidates.append(Path(local_appdata) / "LichtFeldStudio" / "asset_manager")

    candidates.append(
        Path.home() / ".local" / "share" / "LichtFeldStudio" / "asset_manager"
    )
    candidates.append(Path(tempfile.gettempdir()) / "LichtFeldStudio" / "asset_manager")
    return _dedupe_paths(candidates)


def _path_accepts_writes(path: Path) -> bool:
    probe_path: Optional[Path] = None
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".lfs-write-test-",
            dir=path,
            delete=False,
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
            except Exception:
                pass
        return False
    except Exception as exc:
        _log.debug("Asset Manager storage path probe failed: %s (%s)", path, exc)
        return False


def _copy_existing_storage(source_dir: Path, target_dir: Path) -> None:
    if source_dir == target_dir:
        return

    source_library = source_dir / "library.json"
    target_library = target_dir / "library.json"
    try:
        if source_library.exists() and not target_library.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_library, target_library)
            _log.info(
                "Copied Asset Manager catalog from %s to writable storage %s",
                source_library,
                target_library,
            )
    except Exception as exc:
        _log.warning(
            "Failed to copy Asset Manager catalog from %s to %s: %s",
            source_library,
            target_library,
            exc,
        )

    source_thumbnails = source_dir / "thumbnails"
    target_thumbnails = target_dir / "thumbnails"
    try:
        if source_thumbnails.exists() and not target_thumbnails.exists():
            shutil.copytree(source_thumbnails, target_thumbnails)
    except Exception as exc:
        _log.debug(
            "Failed to copy Asset Manager thumbnails from %s to %s: %s",
            source_thumbnails,
            target_thumbnails,
            exc,
        )


def resolve_asset_manager_storage_path() -> Path:
    for candidate in _storage_candidates():
        if _path_accepts_writes(candidate):
            if candidate != LEGACY_STORAGE_PATH:
                _copy_existing_storage(LEGACY_STORAGE_PATH, candidate)
                _log.warning(
                    "Asset Manager catalog path %s is not writable; using %s",
                    LEGACY_STORAGE_PATH,
                    candidate,
                )
            return candidate

    return LEGACY_STORAGE_PATH


def resolve_asset_manager_library_path() -> Path:
    return resolve_asset_manager_storage_path() / "library.json"


@dataclass
class Folder:
    """A folder container for scenes and assets."""

    id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    scene_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    thumbnail_asset_id: Optional[str] = None
    watch_directories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Folder":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
            scene_ids=data.get("scene_ids", []),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            thumbnail_asset_id=data.get("thumbnail_asset_id"),
            watch_directories=data.get("watch_directories", []),
        )


@dataclass
class Scene:
    """A scene within a folder."""

    id: str
    folder_id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_asset_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    thumbnail_asset_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scene":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            folder_id=data["folder_id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
            dataset_asset_id=data.get("dataset_asset_id"),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            thumbnail_asset_id=data.get("thumbnail_asset_id"),
        )


@dataclass
class Asset:
    """An asset file (dataset, checkpoint, etc.)."""

    id: str
    folder_id: Optional[str] = None
    scene_id: Optional[str] = None
    name: str = ""
    type: str = ""  # dataset, checkpoint, image, mesh, etc.
    role: str = ""  # source, output, intermediate, thumbnail, etc.
    path: str = ""  # Relative path within folder
    absolute_path: str = ""  # Absolute path on filesystem
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    file_size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    thumbnail_path: Optional[str] = None
    geometry_metadata: Dict[str, Any] = field(default_factory=dict)
    dataset_metadata: Dict[str, Any] = field(default_factory=dict)
    transform_metadata: Dict[str, Any] = field(default_factory=dict)
    exists: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Asset":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            folder_id=data.get("folder_id"),
            scene_id=data.get("scene_id"),
            name=data.get("name", ""),
            type=data.get("type", ""),
            role=data.get("role", ""),
            path=data.get("path", ""),
            absolute_path=data.get("absolute_path", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
            file_size_bytes=data.get("file_size_bytes", 0),
            tags=data.get("tags", []),
            thumbnail_path=data.get("thumbnail_path"),
            geometry_metadata=data.get("geometry_metadata", {}),
            dataset_metadata=data.get("dataset_metadata", {}),
            transform_metadata=data.get("transform_metadata", {}),
            exists=data.get("exists", True),
        )


class AssetIndex:
    """JSON persistence layer for the Asset Manager catalog."""

    def __init__(self, library_path: Optional[Path] = None):
        """Initialize with path to library.json.

        Args:
            library_path: Path to library.json. Defaults to ~/.lichtfeld/asset_manager/library.json
        """
        self._library_path = library_path or resolve_asset_manager_library_path()
        self._library_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _ASSET_INDEX_LOCK

        # In-memory catalog storage
        self._version: str = LIBRARY_VERSION
        self._created_at: str = datetime.now().isoformat()
        self._modified_at: str = datetime.now().isoformat()
        self._folders: Dict[str, Folder] = {}
        self._scenes: Dict[str, Scene] = {}
        self._assets: Dict[str, Asset] = {}
        self._collections: Dict[str, Dict[str, Any]] = {}
        self._tags: Dict[str, Dict[str, Any]] = {}

    @property
    def library_path(self) -> Path:
        """Return the backing library.json path."""
        return self._library_path

    @property
    @_synchronized
    def folders(self) -> Dict[str, Dict[str, Any]]:
        """Return folders as dictionaries for backward compatibility."""
        return {fid: f.to_dict() for fid, f in self._folders.items()}

    @property
    @_synchronized
    def scenes(self) -> Dict[str, Dict[str, Any]]:
        """Return scenes as dictionaries for backward compatibility."""
        return {sid: s.to_dict() for sid, s in self._scenes.items()}

    @property
    @_synchronized
    def assets(self) -> Dict[str, Dict[str, Any]]:
        """Return assets as dictionaries for backward compatibility."""
        return {aid: a.to_dict() for aid, a in self._assets.items()}

    @property
    @_synchronized
    def collections(self) -> Dict[str, Dict[str, Any]]:
        """Return collections."""
        return dict(self._collections)

    @property
    @_synchronized
    def tags(self) -> Dict[str, Dict[str, Any]]:
        """Return tags."""
        return dict(self._tags)

    @_synchronized
    def load(self) -> bool:
        """Load library.json, create default if missing.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self._library_path.exists():
            _log.info(
                "Library not found at %s, creating default catalog", self._library_path
            )
            self.ensure_default_catalog()
            return self.save()

        try:
            with open(self._library_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._version = data.get("version", LIBRARY_VERSION)
            self._created_at = data.get("created_at", datetime.now().isoformat())
            self._modified_at = data.get("modified_at", datetime.now().isoformat())

            # Load folders
            self._folders = {
                fid: Folder.from_dict(f) for fid, f in data.get("folders", {}).items()
            }

            # Load scenes
            self._scenes = {
                sid: Scene.from_dict(s) for sid, s in data.get("scenes", {}).items()
            }

            # Load assets
            self._assets = {
                aid: Asset.from_dict(a) for aid, a in data.get("assets", {}).items()
            }

            # Load collections and tags
            self._collections = data.get("collections", {})
            self._tags = data.get("tags", {})
            self.rebuild_tag_index(save=False)
            self._ensure_default_folder()

            _log.info(
                "Loaded library with %d folders, %d scenes, %d assets",
                len(self._folders),
                len(self._scenes),
                len(self._assets),
            )
            return True

        except json.JSONDecodeError as exc:
            _log.error("Failed to parse library.json: %s", exc)
            return False
        except Exception as exc:
            _log.error("Failed to load library: %s", exc)
            return False

    @_synchronized
    def save(self) -> bool:
        """Atomic save with backup (.json.bak).

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            self._modified_at = datetime.now().isoformat()

            data = {
                "version": self._version,
                "created_at": self._created_at,
                "modified_at": self._modified_at,
                "folders": {fid: f.to_dict() for fid, f in self._folders.items()},
                "scenes": {sid: s.to_dict() for sid, s in self._scenes.items()},
                "assets": {aid: a.to_dict() for aid, a in self._assets.items()},
                "collections": self._collections,
                "tags": self._tags,
            }

            # Ensure parent directory exists
            self._library_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to a temp file in the same directory (same filesystem guarantees
            # atomic rename).  Use tempfile so we never collide with an existing
            # file and we get a guaranteed unique name.
            import tempfile as _tf

            fd, temp_path_str = _tf.mkstemp(
                suffix=".tmp",
                prefix=self._library_path.stem + ".",
                dir=str(self._library_path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                # Clean up the temp file if writing failed
                try:
                    os.unlink(temp_path_str)
                except Exception:
                    pass
                raise

            # Rotate: move current to backup if it still exists.
            # Use a try/except to tolerate the race where the file disappears
            # between the exists() check and the move.
            backup_path = self._library_path.with_suffix(".json.bak")
            try:
                if self._library_path.exists():
                    shutil.move(str(self._library_path), str(backup_path))
            except FileNotFoundError:
                pass  # Nothing to back up — proceed with the new file

            # Atomic rename temp -> final
            os.rename(temp_path_str, str(self._library_path))

            _log.info(
                "Saved library to %s (%d folders, %d scenes, %d assets)",
                self._library_path,
                len(self._folders),
                len(self._scenes),
                len(self._assets),
            )
            return True

        except Exception as exc:
            _log.error(
                "Failed to save library to %s: %s",
                self._library_path,
                exc,
                exc_info=True,
            )
            return False

    @_synchronized
    def ensure_default_catalog(self) -> None:
        """Create empty catalog structure."""
        self._version = LIBRARY_VERSION
        self._created_at = datetime.now().isoformat()
        self._modified_at = datetime.now().isoformat()
        self._folders = {}
        self._scenes = {}
        self._assets = {}
        self._collections = {}
        self._tags = {}
        self._ensure_default_folder()
        _log.debug("Initialized default catalog")

    def _ensure_default_folder(self) -> bool:
        """Guarantee that the catalog always contains the canonical default folder."""
        for folder in self._folders.values():
            if folder.id == DEFAULT_FOLDER_ID:
                if folder.name != DEFAULT_FOLDER_NAME:
                    folder.name = DEFAULT_FOLDER_NAME
                    folder.modified_at = datetime.now().isoformat()
                return False

        for folder in self._folders.values():
            if str(folder.name).strip().lower() == DEFAULT_FOLDER_NAME.lower():
                return False

        self._folders[DEFAULT_FOLDER_ID] = Folder(
            id=DEFAULT_FOLDER_ID,
            name=DEFAULT_FOLDER_NAME,
        )
        return True

    # -------------------------------------------------------------------------
    # Folder CRUD
    # -------------------------------------------------------------------------

    @_synchronized
    def create_folder(
        self, name: str, description: str = "", tags: Optional[List[str]] = None
    ) -> Folder:
        """Create a new folder.

        Args:
            name: Folder name
            description: Folder description
            tags: Optional list of tags

        Returns:
            The created Folder instance
        """
        folder = Folder(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            tags=tags or [],
        )
        self._folders[folder.id] = folder
        self.save()
        return folder

    @_synchronized
    def update_folder(self, folder_id: str, **kwargs) -> Optional[Folder]:
        """Update a folder.

        Args:
            folder_id: Folder ID to update
            **kwargs: Fields to update

        Returns:
            Updated Folder or None if not found
        """
        if folder_id not in self._folders:
            return None

        folder = self._folders[folder_id]
        for key, value in kwargs.items():
            if hasattr(folder, key):
                setattr(folder, key, value)
        folder.modified_at = datetime.now().isoformat()
        self.save()
        return folder

    @_synchronized
    def delete_folder(self, folder_id: str) -> bool:
        """Delete a folder and all associated scenes and assets.

        Args:
            folder_id: Folder ID to delete

        Returns:
            True if deleted, False if not found
        """
        if folder_id not in self._folders:
            return False

        # Delete associated scenes
        scenes_to_delete = [
            sid for sid, s in self._scenes.items() if s.folder_id == folder_id
        ]
        for sid in scenes_to_delete:
            self.delete_scene(sid)

        # Delete associated assets (not tied to scenes)
        assets_to_delete = [
            aid
            for aid, a in self._assets.items()
            if a.folder_id == folder_id and a.scene_id is None
        ]
        for aid in assets_to_delete:
            del self._assets[aid]

        del self._folders[folder_id]
        self.save()
        return True

    @_synchronized
    def get_folder(self, folder_id: str) -> Optional[Folder]:
        """Get a folder by ID.

        Args:
            folder_id: Folder ID

        Returns:
            Folder or None if not found
        """
        return self._folders.get(folder_id)

    @_synchronized
    def get_watch_dirs(self, folder_id: str) -> List[str]:
        """Get watched directories for a folder.

        Args:
            folder_id: Folder ID

        Returns:
            List of watched directory paths
        """
        folder = self._folders.get(folder_id)
        if folder is None:
            return []
        return list(folder.watch_directories)

    @_synchronized
    def set_watch_dirs(self, folder_id: str, paths: List[str]) -> bool:
        """Set watched directories for a folder.

        Args:
            folder_id: Folder ID
            paths: List of directory paths to watch

        Returns:
            True if updated, False if folder not found
        """
        if folder_id not in self._folders:
            return False
        folder = self._folders[folder_id]
        previous_paths = list(folder.watch_directories)
        previous_modified_at = folder.modified_at
        folder.watch_directories = list(paths)
        folder.modified_at = datetime.now().isoformat()
        if not self.save():
            folder.watch_directories = previous_paths
            folder.modified_at = previous_modified_at
            return False
        return True

    @_synchronized
    def list_folders(self) -> List[Folder]:
        """List all folders.

        Returns:
            List of all folders
        """
        return list(self._folders.values())

    @_synchronized
    def find_or_create_folder(self, name: str) -> Folder:
        """Find a folder by name or create a new one.

        Args:
            name: Folder name to find or create

        Returns:
            Existing or newly created Folder instance
        """
        for folder in self._folders.values():
            if folder.name == name:
                return folder
        return self.create_folder(name=name)

    # -------------------------------------------------------------------------
    # Scene CRUD
    # -------------------------------------------------------------------------

    @_synchronized
    def create_scene(
        self,
        folder_id: str,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Optional[Scene]:
        """Create a new scene within a folder.

        Args:
            folder_id: Parent folder ID
            name: Scene name
            description: Scene description
            tags: Optional list of tags

        Returns:
            The created Scene instance or None if folder not found
        """
        if folder_id not in self._folders:
            return None

        scene = Scene(
            id=str(uuid.uuid4()),
            folder_id=folder_id,
            name=name,
            description=description,
            tags=tags or [],
        )
        self._scenes[scene.id] = scene
        self._folders[folder_id].scene_ids.append(scene.id)
        self._folders[folder_id].modified_at = datetime.now().isoformat()
        if not self.save():
            _log.error("Failed to save library during scene creation for %s", scene.id)
            # Clean up in-memory state
            del self._scenes[scene.id]
            self._folders[folder_id].scene_ids.remove(scene.id)
            return None
        return scene

    @_synchronized
    def update_scene(self, scene_id: str, **kwargs) -> Optional[Scene]:
        """Update a scene.

        Args:
            scene_id: Scene ID to update
            **kwargs: Fields to update

        Returns:
            Updated Scene or None if not found
        """
        if scene_id not in self._scenes:
            return None

        scene = self._scenes[scene_id]
        for key, value in kwargs.items():
            if hasattr(scene, key):
                setattr(scene, key, value)
        scene.modified_at = datetime.now().isoformat()
        if not self.save():
            _log.error("Failed to save library during scene update for %s", scene_id)
            return None
        return scene

    @_synchronized
    def delete_scene(self, scene_id: str) -> bool:
        """Delete a scene and all associated assets.

        Args:
            scene_id: Scene ID to delete

        Returns:
            True if deleted, False if not found
        """
        if scene_id not in self._scenes:
            return False

        scene = self._scenes[scene_id]

        # Delete associated assets
        assets_to_delete = [
            aid for aid, a in self._assets.items() if a.scene_id == scene_id
        ]
        for aid in assets_to_delete:
            del self._assets[aid]

        # Remove from folder
        if scene.folder_id in self._folders:
            folder = self._folders[scene.folder_id]
            if scene_id in folder.scene_ids:
                folder.scene_ids.remove(scene_id)
                folder.modified_at = datetime.now().isoformat()

        del self._scenes[scene_id]
        self.save()
        return True

    @_synchronized
    def get_scene(self, scene_id: str) -> Optional[Scene]:
        """Get a scene by ID.

        Args:
            scene_id: Scene ID

        Returns:
            Scene or None if not found
        """
        return self._scenes.get(scene_id)

    @_synchronized
    def list_scenes(self, folder_id: Optional[str] = None) -> List[Scene]:
        """List scenes, optionally filtered by folder.

        Args:
            folder_id: Optional folder ID to filter by

        Returns:
            List of scenes
        """
        scenes = list(self._scenes.values())
        if folder_id:
            scenes = [s for s in scenes if s.folder_id == folder_id]
        return scenes

    @_synchronized
    def find_or_create_scene(self, folder_id: str, name: str) -> Optional[Scene]:
        """Find a scene by name within a folder or create a new one.

        Args:
            folder_id: Parent folder ID
            name: Scene name to find or create

        Returns:
            Existing or newly created Scene instance, or None if folder not found
        """
        if folder_id not in self._folders:
            return None
        for scene in self._scenes.values():
            if scene.folder_id == folder_id and scene.name == name:
                return scene
        return self.create_scene(folder_id=folder_id, name=name)

    # -------------------------------------------------------------------------
    # Asset CRUD
    # -------------------------------------------------------------------------

    @_synchronized
    def create_asset(
        self,
        folder_id: Optional[str],
        name: str,
        type: str,
        path: str,
        absolute_path: str,
        scene_id: Optional[str] = None,
        role: str = "",
        tags: Optional[List[str]] = None,
        file_size_bytes: int = 0,
        thumbnail_path: Optional[str] = None,
        geometry_metadata: Optional[Dict[str, Any]] = None,
        dataset_metadata: Optional[Dict[str, Any]] = None,
        transform_metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
        modified_at: Optional[str] = None,
        exists: Optional[bool] = None,
    ) -> Optional[Asset]:
        """Create a new asset.

        Args:
            folder_id: Parent folder ID
            name: Asset name
            type: Asset type (dataset, checkpoint, etc.)
            path: Relative path within folder
            absolute_path: Absolute path on filesystem
            scene_id: Optional parent scene ID
            role: Asset role (source, output, etc.)
            tags: Optional list of tags
            file_size_bytes: File size in bytes

        Returns:
            The created Asset instance or None if folder not found
        """
        if folder_id is not None and folder_id not in self._folders:
            _log.error("Cannot create asset: folder_id %s not found", folder_id)
            return None
        if scene_id is not None and scene_id not in self._scenes:
            _log.error("Cannot create asset: scene_id %s not found", scene_id)
            return None

        normalized_abs_path = os.path.abspath(absolute_path or path)
        existing_asset = self.find_asset_by_path(
            normalized_abs_path,
            folder_id=folder_id,
        )
        if existing_asset is not None:
            merged_tags = list(
                dict.fromkeys((existing_asset.tags or []) + (tags or []))
            )
            updated = self.update_asset(
                existing_asset.id,
                folder_id=folder_id
                if folder_id is not None
                else existing_asset.folder_id,
                scene_id=scene_id if scene_id is not None else existing_asset.scene_id,
                name=name or existing_asset.name,
                type=type or existing_asset.type,
                role=role or existing_asset.role,
                path=path,
                absolute_path=normalized_abs_path,
                file_size_bytes=file_size_bytes or existing_asset.file_size_bytes,
                thumbnail_path=thumbnail_path
                if thumbnail_path is not None
                else existing_asset.thumbnail_path,
                geometry_metadata=geometry_metadata
                if geometry_metadata is not None
                else existing_asset.geometry_metadata,
                dataset_metadata=dataset_metadata
                if dataset_metadata is not None
                else existing_asset.dataset_metadata,
                tags=merged_tags,
                created_at=created_at or existing_asset.created_at,
                exists=os.path.exists(normalized_abs_path)
                if exists is None
                else exists,
            )
            return updated

        asset = Asset(
            id=str(uuid.uuid4()),
            folder_id=folder_id,
            scene_id=scene_id,
            name=name,
            type=type,
            role=role,
            path=path,
            absolute_path=normalized_abs_path,
            created_at=created_at or datetime.now().isoformat(),
            modified_at=modified_at or datetime.now().isoformat(),
            tags=tags or [],
            file_size_bytes=file_size_bytes,
            thumbnail_path=thumbnail_path,
            geometry_metadata=geometry_metadata or {},
            dataset_metadata=dataset_metadata or {},
            transform_metadata=transform_metadata or {},
            exists=os.path.exists(normalized_abs_path) if exists is None else exists,
        )
        self._assets[asset.id] = asset

        # Update parent modified times
        if scene_id and scene_id in self._scenes:
            self._scenes[scene_id].modified_at = datetime.now().isoformat()

        self.rebuild_tag_index(save=False)
        if not self.save():
            _log.error("Failed to save library during asset creation for %s", asset.id)
            # Clean up in-memory state to maintain consistency with disk
            del self._assets[asset.id]
            return None
        return asset

    @_synchronized
    def update_asset(self, asset_id: str, **kwargs) -> Optional[Asset]:
        """Update an asset.

        Args:
            asset_id: Asset ID to update
            **kwargs: Fields to update

        Returns:
            Updated Asset or None if not found
        """
        if asset_id not in self._assets:
            return None

        asset = self._assets[asset_id]
        explicit_modified_at = kwargs.pop("modified_at", None)
        for key, value in kwargs.items():
            if hasattr(asset, key):
                setattr(asset, key, value)
        asset.modified_at = explicit_modified_at or datetime.now().isoformat()
        self.rebuild_tag_index(save=False)
        if not self.save():
            _log.error("Failed to save library during asset update for %s", asset_id)
            return None
        return asset

    @_synchronized
    def delete_asset(self, asset_id: str) -> bool:
        """Delete an asset.

        Args:
            asset_id: Asset ID to delete

        Returns:
            True if deleted, False if not found
        """
        if asset_id not in self._assets:
            return False

        asset = self._assets[asset_id]
        asset_scene_id = asset.scene_id
        asset_folder_id = asset.folder_id
        is_dataset = asset.type == "dataset" or asset.role == "source_dataset"

        for scene in self._scenes.values():
            if scene.dataset_asset_id == asset_id:
                scene.dataset_asset_id = None
                scene.modified_at = datetime.now().isoformat()

        del self._assets[asset_id]

        if is_dataset and asset_scene_id in self._scenes:
            scene_has_assets = any(
                a.scene_id == asset_scene_id for a in self._assets.values()
            )
            scene = self._scenes[asset_scene_id]
            if (
                not scene_has_assets
                and scene.dataset_asset_id is None
            ):
                folder = self._folders.get(scene.folder_id)
                if folder and asset_scene_id in folder.scene_ids:
                    folder.scene_ids.remove(asset_scene_id)
                    folder.modified_at = datetime.now().isoformat()
                del self._scenes[asset_scene_id]

        if asset_folder_id in self._folders:
            folder_has_scenes = bool(self._folders[asset_folder_id].scene_ids)
            folder_has_assets = any(
                a.folder_id == asset_folder_id for a in self._assets.values()
            )
            if not folder_has_scenes and not folder_has_assets:
                del self._folders[asset_folder_id]

        self.rebuild_tag_index(save=False)
        if not self.save():
            _log.error("Failed to save library during asset deletion for %s", asset_id)
            return False
        return True

    @_synchronized
    def remove_asset(self, asset_id: str) -> bool:
        """Backward-compatible alias for delete_asset."""
        return self.delete_asset(asset_id)

    @_synchronized
    def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Get an asset by ID.

        Args:
            asset_id: Asset ID

        Returns:
            Asset or None if not found
        """
        return self._assets.get(asset_id)

    @_synchronized
    def find_asset_by_path(
        self,
        absolute_path: str,
        folder_id: Optional[str] = None,
    ) -> Optional[Asset]:
        """Find an asset by its absolute path.

        Args:
            absolute_path: Absolute file path
            folder_id: Optional folder ID to scope the lookup

        Returns:
            Asset or None if not found
        """
        normalized = os.path.abspath(absolute_path)
        for asset in self._assets.values():
            if folder_id is not None and asset.folder_id != folder_id:
                continue
            if os.path.abspath(asset.absolute_path) == normalized:
                return asset
        return None

    @_synchronized
    def rebuild_tag_index(self, save: bool = True) -> None:
        """Recompute tag counts from current catalog contents."""
        tag_counts: Dict[str, Dict[str, Any]] = {}

        def _accumulate(values: List[str]) -> None:
            for raw_tag in values or []:
                tag = str(raw_tag).strip()
                if not tag:
                    continue
                entry = tag_counts.setdefault(
                    tag,
                    {
                        "label": tag,
                        "count": 0,
                    },
                )
                entry["count"] += 1

        for folder in self._folders.values():
            _accumulate(folder.tags)
        for scene in self._scenes.values():
            _accumulate(scene.tags)
        for asset in self._assets.values():
            _accumulate(asset.tags)

        self._tags = tag_counts
        if save:
            self.save()

    @_synchronized
    def add_tag_to_asset(self, asset_id: str, tag: str) -> Optional[Asset]:
        """Add a tag to an asset if it is not already present."""
        asset = self._assets.get(asset_id)
        if asset is None:
            return None
        normalized = tag.strip()
        if not normalized:
            return asset
        if normalized not in asset.tags:
            asset.tags.append(normalized)
        asset.modified_at = datetime.now().isoformat()
        self.rebuild_tag_index(save=False)
        self.save()
        return asset

    @_synchronized
    def remove_tag_from_asset(self, asset_id: str, tag: str) -> Optional[Asset]:
        """Remove a tag from an asset."""
        asset = self._assets.get(asset_id)
        if asset is None:
            return None
        normalized = tag.strip()
        if normalized in asset.tags:
            asset.tags.remove(normalized)
            asset.modified_at = datetime.now().isoformat()
        self.rebuild_tag_index(save=False)
        if not self.save():
            _log.error("Failed to save library during tag removal for %s", asset.id)
            # Restore the tag on failure to maintain consistency
            if normalized not in asset.tags:
                asset.tags.append(normalized)
            return None
        return asset

    @_synchronized
    def list_assets(
        self,
        folder_id: Optional[str] = None,
        scene_id: Optional[str] = None,
        type: Optional[str] = None,
        role: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Asset]:
        """List assets with optional filters.

        Args:
            folder_id: Optional folder ID to filter by
            scene_id: Optional scene ID to filter by
            type: Optional asset type to filter by
            role: Optional asset role to filter by
            tags: Optional tags to filter by (all must match)

        Returns:
            List of assets
        """
        assets = list(self._assets.values())
        if folder_id:
            assets = [a for a in assets if a.folder_id == folder_id]
        if scene_id:
            assets = [a for a in assets if a.scene_id == scene_id]
        if type:
            assets = [a for a in assets if a.type == type]
        if role:
            assets = [a for a in assets if a.role == role]
        if tags:
            assets = [a for a in assets if all(t in a.tags for t in tags)]
        return assets

    @_synchronized
    def mark_missing_files(self) -> Tuple[int, int]:
        """Update exists flag for all assets based on file existence.

        Returns:
            Tuple of (missing_count, total_count)
        """
        missing_count = 0
        total_count = len(self._assets)
        changed = False

        for asset in self._assets.values():
            exists = os.path.exists(asset.absolute_path)
            if not exists:
                missing_count += 1
            if asset.exists != exists:
                asset.exists = exists
                asset.modified_at = datetime.now().isoformat()
                changed = True

        if changed:
            self.save()

        _log.info("Marked %d/%d assets as missing", missing_count, total_count)
        return missing_count, total_count

    # -------------------------------------------------------------------------
    # Search/Filter Methods
    # -------------------------------------------------------------------------

    @_synchronized
    def search_folders(self, query: str) -> List[Folder]:
        """Search folders by name, description, or tags.

        Args:
            query: Search query string

        Returns:
            List of matching folders
        """
        query_lower = query.lower()
        results = []
        for folder in self._folders.values():
            searchable = (
                f"{folder.name} {folder.description} {' '.join(folder.tags)}".lower()
            )
            if query_lower in searchable:
                results.append(folder)
        return results

    @_synchronized
    def search_scenes(
        self, query: str, folder_id: Optional[str] = None
    ) -> List[Scene]:
        """Search scenes by name, description, or tags.

        Args:
            query: Search query string
            folder_id: Optional folder ID to filter by

        Returns:
            List of matching scenes
        """
        query_lower = query.lower()
        results = []
        scenes = self.list_scenes(folder_id)
        for scene in scenes:
            searchable = (
                f"{scene.name} {scene.description} {' '.join(scene.tags)}".lower()
            )
            if query_lower in searchable:
                results.append(scene)
        return results

    @_synchronized
    def search_assets(
        self,
        query: str,
        folder_id: Optional[str] = None,
        type: Optional[str] = None,
    ) -> List[Asset]:
        """Search assets by name, path, or tags.

        Args:
            query: Search query string
            folder_id: Optional folder ID to filter by
            type: Optional asset type to filter by

        Returns:
            List of matching assets
        """
        query_lower = query.lower()
        results = []
        assets = self.list_assets(folder_id=folder_id, type=type)
        for asset in assets:
            searchable = f"{asset.name} {asset.path} {' '.join(asset.tags)}".lower()
            if query_lower in searchable:
                results.append(asset)
        return results

    @_synchronized
    def get_recent_assets(self, limit: int = 10) -> List[Asset]:
        """Get most recently modified assets.

        Args:
            limit: Maximum number of assets to return

        Returns:
            List of recently modified assets
        """
        sorted_assets = sorted(
            self._assets.values(),
            key=lambda a: a.modified_at,
            reverse=True,
        )
        return sorted_assets[:limit]

    @_synchronized
    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics.

        Returns:
            Dictionary with catalog statistics
        """
        total_size = sum(a.file_size_bytes for a in self._assets.values())
        missing_count = sum(1 for a in self._assets.values() if not a.exists)

        return {
            "version": self._version,
            "created_at": self._created_at,
            "modified_at": self._modified_at,
            "folder_count": len(self._folders),
            "scene_count": len(self._scenes),
            "asset_count": len(self._assets),
            "total_size_bytes": total_size,
            "missing_files_count": missing_count,
        }
