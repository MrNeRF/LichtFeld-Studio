# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Index module for JSON persistence of the Asset Manager catalog."""

import json
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from .environment import value as environment_value

_log = logging.getLogger(__name__)
_T = TypeVar("_T")
_ASSET_INDEX_LOCK = threading.RLock()

LIBRARY_VERSION = "1.2.0"
SUPPORTED_ASSET_EXTENSION = ".licht"
DEFAULT_FOLDER_ID = "default"
DEFAULT_FOLDER_NAME = "Default"


def _normalize_watch_directories(paths: Any) -> List[str]:
    if not isinstance(paths, (list, tuple)):
        return []
    normalized_paths: List[str] = []
    seen = set()
    for path in paths:
        text = str(path).strip()
        if not text:
            continue
        normalized = os.path.abspath(os.path.expanduser(text))
        key = os.path.normcase(normalized)
        if key in seen:
            continue
        seen.add(key)
        normalized_paths.append(normalized)
    return normalized_paths


def _synchronized(method: Callable[..., _T]) -> Callable[..., _T]:
    """Serialize access to the in-memory catalog and backing JSON file."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def resolve_asset_manager_storage_path() -> Path:
    override = environment_value("LFS_ASSET_MANAGER_DIR")
    if override:
        return Path(override).expanduser()

    # UserPaths is the single source of truth for platform, portable, and LFS_HOME
    # storage policy. Keep this import lazy so the pure-Python catalog remains easy
    # to exercise with an explicit library_path in tests and tools.
    import lichtfeld as lf

    return Path(lf.io.asset_library_dir())


def is_supported_asset_path(path: str) -> bool:
    """Return whether a path is a project format supported by Asset Manager."""
    return Path(path).suffix.lower() == SUPPORTED_ASSET_EXTENSION


def _serialize_fingerprint(value: Any) -> Dict[str, Any]:
    return {
        "kind": value.kind.name,
        "size": int(value.size),
        "mtime_unix_ns": int(value.mtime_unix_ns),
        "head_xxh3": value.head_xxh3.to_hex(),
        "tail_xxh3": value.tail_xxh3.to_hex(),
        "full_xxh3": value.full_xxh3.to_hex() if value.full_xxh3 is not None else None,
    }


def _deserialize_fingerprint(data: Dict[str, Any]) -> Any:
    import lichtfeld as lf

    value = lf.io.ReferenceFingerprint()
    value.kind = getattr(lf.io.FingerprintKind, data["kind"])
    value.size = int(data["size"])
    value.mtime_unix_ns = int(data["mtime_unix_ns"])
    value.head_xxh3 = lf.io.Hash128.from_hex(data["head_xxh3"])
    value.tail_xxh3 = lf.io.Hash128.from_hex(data["tail_xxh3"])
    full_hash = data.get("full_xxh3")
    value.full_xxh3 = lf.io.Hash128.from_hex(full_hash) if full_hash else None
    return value


def _fingerprint_content_key(data: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        data["kind"],
        int(data["size"]),
        data["head_xxh3"],
        data["tail_xxh3"],
        data.get("full_xxh3"),
    )


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
    watch_directories: List[str] = field(default_factory=list)
    notes: str = ""

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
            watch_directories=_normalize_watch_directories(
                data.get("watch_directories", [])
            ),
            notes=data.get("notes", ""),
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
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scene":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            folder_id=data.get("folder_id") or DEFAULT_FOLDER_ID,
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
            notes=data.get("notes", ""),
        )


@dataclass
class Asset:
    """A catalog record. New Asset Manager records are LichtFeld projects."""

    id: str
    fingerprint: Dict[str, Any]
    folder_id: Optional[str] = None
    scene_id: Optional[str] = None
    name: str = ""
    path: str = ""  # Relative path within folder
    absolute_path: str = ""  # Absolute path on filesystem
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    file_size_bytes: int = 0
    verification_disposition: Optional[str] = None
    exists: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize a LichtFeld project catalog row."""
        return dict(
            id=self.id,
            folder_id=self.folder_id,
            scene_id=self.scene_id,
            name=self.name,
            path=self.path,
            absolute_path=self.absolute_path,
            created_at=self.created_at,
            modified_at=self.modified_at,
            file_size_bytes=self.file_size_bytes,
            fingerprint=self.fingerprint,
            verification_disposition=self.verification_disposition,
            exists=self.exists,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Asset":
        """Create from dictionary."""
        fingerprint = data.get("fingerprint")
        if not isinstance(fingerprint, dict):
            raise ValueError("Asset Manager project records require a fingerprint")
        if fingerprint.get("kind") != "FILE":
            raise ValueError("Asset Manager project fingerprints must describe a file")
        for key in ("size", "mtime_unix_ns"):
            int(fingerprint[key])
        for key in ("head_xxh3", "tail_xxh3"):
            value = fingerprint[key]
            if not isinstance(value, str) or len(value) != 32:
                raise ValueError(f"Invalid project fingerprint field: {key}")
        full_hash = fingerprint.get("full_xxh3")
        if full_hash is not None and (
            not isinstance(full_hash, str) or len(full_hash) != 32
        ):
            raise ValueError("Invalid project fingerprint field: full_xxh3")
        return cls(
            id=data["id"],
            fingerprint=fingerprint,
            folder_id=data.get("folder_id"),
            scene_id=data.get("scene_id"),
            name=data.get("name", ""),
            path=data.get("path", ""),
            absolute_path=data.get("absolute_path", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
            file_size_bytes=data.get("file_size_bytes", 0),
            verification_disposition=data.get("verification_disposition"),
            exists=data.get("exists", True),
        )


class AssetIndex:
    """JSON persistence layer for the Asset Manager catalog."""

    def __init__(self, library_path: Optional[Path] = None):
        """Initialize with path to library.json.

        Args:
            library_path: Path to library.json. Defaults to the resolved Asset Manager data directory.
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
        self._asset_by_path: Dict[str, str] = {}
        self._asset_by_content_key: Dict[Tuple[Any, ...], str] = {}

    @property
    def library_path(self) -> Path:
        """Return the backing library.json path."""
        return self._library_path

    @property
    @_synchronized
    def folders(self) -> Dict[str, Dict[str, Any]]:
        """Return a serializable snapshot of the project folders."""
        return {fid: f.to_dict() for fid, f in self._folders.items()}

    @property
    @_synchronized
    def scenes(self) -> Dict[str, Dict[str, Any]]:
        """Return a serializable snapshot of the project scenes."""
        return {sid: s.to_dict() for sid, s in self._scenes.items()}

    @property
    @_synchronized
    def assets(self) -> Dict[str, Dict[str, Any]]:
        """Return a serializable snapshot of the .licht projects."""
        return {aid: a.to_dict() for aid, a in self._assets.items()}

    @staticmethod
    def _path_key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def _index_asset(self, asset: Asset) -> None:
        if asset.absolute_path:
            self._asset_by_path[self._path_key(asset.absolute_path)] = asset.id
        content_key = _fingerprint_content_key(asset.fingerprint)
        self._asset_by_content_key.setdefault(content_key, asset.id)

    def _deindex_asset(self, asset: Asset) -> None:
        if asset.absolute_path:
            path_key = self._path_key(asset.absolute_path)
            if self._asset_by_path.get(path_key) == asset.id:
                self._asset_by_path.pop(path_key, None)
        content_key = _fingerprint_content_key(asset.fingerprint)
        if self._asset_by_content_key.get(content_key) == asset.id:
            self._asset_by_content_key.pop(content_key, None)

    def _rebuild_asset_lookups(self) -> None:
        self._asset_by_path = {}
        self._asset_by_content_key = {}
        for asset in self._assets.values():
            self._index_asset(asset)

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

            stored_version = data.get("version", LIBRARY_VERSION)
            self._version = LIBRARY_VERSION
            catalog_changed = stored_version != LIBRARY_VERSION
            self._created_at = data.get("created_at", datetime.now().isoformat())
            self._modified_at = data.get("modified_at", datetime.now().isoformat())

            # Load folders
            self._folders = {
                fid: Folder.from_dict(f) for fid, f in data.get("folders", {}).items()
            }
            if any(
                folder.to_dict() != data["folders"][folder_id]
                for folder_id, folder in self._folders.items()
            ):
                catalog_changed = True
            catalog_changed = self._ensure_default_folder() or catalog_changed

            # Load scenes
            self._scenes = {
                sid: Scene.from_dict(s) for sid, s in data.get("scenes", {}).items()
            }
            if any(
                scene.to_dict() != data["scenes"][scene_id]
                for scene_id, scene in self._scenes.items()
            ):
                catalog_changed = True

            # Load assets
            self._assets = {}
            for asset_id, asset_data in data.get("assets", {}).items():
                asset_path = asset_data.get("absolute_path") or asset_data.get("path") or ""
                if not is_supported_asset_path(asset_path):
                    catalog_changed = True
                    continue
                try:
                    asset = Asset.from_dict(asset_data)
                except (KeyError, TypeError, ValueError):
                    catalog_changed = True
                    continue
                if asset.id != asset_id:
                    catalog_changed = True
                    continue
                if asset.folder_id not in self._folders:
                    asset.folder_id = DEFAULT_FOLDER_ID
                    catalog_changed = True
                if asset.scene_id not in self._scenes:
                    asset.scene_id = None
                    catalog_changed = True
                if asset.to_dict() != asset_data:
                    catalog_changed = True
                self._assets[asset_id] = asset
            self._rebuild_asset_lookups()

            _log.info(
                "Loaded library with %d folders, %d scenes, %d assets",
                len(self._folders),
                len(self._scenes),
                len(self._assets),
            )
            if catalog_changed:
                _log.info("Normalized Asset Manager catalog to .licht projects only")
                if not self.save():
                    _log.error(
                        "Failed to persist normalized Asset Manager catalog at %s",
                        self._library_path,
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
        temp_path_str: Optional[str] = None
        try:
            self._modified_at = datetime.now().isoformat()

            data = {
                "version": self._version,
                "created_at": self._created_at,
                "modified_at": self._modified_at,
                "folders": {fid: f.to_dict() for fid, f in self._folders.items()},
                "scenes": {sid: s.to_dict() for sid, s in self._scenes.items()},
                "assets": {aid: a.to_dict() for aid, a in self._assets.items()},
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

            # Refresh the backup without moving the live catalog away. The
            # final os.replace therefore always replaces a valid destination
            # atomically, including on Windows.
            backup_path = self._library_path.with_suffix(".json.bak")
            backup_temp: Optional[Path] = None
            try:
                if self._library_path.exists():
                    backup_temp = backup_path.with_suffix(backup_path.suffix + ".tmp")
                    shutil.copy2(self._library_path, backup_temp)
                    os.replace(backup_temp, backup_path)
            except FileNotFoundError:
                pass  # Nothing to back up — proceed with the new file
            finally:
                if backup_temp is not None:
                    try:
                        backup_temp.unlink(missing_ok=True)
                    except OSError:
                        pass

            # Atomic replacement preserves the previous destination until the
            # new catalog has been fully flushed.
            os.replace(temp_path_str, self._library_path)

            _log.info(
                "Saved library to %s (%d folders, %d scenes, %d assets)",
                self._library_path,
                len(self._folders),
                len(self._scenes),
                len(self._assets),
            )
            return True

        except Exception as exc:
            if temp_path_str is not None:
                try:
                    Path(temp_path_str).unlink(missing_ok=True)
                except OSError:
                    pass
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
        self._rebuild_asset_lookups()
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
    def create_folder(self, name: str, description: str = "") -> Folder:
        """Create a new folder.

        Args:
            name: Folder name
            description: Folder description

        Returns:
            The created Folder instance
        """
        folder = Folder(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
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

        now = datetime.now().isoformat()
        scenes_to_delete = {
            sid for sid, s in self._scenes.items() if s.folder_id == folder_id
        }
        assets_to_delete = {
            aid
            for aid, a in self._assets.items()
            if a.folder_id == folder_id or a.scene_id in scenes_to_delete
        }

        for aid in assets_to_delete:
            self._deindex_asset(self._assets[aid])
            del self._assets[aid]

        for sid in scenes_to_delete:
            del self._scenes[sid]

        if folder_id == DEFAULT_FOLDER_ID:
            folder = self._folders[folder_id]
            folder.scene_ids = []
            folder.watch_directories = []
            folder.modified_at = now
        else:
            del self._folders[folder_id]
        return self.save()

    @_synchronized
    def get_watch_dirs(self, folder_id: str) -> List[str]:
        """Return the directories scanned for .licht projects in a folder."""
        folder = self._folders.get(folder_id)
        return list(folder.watch_directories) if folder is not None else []

    @_synchronized
    def set_watch_dirs(self, folder_id: str, paths: List[str]) -> bool:
        """Persist normalized, unique .licht discovery roots for a folder."""
        folder = self._folders.get(folder_id)
        if folder is None:
            return False

        normalized_paths = _normalize_watch_directories(paths)

        previous_paths = folder.watch_directories
        previous_modified_at = folder.modified_at
        folder.watch_directories = normalized_paths
        folder.modified_at = datetime.now().isoformat()
        if self.save():
            return True

        folder.watch_directories = previous_paths
        folder.modified_at = previous_modified_at
        return False


    # -------------------------------------------------------------------------
    # Asset CRUD
    # -------------------------------------------------------------------------

    @_synchronized
    def update_asset(
        self,
        asset_id: str,
        *,
        save: bool = True,
        **kwargs,
    ) -> Optional[Asset]:
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
        self._deindex_asset(asset)
        explicit_modified_at = kwargs.pop("modified_at", None)
        for key, value in kwargs.items():
            if hasattr(asset, key):
                setattr(asset, key, value)
        asset.modified_at = explicit_modified_at or datetime.now().isoformat()
        self._index_asset(asset)
        if save:
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
        self._deindex_asset(asset)
        asset_folder_id = asset.folder_id

        del self._assets[asset_id]

        if asset_folder_id in self._folders:
            folder_has_scenes = bool(self._folders[asset_folder_id].scene_ids)
            folder_has_assets = any(
                a.folder_id == asset_folder_id for a in self._assets.values()
            )
            if (
                asset_folder_id != DEFAULT_FOLDER_ID
                and not folder_has_scenes
                and not folder_has_assets
            ):
                del self._folders[asset_folder_id]

        if not self.save():
            _log.error("Failed to save library during asset deletion for %s", asset_id)
            return False
        return True

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
    def register_licht_asset(
        self,
        absolute_path: str,
        *,
        folder_id: Optional[str] = None,
        scene_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Tuple[Optional[Asset], bool]:
        """Register a .licht project by content identity.

        Returns ``(asset, created)``. Importing a copy of known content returns
        the original catalog record instead of creating a path-based duplicate.
        """
        normalized_path = os.path.abspath(absolute_path)
        if not is_supported_asset_path(normalized_path):
            _log.warning("Asset Manager only supports .licht projects: %s", normalized_path)
            return None, False

        import lichtfeld as lf

        observed = lf.io.fingerprint_path(normalized_path)
        if observed.kind != lf.io.FingerprintKind.FILE:
            raise ValueError("Asset Manager .licht entries must be project files")
        fingerprint = _serialize_fingerprint(observed)
        content_key = _fingerprint_content_key(fingerprint)
        existing_id = self._asset_by_content_key.get(content_key)
        existing = self._assets.get(existing_id) if existing_id else None
        if existing is not None:
            if not os.path.exists(existing.absolute_path):
                self._deindex_asset(existing)
                existing.path = normalized_path
                existing.absolute_path = normalized_path
                existing.fingerprint = fingerprint
                existing.verification_disposition = "MATCH_FAST_PATH"
                existing.file_size_bytes = int(fingerprint["size"])
                existing.exists = True
                existing.modified_at = datetime.now().isoformat()
                self._index_asset(existing)
                self.save()
            return existing, False

        # Re-importing the same catalog path adopts its current content identity
        # without creating two rows for one filesystem location.
        existing = self.find_asset_by_path(normalized_path)
        if existing is not None:
            updated = self.update_asset(
                existing.id,
                name=name or existing.name or Path(normalized_path).stem,
                fingerprint=fingerprint,
                verification_disposition="MATCH_FAST_PATH",
                file_size_bytes=int(fingerprint["size"]),
                exists=True,
            )
            return updated, False

        target_folder_id = folder_id or DEFAULT_FOLDER_ID
        if target_folder_id not in self._folders:
            _log.error("Cannot register project: folder_id %s not found", target_folder_id)
            return None, False
        if scene_id is not None and scene_id not in self._scenes:
            _log.error("Cannot register project: scene_id %s not found", scene_id)
            return None, False

        asset = Asset(
            id=str(uuid.uuid4()),
            fingerprint=fingerprint,
            folder_id=target_folder_id,
            scene_id=scene_id,
            name=name or Path(normalized_path).stem,
            path=normalized_path,
            absolute_path=normalized_path,
            file_size_bytes=int(fingerprint["size"]),
            verification_disposition="MATCH_FAST_PATH",
            exists=True,
        )
        self._assets[asset.id] = asset
        self._index_asset(asset)
        if scene_id:
            self._scenes[scene_id].modified_at = datetime.now().isoformat()
        if not self.save():
            self._deindex_asset(asset)
            del self._assets[asset.id]
            return None, False
        return asset, True

    def _verify_asset_no_save(self, asset: Asset) -> bool:
        import lichtfeld as lf

        before = (
            asset.fingerprint,
            asset.verification_disposition,
            asset.exists,
            asset.file_size_bytes,
        )
        try:
            check = lf.io.check_fingerprint(
                asset.absolute_path,
                _deserialize_fingerprint(asset.fingerprint),
            )
            disposition = check.disposition.name
            asset.verification_disposition = disposition
            asset.exists = bool(check.matches)
            if disposition == "MATCH_MTIME_REFRESHED" and check.observed is not None:
                self._deindex_asset(asset)
                asset.fingerprint = _serialize_fingerprint(check.observed)
                asset.file_size_bytes = int(check.observed.size)
                self._index_asset(asset)
        except Exception as exc:
            if not os.path.exists(asset.absolute_path):
                asset.verification_disposition = "MISSING"
                asset.exists = False
            else:
                _log.warning("Failed to verify Asset Manager project %s: %s", asset.absolute_path, exc)
                asset.verification_disposition = "UNVERIFIED"
                asset.exists = False

        after = (
            asset.fingerprint,
            asset.verification_disposition,
            asset.exists,
            asset.file_size_bytes,
        )
        return before != after

    @_synchronized
    def verify_asset(self, asset_id: str) -> Optional[Asset]:
        """Verify one project and persist its typed disposition."""
        asset = self._assets.get(asset_id)
        if asset is None:
            return None
        if self._verify_asset_no_save(asset):
            self.save()
        return asset

    @_synchronized
    def relink_asset(self, asset_id: str, new_path: str) -> bool:
        """Relink a missing project only when the selected content still matches."""
        asset = self._assets.get(asset_id)
        normalized_path = os.path.abspath(new_path)
        if asset is None or not is_supported_asset_path(normalized_path):
            return False

        import lichtfeld as lf

        check = lf.io.check_fingerprint(
            normalized_path,
            _deserialize_fingerprint(asset.fingerprint),
        )
        if not check.matches:
            return False

        self._deindex_asset(asset)
        asset.path = normalized_path
        asset.absolute_path = normalized_path
        asset.verification_disposition = check.disposition.name
        asset.exists = True
        if check.observed is not None:
            asset.fingerprint = _serialize_fingerprint(check.observed)
            asset.file_size_bytes = int(check.observed.size)
        asset.modified_at = datetime.now().isoformat()
        self._index_asset(asset)
        return self.save()

    @_synchronized
    def verify_projects(self) -> Tuple[int, int]:
        """Verify every cataloged .licht project."""
        projects = list(self._assets.values())
        changed = False
        unavailable = 0
        for asset in projects:
            changed = self._verify_asset_no_save(asset) or changed
            if not asset.exists:
                unavailable += 1
        if changed:
            self.save()
        return unavailable, len(projects)

    @_synchronized
    def list_projects(
        self,
        folder_id: Optional[str] = None,
    ) -> List[Asset]:
        """List the .licht projects exposed by Asset Manager."""
        assets = list(self._assets.values())
        if folder_id:
            assets = [asset for asset in assets if asset.folder_id == folder_id]
        return assets

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
        asset_id = self._asset_by_path.get(self._path_key(absolute_path))
        asset = self._assets.get(asset_id) if asset_id else None
        if asset is not None and (folder_id is None or asset.folder_id == folder_id):
            return asset
        return None
