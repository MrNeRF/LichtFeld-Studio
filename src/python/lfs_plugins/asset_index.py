# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Index module for JSON persistence of the Asset Manager catalog."""

import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

LIBRARY_VERSION = "1.0.0"
DEFAULT_LIBRARY_PATH = Path.home() / ".lichtfeld" / "asset_manager" / "library.json"


@dataclass
class Project:
    """A project container for scenes and assets."""

    id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    scene_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    thumbnail_asset_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
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
        )


@dataclass
class Scene:
    """A scene within a project."""

    id: str
    project_id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_asset_id: Optional[str] = None
    run_ids: List[str] = field(default_factory=list)
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
            project_id=data["project_id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
            dataset_asset_id=data.get("dataset_asset_id"),
            run_ids=data.get("run_ids", []),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            thumbnail_asset_id=data.get("thumbnail_asset_id"),
        )


@dataclass
class TrainingRun:
    """A training run associated with a scene."""

    id: str
    project_id: str
    scene_id: str
    name: str
    status: str = "pending"  # pending, running, completed, failed, cancelled
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    source_dataset_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    parent_checkpoint_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_asset_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    is_favorite: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingRun":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            project_id=data["project_id"],
            scene_id=data["scene_id"],
            name=data["name"],
            status=data.get("status", "pending"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
            completed_at=data.get("completed_at"),
            source_dataset_id=data.get("source_dataset_id"),
            parent_run_id=data.get("parent_run_id"),
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            parameters=data.get("parameters", {}),
            metrics=data.get("metrics", {}),
            artifact_asset_ids=data.get("artifact_asset_ids", []),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            is_favorite=data.get("is_favorite", False),
        )


@dataclass
class Asset:
    """An asset file (dataset, checkpoint, video, etc.)."""

    id: str
    project_id: Optional[str] = None
    scene_id: Optional[str] = None
    run_id: Optional[str] = None
    name: str = ""
    type: str = ""  # dataset, checkpoint, video, image, mesh, etc.
    role: str = ""  # source, output, intermediate, thumbnail, etc.
    path: str = ""  # Relative path within project
    absolute_path: str = ""  # Absolute path on filesystem
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    file_size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    collection_ids: List[str] = field(default_factory=list)
    notes: str = ""
    is_favorite: bool = False
    thumbnail_path: Optional[str] = None
    preview_path: Optional[str] = None
    geometry_metadata: Dict[str, Any] = field(default_factory=dict)
    training_metadata: Dict[str, Any] = field(default_factory=dict)
    dataset_metadata: Dict[str, Any] = field(default_factory=dict)
    video_metadata: Dict[str, Any] = field(default_factory=dict)
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
            project_id=data.get("project_id"),
            scene_id=data.get("scene_id"),
            run_id=data.get("run_id"),
            name=data.get("name", ""),
            type=data.get("type", ""),
            role=data.get("role", ""),
            path=data.get("path", ""),
            absolute_path=data.get("absolute_path", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
            file_size_bytes=data.get("file_size_bytes", 0),
            tags=data.get("tags", []),
            collection_ids=data.get("collection_ids", []),
            notes=data.get("notes", ""),
            is_favorite=data.get("is_favorite", False),
            thumbnail_path=data.get("thumbnail_path"),
            preview_path=data.get("preview_path"),
            geometry_metadata=data.get("geometry_metadata", {}),
            training_metadata=data.get("training_metadata", {}),
            dataset_metadata=data.get("dataset_metadata", {}),
            video_metadata=data.get("video_metadata", {}),
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
        self._library_path = library_path or DEFAULT_LIBRARY_PATH
        self._library_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory catalog storage
        self._version: str = LIBRARY_VERSION
        self._created_at: str = datetime.now().isoformat()
        self._modified_at: str = datetime.now().isoformat()
        self._projects: Dict[str, Project] = {}
        self._scenes: Dict[str, Scene] = {}
        self._runs: Dict[str, TrainingRun] = {}
        self._assets: Dict[str, Asset] = {}
        self._collections: Dict[str, Dict[str, Any]] = {}
        self._tags: Dict[str, Dict[str, Any]] = {}

    @property
    def library_path(self) -> Path:
        """Return the backing library.json path."""
        return self._library_path

    @property
    def projects(self) -> Dict[str, Dict[str, Any]]:
        """Return projects as dictionaries for backward compatibility."""
        return {pid: p.to_dict() for pid, p in self._projects.items()}

    @property
    def scenes(self) -> Dict[str, Dict[str, Any]]:
        """Return scenes as dictionaries for backward compatibility."""
        return {sid: s.to_dict() for sid, s in self._scenes.items()}

    @property
    def runs(self) -> Dict[str, Dict[str, Any]]:
        """Return training runs as dictionaries for backward compatibility."""
        return {rid: r.to_dict() for rid, r in self._runs.items()}

    @property
    def assets(self) -> Dict[str, Dict[str, Any]]:
        """Return assets as dictionaries for backward compatibility."""
        return {aid: a.to_dict() for aid, a in self._assets.items()}

    @property
    def collections(self) -> Dict[str, Dict[str, Any]]:
        """Return collections."""
        return self._collections

    @property
    def tags(self) -> Dict[str, Dict[str, Any]]:
        """Return tags."""
        return self._tags

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

            # Load projects
            self._projects = {
                pid: Project.from_dict(p) for pid, p in data.get("projects", {}).items()
            }

            # Load scenes
            self._scenes = {
                sid: Scene.from_dict(s) for sid, s in data.get("scenes", {}).items()
            }

            # Load runs
            self._runs = {
                rid: TrainingRun.from_dict(r) for rid, r in data.get("runs", {}).items()
            }

            # Load assets
            self._assets = {
                aid: Asset.from_dict(a) for aid, a in data.get("assets", {}).items()
            }

            # Load collections and tags
            self._collections = data.get("collections", {})
            self._tags = data.get("tags", {})
            self.rebuild_tag_index(save=False)

            _log.info(
                "Loaded library with %d projects, %d scenes, %d runs, %d assets",
                len(self._projects),
                len(self._scenes),
                len(self._runs),
                len(self._assets),
            )
            return True

        except json.JSONDecodeError as exc:
            _log.error("Failed to parse library.json: %s", exc)
            return False
        except Exception as exc:
            _log.error("Failed to load library: %s", exc)
            return False

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
                "projects": {pid: p.to_dict() for pid, p in self._projects.items()},
                "scenes": {sid: s.to_dict() for sid, s in self._scenes.items()},
                "runs": {rid: r.to_dict() for rid, r in self._runs.items()},
                "assets": {aid: a.to_dict() for aid, a in self._assets.items()},
                "collections": self._collections,
                "tags": self._tags,
            }

            # Write to temp file first
            temp_path = self._library_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Rotate: move current to backup if exists
            backup_path = self._library_path.with_suffix(".json.bak")
            if self._library_path.exists():
                shutil.move(str(self._library_path), str(backup_path))

            # Move temp to final
            shutil.move(str(temp_path), str(self._library_path))

            _log.debug("Saved library to %s", self._library_path)
            return True

        except Exception as exc:
            _log.error("Failed to save library: %s", exc)
            return False

    def ensure_default_catalog(self) -> None:
        """Create empty catalog structure."""
        self._version = LIBRARY_VERSION
        self._created_at = datetime.now().isoformat()
        self._modified_at = datetime.now().isoformat()
        self._projects = {}
        self._scenes = {}
        self._runs = {}
        self._assets = {}
        self._collections = {}
        self._tags = {}
        _log.debug("Initialized default catalog")

    # -------------------------------------------------------------------------
    # Project CRUD
    # -------------------------------------------------------------------------

    def create_project(
        self, name: str, description: str = "", tags: Optional[List[str]] = None
    ) -> Project:
        """Create a new project.

        Args:
            name: Project name
            description: Project description
            tags: Optional list of tags

        Returns:
            The created Project instance
        """
        project = Project(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            tags=tags or [],
        )
        self._projects[project.id] = project
        self.save()
        return project

    def update_project(self, project_id: str, **kwargs) -> Optional[Project]:
        """Update a project.

        Args:
            project_id: Project ID to update
            **kwargs: Fields to update

        Returns:
            Updated Project or None if not found
        """
        if project_id not in self._projects:
            return None

        project = self._projects[project_id]
        for key, value in kwargs.items():
            if hasattr(project, key):
                setattr(project, key, value)
        project.modified_at = datetime.now().isoformat()
        self.save()
        return project

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all associated scenes, runs, and assets.

        Args:
            project_id: Project ID to delete

        Returns:
            True if deleted, False if not found
        """
        if project_id not in self._projects:
            return False

        # Delete associated scenes
        scenes_to_delete = [
            sid for sid, s in self._scenes.items() if s.project_id == project_id
        ]
        for sid in scenes_to_delete:
            self.delete_scene(sid)

        # Delete associated assets (not tied to scenes)
        assets_to_delete = [
            aid
            for aid, a in self._assets.items()
            if a.project_id == project_id and a.scene_id is None
        ]
        for aid in assets_to_delete:
            del self._assets[aid]

        del self._projects[project_id]
        self.save()
        return True

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID.

        Args:
            project_id: Project ID

        Returns:
            Project or None if not found
        """
        return self._projects.get(project_id)

    def list_projects(self) -> List[Project]:
        """List all projects.

        Returns:
            List of all projects
        """
        return list(self._projects.values())

    def find_or_create_project(self, name: str) -> Project:
        """Find a project by name or create a new one.

        Args:
            name: Project name to find or create

        Returns:
            Existing or newly created Project instance
        """
        for project in self._projects.values():
            if project.name == name:
                return project
        return self.create_project(name=name)

    # -------------------------------------------------------------------------
    # Scene CRUD
    # -------------------------------------------------------------------------

    def create_scene(
        self,
        project_id: str,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Optional[Scene]:
        """Create a new scene within a project.

        Args:
            project_id: Parent project ID
            name: Scene name
            description: Scene description
            tags: Optional list of tags

        Returns:
            The created Scene instance or None if project not found
        """
        if project_id not in self._projects:
            return None

        scene = Scene(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name=name,
            description=description,
            tags=tags or [],
        )
        self._scenes[scene.id] = scene
        self._projects[project_id].scene_ids.append(scene.id)
        self._projects[project_id].modified_at = datetime.now().isoformat()
        if not self.save():
            _log.error("Failed to save library during scene creation for %s", scene.id)
            # Clean up in-memory state
            del self._scenes[scene.id]
            self._projects[project_id].scene_ids.remove(scene.id)
            return None
        return scene

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

    def delete_scene(self, scene_id: str) -> bool:
        """Delete a scene and all associated runs and assets.

        Args:
            scene_id: Scene ID to delete

        Returns:
            True if deleted, False if not found
        """
        if scene_id not in self._scenes:
            return False

        scene = self._scenes[scene_id]

        # Delete associated runs
        runs_to_delete = [
            rid for rid, r in self._runs.items() if r.scene_id == scene_id
        ]
        for rid in runs_to_delete:
            self.delete_run(rid)

        # Delete associated assets
        assets_to_delete = [
            aid for aid, a in self._assets.items() if a.scene_id == scene_id
        ]
        for aid in assets_to_delete:
            del self._assets[aid]

        # Remove from project
        if scene.project_id in self._projects:
            project = self._projects[scene.project_id]
            if scene_id in project.scene_ids:
                project.scene_ids.remove(scene_id)
                project.modified_at = datetime.now().isoformat()

        del self._scenes[scene_id]
        self.save()
        return True

    def get_scene(self, scene_id: str) -> Optional[Scene]:
        """Get a scene by ID.

        Args:
            scene_id: Scene ID

        Returns:
            Scene or None if not found
        """
        return self._scenes.get(scene_id)

    def list_scenes(self, project_id: Optional[str] = None) -> List[Scene]:
        """List scenes, optionally filtered by project.

        Args:
            project_id: Optional project ID to filter by

        Returns:
            List of scenes
        """
        scenes = list(self._scenes.values())
        if project_id:
            scenes = [s for s in scenes if s.project_id == project_id]
        return scenes

    def find_or_create_scene(self, project_id: str, name: str) -> Optional[Scene]:
        """Find a scene by name within a project or create a new one.

        Args:
            project_id: Parent project ID
            name: Scene name to find or create

        Returns:
            Existing or newly created Scene instance, or None if project not found
        """
        if project_id not in self._projects:
            return None
        for scene in self._scenes.values():
            if scene.project_id == project_id and scene.name == name:
                return scene
        return self.create_scene(project_id=project_id, name=name)

    # -------------------------------------------------------------------------
    # Run CRUD
    # -------------------------------------------------------------------------

    def create_run(
        self,
        project_id: str,
        scene_id: str,
        name: str,
        source_dataset_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        parent_checkpoint_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        status: str = "pending",
    ) -> Optional[TrainingRun]:
        """Create a new training run.

        Args:
            project_id: Parent project ID
            scene_id: Parent scene ID
            name: Run name
            source_dataset_id: Optional source dataset asset ID
            parent_run_id: Optional parent run ID for branching
            parent_checkpoint_id: Optional parent checkpoint asset ID
            parameters: Optional training parameters
            tags: Optional list of tags
            status: Initial run status (default: "pending")

        Returns:
            The created TrainingRun instance or None if project/scene not found
        """
        if project_id not in self._projects or scene_id not in self._scenes:
            return None

        run = TrainingRun(
            id=str(uuid.uuid4()),
            project_id=project_id,
            scene_id=scene_id,
            name=name,
            status=status,
            source_dataset_id=source_dataset_id,
            parent_run_id=parent_run_id,
            parent_checkpoint_id=parent_checkpoint_id,
            parameters=parameters or {},
            tags=tags or [],
        )
        self._runs[run.id] = run
        self._scenes[scene_id].run_ids.append(run.id)
        self._scenes[scene_id].modified_at = datetime.now().isoformat()
        self.save()
        return run

    def update_run(self, run_id: str, **kwargs) -> Optional[TrainingRun]:
        """Update a training run.

        Args:
            run_id: Run ID to update
            **kwargs: Fields to update

        Returns:
            Updated TrainingRun or None if not found
        """
        if run_id not in self._runs:
            return None

        run = self._runs[run_id]
        for key, value in kwargs.items():
            if hasattr(run, key):
                setattr(run, key, value)
        run.modified_at = datetime.now().isoformat()
        self.save()
        return run

    def set_run_status(self, run_id: str, status: str) -> Optional[TrainingRun]:
        """Set the status of a training run.

        Args:
            run_id: Run ID
            status: New status (pending, running, completed, failed, cancelled)

        Returns:
            Updated TrainingRun or None if not found
        """
        return self.update_run(run_id, status=status)

    def delete_run(self, run_id: str) -> bool:
        """Delete a training run and its associated assets.

        Args:
            run_id: Run ID to delete

        Returns:
            True if deleted, False if not found
        """
        if run_id not in self._runs:
            return False

        run = self._runs[run_id]

        # Delete associated assets
        assets_to_delete = [
            aid for aid, a in self._assets.items() if a.run_id == run_id
        ]
        for aid in assets_to_delete:
            del self._assets[aid]

        # Remove from scene
        if run.scene_id in self._scenes:
            scene = self._scenes[run.scene_id]
            if run_id in scene.run_ids:
                scene.run_ids.remove(run_id)
                scene.modified_at = datetime.now().isoformat()

        del self._runs[run_id]
        self.save()
        return True

    def get_run(self, run_id: str) -> Optional[TrainingRun]:
        """Get a training run by ID.

        Args:
            run_id: Run ID

        Returns:
            TrainingRun or None if not found
        """
        return self._runs.get(run_id)

    def list_runs(
        self,
        project_id: Optional[str] = None,
        scene_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[TrainingRun]:
        """List training runs with optional filters.

        Args:
            project_id: Optional project ID to filter by
            scene_id: Optional scene ID to filter by
            status: Optional status to filter by

        Returns:
            List of training runs
        """
        runs = list(self._runs.values())
        if project_id:
            runs = [r for r in runs if r.project_id == project_id]
        if scene_id:
            runs = [r for r in runs if r.scene_id == scene_id]
        if status:
            runs = [r for r in runs if r.status == status]
        return runs

    # -------------------------------------------------------------------------
    # Asset CRUD
    # -------------------------------------------------------------------------

    def create_asset(
        self,
        project_id: Optional[str],
        name: str,
        type: str,
        path: str,
        absolute_path: str,
        scene_id: Optional[str] = None,
        run_id: Optional[str] = None,
        role: str = "",
        tags: Optional[List[str]] = None,
        file_size_bytes: int = 0,
        thumbnail_path: Optional[str] = None,
        preview_path: Optional[str] = None,
        geometry_metadata: Optional[Dict[str, Any]] = None,
        training_metadata: Optional[Dict[str, Any]] = None,
        dataset_metadata: Optional[Dict[str, Any]] = None,
        video_metadata: Optional[Dict[str, Any]] = None,
        transform_metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
        modified_at: Optional[str] = None,
        exists: Optional[bool] = None,
    ) -> Optional[Asset]:
        """Create a new asset.

        Args:
            project_id: Parent project ID
            name: Asset name
            type: Asset type (dataset, checkpoint, video, etc.)
            path: Relative path within project
            absolute_path: Absolute path on filesystem
            scene_id: Optional parent scene ID
            run_id: Optional parent run ID
            role: Asset role (source, output, etc.)
            tags: Optional list of tags
            file_size_bytes: File size in bytes

        Returns:
            The created Asset instance or None if project not found
        """
        if project_id is not None and project_id not in self._projects:
            _log.error("Cannot create asset: project_id %s not found", project_id)
            return None
        if scene_id is not None and scene_id not in self._scenes:
            _log.error("Cannot create asset: scene_id %s not found", scene_id)
            return None
        if run_id is not None and run_id not in self._runs:
            _log.error("Cannot create asset: run_id %s not found", run_id)
            return None

        normalized_abs_path = os.path.abspath(absolute_path or path)
        existing_asset = self.find_asset_by_path(normalized_abs_path)
        if existing_asset is not None:
            merged_tags = list(
                dict.fromkeys((existing_asset.tags or []) + (tags or []))
            )
            updated = self.update_asset(
                existing_asset.id,
                project_id=project_id
                if project_id is not None
                else existing_asset.project_id,
                scene_id=scene_id if scene_id is not None else existing_asset.scene_id,
                run_id=run_id if run_id is not None else existing_asset.run_id,
                name=name or existing_asset.name,
                type=type or existing_asset.type,
                role=role or existing_asset.role,
                path=path,
                absolute_path=normalized_abs_path,
                file_size_bytes=file_size_bytes or existing_asset.file_size_bytes,
                thumbnail_path=thumbnail_path
                if thumbnail_path is not None
                else existing_asset.thumbnail_path,
                preview_path=preview_path
                if preview_path is not None
                else existing_asset.preview_path,
                geometry_metadata=geometry_metadata
                if geometry_metadata is not None
                else existing_asset.geometry_metadata,
                training_metadata=training_metadata
                if training_metadata is not None
                else existing_asset.training_metadata,
                dataset_metadata=dataset_metadata
                if dataset_metadata is not None
                else existing_asset.dataset_metadata,
                video_metadata=video_metadata
                if video_metadata is not None
                else existing_asset.video_metadata,
                tags=merged_tags,
                created_at=created_at or existing_asset.created_at,
                exists=os.path.exists(normalized_abs_path)
                if exists is None
                else exists,
            )
            if updated and run_id and run_id in self._runs:
                run = self._runs[run_id]
                if updated.id not in run.artifact_asset_ids:
                    run.artifact_asset_ids.append(updated.id)
            return updated

        asset = Asset(
            id=str(uuid.uuid4()),
            project_id=project_id,
            scene_id=scene_id,
            run_id=run_id,
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
            preview_path=preview_path,
            geometry_metadata=geometry_metadata or {},
            training_metadata=training_metadata or {},
            dataset_metadata=dataset_metadata or {},
            video_metadata=video_metadata or {},
            transform_metadata=transform_metadata or {},
            exists=os.path.exists(normalized_abs_path) if exists is None else exists,
        )
        self._assets[asset.id] = asset

        # Update parent modified times
        if scene_id and scene_id in self._scenes:
            self._scenes[scene_id].modified_at = datetime.now().isoformat()
        if run_id and run_id in self._runs:
            self._runs[run_id].modified_at = datetime.now().isoformat()
            if asset.id not in self._runs[run_id].artifact_asset_ids:
                self._runs[run_id].artifact_asset_ids.append(asset.id)

        self.rebuild_tag_index(save=False)
        if not self.save():
            _log.error("Failed to save library during asset creation for %s", asset.id)
            # Clean up in-memory state to maintain consistency with disk
            del self._assets[asset.id]
            if asset.run_id and asset.run_id in self._runs:
                run = self._runs[asset.run_id]
                if asset.id in run.artifact_asset_ids:
                    run.artifact_asset_ids.remove(asset.id)
            return None
        return asset

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
        asset_project_id = asset.project_id
        is_dataset = asset.type == "dataset" or asset.role == "source_dataset"

        # Remove from run's artifact list
        if asset.run_id and asset.run_id in self._runs:
            run = self._runs[asset.run_id]
            if asset_id in run.artifact_asset_ids:
                run.artifact_asset_ids.remove(asset_id)
                run.modified_at = datetime.now().isoformat()

        for run in self._runs.values():
            touched = False
            if run.source_dataset_id == asset_id:
                run.source_dataset_id = None
                touched = True
            if run.parent_checkpoint_id == asset_id:
                run.parent_checkpoint_id = None
                touched = True
            if touched:
                run.modified_at = datetime.now().isoformat()

        for scene in self._scenes.values():
            if scene.dataset_asset_id == asset_id:
                scene.dataset_asset_id = None
                scene.modified_at = datetime.now().isoformat()

        del self._assets[asset_id]

        if is_dataset and asset_scene_id in self._scenes:
            scene_has_assets = any(
                a.scene_id == asset_scene_id for a in self._assets.values()
            )
            scene_has_runs = any(r.scene_id == asset_scene_id for r in self._runs.values())
            scene = self._scenes[asset_scene_id]
            if (
                not scene_has_assets
                and not scene_has_runs
                and scene.dataset_asset_id is None
            ):
                project = self._projects.get(scene.project_id)
                if project and asset_scene_id in project.scene_ids:
                    project.scene_ids.remove(asset_scene_id)
                    project.modified_at = datetime.now().isoformat()
                del self._scenes[asset_scene_id]

        if asset_project_id in self._projects:
            project_has_scenes = bool(self._projects[asset_project_id].scene_ids)
            project_has_assets = any(
                a.project_id == asset_project_id for a in self._assets.values()
            )
            if not project_has_scenes and not project_has_assets:
                del self._projects[asset_project_id]

        self.rebuild_tag_index(save=False)
        if not self.save():
            _log.error("Failed to save library during asset deletion for %s", asset_id)
            return False
        return True

    def remove_asset(self, asset_id: str) -> bool:
        """Backward-compatible alias for delete_asset."""
        return self.delete_asset(asset_id)

    def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Get an asset by ID.

        Args:
            asset_id: Asset ID

        Returns:
            Asset or None if not found
        """
        return self._assets.get(asset_id)

    def find_asset_by_path(self, absolute_path: str) -> Optional[Asset]:
        """Find an asset by its absolute path.

        Args:
            absolute_path: Absolute file path

        Returns:
            Asset or None if not found
        """
        normalized = os.path.abspath(absolute_path)
        for asset in self._assets.values():
            if os.path.abspath(asset.absolute_path) == normalized:
                return asset
        return None

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

        for project in self._projects.values():
            _accumulate(project.tags)
        for scene in self._scenes.values():
            _accumulate(scene.tags)
        for run in self._runs.values():
            _accumulate(run.tags)
        for asset in self._assets.values():
            _accumulate(asset.tags)

        self._tags = tag_counts
        if save:
            self.save()

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

    def list_assets(
        self,
        project_id: Optional[str] = None,
        scene_id: Optional[str] = None,
        run_id: Optional[str] = None,
        type: Optional[str] = None,
        role: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Asset]:
        """List assets with optional filters.

        Args:
            project_id: Optional project ID to filter by
            scene_id: Optional scene ID to filter by
            run_id: Optional run ID to filter by
            type: Optional asset type to filter by
            role: Optional asset role to filter by
            tags: Optional tags to filter by (all must match)

        Returns:
            List of assets
        """
        assets = list(self._assets.values())
        if project_id:
            assets = [a for a in assets if a.project_id == project_id]
        if scene_id:
            assets = [a for a in assets if a.scene_id == scene_id]
        if run_id:
            assets = [a for a in assets if a.run_id == run_id]
        if type:
            assets = [a for a in assets if a.type == type]
        if role:
            assets = [a for a in assets if a.role == role]
        if tags:
            assets = [a for a in assets if all(t in a.tags for t in tags)]
        return assets

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

    def search_projects(self, query: str) -> List[Project]:
        """Search projects by name, description, or tags.

        Args:
            query: Search query string

        Returns:
            List of matching projects
        """
        query_lower = query.lower()
        results = []
        for project in self._projects.values():
            searchable = (
                f"{project.name} {project.description} {' '.join(project.tags)}".lower()
            )
            if query_lower in searchable:
                results.append(project)
        return results

    def search_scenes(
        self, query: str, project_id: Optional[str] = None
    ) -> List[Scene]:
        """Search scenes by name, description, or tags.

        Args:
            query: Search query string
            project_id: Optional project ID to filter by

        Returns:
            List of matching scenes
        """
        query_lower = query.lower()
        results = []
        scenes = self.list_scenes(project_id)
        for scene in scenes:
            searchable = (
                f"{scene.name} {scene.description} {' '.join(scene.tags)}".lower()
            )
            if query_lower in searchable:
                results.append(scene)
        return results

    def search_assets(
        self,
        query: str,
        project_id: Optional[str] = None,
        type: Optional[str] = None,
    ) -> List[Asset]:
        """Search assets by name, path, or tags.

        Args:
            query: Search query string
            project_id: Optional project ID to filter by
            type: Optional asset type to filter by

        Returns:
            List of matching assets
        """
        query_lower = query.lower()
        results = []
        assets = self.list_assets(project_id=project_id, type=type)
        for asset in assets:
            searchable = f"{asset.name} {asset.path} {' '.join(asset.tags)}".lower()
            if query_lower in searchable:
                results.append(asset)
        return results

    def get_favorite_assets(self) -> List[Asset]:
        """Get all favorite assets.

        Returns:
            List of favorite assets
        """
        return [a for a in self._assets.values() if a.is_favorite]

    def get_favorite_runs(self) -> List[TrainingRun]:
        """Get all favorite training runs.

        Returns:
            List of favorite runs
        """
        return [r for r in self._runs.values() if r.is_favorite]

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

    def get_assets_by_collection(self, collection_id: str) -> List[Asset]:
        """Get all assets in a collection.

        Args:
            collection_id: Collection ID

        Returns:
            List of assets in the collection
        """
        return [a for a in self._assets.values() if collection_id in a.collection_ids]

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
            "project_count": len(self._projects),
            "scene_count": len(self._scenes),
            "run_count": len(self._runs),
            "asset_count": len(self._assets),
            "total_size_bytes": total_size,
            "missing_files_count": missing_count,
        }
