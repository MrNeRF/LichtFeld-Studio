# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Retained RmlUI panels for dataset, checkpoint, and URL import flows."""

import asyncio
import logging
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import lichtfeld as lf

_logger = logging.getLogger(__name__)

THREAD_JOIN_TIMEOUT_SEC = 5.0
WATCH_DISCOVERY_PROGRESS_INTERVAL = 128
WATCH_REGISTER_PROGRESS_INTERVAL = 64
WATCH_UI_UPDATE_INTERVAL_SEC = 0.2
WATCH_ASSET_EXTENSIONS = {
    ".ply": "ply",
    ".rad": "rad",
    ".sog": "sog",
    ".spz": "spz",
    ".obj": "mesh",
    ".fbx": "mesh",
    ".gltf": "mesh",
    ".glb": "mesh",
    ".stl": "mesh",
    ".dae": "mesh",
    ".3ds": "mesh",
    ".mesh": "mesh",
    ".ckpt": "checkpoint",
    ".resume": "checkpoint",
    ".usd": "usd",
    ".usda": "usd",
    ".usdc": "usd",
    ".usdz": "usd",
}
WATCH_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
    ".exr",
}
WATCH_SKIP_DIRS = {
    "$recycle.bin",
    ".appledouble",
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".spotlight-v100",
    ".svn",
    ".temporaryitems",
    ".tox",
    ".trashes",
    ".venv",
    "__pycache__",
    "node_modules",
    "system volume information",
    "venv",
}
WATCH_DATASET_SKIP_DIRS = WATCH_SKIP_DIRS | {
    "dense",
    "depth",
    "depths",
    "images",
    "images_reconstruction",
    "mask",
    "masks",
    "reconstruction",
    "sparse",
    "stereo",
}


def _watch_log(
    level: str,
    message: str,
    *args,
    exc_info: bool = False,
    ui_log: bool = True,
) -> None:
    if args:
        try:
            message = message % args
        except Exception:
            pass

    prefixed = f"[AssetManagerWatch] {message}"
    logger_fn = {
        "debug": _logger.debug,
        "info": _logger.info,
        "warn": _logger.warning,
        "error": _logger.error,
    }.get(level, _logger.info)
    logger_fn(prefixed, exc_info=exc_info)

    if ui_log:
        try:
            lf_log = getattr(lf, "log", None)
            lf_level = "warn" if level == "warn" else level
            lf_fn = getattr(lf_log, lf_level, None) if lf_log is not None else None
            if callable(lf_fn):
                lf_fn(prefixed)
        except Exception:
            pass


def _join_thread(thread: Optional[threading.Thread], name: str, timeout: float = THREAD_JOIN_TIMEOUT_SEC) -> None:
    if thread is None:
        return
    if threading.current_thread() is thread:
        return
    try:
        thread.join(timeout=timeout)
        if thread.is_alive():
            _logger.warning("%s thread did not stop within %.1fs", name, timeout)
    except Exception:
        _logger.warning("Failed to join %s thread", name, exc_info=True)


from . import rml_widgets as w
from .asset_manager_integration import (
    get_asset_manager_panel,
    load_asset_index,
    load_scanner,
    load_thumbnails,
    metadata_to_asset_kwargs,
    refresh_active_panel,
    register_catalog_asset_path,
    select_asset_in_active_panel,
)
from .asset_index import resolve_asset_manager_storage_path
from .types import Panel
from .rml_keys import KI_ESCAPE, KI_RETURN
from .ui import RuntimeState
from .url_downloader import (
    URLDownloadError,
    UnsupportedURLError,
    ExtractError,
    normalize_url,
    download_url,
    extract_archive,
    get_url_info,
)


_dataset_import_panel = None
_resume_checkpoint_panel = None
_url_import_panel = None
_watch_dirs_dialog_panel = None
_watch_dirs_dialog_state = {
    "folder_id": None,
    "folder_name": "",
    "watch_dirs": [],
    "version": 0,
}

__lfs_panel_classes__ = [
    "DatasetImportPanel",
    "ResumeCheckpointPanel",
    "URLImportPanel",
    "WatchDirsDialogPanel",
]
__lfs_panel_ids__ = [
    "lfs.dataset_import",
    "lfs.resume_checkpoint",
    "lfs.url_import",
    "lfs.watch_dirs_dialog",
]


def open_dataset_import_panel(
    dataset_path: str,
    *,
    clear_scene_on_load: bool = False,
) -> bool:
    """Open the retained dataset import dialog for the given dataset path."""
    if _dataset_import_panel is None:
        return False
    return _dataset_import_panel.show(
        dataset_path,
        clear_scene_on_load=clear_scene_on_load,
    )


def open_resume_checkpoint_panel(checkpoint_path: str) -> bool:
    """Open the retained checkpoint resume dialog for the given checkpoint path."""
    if _resume_checkpoint_panel is None:
        return False
    return _resume_checkpoint_panel.show(checkpoint_path)


def open_url_import_panel() -> bool:
    """Open the retained URL import panel."""
    if _url_import_panel is None:
        return False
    return _url_import_panel.show()


def open_watch_dirs_dialog(folder_id: str) -> bool:
    """Open the watched directories dialog for the given folder."""
    global _watch_dirs_dialog_panel
    if not folder_id:
        return False
    if _watch_dirs_dialog_panel is None:
        try:
            lf.register_class(WatchDirsDialogPanel)
        except Exception:
            pass
    try:
        if not _load_watch_dirs_dialog_state(folder_id):
            return False
        lf.ui.set_panel_enabled(WatchDirsDialogPanel.id, True)
        if _watch_dirs_dialog_panel is not None:
            _watch_dirs_dialog_panel._sync_from_shared_state()
            _watch_dirs_dialog_panel._dirty_model()
        return True
    except Exception as e:
        _logger.error("Failed to open watch dirs dialog: %s", e, exc_info=True)
        return False


def _set_watch_dirs_dialog_state(
    folder_id: Optional[str],
    folder_name: str = "",
    watch_dirs: Optional[list[str]] = None,
) -> None:
    _watch_dirs_dialog_state["folder_id"] = folder_id
    _watch_dirs_dialog_state["folder_name"] = folder_name
    _watch_dirs_dialog_state["watch_dirs"] = list(watch_dirs or [])
    _watch_dirs_dialog_state["version"] = int(_watch_dirs_dialog_state.get("version") or 0) + 1


def _load_watch_dirs_dialog_state(folder_id: str) -> bool:
    index = load_asset_index()
    if index is None:
        _watch_log("error", "catalog load failed")
        return False
    folder = index.get_folder(folder_id)
    if folder is None:
        _watch_log(
            "error",
            "show aborted: folder not found folder_id=%s available=%s library=%s",
            folder_id,
            list(getattr(index, "folders", {}).keys()),
            _index_library_path(index),
        )
        return False
    folder_name = getattr(folder, "name", "Unnamed Folder")
    watch_dirs = index.get_watch_dirs(folder_id)
    _set_watch_dirs_dialog_state(folder_id, folder_name, watch_dirs)
    _watch_log(
        "info",
        "show loaded folder_id=%s folder_name=%s watch_dirs=%s library=%s",
        folder_id,
        folder_name,
        watch_dirs,
        _index_library_path(index),
    )
    return True


def _tr(key: str) -> str:
    return lf.ui.tr(key)


def _index_library_path(index) -> str:
    return str(getattr(index, "library_path", "<unknown library path>"))


def _safe_count(mapping) -> int:
    try:
        return len(mapping)
    except Exception:
        return -1


def _library_mtime(index) -> str:
    try:
        path = getattr(index, "library_path", None)
        if path is not None and path.exists():
            return str(path.stat().st_mtime)
    except Exception:
        pass
    return "missing"


def _asset_role_for_type(asset_type: Optional[str]) -> str:
    if asset_type == "dataset":
        return "source_dataset"
    if asset_type == "checkpoint":
        return "training_checkpoint"
    if asset_type in {"ply", "rad", "sog", "spz", "mesh", "usd"}:
        return "trained_output"
    return "unknown"


def _lightweight_asset_metadata(path: str, asset_type: str) -> dict[str, Any]:
    """Return cheap metadata for watch discovery; detailed metadata is synced later."""
    path_obj = Path(path)
    metadata = {
        "path": str(path_obj.resolve()),
        "name": path_obj.name,
        "type": asset_type,
        "role": _asset_role_for_type(asset_type),
        "size_bytes": 0,
        "created": None,
        "modified": None,
        "format_specific": {},
    }
    try:
        stat = path_obj.stat()
        metadata["size_bytes"] = 0 if path_obj.is_dir() else stat.st_size
        metadata["created"] = datetime.fromtimestamp(stat.st_ctime).isoformat()
        metadata["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
    except (OSError, PermissionError):
        pass
    return metadata


def _entry_name_lower(entry) -> str:
    try:
        return entry.name.lower()
    except Exception:
        return ""


def _safe_is_file(entry) -> bool:
    try:
        return entry.is_file(follow_symlinks=False)
    except (OSError, PermissionError):
        return False


def _safe_is_dir(entry) -> bool:
    try:
        return entry.is_dir(follow_symlinks=False)
    except (OSError, PermissionError):
        return False


def _directory_has_image_file(path: str) -> bool:
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if not _safe_is_file(entry):
                    continue
                if os.path.splitext(entry.name)[1].lower() in WATCH_IMAGE_EXTENSIONS:
                    return True
    except (OSError, PermissionError):
        return False
    return False


def _directory_has_colmap_file(path: str) -> bool:
    colmap_files = {
        "cameras.bin",
        "images.bin",
        "points3d.bin",
        "cameras.txt",
        "images.txt",
        "points3d.txt",
    }
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                name = _entry_name_lower(entry)
                if _safe_is_file(entry) and name in colmap_files:
                    return True
                if name == "0" and _safe_is_dir(entry):
                    try:
                        with os.scandir(entry.path) as nested_entries:
                            for nested in nested_entries:
                                if (
                                    _safe_is_file(nested)
                                    and _entry_name_lower(nested) in colmap_files
                                ):
                                    return True
                    except (OSError, PermissionError):
                        continue
    except (OSError, PermissionError):
        return False
    return False


def _looks_like_dataset_from_entries(dirs: list[Any], files: list[Any]) -> bool:
    dir_by_name = {_entry_name_lower(entry): entry for entry in dirs}
    file_names = {_entry_name_lower(entry) for entry in files}

    if "database.db" in file_names:
        return True
    if "cameras.txt" in file_names and "images.txt" in file_names:
        return True

    images_dir = dir_by_name.get("images")
    if images_dir is not None and _directory_has_image_file(images_dir.path):
        return True

    sparse_dir = dir_by_name.get("sparse")
    if sparse_dir is not None and _directory_has_colmap_file(sparse_dir.path):
        return True

    return False


def _discover_asset_metadata_fast(
    path: str,
    *,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[str, int, int, int], None]] = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    visited_dirs = 0
    visited_files = 0

    def _cancelled() -> bool:
        return bool(should_cancel is not None and should_cancel())

    def _report(current_path: str, *, force: bool = False) -> None:
        if on_progress is None:
            return
        if not force and visited_dirs % WATCH_DISCOVERY_PROGRESS_INTERVAL != 0:
            return
        on_progress(current_path, visited_dirs, visited_files, len(results))

    def _append_metadata(candidate_path: str, asset_type: str) -> None:
        if _cancelled():
            raise InterruptedError("Scan cancelled")
        try:
            metadata = _lightweight_asset_metadata(candidate_path, asset_type)
        except Exception as exc:
            _watch_log("debug", "metadata skipped path=%s error=%s", candidate_path, exc)
            return
        metadata_path = metadata.get("path")
        if not metadata_path or metadata_path in seen_paths:
            return
        results.append(metadata)
        seen_paths.add(metadata_path)

    root = Path(path)
    if not root.exists():
        return results

    if root.is_file():
        asset_type = WATCH_ASSET_EXTENSIONS.get(root.suffix.lower())
        if asset_type:
            _append_metadata(str(root), asset_type)
        return results

    stack = [str(root)]
    while stack:
        if _cancelled():
            raise InterruptedError("Scan cancelled")

        current_path = stack.pop()
        visited_dirs += 1
        _report(current_path)

        try:
            with os.scandir(current_path) as entries:
                dir_entries = []
                file_entries = []
                for entry in entries:
                    if _safe_is_dir(entry):
                        dir_entries.append(entry)
                    elif _safe_is_file(entry):
                        file_entries.append(entry)
        except (OSError, PermissionError) as exc:
            _watch_log("debug", "walk skipped path=%s error=%s", current_path, exc)
            continue

        is_dataset = _looks_like_dataset_from_entries(dir_entries, file_entries)
        if is_dataset:
            _append_metadata(current_path, "dataset")
            _report(current_path, force=True)

        for entry in file_entries:
            if _cancelled():
                raise InterruptedError("Scan cancelled")
            visited_files += 1
            asset_type = WATCH_ASSET_EXTENSIONS.get(
                os.path.splitext(entry.name)[1].lower()
            )
            if asset_type is None:
                continue
            _append_metadata(entry.path, asset_type)
            _report(entry.path, force=True)

        skip_dirs = WATCH_DATASET_SKIP_DIRS if is_dataset else WATCH_SKIP_DIRS
        for entry in reversed(dir_entries):
            if _entry_name_lower(entry) in skip_dirs:
                continue
            stack.append(entry.path)

    _report(str(root), force=True)
    return results


def _discover_asset_metadata(
    scanner,
    path: str,
    *,
    fast_walk: bool = False,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[str, int, int, int], None]] = None,
) -> list[dict[str, Any]]:
    """Discover assets under a path using the shared scanner contract."""
    if scanner is None:
        _watch_log("error", "discover skipped because scanner is None")
        return []

    if fast_walk:
        metadata_list = _discover_asset_metadata_fast(
            path,
            should_cancel=should_cancel,
            on_progress=on_progress,
        )
        _watch_log(
            "debug",
            "discover complete path=%s count=%d",
            path,
            len(metadata_list),
        )
        return metadata_list

    if hasattr(scanner, "scan_directory_deep"):
        metadata_list = scanner.scan_directory_deep(path)
        _watch_log(
            "debug",
            "discover complete path=%s count=%d",
            path,
            len(metadata_list),
        )
        return metadata_list

    metadata_list: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    root_metadata = scanner.scan_file(path)
    root_type = root_metadata.get("type")
    root_path = root_metadata.get("path")
    if root_type is not None and root_path:
        metadata_list.append(root_metadata)
        seen_paths.add(root_path)

    for metadata in scanner.scan_directory(path, recursive=True):
        metadata_path = metadata.get("path")
        if not metadata_path or metadata_path in seen_paths:
            continue
        metadata_list.append(metadata)
        seen_paths.add(metadata_path)

    _watch_log(
        "debug",
        "discover complete path=%s count=%d",
        path,
        len(metadata_list),
    )
    return metadata_list


def _register_discovered_assets(
    index,
    thumbnails,
    metadata_list: list[dict[str, Any]],
    *,
    folder_id: Optional[str],
    scene_id: Optional[str] = None,
    name_override: Optional[str] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int, int, int], None]] = None,
    log_each_asset: bool = True,
    save_immediately: bool = True,
) -> list[Any]:
    """Create catalog assets from discovered metadata using shared logic."""
    created_assets = []
    skipped_count = 0
    single_asset_override = name_override if len(metadata_list) == 1 else None
    if not folder_id:
        _watch_log(
            "warn",
            "register skipped without selected folder library=%s metadata_count=%d",
            _index_library_path(index),
            len(metadata_list),
        )
        return []

    existing_paths: set[str] = set()
    try:
        for asset in index.assets.values():
            if folder_id is not None and asset.get("folder_id") != folder_id:
                continue
            asset_path = asset.get("absolute_path") or asset.get("path")
            if asset_path:
                existing_paths.add(os.path.abspath(str(asset_path)))
    except Exception:
        existing_paths = set()

    total_count = len(metadata_list)
    for processed_count, metadata in enumerate(metadata_list, start=1):
        if should_cancel is not None and should_cancel():
            raise InterruptedError("Scan cancelled")

        file_path = metadata.get("path")
        if not file_path or not os.path.exists(file_path):
            skipped_count += 1
            if log_each_asset:
                _watch_log(
                    "warn",
                    "register skipped missing path metadata_path=%s name=%s type=%s",
                    file_path,
                    metadata.get("name"),
                    metadata.get("type"),
                )
            if on_progress is not None:
                on_progress(
                    processed_count,
                    total_count,
                    len(created_assets),
                    skipped_count,
                )
            continue
        normalized_path = os.path.abspath(str(file_path))
        if normalized_path in existing_paths:
            skipped_count += 1
            if log_each_asset:
                _watch_log(
                    "info",
                    "register skipped existing folder asset path=%s folder_id=%s",
                    file_path,
                    folder_id,
                )
            if on_progress is not None:
                on_progress(
                    processed_count,
                    total_count,
                    len(created_assets),
                    skipped_count,
                )
            continue

        asset_name = (
            single_asset_override
            or metadata.get("name")
            or Path(file_path).name
        )
        asset = index.create_asset(
            folder_id=folder_id,
            name=asset_name,
            type=metadata.get("type", "unknown"),
            path=file_path,
            absolute_path=file_path,
            scene_id=scene_id,
            role=metadata.get("role", "reference"),
            save=save_immediately,
            check_existing=save_immediately,
            rebuild_tags=save_immediately,
            **metadata_to_asset_kwargs(metadata),
        )
        if asset is None:
            skipped_count += 1
            _watch_log(
                "error",
                "register create_asset returned None path=%s type=%s role=%s library=%s",
                file_path,
                metadata.get("type"),
                metadata.get("role"),
                _index_library_path(index),
            )
            continue
        created_assets.append(asset)
        existing_paths.add(normalized_path)
        if log_each_asset:
            _watch_log(
                "info",
                "register created asset_id=%s name=%s type=%s path=%s",
                asset.id,
                asset.name,
                asset.type,
                file_path,
            )
        if thumbnails is not None:
            def _maybe_await(coro_or_result):
                if asyncio.iscoroutine(coro_or_result):
                    return asyncio.run(coro_or_result)
                return coro_or_result

            try:
                thumb_path = None
                if asset.type == "dataset" and hasattr(thumbnails, "generate_dataset_preview"):
                    thumb_path = _maybe_await(
                        thumbnails.generate_dataset_preview(
                            asset.type,
                            asset.id,
                            file_path,
                            metadata.get("format_specific", {}) or {},
                        )
                    )
                elif hasattr(thumbnails, "generate_rendered_preview"):
                    thumb_path = _maybe_await(
                        thumbnails.generate_rendered_preview(
                            asset.type,
                            asset.id,
                            file_path,
                        )
                    )
                if thumb_path is None:
                    thumb_path = _maybe_await(thumbnails.generate_placeholder(asset.type, asset.id))
                index.update_asset(asset.id, thumbnail_path=str(thumb_path))
            except Exception:
                _watch_log(
                    "warn",
                    "thumbnail generation failed asset_id=%s path=%s",
                    asset.id,
                    file_path,
                    exc_info=True,
                )

        if (
            on_progress is not None
            and (
                processed_count == total_count
                or processed_count % WATCH_REGISTER_PROGRESS_INTERVAL == 0
            )
        ):
            on_progress(
                processed_count,
                total_count,
                len(created_assets),
                skipped_count,
            )
        if processed_count % WATCH_REGISTER_PROGRESS_INTERVAL == 0:
            time.sleep(0)

    if created_assets and not save_immediately:
        try:
            index.rebuild_tag_index(save=False)
        except Exception:
            _watch_log(
                "debug",
                "tag index rebuild after bulk register failed",
                exc_info=True,
            )

    _watch_log(
        "debug",
        "register complete created_count=%d skipped_count=%d metadata_count=%d",
        len(created_assets),
        skipped_count,
        len(metadata_list),
    )
    return created_assets


class _ImportDialogPanel(Panel):
    """Common behavior for retained import dialogs."""

    update_policy = "dirty"
    form_id = ""

    def on_mount(self, doc):
        super().on_mount(doc)
        self._last_lang = lf.ui.get_current_language()
        self._escape_revert = w.EscapeRevertController()
        doc.add_event_listener("keydown", self._on_keydown)
        self._form = doc.get_element_by_id(self.form_id) if self.form_id else None
        if self._form:
            self._form.add_event_listener("submit", self._on_form_submit)
            self._form.add_event_listener("change", self._on_form_change)
        for el in doc.query_selector_all('input[type="text"]'):
            prop = el.get_attribute("data-value", "")
            if not prop:
                continue
            w.bind_select_all_on_focus(el)
            self._escape_revert.bind(
                el,
                prop,
                lambda p=prop: self._capture_bound_input_value(p),
                lambda snapshot, p=prop: self._restore_bound_input_value(p, snapshot),
            )
        self._subscribe_reactive_state()

    def on_unmount(self, _doc):
        self._unsubscribe_reactive_state()

    def _subscribe_reactive_state(self):
        if getattr(self, "_reactive_unsubscribers", None):
            return

        self._reactive_unsubscribers = [
            RuntimeState.language_generation.subscribe(lambda _value: self._request_reactive_update()),
        ]

    def _unsubscribe_reactive_state(self):
        for unsubscribe in getattr(self, "_reactive_unsubscribers", []):
            try:
                unsubscribe()
            except Exception:
                pass
        self._reactive_unsubscribers = []

    def _request_reactive_update(self):
        handle = getattr(self, "_handle", None)
        if handle:
            w.request_model_update(handle)

    def _on_keydown(self, event):
        key = int(event.get_parameter("key_identifier", "0"))
        if key == KI_RETURN and self._can_submit_from_keyboard():
            self._on_do_load()
            event.stop_propagation()
        elif key == KI_ESCAPE:
            self._on_do_cancel()
            event.stop_propagation()

    def _capture_bound_input_value(self, prop: str) -> str:
        return str(getattr(self, f"_{prop}", "") or "")

    def _restore_bound_input_value(self, prop: str, snapshot) -> None:
        attr = f"_{prop}"
        if hasattr(self, attr):
            setattr(self, attr, str(snapshot or ""))
        if hasattr(self, "_dirty_model"):
            self._dirty_model()

    def _on_form_submit(self, event):
        if self._can_submit_from_keyboard():
            self._on_do_load()
        event.stop_propagation()

    def _on_form_change(self, event):
        target = event.target()
        if target is None or not event.get_bool_parameter("linebreak", False):
            return
        if target.tag_name != "input":
            return

        input_type = target.get_attribute("type", "text")
        if input_type not in ("", "text", "password", "search", "email", "url"):
            return

        if self._form and self._can_submit_from_keyboard():
            self._form.submit()
            event.stop_propagation()

    def _can_submit_from_keyboard(self) -> bool:
        return False


class DatasetImportPanel(_ImportDialogPanel):
    """Floating panel for configuring dataset import paths."""

    id = "lfs.dataset_import"
    label = "Load Dataset"
    space = lf.ui.PanelSpace.FLOATING
    order = 11
    template = "rmlui/dataset_import_panel.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (560, 0)
    form_id = "dataset-import-form"

    DEFAULT_MAX_WIDTH = 3840

    def __init__(self):
        global _dataset_import_panel
        _dataset_import_panel = self

        self._handle = None
        self._dataset_path = ""
        self._dataset_info = None
        self._dataset_valid = False
        self._output_path = ""
        self._init_path = ""
        self._ppisp_sidecar_path = ""
        self._centralize_dataset = "off"
        self._max_width = self.DEFAULT_MAX_WIDTH
        self._max_width_str = str(self.DEFAULT_MAX_WIDTH)
        self._apply_auto_crop = False
        self._clear_scene_on_load = False
        self._last_lang = ""

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("dataset_import")
        if model is None:
            return

        model.bind_func("panel_label", lambda: lf.ui.tr("load_dataset_popup.title"))

        model.bind_func("images_path", lambda: self._string_attr("images_path"))
        model.bind_func("sparse_path", lambda: self._string_attr("sparse_path"))
        model.bind_func("masks_path", lambda: self._string_attr("masks_path"))
        model.bind_func("images_count_text", self._images_count_text)
        model.bind_func("mask_count_text", self._mask_count_text)
        model.bind_func("show_masks", lambda: bool(self._dataset_info and getattr(self._dataset_info, "has_masks", False)))
        model.bind_func("can_load", lambda: bool(self._dataset_valid and self._output_path.strip()))

        model.bind("dataset_path", lambda: self._dataset_path, self._set_dataset_path)
        model.bind("output_path", lambda: self._output_path, self._set_output_path)
        model.bind("init_path", lambda: self._init_path, self._set_init_path)
        model.bind("ppisp_sidecar_path", lambda: self._ppisp_sidecar_path, self._set_ppisp_sidecar_path)
        model.bind("centralize_dataset", lambda: self._centralize_dataset, self._set_centralize_dataset)
        model.bind("max_width_str", lambda: self._max_width_str, self._set_max_width_str)
        model.bind("max_width_disabled", lambda: self._max_width == 0, self._set_max_width_disabled)
        model.bind("apply_auto_crop", lambda: self._apply_auto_crop, self._set_apply_auto_crop)

        model.bind_event("browse_dataset", self._on_browse_dataset)
        model.bind_event("browse_output", self._on_browse_output)
        model.bind_event("browse_init", self._on_browse_init)
        model.bind_event("browse_ppisp_sidecar", self._on_browse_ppisp_sidecar)
        model.bind_event("do_load", self._on_do_load)
        model.bind_event("do_cancel", self._on_do_cancel)

        self._handle = model.get_handle()

    def on_update(self, doc):
        del doc
        current_lang = lf.ui.get_current_language()
        if current_lang != self._last_lang:
            self._last_lang = current_lang
            self._dirty_model()
            return True
        return False

    def show(self, dataset_path: str, *, clear_scene_on_load: bool = False) -> bool:
        self._clear_scene_on_load = False
        if not self._apply_dataset_path(dataset_path, reset_output=True):
            return False

        self._clear_scene_on_load = bool(clear_scene_on_load)
        self._init_path = ""
        self._centralize_dataset = "off"
        self._max_width = self.DEFAULT_MAX_WIDTH
        self._max_width_str = str(self.DEFAULT_MAX_WIDTH)
        self._apply_auto_crop = False
        params = lf.optimization_params()
        self._ppisp_sidecar_path = (
            str(params.ppisp_sidecar_path) if params and params.has_params() else ""
        )
        self._dirty_model()
        lf.ui.set_panel_enabled(self.id, True)
        return True

    def _can_submit_from_keyboard(self) -> bool:
        return bool(self._dataset_valid and self._output_path.strip())

    def _preview_base_path(self, dataset_path: str) -> Path:
        path = Path(dataset_path)
        if path.suffix.lower() == ".json":
            return path.parent
        return path

    def _default_output_path(self, dataset_path: str, info) -> str:
        preview_root = self._preview_base_path(dataset_path)
        base_path = Path(info.base_path) if info is not None else preview_root
        return str(base_path / "output")

    def _apply_dataset_path(self, dataset_path: str, reset_output: bool) -> bool:
        next_value = str(dataset_path).strip()
        dataset_changed = next_value != self._dataset_path
        self._dataset_path = next_value
        self._dataset_valid = bool(next_value) and lf.is_dataset_path(next_value)
        self._dataset_info = None

        if dataset_changed:
            self._init_path = ""
            self._ppisp_sidecar_path = ""

        if self._dataset_valid:
            self._dataset_info = lf.detect_dataset_info(str(self._preview_base_path(next_value)))
            if reset_output:
                self._output_path = self._default_output_path(next_value, self._dataset_info)
        elif reset_output:
            self._output_path = ""

        self._dirty_model(
            "dataset_path",
            "images_path",
            "sparse_path",
            "masks_path",
            "images_count_text",
            "mask_count_text",
            "show_masks",
            "output_path",
            "init_path",
            "ppisp_sidecar_path",
            "can_load",
        )
        return self._dataset_valid

    def _dirty_model(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            return
        for field in fields:
            self._handle.dirty(field)

    def _string_attr(self, name: str) -> str:
        if self._dataset_info is None:
            return ""
        value = getattr(self._dataset_info, name, "")
        return str(value) if value is not None else ""

    def _images_count_text(self) -> str:
        if self._dataset_info is None:
            return ""
        return f"({int(getattr(self._dataset_info, 'image_count', 0))} images)"

    def _mask_count_text(self) -> str:
        if self._dataset_info is None or not getattr(self._dataset_info, "has_masks", False):
            return ""
        return f"({int(getattr(self._dataset_info, 'mask_count', 0))} masks)"

    def _set_output_path(self, value):
        next_value = str(value)
        if next_value == self._output_path:
            return
        self._output_path = next_value
        self._dirty_model("output_path", "can_load")

    def _set_dataset_path(self, value):
        next_value = str(value)
        if next_value == self._dataset_path:
            return
        self._apply_dataset_path(next_value, reset_output=True)

    def _set_init_path(self, value):
        next_value = str(value)
        if next_value == self._init_path:
            return
        self._init_path = next_value
        self._dirty_model("init_path")

    def _set_ppisp_sidecar_path(self, value):
        next_value = str(value)
        if next_value == self._ppisp_sidecar_path:
            return
        self._ppisp_sidecar_path = next_value
        self._dirty_model("ppisp_sidecar_path")

    def _set_centralize_dataset(self, value):
        next_value = str(value)
        if next_value == self._centralize_dataset:
            return
        self._centralize_dataset = next_value
        self._dirty_model("centralize_dataset")

    def _set_max_width_str(self, value):
        text = str(value).strip().replace(",", "")
        try:
            parsed = int(text) if text else 0
        except ValueError:
            self._dirty_model("max_width_str")
            return
        if parsed < 0:
            parsed = 0
        self._max_width = parsed
        self._max_width_str = str(parsed)
        self._dirty_model("max_width_str", "max_width_disabled")

    def _set_max_width_disabled(self, value):
        disabled = bool(value)
        if disabled:
            if self._max_width == 0:
                return
            self._max_width = 0
            self._max_width_str = "0"
        else:
            if self._max_width != 0:
                return
            self._max_width = self.DEFAULT_MAX_WIDTH
            self._max_width_str = str(self.DEFAULT_MAX_WIDTH)
        self._dirty_model("max_width_str", "max_width_disabled")

    def _set_apply_auto_crop(self, value):
        enabled = bool(value)
        if enabled == self._apply_auto_crop:
            return
        self._apply_auto_crop = enabled
        self._dirty_model("apply_auto_crop")

    def _on_browse_dataset(self, _handle=None, _ev=None, _args=None):
        path = lf.ui.open_dataset_folder_dialog()
        if path:
            self._set_dataset_path(path)

    def _on_browse_output(self, _handle=None, _ev=None, _args=None):
        path = lf.ui.open_dataset_folder_dialog()
        if path:
            self._set_output_path(path)

    def _on_browse_init(self, _handle=None, _ev=None, _args=None):
        if not self._dataset_path.strip():
            return
        path = lf.ui.open_ply_file_dialog(str(self._preview_base_path(self._dataset_path)))
        if path:
            self._set_init_path(path)

    def _on_browse_ppisp_sidecar(self, _handle=None, _ev=None, _args=None):
        start_dir = str(self._preview_base_path(self._dataset_path)) if self._dataset_path else ""
        if self._ppisp_sidecar_path:
            start_dir = self._ppisp_sidecar_path
        path = lf.ui.open_ppisp_file_dialog(start_dir)
        if path:
            self._set_ppisp_sidecar_path(path)

    def _on_do_load(self, _handle=None, _ev=None, _args=None):
        if not self._dataset_valid or not self._output_path.strip():
            return

        dataset_path = self._dataset_path.strip()
        init_path = self._init_path.strip()
        ppisp_sidecar_path = self._ppisp_sidecar_path.strip()
        centralize_dataset = self._centralize_dataset

        params = lf.optimization_params()
        if params and params.has_params():
            params.ppisp_sidecar_path = ppisp_sidecar_path
            params.ppisp_freeze_from_sidecar = bool(ppisp_sidecar_path)
            if ppisp_sidecar_path:
                params.ppisp = True

        lf.ui.set_panel_enabled(self.id, False)
        register_catalog_asset_path(dataset_path, is_dataset=True, select=True)
        clear_scene_on_load = self._clear_scene_on_load
        self._clear_scene_on_load = False
        if clear_scene_on_load:
            lf.clear_scene()
        lf.load_file(
            dataset_path,
            is_dataset=True,
            output_path=self._output_path.strip(),
            init_path=init_path,
            centralize_dataset=centralize_dataset,
            max_width=self._max_width,
            apply_auto_crop=self._apply_auto_crop,
        )

    def _on_do_cancel(self, _handle=None, _ev=None, _args=None):
        self._clear_scene_on_load = False
        lf.ui.set_panel_enabled(self.id, False)


class ResumeCheckpointPanel(_ImportDialogPanel):
    """Floating panel for configuring checkpoint resume paths."""

    id = "lfs.resume_checkpoint"
    label = "Resume Checkpoint"
    space = lf.ui.PanelSpace.FLOATING
    order = 12
    template = "rmlui/resume_checkpoint_panel.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (580, 0)
    form_id = "resume-checkpoint-form"

    def __init__(self):
        global _resume_checkpoint_panel
        _resume_checkpoint_panel = self

        self._handle = None
        self._checkpoint_path = ""
        self._header = None
        self._stored_dataset_path = ""
        self._dataset_path = ""
        self._output_path = ""
        self._dataset_valid = False
        self._stored_dataset_exists = False
        self._last_lang = ""

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("resume_checkpoint")
        if model is None:
            return

        model.bind_func("panel_label", lambda: lf.ui.tr("resume_checkpoint_popup.title"))
        model.bind_func("checkpoint_filename", self._checkpoint_filename)
        model.bind_func("checkpoint_metadata", self._checkpoint_metadata)
        model.bind_func("stored_path_text", lambda: self._stored_dataset_path)
        model.bind_func("stored_path_class", self._stored_path_class)
        model.bind_func("show_stored_missing", lambda: bool(self._stored_dataset_path and not self._stored_dataset_exists))
        model.bind_func("dataset_status_text", self._dataset_status_text)
        model.bind_func("dataset_status_class", self._dataset_status_class)
        model.bind_func("can_load", lambda: self._dataset_valid)

        model.bind("dataset_path", lambda: self._dataset_path, self._set_dataset_path)
        model.bind("output_path", lambda: self._output_path, self._set_output_path)

        model.bind_event("browse_dataset", self._on_browse_dataset)
        model.bind_event("browse_output", self._on_browse_output)
        model.bind_event("do_load", self._on_do_load)
        model.bind_event("do_cancel", self._on_do_cancel)

        self._handle = model.get_handle()

    def on_update(self, doc):
        del doc
        current_lang = lf.ui.get_current_language()
        if current_lang != self._last_lang:
            self._last_lang = current_lang
            self._dirty_model()
            return True
        return False

    def show(self, checkpoint_path: str) -> bool:
        header = lf.read_checkpoint_header(checkpoint_path)
        if not header:
            return False

        params = lf.read_checkpoint_params(checkpoint_path)
        if not params:
            return False

        self._checkpoint_path = checkpoint_path
        self._header = header
        self._stored_dataset_path = str(params.dataset_path)
        self._dataset_path = self._stored_dataset_path
        self._output_path = str(params.output_path)
        self._stored_dataset_exists = self._validate_dataset(self._stored_dataset_path)
        self._dataset_valid = self._stored_dataset_exists
        self._dirty_model()
        lf.ui.set_panel_enabled(self.id, True)
        return True

    def _can_submit_from_keyboard(self) -> bool:
        return self._dataset_valid and bool(self._checkpoint_path)

    def _dirty_model(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            return
        for field in fields:
            self._handle.dirty(field)

    def _validate_dataset(self, path: str) -> bool:
        return bool(path) and Path(path).is_dir()

    def _checkpoint_filename(self) -> str:
        if not self._checkpoint_path:
            return ""
        return Path(self._checkpoint_path).name

    def _checkpoint_metadata(self) -> str:
        if self._header is None:
            return ""
        return f"(iter {int(self._header.iteration)}, {int(self._header.num_gaussians)} gaussians)"

    def _stored_path_class(self) -> str:
        if self._stored_dataset_path and not self._stored_dataset_exists:
            return "impdlg-value status-error"
        return "impdlg-value text-default"

    def _dataset_status_text(self) -> str:
        if self._dataset_valid:
            return lf.ui.tr("common.ok") or "common.ok"
        return lf.ui.tr("resume_checkpoint_popup.invalid") or "resume_checkpoint_popup.invalid"

    def _dataset_status_class(self) -> str:
        if self._dataset_valid:
            return "impdlg-status status-success"
        return "impdlg-status status-error"

    def _set_dataset_path(self, value):
        next_value = str(value)
        if next_value == self._dataset_path:
            return
        self._dataset_path = next_value
        self._dataset_valid = self._validate_dataset(next_value)
        self._dirty_model("dataset_path", "dataset_status_text", "dataset_status_class", "can_load")

    def _set_output_path(self, value):
        next_value = str(value)
        if next_value == self._output_path:
            return
        self._output_path = next_value
        self._dirty_model("output_path")

    def _on_browse_dataset(self, _handle=None, _ev=None, _args=None):
        path = lf.ui.open_dataset_folder_dialog()
        if path:
            self._set_dataset_path(path)

    def _on_browse_output(self, _handle=None, _ev=None, _args=None):
        path = lf.ui.open_dataset_folder_dialog()
        if path:
            self._set_output_path(path)

    def _on_do_load(self, _handle=None, _ev=None, _args=None):
        if not self._dataset_valid or not self._checkpoint_path:
            return

        lf.ui.set_panel_enabled(self.id, False)
        register_catalog_asset_path(self._dataset_path, is_dataset=True)
        register_catalog_asset_path(
            self._checkpoint_path,
            asset_type="checkpoint",
            role="training_checkpoint",
            select=True,
        )
        lf.load_checkpoint_for_training(
            self._checkpoint_path,
            self._dataset_path,
            self._output_path,
        )

    def _on_do_cancel(self, _handle=None, _ev=None, _args=None):
        lf.ui.set_panel_enabled(self.id, False)


class URLImportPanel(_ImportDialogPanel):
    """Floating panel for importing assets from URL-backed sources."""

    id = "lfs.url_import"
    label = "Import from URL"
    space = lf.ui.PanelSpace.FLOATING
    order = 13
    template = "rmlui/url_import_panel.rml"
    height_mode = lf.ui.PanelHeightMode.FILL
    size = (560, 360)
    form_id = "url-import-form"

    STORAGE_PATH = resolve_asset_manager_storage_path()

    def __init__(self):
        global _url_import_panel
        _url_import_panel = self

        self._handle = None
        self._doc = None
        self._last_lang = ""
        self._url_import_url = ""
        self._url_import_progress = -1.0
        self._url_import_status = ""
        self._url_import_warning = ""
        self._url_import_in_progress = False
        self._url_import_thread: Optional[threading.Thread] = None
        self._url_import_cancelled = False
        self._url_import_session_id = 0
        self._url_import_close_timer: Optional[threading.Timer] = None
        self._url_import_formats_open = False
        self._formats_header = None
        self._formats_arrow = None
        self._formats_content = None

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        self._formats_header = doc.get_element_by_id("url-import-formats-header")
        self._formats_arrow = doc.get_element_by_id("url-import-formats-arrow")
        self._formats_content = doc.get_element_by_id("url-import-formats-content")
        self._sync_formats_ui()

    def on_unmount(self, doc):
        self._unsubscribe_reactive_state()
        # Cancel any in-progress download before unmounting
        if self._url_import_in_progress and not self._url_import_cancelled:
            self._url_import_cancelled = True
        self._cancel_url_import_close_timer()
        _join_thread(self._url_import_close_timer, "URL import close timer")
        self._url_import_close_timer = None
        url_import_thread = self._url_import_thread
        self._url_import_thread = None
        _join_thread(url_import_thread, "URL import worker")
        self._handle = None
        self._doc = None
        self._formats_header = None
        self._formats_arrow = None
        self._formats_content = None
        doc.remove_data_model("url_import")

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("url_import")
        if model is None:
            return

        model.bind_func("panel_label", lambda: _tr("asset_manager.import_from_url"))
        model.bind(
            "url_import_url",
            lambda: self._url_import_url,
            self._set_url_import_url,
        )
        model.bind_func(
            "url_import_progress",
            lambda: str(
                max(
                    0.0,
                    min(
                        1.0,
                        self._url_import_progress
                        if self._url_import_progress >= 0.0
                        else 0.0,
                    ),
                )
            ),
        )
        model.bind_func("url_import_status", lambda: self._url_import_status)
        model.bind_func("url_import_warning_text", lambda: self._url_import_warning)
        model.bind_func("url_import_has_warning", lambda: bool(self._url_import_warning))
        model.bind_func(
            "url_import_show_progress",
            lambda: self._url_import_in_progress or bool(self._url_import_status),
        )
        model.bind_func(
            "url_import_show_progress_bar",
            lambda: self._url_import_in_progress
            and self._url_import_progress >= 0.0
            and not self._url_import_cancelled,
        )
        model.bind_func(
            "url_import_button_enabled",
            lambda: not self._url_import_in_progress and bool(self._url_import_url.strip()),
        )
        model.bind_func(
            "url_import_button_text",
            lambda: _tr("asset_manager.import_button_downloading")
            if self._url_import_in_progress
            else _tr("asset_manager.import_button"),
        )

        model.bind_event("toggle_formats", self._on_toggle_formats)
        model.bind_event("do_load", self._on_do_load)
        model.bind_event("do_cancel", self._on_do_cancel)

        self._handle = model.get_handle()

    def on_update(self, doc):
        del doc
        current_lang = lf.ui.get_current_language()
        if current_lang != self._last_lang:
            self._last_lang = current_lang
            self._dirty_model()
            return True
        return False

    def show(self) -> bool:
        self._reset_state()
        self._dirty_model()
        self._sync_formats_ui()
        lf.ui.set_panel_enabled(self.id, True)
        return True

    def _can_submit_from_keyboard(self) -> bool:
        return not self._url_import_in_progress and bool(self._url_import_url.strip())

    def _dirty_model(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            return
        for field in fields:
            self._handle.dirty(field)

    def _set_url_import_url(self, value):
        next_value = str(value)
        if next_value == self._url_import_url:
            return
        self._url_import_url = next_value
        self._dirty_model("url_import_url", "url_import_button_enabled")

    def _cancel_url_import_close_timer(self) -> None:
        if self._url_import_close_timer is None:
            return
        self._url_import_close_timer.cancel()

    def _schedule_close(self, session_id: int) -> None:
        self._cancel_url_import_close_timer()
        self._url_import_close_timer = threading.Timer(
            1.0,
            lambda: self._close_panel_after_success(session_id),
        )
        self._url_import_close_timer.start()

    def _close_panel_after_success(self, session_id: int) -> None:
        if session_id != self._url_import_session_id:
            return
        self._reset_state()
        self._dirty_model()
        refresh_active_panel()
        lf.ui.set_panel_enabled(self.id, False)

    def _on_toggle_formats(self, _handle=None, _ev=None, _args=None):
        self._url_import_formats_open = not self._url_import_formats_open
        self._sync_formats_ui()

    def _sync_formats_ui(self) -> None:
        if self._formats_header is not None:
            self._formats_header.set_class("is-expanded", self._url_import_formats_open)
        if self._formats_arrow is not None:
            self._formats_arrow.set_class("is-expanded", self._url_import_formats_open)
        if self._formats_content is not None:
            self._formats_content.set_class("collapsed", not self._url_import_formats_open)

    def _reset_state(self) -> None:
        self._cancel_url_import_close_timer()
        self._url_import_url = ""
        self._url_import_progress = -1.0
        self._url_import_status = ""
        self._url_import_warning = ""
        self._url_import_in_progress = False
        self._url_import_cancelled = False
        self._url_import_thread = None
        self._url_import_formats_open = False

    def _should_cancel(self, session_id: int) -> bool:
        return self._url_import_cancelled or session_id != self._url_import_session_id

    def _cleanup_destination(self, dest_dir: Optional[Path], created_dest_dir: bool) -> None:
        if not dest_dir or not created_dest_dir:
            return
        try:
            shutil.rmtree(dest_dir, ignore_errors=True)
        except Exception:
            pass

    def _resolve_catalog_context(self, index) -> tuple[Optional[str], Optional[str]]:
        folder_id = None
        scene_id = None

        panel = get_asset_manager_panel()
        if panel is not None:
            candidate_folder = getattr(panel, "_selected_folder_id", None)
            candidate_scene = getattr(panel, "_selected_scene_id", None)
            if candidate_folder and candidate_folder in index.folders:
                folder_id = candidate_folder
            if candidate_scene and candidate_scene in index.scenes:
                scene = index.scenes[candidate_scene]
                scene_folder_id = scene.get("folder_id")
                if folder_id is None or scene_folder_id == folder_id:
                    scene_id = candidate_scene
                    folder_id = folder_id or scene_folder_id

        return folder_id, scene_id

    def _strip_archive_suffix(self, name: str) -> str:
        for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".zip", ".tar"):
            if name.lower().endswith(suffix):
                return name[:-len(suffix)]
        return name

    def _scan_and_register_asset(self, path: str, *, name: Optional[str] = None):
        index = load_asset_index()
        if index is None:
            raise RuntimeError("Asset Manager backend is unavailable")

        scanner = load_scanner()
        thumbnails = load_thumbnails()
        folder_id, scene_id = self._resolve_catalog_context(index)
        metadata_list = _discover_asset_metadata(scanner, path)
        created_assets = _register_discovered_assets(
            index,
            thumbnails,
            metadata_list,
            folder_id=folder_id,
            scene_id=scene_id,
            name_override=self._strip_archive_suffix(name) if name else None,
        )
        if created_assets:
            first_asset = created_assets[0]
            panel = get_asset_manager_panel()
            if panel is not None:
                select_asset_in_active_panel(
                    first_asset.id,
                    folder_id=first_asset.folder_id,
                    scene_id=first_asset.scene_id,
                )
            else:
                refresh_active_panel()
            return first_asset
        return None

    def _handle_download_error(self, title: str, message: str, session_id: int) -> None:
        if session_id != self._url_import_session_id:
            return
        self._url_import_in_progress = False
        self._url_import_progress = -1.0
        self._url_import_status = ""
        self._url_import_warning = f"{title}: {message}"
        self._dirty_model(
            "url_import_show_progress",
            "url_import_show_progress_bar",
            "url_import_progress",
            "url_import_status",
            "url_import_warning_text",
            "url_import_has_warning",
            "url_import_button_enabled",
            "url_import_button_text",
        )

    def _download_url_worker(self, url: str, session_id: int) -> None:
        dest_dir: Optional[Path] = None
        created_dest_dir = False
        try:
            url_info = get_url_info(url)
            asset_name = str(url_info.get("name") or "").strip() or "imported-asset"
            asset_name = Path(asset_name).name or "imported-asset"

            dest_dir = self.STORAGE_PATH / "assets" / asset_name
            created_dest_dir = not dest_dir.exists()
            dest_dir.mkdir(parents=True, exist_ok=True)

            def on_progress(percent: float, status: str):
                if self._should_cancel(session_id):
                    raise InterruptedError("Download cancelled")
                if session_id != self._url_import_session_id:
                    return
                self._url_import_progress = percent
                self._url_import_status = status
                self._dirty_model(
                    "url_import_show_progress",
                    "url_import_show_progress_bar",
                    "url_import_progress",
                    "url_import_status",
                )

            def on_warning(message: str):
                if session_id != self._url_import_session_id:
                    return
                self._url_import_warning = message
                self._dirty_model("url_import_warning_text", "url_import_has_warning")

            downloaded_file = dest_dir / asset_name
            download_url(
                url,
                downloaded_file,
                on_progress=on_progress,
                on_warning=on_warning,
                should_cancel=lambda: self._should_cancel(session_id),
            )

            if self._should_cancel(session_id):
                raise InterruptedError("Download cancelled")

            self._url_import_progress = 0.0
            self._url_import_status = _tr("asset_manager.status_extracting")
            self._dirty_model(
                "url_import_show_progress",
                "url_import_show_progress_bar",
                "url_import_progress",
                "url_import_status",
            )

            extracted = False
            try:
                extract_archive(
                    downloaded_file,
                    dest_dir,
                    on_progress=on_progress,
                    should_cancel=lambda: self._should_cancel(session_id),
                )
                downloaded_file.unlink(missing_ok=True)
                extracted = True
            except ExtractError:
                extracted = False

            discovery_path = str(dest_dir) if extracted else str(downloaded_file)
            self._scan_and_register_asset(discovery_path, name=asset_name)

            if session_id != self._url_import_session_id:
                return
            self._url_import_in_progress = False
            self._url_import_status = _tr("asset_manager.status_complete")
            self._url_import_progress = 1.0
            self._dirty_model(
                "url_import_show_progress",
                "url_import_show_progress_bar",
                "url_import_status",
                "url_import_progress",
                "url_import_button_enabled",
                "url_import_button_text",
            )
            self._schedule_close(session_id)

        except URLDownloadError as exc:
            self._cleanup_destination(dest_dir, created_dest_dir)
            self._handle_download_error(_tr("asset_manager.error_download_failed"), str(exc), session_id)
        except UnsupportedURLError as exc:
            self._cleanup_destination(dest_dir, created_dest_dir)
            self._handle_download_error(_tr("asset_manager.error_unsupported_url"), str(exc), session_id)
        except ExtractError as exc:
            self._cleanup_destination(dest_dir, created_dest_dir)
            self._handle_download_error(_tr("asset_manager.error_extract_failed"), str(exc), session_id)
        except InterruptedError:
            self._cleanup_destination(dest_dir, created_dest_dir)
            if session_id != self._url_import_session_id:
                return
            self._reset_state()
            self._dirty_model()
            lf.ui.set_panel_enabled(self.id, False)
        except Exception as exc:
            self._cleanup_destination(dest_dir, created_dest_dir)
            self._handle_download_error(_tr("asset_manager.error_unknown"), str(exc), session_id)
        finally:
            if session_id == self._url_import_session_id:
                self._url_import_thread = None

    def _on_do_load(self, _handle=None, _ev=None, _args=None):
        if self._url_import_in_progress:
            return

        url = self._url_import_url.strip()
        if not url:
            self._url_import_warning = _tr("asset_manager.error_empty_url")
            self._dirty_model("url_import_warning_text", "url_import_has_warning")
            return

        try:
            normalize_url(url)
        except UnsupportedURLError as exc:
            self._url_import_warning = _tr("asset_manager.error_unsupported_url")
            self._dirty_model("url_import_warning_text", "url_import_has_warning")
            return
        except Exception as exc:
            self._url_import_warning = str(exc)
            self._dirty_model("url_import_warning_text", "url_import_has_warning")
            return

        self._cancel_url_import_close_timer()
        self._url_import_session_id += 1
        self._url_import_warning = ""
        self._url_import_in_progress = True
        self._url_import_cancelled = False
        self._url_import_progress = 0.0
        self._url_import_status = _tr("asset_manager.status_connecting")
        self._dirty_model(
            "url_import_warning_text",
            "url_import_has_warning",
            "url_import_show_progress",
            "url_import_show_progress_bar",
            "url_import_progress",
            "url_import_status",
            "url_import_button_enabled",
            "url_import_button_text",
        )

        self._url_import_thread = threading.Thread(
            target=self._download_url_worker,
            args=(url, self._url_import_session_id),
            daemon=True,
        )
        self._url_import_thread.start()

    def _on_do_cancel(self, _handle=None, _ev=None, _args=None):
        if not self._url_import_in_progress:
            self._reset_state()
            self._dirty_model()
            lf.ui.set_panel_enabled(self.id, False)
            return

        if self._url_import_cancelled:
            return

        self._url_import_cancelled = True
        self._url_import_status = _tr("asset_manager.status_cancelling")
        self._dirty_model(
            "url_import_show_progress",
            "url_import_show_progress_bar",
            "url_import_status",
        )


class WatchDirsDialogPanel(Panel):
    """Floating panel for editing watched directories per folder."""

    id = "lfs.watch_dirs_dialog"
    label = "Watched Directories"
    space = lf.ui.PanelSpace.FLOATING
    order = 14
    template = "rmlui/watch_dirs_dialog.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (520, 0)
    update_policy = "dirty"

    def __init__(self):
        global _watch_dirs_dialog_panel
        _watch_dirs_dialog_panel = self

        self._handle = None
        self._folder_id: Optional[str] = None
        self._folder_name: str = ""
        self._watch_dirs: list[str] = []
        self._shared_state_version: int = -1
        self._last_lang: str = ""
        self._scan_thread: Optional[threading.Thread] = None
        self._scan_cancel_event = threading.Event()
        self._scan_state_lock = threading.Lock()
        self._scan_active = False
        self._scan_progress = 0.0
        self._scan_status = ""
        self._scan_log: list[dict[str, str]] = []
        self._scan_terminal = False
        self._scan_model_update_scheduled = False

    def _sync_from_shared_state(self) -> None:
        self._folder_id = _watch_dirs_dialog_state.get("folder_id")
        self._folder_name = str(_watch_dirs_dialog_state.get("folder_name") or "")
        self._watch_dirs = list(_watch_dirs_dialog_state.get("watch_dirs") or [])
        self._shared_state_version = int(_watch_dirs_dialog_state.get("version") or 0)

    def _publish_shared_state(self) -> None:
        _set_watch_dirs_dialog_state(self._folder_id, self._folder_name, self._watch_dirs)
        self._shared_state_version = int(_watch_dirs_dialog_state.get("version") or 0)

    def on_mount(self, doc):
        super().on_mount(doc)
        self._last_lang = lf.ui.get_current_language()
        self._subscribe_reactive_state()

    def on_unmount(self, doc):
        self._unsubscribe_reactive_state()
        self._scan_cancel_event.set()
        scan_thread = self._scan_thread
        self._scan_thread = None
        _join_thread(scan_thread, "AssetManagerWatch scan")
        self._handle = None
        doc.remove_data_model("watch_dirs_dialog")

    def _subscribe_reactive_state(self):
        if getattr(self, "_reactive_unsubscribers", None):
            return

        self._reactive_unsubscribers = [
            RuntimeState.language_generation.subscribe(lambda _value: self._request_reactive_update()),
        ]

    def _unsubscribe_reactive_state(self):
        for unsubscribe in getattr(self, "_reactive_unsubscribers", []):
            try:
                unsubscribe()
            except Exception:
                pass
        self._reactive_unsubscribers = []

    def _request_reactive_update(self):
        if self._handle:
            w.request_model_update(self._handle)

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("watch_dirs_dialog")
        if model is None:
            return

        model.bind_func("panel_label", self._panel_label)
        model.bind_record_list("watch_dirs_list")
        model.bind_record_list("scan_log_list")
        model.bind_func("no_watch_dirs", lambda: len(self._watch_dirs) == 0)
        model.bind_func("scan_active", self._get_scan_active)
        model.bind_func("scan_status_visible", self._get_scan_status_visible)
        model.bind_func("scan_status_text", self._get_scan_status_text)
        model.bind_func("scan_progress_value", self._get_scan_progress_value)
        model.bind_func("scan_progress_pct", self._get_scan_progress_pct)
        model.bind_func("scan_save_enabled", self._get_scan_save_enabled)
        model.bind_func("scan_save_label", self._get_scan_save_label)
        model.bind_event("on_browse_add", self._on_browse_add)
        model.bind_event("on_remove_dir", self._on_remove_dir)
        model.bind_event("on_cancel", self._on_cancel)
        model.bind_event("on_save", self._on_save)
        self._handle = model.get_handle()
        self._sync_from_shared_state()
        if self._handle:
            self._handle.update_record_list(
                "watch_dirs_list", [{"path": p} for p in self._watch_dirs]
            )
            self._handle.update_record_list("scan_log_list", self._get_scan_log())
            self._handle.dirty_all()

    def on_update(self, doc):
        del doc
        if self._shared_state_version != int(_watch_dirs_dialog_state.get("version") or 0):
            self._sync_from_shared_state()
            self._dirty_model()
            return True
        current_lang = lf.ui.get_current_language()
        if current_lang != self._last_lang:
            self._last_lang = current_lang
            self._dirty_model()
            return True
        return False

    def _catalog_index(self):
        index = load_asset_index()
        if index is None:
            _watch_log("error", "catalog load failed")
            return None

        return index

    def _scanner(self):
        panel = get_asset_manager_panel()
        scanner = getattr(panel, "_asset_scanner", None) if panel is not None else None
        if scanner is not None:
            return scanner
        scanner = load_scanner()
        _watch_log(
            "warn" if scanner is not None else "error",
            "using fallback scanner" if scanner is not None else "scanner unavailable",
        )
        return scanner

    def _thumbnails(self):
        panel = get_asset_manager_panel()
        thumbnails = (
            getattr(panel, "_asset_thumbnails", None) if panel is not None else None
        )
        if thumbnails is not None:
            return thumbnails
        thumbnails = load_thumbnails()
        _watch_log(
            "warn" if thumbnails is not None else "error",
            "using fallback thumbnails"
            if thumbnails is not None
            else "thumbnail backend unavailable",
        )
        return thumbnails

    def show(self, folder_id: str) -> bool:
        try:
            if not _load_watch_dirs_dialog_state(folder_id):
                return False
            self._sync_from_shared_state()
            if not self._get_scan_active():
                self._set_scan_state(
                    active=False,
                    progress=0.0,
                    status="",
                    terminal=False,
                    reset_log=True,
                )
            lf.ui.set_panel_enabled(self.id, True)
            self._dirty_model()
            return True
        except Exception as e:
            _watch_log("error", "failed to show watch dirs dialog: %s", e, exc_info=True)
            return False

    def _panel_label(self) -> str:
        return f"Watched Directories — {self._folder_name}"

    def _get_scan_active(self) -> bool:
        with self._scan_state_lock:
            return self._scan_active

    def _get_scan_status_visible(self) -> bool:
        with self._scan_state_lock:
            return self._scan_active or bool(self._scan_status)

    def _get_scan_status_text(self) -> str:
        with self._scan_state_lock:
            return self._scan_status

    def _get_scan_progress_value(self) -> str:
        with self._scan_state_lock:
            return f"{self._scan_progress:.4f}".rstrip("0").rstrip(".") or "0"

    def _get_scan_progress_pct(self) -> str:
        with self._scan_state_lock:
            return f"{int(round(self._scan_progress * 100.0))}%"

    def _get_scan_save_enabled(self) -> bool:
        with self._scan_state_lock:
            return self._scan_terminal or ((not self._scan_active) and bool(self._watch_dirs))

    def _get_scan_save_label(self) -> str:
        with self._scan_state_lock:
            if self._scan_active:
                return "Scanning..."
            if self._scan_terminal:
                return "Done"
            return "Save & Scan"

    def _get_scan_terminal(self) -> bool:
        with self._scan_state_lock:
            return self._scan_terminal

    def _get_scan_log(self) -> list[dict[str, str]]:
        with self._scan_state_lock:
            return [dict(entry) for entry in self._scan_log]

    def _dirty_model(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.update_record_list(
                "watch_dirs_list", [{"path": p} for p in self._watch_dirs]
            )
            self._handle.update_record_list("scan_log_list", self._get_scan_log())
            self._handle.dirty_all()
            return
        if "watch_dirs_list" in fields or not fields:
            self._handle.update_record_list(
                "watch_dirs_list", [{"path": p} for p in self._watch_dirs]
            )
        if "scan_log_list" in fields or not fields:
            self._handle.update_record_list("scan_log_list", self._get_scan_log())
        for field in fields:
            self._handle.dirty(field)

    def _dirty_scan_model(self) -> None:
        self._dirty_model(
            "scan_active",
            "scan_status_visible",
            "scan_status_text",
            "scan_progress_value",
            "scan_progress_pct",
            "scan_save_enabled",
            "scan_save_label",
            "scan_log_list",
        )

    def _queue_dirty_scan_model(self) -> None:
        if not self._handle:
            return
        with self._scan_state_lock:
            if self._scan_model_update_scheduled:
                return
            self._scan_model_update_scheduled = True

        def _run_update() -> None:
            with self._scan_state_lock:
                self._scan_model_update_scheduled = False
            self._dirty_scan_model()

        scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
        if not callable(scheduler):
            scheduler = getattr(lf.ui, "_run_on_ui_thread", None)

        if callable(scheduler):
            try:
                scheduler(_run_update)
                return
            except Exception:
                pass

        if threading.current_thread() is threading.main_thread():
            _run_update()
            return

        # Older hosts may not expose a UI-thread scheduler; progress callbacks
        # are throttled before reaching this fallback.
        _run_update()

    def _set_scan_state(
        self,
        *,
        active: Optional[bool] = None,
        progress: Optional[float] = None,
        status: Optional[str] = None,
        terminal: Optional[bool] = None,
        reset_log: bool = False,
    ) -> None:
        with self._scan_state_lock:
            if active is not None:
                self._scan_active = active
            if progress is not None:
                self._scan_progress = max(0.0, min(1.0, float(progress)))
            if status is not None:
                self._scan_status = status
            if terminal is not None:
                self._scan_terminal = terminal
            if reset_log:
                self._scan_log = []
        self._queue_dirty_scan_model()

    def _set_scan_log_entry(self, index: int, path: str, status: str) -> None:
        with self._scan_state_lock:
            while len(self._scan_log) <= index:
                self._scan_log.append({"status": "Queued", "path": ""})
            self._scan_log[index] = {"status": status, "path": path}
        self._queue_dirty_scan_model()

    def _clear_scan_thread(self) -> None:
        if self._scan_thread is threading.current_thread():
            self._scan_thread = None

    def _on_browse_add(self, _handle=None, _ev=None, _args=None):
        if self._get_scan_active():
            return
        self._sync_from_shared_state()
        path = lf.ui.open_dataset_folder_dialog()
        if not path:
            return
        if path in self._watch_dirs:
            _watch_log("info", "browse add ignored duplicate path=%s", path)
            return
        self._watch_dirs.append(path)
        self._publish_shared_state()
        self._set_scan_state(
            active=False,
            progress=0.0,
            status="",
            terminal=False,
            reset_log=True,
        )
        self._dirty_model("watch_dirs_list", "no_watch_dirs", "scan_save_enabled")

    def _on_remove_dir(self, _handle=None, _ev=None, args=None):
        if self._get_scan_active():
            return
        self._sync_from_shared_state()
        if not args:
            return
        try:
            idx = int(args[0])
        except (ValueError, TypeError):
            return
        if 0 <= idx < len(self._watch_dirs):
            self._watch_dirs.pop(idx)
            self._publish_shared_state()
            self._set_scan_state(
                active=False,
                progress=0.0,
                status="",
                terminal=False,
                reset_log=True,
            )
            self._dirty_model("watch_dirs_list", "no_watch_dirs", "scan_save_enabled")

    def _on_cancel(self, _handle=None, _ev=None, _args=None):
        if self._get_scan_active():
            self._scan_cancel_event.set()
            self._set_scan_state(status="Cancelling scan...")
            return
        lf.ui.set_panel_enabled(self.id, False)

    def _on_save(self, _handle=None, _ev=None, _args=None):
        if self._get_scan_terminal():
            lf.ui.set_panel_enabled(self.id, False)
            return
        if self._get_scan_active():
            return
        self._sync_from_shared_state()
        if not self._watch_dirs:
            self._set_scan_state(
                active=False,
                progress=0.0,
                status="No watched directories to scan.",
                terminal=False,
                reset_log=True,
            )
            return
        index = self._catalog_index()
        if index is None or self._folder_id is None:
            _watch_log(
                "error",
                "save aborted index_is_none=%s folder_id=%s",
                index is None,
                self._folder_id,
            )
            lf.ui.set_panel_enabled(self.id, False)
            return

        folder = index.get_folder(self._folder_id)
        if folder is None:
            _watch_log(
                "error",
                "save aborted: folder missing folder_id=%s available=%s library=%s",
                self._folder_id,
                list(getattr(index, "folders", {}).keys()),
                _index_library_path(index),
            )
            return

        if not index.set_watch_dirs(self._folder_id, self._watch_dirs):
            _watch_log(
                "error",
                "persist failed folder_id=%s library=%s mtime=%s",
                self._folder_id,
                _index_library_path(index),
                _library_mtime(index),
            )
            return

        refresh_active_panel()

        scanner = self._scanner()
        self._scan_cancel_event = threading.Event()
        watch_dirs = list(self._watch_dirs)
        total_dirs = len(watch_dirs)
        self._set_scan_state(
            active=True,
            progress=0.0,
            status=f"Preparing scan for {total_dirs} folder{'s' if total_dirs != 1 else ''}...",
            terminal=False,
            reset_log=True,
        )
        for idx, path in enumerate(watch_dirs):
            self._set_scan_log_entry(idx, path, "Queued")
        thread = threading.Thread(
            target=self._scan_worker,
            args=(
                index,
                scanner,
                self._folder_id,
                watch_dirs,
                self._scan_cancel_event,
            ),
            daemon=True,
            name="AssetManagerWatchScan",
        )
        self._scan_thread = thread
        thread.start()

    def _scan_worker(
        self,
        index,
        scanner,
        folder_id: str,
        watch_dirs: list[str],
        cancel_event: Optional[threading.Event] = None,
    ):
        """Background thread: scan watched directories and import new assets."""
        if index is None:
            index = load_asset_index()
        if index is None or scanner is None:
            _watch_log(
                "error",
                "scan worker aborted index_is_none=%s scanner_is_none=%s",
                index is None,
                scanner is None,
            )
            self._set_scan_state(
                active=False,
                progress=0.0,
                status="Scan failed: Asset Manager backend is unavailable.",
                terminal=True,
            )
            self._clear_scan_thread()
            return

        added = 0
        discovered = 0
        total_dirs = max(1, len(watch_dirs))
        for path_index, path in enumerate(watch_dirs):
            if cancel_event is not None and cancel_event.is_set():
                _watch_log(
                    "info",
                    "scan worker cancelled before path=%s folder_id=%s",
                    path,
                    folder_id,
                )
                self._set_scan_log_entry(path_index, path, "Cancelled")
                self._set_scan_state(
                    active=False,
                    progress=path_index / total_dirs,
                    status="Scan cancelled.",
                    terminal=True,
                )
                self._clear_scan_thread()
                return
            try:
                self._set_scan_log_entry(path_index, path, "Scanning")
                self._set_scan_state(
                    active=True,
                    progress=path_index / total_dirs,
                    status=f"Scanning {path}",
                )
                progress_base = path_index / total_dirs
                progress_span = 1.0 / total_dirs
                last_progress_update = 0.0

                def _can_update_progress(*, force: bool = False) -> bool:
                    nonlocal last_progress_update
                    now = time.monotonic()
                    if not force and now - last_progress_update < WATCH_UI_UPDATE_INTERVAL_SEC:
                        return False
                    last_progress_update = now
                    return True

                def _on_discovery_progress(
                    current_path: str,
                    scanned_dirs: int,
                    scanned_files: int,
                    found_count: int,
                ) -> None:
                    if not _can_update_progress():
                        return
                    walk_units = scanned_dirs + (scanned_files // 64)
                    inner_progress = min(0.95, walk_units / (walk_units + 256.0))
                    self._set_scan_log_entry(
                        path_index,
                        path,
                        f"Scanning ({found_count} found)",
                    )
                    self._set_scan_state(
                        active=True,
                        progress=progress_base + progress_span * inner_progress,
                        status=(
                            f"Scanning {current_path} "
                            f"({scanned_dirs:,} folders, {found_count:,} assets found)"
                        ),
                    )

                metadata_list = _discover_asset_metadata(
                    scanner,
                    path,
                    fast_walk=True,
                    should_cancel=(
                        lambda: cancel_event is not None and cancel_event.is_set()
                    ),
                    on_progress=_on_discovery_progress,
                )
                discovered += len(metadata_list)
                if not metadata_list:
                    _watch_log("info", "no importable assets found path=%s", path)
                    self._set_scan_log_entry(path_index, path, "No assets")
                    self._set_scan_state(
                        active=True,
                        progress=(path_index + 1) / total_dirs,
                        status=f"No importable assets found in {path}",
                    )
                    continue
                self._set_scan_log_entry(
                    path_index,
                    path,
                    f"Registering ({len(metadata_list)} found)",
                )
                self._set_scan_state(
                    active=True,
                    progress=progress_base + progress_span * 0.95,
                    status=f"Registering {len(metadata_list):,} assets from {path}",
                )

                def _on_register_progress(
                    processed_count: int,
                    total_count: int,
                    created_count: int,
                    skipped_count: int,
                ) -> None:
                    if not _can_update_progress(
                        force=processed_count >= total_count
                    ):
                        return
                    total = max(1, total_count)
                    register_progress = min(1.0, processed_count / total)
                    self._set_scan_log_entry(
                        path_index,
                        path,
                        f"Registering ({created_count} new, {skipped_count} skipped)",
                    )
                    self._set_scan_state(
                        active=True,
                        progress=(
                            progress_base
                            + progress_span * (0.95 + 0.05 * register_progress)
                        ),
                        status=(
                            f"Registering {processed_count:,}/{total_count:,} "
                            f"assets from {path}"
                        ),
                    )

                created_assets = _register_discovered_assets(
                    index,
                    None,
                    metadata_list,
                    folder_id=folder_id,
                    should_cancel=(
                        lambda: cancel_event is not None and cancel_event.is_set()
                    ),
                    on_progress=_on_register_progress,
                    log_each_asset=False,
                    save_immediately=False,
                )
                added += len(created_assets)
                if created_assets:
                    status = f"Added {len(created_assets)} of {len(metadata_list)}"
                else:
                    status = f"Found {len(metadata_list)}, already cataloged"
                self._set_scan_log_entry(path_index, path, status)
                self._set_scan_state(
                    active=True,
                    progress=(path_index + 1) / total_dirs,
                    status=f"{status}: {path}",
                )
            except InterruptedError:
                _watch_log(
                    "info",
                    "scan worker cancelled while scanning path=%s folder_id=%s",
                    path,
                    folder_id,
                )
                self._set_scan_log_entry(path_index, path, "Cancelled")
                self._set_scan_state(
                    active=False,
                    progress=path_index / total_dirs,
                    status="Scan cancelled.",
                    terminal=True,
                )
                self._clear_scan_thread()
                return
            except Exception as e:
                _watch_log(
                    "warn",
                    "failed to scan watched directory path=%s error=%s",
                    path,
                    e,
                    exc_info=True,
                )
                self._set_scan_log_entry(path_index, path, "Failed")
                self._set_scan_state(
                    active=True,
                    progress=(path_index + 1) / total_dirs,
                    status=f"Failed to scan {path}: {e}",
                )

        try:
            if cancel_event is not None and cancel_event.is_set():
                _watch_log(
                    "info",
                    "scan worker cancelled before final save folder_id=%s discovered=%d added=%d",
                    folder_id,
                    discovered,
                    added,
                )
                self._set_scan_state(
                    active=False,
                    progress=1.0,
                    status=f"Scan cancelled. Found {discovered}, added {added}.",
                    terminal=True,
                )
                return
            if added > 0:
                if index.save():
                    _watch_log(
                        "info",
                        "scan save complete discovered=%d added=%d library=%s mtime=%s",
                        discovered,
                        added,
                        _index_library_path(index),
                        _library_mtime(index),
                    )
                    self._set_scan_state(
                        active=False,
                        progress=1.0,
                        status=f"Scan complete. Found {discovered}, added {added}.",
                        terminal=True,
                    )
                else:
                    _watch_log(
                        "error",
                        "scan save failed discovered=%d added=%d library=%s mtime=%s",
                        discovered,
                        added,
                        _index_library_path(index),
                        _library_mtime(index),
                    )
                    self._set_scan_state(
                        active=False,
                        progress=1.0,
                        status=f"Scan finished, but saving failed. Found {discovered}, added {added}.",
                        terminal=True,
                    )
            else:
                _watch_log(
                    "info",
                    "scan complete with no new assets discovered=%d added=%d library=%s",
                    discovered,
                    added,
                    _index_library_path(index),
                )
                self._set_scan_state(
                    active=False,
                    progress=1.0,
                    status=f"Scan complete. Found {discovered}, no new assets added.",
                    terminal=True,
                )
            refresh_active_panel()
        except Exception as e:
            _watch_log("error", "failed to finalize watch-dir scan: %s", e, exc_info=True)
            self._set_scan_state(
                active=False,
                progress=1.0,
                status=f"Failed to finish scan: {e}",
                terminal=True,
            )
        finally:
            self._clear_scan_thread()
