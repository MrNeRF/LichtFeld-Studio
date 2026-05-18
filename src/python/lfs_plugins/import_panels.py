# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Retained RmlUI panels for dataset, checkpoint, and URL import flows."""

import shutil
import threading
from pathlib import Path
from typing import Any, Optional

import lichtfeld as lf

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
from .types import Panel
from .rml_keys import KI_ESCAPE, KI_RETURN
from .url_downloader import (
    URLDownloadError,
    UnsupportedURLError,
    ExtractError,
    normalize_url,
    download_url,
    extract_archive,
    get_url_info,
    is_archive_url,
)


_dataset_import_panel = None
_resume_checkpoint_panel = None
_url_import_panel = None

__lfs_panel_classes__ = [
    "DatasetImportPanel",
    "ResumeCheckpointPanel",
    "URLImportPanel",
]
__lfs_panel_ids__ = [
    "lfs.dataset_import",
    "lfs.resume_checkpoint",
    "lfs.url_import",
]


def open_dataset_import_panel(dataset_path: str) -> bool:
    """Open the retained dataset import dialog for the given dataset path."""
    if _dataset_import_panel is None:
        return False
    return _dataset_import_panel.show(dataset_path)


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


def _tr(key: str) -> str:
    fallbacks = {
        "asset_manager.import_from_url": "Import from URL",
        "asset_manager.import_button": "Import",
        "asset_manager.import_button_downloading": "Downloading...",
        "asset_manager.status_connecting": "Connecting...",
        "asset_manager.status_extracting": "Extracting...",
        "asset_manager.status_complete": "Complete!",
        "asset_manager.status_cancelling": "Cancelling...",
        "asset_manager.error_empty_url": "Please enter a URL",
        "asset_manager.error_unsupported_url": "Unsupported URL format",
        "asset_manager.error_download_failed": "Download failed",
        "asset_manager.error_extract_failed": "Extraction failed",
        "asset_manager.error_unknown": "An error occurred",
    }
    result = lf.ui.tr(key)
    if not result or result == key:
        return fallbacks.get(key, key)
    return result


class _ImportDialogPanel(Panel):
    """Common behavior for retained import dialogs."""

    update_interval_ms = 200
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

    def show(self, dataset_path: str) -> bool:
        if not self._apply_dataset_path(dataset_path, reset_output=True):
            return False

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

    STORAGE_PATH = Path.home() / ".lichtfeld" / "asset_manager"

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
        # Cancel any in-progress download before unmounting
        if self._url_import_in_progress and not self._url_import_cancelled:
            self._url_import_cancelled = True
        self._cancel_url_import_close_timer()
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
        self._url_import_close_timer = None

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

    def _make_import_metadata(self, managed_root: Path) -> dict[str, Any]:
        return {
            "kind": "url_download",
            "managed_root": str(managed_root.resolve()),
        }

    def _resolve_catalog_context(self, index) -> tuple[Optional[str], Optional[str]]:
        project_id = None
        scene_id = None

        panel = get_asset_manager_panel()
        if panel is not None:
            candidate_project = getattr(panel, "_selected_project_id", None)
            candidate_scene = getattr(panel, "_selected_scene_id", None)
            if candidate_project and candidate_project in index.projects:
                project_id = candidate_project
            if candidate_scene and candidate_scene in index.scenes:
                scene = index.scenes[candidate_scene]
                if project_id is None or scene.project_id == project_id:
                    scene_id = candidate_scene
                    project_id = project_id or scene.project_id

        if project_id is None:
            project = index.find_or_create_project("Default")
            project_id = project.id if project is not None else None

        return project_id, scene_id

    def _generate_thumbnail(self, index, asset, thumbnails) -> None:
        if thumbnails is None or asset is None:
            return
        try:
            thumb_path = thumbnails.generate_placeholder(asset.type, asset.id)
            index.update_asset(asset.id, thumbnail_path=str(thumb_path))
        except Exception:
            pass

    def _strip_archive_suffix(self, name: str) -> str:
        for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".zip", ".tar"):
            if name.lower().endswith(suffix):
                return name[:-len(suffix)]
        return name

    def _scan_and_register_asset(
        self,
        path: str,
        *,
        fallback_role: str,
        override_type: Optional[str],
        override_role: Optional[str],
        import_metadata: dict[str, Any],
        name: Optional[str] = None,
    ):
        index = load_asset_index()
        if index is None:
            raise RuntimeError("Asset Manager backend is unavailable")

        scanner = load_scanner()
        thumbnails = load_thumbnails()
        project_id, scene_id = self._resolve_catalog_context(index)

        metadata = scanner.scan_file(path) if scanner is not None else {}
        asset_kwargs = metadata_to_asset_kwargs(metadata)
        asset_type = (
            override_type
            or metadata.get("type")
            or Path(path).suffix.lstrip(".").lower()
            or "unknown"
        )
        role = override_role or metadata.get("role") or fallback_role

        asset_name = self._strip_archive_suffix(name) if name else Path(path).name

        asset = index.create_asset(
            project_id=project_id,
            name=asset_name,
            type=asset_type,
            path=path,
            absolute_path=path,
            scene_id=scene_id,
            role=role,
            import_metadata=import_metadata,
            **asset_kwargs,
        )
        if asset is not None:
            self._generate_thumbnail(index, asset, thumbnails)
            panel = get_asset_manager_panel()
            if panel is not None:
                select_asset_in_active_panel(
                    asset.id,
                    project_id=asset.project_id,
                    scene_id=asset.scene_id,
                )
            else:
                refresh_active_panel()
        return asset

    def _register_extracted_asset(self, extract_dir: Path, name: Optional[str] = None):
        import_metadata = self._make_import_metadata(extract_dir)

        for ext in (".ply", ".sog", ".spz", ".rad"):
            files = list(extract_dir.rglob(f"*{ext}"))
            if files:
                return self._scan_and_register_asset(
                    str(files[0]),
                    fallback_role="source_dataset",
                    override_type="dataset",
                    override_role=None,
                    import_metadata=import_metadata,
                    name=name,
                )

        cameras_bin = list(extract_dir.rglob("cameras.bin"))
        if cameras_bin:
            dataset_dir = cameras_bin[0].parent.parent
            return self._scan_and_register_asset(
                str(dataset_dir),
                fallback_role="source_dataset",
                override_type="dataset",
                override_role=None,
                import_metadata=import_metadata,
                name=name,
            )

        return self._scan_and_register_asset(
            str(extract_dir),
            fallback_role="source_dataset",
            override_type="dataset",
            override_role=None,
            import_metadata=import_metadata,
            name=name,
        )

    def _register_downloaded_file(self, file_path: Path, name: Optional[str] = None):
        return self._scan_and_register_asset(
            str(file_path),
            fallback_role="source_dataset",
            override_type="dataset",
            override_role=None,
            import_metadata=self._make_import_metadata(file_path.parent),
            name=name,
        )

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

            if is_archive_url(url):
                temp_file = dest_dir / f"download_{asset_name}.tmp"
                download_url(
                    url,
                    temp_file,
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

                extract_archive(
                    temp_file,
                    dest_dir,
                    on_progress=on_progress,
                    should_cancel=lambda: self._should_cancel(session_id),
                )
                temp_file.unlink(missing_ok=True)
                self._register_extracted_asset(dest_dir, name=asset_name)
            else:
                dest_file = dest_dir / asset_name
                download_url(
                    url,
                    dest_file,
                    on_progress=on_progress,
                    on_warning=on_warning,
                    should_cancel=lambda: self._should_cancel(session_id),
                )

                if self._should_cancel(session_id):
                    raise InterruptedError("Download cancelled")

                self._register_downloaded_file(dest_file, name=asset_name)

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
