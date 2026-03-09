# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Retained RmlUI panels for dataset and checkpoint import flows."""

from pathlib import Path

import lichtfeld as lf

from .types import RmlPanel


_dataset_import_panel = None
_resume_checkpoint_panel = None

KI_RETURN = 72
KI_ESCAPE = 81


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


class _ImportDialogPanel(RmlPanel):
    """Common behavior for retained import dialogs."""

    update_interval_ms = 200

    def on_load(self, doc):
        super().on_load(doc)
        doc.add_event_listener("keydown", self._on_keydown)

    def _on_keydown(self, event):
        key = int(event.get_parameter("key_identifier", "0"))
        if key == KI_RETURN:
            if self._can_submit_from_keyboard():
                self._on_do_load()
                event.stop_propagation()
        elif key == KI_ESCAPE:
            self._on_do_cancel()
            event.stop_propagation()

    def _can_submit_from_keyboard(self) -> bool:
        return False


class DatasetImportPanel(_ImportDialogPanel):
    """Floating panel for configuring dataset import paths."""

    idname = "lfs.dataset_import"
    label = "Load Dataset"
    space = "FLOATING"
    order = 11
    rml_template = "rmlui/dataset_import_panel.rml"
    rml_height_mode = "content"
    initial_width = 560

    def __init__(self):
        global _dataset_import_panel
        _dataset_import_panel = self

        self._handle = None
        self._dataset_info = None
        self._output_path = ""
        self._init_path = ""
        self._last_lang = ""

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("dataset_import")
        if model is None:
            return

        tr = lf.ui.tr

        model.bind_func("panel_label", lambda: tr("load_dataset_popup.title"))
        model.bind_func("configure_paths", lambda: tr("load_dataset_popup.configure_paths"))
        model.bind_func("images_dir_label", lambda: tr("load_dataset_popup.images_dir"))
        model.bind_func("sparse_dir_label", lambda: tr("load_dataset_popup.sparse_dir"))
        model.bind_func("masks_dir_label", lambda: tr("load_dataset_popup.masks_dir"))
        model.bind_func("output_dir_label", lambda: tr("load_dataset_popup.output_dir"))
        model.bind_func("init_file_label", lambda: tr("load_dataset_popup.init_file"))
        model.bind_func("help_text", lambda: tr("load_dataset_popup.help_text"))
        model.bind_func("browse_label", lambda: tr("common.browse"))
        model.bind_func("load_label", lambda: tr("common.load"))
        model.bind_func("cancel_label", lambda: tr("common.cancel"))

        model.bind_func("images_path", lambda: self._string_attr("images_path"))
        model.bind_func("sparse_path", lambda: self._string_attr("sparse_path"))
        model.bind_func("masks_path", lambda: self._string_attr("masks_path"))
        model.bind_func("images_count_text", self._images_count_text)
        model.bind_func("mask_count_text", self._mask_count_text)
        model.bind_func("show_masks", lambda: bool(self._dataset_info and getattr(self._dataset_info, "has_masks", False)))
        model.bind_func("can_load", lambda: bool(self._dataset_info and self._output_path.strip()))

        model.bind("output_path", lambda: self._output_path, self._set_output_path)
        model.bind("init_path", lambda: self._init_path, self._set_init_path)

        model.bind_event("browse_output", self._on_browse_output)
        model.bind_event("browse_init", self._on_browse_init)
        model.bind_event("do_load", self._on_do_load)
        model.bind_event("do_cancel", self._on_do_cancel)

        self._handle = model.get_handle()

    def on_update(self, doc):
        del doc
        current_lang = lf.ui.get_current_language()
        if current_lang == self._last_lang:
            return False
        self._last_lang = current_lang
        self._dirty_model()
        return True

    def show(self, dataset_path: str) -> bool:
        info = lf.detect_dataset_info(dataset_path)
        if not info:
            return False

        self._dataset_info = info
        self._output_path = str(Path(info.base_path) / "output")
        self._init_path = ""
        self._last_lang = lf.ui.get_current_language()
        self._dirty_model()
        lf.ui.set_panel_enabled(self.idname, True)
        return True

    def _can_submit_from_keyboard(self) -> bool:
        return bool(self._dataset_info and self._output_path.strip())

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

    def _set_init_path(self, value):
        next_value = str(value)
        if next_value == self._init_path:
            return
        self._init_path = next_value
        self._dirty_model("init_path")

    def _on_browse_output(self, _handle=None, _ev=None, _args=None):
        path = lf.ui.open_dataset_folder_dialog()
        if path:
            self._set_output_path(path)

    def _on_browse_init(self, _handle=None, _ev=None, _args=None):
        if self._dataset_info is None:
            return
        path = lf.ui.open_ply_file_dialog(str(self._dataset_info.base_path))
        if path:
            self._set_init_path(path)

    def _on_do_load(self, _handle=None, _ev=None, _args=None):
        if self._dataset_info is None or not self._output_path.strip():
            return

        base_path = str(self._dataset_info.base_path)
        init_path = self._init_path.strip()

        lf.ui.set_panel_enabled(self.idname, False)
        lf.load_file(
            base_path,
            is_dataset=True,
            output_path=self._output_path.strip(),
            init_path=init_path,
        )

    def _on_do_cancel(self, _handle=None, _ev=None, _args=None):
        lf.ui.set_panel_enabled(self.idname, False)


class ResumeCheckpointPanel(_ImportDialogPanel):
    """Floating panel for configuring checkpoint resume paths."""

    idname = "lfs.resume_checkpoint"
    label = "Resume Checkpoint"
    space = "FLOATING"
    order = 12
    rml_template = "rmlui/resume_checkpoint_panel.rml"
    rml_height_mode = "content"
    initial_width = 580

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

        tr = lf.ui.tr

        model.bind_func("panel_label", lambda: tr("resume_checkpoint_popup.title"))
        model.bind_func("checkpoint_badge", lambda: tr("resume_checkpoint_popup.checkpoint"))
        model.bind_func("configure_paths", lambda: tr("resume_checkpoint_popup.configure_paths"))
        model.bind_func("file_label", lambda: tr("resume_checkpoint_popup.file"))
        model.bind_func("stored_path_label", lambda: tr("resume_checkpoint_popup.stored_path"))
        model.bind_func("not_found_label", lambda: tr("resume_checkpoint_popup.not_found"))
        model.bind_func("dataset_path_label", lambda: tr("resume_checkpoint_popup.dataset_path"))
        model.bind_func("output_path_label", lambda: tr("resume_checkpoint_popup.output_path"))
        model.bind_func("help_text", lambda: tr("resume_checkpoint_popup.help_text"))
        model.bind_func("browse_label", lambda: tr("common.browse"))
        model.bind_func("load_label", lambda: tr("common.load"))
        model.bind_func("cancel_label", lambda: tr("common.cancel"))
        model.bind_func("ok_label", lambda: tr("common.ok"))
        model.bind_func("invalid_label", lambda: tr("resume_checkpoint_popup.invalid"))

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
        if current_lang == self._last_lang:
            return False
        self._last_lang = current_lang
        self._dirty_model()
        return True

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
        self._last_lang = lf.ui.get_current_language()
        self._dirty_model()
        lf.ui.set_panel_enabled(self.idname, True)
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
        tr = lf.ui.tr
        return tr("common.ok") if self._dataset_valid else tr("resume_checkpoint_popup.invalid")

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

        lf.ui.set_panel_enabled(self.idname, False)
        lf.load_checkpoint_for_training(
            self._checkpoint_path,
            self._dataset_path,
            self._output_path,
        )

    def _on_do_cancel(self, _handle=None, _ev=None, _args=None):
        lf.ui.set_panel_enabled(self.idname, False)
