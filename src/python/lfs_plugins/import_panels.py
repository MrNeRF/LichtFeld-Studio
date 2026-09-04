# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""RmlUI panels for dataset and checkpoint import flows."""

import threading
import time
from pathlib import Path

import lichtfeld as lf

from . import rml_widgets as w
from .rml_keys import KI_ESCAPE, KI_RETURN
from .training_confirm import confirm_discard_work_then
from .types import Panel
from .panels import panel_class
from .ui import RuntimeState

_new_project_panel = None
_resume_checkpoint_panel = None
_SOURCE_PROBE_DEBOUNCE_SECONDS = 0.3

__lfs_panel_classes__ = ["NewProjectPanel", "ResumeCheckpointPanel"]
__lfs_panel_ids__ = ["lfs.new_project", "lfs.resume_checkpoint"]


def open_new_project_panel(source_path: str = "") -> bool:
    panel = _new_project_panel or lf.ui.get_panel_object("lfs.new_project")
    if panel is None:
        return False
    return panel.show(source_path)


def open_resume_checkpoint_panel(checkpoint_path: str) -> bool:
    panel = _resume_checkpoint_panel or lf.ui.get_panel_object("lfs.resume_checkpoint")
    if panel is None:
        return False
    return panel.show(checkpoint_path)


def _directory_has_colmap_file(path: str) -> bool:
    """Return whether a sparse directory contains a COLMAP model."""
    filenames = (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "cameras.txt",
        "images.txt",
        "points3D.txt",
    )
    root = Path(path)
    return any(
        (candidate / filename).is_file()
        for candidate in (root, root / "0")
        for filename in filenames
    )


class _ImportDialogPanel(Panel):
    """Common behavior for retained import dialogs."""

    update_policy = "dirty"
    form_id = ""

    def on_mount(self, doc):
        super().on_mount(doc)
        if hasattr(self, "_dialog_mounted"):
            self._dialog_mounted = True
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
        if hasattr(self, "_dialog_mounted"):
            self._dialog_mounted = False
            self._source_generation += 1
            self._source_probe_update_generation += 1
            self._source_probe_cancel.set()
            self._source_probe_active = False
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


@panel_class("new_project")
class NewProjectPanel(_ImportDialogPanel):
    """Floating panel for creating a project and optionally importing a source."""

    form_id = "new-project-form"

    DEFAULT_MAX_WIDTH = 3840
    _INVALID_NAME_CHARACTERS = set('<>:"|?*')
    _WINDOWS_RESERVED_NAMES = {
        "con", "prn", "aux", "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }

    def __init__(self):
        global _new_project_panel
        _new_project_panel = self

        self._handle = None
        self._name = ""
        self._source_path = ""
        self._source_kind = "blank"
        self._dataset_info = None
        self._init_path = ""
        self._ppisp_sidecar_path = ""
        self._centralize_dataset = "off"
        self._max_width = self.DEFAULT_MAX_WIDTH
        self._max_width_str = str(self.DEFAULT_MAX_WIDTH)
        self._min_track_length = 0
        self._min_track_length_str = "0"
        self._apply_auto_crop = False
        self._embed_dataset = False
        self._advanced_expanded = False
        self._last_lang = ""
        self._source_generation = 0
        self._source_probe_due = 0.0
        self._source_probe_update_generation = 0
        self._source_probe_active = False
        self._source_probe_cancel = threading.Event()
        self._dialog_mounted = False
        self._colmap_available = False
        self._target_exists_cached = False
        self._dedupe_name_cache: dict[str, str] = {}
        self._name_derived_from_source = False

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("new_project")
        if model is None:
            return

        model.bind_func("panel_label", lambda: lf.ui.tr("new_project.title"))
        model.bind_func("source_is_dataset", lambda: self._source_kind == "dataset")
        model.bind_func("embed_dataset_visible", lambda: self._source_kind == "dataset")
        model.bind_func("advanced_expanded", lambda: self._advanced_expanded)
        model.bind_func("name_valid", self._name_is_valid)
        model.bind_func("target_exists", self._target_exists)
        model.bind_func("can_create", self._can_create)
        model.bind_func("location_preview", self._location_preview)
        model.bind_func("create_hint", self._create_hint)

        model.bind_func("images_path", lambda: self._string_attr("images_path"))
        model.bind_func("sparse_path", lambda: self._string_attr("sparse_path"))
        model.bind_func("masks_path", lambda: self._string_attr("masks_path"))
        model.bind_func("images_count_text", self._images_count_text)
        model.bind_func("mask_count_text", self._mask_count_text)
        model.bind_func("show_masks", lambda: bool(self._dataset_info and getattr(self._dataset_info, "has_masks", False)))
        model.bind_func("show_min_track_length", self._show_min_track_length)
        model.bind_func("show_min_track_length_warning", self._show_min_track_length_warning)
        model.bind("name", lambda: self._name, self._set_name)
        model.bind("source_path", lambda: self._source_path, self._set_source_path)

        model.bind("init_path", lambda: self._init_path, self._set_init_path)
        model.bind("ppisp_sidecar_path", lambda: self._ppisp_sidecar_path, self._set_ppisp_sidecar_path)
        model.bind("centralize_dataset", lambda: self._centralize_dataset, self._set_centralize_dataset)
        model.bind("max_width_str", lambda: self._max_width_str, self._set_max_width_str)
        model.bind("max_width_disabled", lambda: self._max_width == 0, self._set_max_width_disabled)
        model.bind("min_track_length_str", lambda: self._min_track_length_str, self._set_min_track_length_str)
        model.bind("apply_auto_crop", lambda: self._apply_auto_crop, self._set_apply_auto_crop)
        model.bind("embed_dataset", lambda: self._embed_dataset, self._set_embed_dataset)

        model.bind_event("browse_folder", self._on_browse_folder)
        model.bind_event("browse_file", self._on_browse_file)
        model.bind_event("browse_init", self._on_browse_init)
        model.bind_event("browse_ppisp_sidecar", self._on_browse_ppisp_sidecar)
        model.bind_event("min_track_length_step", self._on_min_track_length_step)
        model.bind_event("toggle_advanced", self._on_toggle_advanced)
        model.bind_event("do_create", self._on_do_create)
        model.bind_event("do_cancel", self._on_do_cancel)

        self._handle = model.get_handle()

    def on_update(self, doc):
        del doc
        changed = False
        current_lang = lf.ui.get_current_language()
        if current_lang != self._last_lang:
            self._last_lang = current_lang
            self._dirty_model()
            changed = True
        if self._source_probe_due and time.monotonic() >= self._source_probe_due:
            self._source_probe_due = 0.0
            self._start_source_probe()
            changed = True
        return changed

    def show(self, source_path: str = "") -> bool:
        self._name = ""
        self._source_path = ""
        self._source_kind = "blank"
        self._dataset_info = None
        self._init_path = ""
        self._centralize_dataset = "off"
        self._max_width = self.DEFAULT_MAX_WIDTH
        self._max_width_str = str(self.DEFAULT_MAX_WIDTH)
        self._min_track_length = 0
        self._min_track_length_str = "0"
        self._apply_auto_crop = False
        self._embed_dataset = bool(getattr(lf.ui, "get_embed_dataset_by_default", lambda: False)())
        self._advanced_expanded = False
        self._source_generation += 1
        self._source_probe_cancel.set()
        self._source_probe_cancel = threading.Event()
        self._source_probe_due = 0.0
        self._colmap_available = False
        self._target_exists_cached = False
        self._dedupe_name_cache = {}
        self._name_derived_from_source = False
        params = lf.optimization_params()
        self._ppisp_sidecar_path = ""
        self._set_source_path(source_path, derive_name=True)
        self._start_source_probe()
        self._ppisp_sidecar_path = (
            str(params.ppisp_sidecar_path) if params and params.has_params() else ""
        )
        self._dirty_model("ppisp_sidecar_path")
        if not self._name:
            self._set_name(self._dedupe_name("untitled"))
        self._dirty_model()
        lf.ui.set_panel_enabled(self.id, True)
        return True

    def _can_submit_from_keyboard(self) -> bool:
        return self._can_create()

    def _preview_base_path(self, dataset_path: str) -> Path:
        path = Path(dataset_path)
        if path.suffix.lower() == ".json":
            return path.parent
        return path

    def _apply_source_path(self, source_path: str) -> None:
        next_value = str(source_path).strip()
        self._source_path = next_value
        self._dataset_info = None
        if not next_value:
            self._source_kind = "blank"
        elif self._is_splat_path(next_value):
            self._source_kind = "splat"
        else:
            self._source_kind = "checking"
            self._source_probe_due = time.monotonic() + _SOURCE_PROBE_DEBOUNCE_SECONDS
            self._schedule_source_probe_update()

        self._dirty_model(
            "source_path",
            "source_is_dataset",
            "embed_dataset_visible",
            "images_path",
            "sparse_path",
            "masks_path",
            "images_count_text",
            "mask_count_text",
            "show_masks",
            "show_min_track_length",
            "show_min_track_length_warning",
            "init_path",
            "ppisp_sidecar_path",
            "can_create",
            "target_exists",
            "location_preview",
            "create_hint",
            "min_track_length_str",
            "embed_dataset_visible",
        )

    @staticmethod
    def _is_splat_path(path: str) -> bool:
        suffix = Path(path).suffix.lower()
        return suffix in {".ply", ".sog", ".spz", ".rad"} or suffix.startswith(".usd")

    def _set_source_path(self, value, derive_name=False):
        previous = self._source_path
        self._apply_source_path(value)
        if derive_name or (not self._name and str(value).strip()):
            source = Path(str(value).strip())
            stem = source.name if self._source_kind == "dataset" else source.stem
            if stem:
                self._set_name(self._dedupe_name(stem))
                self._name_derived_from_source = True
        if previous != self._source_path:
            self._source_probe_active = False
            self._source_generation += 1
            self._source_probe_cancel.set()
            self._source_probe_cancel = threading.Event()
            self._init_path = ""
            self._ppisp_sidecar_path = ""

    def _start_source_probe(self) -> None:
        if self._source_probe_active:
            return
        self._source_probe_due = 0.0
        self._source_probe_active = True
        path = self._source_path
        name = self._name
        location = str(getattr(lf.ui, "get_project_location", lambda: "")() or "")
        generation = self._source_generation
        cancel = self._source_probe_cancel

        def worker() -> None:
            kind = "blank" if not path else "invalid"
            info = None
            colmap = False
            try:
                if cancel.is_set():
                    if cancel is self._source_probe_cancel:
                        self._source_probe_active = False
                    return
                if path and lf.is_dataset_path(path):
                    kind = "dataset"
                    info = lf.detect_dataset_info(str(self._preview_base_path(path)))
                    colmap = bool(info and getattr(info, "sparse_path", "") and
                                  _directory_has_colmap_file(str(info.sparse_path)))
                elif path and Path(path).is_file() and self._is_splat_path(path):
                    kind = "splat"
            except Exception:
                kind = "invalid"
            base = name.strip() or "untitled"
            candidate = Path(location) / f"{base}.licht"
            target_exists = candidate.exists()
            deduped = base
            suffix = 2
            while (Path(location) / f"{deduped}.licht").exists():
                deduped = f"{base}-{suffix}"
                suffix += 1
            result = (kind, info, colmap, target_exists, base, deduped)

            def complete() -> None:
                if generation != self._source_generation or cancel.is_set():
                    return
                if not self._dialog_mounted:
                    self._source_probe_active = False
                    self._source_probe_due = time.monotonic() + 0.3
                    return
                self._source_probe_active = False
                (
                    self._source_kind,
                    self._dataset_info,
                    self._colmap_available,
                    target_exists,
                    base,
                    deduped,
                ) = result
                # A name edit can race the source probe; never publish the
                # old name's filesystem result over the current name.
                if self._name == name:
                    self._target_exists_cached = target_exists
                    self._dedupe_name_cache[base] = deduped
                    if self._name_derived_from_source and deduped != base:
                        self._name = deduped
                        self._target_exists_cached = False
                self._dirty_model()

            scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
            if callable(scheduler):
                scheduler(complete)
            else:
                complete()

        scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
        if callable(scheduler):
            threading.Thread(
                target=worker, daemon=True, name="ImportDatasetProbe"
            ).start()
        else:
            worker()

    def _refresh_target_cache(self) -> None:
        location = str(getattr(lf.ui, "get_project_location", lambda: "")() or "")
        base = self._name.strip() or "untitled"
        candidate = Path(location) / f"{base}.licht"
        self._target_exists_cached = candidate.exists()
        deduped = base
        suffix = 2
        while (Path(location) / f"{deduped}.licht").exists():
            deduped = f"{base}-{suffix}"
            suffix += 1
        self._dedupe_name_cache[base] = deduped

    def _set_name(self, value):
        self._name = str(value)
        self._name_derived_from_source = False
        scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
        if callable(scheduler):
            self._target_exists_cached = False
            self._source_probe_due = time.monotonic() + _SOURCE_PROBE_DEBOUNCE_SECONDS
            self._schedule_source_probe_update()
        else:
            self._refresh_target_cache()
        self._dirty_model("name", "name_valid", "target_exists", "can_create", "location_preview", "create_hint")

    def _schedule_source_probe_update(self) -> None:
        self._source_probe_update_generation += 1
        generation = self._source_probe_update_generation
        due = self._source_probe_due
        delay = max(0.0, due - time.monotonic())

        def fire() -> None:
            if (
                generation != self._source_probe_update_generation
                or self._source_probe_due != due
            ):
                return
            scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
            if callable(scheduler):
                scheduler(self._request_reactive_update)
            else:
                self._request_reactive_update()

        timer = threading.Timer(delay, fire)
        timer.daemon = True
        timer.start()

    def _name_is_valid(self) -> bool:
        name = self._name.strip()
        if not name or name[-1] in ". ":
            return False
        if any(character in name for character in "/\\" + "".join(self._INVALID_NAME_CHARACTERS)):
            return False
        return name.split(".", 1)[0].casefold() not in self._WINDOWS_RESERVED_NAMES

    def _target_path(self) -> Path:
        location = str(getattr(lf.ui, "get_project_location", lambda: "")() or "")
        return Path(location) / f"{self._name.strip()}.licht"

    def _target_exists(self) -> bool:
        return bool(self._name.strip()) and self._target_exists_cached

    def _can_create(self) -> bool:
        return (
            self._name_is_valid()
            and not self._target_exists()
            and self._source_kind in {"blank", "dataset", "splat"}
        )

    def _location_preview(self) -> str:
        template = lf.ui.tr("new_project.location_preview") or "Will be created at: {path}"
        return template.replace("{path}", str(self._target_path()))

    def _create_hint(self) -> str:
        if self._target_exists():
            return lf.ui.tr("new_project.already_exists")
        if not self._name_is_valid():
            return lf.ui.tr("new_project.invalid_name")
        if self._source_kind == "invalid":
            return lf.ui.tr("new_project.source_invalid")
        return ""

    def _dedupe_name(self, name: str) -> str:
        base = name.strip() or "untitled"
        return self._dedupe_name_cache.get(base, base)

    def _dirty_model(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            w.request_model_update(self._handle)
            return
        for field in fields:
            self._handle.dirty(field)
        w.request_model_update(self._handle)

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

    def _show_min_track_length(self) -> bool:
        if self._dataset_info is None:
            return False
        sparse_path = getattr(self._dataset_info, "sparse_path", "")
        if not sparse_path:
            return False
        return self._colmap_available

    def _show_min_track_length_warning(self) -> bool:
        return (
            self._show_min_track_length()
            and bool(self._init_path.strip())
            and self._min_track_length > 0
        )

    def _set_init_path(self, value):
        next_value = str(value)
        if next_value == self._init_path:
            return
        self._init_path = next_value
        self._dirty_model("init_path", "show_min_track_length_warning")

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

    def _set_min_track_length_str(self, value):
        text = str(value).strip().replace(",", "")
        try:
            parsed = int(text) if text else 0
        except ValueError:
            self._dirty_model("min_track_length_str")
            return
        self._set_min_track_length(parsed)

    def _set_min_track_length(self, value):
        parsed = max(0, min(99, int(value)))
        if parsed == self._min_track_length and self._min_track_length_str == str(parsed):
            return
        self._min_track_length = parsed
        self._min_track_length_str = str(parsed)
        self._dirty_model("min_track_length_str", "show_min_track_length_warning")

    def _on_min_track_length_step(self, _handle=None, _ev=None, args=None):
        delta = 0
        if args:
            try:
                delta = int(args[0])
            except (TypeError, ValueError, IndexError):
                delta = 0
        self._set_min_track_length(self._min_track_length + delta)

    def _set_apply_auto_crop(self, value):
        enabled = bool(value)
        if enabled == self._apply_auto_crop:
            return
        self._apply_auto_crop = enabled
        self._dirty_model("apply_auto_crop")

    def _set_embed_dataset(self, value):
        value = bool(value)
        if value == self._embed_dataset:
            return
        self._embed_dataset = value
        self._dirty_model("embed_dataset")

    def _on_browse_init(self, _handle=None, _ev=None, _args=None):
        if self._source_kind != "dataset":
            return
        path = lf.ui.open_ply_file_dialog(str(self._preview_base_path(self._source_path)))
        if path:
            self._set_init_path(path)

    def _on_browse_ppisp_sidecar(self, _handle=None, _ev=None, _args=None):
        start_dir = str(self._preview_base_path(self._source_path)) if self._source_path else ""
        if self._ppisp_sidecar_path:
            start_dir = self._ppisp_sidecar_path
        path = lf.ui.open_ppisp_file_dialog(start_dir)
        if path:
            self._set_ppisp_sidecar_path(path)

    def _on_toggle_advanced(self, _handle=None, _ev=None, _args=None):
        self._advanced_expanded = not self._advanced_expanded
        self._dirty_model("advanced_expanded")

    def _on_browse_folder(self, _handle=None, _ev=None, _args=None):
        path = lf.ui.open_dataset_folder_dialog()
        if path:
            self._set_source_path(path)
            self._source_probe_due = 0.0
            self._start_source_probe()

    def _on_browse_file(self, _handle=None, _ev=None, _args=None):
        path = lf.ui.open_ply_file_dialog("")
        if path:
            self._set_source_path(path)
            self._source_probe_due = 0.0
            self._start_source_probe()

    def _on_do_create(self, _handle=None, _ev=None, _args=None):
        if not self._can_create():
            return

        name = self._name.strip()
        source_path = self._source_path.strip()
        source_kind = self._source_kind
        target = self._target_path()
        init_path = self._init_path.strip()
        ppisp_sidecar_path = self._ppisp_sidecar_path.strip()
        centralize_dataset = self._centralize_dataset
        max_width = self._max_width
        apply_auto_crop = self._apply_auto_crop
        min_track_length = self._min_track_length
        embed_dataset = self._embed_dataset

        def _commit(stop_training: bool) -> None:
            if not self._can_create() or self._target_path() != target:
                self._dirty_model()
                return
            params = lf.optimization_params()
            if params and params.has_params():
                params.ppisp_sidecar_path = ppisp_sidecar_path
                params.ppisp_freeze_from_sidecar = bool(ppisp_sidecar_path)
                if ppisp_sidecar_path:
                    params.ppisp = True

            lf.ui.set_panel_enabled(self.id, False)
            lf.project_create(str(target), discard_changes=True, stop_training=stop_training)
            if source_kind == "dataset":
                load_kwargs = {
                    "path": source_path,
                    "is_dataset": True,
                    "init_path": init_path,
                    "centralize_dataset": centralize_dataset,
                    "max_width": max_width,
                    "apply_auto_crop": apply_auto_crop,
                    "min_track_length": min_track_length,
                    "discard_changes": True,
                }
                if stop_training:
                    load_kwargs["stop_training"] = True
                lf.load_file(**load_kwargs)
                if embed_dataset:
                    lf.project_embed_dataset()
            elif source_kind == "splat":
                lf.load_file(path=source_path, is_dataset=False, discard_changes=True)

        confirm_discard_work_then(
            lf.ui.tr("new_project.title"),
            _commit,
        )

    def _on_do_cancel(self, _handle=None, _ev=None, _args=None):
        lf.ui.set_panel_enabled(self.id, False)

    def _on_do_load(self, _handle=None, _ev=None, _args=None):
        self._on_do_create(_handle, _ev, _args)


@panel_class("resume_checkpoint")
class ResumeCheckpointPanel(_ImportDialogPanel):
    """Floating panel for configuring checkpoint resume paths."""

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

        checkpoint_path = self._checkpoint_path
        dataset_path = self._dataset_path
        output_path = self._output_path

        def _commit(stop_training: bool) -> None:
            del stop_training
            lf.ui.set_panel_enabled(self.id, False)
            lf.load_checkpoint_for_training(
                checkpoint_path,
                dataset_path,
                output_path,
            )

        confirm_discard_work_then(
            lf.ui.tr("resume_checkpoint_popup.title"),
            _commit,
        )

    def _on_do_cancel(self, _handle=None, _ev=None, _args=None):
        lf.ui.set_panel_enabled(self.id, False)
