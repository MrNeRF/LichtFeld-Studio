# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Image preview panel using RmlUI floating window."""

import time
import weakref
from math import gcd
from html import escape
from _lfs_path_io import Path
from typing import Optional
from urllib.parse import quote

import lichtfeld as lf
from . import rml_widgets as w
from .types import Panel
from .panels import panel_class
from .preview_geometry import THUMB_ROW_DP, visible_thumbnail_range, view_to_source, source_to_view
from .preview_metadata import PreviewMetadata, SourceChangedError
from .preview_masking import FrameMaskEditor, MaskApply
from .bert_tokenizer import BertTokenizer
from .default_reset import LabelDefaultReset
from .ui import RuntimeState
from .rml_keys import (
    KI_1, KI_ADD, KI_C, KI_DELETE, KI_DOWN, KI_END, KI_ESCAPE, KI_F, KI_HOME, KI_I,
    KI_LEFT, KI_M, KI_OEM_MINUS, KI_OEM_PLUS, KI_R, KI_RIGHT, KI_SPACE,
    KI_RETURN, KI_SUBTRACT, KI_T, KI_UP, native_key,
)

__lfs_panel_classes__ = ["ImagePreviewPanel"]
__lfs_panel_ids__ = ["lfs.image_preview"]


def __lfs_after_reload__(runtime):
    runtime.ui.on_open_camera_preview(open_camera_preview_by_uid)
    runtime.ui.on_open_image_preview(open_extracted_frame_preview, include_sequence=True)


ZOOM_MIN = 0.01
ZOOM_MAX = 10.0
PRECISE_SCROLL_STEP = 32.0
THUMB_MAX_PX = 256
SEMANTIC_CATEGORIES = {
    "person": "person", "car": "car", "bicycle": "bicycle", "motorcycle": "motorcycle",
    "bus": "bus", "truck": "truck", "boat": "boat", "dog": "dog", "cat": "cat",
    "plant": "potted plant", "chair": "chair", "bottle": "bottle",
    "backpack": "backpack", "suitcase": "suitcase", "sky": "sky",
}


def _sam_key(name):
    return "image_preview.sam_" + name


def _sam_confirmation_rml(message):
    paragraphs = [escape(part).replace("\n", "<br/>") for part in message.split("\n\n")]
    return "".join(f'<div class="modal-note">{"<strong>" + part + "</strong>" if i == 0 else part}</div>'
                   for i, part in enumerate(paragraphs))


_instance = None
_pending_open = None
_RML_PATH_SAFE_CHARS = "/:._-~"


def _encode_rml_path(path: Path | str) -> str:
    return quote(str(path), safe=_RML_PATH_SAFE_CHARS)


@panel_class("image_preview")
class ImagePreviewPanel(Panel):

    def __init__(self):
        global _instance, _pending_open
        _instance = self

        self._image_paths: list[Path] = []
        self._mask_paths: list[Optional[Path]] = []
        self._camera_uids: list[int] = []
        self._camera_nodes: dict[int, tuple[int, str]] = {}
        self._unbound_camera_generation: int | None = None
        self._current_index = 0

        self._zoom = 1.0
        self._fit_to_window = True
        self._show_info = True
        self._show_filmstrip = True
        self._show_overlay = False
        self._show_help = False
        self._show_semantic_options = False
        self._sampled_color = None
        self._sample_request = None
        self._mask_editor = FrameMaskEditor(self._sam_record_edit)
        self._default_reset = LabelDefaultReset(self._sam_reset_default, self._sam_apply_reset)
        self._sam_tokenizer = None
        self._sam_semantic_ui_state = None
        self._sam_method = "manual"
        self._sam_texture = None
        self._sam_texture_key = None
        self._sam_texture_source = ""
        self._sam_thumbnail_sources = set()
        self._sam_controls_state = None
        self._sam_session_active = False
        self._sam_apply_target = None
        self._sam_sequence = None
        self._sam_page = "frame"
        self._sam_sequence_revision = -1
        self._sam_frame_summaries = {}
        self._sam_inspect_pending = False
        self._sam_track_first = self._sam_track_last = 0
        self._sam_track_direction = "both"
        self._sam_syncing_sequence = False
        self._sam_progress = (0, 0, 0)
        self._sam_empty_approved = 0
        self._sam_sequence_error = ""
        self._sam_tracking_request = 0
        self._sam_worker_busy = False
        self._sam_markers_rml = None
        self._sam_selection_options = None
        self._sam_direction_options = None
        self._sam_syncing_selection = False
        self._mask_inverted = False
        self._metadata = PreviewMetadata()
        self._metadata_required = False
        self._metadata_validated = set()
        self._action_error = ""
        self._preview_state = "loading"
        self._reset_preview()
        self._refresh_revision = 0
        self._requested_at = 0.0
        self._displayed_index = -1
        self._filmstrip_range = None
        self._ensure_selection_visible = True
        self._filmstrip_scrolling = False
        self._filmstrip_follow_selection = True
        self._filmstrip_viewport_height = None

        self._pan_x = 0.0
        self._pan_y = 0.0
        self._rotation_quadrants = 0
        self._dragging = False
        self._drag_start_x = 0.0
        self._drag_start_y = 0.0
        self._drag_dp_ratio = 1.0
        self._drag_start_pan_x = 0.0
        self._drag_start_pan_y = 0.0
        self._hover_image = False
        self._color_picker_active = False

        self._doc = None
        self._handle = None
        self._dirty = True
        self._layout_show_info = None
        self._layout_show_filmstrip = None
        self._layout_extra = None
        self._layout_pending = False
        self._image_info_cache: dict[str, tuple[int, int, int]] = {}
        self._path_stat_cache: dict[str, tuple[bool, int, int, int, int, int]] = {}
        self._path_revision_cache: dict[str, str] = {}
        self._filmstrip_top_spacer = 0.0
        self._filmstrip_bottom_spacer = 0.0
        self._last_training_params: tuple[int, int, bool, bool] = (1, 0, False, False)
        self._decorator_cache: dict[str, str] = {}
        self._reactive_unsubscribers = []
        self._pending_view_chrome = None
        if _pending_open is not None:
            self.open(*_pending_open)
            _pending_open = None

    def capture_chrome(self):
        return {
            "camera_index": int(self._current_index),
            "zoom": float(self._zoom),
            "fit_to_window": bool(self._fit_to_window),
            "pan_x": float(self._pan_x),
            "pan_y": float(self._pan_y),
            "rotation_quadrants": int(self._rotation_quadrants),
            "show_filmstrip": bool(self._show_filmstrip),
            "show_info": bool(self._show_info),
            "show_overlay": bool(self._show_overlay),
        }

    def apply_chrome(self, payload):
        self._current_index = 0
        self._zoom = 1.0
        self._fit_to_window = True
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._rotation_quadrants = 0
        self._show_filmstrip = True
        self._show_info = True
        self._show_overlay = False
        self._pending_view_chrome = payload if isinstance(payload, dict) else None
        if self._pending_view_chrome:
            self._apply_view_chrome(self._pending_view_chrome)
        self._mark_dirty()

    def _apply_view_chrome(self, payload):
        index = payload.get("camera_index")
        if isinstance(index, (int, float)):
            index = int(index)
            if self._image_paths:
                self._current_index = max(0, min(index, len(self._image_paths) - 1))
            else:
                self._current_index = max(0, index)
        zoom = payload.get("zoom")
        if isinstance(zoom, (int, float)) and zoom > 0:
            self._zoom = float(zoom)
        if "fit_to_window" in payload:
            self._fit_to_window = bool(payload.get("fit_to_window"))
        if isinstance(payload.get("pan_x"), (int, float)):
            self._pan_x = float(payload["pan_x"])
        if isinstance(payload.get("pan_y"), (int, float)):
            self._pan_y = float(payload["pan_y"])
        rotation = payload.get("rotation_quadrants")
        if isinstance(rotation, (int, float)):
            self._rotation_quadrants = int(rotation) % 4
        if "show_filmstrip" in payload:
            self._show_filmstrip = bool(payload.get("show_filmstrip"))
        if "show_info" in payload:
            self._show_info = bool(payload.get("show_info"))
        if "show_overlay" in payload:
            self._show_overlay = bool(payload.get("show_overlay"))

    def _get_title(self) -> str:
        if self._image_paths:
            dirname = self._image_paths[0].parent.name
            return f"{dirname} \u00b7 {self._current_index + 1:,} / {len(self._image_paths):,}"
        return lf.ui.tr("image_preview.title")

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("image_preview")
        if model is None:
            return

        model.bind_func("panel_label", lambda: self._get_title())
        model.bind_record_list("thumbs")
        model.bind_record_list("semantic_categories")
        model.bind("semantic_text", lambda: self._mask_editor.semantic_text,
                   lambda value: self._sam_semantic_edit("semantic_text", str(value)))
        model.bind("semantic_confidence", lambda: self._mask_editor.semantic_confidence,
                   self._sam_set_semantic_confidence)
        model.bind_event("remove_semantic_category", lambda _h, _e, args: self._sam_remove_category(args[0]) if args else None)
        model.bind_func("filmstrip_top_spacer_height", lambda: f"{self._filmstrip_top_spacer:.1f}dp")
        model.bind_func("filmstrip_bottom_spacer_height", lambda: f"{self._filmstrip_bottom_spacer:.1f}dp")
        self._handle = model.get_handle()
        self._filmstrip_range = None

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        self._default_reset.mount(doc)
        self._sam_markers_rml = None
        self._sam_selection_options = None
        self._sam_direction_options = None
        self._sam_semantic_ui_state = None
        self._layout_show_info = None
        self._layout_show_filmstrip = None
        self._reset_preview()
        self._displayed_index = -1
        self._preview_state = "loading" if self._image_paths else "empty"
        self._ensure_selection_visible = True
        self._filmstrip_viewport_height = None

        for eid, handler in [
            ("nav-prev", lambda _ev: self._navigate(-1)),
            ("nav-next", lambda _ev: self._navigate(1)),
            ("btn-copy-path", lambda _ev: self._copy_path_to_clipboard()),
            ("btn-rotate-left", lambda _ev: self._rotate(-1)),
            ("btn-rotate-right", lambda _ev: self._rotate(1)),
            ("btn-go-to-camera", lambda _ev: self._action_go_to_camera_view()),
            ("btn-gt-compare", lambda _ev: self._action_open_in_gt_compare()),
            ("btn-toggle-training", lambda _ev: self._action_toggle_training()),
            ("btn-show-in-folder", lambda _ev: self._action_show_in_file_manager()),
            ("btn-fit", lambda _ev: self._reset_view_and_refresh()),
            ("btn-actual-size", lambda _ev: self._actual_size()),
            ("btn-zoom-in", lambda _ev: self._zoom_in()),
            ("btn-zoom-out", lambda _ev: self._zoom_out()),
            ("btn-info", lambda _ev: self._toggle_chrome("_show_info")),
            ("btn-thumbnails", lambda _ev: self._toggle_chrome("_show_filmstrip")),
            ("btn-help", lambda _ev: self._toggle_chrome("_show_help")),
            ("btn-sam", lambda _ev: self._toggle_sam()),
            ("sam-page-frame", lambda _ev: self._set_sam_page("frame")),
            ("sam-page-sequence", lambda _ev: self._set_sam_page("sequence")),
            ("sam-positive", lambda _ev: self._sam_tool("positive")),
            ("sam-negative", lambda _ev: self._sam_tool("negative")),
            ("sam-box", lambda _ev: self._sam_tool("box")),
            ("sam-pan", lambda _ev: self._sam_tool("pan")),
            ("sam-undo", lambda _ev: (lf.undo.undo(), self._mark_dirty())),
            ("sam-clear", lambda _ev: self._sam_edit("clear")),
            ("sam-remove", lambda _ev: self._sam_edit("remove_selected")),
            ("sam-flip", lambda _ev: self._sam_edit("flip_selected")),
            ("sam-show", lambda _ev: self._sam_show()),
            ("sam-generate", lambda _ev: self._sam_generate()),
            ("sam-semantic-find", lambda _ev: self._sam_generate(semantic=True)),
            ("sam-semantic-advanced", lambda _ev: self._toggle_chrome("_show_semantic_options")),
            ("sam-method-manual", lambda _ev: self._set_sam_method("manual")),
            ("sam-semantic-heading", lambda _ev: self._set_sam_method("semantic")),
            ("sam-refine", lambda _ev: self._sam_generate(True)),
            ("sam-cancel", lambda _ev: self._sam_cancel()),
            ("sam-apply", lambda _ev: self._sam_apply(False)),
            ("sam-exclude", lambda _ev: self._sam_apply(True)),
            ("sam-track", lambda _ev: self._sam_track()),
            ("sam-approve-range", lambda _ev: self._sam_approve_range()),
            ("sam-save", lambda _ev: self._sam_save(False)),
            ("sam-save-exclude", lambda _ev: self._sam_save(True)),
            ("sam-candidate-0", lambda _ev: self._sam_candidate(0)),
            ("sam-candidate-1", lambda _ev: self._sam_candidate(1)),
            ("sam-candidate-2", lambda _ev: self._sam_candidate(2)),
            ("btn-picker", lambda _ev: self._toggle_picker()),
            ("btn-refresh", lambda _ev: self._refresh_sources()),
            ("btn-apply-render-color", lambda _ev: self._apply_sampled_color("render")),
            ("btn-apply-training-color", lambda _ev: self._apply_sampled_color("training")),
        ]:
            el = doc.get_element_by_id(eid)
            if el:
                el.add_event_listener("click", lambda ev, action=handler: self._activate_if_enabled(ev, action))

        cb_mask = doc.get_element_by_id("cb-mask")
        if cb_mask:
            cb_mask.add_event_listener("change", self._on_mask_checkbox_change)
        for eid in ("sam-track-first", "sam-track-last", "sam-track-direction"):
            element = doc.get_element_by_id(eid)
            if element:
                element.add_event_listener("change", self._sam_track_options_changed)
        approved = doc.get_element_by_id("sam-approve")
        if approved:
            approved.add_event_listener("change", lambda _ev: self._sam_approve())
        selection = doc.get_element_by_id("sam-selection")
        if selection:
            selection.add_event_listener("change", self._sam_select_control)
        category = doc.get_element_by_id("sam-semantic-category")
        if category:
            category.add_event_listener("change", self._sam_add_category)

        filmstrip = doc.get_element_by_id("filmstrip")
        if filmstrip:
            filmstrip.add_event_listener("click", self._on_filmstrip_click)
            filmstrip.add_event_listener("mousescroll", self._on_precise_scroll)
            filmstrip.add_event_listener("scroll", self._on_filmstrip_scroll)
            filmstrip.add_event_listener("mousedown", self._on_filmstrip_interaction)

        for element_id in ("sidebar", "sam-controls", "metadata-info", "sam-status"):
            scrollable = doc.get_element_by_id(element_id)
            if scrollable:
                scrollable.add_event_listener("mousescroll", self._on_precise_scroll)

        img_container = doc.get_element_by_id("image-container")
        if img_container:
            img_container.add_event_listener("mousescroll", self._on_wheel)
            img_container.add_event_listener("mousedown", self._on_img_mousedown)
            img_container.add_event_listener("dblclick", lambda event: self._on_img_mousedown(event, double_click=True))
            img_container.add_event_listener("mouseup", self._on_img_mouseup)
            img_container.add_event_listener("mousemove", self._on_img_mousemove)
            img_container.add_event_listener("mouseover", self._on_img_mouseover)
            img_container.add_event_listener("mouseout", self._on_img_mouseout)

        doc.add_event_listener("keydown", self._on_keydown)
        doc.add_event_listener("resize", self._on_layout_resize)
        # The document listener receives bubbling keys once, including native-window content.

        self._decorator_cache = {}
        self._dirty = True
        self._subscribe_reactive_state()
        self._update_picker_cursor()

    def on_update(self, doc):
        previous_state = (self._preview_state, self._preview_load_failed)
        node = self._current_camera_node()
        if node is not None and self._current_index < len(self._mask_paths):
            mask = Path(node.mask_path) if node.mask_path else None
            if mask != self._mask_paths[self._current_index]:
                self._refresh_sources()
        self._service_sample()
        self._service_metadata()
        self._service_sam()
        if self._mask_editor.enabled:
            controls_state = (lf.keymap.bindings_revision(), lf.undo.can_undo())
            if controls_state != self._sam_controls_state:
                self._sam_controls_state = controls_state
                self._dirty = True
        if (not self._color_picker_active and not self._fit_to_window and self._image_paths and self._hover_image
                and (not self._mask_editor.enabled or self._mask_editor.tool == "pan")):
            lf.ui.set_mouse_cursor_hand()

        self._update_filmstrip(doc, bool(self._image_paths))

        current_params = self._get_training_params()
        if current_params != self._last_training_params:
            self._last_training_params = current_params
            self._dirty = True
        inverted = current_params[3]
        if inverted != self._mask_inverted:
            self._mask_inverted = inverted
            self._dirty = True

        needs_redraw = False
        if (self._preview_state == "loading" or self._preview_pending) and not self._dirty:
            self._update_main_image(doc, bool(self._image_paths))
            needs_redraw = True
        if self._layout_pending and not self._dirty:
            self._layout_pending = False
            self._update_main_image(doc, bool(self._image_paths))
            self._update_sam_geometry(doc)
            self._update_status(doc, bool(self._image_paths))
            needs_redraw = True

        if (self._preview_state, self._preview_load_failed) != previous_state:
            self._dirty = True

        if self._dirty:
            self._dirty = False
            self._refresh_ui(doc)
            return True

        return needs_redraw

    def on_unmount(self, doc):
        self._default_reset.unmount()
        self._sam_cancel()
        if self._sam_sequence:
            lf.ui.sam_preview_end_sequence(self._sam_sequence["run"])
            self._sam_sequence = None
        self._sample_request = None
        self._metadata.close()
        self._unsubscribe_reactive_state()
        doc.remove_data_model("image_preview")
        self._sam_release_thumbnails()
        self._handle = None
        self._doc = None
        self._filmstrip_range = None
        self._reset_preview()

    def _subscribe_reactive_state(self):
        if self._reactive_unsubscribers:
            return

        self._reactive_unsubscribers = [
            RuntimeState.scene_generation.subscribe(self._on_scene_generation_changed),
            RuntimeState.language_generation.subscribe(lambda _value: self._mark_dirty()),
        ]

    def _on_scene_generation_changed(self, _value):
        if self._current_camera_uid() >= 0 and self._current_camera_node() is None:
            self._image_paths = []
            self._mask_paths = []
            self._camera_uids = []
            self._camera_nodes.clear()
            self._unbound_camera_generation = None
            self._image_info_cache.clear()
            self._path_stat_cache.clear()
            self._current_index = 0
            self._color_picker_active = False
            self._reset_view()
            self._close_panel()
        self._mark_dirty()

    def _unsubscribe_reactive_state(self):
        for unsubscribe in self._reactive_unsubscribers:
            try:
                unsubscribe()
            except Exception:
                pass
        self._reactive_unsubscribers = []

    def _request_model_update(self):
        if self._handle:
            w.request_model_update(self._handle)

    def _request_async_update(self):
        self._request_model_update()
        lf.ui.request_redraw(0.03)

    def _reset_preview(self):
        self._requested_preview = ""
        self._ready_preview = ""
        self._preview_identity = None
        self._preview_max_width = 0
        self._preview_pending = False
        self._preview_load_failed = False
        self._active_image_layer = "main-image-a"

    def _mark_dirty(self):
        self._dirty = True
        self._request_model_update()

    def open(self, image_paths: list[Path], mask_paths: list[Optional[Path]],
             start_index: int, camera_uids: list[int] | None = None,
             camera_nodes: dict[int, tuple[int, str]] | None = None, assisted: bool = False, weights: str = "", sequence=None):
        if sequence and self._sam_sequence and sequence["run"] == self._sam_sequence["run"]:
            self._mask_editor.enabled = assisted
            self._mark_dirty()
            if assisted:
                lf.ui.set_panel_enabled("native.video_extractor", False)
            return
        self._reset_preview()
        self._sam_cancel()
        if self._sam_sequence and (not sequence or self._sam_sequence["run"] != sequence["run"]):
            lf.ui.sam_preview_end_sequence(self._sam_sequence["run"])
        self._sam_sequence = sequence
        self._sam_page = "frame"
        self._sam_sequence_revision = -1
        self._sam_method = "manual"
        self._sam_frame_summaries = {}
        self._sam_progress = (0, 0, 0)
        self._sam_empty_approved = 0
        self._sam_tracking_request = 0
        self._sam_track_first, self._sam_track_last = 0, max(0, len(image_paths) - 1)
        self._sam_inspect_pending = bool(sequence)
        self._mask_editor = FrameMaskEditor(self._sam_record_edit)
        self._mask_editor.enabled = assisted
        self._mask_editor.weights = weights
        self._sam_semantic_ui_state = None
        if self._handle:
            self._handle.dirty("semantic_text")
            self._handle.dirty("semantic_confidence")
        if assisted:
            self._show_info = False
        if not image_paths:
            return

        self._image_paths = [Path(p) for p in image_paths]
        self._mask_paths = [Path(p) if p else None for p in mask_paths] if mask_paths else [None] * len(image_paths)
        live_paths = {str(p) for p in self._image_paths + [p for p in self._mask_paths if p]}
        self._metadata.reset()
        self._sample_request = None
        self._sampled_color = None
        self._action_error = ""
        self._displayed_index = -1
        self._preview_state = "loading"
        self._color_picker_active = False
        self._metadata_required = True
        self._metadata_validated.clear()
        self._image_info_cache = {p: v for p, v in self._image_info_cache.items() if p in live_paths}
        self._path_stat_cache = {p: v for p, v in self._path_stat_cache.items() if p in live_paths}
        self._path_revision_cache = {p: v for p, v in self._path_revision_cache.items() if p in live_paths}
        self._camera_uids = list(camera_uids) if camera_uids is not None else [-1] * len(image_paths)
        self._camera_nodes = dict(camera_nodes) if camera_nodes is not None else {}
        self._unbound_camera_generation = (
            RuntimeState.scene_generation.value
            if camera_nodes is None and any(uid >= 0 for uid in self._camera_uids)
            else None
        )
        self._filmstrip_range = None
        self._ensure_selection_visible = True
        self._last_training_params = self._get_training_params()
        self._rotation_quadrants = 0
        self._reset_view()
        if self._pending_view_chrome:
            self._apply_view_chrome(self._pending_view_chrome)
            self._pending_view_chrome = None
        # A restored view keeps zoom and toggles; the image the caller asked for still wins.
        self._current_index = max(0, min(start_index, len(image_paths) - 1))
        self._decorator_cache = {}
        self._update_picker_cursor()
        self._mark_dirty()
        # Hide the source only after the review has accepted its frames, including
        # a deferred open during construction. Failures leave extraction available.
        if assisted:
            lf.ui.set_panel_enabled("native.video_extractor", False)

    def _reset_pan(self):
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._dragging = False

    def _reset_view(self):
        self._zoom = 1.0
        self._fit_to_window = True
        self._reset_pan()

    def _refresh_immediately(self):
        self._mark_dirty()

    def _navigate(self, delta: int):
        new_idx = self._current_index + delta
        if 0 <= new_idx < len(self._image_paths):
            self._current_index = new_idx
            self._ensure_selection_visible = True
            self._sampled_color = None
            self._action_error = ""
            self._reset_pan()
            self._refresh_immediately()

    def _go_to_image(self, index: int):
        if 0 <= index < len(self._image_paths):
            self._current_index = index
            self._ensure_selection_visible = True
            self._sampled_color = None
            self._action_error = ""
            self._reset_pan()
            self._refresh_immediately()

    def _toggle_fit(self):
        if self._fit_to_window:
            self._actual_size()
        else:
            self._reset_view_and_refresh()

    def _reset_view_and_refresh(self):
        self._reset_view()
        self._mark_dirty()

    def _actual_size(self):
        self._zoom = 1.0 / max(1.0, lf.ui.get_ui_scale())
        self._fit_to_window = False
        self._reset_pan()
        self._mark_dirty()

    def _toggle_chrome(self, field):
        setattr(self, field, not getattr(self, field))
        self._mark_dirty()

    def _toggle_picker(self):
        if self._mask_editor.enabled:
            return
        if self._image_paths and self._preview_state == "ready" and self._sample_request is None:
            self._color_picker_active = not self._color_picker_active
            self._update_picker_cursor()
            self._mark_dirty()

    def _copy_path_to_clipboard(self):
        if self._image_paths:
            lf.ui.set_clipboard_text(str(self._image_paths[self._current_index]))

    def _current_camera_uid(self) -> int:
        if 0 <= self._current_index < len(self._camera_uids):
            return self._camera_uids[self._current_index]
        return -1

    def _current_camera_node(self, resolve_unbound: bool = False):
        uid = self._current_camera_uid()
        identity = self._camera_nodes.get(uid)
        if not 0 <= self._current_index < len(self._image_paths):
            return None
        scene = lf.get_scene()
        if not scene:
            return None
        if identity is None and resolve_unbound and self._unbound_camera_generation is not None:
            if RuntimeState.scene_generation.value != self._unbound_camera_generation:
                return None
            for candidate in scene.get_nodes():
                if (candidate.type == lf.scene.NodeType.CAMERA
                        and candidate.camera_uid == uid
                        and Path(candidate.image_path) == self._image_paths[self._current_index]):
                    identity = (candidate.id, candidate.uuid)
                    self._camera_nodes[uid] = identity
                    break
        if identity is None:
            return None
        node = scene.get_node_by_id(identity[0])
        if (node and node.uuid == identity[1]
                and node.type == lf.scene.NodeType.CAMERA and node.camera_uid == uid
                and Path(node.image_path) == self._image_paths[self._current_index]):
            return node
        return None

    def _action_go_to_camera_view(self):
        node = self._current_camera_node(resolve_unbound=True)
        if node is not None:
            lf.ui.go_to_camera_view(node.camera_uid)

    def _action_open_in_gt_compare(self):
        node = self._current_camera_node(resolve_unbound=True)
        if node is None:
            return
        lf.ui.go_to_camera_view(node.camera_uid)
        if not lf.ui.is_gt_comparison_active():
            lf.ui.toggle_gt_comparison()

    def _action_toggle_training(self):
        node = self._current_camera_node(resolve_unbound=True)
        if node is None:
            return
        lf.set_camera_training_enabled(node.name, not node.training_enabled)
        self._mark_dirty()

    def _action_show_in_file_manager(self):
        if not self._image_paths:
            return
        lf.ui.reveal_in_file_manager(str(self._image_paths[self._current_index]))

    def _rotate(self, delta_quadrants: int):
        if not self._image_paths:
            return
        self._rotation_quadrants = (self._rotation_quadrants + delta_quadrants) % 4
        self._mark_dirty()

    def _effective_zoom(self):
        if self._image_paths and self._fit_to_window:
            layout = self._get_image_layout(self._image_paths[self._current_index])
            if layout:
                return layout["scale"]
        return self._zoom

    def _zoom_in(self):
        self._zoom = self._clamp_zoom(self._effective_zoom() * 1.25)
        self._fit_to_window = False
        self._mark_dirty()

    def _zoom_out(self):
        self._zoom = self._clamp_zoom(self._effective_zoom() / 1.25)
        self._fit_to_window = False
        self._mark_dirty()

    @staticmethod
    def _clamp_zoom(zoom):
        ratio = max(1.0, lf.ui.get_ui_scale())
        return max(ZOOM_MIN / ratio, min(ZOOM_MAX / ratio, zoom))

    def _has_valid_overlay(self) -> bool:
        if self._current_index >= len(self._mask_paths):
            return False
        mask_path = self._mask_paths[self._current_index]
        return mask_path is not None and self._path_stat_cache.get(str(mask_path), (False, 0, 0, 0, 0, 0))[0]

    def _close_panel(self):
        self._mask_editor.enabled = False
        self._sam_cancel()
        self._sample_request = None
        self._metadata.close()
        lf.ui.set_panel_enabled("lfs.image_preview", False)

    def _get_image_info(self, path: Path) -> tuple[int, int, int]:
        return self._image_info_cache.get(str(path), (0, 0, 0))

    def _service_metadata(self):
        if not self._image_paths or self._sample_request is not None:
            return
        index = self._current_index
        wanted = [(self._image_paths[index], True)]
        if index < len(self._mask_paths) and self._mask_paths[index]:
            wanted.append((self._mask_paths[index], False))
        if self._filmstrip_range:
            lo, hi = self._filmstrip_range[:2]
            wanted.extend((self._image_paths[i], True) for i in range(lo, hi))
        cached = {str(path): (self._path_stat_cache.get(str(path)), self._image_info_cache.get(str(path)))
                  for path, full in wanted if full}
        result, pending = self._metadata.tick(wanted, lf.ui.get_image_info, cached)
        if result:
            path, stat, info = result
            key = str(path)
            previous = self._path_stat_cache.get(key)
            self._path_stat_cache[key] = stat
            self._path_revision_cache[key] = "-".join(str(v) for v in stat[1:]) if stat[0] else "missing"
            if info is not None:
                self._image_info_cache[key] = info
                self._metadata_validated.add(key)
            elif previous != stat:
                self._image_info_cache.pop(key, None)
            self._mark_dirty()
        if pending:
            # Worker never calls RmlUI. A bounded UI poll also works in native windows.
            self._request_async_update()

    @staticmethod
    def _format_aspect_ratio(w: int, h: int) -> str:
        if w <= 0 or h <= 0:
            return ""
        d = gcd(w, h)
        rw, rh = w // d, h // d
        if rw > 30 or rh > 30:
            ratio = w / h
            common = {
                (16, 9), (16, 10), (4, 3), (3, 2), (21, 9),
                (1, 1), (5, 4), (3, 1), (2, 1),
            }
            best = min(common, key=lambda r: abs(r[0] / r[1] - ratio))
            if abs(best[0] / best[1] - ratio) < 0.05:
                return f"{best[0]}:{best[1]}"
            return f"{w / h:.2f}:1"
        return f"{rw}:{rh}"

    def _get_training_params(self) -> tuple[int, int, bool, bool]:
        inverted = False
        try:
            op = lf.optimization_params()
            inverted = bool(op and op.has_params() and op.invert_masks
                            and op.mask_mode != lf.MaskMode.SEGMENT_AND_IGNORE)
            dp = lf.dataset_params()
            if dp.has_params():
                return (dp.resize_factor, dp.max_width, op.undistort, inverted)
            return (1, 0, False, inverted)
        except Exception:
            pass
        return (1, 0, False, inverted)

    def _make_preview_url(self, path: Path, cam_uid: int,
                          thumb: int = 0, preview_mw: int = 0, dimensions=None, mask_overlay=False) -> str:
        rf, mw, ud = self._last_training_params[:3]
        revision = self._path_revision_cache.get(str(path), "missing")
        width, height, _ = dimensions or self._image_info_cache.get(str(path), (0, 0, 0))
        return (f"preview://cam={cam_uid}&thumb={thumb}&w={width}&h={height}"
                f"&rf={rf}&mw={mw}&pmw={preview_mw}"
                f"&ud={1 if ud else 0}&rev={revision}&refresh={self._refresh_revision}"
                f"{'&mode=mask' if mask_overlay else ''}&path={_encode_rml_path(path)}")

    def _on_filmstrip_click(self, event):
        el = event.target()
        while el:
            idx_str = el.get_attribute("data-index", "")
            if idx_str:
                self._go_to_image(int(idx_str))
                return
            el = el.parent()

    def _on_mask_checkbox_change(self, _event):
        cb = self._doc.get_element_by_id("cb-mask") if self._doc else None
        if cb:
            self._show_overlay = cb.has_attribute("checked")
            self._mark_dirty()

    def _on_precise_scroll(self, event):
        scroll_el = event.current_target()
        if not scroll_el:
            return

        try:
            wheel_delta = float(event.get_parameter("wheel_delta_y", "0"))
        except (TypeError, ValueError):
            return

        is_filmstrip = self._doc and (scroll_el.id == "filmstrip"
                                      or scroll_el is self._doc.get_element_by_id("filmstrip"))
        if is_filmstrip:
            self._on_filmstrip_interaction(event)
        max_scroll = max(0.0, scroll_el.scroll_height - scroll_el.client_height)
        if max_scroll <= 0.0:
            if is_filmstrip or scroll_el.id == "sidebar":
                event.stop_propagation()
            return

        new_scroll = min(max(scroll_el.scroll_top + wheel_delta * PRECISE_SCROLL_STEP, 0.0), max_scroll)
        if abs(new_scroll - scroll_el.scroll_top) > 0.01:
            scroll_el.scroll_top = new_scroll

        if is_filmstrip:
            self._update_filmstrip(self._doc, bool(self._image_paths))

        event.stop_propagation()

    def _on_wheel(self, event):
        delta = float(event.get_parameter("wheel_delta_y", "0"))
        if delta == 0:
            return
        event.stop_propagation()
        self._zoom_at_pointer(1.15 if delta > 0 else 1 / 1.15, event)

    def _zoom_at_pointer(self, factor, event):
        if not self._image_paths:
            return
        path = self._image_paths[self._current_index]
        viewport = self._doc.get_element_by_id("image-viewport") if self._doc else None
        old = self._get_image_layout(path, viewport)
        if not old:
            return
        x = (float(event.get_parameter("mouse_x", "0")) - viewport.absolute_left) / old["dp_ratio"]
        y = (float(event.get_parameter("mouse_y", "0")) - viewport.absolute_top) / old["dp_ratio"]
        source = view_to_source(old, self._rotation_quadrants, x, y)
        self._zoom = self._clamp_zoom(old["scale"] * factor)
        self._fit_to_window = False
        new = self._get_image_layout(path, viewport)
        nx, ny = source_to_view(new, self._rotation_quadrants, *source)
        self._pan_x += x - nx
        self._pan_y += y - ny
        self._mark_dirty()

    def _on_img_mousedown(self, event, double_click=False):
        button = int(event.get_parameter("button", "0"))
        target = event.target()
        while target is not None:
            if target.tag_name in ("button", "input", "textarea", "select"):
                return
            target = target.parent()
        if self._mask_editor.enabled:
            keymap = lf.keymap
            if keymap.is_capturing():
                return
            mouse = {0: keymap.MouseButton.LEFT, 1: keymap.MouseButton.RIGHT,
                     2: keymap.MouseButton.MIDDLE}.get(button)
            action = (keymap.get_action_for_mouse_button(keymap.ToolMode.IMAGE_PREVIEW, mouse,
                                                        self._sam_modifiers(event), double_click)
                      if mouse is not None else keymap.Action.NONE)
            if action == keymap.Action.IMAGE_PREVIEW_REMOVE_POINT and self._sam_method == "manual":
                self._mask_editor.selected = self._sam_hit(event)
                self._sam_edit("remove_selected")
                event.stop_propagation()
                return
        if button != 0 or double_click:
            return

        if self._mask_editor.enabled and self._sam_method == "manual" and self._mask_editor.tool != "pan":
            self._sam_pointer(event)
            return
        if self._color_picker_active and self._image_paths:
            self._perform_color_pick(event)
            return

        if self._fit_to_window:
            return
        self._dragging = True
        self._drag_dp_ratio = max(1.0, lf.ui.get_ui_scale())
        self._drag_start_x = float(event.get_parameter("mouse_x", "0"))
        self._drag_start_y = float(event.get_parameter("mouse_y", "0"))
        self._drag_start_pan_x = self._pan_x
        self._drag_start_pan_y = self._pan_y

    def _on_img_mouseup(self, _event):
        if self._mask_editor.box_drag is not None:
            end = self._sam_source_point(_event, clamp=True)
            if end:
                box = self._mask_editor.dragged_box(end)
                self._mask_editor.set_box(box[:2], box[2:])
            self._mask_editor.box_drag = None
            self._mark_dirty()
        if self._mask_editor.point_drag is not None:
            end = self._sam_source_point(_event, clamp=True)
            if end:
                self._mask_editor.move_selected(*end)
            self._mask_editor.point_drag = None
            self._mark_dirty()
        if self._mask_editor.drag_start is not None:
            end = self._sam_source_point(_event, clamp=True)
            if end:
                self._mask_editor.set_box(self._mask_editor.drag_start, end)
            self._mask_editor.drag_start = None
            self._mark_dirty()
        self._dragging = False

    def _on_img_mousemove(self, event):
        if self._mask_editor.box_drag is not None:
            self._mask_editor.drag_end = self._sam_source_point(event, clamp=True)
            self._mark_dirty()
            return
        if self._mask_editor.point_drag is not None:
            end = self._sam_source_point(event, clamp=True)
            if end:
                self._mask_editor.point_drag = end
            self._mark_dirty()
            return
        if self._mask_editor.drag_start is not None:
            self._mask_editor.drag_end = self._sam_source_point(event, clamp=True)
            self._mark_dirty()
            return
        if not self._dragging:
            return
        mx = float(event.get_parameter("mouse_x", "0"))
        my = float(event.get_parameter("mouse_y", "0"))
        self._pan_x = self._drag_start_pan_x + (mx - self._drag_start_x) / self._drag_dp_ratio
        self._pan_y = self._drag_start_pan_y + (my - self._drag_start_y) / self._drag_dp_ratio
        self._mark_dirty()

    def _on_img_mouseover(self, _event):
        self._hover_image = True

    def _on_img_mouseout(self, _event):
        if self._mask_editor.dragging and self._doc:
            viewport = self._doc.get_element_by_id("image-viewport")
            mx = float(_event.get_parameter("mouse_x", "-1"))
            my = float(_event.get_parameter("mouse_y", "-1"))
            if (viewport and viewport.absolute_left <= mx < viewport.absolute_left + viewport.client_width
                    and viewport.absolute_top <= my < viewport.absolute_top + viewport.client_height):
                return
        self._hover_image = False
        self._dragging = False
        if self._mask_editor.dragging:
            self._mark_dirty()
        self._mask_editor.drag_start = self._mask_editor.drag_end = None
        self._mask_editor.point_drag = None
        self._mask_editor.box_drag = None

    def _on_layout_resize(self, _event):
        self._layout_pending = True
        self._mark_dirty()

    def _get_rotation_degrees(self) -> int:
        return (self._rotation_quadrants % 4) * 90

    def _get_image_layout(self, path: Path, viewport=None):
        if viewport is None:
            viewport = self._doc.get_element_by_id("image-viewport") if self._doc else None

        w, h, _ = self._get_image_info(path)
        if not viewport or w <= 0 or h <= 0:
            return None

        dp_ratio = max(1.0, lf.ui.get_ui_scale())
        vw = max(1.0, viewport.client_width / dp_ratio)
        vh = max(1.0, viewport.client_height / dp_ratio)
        rotated_w, rotated_h = (h, w) if self._rotation_quadrants % 2 else (w, h)

        if self._fit_to_window:
            scale = min(1.0, vw / rotated_w, vh / rotated_h)
        else:
            scale = self._zoom

        dw = max(1.0 / dp_ratio, w * scale)
        dh = max(1.0 / dp_ratio, h * scale)

        ox = (vw - dw) * 0.5
        oy = (vh - dh) * 0.5
        if not self._fit_to_window:
            ox += self._pan_x
            oy += self._pan_y

        return {
            "width": w,
            "height": h,
            "dp_ratio": dp_ratio,
            "viewport_width": vw,
            "viewport_height": vh,
            "scale": scale,
            "draw_width": dw,
            "draw_height": dh,
            "offset_x": ox,
            "offset_y": oy,
        }

    def _apply_zoom(self, img_el, path: Path):
        layout = self._get_image_layout(path)
        if not layout:
            return

        dw = layout["draw_width"]
        dh = layout["draw_height"]
        ox = layout["offset_x"]
        oy = layout["offset_y"]

        img_el.set_property("width", f"{dw:.9g}dp")
        img_el.set_property("height", f"{dh:.9g}dp")
        img_el.set_property("max-width", "none")
        img_el.set_property("max-height", "none")
        img_el.set_property("left", f"{ox:.9g}dp")
        img_el.set_property("top", f"{oy:.9g}dp")
        img_el.set_property("transform", f"rotate({self._get_rotation_degrees()}deg)")
        img_el.remove_property("margin-left")
        img_el.remove_property("margin-top")

    def _get_main_preview_max_width(self, doc, path: Path) -> int:
        if not self._fit_to_window:
            return max(self._get_image_info(path)[:2])
        viewport = doc.get_element_by_id("image-viewport") if doc else None
        layout = self._get_image_layout(path, viewport)
        if not layout:
            return 512
        # Fit starts display-sized. Leaving Fit requests the source once, not a new
        # decode for every zoom step. The current source retains its largest request.
        source_max = max(layout["width"], layout["height"])
        target = min(source_max, source_max * layout["scale"] * layout["dp_ratio"] * 1.25)
        return max(512, (int(target) + 255) // 256 * 256)

    def _set_decorator(self, element, cache_key: str, decorator: str):
        if not element:
            self._decorator_cache.pop(cache_key, None)
            return

        if self._decorator_cache.get(cache_key) != decorator:
            element.set_property("decorator", decorator)
            self._decorator_cache[cache_key] = decorator

    # -- UI refresh --

    def _refresh_ui(self, doc):
        has_images = bool(self._image_paths)

        self._update_localized_labels(doc)
        self._update_main_image(doc, has_images)
        self._update_filmstrip(doc, has_images)
        self._update_sidebar(doc, has_images)
        self._update_nav_arrows(doc, has_images)
        self._update_view_controls(doc, has_images)
        self._update_status(doc, has_images)
        self._update_toolbar(doc, has_images)
        self._update_sam(doc)

        extra = (self._show_help, bool(self._sampled_color), self._action_error,
                 RuntimeState.language_generation.value, self._mask_editor.enabled,
                 self._mask_editor.phase, self._mask_editor.error, bool(self._mask_editor.scores),
                 self._mask_editor.can_refine, self._mask_editor.show_proposal,
                 self._show_overlay, self._has_valid_overlay(), self._sam_method)
        if (self._show_info != self._layout_show_info
                or self._show_filmstrip != self._layout_show_filmstrip or extra != self._layout_extra):
            self._layout_show_info = self._show_info
            self._layout_show_filmstrip = self._show_filmstrip
            self._layout_extra = extra
            # RmlUI resolves chrome visibility after this hook. Measure once more
            # using the resulting viewport, without repeating metadata refresh.
            self._layout_pending = True
        if self._layout_pending:
            self._request_model_update()

        if self._handle:
            self._handle.dirty("panel_label")

    def _update_main_image(self, doc, has_images: bool):
        mask = doc.get_element_by_id("mask-overlay")
        message = doc.get_element_by_id("no-image-text")
        self._displayed_index = -1
        self._preview_state = "empty"
        path = self._image_paths[self._current_index] if has_images else None
        dimensions = self._get_image_info(path) if path else (0, 0, 0)
        validated = path is None or not self._metadata_required or str(path) in self._metadata_validated
        if path and validated and dimensions[0] > 0 and dimensions[1] > 0:
            uid = -1 if self._mask_editor.enabled else self._current_camera_uid()
            # All source/geometry/revision parameters except decode resolution.
            identity = self._make_preview_url(path, uid)
            if identity != self._preview_identity:
                self._reset_preview()
                self._preview_identity = identity
                self._sample_request = None
                self._sampled_color = None
            self._preview_max_width = max(self._preview_max_width, self._get_main_preview_max_width(doc, path))
            url = self._make_preview_url(path, uid, preview_mw=self._preview_max_width)
            if url != self._requested_preview:
                self._requested_preview, self._requested_at = url, time.monotonic()
                self._preview_load_failed = False
            target = self._active_image_layer
            if self._ready_preview and url != self._ready_preview:
                target = "main-image-b" if target == "main-image-a" else "main-image-a"
            # The staging layer is rendered behind the ready layer. Hiding it with
            # display:none would prevent RmlUi from requesting its texture.
            for layer_id in ("main-image-a", "main-image-b"):
                layer = doc.get_element_by_id(layer_id)
                if not layer:
                    continue
                active = layer_id == self._active_image_layer and bool(self._ready_preview)
                staging = layer_id == target and not self._preview_load_failed
                source = self._ready_preview if active else url if staging else ""
                layer.set_attribute("class", "image-layer preview-active" if active else
                                    "image-layer" if source else "image-layer hidden")
                self._set_decorator(layer, layer_id, f"image({source})" if source else "none")
                if source:
                    self._apply_zoom(layer, path)
            state = "error" if self._preview_load_failed else lf.ui.preview_texture_status(url, doc)
            if state == "ready":
                self._ready_preview = url
                self._active_image_layer = target
                self._preview_pending = False
                for layer_id in ("main-image-a", "main-image-b"):
                    layer = doc.get_element_by_id(layer_id)
                    if layer:
                        layer.set_attribute("class", "image-layer preview-active" if layer_id == target else "image-layer hidden")
                        if layer_id != target:
                            self._set_decorator(layer, layer_id, "none")
            elif state == "error" or (state == "unknown" and time.monotonic() - self._requested_at > 5):
                if url == self._ready_preview:
                    self._ready_preview = ""
                self._preview_load_failed = True
                self._preview_pending = False
                if target != self._active_image_layer or not self._ready_preview:
                    layer = doc.get_element_by_id(target)
                    if layer:
                        layer.set_attribute("class", "image-layer hidden")
                        self._set_decorator(layer, target, "none")
            else:
                self._preview_pending = True
            self._preview_state = "ready" if self._ready_preview else "error" if self._preview_load_failed else "loading"
            if self._ready_preview:
                self._displayed_index = self._current_index
        else:
            self._reset_preview()
            for layer_id in ("main-image-a", "main-image-b"):
                layer = doc.get_element_by_id(layer_id)
                if layer:
                    layer.set_attribute("class", "image-layer hidden")
                    self._set_decorator(layer, layer_id, "none")
            if path:
                self._preview_state = "error" if validated and str(path) in self._image_info_cache else "loading"
        if self._preview_state == "loading" or self._preview_pending:
            self._request_async_update()
        if message:
            keys = {"empty": "image_preview.no_images_loaded", "loading": "image_preview.loading",
                    "error": "image_preview.load_failed"}
            message.set_attribute("class", "hidden" if self._preview_state == "ready" else "")
            if self._preview_state != "ready":
                label = doc.get_element_by_id("preview-state-text") or message
                label.set_text(lf.ui.tr(keys[self._preview_state]))
        show_mask = (not self._mask_editor.enabled and self._preview_state == "ready"
                     and self._show_overlay and self._has_valid_overlay())
        if mask:
            mask.set_attribute("class", "visible" if show_mask else "")
            if show_mask:
                mask_path = self._mask_paths[self._current_index]
                # The URL is versioned just like the source, avoiding stale masks after replacement.
                mask_url = self._make_preview_url(mask_path, -1, preview_mw=self._preview_max_width, dimensions=dimensions, mask_overlay=True)
                self._set_decorator(mask, "mask-overlay", f"image({mask_url})")
                self._apply_zoom(mask, path)
            else:
                self._set_decorator(mask, "mask-overlay", "none")

    def _on_filmstrip_scroll(self, _event):
        # Layout and programmatic scrolling also emit scroll events. Only input
        # cancels navigation's visibility request, after the extent is published.
        if self._doc and not self._filmstrip_scrolling:
            self._update_filmstrip(self._doc, bool(self._image_paths))

    def _on_filmstrip_interaction(self, _event):
        self._ensure_selection_visible = False
        self._filmstrip_follow_selection = False

    def _update_filmstrip(self, doc, has_images: bool):
        filmstrip = doc.get_element_by_id("filmstrip")
        if not filmstrip or not self._handle:
            return
        filmstrip.set_attribute("class", "" if self._show_filmstrip else "hidden")
        if not self._show_filmstrip:
            self._sam_release_thumbnails()
            self._filmstrip_range = None
            return
        count = len(self._image_paths) if has_images else 0
        ratio = max(1.0, lf.ui.get_ui_scale())
        height = max(1.0, float(filmstrip.client_height))
        if self._ensure_selection_visible:
            self._filmstrip_follow_selection = True
        if (self._filmstrip_follow_selection and self._filmstrip_viewport_height is not None
                and abs(height - self._filmstrip_viewport_height) > 0.5):
            # Translation, wrapping and model layout can resize the viewport
            # after a successful scroll. Follow navigation until the user scrolls.
            self._ensure_selection_visible = True
        self._filmstrip_viewport_height = height
        top = max(0.0, float(filmstrip.scroll_top))
        actual_top = top
        item_top = self._current_index * THUMB_ROW_DP * ratio
        item_bottom = item_top + THUMB_ROW_DP * ratio
        if self._ensure_selection_visible and count:
            if item_top < top:
                top = item_top
            elif item_bottom > top + height:
                top = item_bottom - height
        lo, hi = visible_thumbnail_range(count, top, height, ratio)
        key = (lo, hi, self._current_index,
               tuple((self._path_revision_cache.get(str(self._image_paths[i]), "missing"),
                      self._image_info_cache.get(str(self._image_paths[i]))) for i in range(lo, hi)),
               self._refresh_revision, self._sam_sequence_revision if self._sam_sequence else None,
               self._mask_editor.enabled, self._mask_editor.show_proposal)
        if key == self._filmstrip_range:
            if self._ensure_selection_visible and count:
                # The unchanged model has had a chance to lay out its spacers.
                self._filmstrip_scrolling = True
                try:
                    filmstrip.scroll_top = top
                finally:
                    self._filmstrip_scrolling = False
                actual_top = max(0.0, float(filmstrip.scroll_top))
                self._ensure_selection_visible = not (
                    actual_top <= item_top + 1.0 and item_bottom <= actual_top + height + 1.0)
                if self._ensure_selection_visible:
                    self._request_model_update()
            return
        self._filmstrip_range = key
        self._filmstrip_top_spacer = lo * THUMB_ROW_DP
        self._filmstrip_bottom_spacer = max(0, count - hi) * THUMB_ROW_DP
        records = []
        mask_sources = set()
        for i in range(lo, hi):
            path = self._image_paths[i]
            uid = self._camera_uids[i] if i < len(self._camera_uids) else -1
            name = path.name + (" · " + self._sam_frame_status(i) if self._sam_sequence else "")
            frame = self._sam_frame_summaries.get(i, {})
            mask_decorator = "none"
            if self._sam_sequence and self._mask_editor.enabled and self._mask_editor.show_proposal and frame.get("revision"):
                width, height, _ = self._image_info_cache.get(str(path), (0, 0, 0))
                source = (f"preview://memory=sam:{self._sam_sequence['run']}:{i}:{frame['revision']}"
                          f"&thumb={THUMB_MAX_PX}&w={width}&h={height}&mode=mask")
                mask_sources.add(source)
                mask_decorator = f"image({source} contain)"
            badge = " ✓" if frame.get("approved") else " !" if frame.get("revision") and not frame.get("reliable") else " ●" if frame.get("revision") else ""
            records.append({"index": i, "label": f"{i + 1:,}" + badge, "name": name,
                            "selected": i == self._current_index,
                            "mask_decorator": mask_decorator,
                            "decorator": f"image({self._make_preview_url(path, uid, THUMB_MAX_PX)} contain)"})
        self._handle.update_record_list("thumbs", records)
        self._sam_release_thumbnails(mask_sources)
        self._handle.dirty("filmstrip_top_spacer_height")
        self._handle.dirty("filmstrip_bottom_spacer_height")
        if self._ensure_selection_visible and count:
            self._ensure_selection_visible = not (
                actual_top <= item_top + 1.0 and item_bottom <= actual_top + height + 1.0)
            if self._ensure_selection_visible:
                self._request_model_update()

    def _sam_release_thumbnails(self, keep=()):
        keep = set(keep)
        for source in self._sam_thumbnail_sources - keep:
            lf.ui.release_rml_texture(source)
        self._sam_thumbnail_sources = keep

    def _update_nav_arrows(self, doc, has_images: bool):
        prev_el = doc.get_element_by_id("nav-prev")
        next_el = doc.get_element_by_id("nav-next")

        if not has_images or len(self._image_paths) <= 1:
            if prev_el:
                prev_el.set_attribute("class", "nav-arrow hidden")
            if next_el:
                next_el.set_attribute("class", "nav-arrow hidden")
            return

        if prev_el:
            cls = "nav-arrow hidden" if self._current_index == 0 else "nav-arrow"
            prev_el.set_attribute("class", cls)
        if next_el:
            cls = "nav-arrow hidden" if self._current_index >= len(self._image_paths) - 1 else "nav-arrow"
            next_el.set_attribute("class", cls)

    def _update_localized_labels(self, doc):
        tr = lf.ui.tr

        for element_id, text in {
            "meta-width-label": tr("image_preview.width_label"),
            "meta-height-label": tr("image_preview.height_label"),
            "meta-megapixels-label": tr("image_preview.megapixels_label"),
            "meta-aspect-label": tr("image_preview.aspect_label"),
            "meta-channels-label": tr("image_preview.channels_label"),
            "meta-format-label": tr("image_preview.format_label"),
            "meta-size-label": tr("image_preview.size_label"),
            "meta-path-label": tr("image_preview.path_label"),
            "hk-navigate": tr("image_preview.navigate"),
            "hk-fit": tr("image_preview.fit"),
            "hk-zoom": tr("image_preview.zoom"),
            "hk-info": tr("image_preview.info"),
            "hk-thumbnails": tr("image_preview.thumbnails"),
            "hk-mask": tr("image_preview.mask"),
            "hk-reset": tr("common.reset"),
            "hk-close": tr("common.close"),
            "hk-color-picker": tr("image_preview.color_picker"),
        }.items():
            _set_text(doc, element_id, text)

        copy_btn = doc.get_element_by_id("btn-copy-path")
        if copy_btn:
            copy_btn.set_attribute("title", tr("image_preview.copy_full_path"))
        rotate_left_btn = doc.get_element_by_id("btn-rotate-left")
        if rotate_left_btn:
            rotate_left_btn.set_attribute("title", tr("image_preview.rotate_left"))
        rotate_right_btn = doc.get_element_by_id("btn-rotate-right")
        if rotate_right_btn:
            rotate_right_btn.set_attribute("title", tr("image_preview.rotate_right"))

        node = self._current_camera_node()
        for eid, key in (
            ("btn-go-to-camera", "scene.go_to_camera_view"),
            ("btn-gt-compare", "scene.open_in_gt_compare"),
            ("btn-show-in-folder", "scene.show_in_file_manager"),
        ):
            el = doc.get_element_by_id(eid)
            if el:
                el.set_attribute("title", tr(key))

        train_btn = doc.get_element_by_id("btn-toggle-training")
        if train_btn:
            train_key = (
                "scene.disable_for_training"
                if node and node.training_enabled
                else "scene.enable_for_training"
            )
            train_btn.set_attribute("title", tr(train_key))
        train_icon = doc.get_element_by_id("btn-toggle-training-icon")
        if train_icon:
            icon_src = (
                "../icon/scene/unlocked.png"
                if node and node.training_enabled
                else "../icon/scene/locked.png"
            )
            train_icon.set_attribute("src", icon_src)

    def _update_sidebar(self, doc, has_images: bool):
        sidebar = doc.get_element_by_id("sidebar")
        if not sidebar:
            return

        if not self._show_info and not self._mask_editor.enabled:
            sidebar.set_attribute("class", "hidden")
            return
        sidebar.set_attribute("class", "sam-sidebar" if self._mask_editor.enabled else "")
        metadata = doc.get_element_by_id("metadata-info")
        if metadata:
            metadata.set_attribute("class", "" if self._show_info else "hidden")

        tr = lf.ui.tr

        _set_text(doc, "sidebar-file-header", tr("image_preview.file_section"))
        _set_text(doc, "sidebar-image-label", tr("image_preview.image_section"))
        _set_text(doc, "sidebar-storage-label", tr("image_preview.storage_section"))

        if has_images:
            path = self._image_paths[self._current_index]
            ext = path.suffix[1:].upper() if path.suffix else "?"
            w, h, c = self._get_image_info(path)

            _set_text(doc, "sidebar-filename", path.name)

            if w > 0 and h > 0:
                _set_text(doc, "sidebar-width", str(w))
                _set_text(doc, "sidebar-height", str(h))
                mp = (w * h) / 1_000_000
                _set_text(doc, "sidebar-mp", f"{mp:.1f} MP")
                _set_text(doc, "sidebar-aspect", self._format_aspect_ratio(w, h))
            else:
                for eid in ("sidebar-width", "sidebar-height", "sidebar-mp", "sidebar-aspect"):
                    _set_text(doc, eid, "—")

            _set_text(doc, "sidebar-channels", self._get_channel_label(c))

            _set_text(doc, "sidebar-format", ext)
            stat = self._path_stat_cache.get(str(path))
            _set_text(doc, "sidebar-size", self._format_size(stat[2]) if stat and stat[0] else "—")

            full_path = str(path)
            display_dir = str(path.parent)
            if len(display_dir) > 30:
                display_dir = "..." + display_dir[-27:]
            _set_text(doc, "sidebar-filepath", display_dir)
            filepath_el = doc.get_element_by_id("sidebar-filepath")
            if filepath_el:
                filepath_el.set_attribute("title", full_path)
        else:
            _set_text(doc, "sidebar-filename", "")
            for eid in ("sidebar-width", "sidebar-height", "sidebar-mp",
                        "sidebar-aspect", "sidebar-channels", "sidebar-format",
                        "sidebar-size", "sidebar-filepath"):
                _set_text(doc, eid, "")

        has_mask = self._has_valid_overlay()
        mask_section = doc.get_element_by_id("sidebar-mask-section")

        if has_mask:
            if mask_section:
                mask_section.set_attribute("class", "sidebar-section-ip")
            _set_text(doc, "sidebar-mask-label", tr("image_preview.mask_section"))
            name = self._mask_paths[self._current_index].name
            _set_text(doc, "sidebar-mask-name", name)
            cb_mask = doc.get_element_by_id("cb-mask")
            if cb_mask:
                is_checked = cb_mask.has_attribute("checked")
                if self._show_overlay and not is_checked:
                    cb_mask.set_attribute("checked", "")
                elif not self._show_overlay and is_checked:
                    cb_mask.remove_attribute("checked")
            _set_text(doc, "cb-mask-label", tr("image_preview.show_mask_overlay"))
        else:
            if mask_section:
                mask_section.set_attribute("class", "sidebar-section-ip hidden")

    def _update_view_controls(self, doc, has_images: bool):
        controls = doc.get_element_by_id("view-controls")
        if controls:
            controls.set_attribute("class", "hidden" if not has_images else "")

        for button_id in ("btn-rotate-left", "btn-rotate-right"):
            button = doc.get_element_by_id(button_id)
            if not button:
                continue
            if has_images:
                if button.has_attribute("disabled"):
                    button.remove_attribute("disabled")
            else:
                if not button.has_attribute("disabled"):
                    button.set_attribute("disabled", "")

    def _refresh_sources(self):
        node = self._current_camera_node(resolve_unbound=True)
        if node is not None and self._current_index < len(self._mask_paths):
            self._mask_paths[self._current_index] = Path(node.mask_path) if node.mask_path else None
        self._metadata.reset()
        self._sample_request = None
        self._color_picker_active = False
        self._displayed_index = -1
        self._preview_state = "loading"
        self._update_picker_cursor()
        self._refresh_revision += 1
        self._action_error = ""
        self._metadata_validated.clear()
        self._metadata_required = True
        self._sampled_color = None
        if self._image_paths:
            # Explicit Refresh also handles tools that preserve a file's timestamps.
            self._image_info_cache.pop(str(self._image_paths[self._current_index]), None)
        self._mark_dirty()

    def _update_toolbar(self, doc, has_images):
        tr = lf.ui.tr
        labels = {"btn-fit": "image_preview.fit", "btn-info": "image_preview.info",
                  "btn-thumbnails": "image_preview.thumbnails", "btn-help": "image_preview.help",
                  "btn-picker": "image_preview.color_picker", "btn-refresh": "image_preview.refresh",
                  "btn-apply-render-color": "image_preview.apply_render_color",
                  "btn-apply-training-color": "image_preview.apply_training_color"}
        for eid, key in labels.items():
            _set_text(doc, eid, tr(key))
        node = self._current_camera_node(resolve_unbound=True) if has_images else None
        for eid in ("btn-go-to-camera", "btn-gt-compare", "btn-toggle-training"):
            el = doc.get_element_by_id(eid)
            if el:
                self._set_disabled(el, node is None)
                if node is None:
                    el.set_attribute("title", tr("image_preview.camera_unavailable"))
        for eid in ("btn-fit", "btn-actual-size", "btn-zoom-in", "btn-zoom-out", "btn-picker", "btn-refresh"):
            el = doc.get_element_by_id(eid)
            if el:
                self._set_disabled(el, not has_images or (eid == "btn-picker" and
                                   (self._mask_editor.enabled or self._preview_state != "ready" or self._sample_request is not None)))
        for eid, active in (("btn-info", self._show_info), ("btn-thumbnails", self._show_filmstrip),
                            ("btn-help", self._show_help), ("btn-fit", self._fit_to_window),
                            ("btn-picker", self._color_picker_active)):
            el = doc.get_element_by_id(eid)
            if el:
                el.set_attribute("aria-pressed", "true" if active else "false")
                el.set_attribute("class", "btn btn--primary" if active else "btn")
        help_el = doc.get_element_by_id("shortcut-help")
        if help_el:
            help_el.set_attribute("class", "" if self._show_help else "hidden")
        sample = doc.get_element_by_id("sampled-color-controls")
        if sample:
            sample.set_attribute("class", "" if self._sampled_color else "hidden")
        if self._sampled_color:
            _set_text(doc, "sampled-color-value", "RGB " + ", ".join(f"{c:.3f}" for c in self._sampled_color))
        for destination in ("render", "training"):
            button = doc.get_element_by_id(f"btn-apply-{destination}-color")
            if button:
                available = self._can_apply_color(destination)
                self._set_disabled(button, not available)
                button.set_attribute("title", "" if available else tr("training_panel.parameters_unavailable"))
        _set_text(doc, "preview-message", self._action_error or
                  (lf.ui.tr("image_preview.load_failed") if self._preview_load_failed and self._ready_preview else ""))

    def _can_apply_color(self, destination):
        if self._sampled_color is None:
            return False
        if destination == "render":
            return lf.get_render_settings() is not None
        params = lf.optimization_params()
        return params is not None and params.has_params()

    @staticmethod
    def _set_disabled(element, disabled):
        if element is None:
            return
        if disabled:
            element.set_attribute("disabled", "")
        elif element.has_attribute("disabled"):
            element.remove_attribute("disabled")

    @staticmethod
    def _activate_if_enabled(event, action):
        element = event.current_target()
        if element is not None and not element.has_attribute("disabled"):
            action(event)

    def _apply_sampled_color(self, destination):
        if self._sampled_color is None:
            return
        color = self._sampled_color
        # Project/scene replacement clears the shared history. Ordinary scene
        # mutations (including rename and its undo) must not invalidate replay.
        try:
            if destination == "render":
                settings = lf.get_render_settings()
                if settings is None:
                    raise RuntimeError(lf.ui.tr("training_panel.parameters_unavailable"))
                previous = tuple(settings.get("background_color"))
                def write(value):
                    # Render settings are snapshots: reuse would restore unrelated old settings.
                    current = lf.get_render_settings()
                    if current is None:
                        raise RuntimeError(lf.ui.tr("training_panel.parameters_unavailable"))
                    current.set("background_color", value)
            else:
                params = lf.optimization_params()
                if params is None or not params.has_params():
                    raise RuntimeError(lf.ui.tr("training_panel.parameters_unavailable"))
                previous = tuple(params.bg_color)
                def write(value):
                    current = lf.optimization_params()
                    if current is None or not current.has_params():
                        raise RuntimeError(lf.ui.tr("training_panel.parameters_unavailable"))
                    current.bg_color = value
            write(color)
            try:
                label = "image_preview.apply_render_color" if destination == "render" else "image_preview.apply_training_color"
                lf.undo.push(lf.ui.tr(label),
                             lambda: write(previous), lambda: write(color),
                             scope="parameters", estimated_bytes=48)
            except Exception:
                write(previous)
                raise
            self._action_error = ""
        except Exception as error:
            self._action_error = str(error)
        self._mark_dirty()

    def _update_status(self, doc, has_images: bool):
        ids = ("st-zoom", "st-position")
        if not has_images:
            for sid in ids:
                _set_text(doc, sid, "")
            return

        _set_text(doc, "st-zoom", f"{lf.ui.tr('image_preview.zoom')} {self._get_zoom_display()}")
        _set_text(doc, "st-position", f"{self._current_index + 1:,} / {len(self._image_paths):,}")

    # -- Assisted masks: the original image is the coordinate authority. --

    def _toggle_sam(self):
        self._mask_editor.enabled = not self._mask_editor.enabled
        if not self._mask_editor.enabled:
            self._sam_cancel()
        else:
            self._color_picker_active = False
            self._show_info = False
        self._update_picker_cursor()
        self._mark_dirty()

    def _sam_tool(self, tool):
        self._mask_editor.tool = tool
        self._mask_editor.drag_start = self._mask_editor.drag_end = None
        self._mask_editor.point_drag = None
        self._mask_editor.box_drag = None
        self._color_picker_active = False
        self._update_picker_cursor()
        self._dragging = False
        self._mark_dirty()

    def _set_sam_method(self, method):
        if method not in ("manual", "semantic") or method == self._sam_method:
            return
        self._sam_method = method
        # End transient gestures, keeping the draft, proposal and selected tool.
        self._sam_tool(self._mask_editor.tool)
        if self._doc:
            controls = self._doc.get_element_by_id("sam-controls")
            if controls:
                controls.scroll_top = 0

    def _sam_record_edit(self, key, before, after):
        panel_ref = weakref.ref(self)
        editor = self._mask_editor
        editor_ref = weakref.ref(editor)

        def require_idle(panel):
            if panel._sam_sequence and lf.ui.sam_preview_poll().get("phase", "idle") in FrameMaskEditor.BUSY_PHASES:
                raise RuntimeError(lf.ui.tr("projects.gallery.error.busy"))

        def restore(state):
            panel = panel_ref()
            owner = editor_ref()
            # Project replacement clears shared history; ordinary scene mutations
            # (including Apply and its undo) must not invalidate these edits.
            if panel is not None and owner is panel._mask_editor:
                require_idle(panel)
                if owner.restore(key, state):
                    panel._sam_invalidate_sequence_frame(key)
                    panel._mark_dirty()

        try:
            require_idle(self)
            lf.undo.push(lf.ui.tr("image_preview.sam_title") + ": " + lf.ui.tr("image_preview.sam_edit"),
                         lambda: restore(before), lambda: restore(after),
                         scope="image_preview", dirty_flags=16,
                         estimated_bytes=512 + 128 * (len(before[0]) + len(after[0])))
        except Exception:
            editor.restore(key, before)
            raise
        self._sam_invalidate_sequence_frame(key)

    def _sam_invalidate_sequence_frame(self, key):
        if not self._sam_sequence:
            return
        index = next((i for i, path in enumerate(self._image_paths) if str(path) == key[0]), None)
        if index is not None:
            lf.ui.sam_preview_invalidate_frame(self._sam_sequence["run"], index)
            self._sam_sequence_revision = -1
            if key == self._mask_editor.key:
                self._sam_apply_target = None
                self._sam_inspect_pending = False
                self._sam_tracking_request = 0
                self._mask_editor.invalidate(keep_refinement=True)

    def _sam_release_texture(self, destroy=True):
        if self._sam_texture is None:
            return
        overlay = self._doc.get_element_by_id("sam-overlay") if self._doc else None
        if overlay:
            self._set_decorator(overlay, "sam-overlay", "none")
        if self._sam_texture_source:
            lf.ui.release_rml_texture(self._sam_texture_source)
        if destroy:
            self._sam_texture.destroy()
            self._sam_texture = None
        self._sam_texture_key = None
        self._sam_texture_source = ""

    @staticmethod
    def _sam_modifiers(event):
        return ((2 if event.get_bool_parameter("ctrl_key", False) else 0)
                | (1 if event.get_bool_parameter("shift_key", False) else 0)
                | (4 if event.get_bool_parameter("alt_key", False) else 0)
                | (8 if (event.get_bool_parameter("meta_key", False)
                         or event.get_bool_parameter("command_key", False)) else 0))

    def _sam_edit(self, action):
        if self._sam_method != "manual":
            return
        if self._sam_apply_target is not None or self._sam_worker_busy or self._mask_editor.busy:
            return
        getattr(self._mask_editor, action)()
        self._mark_dirty()

    def _sam_select_control(self, event):
        if self._sam_syncing_selection or self._sam_apply_target is not None:
            return
        value = event.get_parameter("value", "")
        draft = self._mask_editor.draft
        selected = ("box" if value == "box" and draft and draft.box else
                    int(value) if value.isdecimal() and draft and int(value) < len(draft.points) else None)
        if selected != self._mask_editor.selected:
            self._mask_editor.selected = selected
            self._mark_dirty()

    def _sam_show(self):
        self._mask_editor.show_proposal = not self._mask_editor.show_proposal
        self._mark_dirty()

    def _sam_cancel(self):
        self._sam_tokenizer = None
        self._sam_apply_target = None
        self._sam_inspect_pending = False
        self._sam_tracking_request = 0
        self._sam_sequence_error = ""
        self._sam_release_texture()
        lf.ui.sam_preview_close()
        self._mask_editor.invalidate()
        if self._mask_editor.enabled:
            self._mask_editor.phase = "closing"
        self._mark_dirty()

    def _sam_candidate(self, index):
        if self._sam_apply_target is not None or self._mask_editor.phase != "ready":
            return
        if 0 <= index < len(self._mask_editor.overlays) and index != self._mask_editor.candidate:
            self._mask_editor.candidate = index
            self._mask_editor.refinement = (self._mask_editor.request, index)
            if self._sam_sequence:
                self._sam_retain_candidate()
            self._mark_dirty()

    def _sam_source_point(self, event, clamp=False):
        if not self._doc or not self._image_paths or self._displayed_index != self._current_index:
            return None
        path = self._image_paths[self._current_index]
        viewport = self._doc.get_element_by_id("image-viewport")
        layout = self._get_image_layout(path, viewport) if viewport else None
        if not layout:
            return None
        ratio = layout["dp_ratio"]
        x = (float(event.get_parameter("mouse_x", "0")) - viewport.absolute_left) / ratio
        y = (float(event.get_parameter("mouse_y", "0")) - viewport.absolute_top) / ratio
        sx, sy = view_to_source(layout, self._rotation_quadrants, x, y)
        if not clamp and not (0 <= sx < layout["width"] and 0 <= sy < layout["height"]):
            return None
        return (min(1.0, max(0.0, sx / max(1, layout["width"] - 1))),
                min(1.0, max(0.0, sy / max(1, layout["height"] - 1))))

    def _sam_pointer(self, event):
        if self._sam_method != "manual" or self._sam_apply_target is not None or self._sam_worker_busy or self._mask_editor.busy:
            return
        point = self._sam_source_point(event)
        if point is None:
            return
        hit = self._sam_hit(event)
        if hit is None and self._mask_editor.tool == "box" and self._mask_editor.draft and self._mask_editor.draft.box:
            x0, y0, x1, y1 = self._mask_editor.draft.box
            if x0 < point[0] < x1 and y0 < point[1] < y1:
                hit = "box"
        if hit is not None:
            self._mask_editor.selected = hit
            if isinstance(hit, int):
                self._mask_editor.point_drag = point
            else:
                box = self._mask_editor.draft.box
                layout = self._get_image_layout(self._image_paths[self._current_index])
                sx, sy = (layout["width"] - 1) * layout["scale"], (layout["height"] - 1) * layout["scale"]
                edges = []
                for axis, scale in ((0, sx), (1, sy)):
                    edge = min((axis, axis + 2), key=lambda i: abs(point[axis] - box[i]))
                    if abs(point[axis] - box[edge]) * scale <= 8:
                        edges.append(edge)
                self._mask_editor.box_drag = (box, point, edges)
                self._mask_editor.drag_end = point
            self._mark_dirty()
            return
        if self._mask_editor.tool == "box":
            self._mask_editor.drag_start = self._mask_editor.drag_end = point
        elif self._mask_editor.tool in ("positive", "negative"):
            self._mask_editor.point(*point, 1 if self._mask_editor.tool == "positive" else 0)
        else:
            self._mask_editor.selected = None
        self._mark_dirty()

    def _sam_hit(self, event):
        point = self._sam_source_point(event)
        draft = self._mask_editor.draft
        if point is None or not draft:
            return None
        layout = self._get_image_layout(self._image_paths[self._current_index])
        sx, sy = (layout["width"] - 1) * layout["scale"], (layout["height"] - 1) * layout["scale"]
        distances = [((x - point[0]) * sx) ** 2 + ((y - point[1]) * sy) ** 2 for x, y, _ in draft.points]
        if distances:
            index = min(range(len(distances)), key=distances.__getitem__)
            if distances[index] <= 12 ** 2:
                return index
        if draft.box:
            x0, y0, x1, y1 = draft.box
            x, y = point
            if ((x0 - 8 / max(sx, 1) <= x <= x1 + 8 / max(sx, 1) and min(abs(y-y0), abs(y-y1)) * sy <= 8)
                    or (y0 - 8 / max(sy, 1) <= y <= y1 + 8 / max(sy, 1) and min(abs(x-x0), abs(x-x1)) * sx <= 8)):
                return "box"
        return None

    def _sam_semantic_busy(self):
        return (self._mask_editor.busy or self._sam_worker_busy
                or self._sam_apply_target is not None or bool(self._sam_tracking_request))

    def _sam_semantic_edit(self, name, value):
        editor = self._mask_editor
        if not editor.enabled or self._sam_semantic_busy() or getattr(editor, name) == value:
            return
        setattr(editor, name, value)
        # Search settings describe the next request. Keep the current proposal
        # and sequence approval until a replacement is successfully submitted.
        editor.error = ""
        if name == "semantic_categories":
            self._sam_semantic_ui_state = None
        if self._handle and name in ("semantic_text", "semantic_confidence"):
            self._handle.dirty(name)
        self._mark_dirty()

    def _sam_set_semantic_confidence(self, value):
        try:
            value = int(float(value))
        except (ValueError, TypeError, OverflowError):
            return
        self._sam_semantic_edit("semantic_confidence", max(1, min(99, value)))

    def _sam_reset_default(self, prop):
        if prop == "semantic_confidence":
            return FrameMaskEditor.DEFAULT_SEMANTIC_CONFIDENCE
        return None

    def _sam_apply_reset(self, prop, value):
        if self._sam_reset_default(prop) is None or not self._mask_editor.enabled or self._sam_semantic_busy():
            return False
        self._sam_set_semantic_confidence(value)
        return self._mask_editor.semantic_confidence == value

    def _sam_add_category(self, event):
        value = event.get_parameter("value", "")
        if value not in SEMANTIC_CATEGORIES:
            return
        if self._doc:
            element = self._doc.get_element_by_id("sam-semantic-category")
            if element:
                element.set_attribute("value", "")
        selected = self._mask_editor.semantic_categories
        if value not in selected:
            self._sam_semantic_edit("semantic_categories", selected + (value,))

    def _sam_remove_category(self, value):
        selected = tuple(key for key in self._mask_editor.semantic_categories if key != value)
        self._sam_semantic_edit("semantic_categories", selected)

    def _sam_semantic_payload(self):
        editor = self._mask_editor
        phrases = list(dict.fromkeys(
            [SEMANTIC_CATEGORIES[key] for key in editor.semantic_categories]
            + [line.strip() for line in editor.semantic_text.splitlines() if line.strip()]))
        if not phrases:
            raise ValueError(lf.ui.tr(_sam_key("semantic_invalid")))
        try:
            if self._sam_tokenizer is None:
                self._sam_tokenizer = BertTokenizer.load()
        except (OSError, EOFError, ValueError):
            raise ValueError(lf.ui.tr(_sam_key("semantic_resource_error"))) from None
        try:
            descriptions = list(dict.fromkeys(self._sam_tokenizer.encode(phrase) for phrase in phrases))
        except ValueError:
            raise ValueError(lf.ui.tr(_sam_key("semantic_text_error"))) from None
        if len(descriptions) > 16:
            raise ValueError(lf.ui.tr(_sam_key("semantic_invalid")))
        if 2 + sum(len(ids) - 2 for ids in descriptions) > 256:
            raise ValueError(lf.ui.tr(_sam_key("semantic_text_error")))
        return {"descriptions": [list(ids) for ids in descriptions],
                "confidence": editor.semantic_confidence / 100.0}

    def _sam_generate(self, refine=False, *, semantic=False):
        editor = self._mask_editor
        draft = editor.draft
        if not editor.enabled or not draft or not editor.key or self._preview_state != "ready" or editor.busy or self._sam_worker_busy or self._sam_apply_target is not None:
            return
        if semantic and self._sam_semantic_busy():
            return
        stat = self._path_stat_cache.get(editor.key[0])
        if not stat or not stat[0] or (not semantic and not draft.points and not draft.box):
            return
        if refine and editor.refinement_object_count > 1:
            editor.error = lf.ui.tr(_sam_key("semantic_refine_combined"))
            self._mark_dirty()
            return
        if refine and editor.refinement_object_count == 0:
            refine = False
        base, candidate = editor.refinement if refine and editor.refinement else (0, -1)
        try:
            arguments = {"semantic": self._sam_semantic_payload()} if semantic else {}
            self._sam_apply_target = None
            self._sam_inspect_pending = False
            self._sam_sequence_error = ""
            request = lf.ui.sam_preview_submit(editor.key[0], editor.key[1], stat[1], stat[2],
                                               [] if semantic else draft.points, None if semantic else draft.box,
                                               base, candidate, editor.weights,
                                               **self._sam_sequence_arguments(), **arguments)
            self._sam_tracking_request = 0
            editor.invalidate()
            editor.request = request
            self._sam_session_active = bool(request)
            editor.submitted_signature = editor.signature
            editor.show_proposal = True
            editor.phase = "queued" if request else "closing"
        except Exception as error:
            # Validation or submission failure did not replace the native result.
            # A ready proposal remains usable while the user corrects the query.
            if not semantic or editor.phase != "ready":
                editor.phase = "error"
            editor.error = str(error)
        self._mark_dirty()

    def _sam_apply(self, exclude):
        editor = self._mask_editor
        node = self._current_camera_node(resolve_unbound=True)
        if not node or node.read_only or editor.phase != "ready" or editor.overlay is None or not editor.key or self._sam_apply_target is not None or (not exclude and editor.matched_objects == 0):
            return
        stat = self._path_stat_cache.get(editor.key[0])
        if not stat or not stat[0]:
            return
        try:
            request = lf.ui.sam_preview_submit(editor.key[0], editor.key[1], stat[1], stat[2],
                                               editor.draft.points, editor.draft.box, editor.request, -1,
                                               editor.weights, editor.candidate, bool(exclude),
                                               **self._sam_sequence_arguments())
            self._sam_apply_target = (MaskApply(request, editor.request, node.uuid, editor.key)
                                      if request else None)
            editor.phase = "rendering" if request else "closing"
            editor.error = ""
        except Exception as error:
            editor.phase, editor.error = "ready", str(error)
        self._mark_dirty()

    def _sam_save_message(self, approved=None):
        return lf.ui.tr("image_preview.sam_save_hint").format(
            output=str(Path(self._sam_sequence["output"]) / "masks"),
            approved=self._sam_progress[2] if approved is None else approved)

    def _sam_save(self, exclude, confirmed=False):
        editor = self._mask_editor
        stat = self._path_stat_cache.get(editor.key[0]) if editor.key else None
        if not self._sam_sequence or not stat or not stat[0] or not self._sam_progress[2]:
            return
        if not confirmed:
            run = self._sam_sequence["run"]
            snapshot = lf.ui.sam_preview_poll()
            revision, approved = snapshot.get("sequence_revision"), snapshot.get("approved", 0)
            if not approved or snapshot.get("phase", "idle") in FrameMaskEditor.BUSY_PHASES:
                return
            if not exclude and snapshot.get("empty_approved", 0):
                self._sam_sequence_error = lf.ui.tr("image_preview.sam_empty_keep")
                self._mark_dirty()
                return
            label = lf.ui.tr("image_preview.sam_save_excluded" if exclude else "image_preview.sam_save_approved")
            answered = False
            panel_ref = weakref.ref(self)
            editor_ref = weakref.ref(editor)
            def selected(button, _values):
                nonlocal answered
                if answered:
                    return
                answered = True
                panel = panel_ref()
                if button != label or panel is None:
                    return
                if (panel._sam_sequence and panel._sam_sequence["run"] == run and editor_ref() is panel._mask_editor
                        and revision == lf.ui.sam_preview_poll().get("sequence_revision")):
                    panel._sam_save(exclude, confirmed=True)
                else:
                    panel._sam_sequence_error = lf.ui.tr("image_preview.sam_proposal_expired")
                    panel._mark_dirty()
            message = self._sam_save_message(approved)
            lf.ui.form_dialog(f"sam-save:{run}", label, _sam_confirmation_rml(message),
                              [{"label": lf.ui.tr("common.cancel")}, {"label": label, "style": "primary"}], selected)
            return
        try:
            self._sam_sequence_error = ""
            editor.request = lf.ui.sam_preview_submit(editor.key[0], editor.key[1], stat[1], stat[2], [],
                exclude=exclude, **self._sam_sequence_arguments(publish=True))
            self._sam_tracking_request = 0
            editor.phase = "saving" if editor.request else "closing"
            editor.error = ""
            self._sam_apply_target = None
        except Exception as error:
            editor.phase, editor.error = "error", str(error)
        self._mark_dirty()

    def _sam_sequence_arguments(self, **values):
        return {"sequence": {"run": self._sam_sequence["run"], **values}} if self._sam_sequence else {}

    def _sam_retain_candidate(self):
        editor = self._mask_editor
        stat = self._path_stat_cache.get(editor.key[0]) if editor.key else None
        if not stat or not stat[0] or not self._sam_sequence or editor.phase != "ready":
            return
        try:
            proposal = editor.request
            request = lf.ui.sam_preview_submit(editor.key[0], editor.key[1], stat[1], stat[2],
                editor.draft.points, editor.draft.box, proposal, -1, editor.weights, editor.candidate, False,
                **self._sam_sequence_arguments(retain=True))
            self._sam_apply_target = MaskApply(request, proposal, None, editor.key, kind="retain")
            editor.phase = "storing"
        except Exception as error:
            editor.phase, editor.error = "error", str(error)

    def _sam_inspect_current(self):
        editor = self._mask_editor
        stat = self._path_stat_cache.get(editor.key[0]) if editor.key else None
        self._sam_inspect_pending = False
        if not stat or not stat[0] or not self._sam_sequence:
            return
        try:
            request = lf.ui.sam_preview_submit(editor.key[0], editor.key[1], stat[1], stat[2], [],
                                                **self._sam_sequence_arguments(inspect=self._current_index))
            self._sam_tracking_request = 0
            editor.invalidate()
            editor.request, editor.phase = request, "queued" if request else "closing"
            editor.submitted_signature = editor.signature
            self._sam_apply_target = None
        except Exception as error:
            editor.phase, editor.error = "error", str(error)
        self._mark_dirty()

    def _sam_track(self):
        if self._sam_tracking_request:
            if self._sam_frame_summaries.get(self._current_index, {}).get("revision"):
                self._sam_inspect_current()
            else:
                self._sam_cancel()
            return
        editor = self._mask_editor
        stat = self._path_stat_cache.get(editor.key[0]) if editor.key else None
        if not self._sam_sequence or not stat or not stat[0] or editor.phase != "ready":
            return
        if editor.matched_objects == 0:
            editor.error = lf.ui.tr("image_preview.sam_semantic_no_matches")
            self._mark_dirty()
            return
        if self._doc and not self._sam_track_options_changed(None):
            return
        if not self._sam_track_first <= self._current_index <= self._sam_track_last:
            editor.error = lf.ui.tr("image_preview.sam_range_seed_required")
            self._mark_dirty()
            return
        try:
            self._sam_sequence_error = ""
            request = lf.ui.sam_preview_submit(editor.key[0], editor.key[1], stat[1], stat[2],
                editor.draft.points, editor.draft.box, editor.request, editor.candidate, editor.weights,
                **self._sam_sequence_arguments(first=self._sam_track_first, last=self._sam_track_last,
                                               seed=self._current_index, direction=self._sam_track_direction))
            editor.invalidate()
            self._sam_tracking_request = editor.request = request
            editor.phase = "queued" if request else "closing"
            self._sam_inspect_pending = False
        except Exception as error:
            editor.phase, editor.error = "error", str(error)
        self._mark_dirty()

    def _sam_track_options_changed(self, _event):
        if self._sam_syncing_sequence or not self._doc:
            return False
        valid = True
        try:
            first = int(self._doc.get_element_by_id("sam-track-first").get_attribute("value", "1")) - 1
            last = int(self._doc.get_element_by_id("sam-track-last").get_attribute("value", "1")) - 1
            direction = self._doc.get_element_by_id("sam-track-direction").get_attribute("value", "both")
            if not 0 <= first <= last < len(self._image_paths) or direction not in ("both", "forward", "reverse"):
                raise ValueError()
            self._sam_track_first, self._sam_track_last, self._sam_track_direction = first, last, direction
            self._mask_editor.error = ""
        except ValueError:
            self._mask_editor.error = lf.ui.tr("image_preview.sam_range_invalid")
            valid = False
        self._mark_dirty()
        return valid

    def _sam_approve(self):
        if self._sam_syncing_sequence or not self._sam_sequence or self._mask_editor.phase != "ready":
            return
        run, index = self._sam_sequence["run"], self._current_index
        frames = lf.ui.sam_preview_sequence_frames(run, index, 1)
        if not frames or not frames[0]["revision"]:
            return
        previous, revision = frames[0]["approved"], frames[0]["revision"]
        label = lf.ui.tr("image_preview.sam_revoke" if previous else "image_preview.sam_approve")
        self._sam_commit_approvals(run, [(index, revision, previous)], not previous, label)

    def _sam_commit_approvals(self, run, previous, approved, label):
        panel_ref = weakref.ref(self)
        updates = [(index, revision, approved) for index, revision, _ in previous]
        def write(values):
            panel = panel_ref()
            if panel is not None and panel._sam_sequence and panel._sam_sequence["run"] == run:
                if panel._mask_editor.busy or panel._sam_worker_busy:
                    # Shared history retains the entry when playback is refused.
                    raise RuntimeError(lf.ui.tr("projects.gallery.error.busy"))
                try:
                    accepted = lf.ui.sam_preview_approve_frames(run, values)
                except Exception as error:
                    panel._sam_sequence_error = str(error)
                    panel._mark_dirty()
                    raise
                if accepted:
                    panel._sam_sequence_revision = -1
                    panel._mark_dirty()
                    return True
                panel._sam_sequence_error = lf.ui.tr("image_preview.sam_proposal_expired")
                panel._mark_dirty()
            return False
        try:
            if not write(updates):
                return
        except Exception:
            return
        def restore(values):
            panel = panel_ref()
            if panel is None or not panel._sam_sequence or panel._sam_sequence["run"] != run:
                return
            # PyUndoEntry ignores callback return values. Refuse busy playback
            # explicitly, including work started since the last panel update.
            if lf.ui.sam_preview_poll().get("phase", "idle") in FrameMaskEditor.BUSY_PHASES:
                raise RuntimeError(lf.ui.tr("projects.gallery.error.busy"))
            # On the UI thread no new work can start between this poll and write.
            # A stale proposal is reported by write(), then skipped like an
            # expired draft; it must not trap unrelated history behind it.
            write(values)
        try:
            lf.undo.push(label, lambda: restore(previous), lambda: restore(updates),
                         scope="image_preview", estimated_bytes=256 * len(previous) + 1024)
        except Exception as error:
            self._sam_sequence_error = str(error)
            try:
                write(previous)
            except Exception:
                pass  # write() retains the rollback error for the panel.
            self._mark_dirty()
            raise

    def _sam_approve_range(self):
        editor = self._mask_editor
        if not self._sam_sequence or editor.busy or editor.phase != "ready":
            return
        if self._doc and not self._sam_track_options_changed(None):
            return
        run = self._sam_sequence["run"]
        first, last = self._sam_track_first, self._sam_track_last
        revision = lf.ui.sam_preview_poll().get("sequence_revision")
        frames = []
        for start in range(first, last + 1, 256):
            frames.extend(lf.ui.sam_preview_sequence_frames(run, start, min(256, last - start + 1)))
        if len(frames) != last - first + 1:
            editor.error = lf.ui.tr("image_preview.sam_proposal_expired")
            self._mark_dirty()
            return
        if not any(frame.get("revision") for frame in frames):
            editor.error = lf.ui.tr("image_preview.sam_range_missing")
            self._mark_dirty()
            return
        previous = [(first + i, frame["revision"], False) for i, frame in enumerate(frames)
                    if frame.get("revision") and not frame["approved"]]
        if not previous:
            return
        label = lf.ui.tr("image_preview.sam_approve_range").removesuffix("…")
        message = lf.ui.tr("image_preview.sam_approve_range_hint").format(
            first=first + 1, last=last + 1, count=len(previous),
            uncertain=sum(bool(frame.get("revision")) and not frame["reliable"] and not frame["approved"] for frame in frames),
            missing=sum(not frame.get("revision") for frame in frames))
        panel_ref = weakref.ref(self)
        editor_ref = weakref.ref(editor)
        answered = False
        def selected(button, _values):
            nonlocal answered
            if answered:
                return
            answered = True
            panel = panel_ref()
            if button != label or panel is None:
                return
            owner = editor_ref()
            if (not panel._sam_sequence or panel._sam_sequence["run"] != run
                    or owner is None or panel._mask_editor is not owner or owner.busy
                    or revision != lf.ui.sam_preview_poll().get("sequence_revision")):
                panel._sam_sequence_error = lf.ui.tr("image_preview.sam_proposal_expired")
                panel._mark_dirty()
                return
            panel._sam_commit_approvals(run, previous, True, label)
        lf.ui.form_dialog(f"sam-approve:{run}", label, _sam_confirmation_rml(message),
                          [{"label": lf.ui.tr("common.cancel")}, {"label": label, "style": "primary"}], selected)

    def _service_sam(self):
        editor = self._mask_editor
        if not editor.enabled:
            return
        path = self._image_paths[self._current_index] if self._image_paths else None
        revision = self._path_revision_cache.get(str(path)) if path else None
        old_key = editor.key
        state = lf.ui.sam_preview_poll()
        worker_busy = state.get("phase", "idle") in FrameMaskEditor.BUSY_PHASES
        if worker_busy != self._sam_worker_busy:
            self._sam_worker_busy = worker_busy
            self._mark_dirty()
        if old_key and path and old_key[0] == str(path) and revision is None:
            # Metadata refresh has not established a new source revision yet.
            self._request_async_update()
            return
        if old_key and path and old_key[0] == str(path) and old_key[1] != revision:
            if worker_busy:
                # Keep the old binding until invalidation is admitted. The next
                # poll retries the same revision change without losing its edge.
                if editor.request or editor.overlays:
                    editor.invalidate()
                    self._sam_release_texture()
                    self._mark_dirty()
                message = lf.ui.tr("image_preview.sam_source_changed")
                if self._sam_sequence_error != message:
                    self._sam_sequence_error = message
                    self._mark_dirty()
                self._request_async_update()
                return
            self._sam_invalidate_sequence_frame(old_key)
            if self._sam_sequence_error == lf.ui.tr("image_preview.sam_source_changed"):
                self._sam_sequence_error = ""
            state = lf.ui.sam_preview_poll()
        bound = editor.bind(path, revision)
        if bound:
            self._sam_inspect_pending = bool(self._sam_sequence and editor.key)
            self._mark_dirty()
        self._sam_sync_sequence(state)
        if self._sam_sequence and state.get("phase") == "error" and state.get("error"):
            message = state["error"]
            if state.get("total"):
                message += f" · {state.get('completed', 0)} / {state['total']}"
            if self._sam_sequence_error != message:
                self._sam_sequence_error = message
                self._mark_dirty()
        active = state.get("phase", "idle") != "idle"
        if active != self._sam_session_active:
            self._sam_session_active = active
            self._mark_dirty()
        if state.get("phase") == "closing" and editor.phase != "closing":
            editor.phase = "closing"
            self._mark_dirty()
        target = self._sam_apply_target
        if target and state.get("request") == target.request and state.get("phase") == "error":
            self._sam_apply_target = None
            editor.phase = "ready" if editor.key == target.key and editor.request == target.proposal_request and editor.overlay is not None else "error"
            editor.error = state.get("error", "")
            self._mark_dirty()
        if state.get("phase") in ("mask_ready", "expired") and target and state.get("request") == target.request:
            self._sam_apply_target = None
            if state["phase"] == "mask_ready" and editor.key == target.key and editor.request == target.proposal_request:
                try:
                    if target.kind == "retain":
                        editor.phase = "ready"
                    else:
                        lf.ui.sam_preview_apply(target.camera_uuid, target.request)
                        self._sam_cancel()
                        editor.enabled = False
                        self._show_overlay = True
                        self._refresh_sources()
                except Exception as error:
                    editor.phase, editor.error = "ready", str(error)
                self._mark_dirty()
            elif target.kind != "retain" or editor.key == target.key:
                editor.invalidate()
                editor.phase, editor.error = "error", lf.ui.tr("image_preview.sam_proposal_expired")
                self._mark_dirty()
        if editor.accept(state):
            self._mark_dirty()
        if self._sam_tracking_request and state.get("request") == self._sam_tracking_request:
            if state.get("phase") in ("tracked", "error"):
                self._sam_tracking_request = 0
                self._sam_inspect_pending = state.get("phase") == "tracked"
        elif self._sam_tracking_request and state.get("phase") == "idle":
            # Native retirement may clear the request before the next panel poll.
            self._sam_tracking_request = 0
            self._mark_dirty()
        if editor.request == state.get("request") and state.get("phase") == "saved":
            if editor.overlay is not None:
                self._sam_inspect_pending = True
            elif not self._sam_inspect_pending:
                self._sam_cancel()
        if (self._sam_inspect_pending and editor.key and self._preview_state == "ready"
                and state.get("phase") not in FrameMaskEditor.BUSY_PHASES):
            frame = self._sam_frame_summaries.get(self._current_index, {})
            if frame.get("revision"):
                self._sam_inspect_current()
            else:
                self._sam_inspect_pending = False
        if editor.phase == "closing" and state.get("phase") == "idle":
            editor.phase = "idle"
            self._mark_dirty()
        if editor.busy or self._sam_tracking_request:
            self._request_async_update()

    def _sam_sync_sequence(self, state):
        if not self._sam_sequence:
            return
        progress = (state.get("completed", 0), state.get("total", 0), state.get("approved", 0))
        empty_approved = state.get("empty_approved", 0)
        if progress != self._sam_progress or empty_approved != self._sam_empty_approved:
            self._sam_progress = progress
            self._sam_empty_approved = empty_approved
            self._mark_dirty()
        lo, hi = self._filmstrip_range[:2] if self._filmstrip_range else (self._current_index, self._current_index + 1)
        key = (state.get("sequence_revision", 0), self._current_index, lo, hi)
        if key != self._sam_sequence_revision:
            run = self._sam_sequence["run"]
            frames = []
            for start in range(lo, hi, 256):
                frames.extend(lf.ui.sam_preview_sequence_frames(run, start, min(256, hi - start)))
            self._sam_frame_summaries = {lo + i: frame for i, frame in enumerate(frames)}
            if self._current_index not in self._sam_frame_summaries:
                current = lf.ui.sam_preview_sequence_frames(run, self._current_index, 1)
                if current:
                    self._sam_frame_summaries[self._current_index] = current[0]
            for index, frame in self._sam_frame_summaries.items():
                if frame.get("mask_path"):
                    self._mask_paths[index] = Path(frame["mask_path"])
            self._sam_sequence_revision = key
            self._mark_dirty()

    def _sam_frame_status(self, index):
        frame = self._sam_frame_summaries.get(index, {})
        key = ("approved" if frame.get("approved") else "keyframe" if frame.get("keyframe") else
               "proposal" if frame.get("revision") else "pending")
        status = lf.ui.tr(_sam_key("frame_" + key))
        if frame.get("revision") and not frame.get("reliable") and not frame.get("approved"):
            status += " · " + lf.ui.tr("image_preview.sam_frame_low_confidence")
        return status

    def _update_sam_semantic(self, doc):
        editor = self._mask_editor
        if not editor.enabled:
            return
        semantic = self._sam_method == "semantic"
        for element_id, hidden in (("sam-semantic-controls", not semantic), ("sam-manual-controls", semantic)):
            element = doc.get_element_by_id(element_id)
            if element:
                element.set_attribute("class", "hidden" if hidden else "")
        for element_id, active, key in (("sam-method-manual", not semantic, "method_manual"),
                                         ("sam-semantic-heading", semantic, "semantic_heading")):
            button = doc.get_element_by_id(element_id)
            if button:
                label = lf.ui.tr(_sam_key(key))
                button.set_text(label)
                button.set_attribute("title", label)
                button.set_attribute("aria-pressed", "true" if active else "false")
                button.set_attribute("class", "btn btn--primary" if active else "btn")
        for element_id, style in (("sam-proposals-heading", "section-label-ip"),
                                   ("sam-candidates-row", "sam-tool-row"), ("sam-quality", "sam-note")):
            element = doc.get_element_by_id(element_id)
            if element:
                element.set_attribute("class", style + (" hidden" if semantic else ""))
        for suffix in ("heading", "descriptions", "find", "help"):
            _set_text(doc, "sam-semantic-" + suffix, lf.ui.tr(_sam_key("semantic_" + suffix)))
        advanced = doc.get_element_by_id("sam-semantic-advanced")
        if advanced:
            advanced.set_attribute("aria-expanded", "true" if self._show_semantic_options else "false")
        options = doc.get_element_by_id("sam-semantic-options")
        if options:
            options.set_class("hidden", not self._show_semantic_options)
        busy = self._sam_semantic_busy()
        self._set_disabled(doc.get_element_by_id("sam-semantic-confidence"), busy)
        state = (editor.semantic_categories, RuntimeState.language_generation.value, busy)
        if state != self._sam_semantic_ui_state:
            records = []
            for key in editor.semantic_categories:
                label = lf.ui.tr(_sam_key("category_" + key))
                records.append({"value": key, "label": label, "disabled": busy,
                                "remove_label": lf.ui.tr(_sam_key("semantic_remove_category")).format(name=label)})
            if self._handle:
                self._handle.update_record_list("semantic_categories", records)
            category = doc.get_element_by_id("sam-semantic-category")
            if category:
                category.set_attribute("aria-label", lf.ui.tr(_sam_key("semantic_add_category")))
                options = [("", lf.ui.tr(_sam_key("semantic_add_category")))] + [
                    (key, lf.ui.tr(_sam_key("category_" + key))) for key in SEMANTIC_CATEGORIES
                    if key not in editor.semantic_categories]
                category.set_inner_rml("".join(f'<option value="{key}">{escape(label)}</option>' for key, label in options))
                category.set_attribute("value", "")
                self._set_disabled(category, busy or len(options) == 1)
            self._sam_semantic_ui_state = state
        self._set_disabled(doc.get_element_by_id("sam-semantic-text"), busy)
        has_query = bool(editor.semantic_categories or editor.semantic_text.strip())
        ready = self._preview_state == "ready" and bool(editor.key and has_query)
        self._set_disabled(doc.get_element_by_id("sam-semantic-find"), busy or not ready)
        find = doc.get_element_by_id("sam-semantic-find")
        if find:
            label = lf.ui.tr(_sam_key("semantic_find"))
            binding = lf.keymap.get_trigger_description(lf.keymap.Action.IMAGE_PREVIEW_GENERATE, lf.keymap.ToolMode.IMAGE_PREVIEW)
            find.set_attribute("title", f"{label} ({binding})" if binding else label)

    def _update_sam(self, doc):
        editor = self._mask_editor
        self._update_sam_semantic(doc)
        toggle = doc.get_element_by_id("btn-sam")
        if toggle:
            toggle.set_attribute("aria-pressed", "true" if editor.enabled else "false")
            toggle.set_attribute("class", "btn btn--primary" if editor.enabled else "btn")
        for element_id in ("sam-controls", "sam-apply-controls"):
            bar = doc.get_element_by_id(element_id)
            if bar:
                bar.set_attribute("class", "" if editor.enabled else "hidden")
        for key in ("positive", "negative", "box", "pan", "undo", "clear", "remove", "flip", "generate", "cancel", "show"):
            element = doc.get_element_by_id("sam-" + key)
            if element:
                element.set_text(lf.ui.tr(_sam_key(key)))
                element.set_attribute("aria-pressed", "true" if editor.tool == key else "false")
        _set_text(doc, "btn-sam", lf.ui.tr("image_preview.sam_title"))
        for key in ("heading", "proposals-heading", "apply-heading", "edit-hint"):
            hint = "semantic_help" if key == "edit-hint" and self._sam_method == "semantic" else key.replace("-", "_")
            _set_text(doc, "sam-" + key, lf.ui.tr(_sam_key(hint)))
        if self._sam_sequence:
            _set_text(doc, "sam-apply-heading", lf.ui.tr("image_preview.sam_save_heading"))
        refine = doc.get_element_by_id("sam-refine")
        if refine:
            refine.set_text(lf.ui.tr("image_preview.sam_refine"))
        for element_id, key in (("sam-positive", "positive_hint"), ("sam-negative", "negative_hint"),
                                ("sam-box", "box_hint"), ("sam-pan", "pan_hint"),
                                ("sam-generate", "generate_hint"), ("sam-refine", "refine_hint"),
                                ("sam-remove", "edit_hint"), ("sam-undo", "undo_hint")):
            element = doc.get_element_by_id(element_id)
            if element:
                element.set_attribute("title", lf.ui.tr(_sam_key(key)))
        for element_id, action in (("sam-remove", lf.keymap.Action.IMAGE_PREVIEW_DELETE),
                                   ("sam-generate", lf.keymap.Action.IMAGE_PREVIEW_GENERATE)):
            binding = lf.keymap.get_trigger_description(action, lf.keymap.ToolMode.IMAGE_PREVIEW)
            element = doc.get_element_by_id(element_id)
            if element:
                hint = (lf.ui.tr(_sam_key("generate_hint")) if element_id == "sam-generate"
                        else lf.keymap.get_action_name(action))
                element.set_attribute("title", f"{hint} ({binding})" if binding else hint)
        bindings = []
        actions = ((lf.keymap.Action.IMAGE_PREVIEW_GENERATE,) if self._sam_method == "semantic" else
                   (lf.keymap.Action.IMAGE_PREVIEW_DELETE, lf.keymap.Action.IMAGE_PREVIEW_REMOVE_POINT,
                    lf.keymap.Action.IMAGE_PREVIEW_GENERATE))
        for action in actions:
            binding = lf.keymap.get_trigger_description(action, lf.keymap.ToolMode.IMAGE_PREVIEW)
            if binding:
                bindings.append(f"{binding}: {lf.keymap.get_action_name(action)}")
        _set_text(doc, "sam-shortcuts", " · ".join(bindings))
        node = self._current_camera_node(resolve_unbound=True)
        has_target = bool(node and not node.read_only)
        hint = "tracking_hint" if self._sam_sequence else "hint" if has_target else "apply_dataset"
        _set_text(doc, "sam-hint", lf.ui.tr(_sam_key(hint)))
        _set_text(doc, "sam-apply", lf.ui.tr("image_preview.sam_apply"))
        _set_text(doc, "sam-exclude", lf.ui.tr("image_preview.sam_exclude"))
        applicable = editor.phase == "ready" and (editor.overlay is not None) and has_target
        for element_id, label in (("sam-apply", "apply"), ("sam-exclude", "exclude")):
            button = doc.get_element_by_id(element_id)
            empty_keep = element_id == "sam-apply" and editor.matched_objects == 0
            self._set_disabled(button, not applicable or empty_keep)
            if button:
                hint_key = "semantic_no_matches" if empty_keep and has_target else "apply_dataset" if not has_target else label
                button.set_attribute("title", lf.ui.tr(_sam_key(hint_key)))
        phase = editor.phase if not editor.error else "error"
        has_controls = bool(editor.draft and (editor.draft.points or editor.draft.box))
        status = ("" if phase == "idle" and not has_controls else
                  lf.ui.tr(_sam_key("generate_hint" if phase == "idle" else phase)))
        if phase in ("tracking", "saving", "saved") or (phase == "error" and self._sam_progress[1]):
            status += f" · {self._sam_progress[0]} / {self._sam_progress[1]}"
        if phase == "ready" and editor.matched_objects >= 0:
            status = (lf.ui.tr(_sam_key("semantic_no_matches")) if editor.matched_objects == 0 else
                      lf.ui.tr(_sam_key("semantic_matches")).format(count=editor.matched_objects))
            if editor.semantic_limited:
                status += " " + lf.ui.tr(_sam_key("semantic_limited"))
        status += ": " + editor.error if editor.error else ""
        if self._sam_sequence_error:
            status = lf.ui.tr("image_preview.sam_error") + ": " + self._sam_sequence_error
        _set_text(doc, "sam-status", status)
        status_element = doc.get_element_by_id("sam-status")
        if status_element:
            status_element.set_attribute("title", status)
        ready = bool(editor.draft and (editor.draft.points or editor.draft.box) and self._preview_state == "ready")
        busy = editor.busy or self._sam_worker_busy
        self._set_disabled(doc.get_element_by_id("sam-generate"), not ready or busy)
        self._set_disabled(doc.get_element_by_id("sam-refine"), not ready or busy or not editor.can_refine)
        refine = doc.get_element_by_id("sam-refine")
        if refine and editor.refinement_object_count > 1:
            refine.set_attribute("title", lf.ui.tr(_sam_key("semantic_refine_combined")))
        self._set_disabled(doc.get_element_by_id("sam-cancel"), not self._sam_session_active or editor.phase == "closing")
        applying = self._sam_apply_target is not None or busy
        self._set_disabled(doc.get_element_by_id("sam-clear"), applying or not bool(editor.draft and (editor.draft.points or editor.draft.box)))
        self._set_disabled(doc.get_element_by_id("sam-remove"), applying or editor.selected is None)
        self._set_disabled(doc.get_element_by_id("sam-flip"), applying or not isinstance(editor.selected, int))
        self._set_disabled(doc.get_element_by_id("sam-undo"), applying or not lf.undo.can_undo())
        self._set_disabled(doc.get_element_by_id("sam-show"), not bool(editor.overlays) or editor.matched_objects == 0)
        self._update_sam_selection(doc)
        _set_text(doc, "sam-quality", (lf.ui.tr("image_preview.sam_quality").format(score=round(100 * editor.scores[editor.candidate]))
                                       if editor.scores and editor.matched_objects != 0 else ""))
        show = doc.get_element_by_id("sam-show")
        if show:
            show.set_attribute("aria-pressed", "true" if editor.show_proposal else "false")
        for index in range(3):
            button = doc.get_element_by_id(f"sam-candidate-{index}")
            if button:
                available = index < len(editor.scores)
                text = str(index + 1)
                button.set_text(text)
                description = lf.ui.tr("image_preview.sam_candidate")
                if available:
                    description += f": {round(100 * editor.scores[index])}%"
                button.set_attribute("title", description)
                button.set_attribute("aria-label", f"{index + 1} · {description}")
                button.set_attribute("aria-pressed", "true" if index == editor.candidate and available else "false")
                self._set_disabled(button, not available or busy)
        legend = doc.get_element_by_id("mask-legend")
        if legend:
            proposal = editor.enabled and editor.show_proposal and editor.overlay is not None and editor.matched_objects != 0
            applied = not editor.enabled and self._show_overlay and self._has_valid_overlay()
            legend.set_attribute("class", "sam-note" if proposal or applied else "sam-note hidden")
            key = ("image_preview.sam_proposal_legend" if proposal else
                   "image_preview.mask_excluded_legend" if self._mask_inverted else "image_preview.mask_kept_legend")
            description = lf.ui.tr(key)
            legend.set_text(description)
            legend.set_attribute("title", description)
        self._update_sam_sequence(doc, busy)
        self._update_sam_page(doc)
        self._update_sam_geometry(doc)

    def _set_sam_page(self, page):
        self._sam_page = page
        if self._doc:
            self._doc.get_element_by_id("sam-controls").scroll_top = 0
        self._mark_dirty()

    def _update_sam_page(self, doc):
        sequence = bool(self._sam_sequence)
        pages = doc.get_element_by_id("sam-pages")
        if pages:
            pages.set_class("hidden", not (sequence and self._mask_editor.enabled))
        heading = doc.get_element_by_id("sam-heading")
        if heading:
            heading.set_class("hidden", sequence)
        for page, content in (("frame", "sam-frame-controls"), ("sequence", "sam-tracking")):
            active = page == (self._sam_page if sequence else "frame")
            button = doc.get_element_by_id("sam-page-" + page)
            if button:
                button.set_text(lf.ui.tr(_sam_key("page_" + page)))
                button.set_class("btn--primary", active)
                button.set_attribute("aria-pressed", "true" if active else "false")
            element = doc.get_element_by_id(content)
            if element:
                element.set_class("hidden", not active)
        for row, method in (("sam-generate-row", "manual"), ("sam-semantic-find-row", "semantic")):
            action = doc.get_element_by_id(row)
            if action:
                action.set_class("hidden", self._sam_method != method or (sequence and self._sam_page != "frame"))

    def _update_sam_sequence(self, doc, busy):
        for eid in ("sam-apply", "sam-exclude"):
            button = doc.get_element_by_id(eid)
            if button:
                button.set_class("hidden", bool(self._sam_sequence))
        save = doc.get_element_by_id("sam-save-controls")
        if save:
            save.set_class("hidden", not bool(self._sam_sequence))
        controls = doc.get_element_by_id("sam-tracking")
        if not controls:
            return
        if not self._sam_sequence:
            return
        _set_text(doc, "sam-apply-heading", lf.ui.tr("image_preview.sam_save_heading")
                  + f" · {self._sam_progress[2]} / {len(self._image_paths)}")
        for eid, key in (("sam-tracking-heading", "tracking_heading"), ("sam-track-first-label", "range_first"),
                         ("sam-track-last-label", "range_last"), ("sam-track-direction-label", "direction"),
                         ("sam-approve-label", "approve")):
            _set_text(doc, eid, lf.ui.tr(_sam_key(key)))
        self._sam_syncing_sequence = True
        try:
            for eid, value in (("sam-track-first", self._sam_track_first + 1), ("sam-track-last", self._sam_track_last + 1)):
                element = doc.get_element_by_id(eid)
                # Initialize each run once. A temporarily empty or invalid edit
                # must stay editable and be rejected by Track, not overwritten.
                if element.get_attribute("data-sequence-run", "") != self._sam_sequence["run"]:
                    element.set_attribute("value", str(value))
                    element.set_attribute("data-sequence-run", self._sam_sequence["run"])
                self._set_disabled(element, busy)
            direction = doc.get_element_by_id("sam-track-direction")
            options = "".join(f'<option value="{value}">{escape(lf.ui.tr(_sam_key(value)))}</option>'
                              for value in ("both", "forward", "reverse"))
            # RmlUI select options are not exposed by get_inner_rml(). Rebuilding
            # them on every refresh emits change events and keeps the UI awake.
            if self._sam_direction_options != options:
                direction.set_inner_rml(options)
                self._sam_direction_options = options
            if direction.get_attribute("value", "") != self._sam_track_direction:
                direction.set_attribute("value", self._sam_track_direction)
            self._set_disabled(direction, busy)
            frame = self._sam_frame_summaries.get(self._current_index, {})
            approved = doc.get_element_by_id("sam-approve")
            checked = bool(frame.get("approved"))
            if approved.has_attribute("checked") != checked:
                if checked:
                    approved.set_attribute("checked", "checked")
                else:
                    approved.remove_attribute("checked")
            self._set_disabled(approved, busy or self._mask_editor.phase != "ready" or not frame.get("revision"))
        finally:
            self._sam_syncing_sequence = False
        track = doc.get_element_by_id("sam-track")
        track.set_text(lf.ui.tr(_sam_key("stop_tracking" if self._sam_tracking_request else "track")))
        empty = self._mask_editor.matched_objects == 0
        track.set_attribute("title", lf.ui.tr("image_preview.sam_semantic_no_matches" if empty and not self._sam_tracking_request else "image_preview.sam_tracking_hint"))
        self._set_disabled(track, not self._sam_tracking_request and (busy or self._mask_editor.phase != "ready" or empty))
        approve_range = doc.get_element_by_id("sam-approve-range")
        if approve_range:
            approve_range.set_text(lf.ui.tr("image_preview.sam_approve_range"))
            self._set_disabled(approve_range, busy or self._mask_editor.phase != "ready")
        sources = self._sam_sequence.get("frames", ())
        source = sources[self._current_index] if self._current_index < len(sources) else {}
        time_text = f"≈{source['time']:.3f}" if "time" in source else "—"
        status = lf.ui.tr("image_preview.sam_frame_status").format(frame=self._current_index + 1,
            total=len(self._image_paths), time=time_text, state=self._sam_frame_status(self._current_index), approved=self._sam_progress[2])
        if self._sam_frame_summaries.get(self._current_index, {}).get("mask_path"):
            status += " · " + lf.ui.tr("image_preview.sam_saved")
        _set_text(doc, "sam-sequence-status", status)
        for eid, key in (("sam-save", "save_approved"), ("sam-save-exclude", "save_excluded")):
            button = doc.get_element_by_id(eid)
            button.set_text(lf.ui.tr(_sam_key(key)))
            empty_keep = eid == "sam-save" and self._sam_empty_approved > 0
            button.set_attribute("title", lf.ui.tr("image_preview.sam_empty_keep") if empty_keep else self._sam_save_message())
            self._set_disabled(button, busy or not self._sam_progress[2] or empty_keep)

    def _update_sam_selection(self, doc):
        editor = self._mask_editor
        selection = doc.get_element_by_id("sam-selection")
        if not selection:
            return
        options = [("", lf.ui.tr("image_preview.sam_selection_none"))]
        if editor.draft:
            options += [(str(i), f"{i + 1} · {lf.ui.tr(_sam_key('positive' if label else 'negative'))}")
                        for i, (_, _, label) in enumerate(editor.draft.points)]
            if editor.draft.box:
                options.append(("box", lf.ui.tr("image_preview.sam_box")))
        # RmlUI emits change events even for programmatic option/value updates.
        # The model remains the owner while synchronizing its selection to the view.
        self._sam_syncing_selection = True
        try:
            if options != self._sam_selection_options:
                selection.set_inner_rml("".join(f'<option value="{value}">{escape(text)}</option>' for value, text in options))
                self._sam_selection_options = options
            value = "" if editor.selected is None else str(editor.selected)
            if selection.get_attribute("value", "") != value:
                selection.set_attribute("value", value)
            selection.set_attribute("aria-label", lf.ui.tr("image_preview.sam_heading"))
            self._set_disabled(selection, self._sam_apply_target is not None or len(options) == 1)
        finally:
            self._sam_syncing_selection = False

    def _update_sam_geometry(self, doc):
        editor = self._mask_editor
        overlay = doc.get_element_by_id("sam-overlay")
        if overlay:
            visible = bool(editor.enabled and editor.show_proposal and editor.overlay is not None and editor.matched_objects != 0 and self._preview_state == "ready")
            overlay.set_attribute("class", "image-layer" if visible else "image-layer hidden")
            if visible:
                path = self._image_paths[self._current_index]
                texture_key = (editor.request, editor.candidate)
                if texture_key != self._sam_texture_key:
                    self._sam_release_texture(destroy=False)
                    if self._sam_texture is None:
                        self._sam_texture = lf.ui.DynamicTexture(editor.overlay)
                    else:
                        self._sam_texture.update(editor.overlay)
                    info = self._get_image_info(path)
                    self._sam_texture_source = self._sam_texture.rml_src(info[0], info[1])
                    self._sam_texture_key = texture_key
                self._set_decorator(overlay, "sam-overlay", f"image({self._sam_texture_source})")
                self._apply_zoom(overlay, path)
            elif not editor.enabled or editor.overlay is None or editor.matched_objects == 0:
                self._sam_release_texture(destroy=not editor.enabled)
        markers = doc.get_element_by_id("sam-markers")
        if markers:
            html = []
            if editor.enabled and self._sam_method == "manual" and editor.draft and self._preview_state == "ready":
                layout = self._get_image_layout(self._image_paths[self._current_index])
                if layout:
                    def view(x, y):
                        return source_to_view(layout, self._rotation_quadrants, x * (layout["width"] - 1), y * (layout["height"] - 1))
                    for index, (x, y, label) in enumerate(editor.draft.points):
                        if index == editor.selected and editor.point_drag is not None:
                            x, y = editor.point_drag
                        vx, vy = view(x, y)
                        selected = " sam-point-selected" if editor.selected == index else ""
                        html.append(f'<span class="sam-point sam-point-{label}{selected}" style="left: {vx:.3f}dp; top: {vy:.3f}dp;">{"+" if label else "−"}</span>')
                    box = editor.draft.box
                    if editor.box_drag and editor.drag_end:
                        box = editor.dragged_box(editor.drag_end)
                    elif editor.drag_start and editor.drag_end:
                        box = (*editor.drag_start, *editor.drag_end)
                    if box:
                        a, b = view(*box[:2]), view(*box[2:])
                        selected = " sam-box-selected" if editor.selected == "box" else ""
                        html.append(f'<div class="sam-box{selected}" style="left: {min(a[0], b[0]):.3f}dp; top: {min(a[1], b[1]):.3f}dp; width: {abs(a[0]-b[0]):.3f}dp; height: {abs(a[1]-b[1]):.3f}dp;"></div>')
                        if selected:
                            for x, y in ((a[0], a[1]), (b[0], a[1]), (a[0], b[1]), (b[0], b[1])):
                                html.append(f'<span class="sam-box-handle" style="left: {x:.3f}dp; top: {y:.3f}dp;"></span>')
            content = "".join(html)
            if content != self._sam_markers_rml:
                markers.set_inner_rml(content)
                self._sam_markers_rml = content

    # -- Color Picker --

    def _perform_color_pick(self, event):
        """Sample off-thread; changing either background requires an explicit command."""
        if self._sample_request is not None:
            return
        if self._displayed_index != self._current_index:
            return
        path = self._image_paths[self._current_index]
        viewport = self._doc.get_element_by_id("image-viewport") if self._doc else None
        layout = self._get_image_layout(path, viewport)
        if not layout:
            return

        ratio = layout["dp_ratio"]
        x = (float(event.get_parameter("mouse_x", "0")) - viewport.absolute_left) / ratio
        y = (float(event.get_parameter("mouse_y", "0")) - viewport.absolute_top) / ratio
        sx, sy = view_to_source(layout, self._rotation_quadrants, x, y)
        if not (0 <= sx < layout["width"] and 0 <= sy < layout["height"]):
            return
        ix, iy = int(sx), int(sy)

        future = self._metadata.sample(path, self._path_stat_cache.get(str(path)), ix, iy,
                                       lf.ui.sample_image_color)
        self._sample_request = (str(path), self._path_revision_cache.get(str(path)), future)
        self._sampled_color = None
        self._action_error = ""
        self._color_picker_active = False
        self._update_picker_cursor()
        self._mark_dirty()

    def _service_sample(self):
        if self._sample_request is None:
            return
        path, revision, future = self._sample_request
        if not future.done():
            self._request_async_update()
            return
        self._sample_request = None
        self._mark_dirty()
        if (not self._image_paths or str(self._image_paths[self._current_index]) != path
                or self._path_revision_cache.get(path) != revision):
            return
        try:
            self._sampled_color = future.result()
        except SourceChangedError:
            self._action_error = lf.ui.tr("image_preview.source_changed")
        except Exception as error:
            self._action_error = lf.ui.tr("image_preview.color_sample_failed").format(str(error))

    def _update_picker_cursor(self):
        """Toggle the picker-active CSS class on the image container."""
        if not self._doc:
            return
        container = self._doc.get_element_by_id("image-container")
        if container:
            if self._mask_editor.enabled and self._sam_method == "manual" and self._mask_editor.tool != "pan":
                container.set_attribute("class", "sam-prompt")
            elif self._color_picker_active:
                container.set_attribute("class", "picker-active")
            else:
                container.set_attribute("class", "")

    # -- Keyboard --

    def _on_keydown(self, event):
        key = int(event.get_parameter("key_identifier", "0"))
        target = event.target()
        while target is not None:
            if target.tag_name.lower() in ("input", "textarea", "select"):
                if target.id != "sam-approve" or key not in (KI_LEFT, KI_RIGHT):
                    return
            if target.has_attribute("contenteditable"):
                return
            target = target.parent()
        if self._mask_editor.enabled:
            keymap = lf.keymap
            if keymap.is_capturing():
                return
            code = native_key(key)
            action = (keymap.get_action_for_key(keymap.ToolMode.IMAGE_PREVIEW, code,
                                                self._sam_modifiers(event))
                      if code is not None else keymap.Action.NONE)
            if action == keymap.Action.IMAGE_PREVIEW_DELETE:
                self._sam_edit("remove_selected")
            elif action == keymap.Action.IMAGE_PREVIEW_GENERATE:
                self._sam_generate(semantic=self._sam_method == "semantic")
            else:
                action = keymap.Action.NONE
            if action != keymap.Action.NONE:
                event.stop_propagation()
                return
        if any(event.get_bool_parameter(name, False)
               for name in ("ctrl_key", "alt_key", "meta_key", "command_key")):
            return

        if key == KI_SPACE:
            target = event.target()
            if target is not None and target.tag_name == "button":
                return

        if key == KI_LEFT or key == KI_UP:
            self._navigate(-1)
            event.stop_propagation()
        elif key == KI_RIGHT or key == KI_DOWN:
            self._navigate(1)
            event.stop_propagation()
        elif key == KI_HOME:
            self._go_to_image(0)
            event.stop_propagation()
        elif key == KI_END:
            self._go_to_image(len(self._image_paths) - 1)
            event.stop_propagation()
        elif key == KI_F:
            self._toggle_fit()
            event.stop_propagation()
        elif key == KI_I:
            self._show_info = not self._show_info
            self._mark_dirty()
            event.stop_propagation()
        elif key == KI_T:
            self._show_filmstrip = not self._show_filmstrip
            self._mark_dirty()
            event.stop_propagation()
        elif key == KI_M:
            if self._has_valid_overlay():
                self._show_overlay = not self._show_overlay
                self._mark_dirty()
            event.stop_propagation()
        elif key == KI_1:
            self._actual_size()
            event.stop_propagation()
        elif key == KI_OEM_PLUS or key == KI_ADD:
            self._zoom_in()
            event.stop_propagation()
        elif key == KI_OEM_MINUS or key == KI_SUBTRACT:
            self._zoom_out()
            event.stop_propagation()
        elif key == KI_SPACE:
            if self._fit_to_window:
                self._actual_size()
            else:
                self._reset_view_and_refresh()
            event.stop_propagation()
        elif key == KI_R:
            self._reset_view()
            self._mark_dirty()
            event.stop_propagation()
        elif key == KI_C:
            self._toggle_picker()
            event.stop_propagation()
        elif key == KI_ESCAPE:
            if self._mask_editor.enabled:
                if self._mask_editor.dragging:
                    self._mask_editor.drag_start = self._mask_editor.drag_end = None
                    self._mask_editor.point_drag = None
                    self._mask_editor.box_drag = None
                    self._mark_dirty()
                elif self._mask_editor.selected is not None:
                    self._mask_editor.selected = None
                    self._mark_dirty()
                else:
                    self._toggle_sam()
            elif self._color_picker_active:
                self._color_picker_active = False
                self._update_picker_cursor()
                self._mark_dirty()
            elif self._show_help:
                self._show_help = False
                self._mark_dirty()
            else:
                self._close_panel()
            event.stop_propagation()

    # -- Helpers --

    def _get_zoom_display(self) -> str:
        ratio = max(1.0, lf.ui.get_ui_scale())
        if self._fit_to_window:
            layout = self._get_image_layout(self._image_paths[self._current_index]) if self._image_paths else None
            if layout:
                percentage = round(layout["scale"] * ratio * 100, 6)
                return f"{percentage:.0f}% · {lf.ui.tr('image_preview.fit')}"
            return lf.ui.tr("image_preview.fit")
        return f"{self._zoom * ratio * 100:.0f}%"

    @staticmethod
    def _get_channel_label(channels: int) -> str:
        if channels <= 0:
            return "—"
        if channels == 1:
            return lf.ui.tr("image_preview.channel_gray")
        if channels == 2:
            return lf.ui.tr("image_preview.channel_gray_alpha")
        if channels == 3:
            return "RGB"
        if channels == 4:
            return "RGBA"
        return str(channels)

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        if size_bytes >= 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.0f} KB"
        return f"{size_bytes} B"


def _set_text(doc, element_id: str, text: str):
    el = doc.get_element_by_id(element_id)
    if el:
        el.set_text(text)


def open_image_preview(image_paths: list[Path], mask_paths: list[Path],
                       start_index: int, camera_uids: list[int] | None = None,
                       camera_nodes: dict[int, tuple[int, str]] | None = None, assisted: bool = False, weights: str = "", sequence=None):
    global _pending_open
    _pending_open = (image_paths, mask_paths, start_index, camera_uids, camera_nodes)
    if sequence is not None:
        _pending_open += (assisted, weights, sequence)
    elif assisted:
        _pending_open += (True, weights)
    lf.ui.set_panel_enabled("lfs.image_preview", True)
    panel = _instance
    if panel is None:
        panel = lf.ui.get_panel_object("lfs.image_preview")
    pending = _pending_open
    if panel and pending is not None:
        _pending_open = None
        panel.open(*pending)


def open_extracted_frame_preview(paths, weights="", sequence=None):
    open_image_preview([Path(path) for path in paths], [None] * len(paths), 0, assisted=True, weights=weights, sequence=sequence)


def open_camera_preview_by_uid(cam_uid: int):
    scene = lf.get_scene()
    if not scene:
        return
    nodes = scene.get_nodes()
    target = None
    for node in nodes:
        if node.type == lf.scene.NodeType.CAMERA and node.camera_uid == cam_uid:
            target = node
            break
    if not target or not target.image_path:
        return

    parent = scene.get_node_by_id(target.parent_id) if target.parent_id >= 0 else None
    child_ids = parent.children if parent else [n.id for n in nodes if n.type == lf.scene.NodeType.CAMERA]

    nodes_by_id = {node.id: node for node in nodes}

    image_paths = []
    mask_paths = []
    camera_uids = []
    camera_nodes = {}
    start_index = 0
    for cid in child_ids:
        child = nodes_by_id.get(cid)
        if not child or child.type != lf.scene.NodeType.CAMERA or not child.image_path:
            continue
        if child.id == target.id:
            start_index = len(image_paths)
        image_paths.append(Path(child.image_path))
        mask_paths.append(Path(child.mask_path) if child.mask_path else None)
        camera_uids.append(child.camera_uid)
        camera_nodes[child.camera_uid] = (child.id, child.uuid)

    if image_paths:
        open_image_preview(image_paths, mask_paths, start_index, camera_uids, camera_nodes)
