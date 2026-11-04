# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Image preview panel using RmlUI floating window."""

import time
from math import gcd
from _lfs_path_io import Path
from typing import Optional
from urllib.parse import quote

import lichtfeld as lf
from . import rml_widgets as w
from .types import Panel
from .panels import panel_class
from .preview_geometry import THUMB_ROW_DP, visible_thumbnail_range, view_to_source, source_to_view
from .preview_metadata import PreviewMetadata, SourceChangedError
from .ui import RuntimeState
from .rml_keys import (
    KI_1, KI_ADD, KI_C, KI_DOWN, KI_END, KI_ESCAPE, KI_F, KI_HOME, KI_I,
    KI_LEFT, KI_M, KI_OEM_MINUS, KI_OEM_PLUS, KI_R, KI_RIGHT, KI_SPACE,
    KI_SUBTRACT, KI_T, KI_UP,
)

__lfs_panel_classes__ = ["ImagePreviewPanel"]
__lfs_panel_ids__ = ["lfs.image_preview"]


def __lfs_after_reload__(runtime):
    runtime.ui.on_open_camera_preview(open_camera_preview_by_uid)


ZOOM_MIN = 0.01
ZOOM_MAX = 10.0
PRECISE_SCROLL_STEP = 32.0
THUMB_MAX_PX = 256

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
        self._sampled_color = None
        self._sample_request = None
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
        self._last_training_params: tuple[int, int, bool] = (1, 0, False)
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
        model.bind_func("filmstrip_top_spacer_height", lambda: f"{self._filmstrip_top_spacer:.1f}dp")
        model.bind_func("filmstrip_bottom_spacer_height", lambda: f"{self._filmstrip_bottom_spacer:.1f}dp")
        self._handle = model.get_handle()
        self._filmstrip_range = None

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
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

        filmstrip = doc.get_element_by_id("filmstrip")
        if filmstrip:
            filmstrip.add_event_listener("click", self._on_filmstrip_click)
            filmstrip.add_event_listener("mousescroll", self._on_precise_scroll)
            filmstrip.add_event_listener("scroll", self._on_filmstrip_scroll)
            filmstrip.add_event_listener("mousedown", self._on_filmstrip_interaction)

        sidebar = doc.get_element_by_id("sidebar")
        if sidebar:
            sidebar.add_event_listener("mousescroll", self._on_precise_scroll)

        img_container = doc.get_element_by_id("image-container")
        if img_container:
            img_container.add_event_listener("mousescroll", self._on_wheel)
            img_container.add_event_listener("mousedown", self._on_img_mousedown)
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

    def on_update(self, doc):
        previous_state = (self._preview_state, self._preview_load_failed)
        self._service_sample()
        self._service_metadata()
        if not self._color_picker_active and not self._fit_to_window and self._image_paths and self._hover_image:
            lf.ui.set_mouse_cursor_hand()

        self._update_filmstrip(doc, bool(self._image_paths))

        current_params = self._get_training_params()
        if current_params != self._last_training_params:
            self._last_training_params = current_params
            self._dirty = True

        needs_redraw = False
        if (self._preview_state == "loading" or self._preview_pending) and not self._dirty:
            self._update_main_image(doc, bool(self._image_paths))
            needs_redraw = True
        if self._layout_pending and not self._dirty:
            self._layout_pending = False
            self._update_main_image(doc, bool(self._image_paths))
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
        self._sample_request = None
        self._metadata.close()
        self._unsubscribe_reactive_state()
        doc.remove_data_model("image_preview")
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
             camera_nodes: dict[int, tuple[int, str]] | None = None):
        self._reset_preview()
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
        self._current_index = max(0, min(start_index, len(image_paths) - 1))
        self._last_training_params = self._get_training_params()
        self._rotation_quadrants = 0
        self._reset_view()
        if self._pending_view_chrome:
            self._apply_view_chrome(self._pending_view_chrome)
            self._pending_view_chrome = None
        self._decorator_cache = {}
        self._update_picker_cursor()
        self._mark_dirty()

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

    def _get_training_params(self) -> tuple[int, int, bool]:
        try:
            dp = lf.dataset_params()
            if dp.has_params():
                op = lf.optimization_params()
                return (dp.resize_factor, dp.max_width, op.undistort)
        except Exception:
            pass
        return (1, 0, False)

    def _make_preview_url(self, path: Path, cam_uid: int,
                          thumb: int = 0, preview_mw: int = 0, dimensions=None, mask_overlay=False) -> str:
        rf, mw, ud = self._last_training_params
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

    def _on_img_mousedown(self, event):
        button = int(event.get_parameter("button", "0"))
        if button != 0:
            return
        target = event.target()
        while target is not None:
            if target.tag_name in ("button", "input", "select"):
                return
            target = target.parent()

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
        self._dragging = False

    def _on_img_mousemove(self, event):
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
        self._hover_image = False
        self._dragging = False

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

        extra = (self._show_help, bool(self._sampled_color), self._action_error,
                 RuntimeState.language_generation.value)
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
            uid = self._current_camera_uid()
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
        show_mask = self._preview_state == "ready" and self._show_overlay and self._has_valid_overlay()
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
               self._refresh_revision)
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
        for i in range(lo, hi):
            path = self._image_paths[i]
            uid = self._camera_uids[i] if i < len(self._camera_uids) else -1
            records.append({"index": i, "label": f"{i + 1:,}", "name": path.name,
                            "selected": i == self._current_index,
                            "decorator": f"image({self._make_preview_url(path, uid, THUMB_MAX_PX)} contain)"})
        self._handle.update_record_list("thumbs", records)
        self._handle.dirty("filmstrip_top_spacer_height")
        self._handle.dirty("filmstrip_bottom_spacer_height")
        if self._ensure_selection_visible and count:
            self._ensure_selection_visible = not (
                actual_top <= item_top + 1.0 and item_bottom <= actual_top + height + 1.0)
            if self._ensure_selection_visible:
                self._request_model_update()

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

        if not self._show_info:
            sidebar.set_attribute("class", "hidden")
            return
        sidebar.set_attribute("class", "")

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
                                   (self._preview_state != "ready" or self._sample_request is not None)))
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
            if self._color_picker_active:
                container.set_attribute("class", "picker-active")
            else:
                container.set_attribute("class", "")

    # -- Keyboard --

    def _on_keydown(self, event):
        target = event.target()
        while target is not None:
            if target.tag_name.lower() in ("input", "textarea", "select"):
                return
            if target.has_attribute("contenteditable"):
                return
            target = target.parent()
        if any(
                event.get_bool_parameter(name, False)
                for name in ("ctrl_key", "alt_key", "meta_key", "command_key")):
            return
        key = int(event.get_parameter("key_identifier", "0"))

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
            if self._color_picker_active:
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
                       camera_nodes: dict[int, tuple[int, str]] | None = None):
    global _pending_open
    _pending_open = (image_paths, mask_paths, start_index, camera_uids, camera_nodes)
    lf.ui.set_panel_enabled("lfs.image_preview", True)
    panel = _instance
    if panel is None:
        panel = lf.ui.get_panel_object("lfs.image_preview")
    pending = _pending_open
    if panel and pending is not None:
        _pending_open = None
        panel.open(*pending)


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
