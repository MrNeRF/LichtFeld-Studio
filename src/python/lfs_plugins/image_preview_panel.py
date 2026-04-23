# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Image preview panel using RmlUI floating window."""

import time
from math import gcd
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import lichtfeld as lf
from .types import Panel
from .rml_keys import (
    KI_1, KI_ADD, KI_C, KI_DOWN, KI_END, KI_ESCAPE, KI_F, KI_HOME, KI_I,
    KI_LEFT, KI_M, KI_OEM_MINUS, KI_OEM_PLUS, KI_R, KI_RIGHT, KI_SPACE,
    KI_SUBTRACT, KI_T, KI_UP,
)

__lfs_panel_classes__ = ["ImagePreviewPanel"]
__lfs_panel_ids__ = ["lfs.image_preview"]


def __lfs_after_reload__(runtime):
    runtime.ui.on_open_camera_preview(open_camera_preview_by_uid)


ZOOM_MIN = 0.1
ZOOM_MAX = 10.0
PRECISE_SCROLL_STEP = 32.0
CROSSFADE_DURATION = 0.2
SCROLL_DURATION = 0.15
FILMSTRIP_WINDOW = 40
THUMB_MAX_PX = 256

_instance = None
_RML_PATH_SAFE_CHARS = "/:._-~"


def _encode_rml_path(path: Path | str) -> str:
    return quote(str(path), safe=_RML_PATH_SAFE_CHARS)


class ImagePreviewPanel(Panel):
    id = "lfs.image_preview"
    label = "Image Preview"
    space = lf.ui.PanelSpace.FLOATING
    order = 98
    template = "rmlui/image_preview.rml"
    size = (900, 600)
    update_interval_ms = 16

    def __init__(self):
        global _instance
        _instance = self

        self._image_paths: list[Path] = []
        self._mask_paths: list[Optional[Path]] = []
        self._camera_uids: list[int] = []
        self._current_index = 0

        self._zoom = 1.0
        self._fit_to_window = True
        self._show_info = True
        self._show_filmstrip = True
        self._show_overlay = False

        self._pan_x = 0.0
        self._pan_y = 0.0
        self._rotation_quadrants = 0
        self._dragging = False
        self._drag_start_x = 0.0
        self._drag_start_y = 0.0
        self._drag_start_pan_x = 0.0
        self._drag_start_pan_y = 0.0
        self._hover_image = False
        self._color_picker_active = False

        self._doc = None
        self._dirty = True
        self._prev_image_index = -1

        self._active_layer = "a"
        self._crossfade_pending = False
        self._crossfade_start = 0.0

        self._scroll_target = None
        self._scroll_origin = 0.0
        self._scroll_start_time = 0.0

        self._image_info_cache: dict[str, tuple[int, int, int]] = {}
        self._last_training_params: tuple[int, int, bool] = (1, 0, False)
        self._decorator_cache: dict[str, str] = {}

    def _get_title(self) -> str:
        if self._image_paths:
            dirname = self._image_paths[0].parent.name
            return f"{dirname} \u00b7 {self._current_index + 1} / {len(self._image_paths)}"
        return lf.ui.tr("image_preview.title")

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("image_preview")
        if model is None:
            return

        model.bind_func("panel_label", lambda: self._get_title())
        model.bind_record_list("thumbs")
        self._handle = model.get_handle()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc

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
        ]:
            el = doc.get_element_by_id(eid)
            if el:
                el.add_event_listener("click", handler)

        cb_fit = doc.get_element_by_id("cb-fit")
        if cb_fit:
            cb_fit.add_event_listener("change", self._on_fit_checkbox_change)

        cb_mask = doc.get_element_by_id("cb-mask")
        if cb_mask:
            cb_mask.add_event_listener("change", self._on_mask_checkbox_change)

        filmstrip = doc.get_element_by_id("filmstrip")
        if filmstrip:
            filmstrip.add_event_listener("click", self._on_filmstrip_click)
            filmstrip.add_event_listener("mousescroll", self._on_precise_scroll)

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

        wf = doc.get_element_by_id("window-frame")

        doc.add_event_listener("keydown", self._on_keydown)
        doc.add_event_listener("resize", self._on_layout_resize)
        if wf:
            wf.add_event_listener("keydown", self._on_keydown)

        self._decorator_cache = {}
        self._dirty = True

    def on_update(self, doc):
        if not self._fit_to_window and self._image_paths and self._hover_image:
            lf.ui.set_mouse_cursor_hand()

        current_params = self._get_training_params()
        if current_params != self._last_training_params:
            self._last_training_params = current_params
            self._dirty = True

        needs_redraw = False

        if self._crossfade_pending:
            if time.monotonic() - self._crossfade_start >= CROSSFADE_DURATION:
                self._finalize_crossfade(doc)
            needs_redraw = True

        if self._scroll_target is not None:
            self._tick_scroll(doc)
            needs_redraw = True

        if self._dirty:
            self._dirty = False
            self._refresh_ui(doc)
            return True

        return needs_redraw

    def open(self, image_paths: list[Path], mask_paths: list[Optional[Path]],
             start_index: int, camera_uids: list[int] | None = None):
        if not image_paths:
            return

        self._image_paths = [p.resolve() for p in image_paths]
        self._mask_paths = [p.resolve() if p else None for p in mask_paths] if mask_paths else [None] * len(image_paths)
        self._camera_uids = camera_uids if camera_uids is not None else [-1] * len(image_paths)
        self._current_index = min(start_index, len(image_paths) - 1)
        self._last_training_params = self._get_training_params()
        self._rotation_quadrants = 0
        self._reset_view()
        self._dirty = True
        self._prev_image_index = -1
        self._crossfade_pending = False
        self._scroll_target = None
        self._decorator_cache = {}

    def _reset_pan(self):
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._dragging = False

    def _reset_view(self):
        self._zoom = 1.0
        self._fit_to_window = True
        self._reset_pan()

    def _refresh_immediately(self):
        if self._doc:
            self._dirty = False
            self._refresh_ui(self._doc)
        else:
            self._dirty = True

    def _navigate(self, delta: int):
        new_idx = self._current_index + delta
        if 0 <= new_idx < len(self._image_paths):
            self._current_index = new_idx
            self._reset_pan()
            self._refresh_immediately()

    def _go_to_image(self, index: int):
        if 0 <= index < len(self._image_paths):
            self._current_index = index
            self._reset_pan()
            self._refresh_immediately()

    def _toggle_fit(self):
        self._fit_to_window = not self._fit_to_window
        if self._fit_to_window:
            self._reset_view()
        self._dirty = True

    def _copy_path_to_clipboard(self):
        if self._image_paths:
            lf.ui.set_clipboard_text(str(self._image_paths[self._current_index]))

    def _current_camera_uid(self) -> int:
        if 0 <= self._current_index < len(self._camera_uids):
            return self._camera_uids[self._current_index]
        return -1

    def _current_camera_node(self):
        uid = self._current_camera_uid()
        if uid < 0:
            return None
        scene = lf.get_scene()
        for node in scene.get_nodes():
            if node.type == lf.scene.NodeType.CAMERA and node.camera_uid == uid:
                return node
        return None

    def _action_go_to_camera_view(self):
        uid = self._current_camera_uid()
        if uid >= 0:
            lf.ui.go_to_camera_view(uid)

    def _action_open_in_gt_compare(self):
        uid = self._current_camera_uid()
        if uid < 0:
            return
        lf.ui.go_to_camera_view(uid)
        if not lf.ui.is_gt_comparison_active():
            lf.ui.toggle_gt_comparison()

    def _action_toggle_training(self):
        node = self._current_camera_node()
        if node is None:
            return
        lf.set_camera_training_enabled(node.name, not node.training_enabled)
        self._dirty = True

    def _action_show_in_file_manager(self):
        if not self._image_paths:
            return
        lf.ui.reveal_in_file_manager(str(self._image_paths[self._current_index]))

    def _rotate(self, delta_quadrants: int):
        if not self._image_paths:
            return
        self._rotation_quadrants = (self._rotation_quadrants + delta_quadrants) % 4
        self._dirty = True

    def _zoom_in(self):
        self._zoom = min(ZOOM_MAX, self._zoom * 1.25)
        self._fit_to_window = False
        self._dirty = True

    def _zoom_out(self):
        self._zoom = max(ZOOM_MIN, self._zoom / 1.25)
        self._fit_to_window = False
        self._dirty = True

    def _has_valid_overlay(self) -> bool:
        if self._current_index >= len(self._mask_paths):
            return False
        mask_path = self._mask_paths[self._current_index]
        return mask_path is not None and mask_path.exists()

    def _close_panel(self):
        lf.ui.set_panel_enabled("lfs.image_preview", False)

    def _get_image_info(self, path: Path) -> tuple[int, int, int]:
        key = str(path)
        if key not in self._image_info_cache:
            try:
                self._image_info_cache[key] = lf.ui.get_image_info(key)
            except Exception:
                self._image_info_cache[key] = (0, 0, 0)
        return self._image_info_cache[key]

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
                          thumb: int = 0, preview_mw: int = 0) -> str:
        rf, mw, ud = self._last_training_params
        return (f"preview://cam={cam_uid}&thumb={thumb}"
                f"&rf={rf}&mw={mw}&pmw={preview_mw}"
                f"&ud={1 if ud else 0}&path={_encode_rml_path(path)}")

    def _on_filmstrip_click(self, event):
        el = event.target()
        while el:
            idx_str = el.get_attribute("data-index", "")
            if idx_str:
                self._go_to_image(int(idx_str))
                return
            el = el.parent()

    def _on_fit_checkbox_change(self, _event):
        cb = self._doc.get_element_by_id("cb-fit") if self._doc else None
        if cb:
            self._fit_to_window = cb.has_attribute("checked")
            if self._fit_to_window:
                self._reset_view()
            self._dirty = True

    def _on_mask_checkbox_change(self, _event):
        cb = self._doc.get_element_by_id("cb-mask") if self._doc else None
        if cb:
            self._show_overlay = cb.has_attribute("checked")
            self._dirty = True

    def _on_precise_scroll(self, event):
        scroll_el = event.current_target()
        if not scroll_el:
            return

        try:
            wheel_delta = float(event.get_parameter("wheel_delta_y", "0"))
        except (TypeError, ValueError):
            return

        max_scroll = max(0.0, scroll_el.scroll_height - scroll_el.client_height)
        if max_scroll <= 0.0:
            event.stop_propagation()
            return

        new_scroll = min(max(scroll_el.scroll_top + wheel_delta * PRECISE_SCROLL_STEP, 0.0), max_scroll)
        if abs(new_scroll - scroll_el.scroll_top) > 0.01:
            scroll_el.scroll_top = new_scroll

        event.stop_propagation()

    def _on_wheel(self, event):
        delta = float(event.get_parameter("wheel_delta_y", "0"))
        if delta == 0:
            return
        event.stop_propagation()
        if delta > 0:
            self._zoom = min(ZOOM_MAX, self._zoom * 1.15)
        else:
            self._zoom = max(ZOOM_MIN, self._zoom / 1.15)
        self._fit_to_window = False
        self._dirty = True

    def _on_img_mousedown(self, event):
        button = int(event.get_parameter("button", "0"))
        if button != 0:
            return

        if self._color_picker_active and self._image_paths:
            self._perform_color_pick(event)
            return

        if self._fit_to_window:
            return
        self._dragging = True
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
        self._pan_x = self._drag_start_pan_x + (mx - self._drag_start_x)
        self._pan_y = self._drag_start_pan_y + (my - self._drag_start_y)
        self._dirty = True

    def _on_img_mouseover(self, _event):
        self._hover_image = True

    def _on_img_mouseout(self, _event):
        self._hover_image = False
        self._dragging = False

    def _on_layout_resize(self, _event):
        self._dirty = True

    def _get_active_layer_id(self):
        return "main-image-a" if self._active_layer == "a" else "main-image-b"

    def _get_inactive_layer_id(self):
        return "main-image-b" if self._active_layer == "a" else "main-image-a"

    def _get_rotation_degrees(self) -> int:
        return (self._rotation_quadrants % 4) * 90

    def _get_image_layout(self, path: Path, viewport=None):
        if viewport is None:
            viewport = self._doc.get_element_by_id("image-viewport") if self._doc else None

        w, h, _ = self._get_image_info(path)
        if not viewport or w <= 0 or h <= 0:
            return None

        dp_ratio = max(1.0, lf.ui.get_ui_scale())
        vw = max(1, int(viewport.client_width / dp_ratio))
        vh = max(1, int(viewport.client_height / dp_ratio))
        rotated_w, rotated_h = (h, w) if self._rotation_quadrants % 2 else (w, h)

        if self._fit_to_window:
            scale = min(1.0, vw / rotated_w, vh / rotated_h)
        else:
            scale = self._zoom

        dw = max(1, int(round(w * scale)))
        dh = max(1, int(round(h * scale)))

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

        img_el.set_property("width", f"{dw}dp")
        img_el.set_property("height", f"{dh}dp")
        img_el.set_property("max-width", "none")
        img_el.set_property("max-height", "none")
        img_el.set_property("left", f"{int(round(ox))}dp")
        img_el.set_property("top", f"{int(round(oy))}dp")
        img_el.set_property("transform", f"rotate({self._get_rotation_degrees()}deg)")
        img_el.remove_property("margin-left")
        img_el.remove_property("margin-top")

    def _get_main_preview_max_width(self, doc, path: Path) -> int:
        if not self._fit_to_window:
            return 0

        viewport = doc.get_element_by_id("image-viewport") if doc else None
        if not viewport:
            return 0

        dp_ratio = max(1.0, lf.ui.get_ui_scale())
        vw = int(viewport.client_width / dp_ratio)
        vh = int(viewport.client_height / dp_ratio)
        if vw <= 1 or vh <= 1:
            return 0

        preview_max = max(vw, vh) * 1.25

        w, h, _ = self._get_image_info(path)
        if w > 0 and h > 0:
            preview_max = min(preview_max, max(w, h))

        return max(512, (int(preview_max) + 255) // 256 * 256)

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

        if hasattr(self, '_handle'):
            self._handle.dirty("panel_label")

    def _update_main_image(self, doc, has_images: bool):
        layer_a = doc.get_element_by_id("main-image-a")
        layer_b = doc.get_element_by_id("main-image-b")
        mask_img = doc.get_element_by_id("mask-overlay")
        no_text = doc.get_element_by_id("no-image-text")

        if not has_images:
            for layer in (layer_a, layer_b):
                if layer:
                    layer.set_attribute("class", "image-layer hidden")
            self._set_decorator(layer_a, "main-image-a", "none")
            self._set_decorator(layer_b, "main-image-b", "none")
            if mask_img:
                mask_img.set_attribute("class", "")
            self._set_decorator(mask_img, "mask-overlay", "none")
            if no_text:
                no_text.set_attribute("class", "")
                no_text.set_text(lf.ui.tr("image_preview.no_images_loaded"))
            self._prev_image_index = -1
            self._crossfade_pending = False
            return

        path = self._image_paths[self._current_index]
        uid = self._camera_uids[self._current_index] if self._current_index < len(self._camera_uids) else -1
        preview_mw = self._get_main_preview_max_width(doc, path)
        preview_dec = f"image({self._make_preview_url(path, uid, preview_mw=preview_mw)})"
        active_layer = doc.get_element_by_id(self._get_active_layer_id())
        inactive_layer = doc.get_element_by_id(self._get_inactive_layer_id())

        if no_text:
            no_text.set_attribute("class", "hidden")

        if self._prev_image_index == -1:
            if active_layer:
                self._set_decorator(active_layer, self._get_active_layer_id(), preview_dec)
                active_layer.set_attribute("class", "image-layer")
                self._apply_zoom(active_layer, path)
            if inactive_layer:
                inactive_layer.set_attribute("class", "image-layer hidden")
                self._set_decorator(inactive_layer, self._get_inactive_layer_id(), "none")
            self._prev_image_index = self._current_index
        elif self._prev_image_index != self._current_index:
            if self._crossfade_pending and inactive_layer:
                inactive_layer.set_attribute("class", "image-layer hidden")
                self._set_decorator(inactive_layer, self._get_inactive_layer_id(), "none")

            if inactive_layer:
                self._set_decorator(inactive_layer, self._get_inactive_layer_id(), preview_dec)
                inactive_layer.set_attribute("class", "image-layer")
                self._apply_zoom(inactive_layer, path)
            if active_layer:
                active_layer.set_attribute("class", "image-layer fading-out")

            self._active_layer = "b" if self._active_layer == "a" else "a"
            self._crossfade_pending = True
            self._crossfade_start = time.monotonic()
            self._prev_image_index = self._current_index
        else:
            if active_layer:
                self._set_decorator(active_layer, self._get_active_layer_id(), preview_dec)
                self._apply_zoom(active_layer, path)

        active_layer = doc.get_element_by_id(self._get_active_layer_id())
        show_mask = self._show_overlay and self._has_valid_overlay()
        if mask_img:
            if show_mask:
                mask_path = self._mask_paths[self._current_index]
                self._set_decorator(mask_img, "mask-overlay",
                                    f"image({_encode_rml_path(mask_path)})")
                mask_img.set_attribute("class", "visible")
                self._apply_zoom(mask_img, path)
            else:
                mask_img.set_attribute("class", "")
                self._set_decorator(mask_img, "mask-overlay", "none")

    def _finalize_crossfade(self, doc):
        outgoing_id = self._get_inactive_layer_id()
        outgoing = doc.get_element_by_id(outgoing_id)
        if outgoing:
            outgoing.set_attribute("class", "image-layer hidden")
            self._set_decorator(outgoing, outgoing_id, "none")
        self._crossfade_pending = False

    def _update_filmstrip(self, doc, has_images: bool):
        filmstrip = doc.get_element_by_id("filmstrip")
        if not filmstrip:
            return

        if not self._show_filmstrip:
            filmstrip.set_attribute("class", "hidden")
            return
        filmstrip.set_attribute("class", "")

        if not has_images:
            self._handle.update_record_list("thumbs", [])
            return

        half = FILMSTRIP_WINDOW // 2
        n = len(self._image_paths)
        lo = max(0, self._current_index - half)
        hi = min(n, self._current_index + half)

        records = []
        for i, path in enumerate(self._image_paths):
            if lo <= i < hi:
                uid = self._camera_uids[i] if i < len(self._camera_uids) else -1
                dec = f"image({self._make_preview_url(path, uid, THUMB_MAX_PX)})"
            else:
                dec = "none"
            records.append({
                "index": i,
                "label": f"{i + 1:02d}",
                "selected": i == self._current_index,
                "decorator": dec,
            })
        self._handle.update_record_list("thumbs", records)

        self._scroll_filmstrip_smooth(filmstrip, self._current_index)

    def _scroll_filmstrip_smooth(self, filmstrip, index: int):
        children = filmstrip.children()
        if index < 0 or index >= len(children):
            return
        el = children[index]
        item_top = el.offset_top
        item_bot = item_top + el.offset_height
        view_h = filmstrip.client_height
        if view_h <= 0:
            return
        st = filmstrip.scroll_top
        if item_top < st:
            target = item_top
        elif item_bot > st + view_h:
            target = item_bot - view_h
        else:
            return
        self._scroll_target = target
        self._scroll_origin = st
        self._scroll_start_time = time.monotonic()

    def _tick_scroll(self, doc):
        filmstrip = doc.get_element_by_id("filmstrip")
        if not filmstrip:
            self._scroll_target = None
            return
        t = min(1.0, (time.monotonic() - self._scroll_start_time) / SCROLL_DURATION)
        t = t * (2.0 - t)
        filmstrip.scroll_top = self._scroll_origin + (self._scroll_target - self._scroll_origin) * t
        if t >= 1.0:
            self._scroll_target = None

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
                "../icon/scene/visible.png"
                if node and node.training_enabled
                else "../icon/scene/hidden.png"
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
        _set_text(doc, "sidebar-view-label", tr("image_preview.view_section"))

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
                    _set_text(doc, eid, "")

            _set_text(doc, "sidebar-channels", self._get_channel_label(c))

            _set_text(doc, "sidebar-format", ext)
            if path.exists():
                _set_text(doc, "sidebar-size", self._format_size(path.stat().st_size))
            else:
                _set_text(doc, "sidebar-size", "")

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

        cb_fit = doc.get_element_by_id("cb-fit")
        if cb_fit:
            is_checked = cb_fit.has_attribute("checked")
            if self._fit_to_window and not is_checked:
                cb_fit.set_attribute("checked", "")
            elif not self._fit_to_window and is_checked:
                cb_fit.remove_attribute("checked")
        _set_text(doc, "cb-fit-label", tr("image_preview.fit_to_window"))

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

    def _update_status(self, doc, has_images: bool):
        ids = ("st-w", "st-h", "st-ch", "st-zoom", "st-counter")
        if not has_images:
            for sid in ids:
                _set_text(doc, sid, "")
            return

        path = self._image_paths[self._current_index]
        w, h, c = self._get_image_info(path)

        _set_text(doc, "st-w", f"W {w}" if w > 0 else "")
        _set_text(doc, "st-h", f"H {h}" if h > 0 else "")
        _set_text(doc, "st-ch", f"CH {c}")
        _set_text(doc, "st-zoom", f"{lf.ui.tr('image_preview.zoom')} {self._get_zoom_display()}")
        _set_text(doc, "st-counter", f"{self._current_index + 1} / {len(self._image_paths)}")

    # -- Color Picker --

    def _perform_color_pick(self, event):
        """Sample the average color at the clicked position and set as bg_color."""
        path = self._image_paths[self._current_index]
        viewport = self._doc.get_element_by_id("image-viewport") if self._doc else None
        layout = self._get_image_layout(path, viewport)
        if not layout:
            return

        w = layout["width"]
        h = layout["height"]
        dp_ratio = layout["dp_ratio"]
        scale = layout["scale"]
        ox = layout["offset_x"]
        oy = layout["offset_y"]
        cx = ox + layout["draw_width"] * 0.5
        cy = oy + layout["draw_height"] * 0.5

        # Get click position relative to the viewport
        vp_left = viewport.absolute_left
        vp_top = viewport.absolute_top
        mx = float(event.get_parameter("mouse_x", "0")) / dp_ratio - vp_left / dp_ratio
        my = float(event.get_parameter("mouse_y", "0")) / dp_ratio - vp_top / dp_ratio

        dx = mx - cx
        dy = my - cy
        rotation = self._rotation_quadrants % 4
        if rotation == 1:
            ux = dy
            uy = -dx
        elif rotation == 2:
            ux = -dx
            uy = -dy
        elif rotation == 3:
            ux = -dy
            uy = dx
        else:
            ux = dx
            uy = dy

        ix = int(round(ux / scale + w * 0.5))
        iy = int(round(uy / scale + h * 0.5))

        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return

        try:
            r, g, b = lf.ui.sample_image_color(str(path), ix, iy, 10)
        except Exception:
            return

        color = (r, g, b)
        try:
            params = lf.optimization_params()
            params.bg_color = color
        except Exception:
            pass
        try:
            rs = lf.get_render_settings()
            if rs:
                rs.set("background_color", color)
        except Exception:
            pass

        self._color_picker_active = False
        self._update_picker_cursor()
        self._dirty = True

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
        key = int(event.get_parameter("key_identifier", "0"))

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
            self._dirty = True
            event.stop_propagation()
        elif key == KI_T:
            self._show_filmstrip = not self._show_filmstrip
            self._dirty = True
            event.stop_propagation()
        elif key == KI_M:
            if self._has_valid_overlay():
                self._show_overlay = not self._show_overlay
                self._dirty = True
            event.stop_propagation()
        elif key == KI_1:
            self._zoom = 1.0
            self._fit_to_window = False
            self._reset_pan()
            self._dirty = True
            event.stop_propagation()
        elif key == KI_OEM_PLUS or key == KI_ADD:
            self._zoom_in()
            event.stop_propagation()
        elif key == KI_OEM_MINUS or key == KI_SUBTRACT:
            self._zoom_out()
            event.stop_propagation()
        elif key == KI_SPACE:
            if self._fit_to_window:
                self._fit_to_window = False
                self._zoom = 1.0
                self._reset_pan()
            else:
                self._reset_view()
            self._dirty = True
            event.stop_propagation()
        elif key == KI_R:
            self._reset_view()
            self._dirty = True
            event.stop_propagation()
        elif key == KI_C:
            if self._image_paths:
                self._color_picker_active = not self._color_picker_active
                self._update_picker_cursor()
                self._dirty = True
            event.stop_propagation()
        elif key == KI_ESCAPE:
            if self._color_picker_active:
                self._color_picker_active = False
                self._update_picker_cursor()
                self._dirty = True
            else:
                self._close_panel()
            event.stop_propagation()

    # -- Helpers --

    def _get_zoom_display(self) -> str:
        if self._fit_to_window:
            return lf.ui.tr("image_preview.fit")
        return f"{self._zoom * 100:.0f}%"

    @staticmethod
    def _get_channel_label(channels: int) -> str:
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
                       start_index: int, camera_uids: list[int] | None = None):
    if _instance:
        _instance.open(image_paths, mask_paths, start_index, camera_uids)
    lf.ui.set_panel_enabled("lfs.image_preview", True)


def open_camera_preview_by_uid(cam_uid: int):
    scene = lf.get_scene()
    if not scene:
        return
    target = None
    for node in scene.get_nodes():
        if node.type == lf.scene.NodeType.CAMERA and node.camera_uid == cam_uid:
            target = node
            break
    if not target or not target.image_path:
        return

    parent = scene.get_node_by_id(target.parent_id) if target.parent_id >= 0 else None
    child_ids = parent.children if parent else [n.id for n in scene.get_nodes() if n.type == lf.scene.NodeType.CAMERA]

    image_paths = []
    mask_paths = []
    camera_uids = []
    start_index = 0
    for cid in child_ids:
        child = scene.get_node_by_id(cid)
        if not child or child.type != lf.scene.NodeType.CAMERA or not child.image_path:
            continue
        if child.id == target.id:
            start_index = len(image_paths)
        image_paths.append(Path(child.image_path))
        mask_paths.append(Path(child.mask_path) if child.mask_path else None)
        camera_uids.append(child.camera_uid)

    if image_paths:
        open_image_preview(image_paths, mask_paths, start_index, camera_uids)
