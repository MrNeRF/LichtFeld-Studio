# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Image preview panel with modern UI: filmstrip, info sidebar, and status bar."""

from pathlib import Path
from typing import Optional

import lichtfeld as lf
from .types import Panel

ZOOM_MIN = 0.1
ZOOM_MAX = 10.0
ZOOM_FACTOR = 1.08

FILMSTRIP_THUMB_SIZE = 64
FILMSTRIP_PADDING = 4.0
FILMSTRIP_WIDTH = FILMSTRIP_THUMB_SIZE + FILMSTRIP_PADDING * 2

SIDEBAR_WIDTH = 180.0
SIDEBAR_PADDING = 8.0

TITLE_BAR_HEIGHT = 28.0
STATUS_BAR_HEIGHT = 24.0

OVERLAY_TINT = (1.0, 0.2, 0.2, 0.5)

_instance = None


def tr(key):
    result = lf.ui.tr(key)
    return result if result else key


class ThumbnailCache:
    """LRU cache for thumbnail textures."""

    MAX_CACHED = 50
    THUMB_SIZE = 64

    def __init__(self):
        self._cache: dict[str, tuple[int, int, int]] = {}
        self._lru: list[str] = []

    def get(self, path: str) -> Optional[tuple[int, int, int]]:
        if path in self._cache:
            self._lru.remove(path)
            self._lru.append(path)
            return self._cache[path]
        return None

    def load(self, path: str) -> tuple[int, int, int]:
        if path in self._cache:
            return self.get(path)

        while len(self._cache) >= self.MAX_CACHED:
            old = self._lru.pop(0)
            lf.ui.release_texture(self._cache[old][0])
            del self._cache[old]

        tex_id, w, h = lf.ui.load_thumbnail(path, self.THUMB_SIZE)
        self._cache[path] = (tex_id, w, h)
        self._lru.append(path)
        return (tex_id, w, h)

    def clear(self):
        for tex_id, _, _ in self._cache.values():
            if tex_id:
                lf.ui.release_texture(tex_id)
        self._cache.clear()
        self._lru.clear()


class ImagePreviewPanel(Panel):
    """Modern image preview with filmstrip, info sidebar, and status bar."""

    idname = "lfs.image_preview"
    label = "Image Preview"
    space = "FLOATING"
    order = 98
    options = {"DEFAULT_CLOSED"}

    def __init__(self):
        global _instance
        _instance = self

        self._image_paths: list[Path] = []
        self._mask_paths: list[Optional[Path]] = []
        self._current_index = 0

        self._current_texture: Optional[tuple[int, int, int]] = None
        self._overlay_texture: Optional[tuple[int, int, int]] = None
        self._thumbnail_cache = ThumbnailCache()

        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._fit_to_window = True
        self._show_info = True
        self._show_filmstrip = True
        self._show_overlay = False

        self._focus_next_frame = False
        self._filmstrip_scroll = 0.0

    def open(self, image_paths: list[Path], mask_paths: list[Optional[Path]], start_index: int):
        if not image_paths:
            return

        self._release_textures()
        self._image_paths = image_paths
        self._mask_paths = mask_paths if mask_paths else [None] * len(image_paths)
        self._current_index = min(start_index, len(image_paths) - 1)
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._fit_to_window = True
        self._focus_next_frame = True
        self._filmstrip_scroll = 0.0

        lf.ui.request_keyboard_capture("ImagePreview")
        self._load_current()
        self._preload_adjacent()

    def close(self):
        lf.ui.release_keyboard_capture("ImagePreview")
        self._release_textures()
        self._thumbnail_cache.clear()
        self._image_paths.clear()
        self._mask_paths.clear()

    def _release_textures(self):
        if self._current_texture and self._current_texture[0]:
            lf.ui.release_texture(self._current_texture[0])
        self._current_texture = None

        if self._overlay_texture and self._overlay_texture[0]:
            lf.ui.release_texture(self._overlay_texture[0])
        self._overlay_texture = None

    def _load_current(self):
        if self._current_texture and self._current_texture[0]:
            lf.ui.release_texture(self._current_texture[0])
            self._current_texture = None

        if not self._image_paths:
            return

        path = str(self._image_paths[self._current_index])

        if lf.ui.is_preload_ready(path):
            tex_id, w, h = lf.ui.get_preloaded_texture(path)
        else:
            tex_id, w, h = lf.ui.load_image_texture(path)

        if tex_id:
            self._current_texture = (tex_id, w, h)

        self._load_overlay()

    def _load_overlay(self):
        if self._overlay_texture and self._overlay_texture[0]:
            lf.ui.release_texture(self._overlay_texture[0])
            self._overlay_texture = None

        if self._current_index >= len(self._mask_paths):
            return

        mask_path = self._mask_paths[self._current_index]
        if not mask_path or not mask_path.exists():
            return

        tex_id, w, h = lf.ui.load_image_texture(str(mask_path))
        if tex_id:
            self._overlay_texture = (tex_id, w, h)

    def _preload_adjacent(self):
        for offset in [1, -1]:
            idx = self._current_index + offset
            if 0 <= idx < len(self._image_paths):
                lf.ui.preload_image_async(str(self._image_paths[idx]))

    def _has_valid_overlay(self) -> bool:
        if self._current_index >= len(self._mask_paths):
            return False
        mask_path = self._mask_paths[self._current_index]
        return mask_path is not None and mask_path.exists() and self._overlay_texture is not None

    def _calculate_display_size(self, available_w: float, available_h: float) -> tuple[float, float]:
        if not self._current_texture:
            return (0.0, 0.0)

        _, img_w, img_h = self._current_texture

        if self._fit_to_window:
            scale_x = available_w / img_w
            scale_y = available_h / img_h
            scale = min(scale_x, scale_y) * 0.95
            return (img_w * scale * self._zoom, img_h * scale * self._zoom)
        else:
            return (img_w * self._zoom, img_h * self._zoom)

    def _navigate(self, delta: int):
        new_idx = self._current_index + delta
        if 0 <= new_idx < len(self._image_paths):
            self._current_index = new_idx
            self._load_current()
            self._preload_adjacent()
            self._ensure_filmstrip_visible()

    def _go_to_image(self, index: int):
        if 0 <= index < len(self._image_paths):
            self._current_index = index
            self._pan_x = 0.0
            self._pan_y = 0.0
            self._load_current()
            self._preload_adjacent()
            self._ensure_filmstrip_visible()

    def _ensure_filmstrip_visible(self):
        item_height = FILMSTRIP_THUMB_SIZE + FILMSTRIP_PADDING
        target_y = self._current_index * item_height
        self._filmstrip_scroll = max(0, target_y - item_height * 2)

    def _get_zoom_display(self) -> str:
        if self._fit_to_window and self._current_texture:
            return tr("image_preview.fit")
        return f"{self._zoom * 100:.0f}%"

    def _format_size(self, size_bytes: int) -> str:
        if size_bytes >= 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.0f} KB"
        return f"{size_bytes} B"

    def _get_aspect_ratio_name(self, w: int, h: int) -> str:
        if h == 0:
            return tr("common.na")
        aspect = w / h
        if abs(aspect - 16 / 9) < 0.02:
            return "16:9"
        elif abs(aspect - 4 / 3) < 0.02:
            return "4:3"
        elif abs(aspect - 3 / 2) < 0.02:
            return "3:2"
        elif abs(aspect - 1.0) < 0.02:
            return "1:1"
        elif abs(aspect - 21 / 9) < 0.02:
            return "21:9"
        return f"{aspect:.2f}"

    def _close_panel(self):
        self.close()
        lf.ui.set_panel_enabled("lfs.image_preview", False)

    def draw(self, layout):
        if not self._image_paths:
            layout.text_colored(tr("image_preview.no_images_loaded"), (0.6, 0.6, 0.6, 1.0))
            return

        theme = lf.ui.theme()
        scale = layout.get_dpi_scale()

        content_w, content_h = layout.get_content_region_avail()

        nav_h = TITLE_BAR_HEIGHT * scale
        status_h = STATUS_BAR_HEIGHT * scale
        filmstrip_w = FILMSTRIP_WIDTH * scale if self._show_filmstrip else 0
        sidebar_w = SIDEBAR_WIDTH * scale if self._show_info else 0

        image_area_y = nav_h
        image_area_h = content_h - nav_h - status_h

        self._draw_nav_bar(layout, theme, content_w, nav_h, scale)

        layout.set_cursor_pos((0, nav_h))
        self._draw_filmstrip(layout, theme, filmstrip_w, image_area_h, scale)

        image_area_x = filmstrip_w
        image_area_w = content_w - filmstrip_w - sidebar_w
        self._draw_image_area(layout, theme, image_area_x, image_area_y, image_area_w, image_area_h, scale)

        if self._show_info:
            sidebar_x = content_w - sidebar_w
            layout.set_cursor_pos((sidebar_x, nav_h))
            self._draw_sidebar(layout, theme, sidebar_w, image_area_h, scale)

        layout.set_cursor_pos((0, content_h - status_h))
        self._draw_status_bar(layout, theme, content_w, status_h, scale)

        if layout.is_window_hovered() or layout.is_window_focused():
            layout.capture_mouse_from_app(True)

        self._handle_keyboard()

    def _draw_nav_bar(self, layout, theme, width: float, height: float, scale: float):
        padding = SIDEBAR_PADDING * scale
        text_height = layout.get_text_line_height()
        v_center = (height - text_height) / 2

        layout.set_cursor_pos((0, 0))
        layout.push_style_color("ChildBg", theme.palette.surface_bright)

        if layout.begin_child("##NavBar", (width, height), border=False):
            layout.set_cursor_pos((padding, v_center))

            if layout.small_button("<"):
                self._navigate(-1)
            layout.same_line()
            if layout.small_button(">"):
                self._navigate(1)

            layout.same_line(spacing=padding)
            filename = self._image_paths[self._current_index].name if self._image_paths else ""
            layout.label(f"{filename}  ({self._current_index + 1}/{len(self._image_paths)})")

        layout.end_child()
        layout.pop_style_color()

    def _draw_filmstrip(self, layout, theme, width: float, height: float, scale: float):
        if width <= 0:
            return

        padding = FILMSTRIP_PADDING * scale
        thumb_size = FILMSTRIP_THUMB_SIZE * scale
        item_height = thumb_size + padding

        layout.push_style_color("ChildBg", theme.palette.surface)
        layout.push_style_var_vec2("WindowPadding", (padding, padding))

        if layout.begin_child("##Filmstrip", (width, height), border=False):
            total_height = len(self._image_paths) * item_height
            max_scroll = max(0, total_height - height + padding)

            if layout.is_window_hovered():
                wheel = layout.get_mouse_wheel()
                if wheel != 0.0:
                    self._filmstrip_scroll -= wheel * item_height * 3
                    self._filmstrip_scroll = max(0, min(self._filmstrip_scroll, max_scroll))

            visible_start = max(0, int(self._filmstrip_scroll / item_height))
            visible_end = min(len(self._image_paths), int((self._filmstrip_scroll + height) / item_height) + 1)

            for i in range(visible_start, visible_end):
                y_pos = padding + i * item_height - self._filmstrip_scroll

                layout.set_cursor_pos((padding, y_pos))
                layout.push_id_int(i)

                is_selected = i == self._current_index
                if is_selected:
                    layout.push_style_color("Button", theme.palette.primary_dim)
                    layout.push_style_color("ButtonHovered", theme.palette.primary)
                    layout.push_style_color("ButtonActive", theme.palette.primary)

                path_str = str(self._image_paths[i])
                thumb_data = self._thumbnail_cache.get(path_str)
                if thumb_data is None:
                    thumb_data = self._thumbnail_cache.load(path_str)

                tex_id, tw, th = thumb_data
                if tex_id:
                    if layout.image_button(f"##thumb{i}", tex_id, (thumb_size, thumb_size)):
                        self._go_to_image(i)
                else:
                    if layout.button(f"##thumb{i}", (thumb_size, thumb_size)):
                        self._go_to_image(i)

                if layout.is_item_hovered():
                    layout.set_tooltip(self._image_paths[i].name)

                if is_selected:
                    layout.pop_style_color(3)

                layout.pop_id()

        layout.end_child()
        layout.pop_style_var()
        layout.pop_style_color()

    def _draw_image_area(self, layout, theme, x: float, y: float, width: float, height: float, scale: float):
        layout.set_cursor_pos((x, y))
        layout.push_style_color("ChildBg", theme.palette.background)

        if layout.begin_child("##ImageArea", (width, height), border=False):
            if not self._current_texture:
                cw, ch = layout.get_content_region_avail()
                layout.set_cursor_pos((cw / 2 - 40, ch / 2))
                layout.label(tr("image_preview.no_image"))
            else:
                cw, ch = layout.get_content_region_avail()
                display_w, display_h = self._calculate_display_size(cw, ch)

                x_offset = (cw - display_w) / 2 + self._pan_x
                y_offset = (ch - display_h) / 2 + self._pan_y

                layout.set_cursor_pos((0, 0))
                layout.invisible_button("##ImageInteract", (cw, ch))
                is_hovered = layout.is_item_hovered()
                is_active = layout.is_item_active()

                layout.set_cursor_pos((x_offset, y_offset))
                tex_id, _, _ = self._current_texture
                layout.image(tex_id, (display_w, display_h))

                if self._show_overlay and self._has_valid_overlay():
                    layout.set_cursor_pos((x_offset, y_offset))
                    overlay_tex_id, _, _ = self._overlay_texture
                    layout.image(overlay_tex_id, (display_w, display_h), OVERLAY_TINT)

                self._handle_image_input(layout, is_hovered, is_active)

        layout.end_child()
        layout.pop_style_color()

    def _draw_sidebar(self, layout, theme, width: float, height: float, scale: float):
        padding = SIDEBAR_PADDING * scale

        layout.push_style_color("ChildBg", theme.palette.surface)
        layout.push_style_var_vec2("WindowPadding", (padding, padding))
        layout.push_style_var_vec2("ItemSpacing", (4 * scale, 2 * scale))

        if layout.begin_child("##Sidebar", (width, height), border=False):
            label_color = theme.palette.text_dim
            content_width = width - padding * 3

            layout.indent(padding)

            if self._current_texture:
                _, w, h = self._current_texture
                mp = (w * h) / 1e6

                layout.text_colored(lf.ui.tr("image_preview.image_section"), label_color)
                layout.label(f"{w} x {h}")
                layout.label(f"{mp:.1f} MP · {self._get_aspect_ratio_name(w, h)}")

                layout.spacing()
                layout.separator()
                layout.spacing()

            if self._image_paths:
                path = self._image_paths[self._current_index]
                ext = path.suffix[1:].upper() if path.suffix else "?"

                layout.text_colored(lf.ui.tr("image_preview.file_section"), label_color)
                if path.exists():
                    size_str = self._format_size(path.stat().st_size)
                    layout.label(f"{size_str} · {ext}")
                else:
                    layout.label(ext)

                parent_str = str(path.parent)
                max_chars = int(content_width / (6 * scale))
                if len(parent_str) > max_chars:
                    parent_str = "..." + parent_str[-(max_chars - 3):]
                layout.text_colored(parent_str, label_color)

                layout.spacing()
                layout.separator()
                layout.spacing()

            layout.text_colored(lf.ui.tr("image_preview.view_section"), label_color)
            layout.label(f"{tr('image_preview.zoom')}: {self._get_zoom_display()}")
            changed, self._fit_to_window = layout.checkbox(lf.ui.tr("image_preview.fit_to_window"), self._fit_to_window)

            if self._has_valid_overlay():
                layout.spacing()
                layout.separator()
                layout.spacing()

                layout.text_colored(lf.ui.tr("image_preview.mask_section"), label_color)
                changed, self._show_overlay = layout.checkbox(lf.ui.tr("image_preview.show_mask_overlay"), self._show_overlay)
                mask_path = self._mask_paths[self._current_index]
                if mask_path:
                    name = mask_path.name
                    max_chars = int(content_width / (6 * scale))
                    if len(name) > max_chars:
                        name = name[:max_chars - 3] + "..."
                    layout.text_colored(name, label_color)

            layout.unindent(padding)

        layout.end_child()
        layout.pop_style_var(2)
        layout.pop_style_color()

    def _draw_status_bar(self, layout, theme, width: float, height: float, scale: float):
        padding = SIDEBAR_PADDING * scale
        text_height = layout.get_text_line_height()
        v_center = (height - text_height) / 2

        layout.push_style_color("ChildBg", theme.palette.surface_bright)

        if layout.begin_child("##StatusBar", (width, height), border=False):
            layout.set_cursor_pos((padding, v_center))

            if self._current_texture:
                _, w, h = self._current_texture
                mp = (w * h) / 1e6
                size_str = ""
                if self._image_paths:
                    path = self._image_paths[self._current_index]
                    if path.exists():
                        size_str = f" · {self._format_size(path.stat().st_size)}"
                layout.label(f"{w}x{h} · {mp:.1f} MP{size_str}")

            zoom_text = f"{tr('image_preview.zoom')}: {self._get_zoom_display()}"
            fit_label = tr("image_preview.fit")
            fit_text_w, _ = layout.calc_text_size(fit_label)
            fit_btn_width = max(30 * scale, fit_text_w + 12 * scale)
            zoom_width, _ = layout.calc_text_size(zoom_text)
            right_content_width = zoom_width + fit_btn_width + padding * 3

            layout.same_line(offset=width - right_content_width)
            layout.label(zoom_text)

            layout.same_line()
            if layout.small_button(fit_label):
                self._fit_to_window = True
                self._zoom = 1.0

        layout.end_child()
        layout.pop_style_color()

    def _handle_image_input(self, layout, is_hovered: bool, is_active: bool):
        if not layout.is_window_hovered():
            return

        if is_hovered:
            wheel = layout.get_mouse_wheel()
            if wheel != 0.0:
                old_zoom = self._zoom
                if wheel > 0:
                    self._zoom = min(ZOOM_MAX, self._zoom * ZOOM_FACTOR)
                else:
                    self._zoom = max(ZOOM_MIN, self._zoom / ZOOM_FACTOR)
                if old_zoom != self._zoom:
                    self._fit_to_window = False

            if layout.is_mouse_double_clicked(0):
                self._zoom = 1.0
                self._pan_x = 0.0
                self._pan_y = 0.0
                self._fit_to_window = True

        if is_active and layout.is_mouse_dragging(0):
            dx, dy = layout.get_mouse_delta()
            self._pan_x += dx
            self._pan_y += dy

    def _handle_keyboard(self):
        if lf.ui.is_key_pressed(lf.ui.Key.LEFT, repeat=False):
            self._navigate(-1)
        if lf.ui.is_key_pressed(lf.ui.Key.RIGHT, repeat=False):
            self._navigate(1)
        if lf.ui.is_key_pressed(lf.ui.Key.UP, repeat=False):
            self._navigate(-1)
        if lf.ui.is_key_pressed(lf.ui.Key.DOWN, repeat=False):
            self._navigate(1)
        if lf.ui.is_key_pressed(lf.ui.Key.HOME):
            self._go_to_image(0)
        if lf.ui.is_key_pressed(lf.ui.Key.END):
            self._go_to_image(len(self._image_paths) - 1)

        if lf.ui.is_key_pressed(lf.ui.Key.F):
            self._fit_to_window = not self._fit_to_window
        if lf.ui.is_key_pressed(lf.ui.Key.I):
            self._show_info = not self._show_info
        if lf.ui.is_key_pressed(lf.ui.Key.T):
            self._show_filmstrip = not self._show_filmstrip
        if lf.ui.is_key_pressed(lf.ui.Key.M) and self._has_valid_overlay():
            self._show_overlay = not self._show_overlay

        if lf.ui.is_key_pressed(lf.ui.Key._1):
            self._zoom = 1.0
            self._fit_to_window = False
        if lf.ui.is_key_pressed(lf.ui.Key.EQUAL):
            self._zoom = min(ZOOM_MAX, self._zoom * 1.25)
            self._fit_to_window = False
        if lf.ui.is_key_pressed(lf.ui.Key.MINUS):
            self._zoom = max(ZOOM_MIN, self._zoom / 1.25)
            self._fit_to_window = False
        if lf.ui.is_key_pressed(lf.ui.Key.SPACE):
            if self._fit_to_window:
                self._fit_to_window = False
                self._zoom = 1.0
            else:
                self._fit_to_window = True

        if lf.ui.is_key_pressed(lf.ui.Key.R):
            self._zoom = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0
        if lf.ui.is_key_pressed(lf.ui.Key.ESCAPE):
            self._close_panel()


def open_image_preview(image_paths: list[Path], mask_paths: list[Path], start_index: int):
    """Public API to open image preview with specific images."""
    if _instance:
        _instance.open(image_paths, mask_paths, start_index)
    lf.ui.set_panel_enabled("lfs.image_preview", True)
