"""Viewport overlay controllers for retained status dialogs and custom drawing."""

import math

import lichtfeld as lf

from .. import toolbar as viewport_toolbar

_HOOK_PANEL = "viewport_overlay"
_DOCUMENT_SECTION = "document"
_DRAW_SECTION = "draw"
_HOOK_POSITION = "append"

_MODEL_NAME = "viewport_overlay_status"
_MODEL_MARKER = "data-viewport-overlay-status-bound"
_document_controller = None
_hook_registered = False

_INSET = 30.0
_CORNER_RADIUS = 16.0
_GLOW_MAX = 8.0
_PULSE_SPEED = 3.0
_BOUNCE_SPEED = 4.0
_BOUNCE_AMOUNT = 5.0

_ZONE_PADDING = 120.0
_DASH_LENGTH = 12.0
_GAP_LENGTH = 8.0
_BORDER_THICKNESS = 2.0
_ICON_SIZE = 48.0
_ANIM_SPEED = 30.0
_MIN_VIEWPORT_SIZE = 200.0
_AUTO_DISMISS_DELAY = 3.0

_OVERLAY_FLAGS = (
    lf.ui.UILayout.WindowFlags.NoTitleBar
    | lf.ui.UILayout.WindowFlags.NoResize
    | lf.ui.UILayout.WindowFlags.NoMove
    | lf.ui.UILayout.WindowFlags.NoScrollbar
    | lf.ui.UILayout.WindowFlags.NoInputs
    | lf.ui.UILayout.WindowFlags.NoBackground
    | lf.ui.UILayout.WindowFlags.NoFocusOnAppearing
    | lf.ui.UILayout.WindowFlags.NoBringToFrontOnFocus
)


def _clamp_progress(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _viewport_bottom_inset(layout, base_inset):
    bottom_inset = base_inset
    if lf.ui.is_sequencer_visible():
        dp = layout.get_dpi_scale()
        seq_state = lf.ui.get_sequencer_state()
        film_strip_h = 56.0 if (seq_state and seq_state.show_film_strip) else 0.0
        seq_height = 162.0 * dp + film_strip_h
        bottom_inset = max(base_inset, seq_height + 8.0)
    return bottom_inset


class _OverlayDocumentController:
    def __init__(self):
        self.reset()

    def reset(self):
        self._handle = None
        self._import_state = {}
        self._video_state = {}
        self._last_import_signature = None
        self._last_video_signature = None
        viewport_toolbar.reset_overlay_state()

    def update(self, doc=None):
        if doc is None or not hasattr(doc, "get_element_by_id"):
            doc = lf.ui.rml.get_document(_HOOK_PANEL)
        if doc is None:
            return

        if not self._ensure_model(doc):
            return

        import_state = self._get_import_state()
        video_state = self._get_video_state()

        seconds_since = import_state.get("seconds_since_completion", 0.0)
        import_signature = (
            import_state.get("active", False),
            import_state.get("show_completion", False),
            import_state.get("success", False),
            import_state.get("dataset_type", ""),
            import_state.get("path", ""),
            import_state.get("progress", 0.0),
            import_state.get("stage", ""),
            import_state.get("num_images", 0),
            import_state.get("num_points", 0),
            import_state.get("error", ""),
            round(seconds_since, 1),
        )
        video_signature = (
            video_state.get("active", False),
            video_state.get("progress", 0.0),
            video_state.get("current_frame", 0),
            video_state.get("total_frames", 0),
            video_state.get("stage", ""),
        )

        needs_dirty = False

        if import_signature != self._last_import_signature:
            self._import_state = import_state
            self._last_import_signature = import_signature
            needs_dirty = True

        if (import_state.get("show_completion", False)
                and not import_state.get("active", False)
                and import_state.get("success", False)
                and seconds_since >= _AUTO_DISMISS_DELAY):
            lf.ui.dismiss_import()

        if video_signature != self._last_video_signature:
            self._video_state = video_state
            self._last_video_signature = video_signature
            needs_dirty = True

        viewport_toolbar.update_overlay(doc)

        if needs_dirty:
            self._handle.dirty_all()

    def _ensure_model(self, doc):
        body = doc.get_element_by_id("overlay-body")
        if body is None:
            return False

        if self._handle is not None and body.get_attribute(_MODEL_MARKER, "") == "1":
            if body.get_attribute("data-model", "") != _MODEL_NAME:
                body.set_attribute("data-model", _MODEL_NAME)
            return True

        self._handle = None
        doc.remove_data_model(_MODEL_NAME)
        body.remove_attribute(_MODEL_MARKER)
        viewport_toolbar.reset_overlay_state()

        model = doc.create_data_model(_MODEL_NAME)
        if model is None:
            return False

        model.bind_func("show_import_overlay", self._show_import_overlay)
        model.bind_func("show_import_backdrop", lambda: self._import_state.get("active", False))
        model.bind_func("import_title", self._import_title)
        model.bind_func("import_title_class", self._import_title_class)
        model.bind_func("show_import_path", lambda: bool(self._import_state.get("path", "")))
        model.bind_func("import_path", self._import_path)
        model.bind_func("show_import_progress", self._show_import_progress)
        model.bind_func("import_progress_value", self._import_progress_value)
        model.bind_func("import_progress_pct", self._import_progress_pct)
        model.bind_func("show_import_stage", self._show_import_stage)
        model.bind_func("import_stage", lambda: self._import_state.get("stage", ""))
        model.bind_func("show_import_counts", self._show_import_counts)
        model.bind_func("import_counts", self._import_counts)
        model.bind_func("show_import_error", lambda: bool(self._import_state.get("error", "")))
        model.bind_func("import_error", lambda: self._import_state.get("error", ""))
        model.bind_func("show_import_dismiss", self._show_import_dismiss)
        model.bind_func("import_dismiss_label", lambda: lf.ui.tr("common.ok"))

        model.bind_func("show_video_overlay", lambda: self._video_state.get("active", False))
        model.bind_func("video_title", lambda: lf.ui.tr("progress.exporting_video"))
        model.bind_func("video_progress_value", self._video_progress_value)
        model.bind_func("video_progress_pct", self._video_progress_pct)
        model.bind_func("video_frame_text", self._video_frame_text)
        model.bind_func("show_video_stage", lambda: bool(self._video_state.get("stage", "")))
        model.bind_func("video_stage", lambda: self._video_state.get("stage", ""))
        model.bind_func("video_cancel_label", lambda: lf.ui.tr("common.cancel"))

        viewport_toolbar.bind_overlay_model(model)

        model.bind_event("overlay_action", self._on_overlay_action)
        self._handle = model.get_handle()
        viewport_toolbar.attach_overlay_model_handle(self._handle)
        body.set_attribute("data-model", _MODEL_NAME)
        body.set_attribute(_MODEL_MARKER, "1")
        self._handle.dirty_all()
        return True

    def _get_import_state(self):
        if not hasattr(lf.ui, "get_import_state"):
            return {}

        return dict(lf.ui.get_import_state())

    def _get_video_state(self):
        if not hasattr(lf.ui, "get_video_export_state"):
            return {}
        return dict(lf.ui.get_video_export_state())

    def _show_import_overlay(self):
        state = self._import_state
        return state.get("active", False) or state.get("show_completion", False)

    def _show_import_progress(self):
        state = self._import_state
        if state.get("active", False):
            return True
        return (state.get("show_completion", False)
                and state.get("success", False))

    def _show_import_stage(self):
        return self._import_state.get("active", False) and bool(self._import_state.get("stage", ""))

    def _show_import_counts(self):
        if self._import_state.get("active", False):
            return False
        return (self._import_state.get("num_images", 0) > 0 or
                self._import_state.get("num_points", 0) > 0)

    def _show_import_dismiss(self):
        state = self._import_state
        return (state.get("show_completion", False)
                and not state.get("active", False)
                and not state.get("success", False))

    def _import_title(self):
        state = self._import_state
        show_completion = state.get("show_completion", False)
        if show_completion and not state.get("active", False):
            if state.get("success", False):
                return lf.ui.tr("progress.import_complete_title")
            return lf.ui.tr("progress.import_failed_title")
        dataset_type = state.get("dataset_type", "dataset") or "dataset"
        return lf.ui.tr("progress.importing").replace("%s", dataset_type)

    def _import_title_class(self):
        show_completion = self._import_state.get("show_completion", False)
        if not show_completion or self._import_state.get("active", False):
            return ""
        return "status-success" if self._import_state.get("success", False) else "status-error"

    def _import_path(self):
        path_str = self._import_state.get("path", "")
        return f"Path: {path_str}" if path_str else ""

    def _import_progress_value(self):
        state = self._import_state
        if state.get("show_completion", False) and state.get("success", False):
            return "1"
        return str(_clamp_progress(state.get("progress", 0.0)))

    def _import_progress_pct(self):
        state = self._import_state
        if state.get("show_completion", False) and state.get("success", False):
            return "100%"
        return f"{_clamp_progress(state.get('progress', 0.0)) * 100:.0f}%"

    def _import_counts(self):
        return f"{self._import_state.get('num_images', 0)} images, {self._import_state.get('num_points', 0)} points"

    def _video_progress_value(self):
        return str(_clamp_progress(self._video_state.get("progress", 0.0)))

    def _video_progress_pct(self):
        return f"{_clamp_progress(self._video_state.get('progress', 0.0)) * 100:.0f}%"

    def _video_frame_text(self):
        return f"Frame {self._video_state.get('current_frame', 0)} / {self._video_state.get('total_frames', 0)}"

    def _on_overlay_action(self, _handle, _ev, args):
        if not args:
            return

        action = str(args[0])
        if action == "dismiss_import":
            lf.ui.dismiss_import()
        elif action == "cancel_video_export":
            lf.ui.cancel_video_export()


def _draw_empty_state_overlay(layout):
    if not lf.ui.is_scene_empty() or lf.ui.is_drag_hovering() or lf.ui.is_startup_visible():
        return
    if hasattr(lf.ui, "get_import_state"):
        state = dict(lf.ui.get_import_state())
        if state.get("active", False) or state.get("show_completion", False):
            return

    vp_x, vp_y = layout.get_viewport_pos()
    vp_w, vp_h = layout.get_viewport_size()
    if vp_w < _MIN_VIEWPORT_SIZE or vp_h < _MIN_VIEWPORT_SIZE:
        return

    layout.set_next_window_pos((vp_x, vp_y))
    layout.set_next_window_size((vp_w, vp_h))

    if not layout.begin_window("##EmptyStateOverlay", _OVERLAY_FLAGS):
        layout.end_window()
        return

    theme = lf.ui.theme()
    border_color = theme.palette.overlay_border
    icon_color = theme.palette.overlay_icon
    title_color = theme.palette.overlay_text
    subtitle_color = theme.palette.overlay_text_dim
    hint_color = (subtitle_color[0], subtitle_color[1], subtitle_color[2], 0.5)

    center_x = vp_x + vp_w * 0.5
    center_y = vp_y + vp_h * 0.5
    zone_min_x = vp_x + _ZONE_PADDING
    zone_min_y = vp_y + _ZONE_PADDING
    zone_max_x = vp_x + vp_w - _ZONE_PADDING
    zone_max_y = vp_y + vp_h - _viewport_bottom_inset(layout, _ZONE_PADDING)

    dash_offset = (lf.ui.get_time() * _ANIM_SPEED) % (_DASH_LENGTH + _GAP_LENGTH)

    def draw_dashed_line(start_x, start_y, end_x, end_y):
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return
        nx, ny = dx / length, dy / length
        pos = -dash_offset
        while pos < length:
            d0 = max(0.0, pos)
            d1 = min(length, pos + _DASH_LENGTH)
            if d1 > d0:
                layout.draw_window_line(
                    start_x + nx * d0,
                    start_y + ny * d0,
                    start_x + nx * d1,
                    start_y + ny * d1,
                    border_color,
                    _BORDER_THICKNESS,
                )
            pos += _DASH_LENGTH + _GAP_LENGTH

    draw_dashed_line(zone_min_x, zone_min_y, zone_max_x, zone_min_y)
    draw_dashed_line(zone_max_x, zone_min_y, zone_max_x, zone_max_y)
    draw_dashed_line(zone_max_x, zone_max_y, zone_min_x, zone_max_y)
    draw_dashed_line(zone_min_x, zone_max_y, zone_min_x, zone_min_y)

    icon_y = center_y - 50.0
    layout.draw_window_rect(
        center_x - _ICON_SIZE * 0.5,
        icon_y - _ICON_SIZE * 0.3,
        center_x + _ICON_SIZE * 0.5,
        icon_y + _ICON_SIZE * 0.4,
        icon_color,
        2.0,
    )
    layout.draw_window_line(
        center_x - _ICON_SIZE * 0.5,
        icon_y - _ICON_SIZE * 0.3,
        center_x - _ICON_SIZE * 0.2,
        icon_y - _ICON_SIZE * 0.5,
        icon_color,
        2.0,
    )
    layout.draw_window_line(
        center_x - _ICON_SIZE * 0.2,
        icon_y - _ICON_SIZE * 0.5,
        center_x + _ICON_SIZE * 0.1,
        icon_y - _ICON_SIZE * 0.5,
        icon_color,
        2.0,
    )
    layout.draw_window_line(
        center_x + _ICON_SIZE * 0.1,
        icon_y - _ICON_SIZE * 0.5,
        center_x + _ICON_SIZE * 0.2,
        icon_y - _ICON_SIZE * 0.3,
        icon_color,
        2.0,
    )

    title = lf.ui.tr("startup.drop_files_title")
    subtitle = lf.ui.tr("startup.drop_files_subtitle")
    hint = lf.ui.tr("startup.drop_files_hint")

    title_w, _ = layout.calc_text_size(title)
    subtitle_w, _ = layout.calc_text_size(subtitle)
    hint_w, _ = layout.calc_text_size(hint)

    layout.draw_window_text(center_x - title_w * 0.5, center_y + 10.0, title, title_color)
    layout.draw_window_text(center_x - subtitle_w * 0.5, center_y + 40.0, subtitle, subtitle_color)
    layout.draw_window_text(center_x - hint_w * 0.5, center_y + 70.0, hint, hint_color)

    layout.end_window()


def _draw_drag_drop_overlay(layout):
    if not lf.ui.is_drag_hovering() or lf.ui.is_startup_visible():
        return

    vp_x, vp_y = layout.get_viewport_pos()
    vp_w, vp_h = layout.get_viewport_size()

    layout.set_next_window_pos((vp_x, vp_y))
    layout.set_next_window_size((vp_w, vp_h))

    if not layout.begin_window("##DragDropOverlay", _OVERLAY_FLAGS):
        layout.end_window()
        return

    theme = lf.ui.theme()
    primary = theme.palette.primary
    primary_dim = theme.palette.primary_dim
    overlay_text = theme.palette.overlay_text
    overlay_text_dim = theme.palette.overlay_text_dim

    overlay_color = (primary_dim[0], primary_dim[1], primary_dim[2], 0.7)
    fill_color = (primary[0], primary[1], primary[2], 0.23)

    win_max_x = vp_x + vp_w
    win_max_y = vp_y + vp_h
    zone_min_x = vp_x + _INSET
    zone_min_y = vp_y + _INSET
    zone_max_x = win_max_x - _INSET
    zone_max_y = win_max_y - _viewport_bottom_inset(layout, _INSET)
    center_x = vp_x + vp_w * 0.5
    center_y = vp_y + vp_h * 0.5

    t = lf.ui.get_time()
    pulse = 0.5 + 0.5 * math.sin(t * _PULSE_SPEED)

    layout.draw_window_rect_filled(vp_x, vp_y, win_max_x, win_max_y, overlay_color)

    glow_color = (primary[0], primary[1], primary[2], 0.16 * pulse)
    glow_size = _GLOW_MAX
    while glow_size > 0:
        layout.draw_window_rect_rounded(
            zone_min_x - glow_size,
            zone_min_y - glow_size,
            zone_max_x + glow_size,
            zone_max_y + glow_size,
            glow_color,
            _CORNER_RADIUS + glow_size,
            2.0,
        )
        glow_size -= 2.0

    border_alpha = 0.7 + 0.3 * pulse
    border_color = (primary[0], primary[1], primary[2], border_alpha)
    layout.draw_window_rect_rounded(
        zone_min_x,
        zone_min_y,
        zone_max_x,
        zone_max_y,
        border_color,
        _CORNER_RADIUS,
        3.0,
    )
    layout.draw_window_rect_rounded_filled(
        zone_min_x,
        zone_min_y,
        zone_max_x,
        zone_max_y,
        fill_color,
        _CORNER_RADIUS,
    )

    arrow_y = center_y - 60.0 + _BOUNCE_AMOUNT * math.sin(t * _BOUNCE_SPEED)
    layout.draw_window_triangle_filled(
        center_x,
        arrow_y + 25.0,
        center_x - 20.0,
        arrow_y,
        center_x + 20.0,
        arrow_y,
        overlay_text,
    )
    layout.draw_window_rect_rounded_filled(
        center_x - 8.0,
        arrow_y - 25.0,
        center_x + 8.0,
        arrow_y,
        overlay_text,
        2.0,
    )

    title = lf.ui.tr("startup.drop_to_import")
    subtitle = lf.ui.tr("startup.drop_to_import_subtitle")
    title_w, _ = layout.calc_text_size(title)
    subtitle_w, _ = layout.calc_text_size(subtitle)

    layout.draw_window_text(center_x - title_w * 0.5, center_y + 5.0, title, overlay_text)
    subtitle_color = (overlay_text_dim[0], overlay_text_dim[1], overlay_text_dim[2], 0.5)
    layout.draw_window_text(center_x - subtitle_w * 0.5, center_y + 35.0, subtitle, subtitle_color)

    layout.end_window()


def _sync_viewport_overlay_document(doc):
    global _document_controller
    if _document_controller is None:
        _document_controller = _OverlayDocumentController()
    _document_controller.update(doc)


def _draw_viewport_overlay(layout):
    _draw_empty_state_overlay(layout)
    _draw_drag_drop_overlay(layout)


def register():
    """Register built-in viewport overlay controllers."""
    global _hook_registered
    if _hook_registered:
        return

    lf.ui.add_hook(_HOOK_PANEL, _DOCUMENT_SECTION, _sync_viewport_overlay_document, _HOOK_POSITION)
    lf.ui.add_hook(_HOOK_PANEL, _DRAW_SECTION, _draw_viewport_overlay, _HOOK_POSITION)
    _hook_registered = True


def unregister():
    """Unregister built-in viewport overlay controllers."""
    global _hook_registered
    if not _hook_registered:
        return

    lf.ui.remove_hook(_HOOK_PANEL, _DOCUMENT_SECTION, _sync_viewport_overlay_document)
    lf.ui.remove_hook(_HOOK_PANEL, _DRAW_SECTION, _draw_viewport_overlay)
    _hook_registered = False
    if _document_controller is not None:
        _document_controller.reset()
