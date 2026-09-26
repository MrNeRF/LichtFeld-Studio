"""Viewport overlay controllers for retained status dialogs and custom drawing."""

import math

import lichtfeld as lf

from ..ui import RuntimeState
from ..localization import safe_format

from .. import toolbar as viewport_toolbar
from ..gallery_transfer_overlay import GalleryTransferOverlay

try:
    from ..ui.store import native_value as _native_store_value
except Exception:
    def _native_store_value(_field, fallback):
        return fallback

_HOOK_PANEL = "viewport_overlay"
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


def _viewport_bottom_inset(layout, base_inset):
    bottom_inset = base_inset
    if lf.ui.is_sequencer_visible():
        dp = layout.get_dpi_scale()
        seq_state = lf.ui.get_sequencer_state()
        film_strip_h = 56.0 if (seq_state and seq_state.show_film_strip) else 0.0
        seq_height = 162.0 * dp + film_strip_h
        bottom_inset = max(base_inset, seq_height + 8.0)
    return bottom_inset


def _empty_state_visible(import_visible):
    return (
        not import_visible
        and lf.ui.is_scene_empty()
        and not lf.ui.is_drag_hovering()
        and not lf.ui.is_startup_visible()
    )


def _empty_state_import_hint():
    import_path = lf.ui.tr("menu.file") + " > " + lf.ui.tr("menu.file.import")
    return safe_format(lf.ui.tr("startup.drop_files_hint"), path=import_path)


def _get_import_state():
    native_state = _native_store_value("import_overlay_state", None)
    if isinstance(native_state, dict):
        return dict(native_state)

    if not hasattr(lf.ui, "get_import_state"):
        return {}

    return dict(lf.ui.get_import_state())


class _OverlayDocumentController:
    def __init__(self):
        self.gallery_transfers = GalleryTransferOverlay()
        self.reset()

    def reset(self):
        self._handle = None
        self._empty_state_signature = None
        self.gallery_transfers.reset()
        viewport_toolbar.reset_overlay_state()

    def update(self, doc=None):
        if doc is None or not hasattr(doc, "get_element_by_id"):
            doc = lf.ui.rml.get_document(_HOOK_PANEL)
        if doc is None:
            return []

        if not self._ensure_model(doc):
            return []

        dirty_sources = []
        status_dirty = False

        import_state = _get_import_state()
        import_visible = import_state.get("active", False) or import_state.get("show_completion", False)
        empty_state_signature = (
            RuntimeState.language_generation.value,
            _empty_state_visible(import_visible),
        )
        if empty_state_signature != self._empty_state_signature:
            self._empty_state_signature = empty_state_signature
            dirty_sources.append("empty_state")
            status_dirty = True

        toolbar_sources = viewport_toolbar.update_overlay(doc) or []
        dirty_sources.extend(f"toolbar.{source}" for source in toolbar_sources)
        if self.gallery_transfers.update():
            dirty_sources.append("gallery_transfers")
            status_dirty = True

        if status_dirty:
            self._handle.dirty_all()
        return dirty_sources

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

        model.bind_func("show_empty_state",
                        lambda: self._empty_state_signature is not None and self._empty_state_signature[1])
        model.bind_func("empty_state_title", lambda: lf.ui.tr("startup.drop_files_title"))
        model.bind_func("empty_state_subtitle", lambda: lf.ui.tr("startup.drop_files_subtitle"))
        model.bind_func("empty_state_import_hint", _empty_state_import_hint)

        viewport_toolbar.bind_overlay_model(model)
        self.gallery_transfers.bind_model(model)
        self._handle = model.get_handle()
        viewport_toolbar.attach_overlay_model_handle(self._handle)
        body.set_attribute("data-model", _MODEL_NAME)
        body.set_attribute(_MODEL_MARKER, "1")
        self._handle.dirty_all()
        return True


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


def _sync_viewport_overlay_document(doc=None):
    global _document_controller
    if _document_controller is None:
        _document_controller = _OverlayDocumentController()
    dirty_sources = _document_controller.update(doc)
    debug_log = getattr(getattr(lf, "log", None), "debug", None)
    if callable(debug_log):
        for source in dirty_sources or []:
            debug_log("[PERF] viewport_overlay_document_dirty source=" + str(source))
    return bool(dirty_sources)


def sync_document(doc=None):
    """Synchronize the first-party viewport overlay data model."""
    if not _hook_registered:
        return False
    return _sync_viewport_overlay_document(doc)


def show_gallery_transfers():
    _sync_viewport_overlay_document()
    _document_controller.gallery_transfers.show()


def _draw_viewport_overlay(layout):
    _draw_drag_drop_overlay(layout)


def register():
    """Register built-in viewport overlay controllers."""
    global _hook_registered
    if _hook_registered:
        return

    lf.ui.add_hook(_HOOK_PANEL, _DRAW_SECTION, _draw_viewport_overlay, _HOOK_POSITION)
    _hook_registered = True
    _sync_viewport_overlay_document()


def on_document_unloaded():
    """Drop Python overlay handles before RmlUi frees the native document."""
    global _document_controller
    if _document_controller is not None:
        _document_controller.reset()


def unregister():
    """Unregister built-in viewport overlay controllers."""
    global _hook_registered
    if not _hook_registered:
        return

    lf.ui.remove_hook(_HOOK_PANEL, _DRAW_SECTION, _draw_viewport_overlay)
    _hook_registered = False
    on_document_unloaded()
