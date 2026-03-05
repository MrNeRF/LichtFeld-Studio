"""Drag-drop overlay - pulsing drop zone when dragging files over window."""

import math

import lichtfeld as lf
from ..types import Panel

INSET = 30.0
CORNER_RADIUS = 16.0
GLOW_MAX = 8.0
PULSE_SPEED = 3.0
BOUNCE_SPEED = 4.0
BOUNCE_AMOUNT = 5.0

OVERLAY_FLAGS = (
    lf.ui.UILayout.WindowFlags.NoTitleBar
    | lf.ui.UILayout.WindowFlags.NoResize
    | lf.ui.UILayout.WindowFlags.NoMove
    | lf.ui.UILayout.WindowFlags.NoScrollbar
    | lf.ui.UILayout.WindowFlags.NoInputs
    | lf.ui.UILayout.WindowFlags.NoBackground
    | lf.ui.UILayout.WindowFlags.NoFocusOnAppearing
    | lf.ui.UILayout.WindowFlags.NoBringToFrontOnFocus
)


class DragDropOverlay(Panel):
    """Full-screen overlay when dragging files over window."""

    label = "##DragDrop"
    space = "VIEWPORT_OVERLAY"
    order = 100

    @classmethod
    def poll(cls, context):
        return lf.ui.is_drag_hovering() and not lf.ui.is_startup_visible()

    def draw(self, layout):
        vp_x, vp_y = layout.get_viewport_pos()
        vp_w, vp_h = layout.get_viewport_size()

        layout.set_next_window_pos((vp_x, vp_y))
        layout.set_next_window_size((vp_w, vp_h))

        if not layout.begin_window("##DragDropOverlay", OVERLAY_FLAGS):
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
        zone_min_x = vp_x + INSET
        zone_min_y = vp_y + INSET
        zone_max_x = win_max_x - INSET
        bottom_inset = INSET
        if lf.ui.is_sequencer_visible():
            dp = layout.get_dpi_scale()
            seq_state = lf.ui.get_sequencer_state()
            film_strip_h = 56.0 if (seq_state and seq_state.show_film_strip) else 0.0
            seq_height = 162.0 * dp + film_strip_h
            bottom_inset = max(INSET, seq_height + 8.0)
        zone_max_y = win_max_y - bottom_inset
        center_x = vp_x + vp_w * 0.5
        center_y = vp_y + vp_h * 0.5

        t = lf.ui.get_time()
        pulse = 0.5 + 0.5 * math.sin(t * PULSE_SPEED)

        layout.draw_window_rect_filled(vp_x, vp_y, win_max_x, win_max_y, overlay_color)

        glow_color = (primary[0], primary[1], primary[2], 0.16 * pulse)
        i = GLOW_MAX
        while i > 0:
            layout.draw_window_rect_rounded(
                zone_min_x - i,
                zone_min_y - i,
                zone_max_x + i,
                zone_max_y + i,
                glow_color,
                CORNER_RADIUS + i,
                2.0,
            )
            i -= 2.0

        border_alpha = 0.7 + 0.3 * pulse
        border_color = (primary[0], primary[1], primary[2], border_alpha)
        layout.draw_window_rect_rounded(
            zone_min_x, zone_min_y, zone_max_x, zone_max_y, border_color, CORNER_RADIUS, 3.0
        )
        layout.draw_window_rect_rounded_filled(
            zone_min_x, zone_min_y, zone_max_x, zone_max_y, fill_color, CORNER_RADIUS
        )

        arrow_y = center_y - 60.0 + BOUNCE_AMOUNT * math.sin(t * BOUNCE_SPEED)
        layout.draw_window_triangle_filled(
            center_x, arrow_y + 25.0,
            center_x - 20.0, arrow_y,
            center_x + 20.0, arrow_y,
            overlay_text,
        )
        layout.draw_window_rect_rounded_filled(
            center_x - 8.0, arrow_y - 25.0, center_x + 8.0, arrow_y, overlay_text, 2.0
        )

        title = lf.ui.tr("startup.drop_to_import")
        subtitle = lf.ui.tr("startup.drop_to_import_subtitle")
        title_w, _ = layout.calc_text_size(title)
        subtitle_w, _ = layout.calc_text_size(subtitle)

        layout.draw_window_text(center_x - title_w * 0.5, center_y + 5.0, title, overlay_text)
        subtitle_color = (overlay_text_dim[0], overlay_text_dim[1], overlay_text_dim[2], 0.5)
        layout.draw_window_text(center_x - subtitle_w * 0.5, center_y + 35.0, subtitle, subtitle_color)

        layout.end_window()
