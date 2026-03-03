"""Empty state overlay - animated drop zone for when no scene is loaded."""

import math

import lichtfeld as lf
from ..types import Panel

ZONE_PADDING = 120.0
DASH_LENGTH = 12.0
GAP_LENGTH = 8.0
BORDER_THICKNESS = 2.0
ICON_SIZE = 48.0
ANIM_SPEED = 30.0
MIN_VIEWPORT_SIZE = 200.0

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


class EmptyStateOverlay(Panel):
    """Viewport overlay showing drop zone when scene is empty."""

    label = "##EmptyState"
    space = "VIEWPORT_OVERLAY"
    order = 0

    @classmethod
    def poll(cls, context):
        return lf.ui.is_scene_empty() and not lf.ui.is_drag_hovering()

    def draw(self, layout):
        vp_x, vp_y = layout.get_viewport_pos()
        vp_w, vp_h = layout.get_viewport_size()

        if vp_w < MIN_VIEWPORT_SIZE or vp_h < MIN_VIEWPORT_SIZE:
            return

        layout.set_next_window_pos((vp_x, vp_y))
        layout.set_next_window_size((vp_w, vp_h))

        if not layout.begin_window("##EmptyStateOverlay", OVERLAY_FLAGS):
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
        zone_min_x = vp_x + ZONE_PADDING
        zone_min_y = vp_y + ZONE_PADDING
        zone_max_x = vp_x + vp_w - ZONE_PADDING
        zone_max_y = vp_y + vp_h - ZONE_PADDING

        t = lf.ui.get_time()
        dash_offset = (t * ANIM_SPEED) % (DASH_LENGTH + GAP_LENGTH)

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
                d1 = min(length, pos + DASH_LENGTH)
                if d1 > d0:
                    layout.draw_window_line(
                        start_x + nx * d0, start_y + ny * d0,
                        start_x + nx * d1, start_y + ny * d1,
                        border_color, BORDER_THICKNESS,
                    )
                pos += DASH_LENGTH + GAP_LENGTH

        if lf.ui.is_startup_visible():
            layout.end_window()
            return

        draw_dashed_line(zone_min_x, zone_min_y, zone_max_x, zone_min_y)
        draw_dashed_line(zone_max_x, zone_min_y, zone_max_x, zone_max_y)
        draw_dashed_line(zone_max_x, zone_max_y, zone_min_x, zone_max_y)
        draw_dashed_line(zone_min_x, zone_max_y, zone_min_x, zone_min_y)

        icon_y = center_y - 50.0
        layout.draw_window_rect(
            center_x - ICON_SIZE * 0.5, icon_y - ICON_SIZE * 0.3,
            center_x + ICON_SIZE * 0.5, icon_y + ICON_SIZE * 0.4,
            icon_color, 2.0,
        )
        layout.draw_window_line(
            center_x - ICON_SIZE * 0.5, icon_y - ICON_SIZE * 0.3,
            center_x - ICON_SIZE * 0.2, icon_y - ICON_SIZE * 0.5,
            icon_color, 2.0,
        )
        layout.draw_window_line(
            center_x - ICON_SIZE * 0.2, icon_y - ICON_SIZE * 0.5,
            center_x + ICON_SIZE * 0.1, icon_y - ICON_SIZE * 0.5,
            icon_color, 2.0,
        )
        layout.draw_window_line(
            center_x + ICON_SIZE * 0.1, icon_y - ICON_SIZE * 0.5,
            center_x + ICON_SIZE * 0.2, icon_y - ICON_SIZE * 0.3,
            icon_color, 2.0,
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
