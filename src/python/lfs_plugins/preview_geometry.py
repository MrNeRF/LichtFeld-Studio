# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Source-pixel geometry shared by preview gestures and color sampling."""

from math import ceil, floor


THUMB_ROW_DP = 64.0
THUMB_OVERSCAN = 4


def visible_thumbnail_range(count, scroll_top, viewport_height, dp_ratio=1.0):
    """Rml scroll offsets are physical pixels; spacers are authored in dp."""
    top = max(0.0, scroll_top / dp_ratio)
    height = max(THUMB_ROW_DP, viewport_height / dp_ratio)
    first = min(count, max(0, floor(top / THUMB_ROW_DP) - THUMB_OVERSCAN))
    last = min(count, ceil((top + height) / THUMB_ROW_DP) + THUMB_OVERSCAN)
    return first, max(first, last)


def view_to_source(layout, rotation, x, y):
    """Map viewport-local dp to continuous source coordinates (before rounding)."""
    dx = x - layout["offset_x"] - layout["draw_width"] * 0.5
    dy = y - layout["offset_y"] - layout["draw_height"] * 0.5
    ux, uy = ((dx, dy), (dy, -dx), (-dx, -dy), (-dy, dx))[rotation % 4]
    return (ux / layout["scale"] + layout["width"] * 0.5,
            uy / layout["scale"] + layout["height"] * 0.5)


def source_to_view(layout, rotation, x, y):
    dx = (x - layout["width"] * 0.5) * layout["scale"]
    dy = (y - layout["height"] * 0.5) * layout["scale"]
    rx, ry = ((dx, dy), (-dy, dx), (-dx, -dy), (dy, -dx))[rotation % 4]
    return (rx + layout["offset_x"] + layout["draw_width"] * 0.5,
            ry + layout["offset_y"] + layout["draw_height"] * 0.5)
