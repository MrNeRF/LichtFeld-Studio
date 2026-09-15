# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Projects geometry in dp; shared by DOM sizing and regression tests."""
import math
RESULTS_MIN_HEIGHT = 160.0
SIDEBAR_PADDING = 16.0
GALLERY_SECTION_HEIGHT = 140.0
LOCAL_SECTION_HEIGHT = 77.0
FOLDER_ROW_HEIGHT = 34.0
RESIZE_HANDLES_HEIGHT = 20.0
GALLERY_CARD_GAP = 10.0
GALLERY_CARD_PREFERRED_WIDTH = 208.0
GALLERY_HORIZONTAL_CHROME = 48.0

BREAKPOINT_COMPACT_MAX = 420.0
BREAKPOINT_NARROW_MAX = 640.0
BREAKPOINT_MEDIUM_MAX = 900.0
GRID_GAP = 12.0
GRID_HORIZONTAL_PADDING = 24.0
THUMBNAIL_MIN = 112.0
THUMBNAIL_MAX = 320.0
THUMBNAIL_DEFAULTS = {
    "compact": 112.0,
    "narrow": 136.0,
    "medium": 168.0,
    "wide": 168.0,
}


def breakpoint_for_width(width):
    """Return the stable root class for a panel content width in dp."""
    width = float(width)
    if width < BREAKPOINT_COMPACT_MAX:
        return "compact"
    if width < BREAKPOINT_NARROW_MAX:
        return "narrow"
    if width < BREAKPOINT_MEDIUM_MAX:
        return "medium"
    return "wide"


def grid_columns(width, card_width, gap=GRID_GAP, horizontal_padding=GRID_HORIZONTAL_PADDING):
    """Return grid columns after subtracting the gap for every column."""
    content_width = max(0.0, float(width) - horizontal_padding)
    card_width = max(1.0, float(card_width))
    return max(1, int((content_width + gap) // (card_width + gap)))


def grid_slot_width(width, card_width, gap=GRID_GAP, horizontal_padding=GRID_HORIZONTAL_PADDING):
    """Return the stretched card width for a grid row in dp."""
    content_width = max(0.0, float(width) - horizontal_padding)
    columns = grid_columns(width, card_width, gap, horizontal_padding)
    # RmlUi lays out inline dp values after converting them to native pixels.
    # Leave a tenth of a dp of headroom so an exact final slot does not round
    # up and wrap the last card onto a new row.
    stretched = (content_width - gap * (columns - 1)) / columns
    stretched = math.floor(max(0.0, stretched) * 10.0) / 10.0
    return max(1.0, min(stretched, card_width * 1.15))


def card_geometry(card_width):
    """Return the fixed-ratio thumbnail and card heights in dp."""
    thumbnail_height = float(card_width) * 10.0 / 16.0
    return {
        "thumbnail_width": float(card_width),
        "thumbnail_height": thumbnail_height,
        "height": thumbnail_height + 40.0,
    }


def list_row_height(*, gallery_column_visible=True):
    """Return the list row height for the visible column arrangement."""
    return 40.0 if gallery_column_visible else 48.0


def breakpoint_metrics(width):
    """Return the exact region defaults and limits for a breakpoint."""
    name = breakpoint_for_width(width)
    values = {
        "compact": {
            "toolbar_rows": 2,
            "navigator_mode": "dropdown",
            "navigator_default": 0.0,
            "navigator_min": 0.0,
            "navigator_max": 0.0,
            "inspector_placement": "overlay",
            "inspector_default": 200.0,
            "inspector_min": 120.0,
            "inspector_max": 320.0,
        },
        "narrow": {
            "toolbar_rows": 2,
            "navigator_mode": "dropdown",
            "navigator_default": 0.0,
            "navigator_min": 0.0,
            "navigator_max": 0.0,
            "inspector_placement": "strip",
            "inspector_default": 32.0,
            "inspector_min": 32.0,
            "inspector_max": 32.0,
        },
        "medium": {
            "toolbar_rows": 1,
            "navigator_mode": "column",
            "navigator_default": 160.0,
            "navigator_min": 120.0,
            "navigator_max": 240.0,
            "inspector_placement": "band",
            "inspector_default": 200.0,
            "inspector_min": 120.0,
            "inspector_max": 450.0,
        },
        "wide": {
            "toolbar_rows": 1,
            "navigator_mode": "column",
            "navigator_default": 200.0,
            "navigator_min": 160.0,
            "navigator_max": 240.0,
            "inspector_placement": "column",
            "inspector_default": 280.0,
            "inspector_min": 240.0,
            "inspector_max": 420.0,
        },
    }[name].copy()
    values["breakpoint"] = name
    values["card_width"] = THUMBNAIL_DEFAULTS[name]
    values["card"] = card_geometry(values["card_width"])
    return values


def native_to_dp(value, scale):
    return max(0.0, float(value or 0.0)) / max(0.1, float(scale or 1.0))


def gallery_columns(width, *, preferred=GALLERY_CARD_PREFERRED_WIDTH,
                    horizontal_chrome=GALLERY_HORIZONTAL_CHROME, gap=GALLERY_CARD_GAP):
    content_width = max(preferred, float(width) - horizontal_chrome)
    return max(1, int((content_width + gap) // (preferred + gap)))


def gallery_slot_width(width, *, preferred=GALLERY_CARD_PREFERRED_WIDTH,
                       horizontal_chrome=GALLERY_HORIZONTAL_CHROME, gap=GALLERY_CARD_GAP):
    content_width = max(preferred, float(width) - horizontal_chrome)
    columns = gallery_columns(width, preferred=preferred,
                              horizontal_chrome=horizontal_chrome, gap=gap)
    return max(1.0, (content_width - gap * (columns - 1)) / columns)


def panel_layout(height, *, folder_count=0, folders_collapsed=False, info_height=220.0,
                 toolbar_height=115.0, results_header_height=49.0, sidebar_content_height=None):
    content = (SIDEBAR_PADDING + GALLERY_SECTION_HEIGHT + LOCAL_SECTION_HEIGHT
               + (0 if folders_collapsed else FOLDER_ROW_HEIGHT * folder_count))
    if sidebar_content_height is not None:
        content = sidebar_content_height
    sidebar = min(content, height * 0.4)
    available = max(0.0, height - toolbar_height - results_header_height - RESIZE_HANDLES_HEIGHT)
    # At exceptionally short sizes, the sidebar yields after Info has collapsed.
    sidebar = min(sidebar, max(0.0, available - RESULTS_MIN_HEIGHT))
    info = min(max(0.0, info_height), max(0.0, available - sidebar - RESULTS_MIN_HEIGHT))
    return dict(sidebar=sidebar, info=info, results=available - sidebar - info,
                main_min_height=sidebar + 10.0 + results_header_height + RESULTS_MIN_HEIGHT)


def list_columns(width):
    """Columns that fit the browser width, after the other panel regions yield space."""
    size, modified, folder = width >= 360, width >= 560, width >= 700
    gallery = 24.0 if width < 480 else 128.0
    gaps = 8.0 * (2 + int(size) + int(modified) + int(folder))
    name = width - 24.0 - 32.0 - gaps - gallery - size * 72.0 - modified * 96.0 - folder * 100.0
    return dict(size=size, modified=modified, folder=folder, name=name, gallery=gallery)
