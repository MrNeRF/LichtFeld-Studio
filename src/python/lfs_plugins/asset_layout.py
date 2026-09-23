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

# Gallery now owns a fixed icon slot instead of a measured prose column. These
# boundaries retain at least 360 dp for the results beside a 320 dp Inspector.
# Wide mode starts once the navigator can join it while leaving roughly 480 dp
# for the adaptive list; secondary columns may hide, but core actions remain.
BREAKPOINT_COMPACT_MAX = 360.0
BREAKPOINT_NARROW_MAX = 680.0
BREAKPOINT_MEDIUM_MAX = 1000.0
GRID_GAP = 12.0
GRID_HORIZONTAL_PADDING = 24.0
LIST_ACTION_COLUMN_WIDTH = 32.0
THUMBNAIL_MIN = 112.0
THUMBNAIL_MAX = 320.0
INSPECTOR_COLUMN_MIN = 320.0
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
            "inspector_placement": "bottom",
            "inspector_default": 1000.0,
            "inspector_min": 180.0,
            "inspector_max": 1000.0,
        },
        "narrow": {
            "toolbar_rows": 2,
            "navigator_mode": "dropdown",
            "navigator_default": 0.0,
            "navigator_min": 0.0,
            "navigator_max": 0.0,
            "inspector_placement": "bottom",
            "inspector_default": 1000.0,
            "inspector_min": 180.0,
            "inspector_max": 1000.0,
        },
        "medium": {
            "toolbar_rows": 1,
            "navigator_mode": "dropdown",
            "navigator_default": 0.0,
            "navigator_min": 0.0,
            "navigator_max": 0.0,
            "inspector_placement": "column",
            "inspector_default": INSPECTOR_COLUMN_MIN,
            "inspector_min": INSPECTOR_COLUMN_MIN,
            "inspector_max": 420.0,
        },
        "wide": {
            "toolbar_rows": 1,
            "navigator_mode": "column",
            "navigator_default": 200.0,
            "navigator_min": 160.0,
            "navigator_max": 240.0,
            "inspector_placement": "column",
            "inspector_default": INSPECTOR_COLUMN_MIN,
            "inspector_min": INSPECTOR_COLUMN_MIN,
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


def list_columns(width, measured=None, overrides=None):
    """Hide whole columns when their measured content cannot fit."""
    widths = list_column_widths(width, overrides, measured)
    return dict(size=widths["size"] > 0, modified=widths["modified"] > 0,
                folder=widths["folder"] > 0, name=widths["name"], gallery=widths["gallery"])


def list_column_widths(width, overrides=None, measured=None):
    """Content widths include 8 dp on each side; only Name can shrink."""
    # Used before a document mounts. Mounted panels always supply font measurements.
    metrics = measured or {key: len(sample) * 6.0 + 16.0 for key, sample in (
        ("gallery", "Not published    "), ("size", "1023.9 MB"),
        ("modified", "2000-12-30 23:59"), ("folder", "Projects"))}
    widths = {key: max(float(metrics[key]), float((overrides or {}).get(key, 0)))
              for key in ("gallery", "size", "modified", "folder")}
    # Gallery is a status affordance, not a prose column. Its icon always owns
    # the same compact slot; the complete state remains available as a tooltip.
    widths["gallery"] = 32.0
    for key, threshold in (("size", 360), ("modified", 560), ("folder", 700)):
        if width < threshold:
            widths[key] = 0.0
    # Reserve shell/row insets, the thumbnail and gap, and the row action column.
    available = max(
        0.0,
        float(width) - 24.0 - 16.0 - 32.0 - 8.0 - LIST_ACTION_COLUMN_WIDTH,
    )
    name_minimum = min(max(80.0, float((overrides or {}).get("name", 80.0))), max(80.0, available - 32.0))
    for key in ("folder", "modified", "size"):
        if sum(widths.values()) + name_minimum > available:
            widths[key] = 0.0
    widths["name"] = max(0.0, available - sum(widths.values()))
    # A Name drag consumes spare space only. Measured columns never shrink.
    return {key: math.floor(widths.get(key, 0.0) * 10.0) / 10.0
            for key in ("name", "gallery", "size", "modified", "folder")}
