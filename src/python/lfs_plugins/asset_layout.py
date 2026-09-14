# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager geometry in dp; shared by DOM sizing and regression tests."""
RESULTS_MIN_HEIGHT = 160.0
SIDEBAR_PADDING = 16.0
GALLERY_SECTION_HEIGHT = 140.0
LOCAL_SECTION_HEIGHT = 77.0
FOLDER_ROW_HEIGHT = 34.0
RESIZE_HANDLES_HEIGHT = 20.0
GALLERY_CARD_GAP = 10.0
GALLERY_CARD_PREFERRED_WIDTH = 208.0
GALLERY_HORIZONTAL_CHROME = 48.0


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
    # Match the list shell/row padding, gaps and fixed columns in asset_manager.rcss.
    gallery, size, modified_width, folder_width = 96.0, 48.0, 72.0, 58.0
    modified, folder = width >= 380, width >= 600
    def name_width(gallery_width):
        count = 3 + int(modified) + int(folder)
        return width - 46.0 - 6.0 * (count - 1) - gallery_width - size - modified * modified_width - folder * folder_width
    if name_width(gallery) < 64:
        modified = False
    spare_gallery = width - 46.0 - 6.0 * (2 + int(modified)) - size - modified * modified_width - folder * folder_width - 64.0
    gallery = min(200.0, max(96.0, spare_gallery))
    return dict(modified=modified, folder=folder, name=name_width(gallery), gallery=gallery)
