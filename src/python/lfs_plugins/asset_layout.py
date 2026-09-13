# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager geometry in dp; shared by DOM sizing and regression tests."""
RESULTS_MIN_HEIGHT = 160.0
SIDEBAR_PADDING = 16.0
GALLERY_SECTION_HEIGHT = 140.0
LOCAL_SECTION_HEIGHT = 77.0
FOLDER_ROW_HEIGHT = 34.0
RESIZE_HANDLES_HEIGHT = 20.0


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
    def name_width():
        count = 3 + int(modified) + int(folder)
        return width - 46.0 - 6.0 * (count - 1) - gallery - size - modified * modified_width - folder * folder_width
    if name_width() < 64:
        modified = False
    return dict(modified=modified, folder=folder, name=name_width(), gallery=gallery)
