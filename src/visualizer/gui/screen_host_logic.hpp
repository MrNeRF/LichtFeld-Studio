/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "screen/area_gestures.hpp"

namespace lfs::vis::gui::screen_host_detail {

    inline screen::AreaId displayedViewAt(const bool ui_hidden, const screen::AreaId active_view,
                                          const screen::AreaId screen_view, const screen::Rect& viewport_rect,
                                          const float x, const float y) {
        if (ui_hidden)
            return viewport_rect.contains(x, y) ? active_view : screen::AreaId{};
        return screen_view;
    }

    inline bool cornerGestureZone(const screen::LayoutGeometry& geometry, const bool maximized,
                                  const float corner_size, const float x, const float y) {
        return !maximized && screen::cornerAt(geometry, x, y, corner_size).has_value();
    }

    inline bool shouldCloseMissingPanelEditor(const bool editor_type_registered, const bool was_panel_editor) {
        return !editor_type_registered && was_panel_editor;
    }

    inline void clearDirtyWhenOverlayHidden(const bool overlay_visible, bool& overlay_dirty) {
        if (!overlay_visible)
            overlay_dirty = false;
    }

} // namespace lfs::vis::gui::screen_host_detail
