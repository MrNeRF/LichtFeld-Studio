/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "theme/theme.hpp"

namespace lfs::vis::ui_style {

    // Shared by native drawing and the stroke.* RmlUi theme tokens.
    inline constexpr float normal_stroke = 1.0f;
    inline constexpr float emphasis_stroke = 2.0f;

    struct MenuStyle {
        ThemeColor surface;
        ThemeColor hovered;
        ThemeColor active;
        ThemeColor border;
        ThemeColor text;
        ThemeColor disabled_text;
        ThemeColor center;
        ThemeColor shadow;
        float stroke_width;
        float label_size;
        float submode_label_size;
        float fallback_icon_size;
    };

    // Native menus use the same palette, typography and stroke roles as panels.
    // Resolve on each draw so theme switches and hot reload need no local cache.
    [[nodiscard]] inline MenuStyle menuStyle(const Theme& theme) {
        return {
            .surface = theme.palette.surface,
            .hovered = theme.palette.primary,
            .active = theme.palette.primary_dim,
            .border = theme.palette.border,
            .text = theme.palette.text,
            .disabled_text = theme.palette.text_dim,
            .center = theme.palette.background,
            .shadow = {0.0f, 0.0f, 0.0f, theme.shadows.enabled ? theme.shadows.alpha : 0.0f},
            .stroke_width = normal_stroke,
            .label_size = theme.fonts.small_size,
            .submode_label_size = theme.fonts.small_size,
            .fallback_icon_size = theme.fonts.large_size,
        };
    }

} // namespace lfs::vis::ui_style
