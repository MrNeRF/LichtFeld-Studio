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
        ThemeColor disabled_surface;
        ThemeColor hovered;
        ThemeColor active;
        ThemeColor submode_surface;
        ThemeColor submode_hovered;
        ThemeColor border;
        ThemeColor text;
        ThemeColor disabled_text;
        ThemeColor center;
        ThemeColor halo;
        ThemeColor text_shadow;
        float stroke_width;
        float label_size;
        float submode_label_size;
        float fallback_icon_size;
    };

    // Native menus use the same palette, typography and stroke roles as panels.
    // Resolve on each draw so theme switches and hot reload need no local cache.
    [[nodiscard]] inline MenuStyle menuStyle(const Theme& theme) {
        const auto with_alpha = [](ThemeColor color, float scale) {
            color.w *= scale;
            return color;
        };
        const auto& palette = theme.palette;
        return {
            .surface = with_alpha(palette.surface, 0.92f),
            .disabled_surface = with_alpha(palette.surface, 0.55f),
            .hovered = with_alpha(palette.primary, 0.92f),
            .active = with_alpha(palette.primary_dim, 0.80f),
            .submode_surface = with_alpha(palette.surface, 0.70f),
            .submode_hovered = with_alpha(palette.primary, 0.85f),
            .border = with_alpha(palette.border, 0.50f),
            .text = palette.text,
            .disabled_text = with_alpha(palette.text_dim, 0.40f),
            .center = with_alpha(palette.background, 0.55f),
            .halo = with_alpha(palette.background, 0.30f),
            .text_shadow = {0.0f, 0.0f, 0.0f, 0.45f},
            .stroke_width = normal_stroke,
            .label_size = 11.0f,
            .submode_label_size = 10.0f,
            .fallback_icon_size = 16.0f,
        };
    }

} // namespace lfs::vis::ui_style
