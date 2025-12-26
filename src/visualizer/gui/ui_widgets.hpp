/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include <glm/glm.hpp>
#include <string>
#include <imgui.h>

namespace lfs::vis::gui::widgets {

    // Reusable UI widgets
    bool SliderWithReset(const char* label, float* v, float min, float max, float reset_value);
    bool DragFloat3WithReset(const char* label, float* v, float speed, float reset_value);
    void HelpMarker(const char* desc);
    void TableRow(const char* label, const char* format, ...);

    // Progress display
    void DrawProgressBar(float fraction, const char* overlay_text);
    void DrawLossPlot(const float* values, int count, float min_val, float max_val, const char* label);
    void DrawModeStatusWithContentSwitch(const UIContext& ctx);
    // Mode display helpers
    void DrawModeStatus(const UIContext& ctx);

    // Shadow drawing for floating panels
    void DrawWindowShadow(const ImVec2& pos, const ImVec2& size, float rounding = 6.0f);

    // Vignette effect for viewport
    void DrawViewportVignette(const ImVec2& pos, const ImVec2& size);

    // Icon button with selection state styling
    bool IconButton(const char* id, unsigned int texture, const ImVec2& size, bool selected = false,
                    const char* fallback_label = "?");

    // Semantic colored buttons - subtle tint on surface, stronger on hover
    enum class ButtonStyle { Primary,
                             Success,
                             Warning,
                             Error,
                             Secondary };
    bool ColoredButton(const char* label, ButtonStyle style, const ImVec2& size = {-1, 0});

    // Typography
    void SectionHeader(const char* text, const FontSet& fonts);

    // Tooltip with theme-aware text color (dark text on light themes)
    void SetThemedTooltip(const char* fmt, ...);

} // namespace lfs::vis::gui::widgets
