/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "gui/ui_context.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <imgui.h>

namespace lfs::vis::gui::widgets {

    // Reusable UI widgets
    bool SliderWithReset(const char* label, float* v, float min, float max, float reset_value,
                         const char* tooltip = nullptr, const char* format = "%.3f");
    bool DragFloat3WithReset(const char* label, float* v, float speed, float reset_value,
                             const char* tooltip = nullptr);
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
    LFS_VIS_API bool IconButton(const char* id, unsigned int texture, const ImVec2& size, bool selected = false,
                                const char* fallback_label = "?");

    // Semantic colored buttons - subtle tint on surface, stronger on hover
    enum class ButtonStyle { Primary,
                             Success,
                             Warning,
                             Error,
                             Secondary };
    LFS_VIS_API bool ColoredButton(const char* label, ButtonStyle style, const ImVec2& size = {-1, 0});

    // Typography
    void SectionHeader(const char* text, const FontSet& fonts);

    // Tooltip with theme-aware text color (dark text on light themes)
    LFS_VIS_API void SetThemedTooltip(const char* fmt, ...);

    // Format number with thousand separators (e.g., 1500000 -> "1,500,000")
    std::string formatNumber(int64_t num);

    // InputInt with thousand separator display (shows formatted when not editing)
    LFS_VIS_API bool InputIntFormatted(const char* label, int* v, int step = 0, int step_fast = 0);

    // 2D point picker for chromaticity offset (color grading)
    // Returns true if value changed. Color tint shows which channel (red/green/blue).
    bool ChromaticityPicker2D(const char* label, float* x, float* y, float range = 0.5f,
                              const ImVec4& color_tint = ImVec4(1, 1, 1, 1));

    // Unified chromaticity diagram with 4 draggable control points (R, G, B, Neutral)
    // Shows rg chromaticity space with all color correction points in one widget.
    // Returns true if any value changed.
    LFS_VIS_API bool ChromaticityDiagram(const char* label, float* red_x, float* red_y, float* green_x, float* green_y,
                                         float* blue_x, float* blue_y, float* neutral_x, float* neutral_y,
                                         float range = 0.5f);

    // CRF tone curve preview (read-only visualization)
    // Shows the effect of gamma, toe, and shoulder on the tone curve
    LFS_VIS_API void CRFCurvePreview(const char* label, float gamma, float toe, float shoulder,
                                     float gamma_r = 0.0f, float gamma_g = 0.0f, float gamma_b = 0.0f);

} // namespace lfs::vis::gui::widgets
