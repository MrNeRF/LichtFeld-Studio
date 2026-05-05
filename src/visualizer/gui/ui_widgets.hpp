/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "gui/ui_context.hpp"
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <imgui.h>

namespace lfs::vis::gui::widgets {

    // Shared single-line input helpers with Blender-style text editing affordances.
    LFS_VIS_API bool InputText(const char* label, char* buf, std::size_t buf_size,
                               ImGuiInputTextFlags flags = 0,
                               ImGuiInputTextCallback callback = nullptr, void* user_data = nullptr);
    LFS_VIS_API bool InputTextWithHint(const char* label, const char* hint, char* buf, std::size_t buf_size,
                                       ImGuiInputTextFlags flags = 0,
                                       ImGuiInputTextCallback callback = nullptr, void* user_data = nullptr);
    LFS_VIS_API bool InputFloat(const char* label, float* v, float step = 0.0f, float step_fast = 0.0f,
                                const char* format = "%.3f", ImGuiInputTextFlags flags = 0);
    LFS_VIS_API bool InputInt(const char* label, int* v, int step = 1, int step_fast = 100,
                              ImGuiInputTextFlags flags = 0);
    LFS_VIS_API bool DragFloat(const char* label, float* v, float speed = 1.0f, float min = 0.0f, float max = 0.0f,
                               const char* format = "%.3f", ImGuiSliderFlags flags = 0);
    LFS_VIS_API bool DragInt(const char* label, int* v, float speed = 1.0f, int min = 0, int max = 0,
                             const char* format = "%d", ImGuiSliderFlags flags = 0);
    LFS_VIS_API bool DragFloat2(const char* label, float v[2], float speed = 1.0f, float min = 0.0f, float max = 0.0f,
                                const char* format = "%.3f", ImGuiSliderFlags flags = 0);
    LFS_VIS_API bool DragFloat3(const char* label, float v[3], float speed = 1.0f, float min = 0.0f, float max = 0.0f,
                                const char* format = "%.3f", ImGuiSliderFlags flags = 0);
    LFS_VIS_API bool DragFloat4(const char* label, float v[4], float speed = 1.0f, float min = 0.0f, float max = 0.0f,
                                const char* format = "%.3f", ImGuiSliderFlags flags = 0);
    LFS_VIS_API bool DragInt2(const char* label, int v[2], float speed = 1.0f, int min = 0, int max = 0,
                              const char* format = "%d", ImGuiSliderFlags flags = 0);
    LFS_VIS_API bool DragInt3(const char* label, int v[3], float speed = 1.0f, int min = 0, int max = 0,
                              const char* format = "%d", ImGuiSliderFlags flags = 0);
    LFS_VIS_API bool DragInt4(const char* label, int v[4], float speed = 1.0f, int min = 0, int max = 0,
                              const char* format = "%d", ImGuiSliderFlags flags = 0);

    // Slider helpers enter text edit mode on click-release-without-drag and select the full value.
    LFS_VIS_API bool SliderFloat(const char* label, float* v, float min, float max,
                                 const char* format = "%.3f", ImGuiSliderFlags flags = 0);
    LFS_VIS_API bool SliderInt(const char* label, int* v, int min, int max,
                               const char* format = "%d", ImGuiSliderFlags flags = 0);
    LFS_VIS_API bool SliderFloat2(const char* label, float v[2], float min, float max,
                                  const char* format = "%.3f", ImGuiSliderFlags flags = 0);
    LFS_VIS_API bool SliderFloat3(const char* label, float v[3], float min, float max,
                                  const char* format = "%.3f", ImGuiSliderFlags flags = 0);
    LFS_VIS_API void RequestActiveEditCancel();

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
    void DrawShadowRect(ImDrawList* draw_list, const ImVec2& pos, const ImVec2& size,
                        float rounding = 6.0f, float alpha_scale = 1.0f,
                        float blur_scale = 1.0f, float offset_scale = 1.0f);
    void DrawFloatingWindowShadow(ImDrawList* draw_list, const ImVec2& pos, const ImVec2& size,
                                  float rounding = 6.0f);
    void DrawFloatingWindowShadow(const ImVec2& pos, const ImVec2& size, float rounding = 6.0f);
    void DrawPopoverShadowOverlay(ImDrawList* draw_list, const ImVec2& pos, const ImVec2& size,
                                  float rounding = 6.0f);
    void DrawModalShadow(ImDrawList* draw_list, const ImVec2& pos, const ImVec2& size,
                         float rounding = 6.0f);
    void DrawWindowShadow(const ImVec2& pos, const ImVec2& size, float rounding = 6.0f);

    // Icon button with selection state styling
    LFS_VIS_API bool IconButton(const char* id, ImTextureID texture, const ImVec2& size, bool selected = false,
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
