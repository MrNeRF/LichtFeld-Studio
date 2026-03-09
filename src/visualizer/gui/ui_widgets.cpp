/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/ui_widgets.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/image_io.hpp"
#include "gui/string_keys.hpp"
#include "internal/resource_paths.hpp"
#include "python/python_runtime.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "training/training_manager.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <glad/glad.h>
#include <implot.h>
#include <imgui.h>

namespace lfs::vis::gui::widgets {

    using namespace lfs::core::events;

    namespace {
        struct WidgetIcons {
            unsigned int reset = 0;
            bool initialized = false;
        };

        WidgetIcons g_icons;

        void ensureIconsLoaded() {
            if (g_icons.initialized)
                return;

            try {
                const auto path = lfs::vis::getAssetPath("icon/reset.png");
                const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(path);

                glGenTextures(1, &g_icons.reset);
                glBindTexture(GL_TEXTURE_2D, g_icons.reset);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            } catch (...) {
                g_icons.reset = 0;
            }

            g_icons.initialized = true;
        }

        ImVec4 getIconTint() {
            return theme().isLightTheme() ? ImVec4{0.2f, 0.2f, 0.2f, 0.9f} : ImVec4{1.0f, 1.0f, 1.0f, 0.9f};
        }
    } // namespace

    bool SliderWithReset(const char* label, float* v, float min, float max, float reset_value,
                         const char* tooltip, const char* format) {
        ensureIconsLoaded();

        bool changed = ImGui::SliderFloat(label, v, min, max, format);
        bool slider_hovered = ImGui::IsItemHovered();

        ImGui::SameLine();
        ImGui::PushID(label);

        const float btn_size = ImGui::GetFrameHeight();
        const ImVec2 icon_size(btn_size - 4, btn_size - 4);
        const ImVec4 icon_tint = getIconTint();

        if (g_icons.reset) {
            if (ImGui::ImageButton("##reset", static_cast<ImTextureID>(g_icons.reset), icon_size,
                                   ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0), icon_tint)) {
                *v = reset_value;
                changed = true;
            }
        } else {
            if (ImGui::Button("R", ImVec2(btn_size, btn_size))) {
                *v = reset_value;
                changed = true;
            }
        }

        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", LOC(lichtfeld::Strings::Common::RESET));
        }
        ImGui::PopID();

        if (tooltip && slider_hovered) {
            ImGui::SetTooltip("%s", tooltip);
        }

        return changed;
    }

    bool DragFloat3WithReset(const char* label, float* v, float speed, float reset_value,
                             const char* tooltip) {
        ensureIconsLoaded();

        bool changed = ImGui::DragFloat3(label, v, speed);
        bool drag_hovered = ImGui::IsItemHovered();

        ImGui::SameLine();
        ImGui::PushID(label);

        const float btn_size = ImGui::GetFrameHeight();
        const ImVec2 icon_size(btn_size - 4, btn_size - 4);
        const ImVec4 icon_tint = getIconTint();

        if (g_icons.reset) {
            if (ImGui::ImageButton("##reset", static_cast<ImTextureID>(g_icons.reset), icon_size,
                                   ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0), icon_tint)) {
                v[0] = v[1] = v[2] = reset_value;
                changed = true;
            }
        } else {
            if (ImGui::Button("R", ImVec2(btn_size, btn_size))) {
                v[0] = v[1] = v[2] = reset_value;
                changed = true;
            }
        }

        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", LOC(lichtfeld::Strings::Common::RESET));
        }
        ImGui::PopID();

        if (tooltip && drag_hovered) {
            ImGui::SetTooltip("%s", tooltip);
        }

        return changed;
    }

    void HelpMarker(const char* desc) {
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(desc);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    void TableRow(const char* label, const char* format, ...) {
        ImGui::Text("%s:", label);
        ImGui::SameLine(120 * lfs::python::get_shared_dpi_scale()); // Align values at column 120

        va_list args;
        va_start(args, format);
        ImGui::TextV(format, args);
        va_end(args);
    }

    void DrawProgressBar(float fraction, const char* overlay_text) {
        ImGui::ProgressBar(fraction, ImVec2(-1, 0), overlay_text);
    }

    void DrawLossPlot(const float* values, int count, float min_val, float max_val, const char* label) {
        if (count <= 0)
            return;

        // Ensure we have a valid, non-empty label
        const char* plot_label = (label && strlen(label) > 0) ? label : "Plot##default";

        // Simple line plot using ImGui
        ImGui::PlotLines(
            "Plot##default",
            values,
            count,
            0,
            plot_label,
            min_val,
            max_val,
            ImVec2(ImGui::GetContentRegionAvail().x, 80 * lfs::python::get_shared_dpi_scale()));
    }

    void DrawModeStatus(const UIContext& ctx) {
        using namespace lichtfeld::Strings;

        auto* scene_manager = ctx.viewer->getSceneManager();
        if (!scene_manager) {
            ImGui::Text("%s %s", LOC(Status::MODE), LOC(Status::UNKNOWN));
            return;
        }

        const auto& t = theme();
        const char* mode_str = LOC(Status::UNKNOWN);
        ImVec4 mode_color = t.palette.text_dim;

        // Content determines base mode
        SceneManager::ContentType content = scene_manager->getContentType();

        switch (content) {
        case SceneManager::ContentType::Empty:
            mode_str = LOC(Mode::EMPTY);
            mode_color = t.palette.text_dim;
            break;

        case SceneManager::ContentType::SplatFiles:
            mode_str = LOC(Mode::EDIT_MODE);
            mode_color = t.palette.info;
            break;

        case SceneManager::ContentType::Dataset: {
            // For dataset, check training state from TrainerManager
            auto* trainer_manager = scene_manager->getTrainerManager();
            if (!trainer_manager || !trainer_manager->hasTrainer()) {
                mode_str = LOC(Status::DATASET_NO_TRAINER);
                mode_color = t.palette.text_dim;
            } else {
                // Use trainer state for specific mode
                auto state = trainer_manager->getState();
                switch (state) {
                case TrainerManager::State::Ready:
                    mode_str = LOC(Status::DATASET_READY);
                    mode_color = t.palette.success;
                    break;
                case TrainerManager::State::Running:
                    mode_str = LOC(Status::TRAINING);
                    mode_color = t.palette.warning;
                    break;
                case TrainerManager::State::Paused:
                    mode_str = LOC(Status::TRAINING_PAUSED);
                    mode_color = lighten(t.palette.warning, -0.3f);
                    break;
                case TrainerManager::State::Finished: {
                    const auto reason = trainer_manager->getStateMachine().getFinishReason();
                    switch (reason) {
                    case FinishReason::Completed:
                        mode_str = LOC(Messages::TRAINING_COMPLETE);
                        mode_color = t.palette.success;
                        break;
                    case FinishReason::UserStopped:
                        mode_str = LOC(Messages::TRAINING_STOPPED);
                        mode_color = t.palette.text_dim;
                        break;
                    case FinishReason::Error:
                        mode_str = LOC(Messages::TRAINING_ERROR);
                        mode_color = t.palette.error;
                        break;
                    default:
                        mode_str = LOC(Status::TRAINING_FINISHED);
                        mode_color = t.palette.text_dim;
                    }
                    break;
                }
                case TrainerManager::State::Stopping:
                    mode_str = LOC(Status::STOPPING);
                    mode_color = darken(t.palette.error, 0.3f);
                    break;
                default:
                    mode_str = LOC(Mode::DATASET);
                    mode_color = t.palette.text_dim;
                }
            }
            break;
        }
        }

        ImGui::TextColored(mode_color, "%s %s", LOC(Status::MODE), mode_str);

        // Display scene info
        auto info = scene_manager->getSceneInfo();
        if (info.num_gaussians > 0) {
            ImGui::Text("%s %zu", LOC(Status::GAUSSIANS), info.num_gaussians);
        }

        if (info.source_type == "PLY" && info.num_nodes > 0) {
            ImGui::Text(LOC(Status::PLY_MODELS_COUNT), info.num_nodes);
        }

        // Display training iteration if actively training
        if (content == SceneManager::ContentType::Dataset) {
            auto* trainer_manager = scene_manager->getTrainerManager();
            if (trainer_manager && trainer_manager->isRunning()) {
                int iteration = trainer_manager->getCurrentIteration();
                if (iteration > 0) {
                    ImGui::Text("%s %d", LOC(Status::ITERATION), iteration);
                }
            }
        }
    }

    void DrawModeStatusWithContentSwitch(const UIContext& ctx) {
        DrawModeStatus(ctx);
    }

    void DrawShadowRect(ImDrawList* draw_list, const ImVec2& pos, const ImVec2& size,
                        const float rounding, const float alpha_scale,
                        const float blur_scale, const float offset_scale) {
        const auto& t = theme();
        if (!t.shadows.enabled || !draw_list)
            return;

        constexpr int LAYER_COUNT = 12;
        constexpr float FALLOFF_SCALE = 0.34f;
        constexpr float ROUNDING_SCALE = 0.18f;

        const ImVec4 shadow_tint = t.isLightTheme()
                                       ? ImVec4{0.08f, 0.10f, 0.16f, 1.0f}
                                       : ImVec4{0.0f, 0.0f, 0.0f, 1.0f};
        const float theme_alpha_scale = t.isLightTheme() ? 0.82f : 1.0f;
        const ImVec2& off = t.shadows.offset;
        const float blur = std::max(0.0f, t.shadows.blur * blur_scale);
        const float min_expand = std::max(1.0f, blur * 0.14f);
        const float base_alpha =
            std::clamp(t.shadows.alpha * theme_alpha_scale * alpha_scale, 0.0f, 1.0f);

        for (int i = 0; i < LAYER_COUNT; ++i) {
            const float t_val = static_cast<float>(i) / (LAYER_COUNT - 1);
            const float inv_t = 1.0f - t_val;
            const float falloff = std::pow(inv_t, 1.65f);
            const float alpha = base_alpha * falloff * FALLOFF_SCALE;
            if (alpha <= 0.0025f)
                continue;

            const float expand = min_expand + blur * (0.2f + 0.8f * t_val);
            const ImVec2 layer_off = {
                off.x * offset_scale * (0.45f + 0.55f * t_val),
                off.y * offset_scale * (0.45f + 0.55f * t_val)};
            const ImVec2 p1 = {pos.x + layer_off.x - expand, pos.y + layer_off.y - expand};
            const ImVec2 p2 = {pos.x + size.x + layer_off.x + expand, pos.y + size.y + layer_off.y + expand};
            draw_list->AddRectFilled(p1, p2, toU32WithAlpha(shadow_tint, alpha),
                                     rounding + expand * ROUNDING_SCALE);
        }
    }

    void DrawShadowRectOutside(ImDrawList* draw_list, const ImVec2& pos, const ImVec2& size,
                               const float rounding, const float alpha_scale,
                               const float blur_scale, const float offset_scale) {
        const auto& t = theme();
        if (!t.shadows.enabled || !draw_list)
            return;

        constexpr int LAYER_COUNT = 12;
        constexpr float FALLOFF_SCALE = 0.34f;
        constexpr float ROUNDING_SCALE = 0.18f;

        const ImVec4 shadow_tint = t.isLightTheme()
                                       ? ImVec4{0.08f, 0.10f, 0.16f, 1.0f}
                                       : ImVec4{0.0f, 0.0f, 0.0f, 1.0f};
        const float theme_alpha_scale = t.isLightTheme() ? 0.82f : 1.0f;
        const ImVec2& off = t.shadows.offset;
        const float blur = std::max(0.0f, t.shadows.blur * blur_scale);
        const float min_expand = std::max(1.0f, blur * 0.14f);
        const float base_alpha =
            std::clamp(t.shadows.alpha * theme_alpha_scale * alpha_scale, 0.0f, 1.0f);
        const ImVec2 inner_min = pos;
        const ImVec2 inner_max = {pos.x + size.x, pos.y + size.y};

        for (int i = 0; i < LAYER_COUNT; ++i) {
            const float t_val = static_cast<float>(i) / (LAYER_COUNT - 1);
            const float inv_t = 1.0f - t_val;
            const float falloff = std::pow(inv_t, 1.65f);
            const float alpha = base_alpha * falloff * FALLOFF_SCALE;
            if (alpha <= 0.0025f)
                continue;

            const float expand = min_expand + blur * (0.2f + 0.8f * t_val);
            const ImVec2 layer_off = {
                off.x * offset_scale * (0.45f + 0.55f * t_val),
                off.y * offset_scale * (0.45f + 0.55f * t_val)};
            const ImVec2 p1 = {pos.x + layer_off.x - expand, pos.y + layer_off.y - expand};
            const ImVec2 p2 = {pos.x + size.x + layer_off.x + expand,
                               pos.y + size.y + layer_off.y + expand};
            const float layer_rounding = rounding + expand * ROUNDING_SCALE;
            const ImU32 col = toU32WithAlpha(shadow_tint, alpha);

            const auto draw_band = [&](const ImVec2 clip_min, const ImVec2 clip_max) {
                if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y)
                    return;
                draw_list->PushClipRect(clip_min, clip_max, true);
                draw_list->AddRectFilled(p1, p2, col, layer_rounding);
                draw_list->PopClipRect();
            };

            draw_band(p1, {p2.x, inner_min.y});
            draw_band({p1.x, inner_max.y}, p2);
            draw_band({p1.x, inner_min.y}, {inner_min.x, inner_max.y});
            draw_band({inner_max.x, inner_min.y}, {p2.x, inner_max.y});
        }
    }

    void DrawFloatingWindowShadow(ImDrawList* draw_list, const ImVec2& pos, const ImVec2& size,
                                  const float rounding) {
        DrawShadowRect(draw_list, pos, size, rounding, 0.36f, 0.82f, 0.58f);
    }

    void DrawFloatingWindowShadow(const ImVec2& pos, const ImVec2& size, const float rounding) {
        DrawFloatingWindowShadow(ImGui::GetBackgroundDrawList(), pos, size, rounding);
    }

    void DrawPopoverShadowOverlay(ImDrawList* draw_list, const ImVec2& pos, const ImVec2& size,
                                  const float rounding) {
        DrawShadowRectOutside(draw_list, pos, size, rounding, 0.46f, 0.92f, 0.72f);
    }

    void DrawModalShadow(ImDrawList* draw_list, const ImVec2& pos, const ImVec2& size,
                         const float rounding) {
        DrawShadowRect(draw_list, pos, size, rounding, 0.68f, 1.08f, 0.92f);
    }

    void DrawWindowShadow(const ImVec2& pos, const ImVec2& size, const float rounding) {
        DrawFloatingWindowShadow(pos, size, rounding);
    }

    void DrawViewportVignette(const ImVec2& pos, const ImVec2& size) {
        const auto& t = theme();
        if (!t.vignette.enabled)
            return;

        constexpr float EDGE_SCALE = 0.5f;
        constexpr ImU32 CLEAR_COLOR = IM_COL32(0, 0, 0, 0);

        auto* const draw_list = ImGui::GetBackgroundDrawList();
        const float edge_mult = (1.0f - t.vignette.radius) * EDGE_SCALE * (1.0f + t.vignette.softness);
        const float edge_w = size.x * edge_mult;
        const float edge_h = size.y * edge_mult;
        const ImU32 dark = IM_COL32(0, 0, 0, static_cast<int>(t.vignette.intensity * 255.0f));

        const float x1 = pos.x, y1 = pos.y;
        const float x2 = pos.x + size.x, y2 = pos.y + size.y;

        draw_list->AddRectFilledMultiColor({x1, y1}, {x1 + edge_w, y2}, dark, CLEAR_COLOR, CLEAR_COLOR, dark);
        draw_list->AddRectFilledMultiColor({x2 - edge_w, y1}, {x2, y2}, CLEAR_COLOR, dark, dark, CLEAR_COLOR);
        draw_list->AddRectFilledMultiColor({x1, y1}, {x2, y1 + edge_h}, dark, dark, CLEAR_COLOR, CLEAR_COLOR);
        draw_list->AddRectFilledMultiColor({x1, y2 - edge_h}, {x2, y2}, CLEAR_COLOR, CLEAR_COLOR, dark, dark);
    }

    bool IconButton(const char* id, const unsigned int texture, const ImVec2& size,
                    const bool selected, const char* fallback_label) {
        constexpr float ACTIVE_DARKEN = 0.1f;
        constexpr float TINT_BASE = 0.7f;
        constexpr float TINT_ACCENT = 0.3f;
        constexpr float FALLBACK_PADDING = 8.0f;

        const auto& t = theme();
        const ImVec4 TINT_NORMAL =
            t.isLightTheme() ? ImVec4{0.2f, 0.2f, 0.2f, 0.9f} : ImVec4{1.0f, 1.0f, 1.0f, 0.9f};

        // Make button backgrounds transparent so they blend with toolbar, except when selected
        const ImVec4 bg_normal = selected ? t.button_selected() : ImVec4{0, 0, 0, 0};
        const ImVec4 bg_hovered = selected ? t.button_selected_hovered() : withAlpha(t.palette.surface_bright, 0.3f);
        const ImVec4 bg_active = selected ? darken(t.button_selected(), ACTIVE_DARKEN) : withAlpha(t.palette.surface_bright, 0.5f);
        const ImVec4 tint = selected
                                ? ImVec4{TINT_BASE + t.palette.primary.x * TINT_ACCENT,
                                         TINT_BASE + t.palette.primary.y * TINT_ACCENT,
                                         TINT_BASE + t.palette.primary.z * TINT_ACCENT, 1.0f}
                                : TINT_NORMAL;

        ImGui::PushStyleColor(ImGuiCol_Button, bg_normal);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bg_hovered);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, bg_active);

        const bool clicked = texture
                                 ? ImGui::ImageButton(id, static_cast<ImTextureID>(texture), size, {0, 0}, {1, 1}, {0, 0, 0, 0}, tint)
                                 : ImGui::Button(fallback_label, {size.x + FALLBACK_PADDING, size.y + FALLBACK_PADDING});

        ImGui::PopStyleColor(3);
        return clicked;
    }

    void SectionHeader(const char* text, const FontSet& fonts) {
        const auto& t = theme();
        if (fonts.section)
            ImGui::PushFont(fonts.section);
        ImGui::TextColored(t.palette.text_dim, "%s", text);
        if (fonts.section)
            ImGui::PopFont();
        ImGui::Separator();
    }

    bool ColoredButton(const char* label, const ButtonStyle style, const ImVec2& size) {
        const auto& t = theme();
        const ImVec4& base = t.palette.surface;

        const ImVec4 accent = [&]() {
            switch (style) {
            case ButtonStyle::Primary: return t.palette.primary;
            case ButtonStyle::Success: return t.palette.success;
            case ButtonStyle::Warning: return t.palette.warning;
            case ButtonStyle::Error: return t.palette.error;
            default: return t.palette.text_dim;
            }
        }();

        const auto blend = [&](const float f) {
            return ImVec4{base.x + (accent.x - base.x) * f,
                          base.y + (accent.y - base.y) * f,
                          base.z + (accent.z - base.z) * f, 1.0f};
        };

        ImGui::PushStyleColor(ImGuiCol_Button, blend(t.button.tint_normal));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, blend(t.button.tint_hover));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, blend(t.button.tint_active));

        const bool clicked = ImGui::Button(label, size);

        ImGui::PopStyleColor(3);
        return clicked;
    }

    void SetThemedTooltip(const char* fmt, ...) {
        const auto& t = theme();

        ImGui::PushStyleColor(ImGuiCol_PopupBg, withAlpha(t.palette.surface, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleColor(ImGuiCol_Border, t.palette.border);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 6));

        ImGui::BeginTooltip();

        va_list args;
        va_start(args, fmt);
        ImGui::TextV(fmt, args);
        va_end(args);

        ImGui::EndTooltip();

        ImGui::PopStyleVar(1);
        ImGui::PopStyleColor(3);
    }

    std::string formatNumber(const int64_t num) {
        const bool negative = num < 0;
        std::string result = std::to_string(negative ? -num : num);
        for (int i = static_cast<int>(result.length()) - 3; i > 0; i -= 3) {
            result.insert(i, ",");
        }
        return negative ? "-" + result : result;
    }

    bool InputIntFormatted(const char* label, int* v, const int step, const int step_fast) {
        constexpr size_t BUF_SIZE = 32;
        constexpr float BUTTON_COUNT = 2.0f;
        constexpr float SPACING_COUNT = 3.0f;

        ImGui::PushID(label);

        char buf[BUF_SIZE];
        const std::string formatted = formatNumber(*v);
        std::copy(formatted.begin(), formatted.end(), buf);
        buf[formatted.size()] = '\0';

        const float btn_size = ImGui::GetFrameHeight();
        const float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
        const float btns_width = step != 0 ? (btn_size * BUTTON_COUNT + spacing * SPACING_COUNT) : 0.0f;

        ImGui::SetNextItemWidth(ImGui::CalcItemWidth() - btns_width);

        bool changed = false;
        constexpr auto FLAGS = ImGuiInputTextFlags_CharsDecimal | ImGuiInputTextFlags_AutoSelectAll;
        if (ImGui::InputText("##input", buf, BUF_SIZE, FLAGS)) {
            int parsed = 0;
            bool has_digits = false;
            bool negative = false;
            for (const char* p = buf; *p; ++p) {
                if (*p == '-' && p == buf) {
                    negative = true;
                } else if (*p >= '0' && *p <= '9') {
                    parsed = parsed * 10 + (*p - '0');
                    has_digits = true;
                }
            }
            if (has_digits) {
                *v = negative ? -parsed : parsed;
                changed = true;
            }
        }

        if (step != 0) {
            const int delta = ImGui::GetIO().KeyCtrl ? step_fast : step;
            const ImVec2 btn_sz{btn_size, btn_size};
            ImGui::SameLine(0, spacing);
            if (ImGui::Button("-", btn_sz)) {
                *v -= delta;
                changed = true;
            }
            ImGui::SameLine(0, spacing);
            if (ImGui::Button("+", btn_sz)) {
                *v += delta;
                changed = true;
            }
        }

        ImGui::PopID();
        return changed;
    }

    bool ChromaticityPicker2D(const char* label, float* x, float* y, const float range,
                              const ImVec4& color_tint) {
        constexpr float PICKER_SIZE_BASE = 80.0f;
        constexpr float POINT_RADIUS_BASE = 5.0f;
        constexpr float POINT_OUTLINE_WIDTH = 1.5f;
        constexpr float COLOR_BLEND = 0.8f;
        constexpr float COLOR_OFFSET = 0.2f;

        const auto& t = theme();
        const float dpi = lfs::python::get_shared_dpi_scale();
        const float size = PICKER_SIZE_BASE * dpi;

        ImGui::PushID(label);

        bool changed = false;
        const ImVec2 cursor_pos = ImGui::GetCursorScreenPos();
        auto* const draw_list = ImGui::GetWindowDrawList();

        const ImU32 bg_color = t.isLightTheme() ? IM_COL32(240, 240, 240, 255) : IM_COL32(40, 40, 40, 255);
        const ImU32 grid_color = t.isLightTheme() ? IM_COL32(200, 200, 200, 255) : IM_COL32(70, 70, 70, 255);
        const ImU32 border_color = ImGui::ColorConvertFloat4ToU32(t.palette.border);

        draw_list->AddRectFilled(cursor_pos, ImVec2(cursor_pos.x + size, cursor_pos.y + size), bg_color);

        const float center = size * 0.5f;
        draw_list->AddLine(ImVec2(cursor_pos.x + center, cursor_pos.y),
                           ImVec2(cursor_pos.x + center, cursor_pos.y + size), grid_color);
        draw_list->AddLine(ImVec2(cursor_pos.x, cursor_pos.y + center),
                           ImVec2(cursor_pos.x + size, cursor_pos.y + center), grid_color);

        draw_list->AddRect(cursor_pos, ImVec2(cursor_pos.x + size, cursor_pos.y + size), border_color);

        ImGui::InvisibleButton("##picker", ImVec2(size, size));
        const bool is_active = ImGui::IsItemActive();
        const bool is_hovered = ImGui::IsItemHovered();

        if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            const ImVec2 mouse = ImGui::GetMousePos();
            *x = ((mouse.x - cursor_pos.x) / size - 0.5f) * 2.0f * range;
            *y = -((mouse.y - cursor_pos.y) / size - 0.5f) * 2.0f * range;
            *x = std::clamp(*x, -range, range);
            *y = std::clamp(*y, -range, range);
            changed = true;
        }

        if (is_hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            *x = 0.0f;
            *y = 0.0f;
            changed = true;
        }

        const float px = cursor_pos.x + center + (*x / range) * center;
        const float py = cursor_pos.y + center - (*y / range) * center;
        const float point_radius = POINT_RADIUS_BASE * dpi;

        const ImU32 point_outline = t.isLightTheme() ? IM_COL32(60, 60, 60, 255) : IM_COL32(255, 255, 255, 255);
        const ImU32 point_fill = ImGui::ColorConvertFloat4ToU32(ImVec4(
            color_tint.x * COLOR_BLEND + COLOR_OFFSET, color_tint.y * COLOR_BLEND + COLOR_OFFSET,
            color_tint.z * COLOR_BLEND + COLOR_OFFSET, 1.0f));

        draw_list->AddCircleFilled(ImVec2(px, py), point_radius, point_fill);
        draw_list->AddCircle(ImVec2(px, py), point_radius, point_outline, 0, POINT_OUTLINE_WIDTH);

        ImGui::SameLine();
        ImGui::BeginGroup();
        ImGui::TextColored(ImVec4(color_tint.x, color_tint.y, color_tint.z, 1.0f), "%s", label);
        ImGui::Text("X: %.3f", *x);
        ImGui::Text("Y: %.3f", *y);
        if (is_hovered) {
            ImGui::TextDisabled("(%s)", LOC(lichtfeld::Strings::Common::DOUBLE_CLICK_RESET));
        }
        ImGui::EndGroup();

        ImGui::PopID();
        return changed;
    }

    bool ChromaticityDiagram(const char* label, float* red_x, float* red_y, float* green_x, float* green_y,
                             float* blue_x, float* blue_y, float* neutral_x, float* neutral_y, const float range) {
        constexpr float DIAGRAM_SIZE_BASE = 140.0f;
        constexpr float POINT_RADIUS_BASE = 6.0f;
        constexpr float POINT_OUTLINE_WIDTH = 2.0f;
        constexpr float HIT_RADIUS_BASE = 10.0f;

        const auto& t = theme();
        const float dpi = lfs::python::get_shared_dpi_scale();
        const float size = DIAGRAM_SIZE_BASE * dpi;
        const float point_radius = POINT_RADIUS_BASE * dpi;
        const float hit_radius = HIT_RADIUS_BASE * dpi;

        ImGui::PushID(label);

        bool changed = false;
        const ImVec2 cursor_pos = ImGui::GetCursorScreenPos();
        auto* const draw_list = ImGui::GetWindowDrawList();

        // Draw rg chromaticity background gradient
        constexpr int GRID_RES = 24;
        const float cell_size = size / GRID_RES;
        for (int iy = 0; iy < GRID_RES; ++iy) {
            for (int ix = 0; ix < GRID_RES; ++ix) {
                const float r_chrom = static_cast<float>(ix) / (GRID_RES - 1);
                const float g_chrom = 1.0f - static_cast<float>(iy) / (GRID_RES - 1);
                const float b_chrom = std::max(0.0f, 1.0f - r_chrom - g_chrom);

                // Convert chromaticity to displayable RGB (with some saturation)
                const float intensity = 0.7f;
                float r = r_chrom * intensity + 0.15f;
                float g = g_chrom * intensity + 0.15f;
                float b = b_chrom * intensity + 0.15f;
                const float max_val = std::max({r, g, b});
                if (max_val > 1.0f) {
                    r /= max_val;
                    g /= max_val;
                    b /= max_val;
                }

                const ImU32 cell_color =
                    IM_COL32(static_cast<int>(r * 255), static_cast<int>(g * 255), static_cast<int>(b * 255), 255);
                const ImVec2 p0(cursor_pos.x + ix * cell_size, cursor_pos.y + iy * cell_size);
                const ImVec2 p1(p0.x + cell_size + 1, p0.y + cell_size + 1);
                draw_list->AddRectFilled(p0, p1, cell_color);
            }
        }

        // Border
        const ImU32 border_color = ImGui::ColorConvertFloat4ToU32(t.palette.border);
        draw_list->AddRect(cursor_pos, ImVec2(cursor_pos.x + size, cursor_pos.y + size), border_color, 0.0f, 0, 1.5f);

        // Reference chromaticity positions (where pure R, G, B, Gray would be)
        // In our normalized space: center = (0,0), range maps to half the widget
        const float center = size * 0.5f;

        // Control point data: {x_ptr, y_ptr, color, base_x, base_y, name}
        struct ControlPoint {
            float* x;
            float* y;
            ImU32 fill_color;
            ImU32 outline_color;
            const char* name;
        };

        ControlPoint points[4] = {
            {red_x, red_y, IM_COL32(255, 80, 80, 255), IM_COL32(180, 0, 0, 255), "R"},
            {green_x, green_y, IM_COL32(80, 220, 80, 255), IM_COL32(0, 150, 0, 255), "G"},
            {blue_x, blue_y, IM_COL32(80, 120, 255, 255), IM_COL32(0, 0, 180, 255), "B"},
            {neutral_x, neutral_y, IM_COL32(200, 200, 200, 255), IM_COL32(80, 80, 80, 255), "N"},
        };

        // Handle interaction
        ImGui::InvisibleButton("##diagram", ImVec2(size, size));
        const bool is_hovered = ImGui::IsItemHovered();

        static int dragging_point = -1;

        if (is_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            const ImVec2 mouse = ImGui::GetMousePos();
            float best_dist = hit_radius;
            dragging_point = -1;

            for (int i = 0; i < 4; ++i) {
                const float px = cursor_pos.x + center + (*points[i].x / range) * center;
                const float py = cursor_pos.y + center - (*points[i].y / range) * center;
                const float dx = mouse.x - px;
                const float dy = mouse.y - py;
                const float dist = std::sqrt(dx * dx + dy * dy);
                if (dist < best_dist) {
                    best_dist = dist;
                    dragging_point = i;
                }
            }
        }

        if (dragging_point >= 0 && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            const ImVec2 mouse = ImGui::GetMousePos();
            *points[dragging_point].x = ((mouse.x - cursor_pos.x) / size - 0.5f) * 2.0f * range;
            *points[dragging_point].y = -((mouse.y - cursor_pos.y) / size - 0.5f) * 2.0f * range;
            *points[dragging_point].x = std::clamp(*points[dragging_point].x, -range, range);
            *points[dragging_point].y = std::clamp(*points[dragging_point].y, -range, range);
            changed = true;
        }

        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            dragging_point = -1;
        }

        // Double-click to reset all
        if (is_hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            *red_x = *red_y = *green_x = *green_y = *blue_x = *blue_y = *neutral_x = *neutral_y = 0.0f;
            changed = true;
        }

        // Draw control points
        for (int i = 0; i < 4; ++i) {
            const float px = cursor_pos.x + center + (*points[i].x / range) * center;
            const float py = cursor_pos.y + center - (*points[i].y / range) * center;
            const float r = (i == dragging_point) ? point_radius * 1.3f : point_radius;

            draw_list->AddCircleFilled(ImVec2(px, py), r, points[i].fill_color);
            draw_list->AddCircle(ImVec2(px, py), r, points[i].outline_color, 0, POINT_OUTLINE_WIDTH);

            // Draw label
            const ImVec2 text_pos(px - 3, py - 4);
            draw_list->AddText(text_pos, IM_COL32(0, 0, 0, 255), points[i].name);
        }

        // Side info
        ImGui::SameLine();
        ImGui::BeginGroup();
        ImGui::Text("%s", label);
        if (is_hovered) {
            ImGui::TextDisabled("(%s)", LOC(lichtfeld::Strings::Common::DOUBLE_CLICK_RESET));
        }
        ImGui::EndGroup();

        ImGui::PopID();
        return changed;
    }

    void CRFCurvePreview(const char* label, const float gamma, const float toe, const float shoulder,
                         const float gamma_r, const float gamma_g, const float gamma_b) {
        constexpr float PLOT_WIDTH_BASE = 200.0f;
        constexpr float PLOT_HEIGHT_BASE = 120.0f;
        constexpr int NUM_POINTS = 64;
        constexpr float TOE_FACTOR = 0.5f;
        constexpr float SHOULDER_FACTOR = 0.3f;
        constexpr float MIDPOINT = 0.5f;

        const auto& t = theme();
        const float dpi = lfs::python::get_shared_dpi_scale();
        const ImVec2 plot_size(PLOT_WIDTH_BASE * dpi, PLOT_HEIGHT_BASE * dpi);

        float xs[NUM_POINTS];
        float ys_combined[NUM_POINTS];
        float ys_r[NUM_POINTS];
        float ys_g[NUM_POINTS];
        float ys_b[NUM_POINTS];

        const auto apply_crf = [](const float x, const float g, const float t_param, const float s_param) {
            float y = std::pow(x, 1.0f / g);
            if (t_param != 0.0f && x < MIDPOINT) {
                const float t_factor = 1.0f + t_param * TOE_FACTOR;
                y = y * t_factor - (t_factor - 1.0f) * x * 2.0f * (MIDPOINT - x);
            }
            if (s_param != 0.0f && x > MIDPOINT) {
                const float s_factor = 1.0f - s_param * SHOULDER_FACTOR;
                const float blend = (x - MIDPOINT) * 2.0f;
                y = y * (1.0f - blend * (1.0f - s_factor));
            }
            return std::clamp(y, 0.0f, 1.0f);
        };

        const bool has_per_channel = (gamma_r != 0.0f || gamma_g != 0.0f || gamma_b != 0.0f);

        for (int i = 0; i < NUM_POINTS; ++i) {
            xs[i] = static_cast<float>(i) / (NUM_POINTS - 1);
            ys_combined[i] = apply_crf(xs[i], gamma, toe, shoulder);
            if (has_per_channel) {
                ys_r[i] = apply_crf(xs[i], gamma * (1.0f + gamma_r), toe, shoulder);
                ys_g[i] = apply_crf(xs[i], gamma * (1.0f + gamma_g), toe, shoulder);
                ys_b[i] = apply_crf(xs[i], gamma * (1.0f + gamma_b), toe, shoulder);
            }
        }

        ImPlot::PushStyleColor(ImPlotCol_FrameBg,
                               t.isLightTheme() ? ImVec4(0.95f, 0.95f, 0.95f, 1.0f) : ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        ImPlot::PushStyleColor(ImPlotCol_PlotBg,
                               t.isLightTheme() ? ImVec4(1.0f, 1.0f, 1.0f, 1.0f) : ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
        ImPlot::PushStyleColor(ImPlotCol_PlotBorder, t.palette.border);
        ImPlot::PushStyleColor(ImPlotCol_Line, t.palette.text);

        constexpr auto PLOT_FLAGS = ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMenus |
                                    ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText;
        constexpr auto AXIS_FLAGS = ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_Lock;

        if (ImPlot::BeginPlot(label, plot_size, PLOT_FLAGS)) {
            ImPlot::SetupAxes(nullptr, nullptr, AXIS_FLAGS, AXIS_FLAGS);
            ImPlot::SetupAxesLimits(0.0, 1.0, 0.0, 1.0, ImPlotCond_Always);

            const float diag[] = {0.0f, 1.0f};
            ImPlot::PushStyleColor(ImPlotCol_Line,
                                   t.isLightTheme() ? ImVec4(0.8f, 0.8f, 0.8f, 1.0f) : ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
            ImPlot::PlotLine("##diag", diag, diag, 2);
            ImPlot::PopStyleColor();

            if (has_per_channel) {
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.9f, 0.3f, 0.3f, 0.8f));
                ImPlot::PlotLine("##r", xs, ys_r, NUM_POINTS);
                ImPlot::PopStyleColor();

                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.3f, 0.8f, 0.3f, 0.8f));
                ImPlot::PlotLine("##g", xs, ys_g, NUM_POINTS);
                ImPlot::PopStyleColor();

                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.3f, 0.5f, 0.9f, 0.8f));
                ImPlot::PlotLine("##b", xs, ys_b, NUM_POINTS);
                ImPlot::PopStyleColor();
            } else {
                ImPlot::PushStyleColor(ImPlotCol_Line, t.palette.primary);
                ImPlot::PlotLine("##curve", xs, ys_combined, NUM_POINTS);
                ImPlot::PopStyleColor();
            }

            ImPlot::EndPlot();
        }

        ImPlot::PopStyleColor(4);
    }

} // namespace lfs::vis::gui::widgets
