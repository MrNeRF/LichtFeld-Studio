/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/startup_overlay.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/string_keys.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <imgui.h>
#ifdef _WIN32
#include <shellapi.h>
#include <windows.h>
#endif

namespace lfs::vis::gui {

    void StartupOverlay::loadTextures() {
        const auto load = [](const std::filesystem::path& path, rendering::Texture& tex, int& w, int& h) {
            try {
                const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(path);
                assert(channels == 4);
                glGenTextures(1, tex.ptr());
                glBindTexture(GL_TEXTURE_2D, tex.get());
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
                lfs::core::free_image(data);
                glBindTexture(GL_TEXTURE_2D, 0);
                w = width;
                h = height;
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load overlay texture {}: {}", lfs::core::path_to_utf8(path), e.what());
            }
        };
        load(lfs::vis::getAssetPath("lichtfeld-splash-logo.png"), logo_light_texture_, logo_width_, logo_height_);
        load(lfs::vis::getAssetPath("lichtfeld-splash-logo-dark.png"), logo_dark_texture_, logo_width_, logo_height_);
        load(lfs::vis::getAssetPath("core11-logo.png"), core11_light_texture_, core11_width_, core11_height_);
        load(lfs::vis::getAssetPath("core11-logo-dark.png"), core11_dark_texture_, core11_width_, core11_height_);
        load(lfs::vis::getAssetPath("icon/discord.png"), discord_icon_texture_, discord_icon_width_, discord_icon_height_);
        load(lfs::vis::getAssetPath("icon/x-twitter.png"), x_icon_texture_, x_icon_width_, x_icon_height_);
        load(lfs::vis::getAssetPath("icon/heart.png"), heart_icon_texture_, heart_icon_width_, heart_icon_height_);
    }

    void StartupOverlay::destroyTextures() {
        logo_light_texture_ = {};
        logo_dark_texture_ = {};
        core11_light_texture_ = {};
        core11_dark_texture_ = {};
        discord_icon_texture_ = {};
        x_icon_texture_ = {};
        heart_icon_texture_ = {};
    }

    void StartupOverlay::openURL(const char* url) {
#ifdef _WIN32
        ShellExecuteA(nullptr, "open", url, nullptr, nullptr, SW_SHOWNORMAL);
#else
        std::string cmd = "xdg-open \"" + std::string(url) + "\" &";
        std::system(cmd.c_str());
#endif
    }

    void StartupOverlay::render(const ViewportLayout& viewport, ImFont* font_small, bool drag_hovering) {
        if (!visible_)
            return;

        static constexpr float MIN_VIEWPORT_SIZE = 100.0f;
        if (viewport.size.x < MIN_VIEWPORT_SIZE || viewport.size.y < MIN_VIEWPORT_SIZE)
            return;

        static constexpr float MAIN_LOGO_SCALE = 1.3f;
        static constexpr float CORE11_LOGO_SCALE = 0.5f;
        static constexpr float CORNER_RADIUS = 12.0f;
        static constexpr float PADDING_X = 40.0f;
        static constexpr float PADDING_Y = 28.0f;
        static constexpr float GAP_LOGO_TEXT = 20.0f;
        static constexpr float GAP_TEXT_CORE11 = 10.0f;
        static constexpr float GAP_CORE11_SOCIAL = 16.0f;
        static constexpr float GAP_SOCIAL_LANG = 14.0f;
        static constexpr float GAP_LANG_HINT = 12.0f;
        static constexpr float SOCIAL_ICON_SIZE = 20.0f;
        static constexpr float SOCIAL_ICON_TEXT_GAP = 6.0f;
        static constexpr float SOCIAL_ITEM_GAP = 24.0f;
        static constexpr float LANG_COMBO_WIDTH = 140.0f;

        const auto& t = theme();
        const bool is_dark_theme = (t.name == "Dark");
        const GLuint logo_texture = is_dark_theme ? logo_light_texture_.get() : logo_dark_texture_.get();
        const GLuint core11_texture = is_dark_theme ? core11_light_texture_.get() : core11_dark_texture_.get();

        const float main_logo_w = static_cast<float>(logo_width_) * MAIN_LOGO_SCALE;
        const float main_logo_h = static_cast<float>(logo_height_) * MAIN_LOGO_SCALE;
        const float core11_w = static_cast<float>(core11_width_) * CORE11_LOGO_SCALE;
        const float core11_h = static_cast<float>(core11_height_) * CORE11_LOGO_SCALE;

        const char* supported_text = LOC(lichtfeld::Strings::Startup::SUPPORTED_BY);
        const char* click_hint = LOC(lichtfeld::Strings::Startup::CLICK_TO_CONTINUE);
        if (font_small)
            ImGui::PushFont(font_small);
        const ImVec2 supported_size = ImGui::CalcTextSize(supported_text);
        const ImVec2 hint_size = ImGui::CalcTextSize(click_hint);
        const ImVec2 lang_label_size = ImGui::CalcTextSize(LOC(lichtfeld::Strings::Preferences::LANGUAGE));
        if (font_small)
            ImGui::PopFont();

        const char* discord_label = "Discord";
        const char* x_label = "@janusch_patas";
        const char* donate_label = "Donate";
        if (font_small)
            ImGui::PushFont(font_small);
        const ImVec2 discord_label_size = ImGui::CalcTextSize(discord_label);
        const ImVec2 x_label_size = ImGui::CalcTextSize(x_label);
        const ImVec2 donate_label_size = ImGui::CalcTextSize(donate_label);
        if (font_small)
            ImGui::PopFont();
        const float social_row_width = SOCIAL_ICON_SIZE + SOCIAL_ICON_TEXT_GAP + discord_label_size.x +
                                       SOCIAL_ITEM_GAP +
                                       SOCIAL_ICON_SIZE + SOCIAL_ICON_TEXT_GAP + x_label_size.x +
                                       SOCIAL_ITEM_GAP +
                                       SOCIAL_ICON_SIZE + SOCIAL_ICON_TEXT_GAP + donate_label_size.x;
        const float social_row_height = std::max(SOCIAL_ICON_SIZE, discord_label_size.y);

        if (font_small)
            ImGui::PushFont(font_small);
        const float lang_row_height = ImGui::GetFrameHeight() + 4.0f;
        if (font_small)
            ImGui::PopFont();
        const float content_width = std::max({main_logo_w, core11_w, supported_size.x, hint_size.x,
                                              LANG_COMBO_WIDTH + lang_label_size.x + 8.0f, social_row_width});
        const float content_height = main_logo_h + GAP_LOGO_TEXT + supported_size.y + GAP_TEXT_CORE11 +
                                     core11_h + GAP_CORE11_SOCIAL + social_row_height + GAP_SOCIAL_LANG +
                                     lang_row_height + GAP_LANG_HINT + hint_size.y;
        const float overlay_width = content_width + PADDING_X * 2.0f;
        const float overlay_height = content_height + PADDING_Y * 2.0f;

        const float center_x = viewport.pos.x + viewport.size.x * 0.5f;
        const float center_y = viewport.pos.y + viewport.size.y * 0.5f;
        const ImVec2 overlay_pos(center_x - overlay_width * 0.5f, center_y - overlay_height * 0.5f);

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, CORNER_RADIUS);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {PADDING_X, PADDING_Y});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, t.palette.surface);
        ImGui::PushStyleColor(ImGuiCol_Border, t.palette.border);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, t.palette.background);
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, lighten(t.palette.background, 0.05f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, lighten(t.palette.background, 0.08f));
        ImGui::PushStyleColor(ImGuiCol_PopupBg, t.palette.surface);
        ImGui::PushStyleColor(ImGuiCol_Header, t.palette.primary);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, lighten(t.palette.primary, 0.1f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, t.palette.primary);

        ImGui::SetNextWindowPos(overlay_pos);
        ImGui::SetNextWindowSize({overlay_width, overlay_height});

        bool overlay_item_active = false;
        bool overlay_focused = false;
        if (ImGui::Begin("##StartupOverlay", nullptr,
                         ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                             ImGuiWindowFlags_NoCollapse)) {
            overlay_focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            const ImVec2 window_pos = ImGui::GetWindowPos();
            const float window_center_x = window_pos.x + overlay_width * 0.5f;
            float y = window_pos.y + PADDING_Y;

            if (logo_texture && logo_width_ > 0) {
                const float x = window_center_x - main_logo_w * 0.5f;
                draw_list->AddImage(static_cast<ImTextureID>(logo_texture),
                                    {x, y}, {x + main_logo_w, y + main_logo_h});
                y += main_logo_h + GAP_LOGO_TEXT;
            }

            if (font_small)
                ImGui::PushFont(font_small);
            draw_list->AddText({window_center_x - supported_size.x * 0.5f, y},
                               toU32WithAlpha(t.palette.text_dim, 0.85f), supported_text);
            y += supported_size.y + GAP_TEXT_CORE11;

            if (core11_texture && core11_width_ > 0) {
                const float x = window_center_x - core11_w * 0.5f;
                draw_list->AddImage(static_cast<ImTextureID>(core11_texture),
                                    {x, y}, {x + core11_w, y + core11_h});
                y += core11_h + GAP_CORE11_SOCIAL;
            }
            {
                const float social_x = window_center_x - social_row_width * 0.5f;
                const float text_y = y + (SOCIAL_ICON_SIZE - discord_label_size.y) * 0.5f;
                float sx = social_x;

                const ImVec4 tint_vec = t.palette.text_dim;
                const auto tint = toU32WithAlpha(tint_vec, 0.85f);

                const float discord_icon_w = discord_icon_height_ > 0
                                                 ? SOCIAL_ICON_SIZE * static_cast<float>(discord_icon_width_) / static_cast<float>(discord_icon_height_)
                                                 : SOCIAL_ICON_SIZE;
                if (discord_icon_texture_.get()) {
                    draw_list->AddImage(static_cast<ImTextureID>(discord_icon_texture_.get()),
                                        {sx, y}, {sx + discord_icon_w, y + SOCIAL_ICON_SIZE},
                                        {0, 0}, {1, 1}, toU32WithAlpha(tint_vec, 0.85f));
                }
                sx += discord_icon_w + SOCIAL_ICON_TEXT_GAP;
                draw_list->AddText({sx, text_y}, tint, discord_label);
                sx += discord_label_size.x;

                const float discord_hit_w = sx - social_x;
                ImGui::SetCursorScreenPos({social_x, y});
                if (ImGui::InvisibleButton("##discord_link", {discord_hit_w, SOCIAL_ICON_SIZE})) {
                    openURL("https://discord.gg/NqwTqVYVmj");
                }
                overlay_item_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    draw_list->AddLine({social_x + discord_icon_w + SOCIAL_ICON_TEXT_GAP, text_y + discord_label_size.y},
                                       {sx, text_y + discord_label_size.y},
                                       toU32WithAlpha(tint_vec, 0.6f));
                }

                sx += SOCIAL_ITEM_GAP;
                const float x_start = sx;

                const float x_icon_w = x_icon_height_ > 0
                                           ? SOCIAL_ICON_SIZE * static_cast<float>(x_icon_width_) / static_cast<float>(x_icon_height_)
                                           : SOCIAL_ICON_SIZE;
                if (x_icon_texture_.get()) {
                    draw_list->AddImage(static_cast<ImTextureID>(x_icon_texture_.get()),
                                        {sx, y}, {sx + x_icon_w, y + SOCIAL_ICON_SIZE},
                                        {0, 0}, {1, 1}, toU32WithAlpha(tint_vec, 0.85f));
                }
                sx += x_icon_w + SOCIAL_ICON_TEXT_GAP;
                draw_list->AddText({sx, text_y}, tint, x_label);
                sx += x_label_size.x;

                const float x_hit_w = sx - x_start;
                ImGui::SetCursorScreenPos({x_start, y});
                if (ImGui::InvisibleButton("##x_link", {x_hit_w, SOCIAL_ICON_SIZE})) {
                    openURL("https://x.com/janusch_patas");
                }
                overlay_item_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    draw_list->AddLine({x_start + x_icon_w + SOCIAL_ICON_TEXT_GAP, text_y + x_label_size.y},
                                       {sx, text_y + x_label_size.y},
                                       toU32WithAlpha(tint_vec, 0.6f));
                }

                sx += SOCIAL_ITEM_GAP;
                const float donate_start = sx;

                if (heart_icon_texture_.get()) {
                    draw_list->AddImage(static_cast<ImTextureID>(heart_icon_texture_.get()),
                                        {sx, y}, {sx + SOCIAL_ICON_SIZE, y + SOCIAL_ICON_SIZE},
                                        {0, 0}, {1, 1}, IM_COL32(220, 50, 50, 230));
                }

                sx += SOCIAL_ICON_SIZE + SOCIAL_ICON_TEXT_GAP;
                draw_list->AddText({sx, text_y}, tint, donate_label);
                sx += donate_label_size.x;

                const float donate_hit_w = sx - donate_start;
                ImGui::SetCursorScreenPos({donate_start, y});
                if (ImGui::InvisibleButton("##donate_link", {donate_hit_w, SOCIAL_ICON_SIZE})) {
                    openURL("https://lichtfeld.io/#support-the-project");
                }
                overlay_item_active |= ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    draw_list->AddLine({donate_start + SOCIAL_ICON_SIZE + SOCIAL_ICON_TEXT_GAP, text_y + donate_label_size.y},
                                       {sx, text_y + donate_label_size.y},
                                       toU32WithAlpha(tint_vec, 0.6f));
                }

                y += social_row_height + GAP_SOCIAL_LANG;
            }

            const float lang_total_width = lang_label_size.x + 8.0f + LANG_COMBO_WIDTH;
            const float content_area_width = overlay_width - 2.0f * PADDING_X;
            const float lang_indent = (content_area_width - lang_total_width) * 0.5f;
            ImGui::SetCursorPosY(y - window_pos.y);
            ImGui::SetCursorPosX(PADDING_X + lang_indent);
            ImGui::TextColored(t.palette.text_dim, "%s", LOC(lichtfeld::Strings::Preferences::LANGUAGE));
            ImGui::SameLine(0.0f, 8.0f);
            ImGui::SetNextItemWidth(LANG_COMBO_WIDTH);

            auto& loc = lfs::event::LocalizationManager::getInstance();
            const auto& current_lang = loc.getCurrentLanguage();
            const auto available_langs = loc.getAvailableLanguages();
            const auto lang_names = loc.getAvailableLanguageNames();

            std::string current_name = current_lang;
            for (size_t i = 0; i < available_langs.size(); ++i) {
                if (available_langs[i] == current_lang) {
                    current_name = lang_names[i];
                    break;
                }
            }

            if (ImGui::BeginCombo("##LangCombo", current_name.c_str())) {
                for (size_t i = 0; i < available_langs.size(); ++i) {
                    const bool is_selected = (available_langs[i] == current_lang);
                    if (ImGui::Selectable(lang_names[i].c_str(), is_selected)) {
                        loc.setLanguage(available_langs[i]);
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            overlay_item_active |= ImGui::IsItemActive();

            y += lang_row_height + GAP_LANG_HINT;

            draw_list->AddText({window_center_x - hint_size.x * 0.5f, y},
                               toU32WithAlpha(t.palette.text_dim, 0.5f), click_hint);
            if (font_small)
                ImGui::PopFont();
        }
        ImGui::End();
        ImGui::PopStyleColor(9);
        ImGui::PopStyleVar(5);

        const auto& io = ImGui::GetIO();
        const bool mouse_action = ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
                                  ImGui::IsMouseClicked(ImGuiMouseButton_Right) ||
                                  ImGui::IsMouseClicked(ImGuiMouseButton_Middle) ||
                                  std::abs(io.MouseWheel) > 0.0f || std::abs(io.MouseWheelH) > 0.0f;
        const bool key_action = io.InputQueueCharacters.Size > 0 ||
                                ImGui::IsKeyPressed(ImGuiKey_Escape) ||
                                ImGui::IsKeyPressed(ImGuiKey_Space) ||
                                ImGui::IsKeyPressed(ImGuiKey_Enter);

        ++shown_frames_;
        const bool any_popup_open = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
        if (shown_frames_ > 2 && !any_popup_open && !overlay_item_active && !drag_hovering) {
            if (!overlay_focused || mouse_action || key_action)
                visible_ = false;
        }
    }

} // namespace lfs::vis::gui
