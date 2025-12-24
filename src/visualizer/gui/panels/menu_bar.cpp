/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/menu_bar.hpp"
#include "config.h"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/tensor_trace.hpp"
#include "core/training_snapshot.hpp"
#ifdef WIN32
#include <winsock2.h>
#endif
#include "gui/utils/windows_utils.hpp"
#include "theme/theme.hpp"

#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <imgui.h>

#include <cstdlib>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

namespace lfs::vis::gui {

    MenuBar::MenuBar() = default;

    MenuBar::~MenuBar() {
        for (const auto& [id, thumb] : thumbnails_) {
            if (thumb.texture)
                glDeleteTextures(1, &thumb.texture);
        }
    }

    void MenuBar::startThumbnailDownload(const std::string& video_id) {
        if (video_id.empty() || thumbnails_.contains(video_id))
            return;

        auto& thumb = thumbnails_[video_id];
        thumb.state = Thumbnail::State::LOADING;

        thumb.download_future = std::async(std::launch::async, [video_id]() -> std::vector<uint8_t> {
            httplib::Client cli("https://img.youtube.com");
            cli.set_connection_timeout(5);
            cli.set_read_timeout(5);

            if (const auto res = cli.Get("/vi/" + video_id + "/mqdefault.jpg"))
                if (res->status == 200)
                    return {res->body.begin(), res->body.end()};
            return {};
        });
    }

    void MenuBar::updateThumbnails() {
        for (auto& [id, thumb] : thumbnails_) {
            if (thumb.state != Thumbnail::State::LOADING || !thumb.download_future.valid())
                continue;
            if (thumb.download_future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
                continue;

            const auto data = thumb.download_future.get();
            if (data.empty()) {
                thumb.state = Thumbnail::State::FAILED;
                continue;
            }

            try {
                auto [pixels, w, h, c] = lfs::core::load_image_from_memory(data.data(), data.size());
                if (!pixels) {
                    thumb.state = Thumbnail::State::FAILED;
                    continue;
                }

                GLuint tex = 0;
                glGenTextures(1, &tex);
                glBindTexture(GL_TEXTURE_2D, tex);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
                glBindTexture(GL_TEXTURE_2D, 0);

                lfs::core::free_image(pixels);
                thumb.texture = tex;
                thumb.state = Thumbnail::State::READY;
            } catch (...) {
                thumb.state = Thumbnail::State::FAILED;
            }
        }
    }

    void MenuBar::renderVideoCard(const char* title, const char* video_id, const char* url) {
        constexpr float CARD_WIDTH = 160.0f;
        constexpr float CARD_HEIGHT = 90.0f;
        constexpr float CARD_ROUNDING = 4.0f;
        constexpr float PLAY_ICON_RADIUS = 15.0f;

        if (!thumbnails_.contains(video_id))
            startThumbnailDownload(video_id);

        auto& thumb = thumbnails_[video_id];
        const auto& t = theme();
        const ImVec2 card_size(CARD_WIDTH, CARD_HEIGHT + 30.0f);
        const ImVec2 cursor = ImGui::GetCursorScreenPos();

        if (ImGui::InvisibleButton(video_id, card_size))
            openURL(url);

        const bool hovered = ImGui::IsItemHovered();
        if (hovered)
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

        auto* const dl = ImGui::GetWindowDrawList();
        const ImU32 bg = hovered ? toU32(lighten(t.palette.surface_bright, 0.1f)) : toU32(t.palette.surface_bright);
        dl->AddRectFilled(cursor, {cursor.x + card_size.x, cursor.y + card_size.y}, bg, CARD_ROUNDING);

        if (thumb.state == Thumbnail::State::READY && thumb.texture) {
            dl->AddImage(static_cast<ImTextureID>(thumb.texture),
                         cursor, {cursor.x + CARD_WIDTH, cursor.y + CARD_HEIGHT});
        } else {
            dl->AddRectFilled(cursor, {cursor.x + CARD_WIDTH, cursor.y + CARD_HEIGHT},
                              toU32(darken(t.palette.surface_bright, 0.1f)), CARD_ROUNDING);

            const float cx = cursor.x + CARD_WIDTH * 0.5f;
            const float cy = cursor.y + CARD_HEIGHT * 0.5f;
            dl->AddTriangleFilled(
                {cx - PLAY_ICON_RADIUS * 0.4f, cy - PLAY_ICON_RADIUS * 0.6f},
                {cx - PLAY_ICON_RADIUS * 0.4f, cy + PLAY_ICON_RADIUS * 0.6f},
                {cx + PLAY_ICON_RADIUS * 0.6f, cy},
                toU32(withAlpha(t.palette.text, 0.6f)));

            if (thumb.state == Thumbnail::State::LOADING)
                dl->AddText({cursor.x + 4, cursor.y + CARD_HEIGHT - 16}, toU32(t.palette.text_dim), "Loading...");
        }

        dl->AddText({cursor.x + 4, cursor.y + CARD_HEIGHT + 4.0f}, toU32(t.palette.text), title);
    }

    void MenuBar::render() {
        const auto& t = theme();

        // Use regular font for entire menu bar
        if (fonts_.regular)
            ImGui::PushFont(fonts_.regular);

        ImGui::PushStyleColor(ImGuiCol_MenuBarBg, t.menu_background());
        ImGui::PushStyleColor(ImGuiCol_Header, t.menu_active());
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, t.menu_hover());
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, t.menu_active());
        ImGui::PushStyleColor(ImGuiCol_PopupBg, t.menu_popup_background());
        ImGui::PushStyleColor(ImGuiCol_Border, t.menu_border());
        ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, t.menu.popup_rounding);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, t.menu.popup_border_size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, t.menu.popup_padding);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, t.menu.frame_padding);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, t.menu.item_spacing);

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                const bool can_clear = !can_clear_ || can_clear_();
                if (ImGui::MenuItem("New Project", nullptr, false, can_clear) && on_new_project_) {
                    on_new_project_();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Import Dataset") && on_import_dataset_) {
                    on_import_dataset_();
                }
                if (ImGui::MenuItem("Import Ply") && on_import_ply_) {
                    on_import_ply_();
                }
                if (ImGui::MenuItem("Import Checkpoint") && on_import_checkpoint_) {
                    on_import_checkpoint_();
                }
                if (ImGui::MenuItem("Import Config") && on_import_config_) {
                    on_import_config_();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Export...") && on_export_) {
                    on_export_();
                }
                if (ImGui::MenuItem("Export Config...") && on_export_config_) {
                    on_export_config_();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Exit") && on_exit_) {
                    on_exit_();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Edit")) {
                if (ImGui::MenuItem("Input Settings...")) {
                    show_input_settings_ = true;
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("View")) {
                if (ImGui::BeginMenu("Theme")) {
                    const bool is_dark = (theme().name == "Dark");
                    if (ImGui::MenuItem("Dark", nullptr, is_dark)) {
                        setTheme(darkTheme());
                    }
                    if (ImGui::MenuItem("Light", nullptr, !is_dark)) {
                        setTheme(lightTheme());
                    }
                    ImGui::EndMenu();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Debug Info...")) {
                    show_debug_window_ = true;
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Help")) {
                if (ImGui::MenuItem("Getting Started")) {
                    show_getting_started_ = true;
                }
                if (ImGui::MenuItem("About LichtFeld Studio")) {
                    show_about_window_ = true;
                }
                ImGui::EndMenu();
            }

            const float h = ImGui::GetWindowHeight();
            ImGui::GetWindowDrawList()->AddLine({0, h - 1}, {ImGui::GetWindowWidth(), h - 1},
                                                t.menu_bottom_border_u32(), 1.0f);

            ImGui::EndMainMenuBar();
        }

        ImGui::PopStyleVar(5);
        ImGui::PopStyleColor(6);
        if (fonts_.regular)
            ImGui::PopFont();

        renderGettingStartedWindow();
        renderAboutWindow();
        renderInputSettingsWindow();
        renderDebugWindow();
    }

    void MenuBar::openURL(const char* url) {
#ifdef _WIN32
        ShellExecuteA(nullptr, "open", url, nullptr, nullptr, SW_SHOWNORMAL);
#else
        std::string cmd = "xdg-open " + std::string(url);
        system(cmd.c_str());
#endif
    }

    void MenuBar::renderGettingStartedWindow() {
        if (!show_getting_started_)
            return;

        constexpr ImGuiWindowFlags WINDOW_FLAGS = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize;
        constexpr float WINDOW_ROUNDING = 8.0f;
        constexpr float WINDOW_PADDING = 20.0f;
        constexpr float ITEM_SPACING_Y = 12.0f;
        constexpr float VIDEO_SPACING = 16.0f;
        constexpr float INDENT = 25.0f;

        const auto& t = theme();

        ImGui::SetNextWindowSize(ImVec2(560, 0), ImGuiCond_Once);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, WINDOW_ROUNDING);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {WINDOW_PADDING, WINDOW_PADDING});
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8.0f, ITEM_SPACING_Y});
        ImGui::PushStyleColor(ImGuiCol_WindowBg, withAlpha(t.palette.surface, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, t.palette.surface);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, t.palette.surface_bright);
        ImGui::PushStyleColor(ImGuiCol_Border, withAlpha(t.palette.info, 0.3f));

        if (ImGui::Begin("Getting Started", &show_getting_started_, WINDOW_FLAGS)) {
            updateThumbnails();

            if (fonts_.heading)
                ImGui::PushFont(fonts_.heading);
            ImGui::TextColored(t.palette.info, "QUICK START GUIDE");
            if (fonts_.heading)
                ImGui::PopFont();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextWrapped("Learn how to prepare datasets and get started with LichtFeld Studio:");
            ImGui::Spacing();
            ImGui::Spacing();

            renderVideoCard("Reality Scan Dataset", "JWmkhTlbDvg", "https://www.youtube.com/watch?v=JWmkhTlbDvg");
            ImGui::SameLine(0.0f, VIDEO_SPACING);
            renderVideoCard("COLMAP Tutorial", "-3TBbukYN00", "https://www.youtube.com/watch?v=-3TBbukYN00");
            ImGui::SameLine(0.0f, VIDEO_SPACING);
            renderVideoCard("LichtFeld Tutorial", "aX8MTlr9Ypc", "https://www.youtube.com/watch?v=aX8MTlr9Ypc");

            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (fonts_.section)
                ImGui::PushFont(fonts_.section);
            ImGui::TextColored(t.palette.text_dim, "WIKI & FAQ");
            if (fonts_.section)
                ImGui::PopFont();
            ImGui::Spacing();

            static constexpr const char* WIKI_URL = "https://github.com/MrNeRF/LichtFeld-Studio/wiki";
            ImGui::Indent(INDENT);
            ImGui::PushStyleColor(ImGuiCol_Text, lighten(t.palette.info, 0.3f));
            ImGui::TextWrapped("%s", WIKI_URL);
            ImGui::PopStyleColor();

            if (ImGui::IsItemHovered())
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            if (ImGui::IsItemClicked())
                openURL(WIKI_URL);

            ImGui::Unindent(INDENT);
        }
        ImGui::End();

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(3);
    }

    void MenuBar::renderAboutWindow() {
        if (!show_about_window_) {
            return;
        }

        constexpr ImGuiWindowFlags WINDOW_FLAGS = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize;
        ImGui::SetNextWindowSize(ImVec2(750, 0), ImGuiCond_Once);

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f, 20.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 10.0f));
        const auto& t = theme();
        ImGui::PushStyleColor(ImGuiCol_WindowBg, withAlpha(t.palette.surface, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, t.palette.surface);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, t.palette.surface_bright);
        ImGui::PushStyleColor(ImGuiCol_Border, withAlpha(t.palette.info, 0.3f));
        ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, t.palette.surface_bright);
        ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, lighten(t.palette.surface_bright, 0.15f));

        static constexpr const char* REPO_URL = "https://github.com/MrNeRF/LichtFeld-Studio";
        static constexpr const char* WEBSITE_URL = "https://lichtfeld.io";

        if (ImGui::Begin("About LichtFeld Studio", &show_about_window_, WINDOW_FLAGS)) {
            if (fonts_.heading)
                ImGui::PushFont(fonts_.heading);
            ImGui::TextColored(t.palette.info, "LICHTFELD STUDIO");
            if (fonts_.heading)
                ImGui::PopFont();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextWrapped(
                "A high-performance C++ and CUDA implementation of 3D Gaussian Splatting for "
                "real-time neural rendering, training, and visualization.");

            ImGui::Spacing();
            ImGui::Spacing();

            if (fonts_.section)
                ImGui::PushFont(fonts_.section);
            ImGui::TextColored(t.palette.text_dim, "BUILD INFORMATION");
            if (fonts_.section)
                ImGui::PopFont();
            ImGui::Spacing();

            constexpr ImGuiTableFlags TABLE_FLAGS = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;
            if (ImGui::BeginTable("build_info_table", 2, TABLE_FLAGS)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                const ImVec4 LABEL_COLOR = t.palette.text_dim;

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(LABEL_COLOR, "Version");
                ImGui::TableNextColumn();
                ImGui::TextWrapped("%s", GIT_TAGGED_VERSION);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                }
                if (ImGui::IsItemClicked()) {
                    ImGui::SetClipboardText(GIT_TAGGED_VERSION);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(LABEL_COLOR, "Commit");
                ImGui::TableNextColumn();
                ImGui::Text("%s", GIT_COMMIT_HASH_SHORT);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                }
                if (ImGui::IsItemClicked()) {
                    ImGui::SetClipboardText(GIT_COMMIT_HASH_SHORT);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(LABEL_COLOR, "Build Type");
                ImGui::TableNextColumn();
#ifdef DEBUG_BUILD
                ImGui::Text("Debug");
#else
                ImGui::Text("Release");
#endif

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(LABEL_COLOR, "Platform");
                ImGui::TableNextColumn();
#ifdef PLATFORM_WINDOWS
                ImGui::Text("Windows");
#elif defined(PLATFORM_LINUX)
                ImGui::Text("Linux");
#else
                ImGui::Text("Unknown");
#endif

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(LABEL_COLOR, "CUDA-GL Interop");
                ImGui::TableNextColumn();
#ifdef CUDA_GL_INTEROP_ENABLED
                ImGui::Text("Enabled");
#else
                ImGui::Text("Disabled");
#endif

                ImGui::EndTable();
            }

            ImGui::Spacing();
            ImGui::Spacing();

            if (fonts_.section)
                ImGui::PushFont(fonts_.section);
            ImGui::TextColored(t.palette.text_dim, "LINKS");
            if (fonts_.section)
                ImGui::PopFont();
            ImGui::Spacing();

            const ImVec4 LINK_COLOR = lighten(t.palette.info, 0.3f);

            ImGui::Text("Repository:");
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, LINK_COLOR);
            ImGui::Text("%s", REPO_URL);
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            }
            if (ImGui::IsItemClicked()) {
                openURL(REPO_URL);
            }

            ImGui::Text("Website:");
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, LINK_COLOR);
            ImGui::Text("%s", WEBSITE_URL);
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            }
            if (ImGui::IsItemClicked()) {
                openURL(WEBSITE_URL);
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored(t.palette.text_dim, "LichtFeld Studio Authors");
            ImGui::SameLine();
            ImGui::TextColored(darken(t.palette.text_dim, 0.15f), " | ");
            ImGui::SameLine();
            ImGui::TextColored(t.palette.text_dim, "Licensed under GPLv3");
        }
        ImGui::End();

        ImGui::PopStyleColor(7);
        ImGui::PopStyleVar(3);
    }

    void MenuBar::setOnNewProject(std::function<void()> callback) {
        on_new_project_ = std::move(callback);
    }

    void MenuBar::setOnImportDataset(std::function<void()> callback) {
        on_import_dataset_ = std::move(callback);
    }

    void MenuBar::setOnImportPLY(std::function<void()> callback) {
        on_import_ply_ = std::move(callback);
    }

    void MenuBar::setOnImportCheckpoint(std::function<void()> callback) {
        on_import_checkpoint_ = std::move(callback);
    }

    void MenuBar::setOnImportConfig(std::function<void()> callback) {
        on_import_config_ = std::move(callback);
    }

    void MenuBar::setOnExport(std::function<void()> callback) {
        on_export_ = std::move(callback);
    }

    void MenuBar::setOnExportConfig(std::function<void()> callback) {
        on_export_config_ = std::move(callback);
    }

    void MenuBar::setOnExit(std::function<void()> callback) {
        on_exit_ = std::move(callback);
    }

    void MenuBar::setCanClearCheck(std::function<bool()> check) {
        can_clear_ = std::move(check);
    }

    namespace {
        const char* getToolModeName(input::ToolMode mode) {
            switch (mode) {
            case input::ToolMode::GLOBAL: return "Global";
            case input::ToolMode::SELECTION: return "Selection Tool";
            case input::ToolMode::BRUSH: return "Brush Tool";
            case input::ToolMode::TRANSLATE: return "Translate";
            case input::ToolMode::ROTATE: return "Rotate";
            case input::ToolMode::SCALE: return "Scale";
            case input::ToolMode::ALIGN: return "Align Tool";
            case input::ToolMode::CROP_BOX: return "Crop Box";
            default: return "Unknown";
            }
        }
    } // namespace

    void MenuBar::renderBindingRow(const input::Action action, const input::ToolMode mode) {
        static constexpr ImVec4 COLOR_ACTION{0.9f, 0.9f, 0.9f, 1.0f};
        static constexpr ImVec4 COLOR_BINDING{0.4f, 0.7f, 1.0f, 1.0f};
        static constexpr ImVec4 COLOR_WAITING{1.0f, 0.8f, 0.2f, 1.0f};
        static constexpr ImVec4 COLOR_REBIND{0.2f, 0.4f, 0.6f, 1.0f};
        static constexpr ImVec4 COLOR_REBIND_HOVER{0.3f, 0.5f, 0.7f, 1.0f};
        static constexpr ImVec4 COLOR_REBIND_ACTIVE{0.4f, 0.6f, 0.8f, 1.0f};
        static constexpr ImVec4 COLOR_CANCEL{0.6f, 0.2f, 0.2f, 1.0f};
        static constexpr ImVec4 COLOR_CANCEL_HOVER{0.7f, 0.3f, 0.3f, 1.0f};
        static constexpr ImVec4 COLOR_CANCEL_ACTIVE{0.8f, 0.4f, 0.4f, 1.0f};

        const bool is_rebinding = rebinding_action_.has_value() &&
                                  *rebinding_action_ == action &&
                                  rebinding_mode_ == mode;

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(COLOR_ACTION, "%s", input::getActionName(action).c_str());

        ImGui::TableNextColumn();
        if (is_rebinding) {
            if (waiting_for_double_click_) {
                ImGui::TextColored(COLOR_WAITING, "Click again for double-click...");
            } else {
                ImGui::TextColored(COLOR_WAITING, "Press key or click mouse...");
            }
        } else {
            const std::string desc = input_bindings_->getTriggerDescription(action, mode);
            ImGui::TextColored(COLOR_BINDING, "%s", desc.c_str());
        }

        ImGui::TableNextColumn();
        const int unique_id = static_cast<int>(action) * 100 + static_cast<int>(mode);
        if (is_rebinding) {
            ImGui::PushStyleColor(ImGuiCol_Button, COLOR_CANCEL);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, COLOR_CANCEL_HOVER);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, COLOR_CANCEL_ACTIVE);
            char label[32];
            snprintf(label, sizeof(label), "Cancel##%d", unique_id);
            if (ImGui::Button(label, ImVec2(-1, 0))) {
                cancelCapture();
            }
            ImGui::PopStyleColor(3);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, COLOR_REBIND);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, COLOR_REBIND_HOVER);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, COLOR_REBIND_ACTIVE);
            char label[32];
            snprintf(label, sizeof(label), "Rebind##%d", unique_id);
            if (ImGui::Button(label, ImVec2(-1, 0))) {
                rebinding_action_ = action;
                rebinding_mode_ = mode;
            }
            ImGui::PopStyleColor(3);
        }
    }

    void MenuBar::captureKey(const int key, const int mods) {
        if (!rebinding_action_.has_value() || !input_bindings_) {
            return;
        }

        if (key == GLFW_KEY_ESCAPE) {
            cancelCapture();
            return;
        }

        // Ignore modifier-only keys
        if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT ||
            key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL ||
            key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT ||
            key == GLFW_KEY_LEFT_SUPER || key == GLFW_KEY_RIGHT_SUPER) {
            return;
        }

        const input::KeyTrigger trigger{key, mods, false};
        input_bindings_->setBinding(rebinding_mode_, *rebinding_action_, trigger);
        rebinding_action_.reset();
    }

    void MenuBar::captureMouseButton(const int button, const int mods) {
        if (!rebinding_action_.has_value() || !input_bindings_) {
            return;
        }

        if (waiting_for_double_click_) {
            // Second click - check if it's the same button
            if (button == pending_button_ && mods == pending_mods_) {
                // This is a double-click!
                const auto mouse_btn = static_cast<input::MouseButton>(button);
                const input::MouseButtonTrigger trigger{mouse_btn, mods, true};
                input_bindings_->setBinding(rebinding_mode_, *rebinding_action_, trigger);
                rebinding_action_.reset();
                waiting_for_double_click_ = false;
                pending_button_ = -1;
                return;
            }
            // Different button - commit the first click as single and start new wait
        }

        // First click - start waiting for potential second click
        waiting_for_double_click_ = true;
        pending_button_ = button;
        pending_mods_ = mods;
        first_click_time_ = std::chrono::steady_clock::now();
    }

    void MenuBar::updateCapture() {
        if (!waiting_for_double_click_ || !rebinding_action_.has_value() || !input_bindings_) {
            return;
        }

        // Check if we've waited long enough for a double-click
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - first_click_time_).count();

        if (elapsed >= DOUBLE_CLICK_WAIT_TIME) {
            // Timeout - commit as single-click (drag) binding
            const auto mouse_btn = static_cast<input::MouseButton>(pending_button_);
            const input::MouseDragTrigger trigger{mouse_btn, pending_mods_};
            input_bindings_->setBinding(rebinding_mode_, *rebinding_action_, trigger);
            rebinding_action_.reset();
            waiting_for_double_click_ = false;
            pending_button_ = -1;
        }
    }

    void MenuBar::cancelCapture() {
        rebinding_action_.reset();
        waiting_for_double_click_ = false;
        pending_button_ = -1;
    }

    void MenuBar::renderInputSettingsWindow() {
        if (!show_input_settings_) {
            cancelCapture();
            return;
        }

        // Check for double-click timeout each frame
        updateCapture();

        constexpr ImGuiWindowFlags WINDOW_FLAGS = ImGuiWindowFlags_NoDocking;
        ImGui::SetNextWindowSize(ImVec2(600, 600), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSizeConstraints(ImVec2(400, 300), ImVec2(FLT_MAX, FLT_MAX));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f, 20.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 10.0f));
        const auto& t = theme();
        ImGui::PushStyleColor(ImGuiCol_WindowBg, withAlpha(t.palette.surface, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, t.palette.surface);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, t.palette.surface_bright);
        ImGui::PushStyleColor(ImGuiCol_Border, withAlpha(t.palette.info, 0.3f));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, darken(t.palette.surface, 0.05f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, t.palette.surface_bright);
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, lighten(t.palette.surface_bright, 0.05f));
        ImGui::PushStyleColor(ImGuiCol_PopupBg, withAlpha(darken(t.palette.surface, 0.1f), 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Header, withAlpha(t.palette.info, 0.3f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, withAlpha(t.palette.info, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, withAlpha(t.palette.info, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, t.palette.surface_bright);
        ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, lighten(t.palette.surface_bright, 0.15f));

        if (ImGui::Begin("Input Settings", &show_input_settings_, WINDOW_FLAGS)) {
            if (fonts_.heading)
                ImGui::PushFont(fonts_.heading);
            ImGui::TextColored(t.palette.info, "INPUT SETTINGS");
            if (fonts_.heading)
                ImGui::PopFont();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (input_bindings_) {
                ImGui::Text("Active Profile:");
                ImGui::SameLine();

                const auto profiles = input_bindings_->getAvailableProfiles();
                const auto& current = input_bindings_->getCurrentProfileName();
                const bool is_rebinding = rebinding_action_.has_value();

                if (is_rebinding) {
                    ImGui::BeginDisabled();
                }

                if (ImGui::BeginCombo("##profile", current.c_str())) {
                    for (const auto& profile : profiles) {
                        const bool is_selected = (profile == current);
                        if (ImGui::Selectable(profile.c_str(), is_selected)) {
                            input_bindings_->loadProfile(profile);
                        }
                        if (is_selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }

                if (is_rebinding) {
                    ImGui::EndDisabled();
                }

                ImGui::Spacing();
                ImGui::Spacing();

                if (fonts_.section)
                    ImGui::PushFont(fonts_.section);
                ImGui::TextColored(t.palette.text_dim, "TOOL MODE");
                if (fonts_.section)
                    ImGui::PopFont();
                if (fonts_.small_font)
                    ImGui::PushFont(fonts_.small_font);
                ImGui::TextColored(t.palette.text_dim, "Select tool mode to view/edit bindings");
                if (fonts_.small_font)
                    ImGui::PopFont();
                ImGui::Spacing();

                // Tool mode selector
                static constexpr input::ToolMode TOOL_MODES[] = {
                    input::ToolMode::GLOBAL,
                    input::ToolMode::SELECTION,
                    input::ToolMode::BRUSH,
                    input::ToolMode::ALIGN,
                    input::ToolMode::CROP_BOX,
                };

                if (is_rebinding) {
                    ImGui::BeginDisabled();
                }

                if (ImGui::BeginCombo("##toolmode", getToolModeName(selected_tool_mode_))) {
                    for (const auto mode : TOOL_MODES) {
                        const bool is_selected = (mode == selected_tool_mode_);
                        if (ImGui::Selectable(getToolModeName(mode), is_selected)) {
                            selected_tool_mode_ = mode;
                        }
                        if (is_selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }

                if (is_rebinding) {
                    ImGui::EndDisabled();
                }

                ImGui::Spacing();
                ImGui::Spacing();

                if (fonts_.section)
                    ImGui::PushFont(fonts_.section);
                ImGui::TextColored(t.palette.text_dim, "CURRENT BINDINGS");
                if (fonts_.section)
                    ImGui::PopFont();
                if (fonts_.small_font)
                    ImGui::PushFont(fonts_.small_font);
                if (selected_tool_mode_ == input::ToolMode::GLOBAL) {
                    ImGui::TextColored(t.palette.text_dim, "Global bindings apply everywhere unless overridden");
                } else {
                    ImGui::TextColored(t.palette.text_dim, "Tool-specific bindings override global bindings");
                }
                if (fonts_.small_font)
                    ImGui::PopFont();
                ImGui::Spacing();

                constexpr ImGuiTableFlags TABLE_FLAGS = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY;
                constexpr float FOOTER_HEIGHT = 150.0f; // Space for buttons below table
                const float available_height = ImGui::GetContentRegionAvail().y - FOOTER_HEIGHT;
                const float table_height = std::max(200.0f, available_height);

                if (ImGui::BeginTable("bindings_table", 3, TABLE_FLAGS, ImVec2(0, table_height))) {
                    ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 180.0f);
                    ImGui::TableSetupColumn("Binding", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                    ImGui::TableHeadersRow();

                    const ImU32 SECTION_BG_COLOR = toU32(withAlpha(t.palette.info, 0.2f));
                    const ImVec4 SECTION_TEXT_COLOR = lighten(t.palette.info, 0.2f);

                    const auto renderSectionHeader = [SECTION_BG_COLOR, SECTION_TEXT_COLOR](const char* title) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, SECTION_BG_COLOR);
                        ImGui::TextColored(SECTION_TEXT_COLOR, "%s", title);
                        ImGui::TableNextColumn();
                        ImGui::TableNextColumn();
                    };

                    const auto mode = selected_tool_mode_;

                    // Navigation - always relevant for all tools
                    renderSectionHeader("NAVIGATION");
                    renderBindingRow(input::Action::CAMERA_ORBIT, mode);
                    renderBindingRow(input::Action::CAMERA_PAN, mode);
                    renderBindingRow(input::Action::CAMERA_ZOOM, mode);
                    renderBindingRow(input::Action::CAMERA_SET_PIVOT, mode);
                    renderBindingRow(input::Action::CAMERA_MOVE_FORWARD, mode);
                    renderBindingRow(input::Action::CAMERA_MOVE_BACKWARD, mode);
                    renderBindingRow(input::Action::CAMERA_MOVE_LEFT, mode);
                    renderBindingRow(input::Action::CAMERA_MOVE_RIGHT, mode);
                    renderBindingRow(input::Action::CAMERA_MOVE_UP, mode);
                    renderBindingRow(input::Action::CAMERA_MOVE_DOWN, mode);
                    renderBindingRow(input::Action::CAMERA_SPEED_UP, mode);
                    renderBindingRow(input::Action::CAMERA_SPEED_DOWN, mode);
                    renderBindingRow(input::Action::ZOOM_SPEED_UP, mode);
                    renderBindingRow(input::Action::ZOOM_SPEED_DOWN, mode);

                    if (mode == input::ToolMode::GLOBAL) {
                        // These only make sense globally
                        renderBindingRow(input::Action::CAMERA_RESET_HOME, mode);
                        renderBindingRow(input::Action::CAMERA_NEXT_VIEW, mode);
                        renderBindingRow(input::Action::CAMERA_PREV_VIEW, mode);
                    }

                    // Tool-specific actions
                    if (mode == input::ToolMode::GLOBAL ||
                        mode == input::ToolMode::SELECTION ||
                        mode == input::ToolMode::BRUSH) {
                        renderSectionHeader("SELECTION");
                        renderBindingRow(input::Action::SELECTION_REPLACE, mode);
                        renderBindingRow(input::Action::SELECTION_ADD, mode);
                        renderBindingRow(input::Action::SELECTION_REMOVE, mode);

                        if (mode == input::ToolMode::GLOBAL) {
                            renderBindingRow(input::Action::SELECT_MODE_CENTERS, mode);
                            renderBindingRow(input::Action::SELECT_MODE_RECTANGLE, mode);
                            renderBindingRow(input::Action::SELECT_MODE_POLYGON, mode);
                            renderBindingRow(input::Action::SELECT_MODE_LASSO, mode);
                            renderBindingRow(input::Action::SELECT_MODE_RINGS, mode);
                        }

                        if (mode == input::ToolMode::GLOBAL || mode == input::ToolMode::SELECTION) {
                            renderBindingRow(input::Action::TOGGLE_DEPTH_MODE, mode);
                            renderBindingRow(input::Action::DEPTH_ADJUST_FAR, mode);
                            renderBindingRow(input::Action::DEPTH_ADJUST_SIDE, mode);
                        }
                    }

                    if (mode == input::ToolMode::BRUSH) {
                        renderSectionHeader("BRUSH");
                        renderBindingRow(input::Action::CYCLE_BRUSH_MODE, mode);
                        renderBindingRow(input::Action::BRUSH_RESIZE, mode);
                    }

                    if (mode == input::ToolMode::CROP_BOX) {
                        renderSectionHeader("CROP BOX");
                        renderBindingRow(input::Action::APPLY_CROP_BOX, mode);
                    }

                    // Editing - available in all modes
                    renderSectionHeader("EDITING");
                    // Delete action depends on mode: GLOBAL/transform = delete node, others = delete Gaussians
                    if (mode == input::ToolMode::GLOBAL ||
                        mode == input::ToolMode::TRANSLATE ||
                        mode == input::ToolMode::ROTATE ||
                        mode == input::ToolMode::SCALE) {
                        renderBindingRow(input::Action::DELETE_NODE, mode);
                    } else {
                        renderBindingRow(input::Action::DELETE_SELECTED, mode);
                    }
                    renderBindingRow(input::Action::UNDO, mode);
                    renderBindingRow(input::Action::REDO, mode);
                    renderBindingRow(input::Action::COPY_SELECTION, mode);
                    renderBindingRow(input::Action::PASTE_SELECTION, mode);
                    renderBindingRow(input::Action::INVERT_SELECTION, mode);
                    renderBindingRow(input::Action::DESELECT_ALL, mode);

                    if (mode == input::ToolMode::GLOBAL) {
                        renderSectionHeader("VIEW");
                        renderBindingRow(input::Action::TOGGLE_SPLIT_VIEW, mode);
                        renderBindingRow(input::Action::TOGGLE_GT_COMPARISON, mode);
                        renderBindingRow(input::Action::CYCLE_PLY, mode);
                        renderBindingRow(input::Action::CYCLE_SELECTION_VIS, mode);
                    }

                    ImGui::EndTable();
                }
            } else {
                ImGui::TextColored(lighten(t.palette.error, 0.2f), "Input bindings not available");
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (input_bindings_) {
                const ImVec4 BTN_SAVE = darken(t.palette.success, 0.3f);
                const ImVec4 BTN_SAVE_HOVER = darken(t.palette.success, 0.15f);
                const ImVec4 BTN_SAVE_ACTIVE = darken(t.palette.success, 0.2f);
                const ImVec4 BTN_RESET = darken(t.palette.error, 0.3f);
                const ImVec4 BTN_RESET_HOVER = darken(t.palette.error, 0.15f);
                const ImVec4 BTN_RESET_ACTIVE = darken(t.palette.error, 0.2f);

                ImGui::PushStyleColor(ImGuiCol_Button, BTN_SAVE);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, BTN_SAVE_HOVER);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, BTN_SAVE_ACTIVE);
                if (ImGui::Button("Save Current Profile")) {
                    input_bindings_->saveProfile(input_bindings_->getCurrentProfileName());
                }
                ImGui::PopStyleColor(3);

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Button, BTN_RESET);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, BTN_RESET_HOVER);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, BTN_RESET_ACTIVE);
                if (ImGui::Button("Reset to Default")) {
                    const auto config_dir = input::InputBindings::getConfigDir();
                    const auto saved_path = config_dir / "Default.json";
                    if (std::filesystem::exists(saved_path)) {
                        std::filesystem::remove(saved_path);
                    }
                    input_bindings_->loadProfile("Default");
                    // Save hardcoded defaults to disk
                    input_bindings_->saveProfile("Default");
                }
                ImGui::PopStyleColor(3);

                ImGui::Spacing();

                const ImVec4 BTN_IO = darken(t.palette.secondary, 0.2f);
                const ImVec4 BTN_IO_HOVER = darken(t.palette.secondary, 0.05f);
                const ImVec4 BTN_IO_ACTIVE = darken(t.palette.secondary, 0.1f);

                ImGui::PushStyleColor(ImGuiCol_Button, BTN_IO);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, BTN_IO_HOVER);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, BTN_IO_ACTIVE);
                if (ImGui::Button("Export...")) {
                    const auto path = SaveJsonFileDialog("input_bindings");
                    if (!path.empty()) {
                        input_bindings_->saveProfileToFile(path);
                    }
                }
                ImGui::PopStyleColor(3);

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Button, BTN_IO);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, BTN_IO_HOVER);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, BTN_IO_ACTIVE);
                if (ImGui::Button("Import...")) {
                    if (const auto path = OpenJsonFileDialog(); !path.empty() && std::filesystem::exists(path)) {
                        input_bindings_->loadProfileFromFile(path);
                    }
                }
                ImGui::PopStyleColor(3);

                ImGui::Spacing();
            }

            ImGui::TextColored(t.palette.text_dim, "Save to persist custom bindings");
            ImGui::TextColored(t.palette.text_dim, "Tip: Double-click to bind double-click action");
        }
        ImGui::End();

        ImGui::PopStyleColor(14);
        ImGui::PopStyleVar(3);
    }

    void MenuBar::renderDebugWindow() {
        if (!show_debug_window_)
            return;

        // Fixed dimensions to prevent DPI-related resize feedback loop
        constexpr float WINDOW_WIDTH = 450.0f;
        constexpr float WINDOW_HEIGHT = 400.0f;
        constexpr ImGuiWindowFlags WINDOW_FLAGS = ImGuiWindowFlags_NoDocking |
                                                  ImGuiWindowFlags_NoResize |
                                                  ImGuiWindowFlags_NoScrollbar;
        const auto& t = theme();

        ImGui::SetNextWindowSize({WINDOW_WIDTH, WINDOW_HEIGHT}, ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f, 20.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 10.0f));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, withAlpha(t.palette.surface, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, t.palette.surface);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, t.palette.surface_bright);
        ImGui::PushStyleColor(ImGuiCol_Border, withAlpha(t.palette.info, 0.3f));

        if (ImGui::Begin("Debug Info", &show_debug_window_, WINDOW_FLAGS)) {
            if (fonts_.heading)
                ImGui::PushFont(fonts_.heading);
            ImGui::TextColored(t.palette.info, "DEBUG INFORMATION");
            if (fonts_.heading)
                ImGui::PopFont();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // GPU Memory Section
            if (fonts_.section)
                ImGui::PushFont(fonts_.section);
            ImGui::TextColored(t.palette.text_dim, "GPU MEMORY");
            if (fonts_.section)
                ImGui::PopFont();
            ImGui::Spacing();

            const auto mem = lfs::core::debug::get_memory_snapshot();
            ImGui::Text("Used: %.2f GB / %.2f GB (%.1f%%)",
                        mem.gpu_used_bytes / 1e9,
                        mem.gpu_total_bytes / 1e9,
                        mem.gpu_usage_percent());
            ImGui::Text("Free: %.2f GB", mem.gpu_free_bytes / 1e9);

            // Progress bar for memory usage
            const float usage_ratio = mem.gpu_total_bytes > 0
                                          ? static_cast<float>(mem.gpu_used_bytes) / static_cast<float>(mem.gpu_total_bytes)
                                          : 0.0f;
            const ImVec4 bar_color = usage_ratio > 0.9f   ? t.palette.error
                                     : usage_ratio > 0.7f ? t.palette.warning
                                                          : t.palette.success;
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, bar_color);
            ImGui::ProgressBar(usage_ratio, ImVec2(-1, 0));
            ImGui::PopStyleColor();

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Tensor Operation Tracing Section
            if (fonts_.section)
                ImGui::PushFont(fonts_.section);
            ImGui::TextColored(t.palette.text_dim, "TENSOR OP TRACING");
            if (fonts_.section)
                ImGui::PopFont();
            ImGui::Spacing();

            auto& tracer = lfs::core::debug::TensorOpTracer::instance();
            bool tracing_enabled = tracer.is_enabled();

            if (ImGui::Checkbox("Enable Operation Tracing", &tracing_enabled)) {
                tracer.set_enabled(tracing_enabled);
            }
            ImGui::SameLine();
            ImGui::TextColored(t.palette.text_dim, "(Performance impact)");

            if (tracing_enabled) {
                const auto& history = tracer.get_history();
                ImGui::Text("Recorded operations: %zu", history.size());

                if (ImGui::Button("Clear History")) {
                    tracer.clear_history();
                }
                ImGui::SameLine();
                if (ImGui::Button("Print to Log")) {
                    tracer.print_history(50);
                }

                // Show last few operations
                if (!history.empty()) {
                    ImGui::Spacing();
                    if (fonts_.small_font)
                        ImGui::PushFont(fonts_.small_font);
                    ImGui::TextColored(t.palette.text_dim, "Recent operations:");
                    const size_t show_count = std::min(size_t{5}, history.size());
                    for (size_t i = history.size() - show_count; i < history.size(); ++i) {
                        const auto& op = history[i];
                        std::string name_tag = op.tensor_name.empty() ? "" : std::format(" [{}]", op.tensor_name);
                        std::string location = op.file.empty() ? "" : std::format(" @ {}:{}", op.file, op.line);
                        ImGui::Text("  %s(%s) -> %s [%.2fms]%s%s",
                                    op.op_name.c_str(),
                                    op.input_shapes.c_str(),
                                    op.output_shape.c_str(),
                                    op.duration_ms,
                                    name_tag.c_str(),
                                    location.c_str());
                    }
                    if (fonts_.small_font)
                        ImGui::PopFont();
                }
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Build Info Section
            if (fonts_.section)
                ImGui::PushFont(fonts_.section);
            ImGui::TextColored(t.palette.text_dim, "BUILD FLAGS");
            if (fonts_.section)
                ImGui::PopFont();
            ImGui::Spacing();

#ifdef TENSOR_VALIDATION_ENABLED
            ImGui::TextColored(t.palette.success, "[x] Tensor Validation");
#else
            ImGui::TextColored(t.palette.text_dim, "[ ] Tensor Validation");
#endif

#ifdef CUDA_DEBUG_SYNC
            ImGui::TextColored(t.palette.success, "[x] CUDA Debug Sync");
#else
            ImGui::TextColored(t.palette.text_dim, "[ ] CUDA Debug Sync");
#endif

#ifdef TENSOR_OP_TRACING
            ImGui::TextColored(t.palette.success, "[x] Tensor Op Tracing (compile-time)");
#else
            ImGui::TextColored(t.palette.text_dim, "[ ] Tensor Op Tracing (compile-time)");
#endif

#ifdef DEBUG_BUILD
            ImGui::TextColored(t.palette.success, "[x] Debug Build");
#else
            ImGui::TextColored(t.palette.text_dim, "[ ] Debug Build");
#endif

            ImGui::Spacing();
            if (fonts_.small_font)
                ImGui::PushFont(fonts_.small_font);
            ImGui::TextColored(t.palette.text_dim, "Rebuild with -DENABLE_TENSOR_VALIDATION=ON");
            ImGui::TextColored(t.palette.text_dim, "or -DENABLE_CUDA_DEBUG_SYNC=ON to enable");
            if (fonts_.small_font)
                ImGui::PopFont();
        }
        ImGui::End();

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(3);
    }

} // namespace lfs::vis::gui
