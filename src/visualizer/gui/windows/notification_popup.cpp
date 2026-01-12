/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "notification_popup.hpp"
#include "core/events.hpp"
#include "core/path_utils.hpp"
#include "gui/dpi_scale.hpp"
#include "gui/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_widgets.hpp"
#include "theme/theme.hpp"
#include <cmath>
#include <format>
#include <imgui.h>

namespace lfs::vis::gui {

    namespace {
        // Base dimensions (scaled by DPI factor at runtime)
        constexpr float BASE_BUTTON_WIDTH = 100.0f;
        constexpr float TEXT_WRAP_WIDTH = 30.0f; // Multiplier for font size, not pixels
        constexpr float POPUP_ALPHA = 0.98f;
        constexpr float BORDER_SIZE = 2.0f;
        constexpr ImVec2 BASE_WINDOW_PADDING = {20.0f, 16.0f};
        constexpr ImGuiWindowFlags POPUP_FLAGS = ImGuiWindowFlags_AlwaysAutoResize |
                                                 ImGuiWindowFlags_NoCollapse |
                                                 ImGuiWindowFlags_NoDocking |
                                                 ImGuiWindowFlags_NoSavedSettings;
    } // namespace

    using namespace lfs::core::events;

    NotificationPopup::NotificationPopup() {
        setupEventHandlers();
    }

    std::string NotificationPopup::formatDuration(const float seconds) {
        const float clamped = std::max(0.0f, seconds);
        const int total = static_cast<int>(std::round(clamped));
        const int hours = total / 3600;
        const int minutes = (total % 3600) / 60;
        const int secs = total % 60;

        if (hours > 0) {
            return std::format("{}h {}m {}s", hours, minutes, secs);
        }
        if (minutes > 0) {
            return std::format("{}m {}s", minutes, secs);
        }
        if (clamped >= 1.0f) {
            return std::format("{}s", secs);
        }
        return std::format("{:.1f}s", clamped);
    }

    void NotificationPopup::setupEventHandlers() {
        state::DatasetLoadCompleted::when([this](const auto& e) {
            if (!e.success && e.error.has_value()) {
                show(Type::FAILURE, "Failed to Load Dataset", *e.error);
            }
        });

        state::CudaVersionUnsupported::when([this](const auto& e) {
            show(Type::WARNING, "Unsupported CUDA Driver",
                 std::format("Your CUDA driver version ({}.{}) is not supported.\n\n"
                             "LichtFeld Studio requires CUDA {}.{} or later\n"
                             "(NVIDIA driver 570+).\n\n"
                             "Please update your NVIDIA driver for full functionality.",
                             e.major, e.minor, e.min_major, e.min_minor));
        });

        state::ConfigLoadFailed::when([this](const auto& e) {
            show(Type::FAILURE, "Invalid Config File",
                 std::format("Could not load '{}':\n\n{}", lfs::core::path_to_utf8(e.path.filename()), e.error));
        });

        state::FileDropFailed::when([this](const auto& e) {
            constexpr size_t MAX_DISPLAY = 5;
            namespace Notif = lichtfeld::Strings::Notification;

            const size_t count = e.files.size();
            const size_t display_count = std::min(count, MAX_DISPLAY);

            std::string file_list;
            file_list.reserve(display_count * 64);

            for (size_t i = 0; i < display_count; ++i) {
                const std::filesystem::path p(e.files[i]);
                const bool is_dir = std::filesystem::is_directory(p);
                file_list += std::format("  - {} ({})\n", lfs::core::path_to_utf8(p.filename()),
                                         is_dir ? LOC(Notif::DIRECTORY) : LOC(Notif::FILE));
            }
            if (count > MAX_DISPLAY) {
                file_list += std::format("  {} {}\n", LOC(Notif::AND_MORE), count - MAX_DISPLAY);
            }

            const bool single_dir = count == 1 && std::filesystem::is_directory(e.files[0]);
            const char* item_type = count == 1 ? (single_dir ? LOC(Notif::DIRECTORY) : LOC(Notif::FILE))
                                               : LOC(Notif::ITEMS);

            show(Type::FAILURE, LOC(Notif::CANNOT_OPEN),
                 std::format("{} {}:\n\n{}\n{}", LOC(Notif::DROPPED_NOT_RECOGNIZED), item_type, file_list, e.error));
        });

        state::TrainingCompleted::when([this](const auto& e) {
            if (e.success) {
                const auto message = std::format(
                    "Training completed successfully!\n\n"
                    "Iterations: {}\n"
                    "Final loss: {:.6f}\n"
                    "Duration: {}",
                    e.iteration, e.final_loss, formatDuration(e.elapsed_seconds));

                show(Type::INFO, "Training Complete", message,
                     []() { cmd::SwitchToLatestCheckpoint{}.emit(); });
            } else {
                // Format error message for better clarity
                std::string error_msg = e.error.value_or("Unknown error occurred during training.");

                // Check if this is an OOM error and format it clearly
                if (error_msg.find("OUT_OF_MEMORY") != std::string::npos) {
                    error_msg = "Out of GPU memory!\n\n"
                                "The scene is too large for available GPU memory.\n\n"
                                "Suggestions:\n"
                                "  - Reduce image resolution (--resize-factor)\n"
                                "  - Enable tile mode (--tile-mode 2 or 4)\n"
                                "  - Reduce max Gaussians (--max-cap)";
                }

                show(Type::FAILURE, "Training Failed", error_msg);
            }
        });
    }

    void NotificationPopup::show(const Type type, const std::string& title,
                                 const std::string& message, Callback on_close) {
        pending_.push_back({type, title, message, std::move(on_close)});
    }

    void NotificationPopup::render(const ImVec2& viewport_pos, const ImVec2& viewport_size) {
        if (!popup_open_ && !pending_.empty()) {
            current_ = std::move(pending_.front());
            pending_.pop_front();
            popup_open_ = true;
            ImGui::OpenPopup(current_.title.c_str());
        }

        if (!popup_open_) {
            return;
        }

        const auto& t = theme();
        const float scale = getDpiScale();

        ImVec4 accent;
        widgets::ButtonStyle btn_style;
        const char* type_label;
        switch (current_.type) {
        case Type::FAILURE:
            accent = t.palette.error;
            btn_style = widgets::ButtonStyle::Error;
            type_label = "Error";
            break;
        case Type::WARNING:
            accent = t.palette.warning;
            btn_style = widgets::ButtonStyle::Warning;
            type_label = "Warning";
            break;
        default:
            accent = t.palette.success;
            btn_style = widgets::ButtonStyle::Success;
            type_label = "Success";
            break;
        }

        const ImVec4 popup_bg = {t.palette.surface.x, t.palette.surface.y,
                                 t.palette.surface.z, POPUP_ALPHA};
        const ImVec4 title_bg = darken(t.palette.surface, 0.1f);
        const ImVec4 title_bg_active = darken(t.palette.surface, 0.05f);

        const ImVec2 center = (viewport_size.x > 0 && viewport_size.y > 0)
                                  ? ImVec2{viewport_pos.x + viewport_size.x * 0.5f, viewport_pos.y + viewport_size.y * 0.5f}
                                  : ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, {0.5f, 0.5f});

        ImGui::PushStyleColor(ImGuiCol_WindowBg, popup_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, title_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, title_bg_active);
        ImGui::PushStyleColor(ImGuiCol_Border, accent);
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, BORDER_SIZE);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(BASE_WINDOW_PADDING.x * scale, BASE_WINDOW_PADDING.y * scale));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.popup_rounding);

        if (ImGui::BeginPopupModal(current_.title.c_str(), nullptr, POPUP_FLAGS)) {
            ImGui::TextColored(accent, "%s", type_label);
            ImGui::SameLine();
            ImGui::TextColored(t.palette.text_dim, "|");
            ImGui::SameLine();
            ImGui::TextUnformatted(current_.title.c_str());

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::PushTextWrapPos(ImGui::GetFontSize() * TEXT_WRAP_WIDTH);
            ImGui::TextUnformatted(current_.message.c_str());
            ImGui::PopTextWrapPos();

            ImGui::Spacing();
            ImGui::Spacing();

            const float button_width = BASE_BUTTON_WIDTH * scale;
            const float avail = ImGui::GetContentRegionAvail().x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - button_width) * 0.5f);

            if (widgets::ColoredButton("OK", btn_style, {button_width, 0}) ||
                ImGui::IsKeyPressed(ImGuiKey_Enter)) {
                popup_open_ = false;
                if (current_.on_close) {
                    current_.on_close();
                }
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(5);
    }

} // namespace lfs::vis::gui
