/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "notification_popup.hpp"
#include "core/events.hpp"
#include "gui/ui_widgets.hpp"
#include "theme/theme.hpp"
#include <cmath>
#include <format>
#include <imgui.h>

namespace lfs::vis::gui {

    namespace {
        constexpr float BUTTON_WIDTH = 100.0f;
        constexpr float TEXT_WRAP_WIDTH = 30.0f;
        constexpr float POPUP_ALPHA = 0.98f;
        constexpr float BORDER_SIZE = 2.0f;
        constexpr ImVec2 WINDOW_PADDING = {20.0f, 16.0f};
        constexpr ImGuiWindowFlags POPUP_FLAGS = ImGuiWindowFlags_AlwaysAutoResize |
                                                 ImGuiWindowFlags_NoCollapse |
                                                 ImGuiWindowFlags_NoDocking;
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

        state::ConfigLoadFailed::when([this](const auto& e) {
            show(Type::FAILURE, "Invalid Config File",
                 std::format("Could not load '{}':\n\n{}", e.path.filename().string(), e.error));
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
                show(Type::FAILURE, "Training Failed",
                     e.error.value_or("Unknown error occurred during training."));
            }
        });
    }

    void NotificationPopup::show(const Type type, const std::string& title,
                                 const std::string& message, Callback on_close) {
        pending_.push_back({type, title, message, std::move(on_close)});
    }

    void NotificationPopup::render() {
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

        const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, {0.5f, 0.5f});

        ImGui::PushStyleColor(ImGuiCol_WindowBg, popup_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, title_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, title_bg_active);
        ImGui::PushStyleColor(ImGuiCol_Border, accent);
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.text);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, BORDER_SIZE);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, WINDOW_PADDING);
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

            const float avail = ImGui::GetContentRegionAvail().x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - BUTTON_WIDTH) * 0.5f);

            if (widgets::ColoredButton("OK", btn_style, {BUTTON_WIDTH, 0}) ||
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
