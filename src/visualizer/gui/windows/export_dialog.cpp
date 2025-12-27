/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/windows/export_dialog.hpp"
#include "gui/dpi_scale.hpp"
#include "gui/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include <imgui.h>

namespace lfs::vis::gui {

    using namespace lichtfeld::Strings;

    namespace {
        // Base sizes (will be scaled by DPI factor)
        constexpr float BASE_WINDOW_WIDTH = 380.0f;
        constexpr float FRAME_DARKEN = 0.1f;
        constexpr float BUTTON_LIGHTEN = 0.1f;
        constexpr ImVec2 BASE_WINDOW_PADDING = {16.0f, 12.0f};
        constexpr ImVec2 BASE_ITEM_SPACING = {8.0f, 6.0f};
        constexpr ImVec2 BASE_EXPORT_BUTTON_SIZE = {130.0f, 28.0f};
        constexpr ImVec2 BASE_CANCEL_BUTTON_SIZE = {80.0f, 28.0f};

        void pushInputStyle(const Theme& t) {
            ImGui::PushStyleColor(ImGuiCol_CheckMark, t.palette.primary);
            ImGui::PushStyleColor(ImGuiCol_FrameBg, darken(t.palette.surface, FRAME_DARKEN));
            ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, t.palette.surface_bright);
            ImGui::PushStyleColor(ImGuiCol_FrameBgActive, t.palette.primary_dim);
        }

        void popInputStyle() {
            ImGui::PopStyleColor(4);
        }

        void pushButtonStyle(const Theme& t) {
            ImGui::PushStyleColor(ImGuiCol_Button, t.button_normal());
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.button_hovered());
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, t.button_active());
        }

        void popButtonStyle() {
            ImGui::PopStyleColor(3);
        }
    } // namespace

    using ExportFormat = lfs::core::ExportFormat;

    void ExportDialog::render(bool* p_open, SceneManager* scene_manager) {
        if (!p_open || !*p_open)
            return;

        const auto& t = theme();
        const float scale = getDpiScale();

        constexpr ImGuiWindowFlags WINDOW_FLAGS =
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking;

        ImGui::SetNextWindowSize(ImVec2(BASE_WINDOW_WIDTH * scale, 0), ImGuiCond_FirstUseEver);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, t.palette.surface);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, t.palette.surface);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, t.palette.surface_bright);
        ImGui::PushStyleColor(ImGuiCol_Border, t.palette.border);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(BASE_WINDOW_PADDING.x * scale, BASE_WINDOW_PADDING.y * scale));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(BASE_ITEM_SPACING.x * scale, BASE_ITEM_SPACING.y * scale));

        if (!ImGui::Begin(LOC(Export::TITLE), p_open, WINDOW_FLAGS)) {
            ImGui::End();
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor(4);
            return;
        }

        // Collect SPLAT nodes
        std::vector<const SceneNode*> splat_nodes;
        if (scene_manager) {
            for (const auto* node : scene_manager->getScene().getNodes()) {
                if (node->type == NodeType::SPLAT && node->model) {
                    splat_nodes.push_back(node);
                }
            }
        }

        // Compute max SH degree from selected nodes
        const auto updateMaxSHDegree = [&]() {
            max_sh_degree_ = 0;
            for (const auto* node : splat_nodes) {
                if (selected_nodes_.contains(node->name) && node->model) {
                    max_sh_degree_ = std::max(max_sh_degree_, node->model->get_max_sh_degree());
                }
            }
            export_sh_degree_ = std::min(export_sh_degree_, max_sh_degree_);
        };

        // Initialize selection on first open
        if (!initialized_ && !splat_nodes.empty()) {
            selected_nodes_.clear();
            for (const auto* node : splat_nodes) {
                selected_nodes_.insert(node->name);
            }
            updateMaxSHDegree();
            export_sh_degree_ = max_sh_degree_;
            initialized_ = true;
        }

        // Format selection
        ImGui::TextColored(t.palette.text_dim, "%s", LOC(lichtfeld::Strings::ExportDialog::FORMAT));
        ImGui::Spacing();

        pushInputStyle(t);
        int format_idx = static_cast<int>(selected_format_);
        ImGui::RadioButton(LOC(Export::FORMAT_PLY_STANDARD), &format_idx, static_cast<int>(ExportFormat::PLY));
        ImGui::RadioButton(LOC(Export::FORMAT_SOG_SUPERSPLAT), &format_idx, static_cast<int>(ExportFormat::SOG));
        ImGui::RadioButton(LOC(Export::FORMAT_SPZ_NIANTIC), &format_idx, static_cast<int>(ExportFormat::SPZ));
        ImGui::RadioButton(LOC(Export::FORMAT_HTML_VIEWER), &format_idx, static_cast<int>(ExportFormat::HTML_VIEWER));
        selected_format_ = static_cast<ExportFormat>(format_idx);
        popInputStyle();

        ImGui::Spacing();
        ImGui::Spacing();

        // Model selection
        ImGui::TextColored(t.palette.text_dim, "%s", LOC(lichtfeld::Strings::ExportDialog::MODELS));
        ImGui::Spacing();

        if (splat_nodes.empty()) {
            ImGui::TextColored(t.palette.text_dim, "%s", LOC(lichtfeld::Strings::ExportDialog::NO_MODELS));
        } else {
            pushButtonStyle(t);
            if (ImGui::SmallButton(LOC(Export::ALL))) {
                for (const auto* node : splat_nodes) {
                    selected_nodes_.insert(node->name);
                }
                updateMaxSHDegree();
            }
            ImGui::SameLine();
            if (ImGui::SmallButton(LOC(Export::NONE))) {
                selected_nodes_.clear();
                updateMaxSHDegree();
            }
            popButtonStyle();

            ImGui::Spacing();

            pushInputStyle(t);
            for (const auto* node : splat_nodes) {
                bool selected = selected_nodes_.contains(node->name);
                if (ImGui::Checkbox(node->name.c_str(), &selected)) {
                    if (selected) {
                        selected_nodes_.insert(node->name);
                    } else {
                        selected_nodes_.erase(node->name);
                    }
                    updateMaxSHDegree();
                }
                ImGui::SameLine();
                ImGui::TextColored(t.palette.text_dim, "(%zu)", node->gaussian_count);
            }
            popInputStyle();
        }

        ImGui::Spacing();
        ImGui::Spacing();

        // SH Degree selection
        ImGui::TextColored(t.palette.text_dim, "%s", LOC(lichtfeld::Strings::ExportDialog::SH_DEGREE));
        ImGui::Spacing();

        pushInputStyle(t);
        if (max_sh_degree_ > 0) {
            ImGui::SliderInt("##sh_degree", &export_sh_degree_, 0, max_sh_degree_, "Degree %d");
        } else {
            ImGui::BeginDisabled();
            ImGui::SliderInt("##sh_degree", &export_sh_degree_, 0, 0, "Degree 0");
            ImGui::EndDisabled();
        }
        popInputStyle();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Export button
        const bool can_export = !selected_nodes_.empty();

        if (!can_export) {
            ImGui::TextColored(t.palette.error, "%s", LOC(Export::SELECT_AT_LEAST_ONE));
            ImGui::Spacing();
        }

        const ImVec4 btn_color = can_export ? t.palette.primary : t.palette.surface_bright;
        const ImVec4 btn_hover = can_export ? lighten(t.palette.primary, BUTTON_LIGHTEN) : t.palette.surface_bright;
        const ImVec4 btn_active = can_export ? darken(t.palette.primary, BUTTON_LIGHTEN) : t.palette.surface_bright;
        const ImVec4 btn_text = can_export ? ImVec4(1, 1, 1, 1) : t.palette.text_dim;

        ImGui::PushStyleColor(ImGuiCol_Button, btn_color);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, btn_hover);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, btn_active);
        ImGui::PushStyleColor(ImGuiCol_Text, btn_text);

        ImGui::BeginDisabled(!can_export);
        const char* label = selected_nodes_.size() > 1 ? LOC(lichtfeld::Strings::ExportDialog::EXPORT_MERGED) : LOC(Export::EXPORT);
        if (ImGui::Button(label, ImVec2(BASE_EXPORT_BUTTON_SIZE.x * scale, BASE_EXPORT_BUTTON_SIZE.y * scale))) {
            if (on_browse_) {
                const std::string default_name = selected_nodes_.size() == 1
                                                     ? *selected_nodes_.begin()
                                                     : "merged";
                std::vector<std::string> nodes(selected_nodes_.begin(), selected_nodes_.end());
                on_browse_(selected_format_, default_name, nodes, export_sh_degree_);
            }
            *p_open = false;
            initialized_ = false;
        }
        ImGui::EndDisabled();
        ImGui::PopStyleColor(4);

        ImGui::SameLine();

        pushButtonStyle(t);
        if (ImGui::Button(LOC(Export::CANCEL), ImVec2(BASE_CANCEL_BUTTON_SIZE.x * scale, BASE_CANCEL_BUTTON_SIZE.y * scale))) {
            *p_open = false;
            initialized_ = false;
        }
        popButtonStyle();

        ImGui::End();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(4);
    }

} // namespace lfs::vis::gui
