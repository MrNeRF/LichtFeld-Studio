/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/mesh2splat_panel.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "core/mesh2splat.hpp"
#include "core/scene.hpp"
#include "gui/async_task_manager.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_widgets.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "visualizer_impl.hpp"

#include <cassert>
#include <glm/glm.hpp>
#include <imgui.h>

using namespace lichtfeld::Strings;

namespace lfs::vis::gui::panels {

    Mesh2SplatPanel::Mesh2SplatPanel(VisualizerImpl* viewer)
        : viewer_(viewer) {
        assert(viewer_);
    }

    bool Mesh2SplatPanel::poll(const PanelDrawContext& ctx) {
        (void)ctx;
        return true;
    }

    int Mesh2SplatPanel::computeResolutionTarget() const {
        assert(resolution_index_ >= 0 && resolution_index_ < kResolutionOptionCount);
        const int max_res = kResolutionOptions[resolution_index_];
        constexpr int kMinRes = lfs::core::Mesh2SplatOptions::kMinResolution;
        return static_cast<int>(kMinRes + quality_ * (max_res - kMinRes));
    }

    void Mesh2SplatPanel::triggerConversion() {
        auto& async = viewer_->getGuiManager()->asyncTasks();
        if (async.isMesh2SplatActive() || selected_mesh_name_.empty())
            return;

        auto* scene_manager = viewer_->getSceneManager();
        if (!scene_manager)
            return;

        const auto& scene = scene_manager->getScene();
        const auto* node = scene.getNode(selected_mesh_name_);
        if (!node || !node->mesh)
            return;

        lfs::core::Mesh2SplatOptions options;
        options.resolution_target = computeResolutionTarget();
        options.sigma = gaussian_scale_;

        glm::vec3 cam_pos = viewer_->getViewport().getTranslation();
        float cam_len = glm::length(cam_pos);
        if (cam_len > 1e-6f)
            options.light_dir = cam_pos / cam_len;

        async.startMesh2Splat(node->mesh, selected_mesh_name_, options);
        has_initial_conversion_ = true;
    }

    void Mesh2SplatPanel::draw(const PanelDrawContext& /*ctx*/) {
        drawMeshSelector();
        ImGui::Spacing();
        drawParameters();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        drawConvertButton();
        drawProgress();
        drawError();
    }

    void Mesh2SplatPanel::drawMeshSelector() {
        ImGui::TextUnformatted(LOC(Mesh2Splat::SOURCE_MESH));

        auto* scene_manager = viewer_->getSceneManager();
        if (!scene_manager) {
            ImGui::TextDisabled("%s", LOC(Mesh2Splat::NO_MESHES));
            return;
        }

        const auto& scene = scene_manager->getScene();
        auto nodes = scene.getNodes();

        std::vector<const lfs::core::SceneNode*> mesh_nodes;
        for (const auto* node : nodes) {
            if (node->type == lfs::core::NodeType::MESH && node->mesh)
                mesh_nodes.push_back(node);
        }

        if (mesh_nodes.empty()) {
            ImGui::TextDisabled("%s", LOC(Mesh2Splat::NO_MESHES));
            selected_mesh_name_.clear();
            return;
        }

        bool selection_valid = false;
        int current_idx = 0;
        for (int i = 0; i < static_cast<int>(mesh_nodes.size()); ++i) {
            if (mesh_nodes[i]->name == selected_mesh_name_) {
                current_idx = i;
                selection_valid = true;
                break;
            }
        }
        if (!selection_valid) {
            selected_mesh_name_ = mesh_nodes[0]->name;
            current_idx = 0;
        }

        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##mesh_selector", selected_mesh_name_.c_str())) {
            for (int i = 0; i < static_cast<int>(mesh_nodes.size()); ++i) {
                const bool is_selected = (i == current_idx);
                if (ImGui::Selectable(mesh_nodes[i]->name.c_str(), is_selected))
                    selected_mesh_name_ = mesh_nodes[i]->name;
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }

    void Mesh2SplatPanel::drawParameters() {
        bool reconvert = false;

        ImGui::TextUnformatted(LOC(Mesh2Splat::GAUSSIAN_SCALE));
        widgets::HelpMarker(LOC(Mesh2Splat::TOOLTIP_GAUSSIAN_SCALE));
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##gaussian_scale", &gaussian_scale_, 0.1f, 2.0f, "%.2f");
        reconvert |= ImGui::IsItemDeactivatedAfterEdit();

        ImGui::Spacing();
        ImGui::SeparatorText(LOC(Mesh2Splat::SAMPLING_DENSITY_HEADER));

        ImGui::TextUnformatted(LOC(Mesh2Splat::SAMPLING_DENSITY));
        widgets::HelpMarker(LOC(Mesh2Splat::TOOLTIP_QUALITY));
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##quality", &quality_, 0.0f, 1.0f, "%.2f");
        reconvert |= ImGui::IsItemDeactivatedAfterEdit();

        ImGui::Spacing();

        ImGui::TextUnformatted(LOC(Mesh2Splat::MAX_RESOLUTION));
        widgets::HelpMarker(LOC(Mesh2Splat::TOOLTIP_MAX_RESOLUTION));
        ImGui::SetNextItemWidth(-1);

        static const char* resolution_labels[] = {"128", "256", "512", "1024", "2048", "4096"};
        reconvert |= ImGui::Combo("##max_resolution", &resolution_index_, resolution_labels, kResolutionOptionCount);

        ImGui::Spacing();

        ImGui::Spacing();
        const int target = computeResolutionTarget();
        ImGui::Text("%s: %d", LOC(Mesh2Splat::EFFECTIVE_RESOLUTION), target);

        if (reconvert && has_initial_conversion_)
            triggerConversion();
    }

    void Mesh2SplatPanel::drawConvertButton() {
        auto& async = viewer_->getGuiManager()->asyncTasks();
        const bool converting = async.isMesh2SplatActive();
        const bool no_mesh = selected_mesh_name_.empty();
        const bool disabled = converting || no_mesh;

        if (disabled)
            ImGui::BeginDisabled();

        if (widgets::ColoredButton(LOC(Mesh2Splat::CONVERT), widgets::ButtonStyle::Primary, {-1, 0}))
            triggerConversion();

        if (disabled)
            ImGui::EndDisabled();
    }

    void Mesh2SplatPanel::drawProgress() {
        auto& async = viewer_->getGuiManager()->asyncTasks();
        if (!async.isMesh2SplatActive())
            return;

        ImGui::Spacing();
        const float progress = async.getMesh2SplatProgress();
        const std::string stage = async.getMesh2SplatStage();
        widgets::DrawProgressBar(progress, stage.c_str());
    }

    void Mesh2SplatPanel::drawError() {
        auto& async = viewer_->getGuiManager()->asyncTasks();
        const std::string error = async.getMesh2SplatError();
        if (error.empty())
            return;

        ImGui::Spacing();
        const auto& t = theme();
        ImGui::PushStyleColor(ImGuiCol_Text, t.palette.error);
        ImGui::TextWrapped("%s", error.c_str());
        ImGui::PopStyleColor();
    }

} // namespace lfs::vis::gui::panels
