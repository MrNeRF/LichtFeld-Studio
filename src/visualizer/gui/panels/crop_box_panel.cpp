/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/crop_box_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "rendering/rendering_manager.hpp"
#include "visualizer_impl.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

namespace gs::gui::panels {

    // Apply rotation to crop box transform
    static void updateRotationMatrix(geometry::EuclideanTransform& transform,
                                     const glm::vec3& min_bounds,
                                     const glm::vec3& max_bounds,
                                     float delta_rot_x, float delta_rot_y, float delta_rot_z) {
        float rad_x = glm::radians(delta_rot_x);
        float rad_y = glm::radians(delta_rot_y);
        float rad_z = glm::radians(delta_rot_z);

        geometry::EuclideanTransform rotate(rad_x, rad_y, rad_z, 0.0f, 0.0f, 0.0f);

        glm::vec3 center = (min_bounds + max_bounds) * 0.5f;

        geometry::EuclideanTransform translate_to_origin(-center);
        geometry::EuclideanTransform translate_back = translate_to_origin.inv();

        transform = translate_back * rotate * translate_to_origin * transform;
    }

    void DrawCropBoxControls(const UIContext& ctx) {
        auto render_manager = ctx.viewer->getRenderingManager();
        if (!render_manager)
            return;

        if (!ImGui::CollapsingHeader("Crop Box")) {
            return;
        }

        auto settings = render_manager->getSettings();
        bool settings_changed = false;

        if (ImGui::Checkbox("Show Crop Box", &settings.show_crop_box)) {
            settings_changed = true;
        }

        if (ImGui::Checkbox("Use Crop Box", &settings.use_crop_box)) {
            settings_changed = true;
        }

        ImVec4 orangishColor(1.0f, 0.55f, 0.0f, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Button, orangishColor);

        // Add the "Crop Active PLY" button below the checkboxes
        if (ImGui::Button("Crop Active PLY")) {
            gs::geometry::BoundingBox crop_box;
            crop_box.setBounds(settings.crop_min, settings.crop_max);
            geometry::EuclideanTransform transform(settings.crop_transform.inv());
            crop_box.setworld2BBox(transform);
            // Emit event for bounds change
            events::cmd::CropPLY{
                .crop_box = crop_box}
                .emit();
        }
        ImGui::PopStyleColor(1); // pop orange color

        if (settings.show_crop_box) {
            // Appearance controls
            float bbox_color[3] = {settings.crop_color.x, settings.crop_color.y, settings.crop_color.z};
            if (ImGui::ColorEdit3("Box Color", bbox_color)) {
                settings.crop_color = glm::vec3(bbox_color[0], bbox_color[1], bbox_color[2]);
                settings_changed = true;
            }

            float available_width = ImGui::GetContentRegionAvail().x;
            float button_width = 120.0f;
            float slider_width = available_width - button_width - ImGui::GetStyle().ItemSpacing.x;

            ImGui::SetNextItemWidth(slider_width);
            if (ImGui::SliderFloat("Line Width", &settings.crop_line_width, 0.5f, 10.0f)) {
                settings_changed = true;
            }

            if (ImGui::Button("Reset to Default")) {
                settings.crop_min = glm::vec3(-1.0f, -1.0f, -1.0f);
                settings.crop_max = glm::vec3(1.0f, 1.0f, 1.0f);
                settings.crop_transform = geometry::EuclideanTransform();
                settings_changed = true;
            }

            // Rotation controls
            if (ImGui::TreeNode("Rotation")) {
                ImGui::Text("Ctrl+click for faster steps");
                ImGui::Text("Rotation around crop box axes:");

                const float rotation_step = 1.0f;
                const float rotation_step_fast = 15.0f;

                static float rotate_timer_x = 0.0f;
                static float rotate_timer_y = 0.0f;
                static float rotate_timer_z = 0.0f;

                float step = ImGui::GetIO().KeyCtrl ? rotation_step_fast : rotation_step;
                float repeat_rate = 0.05f;

                float diff_x = 0, diff_y = 0, diff_z = 0;

                // X-axis rotation
                ImGui::Text("X-axis:");
                ImGui::SameLine();
                ImGui::Text("RotX");

                if (ImGui::ArrowButton("##RotX_Up", ImGuiDir_Up)) {
                    diff_x = step;
                    rotate_timer_x = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_x += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_x >= repeat_rate) {
                        diff_x = step;
                        rotate_timer_x = 0.0f;
                    }
                }

                ImGui::SameLine();
                if (ImGui::ArrowButton("##RotX_Down", ImGuiDir_Down)) {
                    diff_x = -step;
                    rotate_timer_x = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_x += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_x >= repeat_rate) {
                        diff_x = -step;
                        rotate_timer_x = 0.0f;
                    }
                }

                // Y-axis rotation
                ImGui::Text("Y-axis:");
                ImGui::SameLine();
                ImGui::Text("RotY");

                if (ImGui::ArrowButton("##RotY_Up", ImGuiDir_Up)) {
                    diff_y = step;
                    rotate_timer_y = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_y += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_y >= repeat_rate) {
                        diff_y = step;
                        rotate_timer_y = 0.0f;
                    }
                }

                ImGui::SameLine();
                if (ImGui::ArrowButton("##RotY_Down", ImGuiDir_Down)) {
                    diff_y = -step;
                    rotate_timer_y = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_y += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_y >= repeat_rate) {
                        diff_y = -step;
                        rotate_timer_y = 0.0f;
                    }
                }

                // Z-axis rotation
                ImGui::Text("Z-axis:");
                ImGui::SameLine();
                ImGui::Text("RotZ");

                if (ImGui::ArrowButton("##RotZ_Up", ImGuiDir_Up)) {
                    diff_z = step;
                    rotate_timer_z = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_z += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_z >= repeat_rate) {
                        diff_z = step;
                        rotate_timer_z = 0.0f;
                    }
                }

                ImGui::SameLine();
                if (ImGui::ArrowButton("##RotZ_Down", ImGuiDir_Down)) {
                    diff_z = -step;
                    rotate_timer_z = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    rotate_timer_z += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_z >= repeat_rate) {
                        diff_z = -step;
                        rotate_timer_z = 0.0f;
                    }
                }

                if (diff_x != 0 || diff_y != 0 || diff_z != 0) {
                    updateRotationMatrix(settings.crop_transform, settings.crop_min, settings.crop_max,
                                         diff_x, diff_y, diff_z);
                    settings_changed = true;
                }

                ImGui::TreePop();
            }

            // Bounds controls
            if (ImGui::TreeNode("Bounds")) {
                float min_bounds[3] = {settings.crop_min.x, settings.crop_min.y, settings.crop_min.z};
                float max_bounds[3] = {settings.crop_max.x, settings.crop_max.y, settings.crop_max.z};

                bool bounds_changed = false;

                const float max_box_size = 200.0f;
                const float min_range = -max_box_size * 0.5f;
                const float max_range = max_box_size * 0.5f;
                const float bound_step = 0.01f;
                const float bound_step_fast = 0.1f;

                ImGui::Text("Ctrl+click for faster steps");
                ImGui::Text("Min Bounds:");

                // calculate the exact width to hold 0000.000 string in the text box + extra
                float text_width = ImGui::CalcTextSize("0000.000").x + ImGui::GetStyle().FramePadding.x * 2.0f + 50.0f;

                // Min bounds
                ImGui::Text("X:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                bounds_changed |= ImGui::InputFloat("##MinX", &min_bounds[0], bound_step, bound_step_fast, "%.3f");
                min_bounds[0] = std::clamp(min_bounds[0], min_range, max_range);
                min_bounds[0] = std::min(min_bounds[0], max_bounds[0]);

                ImGui::Text("Y:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                bounds_changed |= ImGui::InputFloat("##MinY", &min_bounds[1], bound_step, bound_step_fast, "%.3f");
                min_bounds[1] = std::clamp(min_bounds[1], min_range, max_range);
                min_bounds[1] = std::min(min_bounds[1], max_bounds[1]);

                ImGui::Text("Z:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                bounds_changed |= ImGui::InputFloat("##MinZ", &min_bounds[2], bound_step, bound_step_fast, "%.3f");
                min_bounds[2] = std::clamp(min_bounds[2], min_range, max_range);
                min_bounds[2] = std::min(min_bounds[2], max_bounds[2]);

                ImGui::Separator();
                ImGui::Text("Max Bounds:");

                // Max bounds
                ImGui::Text("X:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                bounds_changed |= ImGui::InputFloat("##MaxX", &max_bounds[0], bound_step, bound_step_fast, "%.3f");
                max_bounds[0] = std::clamp(max_bounds[0], min_range, max_range);
                max_bounds[0] = std::max(max_bounds[0], min_bounds[0]);

                ImGui::Text("Y:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                bounds_changed |= ImGui::InputFloat("##MaxY", &max_bounds[1], bound_step, bound_step_fast, "%.3f");
                max_bounds[1] = std::clamp(max_bounds[1], min_range, max_range);
                max_bounds[1] = std::max(max_bounds[1], min_bounds[1]);

                ImGui::Text("Z:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                bounds_changed |= ImGui::InputFloat("##MaxZ", &max_bounds[2], bound_step, bound_step_fast, "%.3f");
                max_bounds[2] = std::clamp(max_bounds[2], min_range, max_range);
                max_bounds[2] = std::max(max_bounds[2], min_bounds[2]);

                if (bounds_changed) {
                    settings.crop_min = glm::vec3(min_bounds[0], min_bounds[1], min_bounds[2]);
                    settings.crop_max = glm::vec3(max_bounds[0], max_bounds[1], max_bounds[2]);
                    settings_changed = true;

                    // Emit event for bounds change
                    events::ui::CropBoxChanged{
                        .min_bounds = settings.crop_min,
                        .max_bounds = settings.crop_max,
                        .enabled = settings.use_crop_box}
                        .emit();
                }

                // Display info
                glm::vec3 center = (settings.crop_min + settings.crop_max) * 0.5f;
                glm::vec3 size = settings.crop_max - settings.crop_min;

                ImGui::Text("Center: (%.3f, %.3f, %.3f)", center.x, center.y, center.z);
                ImGui::Text("Size: (%.3f, %.3f, %.3f)", size.x, size.y, size.z);

                ImGui::TreePop();
            }
        }

        if (settings_changed) {
            render_manager->updateSettings(settings);
        }
    }

    const CropBoxState& getCropBoxState() {
        return CropBoxState::getInstance();
    }
} // namespace gs::gui::panels