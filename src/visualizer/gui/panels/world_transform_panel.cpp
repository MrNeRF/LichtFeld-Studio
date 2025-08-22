#include "gui/panels/world_transform_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "rendering/rendering_manager.hpp"
#include "tools/translation_gizmo_tool.hpp"
#include "visualizer_impl.hpp"
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>

namespace gs::gui::panels {

    void DrawWorldTransformControls(const UIContext& ctx) {
        auto render_manager = ctx.viewer->getRenderingManager();
        if (!render_manager)
            return;

        if (!ImGui::CollapsingHeader("World Transform")) {
            return;
        }

        auto settings = render_manager->getSettings();
        bool settings_changed = false;

        /* COMMENTED OUT FOR NOW - WE NEED FIRST ORIENT THE WORLD CORRECTLY BEFORE WE ENABLE IT
        // SINGLE CHECKBOX TO CONTROL GIZMO
        if (ImGui::Checkbox("Show Translation Gizmo", &settings.show_translation_gizmo)) {
            settings_changed = true;

            // Sync tool interaction state with visibility
            auto* gizmo_tool = ctx.viewer->getTranslationGizmoTool();
            if (gizmo_tool) {
                gizmo_tool->setEnabled(settings.show_translation_gizmo);

                // When enabling, sync gizmo position with current world transform
                if (settings.show_translation_gizmo) {
                    // The gizmo tool already tracks current_transform_ internally
                    // Just make sure it's in sync
                    auto current_transform = gizmo_tool->getTransform();
                    if (current_transform.isIdentity() && !settings.world_transform.isIdentity()) {
                        // If gizmo is at identity but world transform isn't, sync them
                        // This is handled internally by the tool
                    }
                }
            }
        }

        if (settings.show_translation_gizmo) {
            // Gizmo scale slider
            if (ImGui::SliderFloat("Gizmo Scale", &settings.gizmo_scale, 0.5f, 3.0f)) {
                settings_changed = true;
            }

            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Gizmo Controls:");
            ImGui::BulletText("Drag arrows: Move along axis");
            ImGui::BulletText("Drag planes: Move in 2D plane");
            ImGui::BulletText("R: Reset position");
        }
        */

        ImGui::Separator();

        // Store transform UI values statically
        static float rotation_degrees[3] = {0.0f, 0.0f, 0.0f};
        static float translation[3] = {0.0f, 0.0f, 0.0f};
        static bool transform_initialized = false;

        // Initialize from current transform if needed
        if (!transform_initialized) {
            glm::mat3 rot_mat = settings.world_transform.getRotationMat();
            glm::vec3 euler = glm::eulerAngles(glm::quat_cast(rot_mat));
            rotation_degrees[0] = glm::degrees(euler.x);
            rotation_degrees[1] = glm::degrees(euler.y);
            rotation_degrees[2] = glm::degrees(euler.z);

            glm::vec3 trans = settings.world_transform.getTranslation();
            translation[0] = trans.x;
            translation[1] = trans.y;
            translation[2] = trans.z;
            transform_initialized = true;
        }

        /* COMMENTED OUT - PART OF GIZMO FUNCTIONALITY
        // If gizmo is active and being dragged, update UI values from gizmo
        auto* gizmo_tool = ctx.viewer->getTranslationGizmoTool();
        if (settings.show_translation_gizmo && gizmo_tool) {
            if (gizmo_tool->isInteracting()) {
                // Get transform from gizmo tool
                auto gizmo_transform = gizmo_tool->getTransform();
                glm::vec3 trans = gizmo_transform.getTranslation();
                translation[0] = trans.x;
                translation[1] = trans.y;
                translation[2] = trans.z;

                // Update rotation if needed
                glm::mat3 rot_mat = gizmo_transform.getRotationMat();
                glm::vec3 euler = glm::eulerAngles(glm::quat_cast(rot_mat));
                rotation_degrees[0] = glm::degrees(euler.x);
                rotation_degrees[1] = glm::degrees(euler.y);
                rotation_degrees[2] = glm::degrees(euler.z);
            }
        }
        */

        // Transform status
        if (!settings.world_transform.isIdentity()) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Transform Active");
        } else {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Identity Transform");
        }

        ImGui::Separator();

        // Rotation controls
        ImGui::Text("Rotation (degrees):");
        if (ImGui::DragFloat3("##world_rotation", rotation_degrees, 0.1f, -360.0f, 360.0f, "%.1f")) {
            settings_changed = true;
        }

        // Translation controls
        ImGui::Text("Translation:");
        if (ImGui::DragFloat3("##world_translation", translation, 0.01f, -100.0f, 100.0f, "%.3f")) {
            settings_changed = true;
        }

        if (settings_changed) {
            // Update transform from UI values
            glm::vec3 rot_rad(glm::radians(rotation_degrees[0]),
                              glm::radians(rotation_degrees[1]),
                              glm::radians(rotation_degrees[2]));
            settings.world_transform = geometry::EuclideanTransform(
                rot_rad.x, rot_rad.y, rot_rad.z,
                translation[0], translation[1], translation[2]);

            render_manager->updateSettings(settings);
        }

        ImGui::Separator();

        // Reset button
        if (ImGui::Button("Reset Transform", ImVec2(-1, 0))) {
            settings.world_transform = geometry::EuclideanTransform();
            rotation_degrees[0] = rotation_degrees[1] = rotation_degrees[2] = 0.0f;
            translation[0] = translation[1] = translation[2] = 0.0f;

            /* COMMENTED OUT - GIZMO RESET
            // Also reset gizmo position if it exists
            auto* gizmo_tool = ctx.viewer->getTranslationGizmoTool();
            if (gizmo_tool) {
                // Tool will handle this internally via updateWorldTransform
            }
            */

            render_manager->updateSettings(settings);
        }

        // Show transform matrix info
        if (ImGui::TreeNode("Transform Matrix")) {
            glm::mat3 rot = settings.world_transform.getRotationMat();
            ImGui::Text("Rotation:");
            ImGui::Text("[%.3f, %.3f, %.3f]", rot[0][0], rot[1][0], rot[2][0]);
            ImGui::Text("[%.3f, %.3f, %.3f]", rot[0][1], rot[1][1], rot[2][1]);
            ImGui::Text("[%.3f, %.3f, %.3f]", rot[0][2], rot[1][2], rot[2][2]);

            glm::vec3 trans = settings.world_transform.getTranslation();
            ImGui::Separator();
            ImGui::Text("Translation: [%.3f, %.3f, %.3f]", trans.x, trans.y, trans.z);

            ImGui::TreePop();
        }
    }
} // namespace gs::gui::panels