#include "gui/panels/main_panel.hpp"
#include "core/events.hpp"
#include "gui/panels/tools_panel.hpp"
#include "gui/panels/training_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <imgui.h>

namespace gs::gui::panels {

    void DrawMainPanel(const UIContext& ctx) {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));

        // Simplified flags - positioning is handled in GuiManager::render()
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar |
                                 ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoCollapse |
                                 ImGuiWindowFlags_NoTitleBar; // Add this to remove title bar

        if (ImGui::Begin("Rendering Setting", nullptr, flags)) {
            // Add a custom title
            ImGui::Text("Rendering Settings");
            ImGui::Separator();

            DrawWindowControls(ctx);
            ImGui::Separator();

            widgets::DrawModeStatus(ctx);
            ImGui::Separator();

            DrawRenderingSettings(ctx);
            ImGui::Separator();

            if (ctx.viewer->getTrainer()) {
                DrawTrainingControls(ctx);
                ImGui::Separator();
            }

            DrawProgressInfo(ctx);
            ImGui::Separator();

            DrawToolsPanel(ctx);
        }
        ImGui::End();

        ImGui::PopStyleColor();
    }

    void DrawWindowControls(const UIContext& ctx) {
        if (ImGui::Button("Open Scripting Console", ImVec2(-1, 0))) {
            (*ctx.window_states)["console"] = true;
        }

        if (ImGui::Button("Open Camera Controls", ImVec2(-1, 0))) {
            (*ctx.window_states)["camera_controls"] = true;
        }

        ImGui::Separator();
        ImGui::Text("Windows");
        ImGui::Checkbox("Scripting Console", &(*ctx.window_states)["console"]);
        ImGui::Checkbox("Camera Controls", &(*ctx.window_states)["camera_controls"]);
        ImGui::Checkbox("Scene Panel", &(*ctx.window_states)["scene_panel"]);
    }

    void DrawRenderingSettings(const UIContext& ctx) {
        auto render_manager = ctx.viewer->getRenderingManager();
        if (!render_manager)
            return;

        ImGui::Text("Rendering Settings");
        ImGui::Separator();

        // Get current render settings
        auto settings = render_manager->getSettings();
        bool settings_changed = false;

        // Point Cloud Mode checkbox
        if (ImGui::Checkbox("Point Cloud Mode", &settings.point_cloud_mode)) {
            settings_changed = true;

            events::ui::PointCloudModeChanged{
                .enabled = settings.point_cloud_mode,
                .voxel_size = settings.voxel_size}
                .emit();
        }

        // Show voxel size slider only when in point cloud mode
        if (settings.point_cloud_mode) {
            if (widgets::SliderWithReset("Voxel Size", &settings.voxel_size, 0.001f, 0.1f, 0.01f)) {
                settings_changed = true;

                events::ui::PointCloudModeChanged{
                    .enabled = settings.point_cloud_mode,
                    .voxel_size = settings.voxel_size}
                    .emit();
            }
        }

        // Background Color
        ImGui::Separator();
        ImGui::Text("Background");
        float bg_color[3] = {settings.background_color.x, settings.background_color.y, settings.background_color.z};
        if (ImGui::ColorEdit3("Color##Background", bg_color)) {
            settings.background_color = glm::vec3(bg_color[0], bg_color[1], bg_color[2]);
            settings_changed = true;
        }

        // Coordinate Axes
        ImGui::Separator();
        if (ImGui::Checkbox("Show Coordinate Axes", &settings.show_coord_axes)) {
            settings_changed = true;
        }

        if (settings.show_coord_axes) {
            ImGui::Indent();

            if (ImGui::SliderFloat("Axes Size", &settings.axes_size, 0.5f, 10.0f)) {
                settings_changed = true;
            }

            ImGui::Text("Visible Axes:");
            bool axes_changed = false;
            if (ImGui::Checkbox("X##axis", &settings.axes_visibility[0])) {
                axes_changed = true;
            }
            ImGui::SameLine();
            if (ImGui::Checkbox("Y##axis", &settings.axes_visibility[1])) {
                axes_changed = true;
            }
            ImGui::SameLine();
            if (ImGui::Checkbox("Z##axis", &settings.axes_visibility[2])) {
                axes_changed = true;
            }

            if (axes_changed) {
                settings_changed = true;
            }

            ImGui::Unindent();
        }

        // Grid checkbox and settings
        ImGui::Separator();
        if (ImGui::Checkbox("Show Grid", &settings.show_grid)) {
            settings_changed = true;

            // Emit grid settings changed event
            events::ui::GridSettingsChanged{
                .enabled = settings.show_grid,
                .plane = static_cast<int>(settings.grid_plane),
                .opacity = settings.grid_opacity}
                .emit();
        }

        // Show grid settings only when grid is enabled
        if (settings.show_grid) {
            ImGui::Indent();

            // Grid plane selection
            const char* planes[] = {"YZ (X-plane)", "XZ (Y-plane)", "XY (Z-plane)"};
            int current_plane = static_cast<int>(settings.grid_plane);
            if (ImGui::Combo("Plane", &current_plane, planes, IM_ARRAYSIZE(planes))) {
                settings.grid_plane = current_plane;
                settings_changed = true;

                events::ui::GridSettingsChanged{
                    .enabled = settings.show_grid,
                    .plane = current_plane,
                    .opacity = settings.grid_opacity}
                    .emit();
            }

            // Grid opacity
            if (ImGui::SliderFloat("Grid Opacity", &settings.grid_opacity, 0.0f, 1.0f)) {
                settings_changed = true;

                events::ui::GridSettingsChanged{
                    .enabled = settings.show_grid,
                    .plane = static_cast<int>(settings.grid_plane),
                    .opacity = settings.grid_opacity}
                    .emit();
            }

            ImGui::Unindent();
        }

        // Apply settings changes if any
        if (settings_changed) {
            render_manager->updateSettings(settings);
        }

        ImGui::Separator();

        // Use direct accessors from RenderingManager
        float scaling_modifier = settings.scaling_modifier;
        if (widgets::SliderWithReset("Scale", &scaling_modifier, 0.01f, 3.0f, 1.0f)) {
            render_manager->setScalingModifier(scaling_modifier);

            events::ui::RenderSettingsChanged{
                .fov = std::nullopt,
                .scaling_modifier = scaling_modifier,
                .antialiasing = std::nullopt,
                .background_color = std::nullopt}
                .emit();
        }

        float fov = settings.fov;
        if (widgets::SliderWithReset("FoV", &fov, 45.0f, 120.0f, 75.0f)) {
            render_manager->setFov(fov);

            events::ui::RenderSettingsChanged{
                .fov = fov,
                .scaling_modifier = std::nullopt,
                .antialiasing = std::nullopt,
                .background_color = std::nullopt}
                .emit();
        }

        // Display current FPS and VSync control on the same line
        float average_fps = ctx.viewer->getAverageFPS();
        if (average_fps > 0.0f) {
            ImGui::Text("FPS: %6.1f", average_fps);

            // Add VSync checkbox on the same line
            ImGui::SameLine();
            ImGui::Spacing();
            ImGui::SameLine();

            // Get current VSync state from viewer
            bool vsync_enabled = ctx.viewer->getVSyncEnabled();

            if (ImGui::Checkbox("VSync", &vsync_enabled)) {
                // Set VSync through the viewer's public interface
                ctx.viewer->setVSync(vsync_enabled);
            }

            // Add tooltip
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Toggle Vertical Synchronization\n%s",
                                  vsync_enabled ? "FPS capped to monitor refresh rate"
                                                : "Uncapped FPS");
            }
        }

#ifdef CUDA_GL_INTEROP_ENABLED
        ImGui::Text("Render Mode: GPU Direct (Interop)");
#else
        ImGui::Text("Render Mode: CPU Copy");
#endif
    }

    void DrawProgressInfo(const UIContext& ctx) {
        // Get training info from TrainerManager instead of ViewerStateManager
        auto* trainer_manager = ctx.viewer->getTrainerManager();
        if (!trainer_manager) {
            return;
        }

        int current_iter = trainer_manager->getCurrentIteration();
        int total_iter = trainer_manager->getTotalIterations();
        int num_splats = trainer_manager->getNumSplats();

        // Fix: Convert deque to vector
        std::deque<float> loss_deque = trainer_manager->getLossBuffer();
        std::vector<float> loss_data(loss_deque.begin(), loss_deque.end());

        float fraction = total_iter > 0 ? float(current_iter) / float(total_iter) : 0.0f;
        char overlay_text[64];
        std::snprintf(overlay_text, sizeof(overlay_text), "%d / %d", current_iter, total_iter);

        widgets::DrawProgressBar(fraction, overlay_text);

        if (loss_data.size() > 0) {
            auto [min_it, max_it] = std::minmax_element(loss_data.begin(), loss_data.end());
            float min_val = *min_it, max_val = *max_it;

            if (min_val == max_val) {
                min_val -= 1.0f;
                max_val += 1.0f;
            } else {
                float margin = (max_val - min_val) * 0.05f;
                min_val -= margin;
                max_val += margin;
            }

            char loss_label[64];
            std::snprintf(loss_label, sizeof(loss_label), "Loss: %.4f", loss_data.back());

            widgets::DrawLossPlot(loss_data.data(), static_cast<int>(loss_data.size()),
                                  min_val, max_val, loss_label);
        }

        ImGui::Text("num Splats: %d", num_splats);
    }
} // namespace gs::gui::panels
