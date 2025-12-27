/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <glad/glad.h>

#include "core/editor_context.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "gui/dpi_scale.hpp"
#include "gui/localization_manager.hpp"
#include "gui/panels/main_panel.hpp"
#include "gui/panels/tools_panel.hpp"
#include "gui/panels/training_panel.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_widgets.hpp"
#include "internal/resource_paths.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "tools/selection_tool.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <format>
#include <imgui.h>
#ifdef WIN32
#include <windows.h>
#endif

namespace lfs::vis::gui::panels {

    using namespace lfs::core::events;
    using namespace lichtfeld::Strings;

    namespace {
        // Selection group icons (Tabler Icons - MIT license)
        struct SelectionIcons {
            unsigned int locked = 0;
            unsigned int unlocked = 0;
            bool initialized = false;
        };

        SelectionIcons g_selection_icons;

        unsigned int loadIcon(const std::string& name) {
            try {
                const auto path = lfs::vis::getAssetPath("icon/scene/" + name);
                const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(path);

                unsigned int texture_id;
                glGenTextures(1, &texture_id);
                glBindTexture(GL_TEXTURE_2D, texture_id);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

                const GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
                glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

                lfs::core::free_image(data);
                glBindTexture(GL_TEXTURE_2D, 0);
                return texture_id;
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load icon {}: {}", name, e.what());
                return 0;
            }
        }

        void ensureIconsLoaded() {
            if (g_selection_icons.initialized)
                return;
            g_selection_icons.locked = loadIcon("locked.png");
            g_selection_icons.unlocked = loadIcon("unlocked.png");
            g_selection_icons.initialized = true;
        }
    } // namespace

    void DrawWindowControls(const UIContext& ctx) {

        ImGui::Text("%s", LOC(MainPanel::WINDOWS));
        ImGui::Checkbox(LOC(MainPanel::SCENE_PANEL), &(*ctx.window_states)["scene_panel"]);
    }

    void DrawSystemConsoleButton(const UIContext& ctx) {

#ifdef WIN32
        // On non-Windows platforms, dont show the console toggle button

        if (!ctx.window_states->at("system_console")) {
            if (ImGui::Button(LOC(MainPanel::SHOW_CONSOLE), ImVec2(-1, 0))) {
                HWND hwnd = GetConsoleWindow();
                Sleep(1);
                HWND owner = GetWindow(hwnd, GW_OWNER);

                if (owner == NULL) {
                    ShowWindow(hwnd, SW_SHOW); // Windows 10
                } else {
                    ShowWindow(owner, SW_SHOW); // Windows 11
                }
                ctx.window_states->at("system_console") = true;
            }
        } else {
            if (ImGui::Button(LOC(MainPanel::HIDE_CONSOLE), ImVec2(-1, 0))) {
                HWND hwnd = GetConsoleWindow();
                Sleep(1);
                HWND owner = GetWindow(hwnd, GW_OWNER);

                if (owner == NULL) {
                    ShowWindow(hwnd, SW_HIDE); // Windows 10
                } else {
                    ShowWindow(owner, SW_HIDE); // Windows 11
                }
                ctx.window_states->at("system_console") = false;
            }
        }
#endif // Win32
    }

    void DrawRenderingSettings(const UIContext& ctx) {
        auto render_manager = ctx.viewer->getRenderingManager();
        if (!render_manager)
            return;

        // Use EditorContext to check pre-training mode
        const bool force_point_cloud = ctx.editor && ctx.editor->forcePointCloudMode();

        // Get current render settings
        auto settings = render_manager->getSettings();
        bool settings_changed = false;

        // In pre-training mode, point cloud mode is always on
        if (force_point_cloud && !settings.point_cloud_mode) {
            settings.point_cloud_mode = true;
            settings_changed = true;
        }

        // Background Color
        ImGui::Separator();
        ImGui::Text("%s", LOC(MainPanel::BACKGROUND));
        float bg_color[3] = {settings.background_color.x, settings.background_color.y, settings.background_color.z};
        if (ImGui::ColorEdit3(LOC(MainPanel::COLOR), bg_color)) {
            settings.background_color = glm::vec3(bg_color[0], bg_color[1], bg_color[2]);
            settings_changed = true;
        }

        // Coordinate Axes
        ImGui::Separator();
        if (ImGui::Checkbox(LOC(MainPanel::SHOW_COORD_AXES), &settings.show_coord_axes)) {
            settings_changed = true;
        }

        if (settings.show_coord_axes) {
            ImGui::Indent();

            if (ImGui::SliderFloat(LOC(MainPanel::AXES_SIZE), &settings.axes_size, 0.5f, 10.0f)) {
                settings_changed = true;
            }

            ImGui::Text("%s", LOC(MainPanel::VISIBLE_AXES));
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

        // Pivot Point
        ImGui::Separator();
        if (ImGui::Checkbox(LOC(MainPanel::SHOW_PIVOT), &settings.show_pivot)) {
            settings_changed = true;
        }

        // Grid checkbox and settings
        ImGui::Separator();
        if (ImGui::Checkbox(LOC(MainPanel::SHOW_GRID), &settings.show_grid)) {
            settings_changed = true;

            // Emit grid settings changed event
            ui::GridSettingsChanged{
                .enabled = settings.show_grid,
                .plane = static_cast<int>(settings.grid_plane),
                .opacity = settings.grid_opacity}
                .emit();
        }

        // Show grid settings only when grid is enabled
        if (settings.show_grid) {
            ImGui::Indent();

            // Grid plane selection
            const char* planes[] = {LOC(MainPanel::PLANE_YZ), LOC(MainPanel::PLANE_XZ), LOC(MainPanel::PLANE_XY)};
            int current_plane = static_cast<int>(settings.grid_plane);
            if (ImGui::Combo(LOC(MainPanel::PLANE), &current_plane, planes, IM_ARRAYSIZE(planes))) {
                settings.grid_plane = current_plane;
                settings_changed = true;

                ui::GridSettingsChanged{
                    .enabled = settings.show_grid,
                    .plane = current_plane,
                    .opacity = settings.grid_opacity}
                    .emit();
            }

            // Grid opacity
            if (ImGui::SliderFloat(LOC(MainPanel::GRID_OPACITY), &settings.grid_opacity, 0.0f, 1.0f)) {
                settings_changed = true;

                ui::GridSettingsChanged{
                    .enabled = settings.show_grid,
                    .plane = static_cast<int>(settings.grid_plane),
                    .opacity = settings.grid_opacity}
                    .emit();
            }

            ImGui::Unindent();
        }

        // Camera Frustums
        ImGui::Separator();
        if (ImGui::Checkbox(LOC(MainPanel::CAMERA_FRUSTUMS), &settings.show_camera_frustums)) {
            settings_changed = true;
        }

        if (settings.show_camera_frustums) {
            ImGui::Indent();
            const std::string scale_label = std::string(LOC(Tooltip::SCALE_CAMERA)) + "##camera";
            if (widgets::SliderWithReset(scale_label.c_str(), &settings.camera_frustum_scale, 0.01f, 10.0f, 0.25f)) {
                settings_changed = true;
            }
            ImGui::Unindent();
        }

        // Point Cloud Mode
        ImGui::Separator();
        ImGui::BeginDisabled(force_point_cloud);
        if (ImGui::Checkbox(LOC(MainPanel::POINT_CLOUD_MODE), &settings.point_cloud_mode)) {
            settings_changed = true;
            ui::PointCloudModeChanged{
                .enabled = settings.point_cloud_mode,
                .voxel_size = settings.voxel_size}
                .emit();
        }
        ImGui::EndDisabled();
        if (force_point_cloud && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("%s", LOC(Tooltip::POINT_CLOUD_FORCED));
        }

        if (settings.point_cloud_mode || force_point_cloud) {
            ImGui::Indent();
            if (widgets::SliderWithReset(LOC(Tooltip::POINT_SIZE), &settings.voxel_size, 0.001f, 0.1f, 0.03f)) {
                settings_changed = true;
                ui::PointCloudModeChanged{
                    .enabled = settings.point_cloud_mode,
                    .voxel_size = settings.voxel_size}
                    .emit();
            }
            ImGui::Unindent();
        }

        // Selection Colors
        ImGui::Separator();
        if (ImGui::CollapsingHeader(LOC(MainPanel::SELECTION_COLORS))) {
            ImGui::Indent();

            auto color_edit = [&](const char* label, glm::vec3& color) {
                float c[3] = {color.x, color.y, color.z};
                if (ImGui::ColorEdit3(label, c)) {
                    color = glm::vec3(c[0], c[1], c[2]);
                    settings_changed = true;
                }
            };

            color_edit(LOC(MainPanel::COMMITTED), settings.selection_color_committed);
            color_edit(LOC(MainPanel::PREVIEW), settings.selection_color_preview);
            color_edit(LOC(MainPanel::CENTER_MARKER), settings.selection_color_center_marker);

            ImGui::Unindent();
        }

        // Selection Behavior
        ImGui::Separator();
        if (ImGui::Checkbox(LOC(MainPanel::DESATURATE_UNSELECTED), &settings.desaturate_unselected)) {
            settings_changed = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", LOC(Tooltip::DESATURATE_UNSELECTED));
        }

        // Apply settings changes if any
        if (settings_changed) {
            render_manager->updateSettings(settings);
        }

        ImGui::Separator();

        float fov = settings.fov;
        if (widgets::SliderWithReset(LOC(MainPanel::FOV), &fov, 45.0f, 120.0f, 75.0f)) {
            render_manager->setFov(fov);

            ui::RenderSettingsChanged{
                .fov = fov,
                .scaling_modifier = std::nullopt,
                .antialiasing = std::nullopt,
                .background_color = std::nullopt,
                .equirectangular = std::nullopt}
                .emit();
        }

        // SH DEGREE selection
        const char* sh_degrees[] = {"0", "1", "2", "3"};
        int current_sh_degree = static_cast<int>(settings.sh_degree);
        if (ImGui::Combo(LOC(MainPanel::SH_DEGREE), &current_sh_degree, sh_degrees, IM_ARRAYSIZE(sh_degrees))) {
            settings.sh_degree = current_sh_degree;
            settings_changed = true;

            ui::RenderSettingsChanged{
                .sh_degree = current_sh_degree,
                .fov = std::nullopt,
                .scaling_modifier = std::nullopt,
                .antialiasing = std::nullopt,
                .background_color = std::nullopt,
                .equirectangular = std::nullopt}
                .emit();
        }

        if (ImGui::Checkbox(LOC(MainPanel::EQUIRECTANGULAR), &settings.equirectangular)) {
            settings_changed = true;

            ui::RenderSettingsChanged{
                .sh_degree = std::nullopt,
                .fov = std::nullopt,
                .scaling_modifier = std::nullopt,
                .antialiasing = std::nullopt,
                .background_color = std::nullopt,
                .equirectangular = settings.equirectangular}
                .emit();
        }

        // GUT (Gaussian Unscented Transform) for non-pinhole cameras
        if (ImGui::Checkbox(LOC(MainPanel::GUT_MODE), &settings.gut)) {
            settings_changed = true;
            render_manager->updateSettings(settings);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", LOC(Tooltip::GUT_MODE));
        }

        // Mip Filter (anti-aliasing for distant/small Gaussians)
        if (ImGui::Checkbox(LOC(MainPanel::MIP_FILTER), &settings.mip_filter)) {
            settings_changed = true;
            render_manager->updateSettings(settings);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", LOC(Tooltip::MIP_FILTER));
        }
    }

    void DrawSelectionGroups(const UIContext& ctx) {
        // Only show when selection tool is active
        auto* selection_tool = ctx.viewer->getSelectionTool();
        if (!selection_tool || !selection_tool->isEnabled())
            return;

        auto* scene_manager = ctx.viewer->getSceneManager();
        if (!scene_manager)
            return;

        // Lazy-load icons
        ensureIconsLoaded();

        Scene& scene = scene_manager->getScene();

        if (!ImGui::CollapsingHeader(LOC(MainPanel::SELECTION_GROUPS), ImGuiTreeNodeFlags_DefaultOpen))
            return;

        const float button_width = ImGui::GetContentRegionAvail().x;

        if (ImGui::Button(LOC(MainPanel::ADD_GROUP), ImVec2(button_width, 0))) {
            if (selection_tool->hasActivePolygon()) {
                selection_tool->clearPolygon();
            }
            scene.addSelectionGroup("", glm::vec3(0.0f));
        }

        const auto& groups = scene.getSelectionGroups();
        const uint8_t active_id = scene.getActiveSelectionGroup();

        if (groups.empty()) {
            ImGui::TextColored(theme().palette.text_dim, "%s", LOC(MainPanel::NO_SELECTION_GROUPS));
        } else {
            scene.updateSelectionGroupCounts();

            for (const auto& group : groups) {
                ImGui::PushID(group.id);

                const bool is_active = (group.id == active_id);

                // Highlight active group with colored background
                if (is_active) {
                    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(group.color.r * 0.4f, group.color.g * 0.4f, group.color.b * 0.4f, 0.6f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(group.color.r * 0.5f, group.color.g * 0.5f, group.color.b * 0.5f, 0.7f));
                }

                // Lock button with icon
                const bool is_locked = group.locked;
                const unsigned int lock_tex = is_locked ? g_selection_icons.locked : g_selection_icons.unlocked;
                const ImVec4 lock_tint = is_locked
                                             ? ImVec4(1.0f, 0.7f, 0.3f, 1.0f)  // Orange for locked
                                             : ImVec4(0.6f, 0.6f, 0.6f, 0.6f); // Gray for unlocked

                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, withAlpha(theme().palette.surface_bright, 0.5f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, withAlpha(theme().palette.surface_bright, 0.7f));

                const float LOCK_ICON_SIZE = 14.0f * getDpiScale();
                if (lock_tex) {
                    if (ImGui::ImageButton("##lock", static_cast<ImTextureID>(lock_tex),
                                           ImVec2(LOCK_ICON_SIZE, LOCK_ICON_SIZE),
                                           ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0), lock_tint)) {
                        scene.setSelectionGroupLocked(group.id, !is_locked);
                    }
                } else {
                    if (ImGui::SmallButton(is_locked ? "L" : "U")) {
                        scene.setSelectionGroupLocked(group.id, !is_locked);
                    }
                }
                ImGui::PopStyleColor(3);

                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", is_locked ? LOC(Tooltip::LOCKED) : LOC(Tooltip::UNLOCKED));
                }
                ImGui::SameLine();

                // Color picker
                ImVec4 color(group.color.r, group.color.g, group.color.b, 1.0f);
                if (ImGui::ColorEdit3("##color", &color.x,
                                      ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_NoAlpha)) {
                    scene.setSelectionGroupColor(group.id, glm::vec3(color.x, color.y, color.z));
                }
                ImGui::SameLine();

                // Selectable group - show active marker
                const std::string label = is_active
                                              ? std::format("> {} ({})", group.name, group.count)
                                              : std::format("  {} ({})", group.name, group.count);
                if (ImGui::Selectable(label.c_str(), is_active)) {
                    if (selection_tool->hasActivePolygon()) {
                        selection_tool->clearPolygon();
                    }
                    scene.setActiveSelectionGroup(group.id);
                }

                if (is_active) {
                    ImGui::PopStyleColor(2);
                }

                // Context menu
                if (ImGui::BeginPopupContextItem("##ctx")) {
                    if (ImGui::MenuItem(is_locked ? LOC(lichtfeld::Strings::SelectionGroup::UNLOCK) : LOC(lichtfeld::Strings::SelectionGroup::LOCK))) {
                        scene.setSelectionGroupLocked(group.id, !is_locked);
                    }
                    if (ImGui::MenuItem(LOC(MainPanel::CLEAR))) {
                        if (is_active && selection_tool->hasActivePolygon()) {
                            selection_tool->clearPolygon();
                        }
                        scene.clearSelectionGroup(group.id);
                    }
                    if (ImGui::MenuItem(LOC(Common::DELETE_ITEM))) {
                        if (is_active && selection_tool->hasActivePolygon()) {
                            selection_tool->clearPolygon();
                        }
                        scene.removeSelectionGroup(group.id);
                    }
                    ImGui::EndPopup();
                }

                ImGui::PopID();
            }
        }
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
    }
} // namespace lfs::vis::gui::panels
