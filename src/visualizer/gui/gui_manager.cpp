/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gui_manager.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "gui/panels/main_panel.hpp"
#include "gui/panels/scene_panel.hpp"
#include "gui/panels/tools_panel.hpp"
#include "gui/panels/training_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "gui/windows/file_browser.hpp"
#include "gui/windows/project_changed_dialog_box.hpp"

#include "internal/resource_paths.hpp"
#include "visualizer_impl.hpp"

#include <GLFW/glfw3.h>
#include <chrono>
#include <cstdarg>
#include <format>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_internal.h>

namespace gs::gui {

    GuiManager::GuiManager(visualizer::VisualizerImpl* viewer)
        : viewer_(viewer) {

        // Create components
        file_browser_ = std::make_unique<FileBrowser>();
        project_changed_dialog_box_ = std::make_unique<ProjectChangedDialogBox>();
        scene_panel_ = std::make_unique<ScenePanel>(viewer->trainer_manager_);
        save_project_browser_ = std::make_unique<SaveProjectBrowser>();
        menu_bar_ = std::make_unique<MenuBar>();

        // Initialize window states
        window_states_["file_browser"] = false;
        window_states_["scene_panel"] = true;
        window_states_["project_changed_dialog_box"] = false;
        window_states_["save_project_browser_before_exit"] = false;
        window_states_["show_save_browser"] = false;
        window_states_["system_console"] = false;
        window_states_["training_tab"] = false;

        // Initialize speed overlay state
        speed_overlay_visible_ = false;
        speed_overlay_duration_ = std::chrono::milliseconds(3000); // 3 seconds

        // Initialize focus state
        viewport_has_focus_ = false;

        setupEventHandlers();
    }

    GuiManager::~GuiManager() {
        // Cleanup handled automatically
    }

    void GuiManager::initMenuBar() {
        menu_bar_->setOnImportDataset([this]() {
            window_states_["file_browser"] = true;
#ifdef WIN32
            // show native windows file dialog for project file selection
            OpenDatasetFolderDialog();

            // hide the file browser
            events::cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif // WIN32
        });

        menu_bar_->setOnOpenProject([this]() {
            window_states_["file_browser"] = true;
#ifdef WIN32
            // show native windows file dialog for project file selection
            OpenProjectFileDialog();

            // hide the file browser
            events::cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif // WIN32
        });

        menu_bar_->setOnImportPLY([this]() {
            window_states_["file_browser"] = true;
#ifdef WIN32
            // show native windows file dialog for project file selection
            OpenPlyFileDialog();

            // hide the file browser
            events::cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif // WIN32
        });

        menu_bar_->setOnSaveProjectAs([this]() {
            window_states_["show_save_browser"] = true;
        });

        menu_bar_->setOnSaveProject([this]() {
            if (viewer_->project_) {
                events::cmd::SaveProject{viewer_->project_->getProjectOutputFolder().string()}.emit();
            }
        });

        menu_bar_->setOnExit([this]() {
            glfwSetWindowShouldClose(viewer_->getWindow(), true);
        });
    }

    void GuiManager::init() {
        // ImGui initialization
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
        io.ConfigWindowsMoveFromTitleBarOnly = true;

        // Platform/Renderer initialization
        ImGui_ImplGlfw_InitForOpenGL(viewer_->getWindow(), true);
        ImGui_ImplOpenGL3_Init("#version 430");

        float xscale, yscale;
        glfwGetWindowContentScale(viewer_->getWindow(), &xscale, &yscale);

        // some clamping / safety net for weird DPI values
        xscale = std::clamp(xscale, 1.0f, 2.0f);

        // Set application icon - use the resource path helper
        try {
            const auto icon_path = gs::visualizer::getAssetPath("lichtfeld-icon.png");
            const auto [data, width, height, channels] = load_image_with_alpha(icon_path);

            GLFWimage image{width, height, data};
            glfwSetWindowIcon(viewer_->getWindow(), 1, &image);
            free_image(data);
        } catch (const std::exception& e) {
            LOG_WARN("Could not load application icon: {}", e.what());
        }

        // Load fonts - use the resource path helper
        try {
            auto font_path = gs::visualizer::getAssetPath("JetBrainsMono-Regular.ttf");
            io.Fonts->AddFontFromFileTTF(font_path.string().c_str(), 14.0f * xscale);
        } catch (const std::exception& e) {
            // If font loading fails, just use the default font
            LOG_WARN("Could not load custom font: {}", e.what());
            LOG_DEBUG("Using default ImGui font");
        }

        applyDefaultStyle();

        // Configure file browser callback
        setFileSelectedCallback([this](const std::filesystem::path& path, bool is_dataset) {
            if (path.extension() == gs::management::Project::EXTENSION) {
                events::cmd::LoadProject{.path = path}.emit();
            } else {
                events::cmd::LoadFile{.path = path, .is_dataset = is_dataset}.emit();
            }

            window_states_["file_browser"] = false;
        });

        handleProjectChangedDialogCallback([this](bool save) {
            if (save) {
                window_states_["save_project_browser_before_exit"] = true;
            } else {
                force_exit_ = true;
                glfwSetWindowShouldClose(viewer_->getWindow(), true);
                LOG_INFO("Exiting LichtFeldStudio gracefully without saving");
            }
            window_states_["project_changed_dialog_box"] = false;
        });

        scene_panel_->setOnDatasetLoad([this](const std::filesystem::path& path) {
            if (path.empty()) {
                window_states_["file_browser"] = true;
            } else {
                events::cmd::LoadFile{.path = path, .is_dataset = true}.emit();
            }
        });

        initMenuBar();
    }

    void GuiManager::shutdown() {
        if (ImGui::GetCurrentContext()) {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }
    }

    void GuiManager::render() {
        // Start frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();

        // CRITICAL: Check mouse state BEFORE ImGui::NewFrame()
        // This is important because ImGui updates WantCaptureMouse during NewFrame()
        ImVec2 mouse_pos = ImGui::GetMousePos();
        bool mouse_in_viewport = isPositionInViewport(mouse_pos.x, mouse_pos.y);

        ImGui::NewFrame();

        if (menu_bar_) {
            menu_bar_->render();
        }

        // Override ImGui's mouse capture for right/middle buttons when in viewport
        // This ensures that camera controls work properly
        if (mouse_in_viewport && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
                ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
                ImGui::GetIO().WantCaptureMouse = false;
            }
        }

        // CRITICAL FIX: In point cloud mode, always disable ImGui mouse capture in viewport
        auto* rendering_manager = viewer_->getRenderingManager();
        if (rendering_manager) {
            const auto& settings = rendering_manager->getSettings();
            if (settings.point_cloud_mode && mouse_in_viewport &&
                !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
                ImGui::GetIO().WantCaptureMouse = false;
                ImGui::GetIO().WantCaptureKeyboard = false;
            }
        }

        // Create main dockspace
        const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(main_viewport->WorkPos);
        ImGui::SetNextWindowSize(main_viewport->WorkSize);
        ImGui::SetNextWindowViewport(main_viewport->ID);

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking |
                                        ImGuiWindowFlags_NoTitleBar |
                                        ImGuiWindowFlags_NoCollapse |
                                        ImGuiWindowFlags_NoResize |
                                        ImGuiWindowFlags_NoMove |
                                        ImGuiWindowFlags_NoBringToFrontOnFocus |
                                        ImGuiWindowFlags_NoNavFocus |
                                        ImGuiWindowFlags_NoBackground;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

        ImGui::Begin("DockSpace", nullptr, window_flags);
        ImGui::PopStyleVar(3);

        // DockSpace ID
        ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");

        // Create dockspace
        ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

        // Set up default layout on first run
        static bool first_time = true;
        if (first_time) {
            first_time = false;
            ImGui::DockBuilderRemoveNode(dockspace_id);
            ImGui::DockBuilderAddNode(dockspace_id, dockspace_flags | ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dockspace_id, main_viewport->WorkSize);

            ImGuiID dock_id_left = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.25f, nullptr, &dockspace_id);
            ImGuiID dock_id_right = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Right, 0.2f, nullptr, &dockspace_id);

            // Dock windows
            ImGui::DockBuilderDockWindow("Rendering", dock_id_left);
            ImGui::DockBuilderDockWindow("Scene", dock_id_right);
            ImGui::DockBuilderDockWindow("Training", dock_id_left);

            ImGui::DockBuilderFinish(dockspace_id);
        }

        ImGui::End();

        // Create context for this frame
        UIContext ctx{
            .viewer = viewer_,
            .file_browser = file_browser_.get(),
            .window_states = &window_states_};

        // Draw docked panels
        if (show_main_panel_) {
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));
            if (ImGui::Begin("Rendering", nullptr)) {
                // Draw contents without the manual sizing/positioning
                widgets::DrawModeStatusWithContentSwitch(ctx);
                ImGui::Separator();
                panels::DrawRenderingSettings(ctx);
                ImGui::Separator();
                panels::DrawToolsPanel(ctx);
                panels::DrawSystemConsoleButton(ctx);
            }
            ImGui::End();

            if (viewer_->getTrainer() && !window_states_["training_tab"]) {
                ImGui::SetWindowFocus("Rendering");
                window_states_["training_tab"] = true;
            }

            if (!viewer_->getTrainer()) {
                window_states_["training_tab"] = false;
            }

            if (window_states_["training_tab"]) {
                if (ImGui::Begin("Training", nullptr)) {
                    panels::DrawTrainingControls(ctx);
                    ImGui::Separator();
                    panels::DrawProgressInfo(ctx);
                }
                ImGui::End();
            }

            ImGui::PopStyleColor();
        }

        // Draw Scene panel
        if (window_states_["scene_panel"]) {
            scene_panel_->render(&window_states_["scene_panel"]);
        }

        // Render floating windows (these remain movable)
        if (window_states_["file_browser"]) {
            file_browser_->render(&window_states_["file_browser"]);
        }

        if (window_states_["project_changed_dialog_box"]) {
            project_changed_dialog_box_->render(&window_states_["project_changed_dialog_box"]);
        }

        if (window_states_["save_project_browser_before_exit"]) {
#ifdef WIN32
            bool was_project_saved = save_project_browser_->SaveProjectFileDialog(&window_states_["save_project_browser_before_exit"]);
#else
            bool was_project_saved = save_project_browser_->render(&window_states_["save_project_browser_before_exit"]);
#endif
            if (was_project_saved) {
                force_exit_ = true;
                glfwSetWindowShouldClose(viewer_->getWindow(), true);
                LOG_INFO("Exiting LichtFeldStudio gracefully after project save");
            }
        }

        if (window_states_["show_save_browser"]) {
#ifdef WIN32
            save_project_browser_->SaveProjectFileDialog(&window_states_["show_save_browser"]);
#else
            save_project_browser_->render(&window_states_["show_save_browser"]);
#endif
        }

        if (menu_bar_ && viewer_) {
            auto project = viewer_->getProject();
            menu_bar_->setIsProjectTemp(project ? project->getIsTempProject() : false);
        }

        // Render speed overlay if visible
        renderSpeedOverlay();

        // Render split view indicator if enabled
        if (rendering_manager) {
            auto split_info = rendering_manager->getSplitViewInfo();
            if (split_info.enabled) {
                // Create a small overlay showing what is being compared
                const ImGuiViewport* viewport = ImGui::GetMainViewport();

                // Position at top center
                ImVec2 overlay_pos(
                    viewport->WorkPos.x + (viewport->WorkSize.x - 400.0f) * 0.5f,
                    viewport->WorkPos.y + 10.0f);

                ImGui::SetNextWindowPos(overlay_pos, ImGuiCond_Always);
                ImGui::SetNextWindowSize(ImVec2(400.0f, 40.0f), ImGuiCond_Always);

                ImGuiWindowFlags overlay_flags =
                    ImGuiWindowFlags_NoTitleBar |
                    ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoScrollbar |
                    ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoSavedSettings |
                    ImGuiWindowFlags_NoInputs |
                    ImGuiWindowFlags_NoFocusOnAppearing;

                ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.8f));
                ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0f, 1.0f, 0.0f, 0.5f));
                ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 5.0f);

                if (ImGui::Begin("##SplitViewIndicator", nullptr, overlay_flags)) {
                    ImGui::SetCursorPos(ImVec2(10, 10));

                    // Draw the split view info - check mode
                    const auto& settings = rendering_manager->getSettings();
                    if (settings.split_view_mode == visualizer::SplitViewMode::GTComparison) {
                        // GT comparison mode
                        int cam_id = rendering_manager->getCurrentCameraId();
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f),
                                           "GT Comparison - Camera %d", cam_id);
                        ImGui::SameLine();
                        ImGui::TextDisabled(" (G: toggle, V: cycle modes, arrows: change camera)");
                    } else {
                        // PLY comparison mode
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f),
                                           "Split View: %s | %s",
                                           split_info.left_name.c_str(),
                                           split_info.right_name.c_str());
                        ImGui::SameLine();
                        ImGui::TextDisabled(" (T: cycle, V: exit)");
                    }
                }
                ImGui::End();

                ImGui::PopStyleVar(2);
                ImGui::PopStyleColor(2);
            }
        }

        // Get the viewport region for 3D rendering
        updateViewportRegion();

        // Update viewport focus based on mouse position
        updateViewportFocus();

        // Render viewport gizmo BEFORE focus indicator - always render regardless of focus
        if (show_viewport_gizmo_ && viewport_size_.x > 0 && viewport_size_.y > 0) {
            if (rendering_manager) {
                auto* engine = rendering_manager->getRenderingEngine();
                if (engine) {
                    const auto& viewport = viewer_->getViewport();
                    glm::mat3 camera_rotation = viewport.getRotationMatrix();

                    engine->renderViewportGizmo(
                        camera_rotation,
                        glm::vec2(viewport_pos_.x, viewport_pos_.y),
                        glm::vec2(viewport_size_.x, viewport_size_.y));
                }
            }
        }

        // Draw viewport focus indicator AFTER gizmo
        // DISABLE in point cloud mode
        bool draw_focus = viewport_has_focus_ && viewport_size_.x > 0 && viewport_size_.y > 0;
        if (rendering_manager) {
            const auto& settings = rendering_manager->getSettings();
            if (settings.point_cloud_mode) {
                draw_focus = false;
            }
        }

        if (draw_focus) {
            ImDrawList* draw_list = ImGui::GetForegroundDrawList();

            // The viewport_pos_ is already relative to the window, so we just need to add the window position
            const ImGuiViewport* main_vp = ImGui::GetMainViewport();
            ImVec2 screen_pos = ImVec2(
                main_vp->WorkPos.x + viewport_pos_.x,
                main_vp->WorkPos.y + viewport_pos_.y);

            // Animated glow
            float time = static_cast<float>(ImGui::GetTime());
            float pulse = (sin(time * 3.0f) + 1.0f) * 0.5f;

            // Outer glow
            draw_list->AddRect(
                screen_pos,
                ImVec2(screen_pos.x + viewport_size_.x, screen_pos.y + viewport_size_.y),
                IM_COL32(51, 153, 255, 127 + (int)(pulse * 76)), // Blue with pulsing alpha
                0.0f, 0, 3.0f);

            // Inner highlight
            draw_list->AddRect(
                ImVec2(screen_pos.x + 1, screen_pos.y + 1),
                ImVec2(screen_pos.x + viewport_size_.x - 1, screen_pos.y + viewport_size_.y - 1),
                IM_COL32(102, 204, 255, 76 + (int)(pulse * 50)),
                0.0f, 0, 1.0f);
        }

        // End frame
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Update and Render additional Platform Windows (for multi-viewport)
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }
    }

    void GuiManager::updateViewportRegion() {
        const ImGuiViewport* main_viewport = ImGui::GetMainViewport();

        // Start with full window
        float left = 0;
        float top = 0;
        float right = main_viewport->WorkSize.x;
        float bottom = main_viewport->WorkSize.y;

        // Find our docked windows and calculate the remaining space
        ImGuiWindow* settings_window = ImGui::FindWindowByName("Rendering Settings");
        ImGuiWindow* scene_window = ImGui::FindWindowByName("Scene");

        if (settings_window && settings_window->DockNode && settings_window->Active) {
            // Settings panel is on the left
            float panel_right = settings_window->Pos.x + settings_window->Size.x - main_viewport->WorkPos.x;
            left = std::max(left, panel_right);
        }

        if (scene_window && scene_window->DockNode && scene_window->Active) {
            // Scene panel is on the right
            float panel_left = scene_window->Pos.x - main_viewport->WorkPos.x;
            right = std::min(right, panel_left);
        }

        // Store relative to window
        viewport_pos_ = ImVec2(left, top);
        viewport_size_ = ImVec2(right - left, bottom - top);
    }

    void GuiManager::updateViewportFocus() {
        // Check if mouse is in viewport area
        ImVec2 mouse_pos = ImGui::GetMousePos();
        const ImGuiViewport* main_viewport = ImGui::GetMainViewport();

        // Convert mouse position to window-relative coordinates
        float mouse_x = mouse_pos.x - main_viewport->WorkPos.x;
        float mouse_y = mouse_pos.y - main_viewport->WorkPos.y;

        // Check if mouse is within viewport bounds
        bool mouse_in_viewport = (mouse_x >= viewport_pos_.x &&
                                  mouse_x < viewport_pos_.x + viewport_size_.x &&
                                  mouse_y >= viewport_pos_.y &&
                                  mouse_y < viewport_pos_.y + viewport_size_.y);

        // Check if ImGui wants input
        bool imgui_wants_input = ImGui::GetIO().WantCaptureMouse ||
                                 ImGui::GetIO().WantCaptureKeyboard ||
                                 ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);

        // Special handling for clicks - if clicking in viewport, it gets focus immediately
        bool mouse_clicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
                             ImGui::IsMouseClicked(ImGuiMouseButton_Right) ||
                             ImGui::IsMouseClicked(ImGuiMouseButton_Middle);

        if (mouse_clicked && mouse_in_viewport && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
            // Clicking in viewport gives it focus immediately
            viewport_has_focus_ = true;
            // Tell ImGui we don't want this mouse event
            ImGui::GetIO().WantCaptureMouse = false;
        } else {
            // Normal focus rules: viewport has focus when mouse is over it and ImGui doesn't want input
            viewport_has_focus_ = mouse_in_viewport && !imgui_wants_input;
        }
    }

    ImVec2 GuiManager::getViewportPos() const {
        return viewport_pos_;
    }

    ImVec2 GuiManager::getViewportSize() const {
        return viewport_size_;
    }

    bool GuiManager::isMouseInViewport() const {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        return mouse_pos.x >= viewport_pos_.x &&
               mouse_pos.y >= viewport_pos_.y &&
               mouse_pos.x < viewport_pos_.x + viewport_size_.x &&
               mouse_pos.y < viewport_pos_.y + viewport_size_.y;
    }

    bool GuiManager::isViewportFocused() const {
        return viewport_has_focus_;
    }

    bool GuiManager::isPositionInViewport(double x, double y) const {
        const ImGuiViewport* main_viewport = ImGui::GetMainViewport();

        // Convert to window-relative coordinates
        float rel_x = static_cast<float>(x) - main_viewport->WorkPos.x;
        float rel_y = static_cast<float>(y) - main_viewport->WorkPos.y;

        // Check if within viewport bounds
        return (rel_x >= viewport_pos_.x &&
                rel_x < viewport_pos_.x + viewport_size_.x &&
                rel_y >= viewport_pos_.y &&
                rel_y < viewport_pos_.y + viewport_size_.y);
    }

    void GuiManager::renderSpeedOverlay() {
        // Check if overlay should be hidden
        if (speed_overlay_visible_) {
            auto now = std::chrono::steady_clock::now();
            if (now - speed_overlay_start_time_ >= speed_overlay_duration_) {
                speed_overlay_visible_ = false;
            }
        } else {
            return;
        }

        // Get viewport for positioning
        const ImGuiViewport* viewport = ImGui::GetMainViewport();

        // Position overlay in the center-top of the viewport
        const float overlay_width = 300.0f;
        const float overlay_height = 80.0f;
        const float padding = 20.0f;

        ImVec2 overlay_pos(
            viewport->WorkPos.x + (viewport->WorkSize.x - overlay_width) * 0.5f,
            viewport->WorkPos.y + padding);

        // Create overlay window
        ImGui::SetNextWindowPos(overlay_pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(overlay_width, overlay_height), ImGuiCond_Always);

        // Window flags to make it non-interactive and styled nicely
        ImGuiWindowFlags overlay_flags =
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoInputs |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoBringToFrontOnFocus;

        // Apply semi-transparent background
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1.0f, 1.0f, 1.0f, 0.3f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);

        if (ImGui::Begin("##SpeedOverlay", nullptr, overlay_flags)) {
            // Calculate fade effect based on remaining time
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - speed_overlay_start_time_);
            auto remaining = speed_overlay_duration_ - elapsed;

            float fade_alpha = 1.0f;
            if (remaining < std::chrono::milliseconds(500)) {
                // Fade out in the last 500ms
                fade_alpha = static_cast<float>(remaining.count()) / 500.0f;
            }

            // Center the text
            ImVec2 window_size = ImGui::GetWindowSize();

            // Speed text
            std::string speed_text = std::format("WASD Speed: {:.2f}", current_speed_);
            ImVec2 speed_text_size = ImGui::CalcTextSize(speed_text.c_str());
            ImGui::SetCursorPos(ImVec2(
                (window_size.x - speed_text_size.x) * 0.5f,
                window_size.y * 0.3f));

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, fade_alpha));
            ImGui::Text("%s", speed_text.c_str());
            ImGui::PopStyleColor();

            // Max speed text
            std::string max_text = std::format("Max: {:.3f}", max_speed_);
            ImVec2 max_text_size = ImGui::CalcTextSize(max_text.c_str());
            ImGui::SetCursorPos(ImVec2(
                (window_size.x - max_text_size.x) * 0.5f,
                window_size.y * 0.6f));

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.8f, fade_alpha * 0.8f));
            ImGui::Text("%s", max_text.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::End();

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);
    }

    void GuiManager::showSpeedOverlay(float current_speed, float max_speed) {
        current_speed_ = current_speed;
        max_speed_ = max_speed;
        speed_overlay_visible_ = true;
        speed_overlay_start_time_ = std::chrono::steady_clock::now();
    }

    void GuiManager::setupEventHandlers() {
        using namespace events;

        // Handle window visibility
        cmd::ShowWindow::when([this](const auto& e) {
            showWindow(e.window_name, e.show);
        });

        // Handle speed change events
        ui::SpeedChanged::when([this](const auto& e) {
            showSpeedOverlay(e.current_speed, e.max_speed);
        });
    }

    void GuiManager::applyDefaultStyle() {
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
        style.WindowPadding = ImVec2(6.0f, 6.0f);
        style.WindowRounding = 6.0f;
        style.WindowBorderSize = 0.0f;
        style.FrameRounding = 2.0f;

        ImGui::StyleColorsLight();
    }

    void GuiManager::showWindow(const std::string& name, bool show) {
        window_states_[name] = show;
    }

    void GuiManager::toggleWindow(const std::string& name) {
        window_states_[name] = !window_states_[name];
    }

    bool GuiManager::wantsInput() const {
        ImGuiIO& io = ImGui::GetIO();
        return io.WantCaptureMouse || io.WantCaptureKeyboard;
    }

    bool GuiManager::isAnyWindowActive() const {
        return ImGui::IsAnyItemActive() ||
               ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) ||
               ImGui::GetIO().WantCaptureMouse ||
               ImGui::GetIO().WantCaptureKeyboard;
    }

    void GuiManager::setFileSelectedCallback(std::function<void(const std::filesystem::path&, bool)> callback) {
        if (file_browser_) {
            file_browser_->setOnFileSelected(callback);
        }
    }

    void GuiManager::handleProjectChangedDialogCallback(std::function<void(bool)> callback) {
        if (project_changed_dialog_box_) {
            project_changed_dialog_box_->setOnDialogClose(callback);
        }
    }
} // namespace gs::gui