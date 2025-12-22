/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// glad must be included before OpenGL headers
// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/gui_manager.hpp"
#include "command/command_history.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/sogs.hpp"
#include "core/splat_data_export.hpp"
#include "gui/html_viewer_export.hpp"
#include "gui/panels/main_panel.hpp"
#include "gui/panels/scene_panel.hpp"
#include "gui/panels/tools_panel.hpp"
#include "gui/panels/training_panel.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "gui/windows/file_browser.hpp"
#include "io/exporter.hpp"

#include "input/input_controller.hpp"
#include "internal/resource_paths.hpp"
#include "tools/align_tool.hpp"

#include "core/events.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "tools/brush_tool.hpp"
#include "tools/selection_tool.hpp"
#include "training/training_state.hpp"
#include "visualizer_impl.hpp"
#include <cuda_runtime.h>

#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <format>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_internal.h>
#include <ImGuizmo.h>

namespace lfs::vis::gui {

    // Import commonly used types
    using ToolType = lfs::vis::ToolType;
    using ExportFormat = lfs::core::ExportFormat;

    // Gizmo axis/plane visibility threshold (near-zero to always show)
    constexpr float GIZMO_AXIS_LIMIT = 0.0001f;

    // Truncate SH to target degree. shN has (d+1)²-1 coefficients for degree d.
    void truncateSHDegree(lfs::core::SplatData& splat, const int target_degree) {
        if (target_degree >= splat.get_max_sh_degree())
            return;

        if (target_degree == 0) {
            splat.shN() = lfs::core::Tensor{};
        } else {
            const size_t keep_coeffs = static_cast<size_t>((target_degree + 1) * (target_degree + 1) - 1);
            auto& shN = splat.shN();
            if (shN.is_valid() && shN.ndim() >= 2 && shN.shape()[1] > keep_coeffs) {
                if (shN.ndim() == 3) {
                    shN = shN.slice(1, 0, static_cast<int64_t>(keep_coeffs)).contiguous();
                } else {
                    constexpr size_t CHANNELS = 3;
                    shN = shN.slice(1, 0, static_cast<int64_t>(keep_coeffs * CHANNELS)).contiguous();
                }
            }
        }
        splat.set_max_sh_degree(target_degree);
        splat.set_active_sh_degree(target_degree);
    }

    GuiManager::GuiManager(VisualizerImpl* viewer)
        : viewer_(viewer) {

        // Create components
        file_browser_ = std::make_unique<FileBrowser>();
        scene_panel_ = std::make_unique<ScenePanel>(viewer->trainer_manager_);
        menu_bar_ = std::make_unique<MenuBar>();
        export_dialog_ = std::make_unique<ExportDialog>();
        notification_popup_ = std::make_unique<NotificationPopup>();
        save_directory_popup_ = std::make_unique<SaveDirectoryPopup>();
        exit_confirmation_popup_ = std::make_unique<ExitConfirmationPopup>();

        // Initialize window states
        window_states_["file_browser"] = false;
        window_states_["scene_panel"] = true;
        window_states_["system_console"] = false;
        window_states_["training_tab"] = false;
        window_states_["export_dialog"] = false;

        // Initialize speed overlay state
        speed_overlay_visible_ = false;
        speed_overlay_duration_ = std::chrono::milliseconds(3000); // 3 seconds

        // Initialize focus state
        viewport_has_focus_ = false;

        setupEventHandlers();
    }

    GuiManager::~GuiManager() {
        // Cancel and wait for export thread if running
        if (export_state_.active.load()) {
            cancelExport();
            if (export_state_.thread && export_state_.thread->joinable()) {
                export_state_.thread->join();
            }
        }
        export_state_.thread.reset();
    }

    void GuiManager::initMenuBar() {
        menu_bar_->setOnImportDataset([this]() {
            const auto path = OpenDatasetFolderDialogNative();
            if (!path.empty() && std::filesystem::is_directory(path)) {
                save_directory_popup_->show(path);
            }
        });

        menu_bar_->setOnImportPLY([this]() {
            const auto path = OpenPlyFileDialogNative();
            if (!path.empty()) {
                lfs::core::events::cmd::LoadFile{.path = path, .is_dataset = false}.emit();
            }
        });

        menu_bar_->setOnExport([this]() {
            window_states_["export_dialog"] = true;
        });

        // Export dialog: when user clicks Export, show native file dialog and perform export
        export_dialog_->setOnBrowse([this](ExportFormat format,
                                           const std::string& default_name,
                                           const std::vector<std::string>& node_names,
                                           int sh_degree) {
            if (isExporting())
                return;

            // Show native file dialog based on format
            std::filesystem::path path;
            switch (format) {
            case ExportFormat::PLY:
                path = SavePlyFileDialog(default_name);
                break;
            case ExportFormat::SOG:
                path = SaveSogFileDialog(default_name);
                break;
            case ExportFormat::SPZ:
                path = SaveSpzFileDialog(default_name);
                break;
            case ExportFormat::HTML_VIEWER:
                path = SaveHtmlFileDialog(default_name);
                break;
            }

            if (path.empty())
                return;

            // Perform the export
            auto* const scene_manager = viewer_->getSceneManager();
            if (!scene_manager || node_names.empty())
                return;

            // Collect splats with transforms for selected nodes
            const auto& scene = scene_manager->getScene();
            std::vector<std::pair<const lfs::core::SplatData*, glm::mat4>> splats;
            for (const auto& name : node_names) {
                const auto* node = scene.getNode(name);
                if (node && node->type == NodeType::SPLAT && node->model) {
                    splats.emplace_back(node->model.get(), scene.getWorldTransform(node->id));
                }
            }

            if (splats.empty())
                return;

            // Merge selected splats
            auto merged = Scene::mergeSplatsWithTransforms(splats);
            if (!merged)
                return;

            // Truncate SH data if needed
            if (sh_degree < merged->get_max_sh_degree()) {
                truncateSHDegree(*merged, sh_degree);
            }

            // Start async export
            startAsyncExport(format, path, std::move(merged));
        });

        menu_bar_->setOnExit([this]() {
            requestExitConfirmation();
        });

        menu_bar_->setOnNewProject([this]() {
            lfs::core::events::cmd::ClearScene{}.emit();
        });

        menu_bar_->setCanClearCheck([this]() {
            auto* trainer_mgr = viewer_->getTrainerManager();
            if (!trainer_mgr)
                return true;
            return trainer_mgr->canPerform(TrainingAction::ClearScene);
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
            const auto icon_path = lfs::vis::getAssetPath("lichtfeld-icon.png");
            const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(icon_path);

            GLFWimage image{width, height, data};
            glfwSetWindowIcon(viewer_->getWindow(), 1, &image);
            lfs::core::free_image(data);
        } catch (const std::exception& e) {
            LOG_WARN("Could not load application icon: {}", e.what());
        }

        // Apply theme first to get font settings
        applyDefaultStyle();

        // Load fonts
        const auto& t = theme();
        try {
            const auto regular_path = lfs::vis::getAssetPath("fonts/" + t.fonts.regular_path);
            const auto bold_path = lfs::vis::getAssetPath("fonts/" + t.fonts.bold_path);

            constexpr size_t MIN_FONT_FILE_SIZE = 100;
            const auto load_font = [&](const std::filesystem::path& path, const float size) -> ImFont* {
                if (!std::filesystem::exists(path) || std::filesystem::file_size(path) < MIN_FONT_FILE_SIZE) {
                    LOG_WARN("Font file invalid: {}", path.string());
                    return nullptr;
                }
                return io.Fonts->AddFontFromFileTTF(path.string().c_str(), size);
            };

            font_regular_ = load_font(regular_path, t.fonts.base_size * xscale);
            font_bold_ = load_font(bold_path, t.fonts.base_size * xscale);
            font_heading_ = load_font(bold_path, t.fonts.heading_size * xscale);
            font_small_ = load_font(regular_path, t.fonts.small_size * xscale);
            font_section_ = load_font(bold_path, t.fonts.section_size * xscale);

            const bool all_loaded = font_regular_ && font_bold_ && font_heading_ && font_small_ && font_section_;
            if (!all_loaded) {
                LOG_WARN("Some fonts failed to load, using fallback");
                ImFont* const fallback = font_regular_ ? font_regular_ : io.Fonts->AddFontDefault();
                if (!font_regular_)
                    font_regular_ = fallback;
                if (!font_bold_)
                    font_bold_ = fallback;
                if (!font_heading_)
                    font_heading_ = fallback;
                if (!font_small_)
                    font_small_ = fallback;
                if (!font_section_)
                    font_section_ = fallback;
            } else {
                LOG_INFO("Loaded fonts: {} and {}", t.fonts.regular_path, t.fonts.bold_path);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Font loading failed: {}", e.what());
            ImFont* const fallback = io.Fonts->AddFontDefault();
            font_regular_ = font_bold_ = font_heading_ = font_small_ = font_section_ = fallback;
        }

        save_directory_popup_->setOnConfirm([this](const std::filesystem::path& dataset_path,
                                                   const std::filesystem::path& output_path) {
            if (const auto result = services().params().ensureLoaded(); !result) {
                LOG_ERROR("Failed to load parameter files: {}", result.error());
                return;
            }
            services().params().resetToDefaults();
            const auto params = services().params().createForDataset(dataset_path, output_path);
            viewer_->setParameters(params);
            lfs::core::events::cmd::LoadFile{.path = dataset_path, .is_dataset = true}.emit();
        });

        setFileSelectedCallback([this](const std::filesystem::path& path, const bool is_dataset) {
            window_states_["file_browser"] = false;
            if (is_dataset) {
                save_directory_popup_->show(path);
            } else {
                lfs::core::events::cmd::LoadFile{.path = path, .is_dataset = false}.emit();
            }
        });

        scene_panel_->setOnDatasetLoad([this](const std::filesystem::path& path) {
            if (path.empty()) {
                window_states_["file_browser"] = true;
            } else {
                save_directory_popup_->show(path);
            }
        });

        initMenuBar();
        menu_bar_->setFonts({font_regular_, font_bold_, font_heading_, font_small_, font_section_});
        drag_drop_.init(viewer_->getWindow());
        drag_drop_.setFileDropCallback([this](const std::vector<std::string>& paths) {
            if (auto* const ic = viewer_->getInputController())
                ic->handleFileDrop(paths);
        });
    }

    void GuiManager::shutdown() {
        drag_drop_.shutdown();
        panels::ShutdownGizmoToolbar(gizmo_toolbar_state_);

        if (ImGui::GetCurrentContext()) {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }
    }

    void GuiManager::render() {
        drag_drop_.pollEvents();
        drag_drop_hovering_ = drag_drop_.isDragHovering();

        // Start frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();

        // Check mouse state before ImGui::NewFrame() updates WantCaptureMouse
        ImVec2 mouse_pos = ImGui::GetMousePos();
        bool mouse_in_viewport = isPositionInViewport(mouse_pos.x, mouse_pos.y);

        ImGui::NewFrame();

        // Hot-reload themes (check once per second)
        {
            static auto last_check = std::chrono::steady_clock::now();
            const auto now = std::chrono::steady_clock::now();
            if (now - last_check > std::chrono::seconds(1)) {
                checkThemeFileChanges();
                last_check = now;
            }
        }

        // Initialize ImGuizmo for this frame
        ImGuizmo::BeginFrame();

        if (menu_bar_ && !ui_hidden_) {
            // Lazily connect input bindings (input controller may not be ready during init)
            if (!menu_bar_input_bindings_set_) {
                if (auto* input_controller = viewer_->getInputController()) {
                    menu_bar_->setInputBindings(&input_controller->getBindings());
                    menu_bar_input_bindings_set_ = true;
                }
            }
            menu_bar_->render();
        }

        // Override ImGui's mouse capture for gizmo interaction
        // If ImGuizmo is being used or hovered, let it handle the mouse
        if (ImGuizmo::IsOver() || ImGuizmo::IsUsing()) {
            ImGui::GetIO().WantCaptureMouse = false;
            ImGui::GetIO().WantCaptureKeyboard = false;
        }

        // Override ImGui's mouse capture for right/middle buttons when in viewport
        // This ensures that camera controls work properly
        if (mouse_in_viewport && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
                ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
                ImGui::GetIO().WantCaptureMouse = false;
            }
        }

        // In point cloud mode, disable ImGui mouse capture in viewport
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

        ImGui::End();

        // Update editor context state for this frame
        auto& editor_ctx = viewer_->getEditorContext();
        editor_ctx.update(viewer_->getSceneManager(), viewer_->getTrainerManager());

        // Create context for this frame
        UIContext ctx{
            .viewer = viewer_,
            .file_browser = file_browser_.get(),
            .window_states = &window_states_,
            .editor = &editor_ctx,
            .fonts = {font_regular_, font_bold_, font_heading_, font_small_, font_section_}};

        // Right panel
        if (show_main_panel_ && !ui_hidden_) {
            const auto* const vp = ImGui::GetMainViewport();
            const float panel_h = vp->WorkSize.y - STATUS_BAR_HEIGHT;
            const float min_w = vp->WorkSize.x * RIGHT_PANEL_MIN_RATIO;
            const float max_w = vp->WorkSize.x * RIGHT_PANEL_MAX_RATIO;

            // on windows, when window is minimized, WorkSize can be zero
            if (min_w != 0 || max_w != 0)
                right_panel_width_ = std::clamp(right_panel_width_, min_w, max_w);

            const float panel_x = vp->WorkPos.x + vp->WorkSize.x - right_panel_width_;
            ImGui::SetNextWindowPos({panel_x, vp->WorkPos.y}, ImGuiCond_Always);
            ImGui::SetNextWindowSize({right_panel_width_, panel_h}, ImGuiCond_Always);

            constexpr ImGuiWindowFlags PANEL_FLAGS =
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDocking |
                ImGuiWindowFlags_NoTitleBar;

            const auto& t = theme();
            ImGui::PushStyleColor(ImGuiCol_WindowBg, withAlpha(t.palette.surface, 0.95f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {8.0f, 8.0f});

            // Left edge resize handle
            constexpr float EDGE_GRAB_W = 8.0f;
            const auto& io = ImGui::GetIO();
            hovering_panel_edge_ = io.MousePos.x >= panel_x - EDGE_GRAB_W &&
                                   io.MousePos.x <= panel_x + EDGE_GRAB_W &&
                                   io.MousePos.y >= vp->WorkPos.y &&
                                   io.MousePos.y <= vp->WorkPos.y + panel_h;

            if (hovering_panel_edge_ && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                resizing_panel_ = true;
            if (resizing_panel_ && !ImGui::IsMouseDown(ImGuiMouseButton_Left))
                resizing_panel_ = false;
            if (resizing_panel_) {
                right_panel_width_ = std::clamp(right_panel_width_ - io.MouseDelta.x, min_w, max_w);
                updateViewportRegion();
            }
            if (hovering_panel_edge_ || resizing_panel_)
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);

            if (ImGui::Begin("##RightPanel", nullptr, PANEL_FLAGS)) {
                const float avail_h = ImGui::GetContentRegionAvail().y;
                constexpr float SPLITTER_H = 6.0f, MIN_H = 80.0f;

                // Scene panel
                const float scene_h = std::max(MIN_H, avail_h * scene_panel_ratio_ - SPLITTER_H * 0.5f);
                ImGui::PushStyleColor(ImGuiCol_ChildBg, {0, 0, 0, 0});
                if (ImGui::BeginChild("##ScenePanel", {0, scene_h}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
                    widgets::SectionHeader("SCENE", ctx.fonts);
                    scene_panel_->renderContent(&ctx);
                }
                ImGui::EndChild();
                ImGui::PopStyleColor();

                // Splitter
                ImGui::PushStyleColor(ImGuiCol_Button, withAlpha(t.palette.border, 0.4f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, withAlpha(t.palette.info, 0.6f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, withAlpha(t.palette.info, 0.8f));
                ImGui::Button("##Splitter", {-1, SPLITTER_H});
                if (ImGui::IsItemActive())
                    scene_panel_ratio_ = std::clamp(scene_panel_ratio_ + ImGui::GetIO().MouseDelta.y / avail_h, 0.15f, 0.85f);
                if (ImGui::IsItemHovered())
                    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
                ImGui::PopStyleColor(3);

                // Rendering content helper
                const auto draw_rendering = [&ctx] {
                    panels::DrawRenderingSettings(ctx);
                    ImGui::Separator();
                    panels::DrawSelectionGroups(ctx);
                    ImGui::Separator();
                    panels::DrawToolsPanel(ctx);
                    panels::DrawSystemConsoleButton(ctx);
                };

                // Rendering panel
                ImGui::PushStyleColor(ImGuiCol_ChildBg, {0, 0, 0, 0});
                if (viewer_->getTrainer()) {
                    if (ImGui::BeginTabBar("##BottomTabs")) {
                        if (ImGui::BeginTabItem("Rendering")) {
                            if (ImGui::BeginChild("##RenderingPanel", {0, 0}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
                                draw_rendering();
                            }
                            ImGui::EndChild();
                            ImGui::EndTabItem();
                        }
                        const ImGuiTabItemFlags flags = focus_training_panel_
                                                            ? ImGuiTabItemFlags_SetSelected
                                                            : ImGuiTabItemFlags_None;
                        if (focus_training_panel_)
                            focus_training_panel_ = false;
                        if (ImGui::BeginTabItem("Training", nullptr, flags)) {
                            panels::DrawTrainingControls(ctx);
                            if (ImGui::BeginChild("##TrainingPanel", {0, 0}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
                                panels::DrawTrainingParams(ctx);
                                panels::DrawTrainingStatus(ctx);
                                ImGui::Separator();
                                panels::DrawProgressInfo(ctx);
                            }
                            ImGui::EndChild();
                            ImGui::EndTabItem();
                        }
                        ImGui::EndTabBar();
                    }
                } else {
                    widgets::SectionHeader("RENDERING", ctx.fonts);
                    if (ImGui::BeginChild("##RenderingPanel", {0, 0}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
                        draw_rendering();
                    }
                    ImGui::EndChild();
                }
                ImGui::PopStyleColor();
            }
            ImGui::End();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor();
        } else {
            hovering_panel_edge_ = false;
            resizing_panel_ = false;
        }

        // Render floating windows
        if (window_states_["file_browser"]) {
            file_browser_->render(&window_states_["file_browser"]);
        }

        // Export dialog
        if (window_states_["export_dialog"]) {
            export_dialog_->render(&window_states_["export_dialog"], viewer_->getSceneManager());
        }

        // Utility toolbar (always visible)
        const bool is_fullscreen = viewer_->getWindowManager() && viewer_->getWindowManager()->isFullscreen();
        panels::DrawUtilityToolbar(gizmo_toolbar_state_, viewport_pos_, viewport_size_, ui_hidden_, is_fullscreen,
                                   viewer_->getRenderingManager(), &viewer_->getViewport());

        // Gizmo toolbar (only when node selected and UI visible)
        auto* const scene_manager = ctx.viewer->getSceneManager();
        if (scene_manager && scene_manager->hasSelectedNode() && !ui_hidden_) {
            panels::DrawGizmoToolbar(ctx, gizmo_toolbar_state_, viewport_pos_, viewport_size_);

            // Get current tool from EditorContext (single source of truth)
            const auto current_tool = ctx.editor->getActiveTool();
            const bool is_transform_tool = (current_tool == ToolType::Translate ||
                                            current_tool == ToolType::Rotate ||
                                            current_tool == ToolType::Scale);
            show_node_gizmo_ = is_transform_tool;
            if (is_transform_tool) {
                node_gizmo_operation_ = gizmo_toolbar_state_.current_operation;
            }

            auto* brush_tool = ctx.viewer->getBrushTool();
            auto* align_tool = ctx.viewer->getAlignTool();
            auto* selection_tool = ctx.viewer->getSelectionTool();
            const bool is_brush_mode = (current_tool == ToolType::Brush);
            const bool is_align_mode = (current_tool == ToolType::Align);
            const bool is_selection_mode = (current_tool == ToolType::Selection);
            const bool is_cropbox_mode = (current_tool == ToolType::CropBox);

            // Materialize deletions when switching away from selection or cropbox tools
            const bool was_selection_mode = (previous_tool_ == ToolType::Selection);
            const bool was_cropbox_mode = (previous_tool_ == ToolType::CropBox);
            if ((was_selection_mode || was_cropbox_mode) && current_tool != previous_tool_) {
                if (auto* sm = ctx.viewer->getSceneManager()) {
                    sm->applyDeleted();
                }
            }

            // Auto-create and select cropbox when switching to CropBox tool
            if (is_cropbox_mode && !was_cropbox_mode) {
                if (auto* sm = ctx.viewer->getSceneManager()) {
                    sm->ensureCropBoxForSelectedNode();
                    // Auto-select the cropbox for the current node (or first POINTCLOUD)
                    sm->selectCropBoxForCurrentNode();
                }
            }

            previous_tool_ = current_tool;

            if (brush_tool)
                brush_tool->setEnabled(is_brush_mode);
            if (align_tool)
                align_tool->setEnabled(is_align_mode);
            if (selection_tool)
                selection_tool->setEnabled(is_selection_mode);

            // Update selection mode and auto-toggle ring rendering
            if (is_selection_mode) {
                if (auto* rm = ctx.viewer->getRenderingManager()) {
                    lfs::rendering::SelectionMode mode = lfs::rendering::SelectionMode::Centers;
                    switch (gizmo_toolbar_state_.selection_mode) {
                    case panels::SelectionSubMode::Centers: mode = lfs::rendering::SelectionMode::Centers; break;
                    case panels::SelectionSubMode::Rectangle: mode = lfs::rendering::SelectionMode::Rectangle; break;
                    case panels::SelectionSubMode::Polygon: mode = lfs::rendering::SelectionMode::Polygon; break;
                    case panels::SelectionSubMode::Lasso: mode = lfs::rendering::SelectionMode::Lasso; break;
                    case panels::SelectionSubMode::Rings: mode = lfs::rendering::SelectionMode::Rings; break;
                    }
                    rm->setSelectionMode(mode);

                    // Reset selection state when switching modes
                    if (gizmo_toolbar_state_.selection_mode != previous_selection_mode_) {
                        if (selection_tool)
                            selection_tool->onSelectionModeChanged();

                        // Auto-enable rings when switching to Rings sub-mode
                        if (gizmo_toolbar_state_.selection_mode == panels::SelectionSubMode::Rings) {
                            auto settings = rm->getSettings();
                            settings.show_rings = true;
                            settings.show_center_markers = false;
                            rm->updateSettings(settings);
                        }

                        previous_selection_mode_ = gizmo_toolbar_state_.selection_mode;
                    }
                }
            }

            if (is_cropbox_mode) {
                switch (gizmo_toolbar_state_.cropbox_operation) {
                case panels::CropBoxOperation::Bounds: crop_gizmo_operation_ = ImGuizmo::BOUNDS; break;
                case panels::CropBoxOperation::Translate: crop_gizmo_operation_ = ImGuizmo::TRANSLATE; break;
                case panels::CropBoxOperation::Rotate: crop_gizmo_operation_ = ImGuizmo::ROTATE; break;
                case panels::CropBoxOperation::Scale: crop_gizmo_operation_ = ImGuizmo::SCALE; break;
                }
            }

            // Toggle cropbox visibility with tool
            if (auto* render_manager = ctx.viewer->getRenderingManager()) {
                auto settings = render_manager->getSettings();
                if (is_cropbox_mode != settings.show_crop_box) {
                    settings.show_crop_box = is_cropbox_mode;
                    render_manager->updateSettings(settings);
                }
            }

            // Handle reset cropbox request from toolbar
            if (gizmo_toolbar_state_.reset_cropbox_requested) {
                gizmo_toolbar_state_.reset_cropbox_requested = false;
                auto* const scene_manager = ctx.viewer->getSceneManager();
                auto* const render_manager = ctx.viewer->getRenderingManager();
                if (scene_manager && render_manager) {
                    const NodeId cropbox_id = scene_manager->getSelectedNodeCropBoxId();
                    if (cropbox_id != NULL_NODE) {
                        auto* node = scene_manager->getScene().getMutableNode(
                            scene_manager->getScene().getNodeById(cropbox_id)->name);
                        if (node && node->cropbox) {
                            // Capture state before reset
                            const command::CropBoxState old_state{
                                .min = node->cropbox->min,
                                .max = node->cropbox->max,
                                .local_transform = node->local_transform.get(),
                                .inverse = node->cropbox->inverse};

                            // Apply reset to scene graph
                            node->cropbox->min = glm::vec3(-1.0f);
                            node->cropbox->max = glm::vec3(1.0f);
                            node->cropbox->inverse = false;
                            node->local_transform = glm::mat4(1.0f);
                            node->transform_dirty = true;
                            scene_manager->getScene().invalidateCache();

                            // Capture state after reset
                            const command::CropBoxState new_state{
                                .min = node->cropbox->min,
                                .max = node->cropbox->max,
                                .local_transform = node->local_transform.get(),
                                .inverse = node->cropbox->inverse};

                            // Create undo command
                            auto cmd = std::make_unique<command::CropBoxCommand>(
                                scene_manager, node->name, old_state, new_state);
                            viewer_->getCommandHistory().execute(std::move(cmd));

                            auto settings = render_manager->getSettings();
                            settings.use_crop_box = false;
                            render_manager->updateSettings(settings);
                        }
                    }
                }
            }
        } else {
            show_node_gizmo_ = false;
            auto* brush_tool = ctx.viewer->getBrushTool();
            auto* align_tool = ctx.viewer->getAlignTool();
            if (brush_tool)
                brush_tool->setEnabled(false);
            if (align_tool)
                align_tool->setEnabled(false);
        }

        auto* brush_tool = ctx.viewer->getBrushTool();
        if (brush_tool && brush_tool->isEnabled() && !ui_hidden_) {
            brush_tool->renderUI(ctx, nullptr);
        }

        auto* selection_tool = ctx.viewer->getSelectionTool();
        if (selection_tool && selection_tool->isEnabled() && !ui_hidden_) {
            selection_tool->renderUI(ctx, nullptr);
        }

        auto* align_tool = ctx.viewer->getAlignTool();
        if (align_tool && align_tool->isEnabled() && !ui_hidden_) {
            align_tool->renderUI(ctx, nullptr);
        }

        // Node selection rectangle
        if (auto* const ic = ctx.viewer->getInputController();
            !ui_hidden_ && ic && ic->isNodeRectDragging()) {
            const auto start = ic->getNodeRectStart();
            const auto end = ic->getNodeRectEnd();
            const auto& t = theme();
            auto* const draw_list = ImGui::GetForegroundDrawList();
            draw_list->AddRectFilled({start.x, start.y}, {end.x, end.y}, toU32WithAlpha(t.palette.warning, 0.15f));
            draw_list->AddRect({start.x, start.y}, {end.x, end.y}, toU32WithAlpha(t.palette.warning, 0.85f), 0.0f, 0, 2.0f);
        }

        // Render crop box gizmo over viewport
        renderCropBoxGizmo(ctx);

        // Render node transform gizmo (for translating selected PLY nodes)
        renderNodeTransformGizmo(ctx);

        updateCropFlash();

        // Get the viewport region for 3D rendering
        updateViewportRegion();

        // Update viewport focus based on mouse position
        updateViewportFocus();

        // Draw vignette effect on viewport
        if (viewport_size_.x > 0 && viewport_size_.y > 0) {
            widgets::DrawViewportVignette(viewport_pos_, viewport_size_);
        }

        // Mask viewport corners with background for rounded effect
        if (!ui_hidden_ && viewport_size_.x > 0 && viewport_size_.y > 0) {
            const auto& t = theme();
            const float r = t.viewport.corner_radius;
            if (r > 0.0f) {
                auto* const dl = ImGui::GetForegroundDrawList();
                const ImU32 bg = toU32(t.palette.background);
                const float x1 = viewport_pos_.x, y1 = viewport_pos_.y;
                const float x2 = x1 + viewport_size_.x, y2 = y1 + viewport_size_.y;

                // Draw corner wedge: corner -> edge -> arc -> corner
                constexpr int CORNER_ARC_SEGMENTS = 12;
                const auto maskCorner = [&](const ImVec2 corner, const ImVec2 edge,
                                            const ImVec2 center, const float a0, const float a1) {
                    dl->PathLineTo(corner);
                    dl->PathLineTo(edge);
                    dl->PathArcTo(center, r, a0, a1, CORNER_ARC_SEGMENTS);
                    dl->PathLineTo(corner);
                    dl->PathFillConvex(bg);
                };
                maskCorner({x1, y1}, {x1, y1 + r}, {x1 + r, y1 + r}, IM_PI, IM_PI * 1.5f);
                maskCorner({x2, y1}, {x2 - r, y1}, {x2 - r, y1 + r}, IM_PI * 1.5f, IM_PI * 2.0f);
                maskCorner({x1, y2}, {x1 + r, y2}, {x1 + r, y2 - r}, IM_PI * 0.5f, IM_PI);
                maskCorner({x2, y2}, {x2, y2 - r}, {x2 - r, y2 - r}, 0.0f, IM_PI * 0.5f);

                if (t.viewport.border_size > 0.0f) {
                    dl->AddRect({x1, y1}, {x2, y2}, t.viewport_border_u32(), r,
                                ImDrawFlags_RoundCornersAll, t.viewport.border_size);
                }
            }
        }

        // Render status bar at bottom of viewport
        if (!ui_hidden_) {
            renderStatusBar(ctx);
        }

        // Render viewport gizmo and handle drag-to-orbit
        if (show_viewport_gizmo_ && !ui_hidden_ && viewport_size_.x > 0 && viewport_size_.y > 0) {
            if (rendering_manager) {
                if (auto* const engine = rendering_manager->getRenderingEngine()) {
                    auto& viewport = viewer_->getViewport();
                    const glm::vec2 vp_pos(viewport_pos_.x, viewport_pos_.y);
                    const glm::vec2 vp_size(viewport_size_.x, viewport_size_.y);

                    // Gizmo bounds (lower-right corner)
                    const float gizmo_x = vp_pos.x + vp_size.x - VIEWPORT_GIZMO_SIZE - VIEWPORT_GIZMO_MARGIN_X;
                    const float gizmo_y = vp_pos.y + vp_size.y - VIEWPORT_GIZMO_SIZE - VIEWPORT_GIZMO_MARGIN_Y;

                    const ImVec2 mouse = ImGui::GetMousePos();
                    const bool mouse_in_gizmo = mouse.x >= gizmo_x && mouse.x <= gizmo_x + VIEWPORT_GIZMO_SIZE &&
                                                mouse.y >= gizmo_y && mouse.y <= gizmo_y + VIEWPORT_GIZMO_SIZE;

                    const int hovered_axis = engine->hitTestViewportGizmo(glm::vec2(mouse.x, mouse.y), vp_pos, vp_size);
                    engine->setViewportGizmoHover(hovered_axis);

                    if (!ImGui::GetIO().WantCaptureMouse) {
                        const glm::vec2 mouse_pos(mouse.x, mouse.y);
                        const float time = static_cast<float>(ImGui::GetTime());

                        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mouse_in_gizmo) {
                            if (hovered_axis >= 0 && hovered_axis <= 5) {
                                // Snap to axis view
                                const int axis = hovered_axis % 3;
                                const bool negative = hovered_axis >= 3;
                                const glm::mat3 rotation = engine->getAxisViewRotation(axis, negative);
                                const float dist = glm::length(viewport.camera.pivot - viewport.camera.t);

                                viewport.camera.pivot = glm::vec3(0.0f);
                                viewport.camera.R = rotation;
                                viewport.camera.t = -rotation[2] * dist;

                                const auto& settings = rendering_manager->getSettings();
                                lfs::core::events::ui::GridSettingsChanged{
                                    .enabled = settings.show_grid,
                                    .plane = axis,
                                    .opacity = settings.grid_opacity}
                                    .emit();

                                rendering_manager->markDirty();
                            } else {
                                // Drag to orbit
                                viewport_gizmo_dragging_ = true;
                                viewport.camera.startRotateAroundCenter(mouse_pos, time);
                                if (GLFWwindow* const window = glfwGetCurrentContext()) {
                                    glfwGetCursorPos(window, &gizmo_drag_start_cursor_.x, &gizmo_drag_start_cursor_.y);
                                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                                }
                            }
                        }

                        if (viewport_gizmo_dragging_) {
                            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                                viewport.camera.updateRotateAroundCenter(mouse_pos, time);
                                rendering_manager->markDirty();
                            } else {
                                viewport.camera.endRotateAroundCenter();
                                viewport_gizmo_dragging_ = false;

                                // Release cursor, restore position
                                if (GLFWwindow* const window = glfwGetCurrentContext()) {
                                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                                    glfwSetCursorPos(window, gizmo_drag_start_cursor_.x, gizmo_drag_start_cursor_.y);
                                }
                            }
                        }
                    }

                    engine->renderViewportGizmo(viewport.getRotationMatrix(), vp_pos, vp_size);

                    // Drag feedback overlay
                    if (viewport_gizmo_dragging_) {
                        const float center_x = gizmo_x + VIEWPORT_GIZMO_SIZE * 0.5f;
                        const float center_y = gizmo_y + VIEWPORT_GIZMO_SIZE * 0.5f;
                        constexpr float OVERLAY_RADIUS = VIEWPORT_GIZMO_SIZE * 0.46f; // Match gizmo content + 2px
                        ImGui::GetBackgroundDrawList()->AddCircleFilled(
                            ImVec2(center_x, center_y), OVERLAY_RADIUS,
                            IM_COL32(180, 180, 180, 51), 32);
                    }
                }
            }
        }

        // Render export progress overlay (blocking overlay on top of everything)
        renderExportOverlay();

        // Render empty state welcome screen when no content loaded
        renderEmptyStateOverlay();

        // Render drag-drop overlay when files are being dragged over the window
        renderDragDropOverlay();

        if (save_directory_popup_) {
            save_directory_popup_->render(viewport_pos_, viewport_size_);
        }

        if (notification_popup_)
            notification_popup_->render();
        if (exit_confirmation_popup_)
            exit_confirmation_popup_->render();

        // End frame
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Clean up GL state after ImGui rendering (ImGui can leave VAO/shader bindings corrupted)
        glBindVertexArray(0);
        glUseProgram(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        // Clear any errors ImGui might have generated
        while (glGetError() != GL_NO_ERROR) {}

        // Update and Render additional Platform Windows (for multi-viewport)
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);

            // Clean up GL state after multi-viewport rendering too
            glBindVertexArray(0);
            glUseProgram(0);
            glBindTexture(GL_TEXTURE_2D, 0);
            while (glGetError() != GL_NO_ERROR) {}
        }
    }

    void GuiManager::updateViewportRegion() {
        constexpr float PANEL_GAP = 2.0f;
        const auto* const vp = ImGui::GetMainViewport();
        const float w = (show_main_panel_ && !ui_hidden_)
                            ? vp->WorkSize.x - right_panel_width_ - PANEL_GAP
                            : vp->WorkSize.x;
        const float h = ui_hidden_ ? vp->WorkSize.y : vp->WorkSize.y - STATUS_BAR_HEIGHT;
        viewport_pos_ = {vp->WorkPos.x, vp->WorkPos.y};
        viewport_size_ = {w, h};
    }

    void GuiManager::updateViewportFocus() {
        // Viewport has focus unless actively using a GUI widget
        viewport_has_focus_ = !ImGui::IsAnyItemActive();
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

    bool GuiManager::isPositionInViewportGizmo(const double x, const double y) const {
        if (!show_viewport_gizmo_ || ui_hidden_)
            return false;

        const float gizmo_x = viewport_pos_.x + viewport_size_.x - VIEWPORT_GIZMO_SIZE - VIEWPORT_GIZMO_MARGIN_X;
        const float gizmo_y = viewport_pos_.y + viewport_size_.y - VIEWPORT_GIZMO_SIZE - VIEWPORT_GIZMO_MARGIN_Y;

        return x >= gizmo_x && x <= gizmo_x + VIEWPORT_GIZMO_SIZE &&
               y >= gizmo_y && y <= gizmo_y + VIEWPORT_GIZMO_SIZE;
    }

    void GuiManager::renderStatusBar(const UIContext& ctx) {
        auto* const rm = ctx.viewer->getRenderingManager();
        auto* const sm = ctx.viewer->getSceneManager();
        if (!rm)
            return;

        constexpr float PADDING = 8.0f, SPACING = 20.0f, FADE_DURATION_MS = 500.0f;
        const auto& t = theme();
        const auto* const vp = ImGui::GetMainViewport();
        const ImVec2 pos{vp->WorkPos.x, vp->WorkPos.y + vp->WorkSize.y - STATUS_BAR_HEIGHT};
        const ImVec2 size{vp->WorkSize.x, STATUS_BAR_HEIGHT};
        const auto now = std::chrono::steady_clock::now();

        // Fade alpha helper
        const auto fade_alpha = [&](auto start_time) {
            const auto remaining = speed_overlay_duration_ -
                                   std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
            return (remaining.count() < FADE_DURATION_MS) ? remaining.count() / FADE_DURATION_MS : 1.0f;
        };

        // Update overlay timers
        if (speed_overlay_visible_ && now - speed_overlay_start_time_ >= speed_overlay_duration_)
            speed_overlay_visible_ = false;
        if (zoom_speed_overlay_visible_ && now - zoom_speed_overlay_start_time_ >= speed_overlay_duration_)
            zoom_speed_overlay_visible_ = false;

        ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(size, ImGuiCond_Always);

        constexpr ImGuiWindowFlags FLAGS =
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoFocusOnAppearing;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, withAlpha(t.palette.background, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_Border, withAlpha(t.palette.border, 0.6f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {PADDING, 3.0f});
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {6.0f, 0.0f});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);

        if (ImGui::Begin("##StatusBar", nullptr, FLAGS)) {
            // Use regular font as base for status bar
            if (ctx.fonts.regular)
                ImGui::PushFont(ctx.fonts.regular);

            ImGui::GetWindowDrawList()->AddLine(pos, {pos.x + size.x, pos.y},
                                                toU32(withAlpha(t.palette.surface_bright, 0.4f)), 1.0f);

            // Mode (bold)
            const char* mode = "Empty";
            ImVec4 color = t.palette.text_dim;
            TrainerManager* trainer_mgr = nullptr;
            bool show_training_progress = false;
            char mode_buf[64] = {};

            if (sm) {
                switch (sm->getContentType()) {
                case SceneManager::ContentType::SplatFiles:
                    mode = "Viewer";
                    color = t.palette.info;
                    break;
                case SceneManager::ContentType::Dataset:
                    trainer_mgr = sm->getTrainerManager();
                    if (trainer_mgr && trainer_mgr->hasTrainer()) {
                        const auto state = trainer_mgr->getState();
                        const int current_iter = trainer_mgr->getCurrentIteration();
                        const char* base_mode = "Dataset";
                        switch (state) {
                        case TrainerManager::State::Running:
                            base_mode = "Training";
                            color = t.palette.warning;
                            show_training_progress = true;
                            break;
                        case TrainerManager::State::Paused:
                            base_mode = "Paused";
                            color = t.palette.text_dim;
                            show_training_progress = true;
                            break;
                        case TrainerManager::State::Ready:
                            base_mode = (current_iter > 0) ? "Resume" : "Ready";
                            color = t.palette.success;
                            break;
                        case TrainerManager::State::Finished: {
                            const auto reason = trainer_mgr->getStateMachine().getFinishReason();
                            if (reason == FinishReason::Completed) {
                                base_mode = "Complete";
                                color = t.palette.success;
                                show_training_progress = true;
                            } else {
                                base_mode = "Error";
                                color = t.palette.error;
                            }
                            break;
                        }
                        default: break;
                        }
                        // Add strategy and method info
                        const char* strategy = trainer_mgr->getStrategyType();
                        const bool gut = trainer_mgr->isGutEnabled();
                        snprintf(mode_buf, sizeof(mode_buf), "%s (%s/%s)",
                                 base_mode,
                                 strcmp(strategy, "mcmc") == 0 ? "MCMC" : "Default",
                                 gut ? "GUT" : "3DGS");
                        mode = mode_buf;
                    } else {
                        mode = "Dataset";
                    }
                    break;
                default: break;
                }
            }
            if (ctx.fonts.bold)
                ImGui::PushFont(ctx.fonts.bold);
            ImGui::TextColored(color, "%s", mode);
            if (ctx.fonts.bold)
                ImGui::PopFont();

            // Training progress display
            if (show_training_progress && trainer_mgr) {
                constexpr float SECTION_GAP = 12.0f;
                constexpr float LABEL_GAP = 6.0f;
                constexpr float SEP_GAP = 20.0f;
                constexpr float BAR_WIDTH = 120.0f;
                constexpr float BAR_HEIGHT = 12.0f;
                constexpr float BAR_ROUNDING = 2.0f;

                const int current_iter = trainer_mgr->getCurrentIteration();
                const int total_iter = trainer_mgr->getTotalIterations();
                const float progress = total_iter > 0
                                           ? static_cast<float>(current_iter) / static_cast<float>(total_iter)
                                           : 0.0f;

                ImDrawList* const draw_list = ImGui::GetWindowDrawList();
                ImFont* const font = ctx.fonts.bold ? ctx.fonts.bold : ImGui::GetFont();
                const float font_size = font->FontSize;
                const float y_text = pos.y + (STATUS_BAR_HEIGHT - font_size) * 0.5f;
                const float y_bar = pos.y + (STATUS_BAR_HEIGHT - BAR_HEIGHT) * 0.5f;
                const ImU32 col_text = toU32(t.palette.text);
                const ImU32 col_dim = toU32(t.palette.text_dim);
                const ImU32 col_bar_bg = toU32(withAlpha(t.palette.surface_bright, 0.5f));
                const ImU32 col_bar_fill = toU32(t.palette.primary);

                // Helper: draw label and advance cursor
                const auto draw_label = [&](float& x, const char* label) {
                    draw_list->AddText(font, font_size, {x, y_text}, col_dim, label);
                    x += font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, label).x + LABEL_GAP;
                };

                // Helper: draw value and advance cursor
                const auto draw_value = [&](float& x, const char* value) {
                    draw_list->AddText(font, font_size, {x, y_text}, col_text, value);
                    x += font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, value).x;
                };

                // Helper: draw separator and advance cursor
                const auto draw_sep = [&](float& x) {
                    draw_list->AddText(font, font_size, {x, y_text}, col_dim, "|");
                    x += SEP_GAP;
                };

                // Helper: format count with K/M suffix
                const auto fmt_count = [](const int n, char* buf, const size_t size) {
                    if (n >= 1'000'000)
                        snprintf(buf, size, "%.2fM", n / 1e6);
                    else if (n >= 1'000)
                        snprintf(buf, size, "%.0fK", n / 1e3);
                    else
                        snprintf(buf, size, "%d", n);
                };

                // Helper: format time as h:mm:ss or m:ss
                const auto fmt_time = [](const float secs, char* buf, const size_t size) {
                    if (secs < 0.0f) {
                        snprintf(buf, size, "--:--");
                        return;
                    }
                    const int total = static_cast<int>(secs);
                    const int h = total / 3600, m = (total % 3600) / 60, s = total % 60;
                    if (h > 0)
                        snprintf(buf, size, "%d:%02d:%02d", h, m, s);
                    else
                        snprintf(buf, size, "%d:%02d", m, s);
                };

                ImGui::SameLine(0.0f, SPACING);
                ImGui::TextColored(t.palette.text_dim, "|");
                ImGui::SameLine(0.0f, SPACING);

                float x = ImGui::GetCursorScreenPos().x;

                // Progress bar
                draw_list->AddRectFilled({x, y_bar}, {x + BAR_WIDTH, y_bar + BAR_HEIGHT},
                                         col_bar_bg, BAR_ROUNDING);
                if (progress > 0.0f) {
                    draw_list->AddRectFilled({x, y_bar}, {x + BAR_WIDTH * progress, y_bar + BAR_HEIGHT},
                                             col_bar_fill, BAR_ROUNDING);
                }
                char pct_buf[8];
                snprintf(pct_buf, sizeof(pct_buf), "%.0f%%", progress * 100.0f);
                const ImVec2 pct_size = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, pct_buf);
                draw_list->AddText(font, font_size,
                                   {x + (BAR_WIDTH - pct_size.x) * 0.5f, y_bar + (BAR_HEIGHT - pct_size.y) * 0.5f},
                                   col_text, pct_buf);
                x += BAR_WIDTH + SECTION_GAP;

                // Step
                char iter_buf[32];
                snprintf(iter_buf, sizeof(iter_buf), "%d/%d", current_iter, total_iter);
                draw_label(x, "Step");
                draw_value(x, iter_buf);
                x += SECTION_GAP;
                draw_sep(x);

                // Loss
                char loss_buf[16];
                snprintf(loss_buf, sizeof(loss_buf), "%.4f", trainer_mgr->getCurrentLoss());
                draw_label(x, "Loss");
                draw_value(x, loss_buf);
                x += SECTION_GAP;
                draw_sep(x);

                // Gaussians
                char cur_buf[16], max_buf[16], gauss_buf[32];
                fmt_count(trainer_mgr->getNumSplats(), cur_buf, sizeof(cur_buf));
                fmt_count(trainer_mgr->getMaxGaussians(), max_buf, sizeof(max_buf));
                snprintf(gauss_buf, sizeof(gauss_buf), "%s/%s", cur_buf, max_buf);
                draw_label(x, "Gaussians");
                draw_value(x, gauss_buf);
                x += SECTION_GAP;
                draw_sep(x);

                // Time
                char elapsed_buf[16], eta_buf[16];
                fmt_time(trainer_mgr->getElapsedSeconds(), elapsed_buf, sizeof(elapsed_buf));
                fmt_time(trainer_mgr->getEstimatedRemainingSeconds(), eta_buf, sizeof(eta_buf));
                draw_value(x, elapsed_buf);
                x += LABEL_GAP;
                draw_label(x, "ETA");
                draw_value(x, eta_buf);

                ImGui::SameLine();
                ImGui::Dummy({x - ImGui::GetCursorScreenPos().x + SECTION_GAP, STATUS_BAR_HEIGHT});
            }

            // Splat count (visible / total) - skip during training
            if (sm && !show_training_progress) {
                size_t total = 0, visible = 0;
                for (const auto* n : sm->getScene().getNodes()) {
                    if (n->type == NodeType::SPLAT) {
                        total += n->gaussian_count;
                        if (n->visible.get())
                            visible += n->gaussian_count;
                    }
                }
                if (total > 0) {
                    ImGui::SameLine(0.0f, SPACING);
                    ImGui::TextColored(t.palette.text_dim, "|");
                    ImGui::SameLine();
                    const auto fmt = [](const size_t n) -> std::string {
                        if (n >= 1'000'000)
                            return std::format("{:.2f}M", n / 1e6);
                        if (n >= 1'000)
                            return std::format("{:.1f}K", n / 1e3);
                        return std::to_string(n);
                    };
                    if (ctx.fonts.bold)
                        ImGui::PushFont(ctx.fonts.bold);
                    if (visible == total)
                        ImGui::Text("%s Gaussians", fmt(total).c_str());
                    else
                        ImGui::Text("%s / %s Gaussians", fmt(visible).c_str(), fmt(total).c_str());
                    if (ctx.fonts.bold)
                        ImGui::PopFont();
                }
            }

            // Split view / GT comparison
            if (const auto info = rm->getSplitViewInfo(); info.enabled) {
                ImGui::SameLine(0.0f, SPACING);
                ImGui::TextColored(t.palette.text_dim, "|");
                ImGui::SameLine();
                if (rm->getSettings().split_view_mode == SplitViewMode::GTComparison) {
                    const int cam_id = rm->getCurrentCameraId();
                    auto* trainer_mgr = sm ? sm->getTrainerManager() : nullptr;
                    auto cam = trainer_mgr ? trainer_mgr->getCamById(cam_id) : nullptr;

                    if (cam) {
                        // Image filename and extension
                        const auto& path = cam->image_path();
                        const std::string filename = path.filename().string();
                        std::string ext = path.extension().string();
                        if (!ext.empty() && ext[0] == '.')
                            ext = ext.substr(1);
                        for (auto& c : ext)
                            c = static_cast<char>(std::toupper(c));

                        // Infer channels from extension
                        const char* channels = "RGB";
                        if (ext == "PNG" || ext == "WEBP" || ext == "TIFF" || ext == "TIF" || ext == "EXR")
                            channels = "RGBA";

                        ImGui::TextColored(t.palette.info, "%s", filename.c_str());
                        ImGui::SameLine(0.0f, SPACING);
                        ImGui::TextColored(t.palette.text_dim, "|");
                        ImGui::SameLine(0.0f, SPACING);

                        // Dimensions and channels
                        ImGui::TextColored(t.palette.text_dim, "%dx%d %s",
                                           cam->image_width(), cam->image_height(), channels);
                        ImGui::SameLine(0.0f, SPACING);

                        // Format
                        ImGui::TextColored(t.palette.text_dim, "%s", ext.c_str());

                        // Mask indicator
                        if (cam->has_mask()) {
                            ImGui::SameLine(0.0f, SPACING);
                            ImGui::TextColored(t.palette.success, "MASK");
                        }

                        // Camera model (always show)
                        ImGui::SameLine(0.0f, SPACING);
                        const auto model_type = cam->camera_model_type();
                        const char* model = "PINHOLE";
                        switch (model_type) {
                        case lfs::core::CameraModelType::FISHEYE: model = "FISHEYE"; break;
                        case lfs::core::CameraModelType::ORTHO: model = "ORTHO"; break;
                        case lfs::core::CameraModelType::EQUIRECTANGULAR: model = "EQUIRECT"; break;
                        case lfs::core::CameraModelType::THIN_PRISM_FISHEYE: model = "THIN_PRISM"; break;
                        default: break;
                        }
                        const bool is_pinhole = (model_type == lfs::core::CameraModelType::PINHOLE);
                        ImGui::TextColored(is_pinhole ? t.palette.text_dim : t.palette.warning, "%s", model);
                    } else {
                        ImGui::TextColored(t.palette.warning, "GT Compare");
                        ImGui::SameLine(0.0f, 4.0f);
                        ImGui::TextColored(t.palette.text_dim, "Cam %d", cam_id);
                    }
                } else {
                    ImGui::TextColored(t.palette.warning, "Split:");
                    ImGui::SameLine(0.0f, 4.0f);
                    ImGui::Text("%s | %s", info.left_name.c_str(), info.right_name.c_str());
                }
            }

            // WASD speed
            if (speed_overlay_visible_) {
                const float a = fade_alpha(speed_overlay_start_time_);
                ImGui::SameLine(0.0f, SPACING);
                ImGui::TextColored(withAlpha(t.palette.text_dim, a), "|");
                ImGui::SameLine();
                ImGui::TextColored(withAlpha(t.palette.info, a), "WASD: %.0f", current_speed_);
            }

            // Zoom speed
            if (zoom_speed_overlay_visible_) {
                const float a = fade_alpha(zoom_speed_overlay_start_time_);
                ImGui::SameLine(0.0f, SPACING);
                ImGui::TextColored(withAlpha(t.palette.text_dim, a), "|");
                ImGui::SameLine();
                ImGui::TextColored(withAlpha(t.palette.info, a), "Zoom: %.0f", zoom_speed_ * 10.0f);
            }

            // GPU Memory and FPS (right-aligned)
            constexpr float GPU_MEM_WARN_PCT = 50.0f;
            constexpr float GPU_MEM_CRIT_PCT = 75.0f;
            constexpr float FPS_GOOD = 30.0f;
            constexpr float FPS_WARN = 15.0f;
            constexpr float BYTES_TO_GB = 1e-9f;

            ImFont* const font = ctx.fonts.bold ? ctx.fonts.bold : ImGui::GetFont();

            size_t free_mem = 0, total_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);
            const float used_gb = static_cast<float>(total_mem - free_mem) * BYTES_TO_GB;
            const float total_gb = static_cast<float>(total_mem) * BYTES_TO_GB;
            const float pct_used = (used_gb / total_gb) * 100.0f;

            const ImVec4 mem_color = pct_used < GPU_MEM_WARN_PCT   ? t.palette.success
                                     : pct_used < GPU_MEM_CRIT_PCT ? t.palette.warning
                                                                   : t.palette.error;

            char mem_buf[32];
            snprintf(mem_buf, sizeof(mem_buf), "%.1f/%.1fGB", used_gb, total_gb);

            const float fps = rm->getAverageFPS();
            const ImVec4 fps_color = fps >= FPS_GOOD   ? t.palette.success
                                     : fps >= FPS_WARN ? t.palette.warning
                                                       : t.palette.error;
            char fps_buf[16];
            snprintf(fps_buf, sizeof(fps_buf), "%.0f", fps);

            const float mem_val_w = font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, mem_buf).x;
            const float mem_label_w = font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, "GPU ").x;
            const float fps_w = font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, fps_buf).x;
            const float fps_label_w = font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, " FPS").x;

            ImGui::SameLine(size.x - (mem_label_w + mem_val_w + SPACING + fps_w + fps_label_w) - PADDING * 2);

            ImGui::TextColored(t.palette.text_dim, "GPU ");
            ImGui::SameLine(0.0f, 0.0f);
            if (ctx.fonts.bold)
                ImGui::PushFont(ctx.fonts.bold);
            ImGui::TextColored(mem_color, "%s", mem_buf);
            if (ctx.fonts.bold)
                ImGui::PopFont();

            ImGui::SameLine(0.0f, SPACING);

            if (ctx.fonts.bold)
                ImGui::PushFont(ctx.fonts.bold);
            ImGui::TextColored(fps_color, "%s", fps_buf);
            ImGui::SameLine(0.0f, 0.0f);
            ImGui::TextColored(t.palette.text_dim, " FPS");
            if (ctx.fonts.bold)
                ImGui::PopFont();

            if (ctx.fonts.regular)
                ImGui::PopFont();
        }
        ImGui::End();

        ImGui::PopStyleVar(4);
        ImGui::PopStyleColor(2);
    }

    void GuiManager::showSpeedOverlay(const float current_speed, float /*max_speed*/) {
        current_speed_ = current_speed;
        speed_overlay_visible_ = true;
        speed_overlay_start_time_ = std::chrono::steady_clock::now();
    }

    void GuiManager::showZoomSpeedOverlay(const float zoom_speed, float /*max_zoom_speed*/) {
        zoom_speed_ = zoom_speed;
        zoom_speed_overlay_visible_ = true;
        zoom_speed_overlay_start_time_ = std::chrono::steady_clock::now();
    }

    void GuiManager::triggerCropFlash() {
        crop_flash_active_ = true;
        crop_flash_start_ = std::chrono::steady_clock::now();
    }

    void GuiManager::updateCropFlash() {
        if (!crop_flash_active_)
            return;

        auto* const sm = viewer_->getSceneManager();
        auto* const rm = viewer_->getRenderingManager();
        if (!sm || !rm)
            return;

        constexpr int DURATION_MS = 400;
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - crop_flash_start_)
                                    .count();

        const NodeId cropbox_id = sm->getSelectedNodeCropBoxId();
        if (cropbox_id == NULL_NODE) {
            crop_flash_active_ = false;
            return;
        }

        auto* node = sm->getScene().getMutableNode(sm->getScene().getNodeById(cropbox_id)->name);
        if (!node || !node->cropbox) {
            crop_flash_active_ = false;
            return;
        }

        if (elapsed_ms >= DURATION_MS) {
            crop_flash_active_ = false;
            node->cropbox->flash_intensity = 0.0f;
        } else {
            node->cropbox->flash_intensity = 1.0f - static_cast<float>(elapsed_ms) / DURATION_MS;
        }
        sm->getScene().invalidateCache();
        rm->markDirty();
    }

    void GuiManager::deactivateAllTools() {
        if (auto* const t = viewer_->getSelectionTool())
            t->setEnabled(false);
        if (auto* const t = viewer_->getBrushTool())
            t->setEnabled(false);
        if (auto* const t = viewer_->getAlignTool())
            t->setEnabled(false);

        if (auto* const sm = viewer_->getSceneManager()) {
            sm->applyDeleted();
        }

        auto& editor = viewer_->getEditorContext();
        if (editor.getActiveTool() == ToolType::CropBox) {
            if (auto* const rm = viewer_->getRenderingManager()) {
                auto settings = rm->getSettings();
                settings.show_crop_box = false;
                settings.use_crop_box = false;
                rm->updateSettings(settings);
            }
        }

        editor.setActiveTool(ToolType::None);
        gizmo_toolbar_state_.current_operation = ImGuizmo::TRANSLATE;
    }

    void GuiManager::setupEventHandlers() {
        using namespace lfs::core::events;

        cmd::ShowWindow::when([this](const auto& e) {
            showWindow(e.window_name, e.show);
        });

        cmd::ShowDatasetLoadPopup::when([this](const auto& e) {
            if (save_directory_popup_) {
                save_directory_popup_->show(e.dataset_path);
            }
        });

        ui::NodeSelected::when([this](const auto&) {
            if (auto* const t = viewer_->getSelectionTool())
                t->setEnabled(false);
            if (auto* const t = viewer_->getBrushTool())
                t->setEnabled(false);
            if (auto* const t = viewer_->getAlignTool())
                t->setEnabled(false);
            if (auto* const sm = viewer_->getSceneManager())
                sm->syncCropBoxToRenderSettings();
        });
        ui::NodeDeselected::when([this](const auto&) {
            if (auto* const t = viewer_->getSelectionTool())
                t->setEnabled(false);
            if (auto* const t = viewer_->getBrushTool())
                t->setEnabled(false);
            if (auto* const t = viewer_->getAlignTool())
                t->setEnabled(false);
        });
        state::PLYRemoved::when([this](const auto&) { deactivateAllTools(); });
        state::SceneCleared::when([this](const auto&) { deactivateAllTools(); });

        // Handle speed change events
        ui::SpeedChanged::when([this](const auto& e) {
            showSpeedOverlay(e.current_speed, e.max_speed);
        });

        ui::ZoomSpeedChanged::when([this](const auto& e) {
            showZoomSpeedOverlay(e.zoom_speed, e.max_zoom_speed);
        });

        lfs::core::events::tools::SetToolbarTool::when([this](const auto& e) {
            auto& editor = viewer_->getEditorContext();
            editor.setActiveTool(static_cast<ToolType>(e.tool_mode));
            if (editor.getActiveTool() == ToolType::CropBox) {
                gizmo_toolbar_state_.cropbox_operation = panels::CropBoxOperation::Bounds;
                crop_gizmo_operation_ = ImGuizmo::BOUNDS;
            }
        });

        cmd::ApplyCropBox::when([this](const auto&) {
            if (viewer_->getEditorContext().getActiveTool() != ToolType::CropBox) {
                return;
            }
            auto* const sm = viewer_->getSceneManager();
            if (!sm)
                return;

            const NodeId cropbox_id = sm->getSelectedNodeCropBoxId();
            if (cropbox_id == NULL_NODE)
                return;

            const auto* cropbox_node = sm->getScene().getNodeById(cropbox_id);
            if (!cropbox_node || !cropbox_node->cropbox)
                return;

            const glm::mat4 world_transform = sm->getScene().getWorldTransform(cropbox_id);

            lfs::geometry::BoundingBox crop_box;
            crop_box.setBounds(cropbox_node->cropbox->min, cropbox_node->cropbox->max);
            crop_box.setworld2BBox(glm::inverse(world_transform));
            cmd::CropPLY{.crop_box = crop_box, .inverse = cropbox_node->cropbox->inverse}.emit();
            triggerCropFlash();
        });

        // Handle Ctrl+T to toggle crop inverse mode
        cmd::ToggleCropInverse::when([this](const auto&) {
            if (viewer_->getEditorContext().getActiveTool() != ToolType::CropBox) {
                return;
            }
            auto* const sm = viewer_->getSceneManager();
            if (!sm)
                return;

            const NodeId cropbox_id = sm->getSelectedNodeCropBoxId();
            if (cropbox_id == NULL_NODE)
                return;

            auto* node = sm->getScene().getMutableNode(sm->getScene().getNodeById(cropbox_id)->name);
            if (!node || !node->cropbox)
                return;

            // Capture state before toggle
            const command::CropBoxState old_state{
                .min = node->cropbox->min,
                .max = node->cropbox->max,
                .local_transform = node->local_transform.get(),
                .inverse = node->cropbox->inverse};

            // Toggle crop inverse
            node->cropbox->inverse = !node->cropbox->inverse;
            sm->getScene().invalidateCache();

            // Capture state after toggle
            const command::CropBoxState new_state{
                .min = node->cropbox->min,
                .max = node->cropbox->max,
                .local_transform = node->local_transform.get(),
                .inverse = node->cropbox->inverse};

            auto cmd = std::make_unique<command::CropBoxCommand>(
                sm, node->name, old_state, new_state);
            viewer_->getCommandHistory().execute(std::move(cmd));
        });

        // Cycle: normal -> center markers -> rings -> normal
        cmd::CycleSelectionVisualization::when([this](const auto&) {
            if (viewer_->getEditorContext().getActiveTool() != ToolType::Selection)
                return;
            auto* const rm = viewer_->getRenderingManager();
            if (!rm)
                return;

            auto settings = rm->getSettings();
            const bool centers = settings.show_center_markers;
            const bool rings = settings.show_rings;

            settings.show_center_markers = !centers && !rings;
            settings.show_rings = centers && !rings;
            rm->updateSettings(settings);
        });

        ui::FocusTrainingPanel::when([this](const auto&) {
            focus_training_panel_ = true;
        });

        ui::ToggleUI::when([this](const auto&) {
            ui_hidden_ = !ui_hidden_;
        });

        ui::ToggleFullscreen::when([this](const auto&) {
            if (auto* wm = viewer_->getWindowManager()) {
                wm->toggleFullscreen();
            }
        });
    }

    void GuiManager::setSelectionSubMode(panels::SelectionSubMode mode) {
        if (viewer_->getEditorContext().getActiveTool() == ToolType::Selection) {
            gizmo_toolbar_state_.selection_mode = mode;
        }
    }

    panels::ToolType GuiManager::getCurrentToolMode() const {
        return viewer_->getEditorContext().getActiveTool();
    }

    bool GuiManager::isCapturingInput() const {
        return menu_bar_ && menu_bar_->isCapturingInput();
    }

    bool GuiManager::isModalWindowOpen() const {
        return menu_bar_ && menu_bar_->isInputSettingsOpen();
    }

    void GuiManager::captureKey(int key, int mods) {
        if (menu_bar_) {
            menu_bar_->captureKey(key, mods);
        }
    }

    void GuiManager::captureMouseButton(int button, int mods) {
        if (menu_bar_) {
            menu_bar_->captureMouseButton(button, mods);
        }
    }

    void GuiManager::applyDefaultStyle() {
        // Initialize theme system and apply to ImGui
        setTheme(darkTheme());
    }

    void GuiManager::showWindow(const std::string& name, bool show) {
        window_states_[name] = show;
    }

    void GuiManager::toggleWindow(const std::string& name) {
        window_states_[name] = !window_states_[name];
    }

    bool GuiManager::wantsInput() const {
        // Block all input while exporting
        if (export_state_.active.load()) {
            return true;
        }
        ImGuiIO& io = ImGui::GetIO();
        return io.WantCaptureMouse || io.WantCaptureKeyboard;
    }

    bool GuiManager::isAnyWindowActive() const {
        // Block all interaction while exporting
        if (export_state_.active.load()) {
            return true;
        }
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

    void GuiManager::renderCropBoxGizmo(const UIContext& ctx) {
        if (isModalWindowOpen())
            return;

        auto* const render_manager = ctx.viewer->getRenderingManager();
        auto* const scene_manager = ctx.viewer->getSceneManager();
        if (!render_manager || !scene_manager)
            return;

        const auto& settings = render_manager->getSettings();
        if (!settings.show_crop_box)
            return;

        // Get selected cropbox from scene graph
        const NodeId cropbox_id = scene_manager->getSelectedNodeCropBoxId();
        if (cropbox_id == NULL_NODE)
            return;

        const auto* cropbox_node = scene_manager->getScene().getNodeById(cropbox_id);
        if (!cropbox_node || !cropbox_node->visible || !cropbox_node->cropbox)
            return;
        if (!scene_manager->getScene().isNodeEffectivelyVisible(cropbox_id))
            return;

        // Camera setup
        auto& viewport = ctx.viewer->getViewport();
        const glm::mat4 view = viewport.getViewMatrix();
        const glm::ivec2 vp_size(static_cast<int>(viewport_size_.x), static_cast<int>(viewport_size_.y));
        const glm::mat4 projection = lfs::rendering::createProjectionMatrix(
            vp_size, settings.fov, settings.orthographic, settings.ortho_scale);

        // Get cropbox state from scene graph
        const glm::vec3 cropbox_min = cropbox_node->cropbox->min;
        const glm::vec3 cropbox_max = cropbox_node->cropbox->max;
        const glm::mat4 world_transform = scene_manager->getScene().getWorldTransform(cropbox_id);

        // Build gizmo matrix: T * R * S (ImGuizmo expects size in scale component)
        const glm::vec3 original_size = cropbox_max - cropbox_min;
        const glm::vec3 original_center = (cropbox_min + cropbox_max) * 0.5f;
        const glm::vec3 translation = glm::vec3(world_transform[3]);
        const glm::mat3 rotation3x3 = glm::mat3(world_transform);

        glm::mat4 gizmo_matrix = glm::translate(glm::mat4(1.0f), translation + rotation3x3 * original_center);
        gizmo_matrix = gizmo_matrix * glm::mat4(rotation3x3);
        gizmo_matrix = glm::scale(gizmo_matrix, original_size);

        // ImGuizmo setup
        ImGuizmo::SetOrthographic(settings.orthographic);
        ImGuizmo::SetRect(viewport_pos_.x, viewport_pos_.y, viewport_size_.x, viewport_size_.y);
        ImGuizmo::SetAxisLimit(GIZMO_AXIS_LIMIT);
        ImGuizmo::SetPlaneLimit(GIZMO_AXIS_LIMIT);

        if (crop_gizmo_operation_ == ImGuizmo::BOUNDS) {
            ImGuizmo::SetAxisMask(true, true, true);
        } else {
            static bool s_hovered_axis = false;
            const bool is_using = ImGuizmo::IsUsing();
            if (!is_using) {
                s_hovered_axis = ImGuizmo::IsOver(ImGuizmo::TRANSLATE_X) ||
                                 ImGuizmo::IsOver(ImGuizmo::TRANSLATE_Y) ||
                                 ImGuizmo::IsOver(ImGuizmo::TRANSLATE_Z);
                ImGuizmo::SetAxisMask(false, false, false);
            } else {
                ImGuizmo::SetAxisMask(s_hovered_axis, s_hovered_axis, s_hovered_axis);
            }
        }

        // Clip to viewport
        ImDrawList* overlay_drawlist = ImGui::GetForegroundDrawList();
        const ImVec2 clip_min(viewport_pos_.x, viewport_pos_.y);
        const ImVec2 clip_max(clip_min.x + viewport_size_.x, clip_min.y + viewport_size_.y);
        overlay_drawlist->PushClipRect(clip_min, clip_max, true);
        ImGuizmo::SetDrawlist(overlay_drawlist);

        glm::mat4 delta_matrix;
        constexpr float LOCAL_BOUNDS[6] = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};

        const ImGuizmo::MODE gizmo_mode = (crop_gizmo_operation_ == ImGuizmo::TRANSLATE)
                                              ? ImGuizmo::WORLD
                                              : ImGuizmo::LOCAL;

        const bool gizmo_changed = ImGuizmo::Manipulate(
            glm::value_ptr(view), glm::value_ptr(projection),
            crop_gizmo_operation_, gizmo_mode, glm::value_ptr(gizmo_matrix),
            glm::value_ptr(delta_matrix), nullptr,
            crop_gizmo_operation_ == ImGuizmo::BOUNDS ? LOCAL_BOUNDS : nullptr);

        const bool is_using = ImGuizmo::IsUsing();

        // Capture state when manipulation starts
        if (is_using && !cropbox_gizmo_active_) {
            cropbox_gizmo_active_ = true;
            cropbox_node_name_ = cropbox_node->name;
            cropbox_state_before_drag_ = command::CropBoxState{
                .min = cropbox_node->cropbox->min,
                .max = cropbox_node->cropbox->max,
                .local_transform = cropbox_node->local_transform.get(),
                .inverse = cropbox_node->cropbox->inverse};
        }

        if (gizmo_changed) {
            // Extract transform from gizmo matrix
            float mat_trans[3], mat_rot[3], mat_scale[3];
            ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(gizmo_matrix), mat_trans, mat_rot, mat_scale);

            const float rad_x = glm::radians(mat_rot[0]);
            const float rad_y = glm::radians(mat_rot[1]);
            const float rad_z = glm::radians(mat_rot[2]);

            glm::vec3 new_size(mat_scale[0], mat_scale[1], mat_scale[2]);
            new_size = glm::max(new_size, glm::vec3(0.001f));

            const glm::mat3 new_rotation = glm::mat3(
                glm::rotate(glm::mat4(1.0f), rad_x, glm::vec3(1, 0, 0)) *
                glm::rotate(glm::mat4(1.0f), rad_y, glm::vec3(0, 1, 0)) *
                glm::rotate(glm::mat4(1.0f), rad_z, glm::vec3(0, 0, 1)));

            const glm::vec3 world_center(mat_trans[0], mat_trans[1], mat_trans[2]);

            // Update scene graph directly
            auto* mutable_node = scene_manager->getScene().getMutableNode(cropbox_node->name);
            if (mutable_node && mutable_node->cropbox) {
                // Compute new bounds and world transform
                glm::vec3 new_min, new_max;
                glm::mat4 new_world_transform;

                if (crop_gizmo_operation_ == ImGuizmo::SCALE || crop_gizmo_operation_ == ImGuizmo::BOUNDS) {
                    const glm::vec3 new_half = new_size * 0.5f;
                    new_min = -new_half;
                    new_max = new_half;
                    const glm::vec3 new_center = (new_min + new_max) * 0.5f;
                    const glm::vec3 transform_trans = world_center - new_rotation * new_center;
                    new_world_transform = glm::translate(glm::mat4(1.0f), transform_trans) * glm::mat4(new_rotation);
                } else {
                    new_min = cropbox_min;
                    new_max = cropbox_max;
                    const glm::vec3 transform_trans = world_center - new_rotation * original_center;
                    new_world_transform = glm::translate(glm::mat4(1.0f), transform_trans) * glm::mat4(new_rotation);
                }

                mutable_node->cropbox->min = new_min;
                mutable_node->cropbox->max = new_max;

                // Convert world transform to local
                if (mutable_node->parent_id != NULL_NODE) {
                    const glm::mat4 parent_world = scene_manager->getScene().getWorldTransform(mutable_node->parent_id);
                    mutable_node->local_transform = glm::inverse(parent_world) * new_world_transform;
                } else {
                    mutable_node->local_transform = new_world_transform;
                }
                mutable_node->transform_dirty = true;
                scene_manager->getScene().invalidateCache();
                render_manager->markDirty();
            }
        }

        // Create undo command when manipulation ends
        if (!is_using && cropbox_gizmo_active_) {
            cropbox_gizmo_active_ = false;

            if (cropbox_state_before_drag_.has_value()) {
                auto* node = scene_manager->getScene().getMutableNode(cropbox_node_name_);
                if (node && node->cropbox) {
                    const command::CropBoxState new_state{
                        .min = node->cropbox->min,
                        .max = node->cropbox->max,
                        .local_transform = node->local_transform.get(),
                        .inverse = node->cropbox->inverse};

                    auto cmd = std::make_unique<command::CropBoxCommand>(
                        scene_manager, cropbox_node_name_, *cropbox_state_before_drag_, new_state);
                    viewer_->getCommandHistory().execute(std::move(cmd));

                    using namespace lfs::core::events;
                    ui::CropBoxChanged{
                        .min_bounds = node->cropbox->min,
                        .max_bounds = node->cropbox->max,
                        .enabled = settings.use_crop_box}
                        .emit();
                }
                cropbox_state_before_drag_.reset();
            }
        }

        overlay_drawlist->PopClipRect();
    }

    void GuiManager::renderNodeTransformGizmo(const UIContext& ctx) {
        if (isModalWindowOpen())
            return;
        if (!show_node_gizmo_)
            return;

        auto* scene_manager = ctx.viewer->getSceneManager();
        if (!scene_manager || !scene_manager->hasSelectedNode())
            return;

        // Check visibility of at least one selected node
        const auto& scene = scene_manager->getScene();
        const auto selected_names = scene_manager->getSelectedNodeNames();
        bool any_visible = false;
        for (const auto& name : selected_names) {
            if (const auto* node = scene.getNode(name)) {
                if (scene.isNodeEffectivelyVisible(node->id)) {
                    any_visible = true;
                    break;
                }
            }
        }
        if (!any_visible)
            return;

        auto* render_manager = ctx.viewer->getRenderingManager();
        if (!render_manager)
            return;

        const auto& settings = render_manager->getSettings();
        const bool is_multi_selection = (selected_names.size() > 1);

        // Camera matrices
        auto& viewport = ctx.viewer->getViewport();
        const glm::mat4 view = viewport.getViewMatrix();
        const glm::ivec2 vp_size(static_cast<int>(viewport_size_.x), static_cast<int>(viewport_size_.y));
        const glm::mat4 projection = lfs::rendering::createProjectionMatrix(
            vp_size, settings.fov, settings.orthographic, settings.ortho_scale);

        const bool use_world_space =
            (gizmo_toolbar_state_.transform_space == panels::TransformSpace::World) || is_multi_selection;

        const glm::vec3 gizmo_position = is_multi_selection
                                             ? scene_manager->getSelectionWorldCenter()
                                             : glm::vec3(scene_manager->getSelectedNodeWorldTransform() *
                                                         glm::vec4(scene_manager->getSelectionCenter(), 1.0f));

        glm::mat4 gizmo_matrix(1.0f);
        gizmo_matrix[3] = glm::vec4(gizmo_position, 1.0f);

        if (!is_multi_selection && !use_world_space) {
            const glm::mat3 rotation_scale(scene_manager->getSelectedNodeWorldTransform());
            gizmo_matrix[0] = glm::vec4(rotation_scale[0], 0.0f);
            gizmo_matrix[1] = glm::vec4(rotation_scale[1], 0.0f);
            gizmo_matrix[2] = glm::vec4(rotation_scale[2], 0.0f);
        }

        ImGuizmo::SetOrthographic(settings.orthographic);
        ImGuizmo::SetRect(viewport_pos_.x, viewport_pos_.y, viewport_size_.x, viewport_size_.y);
        ImGuizmo::SetAxisLimit(GIZMO_AXIS_LIMIT);
        ImGuizmo::SetPlaneLimit(GIZMO_AXIS_LIMIT);

        static bool s_node_hovered_axis = false;
        const bool is_using = ImGuizmo::IsUsing();

        if (!is_using) {
            s_node_hovered_axis = ImGuizmo::IsOver(ImGuizmo::TRANSLATE_X) ||
                                  ImGuizmo::IsOver(ImGuizmo::TRANSLATE_Y) ||
                                  ImGuizmo::IsOver(ImGuizmo::TRANSLATE_Z);
            ImGuizmo::SetAxisMask(false, false, false);
        } else {
            ImGuizmo::SetAxisMask(s_node_hovered_axis, s_node_hovered_axis, s_node_hovered_axis);
        }

        ImDrawList* overlay_drawlist = ImGui::GetForegroundDrawList();
        const ImVec2 clip_min(viewport_pos_.x, viewport_pos_.y);
        const ImVec2 clip_max(clip_min.x + viewport_size_.x, clip_min.y + viewport_size_.y);
        overlay_drawlist->PushClipRect(clip_min, clip_max, true);
        ImGuizmo::SetDrawlist(overlay_drawlist);

        const ImGuizmo::MODE gizmo_mode = use_world_space ? ImGuizmo::WORLD : ImGuizmo::LOCAL;

        glm::mat4 delta_matrix;
        const bool gizmo_changed = ImGuizmo::Manipulate(
            glm::value_ptr(view), glm::value_ptr(projection),
            node_gizmo_operation_, gizmo_mode,
            glm::value_ptr(gizmo_matrix), glm::value_ptr(delta_matrix), nullptr);

        // Capture state for undo when drag starts
        if (is_using && !node_gizmo_active_) {
            node_gizmo_active_ = true;
            gizmo_pivot_ = gizmo_position;
            gizmo_cumulative_rotation_ = glm::mat3(1.0f);
            node_gizmo_node_names_ = selected_names;
            node_transforms_before_drag_.clear();
            node_transforms_before_drag_.reserve(selected_names.size());
            for (const auto& name : selected_names) {
                node_transforms_before_drag_.push_back(scene_manager->getNodeTransform(name));
            }
        }

        if (gizmo_changed) {
            if (is_multi_selection) {
                // Accumulate delta rotation
                const glm::mat3 delta_rot(delta_matrix);
                gizmo_cumulative_rotation_ = delta_rot * gizmo_cumulative_rotation_;

                // Compute total translation from gizmo movement
                const glm::vec3 new_gizmo_pos(gizmo_matrix[3]);
                const glm::vec3 total_translation = new_gizmo_pos - gizmo_pivot_;

                for (size_t i = 0; i < node_gizmo_node_names_.size(); ++i) {
                    const glm::mat4& original = node_transforms_before_drag_[i];
                    const glm::vec3 original_pos(original[3]);
                    const glm::mat3 original_rot(original);

                    // Rotate position around pivot using cumulative rotation
                    const glm::vec3 offset = original_pos - gizmo_pivot_;
                    const glm::vec3 rotated_offset = gizmo_cumulative_rotation_ * offset;
                    const glm::vec3 new_pos = gizmo_pivot_ + rotated_offset + total_translation;

                    // Combine rotations
                    const glm::mat3 new_rot = gizmo_cumulative_rotation_ * original_rot;

                    glm::mat4 new_transform(new_rot);
                    new_transform[3] = glm::vec4(new_pos, 1.0f);

                    scene_manager->setNodeTransform(node_gizmo_node_names_[i], new_transform);
                }
            } else {
                // Single selection
                const glm::vec3 center = scene_manager->getSelectionCenter();
                const glm::mat4 node_transform = scene_manager->getSelectedNodeTransform();
                const glm::vec3 new_gizmo_pos_world = glm::vec3(gizmo_matrix[3]);

                // Convert world position to parent space
                const auto& scene = scene_manager->getScene();
                const auto* node = scene.getNode(*selected_names.begin());
                const glm::mat4 parent_world_inv = (node && node->parent_id != NULL_NODE)
                                                       ? glm::inverse(scene.getWorldTransform(node->parent_id))
                                                       : glm::mat4(1.0f);
                const glm::vec3 new_gizmo_pos = glm::vec3(parent_world_inv * glm::vec4(new_gizmo_pos_world, 1.0f));

                glm::mat4 new_transform;
                if (use_world_space) {
                    const glm::mat3 old_rs(node_transform);
                    const glm::mat3 delta_rs(delta_matrix);
                    const glm::mat3 new_rs = delta_rs * old_rs;
                    new_transform = glm::mat4(new_rs);
                    new_transform[3] = glm::vec4(new_gizmo_pos - new_rs * center, 1.0f);
                } else {
                    const glm::mat3 new_rs(gizmo_matrix);
                    new_transform = gizmo_matrix;
                    new_transform[3] = glm::vec4(new_gizmo_pos - new_rs * center, 1.0f);
                }
                scene_manager->setSelectedNodeTransform(new_transform);
            }
        }

        // Create undo command when drag ends
        if (!is_using && node_gizmo_active_) {
            node_gizmo_active_ = false;

            const size_t count = node_gizmo_node_names_.size();
            std::vector<glm::mat4> final_transforms;
            final_transforms.reserve(count);
            for (const auto& name : node_gizmo_node_names_) {
                final_transforms.push_back(scene_manager->getNodeTransform(name));
            }

            bool any_changed = false;
            for (size_t i = 0; i < count; ++i) {
                if (node_transforms_before_drag_[i] != final_transforms[i]) {
                    any_changed = true;
                    break;
                }
            }

            if (any_changed) {
                if (count == 1) {
                    auto cmd = std::make_unique<command::TransformCommand>(
                        node_gizmo_node_names_[0],
                        node_transforms_before_drag_[0], final_transforms[0]);
                    services().commands().execute(std::move(cmd));
                } else {
                    auto cmd = std::make_unique<command::MultiTransformCommand>(
                        node_gizmo_node_names_,
                        node_transforms_before_drag_, std::move(final_transforms));
                    services().commands().execute(std::move(cmd));
                }
            }
        }

        overlay_drawlist->PopClipRect();
    }

    void GuiManager::startAsyncExport(ExportFormat format,
                                      const std::filesystem::path& path,
                                      std::unique_ptr<lfs::core::SplatData> data) {
        if (!data) {
            LOG_ERROR("No splat data to export");
            return;
        }

        export_state_.active.store(true);
        export_state_.cancel_requested.store(false);
        export_state_.progress.store(0.0f);
        {
            const std::lock_guard lock(export_state_.mutex);
            export_state_.format = format;
            export_state_.stage = "Starting";
            export_state_.error.clear();
        }

        auto splat_data = std::make_shared<lfs::core::SplatData>(std::move(*data));
        LOG_INFO("Export started: {} (format: {})", path.string(), static_cast<int>(format));

        export_state_.thread = std::make_unique<std::jthread>(
            [this, format, path, splat_data](std::stop_token stop_token) {
                auto update_progress = [this, &stop_token](float progress, const std::string& stage) -> bool {
                    export_state_.progress.store(progress);
                    {
                        const std::lock_guard lock(export_state_.mutex);
                        export_state_.stage = stage;
                    }
                    if (stop_token.stop_requested() || export_state_.cancel_requested.load()) {
                        LOG_INFO("Export cancelled");
                        return false;
                    }
                    return true;
                };

                bool success = false;
                std::string error_msg;

                switch (format) {
                case ExportFormat::PLY: {
                    update_progress(0.1f, "Writing PLY");
                    lfs::core::save_ply(*splat_data, path.parent_path(), 0, true, path.stem().string());
                    success = true;
                    update_progress(1.0f, "Complete");
                    break;
                }
                case ExportFormat::SOG: {
                    const lfs::core::SogWriteOptions options{
                        .iterations = 10,
                        .output_path = path,
                        .progress_callback = update_progress};
                    if (auto result = lfs::core::write_sog(*splat_data, options); result) {
                        success = true;
                    } else {
                        error_msg = result.error();
                    }
                    break;
                }
                case ExportFormat::SPZ: {
                    update_progress(0.1f, "Writing SPZ");
                    const lfs::io::SpzSaveOptions options{.output_path = path};
                    if (auto result = lfs::io::save_spz(*splat_data, options); result) {
                        success = true;
                        update_progress(1.0f, "Complete");
                    } else {
                        error_msg = result.error().message;
                    }
                    break;
                }
                case ExportFormat::HTML_VIEWER: {
                    const HtmlViewerExportOptions options{
                        .output_path = path,
                        .progress_callback = [&update_progress](float p, const std::string& s) {
                            update_progress(p, s);
                        }};
                    if (auto result = export_html_viewer(*splat_data, options); result) {
                        success = true;
                    } else {
                        error_msg = result.error();
                    }
                    break;
                }
                }

                if (success) {
                    LOG_INFO("Export completed: {}", path.string());
                    const std::lock_guard lock(export_state_.mutex);
                    export_state_.stage = "Complete";
                } else {
                    LOG_ERROR("Export failed: {}", error_msg);
                    const std::lock_guard lock(export_state_.mutex);
                    export_state_.error = error_msg;
                    export_state_.stage = "Failed";
                }

                export_state_.active.store(false);
            });
    }

    void GuiManager::cancelExport() {
        if (!export_state_.active.load())
            return;

        LOG_INFO("Cancelling export");
        export_state_.cancel_requested.store(true);
        if (export_state_.thread && export_state_.thread->joinable()) {
            export_state_.thread->request_stop();
        }
    }

    void GuiManager::renderExportOverlay() {
        if (!export_state_.active.load()) {
            if (export_state_.thread && export_state_.thread->joinable()) {
                export_state_.thread->join();
                export_state_.thread.reset();
            }
            return;
        }

        constexpr float OVERLAY_WIDTH = 350.0f;
        constexpr float OVERLAY_HEIGHT = 100.0f;
        constexpr float BUTTON_WIDTH = 100.0f;
        constexpr float BUTTON_HEIGHT = 30.0f;
        constexpr float PROGRESS_BAR_HEIGHT = 20.0f;

        const ImGuiViewport* const viewport = ImGui::GetMainViewport();
        const ImVec2 overlay_pos(
            viewport->WorkPos.x + (viewport->WorkSize.x - OVERLAY_WIDTH) * 0.5f,
            viewport->WorkPos.y + (viewport->WorkSize.y - OVERLAY_HEIGHT) * 0.5f);

        ImGui::SetNextWindowPos(overlay_pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(OVERLAY_WIDTH, 0), ImGuiCond_Always);

        constexpr ImGuiWindowFlags FLAGS = ImGuiWindowFlags_NoTitleBar |
                                           ImGuiWindowFlags_NoResize |
                                           ImGuiWindowFlags_NoMove |
                                           ImGuiWindowFlags_NoScrollbar |
                                           ImGuiWindowFlags_NoCollapse |
                                           ImGuiWindowFlags_AlwaysAutoResize;

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.1f, 0.95f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 15));

        if (ImGui::Begin("##ExportProgress", nullptr, FLAGS)) {
            ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
            {
                const std::lock_guard lock(export_state_.mutex);
                const char* format_name = "file";
                switch (export_state_.format) {
                case ExportFormat::PLY: format_name = "PLY"; break;
                case ExportFormat::SOG: format_name = "SOG"; break;
                case ExportFormat::SPZ: format_name = "SPZ"; break;
                case ExportFormat::HTML_VIEWER: format_name = "HTML"; break;
                }
                ImGui::Text("Exporting %s...", format_name);
            }
            ImGui::PopFont();

            ImGui::Spacing();

            const float progress = export_state_.progress.load();
            ImGui::ProgressBar(progress, ImVec2(-1, PROGRESS_BAR_HEIGHT), "");

            ImGui::Text("%.0f%%", progress * 100.0f);
            ImGui::SameLine();

            {
                const std::lock_guard lock(export_state_.mutex);
                ImGui::TextUnformatted(export_state_.stage.c_str());
            }

            ImGui::Spacing();

            ImGui::SetCursorPosX((OVERLAY_WIDTH - BUTTON_WIDTH) * 0.5f - ImGui::GetStyle().WindowPadding.x);
            if (ImGui::Button("Cancel", ImVec2(BUTTON_WIDTH, BUTTON_HEIGHT))) {
                cancelExport();
            }

            ImGui::End();
        }

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();

        ImDrawList* const draw_list = ImGui::GetBackgroundDrawList();
        draw_list->AddRectFilled(
            viewport->WorkPos,
            ImVec2(viewport->WorkPos.x + viewport->WorkSize.x,
                   viewport->WorkPos.y + viewport->WorkSize.y),
            IM_COL32(0, 0, 0, 100));
    }

    void GuiManager::renderEmptyStateOverlay() {
        const auto* const scene_manager = viewer_->getSceneManager();
        if (!scene_manager || !scene_manager->isEmpty() || drag_drop_hovering_)
            return;
        if (viewport_size_.x < 200.0f || viewport_size_.y < 200.0f)
            return;

        constexpr float ZONE_PADDING = 120.0f;
        constexpr float DASH_LENGTH = 12.0f;
        constexpr float GAP_LENGTH = 8.0f;
        constexpr float BORDER_THICKNESS = 2.0f;
        constexpr float ICON_SIZE = 48.0f;
        constexpr float ANIM_SPEED = 30.0f;
        constexpr ImU32 BORDER_COLOR = IM_COL32(100, 140, 180, 80);
        constexpr ImU32 ICON_COLOR = IM_COL32(120, 160, 200, 120);
        constexpr ImU32 TITLE_COLOR = IM_COL32(180, 200, 220, 200);
        constexpr ImU32 SUBTITLE_COLOR = IM_COL32(140, 160, 180, 150);
        constexpr ImU32 HINT_COLOR = IM_COL32(100, 120, 140, 120);

        ImDrawList* const draw_list = ImGui::GetBackgroundDrawList();
        const float center_x = viewport_pos_.x + viewport_size_.x * 0.5f;
        const float center_y = viewport_pos_.y + viewport_size_.y * 0.5f;
        const ImVec2 zone_min(viewport_pos_.x + ZONE_PADDING, viewport_pos_.y + ZONE_PADDING);
        const ImVec2 zone_max(viewport_pos_.x + viewport_size_.x - ZONE_PADDING,
                              viewport_pos_.y + viewport_size_.y - ZONE_PADDING);

        const float time = static_cast<float>(ImGui::GetTime());
        const float dash_offset = std::fmod(time * ANIM_SPEED, DASH_LENGTH + GAP_LENGTH);

        // Dashed border
        const auto drawDashedLine = [&](const ImVec2& start, const ImVec2& end) {
            const float dx = end.x - start.x;
            const float dy = end.y - start.y;
            const float length = std::sqrt(dx * dx + dy * dy);
            const float nx = dx / length;
            const float ny = dy / length;
            for (float pos = -dash_offset; pos < length; pos += DASH_LENGTH + GAP_LENGTH) {
                const float d0 = std::max(0.0f, pos);
                const float d1 = std::min(length, pos + DASH_LENGTH);
                if (d1 > d0) {
                    draw_list->AddLine(ImVec2(start.x + nx * d0, start.y + ny * d0),
                                       ImVec2(start.x + nx * d1, start.y + ny * d1),
                                       BORDER_COLOR, BORDER_THICKNESS);
                }
            }
        };
        drawDashedLine(zone_min, {zone_max.x, zone_min.y});
        drawDashedLine({zone_max.x, zone_min.y}, zone_max);
        drawDashedLine(zone_max, {zone_min.x, zone_max.y});
        drawDashedLine({zone_min.x, zone_max.y}, zone_min);

        // Folder icon
        const float icon_y = center_y - 50.0f;
        draw_list->AddRect({center_x - ICON_SIZE * 0.5f, icon_y - ICON_SIZE * 0.3f},
                           {center_x + ICON_SIZE * 0.5f, icon_y + ICON_SIZE * 0.4f},
                           ICON_COLOR, 4.0f, 0, 2.0f);
        draw_list->AddLine({center_x - ICON_SIZE * 0.5f, icon_y - ICON_SIZE * 0.3f},
                           {center_x - ICON_SIZE * 0.2f, icon_y - ICON_SIZE * 0.5f}, ICON_COLOR, 2.0f);
        draw_list->AddLine({center_x - ICON_SIZE * 0.2f, icon_y - ICON_SIZE * 0.5f},
                           {center_x + ICON_SIZE * 0.1f, icon_y - ICON_SIZE * 0.5f}, ICON_COLOR, 2.0f);
        draw_list->AddLine({center_x + ICON_SIZE * 0.1f, icon_y - ICON_SIZE * 0.5f},
                           {center_x + ICON_SIZE * 0.2f, icon_y - ICON_SIZE * 0.3f}, ICON_COLOR, 2.0f);

        // Text
        const auto calcTextSize = [this](const char* text, ImFont* font) {
            if (font)
                ImGui::PushFont(font);
            const ImVec2 size = ImGui::CalcTextSize(text);
            if (font)
                ImGui::PopFont();
            return size;
        };

        static constexpr const char* TITLE = "Drop files here to get started";
        static constexpr const char* SUBTITLE = "PLY, SOG, COLMAP dataset, or Project file";
        static constexpr const char* HINT = "Or use File > Import Dataset / Import Ply";

        const ImVec2 title_size = calcTextSize(TITLE, font_heading_);
        const ImVec2 subtitle_size = calcTextSize(SUBTITLE, font_bold_);
        const ImVec2 hint_size = calcTextSize(HINT, font_heading_);

        if (font_heading_)
            ImGui::PushFont(font_heading_);
        draw_list->AddText({center_x - title_size.x * 0.5f, center_y + 10.0f}, TITLE_COLOR, TITLE);
        if (font_heading_)
            ImGui::PopFont();

        if (font_bold_)
            ImGui::PushFont(font_bold_);
        draw_list->AddText({center_x - subtitle_size.x * 0.5f, center_y + 40.0f}, SUBTITLE_COLOR, SUBTITLE);
        if (font_bold_)
            ImGui::PopFont();

        if (font_heading_)
            ImGui::PushFont(font_heading_);
        draw_list->AddText({center_x - hint_size.x * 0.5f, center_y + 70.0f}, HINT_COLOR, HINT);
        if (font_heading_)
            ImGui::PopFont();
    }

    void GuiManager::renderDragDropOverlay() {
        if (!drag_drop_hovering_)
            return;

        constexpr float INSET = 30.0f;
        constexpr float CORNER_RADIUS = 16.0f;
        constexpr float GLOW_MAX = 8.0f;
        constexpr float PULSE_SPEED = 3.0f;
        constexpr float BOUNCE_SPEED = 4.0f;
        constexpr float BOUNCE_AMOUNT = 5.0f;
        constexpr ImU32 OVERLAY_COLOR = IM_COL32(20, 60, 100, 180);
        constexpr ImU32 FILL_COLOR = IM_COL32(30, 80, 140, 60);
        constexpr ImU32 ICON_COLOR = IM_COL32(255, 255, 255, 230);
        constexpr ImU32 TITLE_COLOR = IM_COL32(255, 255, 255, 255);
        constexpr ImU32 SUBTITLE_COLOR = IM_COL32(200, 220, 240, 200);

        const ImGuiViewport* const vp = ImGui::GetMainViewport();
        ImDrawList* const draw_list = ImGui::GetForegroundDrawList();
        const ImVec2 win_max(vp->WorkPos.x + vp->WorkSize.x, vp->WorkPos.y + vp->WorkSize.y);
        const ImVec2 zone_min(vp->WorkPos.x + INSET, vp->WorkPos.y + INSET);
        const ImVec2 zone_max(win_max.x - INSET, win_max.y - INSET);
        const float center_x = vp->WorkPos.x + vp->WorkSize.x * 0.5f;
        const float center_y = vp->WorkPos.y + vp->WorkSize.y * 0.5f;

        const float time = static_cast<float>(ImGui::GetTime());
        const float pulse = 0.5f + 0.5f * std::sin(time * PULSE_SPEED);

        draw_list->AddRectFilled(vp->WorkPos, win_max, OVERLAY_COLOR);

        // Glow effect
        const ImU32 glow_color = IM_COL32(80, 160, 255, static_cast<uint8_t>(40.0f * pulse));
        for (float i = GLOW_MAX; i > 0.0f; i -= 2.0f) {
            draw_list->AddRect({zone_min.x - i, zone_min.y - i}, {zone_max.x + i, zone_max.y + i},
                               glow_color, CORNER_RADIUS + i, 0, 2.0f);
        }

        const uint8_t border_alpha = static_cast<uint8_t>(180.0f + 75.0f * pulse);
        draw_list->AddRect(zone_min, zone_max, IM_COL32(100, 180, 255, border_alpha), CORNER_RADIUS, 0, 3.0f);
        draw_list->AddRectFilled(zone_min, zone_max, FILL_COLOR, CORNER_RADIUS);

        // Animated arrow
        const float arrow_y = center_y - 60.0f + BOUNCE_AMOUNT * std::sin(time * BOUNCE_SPEED);
        draw_list->AddTriangleFilled({center_x, arrow_y + 25.0f}, {center_x - 20.0f, arrow_y},
                                     {center_x + 20.0f, arrow_y}, ICON_COLOR);
        draw_list->AddRectFilled({center_x - 8.0f, arrow_y - 25.0f}, {center_x + 8.0f, arrow_y}, ICON_COLOR, 2.0f);

        // Text
        const auto calcTextSize = [this](const char* text, ImFont* font) {
            if (font)
                ImGui::PushFont(font);
            const ImVec2 size = ImGui::CalcTextSize(text);
            if (font)
                ImGui::PopFont();
            return size;
        };

        static constexpr const char* TITLE = "Drop to Import";
        static constexpr const char* SUBTITLE = "PLY, SOG, COLMAP dataset, or Project";

        const ImVec2 title_size = calcTextSize(TITLE, font_heading_);
        const ImVec2 subtitle_size = calcTextSize(SUBTITLE, font_small_);

        if (font_heading_)
            ImGui::PushFont(font_heading_);
        draw_list->AddText({center_x - title_size.x * 0.5f, center_y + 5.0f}, TITLE_COLOR, TITLE);
        if (font_heading_)
            ImGui::PopFont();

        if (font_small_)
            ImGui::PushFont(font_small_);
        draw_list->AddText({center_x - subtitle_size.x * 0.5f, center_y + 35.0f}, SUBTITLE_COLOR, SUBTITLE);
        if (font_small_)
            ImGui::PopFont();
    }

    void GuiManager::requestExitConfirmation() {
        if (!exit_confirmation_popup_)
            return;
        exit_confirmation_popup_->show([this]() {
            force_exit_ = true;
            glfwSetWindowShouldClose(viewer_->getWindow(), true);
        });
    }

    bool GuiManager::isExitConfirmationPending() const {
        return exit_confirmation_popup_ && exit_confirmation_popup_->isOpen();
    }

} // namespace lfs::vis::gui
