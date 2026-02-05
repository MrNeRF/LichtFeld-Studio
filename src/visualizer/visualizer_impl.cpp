/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer_impl.hpp"
#include "core/animatable_property.hpp"
#include "core/data_loading_service.hpp"
#include "core/event_bus.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/services.hpp"
#include "gui/panels/tools_panel.hpp"
#include "gui/panels/windows_console_utils.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/align_ops.hpp"
#include "operator/ops/brush_ops.hpp"
#include "operator/ops/edit_ops.hpp"
#include "operator/ops/selection_ops.hpp"
#include "operator/ops/transform_ops.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"
#include "scene/scene_manager.hpp"
#include "tools/align_tool.hpp"
#include "tools/brush_tool.hpp"
#include "tools/builtin_tools.hpp"
#include "tools/selection_tool.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
#ifdef WIN32
#include <windows.h>
#endif

namespace lfs::vis {

    using namespace lfs::core::events;

    VisualizerImpl::VisualizerImpl(const ViewerOptions& options)
        : options_(options),
          viewport_(options.width, options.height),
          window_manager_(std::make_unique<WindowManager>(options.title, options.width, options.height,
                                                          options.monitor_x, options.monitor_y,
                                                          options.monitor_width, options.monitor_height)) {

        LOG_DEBUG("Creating visualizer with window size {}x{}", options.width, options.height);

        // Create scene manager - it creates its own Scene internally
        scene_manager_ = std::make_unique<SceneManager>();

        // Create trainer manager
        trainer_manager_ = std::make_shared<TrainerManager>();
        trainer_manager_->setViewer(this);

        // Create support components
        gui_manager_ = std::make_unique<gui::GuiManager>(this);

        // Create rendering manager with initial antialiasing setting
        rendering_manager_ = std::make_unique<RenderingManager>();

        // Set initial antialiasing
        RenderSettings initial_settings;
        initial_settings.antialiasing = options.antialiasing;
        initial_settings.gut = options.gut;
        rendering_manager_->updateSettings(initial_settings);

        // Create data loading service
        data_loader_ = std::make_unique<DataLoadingService>(scene_manager_.get());

        // Create parameter manager (lazy-loads JSON files on first use)
        parameter_manager_ = std::make_unique<ParameterManager>();

        // Create main loop
        main_loop_ = std::make_unique<MainLoop>();

        // Register services in the service locator
        services().set(scene_manager_.get());
        services().set(trainer_manager_.get());
        services().set(rendering_manager_.get());
        services().set(window_manager_.get());
        services().set(gui_manager_.get());
        services().set(parameter_manager_.get());
        services().set(&editor_context_);

        registerBuiltinTools();

        // Initialize operator system
        op::operators().setSceneManager(scene_manager_.get());
        op::registerTransformOperators();
        op::registerAlignOperators();
        op::registerSelectionOperators();
        op::registerBrushOperators();
        op::registerEditOperators();

        python::set_trainer_manager(trainer_manager_.get());
        python::set_parameter_manager(parameter_manager_.get());
        python::set_rendering_manager(rendering_manager_.get());
        python::set_editor_context(&editor_context_);
        python::set_operator_callbacks(&editor_context_); // Also set as IOperatorCallbacks for callback dispatch
        python::set_gui_manager(gui_manager_.get());
        python::set_selected_camera_callback([]() -> int {
            const auto* gm = python::get_gui_manager();
            return gm ? gm->getHighlightedCameraUid() : -1;
        });
        python::set_invert_masks_callback([]() -> bool {
            auto* pm = python::get_parameter_manager();
            return pm && pm->getActiveParams().invert_masks;
        });
        python::set_sequencer_callbacks(
            []() {
                const auto* gm = python::get_gui_manager();
                return gm ? gm->panelLayout().isShowSequencer() : false;
            },
            [](bool visible) {
                if (auto* gm = python::get_gui_manager())
                    gm->panelLayout().setShowSequencer(visible);
            });

        // Overlay state callbacks (for Python overlay panels)
        python::set_overlay_callbacks(
            []() {
                const auto* gm = python::get_gui_manager();
                return gm ? gm->isDragHovering() : false;
            },
            []() {
                const auto* gm = python::get_gui_manager();
                return gm ? gm->isStartupVisible() : false;
            },
            []() -> python::OverlayExportState {
                const auto* gm = python::get_gui_manager();
                if (!gm)
                    return {};
                python::OverlayExportState state;
                const auto& tasks = gm->asyncTasks();
                state.active = tasks.isExporting();
                state.progress = tasks.getExportProgress();
                state.stage = tasks.getExportStage();
                const auto fmt = tasks.getExportFormat();
                state.format = fmt == core::ExportFormat::PLY   ? "PLY"
                               : fmt == core::ExportFormat::SOG ? "SOG"
                                                                : "file";
                return state;
            },
            []() {
                if (auto* gm = python::get_gui_manager())
                    gm->asyncTasks().cancelExport();
            },
            []() -> python::OverlayImportState {
                const auto* gm = python::get_gui_manager();
                if (!gm)
                    return {};
                python::OverlayImportState state;
                const auto& tasks = gm->asyncTasks();
                state.active = tasks.isImporting();
                state.show_completion = tasks.isImportCompletionShowing();
                state.progress = tasks.getImportProgress();
                state.stage = tasks.getImportStage();
                state.dataset_type = tasks.getImportDatasetType();
                state.path = tasks.getImportPath();
                state.success = tasks.getImportSuccess();
                state.error = tasks.getImportError();
                state.num_images = tasks.getImportNumImages();
                state.num_points = tasks.getImportNumPoints();
                state.seconds_since_completion = tasks.getImportSecondsSinceCompletion();
                return state;
            },
            []() {
                if (auto* gm = python::get_gui_manager())
                    gm->asyncTasks().dismissImport();
            },
            []() -> python::OverlayVideoExportState {
                const auto* gm = python::get_gui_manager();
                if (!gm)
                    return {};
                python::OverlayVideoExportState state;
                const auto& tasks = gm->asyncTasks();
                state.active = tasks.isExportingVideo();
                state.progress = tasks.getVideoExportProgress();
                state.current_frame = tasks.getVideoExportCurrentFrame();
                state.total_frames = tasks.getVideoExportTotalFrames();
                state.stage = tasks.getVideoExportStage();
                return state;
            },
            []() {
                if (auto* gm = python::get_gui_manager())
                    gm->asyncTasks().cancelVideoExport();
            });

        // Section drawing callbacks (for Python-first UI)
        python::set_section_draw_callbacks({
            .draw_tools_section = []() {
                auto* gm = python::get_gui_manager();
                if (!gm)
                    return;
                auto* viewer = gm->getViewer();
                if (!viewer)
                    return;
                gui::UIContext ctx{
                    .viewer = viewer,
                    .file_browser = nullptr,
                    .window_states = nullptr,
                    .editor = python::get_editor_context(),
                    .sequencer_controller = nullptr,
                    .fonts = {}};
                gui::panels::DrawToolsPanel(ctx); },
            .draw_console_button = []() {
                auto* gm = python::get_gui_manager();
                if (!gm)
                    return;
                auto* viewer = gm->getViewer();
                if (!viewer)
                    return;
                gui::UIContext ctx{
                    .viewer = viewer,
                    .file_browser = nullptr,
                    .window_states = gm->getWindowStates(),
                    .editor = python::get_editor_context(),
                    .sequencer_controller = nullptr,
                    .fonts = {}};
                gui::panels::DrawSystemConsoleButton(ctx); },
        });

        // Sequencer timeline callbacks
        python::set_sequencer_timeline_callbacks(
            []() -> bool {
                auto* gm = python::get_gui_manager();
                return gm ? !gm->sequencer().timeline().empty() : false;
            },
            [](const std::string& path) -> bool {
                auto* gm = python::get_gui_manager();
                return gm ? gm->sequencer().timeline().saveToJson(path) : false;
            },
            [](const std::string& path) -> bool {
                auto* gm = python::get_gui_manager();
                return gm ? gm->sequencer().timeline().loadFromJson(path) : false;
            },
            []() {
                if (auto* gm = python::get_gui_manager())
                    gm->sequencer().timeline().clear();
            },
            [](float speed) {
                if (auto* gm = python::get_gui_manager())
                    gm->sequencer().setPlaybackSpeed(speed);
            });

        // Sequencer UI state callback - provides read access to C++ state
        // Returns a copy of the current C++ state for Python to read
        // Note: Python modifications require calling set_sequencer_state() to persist
        python::set_sequencer_ui_state_callback([]() -> python::SequencerUIStateData* {
            auto* gm = python::get_gui_manager();
            if (!gm)
                return nullptr;

            // Return address of the actual C++ state (first 6 fields match layout)
            // This works because SequencerUIStateData has compatible layout for the exposed fields
            auto& state = gm->getSequencerUIState();
            static python::SequencerUIStateData s_state;
            s_state.show_camera_path = state.show_camera_path;
            s_state.snap_to_grid = state.snap_to_grid;
            s_state.snap_interval = state.snap_interval;
            s_state.playback_speed = state.playback_speed;
            s_state.follow_playback = state.follow_playback;
            s_state.pip_preview_scale = state.pip_preview_scale;
            return &s_state;
        });

        python::set_pivot_mode_callbacks(
            []() -> int {
                const auto* gm = python::get_gui_manager();
                return gm ? static_cast<int>(gm->gizmo().getPivotMode()) : 0;
            },
            [](int mode) {
                if (auto* gm = python::get_gui_manager())
                    gm->gizmo().setPivotMode(static_cast<PivotMode>(mode));
            });
        python::set_transform_space_callbacks(
            []() -> int {
                const auto* gm = python::get_gui_manager();
                return gm ? static_cast<int>(gm->gizmo().getTransformSpace()) : 0;
            },
            [](int space) {
                if (auto* gm = python::get_gui_manager())
                    gm->gizmo().setTransformSpace(static_cast<TransformSpace>(space));
            });
        python::set_thumbnail_callbacks(
            [](const char* video_id) {
                if (auto* gm = python::get_gui_manager())
                    gm->requestThumbnail(video_id);
            },
            []() {
                if (auto* gm = python::get_gui_manager())
                    gm->processThumbnails();
            },
            [](const char* video_id) -> bool {
                const auto* gm = python::get_gui_manager();
                return gm ? gm->isThumbnailReady(video_id) : false;
            },
            [](const char* video_id) -> uint64_t {
                const auto* gm = python::get_gui_manager();
                return gm ? gm->getThumbnailTexture(video_id) : 0;
            });
        python::set_scene_manager(scene_manager_.get());

        python::set_export_callback([](int format, const char* path, const char** node_names,
                                       int node_count, int sh_degree) {
            if (auto* gm = python::get_gui_manager()) {
                std::vector<std::string> names;
                names.reserve(node_count);
                for (int i = 0; i < node_count; ++i) {
                    names.emplace_back(node_names[i]);
                }
                gm->asyncTasks().performExport(static_cast<lfs::core::ExportFormat>(format),
                                               std::filesystem::path(path), names, sh_degree);
            }
        });

        // Setup connections
        setupEventHandlers();
        setupComponentConnections();
    }

    VisualizerImpl::~VisualizerImpl() {
        // Clear event handlers before destroying components to prevent use-after-free
        lfs::core::event::bus().clear_all();
        services().clear();

        // Clear operator system
        op::unregisterEditOperators();
        op::unregisterBrushOperators();
        op::unregisterSelectionOperators();
        op::unregisterAlignOperators();
        op::unregisterTransformOperators();
        op::operators().clear();

        python::set_sequencer_callbacks(nullptr, nullptr);
        python::set_overlay_callbacks(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        python::set_section_draw_callbacks({});
        python::set_sequencer_ui_state_callback(nullptr);
        python::set_pivot_mode_callbacks(nullptr, nullptr);
        python::set_transform_space_callbacks(nullptr, nullptr);
        python::set_selected_camera_callback(nullptr);
        python::set_invert_masks_callback(nullptr);
        python::set_gui_manager(nullptr);
        trainer_manager_.reset();
        brush_tool_.reset();
        tool_context_.reset();
        if (gui_manager_) {
            gui_manager_->shutdown();
        }
        LOG_DEBUG("Visualizer destroyed");
    }

    void VisualizerImpl::initializeTools() {
        if (tools_initialized_) {
            LOG_TRACE("Tools already initialized, skipping");
            return;
        }

        tool_context_ = std::make_unique<ToolContext>(
            rendering_manager_.get(),
            scene_manager_.get(),
            &viewport_,
            window_manager_->getWindow());

        // Connect tool context to input controller
        if (input_controller_) {
            input_controller_->setToolContext(tool_context_.get());
        }

        brush_tool_ = std::make_shared<tools::BrushTool>();
        if (!brush_tool_->initialize(*tool_context_)) {
            LOG_ERROR("Failed to initialize brush tool");
            brush_tool_.reset();
        } else if (input_controller_) {
            input_controller_->setBrushTool(brush_tool_);
        }

        align_tool_ = std::make_shared<tools::AlignTool>();
        if (!align_tool_->initialize(*tool_context_)) {
            LOG_ERROR("Failed to initialize align tool");
            align_tool_.reset();
        } else if (input_controller_) {
            input_controller_->setAlignTool(align_tool_);
        }

        selection_tool_ = std::make_shared<tools::SelectionTool>();
        if (!selection_tool_->initialize(*tool_context_)) {
            LOG_ERROR("Failed to initialize selection tool");
            selection_tool_.reset();
        } else if (input_controller_) {
            input_controller_->setSelectionTool(selection_tool_);
            selection_tool_->setInputBindings(&input_controller_->getBindings());
        }

        tools_initialized_ = true;
    }

    void VisualizerImpl::setupComponentConnections() {
        // Set up main loop callbacks
        main_loop_->setInitCallback([this]() { return initialize(); });
        main_loop_->setUpdateCallback([this]() { update(); });
        main_loop_->setRenderCallback([this]() { render(); });
        main_loop_->setShutdownCallback([this]() { shutdown(); });
        main_loop_->setShouldCloseCallback([this]() { return allowclose(); });

        gui_manager_->setFileSelectedCallback([this](const std::filesystem::path& path, bool is_dataset) {
            lfs::core::events::cmd::LoadFile{.path = path, .is_dataset = is_dataset}.emit();
        });
    }

    void VisualizerImpl::setupEventHandlers() {
        using namespace lfs::core::events;

        // NOTE: Training control commands (Start/Pause/Resume/Stop/SaveCheckpoint)
        // are now handled by TrainerManager::setupEventHandlers()

        cmd::ResetTraining::when([this](const auto&) {
            if (!scene_manager_ || !scene_manager_->hasDataset()) {
                LOG_WARN("Cannot reset: no dataset");
                return;
            }
            if (trainer_manager_ && trainer_manager_->isTrainingActive()) {
                pending_reset_ = true;
                trainer_manager_->stopTraining();
                return;
            }
            performReset();
        });

        cmd::ClearScene::when([this](const auto&) {
            if (auto* const param_mgr = services().paramsOrNull()) {
                param_mgr->resetToDefaults();
            }
        });

        // Undo/Redo commands (require command_history_ which lives here)
        cmd::Undo::when([this](const auto&) { undo(); });
        cmd::Redo::when([this](const auto&) { redo(); });

        // Selection operations (require command_history_ and tools)
        cmd::DeleteSelected::when([this](const auto&) { deleteSelectedGaussians(); });
        cmd::InvertSelection::when([this](const auto&) { invertSelection(); });
        cmd::DeselectAll::when([this](const auto&) { deselectAll(); });
        cmd::SelectAll::when([this](const auto&) { selectAll(); });
        cmd::CopySelection::when([this](const auto&) { copySelection(); });
        cmd::PasteSelection::when([this](const auto&) { pasteSelection(); });
        cmd::SelectRect::when([this](const auto& e) { selectRect(e.x0, e.y0, e.x1, e.y1, e.mode); });
        cmd::ApplySelectionMask::when([this](const auto& e) { applySelectionMask(e.mask); });

        // NOTE: ui::RenderSettingsChanged, ui::CameraMove, state::SceneChanged,
        // ui::PointCloudModeChanged are handled by RenderingManager::setupEventHandlers()

        // Window redraw requests on scene/mode changes
        state::SceneChanged::when([this](const auto&) {
            if (window_manager_) {
                window_manager_->requestRedraw();
            }
        });

        ui::PointCloudModeChanged::when([this](const auto&) {
            if (window_manager_) {
                window_manager_->requestRedraw();
            }
        });

        ui::AppearanceModelLoaded::when([this](const auto& e) {
            if (rendering_manager_) {
                auto settings = rendering_manager_->getSettings();
                settings.apply_appearance_correction = true;
                settings.ppisp_mode =
                    e.has_controller ? RenderSettings::PPISPMode::AUTO : RenderSettings::PPISPMode::MANUAL;
                rendering_manager_->updateSettings(settings);
            }
        });

        // Trainer ready signal
        internal::TrainerReady::when([this](const auto&) {
            internal::TrainingReadyToStart{}.emit();
        });

        // Training started - switch to splat rendering and select training model
        state::TrainingStarted::when([this](const auto&) {
            ui::PointCloudModeChanged{
                .enabled = false,
                .voxel_size = 0.03f}
                .emit();

            // Select the training model so it's visible
            if (scene_manager_) {
                const auto& scene = scene_manager_->getScene();
                const auto& model_name = scene.getTrainingModelNodeName();
                if (!model_name.empty()) {
                    scene_manager_->selectNode(model_name);
                    LOG_INFO("Selected training model '{}' for training", model_name);
                }
            }

            LOG_INFO("Switched to splat rendering mode (training started)");
        });

        // Training completed - update content type
        state::TrainingCompleted::when([this](const auto& event) {
            handleTrainingCompleted(event);
        });

        // File loading commands
        cmd::LoadFile::when([this](const auto& cmd) {
            handleLoadFileCommand(cmd);
        });

        cmd::LoadConfigFile::when([this](const auto& cmd) {
            handleLoadConfigFile(cmd.path);
        });

        // RequestExit handled by Python file_menu.py

        cmd::ForceExit::when([this](const auto&) {
            if (gui_manager_) {
                gui_manager_->setForceExit(true);
            }
            if (window_manager_ && window_manager_->getWindow()) {
                glfwSetWindowShouldClose(window_manager_->getWindow(), GLFW_TRUE);
            }
        });

        cmd::SwitchToLatestCheckpoint::when([this](const auto&) {
            handleSwitchToLatestCheckpoint();
        });

        // Signal bridge event handlers
        state::TrainingProgress::when([](const auto& event) {
            python::update_training_progress(event.iteration, event.loss, event.num_gaussians);
        });

        state::TrainingStarted::when([this](const auto& event) {
            python::update_trainer_loaded(true, event.total_iterations);
            python::update_training_state(true, "running");
        });

        state::TrainingPaused::when([](const auto&) {
            python::update_training_state(false, "paused");
        });

        state::TrainingResumed::when([](const auto&) {
            python::update_training_state(true, "running");
        });

        state::TrainingCompleted::when([](const auto& event) {
            const char* state = !event.success       ? "error"
                                : event.user_stopped ? "stopped"
                                                     : "completed";
            python::update_training_state(false, state);
        });

        internal::TrainerReady::when([this](const auto&) {
            python::update_trainer_loaded(true, trainer_manager_->getTotalIterations());
            python::update_training_state(false, "ready");
        });

        state::EvaluationCompleted::when([](const auto& event) {
            python::update_psnr(event.psnr);
        });

        state::SceneLoaded::when([](const auto& event) {
            python::update_scene(true, event.path.string().c_str());
        });

        state::SceneCleared::when([](const auto&) {
            python::update_scene(false, "");
        });

        state::SelectionChanged::when([](const auto& event) {
            python::update_selection(event.has_selection, event.count);
        });
    }

    bool VisualizerImpl::initialize() {
        // Track if we're fully initialized
        static bool fully_initialized = false;
        if (fully_initialized) {
            LOG_TRACE("Already fully initialized");
            return true;
        }

        // Initialize Python early so signal bridge is ready before trainer creation
        python::ensure_initialized();

        // Initialize window first and ensure it has proper size
        if (!window_initialized_) {
            if (!window_manager_->init()) {
                return false;
            }
            window_initialized_ = true;

            // Poll events to get actual window dimensions
            window_manager_->pollEvents();
            window_manager_->updateWindowSize();

            // Update viewport with actual window size
            viewport_.windowSize = window_manager_->getWindowSize();
            viewport_.frameBufferSize = window_manager_->getFramebufferSize();

            // Validate we got reasonable dimensions
            if (viewport_.windowSize.x <= 0 || viewport_.windowSize.y <= 0) {
                LOG_WARN("Window manager returned invalid size, using options fallback: {}x{}",
                         options_.width, options_.height);
                viewport_.windowSize = glm::ivec2(options_.width, options_.height);
                viewport_.frameBufferSize = glm::ivec2(options_.width, options_.height);
            }

            LOG_DEBUG("Window initialized with actual size: {}x{}",
                      viewport_.windowSize.x, viewport_.windowSize.y);
        }

        // Initialize GUI (sets up ImGui callbacks)
        if (!gui_initialized_) {
            gui_manager_->init();
            gui_initialized_ = true;
        }

        // Create simplified input controller AFTER ImGui is initialized
        // NOTE: InputController uses services() for TrainerManager, RenderingManager, GuiManager
        if (!input_controller_) {
            input_controller_ = std::make_unique<InputController>(
                window_manager_->getWindow(), viewport_);
            input_controller_->initialize();
            python::set_keymap_bindings(&input_controller_->getBindings());
        }

        // Initialize rendering with proper viewport dimensions
        if (!rendering_manager_->isInitialized()) {
            // Pass viewport dimensions to rendering manager
            rendering_manager_->setInitialViewportSize(viewport_.windowSize);
            rendering_manager_->initialize();
        }

        // Initialize tools AFTER rendering is initialized (only once!)
        if (!tools_initialized_) {
            initializeTools();
        }

        // Start IPC server for MCP selection commands
        if (!selection_server_) {
            selection_server_ = std::make_unique<SelectionServer>();
            selection_server_->start();
            if (rendering_manager_) {
                rendering_manager_->setOutputScreenPositions(true);
            }

            // Set up view callback for Python rendering API
            vis::set_view_callback([this]() -> std::optional<vis::ViewInfo> {
                if (!rendering_manager_)
                    return std::nullopt;

                const auto& settings = rendering_manager_->getSettings();
                const auto R = viewport_.getRotationMatrix();
                const auto T = viewport_.getTranslation();

                vis::ViewInfo info;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        info.rotation[i * 3 + j] = R[j][i];
                info.translation = {T.x, T.y, T.z};
                info.width = viewport_.windowSize.x;
                info.height = viewport_.windowSize.y;
                info.fov = lfs::rendering::focalLengthToVFov(settings.focal_length_mm);
                return info;
            });

            // Set up viewport render callback for Python rendering API
            vis::set_viewport_render_callback([this]() -> std::optional<vis::ViewportRender> {
                if (!rendering_manager_)
                    return std::nullopt;

                const auto& result = rendering_manager_->getCachedResult();
                if (!result.valid || !result.image)
                    return std::nullopt;

                return vis::ViewportRender{result.image, result.screen_positions};
            });

            vis::set_render_settings_callbacks(
                [this]() -> std::optional<vis::RenderSettingsProxy> {
                    if (!rendering_manager_)
                        return std::nullopt;

                    const auto& s = rendering_manager_->getSettings();
                    vis::RenderSettingsProxy proxy;
                    proxy.focal_length_mm = s.focal_length_mm;
                    proxy.scaling_modifier = s.scaling_modifier;
                    proxy.antialiasing = s.antialiasing;
                    proxy.mip_filter = s.mip_filter;
                    proxy.sh_degree = s.sh_degree;
                    proxy.render_scale = s.render_scale;
                    proxy.show_crop_box = s.show_crop_box;
                    proxy.use_crop_box = s.use_crop_box;
                    proxy.desaturate_unselected = s.desaturate_unselected;
                    proxy.desaturate_cropping = s.desaturate_cropping;
                    proxy.background_color = {s.background_color.r, s.background_color.g, s.background_color.b};
                    proxy.show_coord_axes = s.show_coord_axes;
                    proxy.axes_size = s.axes_size;
                    proxy.show_grid = s.show_grid;
                    proxy.grid_plane = s.grid_plane;
                    proxy.grid_opacity = s.grid_opacity;
                    proxy.point_cloud_mode = s.point_cloud_mode;
                    proxy.voxel_size = s.voxel_size;
                    proxy.show_rings = s.show_rings;
                    proxy.ring_width = s.ring_width;
                    proxy.show_center_markers = s.show_center_markers;
                    proxy.show_camera_frustums = s.show_camera_frustums;
                    proxy.camera_frustum_scale = s.camera_frustum_scale;
                    proxy.train_camera_color = {s.train_camera_color.r, s.train_camera_color.g, s.train_camera_color.b};
                    proxy.eval_camera_color = {s.eval_camera_color.r, s.eval_camera_color.g, s.eval_camera_color.b};
                    proxy.show_pivot = s.show_pivot;
                    proxy.split_position = s.split_position;
                    proxy.gut = s.gut;
                    proxy.equirectangular = s.equirectangular;
                    proxy.orthographic = s.orthographic;
                    proxy.ortho_scale = s.ortho_scale;
                    proxy.selection_color_committed = {s.selection_color_committed.r, s.selection_color_committed.g, s.selection_color_committed.b};
                    proxy.selection_color_preview = {s.selection_color_preview.r, s.selection_color_preview.g, s.selection_color_preview.b};
                    proxy.selection_color_center_marker = {s.selection_color_center_marker.r, s.selection_color_center_marker.g, s.selection_color_center_marker.b};
                    proxy.depth_clip_enabled = s.depth_clip_enabled;
                    proxy.depth_clip_far = s.depth_clip_far;
                    proxy.apply_appearance_correction = s.apply_appearance_correction;
                    proxy.ppisp_mode = static_cast<int>(s.ppisp_mode);
                    proxy.ppisp = s.ppisp_overrides;
                    return proxy;
                },
                [this](const vis::RenderSettingsProxy& proxy) {
                    if (!rendering_manager_)
                        return;

                    auto s = rendering_manager_->getSettings();
                    s.focal_length_mm = proxy.focal_length_mm;
                    s.scaling_modifier = proxy.scaling_modifier;
                    s.antialiasing = proxy.antialiasing;
                    s.mip_filter = proxy.mip_filter;
                    s.sh_degree = proxy.sh_degree;
                    s.render_scale = proxy.render_scale;
                    s.show_crop_box = proxy.show_crop_box;
                    s.use_crop_box = proxy.use_crop_box;
                    s.desaturate_unselected = proxy.desaturate_unselected;
                    s.desaturate_cropping = proxy.desaturate_cropping;
                    s.background_color = glm::vec3(proxy.background_color[0], proxy.background_color[1], proxy.background_color[2]);
                    s.show_coord_axes = proxy.show_coord_axes;
                    s.axes_size = proxy.axes_size;
                    s.show_grid = proxy.show_grid;
                    s.grid_plane = proxy.grid_plane;
                    s.grid_opacity = proxy.grid_opacity;
                    s.point_cloud_mode = proxy.point_cloud_mode;
                    s.voxel_size = proxy.voxel_size;
                    s.show_rings = proxy.show_rings;
                    s.ring_width = proxy.ring_width;
                    s.show_center_markers = proxy.show_center_markers;
                    s.show_camera_frustums = proxy.show_camera_frustums;
                    s.camera_frustum_scale = proxy.camera_frustum_scale;
                    s.train_camera_color = glm::vec3(proxy.train_camera_color[0], proxy.train_camera_color[1], proxy.train_camera_color[2]);
                    s.eval_camera_color = glm::vec3(proxy.eval_camera_color[0], proxy.eval_camera_color[1], proxy.eval_camera_color[2]);
                    s.show_pivot = proxy.show_pivot;
                    s.split_position = proxy.split_position;
                    s.gut = proxy.gut;
                    s.equirectangular = proxy.equirectangular;
                    s.orthographic = proxy.orthographic;
                    s.ortho_scale = proxy.ortho_scale;
                    s.selection_color_committed = glm::vec3(proxy.selection_color_committed[0], proxy.selection_color_committed[1], proxy.selection_color_committed[2]);
                    s.selection_color_preview = glm::vec3(proxy.selection_color_preview[0], proxy.selection_color_preview[1], proxy.selection_color_preview[2]);
                    s.selection_color_center_marker = glm::vec3(proxy.selection_color_center_marker[0], proxy.selection_color_center_marker[1], proxy.selection_color_center_marker[2]);
                    s.depth_clip_enabled = proxy.depth_clip_enabled;
                    s.depth_clip_far = proxy.depth_clip_far;
                    s.apply_appearance_correction = proxy.apply_appearance_correction;
                    s.ppisp_mode = static_cast<vis::RenderSettings::PPISPMode>(proxy.ppisp_mode);
                    s.ppisp_overrides = proxy.ppisp;
                    rendering_manager_->updateSettings(s);
                });

            // Set up generic capability invocation callback (runs on IPC thread, waits for main thread)
            selection_server_->setInvokeCapabilityCallback(
                [this](const std::string& name, const std::string& args) -> CapabilityInvokeResult {
                    std::mutex mtx;
                    std::condition_variable cv;
                    CapabilityInvokeResult result;
                    bool done = false;

                    // Queue request for main thread
                    {
                        std::lock_guard lock(capability_request_mutex_);
                        pending_capability_request_ = CapabilityRequest{name, args, &result, &mtx, &cv, &done};
                    }

                    // Wait for main thread to process
                    std::unique_lock lock(mtx);
                    cv.wait(lock, [&done] { return done; });

                    return result;
                });
        }

        // Create selection service
        if (!selection_service_) {
            selection_service_ = std::make_unique<SelectionService>(scene_manager_.get(), rendering_manager_.get());
            python::set_selection_service(selection_service_.get());
        }

        fully_initialized = true;
        return true;
    }

    void VisualizerImpl::update() {
        window_manager_->updateWindowSize();

        // Process MCP selection commands from the IPC server
        if (selection_server_) {
            selection_server_->process_pending_commands();
        }

        // Process pending capability request from IPC thread
        {
            std::lock_guard lock(capability_request_mutex_);
            if (pending_capability_request_) {
                auto& req = *pending_capability_request_;
                *req.result = processCapabilityRequest(req.name, req.args);

                // Signal completion
                {
                    std::lock_guard done_lock(*req.mtx);
                    *req.done = true;
                }
                req.cv->notify_one();
                pending_capability_request_.reset();
            }
        }

        if (gui_manager_) {
            const auto& size = gui_manager_->getViewportSize();
            viewport_.windowSize = {static_cast<int>(size.x), static_cast<int>(size.y)};
        } else {
            viewport_.windowSize = window_manager_->getWindowSize();
        }
        viewport_.frameBufferSize = window_manager_->getFramebufferSize();

        // Update editor context state from scene/trainer
        editor_context_.update(scene_manager_.get(), trainer_manager_.get());

        if (brush_tool_ && brush_tool_->isEnabled() && tool_context_) {
            brush_tool_->update(*tool_context_);
        }
        if (selection_tool_ && selection_tool_->isEnabled() && tool_context_) {
            selection_tool_->update(*tool_context_);
        }

        if (pending_reset_ && trainer_manager_ && !trainer_manager_->isTrainingActive()) {
            pending_reset_ = false;
            trainer_manager_->waitForCompletion();
            performReset();
        }

        // Auto-start training if --train flag was passed
        if (pending_auto_train_ && trainer_manager_ && trainer_manager_->canStart()) {
            pending_auto_train_ = false;
            LOG_INFO("Auto-starting training (--train flag)");
            cmd::StartTraining{}.emit();
        }
    }

    void VisualizerImpl::render() {
        // Calculate delta time for input updates
        static auto last_frame_time = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        float delta_time = std::chrono::duration<float>(now - last_frame_time).count();
        last_frame_time = now;

        // Clamp delta time to prevent huge jumps (min 30 FPS)
        delta_time = std::min(delta_time, 1.0f / 30.0f);

        // Tick Python frame callback for animations
        if (python::has_frame_callback()) {
            python::tick_frame_callback(delta_time);
            if (rendering_manager_) {
                rendering_manager_->markDirty();
            }
        }

        // Update input controller with viewport bounds
        if (gui_manager_) {
            auto pos = gui_manager_->getViewportPos();
            auto size = gui_manager_->getViewportSize();
            input_controller_->updateViewportBounds(pos.x, pos.y, size.x, size.y);
            if (tool_context_) {
                tool_context_->updateViewportBounds(pos.x, pos.y, size.x, size.y);
            }
        }

        // Update point cloud mode in input controller
        auto* rendering_manager = getRenderingManager();
        if (rendering_manager) {
            const auto& settings = rendering_manager->getSettings();
            input_controller_->setPointCloudMode(settings.point_cloud_mode);
        }

        if (input_controller_) {
            input_controller_->update(delta_time);
        }

        // Get viewport region from GUI
        ViewportRegion viewport_region;
        bool has_viewport_region = false;
        if (gui_manager_) {
            ImVec2 pos = gui_manager_->getViewportPos();
            ImVec2 size = gui_manager_->getViewportSize();

            viewport_region.x = pos.x;
            viewport_region.y = pos.y;
            viewport_region.width = size.x;
            viewport_region.height = size.y;

            has_viewport_region = true;
        }

        // viewport_region accounts for toolbar offset - required for all render modes
        RenderingManager::RenderContext context{
            .viewport = viewport_,
            .settings = rendering_manager_->getSettings(),
            .viewport_region = has_viewport_region ? &viewport_region : nullptr,
            .has_focus = gui_manager_ && gui_manager_->isViewportFocused(),
            .scene_manager = scene_manager_.get()};

        if (gui_manager_) {
            rendering_manager_->setCropboxGizmoActive(gui_manager_->gizmo().isCropboxGizmoActive());
            rendering_manager_->setEllipsoidGizmoActive(gui_manager_->gizmo().isEllipsoidGizmoActive());
        }

        rendering_manager_->renderFrame(context, scene_manager_.get());
        gui_manager_->render();
        window_manager_->swapBuffers();

        python::flush_signals();

        // Render-on-demand: VSync handles frame pacing, waitEvents saves CPU when idle
        const bool is_training = trainer_manager_ && trainer_manager_->isRunning();
        const bool needs_render = rendering_manager_->needsRender();
        const bool continuous_input = input_controller_ && input_controller_->isContinuousInputActive();
        const bool has_python_animation = python::has_frame_callback();
        const bool needs_gui_animation = gui_manager_ && gui_manager_->needsAnimationFrame();

        if (needs_render || continuous_input || has_python_animation || needs_gui_animation) {
            window_manager_->pollEvents();
        } else if (is_training) {
            // Training: longer wait to reduce GPU load and memory fragmentation
            constexpr double TRAINING_WAIT_SEC = 0.1; // ~10 Hz
            window_manager_->waitEvents(TRAINING_WAIT_SEC);
        } else {
            // Idle: long wait to minimize CPU usage (VSync still applies on wake)
            constexpr double IDLE_WAIT_SEC = 0.5;
            window_manager_->waitEvents(IDLE_WAIT_SEC);
        }
    }

    bool VisualizerImpl::allowclose() {
        if (!window_manager_->shouldClose()) {
            return false;
        }

        if (!gui_manager_) {
            return true;
        }

        if (gui_manager_->isForceExit()) {
#ifdef WIN32
            const HWND hwnd = GetConsoleWindow();
            Sleep(1);
            const HWND owner = GetWindow(hwnd, GW_OWNER);
            DWORD process_id = 0;
            GetWindowThreadProcessId(hwnd, &process_id);
            if (GetCurrentProcessId() != process_id) {
                ShowWindow(owner ? owner : hwnd, SW_SHOW);
            }
#endif
            return true;
        }

        if (!gui_manager_->isExitConfirmationPending()) {
            gui_manager_->requestExitConfirmation();
        }
        window_manager_->cancelClose();
        return false;
    }

    void VisualizerImpl::shutdown() {
        // Stop training before GPU resources are freed
        if (trainer_manager_) {
            if (trainer_manager_->isTrainingActive()) {
                trainer_manager_->stopTraining();
                trainer_manager_->waitForCompletion();
            }
            trainer_manager_.reset();
        }

        // Shutdown tools
        if (brush_tool_) {
            brush_tool_->shutdown();
            brush_tool_.reset();
        }

        // Clean up tool context
        tool_context_.reset();

        op::undoHistory().clear();

        tools_initialized_ = false;
    }

    void VisualizerImpl::undo() {
        op::undoHistory().undo();
        if (rendering_manager_) {
            rendering_manager_->markDirty();
        }
    }

    void VisualizerImpl::redo() {
        op::undoHistory().redo();
        if (rendering_manager_) {
            rendering_manager_->markDirty();
        }
    }

    void VisualizerImpl::deleteSelectedGaussians() {
        if (!scene_manager_)
            return;

        auto& scene = scene_manager_->getScene();
        auto selection = scene.getSelectionMask();

        if (!selection || !selection->is_valid()) {
            LOG_INFO("No Gaussians selected to delete");
            return;
        }

        auto nodes = scene.getVisibleNodes();
        if (nodes.empty())
            return;

        auto entry = std::make_unique<op::SceneSnapshot>(*scene_manager_, "edit.delete");
        entry->captureTopology();
        entry->captureSelection();

        size_t offset = 0;
        bool any_deleted = false;

        for (const auto* node : nodes) {
            if (!node || !node->model)
                continue;

            const size_t node_size = node->model->size();
            if (node_size == 0)
                continue;

            auto node_selection = selection->slice(0, offset, offset + node_size);
            auto bool_mask = node_selection.to(lfs::core::DataType::Bool);
            node->model->soft_delete(bool_mask);

            any_deleted = true;
            offset += node_size;
        }

        if (any_deleted) {
            LOG_INFO("Deleted selected Gaussians");
            scene.markDirty();
            scene.clearSelection();

            entry->captureAfter();
            op::undoHistory().push(std::move(entry));

            if (rendering_manager_) {
                rendering_manager_->markDirty();
            }
        }
    }

    void VisualizerImpl::invertSelection() {
        if (!scene_manager_)
            return;
        auto& scene = scene_manager_->getScene();
        const size_t total = scene.getTotalGaussianCount();
        if (total == 0)
            return;

        auto entry = std::make_unique<op::SceneSnapshot>(*scene_manager_, "select.invert");
        entry->captureSelection();

        const auto old_mask = scene.getSelectionMask();
        const auto ones = lfs::core::Tensor::ones({total}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
        auto new_mask = std::make_shared<lfs::core::Tensor>(
            (old_mask && old_mask->is_valid()) ? ones - *old_mask : ones);

        scene.setSelectionMask(new_mask);

        entry->captureAfter();
        op::undoHistory().push(std::move(entry));

        if (rendering_manager_)
            rendering_manager_->markDirty();
    }

    void VisualizerImpl::deselectAll() {
        if (!scene_manager_)
            return;
        auto& scene = scene_manager_->getScene();
        if (!scene.hasSelection())
            return;

        auto entry = std::make_unique<op::SceneSnapshot>(*scene_manager_, "select.none");
        entry->captureSelection();

        scene.clearSelection();

        entry->captureAfter();
        op::undoHistory().push(std::move(entry));

        if (rendering_manager_)
            rendering_manager_->markDirty();
    }

    void VisualizerImpl::selectAll() {
        if (!scene_manager_)
            return;

        const auto tool = editor_context_.getActiveTool();
        const bool is_selection_tool = (tool == ToolType::Selection || tool == ToolType::Brush);

        if (is_selection_tool) {
            auto& scene = scene_manager_->getScene();
            const size_t total = scene.getTotalGaussianCount();
            if (total == 0)
                return;

            const auto& selected_name = scene_manager_->getSelectedNodeName();
            if (selected_name.empty())
                return;

            const int node_index = scene.getVisibleNodeIndex(selected_name);
            if (node_index < 0)
                return;

            const auto transform_indices = scene.getTransformIndices();
            if (!transform_indices || transform_indices->numel() != total)
                return;

            auto entry = std::make_unique<op::SceneSnapshot>(*scene_manager_, "select.all");
            entry->captureSelection();

            auto new_mask = std::make_shared<lfs::core::Tensor>(transform_indices->eq(node_index));
            scene.setSelectionMask(new_mask);

            entry->captureAfter();
            op::undoHistory().push(std::move(entry));
        } else {
            const auto& scene = scene_manager_->getScene();
            const auto nodes = scene.getNodes();
            std::vector<std::string> splat_names;
            splat_names.reserve(nodes.size());
            for (const auto* node : nodes) {
                if (node->type == NodeType::SPLAT) {
                    splat_names.push_back(node->name);
                }
            }
            if (!splat_names.empty()) {
                scene_manager_->selectNodes(splat_names);
            }
        }
        if (rendering_manager_)
            rendering_manager_->markDirty();
    }

    void VisualizerImpl::copySelection() {
        if (!scene_manager_)
            return;

        const auto tool = editor_context_.getActiveTool();
        const bool is_selection_tool = (tool == ToolType::Selection || tool == ToolType::Brush);

        if (is_selection_tool && scene_manager_->getScene().hasSelection()) {
            scene_manager_->copySelectedGaussians();
        } else {
            scene_manager_->copySelectedNodes();
        }
    }

    void VisualizerImpl::pasteSelection() {
        if (!scene_manager_)
            return;

        const auto pasted = scene_manager_->hasGaussianClipboard()
                                ? scene_manager_->pasteGaussians()
                                : scene_manager_->pasteNodes();

        if (pasted.empty())
            return;

        // Selection tool state is managed by Python operator
        scene_manager_->getScene().resetSelectionState();

        scene_manager_->clearSelection();
        for (const auto& name : pasted) {
            scene_manager_->addToSelection(name);
        }

        if (rendering_manager_)
            rendering_manager_->markDirty();
    }

    void VisualizerImpl::selectRect(float x0, float y0, float x1, float y1, const std::string& mode) {
        if (!selection_service_)
            return;

        SelectionMode sel_mode = SelectionMode::Replace;
        if (mode == "add")
            sel_mode = SelectionMode::Add;
        else if (mode == "remove")
            sel_mode = SelectionMode::Remove;

        selection_service_->selectRect(x0, y0, x1, y1, sel_mode);
    }

    void VisualizerImpl::applySelectionMask(const std::vector<uint8_t>& mask) {
        if (!selection_service_)
            return;

        selection_service_->applyMask(mask, SelectionMode::Replace);
    }

    void VisualizerImpl::run() {
        main_loop_->run();
    }

    void VisualizerImpl::setParameters(const lfs::core::param::TrainingParameters& params) {
        data_loader_->setParameters(params);
        if (parameter_manager_) {
            parameter_manager_->setSessionDefaults(params);
        }
        pending_auto_train_ = params.optimization.auto_train;
    }

    std::expected<void, std::string> VisualizerImpl::loadPLY(const std::filesystem::path& path) {
        LOG_TIMER("LoadPLY");

        // Ensure full initialization before loading PLY
        // This will only initialize once due to the guard in initialize()
        if (!initialize()) {
            return std::unexpected("Failed to initialize visualizer");
        }

        LOG_INFO("Loading PLY file: {}", lfs::core::path_to_utf8(path));
        return data_loader_->loadPLY(path);
    }

    std::expected<void, std::string> VisualizerImpl::addSplatFile(const std::filesystem::path& path) {
        if (!initialize()) {
            return std::unexpected("Failed to initialize visualizer");
        }
        try {
            data_loader_->addSplatFileToScene(path);
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to add splat file: {}", e.what()));
        }
    }

    std::expected<void, std::string> VisualizerImpl::loadDataset(const std::filesystem::path& path) {
        LOG_TIMER("LoadDataset");

        if (!initialize()) {
            return std::unexpected("Failed to initialize visualizer");
        }

        LOG_INFO("Loading dataset: {}", lfs::core::path_to_utf8(path));
        return data_loader_->loadDataset(path);
    }

    std::expected<void, std::string> VisualizerImpl::loadCheckpointForTraining(const std::filesystem::path& path) {
        LOG_TIMER("LoadCheckpointForTraining");

        // Ensure full initialization before loading checkpoint
        if (!initialize()) {
            return std::unexpected("Failed to initialize visualizer");
        }

        LOG_INFO("Loading checkpoint for training: {}", lfs::core::path_to_utf8(path));
        return data_loader_->loadCheckpointForTraining(path);
    }

    void VisualizerImpl::consolidateModels() {
        scene_manager_->consolidateNodeModels();
    }

    void VisualizerImpl::clearScene() {
        data_loader_->clearScene();
    }

    void VisualizerImpl::performReset() {
        assert(scene_manager_ && scene_manager_->hasDataset());

        const auto& path = scene_manager_->getDatasetPath();
        if (path.empty()) {
            LOG_ERROR("Cannot reset: empty path");
            return;
        }

        const auto& init_path = data_loader_->getParameters().init_path;
        if (auto* const param_mgr = services().paramsOrNull(); param_mgr && param_mgr->ensureLoaded()) {
            auto params = param_mgr->createForDataset(path, {});
            if (trainer_manager_) {
                params.dataset = trainer_manager_->getEditableDatasetParams();
                params.dataset.data_path = path;
                params.init_path = init_path;
            }
            data_loader_->setParameters(params);
        }

        if (const auto result = data_loader_->loadDataset(path); !result) {
            LOG_ERROR("Reset reload failed: {}", result.error());
        }
    }

    void VisualizerImpl::handleLoadFileCommand([[maybe_unused]] const lfs::core::events::cmd::LoadFile& cmd) {
        // File loading is handled by the data_loader_ service
    }

    void VisualizerImpl::handleLoadConfigFile(const std::filesystem::path& path) {
        auto result = lfs::core::param::read_optim_params_from_json(path);
        if (!result) {
            state::ConfigLoadFailed{.path = path, .error = result.error()}.emit();
            return;
        }
        parameter_manager_->importParams(*result);
    }

    void VisualizerImpl::handleTrainingCompleted([[maybe_unused]] const state::TrainingCompleted& event) {
        if (scene_manager_) {
            scene_manager_->changeContentType(SceneManager::ContentType::Dataset);
        }
    }

    void VisualizerImpl::handleSwitchToLatestCheckpoint() {
        LOG_WARN("Switch to latest checkpoint not implemented without project management");
    }

    CapabilityInvokeResult VisualizerImpl::processCapabilityRequest(const std::string& name, const std::string& args) {
        LOG_INFO("processCapabilityRequest: {} args={}", name, args);

        if (!scene_manager_) {
            LOG_WARN("processCapabilityRequest: scene_manager_ is NULL");
            return {false, "", "No scene available"};
        }

        python::SceneContextGuard ctx(&scene_manager_->getScene());
        auto result = python::invoke_capability(name, args);

        if (result.success && rendering_manager_) {
            rendering_manager_->markDirty();
        }

        return {result.success, result.result_json, result.error};
    }

} // namespace lfs::vis
