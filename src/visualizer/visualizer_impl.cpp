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
#include "ipc/render_settings_convert.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/align_ops.hpp"
#include "operator/ops/brush_ops.hpp"
#include "operator/ops/edit_ops.hpp"
#include "operator/ops/scene_ops.hpp"
#include "operator/ops/selection_ops.hpp"
#include "operator/ops/transform_ops.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "scene/scene_manager.hpp"
#include "tools/align_tool.hpp"
#include "tools/brush_tool.hpp"
#include "tools/builtin_tools.hpp"
#include "tools/selection_tool.hpp"
#include <io/filesystem_utils.hpp>
// clang-format off
#include <glad/glad.h>
// clang-format on
#include <SDL3/SDL_events.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#ifdef WIN32
#include <windows.h>
#endif

namespace lfs::vis {

    using namespace lfs::core::events;

    namespace {

        void wakeEventLoopViaServices() {
            if (auto* const window_manager = services().windowOrNull()) {
                window_manager->wakeEventLoop();
            }
        }
        std::optional<glm::mat3> buildValidatedViewRotation(const glm::vec3& eye,
                                                            const glm::vec3& target,
                                                            const glm::vec3& requested_up) {
            return lfs::rendering::tryMakeVisualizerLookAtRotation(eye, target, requested_up);
        }

    } // namespace

    VisualizerImpl::VisualizerImpl(const ViewerOptions& options)
        : options_(options),
          viewport_(options.width, options.height),
          window_manager_(std::make_unique<WindowManager>(options.title, options.width, options.height,
                                                          options.monitor_x, options.monitor_y,
                                                          options.monitor_width, options.monitor_height)) {
        viewer_thread_id_ = std::this_thread::get_id();

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
        op::registerSceneOperators();

        setupPythonBridge();
        setupEventHandlers();
        setupComponentConnections();
    }

    VisualizerImpl::~VisualizerImpl() {
        // Clear event handlers before destroying components to prevent use-after-free
        lfs::core::event::bus().clear_all();
        services().clear();

        // Clear operator system
        op::unregisterEditOperators();
        op::unregisterSceneOperators();
        op::unregisterBrushOperators();
        op::unregisterSelectionOperators();
        op::unregisterAlignOperators();
        op::unregisterTransformOperators();
        op::operators().clear();

        callback_cleanup_.clear();
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

    void VisualizerImpl::setupPythonBridge() {
        python::set_trainer_manager(trainer_manager_.get());
        callback_cleanup_.add([] { python::set_trainer_manager(nullptr); });
        python::set_parameter_manager(parameter_manager_.get());
        callback_cleanup_.add([] { python::set_parameter_manager(nullptr); });
        python::set_rendering_manager(rendering_manager_.get());
        callback_cleanup_.add([] { python::set_rendering_manager(nullptr); });
        python::set_editor_context(&editor_context_);
        callback_cleanup_.add([] { python::set_editor_context(nullptr); });
        python::set_operator_callbacks(&editor_context_);
        callback_cleanup_.add([] { python::set_operator_callbacks(nullptr); });
        python::set_gui_manager(gui_manager_.get());
        callback_cleanup_.add([] { python::set_gui_manager(nullptr); });
        python::set_main_loop_wake_callback(&wakeEventLoopViaServices);
        callback_cleanup_.add([] { python::set_main_loop_wake_callback(nullptr); });
        python::set_mesh2splat_callbacks(
            [](std::shared_ptr<core::MeshData> mesh, std::string name, core::Mesh2SplatOptions opts) {
                auto* gm = python::get_gui_manager();
                if (!gm)
                    return;
                gm->asyncTasks().startMesh2Splat(std::move(mesh), name, opts);
            },
            []() -> bool {
                auto* gm = python::get_gui_manager();
                return gm && gm->asyncTasks().isMesh2SplatActive();
            },
            []() -> float {
                auto* gm = python::get_gui_manager();
                return gm ? gm->asyncTasks().getMesh2SplatProgress() : 0.0f;
            },
            []() -> std::string {
                auto* gm = python::get_gui_manager();
                return gm ? gm->asyncTasks().getMesh2SplatStage() : std::string{};
            },
            []() -> std::string {
                auto* gm = python::get_gui_manager();
                return gm ? gm->asyncTasks().getMesh2SplatError() : std::string{};
            });
        callback_cleanup_.add([] { python::set_mesh2splat_callbacks(nullptr, nullptr, nullptr, nullptr, nullptr); });
        python::set_splat_simplify_callbacks(
            [](std::string name, core::SplatSimplifyOptions opts) {
                auto* gm = python::get_gui_manager();
                if (!gm)
                    return;
                gm->asyncTasks().startSplatSimplify(name, opts);
            },
            []() {
                auto* gm = python::get_gui_manager();
                if (gm)
                    gm->asyncTasks().cancelSplatSimplify();
            },
            []() -> bool {
                auto* gm = python::get_gui_manager();
                return gm && gm->asyncTasks().isSplatSimplifyActive();
            },
            []() -> float {
                auto* gm = python::get_gui_manager();
                return gm ? gm->asyncTasks().getSplatSimplifyProgress() : 0.0f;
            },
            []() -> std::string {
                auto* gm = python::get_gui_manager();
                return gm ? gm->asyncTasks().getSplatSimplifyStage() : std::string{};
            },
            []() -> std::string {
                auto* gm = python::get_gui_manager();
                return gm ? gm->asyncTasks().getSplatSimplifyError() : std::string{};
            });
        callback_cleanup_.add([] { python::set_splat_simplify_callbacks(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr); });
        python::set_selected_camera_callback([]() -> int {
            const auto* gm = python::get_gui_manager();
            return gm ? gm->getHighlightedCameraUid() : -1;
        });
        callback_cleanup_.add([] { python::set_selected_camera_callback(nullptr); });
        python::set_invert_masks_callback([]() -> bool {
            auto* pm = python::get_parameter_manager();
            return pm && pm->getActiveParams().invert_masks;
        });
        callback_cleanup_.add([] { python::set_invert_masks_callback(nullptr); });
        python::set_sequencer_callbacks(
            []() {
                const auto* gm = python::get_gui_manager();
                return gm ? gm->panelLayout().isShowSequencer() : false;
            },
            [](bool visible) {
                if (auto* gm = python::get_gui_manager())
                    gm->panelLayout().setShowSequencer(visible);
            });
        callback_cleanup_.add([] { python::set_sequencer_callbacks(nullptr, nullptr); });

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
                state.format = fmt == core::ExportFormat::PLY           ? "PLY"
                               : fmt == core::ExportFormat::SOG         ? "SOG"
                               : fmt == core::ExportFormat::SPZ         ? "SPZ"
                               : fmt == core::ExportFormat::HTML_VIEWER ? "HTML"
                               : fmt == core::ExportFormat::USD         ? "USD"
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
        callback_cleanup_.add([] { python::set_overlay_callbacks(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr); });

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
                    .window_states = gm->getWindowStates(),
                    .editor = python::get_editor_context(),
                    .sequencer_controller = nullptr,
                    .fonts = {}};
                gui::panels::DrawSystemConsoleButton(ctx); },
            .toggle_system_console = []() {
                auto* gm = python::get_gui_manager();
                if (!gm)
                    return;
                auto* viewer = gm->getViewer();
                if (!viewer)
                    return;
                gui::UIContext ctx{
                    .viewer = viewer,
                    .window_states = gm->getWindowStates(),
                    .editor = python::get_editor_context(),
                    .sequencer_controller = nullptr,
                    .fonts = {}};
                gui::panels::ToggleSystemConsole(ctx); },
        });
        callback_cleanup_.add([] { python::set_section_draw_callbacks({}); });

        python::set_sequencer_timeline_callbacks(
            []() -> bool {
                auto* gm = python::get_gui_manager();
                return gm ? (gm->sequencer().timeline().realKeyframeCount() > 0 ||
                             gm->sequencer().timeline().hasAnimationClip())
                          : false;
            },
            [](const std::string& path) -> bool {
                auto* gm = python::get_gui_manager();
                return gm ? gm->sequencer().saveToJson(path) : false;
            },
            [](const std::string& path) -> bool {
                auto* gm = python::get_gui_manager();
                if (!gm)
                    return false;
                const bool loaded = gm->sequencer().loadFromJson(path);
                if (loaded) {
                    lfs::core::events::state::KeyframeListChanged{
                        .count = gm->sequencer().timeline().realKeyframeCount()}
                        .emit();
                }
                return loaded;
            },
            []() {
                if (auto* gm = python::get_gui_manager()) {
                    gm->sequencer().clear();
                    lfs::core::events::state::KeyframeListChanged{.count = 0}.emit();
                }
            },
            [](float speed) {
                if (auto* gm = python::get_gui_manager()) {
                    gm->sequencer().setPlaybackSpeed(speed);
                    gm->getSequencerUIState().playback_speed = gm->sequencer().playbackSpeed();
                }
            });
        callback_cleanup_.add([] { python::set_sequencer_timeline_callbacks(nullptr, nullptr, nullptr, nullptr, nullptr); });

        sequencer_ui_state_ = std::make_unique<python::SequencerUIStateData>();
        python::set_sequencer_ui_state_callback([this]() -> python::SequencerUIStateData* {
            auto* gm = python::get_gui_manager();
            if (!gm)
                return nullptr;

            auto& state = gm->getSequencerUIState();
            auto& s = *sequencer_ui_state_;

            if (sequencer_ui_initialized_) {
                state.show_camera_path = s.show_camera_path;
                state.snap_to_grid = s.snap_to_grid;
                state.snap_interval = s.snap_interval;
                state.playback_speed = s.playback_speed;
                gm->sequencer().setPlaybackSpeed(state.playback_speed);
                state.playback_speed = gm->sequencer().playbackSpeed();
                state.follow_playback = s.follow_playback;
                state.show_pip_preview = s.show_pip_preview;
                state.pip_preview_scale = s.pip_preview_scale;
                state.show_film_strip = s.show_film_strip;
                state.equirectangular = s.equirectangular;
            }

            s.show_camera_path = state.show_camera_path;
            s.snap_to_grid = state.snap_to_grid;
            s.snap_interval = state.snap_interval;
            s.playback_speed = gm->sequencer().playbackSpeed();
            s.follow_playback = state.follow_playback;
            s.show_pip_preview = state.show_pip_preview;
            s.pip_preview_scale = state.pip_preview_scale;
            s.show_film_strip = state.show_film_strip;
            s.equirectangular = state.equirectangular;
            const auto sel = gm->sequencer().selectedKeyframe();
            s.selected_keyframe = sel.has_value() ? static_cast<int>(*sel) : -1;
            sequencer_ui_initialized_ = true;
            return &s;
        });
        callback_cleanup_.add([] { python::set_sequencer_ui_state_callback({}); });

        python::set_pivot_mode_callbacks(
            []() -> int {
                const auto* gm = python::get_gui_manager();
                return gm ? static_cast<int>(gm->gizmo().getPivotMode()) : 0;
            },
            [](int mode) {
                if (auto* gm = python::get_gui_manager())
                    gm->gizmo().setPivotMode(static_cast<PivotMode>(mode));
            });
        callback_cleanup_.add([] { python::set_pivot_mode_callbacks(nullptr, nullptr); });
        python::set_transform_space_callbacks(
            []() -> int {
                const auto* gm = python::get_gui_manager();
                return gm ? static_cast<int>(gm->gizmo().getTransformSpace()) : 0;
            },
            [](int space) {
                if (auto* gm = python::get_gui_manager())
                    gm->gizmo().setTransformSpace(static_cast<TransformSpace>(space));
            });
        callback_cleanup_.add([] { python::set_transform_space_callbacks(nullptr, nullptr); });
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
        callback_cleanup_.add([] { python::set_thumbnail_callbacks(nullptr, nullptr, nullptr, nullptr); });
        python::set_scene_manager(scene_manager_.get());
        callback_cleanup_.add([] { python::set_scene_manager(nullptr); });

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
        callback_cleanup_.add([] { python::set_export_callback(nullptr); });
    }

    void VisualizerImpl::setupViewContextBridge() {
        if (view_context_bridge_initialized_)
            return;

        view_context_bridge_initialized_ = true;

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
            const auto P = viewport_.camera.getPivot();
            info.pivot = {P.x, P.y, P.z};
            info.width = viewport_.windowSize.x;
            info.height = viewport_.windowSize.y;
            info.fov = lfs::rendering::focalLengthToVFov(settings.focal_length_mm);
            return info;
        });
        callback_cleanup_.add([] { vis::set_view_callback(nullptr); });

        vis::set_set_view_callback([this](const vis::SetViewParams& params) {
            const glm::vec3 eye(params.eye[0], params.eye[1], params.eye[2]);
            const glm::vec3 target(params.target[0], params.target[1], params.target[2]);
            const glm::vec3 up(params.up[0], params.up[1], params.up[2]);

            const auto rotation = buildValidatedViewRotation(eye, target, up);
            if (!rotation) {
                LOG_WARN("Ignoring set_view request with degenerate or non-finite eye/target/up vectors");
                return;
            }

            viewport_.camera.R = *rotation;
            viewport_.camera.t = eye;
            viewport_.camera.setPivot(target);

            if (rendering_manager_)
                rendering_manager_->markDirty(DirtyFlag::CAMERA);
        });
        callback_cleanup_.add([] { vis::set_set_view_callback(nullptr); });

        vis::set_set_fov_callback([this](float fov_degrees) {
            if (rendering_manager_)
                rendering_manager_->setFocalLength(lfs::rendering::vFovToFocalLength(fov_degrees));
        });
        callback_cleanup_.add([] { vis::set_set_fov_callback(nullptr); });

        const auto get_screen_positions = [this]() -> std::shared_ptr<lfs::core::Tensor> {
            if (!scene_manager_) {
                return nullptr;
            }
            auto* const selection_service = scene_manager_->getSelectionService();
            return selection_service ? selection_service->getScreenPositions() : nullptr;
        };

        vis::set_viewport_render_callback([this, get_screen_positions]() -> std::optional<vis::ViewportRender> {
            if (!rendering_manager_)
                return std::nullopt;

            auto image = rendering_manager_->getViewportImageIfAvailable();
            if (!image)
                return std::nullopt;

            return vis::ViewportRender{std::move(image), get_screen_positions()};
        });
        callback_cleanup_.add([] { vis::set_viewport_render_callback(nullptr); });

        vis::set_capture_viewport_render_callback([this, get_screen_positions]() -> std::optional<vis::ViewportRender> {
            if (!rendering_manager_)
                return std::nullopt;

            auto image = rendering_manager_->captureViewportImage();
            if (!image)
                return std::nullopt;

            return vis::ViewportRender{std::move(image), get_screen_positions()};
        });
        callback_cleanup_.add([] { vis::set_capture_viewport_render_callback(nullptr); });

        vis::set_render_settings_callbacks(
            [this]() -> std::optional<vis::RenderSettingsProxy> {
                return rendering_manager_ ? std::optional{vis::to_proxy(rendering_manager_->getSettings())}
                                          : std::nullopt;
            },
            [this](const vis::RenderSettingsProxy& proxy) {
                if (!rendering_manager_)
                    return;
                auto s = rendering_manager_->getSettings();
                vis::apply_proxy(s, proxy);
                rendering_manager_->updateSettings(s);
            });
        callback_cleanup_.add([] { vis::set_render_settings_callbacks(nullptr, nullptr); });
    }

    void VisualizerImpl::setupComponentConnections() {
        // Set up main loop callbacks
        main_loop_->setInitCallback([this]() { return initialize(); });
        main_loop_->setUpdateCallback([this]() { update(); });
        main_loop_->setRenderCallback([this]() { render(); });
        main_loop_->setShutdownCallback([this]() { shutdown(); });
        main_loop_->setShouldCloseCallback([this]() { return allowclose(); });
    }

    void VisualizerImpl::beginShutdown([[maybe_unused]] const std::string_view reason) {
        std::vector<WorkItem> pending_work;
        std::vector<WorkItem> pending_render_work;
        {
            std::lock_guard lock(work_queue_mutex_);
            if (shutdown_started_)
                return;
            shutdown_started_ = true;
            accepting_work_ = false;
            pending_work.swap(work_queue_);
            pending_render_work.swap(render_work_queue_);
        }

        for (auto& work : pending_work) {
            if (work.cancel)
                work.cancel();
        }
        for (auto& work : pending_render_work) {
            if (work.cancel)
                work.cancel();
        }

        std::function<void()> shutdown_callback;
        {
            std::lock_guard lock(shutdown_callback_mutex_);
            shutdown_callback = shutdown_requested_callback_;
        }
        if (shutdown_callback)
            shutdown_callback();
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

        // NOTE: ui::RenderSettingsChanged, ui::CameraMove, state::SceneChanged,
        // ui::PointCloudModeChanged are handled by RenderingManager::setupEventHandlers()

        // Window redraw requests on scene/mode changes
        state::SceneChanged::when([this](const auto& event) {
            python::set_scene_mutation_flags(event.mutation_flags);
            python::bump_scene_generation();
            wakeMainLoop();
        });

        ui::PointCloudModeChanged::when([this](const auto&) {
            wakeMainLoop();
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

        // Training started - switch to splat rendering without hijacking scene selection
        state::TrainingStarted::when([this](const auto&) {
            ui::PointCloudModeChanged{
                .enabled = false,
                .voxel_size = 0.01f}
                .emit();

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
            if (window_manager_) {
                window_manager_->requestClose();
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
            const std::string path_utf8 = core::path_to_utf8(event.path);
            python::update_scene(true, path_utf8.c_str());
        });

        state::SceneCleared::when([](const auto&) {
            python::update_scene(false, "");
        });
    }

    bool VisualizerImpl::initialize() {
        if (fully_initialized_) {
            LOG_TRACE("Already fully initialized");
            return true;
        }

        // Initialize window first and ensure it has proper size
        if (!window_initialized_) {
            if (!window_manager_->init()) {
                return false;
            }
            window_initialized_ = true;

            window_manager_->pollEvents();
            window_manager_->updateWindowSize();

            viewport_.windowSize = window_manager_->getWindowSize();
            viewport_.frameBufferSize = window_manager_->getFramebufferSize();

            if (viewport_.windowSize.x <= 0 || viewport_.windowSize.y <= 0) {
                LOG_WARN("Window manager returned invalid size, using options fallback: {}x{}",
                         options_.width, options_.height);
                viewport_.windowSize = glm::ivec2(options_.width, options_.height);
                viewport_.frameBufferSize = glm::ivec2(options_.width, options_.height);
            }

            LOG_DEBUG("Window initialized with actual size: {}x{}",
                      viewport_.windowSize.x, viewport_.windowSize.y);
        }

        // Initialize rendering early so we can show a frame before font atlas build
        if (!rendering_manager_->isInitialized()) {
            rendering_manager_->initialize();
        }

        // Render one frame (grid + background) before the expensive GUI/font init
        {
            RenderingManager::RenderContext ctx{
                .viewport = viewport_,
                .settings = rendering_manager_->getSettings(),
                .viewport_region = nullptr,
                .scene_manager = scene_manager_.get()};
            rendering_manager_->renderFrame(ctx);
            window_manager_->swapBuffers();
        }

        // Initialize GUI (sets up ImGui, builds font atlas)
        if (!gui_initialized_) {
            gui_manager_->init();
            gui_initialized_ = true;
        }

        // InputController requires ImGui to be initialized
        if (!input_controller_) {
            input_controller_ = std::make_unique<InputController>(
                window_manager_->getWindow(), viewport_);
            input_controller_->initialize();
            window_manager_->setInputController(input_controller_.get());
            python::set_keymap_bindings(&input_controller_->getBindings());
            callback_cleanup_.add([] { python::set_keymap_bindings(nullptr); });
        }

        // Initialize tools AFTER rendering is initialized
        if (!tools_initialized_) {
            initializeTools();
        }

        setupViewContextBridge();

        if (scene_manager_)
            scene_manager_->initSelectionService();

        python::ensure_initialized();
        python::preload_user_plugins_async();

        window_manager_->showWindow();

        fully_initialized_ = true;
        return true;
    }

    void VisualizerImpl::update() {
        window_manager_->updateWindowSize();

        // Process MCP work queue
        {
            std::vector<WorkItem> work;
            {
                std::lock_guard lock(work_queue_mutex_);
                work.swap(work_queue_);
            }
            for (size_t i = 0; i < work.size(); ++i) {
                try {
                    if (work[i].run)
                        work[i].run();
                } catch (...) {
                    for (size_t j = i + 1; j < work.size(); ++j) {
                        if (work[j].cancel)
                            work[j].cancel();
                    }
                    throw;
                }
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

        if (!gui_frame_rendered_) {
            // Wait for at least one GUI frame to render before loading data
        } else if (!pending_view_paths_.empty()) {
            auto paths = std::exchange(pending_view_paths_, {});
            LOG_INFO("Loading {} splat file(s)", paths.size());
            if (const auto result = data_loader_->loadPLY(paths[0]); !result) {
                LOG_ERROR("Failed to load {}: {}", lfs::core::path_to_utf8(paths[0]), result.error());
            } else {
                for (size_t i = 1; i < paths.size(); ++i) {
                    try {
                        data_loader_->addSplatFileToScene(paths[i]);
                    } catch (const std::exception& e) {
                        LOG_ERROR("Failed to add {}: {}", lfs::core::path_to_utf8(paths[i]), e.what());
                    }
                }
                if (paths.size() > 1) {
                    scene_manager_->consolidateNodeModels();
                }
            }
        } else if (!pending_dataset_path_.empty()) {
            auto path = std::exchange(pending_dataset_path_, {});
            LOG_INFO("Loading dataset: {}", lfs::core::path_to_utf8(path));
            if (const auto result = data_loader_->loadDataset(path); !result) {
                LOG_ERROR("Failed to load dataset: {}", result.error());
            }
        }

        // Auto-start training if --train flag was passed
        if (pending_auto_train_ && trainer_manager_ && trainer_manager_->canStart()) {
            pending_auto_train_ = false;
            LOG_INFO("Auto-starting training (--train flag)");
            cmd::StartTraining{}.emit();
        }
    }

    void VisualizerImpl::render() {

        auto now = std::chrono::high_resolution_clock::now();
        float delta_time = std::chrono::duration<float>(now - last_frame_time_).count();
        last_frame_time_ = now;

        // Clamp delta time to prevent huge jumps (min 30 FPS)
        delta_time = std::min(delta_time, 1.0f / 30.0f);

        // Tick Python frame callback for animations
        if (python::has_frame_callback()) {
            python::tick_frame_callback(delta_time);
            if (rendering_manager_) {
                rendering_manager_->markDirty(DirtyFlag::ALL);
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
            auto pos = gui_manager_->getViewportPos();
            auto size = gui_manager_->getViewportSize();

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
            .scene_manager = scene_manager_.get()};

        if (gui_manager_) {
            rendering_manager_->setCropboxGizmoActive(gui_manager_->gizmo().isCropboxGizmoActive());
            rendering_manager_->setEllipsoidGizmoActive(gui_manager_->gizmo().isEllipsoidGizmoActive());
        }

        rendering_manager_->renderFrame(context);

        gui_manager_->render();

        const bool resize_done = rendering_manager_->consumeResizeCompleted();
        if (resize_done)
            glFinish();

        {
            std::vector<WorkItem> render_work;
            {
                std::lock_guard lock(work_queue_mutex_);
                render_work.swap(render_work_queue_);
            }
            if (!render_work.empty()) {
                processing_render_work_ = true;
                for (size_t i = 0; i < render_work.size(); ++i) {
                    try {
                        if (render_work[i].run)
                            render_work[i].run();
                    } catch (...) {
                        for (size_t j = i + 1; j < render_work.size(); ++j) {
                            if (render_work[j].cancel)
                                render_work[j].cancel();
                        }
                        processing_render_work_ = false;
                        throw;
                    }
                }
                processing_render_work_ = false;
            }
        }

        window_manager_->swapBuffers();

        python::flush_signals();
        gui_frame_rendered_ = true;

        // Render-on-demand: VSync handles frame pacing, waitEvents saves CPU when idle
        const bool is_training = trainer_manager_ && trainer_manager_->isRunning();
        const bool needs_render = rendering_manager_->pollDirtyState();
        const bool continuous_input = input_controller_ && input_controller_->isContinuousInputActive();
        const bool has_python_animation = python::has_frame_callback();
        const bool has_python_overlay = python::has_viewport_draw_handlers();
        const bool has_python_redraw = python::consume_redraw_request();
        const bool needs_gui_animation = gui_manager_ && gui_manager_->needsAnimationFrame();

        if (needs_render || continuous_input || has_python_animation || has_python_overlay ||
            has_python_redraw || needs_gui_animation) {
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
            beginShutdown();
            return true;
        }

        if (gui_manager_->isForceExit()) {
            beginShutdown();
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
        beginShutdown();

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
            rendering_manager_->markDirty(DirtyFlag::ALL);
        }
    }

    void VisualizerImpl::redo() {
        op::undoHistory().redo();
        if (rendering_manager_) {
            rendering_manager_->markDirty(DirtyFlag::ALL);
        }
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
        pending_view_paths_ = params.view_paths;
        pending_dataset_path_ = params.dataset.data_path;
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
        auto result = data_loader_->loadCheckpointForTraining(path);
        if (result) {
            pending_view_paths_.clear();
            pending_dataset_path_.clear();
        }
        return result;
    }

    void VisualizerImpl::consolidateModels() {
        scene_manager_->consolidateNodeModels();
    }

    void VisualizerImpl::clearScene() {
        data_loader_->clearScene();
    }

    void VisualizerImpl::wakeMainLoop() const {
        if (window_manager_)
            window_manager_->wakeEventLoop();
    }

    bool VisualizerImpl::postWork(WorkItem work) {
        {
            std::lock_guard lock(work_queue_mutex_);
            if (!accepting_work_)
                return false;
            work_queue_.push_back(std::move(work));
        }

        wakeMainLoop();

        return true;
    }

    bool VisualizerImpl::postRenderWork(WorkItem work) {
        {
            std::lock_guard lock(work_queue_mutex_);
            if (!accepting_work_)
                return false;
            render_work_queue_.push_back(std::move(work));
        }

        wakeMainLoop();

        return true;
    }

    bool VisualizerImpl::acceptsPostedWork() const {
        std::lock_guard lock(work_queue_mutex_);
        return accepting_work_;
    }

    void VisualizerImpl::setShutdownRequestedCallback(std::function<void()> callback) {
        std::lock_guard lock(shutdown_callback_mutex_);
        shutdown_requested_callback_ = std::move(callback);
    }

    std::expected<void, std::string> VisualizerImpl::startTraining() {
        if (!trainer_manager_)
            return std::unexpected("Trainer manager not initialized");
        if (!trainer_manager_->startTraining())
            return std::unexpected("Failed to start training");
        return {};
    }

    std::expected<std::filesystem::path, std::string> VisualizerImpl::saveCheckpoint(
        const std::optional<std::filesystem::path>& path) {
        if (!trainer_manager_ || !trainer_manager_->getTrainer())
            return std::unexpected("No active training session");

        auto* const trainer = trainer_manager_->getTrainer();
        if (trainer_manager_->isTrainingActive()) {
            if (path) {
                return std::unexpected(
                    "Custom checkpoint output paths are not supported while training is active");
            }
            return std::unexpected(
                "Cannot report checkpoint save success while training is active; "
                "use the async training checkpoint action or stop training first");
        }

        const int iteration = trainer->get_current_iteration();
        if (path) {
            if (auto result = trainer->save_checkpoint_to(*path, iteration); !result)
                return std::unexpected(result.error());
            return *path;
        }

        if (auto result = trainer->save_checkpoint(iteration); !result)
            return std::unexpected(result.error());
        return trainer->get_output_path();
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

                auto ply_in_sparse = lfs::io::find_file_in_paths(
                    lfs::io::get_colmap_search_paths(path), "points3D.ply");
                if (!ply_in_sparse.empty()) {
                    params.init_path = std::nullopt;
                    LOG_INFO("Reset: using points3D.ply from {}", lfs::core::path_to_utf8(ply_in_sparse));
                } else {
                    params.init_path = init_path;
                }
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
        result->apply_step_scaling();
        parameter_manager_->importParams(*result);
    }

    void VisualizerImpl::handleTrainingCompleted([[maybe_unused]] const state::TrainingCompleted& event) {
        if (scene_manager_) {
            scene_manager_->changeContentType(SceneManager::ContentType::Dataset);
        }
    }

    void VisualizerImpl::handleSwitchToLatestCheckpoint() {
        // This event is emitted by the training flow even when no project/checkpoint manager is active.
        // In the plain dataset workflow there is nothing to switch, so treat it as a no-op.
    }

} // namespace lfs::vis
