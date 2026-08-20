/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer_impl.hpp"
#include "core/animatable_property.hpp"
#include "core/cuda_error.hpp"
#include "core/data_loading_service.hpp"
#include "core/error.hpp"
#include "core/error_bus.hpp"
#include "core/error_reporter.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/guarded_task.hpp"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/path_utils.hpp"
#include "core/services.hpp"
#include "gui/error_event_bridge.hpp"
#include "gui/panel_registry.hpp"
#include "gui/panels/tools_panel.hpp"
#include "gui/panels/windows_console_utils.hpp"
#include "gui/string_keys.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "ipc/render_settings_convert.hpp"
#include "ipc/view_context.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "operator/ops/align_ops.hpp"
#include "operator/ops/edit_ops.hpp"
#include "operator/ops/scene_ops.hpp"
#include "operator/ops/selection_ops.hpp"
#include "operator/ops/transform_ops.hpp"
#include "preferences.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/model_renderability.hpp"
#include "rendering/scene_upscaler_registry.hpp"
#include "scene/scene_manager.hpp"
#include "tools/align_tool.hpp"
#include "tools/builtin_tools.hpp"
#include "tools/selection_tool.hpp"
#include "tools/unified_tool_registry.hpp"
#include "visualizer/app_store.hpp"
#include "window/vulkan_context.hpp"
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_messagebox.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#ifdef WIN32
#include <windows.h>
#endif

namespace lfs::vis {

    using namespace lfs::core::events;

    namespace ErrModalKeys = lichtfeld::Strings::ErrorModal;

    namespace {

        // Generic ErrorBus publisher. Default operation is the frame-loop tag
        // so existing renderer callers stay on kRenderFrame.
        lfs::ErrorNotification makeFrameNotification(const lfs::ErrorCode code,
                                                     const lfs::ErrorDomain domain,
                                                     const lfs::Severity severity,
                                                     const lfs::ErrorSurface surface,
                                                     std::string user_message, std::string detail,
                                                     std::vector<lfs::ErrorAction> actions,
                                                     const lfs::core::SourceSite site,
                                                     const char* operation = gui::error_op::kRenderFrame) {
            lfs::Error base = lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = domain,
                .severity = severity,
                .retryability = lfs::Retryability::NotRetryable,
                .operation_id = lfs::OperationId{},
                .user_message = std::move(user_message),
                .detail = std::move(detail),
                .detection = site,
                .fields = lfs::SmallFields{},
                .native = std::nullopt,
            });
            return lfs::ErrorNotification{
                .error = std::move(base).with_context(operation, site),
                .surface = surface,
                .actions = std::move(actions),
                .operation_id = lfs::OperationId::generate(),
            };
        }

        template <typename T>
        [[nodiscard]] lfs::Result<T> visualizerFailure(
            const lfs::ErrorCode code,
            std::string user_message,
            std::string detail,
            const std::string_view field) {
            auto error = lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::App,
                .severity = lfs::Severity::Error,
                .retryability =
                    lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message =
                    std::move(user_message),
                .detail = std::move(detail),
                .detection =
                    LFS_SOURCE_SITE_CURRENT(),
                .fields =
                    lfs::SmallFields{}.add(
                        "field", field),
                .native = std::nullopt,
            });
            if constexpr (std::same_as<T, void>) {
                return lfs::Status::failure(
                    std::move(error));
            } else {
                return error;
            }
        }

        [[nodiscard]] bool
        isDirtyProjectSwitchError(
            const lfs::Error& error) noexcept {
            return error.code() ==
                       lfs::ErrorCode::
                           FailedPrecondition &&
                   error.user_message() ==
                       "The current project has unsaved changes.";
        }

        [[nodiscard]] bool
        isTrainingProjectSwitchError(
            const lfs::Error& error) noexcept {
            return error.code() ==
                       lfs::ErrorCode::
                           FailedPrecondition &&
                   error.user_message() ==
                       "Stop training before switching projects.";
        }

        void wakeEventLoopViaServices() {
            if (auto* const window_manager = services().windowOrNull()) {
                window_manager->wakeEventLoop();
            }
        }

        constexpr double kResizeSettleMinWaitSeconds = 0.001;
        constexpr double kTooltipRevealMinWaitSeconds = 0.001;
        constexpr double kScheduledRedrawMinWaitSeconds = 0.001;
        constexpr double kGuiScheduledUpdateMinWaitSeconds = 0.001;
        // Backstop cap for GUI-only continuous demand (python_redraw / gui_animation only).
        // MAILBOX presents never block, so free-running would pin the GPU at full GUI cost.
        constexpr double kGuiAnimationFrameInterval = 1.0 / 30.0;
#if defined(__linux__)
        constexpr auto kWindowResizePaintDemandWindow = std::chrono::milliseconds(160);
#endif

        std::optional<glm::mat3> buildValidatedViewRotation(const glm::vec3& eye,
                                                            const glm::vec3& target,
                                                            const glm::vec3& requested_up) {
            return lfs::rendering::tryMakeVisualizerLookAtRotation(eye, target, requested_up);
        }

        [[nodiscard]] bool shouldPreserveResetTransform(const core::SceneNode& node) {
            return node.type == core::NodeType::DATASET ||
                   node.type == core::NodeType::POINTCLOUD ||
                   node.type == core::NodeType::SPLAT ||
                   node.type == core::NodeType::CROPBOX ||
                   node.type == core::NodeType::ELLIPSOID;
        }

        [[nodiscard]] std::unordered_map<std::string, glm::mat4> collectResetTransforms(const core::Scene& scene) {
            std::unordered_map<std::string, glm::mat4> transforms;
            for (const auto* node : scene.getNodes()) {
                if (node && shouldPreserveResetTransform(*node)) {
                    transforms.emplace(node->name, node->transform());
                }
            }
            return transforms;
        }

        void cancelRemainingWork(std::vector<Visualizer::WorkItem>& work,
                                 const size_t first,
                                 const std::string_view queue_name,
                                 const std::thread::id owner_thread) noexcept {
            for (size_t i = first; i < work.size(); ++i) {
                if (!work[i].cancel)
                    continue;
                lfs::core::run_guarded<void>(
                    lfs::core::TaskContext{
                        .name = std::format("{}.cancel", queue_name),
                        .domain = lfs::ErrorDomain::Core,
                        .operation_id = lfs::OperationId::generate(),
                        .site = LFS_SOURCE_SITE_CURRENT(),
                        .expected_thread = owner_thread,
                    },
                    [&work, i]() -> lfs::Result<void> {
                        work[i].cancel();
                        return {};
                    },
                    [](lfs::Result<void>&& result) {
                        if (!result) {
                            lfs::core::ErrorReporter::get().report(result.error(), lfs::core::ReportChannel::OwnerLog);
                        }
                    });
            }
        }

        // Posted work is an external callback boundary. A failing item may report
        // through its own promise, but must never unwind the GUI frame loop.
        void runPostedWork(std::vector<Visualizer::WorkItem>& work,
                           const std::string_view queue_name,
                           const std::thread::id owner_thread) noexcept {
            for (size_t i = 0; i < work.size(); ++i) {
                if (!work[i].run)
                    continue;
                bool item_failed = false;
                lfs::core::run_guarded<void>(
                    lfs::core::TaskContext{
                        .name = std::string(queue_name),
                        .domain = lfs::ErrorDomain::Core,
                        .operation_id = lfs::OperationId::generate(),
                        .site = LFS_SOURCE_SITE_CURRENT(),
                        .expected_thread = owner_thread,
                    },
                    [&work, i]() -> lfs::Result<void> {
                        work[i].run();
                        return {};
                    },
                    [&item_failed](lfs::Result<void>&& result) {
                        if (!result) {
                            item_failed = true;
                            lfs::core::ErrorReporter::get().report(result.error(), lfs::core::ReportChannel::OwnerLog);
                        }
                    });
                if (item_failed) {
                    cancelRemainingWork(work, i + 1, queue_name, owner_thread);
                    return;
                }
            }
        }

    } // namespace

    VisualizerImpl::VisualizerImpl(const ViewerOptions& options)
        : options_(options),
          viewport_(options.width, options.height),
          window_manager_(std::make_unique<WindowManager>(options.title, options.width, options.height,
                                                          options.monitor_x, options.monitor_y,
                                                          options.monitor_width, options.monitor_height,
                                                          options.graphics_backend)) {
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
        rendering_manager_->setWakeCallback([this] {
            wakeMainLoop();
        });

        // Set initial antialiasing
        RenderSettings initial_settings;
        initial_settings.antialiasing = options.antialiasing;
        initial_settings.scene_upscaler = options.safe_mode
                                              ? "native"
                                              : loadSceneUpscalerPreference();
        initial_settings.scene_upscaler_preset = options.safe_mode
                                                     ? "native"
                                                     : loadSceneUpscalerPresetPreference(
                                                           initial_settings.scene_upscaler);
        if (const auto backend = sceneUpscalerBackendFromId(initial_settings.scene_upscaler)) {
            const auto preset = sceneUpscalerPreset(*backend, initial_settings.scene_upscaler_preset)
                                    .value_or(defaultSceneUpscalerPreset(*backend));
            initial_settings.scene_upscaler_preset = std::string(preset.id);
            initial_settings.scene_upscaler_scale = preset.input_scale;
        }
        initial_settings.gut = options.gut;
        initial_settings.raster_backend = options.gut
                                              ? lfs::rendering::GaussianRasterBackend::ThreeDgut
                                              : lfs::rendering::GaussianRasterBackend::ThreeDgs;
        rendering_manager_->updateSettings(initial_settings);

        // Create data loading service
        data_loader_ = std::make_unique<DataLoadingService>(scene_manager_.get());

        // Create parameter manager (lazy-loads JSON files on first use)
        parameter_manager_ = std::make_unique<ParameterManager>();

        project_lifecycle_ =
            std::make_unique<project::ProjectLifecycle>(
                *this,
                options_.project_lifecycle_settings_path);

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
        op::registerEditOperators();
        op::registerSceneOperators();

        setupPythonBridge();
        setupEventHandlers();
        setupComponentConnections();
    }

    VisualizerImpl::~VisualizerImpl() {
        // ProjectLifecycle owns worker threads and calls back into the viewer
        // while shutting down. Destroy it before invalidating the service
        // locator or releasing any of the components it observes.
        project_lifecycle_.reset();

        // Clear event handlers before destroying components to prevent use-after-free
        lfs::event::EventBridge::instance().clear_all();
        services().clear();

        // Clear operator system
        op::unregisterEditOperators();
        op::unregisterSceneOperators();
        op::unregisterSelectionOperators();
        op::unregisterAlignOperators();
        op::unregisterTransformOperators();
        op::operators().clear();

        callback_cleanup_.clear();
        stopForceExitCompletionWatcher();
        trainer_manager_.reset();
        tool_context_.reset();
        if (gui_manager_) {
            gui_manager_->shutdown();
            // GuiManager owns AsyncTaskManager, which retains a reference to
            // job_registry_. Destroy the GUI while that registry and the
            // window/rendering resources it depends on are still alive.
            gui_manager_.reset();
        }
        // Tear down viewport interop after the viewport pass is reset above, while
        // the window/Vulkan context is still alive. Belt-and-braces again in
        // ~RenderingManager once already shut down.
        if (rendering_manager_) {
            VulkanContext* context = nullptr;
            if (window_manager_) {
                context = window_manager_->getVulkanContext();
            }
            rendering_manager_->shutdownViewportInterop(context);
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
        }

        tools_initialized_ = true;
    }

    void VisualizerImpl::setupPythonBridge() {
        python::set_visualizer(this);
        callback_cleanup_.add([] { python::set_visualizer(nullptr); });
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
        core::reactive::Store::set_wake_callback(&wakeEventLoopViaServices);
        callback_cleanup_.add([] { core::reactive::Store::set_wake_callback(nullptr); });
        python::set_scene_generation_callback([](const uint64_t generation) {
            app_store().scene_generation.set(generation);
        });
        callback_cleanup_.add([] { python::set_scene_generation_callback(nullptr); });
        app_store().scene_generation.set(python::get_scene_generation());
        auto active_tool_poll_cache_token = std::make_shared<core::reactive::SubscriptionToken>(
            app_store().active_tool.subscribe([](const std::string&) {
                gui::PanelRegistry::instance().invalidate_poll_cache();
            }));
        callback_cleanup_.add([active_tool_poll_cache_token] {
            active_tool_poll_cache_token->reset();
        });
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
                state.outcome = tasks.getExportOutcome();
                const auto fmt = tasks.getExportFormat();
                state.format = fmt == core::ExportFormat::PLY           ? "PLY"
                               : fmt == core::ExportFormat::SOG         ? "SOG"
                               : fmt == core::ExportFormat::SPZ         ? "SPZ"
                               : fmt == core::ExportFormat::HTML_VIEWER ? "HTML"
                               : fmt == core::ExportFormat::USD         ? "USD"
                               : fmt == core::ExportFormat::NUREC_USDZ  ? "USDZ"
                               : fmt == core::ExportFormat::RAD         ? "RAD"
                               : fmt == core::ExportFormat::COLMAP      ? "COLMAP"
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
                if (const auto project_open =
                        gm->getViewer()->jobs().active(JobType::ProjectOpen);
                    project_open) {
                    state.active = true;
                    state.progress = project_open->progress;
                    state.stage = project_open->stage;
                    state.dataset_type = "project";
                    return state;
                }
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
                state.sequence_fps = s.sequence_fps;
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
            s.sequence_fps = state.sequence_fps;
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
        python::set_multi_transform_mode_callbacks(
            []() -> int {
                const auto* gm = python::get_gui_manager();
                return gm ? static_cast<int>(gm->gizmo().getMultiTransformMode()) : 0;
            },
            [](int mode) {
                if (auto* gm = python::get_gui_manager()) {
                    const auto normalized_mode =
                        gui::normalizeMultiTransformMode(static_cast<gui::MultiTransformMode>(mode));
                    gm->gizmo().setMultiTransformMode(normalized_mode);
                }
            });
        callback_cleanup_.add([] { python::set_multi_transform_mode_callbacks(nullptr, nullptr); });
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
                                       int node_count, int sh_degree, bool rad_flip_y,
                                       bool rad_streamable, int spz_version,
                                       bool include_provenance) {
            if (auto* gm = python::get_gui_manager()) {
                std::vector<std::string> names;
                names.reserve(node_count);
                for (int i = 0; i < node_count; ++i) {
                    names.emplace_back(node_names[i]);
                }
                gm->asyncTasks().performExport(static_cast<lfs::core::ExportFormat>(format),
                                               lfs::core::utf8_to_path(path), names, sh_degree,
                                               rad_flip_y,
                                               rad_streamable,
                                               spz_version,
                                               include_provenance);
            }
        });
        callback_cleanup_.add([] { python::set_export_callback(nullptr); });

        // Asset Manager save callback - implementation handled by Python runtime
        callback_cleanup_.add([] { python::set_save_asset_callback(nullptr); });
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
            info.orthographic = settings.orthographic;
            info.ortho_scale = settings.ortho_scale;
            return info;
        });
        callback_cleanup_.add([] { vis::set_view_callback(nullptr); });

        vis::set_view_for_panel_callback([this](const vis::SplitViewPanelId panel) -> std::optional<vis::ViewInfo> {
            if (!rendering_manager_)
                return std::nullopt;

            const auto& settings = rendering_manager_->getSettings();
            const Viewport& vp = rendering_manager_->resolvePanelViewport(viewport_, panel);
            const auto R = vp.getRotationMatrix();
            const auto T = vp.getTranslation();

            vis::ViewInfo info;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    info.rotation[i * 3 + j] = R[j][i];
            info.translation = {T.x, T.y, T.z};
            const auto P = vp.camera.getPivot();
            info.pivot = {P.x, P.y, P.z};

            const int total_width = viewport_.windowSize.x;
            int panel_width = total_width;
            if (rendering_manager_->isSplitViewActive() && total_width > 0) {
                const float split_pos = std::clamp(settings.split_position, 0.0f, 1.0f);
                const int divider = static_cast<int>(static_cast<float>(total_width) * split_pos);
                panel_width = (panel == vis::SplitViewPanelId::Left)
                                  ? std::max(1, divider)
                                  : std::max(1, total_width - divider);
            }
            info.width = panel_width;
            info.height = viewport_.windowSize.y;
            info.fov = lfs::rendering::focalLengthToVFov(settings.focal_length_mm);
            info.orthographic = settings.orthographic;
            info.ortho_scale = settings.ortho_scale;
            return info;
        });
        callback_cleanup_.add([] { vis::set_view_for_panel_callback(nullptr); });

        vis::set_set_view_callback([this](const vis::SetViewParams& params) {
            const glm::vec3 eye(params.eye[0], params.eye[1], params.eye[2]);
            const glm::vec3 target(params.target[0], params.target[1], params.target[2]);
            const glm::vec3 up(params.up[0], params.up[1], params.up[2]);

            const auto rotation = buildValidatedViewRotation(eye, target, up);
            if (!rotation) {
                LOG_WARN("Ignoring set_view request with degenerate or non-finite eye/target/up vectors");
                return;
            }

            viewport_.setViewMatrix(*rotation, eye);
            viewport_.camera.setPivot(target);

            if (rendering_manager_)
                rendering_manager_->markCameraPoseChanged();
        });
        callback_cleanup_.add([] { vis::set_set_view_callback(nullptr); });

        vis::set_set_view_for_panel_callback([this](const vis::SplitViewPanelId panel,
                                                    const vis::SetViewParams& params) {
            if (!rendering_manager_)
                return;

            const glm::vec3 eye(params.eye[0], params.eye[1], params.eye[2]);
            const glm::vec3 target(params.target[0], params.target[1], params.target[2]);
            const glm::vec3 up(params.up[0], params.up[1], params.up[2]);

            const auto rotation = buildValidatedViewRotation(eye, target, up);
            if (!rotation) {
                LOG_WARN("Ignoring set_view request with degenerate or non-finite eye/target/up vectors");
                return;
            }

            Viewport& vp = rendering_manager_->resolvePanelViewport(viewport_, panel);
            vp.setViewMatrix(*rotation, eye);
            vp.camera.setPivot(target);

            rendering_manager_->markCameraPoseChanged();
        });
        callback_cleanup_.add([] { vis::set_set_view_for_panel_callback(nullptr); });

        vis::set_set_fov_callback([this](float fov_degrees) {
            if (rendering_manager_)
                rendering_manager_->setFocalLength(lfs::rendering::vFovToFocalLength(fov_degrees));
        });
        callback_cleanup_.add([] { vis::set_set_fov_callback(nullptr); });

        const auto get_screen_positions = [this]() -> std::shared_ptr<lfs::core::Tensor> {
            if (!scene_manager_) {
                return nullptr;
            }
            if (!hasRenderableGaussians(scene_manager_->getModelForRendering())) {
                return nullptr;
            }
            if (const auto* tm = scene_manager_->getTrainerManager()) {
                if (tm->isCompletionPending() ||
                    tm->getState() == TrainingState::Stopping) {
                    return nullptr;
                }
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
            if (!isOnViewerThread()) {
                LOG_ERROR(
                    "Viewport capture requested off the viewer thread; returning empty");
                return std::nullopt;
            }
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
                const std::string previous_upscaler = s.scene_upscaler;
                const std::string previous_preset = s.scene_upscaler_preset;
                vis::apply_proxy(s, proxy);
                rendering_manager_->updateSettings(s);
                const auto& applied = rendering_manager_->getSettings();
                if (applied.scene_upscaler != previous_upscaler ||
                    applied.scene_upscaler_preset != previous_preset) {
                    saveSceneUpscalerPreference(applied.scene_upscaler,
                                                applied.scene_upscaler_preset);
                }
                wakeMainLoop();
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
        main_loop_->setWakeCallback([this] { wakeMainLoop(); });
        main_loop_->setInterruptCallback([this]() {
            // First interrupt is ForceExit: discard, stop
            // training, close. The handler itself only
            // woke this loop; a second interrupt _exit(1)s.
            if (gui_manager_) {
                gui_manager_->setForceExit(true);
                gui_manager_->dismissExitConfirmation();
            }
            if (project_lifecycle_) {
                project_lifecycle_
                    ->markApplicationClosePending();
                project_lifecycle_
                    ->setSuppressTrainingAdoption(true);
            }
            if (trainer_manager_ &&
                (trainer_manager_->isTrainingActive() ||
                 trainer_manager_
                     ->isCompletionPending())) {
                pending_training_action_ =
                    PendingTrainingAction::CloseDiscard;
                trainer_manager_->suppressCompletionNotification();
                if (trainer_manager_->canStop()) {
                    trainer_manager_->stopTraining();
                }
            }
            requestApplicationClose();
        });
        main_loop_->setFrameErrorCallback([this](std::exception_ptr eptr) {
            handleFrameException(std::move(eptr));
        });
        main_loop_->setFrameCompletedCallback([this]() { onFrameCompleted(); });
    }

    void VisualizerImpl::handleFrameException(std::exception_ptr eptr) noexcept {
        try {
            std::rethrow_exception(eptr);
        } catch (const lfs::core::MemoryAllocationError& e) {
            if (lfs::core::cuda_is_unavailable()) {
                return;
            }
            const auto fx = frame_state_.on_fault(FrameFault::OomPressure);
            if (fx.run_reclaim_episode) {
                // GPU memory shortage reached the frame loop. Reclaim render-safe
                // caches once and keep running; the next frame is the retry.
                auto& coordinator = lfs::core::MemoryPressureCoordinator::instance();
                const size_t freed = coordinator.run_episode(
                    e.failure(), lfs::core::PressureContext::RenderThread);
                LOG_ERROR("GPU memory pressure during frame (attempt {}): {}. Freed {:.1f} MiB; "
                          "reducing preview quality and retrying.",
                          frame_state_.consecutive_oom_faults(), e.what(),
                          static_cast<double>(freed) / (1024.0 * 1024.0));
            }
            applyFrameStateEffects(fx);
        } catch (const lfs::Exception& e) {
            const auto fault = e.error().code() == lfs::ErrorCode::DeviceLost
                                   ? FrameFault::DeviceLost
                                   : FrameFault::RendererInternal;
            const auto fx = frame_state_.on_fault(fault);
            if (fx.publish_internal_modal) {
                publishRendererInternalModal(e.error());
            } else if (fx.publish_renderer_dead_modal) {
                applyFrameStateEffects(fx);
            } else {
                logRateLimitedFrameError(e);
            }
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): frame-fault boundary — routes the fault
            // through the FrameStateMachine and surfaces it via ErrorBus/log, not a
            // swallow (the untyped sibling of the lfs::Exception clause above).
            const auto fx = frame_state_.on_fault(FrameFault::RendererInternal);
            if (fx.publish_internal_modal) {
                publishRendererInternalModal(lfs::ErrorCode::Internal, std::string(e.what()));
            } else {
                logRateLimitedFrameError(e);
            }
        } catch (...) {
            const auto fx = frame_state_.on_fault(FrameFault::RendererInternal);
            if (fx.publish_internal_modal) {
                publishRendererInternalModal(lfs::ErrorCode::Internal, "Unknown frame error");
            } else {
                LOG_ERROR("Frame failed with an unknown error");
            }
        }
    }

    void VisualizerImpl::logRateLimitedFrameError(const std::exception& e) noexcept {
        const auto now = std::chrono::steady_clock::now();
        if (last_frame_error_log_.time_since_epoch().count() == 0 ||
            now - last_frame_error_log_ >= std::chrono::seconds(5)) {
            LOG_ERROR("Frame failed: {}{}", e.what(),
                      suppressed_frame_errors_ > 0
                          ? std::format(" ({} similar errors suppressed)", suppressed_frame_errors_)
                          : std::string{});
            last_frame_error_log_ = now;
            suppressed_frame_errors_ = 0;
        } else {
            ++suppressed_frame_errors_;
        }
    }

    void VisualizerImpl::applyFrameStateEffects(const FrameStateMachine::Effects& fx) noexcept {
        if (fx.publish_pressure_toast) {
            lfs::ErrorBus::instance().publish(makeFrameNotification(
                lfs::ErrorCode::ResourceExhausted, lfs::ErrorDomain::Rendering,
                lfs::Severity::Warning, lfs::ErrorSurface::Toast,
                LOC(ErrModalKeys::GPU_PRESSURE_RETRYING), std::string{}, {},
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (fx.publish_oom_modal) {
            LOG_ERROR("Viewport rendering paused: GPU out of memory after {} consecutive "
                      "reclaim attempts failed to recover",
                      frame_state_.consecutive_oom_faults());
            std::vector<lfs::ErrorAction> actions;
            actions.push_back(lfs::ErrorAction{
                .kind = lfs::ErrorActionKind::Retry,
                .label = {},
                .on_invoke = [this](lfs::OperationId) {
                    frame_state_.on_retry_action();
                    if (rendering_manager_)
                        rendering_manager_->markDirty(DirtyFlag::ALL);
                    wakeMainLoop();
                },
            });
            actions.push_back(gui::openLogAction());
            lfs::ErrorBus::instance().publish(makeFrameNotification(
                lfs::ErrorCode::ResourceExhausted, lfs::ErrorDomain::Rendering, lfs::Severity::Error,
                lfs::ErrorSurface::Modal, LOC(ErrModalKeys::OOM_RENDER_PAUSED), std::string{},
                std::move(actions), LFS_SOURCE_SITE_CURRENT()));
        }
        if (fx.publish_renderer_dead_modal) {
            publishRendererDeadModal(fx.dead_cause);
        }
    }

    std::vector<lfs::ErrorAction> VisualizerImpl::rendererInternalActions() {
        std::vector<lfs::ErrorAction> actions;
        actions.push_back(lfs::ErrorAction{
            .kind = lfs::ErrorActionKind::Retry,
            .label = {},
            .on_invoke = [this](lfs::OperationId) {
                frame_state_.on_retry_action();
                if (rendering_manager_)
                    rendering_manager_->markDirty(DirtyFlag::ALL);
                wakeMainLoop();
            },
        });
        actions.push_back(lfs::ErrorAction{
            .kind = lfs::ErrorActionKind::StopRenderer,
            .label = {},
            .on_invoke = [this](lfs::OperationId) { frame_state_.on_stop_renderer_action(); },
        });
        actions.push_back(gui::openLogAction());
        return actions;
    }

    // §1.5.2: the lfs::Exception path carries THIS structured error (detection
    // site, native VK info, fields, developer chain) into the notification, not a
    // synthetic one — only the context frame is appended.
    void VisualizerImpl::publishRendererInternalModal(const lfs::Error& error) noexcept {
        LOG_ERROR("Viewport renderer stopped after repeated internal frame errors: {}",
                  lfs::format_for_developer(error));
        lfs::Error contextual = error;
        lfs::ErrorBus::instance().publish(lfs::ErrorNotification{
            .error = std::move(contextual).with_context(gui::error_op::kRenderFrame, LFS_SOURCE_SITE_CURRENT()),
            .surface = lfs::ErrorSurface::Modal,
            .actions = rendererInternalActions(),
            .operation_id = lfs::OperationId::generate(),
        });
    }

    void VisualizerImpl::publishRendererInternalModal(const lfs::ErrorCode code,
                                                      std::string detail) noexcept {
        LOG_ERROR("Viewport renderer stopped after repeated internal frame errors: {}", detail);
        lfs::ErrorBus::instance().publish(makeFrameNotification(
            code, lfs::ErrorDomain::Rendering, lfs::Severity::Error, lfs::ErrorSurface::Modal,
            LOC(ErrModalKeys::RENDERER_FAILED_BODY), std::move(detail), rendererInternalActions(),
            LFS_SOURCE_SITE_CURRENT()));
    }

    void VisualizerImpl::publishRendererDeadModal(const RendererTerminalState cause) noexcept {
        auto* const ctx = window_manager_ ? window_manager_->getVulkanContext() : nullptr;
        std::string detail = ctx ? ctx->lastError() : std::string{};
        const bool device_lost = cause == RendererTerminalState::DeviceLost;
        const lfs::ErrorCode code =
            device_lost ? lfs::ErrorCode::DeviceLost : lfs::ErrorCode::DeadlineExceeded;
        const char* const body_key = device_lost ? ErrModalKeys::RENDERER_DEVICE_LOST_BODY
                                                 : ErrModalKeys::RENDERER_STALLED_BODY;

        // AMB-P3-1: emit the correlated durable record REGARDLESS — a dead context
        // cannot present the bus modal and the OS dialog can fail on Wayland.
        LOG_ERROR("Renderer terminal ({}): {}", device_lost ? "device lost" : "stalled",
                  detail.empty() ? std::string_view{"no detail"} : std::string_view{detail});

        std::vector<lfs::ErrorAction> actions;
        actions.push_back(gui::openLogAction());
        actions.push_back(lfs::ErrorAction{.kind = lfs::ErrorActionKind::Dismiss});
        lfs::ErrorBus::instance().publish(makeFrameNotification(
            code, lfs::ErrorDomain::Vulkan, lfs::Severity::Fatal, lfs::ErrorSurface::Modal,
            LOC(body_key), std::move(detail), std::move(actions), LFS_SOURCE_SITE_CURRENT()));

        if (device_lost) {
            SDL_Window* const window = window_manager_ ? window_manager_->getWindow() : nullptr;
            if (!SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR,
                                          LOC(ErrModalKeys::RENDERER_DEVICE_LOST),
                                          LOC(ErrModalKeys::RENDERER_DEVICE_LOST_BODY), window)) {
                LOG_ERROR("Failed to present device-lost dialog: {}", SDL_GetError());
            }
        }
    }

    void VisualizerImpl::onFrameCompleted() noexcept {
        frame_state_.on_frame_success();
        lfs::core::MemoryPressureCoordinator::instance().maybe_recover();
        if (auto* const ctx = window_manager_ ? window_manager_->getVulkanContext() : nullptr) {
            applyFrameStateEffects(frame_state_.on_renderer_terminal(ctx->rendererTerminalState()));
        }
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

        python::request_plugin_preload_stop();

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

        internal::GuiPanelsReady::when(
            [this](const auto& event) {
                gui_session_restore_.onPanelsReady(
                    event.registration_revision);
                tryApplyProjectSessionRestore();
            });

        // NOTE: Training control and project-save commands
        // are now handled by TrainerManager::setupEventHandlers()

        cmd::ResetTraining::when([this](const auto&) {
            if (!scene_manager_ || !scene_manager_->hasDataset()) {
                LOG_WARN("Cannot reset: no dataset");
                return;
            }
            if (trainer_manager_ &&
                (trainer_manager_->isTrainingActive() || trainer_manager_->isCompletionPending())) {
                trainer_manager_->suppressCompletionNotification();
                if (pending_training_action_ == PendingTrainingAction::None) {
                    pending_training_action_ = PendingTrainingAction::Reset;
                }
                if (trainer_manager_->canStop()) {
                    trainer_manager_->stopTraining();
                }
                return;
            }
            performReset();
        });

        cmd::NewProject::when([this](const auto& command) {
            handleNewProject(
                command.discard_changes
                    ? ProjectSwitchDisposition::
                          DiscardChanges
                    : ProjectSwitchDisposition::
                          RequireClean,
                command.stop_training);
        });

        const auto publish_project_error =
            [](std::string action, const auto& value,
               const char* operation) {
                if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                             lfs::Error>) {
                    LOG_ERROR("{} failed: {}", action,
                              lfs::format_for_developer(value));
                    lfs::Error contextual = value;
                    lfs::ErrorBus::instance().publish(
                        lfs::ErrorNotification{
                            .error = std::move(contextual).with_context(operation, LFS_SOURCE_SITE_CURRENT()),
                            .surface = lfs::ErrorSurface::Toast,
                            .actions = {},
                            .operation_id = lfs::OperationId::generate(),
                        });
                } else {
                    const std::string detail = std::string(value);
                    LOG_ERROR("{} failed: {}", action, detail);
                    lfs::ErrorBus::instance().publish(
                        makeFrameNotification(
                            lfs::ErrorCode::Unavailable,
                            lfs::ErrorDomain::App,
                            lfs::Severity::Error,
                            lfs::ErrorSurface::Toast,
                            std::format("{} failed.", action),
                            detail, {},
                            LFS_SOURCE_SITE_CURRENT(),
                            operation));
                }
            };

        cmd::SwitchToEditMode::when(
            [this, publish_project_error](const auto&) {
                if (project_lifecycle_) {
                    if (auto prepared =
                            project_lifecycle_
                                ->prepareForEditModeTransition();
                        !prepared) {
                        publish_project_error(
                            "Switch to Edit Mode",
                            prepared.error(),
                            gui::error_op::kProjectSettings);
                        return;
                    }
                }
                if (scene_manager_) {
                    scene_manager_->switchToEditMode();
                }
            });

        cmd::ProjectSave::when(
            [this, publish_project_error](const auto&) {
                auto has_path = projectHasPath();
                if (!has_path) {
                    publish_project_error(
                        "Save Project",
                        has_path.error(),
                        gui::error_op::kSave);
                    return;
                }
                if (!*has_path) {
                    const auto path =
                        gui::SaveProjectFileDialog();
                    if (path.empty()) {
                        return;
                    }
                    if (auto saved =
                            projectSaveAsFromDialog(path, true);
                        !saved) {
                        publish_project_error(
                            "Save Project",
                            saved.error(),
                            gui::error_op::kSave);
                    }
                    return;
                }
                if (auto saved = projectSave(true);
                    !saved) {
                    publish_project_error(
                        "Save Project",
                        saved.error(),
                        gui::error_op::kSave);
                }
            });

        cmd::ProjectSaveAs::when(
            [this, publish_project_error](
                const auto& command) {
                project_save_as_started_ = false;
                auto path = command.path;
                if (path.empty()) {
                    std::string default_name =
                        "project.licht";
                    std::filesystem::path
                        default_directory;
                    if (auto info = projectGetInfo();
                        info && info->path) {
                        default_name =
                            info->path->filename().string();
                        default_directory =
                            info->path->parent_path();
                    }
                    path =
                        gui::SaveProjectFileDialog(
                            default_name,
                            default_directory);
                }
                if (path.empty()) {
                    return;
                }
                if (auto saved =
                        command.path.empty()
                            ? projectSaveAsFromDialog(path, true)
                            : projectSaveAs(path, true);
                    !saved) {
                    publish_project_error(
                        "Save Project As",
                        saved.error(),
                        gui::error_op::kSave);
                    return;
                }
                project_save_as_started_ = true;
            });

        cmd::ProjectOpen::when(
            [this](const auto& command) {
                auto path = command.path;
                if (path.empty()) {
                    path =
                        gui::OpenProjectFileDialog();
                }
                if (path.empty()) {
                    return;
                }
                handleOpenProject(
                    path,
                    command.discard_changes
                        ? ProjectSwitchDisposition::
                              DiscardChanges
                        : ProjectSwitchDisposition::
                              RequireClean,
                    command.stop_training);
            });

        cmd::ProjectCompact::when(
            [this, publish_project_error](
                const auto&) {
                if (auto compacted =
                        projectCompact();
                    !compacted) {
                    publish_project_error(
                        "Compact Project",
                        compacted.error(),
                        gui::error_op::kCompact);
                }
            });

        cmd::RequestExit::when([this](const auto&) {
            abandonSaveAndExitAttempt();
            requestApplicationClose();
        });

        cmd::SaveAndExit::when(
            [this, publish_project_error](
                const auto&) {
                auto has_path = projectHasPath();
                if (!has_path) {
                    publish_project_error(
                        "Save Project",
                        has_path.error(),
                        gui::error_op::kSave);
                    abandonSaveAndExitAttempt();
                    return;
                }
                if (!*has_path) {
                    publish_project_error(
                        "Save Project",
                        "The project has no path; use Save As.",
                        gui::error_op::kSave);
                    abandonSaveAndExitAttempt();
                    return;
                }
                if (auto saved = projectSave(false);
                    !saved) {
                    publish_project_error(
                        "Save Project",
                        saved.error(),
                        gui::error_op::kSave);
                    abandonSaveAndExitAttempt();
                    return;
                }
                if (gui_manager_) {
                    gui_manager_->dismissExitConfirmation();
                }
                requestApplicationClose();
            });

        cmd::SaveAsAndExit::when(
            [this](const auto&) {
                completeSaveAsAndExit(
                    gui::SaveProjectFileDialog());
            });

        cmd::CancelExit::when([this](const auto&) {
            if (pending_training_action_ ==
                    PendingTrainingAction::CloseSave ||
                pending_training_action_ ==
                    PendingTrainingAction::
                        CloseDiscard) {
                pending_training_action_ =
                    PendingTrainingAction::None;
                pending_training_action_posted_ =
                    false;
            }
            abandonSaveAndExitAttempt();
        });

        cmd::ForceExit::when([this](const auto& event) {
            if (gui_manager_) {
                gui_manager_->setForceExit(true);
                gui_manager_->dismissExitConfirmation();
            }
            if (project_lifecycle_) {
                project_lifecycle_
                    ->markApplicationClosePending();
                project_lifecycle_
                    ->setSuppressTrainingAdoption(true);
                if (event.discard_autosave) {
                    project_lifecycle_
                        ->markCloseDiscardRequested();
                }
            }
            if (trainer_manager_ &&
                (trainer_manager_->isTrainingActive() ||
                 trainer_manager_
                     ->isCompletionPending())) {
                pending_training_action_ =
                    PendingTrainingAction::CloseDiscard;
                trainer_manager_->suppressCompletionNotification();
                if (trainer_manager_->canStop()) {
                    trainer_manager_->stopTraining();
                }
            }
            requestApplicationClose();
        });

        cmd::StopSaveAndExit::when([this](const auto&) {
            if (trainer_manager_ &&
                (trainer_manager_->isTrainingActive() ||
                 trainer_manager_
                     ->isCompletionPending())) {
                auto has_path = projectHasPath();
                if (!has_path || !*has_path) {
                    const auto path =
                        gui::SaveProjectFileDialog();
                    if (path.empty()) {
                        abandonSaveAndExitAttempt();
                        return;
                    }
                    armStopSaveAndExit(path);
                    return;
                }
                armStopSaveAndExit();
                return;
            }
            auto has_path = projectHasPath();
            if (has_path && *has_path) {
                lfs::core::events::cmd::SaveAndExit{}
                    .emit();
            } else {
                lfs::core::events::cmd::SaveAsAndExit{}
                    .emit();
            }
        });

        cmd::SetReopenLastProject::when(
            [this, publish_project_error](
                const auto& command) {
                if (!project_lifecycle_) {
                    publish_project_error(
                        "Update Project Settings",
                        "Project lifecycle is unavailable.",
                        gui::error_op::kProjectSettings);
                    return;
                }
                if (auto saved =
                        project_lifecycle_
                            ->setReopenLastProject(
                                command.enabled);
                    !saved) {
                    publish_project_error(
                        "Update Project Settings",
                        saved.error(),
                        gui::error_op::kProjectSettings);
                }
            });

        cmd::SetAutoSaveOnClose::when(
            [this, publish_project_error](
                const auto& command) {
                if (!project_lifecycle_) {
                    publish_project_error(
                        "Update Project Settings",
                        "Project lifecycle is unavailable.",
                        gui::error_op::kProjectSettings);
                    return;
                }
                if (auto saved =
                        project_lifecycle_
                            ->setAutoSaveOnClose(
                                command.enabled);
                    !saved) {
                    publish_project_error(
                        "Update Project Settings",
                        saved.error(),
                        gui::error_op::kProjectSettings);
                }
            });

        cmd::SetProjectAutosaveInterval::when(
            [this, publish_project_error](
                const auto& command) {
                if (!project_lifecycle_) {
                    publish_project_error(
                        "Update Project Settings",
                        "Project lifecycle is unavailable.",
                        gui::error_op::kProjectSettings);
                    return;
                }
                if (auto saved =
                        project_lifecycle_
                            ->setAutosaveIntervalSeconds(
                                command.seconds);
                    !saved) {
                    publish_project_error(
                        "Update Project Settings",
                        saved.error(),
                        gui::error_op::kProjectSettings);
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
            if (project_lifecycle_) {
                project_lifecycle_->markSceneMutation(
                    event.mutation_flags);
            }
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

        const auto sync_viewer_mip_filter_with_training = [this] {
            if (!rendering_manager_ || !trainer_manager_)
                return;
            const auto* trainer = trainer_manager_->getTrainer();
            if (!trainer)
                return;

            auto settings = rendering_manager_->getSettings();
            const bool training_mip_filter = trainer->getParams().optimization.mip_filter;
            if (settings.mip_filter == training_mip_filter)
                return;

            settings.mip_filter = training_mip_filter;
            rendering_manager_->updateSettings(settings);
            LOG_INFO("Synced viewer mip filter with training: {}", training_mip_filter ? "enabled" : "disabled");
        };

        // Trainer ready signal
        internal::TrainerReady::when([this, sync_viewer_mip_filter_with_training](const auto&) {
            sync_viewer_mip_filter_with_training();
            internal::TrainingReadyToStart{}.emit();
        });

        // Training started - switch to splat rendering without hijacking scene selection
        state::TrainingStarted::when([this, sync_viewer_mip_filter_with_training](const auto&) {
            sync_viewer_mip_filter_with_training();

            ui::PointCloudModeChanged{
                .enabled = false,
                .voxel_size = 0.01f}
                .emit();

            LOG_INFO("Switched to splat rendering mode (training started)");
        });

        state::TrainingResumed::when([sync_viewer_mip_filter_with_training](const auto&) {
            sync_viewer_mip_filter_with_training();
        });

        // Training completed - update content type
        state::TrainingCompleted::when([this](const auto& event) {
            handleTrainingCompleted(event);
        });

        // File loading commands
        cmd::LoadConfigFile::when([this](const auto& cmd) {
            handleLoadConfigFile(cmd.path);
        });

        // Signal bridge event handlers
        state::TrainingProgress::when([](const auto& event) {
            auto& store = app_store();
            lfs::core::reactive::BatchUpdate batch(store.store());
            store.iteration.set(event.iteration);
            store.loss.set(event.loss);
            store.num_gaussians.set(static_cast<std::int64_t>(event.num_gaussians));
        });

        state::TrainingStarted::when([this](const auto& event) {
            auto& store = app_store();
            lfs::core::reactive::BatchUpdate batch(store.store());
            store.trainer_loaded.set(true);
            store.training_running.set(true);
            store.training_state.set("running");
            store.total_iterations.set(event.total_iterations);
            python::update_trainer_loaded(true, event.total_iterations);
            python::update_training_state(true, "running");
        });

        state::TrainingPaused::when([](const auto&) {
            auto& store = app_store();
            lfs::core::reactive::BatchUpdate batch(store.store());
            store.training_running.set(false);
            store.training_state.set("paused");
            python::update_training_state(false, "paused");
        });

        state::TrainingResumed::when([](const auto&) {
            auto& store = app_store();
            lfs::core::reactive::BatchUpdate batch(store.store());
            store.training_running.set(true);
            store.training_state.set("running");
            python::update_training_state(true, "running");
        });

        state::TrainingCompleted::when([](const auto& event) {
            const char* state = !event.success       ? "error"
                                : event.user_stopped ? "stopped"
                                                     : "completed";
            auto& store = app_store();
            lfs::core::reactive::BatchUpdate batch(store.store());
            store.training_running.set(false);
            store.training_state.set(state);
            python::update_training_state(false, state);
        });

        internal::TrainerReady::when([this](const auto&) {
            auto& store = app_store();
            lfs::core::reactive::BatchUpdate batch(store.store());
            store.trainer_loaded.set(true);
            store.training_running.set(false);
            store.training_state.set("ready");
            store.total_iterations.set(trainer_manager_->getTotalIterations());
            store.iteration.set(trainer_manager_->getCurrentIteration());
            store.loss.set(trainer_manager_->getCurrentLoss());
            store.num_gaussians.set(
                static_cast<std::int64_t>(trainer_manager_->getNumSplats()));
            if (const auto last =
                    trainer_manager_->getLastEvaluationMetrics()) {
                store.eval_psnr.set(last->psnr);
                store.eval_ssim.set(last->ssim);
            }
            python::update_trainer_loaded(true, trainer_manager_->getTotalIterations(),
                                          trainer_manager_->getCurrentIteration());
            python::update_training_state(false, "ready");
        });

        state::EvaluationCompleted::when([](const auto& event) {
            auto& store = app_store();
            lfs::core::reactive::BatchUpdate batch(store.store());
            store.eval_psnr.set(event.psnr);
            store.eval_ssim.set(event.ssim);
        });

        state::SceneLoaded::when([](const auto& event) {
            app_store().scene_generation.set(python::get_scene_generation());
            const std::string path_utf8 = core::path_to_utf8(event.path);
            python::update_scene(true, path_utf8.c_str());
        });

        state::SceneCleared::when([](const auto&) {
            app_store().scene_generation.set(python::get_scene_generation());
            python::update_scene(false, "");
        });

        // Asset Manager save event handler
        cmd::SaveAsset::when([this](const auto& cmd) {
            python::invoke_save_asset(cmd.node_name);
        });

        cmd::SaveAssetById::when([this](const auto& cmd) {
            if (!scene_manager_)
                return;
            const auto* node = scene_manager_->getScene().getNodeById(static_cast<core::NodeId>(cmd.node_id));
            if (node)
                python::invoke_save_asset(node->name);
        });
    }

    bool VisualizerImpl::initialize() {
        if (fully_initialized_) {
            LOG_TRACE("Already fully initialized");
            return true;
        }

        // Initialize window first and ensure it has proper size
        if (!window_initialized_) {
            {
                LOG_TIMER("startup.window_manager.init"); // wraps Vulkan instance/device/swapchain bring-up
                if (!window_manager_->init()) {
                    return false;
                }
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

        // Initialize GUI systems.
        if (!gui_initialized_) {
            LOG_TIMER("startup.gui_manager.init");
            gui_manager_->init();
            gui_initialized_ = true;
        }

        // InputController requires the GUI focus state to be initialized.
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

        {
            LOG_TIMER("startup.python.ensure_initialized");
            (void)python::ensure_initialized();
        }
        {
            LOG_TIMER("startup.python.builtin_ui_registered");
            python::ensure_builtin_ui_registered();
        }
        {
            LOG_TIMER("startup.window.showWindow");
            window_manager_->showWindow();
        }

        fully_initialized_ = true;
        if (!startup_project_open_attempted_) {
            startup_project_open_attempted_ = true;
            if (project_lifecycle_) {
                project_lifecycle_->openStartupProject(
                    options_.startup_project);
            }
        }
        return true;
    }

    void VisualizerImpl::update() {
        const auto update_started_at = std::chrono::steady_clock::now();
        const bool preload_running_at_start = python::is_plugin_preload_running();
        update_work_processed_ = false;
        window_manager_->updateWindowSize();

        if (fully_initialized_ && gui_frame_rendered_ && !startup_plugin_preload_started_) {
            startup_plugin_preload_started_ = true;
            LOG_TIMER("startup.python.preload_plugins_async");
            python::preload_user_plugins_async();
        }

        const auto plugin_load_status = python::get_startup_plugin_load_status();
        if (gui_manager_ &&
            plugin_load_status.revision != startup_plugin_load_status_revision_) {
            const bool plugin_load_started = plugin_load_status.state != "not_started";
            gui_manager_->setStartupPluginLoadState(
                plugin_load_started,
                plugin_load_status.active,
                plugin_load_status.progress,
                plugin_load_status.detail);
            assert(!plugin_load_started || !gui_manager_->isStartupBlockingInput());
            startup_plugin_load_status_revision_ = plugin_load_status.revision;
            if (!plugin_load_status.active) {
                MainLoop::installInterruptHandlers();
            }
        }
        if (startup_plugin_preload_started_ &&
            !gui_panels_ready_emitted_ &&
            project::
                pluginPreloadTerminalForGuiPanels(
                    startup_plugin_preload_started_,
                    plugin_load_status.state)) {
            MainLoop::installInterruptHandlers();
            gui_panels_ready_emitted_ = true;
            internal::GuiPanelsReady{
                .registration_revision =
                    gui::PanelRegistry::instance()
                        .registration_revision()}
                .emit();
        }

        // Process MCP work queue
        {
            std::vector<WorkItem> work;
            {
                std::lock_guard lock(work_queue_mutex_);
                work.swap(work_queue_);
            }
            update_work_processed_ = !work.empty();
            runPostedWork(work, "viewer", viewer_thread_id_);
        }
        if (project_lifecycle_) {
            project_lifecycle_
                ->updateMaintenance();
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

        if (pending_training_completion_refresh_frames_ > 0 &&
            (!trainer_manager_ || !trainer_manager_->isTrainingActive())) {
            --pending_training_completion_refresh_frames_;
            if (rendering_manager_) {
                rendering_manager_->markDirty(DirtyFlag::ALL);
            }
            wakeMainLoop();
        }

        if (selection_tool_ && selection_tool_->isEnabled() && tool_context_) {
            selection_tool_->update(*tool_context_);
        }

        if (!gui_frame_rendered_) {
            // Wait for at least one GUI frame to render before loading data
        } else if (!pending_view_paths_.empty()) {
            auto paths = std::exchange(pending_view_paths_, {});
            LOG_INFO("Loading {} splat file(s)", paths.size());
            if (const auto result = data_loader_->loadSplatFiles(paths); !result) {
                LOG_ERROR("Failed to load startup splat batch: {}", result.error());
            }
        } else if (!pending_dataset_path_.empty()) {
            auto path = std::exchange(pending_dataset_path_, {});
            LOG_INFO("Queueing dataset import: {}", lfs::core::path_to_utf8(path));
            const auto& params = data_loader_->getParameters();
            cmd::LoadFile{
                .path = path,
                .is_dataset = true,
                .output_path = params.dataset.output_path,
                .init_path = params.init_path.value_or(std::string{}),
                .centralize_dataset = params.dataset.centralize_dataset,
            }
                .emit();
        }

        // Auto-start training if --train flag was passed
        if (pending_auto_train_ && trainer_manager_ && trainer_manager_->canStart()) {
            pending_auto_train_ = false;
            LOG_INFO("Auto-starting training (--train flag)");
            cmd::StartTraining{}.emit();
        }

        const bool preload_running_at_end = python::is_plugin_preload_running();
        // The transition update can also consume pending startup assets (for example,
        // a --view PLY) after it starts the worker. Sample only steady-state preload
        // updates so unrelated startup I/O is not attributed to plugin loading.
        if (preload_running_at_start) {
            plugin_preload_timing_active_ = true;
            plugin_preload_max_update_stall_ = std::max(
                plugin_preload_max_update_stall_,
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - update_started_at));
        }
        if (plugin_preload_timing_active_ && !preload_running_at_end) {
            const double max_stall_ms =
                std::chrono::duration<double, std::milli>(
                    plugin_preload_max_update_stall_)
                    .count();
            LOG_DEBUG("Plugin preload frame budget: max VisualizerImpl::update stall {:.3f} ms",
                      max_stall_ms);
            plugin_preload_timing_active_ = false;
        }
    }

    void VisualizerImpl::processRenderWorkQueue() {
        std::vector<WorkItem> render_work;
        {
            std::lock_guard lock(work_queue_mutex_);
            render_work.swap(render_work_queue_);
        }
        if (render_work.empty())
            return;

        processing_render_work_ = true;
        runPostedWork(render_work, "render", viewer_thread_id_);
        processing_render_work_ = false;
    }

    bool VisualizerImpl::hasPendingRenderWork() {
        std::lock_guard lock(work_queue_mutex_);
        return !render_work_queue_.empty();
    }

    bool VisualizerImpl::inputFrameRequestsRender() const {
        if (!window_manager_)
            return false;

        const auto& input = window_manager_->frameInput();
        if (!input.had_event)
            return false;
        if (input.window_event)
            return true;

        const bool mouse_button_event = input.mouse_clicked[0] || input.mouse_clicked[1] ||
                                        input.mouse_clicked[2] || input.mouse_released[0] ||
                                        input.mouse_released[1] || input.mouse_released[2];
        const bool keyboard_event = !input.keys_pressed.empty() ||
                                    !input.keys_repeated.empty() ||
                                    !input.keys_released.empty() ||
                                    !input.text_codepoints.empty() ||
                                    !input.text_inputs.empty() ||
                                    input.has_text_editing;
        if (mouse_button_event || keyboard_event || input.mouse_wheel != 0.0f)
            return true;

        if (!input.mouse_moved)
            return false;

        if (input.mouse_down[0] || input.mouse_down[1] || input.mouse_down[2])
            return true;

        if (selection_tool_ && selection_tool_->isEnabled() && gui_manager_ &&
            gui_manager_->isPositionInViewport(input.mouse_x, input.mouse_y)) {
            return true;
        }

        if (gui_manager_ && gui_manager_->passiveMouseMoveNeedsRender(input.mouse_x, input.mouse_y)) {
            return true;
        }

        const auto targets = window_manager_->inputRouter().pointerTargets(input.mouse_x, input.mouse_y);
        const bool targets_gui = targets.hover_target == input::InputTarget::Gui ||
                                 targets.pointer_target == input::InputTarget::Gui;
        if (!targets_gui)
            return false;

        return !gui_manager_;
    }

    VisualizerImpl::FrameDemand VisualizerImpl::collectFrameDemand(const bool viewport_export_locked,
                                                                   const bool drained_store_dirty,
                                                                   const bool consume_python_redraw) {
        FrameDemand demand;
        demand.viewport_export_locked = viewport_export_locked;
        demand.scene_dirty = rendering_manager_ && rendering_manager_->pollDirtyState();
        demand.continuous_input = input_controller_ && input_controller_->isContinuousInputActive();
        const bool plugin_preload_running = python::is_plugin_preload_running();
        demand.python_animation = !plugin_preload_running &&
                                  (python::has_frame_callback() ||
                                   python::has_scene_time_callback());
        demand.python_overlay = !plugin_preload_running && python::has_viewport_draw_handlers();
        demand.python_redraw = consume_python_redraw ? python::consume_redraw_request()
                                                     : python::has_redraw_request();
        demand.gui_animation = (gui_manager_ && gui_manager_->needsAnimationFrame()) || plugin_preload_running;
        demand.input_event = inputFrameRequestsRender();
        demand.posted_work = update_work_processed_;
        demand.render_work = hasPendingRenderWork();
        demand.store_dirty = drained_store_dirty || app_store().store().has_dirty();
        if (auto* vulkan_context = window_manager_ ? window_manager_->getVulkanContext() : nullptr) {
            demand.swapchain_resize_pending = vulkan_context->hasPendingSwapchainResize();
            demand.swapchain_resize_ready = demand.swapchain_resize_pending &&
                                            vulkan_context->pendingSwapchainResizeReady();
        }
#if defined(__linux__)
        if (window_manager_) {
            demand.window_resize_paint_pending =
                window_manager_->hasRecentWindowSizeChange(kWindowResizePaintDemandWindow);
        }
#endif
        demand.viewport_resize_deferring = rendering_manager_ &&
                                           rendering_manager_->isViewportResizeDeferring();
        demand.viewport_resize_settle_ready = rendering_manager_ &&
                                              rendering_manager_->viewportResizeSettleReady();
        return demand;
    }

    void VisualizerImpl::waitForNextEvent(const bool is_training) {
        if (!window_manager_)
            return;

        auto wait_seconds = is_training ? 0.1 : 0.5;
        if (rendering_manager_ && rendering_manager_->hasPendingViewportResizeSettle()) {
            const double settle_wait = rendering_manager_->secondsUntilViewportResizeSettleReady();
            wait_seconds = std::min(wait_seconds,
                                    std::max(kResizeSettleMinWaitSeconds, settle_wait));
        }

        // Wake exactly when a pending tooltip is due so the reveal costs a single
        // frame instead of rendering continuously through the hover delay.
        // Also min against RmlUi scheduled-update deadlines (CSS transitions, timers)
        // so finite-delay panels wake on time without gui_animation spin.
        if (gui_manager_) {
            if (const auto tooltip_wait = gui_manager_->secondsUntilTooltipReveal())
                wait_seconds = std::min(wait_seconds,
                                        std::max(kTooltipRevealMinWaitSeconds, *tooltip_wait));
            if (const auto anim_wait = gui_manager_->secondsUntilNextAnimationFrame())
                wait_seconds = std::min(wait_seconds,
                                        std::max(kGuiScheduledUpdateMinWaitSeconds, *anim_wait));
        }

        // Wake no later than a Python-scheduled redraw deadline (paced overlay animation).
        if (const auto redraw_wait = python::seconds_until_scheduled_redraw())
            wait_seconds = std::min(wait_seconds,
                                    std::max(kScheduledRedrawMinWaitSeconds, *redraw_wait));

        window_manager_->waitEvents(wait_seconds);
    }

    void VisualizerImpl::render() {

        auto now = std::chrono::high_resolution_clock::now();
        float delta_time = std::chrono::duration<float>(now - last_frame_time_).count();
        last_frame_time_ = now;

        // Clamp delta time to prevent huge jumps (min 30 FPS)
        delta_time = std::min(delta_time, 1.0f / 30.0f);

        const bool viewport_export_locked = gui_manager_ && gui_manager_->isViewportExportLocked();
        if (window_manager_) {
            window_manager_->updateWindowSize("render_begin");
        }
        if (viewport_export_locked && window_manager_) {
            window_manager_->pollEvents();
        }

        // Tick Python frame callback for animations
        if (!python::is_plugin_preload_running() && python::has_frame_callback()) {
            python::tick_frame_callback(delta_time);
            if (rendering_manager_) {
                rendering_manager_->markDirty(DirtyFlag::ALL);
            }
        }

        if (!python::is_plugin_preload_running()) {
            if (python::has_scene_time_callback()) {
                live_scene_clip_time_ += delta_time;
                python::tick_scene_time_callback(live_scene_clip_time_);
                if (rendering_manager_) {
                    rendering_manager_->markDirty(DirtyFlag::ALL);
                }
            } else {
                live_scene_clip_time_ = 0.0f;
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
            const bool startup_overlay_blocking =
                gui_manager_ && gui_manager_->isStartupBlockingInput();
            if (!viewport_export_locked && !startup_overlay_blocking) {
                input_controller_->update(delta_time);
            }
        }

        if (gui_manager_) {
            gui_manager_->updateInteractiveTransitions();
        }
        const bool interactive_transition_settling =
            gui_manager_ && gui_manager_->isInteractiveTransitionSettling();

        // Get viewport region from GUI. This accounts for menu/tool/status panels and must be
        // shared by every graphics backend so camera aspect and render resolution match the viewport.
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

        RenderingManager::RenderContext context{
            .viewport = viewport_,
            .settings = rendering_manager_->getSettings(),
            .logical_screen_size = window_manager_->getWindowSize(),
            .viewport_region = has_viewport_region ? &viewport_region : nullptr,
            .scene_manager = scene_manager_.get(),
            .vulkan_context = window_manager_->getVulkanContext()};

        if (gui_manager_) {
            rendering_manager_->setCropboxGizmoActive(gui_manager_->gizmo().isCropboxGizmoActive());
            rendering_manager_->setEllipsoidGizmoActive(gui_manager_->gizmo().isEllipsoidGizmoActive());
        }

        bool store_dirty = false;
        {
            LOG_TIMER_THRESHOLD("gui_render.reactive_store_drain", 0.05);
            store_dirty = app_store().store().drain_dirty_into_frame();
        }

        if (gui_manager_)
            gui_manager_->sequencerUI().tickPlaybackBeforeSceneRender();

        const bool is_training = trainer_manager_ && trainer_manager_->isRunning();
        const FrameDemand frame_demand = collectFrameDemand(viewport_export_locked, store_dirty);
        if (gui_frame_rendered_ && !frame_demand.shouldRenderFrame()) {
            LOG_PERF("loop_idle skip_gui_render=true needs_render={} continuous_input={} py_anim={} py_overlay={} py_redraw={} gui_anim={} input_event={} posted_work={} render_work={} store_dirty={} swapchain_resize_pending={} swapchain_resize_ready={} window_resize_paint_pending={} viewport_resize_deferring={} viewport_resize_settle_ready={}",
                     frame_demand.scene_dirty,
                     frame_demand.continuous_input,
                     frame_demand.python_animation,
                     frame_demand.python_overlay,
                     frame_demand.python_redraw,
                     frame_demand.gui_animation,
                     frame_demand.input_event,
                     frame_demand.posted_work,
                     frame_demand.render_work,
                     frame_demand.store_dirty,
                     frame_demand.swapchain_resize_pending,
                     frame_demand.swapchain_resize_ready,
                     frame_demand.window_resize_paint_pending,
                     frame_demand.viewport_resize_deferring,
                     frame_demand.viewport_resize_settle_ready);
            if (!python::is_plugin_preload_running()) {
                python::flush_signals();
            }
            waitForNextEvent(is_training);
            return;
        }

        std::optional<std::chrono::steady_clock::time_point>
            project_frame_started;
        if (!viewport_export_locked && !interactive_transition_settling &&
            !frame_state_.scene_render_suspended()) {
            if (!python::is_plugin_preload_running() && frame_demand.python_redraw && gui_manager_)
                gui_manager_->syncVisiblePanelsBeforeSceneRender();

            project_frame_started =
                std::chrono::steady_clock::now();
            const auto vulkan_frame = rendering_manager_->renderVulkanFrame(context);
            {
                auto& interop = rendering_manager_->viewportInterop();
                if (vulkan_frame.external_image != VK_NULL_HANDLE) {
                    interop.setExternalSceneImage(vulkan_frame.external_image,
                                                  vulkan_frame.external_image_view,
                                                  vulkan_frame.external_image_layout,
                                                  vulkan_frame.size,
                                                  vulkan_frame.flip_y,
                                                  vulkan_frame.external_image_generation,
                                                  vulkan_frame.completion_semaphore,
                                                  vulkan_frame.completion_value,
                                                  vulkan_frame.alloc_size);
                } else {
                    interop.setSceneImage(
                        vulkan_frame.image,
                        vulkan_frame.size,
                        vulkan_frame.flip_y,
                        vulkan_frame.split_left_image_generation != 0
                            ? vulkan_frame.split_left_image_generation
                            : vulkan_frame.image_generation,
                        vulkan_frame.completion_semaphore,
                        vulkan_frame.completion_value);
                }
                if (vulkan_frame.split_right_image) {
                    interop.setSplitRightImage(
                        vulkan_frame.split_right_image,
                        vulkan_frame.split_right_size,
                        vulkan_frame.split_right_flip_y,
                        vulkan_frame.image_generation);
                } else {
                    interop.clearSplitRightImage();
                }

                // Splat depth -> R32_SFLOAT interop slot for the depth-blit pass.
                const auto mesh_frame = rendering_manager_->getVulkanMeshFrame();
                if (mesh_frame.depth_blit.depth && mesh_frame.depth_blit.depth->is_valid() &&
                    mesh_frame.depth_blit.depth->ndim() == 3 &&
                    mesh_frame.depth_blit.depth->size(0) == 1) {
                    const auto& d = *mesh_frame.depth_blit.depth;
                    interop.setDepthBlitImage(
                        mesh_frame.depth_blit.depth,
                        glm::ivec2(static_cast<int>(d.size(2)), static_cast<int>(d.size(1))),
                        vulkan_frame.image_generation);
                } else {
                    interop.clearDepthBlitImage();
                }
            }
        } else if (interactive_transition_settling) {
            LOG_DEBUG("Skipping Vulkan viewport render during interactive transition settle: scene_dirty={}, gui_animation={}, input_event={}, render_work={}, store_dirty={}",
                      frame_demand.scene_dirty,
                      frame_demand.gui_animation,
                      frame_demand.input_event,
                      frame_demand.render_work,
                      frame_demand.store_dirty);
        }
        if (gui_manager_) {
            LOG_TIMER("VisualizerImpl::render.gui_frame_total_with_swapchain_wait");
            window_manager_->updateWindowSize("pre_gui_render");
            gui_manager_->render();
            window_manager_->refreshResizeCursor();
            // Count presented frames (GUI-only included). Scene FPS still comes
            // from framerate_controller_ inside renderVulkanFrame; this is
            // measurement-only and does not affect pacing.
            if (rendering_manager_) {
                rendering_manager_->notePresentedFrame();
            }
        } else {
            processRenderWorkQueue();
        }
        if (project_frame_started && project_lifecycle_) {
            project_lifecycle_->noteProjectFrameRendered(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    *project_frame_started)
                    .count());
        }

        if (!python::is_plugin_preload_running()) {
            python::flush_signals();
        }
        if (!gui_frame_rendered_) {
            gui_session_restore_.onFirstGuiFrame();
            tryApplyProjectSessionRestore();
        }
        gui_frame_rendered_ = true;
        update_work_processed_ = false;

        // Render-on-demand: VSync handles frame pacing, waitEvents saves CPU when idle
        const FrameDemand next_demand = collectFrameDemand(viewport_export_locked, false, false);

        // Continuous demand that is only python_redraw and/or gui_animation — pace it
        // so GUI-only animation does not free-run against a MAILBOX swapchain.
        const bool gui_only_animation =
            next_demand.needsContinuousLoop() &&
            !python::is_plugin_preload_running() &&
            !next_demand.scene_dirty && !next_demand.continuous_input &&
            !next_demand.python_animation && !next_demand.python_overlay &&
            !next_demand.input_event && !next_demand.posted_work &&
            !next_demand.render_work && !next_demand.store_dirty &&
            !next_demand.swapchain_resize_pending && !next_demand.swapchain_resize_ready &&
            !next_demand.window_resize_paint_pending && !next_demand.viewport_resize_deferring &&
            !next_demand.viewport_resize_settle_ready && !next_demand.viewport_export_locked;

        const auto py_redraw_due = python::seconds_until_scheduled_redraw();
        const double py_redraw_due_in = py_redraw_due ? *py_redraw_due : -1.0;

        LOG_PERF("loop_end needs_render={} continuous_input={} py_anim={} py_overlay={} py_redraw={} gui_anim={} input_event={} posted_work={} render_work={} store_dirty={} swapchain_resize_pending={} swapchain_resize_ready={} window_resize_paint_pending={} viewport_resize_deferring={} viewport_resize_settle_ready={} gui_only_throttle={} py_redraw_due_in={:.4f}",
                 next_demand.scene_dirty,
                 next_demand.continuous_input,
                 next_demand.python_animation,
                 next_demand.python_overlay,
                 next_demand.python_redraw,
                 next_demand.gui_animation,
                 next_demand.input_event,
                 next_demand.posted_work,
                 next_demand.render_work,
                 next_demand.store_dirty,
                 next_demand.swapchain_resize_pending,
                 next_demand.swapchain_resize_ready,
                 next_demand.window_resize_paint_pending,
                 next_demand.viewport_resize_deferring,
                 next_demand.viewport_resize_settle_ready,
                 gui_only_animation,
                 py_redraw_due_in);

        if (next_demand.needsContinuousLoop()) {
            if (gui_only_animation) {
                // GUI-only animation must not free-run against a MAILBOX swapchain.
                // Cap at kGuiAnimationFrameInterval; waitEvents still wakes instantly on input.
                const double elapsed = std::chrono::duration<double>(
                                           std::chrono::high_resolution_clock::now() - last_frame_time_)
                                           .count();
                if (elapsed >= kGuiAnimationFrameInterval) {
                    window_manager_->pollEvents();
                } else {
                    window_manager_->waitEvents(kGuiAnimationFrameInterval - elapsed);
                }
            } else {
                window_manager_->pollEvents();
            }
        } else {
            // Idle: wait to minimize CPU/GPU work. Mouse-motion-only viewport wakes
            // are filtered at the top of the next loop without presenting a GUI frame.
            waitForNextEvent(is_training);
        }
    }

    bool VisualizerImpl::allowclose() {
        if (!window_manager_->shouldClose()) {
            return false;
        }

        const auto training_blocks_close = [this] {
            if (!trainer_manager_) {
                return false;
            }
            if (trainer_manager_->isCompletionPending()) {
                return true;
            }
            if (!trainer_manager_->isTrainingActive()) {
                return false;
            }
            return !trainer_manager_->isPausedAtCheckpointBaseline();
        };

        const auto finish_close = [this] {
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
        };

        // ForceExit / Discard: skip save. If training is still
        // running, stop and wait (terminal write may still run as
        // crash insurance but is not adopted).
        if (gui_manager_ && gui_manager_->isForceExit()) {
            if (project_lifecycle_) {
                project_lifecycle_
                    ->markApplicationClosePending();
                project_lifecycle_
                    ->setSuppressTrainingAdoption(true);
            }
            if (training_blocks_close()) {
                armForceExitCompletionWatcher();
                if (force_exit_wait_expired_.load(
                        std::memory_order_acquire)) {
                    return finish_close();
                }
                pending_training_action_ =
                    PendingTrainingAction::CloseDiscard;
                trainer_manager_->suppressCompletionNotification();
                if (trainer_manager_->canStop()) {
                    trainer_manager_->stopTraining();
                }
                window_manager_->cancelClose();
                return false;
            }
            return finish_close();
        }

        // Training Running/Paused/Stopping/completion-pending:
        // prompt first; do not stop until the user chooses.
        if (training_blocks_close()) {
            if (project_lifecycle_) {
                project_lifecycle_
                    ->markApplicationClosePending();
            }
            if (pending_training_action_ ==
                    PendingTrainingAction::CloseSave ||
                pending_training_action_ ==
                    PendingTrainingAction::CloseDiscard) {
                window_manager_->cancelClose();
                return false;
            }
            if (!gui_manager_) {
                LOG_ERROR(
                    "Training is active and no GUI prompt is available; refusing to close");
                window_manager_->cancelClose();
                return false;
            }
            if (!gui_manager_->isExitConfirmationPending()) {
                gui_manager_->requestExitConfirmation(
                    true);
            }
            window_manager_->cancelClose();
            return false;
        }

        if (project_lifecycle_) {
            const auto close_save =
                project_lifecycle_
                    ->beginOrPollCloseSave();
            switch (close_save) {
            case project::ProjectLifecycle::
                CloseSaveStatus::NotDirty:
            case project::ProjectLifecycle::
                CloseSaveStatus::Succeeded:
                return finish_close();
            case project::ProjectLifecycle::
                CloseSaveStatus::Saving:
                if (gui_manager_ &&
                    gui_manager_->isForceExit()) {
                    project_lifecycle_
                        ->markApplicationClosePending();
                    project_lifecycle_
                        ->setSuppressTrainingAdoption(
                            true);
                    return finish_close();
                }
                if (!close_save_notice_posted_) {
                    close_save_notice_posted_ = true;
                    lfs::ErrorBus::instance().publish(
                        makeFrameNotification(
                            lfs::ErrorCode::Unavailable,
                            lfs::ErrorDomain::App,
                            lfs::Severity::Info,
                            lfs::ErrorSurface::
                                StatusOnly,
                            "Saving project before exit...",
                            "The window will close after the durable .licht head is published.",
                            {},
                            LFS_SOURCE_SITE_CURRENT()));
                }
                window_manager_->cancelClose();
                return false;
            case project::ProjectLifecycle::
                CloseSaveStatus::Failed: {
                const auto detail =
                    project_lifecycle_
                        ->closeSaveError();
                lfs::ErrorBus::instance().publish(
                    makeFrameNotification(
                        lfs::ErrorCode::Unavailable,
                        lfs::ErrorDomain::IO,
                        lfs::Severity::Error,
                        lfs::ErrorSurface::Toast,
                        "The project could not be saved before exit.",
                        detail,
                        {},
                        LFS_SOURCE_SITE_CURRENT(),
                        gui::error_op::kSave));
                break;
            }
            case project::ProjectLifecycle::
                CloseSaveStatus::NeedsPrompt:
                break;
            }
        } else {
            return finish_close();
        }

        if (!gui_manager_) {
            LOG_ERROR(
                "A dirty project could not be saved and no GUI prompt is available; refusing to close");
            window_manager_->cancelClose();
            return false;
        }
        if (!gui_manager_->isExitConfirmationPending()) {
            gui_manager_->requestExitConfirmation(false);
        }
        window_manager_->cancelClose();
        return false;
    }

    void VisualizerImpl::shutdown() {
        if (trainer_manager_ &&
            (trainer_manager_->isTrainingActive() || trainer_manager_->isCompletionPending())) {
            LOG_CRITICAL("Shutdown reached before the training worker was reaped");
            return;
        }

        beginShutdown();

        if (trainer_manager_) {
            trainer_manager_.reset();
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

        if (params.cli_bg_color_set && rendering_manager_) {
            auto render_settings = rendering_manager_->getSettings();
            const auto& color = params.optimization.bg_color;
            render_settings.background_color = glm::vec3(color[0], color[1], color[2]);
            rendering_manager_->updateSettings(render_settings);
        }
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

    std::expected<void, std::string> VisualizerImpl::clearScene() {
        if (!data_loader_) {
            return std::unexpected("No data loader available");
        }

        if (data_loader_->clearScene()) {
            return {};
        }

        if (trainer_manager_ && scene_manager_ &&
            scene_manager_->getContentType() == SceneManager::ContentType::Dataset &&
            !trainer_manager_->canPerform(TrainingAction::ClearScene)) {
            return std::unexpected(
                std::string(trainer_manager_->getActionBlockedReason(TrainingAction::ClearScene)));
        }

        return std::unexpected("Scene clear request was rejected");
    }

    bool VisualizerImpl::shouldDeferProjectSwitchForTraining() const {
        return trainer_manager_ &&
               (trainer_manager_->isTrainingActive() ||
                !trainer_manager_->canPerform(TrainingAction::ClearScene) ||
                trainer_manager_->isCompletionPending());
    }

    void VisualizerImpl::requestStopThenPendingAction() {
        if (!trainer_manager_) {
            return;
        }
        trainer_manager_->suppressCompletionNotification();
        if (trainer_manager_->canStop()) {
            trainer_manager_->stopTraining();
        }
    }

    void VisualizerImpl::handleNewProject(
        const ProjectSwitchDisposition disposition,
        const bool stop_training) {
        if (pending_training_action_ ==
                PendingTrainingAction::CloseSave ||
            pending_training_action_ ==
                PendingTrainingAction::CloseDiscard) {
            return;
        }
        if (!project_lifecycle_) {
            return;
        }
        if (auto preflight =
                project_lifecycle_
                    ->preflightSwitch(
                        disposition, stop_training);
            !preflight) {
            if (isDirtyProjectSwitchError(
                    preflight.error())) {
                lfs::core::events::cmd::
                    ShowProjectSwitchConfirmation{
                        .new_project = true,
                        .path = {}}
                        .emit();
                return;
            }
            if (!stop_training &&
                isTrainingProjectSwitchError(
                    preflight.error()) &&
                trainer_manager_ &&
                trainer_manager_->isTrainingActive()) {
                lfs::core::events::cmd::
                    ShowStopTrainingConfirmation{
                        .new_project = true,
                        .path = {},
                        .discard_changes =
                            disposition ==
                            ProjectSwitchDisposition::
                                DiscardChanges}
                        .emit();
                return;
            }
            LOG_ERROR(
                "New Project preflight failed: {}",
                lfs::format_for_developer(
                    preflight.error()));
            lfs::Error contextual = preflight.error();
            lfs::ErrorBus::instance().publish(lfs::ErrorNotification{
                .error = std::move(contextual).with_context(gui::error_op::kNewProject, LFS_SOURCE_SITE_CURRENT()),
                .surface = lfs::ErrorSurface::Toast,
                .actions = {},
                .operation_id = lfs::OperationId::generate(),
            });
            return;
        }
        if (gui_manager_) {
            gui_manager_->asyncTasks().cancelImport();
        }

        pending_view_paths_.clear();
        pending_dataset_path_.clear();
        pending_auto_train_ = false;
        pending_open_path_.clear();
        if (pending_training_action_ == PendingTrainingAction::Reset) {
            pending_training_action_ = PendingTrainingAction::None;
        }

        if (shouldDeferProjectSwitchForTraining()) {
            pending_training_action_ = PendingTrainingAction::NewProject;
            pending_new_project_disposition_ =
                disposition;
            requestStopThenPendingAction();
            return;
        }

        pending_training_action_ = PendingTrainingAction::None;
        performNewProject(disposition);
    }

    void VisualizerImpl::performNewProject(
        const ProjectSwitchDisposition disposition) {
        if (!project_lifecycle_) {
            return;
        }
        if (auto created =
                project_lifecycle_->newProject(
                    disposition);
            !created) {
            LOG_ERROR(
                "New Project failed: {}",
                lfs::format_for_developer(
                    created.error()));
            lfs::Error contextual = created.error();
            lfs::ErrorBus::instance().publish(lfs::ErrorNotification{
                .error = std::move(contextual).with_context(gui::error_op::kNewProject, LFS_SOURCE_SITE_CURRENT()),
                .surface = lfs::ErrorSurface::Toast,
                .actions = {},
                .operation_id = lfs::OperationId::generate(),
            });
        }
    }

    void VisualizerImpl::handleOpenProject(
        const std::filesystem::path& path,
        const ProjectSwitchDisposition disposition,
        const bool stop_training) {
        if (pending_training_action_ ==
                PendingTrainingAction::CloseSave ||
            pending_training_action_ ==
                PendingTrainingAction::CloseDiscard) {
            return;
        }
        if (!project_lifecycle_) {
            return;
        }
        if (auto preflight =
                project_lifecycle_
                    ->preflightSwitch(
                        disposition, stop_training);
            !preflight) {
            if (isDirtyProjectSwitchError(
                    preflight.error())) {
                lfs::core::events::cmd::
                    ShowProjectSwitchConfirmation{
                        .new_project = false,
                        .path = path}
                        .emit();
                return;
            }
            if (!stop_training &&
                isTrainingProjectSwitchError(
                    preflight.error()) &&
                trainer_manager_ &&
                trainer_manager_->isTrainingActive()) {
                lfs::core::events::cmd::
                    ShowStopTrainingConfirmation{
                        .new_project = false,
                        .path = path,
                        .discard_changes =
                            disposition ==
                            ProjectSwitchDisposition::
                                DiscardChanges}
                        .emit();
                return;
            }
            LOG_ERROR(
                "Open Project preflight failed: {}",
                lfs::format_for_developer(
                    preflight.error()));
            lfs::Error contextual = preflight.error();
            lfs::ErrorBus::instance().publish(lfs::ErrorNotification{
                .error = std::move(contextual).with_context(gui::error_op::kOpenProject, LFS_SOURCE_SITE_CURRENT()),
                .surface = lfs::ErrorSurface::Toast,
                .actions = {},
                .operation_id = lfs::OperationId::generate(),
            });
            return;
        }
        if (gui_manager_) {
            gui_manager_->asyncTasks().cancelImport();
        }

        if (shouldDeferProjectSwitchForTraining()) {
            pending_training_action_ =
                PendingTrainingAction::OpenProject;
            pending_open_path_ = path;
            pending_open_disposition_ = disposition;
            requestStopThenPendingAction();
            return;
        }

        pending_training_action_ = PendingTrainingAction::None;
        performOpenProject(path, disposition);
    }

    void VisualizerImpl::performOpenProject(
        const std::filesystem::path& path,
        const ProjectSwitchDisposition disposition) {
        if (auto opened = projectOpen(path, disposition);
            !opened) {
            if (isDirtyProjectSwitchError(
                    opened.error())) {
                lfs::core::events::cmd::
                    ShowProjectSwitchConfirmation{
                        .new_project = false,
                        .path = path}
                        .emit();
                return;
            }
            LOG_ERROR(
                "Open Project failed: {}",
                lfs::format_for_developer(
                    opened.error()));
            lfs::Error contextual = opened.error();
            lfs::ErrorBus::instance().publish(lfs::ErrorNotification{
                .error = std::move(contextual).with_context(gui::error_op::kOpenProject, LFS_SOURCE_SITE_CURRENT()),
                .surface = lfs::ErrorSurface::Toast,
                .actions = {},
                .operation_id = lfs::OperationId::generate(),
            });
        }
    }

    void VisualizerImpl::resetProjectState(const bool reset_panel_registry) {
        if (trainer_manager_) {
            trainer_manager_->clearRestoredProjectMetrics();
        }
        if (auto* const param_mgr = services().paramsOrNull()) {
            param_mgr->clearSession();
        }

        if (data_loader_) {
            data_loader_->setParameters({});
        }

        pending_view_paths_.clear();
        pending_dataset_path_.clear();
        pending_auto_train_ = false;
        pending_training_action_ = PendingTrainingAction::None;
        pending_training_action_posted_ = false;
        pending_new_project_disposition_ =
            ProjectSwitchDisposition::RequireClean;
        pending_open_path_.clear();
        pending_open_disposition_ =
            ProjectSwitchDisposition::RequireClean;
        gui_session_restore_.clear();
        pending_project_tools_restore_.reset();
        hydration_terminal_restore_ticket_.reset();
        retained_project_session_ = {};
        camera_bookmarks_.clear();
        if (reset_panel_registry) {
            gui::PanelRegistry::instance()
                .reset_project_state();
        }
    }

    lfs::Result<
        lfs::io::project::ProjectSessionChapters>
    VisualizerImpl::captureProjectSession(
        lfs::io::project::ReferencesChapter* references,
        const std::filesystem::path& project_root)
        const {
        return project::captureGuiSession(
            *this, retained_project_session_,
            camera_bookmarks_, references,
            project_root);
    }

    project::GuiSessionRestoreTicket VisualizerImpl::
        stagePreparedProjectSessionRestore(
            project::PreparedGuiSessionRestore
                prepared) {
        const auto ticket = gui_session_restore_.stagePrepared(
            std::move(prepared));
        tryApplyProjectSessionRestore();
        return ticket;
    }

    void VisualizerImpl::
        tryApplyProjectSessionRestore() {
        auto prepared =
            gui_session_restore_.takeReady();
        if (!prepared)
            return;
        auto retained = prepared->chapters;
        const auto ticket = prepared->ticket;
        project::applyGuiSession(
            *this, *prepared, camera_bookmarks_);
        pending_project_tools_restore_ =
            std::move(*prepared);
        retained_project_session_ =
            std::move(retained);
        LOG_INFO(
            "Applied GUIL/VIEW/EDTR/SEQR/METR after first GUI frame and panel registration revision {}",
            gui_session_restore_
                .panelsRegistrationRevision());
        // Tools apply from whichever of owner-ready and hydration-terminal
        // finishes last for this ticket.
        if (hydration_terminal_restore_ticket_ == ticket &&
            ticket != 0) {
            tryApplyProjectSessionTools(ticket);
        }
    }

    void VisualizerImpl::tryApplyProjectSessionTools(
        const project::GuiSessionRestoreTicket ticket) {
        if (!pending_project_tools_restore_ ||
            pending_project_tools_restore_->ticket != ticket) {
            return;
        }
        auto prepared =
            std::move(*pending_project_tools_restore_);
        pending_project_tools_restore_.reset();
        editor_context_.update(
            scene_manager_.get(), trainer_manager_.get());
        project::applyGuiSessionTools(*this, prepared);
    }

    void VisualizerImpl::noteHydrationTerminalForRestoreTicket(
        const project::GuiSessionRestoreTicket ticket) {
        if (ticket == 0)
            return;
        hydration_terminal_restore_ticket_ = ticket;
        tryApplyProjectSessionTools(ticket);
    }

    void VisualizerImpl::noteGuiSessionRestoreOwnerReady(
        const std::uint64_t panels_registration_revision) {
        gui_session_restore_.onFirstGuiFrame();
        gui_session_restore_.onPanelsReady(
            panels_registration_revision);
        tryApplyProjectSessionRestore();
    }

    void VisualizerImpl::bindTrainerProjectSnapshotTarget() {
        if (project_lifecycle_) {
            project_lifecycle_->bindTrainerSnapshotTarget();
        }
    }

    void VisualizerImpl::deactivateProjectTools() {
        lfs::core::events::tools::SetToolbarTool{
            .tool_mode = static_cast<int>(ToolType::None)}
            .emit();
        editor_context_.setActiveTool(ToolType::None);
        editor_context_.clearActiveOperator();
        UnifiedToolRegistry::instance().clearActiveTool();
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
        if (trainer_manager_->isPaused()) {
            if (project_lifecycle_) {
                if (auto* const trainer = getTrainer()) {
                    const auto policy =
                        trainer->trainer_project_save_policy();
                    if (!policy.on_completion &&
                        !policy.on_stop_or_error &&
                        !policy.at_step_boundaries) {
                        if (auto prepared =
                                project_lifecycle_
                                    ->prepareTrainingStartProject();
                            !prepared) {
                            return std::unexpected(
                                lfs::format_for_developer(
                                    prepared.error()));
                        }
                    }
                }
            }
            trainer_manager_->resumeTraining();
            return {};
        }
        if (project_lifecycle_) {
            if (auto prepared =
                    project_lifecycle_
                        ->prepareTrainingStartProject();
                !prepared) {
                return std::unexpected(
                    lfs::format_for_developer(
                        prepared.error()));
            }
        }
        if (!trainer_manager_->startTraining())
            return std::unexpected("Failed to start training");
        return {};
    }

    std::optional<int>
    VisualizerImpl::trainingStartOverwriteConflict() {
        if (!project_lifecycle_) {
            return std::nullopt;
        }
        return project_lifecycle_
            ->trainingStartOverwriteConflict();
    }

    lfs::Result<void>
    VisualizerImpl::projectSave(
        const bool regenerate_preview) {
        if (!project_lifecycle_) {
            return visualizerFailure<void>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle is unavailable.",
                "The visualizer did not initialize its project lifecycle service",
                "project.lifecycle");
        }
        return project_lifecycle_->save(
            regenerate_preview);
    }

    lfs::Result<void>
    VisualizerImpl::projectSaveAs(
        const std::filesystem::path& path,
        const bool regenerate_preview) {
        if (!project_lifecycle_) {
            return visualizerFailure<void>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle is unavailable.",
                "The visualizer did not initialize its project lifecycle service",
                "project.lifecycle");
        }
        return project_lifecycle_->saveAs(
            path, regenerate_preview);
    }

    lfs::Result<void>
    VisualizerImpl::projectSaveAsExplicit(
        const std::filesystem::path& path,
        const bool regenerate_preview) {
        if (!project_lifecycle_) {
            return visualizerFailure<void>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle is unavailable.",
                "The visualizer did not initialize its project lifecycle service",
                "project.lifecycle");
        }
        return project_lifecycle_->saveAs(
            path, regenerate_preview, true);
    }

    lfs::Result<void>
    VisualizerImpl::projectSaveAsFromDialog(
        const std::filesystem::path& path,
        const bool regenerate_preview) {
        if (!project_lifecycle_) {
            return visualizerFailure<void>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle is unavailable.",
                "The visualizer did not initialize its project lifecycle service",
                "project.lifecycle");
        }
        return project_lifecycle_->saveAs(
            path, regenerate_preview, true);
    }

    bool VisualizerImpl::projectContainsEmbeddedSecrets()
        const {
        return project_lifecycle_ &&
               project_lifecycle_->containsEmbeddedSecrets();
    }

    void VisualizerImpl::abandonSaveAndExitAttempt() {
        pending_close_save_path_.reset();
        if (project_lifecycle_) {
            project_lifecycle_->resetCloseSaveAttempt();
        }
        close_save_notice_posted_ = false;
        if (gui_manager_) {
            gui_manager_->setForceExit(false);
            gui_manager_->dismissExitConfirmation();
        }
        if (window_manager_) {
            window_manager_->cancelClose();
        }
    }

    void VisualizerImpl::armStopSaveAndExit(
        std::optional<std::filesystem::path>
            untitled_destination) {
        if (gui_manager_) {
            gui_manager_->setForceExit(false);
            gui_manager_->dismissExitConfirmation();
        }
        if (project_lifecycle_) {
            project_lifecycle_
                ->markApplicationClosePending();
            project_lifecycle_
                ->setSuppressTrainingAdoption(false);
        }
        if (untitled_destination) {
            pending_close_save_path_ =
                *untitled_destination;
            if (project_lifecycle_) {
                project_lifecycle_
                    ->bindTrainerSnapshotTarget(
                        *untitled_destination, true);
            }
        }
        pending_training_action_ =
            PendingTrainingAction::CloseSave;
        if (trainer_manager_) {
            trainer_manager_->suppressCompletionNotification();
            if (trainer_manager_->canStop()) {
                trainer_manager_->stopTraining();
            }
        }
    }

    void VisualizerImpl::completeSaveAsAndExit(
        const std::filesystem::path& path) {
        if (path.empty()) {
            abandonSaveAndExitAttempt();
            return;
        }
        if (auto saved =
                projectSaveAsFromDialog(path, false);
            !saved) {
            lfs::Error contextual = saved.error();
            lfs::ErrorBus::instance().publish(lfs::ErrorNotification{
                .error = std::move(contextual).with_context(gui::error_op::kSave, LFS_SOURCE_SITE_CURRENT()),
                .surface = lfs::ErrorSurface::Toast,
                .actions = {},
                .operation_id = lfs::OperationId::generate(),
            });
            abandonSaveAndExitAttempt();
            return;
        }
        if (gui_manager_) {
            gui_manager_->dismissExitConfirmation();
        }
        requestApplicationClose();
    }

    void VisualizerImpl::armForceExitCompletionWatcher() {
        if (force_exit_watcher_armed_) {
            return;
        }
        force_exit_watcher_armed_ = true;
        force_exit_wait_expired_.store(
            false, std::memory_order_release);
        force_exit_completion_watcher_ = std::jthread(
            [this](const std::stop_token stop) {
                const auto deadline =
                    std::chrono::steady_clock::now() +
                    force_exit_completion_timeout_;
                while (!stop.stop_requested()) {
                    if (!trainer_manager_ ||
                        !trainer_manager_
                             ->isCompletionPending()) {
                        break;
                    }
                    if (std::chrono::steady_clock::now() >=
                        deadline) {
                        force_exit_wait_expired_.store(
                            true,
                            std::memory_order_release);
                        break;
                    }
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(50));
                }
                if (!stop.stop_requested()) {
                    requestApplicationClose();
                }
            });
    }

    void VisualizerImpl::stopForceExitCompletionWatcher() {
        if (force_exit_completion_watcher_.joinable()) {
            force_exit_completion_watcher_.request_stop();
            force_exit_completion_watcher_.join();
        }
        force_exit_watcher_armed_ = false;
    }

    lfs::Result<ProjectOpenOutcome>
    VisualizerImpl::projectOpen(
        const std::filesystem::path& path,
        const ProjectSwitchDisposition disposition) {
        if (!project_lifecycle_) {
            return visualizerFailure<ProjectOpenOutcome>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle is unavailable.",
                "The visualizer did not initialize its project lifecycle service",
                "project.lifecycle");
        }
        return project_lifecycle_->open(
            path, disposition);
    }

    lfs::Result<ProjectInfo>
    VisualizerImpl::projectGetInfo() {
        if (!project_lifecycle_) {
            return visualizerFailure<ProjectInfo>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle is unavailable.",
                "The visualizer did not initialize its project lifecycle service",
                "project.lifecycle");
        }
        return project_lifecycle_->info();
    }

    lfs::Result<ProjectWritePoll>
    VisualizerImpl::projectPollWrite() {
        if (!project_lifecycle_) {
            return visualizerFailure<ProjectWritePoll>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle is unavailable.",
                "The visualizer did not initialize its project lifecycle service",
                "project.lifecycle");
        }
        return project_lifecycle_->pollWrite();
    }

    bool VisualizerImpl::consumeProjectSaveAsStarted() {
        const bool started = project_save_as_started_;
        project_save_as_started_ = false;
        return started;
    }

    void VisualizerImpl::projectWaitWrite() {
        if (!project_lifecycle_) {
            return;
        }
        project_lifecycle_->joinPendingWrite();
    }

    lfs::Result<ProjectMenuInfo>
    VisualizerImpl::projectGetMenuInfo() {
        if (!project_lifecycle_) {
            return visualizerFailure<ProjectMenuInfo>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle is unavailable.",
                "The visualizer did not initialize its project lifecycle service",
                "project.lifecycle");
        }
        return project_lifecycle_->menuInfo();
    }

    lfs::Result<void>
    VisualizerImpl::projectClearRecentFiles() {
        if (!project_lifecycle_) {
            return visualizerFailure<void>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle is unavailable.",
                "The visualizer did not initialize its project lifecycle service",
                "project.lifecycle");
        }
        return project_lifecycle_->clearRecentProjects();
    }

    lfs::Result<void>
    VisualizerImpl::projectRemoveRecentFile(
        const std::filesystem::path& path) {
        if (!project_lifecycle_) {
            return visualizerFailure<void>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle is unavailable.",
                "The visualizer did not initialize its project lifecycle service",
                "project.lifecycle");
        }
        return project_lifecycle_->removeRecentProject(path);
    }

    lfs::Result<bool>
    VisualizerImpl::projectIsDirty() {
        if (!project_lifecycle_) {
            return visualizerFailure<bool>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle is unavailable.",
                "The visualizer did not initialize its project lifecycle service",
                "project.lifecycle");
        }
        return project_lifecycle_->isDirty();
    }

    lfs::Result<bool>
    VisualizerImpl::projectHasPath() {
        if (!project_lifecycle_) {
            return visualizerFailure<bool>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle is unavailable.",
                "The visualizer did not initialize its project lifecycle service",
                "project.lifecycle");
        }
        return project_lifecycle_->hasSourcePath();
    }

    lfs::Result<void>
    VisualizerImpl::projectCompact() {
        if (!project_lifecycle_) {
            return visualizerFailure<void>(
                lfs::ErrorCode::Unavailable,
                "Project lifecycle is unavailable.",
                "The visualizer did not initialize its project lifecycle service",
                "project.lifecycle");
        }
        return project_lifecycle_->compact();
    }

    void VisualizerImpl::performReset() {
        assert(scene_manager_ && scene_manager_->hasDataset());

        const auto& path = scene_manager_->getDatasetPath();
        if (path.empty()) {
            LOG_ERROR("Cannot reset: empty path");
            return;
        }

        const auto preserved_camera = viewport_.camera;
        const auto preserved_transforms = collectResetTransforms(scene_manager_->getScene());

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

        const auto restore_camera = [this, &preserved_camera]() {
            viewport_.camera = preserved_camera;
            if (selection_tool_ && selection_tool_->isEnabled()) {
                selection_tool_->syncDepthFilterToCamera(viewport_);
            }
            if (rendering_manager_) {
                rendering_manager_->markCameraPoseChanged();
            }
            ui::CameraMove{
                .rotation = viewport_.getRotationMatrix(),
                .translation = viewport_.getTranslation()}
                .emit();
            wakeMainLoop();
        };

        if (const auto result = data_loader_->loadDataset(path); !result) {
            LOG_ERROR("Reset reload failed: {}", result.error());
            restore_camera();
            return;
        }

        if (!preserved_transforms.empty()) {
            auto& scene = scene_manager_->getScene();
            for (const auto& [name, transform] : preserved_transforms) {
                if (scene.getNode(name)) {
                    scene.setNodeTransform(name, transform);
                }
            }
        }

        restore_camera();
    }

    void VisualizerImpl::handleLoadConfigFile(const std::filesystem::path& path) {
        auto result = lfs::core::param::read_optim_params_from_json(path);
        if (!result) {
            state::ConfigLoadFailed{.path = path, .error = result.error()}.emit();
            return;
        }
        result->apply_step_scaling();
        parameter_manager_->importParams(*result);
        parameter_manager_->markDirty();

        // Bump scene generation so all panels (e.g. training panel) pick up
        // the new parameter values.  Without this, importing a config after a
        // dataset is already loaded leaves the UI showing stale defaults.
        python::bump_scene_generation();
    }

    void VisualizerImpl::handleTrainingCompleted([[maybe_unused]] const state::TrainingCompleted& event) {
        if (scene_manager_) {
            auto& scene = scene_manager_->getScene();
            if (const auto* model_node = scene.getNodeByUuid(scene.getTrainingModelNodeUuid());
                model_node && !model_node->visible) {
                scene.setNodeVisibility(model_node->id, true);
            }
        }

        pending_training_completion_refresh_frames_ = 3;
        if (rendering_manager_) {
            rendering_manager_->markDirty(DirtyFlag::ALL);
        }
        wakeMainLoop();
        schedulePendingTrainingAction();
    }

    void VisualizerImpl::schedulePendingTrainingAction() {
        if (pending_training_action_ == PendingTrainingAction::None || pending_training_action_posted_) {
            return;
        }
        pending_training_action_posted_ = postWork({
            .run = [this] { performPendingTrainingAction(); },
            .cancel = [this] { pending_training_action_posted_ = false; },
        });
    }

    void VisualizerImpl::performPendingTrainingAction() {
        pending_training_action_posted_ = false;
        const auto action = std::exchange(pending_training_action_, PendingTrainingAction::None);
        switch (action) {
        case PendingTrainingAction::Reset:
            performReset();
            break;
        case PendingTrainingAction::NewProject:
            performNewProject(
                pending_new_project_disposition_);
            pending_new_project_disposition_ =
                ProjectSwitchDisposition::
                    RequireClean;
            break;
        case PendingTrainingAction::OpenProject: {
            const auto path =
                std::exchange(
                    pending_open_path_, {});
            const auto disposition =
                std::exchange(
                    pending_open_disposition_,
                    ProjectSwitchDisposition::
                        RequireClean);
            if (!path.empty()) {
                performOpenProject(
                    path, disposition);
            }
            break;
        }
        case PendingTrainingAction::CloseSave: {
            if (pending_close_save_path_) {
                const auto path =
                    std::exchange(
                        pending_close_save_path_,
                        std::nullopt);
                completeSaveAsAndExit(*path);
                break;
            }
            auto has_path = projectHasPath();
            if (has_path && *has_path) {
                lfs::core::events::cmd::SaveAndExit{}
                    .emit();
            } else {
                lfs::core::events::cmd::SaveAsAndExit{}
                    .emit();
            }
            break;
        }
        case PendingTrainingAction::CloseDiscard:
            if (gui_manager_) {
                gui_manager_->setForceExit(true);
            }
            if (project_lifecycle_) {
                project_lifecycle_
                    ->markApplicationClosePending();
                project_lifecycle_
                    ->setSuppressTrainingAdoption(true);
            }
            requestApplicationClose();
            break;
        case PendingTrainingAction::None:
            break;
        }
    }

    void VisualizerImpl::requestApplicationClose() {
        if (window_manager_) {
            window_manager_->requestClose();
        }
        wakeMainLoop();
    }

} // namespace lfs::vis
