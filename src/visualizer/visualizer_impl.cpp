/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer_impl.hpp"
#include "command/commands/crop_command.hpp"
#include "command/commands/selection_command.hpp"
#include "core/data_loading_service.hpp"
#include "core/event_bus.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/services.hpp"
#include "scene/scene_manager.hpp"
#include "tools/align_tool.hpp"
#include "tools/brush_tool.hpp"
#include "tools/selection_tool.hpp"
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
        services().set(&command_history_);
        services().set(gui_manager_.get());
        services().set(parameter_manager_.get());

        // Setup connections
        setupEventHandlers();
        setupComponentConnections();
    }

    VisualizerImpl::~VisualizerImpl() {
        // Clear event handlers before destroying components to prevent use-after-free
        lfs::core::event::bus().clear_all();
        services().clear();

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
            window_manager_->getWindow(),
            &command_history_);

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
            // Connect input bindings to selection tool for customizable scroll actions
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
                trainer_manager_->stopTraining();
                trainer_manager_->waitForCompletion();
            }
            const auto& path = scene_manager_->getDatasetPath();
            if (path.empty()) {
                LOG_ERROR("Cannot reset: empty path");
                return;
            }
            // Preserve user-modified params
            if (auto* const param_mgr = services().paramsOrNull(); param_mgr && param_mgr->ensureLoaded()) {
                auto params = param_mgr->createForDataset(path, {});
                if (trainer_manager_) {
                    params.dataset = trainer_manager_->getEditableDatasetParams();
                    params.dataset.data_path = path;
                }
                data_loader_->setParameters(params);
            }
            LOG_DEBUG("Resetting: reloading {}", lfs::core::path_to_utf8(path));
            if (const auto result = data_loader_->loadDataset(path); !result) {
                LOG_ERROR("Reload failed: {}", result.error());
            }
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

        cmd::SwitchToLatestCheckpoint::when([this](const auto&) {
            handleSwitchToLatestCheckpoint();
        });
    }

    bool VisualizerImpl::initialize() {
        // Track if we're fully initialized
        static bool fully_initialized = false;
        if (fully_initialized) {
            LOG_TRACE("Already fully initialized");
            return true;
        }

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

        fully_initialized = true;
        return true;
    }

    void VisualizerImpl::update() {
        window_manager_->updateWindowSize();

        if (gui_manager_) {
            const auto& size = gui_manager_->getViewportSize();
            viewport_.windowSize = {static_cast<int>(size.x), static_cast<int>(size.y)};
        } else {
            viewport_.windowSize = window_manager_->getWindowSize();
        }
        viewport_.frameBufferSize = window_manager_->getFramebufferSize();

        if (brush_tool_ && brush_tool_->isEnabled() && tool_context_) {
            brush_tool_->update(*tool_context_);
        }
        if (selection_tool_ && selection_tool_->isEnabled() && tool_context_) {
            selection_tool_->update(*tool_context_);
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

        rendering_manager_->renderFrame(context, scene_manager_.get());

        gui_manager_->render();

        window_manager_->swapBuffers();

        // Render-on-demand: VSync handles frame pacing, waitEvents saves CPU when idle
        const bool is_training = trainer_manager_ && trainer_manager_->isRunning();
        const bool needs_render = rendering_manager_->needsRender();
        const bool continuous_input = input_controller_ && input_controller_->isContinuousInputActive();

        if (needs_render || continuous_input) {
            // Dirty or active input (WASD/orbit/pan): poll for smooth interaction
            window_manager_->pollEvents();
        } else if (is_training) {
            // Training: short wait for UI responsiveness
            constexpr double TRAINING_WAIT_SEC = 0.016; // ~60 Hz
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

        // User confirmed exit
        if (gui_manager_->isForceExit()) {
#ifdef WIN32
            // Restore console visibility on Windows
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

        // Show confirmation or wait for pending dialog
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
        if (selection_tool_) {
            selection_tool_->shutdown();
            selection_tool_.reset();
        }

        // Clean up tool context
        tool_context_.reset();

        command_history_.clear();

        tools_initialized_ = false;
    }

    void VisualizerImpl::undo() {
        command_history_.undo();
        if (rendering_manager_) {
            rendering_manager_->markDirty();
        }
    }

    void VisualizerImpl::redo() {
        command_history_.redo();
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

        // Get all visible nodes and apply deletion
        auto nodes = scene.getVisibleNodes();
        if (nodes.empty())
            return;

        size_t offset = 0;
        bool any_deleted = false;

        for (const auto* node : nodes) {
            if (!node || !node->model)
                continue;

            const size_t node_size = node->model->size();
            if (node_size == 0)
                continue;

            // Extract selection for this node
            auto node_selection = selection->slice(0, offset, offset + node_size);

            // Convert selection (uint8) to bool tensor for soft_delete
            auto bool_mask = node_selection.to(lfs::core::DataType::Bool);

            // Get old state for undo (clone before modifying)
            auto old_deleted = node->model->has_deleted_mask()
                                   ? node->model->deleted().clone()
                                   : lfs::core::Tensor::zeros({node_size}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);

            // Apply soft delete (OR with existing deleted mask)
            node->model->soft_delete(bool_mask);

            // Get new state for redo
            auto new_deleted = node->model->deleted().clone();

            // Create undo command (uses services() internally)
            auto cmd = std::make_unique<command::CropCommand>(
                node->name, std::move(old_deleted), std::move(new_deleted));
            command_history_.execute(std::move(cmd));

            any_deleted = true;
            offset += node_size;
        }

        if (any_deleted) {
            LOG_INFO("Deleted selected Gaussians");
            // Mark scene as dirty to rebuild combined model with updated deletion masks
            scene.markDirty();
            // Clear selection after deletion
            scene.clearSelection();
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

        const auto old_mask = scene.getSelectionMask();
        const auto ones = lfs::core::Tensor::ones({total}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
        auto new_mask = std::make_shared<lfs::core::Tensor>(
            (old_mask && old_mask->is_valid()) ? ones - *old_mask : ones);

        scene.setSelectionMask(new_mask);
        command_history_.execute(std::make_unique<command::SelectionCommand>(
            scene_manager_.get(),
            old_mask ? std::make_shared<lfs::core::Tensor>(old_mask->clone()) : nullptr,
            new_mask));
        if (rendering_manager_)
            rendering_manager_->markDirty();
    }

    void VisualizerImpl::deselectAll() {
        if (selection_tool_)
            selection_tool_->clearPolygon();

        if (!scene_manager_)
            return;
        auto& scene = scene_manager_->getScene();
        if (!scene.hasSelection())
            return;

        const auto old_mask = scene.getSelectionMask();
        scene.clearSelection();
        command_history_.execute(std::make_unique<command::SelectionCommand>(
            scene_manager_.get(),
            old_mask ? std::make_shared<lfs::core::Tensor>(old_mask->clone()) : nullptr,
            nullptr));
        if (rendering_manager_)
            rendering_manager_->markDirty();
    }

    void VisualizerImpl::selectAll() {
        if (!scene_manager_)
            return;

        const auto tool = editor_context_.getActiveTool();
        const bool is_selection_tool = (tool == ToolType::Selection || tool == ToolType::Brush);

        if (is_selection_tool) {
            // Select all gaussians for the active node
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

            const auto old_mask = scene.getSelectionMask();
            auto new_mask = std::make_shared<lfs::core::Tensor>(transform_indices->eq(node_index));
            scene.setSelectionMask(new_mask);
            command_history_.execute(std::make_unique<command::SelectionCommand>(
                scene_manager_.get(),
                old_mask ? std::make_shared<lfs::core::Tensor>(old_mask->clone()) : nullptr,
                new_mask));
        } else {
            // Select all SPLAT nodes
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

        if (selection_tool_) {
            selection_tool_->clearPolygon();
            selection_tool_->setEnabled(false);
        }
        scene_manager_->getScene().resetSelectionState();

        scene_manager_->clearSelection();
        for (const auto& name : pasted) {
            scene_manager_->addToSelection(name);
        }

        if (rendering_manager_)
            rendering_manager_->markDirty();
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

} // namespace lfs::vis