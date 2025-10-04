/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer_impl.hpp"
#include "core/data_loading_service.hpp"
#include "core/logger.hpp"
#include "scene/scene_manager.hpp"
#include "tools/translation_gizmo_tool.hpp"
#include <stdexcept>
#ifdef WIN32
#include <windows.h>
#endif

namespace gs::visualizer {

    VisualizerImpl::VisualizerImpl(const ViewerOptions& options)
        : options_(options),
          viewport_(options.width, options.height),
          window_manager_(std::make_unique<WindowManager>(options.title, options.width, options.height)) {

        LOG_DEBUG("Creating visualizer with window size {}x{}", options.width, options.height);

        // Create scene manager - it creates its own Scene internally
        scene_manager_ = std::make_unique<SceneManager>();

        // Create trainer manager
        trainer_manager_ = std::make_shared<TrainerManager>();
        trainer_manager_->setViewer(this);
        scene_manager_->setTrainerManager(trainer_manager_.get());

        // Create support components
        gui_manager_ = std::make_unique<gui::GuiManager>(this);

        // Create rendering manager with initial antialiasing setting
        rendering_manager_ = std::make_unique<RenderingManager>();

        // Connect scene manager to rendering manager
        scene_manager_->setRenderingManager(rendering_manager_.get());

        // Set initial antialiasing
        RenderSettings initial_settings;
        initial_settings.antialiasing = options.antialiasing;
        initial_settings.gut = options.gut;
        rendering_manager_->updateSettings(initial_settings);

        // Create data loading service
        data_loader_ = std::make_unique<DataLoadingService>(scene_manager_.get());

        // Create main loop
        main_loop_ = std::make_unique<MainLoop>();

        // Setup connections
        setupEventHandlers();
        setupComponentConnections();
    }

    VisualizerImpl::~VisualizerImpl() {
        trainer_manager_.reset();
        translation_gizmo_tool_.reset();
        tool_context_.reset();
        if (gui_manager_) {
            gui_manager_->shutdown();
        }
        LOG_INFO("Visualizer destroyed");
    }

    void VisualizerImpl::initializeTools() {
        if (tools_initialized_) {
            LOG_TRACE("Tools already initialized, skipping");
            return;
        }

        // Create the tool context
        tool_context_ = std::make_unique<ToolContext>(
            rendering_manager_.get(),
            scene_manager_.get(),
            &viewport_,
            window_manager_->getWindow());

        // Create translation gizmo tool
        translation_gizmo_tool_ = std::make_shared<tools::TranslationGizmoTool>();

        // Initialize the tool with the context
        if (!translation_gizmo_tool_->initialize(*tool_context_)) {
            LOG_ERROR("Failed to initialize translation gizmo tool");
            translation_gizmo_tool_.reset();
        } else {
            // Connect tool to input controller
            if (input_controller_) {
                input_controller_->setTranslationGizmoTool(translation_gizmo_tool_);
                input_controller_->setToolContext(tool_context_.get());
            }
            LOG_DEBUG("Translation gizmo tool initialized successfully");
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
            events::cmd::LoadFile{.path = path, .is_dataset = is_dataset}.emit();
        });
    }

    void VisualizerImpl::setupEventHandlers() {
        using namespace events;

        // Training commands
        cmd::StartTraining::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->startTraining();
            }
        });

        cmd::PauseTraining::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->pauseTraining();
            }
        });

        cmd::ResumeTraining::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->resumeTraining();
            }
        });

        cmd::StopTraining::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->stopTraining();
            }
        });

        cmd::SaveCheckpoint::when([this](const auto&) {
            if (trainer_manager_) {
                trainer_manager_->requestSaveCheckpoint();
            }
        });

        // Render settings changes
        ui::RenderSettingsChanged::when([this]([[maybe_unused]] const auto& event) {
            if (rendering_manager_) {
                // The rendering manager handles this internally now
                // Just need to mark dirty which happens in its event handler
            }
        });

        // Camera moves - mark dirty
        ui::CameraMove::when([this](const auto&) {
            if (rendering_manager_) {
                rendering_manager_->markDirty();
            }
        });

        // Scene changes - mark dirty
        state::SceneChanged::when([this](const auto&) {
            if (window_manager_) {
                window_manager_->requestRedraw();
            }
            if (rendering_manager_) {
                rendering_manager_->markDirty();
            }
        });

        internal::TrainerReady::when([this](const auto&) {
            internal::TrainingReadyToStart{}.emit();
        });

        // Training progress - don't mark dirty, let throttling handle it
        state::TrainingProgress::when([this]([[maybe_unused]] const auto& event) {
            // Just update loss buffer, don't force render
            // The 1 FPS throttle will handle rendering
        });

        // Listen for file load commands
        cmd::LoadProject::when([this](const auto& cmd) {
            handleLoadProjectCommand(cmd);
        });

        // Listen to TrainingCompleted
        events::state::TrainingCompleted::when([this](const auto& event) {
            handleTrainingCompleted(event);
        });

        // Listen to load dataset (we need to update project)
        cmd::LoadFile::when([this](const auto& cmd) {
            handleLoadFileCommand(cmd);
        });

        // Listen to save project
        cmd::SaveProject::when([this](const auto& cmd) {
            handleSaveProject(cmd);
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

            // CRITICAL: Poll events once to get actual window dimensions from the OS
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
        if (!input_controller_) {
            input_controller_ = std::make_unique<InputController>(
                window_manager_->getWindow(), viewport_);
            input_controller_->initialize();
            input_controller_->setTrainingManager(trainer_manager_);
            input_controller_->setRenderingManager(rendering_manager_.get());
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

        // Update the main viewport with window size
        viewport_.windowSize = window_manager_->getWindowSize();
        viewport_.frameBufferSize = window_manager_->getFramebufferSize();

        // Update gizmo tool if active
        if (translation_gizmo_tool_ && translation_gizmo_tool_->isEnabled() && tool_context_) {
            translation_gizmo_tool_->update(*tool_context_);
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

        // Create render context
        RenderingManager::RenderContext context{
            .viewport = viewport_,
            .settings = rendering_manager_->getSettings(),
            .viewport_region = has_viewport_region ? &viewport_region : nullptr,
            .has_focus = gui_manager_ && gui_manager_->isViewportFocused(),
            .scene_manager = scene_manager_.get()};

        rendering_manager_->renderFrame(context, scene_manager_.get());

        gui_manager_->render();

        window_manager_->swapBuffers();
        window_manager_->pollEvents();
    }

    bool VisualizerImpl::allowclose() {
        // If we are trying to close and the project is temporary, show dialog
        if (window_manager_->shouldClose() && !gui_manager_->isForceExit()) {
            if (project_) {
                if (project_->getIsTempProject() && !project_->getIsProjectEmpty()) {
                    gui_manager_->showWindow("project_changed_dialog_box", true);
                    window_manager_->cancelClose();
                }
            }
        }
        // If we are trying to close and the project is temporary and we are forcing exit, unlock project
        if (window_manager_->shouldClose() && gui_manager_->isForceExit()) {
            if (project_) {
                if (project_->getIsTempProject()) {
                    project_->unlockProject();
                }
            }
        }

#ifdef WIN32
        // show console in case it was hidden to prevent cmd window to stay hidden/in memory after closing the application
        if (window_manager_->shouldClose()) {
            HWND hwnd = GetConsoleWindow();
            Sleep(1);
            HWND owner = GetWindow(hwnd, GW_OWNER);
            DWORD dwProcessId;
            GetWindowThreadProcessId(hwnd, &dwProcessId);

            // show console if we started from console
            if (GetCurrentProcessId() != dwProcessId) {
                if (owner == NULL) {
                    ShowWindow(hwnd, SW_SHOW); // Windows 10
                } else {
                    ShowWindow(owner, SW_SHOW); // Windows 11
                }
            }
        }
#endif

        return window_manager_->shouldClose();
    }

    void VisualizerImpl::shutdown() {
        // Shutdown tools
        if (translation_gizmo_tool_) {
            translation_gizmo_tool_->shutdown();
            translation_gizmo_tool_.reset();
        }

        // Clean up tool context
        tool_context_.reset();

        tools_initialized_ = false;
    }

    bool VisualizerImpl::LoadProject() {
        if (project_) {
            try {
                LOG_TIMER("LoadProject");
                // write to project file on every change - maybe configurable in the future?
                project_->setUpdateFileOnChange(true);
                // slicing intended
                auto dataset = static_cast<const param::DatasetConfig&>(project_->getProjectData().data_set_info);
                if (!dataset.data_path.empty()) {
                    LOG_DEBUG("Loading dataset from project: {}", dataset.data_path.string());
                    auto result = data_loader_->loadDataset(dataset.data_path);
                    if (!result) {
                        LOG_ERROR("Failed to load dataset from project: {}", result.error());
                        throw std::runtime_error(std::format("Failed to load dataset from project: {}", result.error()));
                    }
                }
                // update the project of all the different managers
                updateProjectOnModules();
                // load plys
                LoadProjectPlys();

                auto plys = project_->getPlys();
                LOG_INFO("Project loaded successfully with {} PLY files", plys.size());
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to load project: {}", e.what());
                throw std::runtime_error(std::format("Failed to load project: {}", e.what()));
            }

            return true;
        }
        return false;
    }

    void VisualizerImpl::LoadProjectPlys() {
        if (!project_) {
            LOG_ERROR("LoadProjectPlys: project is not initialized");
            return;
        }

        auto plys = project_->getPlys();
        LOG_DEBUG("Loading {} PLY files from project", plys.size());

        // sort according to iter numbers
        std::sort(plys.begin(), plys.end(),
                  [](const gs::management::PlyData& a, const gs::management::PlyData& b) {
                      return a.ply_training_iter_number < b.ply_training_iter_number;
                  });

        if (!plys.empty()) {
            scene_manager_->changeContentType(SceneManager::ContentType::SplatFiles);
        }

        // set all of the nodes to invisible except the last one
        for (auto it = plys.begin(); it != plys.end(); ++it) {
            std::string ply_name = it->ply_name;
            if (!std::filesystem::exists(it->ply_path)) {
                LOG_ERROR("ply path not exists {}. skip loading", it->ply_path.string());
                continue;
            }
            bool is_last = (std::next(it) == plys.end());

            LOG_TRACE("Adding PLY '{}' to scene (visible: {})", ply_name, is_last);
            try {
                scene_manager_->addSplatFile(it->ply_path, ply_name, is_last);
                scene_manager_->setPLYVisibility(ply_name, is_last);
            } catch (const std::exception& e) {
                LOG_ERROR("failed loading ply path {}. reason {} ", it->ply_path.string(), e.what());
            }
        }
    }

    void VisualizerImpl::run() {
        // The main loop will call initialize() as its init callback
        // Don't duplicate initialization here
        LoadProject();
        main_loop_->run();
    }

    void VisualizerImpl::setParameters(const param::TrainingParameters& params) {
        data_loader_->setParameters(params);
    }

    std::expected<void, std::string> VisualizerImpl::loadPLY(const std::filesystem::path& path) {
        LOG_TIMER("LoadPLY");

        // Ensure full initialization before loading PLY
        // This will only initialize once due to the guard in initialize()
        if (!initialize()) {
            return std::unexpected("Failed to initialize visualizer");
        }

        LOG_INFO("Loading PLY file: {}", path.string());
        return data_loader_->loadPLY(path);
    }

    std::expected<void, std::string> VisualizerImpl::loadDataset(const std::filesystem::path& path) {
        LOG_TIMER("LoadDataset");

        // Ensure full initialization before loading dataset
        // This will only initialize once due to the guard in initialize()
        if (!initialize()) {
            return std::unexpected("Failed to initialize visualizer");
        }

        LOG_INFO("Loading dataset: {}", path.string());
        auto result = data_loader_->loadDataset(path);
        if (result && project_) {
            auto data_config = project_->getProjectData().data_set_info;
            if (data_config.data_path.empty() || data_config.data_path == path) { // empty project or same data
                data_config.data_path = path;
                project_->setDataInfo(data_config);
            } else {
                project_ = gs::management::CreateTempNewProject(data_config, project_->getOptimizationParams());
                updateProjectOnModules();
            }
        }

        return result;
    }

    void VisualizerImpl::clearScene() {
        data_loader_->clearScene();
    }

    bool VisualizerImpl::openProject(const std::filesystem::path& path) {
        LOG_TIMER("OpenProject");

        auto project = std::make_shared<gs::management::Project>();

        if (!project) {
            LOG_ERROR("Failed to create project object");
            throw std::runtime_error("Failed to create project object");
        }

        if (!project->readFromFile(path)) {
            LOG_ERROR("Failed to read project file: {}", path.string());
            throw std::runtime_error(std::format("Failed to read project file: {}", path.string()));
        }

        if (!project->validateProjectData()) {
            LOG_ERROR("Failed to validate project data from: {}", path.string());
            throw std::runtime_error(std::format("Failed to validate project data from: {}", path.string()));
        }

        project_ = project;
        LOG_INFO("Project opened successfully: {}", path.string());

        return true;
    }

    bool VisualizerImpl::closeProject(const std::filesystem::path& path) {
        if (!project_) {
            LOG_WARN("No project to close");
            return false;
        }

        if (!path.empty()) {
            project_->setProjectFileName(path);
        }

        bool success = project_->writeToFile();
        if (success) {
            LOG_INFO("Project saved successfully");
        } else {
            LOG_ERROR("Failed to save project");
        }

        return success;
    }

    void VisualizerImpl::attachProject(std::shared_ptr<gs::management::Project> _project) {
        project_ = _project;
        updateProjectOnModules();
    }

    std::shared_ptr<gs::management::Project> VisualizerImpl::getProject() {
        return project_;
    }

    void VisualizerImpl::handleLoadProjectCommand(const events::cmd::LoadProject& cmd) {
        try {
            bool success = openProject(cmd.path);
            if (!success) {
                throw std::runtime_error(std::format("Failed opening project: {}", cmd.path.string()));
            }

            success = LoadProject();
            if (!success) {
                throw std::runtime_error(std::format("Failed to load project content: {}", cmd.path.string()));
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Error handling LoadProject command: {}", e.what());

            // Re-throw to let higher level handle it
            throw;
        }
    }

    void VisualizerImpl::handleLoadFileCommand(const events::cmd::LoadFile& cmd) {
        if (cmd.is_dataset && project_) {
            auto data_config = project_->getProjectData().data_set_info;
            data_config.data_path = cmd.path;

            if (project_->getIsTempProject()) {
                data_config.output_path.clear();
                project_ = gs::management::CreateTempNewProject(data_config, project_->getOptimizationParams());
            } else { // else: project already exits (with output dir) - only need to replace data path
                project_->setDataInfo(data_config);
            }

            updateProjectOnModules();
        }
    }

    void VisualizerImpl::handleSaveProject(const events::cmd::SaveProject& cmd) {
        if (project_) {
            const auto& dst_dir = cmd.project_dir;
            if (!std::filesystem::exists(dst_dir)) {
                bool success = std::filesystem::create_directories(dst_dir);
                if (!success) {
                    LOG_ERROR("Directory creation failed {}", dst_dir.string());
                    return;
                }
                LOG_INFO("created directory successfully {}", dst_dir.string());
            }
            if (project_->getIsTempProject()) {
                if (!project_->portProjectToDir(dst_dir)) {
                    LOG_ERROR("porting project failed. Dst dir {} ", project_->getProjectOutputFolder().string());
                }
                project_->setIsTempProject(false);

                for (const auto& ply : project_->getPlys()) {
                    scene_manager_->updatePlyPath(ply.ply_name, ply.ply_path);
                }

            } else {
                if (!project_->writeToFile()) {
                    LOG_ERROR("save project failed {} ", project_->getProjectFileName().string());
                }
            }
            LOG_INFO("Project was saved successfully to {}", project_->getProjectFileName().string());
        }
    }

    void VisualizerImpl::handleTrainingCompleted([[maybe_unused]] const events::state::TrainingCompleted& event) {

        if (!scene_manager_) {
            LOG_ERROR("scene manager is not initialized");
            return;
        }
        if (!project_) {
            LOG_ERROR("project is not initialized");
            return;
        }
        // load plys
        LoadProjectPlys();

        if (scene_manager_) {
            scene_manager_->changeContentType(SceneManager::ContentType::Dataset);
        }
    }

    void VisualizerImpl::updateProjectOnModules() {
        if (trainer_manager_) {
            trainer_manager_->setProject(project_);
        }
        if (scene_manager_) {
            scene_manager_->setProject(project_);
        }
    }
} // namespace gs::visualizer