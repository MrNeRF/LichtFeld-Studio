#include "config.h" // Include generated config
#include "core/ply_loader.hpp"
#include "core/training_setup.hpp"
#include "visualizer/detail.hpp"
#include "visualizer/gui_manager.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <print>
#include <sstream>
#include <thread>

#include <cuda_runtime.h>

#ifdef CUDA_GL_INTEROP_ENABLED
#include "visualizer/cuda_gl_interop.hpp"
#endif

namespace gs {

    ViewerDetail::ViewerDetail(std::string title, int width, int height)
        : title_(title),
          viewport_(width, height),
          window_manager_(std::make_unique<WindowManager>(title, width, height)) {
    }

    ViewerDetail::~ViewerDetail() {
        std::cout << "Viewer destroyed." << std::endl;
    }

    bool ViewerDetail::init() {
        if (!window_manager_->init()) {
            return false;
        }

        // Create input handler
        input_handler_ = std::make_unique<InputHandler>(window_manager_->getWindow());

        // Create camera controller
        camera_controller_ = std::make_unique<CameraController>(viewport_);
        camera_controller_->connectToInputHandler(*input_handler_);

        return true;
    }

    void ViewerDetail::updateWindowSize() {
        window_manager_->updateWindowSize();
        viewport_.windowSize = window_manager_->getWindowSize();
        viewport_.frameBufferSize = window_manager_->getFramebufferSize();
    }

    float ViewerDetail::getGPUUsage() {
        size_t free_byte, total_byte;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_byte, &total_byte);
        size_t used_byte = total_byte - free_byte;
        float gpuUsage = used_byte / (float)total_byte * 100;

        return gpuUsage;
    }

    void ViewerDetail::setFrameRate(const int fps) {
        targetFPS = fps;
        frameTime = 1000 / targetFPS;
    }

    void ViewerDetail::controlFrameRate() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();
        if (duration < frameTime) {
            std::this_thread::sleep_for(std::chrono::milliseconds(frameTime - duration));
        }
        lastTime = std::chrono::high_resolution_clock::now();
    }

    void ViewerDetail::run() {

        if (!init()) {
            std::cerr << "Viewer initialization failed!" << std::endl;
            return;
        }

        std::string shader_path = std::string(PROJECT_ROOT_PATH) + "/include/visualizer/shaders/";
        quadShader_ = std::make_shared<Shader>(
            (shader_path + "/screen_quad.vert").c_str(),
            (shader_path + "/screen_quad.frag").c_str(),
            true);

        // Initialize screen renderer with interop support if available
#ifdef CUDA_GL_INTEROP_ENABLED
        screen_renderer_ = std::make_shared<ScreenQuadRendererInterop>(true);
        std::cout << "CUDA-OpenGL interop enabled for rendering" << std::endl;
#else
        screen_renderer_ = std::make_shared<ScreenQuadRenderer>();
        std::cout << "Using CPU copy for rendering (interop not available)" << std::endl;
#endif

        while (!window_manager_->shouldClose()) {

            // Clear with a dark background
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            controlFrameRate();

            updateWindowSize();

            draw();

            window_manager_->swapBuffers();
            window_manager_->pollEvents();
        }
    }

    bool GSViewer::handleFileDrop(const InputHandler::FileDropEvent& event) {
        // Process each dropped file
        for (const auto& path_str : event.paths) {
            std::filesystem::path filepath(path_str);

            // Check if it's a PLY file
            if (filepath.extension() == ".ply" || filepath.extension() == ".PLY") {
                std::println("Dropped PLY file: {}", filepath.string());

                // Load the PLY file
                loadPLYFile(filepath);

                // Log the action
                if (gui_manager_) {
                    gui_manager_->showScriptingConsole(true);
                    gui_manager_->addConsoleLog("Info: Loaded PLY file via drag-and-drop: %s",
                                                filepath.filename().string().c_str());
                }

                // Only process the first PLY file if multiple files were dropped
                return true;
            }
            if (std::filesystem::is_directory(filepath)) {
                // Check if it's a dataset directory
                bool is_colmap_dataset = false;
                bool is_transforms_dataset = false;

                // Check for COLMAP dataset structure
                if (std::filesystem::exists(filepath / "sparse" / "0" / "cameras.bin") ||
                    std::filesystem::exists(filepath / "sparse" / "cameras.bin")) {
                    is_colmap_dataset = true;
                }

                // Check for transforms dataset
                if (std::filesystem::exists(filepath / "transforms.json") ||
                    std::filesystem::exists(filepath / "transforms_train.json")) {
                    is_transforms_dataset = true;
                }

                if (is_colmap_dataset || is_transforms_dataset) {
                    std::println("Dropped dataset directory: {}", filepath.string());

                    // Load the dataset
                    loadDataset(filepath);

                    // Log the action
                    if (gui_manager_) {
                        gui_manager_->showScriptingConsole(true);
                        gui_manager_->addConsoleLog("Info: Loaded %s dataset via drag-and-drop: %s",
                                                    is_colmap_dataset ? "COLMAP" : "Transforms",
                                                    filepath.filename().string().c_str());
                    }

                    // Only process the first valid dataset if multiple were dropped
                    return true;
                }
            }
        }
        return false;
    }

    GSViewer::GSViewer(std::string title, int width, int height)
        : ViewerDetail(title, width, height),
          trainer_(nullptr) {

        config_ = std::make_shared<RenderingConfig>();
        info_ = std::make_shared<TrainingInfo>();
        notifier_ = std::make_shared<ViewerNotifier>();

        scene_ = std::make_unique<Scene>();

        setFrameRate(30);

        // Create GUI manager
        gui_manager_ = std::make_unique<gui::GuiManager>(this);

        // Set up default script executor with basic functionality
        gui_manager_->setScriptExecutor([this](const std::string& command) -> std::string {
            std::ostringstream result;

            // Basic command parsing
            if (command.empty()) {
                return "";
            }

            // Handle basic commands
            if (command == "help" || command == "h") {
                result << "Available commands:\n";
                result << "  help, h - Show this help\n";
                result << "  clear - Clear console\n";
                result << "  status - Show training status\n";
                result << "  model_info - Show model information\n";
                result << "  tensor_info <name> - Show tensor information\n";
                result << "  gpu_info - Show GPU information\n";
                return result.str();
            }

            if (command == "clear") {
                // Handled internally by the console
                return "";
            }

            if (command == "status") {
                if (!trainer_) {
                    return "No trainer available (viewer mode)";
                }
                result << "Training Status:\n";
                result << "  Running: " << (trainer_->is_running() ? "Yes" : "No") << "\n";
                result << "  Paused: " << (trainer_->is_paused() ? "Yes" : "No") << "\n";
                result << "  Complete: " << (trainer_->is_training_complete() ? "Yes" : "No") << "\n";
                result << "  Current Iteration: " << trainer_->get_current_iteration() << "\n";
                result << "  Current Loss: " << std::fixed << std::setprecision(6) << trainer_->get_current_loss();
                return result.str();
            }

            if (command == "model_info") {
                if (!scene_->hasModel()) {
                    return "No model available";
                }

                result << "Model Information:\n";

                const SplatData* model = scene_->getModel();
                if (model) {
                    result << "  Number of Gaussians: " << model->size() << "\n";
                    result << "  Positions shape: [" << model->get_means().size(0) << ", " << model->get_means().size(1) << "]\n";
                    result << "  Device: " << model->get_means().device() << "\n";
                    result << "  Dtype: " << model->get_means().dtype() << "\n";
                    result << "  Active SH degree: " << model->get_active_sh_degree() << "\n";
                    result << "  Scene scale: " << model->get_scene_scale();

                    if (scene_->getMode() == Scene::Mode::Viewing) {
                        result << "\n  Mode: Viewer (no training)";
                    }
                }

                return result.str();
            }

            if (command == "gpu_info") {
                size_t free_byte, total_byte;
                cudaDeviceSynchronize();
                cudaMemGetInfo(&free_byte, &total_byte);

                double free_gb = free_byte / 1024.0 / 1024.0 / 1024.0;
                double total_gb = total_byte / 1024.0 / 1024.0 / 1024.0;
                double used_gb = total_gb - free_gb;

                result << "GPU Memory Info:\n";
                result << "  Total: " << std::fixed << std::setprecision(2) << total_gb << " GB\n";
                result << "  Used: " << used_gb << " GB\n";
                result << "  Free: " << free_gb << " GB\n";
                result << "  Usage: " << std::setprecision(1) << (used_gb / total_gb * 100.0) << "%";
                return result.str();
            }

            // Handle tensor_info command
            if (command.substr(0, 11) == "tensor_info") {
                if (!scene_->hasModel()) {
                    return "No model available";
                }

                std::string tensor_name = "";
                if (command.length() > 12) {
                    tensor_name = command.substr(12); // Get parameter after "tensor_info "
                }

                if (tensor_name.empty()) {
                    return "Usage: tensor_info <tensor_name>\nAvailable: means, scaling, rotation, shs, opacity";
                }

                std::string tensor_result;
                SplatData* model = scene_->getMutableModel();
                if (!model) {
                    tensor_result = "Model not available";
                    return tensor_result;
                }

                torch::Tensor tensor;
                if (tensor_name == "means" || tensor_name == "positions") {
                    tensor = model->get_means();
                } else if (tensor_name == "scales" || tensor_name == "scaling") {
                    tensor = model->get_scaling();
                } else if (tensor_name == "rotations" || tensor_name == "rotation" || tensor_name == "quats") {
                    tensor = model->get_rotation();
                } else if (tensor_name == "features" || tensor_name == "colors" || tensor_name == "shs") {
                    tensor = model->get_shs();
                } else if (tensor_name == "opacities" || tensor_name == "opacity") {
                    tensor = model->get_opacity();
                } else {
                    tensor_result = "Unknown tensor: " + tensor_name + "\nAvailable: means, scaling, rotation, shs, opacity";
                    return tensor_result;
                }

                std::ostringstream oss;
                oss << "Tensor '" << tensor_name << "' info:\n";
                oss << "  Shape: [";
                for (int i = 0; i < tensor.dim(); i++) {
                    if (i > 0)
                        oss << ", ";
                    oss << tensor.size(i);
                }
                oss << "]\n";
                oss << "  Device: " << tensor.device() << "\n";
                oss << "  Dtype: " << tensor.dtype() << "\n";
                oss << "  Requires grad: " << (tensor.requires_grad() ? "Yes" : "No") << "\n";

                // Show some statistics if tensor is on CPU or we can move it
                try {
                    auto cpu_tensor = tensor.cpu();
                    auto flat = cpu_tensor.flatten();
                    if (flat.numel() > 0) {
                        oss << "  Min: " << torch::min(flat).item<float>() << "\n";
                        oss << "  Max: " << torch::max(flat).item<float>() << "\n";
                        oss << "  Mean: " << torch::mean(flat).item<float>() << "\n";
                        oss << "  Std: " << torch::std(flat).item<float>();
                    }
                } catch (...) {
                    oss << "  (Statistics unavailable)";
                }

                tensor_result = oss.str();
                return tensor_result;
            }

            return "Unknown command: '" + command + "'. Type 'help' for available commands.";
        });

        // Set up file selection callback
        gui_manager_->setFileSelectedCallback([this](const std::filesystem::path& path, bool is_dataset) {
            if (is_dataset) {
                loadDataset(path);
            } else {
                loadPLYFile(path);
            }
        });
    }

    GSViewer::~GSViewer() {
        // Stop training thread if running
        if (training_thread_ && training_thread_->joinable()) {
            std::cout << "Viewer closing - stopping training thread..." << std::endl;
            training_thread_->request_stop();
            training_thread_->join();
        }

        // If trainer is still running, request it to stop
        if (trainer_ && trainer_->is_running()) {
            std::cout << "Viewer closing - stopping training..." << std::endl;
            trainer_->request_stop();

            // Give the training thread a moment to acknowledge the stop request
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Cleanup GUI
        if (gui_manager_) {
            gui_manager_->shutdown();
        }

        std::cout << "GSViewer destroyed." << std::endl;
    }

    void GSViewer::setTrainer(Trainer* trainer) {
        trainer_ = trainer;
        if (scene_) {
            scene_->linkToTrainer(trainer);
        }
    }

    void GSViewer::setStandaloneModel(std::unique_ptr<SplatData> model) {
        if (scene_) {
            scene_->setModel(std::move(model));
        }
    }

    void GSViewer::setAntiAliasing(bool enable) {
        anti_aliasing_ = enable;
    }

    void GSViewer::loadPLYFile(const std::filesystem::path& path) {
        try {
            std::println("Loading PLY file: {}", path.string());

            // Clear any existing data
            clearCurrentData();

            // Load the PLY file
            auto splat_result = gs::load_ply(path);
            if (!splat_result) {
                gui_manager_->addConsoleLog("Error: Failed to load PLY: %s", splat_result.error().c_str());
                return;
            }

            scene_->setModel(std::make_unique<SplatData>(std::move(*splat_result)));
            current_ply_path_ = path;
            current_mode_ = ViewerMode::PLYViewer;

            gui_manager_->addConsoleLog("Info: Loaded PLY with %lld Gaussians from %s",
                                        scene_->getStandaloneModel()->size(),
                                        path.filename().string().c_str());

        } catch (const std::exception& e) {
            gui_manager_->addConsoleLog("Error: Exception loading PLY: %s", e.what());
        }
    }

    void GSViewer::loadDataset(const std::filesystem::path& path) {
        try {
            std::println("Loading dataset from: {}", path.string());

            // Clear any existing data
            clearCurrentData();

            // Use the parameters that were passed to the viewer
            param::TrainingParameters dataset_params = params_;
            dataset_params.dataset.data_path = path; // Override with the selected path

            // Setup training
            auto setup_result = gs::setupTraining(dataset_params);
            if (!setup_result) {
                gui_manager_->addConsoleLog("Error: Failed to setup training: %s", setup_result.error().c_str());
                return;
            }

            // Store the trainer (but don't take ownership yet)
            auto trainer_ptr = setup_result->trainer.get();

            // Link the trainer to this viewer
            trainer_ptr->setViewer(this);

            // Now take ownership
            trainer_ = setup_result->trainer.release();

            // Link scene to trainer
            scene_->linkToTrainer(trainer_);

            current_dataset_path_ = path;
            current_mode_ = ViewerMode::Training;

            // Get dataset info
            size_t num_images = setup_result->dataset->size().value();
            size_t num_gaussians = trainer_->get_strategy().get_model().size();

            gui_manager_->addConsoleLog("Info: Loaded dataset with %zu images and %zu initial Gaussians",
                                        num_images, num_gaussians);
            gui_manager_->addConsoleLog("Info: Ready to start training from %s",
                                        path.filename().string().c_str());
            gui_manager_->addConsoleLog("Info: Using parameters from command line/config");

        } catch (const std::exception& e) {
            gui_manager_->addConsoleLog("Error: Exception loading dataset: %s", e.what());
        }
    }

    void GSViewer::clearCurrentData() {
        // Stop any ongoing training thread
        if (training_thread_ && training_thread_->joinable()) {
            std::println("Stopping training thread...");
            training_thread_->request_stop();
            training_thread_->join();
            training_thread_.reset();
        }

        // Stop any ongoing training via trainer
        if (trainer_ && trainer_->is_running()) {
            trainer_->request_stop();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Clear scene
        scene_->clearModel();

        // Clear data
        trainer_ = nullptr;

        // Reset state
        current_mode_ = ViewerMode::Empty;
        current_ply_path_.clear();
        current_dataset_path_.clear();

        // Clear training info
        if (info_) {
            info_->curr_iterations_ = 0;
            info_->total_iterations_ = 0;
            info_->num_splats_ = 0;
            info_->loss_buffer_.clear();
        }
    }

    void GSViewer::startTraining() {
        if (!trainer_ || training_thread_) {
            return;
        }

        // Set the ready flag for trainer to start
        if (notifier_) {
            notifier_->ready = true;
        }

        // Then start training in a separate thread
        training_thread_ = std::make_unique<std::jthread>(
            [trainer_ptr = trainer_](std::stop_token stop_token) {
                std::println("Training thread started");
                auto train_result = trainer_ptr->train(stop_token);
                if (!train_result) {
                    std::println(stderr, "Training error: {}", train_result.error());
                }
                std::println("Training thread finished");
            });

        std::println("Training thread launched");
    }

    bool GSViewer::isGuiActive() const {
        return gui_manager_ && gui_manager_->isAnyWindowActive();
    }

    GSViewer::ViewerMode GSViewer::getCurrentMode() const {
        switch (scene_->getMode()) {
        case Scene::Mode::Empty:
            return ViewerMode::Empty;
        case Scene::Mode::Viewing:
            return ViewerMode::PLYViewer;
        case Scene::Mode::Training:
            return ViewerMode::Training;
        default:
            return ViewerMode::Empty;
        }
    }

    void GSViewer::drawFrame() {
        // Only render if we have a model to render
        if (!scene_->hasModel()) {
            return;
        }

        // Build render request
        RenderingPipeline::RenderRequest request{
            .view_rotation = viewport_.getRotationMatrix(),
            .view_translation = viewport_.getTranslation(),
            .viewport_size = viewport_.windowSize,
            .fov = config_->fov,
            .scaling_modifier = config_->scaling_modifier,
            .antialiasing = anti_aliasing_,
            .render_mode = RenderMode::RGB};

        RenderingPipeline::RenderResult result;

        if (trainer_ && trainer_->is_running()) {
            std::shared_lock<std::shared_mutex> lock(trainer_->getRenderMutex());
            result = scene_->render(request);
        } else {
            result = scene_->render(request);
        }

        if (result.valid) {
            RenderingPipeline::uploadToScreen(result, *screen_renderer_, viewport_.windowSize);
            screen_renderer_->render(quadShader_, viewport_);
        }
    }

    void GSViewer::draw() {
        // Initialize GUI on first draw
        static bool gui_initialized = false;
        if (!gui_initialized) {
            gui_manager_->init();
            gui_initialized = true;

            // Set up input handlers after GUI is initialized

            // GUI gets first priority
            input_handler_->addMouseButtonHandler(
                [this](const InputHandler::MouseButtonEvent& event) {
                    return isGuiActive(); // Consume if GUI is active
                });

            input_handler_->addMouseMoveHandler(
                [this](const InputHandler::MouseMoveEvent& event) {
                    return isGuiActive(); // Consume if GUI is active
                });

            input_handler_->addMouseScrollHandler(
                [this](const InputHandler::MouseScrollEvent& event) {
                    return isGuiActive(); // Consume if GUI is active
                });

            // File drop handler
            input_handler_->addFileDropHandler(
                [this](const InputHandler::FileDropEvent& event) {
                    return handleFileDrop(event);
                });
        }

        // Draw the 3D frame first
        drawFrame();

        // Then render GUI on top
        gui_manager_->render();
    }

} // namespace gs