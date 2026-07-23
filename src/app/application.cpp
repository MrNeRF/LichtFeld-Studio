/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/application.hpp"
#include "app/headless_run_coordinator.hpp"
#include "control/command_api.hpp"
#include "core/checkpoint_format.hpp"
#include "core/crash_handler.hpp"
#include "core/cuda_version.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/image_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "io/cache_image_loader.hpp"
#include "tcp/include/tcp_publisher.hpp"
#include "tcp/include/tcp_responder.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include "visualizer/training/training_manager.hpp"
#include "visualizer/visualizer.hpp"

#include "app/mcp_gui_tools.hpp"
#include "io/loader.hpp"
#include "io/video/video_encoder.hpp"
#include "mcp/mcp_http_server.hpp"
#include "mcp/mcp_tools.hpp"
#include "python/runner.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "sequencer/timeline.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "visualizer/gui/panels/python_scripts_panel.hpp"
#include "visualizer/gui/video_widget_interface.hpp"
#include "visualizer/gui/windows/video_extractor_dialog.hpp"
#include <cmath>
#include <condition_variable>
#include <cuda_runtime.h>
#include <future>
#include <mutex>
#include <rasterization_api.h>

#ifdef WIN32
#include <windows.h>
#endif

namespace lfs::app {

    namespace {

        bool checkCudaDriverVersion();

        std::expected<core::param::TrainingParameters, std::string> loadCheckpointParams(const core::param::TrainingParameters& params, core::Scene& scene) {
            LOG_INFO("Resuming from checkpoint: {}", core::path_to_utf8(*params.resume_checkpoint));

            auto params_result = core::load_checkpoint_params(*params.resume_checkpoint);
            if (!params_result) {
                return std::unexpected(std::format("Failed to load checkpoint params: {}", params_result.error()));
            }
            auto checkpoint_params = std::move(*params_result);

            if (!params.dataset.data_path.empty())
                checkpoint_params.dataset.data_path = params.dataset.data_path;
            if (!params.dataset.output_path.empty())
                checkpoint_params.dataset.output_path = params.dataset.output_path;
            if (!params.dataset.output_name.empty())
                checkpoint_params.dataset.output_name = params.dataset.output_name;

            if (checkpoint_params.dataset.data_path.empty()) {
                return std::unexpected("Checkpoint has no dataset path and none provided via --data-path");
            }
            if (!std::filesystem::exists(checkpoint_params.dataset.data_path)) {
                return std::unexpected(std::format("Dataset path does not exist: {}", core::path_to_utf8(checkpoint_params.dataset.data_path)));
            }

            if (const auto result = training::validateDatasetPath(checkpoint_params); !result) {
                return std::unexpected(std::format("Dataset validation failed: {}", result.error()));
            }

            if (const auto result = training::loadTrainingDataIntoScene(checkpoint_params, scene); !result) {
                return std::unexpected(std::format("Failed to load training data: {}", result.error()));
            }

            for (const auto* node : scene.getNodes()) {
                if (node->type == core::NodeType::POINTCLOUD) {
                    scene.removeNode(node->name, false);
                    break;
                }
            }

            auto splat_result = core::load_checkpoint_splat_data(*params.resume_checkpoint);
            if (!splat_result) {
                return std::unexpected(std::format("Failed to load checkpoint splat data: {}", splat_result.error()));
            }

            auto splat_data = std::make_unique<core::SplatData>(std::move(*splat_result));
            scene.addSplat("Model", std::move(splat_data), core::NULL_NODE);
            scene.setTrainingModelNode("Model");

            checkpoint_params.resume_checkpoint = *params.resume_checkpoint;
            return checkpoint_params;
        }

        int runHeadlessWithTCP(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
            if (params->dataset.data_path.empty() && !params->resume_checkpoint) {
                LOG_ERROR("Headless with TCP mode requires --data-path or --resume");
                return 1;
            }

            checkCudaDriverVersion();
            lfs::event::CommandCenterBridge::instance().set(&lfs::training::CommandCenter::instance());
            HeadlessRunCoordinator coordinator;

            {
                core::Scene scene;
                std::optional<core::param::TrainingParameters> checkpoint_params{std::nullopt};

                if (params->resume_checkpoint) {
                    const auto ckpt_params_result = loadCheckpointParams(*params, scene);
                    if (!ckpt_params_result) {
                        LOG_ERROR("Failed to load checkpoint: {}", ckpt_params_result.error());
                        return 1;
                    }
                    checkpoint_params = *ckpt_params_result;
                } else {
                    LOG_INFO("Starting headless with TCP training...");

                    if (const auto result = training::loadTrainingDataIntoScene(*params, scene); !result) {
                        LOG_ERROR("Failed to load training data: {}", result.error());
                        return 1;
                    }

                    if (const auto result = training::initializeTrainingModel(*params, scene); !result) {
                        LOG_ERROR("Failed to initialize model: {}", result.error());
                        return 1;
                    }
                }

                auto manager = std::make_shared<vis::TrainerManager>();
                {
                    auto trainer = std::make_unique<training::Trainer>(scene);

                    if (!params->python_scripts.empty()) {
                        trainer->set_python_scripts(params->python_scripts);
                        vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
                    }

                    trainer->setParams(checkpoint_params ? *checkpoint_params : *params); // Load checkpoint into trainer is called internally
                    manager->setTrainer(std::move(trainer));
                }

                core::Tensor::trim_memory_pool();

                try {
                    tcp::ResponderServer responder(params->server.tcp_server_connection_port, manager);
                    tcp::PublisherServer publisher(params->server.tcp_broadcast_connection_port, manager);

                    responder.start();
                    publisher.start();
                    LOG_INFO("Responder server listening on {}", responder.getEndpoint());
                    LOG_INFO("Publisher server listening on {}", publisher.getEndpoint());

                    std::mutex completion_mutex;
                    std::condition_variable completion_cv;
                    std::optional<core::events::state::TrainingCompleted> completion;
                    lfs::event::ScopedHandler completion_subscription;
                    completion_subscription.subscribe<core::events::state::TrainingCompleted>(
                        [&](const core::events::state::TrainingCompleted& event) {
                            {
                                std::lock_guard lock(completion_mutex);
                                if (completion) {
                                    return;
                                }
                                completion = event;
                            }
                            completion_cv.notify_all();
                        });

                    if (!manager->startTraining()) {
                        throw std::runtime_error("Failed to start TCP headless training");
                    }

                    bool stop_requested = false;
                    std::optional<std::chrono::steady_clock::time_point> stop_deadline;
                    std::string completion_error;
                    std::unique_lock completion_lock(completion_mutex);
                    while (!completion) {
                        completion_cv.wait_for(completion_lock, std::chrono::milliseconds(100));
                        if (completion) {
                            break;
                        }

                        if (coordinator.interrupted() && !stop_requested) {
                            stop_requested = true;
                            stop_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                            completion_lock.unlock();
                            LOG_INFO("Interrupt signal received, requesting TCP training stop");
                            manager->stopTraining();
                            completion_lock.lock();
                        }

                        if (manager->isFinished()) {
                            completion_error = "Training reached a terminal state without a completion event";
                            break;
                        }
                        if (stop_deadline && std::chrono::steady_clock::now() >= *stop_deadline) {
                            completion_error = "Timed out waiting for interrupted TCP training to save and stop";
                            break;
                        }
                    }
                    completion_lock.unlock();

                    if (!manager->waitForCompletion()) {
                        throw std::runtime_error("Training worker did not finish after terminal event");
                    }

                    if (!completion_error.empty()) {
                        throw std::runtime_error(completion_error);
                    }
                    if (!completion) {
                        throw std::runtime_error("TCP training completion was not observed");
                    }

                    publisher.stop();
                    responder.stop();
                    responder.join();
                } catch (const std::exception& e) {
                    LOG_ERROR("Headless TCP lifecycle failed: {}", e.what());
                    return 1;
                }

                if (manager->getStateMachine().getFinishReason() == vis::FinishReason::Error) {
                    LOG_ERROR("Training error: {}", manager->getLastError());
                    if (!params->python_scripts.empty()) {
                        core::teardown_gpu_before_exit();
                        python::finalize();
                        core::flush_and_exit(1);
                    }
                    return 1;
                }

                LOG_INFO("Headless with TCP training completed");
            }

            core::teardown_gpu_before_exit();

            const int exit_code = coordinator.interrupted() ? coordinator.interrupted_exit_code() : 0;
            if (!params->python_scripts.empty()) {
                python::finalize();
                core::flush_and_exit(exit_code);
            }
            return exit_code;
        }

        int runHeadless(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
            if (params->dataset.data_path.empty() && !params->resume_checkpoint) {
                LOG_ERROR("Headless mode requires --data-path or --resume");
                return 1;
            }

            checkCudaDriverVersion();
            lfs::event::CommandCenterBridge::instance().set(&lfs::training::CommandCenter::instance());
            HeadlessRunCoordinator coordinator;

            {
                core::Scene scene;

                if (params->resume_checkpoint) {
                    const auto ckpt_params_result = loadCheckpointParams(*params, scene);
                    if (!ckpt_params_result) {
                        LOG_ERROR("Failed to load checkpoint: {}", ckpt_params_result.error());
                        return 1;
                    }

                    auto trainer = std::make_unique<training::Trainer>(scene);

                    if (!params->python_scripts.empty()) {
                        trainer->set_python_scripts(params->python_scripts);
                        vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
                    }

                    if (const auto result = trainer->initialize(*ckpt_params_result); !result) {
                        LOG_ERROR("Failed to initialize trainer: {}", result.error());
                        return 1;
                    }

                    const auto ckpt_result = trainer->load_checkpoint(*params->resume_checkpoint);
                    if (!ckpt_result) {
                        LOG_ERROR("Failed to restore checkpoint state: {}", ckpt_result.error());
                        return 1;
                    }
                    LOG_INFO("Resumed from iteration {}", *ckpt_result);

                    core::Tensor::trim_memory_pool();

                    if (const auto result = trainer->train(coordinator.stop_token()); !result) {
                        LOG_ERROR("Training error: {}", lfs::format_for_developer(result.error()));
                        if (!params->python_scripts.empty()) {
                            core::teardown_gpu_before_exit();
                            python::finalize();
                            core::flush_and_exit(1);
                        }
                        return 1;
                    }
                    trainer->shutdown();
                    static_cast<void>(trainer.release());
                } else {
                    LOG_INFO("Starting headless training...");

                    if (const auto result = training::loadTrainingDataIntoScene(*params, scene); !result) {
                        LOG_ERROR("Failed to load training data: {}", result.error());
                        return 1;
                    }

                    if (const auto result = training::initializeTrainingModel(*params, scene); !result) {
                        LOG_ERROR("Failed to initialize model: {}", result.error());
                        return 1;
                    }

                    auto trainer = std::make_unique<training::Trainer>(scene);

                    if (!params->python_scripts.empty()) {
                        trainer->set_python_scripts(params->python_scripts);
                        vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
                    }

                    if (const auto result = trainer->initialize(*params); !result) {
                        LOG_ERROR("Failed to initialize trainer: {}", result.error());
                        return 1;
                    }

                    core::Tensor::trim_memory_pool();

                    if (const auto result = trainer->train(coordinator.stop_token()); !result) {
                        LOG_ERROR("Training error: {}", lfs::format_for_developer(result.error()));
                        if (!params->python_scripts.empty()) {
                            core::teardown_gpu_before_exit();
                            python::finalize();
                            core::flush_and_exit(1);
                        }
                        return 1;
                    }
                    trainer->shutdown();
                    static_cast<void>(trainer.release());
                }

                LOG_INFO("Headless training completed");
                core::teardown_gpu_before_exit();
                core::flush_and_exit(0);
            }

            core::teardown_gpu_before_exit();

            const int exit_code = coordinator.interrupted() ? coordinator.interrupted_exit_code() : 0;
            if (!params->python_scripts.empty()) {
                python::finalize();
                core::flush_and_exit(exit_code);
            }
            return exit_code;
        }

        // Renders a sequencer camera path against a trained scene to a video file, headless.
        int runHeadlessRender(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
            const auto& cfg = *params->render_path;

            checkCudaDriverVersion();

            // Load the trained scene.
            std::shared_ptr<core::SplatData> model;
            const auto load_ext = cfg.load_path.extension().string();
            if (load_ext == ".resume") {
                auto splat_result = core::load_checkpoint_splat_data(cfg.load_path);
                if (!splat_result) {
                    LOG_ERROR("Failed to load checkpoint: {}", splat_result.error());
                    return 1;
                }
                model = std::make_shared<core::SplatData>(std::move(*splat_result));
            } else {
                auto loader = lfs::io::Loader::create();
                auto load_result = loader->load(cfg.load_path);
                if (!load_result) {
                    LOG_ERROR("Failed to load scene: {}", load_result.error().message);
                    return 1;
                }
                if (auto* const splat = std::get_if<std::shared_ptr<core::SplatData>>(&load_result->data)) {
                    model = *splat;
                } else {
                    LOG_ERROR("--render-load is not a Gaussian splat scene: {}", core::path_to_utf8(cfg.load_path));
                    return 1;
                }
            }
            if (!model || model->size() == 0) {
                LOG_ERROR("Loaded scene has no Gaussians: {}", core::path_to_utf8(cfg.load_path));
                return 1;
            }

            lfs::sequencer::Timeline timeline;
            if (!timeline.loadFromJson(core::path_to_utf8(cfg.camera_path)) || timeline.empty()) {
                LOG_ERROR("Failed to load camera path (or it has no keyframes): {}", core::path_to_utf8(cfg.camera_path));
                return 1;
            }

            // Solid black background, matching Trainer's default bg_color init.
            auto background = core::Tensor::empty({3}, core::Device::CPU, core::DataType::Float32);
            {
                auto* const bg_ptr = background.ptr<float>();
                bg_ptr[0] = bg_ptr[1] = bg_ptr[2] = 0.0f;
            }
            background = background.to(core::Device::CUDA);

            lfs::io::video::VideoEncoder encoder;
            lfs::io::video::VideoExportOptions options;
            options.preset = lfs::io::video::VideoPreset::CUSTOM;
            options.width = cfg.width;
            options.height = cfg.height;
            options.framerate = cfg.fps;
            options.crf = cfg.crf;
            if (const auto open_result = encoder.open(cfg.output_path, options); !open_result) {
                LOG_ERROR("Failed to open video encoder: {}", open_result.error());
                return 1;
            }

            const float duration = timeline.duration();
            const int total_frames = static_cast<int>(std::ceil(duration * cfg.fps)) + 1;
            LOG_INFO("Rendering {} frame(s) ({:.2f}s @ {}fps) from {} to {}",
                     total_frames, duration, cfg.fps,
                     core::path_to_utf8(cfg.camera_path), core::path_to_utf8(cfg.output_path));

            for (int frame = 0; frame < total_frames; ++frame) {
                const float t = std::min(static_cast<float>(frame) / static_cast<float>(cfg.fps), duration);
                const auto cam_state = timeline.evaluate(t);

                // CameraState is camera-to-world; Camera's R/T are world-to-camera, so invert.
                const glm::mat3 r_c2w = glm::mat3_cast(cam_state.rotation);
                const glm::mat3 r_w2c = glm::transpose(r_c2w);
                const glm::vec3 t_w2c = -(r_w2c * cam_state.position);

                std::vector<float> r_flat(9);
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        r_flat[row * 3 + col] = r_w2c[col][row]; // glm is column-major
                    }
                }
                auto R = core::Tensor::from_vector(r_flat, {3, 3}, core::Device::CPU);
                auto T = core::Tensor::from_vector(
                    std::vector<float>{t_w2c.x, t_w2c.y, t_w2c.z}, {3}, core::Device::CPU);

                const auto [focal_x, focal_y] = rendering::computePixelFocalLengths(
                    {cfg.width, cfg.height}, cam_state.focal_length_mm);

                core::Camera camera(
                    R, T,
                    focal_x, focal_y,
                    static_cast<float>(cfg.width) * 0.5f, static_cast<float>(cfg.height) * 0.5f,
                    core::Tensor::empty({0}, core::Device::CPU),
                    core::Tensor::empty({0}, core::Device::CPU),
                    core::CameraModelType::PINHOLE,
                    std::format("frame_{:06d}", frame),
                    {}, {},
                    cfg.width, cfg.height,
                    frame);

                auto render_output = training::fast_rasterize(camera, *model, background);
                auto image = render_output.image;
                if (image.dtype() != core::DataType::Float32) {
                    image = image.to(core::DataType::Float32);
                }
                if (image.device() != core::Device::CUDA) {
                    image = image.cuda();
                }
                auto image_hwc = image.permute({1, 2, 0}).contiguous();

                const auto write_result = encoder.writeFrameGpu(image_hwc.data_ptr(), cfg.width, cfg.height, nullptr);
                if (!write_result) {
                    LOG_ERROR("Failed to encode frame {}: {}", frame, write_result.error());
                    if (const auto close_result = encoder.close(); !close_result)
                        LOG_WARN("Failed to finalize partial video: {}", close_result.error());
                    return 1;
                }
                LOG_INFO("Encoded frame {}/{}", frame + 1, total_frames);
            }

            if (const auto close_result = encoder.close(); !close_result) {
                LOG_ERROR("Failed to finalize video: {}", close_result.error());
                return 1;
            }

            LOG_INFO("Wrote {} frame(s) to {}", total_frames, core::path_to_utf8(cfg.output_path));
            return 0;
        }

        bool checkCudaDriverVersion() {
            const auto info = lfs::core::check_cuda_version();
            if (info.query_failed) {
                LOG_WARN("Failed to query CUDA driver version");
                return true;
            }

            LOG_INFO("CUDA driver version: {}.{}", info.major, info.minor);
            if (!info.supported) {
                LOG_WARN("CUDA {}.{} unsupported. Requires 12.8+ (driver 570+)", info.major, info.minor);
                return false;
            }
            return true;
        }

        std::future<void>& cudaWarmupFuture() {
            static std::future<void> fut;
            return fut;
        }

        void warmupCudaSync() {
            checkCudaDriverVersion();

            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
                LOG_INFO("GPU: {} (SM {}.{}, {} MB)", prop.name, prop.major, prop.minor,
                         prop.totalGlobalMem / (1024 * 1024));
            }

            LOG_INFO("Initializing CUDA...");
            fast_lfs::rasterization::warmup_kernels();
            lfs::diagnostics::VramProfiler::instance().captureCudaWarmupDelta();
        }

        void warmupCudaAsync() {
            checkCudaDriverVersion();

            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
                LOG_INFO("GPU: {} (SM {}.{}, {} MB)", prop.name, prop.major, prop.minor,
                         prop.totalGlobalMem / (1024 * 1024));
            }

            LOG_INFO("Initializing CUDA (async)...");
            cudaWarmupFuture() = std::async(std::launch::async, [] {
                fast_lfs::rasterization::warmup_kernels();
                lfs::diagnostics::VramProfiler::instance().captureCudaWarmupDelta();
            });
        }

        int runGui(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
            if (!params->python_scripts.empty()) {
                vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
            }

            const bool disable_splash =
#ifdef LFS_BUILD_PORTABLE
                false;
#else
                params->optimization.no_splash;
#endif

            // Warm up on every path, not just import/resume: warmup_kernels forces the
            // lazily-loaded cubins to upload so captureCudaWarmupDelta can attribute that
            // module memory (the cuda.modules row). Without it the modules land in the
            // unattributed NVML residual. warmupCudaAsync runs checkCudaDriverVersion itself.
            warmupCudaAsync();

            lfs::event::CommandCenterBridge::instance().set(&lfs::training::CommandCenter::instance());

            lfs::gui::setVideoWidgetFactory([] {
                return std::make_unique<lfs::gui::VideoExtractorDialog>();
            });
            lfs::gui::setVideoEncoderFactory([] {
                return std::make_unique<lfs::io::video::VideoEncoder>();
            });

            constexpr auto graphics_backend = lfs::vis::GraphicsBackend::Vulkan;
            auto viewer = vis::Visualizer::create({
                .title = "LichtFeld Studio",
                .width = 1280,
                .height = 720,
                .antialiasing = false,
                .show_startup_overlay = !disable_splash,
                .gut = params->optimization.gut,
                .graphics_backend = graphics_backend,
            });

            viewer->setParameters(*params);

            for (const auto& vp : params->view_paths) {
                if (!std::filesystem::exists(vp)) {
                    LOG_ERROR("File not found: {}", lfs::core::path_to_utf8(vp));
                    return 1;
                }
            }
            if (!params->dataset.data_path.empty() && !std::filesystem::exists(params->dataset.data_path)) {
                LOG_ERROR("Dataset not found: {}", lfs::core::path_to_utf8(params->dataset.data_path));
                return 1;
            }

            if (params->import_cameras_path || params->resume_checkpoint) {
                if (auto& fut = cudaWarmupFuture(); fut.valid())
                    fut.wait();
            }

            if (params->import_cameras_path) {
                LOG_INFO("Importing COLMAP cameras: {}", lfs::core::path_to_utf8(*params->import_cameras_path));
                lfs::core::events::cmd::ImportColmapCameras{.sparse_path = *params->import_cameras_path}.emit();
            } else if (params->resume_checkpoint) {
                LOG_INFO("Loading checkpoint: {}", lfs::core::path_to_utf8(*params->resume_checkpoint));
                if (const auto result = viewer->loadCheckpointForTraining(*params->resume_checkpoint); !result) {
                    LOG_ERROR("Failed to load checkpoint: {}", result.error());
                    return 1;
                }
            }

            mcp::register_core_tools();
            mcp::register_core_resources();
            register_gui_scene_tools(viewer.get());
            register_gui_scene_resources(viewer.get());

            mcp::McpHttpServer mcp_http({.enable_resources = true});
            viewer->setShutdownRequestedCallback([&mcp_http]() {
                mcp_http.stop();
            });
            if (!mcp_http.start())
                LOG_ERROR("Failed to start MCP HTTP server");

            viewer->run();

            mcp_http.stop();

            python::finalize();

            viewer.reset();

            core::teardown_gpu_before_exit();

            core::flush_and_exit(0);
        }

#ifdef WIN32
        void hideConsoleWindow() {
            HWND hwnd = GetConsoleWindow();
            Sleep(1);
            HWND owner = GetWindow(hwnd, GW_OWNER);
            DWORD processId;
            GetWindowThreadProcessId(hwnd, &processId);

            if (GetCurrentProcessId() == processId) {
                ShowWindow(owner ? owner : hwnd, SW_HIDE);
            }
        }
#endif

    } // namespace

    int Application::run(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
        // Pre-initialize CacheLoader for the exe module.
        // On Windows, lfs_io (static lib) is linked into both the exe and
        // lfs_visualizer.dll, giving each its own CacheLoader singleton.
        // The callback below executes in the exe's context, so the exe's
        // copy must be initialized before it is invoked.
        lfs::io::CacheLoader::getInstance(
            params->dataset.loading_params.use_cpu_memory,
            params->dataset.loading_params.use_fs_cache);

        lfs::core::set_image_loader([](const lfs::core::ImageLoadParams& p) {
            return lfs::io::CacheLoader::getInstance().load_cached_image(
                p.path,
                {.resize_factor = p.resize_factor,
                 .max_width = p.max_width,
                 .cuda_stream = p.stream,
                 .output_uint8 = p.output_uint8});
        });

        if (params->render_path) {
            const int result = runHeadlessRender(std::move(params));
            if (result == 0) {
                core::teardown_gpu_before_exit();
            }
            return result;
        }

        if (params->optimization.headless && params->server.tcp_connection) {
            return runHeadlessWithTCP(std::move(params));
        }

        if (params->optimization.headless) {
            return runHeadless(std::move(params));
        }

#ifdef WIN32
        hideConsoleWindow();
#endif

        return runGui(std::move(params));
    }

} // namespace lfs::app
