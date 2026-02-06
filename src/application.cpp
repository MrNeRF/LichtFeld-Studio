/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/application.hpp"
#include "core/argument_parser.hpp"
#include "core/logger.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "training/control/control_boundary.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include "visualizer/gui/panels/python_scripts_panel.hpp"
#include "visualizer/scene/scene.hpp" // Scene for unified data storage
#include "visualizer/visualizer.hpp"
#include <cstring>
#ifdef WIN32
#include <windows.h>
#endif
#include "python/runner.hpp"

namespace lfs::core {

    int run_headless_app(std::unique_ptr<param::TrainingParameters> params) {
        if (params->dataset.data_path.empty()) {
            LOG_ERROR("Headless mode requires --data-path");
            return -1;
        }

        LOG_INFO("Starting headless training (using unified Scene)...");

        // Create Scene to hold all training data
        lfs::vis::Scene scene;

        // Load training data into Scene
        auto load_result = lfs::training::loadTrainingDataIntoScene(*params, scene);
        if (!load_result) {
            LOG_ERROR("Failed to load training data: {}", load_result.error());
            return -1;
        }

        // Initialize training model (converts PointCloud to SplatData)
        auto model_init_result = lfs::training::initializeTrainingModel(*params, scene);
        if (!model_init_result) {
            LOG_ERROR("Failed to initialize training model: {}", model_init_result.error());
            return -1;
        }

        // Create Trainer from Scene
        auto trainer = std::make_unique<lfs::training::Trainer>(scene);

        // Set Python scripts if provided
        if (!params->python_scripts.empty()) {
            trainer->set_python_scripts(params->python_scripts);
            lfs::vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
        }

        // Initialize trainer with parameters (creates strategy internally)
        auto init_result = trainer->initialize(*params);
        if (!init_result) {
            LOG_ERROR("Failed to initialize trainer: {}", init_result.error());
            return -1;
        }

        // Free cached CUDA memory before starting training loop
        LOG_INFO("Releasing cached CUDA memory before training...");
        CudaMemoryPool::instance().trim_cached_memory();

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        LOG_INFO("CUDA memory after cleanup: {:.2f} GB free / {:.2f} GB total",
                 free_mem / (1024.0 * 1024.0 * 1024.0),
                 total_mem / (1024.0 * 1024.0 * 1024.0));

        auto train_result = trainer->train();
        if (!train_result) {
            LOG_ERROR("Training error: {}", train_result.error());
            return -1;
        }

        LOG_INFO("Headless training completed successfully");

        if (!params->python_scripts.empty()) {
            lfs::python::finalize();
        }

        return 0;
    }

    int run_gui_app(std::unique_ptr<param::TrainingParameters> params) {
        LOG_INFO("Starting viewer mode...");

        // Populate script manager state for GUI access
        if (!params->python_scripts.empty()) {
            lfs::vis::gui::panels::PythonScriptManagerState::getInstance().setScripts(params->python_scripts);
        }

        // Create visualizer with options
        auto viewer = lfs::vis::Visualizer::create({.title = "LichtFeld Studio",
                                                    .width = 1280,
                                                    .height = 720,
                                                    .antialiasing = false,
                                                    .enable_cuda_interop = true,
                                                    .gut = params->optimization.gut});

        // Set parameters
        viewer->setParameters(*params);

        // Load viewer files
        if (!params->view_paths.empty()) {
            LOG_INFO("Loading {} splat file(s)", params->view_paths.size());
            if (const auto result = viewer->loadPLY(params->view_paths[0]); !result) {
                LOG_ERROR("Failed to load {}: {}", params->view_paths[0].string(), result.error());
                return -1;
            }
            for (size_t i = 1; i < params->view_paths.size(); ++i) {
                if (const auto result = viewer->addSplatFile(params->view_paths[i]); !result) {
                    LOG_ERROR("Failed to load {}: {}", params->view_paths[i].string(), result.error());
                    return -1;
                }
            }
        } else if (params->resume_checkpoint.has_value()) {
            LOG_INFO("Loading checkpoint for training: {}", params->resume_checkpoint->string());
            auto result = viewer->loadCheckpointForTraining(*params->resume_checkpoint);
            if (!result) {
                LOG_ERROR("Failed to load checkpoint: {}", result.error());
                return -1;
            }
        } else if (!params->dataset.data_path.empty()) {
            LOG_INFO("Loading dataset: {}", params->dataset.data_path.string());
            auto result = viewer->loadDataset(params->dataset.data_path);
            if (!result) {
                LOG_ERROR("Failed to load dataset: {}", result.error());
                return -1;
            }
        }

        // Run the viewer
        viewer->run();

        LOG_INFO("Viewer closed");
        return 0;
    }

    int Application::run(std::unique_ptr<param::TrainingParameters> params) {
        int result = 0;

        // no gui
        if (params->optimization.headless) {
            result = run_headless_app(std::move(params));
        } else {
#ifdef WIN32
            // hide console window on windows
            HWND hwnd = GetConsoleWindow();
            Sleep(1);
            HWND owner = GetWindow(hwnd, GW_OWNER);
            DWORD dwProcessId;
            GetWindowThreadProcessId(hwnd, &dwProcessId);

            // Only hide if did not start from console
            if (GetCurrentProcessId() == dwProcessId) {
                if (owner == NULL) {
                    ShowWindow(hwnd, SW_HIDE); // Windows 10
                } else {
                    ShowWindow(owner, SW_HIDE); // Windows 11
                }
            }
#endif
            // gui app
            result = run_gui_app(std::move(params));
        }

        // Ensure explicit shutdown of pinned memory allocator before CUDA context destruction
        // This must happen after run_*_app returns so that all local objects (Scene, Visualizer)
        // have been destroyed and returned their memory to the allocator.
        CudaMemoryPool::instance().shutdown();
        PinnedMemoryAllocator::instance().shutdown();

        return result;
    }
} // namespace lfs::core