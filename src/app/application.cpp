/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/application.hpp"
#include "app/splash_screen.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "rendering/framebuffer_factory.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include "visualizer/scene/scene.hpp"
#include "visualizer/visualizer.hpp"

#include <cuda_runtime.h>
#include <rasterization_api.h>

#ifdef WIN32
#include <windows.h>
#endif

namespace lfs::app {

    namespace {

        int runHeadless(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
            if (params->dataset.data_path.empty()) {
                LOG_ERROR("Headless mode requires --data-path");
                return 1;
            }

            LOG_INFO("Starting headless training...");

            vis::Scene scene;

            if (const auto result = training::loadTrainingDataIntoScene(*params, scene); !result) {
                LOG_ERROR("Failed to load training data: {}", result.error());
                return 1;
            }

            if (const auto result = training::initializeTrainingModel(*params, scene); !result) {
                LOG_ERROR("Failed to initialize model: {}", result.error());
                return 1;
            }

            auto trainer = std::make_unique<training::Trainer>(scene);
            if (const auto result = trainer->initialize(*params); !result) {
                LOG_ERROR("Failed to initialize trainer: {}", result.error());
                return 1;
            }

            lfs::core::CudaMemoryPool::instance().trim_cached_memory();

            if (const auto result = trainer->train(); !result) {
                LOG_ERROR("Training error: {}", result.error());
                return 1;
            }

            LOG_INFO("Headless training completed");
            return 0;
        }

        void warmupCuda() {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
                LOG_INFO("GPU: {} (SM {}.{}, {} MB)", prop.name, prop.major, prop.minor,
                         prop.totalGlobalMem / (1024 * 1024));
            }
            LOG_INFO("Initializing CUDA...");
            fast_lfs::rasterization::warmup_kernels();
        }

        int runGui(std::unique_ptr<lfs::core::param::TrainingParameters> params) {
            if (params->optimization.no_interop) {
                LOG_INFO("CUDA-GL interop disabled");
                lfs::rendering::disableInterop();
            }

            if (params->optimization.no_splash) {
                warmupCuda();
            } else {
                SplashScreen::runWithDelay([]() { warmupCuda(); return 0; });
            }

            auto viewer = vis::Visualizer::create({
                .title = "LichtFeld Studio",
                .width = 1280,
                .height = 720,
                .antialiasing = false,
                .enable_cuda_interop = true,
                .gut = params->optimization.gut,
            });

            viewer->setParameters(*params);

            if (!params->view_paths.empty()) {
                LOG_INFO("Loading {} splat file(s)", params->view_paths.size());
                if (const auto result = viewer->loadPLY(params->view_paths[0]); !result) {
                    LOG_ERROR("Failed to load {}: {}", params->view_paths[0].string(), result.error());
                    return 1;
                }
                for (size_t i = 1; i < params->view_paths.size(); ++i) {
                    if (const auto result = viewer->addSplatFile(params->view_paths[i]); !result) {
                        LOG_ERROR("Failed to load {}: {}", params->view_paths[i].string(), result.error());
                        return 1;
                    }
                }
            } else if (params->resume_checkpoint) {
                LOG_INFO("Loading checkpoint: {}", params->resume_checkpoint->string());
                if (const auto result = viewer->loadCheckpointForTraining(*params->resume_checkpoint); !result) {
                    LOG_ERROR("Failed to load checkpoint: {}", result.error());
                    return 1;
                }
            } else if (!params->dataset.data_path.empty()) {
                LOG_INFO("Loading dataset: {}", params->dataset.data_path.string());
                if (const auto result = viewer->loadDataset(params->dataset.data_path); !result) {
                    LOG_ERROR("Failed to load dataset: {}", result.error());
                    return 1;
                }
            }

            viewer->run();
            return 0;
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
        if (params->optimization.headless) {
            return runHeadless(std::move(params));
        }

#ifdef WIN32
        hideConsoleWindow();
#endif

        return runGui(std::move(params));
    }

} // namespace lfs::app
