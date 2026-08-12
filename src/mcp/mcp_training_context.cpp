/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcp_training_context.hpp"
#include "llm_client.hpp"
#include "mcp_tools.hpp"
#include "shared_scene_tools.hpp"

#include "core/checkpoint_format.hpp"
#include "core/error.hpp"
#include "core/error_reporter.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/guarded_task.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/provenance.hpp"
#include "core/tensor/internal/tensor_ops.hpp"
#include "io/exporter.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"
#include "rendering/selection_ops.hpp"
#include "training/checkpoint.hpp"
#include "training/dataset.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/gsplat/Ops.h"
#include "training/rasterization/gsplat_rasterizer.hpp"
#include "training/training_setup.hpp"
#include "visualizer/selection/selection_group_mask.hpp"

#include <algorithm>
#include <cassert>
#include <optional>
#include <sstream>

namespace lfs::mcp {

    namespace {
        void release_training_thread_local_cuda_caches() noexcept {
            (void)lfs::training::release_fast_rasterizer_thread_local_caches();
            (void)lfs::training::release_gsplat_rasterizer_thread_local_caches();
            (void)gsplat_lfs::release_intersect_thread_local_cache();
            (void)lfs::core::tensor_ops::release_nan_check_thread_buffers();
            lfs::training::release_fastgs_sort_workspace_buffers();
        }

        std::expected<std::pair<core::SplatData*, std::shared_ptr<core::Camera>>, std::string>
        resolve_model_and_camera(const std::shared_ptr<core::Scene>& scene,
                                 int camera_index) {
            if (!scene) {
                return std::unexpected("No scene loaded");
            }

            auto* model = scene->getTrainingModel();
            if (!model) {
                return std::unexpected("No model loaded");
            }

            auto cameras = scene->getAllCameras();
            if (cameras.empty()) {
                return std::unexpected("No cameras available");
            }

            if (camera_index < 0 || camera_index >= static_cast<int>(cameras.size())) {
                camera_index = 0;
            }

            auto camera = cameras[camera_index];
            if (!camera) {
                return std::unexpected("Failed to get camera");
            }

            return std::pair{model, std::move(camera)};
        }

        std::expected<core::Tensor, std::string> compute_screen_positions_for_scene(
            const std::shared_ptr<core::Scene>& scene,
            const int camera_index) {
            auto resolved = resolve_model_and_camera(scene, camera_index);
            if (!resolved) {
                return std::unexpected(resolved.error());
            }
            return std::unexpected(
                "Headless CUDA screen-position rendering has been removed; use the live Vulkan selection path");
        }

        std::expected<std::string, std::string> render_to_base64_for_scene(
            const std::shared_ptr<core::Scene>& scene,
            const int camera_index,
            const int width,
            const int height) {
            auto resolved = resolve_model_and_camera(scene, camera_index);
            if (!resolved) {
                return std::unexpected(resolved.error());
            }
            (void)width;
            (void)height;
            return std::unexpected(
                "Headless CUDA scene rendering has been removed; use live Vulkan viewport capture");
        }

    } // namespace

    TrainingContext& TrainingContext::instance() {
        static TrainingContext inst;
        return inst;
    }

    TrainingContext::~TrainingContext() {
        shutdown();
    }

    std::expected<void, std::string> TrainingContext::load_dataset(
        const std::filesystem::path& path,
        const core::param::TrainingParameters& params) {

        std::lock_guard lock(mutex_);

        stop_training_locked();

        params_ = params;
        params_.dataset.data_path = path;

        scene_ = std::make_shared<core::Scene>();

        if (auto result = training::loadTrainingDataIntoScene(params_, *scene_); !result) {
            scene_.reset();
            return std::unexpected(result.error());
        }

        if (auto result = training::initializeTrainingModel(params_, *scene_); !result) {
            scene_.reset();
            return std::unexpected(result.error());
        }

        trainer_ = std::make_shared<training::Trainer>(*scene_);

        if (auto result = trainer_->initialize(params_); !result) {
            trainer_.reset();
            scene_.reset();
            return std::unexpected(result.error());
        }

        LOG_INFO("MCP: Loaded dataset from {}", core::path_to_utf8(path));
        return {};
    }

    std::expected<void, std::string> TrainingContext::load_checkpoint(
        const std::filesystem::path& path) {

        std::lock_guard lock(mutex_);

        stop_training_locked();

        auto header_result = core::load_checkpoint_header(path);
        if (!header_result) {
            return std::unexpected(header_result.error());
        }

        auto params_result = core::load_checkpoint_params(path);
        if (!params_result) {
            return std::unexpected(params_result.error());
        }
        params_ = std::move(*params_result);

        auto splat_result = core::load_checkpoint_splat_data(path);
        if (!splat_result) {
            return std::unexpected(splat_result.error());
        }

        scene_ = std::make_shared<core::Scene>();
        scene_->setTrainingModel(
            std::make_unique<core::SplatData>(std::move(*splat_result)),
            "checkpoint");

        trainer_ = std::make_shared<training::Trainer>(*scene_);

        if (auto result = trainer_->initialize(params_); !result) {
            trainer_.reset();
            scene_.reset();
            return std::unexpected(result.error());
        }

        LOG_INFO("MCP: Loaded checkpoint from {}", core::path_to_utf8(path));
        return {};
    }

    std::expected<void, std::string> TrainingContext::save_checkpoint(
        const std::filesystem::path& path) {

        std::lock_guard lock(mutex_);

        if (!trainer_) {
            return std::unexpected("No training session to save");
        }

        auto result = training::save_checkpoint(
            path,
            trainer_->get_current_iteration(),
            trainer_->get_strategy(),
            params_,
            nullptr);

        if (!result) {
            return std::unexpected(result.error());
        }

        LOG_INFO("MCP: Saved checkpoint to {}", core::path_to_utf8(path));
        return {};
    }

    std::expected<void, std::string> TrainingContext::save_ply(
        const std::filesystem::path& path,
        const bool include_provenance) {

        std::lock_guard lock(mutex_);

        if (!scene_) {
            return std::unexpected("No scene to save");
        }

        auto* model = scene_->getTrainingModel();
        if (!model) {
            return std::unexpected("No model to save");
        }

        auto stamp = include_provenance ? core::make_provenance_stamp()
                                        : core::make_minimal_provenance_stamp();
        if (include_provenance && trainer_) {
            const int iteration = trainer_->get_current_iteration();
            if (iteration > 0)
                stamp.iteration = iteration;
            const auto strategy = core::param::canonical_strategy_name(
                trainer_->getParams().optimization.strategy);
            if (!strategy.empty())
                stamp.strategy = std::string(strategy);
        }

        io::PlySaveOptions options{.output_path = path, .binary = true, .provenance = std::move(stamp)};
        auto result = io::save_ply(*model, options);
        if (!result) {
            return std::unexpected(result.error().message);
        }

        LOG_INFO("MCP: Saved PLY to {}", core::path_to_utf8(path));
        return {};
    }

    std::expected<std::string, std::string> TrainingContext::render_to_base64(
        int camera_index,
        int width,
        int height) {
        return render_to_base64_for_scene(scene(), camera_index, width, height);
    }

    std::expected<core::Tensor, std::string> TrainingContext::compute_screen_positions(
        int camera_index) {
        return compute_screen_positions_for_scene(scene(), camera_index);
    }

    std::expected<void, std::string> TrainingContext::start_training() {
        std::lock_guard lock(mutex_);

        if (!trainer_) {
            return std::unexpected("No trainer initialized");
        }

        if (training_thread_ && training_active_.load(std::memory_order_acquire)) {
            return std::unexpected("Training already running");
        }

        // A naturally completed jthread remains joinable until its owner reaps
        // it. Join that finished generation before installing the next one.
        training_thread_.reset();

        auto trainer = trainer_;
        training_active_.store(true, std::memory_order_release);
        try {
            last_training_error_.clear();
            training_thread_ = std::make_unique<std::jthread>([this, trainer](std::stop_token stop) {
                lfs::core::run_guarded<void>(
                    lfs::core::TaskContext{
                        .name = "mcp.training-worker",
                        .domain = lfs::ErrorDomain::Training,
                        .operation_id = lfs::OperationId::generate(),
                        .site = LFS_SOURCE_SITE_CURRENT(),
                    },
                    [trainer, stop]() -> lfs::Result<void> {
                        return trainer->train(stop);
                    },
                    [this](lfs::Result<void>&& result) {
                        if (!result) {
                            last_training_error_.set(result.error());
                            lfs::core::ErrorReporter::get().report(result.error(), lfs::core::ReportChannel::OwnerLog);
                        }
                        training_active_.store(false, std::memory_order_release);
                    });
                release_training_thread_local_cuda_caches();
            });
        } catch (const std::exception& e) {
            training_active_.store(false, std::memory_order_release);
            std::string message = std::string("Failed to start training thread: ") + e.what();
            last_training_error_.set(lfs::make_legacy_error(message, lfs::LegacyErrorContext{
                                                                         .code = lfs::ErrorCode::FailedPrecondition,
                                                                         .domain = lfs::ErrorDomain::Training,
                                                                         .operation = "mcp.training.start",
                                                                         .source = LFS_SOURCE_SITE_CURRENT(),
                                                                         .operation_id = lfs::OperationId::generate(),
                                                                     }));
            return std::unexpected(std::move(message));
        }

        LOG_INFO("MCP: Training started");
        return {};
    }

    void TrainingContext::stop_training() {
        std::lock_guard lock(mutex_);
        stop_training_locked();
    }

    void TrainingContext::stop_training_locked() {
        if (training_thread_) {
            training_thread_->request_stop();
            training_thread_.reset();
        }
        training_active_.store(false, std::memory_order_release);
    }

    void TrainingContext::pause_training() {
        auto trainer = this->trainer();
        if (trainer) {
            trainer->request_pause();
        }
    }

    void TrainingContext::resume_training() {
        auto trainer = this->trainer();
        if (trainer) {
            trainer->request_resume();
        }
    }

    void TrainingContext::shutdown() {
        std::lock_guard lock(mutex_);
        stop_training_locked();
        trainer_.reset();
        scene_.reset();
    }

} // namespace lfs::mcp
