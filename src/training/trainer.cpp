/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "trainer.hpp"
#include "components/bilateral_grid.hpp"
#include "components/ppisp.hpp"
#include "components/ppisp_controller_pool.hpp"
#include "components/ppisp_file.hpp"
#include "components/sparsity_optimizer.hpp"
#include "control/command_api.hpp"
#include "control/control_boundary.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "io/cache_image_loader.hpp"
#include "io/exporter.hpp"
#include "io/filesystem_utils.hpp"
#include "lfs/kernels/ssim.cuh"
#include "losses/losses.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "python/runner.hpp"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/gsplat_rasterizer.hpp"
#include "strategies/adc.hpp"
#include "strategies/mcmc.hpp"
#include "strategies/strategy_factory.hpp"
#include "training/kernels/grad_alpha.hpp"
#include "visualizer/scene/scene.hpp"

#include <filesystem>

#include <atomic>
#include <cmath>
#include <cuda_runtime.h>
#include <expected>
#include <memory>
#include <nvtx3/nvToolsExt.h>

namespace lfs::training {

    namespace {
        PPISPRenderOverrides toRenderOverrides(const PPISPViewportOverrides& ov) {
            PPISPRenderOverrides r;
            r.exposure_offset = ov.exposure_offset;
            r.vignette_enabled = ov.vignette_enabled;
            r.vignette_strength = ov.vignette_strength;
            r.wb_temperature = ov.wb_temperature;
            r.wb_tint = ov.wb_tint;
            r.color_red_x = ov.color_red_x;
            r.color_red_y = ov.color_red_y;
            r.color_green_x = ov.color_green_x;
            r.color_green_y = ov.color_green_y;
            r.color_blue_x = ov.color_blue_x;
            r.color_blue_y = ov.color_blue_y;
            r.gamma_multiplier = ov.gamma_multiplier;
            r.gamma_red = ov.gamma_red;
            r.gamma_green = ov.gamma_green;
            r.gamma_blue = ov.gamma_blue;
            r.crf_toe = ov.crf_toe;
            r.crf_shoulder = ov.crf_shoulder;
            return r;
        }
    } // namespace

    // Tile configuration for memory-efficient training
    enum class TileMode {
        One = 1, // 1 tile  - 1x1 - Render full image (no tiling)
        Two = 2, // 2 tiles - 2x1 - Two horizontal tiles
        Four = 4 // 4 tiles - 2x2 - Four tiles in a grid
    };

    void Trainer::cleanup() {
        LOG_DEBUG("Cleaning up trainer for re-initialization");

        // Stop any ongoing operations
        stop_requested_ = true;

        // Sync callback stream to avoid race conditions
        if (callback_stream_) {
            cudaStreamSynchronize(callback_stream_);
        }
        callback_busy_ = false;

        // Reset all components
        progress_.reset();
        bilateral_grid_.reset();
        ppisp_.reset();
        ppisp_controller_pool_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();

        // Clear datasets (will be recreated)
        train_dataset_.reset();
        val_dataset_.reset();

        // Reset flags
        pause_requested_ = false;
        save_requested_ = false;
        stop_requested_ = false;
        is_paused_ = false;
        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false;
        current_iteration_ = 0;
        current_loss_ = 0.0f;

        LOG_DEBUG("Trainer cleanup complete");
    }

    std::expected<void, std::string> Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return {};
        }

        try {
            BilateralGrid::Config config;
            config.lr = params_.optimization.bilateral_grid_lr;

            bilateral_grid_ = std::make_unique<BilateralGrid>(
                static_cast<int>(train_dataset_size_),
                params_.optimization.bilateral_grid_X,
                params_.optimization.bilateral_grid_Y,
                params_.optimization.bilateral_grid_W,
                params_.optimization.iterations,
                config);

            LOG_INFO("Bilateral grid initialized: {}x{}x{} for {} images",
                     params_.optimization.bilateral_grid_X,
                     params_.optimization.bilateral_grid_Y,
                     params_.optimization.bilateral_grid_W,
                     train_dataset_size_);

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init bilateral grid: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::initialize_ppisp() {
        if (!params_.optimization.use_ppisp) {
            return {};
        }

        try {
            PPISPConfig config;
            config.lr = params_.optimization.ppisp_lr;
            config.warmup_steps = params_.optimization.ppisp_warmup_steps;

            ppisp_ = std::make_unique<PPISP>(params_.optimization.iterations, config);
            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam) {
                    ppisp_->register_frame(cam->uid(), cam->camera_id());
                }
            }
            ppisp_->finalize();

            LOG_INFO("PPISP initialized: {} cameras (physical), {} frames, lr={:.2e}, warmup={}",
                     ppisp_->num_cameras(), ppisp_->num_frames(), params_.optimization.ppisp_lr, config.warmup_steps);

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init PPISP: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::initialize_ppisp_controller() {
        if (!params_.optimization.ppisp_use_controller || !params_.optimization.use_ppisp) {
            return {};
        }

        if (!ppisp_) {
            return std::unexpected("PPISP must be initialized before controller");
        }

        try {
            PPISPControllerPool::Config config;
            config.lr = params_.optimization.ppisp_controller_lr;

            if (params_.optimization.ppisp_controller_activation_step < 0) {
                params_.optimization.ppisp_controller_activation_step =
                    std::max(0, static_cast<int>(params_.optimization.iterations) - 5000);
            }
            int distillation_iters =
                static_cast<int>(params_.optimization.iterations) - params_.optimization.ppisp_controller_activation_step;
            int num_cameras = ppisp_->num_cameras();

            ppisp_controller_pool_ = std::make_unique<PPISPControllerPool>(num_cameras, distillation_iters, config);

            size_t max_h = 0, max_w = 0;
            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam) {
                    max_h = std::max(max_h, static_cast<size_t>(cam->image_height()));
                    max_w = std::max(max_w, static_cast<size_t>(cam->image_width()));
                }
            }
            ppisp_controller_pool_->allocate_buffers(max_h, max_w);

            LOG_INFO("PPISP controller pool initialized: num_cameras={}, activation_step={}, lr={:.2e}, max_image={}x{}",
                     num_cameras, params_.optimization.ppisp_controller_activation_step,
                     params_.optimization.ppisp_controller_lr, static_cast<int>(max_h), static_cast<int>(max_w));

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init PPISP controller pool: {}", e.what()));
        }
    }

    // Compute photometric loss AND gradient manually
    std::expected<std::pair<lfs::core::Tensor, lfs::core::Tensor>, std::string> Trainer::compute_photometric_loss_with_gradient(
        const lfs::core::Tensor& rendered,
        const lfs::core::Tensor& gt_image,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::PhotometricLoss photometric_loss;
        lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
        auto result = photometric_loss.forward(rendered, gt_image, params);
        if (!result) {
            return std::unexpected(result.error());
        }
        auto [loss_tensor, ctx] = *result;
        return std::make_pair(loss_tensor, ctx.grad_image);
    }

    std::expected<void, std::string> Trainer::validate_masks() {
        const auto& opt = params_.optimization;
        if (opt.mask_mode == lfs::core::param::MaskMode::None) {
            return {};
        }

        const bool alpha_available = scene_ && scene_->imagesHaveAlpha();
        if (opt.use_alpha_as_mask && alpha_available) {
            LOG_INFO("Using alpha channel as mask source{}", opt.invert_masks ? " (inverted)" : "");
            return {};
        }

        size_t masks_found = 0;
        for (const auto& cam : train_dataset_->get_cameras()) {
            if (cam && cam->has_mask()) {
                ++masks_found;
            }
        }

        if (masks_found == 0) {
            return std::unexpected(std::format(
                "Mask mode enabled but no masks found in {}/masks/",
                lfs::core::path_to_utf8(params_.dataset.data_path)));
        }

        LOG_INFO("Found {} masks{}", masks_found, opt.invert_masks ? " (inverted)" : "");
        return {};
    }

    std::expected<Trainer::MaskLossResult, std::string> Trainer::compute_photometric_loss_with_mask(
        const lfs::core::Tensor& rendered,
        const lfs::core::Tensor& gt_image,
        const lfs::core::Tensor& mask,
        const lfs::core::Tensor& alpha,
        const lfs::core::param::OptimizationParameters& opt_params) {

        using namespace lfs::core;
        constexpr float EPSILON = 1e-8f;
        constexpr float ALPHA_CONSISTENCY_WEIGHT = 10.0f;

        const auto mode = opt_params.mask_mode;
        const Tensor mask_2d = mask.ndim() == 3 ? mask.squeeze(0) : mask;

        Tensor loss, grad, grad_alpha;

        if (mode == param::MaskMode::Segment || mode == param::MaskMode::Ignore) {
            if (opt_params.lambda_dssim > 0.0f) {
                // Use FUSED masked L1+SSIM kernel
                auto [loss_tensor, ctx] = lfs::training::kernels::masked_fused_l1_ssim_forward(
                    rendered, gt_image, mask_2d, opt_params.lambda_dssim, masked_fused_workspace_);

                grad = lfs::training::kernels::masked_fused_l1_ssim_backward(ctx, masked_fused_workspace_);
                loss = loss_tensor;

                // Squeeze gradient to match input dimensions (loss is scalar, no adjustment needed)
                if (grad.ndim() == 4 && rendered.ndim() == 3) {
                    grad = grad.squeeze(0);
                }
            } else {
                // Pure L1 with mask (no SSIM)
                const Tensor mask_expanded = mask_2d.unsqueeze(0).expand({static_cast<int>(rendered.shape()[0]),
                                                                          static_cast<int>(mask_2d.shape()[0]),
                                                                          static_cast<int>(mask_2d.shape()[1])});
                const Tensor mask_sum = mask_expanded.sum() + EPSILON;

                const Tensor l1_diff = (rendered - gt_image).abs();
                const Tensor masked_l1 = (l1_diff * mask_expanded).sum() / mask_sum;
                const Tensor sign_diff = (rendered - gt_image).sign();
                grad = sign_diff * mask_expanded / mask_sum;
                loss = masked_l1;
            }

            // Segment: opacity penalty for background
            if (mode == param::MaskMode::Segment && alpha.is_valid()) {
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor bg_mask = Tensor::full(mask_2d.shape(), 1.0f, mask_2d.device()) - mask_2d;
                const Tensor penalty_weights = bg_mask.pow(opt_params.mask_opacity_penalty_power);
                const Tensor penalty = (alpha_2d * penalty_weights).mean() * opt_params.mask_opacity_penalty_weight;

                const float inv_pixels = opt_params.mask_opacity_penalty_weight / static_cast<float>(alpha_2d.numel());
                grad_alpha = penalty_weights * inv_pixels;
                loss = loss + penalty;
            }

        } else if (mode == param::MaskMode::AlphaConsistent) {
            // Standard photometric loss
            lfs::training::losses::PhotometricLoss photo_loss_fn;
            const lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
            auto result = photo_loss_fn.forward(rendered, gt_image, params);
            if (!result) {
                return std::unexpected(result.error());
            }
            auto [photo_loss, ctx] = *result;
            loss = photo_loss;
            grad = ctx.grad_image;

            // Alpha should match mask
            if (alpha.is_valid()) {
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor alpha_loss = (alpha_2d - mask_2d).abs().mean() * ALPHA_CONSISTENCY_WEIGHT;
                loss = loss + alpha_loss;
                grad_alpha = (alpha_2d - mask_2d).sign() * (ALPHA_CONSISTENCY_WEIGHT / static_cast<float>(alpha_2d.numel()));
            }
        } else {
            auto fallback = compute_photometric_loss_with_gradient(rendered, gt_image, opt_params);
            if (!fallback) {
                return std::unexpected(fallback.error());
            }
            return MaskLossResult{.loss = fallback->first, .grad_image = fallback->second, .grad_alpha = {}};
        }

        return MaskLossResult{.loss = loss, .grad_image = grad, .grad_alpha = grad_alpha};
    }

    // Returns GPU tensor for loss - NO SYNC!
    std::expected<lfs::core::Tensor, std::string> Trainer::compute_scale_reg_loss(
        lfs::core::SplatData& splatData,
        AdamOptimizer& optimizer,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::ScaleRegularization::Params params{.weight = opt_params.scale_reg};
        return lfs::training::losses::ScaleRegularization::forward(splatData.scaling_raw(), optimizer.get_grad(ParamType::Scaling), params);
    }

    // Returns GPU tensor for loss - NO SYNC!
    std::expected<lfs::core::Tensor, std::string> Trainer::compute_opacity_reg_loss(
        lfs::core::SplatData& splatData,
        AdamOptimizer& optimizer,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::OpacityRegularization::Params params{.weight = opt_params.opacity_reg};
        return lfs::training::losses::OpacityRegularization::forward(splatData.opacity_raw(), optimizer.get_grad(ParamType::Opacity), params);
    }

    std::expected<std::pair<lfs::core::Tensor, SparsityLossContext>, std::string>
    Trainer::compute_sparsity_loss_forward(const int iter, const lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_apply_loss(iter)) {
            auto zero = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            return std::make_pair(std::move(zero), SparsityLossContext{});
        }

        if (!sparsity_optimizer_->is_initialized()) {
            if (auto result = sparsity_optimizer_->initialize(splat_data.opacity_raw()); !result) {
                return std::unexpected(result.error());
            }
            LOG_DEBUG("Sparsity optimizer initialized at iteration {}", iter);
        }

        return sparsity_optimizer_->compute_loss_forward(splat_data.opacity_raw());
    }

    std::expected<void, std::string> Trainer::handle_sparsity_update(const int iter, lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_update(iter)) {
            return {};
        }
        return sparsity_optimizer_->update_state(splat_data.opacity_raw());
    }

    std::expected<void, std::string> Trainer::apply_sparsity_pruning(const int iter, lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_prune(iter)) {
            return {};
        }

        auto mask_result = sparsity_optimizer_->get_prune_mask(splat_data.opacity_raw());
        if (!mask_result) {
            return std::unexpected(mask_result.error());
        }

        const int n_before = static_cast<int>(splat_data.size());
        strategy_->remove_gaussians(*mask_result);
        const int n_after = static_cast<int>(splat_data.size());

        LOG_INFO("Sparsity pruning: {} -> {} Gaussians ({}% reduction)",
                 n_before, n_after, static_cast<int>(100.0f * (n_before - n_after) / n_before));

        sparsity_optimizer_.reset();
        return {};
    }

    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy,
                     std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits)
        : base_dataset_(std::move(dataset)),
          strategy_(std::move(strategy)),
          provided_splits_(std::move(provided_splits)) {
        // Check CUDA availability
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        cudaStreamCreateWithFlags(&callback_stream_, cudaStreamNonBlocking);

        LOG_DEBUG("Trainer constructed with {} cameras", base_dataset_->get_cameras().size());
    }

    // New constructor - Scene owns all data
    Trainer::Trainer(lfs::vis::Scene& scene)
        : scene_(&scene) {
        // Check CUDA availability
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        cudaStreamCreateWithFlags(&callback_stream_, cudaStreamNonBlocking);

        if (!scene.hasTrainingData()) {
            throw std::runtime_error("Scene has no cameras");
        }

        LOG_DEBUG("Trainer constructed from Scene with {} cameras", scene.getAllCameras().size());
    }

    std::expected<void, std::string> Trainer::initialize(const lfs::core::param::TrainingParameters& params) {
        // Thread-safe initialization using mutex
        std::lock_guard<std::mutex> lock(init_mutex_);

        // Check again after acquiring lock (double-checked locking pattern)
        if (initialized_.load()) {
            LOG_INFO("Re-initializing trainer with new parameters");
            // Clean up existing state for re-initialization
            cleanup();
        }

        LOG_INFO("Initializing trainer with {} iterations", params.optimization.iterations);

        try {
            params_ = params;

            // Create DatasetConfig for lfs::training::CameraDataset
            lfs::training::DatasetConfig dataset_config;
            dataset_config.resize_factor = params.dataset.resize_factor;
            dataset_config.max_width = params.dataset.max_width;
            dataset_config.test_every = params.dataset.test_every;

            // Get source cameras from Scene nodes or base_dataset_
            std::vector<std::shared_ptr<lfs::core::Camera>> source_cameras;
            if (scene_) {
                source_cameras = scene_->getActiveCameras();
                if (source_cameras.empty()) {
                    return std::unexpected("Scene has no active cameras enabled for training");
                }
            } else if (base_dataset_) {
                source_cameras = base_dataset_->get_cameras();
            } else {
                return std::unexpected("No camera source available");
            }

            // Handle dataset split based on evaluation flag
            if (params.optimization.enable_eval) {
                // Create train/val split
                train_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::TRAIN,
                    provided_splits_ ? std::make_optional(std::get<0>(*provided_splits_)) : std::nullopt);
                val_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::VAL,
                    provided_splits_ ? std::make_optional(std::get<1>(*provided_splits_)) : std::nullopt);

                LOG_INFO("Created train/val split: {} train, {} val images",
                         train_dataset_->size(),
                         val_dataset_->size());
            } else {
                // Use all images for training
                train_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::ALL);
                val_dataset_ = nullptr;

                LOG_INFO("Using all {} images for training (no evaluation)",
                         train_dataset_->size());
            }

            train_dataset_size_ = train_dataset_->size();

            // If using Scene mode and no strategy yet, create one
            if (scene_ && !strategy_) {
                auto* model = scene_->getTrainingModel();
                if (!model) {
                    return std::unexpected("Scene has no training model set");
                }

                auto result = StrategyFactory::instance().create(params.optimization.strategy, *model);
                if (!result) {
                    return std::unexpected(result.error());
                }
                strategy_ = std::move(*result);
                LOG_DEBUG("Created {} strategy from Scene model", params.optimization.strategy);
            }

            auto& splat = strategy_->get_model();

            int max_cap = params.optimization.max_cap;
            if (max_cap < splat.size()) {
                LOG_WARN("Max cap is less than to {} initial splats {}. Choosing randomly {} splats", max_cap, splat.size(), max_cap);
                lfs::core::random_choose(splat, max_cap);
            }

            // Re-initialize strategy with new parameters
            strategy_->initialize(params.optimization);
            LOG_DEBUG("Strategy initialized");

            // Initialize bilateral grid if enabled
            if (auto result = initialize_bilateral_grid(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize PPISP if enabled
            if (auto result = initialize_ppisp(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize PPISP controller if enabled
            if (auto result = initialize_ppisp_controller(); !result) {
                return std::unexpected(result.error());
            }

            // Validate masks if mask mode is enabled
            if (auto result = validate_masks(); !result) {
                return std::unexpected(result.error());
            }

            // Prepare undistortion for all cameras with distortion
            if (params.optimization.undistort) {
                int prepared = 0;
                for (auto& cam : train_dataset_->get_cameras()) {
                    if (cam && cam->has_distortion()) {
                        cam->prepare_undistortion();
                        ++prepared;
                    }
                }
                if (val_dataset_) {
                    for (auto& cam : val_dataset_->get_cameras()) {
                        if (cam && cam->has_distortion()) {
                            cam->prepare_undistortion();
                        }
                    }
                }
                if (prepared > 0) {
                    LOG_INFO("Prepared undistortion for {} cameras", prepared);
                }
            }

            // Initialize sparsity optimizer
            if (params.optimization.enable_sparsity) {
                constexpr int UPDATE_INTERVAL = 50;
                const int sparsify_steps = params.optimization.sparsify_steps;
                const int stored_iters = static_cast<int>(params.optimization.iterations);

                // Checkpoint already has total iterations; fresh start needs sparsify_steps added
                const bool is_resume = params.resume_checkpoint.has_value();
                const int base_iters = is_resume ? (stored_iters - sparsify_steps) : stored_iters;

                if (!is_resume) {
                    params_.optimization.iterations = static_cast<size_t>(base_iters + sparsify_steps);
                }

                const ADMMSparsityOptimizer::Config config{
                    .sparsify_steps = sparsify_steps,
                    .init_rho = params.optimization.init_rho,
                    .prune_ratio = params.optimization.prune_ratio,
                    .update_every = UPDATE_INTERVAL,
                    .start_iteration = base_iters};

                sparsity_optimizer_ = SparsityOptimizerFactory::create("admm", config);
                if (sparsity_optimizer_) {
                    LOG_INFO("Sparsity: base={}, steps={}, prune={:.0f}%",
                             base_iters, sparsify_steps, params.optimization.prune_ratio * 100);
                }
            }

            // Initialize background color tensor from params
            {
                const auto& bg_color = params.optimization.bg_color;
                background_ = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
                auto* bg_ptr = background_.ptr<float>();
                bg_ptr[0] = bg_color[0];
                bg_ptr[1] = bg_color[1];
                bg_ptr[2] = bg_color[2];
                background_ = background_.to(lfs::core::Device::CUDA);
                LOG_INFO("Background color set to RGB({:.2f}, {:.2f}, {:.2f})", bg_color[0], bg_color[1], bg_color[2]);
            }

            // Load background image if specified
            if (params.optimization.bg_mode == lfs::core::param::BackgroundMode::Image &&
                !params.optimization.bg_image_path.empty() &&
                std::filesystem::exists(params.optimization.bg_image_path)) {
                try {
                    auto& loader = lfs::io::CacheLoader::getInstance();
                    lfs::io::LoadParams load_params{
                        .resize_factor = 1,
                        .max_width = 0, // No max width limit
                        .cuda_stream = nullptr};
                    bg_image_base_ = loader.load_cached_image(params.optimization.bg_image_path, load_params);
                    if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                        bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                    }
                    if (bg_image_base_.shape()[0] != 3) {
                        LOG_WARN("Background image has {} channels, expected 3 (RGB)", bg_image_base_.shape()[0]);
                        bg_image_base_ = {};
                        params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                    } else {
                        LOG_INFO("Background image: {} [{}x{}]",
                                 lfs::core::path_to_utf8(params.optimization.bg_image_path),
                                 bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to load background image: {}", e.what());
                    params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                }
            }

            // Create progress bar based on headless flag
            if (params.optimization.headless) {
                progress_ = std::make_unique<TrainingProgress>(
                    params_.optimization.iterations, // This now includes sparsity steps if enabled
                    /*update_frequency=*/100);
                LOG_DEBUG("Progress bar initialized for {} total iterations", params_.optimization.iterations);
            }

            // Initialize the evaluator - it handles all metrics internally
            evaluator_ = std::make_unique<lfs::training::MetricsEvaluator>(params_);
            LOG_DEBUG("Metrics evaluator initialized");

            // Resume from checkpoint if provided
            if (params_.resume_checkpoint.has_value()) {
                auto resume_result = load_checkpoint(*params_.resume_checkpoint);
                if (!resume_result) {
                    return std::unexpected(std::format("Failed to resume from checkpoint: {}", resume_result.error()));
                }
                LOG_INFO("Resumed training from checkpoint at iteration {}", *resume_result);

                // Reload bg_image if checkpoint restored different settings
                if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Image &&
                    !params_.optimization.bg_image_path.empty() &&
                    std::filesystem::exists(params_.optimization.bg_image_path) &&
                    !bg_image_base_.is_valid()) {
                    try {
                        auto& loader = lfs::io::CacheLoader::getInstance();
                        lfs::io::LoadParams load_params{.resize_factor = 1, .max_width = 0, .cuda_stream = nullptr};
                        bg_image_base_ = loader.load_cached_image(params_.optimization.bg_image_path, load_params);
                        if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                            bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                        }
                        if (bg_image_base_.shape()[0] != 3) {
                            LOG_WARN("Background image has {} channels, expected 3", bg_image_base_.shape()[0]);
                            bg_image_base_ = {};
                            params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                        } else {
                            LOG_INFO("Background image from checkpoint: {} [{}x{}]",
                                     lfs::core::path_to_utf8(params_.optimization.bg_image_path),
                                     bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                        }
                    } catch (const std::exception& e) {
                        LOG_WARN("Failed to load background image from checkpoint: {}", e.what());
                        params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                    }
                }
            }

            // Print configuration
            LOG_INFO("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
            LOG_INFO("Strategy: {}", params.optimization.strategy);
            if (params.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                static constexpr const char* MASK_MODE_NAMES[] = {"none", "segment", "ignore", "alpha_consistent"};
                LOG_INFO("Mask mode: {}", MASK_MODE_NAMES[static_cast<int>(params.optimization.mask_mode)]);
            }
            if (current_iteration_ > 0) {
                LOG_INFO("Starting from iteration: {}", current_iteration_.load());
            }

            // Expose initial snapshot for Python control (iteration 0)
            {
                lfs::training::HookContext ctx{
                    .iteration = current_iteration_.load(),
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(current_iteration_.load()) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::SafeControl);
            }

            // Default Python control script if none provided
            if (python_scripts_.empty()) {
                std::filesystem::path default_script = std::filesystem::path(PROJECT_ROOT_PATH) / "src/python/sample/command_demo.py";
                if (std::filesystem::exists(default_script)) {
                    set_python_scripts({default_script});
                    LOG_INFO("No Python scripts specified; using default: {}", default_script.string());
                } else {
                    LOG_WARN("Default Python script not found: {}", default_script.string());
                }
            }

            // Execute configured Python scripts to register iteration callbacks
            if (!python_scripts_.empty()) {
                auto py_result = lfs::python::run_scripts(python_scripts_);
                if (!py_result) {
                    return std::unexpected(std::format("Failed to run Python scripts: {}", py_result.error()));
                }
            }

            initialized_ = true;
            LOG_INFO("Trainer initialization complete");
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize trainer: {}", e.what()));
        }
    }

    Trainer::~Trainer() {
        shutdown();
    }

    void Trainer::shutdown() {
        if (shutdown_complete_.exchange(true)) {
            return;
        }

        LOG_DEBUG("Trainer shutdown");
        stop_requested_ = true;

        lfs::core::image_io::BatchImageSaver::instance().wait_all();

        if (callback_stream_) {
            cudaStreamSynchronize(callback_stream_);
            cudaStreamDestroy(callback_stream_);
            callback_stream_ = nullptr;
        }
        callback_busy_ = false;

        cudaDeviceSynchronize();

        strategy_.reset();
        bilateral_grid_.reset();
        ppisp_.reset();
        ppisp_controller_pool_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();
        progress_.reset();
        train_dataset_.reset();
        val_dataset_.reset();

        // Release GPU memory pools back to system
        lfs::core::CudaMemoryPool::instance().trim_cached_memory();
        lfs::core::GlobalArenaManager::instance().get_arena().full_reset();
        cudaDeviceSynchronize();
        LOG_DEBUG("GPU memory released");

        initialized_ = false;
        is_running_ = false;
        training_complete_ = false;
    }

    void Trainer::setParams(const lfs::core::param::TrainingParameters& params) {
        // Check if background image path changed and needs to be (re)loaded
        const bool bg_image_path_changed =
            params.optimization.bg_image_path != params_.optimization.bg_image_path;
        const bool bg_mode_is_image =
            params.optimization.bg_mode == lfs::core::param::BackgroundMode::Image;

        // Update params first
        params_ = params;

        // Load/reload background image if needed
        if (bg_mode_is_image && bg_image_path_changed &&
            !params.optimization.bg_image_path.empty() &&
            std::filesystem::exists(params.optimization.bg_image_path)) {
            try {
                auto& loader = lfs::io::CacheLoader::getInstance();
                lfs::io::LoadParams load_params{
                    .resize_factor = 1,
                    .max_width = 0,
                    .cuda_stream = nullptr};
                bg_image_base_ = loader.load_cached_image(params.optimization.bg_image_path, load_params);
                if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                    bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                }
                bg_image_cache_.clear();
                if (bg_image_base_.shape()[0] != 3) {
                    LOG_WARN("Background image has {} channels, expected 3 (RGB)", bg_image_base_.shape()[0]);
                    bg_image_base_ = {};
                    params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                } else {
                    LOG_INFO("Background image: {} [{}x{}]",
                             lfs::core::path_to_utf8(params.optimization.bg_image_path),
                             bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                }
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load background image: {}", e.what());
                params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
            }
        }

        if (!bg_mode_is_image && (bg_image_base_.is_valid() || !bg_image_cache_.empty())) {
            bg_image_cache_.clear();
            bg_image_base_ = {};
        }

        // Update background color tensor if changed
        const auto& bg_color = params.optimization.bg_color;
        if (background_.is_valid()) {
            auto bg_cpu = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
            auto* bg_ptr = bg_cpu.ptr<float>();
            bg_ptr[0] = bg_color[0];
            bg_ptr[1] = bg_color[1];
            bg_ptr[2] = bg_color[2];
            background_ = bg_cpu.to(lfs::core::Device::CUDA);
        }
    }

    void Trainer::handle_control_requests(int iter, std::stop_token stop_token) {
        // Check stop token first
        if (stop_token.stop_requested()) {
            stop_requested_ = true;
            return;
        }

        // Handle pause/resume
        if (pause_requested_.load() && !is_paused_.load()) {
            is_paused_ = true;
            if (progress_) {
                progress_->pause();
            }
            LOG_INFO("Training paused at iteration {}", iter);
            LOG_DEBUG("Click 'Resume Training' to continue.");
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            if (progress_) {
                progress_->resume(iter, current_loss_.load(), static_cast<int>(strategy_->get_model().size()));
            }
            LOG_INFO("Training resumed at iteration {}", iter);
        }

        if (save_requested_.exchange(false)) {
            LOG_INFO("Saving checkpoint and PLY at iteration {}...", iter);
            save_ply(params_.dataset.output_path, iter, /*join=*/false);
            auto result = save_checkpoint(iter);
            if (result) {
                auto checkpoint_path = params_.dataset.output_path / "checkpoints" /
                                       std::format("checkpoint_{}.resume", iter);
                LOG_INFO("Checkpoint and PLY saved to {}", lfs::core::path_to_utf8(params_.dataset.output_path));
            } else {
                LOG_ERROR("Failed to save checkpoint: {}", result.error());
            }
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            LOG_INFO("Stopping training permanently at iteration {}...", iter);
            LOG_DEBUG("Saving final model...");
            save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    inline float inv_weight_piecewise(int step, int max_steps) {
        // Phases by fraction of training
        const float phase = std::max(0.f, std::min(1.f, step / float(std::max(1, max_steps))));

        const float limit_hi = 1.0f / 4.0f;  // start limit
        const float limit_mid = 2.0f / 4.0f; // middle limit
        const float limit_lo = 3.0f / 4.0f;  // final limit

        const float weight_hi = 1.0f;  // start weight
        const float weight_mid = 0.5f; // middle weight
        const float weight_lo = 0.0f;  // final weight

        if (phase < limit_hi) {
            return weight_hi; // hold until bypasses the start limit
        } else if (phase < limit_mid) {
            const float t = (phase - limit_hi) / (limit_mid - limit_hi);
            return weight_hi + (weight_mid - weight_hi) * t; // decay to mid value
        } else {
            const float t = (phase - limit_mid) / (limit_lo - limit_mid);
            return weight_mid + (weight_lo - weight_mid) * t; // decay to final value
        }
    }

    namespace {
        constexpr float TWO_PI = static_cast<float>(M_PI * 2.0);
        constexpr float PHASE_OFFSET_G = TWO_PI / 3.0f;
        constexpr float PHASE_OFFSET_B = TWO_PI * 2.0f / 3.0f;
        constexpr float CLAMP_EPS = 1e-4f;
        constexpr int BG_PERIOD_R = 37;
        constexpr int BG_PERIOD_G = 41;
        constexpr int BG_PERIOD_B = 43;
    } // anonymous namespace

    lfs::core::Tensor& Trainer::background_for_step(int iter) {
        if (!params_.optimization.bg_modulation) {
            return background_;
        }

        const float w = inv_weight_piecewise(iter, params_.optimization.iterations);
        if (w <= 0.0f) {
            return background_;
        }

        // Sine-based RGB with prime periods for color diversity
        const float pr = TWO_PI * static_cast<float>(iter % BG_PERIOD_R) / BG_PERIOD_R;
        const float pg = TWO_PI * static_cast<float>(iter % BG_PERIOD_G) / BG_PERIOD_G;
        const float pb = TWO_PI * static_cast<float>(iter % BG_PERIOD_B) / BG_PERIOD_B;

        const float result[3] = {
            std::clamp(0.5f * (1.0f + std::sin(pr)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS),
            std::clamp(0.5f * (1.0f + std::sin(pg + PHASE_OFFSET_G)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS),
            std::clamp(0.5f * (1.0f + std::sin(pb + PHASE_OFFSET_B)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS)};

        if (bg_mix_buffer_.is_empty()) {
            bg_mix_buffer_ = lfs::core::Tensor::empty({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        }

        cudaMemcpyAsync(bg_mix_buffer_.ptr<float>(), result, sizeof(result), cudaMemcpyHostToDevice, nullptr);
        return bg_mix_buffer_;
    }

    lfs::core::Tensor Trainer::get_background_image_for_camera(int width, int height) {
        // Return empty tensor if no background image is loaded
        if (!bg_image_base_.is_valid() || bg_image_base_.is_empty()) {
            return lfs::core::Tensor();
        }

        // Check cache first - key is (height << 32) | width
        const uint64_t cache_key = (static_cast<uint64_t>(height) << 32) | static_cast<uint64_t>(width);
        auto it = bg_image_cache_.find(cache_key);
        if (it != bg_image_cache_.end()) {
            return it->second;
        }

        // Resize background image to match camera dimensions
        const int src_h = static_cast<int>(bg_image_base_.shape()[1]);
        const int src_w = static_cast<int>(bg_image_base_.shape()[2]);
        const int channels = static_cast<int>(bg_image_base_.shape()[0]);

        // If dimensions match, use the original
        if (src_w == width && src_h == height) {
            bg_image_cache_[cache_key] = bg_image_base_;
            return bg_image_base_;
        }

        // Create resized tensor
        auto resized = lfs::core::Tensor::empty(
            {static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)},
            lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);

        // Use bilinear resize kernel
        kernels::launch_bilinear_resize_chw(
            bg_image_base_.ptr<float>(),
            resized.ptr<float>(),
            channels,
            src_h, src_w,
            height, width,
            nullptr // default stream
        );

        // Cache the resized image
        bg_image_cache_[cache_key] = resized;
        LOG_DEBUG("Background image resized: {}x{} -> {}x{}", src_w, src_h, width, height);

        return resized;
    }

    lfs::core::Tensor Trainer::get_random_background_for_camera(int width, int height, int iteration) {
        const size_t required_size = 3 * static_cast<size_t>(height) * static_cast<size_t>(width);

        if (!random_bg_buffer_.is_valid() || random_bg_buffer_.numel() != required_size) {
            random_bg_buffer_ = lfs::core::Tensor::empty(
                {3, static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
        }

        kernels::launch_random_background(
            random_bg_buffer_.ptr<float>(),
            height, width,
            static_cast<uint64_t>(iteration),
            nullptr);

        return random_bg_buffer_;
    }

    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        lfs::core::Camera* cam,
        lfs::core::Tensor gt_image,
        RenderMode render_mode,
        std::stop_token stop_token) {
        try {
            if (params_.optimization.gut) {
                if (cam->camera_model_type() == core::CameraModelType::ORTHO) {
                    return std::unexpected("Training on cameras with ortho model is not supported yet.");
                }
            } else if (!params_.optimization.undistort || !cam->is_undistort_prepared()) {
                if (cam->radial_distortion().numel() != 0 ||
                    cam->tangential_distortion().numel() != 0) {
                    return std::unexpected("Distorted images detected. Use --gut or --undistort to train on cameras with distortion.");
                }
                if (cam->camera_model_type() != core::CameraModelType::PINHOLE) {
                    return std::unexpected("Use --gut or --undistort to train on cameras with non-pinhole model.");
                }
            }

            current_iteration_ = iter;

            // Check control requests at the beginning
            handle_control_requests(iter, stop_token);

            if (on_iteration_start_)
                on_iteration_start_();

            // Python hook: iteration start (safe, pre-forward)
            {
                lfs::training::HookContext ctx{
                    .iteration = iter,
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::IterationStart);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::IterationStart);
                lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::IterationStart, ctx);
                auto view = lfs::training::CommandCenter::instance().snapshot();
                lfs::training::CommandCenter::instance().drain_enqueued(view);
            }

            // Training step entering forward/backward/optimizer region (commands blocked)
            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::Forward);

            // If stop requested, return Stop
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            // If paused, wait
            while (is_paused_.load() && !stop_requested_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                handle_control_requests(iter, stop_token);
            }

            // Check stop again after potential pause
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            nvtxRangePush("background_for_step");
            lfs::core::Tensor& bg = background_for_step(iter);
            nvtxRangePop();

            lfs::core::Tensor bg_image;
            if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Image) {
                bg_image = get_background_image_for_camera(cam->image_width(), cam->image_height());
            } else if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Random) {
                bg_image = get_random_background_for_camera(cam->image_width(), cam->image_height(), iter);
            }

            // Configurable tile-based training to reduce peak memory
            const int full_width = cam->image_width();
            const int full_height = cam->image_height();

            // Read tile mode from parameters (1=1 tile, 2=2 tiles, 4=4 tiles)
            const TileMode tile_mode = static_cast<TileMode>(params_.optimization.tile_mode);

            // Determine tile configuration
            int tile_rows = 1, tile_cols = 1;
            switch (tile_mode) {
            case TileMode::One:
                tile_rows = 1;
                tile_cols = 1;
                break;
            case TileMode::Two:
                tile_rows = 2;
                tile_cols = 1;
                break;
            case TileMode::Four:
                tile_rows = 2;
                tile_cols = 2;
                break;
            }

            const int tile_width = full_width / tile_cols;
            const int tile_height = full_height / tile_rows;
            const int num_tiles = tile_rows * tile_cols;

            core::Tensor loss_tensor_gpu = core::Tensor::zeros({1}, core::Device::CUDA);
            RenderOutput r_output;
            int tiles_processed = 0;

            // Determine controller phase before tile loop (does not depend on tile results)
            const bool known_ppisp_camera = ppisp_ && ppisp_->is_known_camera(cam->camera_id());
            const int ppisp_cam_idx = known_ppisp_camera ? ppisp_->camera_index(cam->camera_id()) : -1;
            const bool in_controller_phase = ppisp_controller_pool_ && known_ppisp_camera &&
                                             params_.optimization.ppisp_use_controller &&
                                             params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                             iter >= params_.optimization.ppisp_controller_activation_step &&
                                             ppisp_cam_idx >= 0 &&
                                             ppisp_cam_idx < ppisp_controller_pool_->num_cameras();

            // Loop over tiles (row-major order)
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
                const int tile_row = tile_idx / tile_cols;
                const int tile_col = tile_idx % tile_cols;
                const int tile_x_offset = tile_col * tile_width;
                const int tile_y_offset = tile_row * tile_height;

                nvtxRangePush(std::format("tile_{}x{}", tile_row, tile_col).c_str());

                // Extract GT image tile
                lfs::core::Tensor gt_tile;
                if (num_tiles == 1) {
                    // No tiling - use full image
                    gt_tile = gt_image;
                } else if (gt_image.shape()[0] == 3) {
                    // CHW layout: gt_image is [3, H, W]
                    // Slice both height and width dimensions
                    auto tile_h = gt_image.slice(1, tile_y_offset, tile_y_offset + tile_height);
                    gt_tile = tile_h.slice(2, tile_x_offset, tile_x_offset + tile_width);
                } else {
                    // HWC layout: gt_image is [H, W, 3]
                    auto tile_h = gt_image.slice(0, tile_y_offset, tile_y_offset + tile_height);
                    gt_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                }

                // Extract background image tile (if using background image)
                lfs::core::Tensor bg_tile;
                if (bg_image.is_valid() && !bg_image.is_empty()) {
                    if (num_tiles == 1) {
                        // No tiling - use full image
                        bg_tile = bg_image;
                    } else {
                        // CHW layout: bg_image is [3, H, W]
                        // Slice both height and width dimensions
                        auto tile_h = bg_image.slice(1, tile_y_offset, tile_y_offset + tile_height);
                        bg_tile = tile_h.slice(2, tile_x_offset, tile_x_offset + tile_width);
                    }
                }

                // Render the tile
                nvtxRangePush("rasterize_forward");

                // Storage for render output (used by both paths)
                RenderOutput output;
                std::optional<FastRasterizeContext> fast_ctx;
                std::optional<GsplatRasterizeContext> gsplat_ctx;

                if (params_.optimization.gut) {
                    const int tw = (num_tiles > 1) ? tile_width : 0;
                    const int th = (num_tiles > 1) ? tile_height : 0;
                    auto rasterize_result = gsplat_rasterize_forward(
                        *cam, strategy_->get_model(), bg,
                        tile_x_offset, tile_y_offset, tw, th,
                        1.0f, false, GsplatRenderMode::RGB, true, bg_tile);

                    if (!rasterize_result) {
                        nvtxRangePop(); // rasterize_forward
                        nvtxRangePop(); // tile
                        return std::unexpected(rasterize_result.error());
                    }

                    output = std::move(rasterize_result->first);
                    gsplat_ctx.emplace(std::move(rasterize_result->second));
                } else {
                    // Standard mode: use fast rasterizer with tiling support
                    auto rasterize_result = fast_rasterize_forward(
                        *cam, strategy_->get_model(), bg,
                        tile_x_offset, tile_y_offset,
                        (num_tiles > 1) ? tile_width : 0, // 0 means full image
                        (num_tiles > 1) ? tile_height : 0,
                        params_.optimization.mip_filter, bg_tile);

                    // Check for OOM error
                    if (!rasterize_result) {
                        const std::string& error = rasterize_result.error();
                        if (error.find("OUT_OF_MEMORY") != std::string::npos) {
                            nvtxRangePop(); // rasterize_forward
                            nvtxRangePop(); // tile

                            // Handle OOM by switching tile mode
                            if (tile_mode == TileMode::Four) {
                                // Already at maximum tiling - can't tile further, return error
                                LOG_ERROR("OUT OF MEMORY at maximum tile mode (2x2). Cannot continue training.");
                                LOG_ERROR("Arena error: {}", error);
                                return std::unexpected(error);
                            } else {
                                // Upgrade to next tile mode
                                TileMode new_mode = (tile_mode == TileMode::One) ? TileMode::Two : TileMode::Four;
                                LOG_WARN("OUT OF MEMORY detected. Switching tile mode from {} to {}",
                                         static_cast<int>(tile_mode), static_cast<int>(new_mode));
                                LOG_WARN("Arena error: {}", error);
                                params_.optimization.tile_mode = static_cast<int>(new_mode);

                                // Retry this step with new tile mode
                                return std::unexpected("OOM_RETRY"); // Signal to retry the step
                            }
                        } else {
                            // Non-OOM error - propagate
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(error);
                        }
                    }

                    output = std::move(rasterize_result->first);
                    fast_ctx.emplace(std::move(rasterize_result->second));

                    if (fast_ctx->forward_ctx.n_visible_primitives == 0) {
                        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
                        arena.end_frame(fast_ctx->forward_ctx.frame_id);
                        nvtxRangePop();
                        nvtxRangePop();
                        continue;
                    }
                }

                r_output = output; // Save last tile for densification
                nvtxRangePop();

                if (in_controller_phase) {
                    // Controller phase: forward through ISP with controller params, photometric loss,
                    // backward only through controller (base params frozen)
                    nvtxRangePush("controller_phase");
                    auto cleanup_controller_tile_context = [&]() {
                        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
                        if (fast_ctx) {
                            arena.end_frame(fast_ctx->forward_ctx.frame_id);
                        } else if (gsplat_ctx) {
                            if (gsplat_ctx->isect_ids_ptr != nullptr) {
                                cudaFree(gsplat_ctx->isect_ids_ptr);
                                gsplat_ctx->isect_ids_ptr = nullptr;
                            }
                            if (gsplat_ctx->flatten_ids_ptr != nullptr) {
                                cudaFree(gsplat_ctx->flatten_ids_ptr);
                                gsplat_ctx->flatten_ids_ptr = nullptr;
                            }
                            arena.end_frame(gsplat_ctx->frame_id);
                        }
                    };

                    lfs::core::Tensor corrected_image = output.image;
                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                    }
                    auto ppisp_input = corrected_image;

                    auto pred = ppisp_controller_pool_->predict(ppisp_cam_idx, corrected_image.unsqueeze(0), 1.0f);
                    corrected_image = ppisp_->apply_with_controller_params(corrected_image, pred, ppisp_cam_idx);

                    // Photometric loss
                    nvtxRangePush("compute_photometric_loss");
                    lfs::core::Tensor tile_loss;
                    lfs::core::Tensor tile_grad;

                    const bool use_mask = params_.optimization.mask_mode != lfs::core::param::MaskMode::None &&
                                          (cam->has_mask() || (params_.optimization.use_alpha_as_mask && scene_ && scene_->imagesHaveAlpha()));
                    if (use_mask) {
                        lfs::core::Tensor mask;
                        if (pipelined_mask_.is_valid() && pipelined_mask_.numel() > 0) {
                            mask = pipelined_mask_;
                        } else {
                            mask = cam->load_and_get_mask(
                                params_.dataset.resize_factor,
                                params_.dataset.max_width,
                                params_.optimization.invert_masks,
                                params_.optimization.mask_threshold);
                        }

                        lfs::core::Tensor mask_tile = mask;
                        if (num_tiles > 1 && mask.ndim() == 2) {
                            auto tile_h = mask.slice(0, tile_y_offset, tile_y_offset + tile_height);
                            mask_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                        }

                        auto result = compute_photometric_loss_with_mask(
                            corrected_image, gt_tile, mask_tile, output.alpha, params_.optimization);
                        if (!result) {
                            cleanup_controller_tile_context();
                            nvtxRangePop();
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->loss;
                        tile_grad = result->grad_image;
                    } else {
                        auto result = compute_photometric_loss_with_gradient(
                            corrected_image, gt_tile, params_.optimization);
                        if (!result) {
                            cleanup_controller_tile_context();
                            nvtxRangePop();
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->first;
                        tile_grad = result->second;
                    }

                    loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                    tiles_processed++;
                    nvtxRangePop(); // compute_photometric_loss

                    // ISP backward for controller params
                    auto ctrl_grad = ppisp_->backward_with_controller_params(ppisp_input, tile_grad, pred, ppisp_cam_idx);
                    ppisp_controller_pool_->backward(ppisp_cam_idx, ctrl_grad);

                    // End arena frame explicitly (normally done inside rasterize_backward which we skip)
                    cleanup_controller_tile_context();

                    nvtxRangePop(); // controller_phase
                } else {
                    // Normal phase: full forward + backward through all components
                    lfs::core::Tensor corrected_image = output.image;
                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        nvtxRangePush("bilateral_grid_forward");
                        corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                        nvtxRangePop();
                    }

                    if (ppisp_ && params_.optimization.use_ppisp) {
                        nvtxRangePush("ppisp_forward");
                        corrected_image = ppisp_->apply(corrected_image, cam->camera_id(), cam->uid());
                        nvtxRangePop();
                    }

                    nvtxRangePush("compute_photometric_loss");
                    lfs::core::Tensor tile_loss;
                    lfs::core::Tensor tile_grad;
                    lfs::core::Tensor tile_grad_alpha;

                    const bool use_mask = params_.optimization.mask_mode != lfs::core::param::MaskMode::None &&
                                          (cam->has_mask() || (params_.optimization.use_alpha_as_mask && scene_ && scene_->imagesHaveAlpha()));
                    if (use_mask) {
                        lfs::core::Tensor mask;
                        if (pipelined_mask_.is_valid() && pipelined_mask_.numel() > 0) {
                            mask = pipelined_mask_;
                        } else {
                            mask = cam->load_and_get_mask(
                                params_.dataset.resize_factor,
                                params_.dataset.max_width,
                                params_.optimization.invert_masks,
                                params_.optimization.mask_threshold);
                        }

                        lfs::core::Tensor mask_tile = mask;
                        if (num_tiles > 1 && mask.ndim() == 2) {
                            auto tile_h = mask.slice(0, tile_y_offset, tile_y_offset + tile_height);
                            mask_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                        }

                        auto result = compute_photometric_loss_with_mask(
                            corrected_image, gt_tile, mask_tile, output.alpha, params_.optimization);
                        if (!result) {
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->loss;
                        tile_grad = result->grad_image;
                        tile_grad_alpha = result->grad_alpha;
                    } else {
                        auto result = compute_photometric_loss_with_gradient(
                            corrected_image, gt_tile, params_.optimization);
                        if (!result) {
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->first;
                        tile_grad = result->second;
                    }

                    loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                    tiles_processed++;
                    nvtxRangePop();

                    lfs::core::Tensor raster_grad = tile_grad;
                    if (ppisp_ && params_.optimization.use_ppisp) {
                        nvtxRangePush("ppisp_backward");
                        lfs::core::Tensor ppisp_input = output.image;
                        if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                            ppisp_input = bilateral_grid_->apply(output.image, cam->uid());
                        }
                        raster_grad = ppisp_->backward(ppisp_input, raster_grad, cam->camera_id(), cam->uid());
                        nvtxRangePop();
                    }

                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        nvtxRangePush("bilateral_grid_backward");
                        raster_grad = bilateral_grid_->backward(output.image, raster_grad, cam->uid());
                        nvtxRangePop();
                    }

                    nvtxRangePush("rasterize_backward");
                    if (gsplat_ctx) {
                        auto grad_alpha = tile_grad_alpha.is_valid()
                                              ? tile_grad_alpha
                                              : lfs::core::Tensor::zeros_like(output.alpha);
                        gsplat_rasterize_backward(*gsplat_ctx, raster_grad, grad_alpha,
                                                  strategy_->get_model(), strategy_->get_optimizer());
                    } else {
                        fast_rasterize_backward(*fast_ctx, raster_grad, strategy_->get_model(),
                                                strategy_->get_optimizer(), tile_grad_alpha);
                    }
                    nvtxRangePop();
                }

                nvtxRangePop(); // End tile
            }

            if (tiles_processed > 1)
                loss_tensor_gpu = loss_tensor_gpu / static_cast<float>(tiles_processed);

            if (tiles_processed == 0) {
                LOG_DEBUG("Skipping iteration {} - no visible primitives", iter);
                return iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()
                           ? StepResult::Continue
                           : StepResult::Stop;
            }

            if (in_controller_phase) {
                // Controller phase: only update controller weights
                nvtxRangePush("controller_optimizer_step");
                ppisp_controller_pool_->optimizer_step(ppisp_cam_idx);
                ppisp_controller_pool_->zero_grad();
                ppisp_controller_pool_->scheduler_step(ppisp_cam_idx);
                nvtxRangePop();
            } else {
                // Normal phase: regularization losses + optimizer steps for all components

                if (params_.optimization.scale_reg > 0.0f) {
                    nvtxRangePush("compute_scale_reg_loss");
                    auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                    if (!scale_loss_result) {
                        return std::unexpected(scale_loss_result.error());
                    }
                    loss_tensor_gpu = loss_tensor_gpu + *scale_loss_result;
                    nvtxRangePop();
                }

                if (params_.optimization.opacity_reg > 0.0f) {
                    nvtxRangePush("compute_opacity_reg_loss");
                    auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                    if (!opacity_loss_result) {
                        return std::unexpected(opacity_loss_result.error());
                    }
                    loss_tensor_gpu = loss_tensor_gpu + *opacity_loss_result;
                    nvtxRangePop();
                }

                if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                    nvtxRangePush("bilateral_grid_tv_and_step");
                    const float tv_weight = params_.optimization.tv_loss_weight;

                    loss_tensor_gpu = loss_tensor_gpu + bilateral_grid_->tv_loss_gpu() * tv_weight;
                    bilateral_grid_->tv_backward(tv_weight);
                    bilateral_grid_->optimizer_step();
                    bilateral_grid_->zero_grad();
                    bilateral_grid_->scheduler_step();

                    nvtxRangePop();
                }

                if (ppisp_ && params_.optimization.use_ppisp) {
                    nvtxRangePush("ppisp_reg_and_step");

                    loss_tensor_gpu = loss_tensor_gpu + ppisp_->reg_loss_gpu();
                    ppisp_->reg_backward();
                    ppisp_->optimizer_step();
                    ppisp_->zero_grad();
                    ppisp_->scheduler_step();

                    nvtxRangePop();
                }
            }

            // Sparsity loss - ALL ON GPU, no CPU sync here
            lfs::core::Tensor sparsity_loss_gpu;
            if (sparsity_optimizer_ && sparsity_optimizer_->should_apply_loss(iter)) {
                nvtxRangePush("sparsity_loss");
                auto sparsity_result = compute_sparsity_loss_forward(iter, strategy_->get_model());
                if (!sparsity_result) {
                    nvtxRangePop();
                    return std::unexpected(sparsity_result.error());
                }
                auto& [loss_tensor, ctx] = *sparsity_result;
                sparsity_loss_gpu = std::move(loss_tensor);

                if (ctx.n > 0) {
                    if (auto result = sparsity_optimizer_->compute_loss_backward(
                            ctx, 1.0f, strategy_->get_optimizer().get_grad(ParamType::Opacity));
                        !result) {
                        nvtxRangePop();
                        return std::unexpected(result.error());
                    }
                }
                nvtxRangePop();
            }

            // Sparsification phase logging (once per phase transition)
            if (params_.optimization.enable_sparsity) {
                const int base_iterations = params_.optimization.iterations - params_.optimization.sparsify_steps;
                if (iter == base_iterations + 1) {
                    LOG_INFO("Entering sparsification: {} Gaussians, target prune={}%",
                             strategy_->get_model().size(), params_.optimization.prune_ratio * 100);
                }
            }

            // Sync loss to CPU only at intervals - single sync point
            constexpr int LOSS_SYNC_INTERVAL = 10;
            float loss_value = 0.0f;
            if (iter % LOSS_SYNC_INTERVAL == 0 || iter == 1) {
                // Accumulate on GPU then sync once
                auto total_loss = sparsity_loss_gpu.numel() > 0
                                      ? (loss_tensor_gpu + sparsity_loss_gpu)
                                      : loss_tensor_gpu;
                loss_value = total_loss.item<float>();

                if (std::isnan(loss_value) || std::isinf(loss_value)) {
                    return std::unexpected(std::format("NaN/Inf loss at iteration {}", iter));
                }

                current_loss_ = loss_value;
                if (progress_) {
                    progress_->update(iter, loss_value,
                                      static_cast<int>(strategy_->get_model().size()),
                                      strategy_->is_refining(iter));
                }
                lfs::core::events::state::TrainingProgress{
                    .iteration = iter,
                    .loss = loss_value,
                    .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                    .is_refining = strategy_->is_refining(iter)}
                    .emit();
            }

            {
                DeferredEvents deferred;
                {
                    std::unique_lock<std::shared_mutex> lock(render_mutex_);

                    // Python hook: pre-optimizer-step (post-backward, pre-step)
                    {
                        lfs::training::HookContext ctx{
                            .iteration = iter,
                            .loss = current_loss_.load(),
                            .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                            .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
                            .trainer = this};
                        lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::OptimizerStep);
                        lfs::training::CommandCenter::instance().update_snapshot(
                            ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                            lfs::training::TrainingPhase::OptimizerStep);
                        lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::PreOptimizerStep, ctx);
                    }

                    // Skip post_backward during sparsification phase
                    const bool in_sparsification = params_.optimization.enable_sparsity &&
                                                   iter > (params_.optimization.iterations - params_.optimization.sparsify_steps);
                    if (!in_sparsification) {
                        strategy_->post_backward(iter, r_output);
                    }

                    // Skip strategy step if we're in controller distillation phase and freeze is enabled
                    const bool freeze_gaussians = ppisp_controller_pool_ &&
                                                  params_.optimization.ppisp_use_controller &&
                                                  params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                                  iter >= params_.optimization.ppisp_controller_activation_step;
                    if (!freeze_gaussians) {
                        strategy_->step(iter);
                    }
                }

                if (auto result = handle_sparsity_update(iter, strategy_->get_model()); !result) {
                    LOG_ERROR("Sparsity update: {}", result.error());
                }
                if (auto result = apply_sparsity_pruning(iter, strategy_->get_model()); !result) {
                    LOG_ERROR("Sparsity pruning: {}", result.error());
                }

                // Clean evaluation - let the evaluator handle everything
                if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                    evaluator_->print_evaluation_header(iter);
                    auto metrics = evaluator_->evaluate(iter,
                                                        strategy_->get_model(),
                                                        val_dataset_,
                                                        background_);
                    LOG_INFO("{}", metrics.to_string());
                }

                // Save checkpoint (not PLY) at specified steps
                for (size_t save_step : params_.optimization.save_steps) {
                    if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                        auto result = save_checkpoint(iter);
                        if (!result) {
                            LOG_WARN("Failed to save checkpoint at iteration {}: {}", iter, result.error());
                        }
                    }
                }

                if (!params_.dataset.timelapse_images.empty() && iter % params_.dataset.timelapse_every == 0) {
                    for (const auto& img_name : params_.dataset.timelapse_images) {
                        auto train_cam = train_dataset_->get_camera_by_filename(img_name);
                        auto val_cam = val_dataset_ ? val_dataset_->get_camera_by_filename(img_name) : std::nullopt;
                        if (train_cam.has_value() || val_cam.has_value()) {
                            lfs::core::Camera* cam_to_use = train_cam.has_value() ? train_cam.value() : val_cam.value();

                            // Image size isn't correct until the image has been loaded once
                            // If we use the camera before it's loaded, it will render images at the non-scaled size
                            if ((cam_to_use->camera_height() == cam_to_use->image_height() && params_.dataset.resize_factor != 1) ||
                                cam_to_use->image_height() > params_.dataset.max_width ||
                                cam_to_use->image_width() > params_.dataset.max_width) {
                                cam_to_use->load_image_size(params_.dataset.resize_factor, params_.dataset.max_width);
                            }

                            RenderOutput rendered_timelapse_output;
                            if (params_.optimization.gut) {
                                rendered_timelapse_output = gsplat_rasterize(*cam_to_use, strategy_->get_model(), background_,
                                                                             1.0f, false, GsplatRenderMode::RGB, true);
                            } else {
                                rendered_timelapse_output = fast_rasterize(*cam_to_use, strategy_->get_model(), background_);
                            }

                            // Get folder name to save in by stripping file extension
                            std::string folder_name = lfs::io::strip_extension(img_name);

                            auto output_path = params_.dataset.output_path / "timelapse" / folder_name;
                            std::filesystem::create_directories(output_path);

                            lfs::core::image_io::save_image_async(output_path / std::format("{:06d}.jpg", iter),
                                                                  rendered_timelapse_output.image);
                        } else {
                            LOG_WARN("Timelapse image '{}' not found in dataset.", img_name);
                        }
                    }
                }
            }

            // Python hook: post-step (after optimizer and side-effects)
            {
                lfs::training::HookContext ctx{
                    .iteration = iter,
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::PostStep, ctx);
            }

            // Return Continue if we should continue training
            if (iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()) {
                return StepResult::Continue;
            } else {
                return StepResult::Stop;
            }
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Training step failed: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::train(std::stop_token stop_token) {
        // Check if initialized
        if (!initialized_.load()) {
            return std::unexpected("Trainer not initialized. Call initialize() before train()");
        }

        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false; // Reset the flag
        lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);

        ready_to_start_ = true; // Skip GUI wait for now

        is_running_ = true; // Now we can start
        LOG_INFO("Starting training loop");
        // initializing image loader
        auto& cache_loader = lfs::io::CacheLoader::getInstance(params_.dataset.loading_params.use_cpu_memory, params_.dataset.loading_params.use_fs_cache);
        cache_loader.reset_cache();
        // in case we call getInstance multiple times and cache parameters/dataset were changed by user
        cache_loader.update_cache_params(params_.dataset.loading_params.use_cpu_memory,
                                         params_.dataset.loading_params.use_fs_cache,
                                         train_dataset_size_,
                                         params_.dataset.loading_params.min_cpu_free_GB,
                                         params_.dataset.loading_params.min_cpu_free_memory_ratio,
                                         params_.dataset.loading_params.print_cache_status,
                                         params_.dataset.loading_params.print_status_freq_num);

        // Notify Python control layer that training is starting
        {
            lfs::training::HookContext ctx{
                .iteration = 0,
                .loss = current_loss_.load(),
                .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                .is_refining = strategy_ ? strategy_->is_refining(0) : false,
                .trainer = this};
            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
            lfs::training::CommandCenter::instance().update_snapshot(
                ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                lfs::training::TrainingPhase::SafeControl);
            lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::TrainingStart, ctx);
        }

        try {
            // Start from current_iteration_ (allows resume from checkpoint)
            int iter = current_iteration_.load() > 0 ? current_iteration_.load() + 1 : 1;
            const RenderMode render_mode = RenderMode::RGB;

            if (progress_) {
                progress_->update(iter, current_loss_.load(),
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Conservative prefetch to avoid VRAM exhaustion
            lfs::io::PipelinedLoaderConfig pipelined_config;
            pipelined_config.jpeg_batch_size = 8;
            pipelined_config.prefetch_count = 8;
            pipelined_config.output_queue_size = 4;
            pipelined_config.io_threads = 2;

            // Non-JPEG images (PNG, WebP) need CPU decoding - use more threads until cache warms
            constexpr float NON_JPEG_THRESHOLD = 0.1f;
            constexpr size_t MIN_COLD_THREADS = 4;
            constexpr size_t COLD_PREFETCH_COUNT = 16;
            const float non_jpeg_ratio = train_dataset_->get_non_jpeg_ratio();
            if (non_jpeg_ratio > NON_JPEG_THRESHOLD) {
                const size_t cold_threads = std::max(MIN_COLD_THREADS,
                                                     static_cast<size_t>(std::thread::hardware_concurrency() / 2));
                pipelined_config.cold_process_threads = cold_threads;
                pipelined_config.prefetch_count = COLD_PREFETCH_COUNT;
                LOG_INFO("{:.0f}% non-JPEG images, using {} cold threads", non_jpeg_ratio * 100.0f, cold_threads);
            }

            const bool alpha_available = scene_ && scene_->imagesHaveAlpha();
            PipelinedMaskConfig mask_pipeline_config;
            if (params_.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                mask_pipeline_config.invert_masks = params_.optimization.invert_masks;
                mask_pipeline_config.mask_threshold = params_.optimization.mask_threshold;
                if (params_.optimization.use_alpha_as_mask && alpha_available) {
                    mask_pipeline_config.use_alpha_as_mask = true;
                    LOG_INFO("Alpha-as-mask enabled (invert={}, threshold={})",
                             mask_pipeline_config.invert_masks, mask_pipeline_config.mask_threshold);
                } else {
                    mask_pipeline_config.load_masks = true;
                    LOG_INFO("Mask file loading enabled (invert={}, threshold={})",
                             mask_pipeline_config.invert_masks, mask_pipeline_config.mask_threshold);
                }
            }

            auto train_dataloader = create_infinite_pipelined_dataloader(
                train_dataset_, pipelined_config, mask_pipeline_config);

            LOG_DEBUG("Starting training iterations");
            while (iter <= params_.optimization.iterations) {
                lfs::core::CudaMemoryPool::instance().set_iteration(iter);

                if (stop_token.stop_requested() || stop_requested_.load())
                    break;
                if (callback_busy_.load())
                    cudaStreamSynchronize(callback_stream_);

                lfs::core::Camera* cam = nullptr;
                lfs::core::Tensor gt_image;
                auto example_opt = train_dataloader->next();
                if (!example_opt) {
                    LOG_ERROR("DataLoader returned nullopt unexpectedly");
                    break;
                }
                auto& example = *example_opt;
                cam = example.data.camera;
                gt_image = std::move(example.data.image);

                // Store pipelined mask for use in train_step
                pipelined_mask_ = example.mask.has_value() ? std::move(*example.mask) : lfs::core::Tensor();

                auto step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                if (!step_result) {
                    // Check if this is an OOM_RETRY signal
                    if (step_result.error() == "OOM_RETRY") {
                        cudaDeviceSynchronize();
                        cudaGetLastError();

                        lfs::core::GlobalArenaManager::instance().get_arena().full_reset();
                        lfs::core::CudaMemoryPool::instance().trim_cached_memory();

                        cudaDeviceSynchronize();
                        cudaGetLastError();

                        LOG_INFO("OOM recovery: retrying iteration {}", iter);
                        step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                        if (!step_result) {
                            // If retry also failed, propagate the error
                            return std::unexpected(step_result.error());
                        }
                    } else {
                        // Regular error - propagate
                        return std::unexpected(step_result.error());
                    }
                }

                // Transition to safe control phase and execute deferred Python callbacks
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().drain_callbacks();

                if (*step_result == StepResult::Stop) {
                    break;
                }

                // Launch callback for async progress update (except first iteration)
                if (iter > 1 && callback_) {
                    callback_busy_ = true;
                    auto err = cudaLaunchHostFunc(
                        callback_stream_,
                        [](void* self) {
                            auto* trainer = static_cast<Trainer*>(self);
                            if (trainer->callback_) {
                                trainer->callback_();
                            }
                            trainer->callback_busy_ = false;
                        },
                        this);
                    if (err != cudaSuccess) {
                        LOG_WARN("Failed to launch callback: {}", cudaGetErrorString(err));
                        callback_busy_ = false;
                    }
                }

                ++iter;
            }

            // Ensure callback is finished before final save
            if (callback_busy_.load()) {
                cudaStreamSynchronize(callback_stream_);
            }

            // Final save if not already saved by stop request
            if (!stop_requested_.load() && !stop_token.stop_requested()) {
                auto final_path = params_.dataset.output_path;
                save_ply(final_path, params_.optimization.iterations, /*join=*/true);
            }

            if (progress_) {
                progress_->complete();
            }
            evaluator_->save_report();
            if (progress_) {
                progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
            }

            is_running_ = false;
            training_complete_ = true;

            cache_loader.clear_cpu_cache();
            lfs::core::image_io::wait_for_pending_saves();

            // Notify training end
            {
                lfs::training::HookContext ctx{
                    .iteration = current_iteration_.load(),
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(current_iteration_.load()) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::TrainingEnd, ctx);
            }

            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::Idle);

            LOG_INFO("Training completed successfully");
            return {};
        } catch (const std::exception& e) {
            is_running_ = false;
            cache_loader.clear_cpu_cache();
            lfs::core::image_io::wait_for_pending_saves();
            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::Idle);

            return std::unexpected(std::format("Training failed: {}", e.what()));
        }
    }

    void Trainer::save_ply(const std::filesystem::path& save_path, const int iter_num, const bool join_threads) {
        const lfs::io::PlySaveOptions ply_options{
            .output_path = save_path / ("splat_" + std::to_string(iter_num) + ".ply"),
            .binary = true,
            .async = !join_threads};

        const auto ply_result = lfs::io::save_ply(strategy_->get_model(), ply_options);
        if (!ply_result) {
            if (ply_result.error().code == lfs::io::ErrorCode::INSUFFICIENT_DISK_SPACE) {
                lfs::core::events::state::DiskSpaceSaveFailed{
                    .iteration = iter_num,
                    .path = ply_options.output_path,
                    .error = ply_result.error().message,
                    .required_bytes = ply_result.error().required_bytes,
                    .available_bytes = ply_result.error().available_bytes,
                    .is_disk_space_error = true,
                    .is_checkpoint = false}
                    .emit();
            }
            LOG_WARN("Failed to save PLY: {}", ply_result.error().message);
            return; // Don't save checkpoint if PLY failed
        }

        // Only save controller if training has reached activation step
        PPISPControllerPool* controller_to_save = nullptr;
        if (ppisp_controller_pool_ && iter_num >= params_.optimization.ppisp_controller_activation_step) {
            controller_to_save = ppisp_controller_pool_.get();
        }

        // Save checkpoint alongside PLY for training resumption
        auto ckpt_result = lfs::training::save_checkpoint(save_path, iter_num, *strategy_, params_,
                                                          bilateral_grid_.get(), ppisp_.get(), controller_to_save);
        if (!ckpt_result) {
            LOG_WARN("Failed to save checkpoint: {}", ckpt_result.error());
        }

        if (ppisp_) {
            const auto ppisp_path = get_ppisp_companion_path(ply_options.output_path);
            const auto ppisp_result = save_ppisp_file(ppisp_path, *ppisp_, controller_to_save);
            if (!ppisp_result) {
                LOG_WARN("Failed to save PPISP file: {}", ppisp_result.error());
            }
        }

        LOG_DEBUG("PLY save initiated: {} (sync={})", lfs::core::path_to_utf8(save_path), join_threads);
    }

    std::expected<void, std::string> Trainer::save_checkpoint(int iteration) {
        if (!strategy_) {
            return std::unexpected("Cannot save checkpoint: no strategy initialized");
        }

        // Only save controller if training has reached activation step
        PPISPControllerPool* controller_to_save = nullptr;
        if (ppisp_controller_pool_ && iteration >= params_.optimization.ppisp_controller_activation_step) {
            controller_to_save = ppisp_controller_pool_.get();
        }

        return lfs::training::save_checkpoint(params_.dataset.output_path, iteration, *strategy_, params_,
                                              bilateral_grid_.get(), ppisp_.get(), controller_to_save);
    }

    lfs::core::Tensor Trainer::applyPPISPForViewport(const lfs::core::Tensor& rgb, const int camera_uid,
                                                     const PPISPViewportOverrides& overrides,
                                                     const bool use_controller) const {
        if (!ppisp_ || !params_.optimization.use_ppisp || rgb.shape().rank() != 3) {
            return rgb;
        }

        const bool is_chw = (rgb.shape()[0] == 3);
        const auto rgb_chw = is_chw ? rgb : rgb.permute({2, 0, 1}).contiguous();
        const bool is_training_camera = ppisp_->is_known_frame(camera_uid);
        const bool has_controller = ppisp_controller_pool_ && params_.optimization.ppisp_use_controller;

        lfs::core::Tensor result;

        if (use_controller && has_controller) {
            constexpr int CONTROLLER_IDX = 0;
            const auto controller_params = ppisp_controller_pool_->predict(CONTROLLER_IDX, rgb_chw.unsqueeze(0), 1.0f);
            result = overrides.isIdentity()
                         ? ppisp_->apply_with_controller_params(rgb_chw, controller_params, CONTROLLER_IDX)
                         : ppisp_->apply_with_controller_params_and_overrides(rgb_chw, controller_params, CONTROLLER_IDX,
                                                                              toRenderOverrides(overrides));
        } else if (is_training_camera) {
            const int camera_id = ppisp_->camera_for_frame(camera_uid);
            result = overrides.isIdentity() ? ppisp_->apply(rgb_chw, camera_id, camera_uid)
                                            : ppisp_->apply_with_overrides(rgb_chw, camera_id, camera_uid,
                                                                           toRenderOverrides(overrides));
        } else {
            const int fallback_camera = ppisp_->any_camera_id();
            const int fallback_frame = ppisp_->any_frame_uid();
            result = overrides.isIdentity() ? ppisp_->apply(rgb_chw, fallback_camera, fallback_frame)
                                            : ppisp_->apply_with_overrides(rgb_chw, fallback_camera, fallback_frame,
                                                                           toRenderOverrides(overrides));
        }

        return is_chw ? result : result.permute({1, 2, 0}).contiguous();
    }

    void Trainer::save_final_ply_and_checkpoint(const int iteration) {
        save_ply(params_.dataset.output_path, iteration, /*join=*/true);
    }

    std::expected<int, std::string> Trainer::load_checkpoint(const std::filesystem::path& checkpoint_path) {
        if (!strategy_) {
            return std::unexpected("Cannot load checkpoint: no strategy initialized");
        }

        // Create bilateral grid before loading if needed (checkpoint may contain grid state)
        if (params_.optimization.use_bilateral_grid && !bilateral_grid_) {
            if (auto init_result = initialize_bilateral_grid(); !init_result) {
                LOG_WARN("Failed to init bilateral grid for resume: {}", init_result.error());
            }
        }

        // Create PPISP before loading if needed
        if (params_.optimization.use_ppisp && !ppisp_) {
            if (auto init_result = initialize_ppisp(); !init_result) {
                LOG_WARN("Failed to init PPISP for resume: {}", init_result.error());
            }
        }

        // Create PPISP controller pool before loading if needed
        if (params_.optimization.ppisp_use_controller && !ppisp_controller_pool_) {
            if (auto init_result = initialize_ppisp_controller(); !init_result) {
                LOG_WARN("Failed to init PPISP controller pool for resume: {}", init_result.error());
            }
        }

        auto result = lfs::training::load_checkpoint(
            checkpoint_path, *strategy_, params_, bilateral_grid_.get(), ppisp_.get(),
            ppisp_controller_pool_.get());
        if (!result) {
            return result;
        }
        current_iteration_ = *result;

        LOG_INFO("Restored training state from checkpoint at iteration {}", *result);
        return result;
    }

} // namespace lfs::training
