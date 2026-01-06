/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "trainer.hpp"
#include "components/bilateral_grid.hpp"
#include "components/sparsity_optimizer.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data_export.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "io/cache_image_loader.hpp"
#include "io/filesystem_utils.hpp"
#include "lfs/kernels/ssim.cuh"
#include "losses/losses.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/gsplat_rasterizer.hpp"
#include "strategies/adc.hpp"
#include "strategies/mcmc.hpp"
#include "visualizer/scene/scene.hpp"

#include <atomic>
#include <cmath>
#include <cuda_runtime.h>
#include <expected>
#include <memory>
#include <nvtx3/nvToolsExt.h>

namespace lfs::training {

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

        // Create reusable event for GPU-side sync (avoids CPU blocking)
        cudaEventCreateWithFlags(&img_sync_event_, cudaEventDisableTiming);

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

        // Create reusable event for GPU-side sync (avoids CPU blocking)
        cudaEventCreateWithFlags(&img_sync_event_, cudaEventDisableTiming);

        // Datasets will be created in initialize() from Scene cameras
        if (!scene.getTrainCameras()) {
            throw std::runtime_error("Scene has no train cameras");
        }

        LOG_DEBUG("Trainer constructed from Scene with {} cameras", scene.getTrainCameras()->get_cameras().size());
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

            // Get source cameras - from Scene (new mode) or base_dataset_ (legacy mode)
            std::vector<std::shared_ptr<lfs::core::Camera>> source_cameras;
            if (scene_) {
                // Scene mode: get cameras from Scene
                auto scene_dataset = scene_->getTrainCameras();
                if (!scene_dataset) {
                    return std::unexpected("Scene has no train cameras");
                }
                source_cameras = scene_dataset->get_cameras();
            } else if (base_dataset_) {
                // Legacy mode: use base_dataset_
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

                if (params.optimization.strategy == "mcmc") {
                    strategy_ = std::make_unique<MCMC>(*model);
                    LOG_DEBUG("Created MCMC strategy from Scene model");
                } else {
                    strategy_ = std::make_unique<ADC>(*model);
                    LOG_DEBUG("Created ADC strategy from Scene model");
                }
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

            // Validate masks if mask mode is enabled
            if (auto result = validate_masks(); !result) {
                return std::unexpected(result.error());
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

            // Initialize background color tensor [3] = [0, 0, 0]
            background_ = lfs::core::Tensor::zeros({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

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

        // Destroy GPU sync event
        if (img_sync_event_) {
            cudaEventDestroy(img_sync_event_);
            img_sync_event_ = nullptr;
        }

        cudaDeviceSynchronize();

        strategy_.reset();
        bilateral_grid_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();
        progress_.reset();

        // Free pinned buffer
        if (bg_rgb_pinned_ != nullptr) {
            cudaFreeHost(bg_rgb_pinned_);
            bg_rgb_pinned_ = nullptr;
        }
        train_dataset_.reset();
        val_dataset_.reset();

        // Release GPU memory pools back to system
        lfs::core::CudaMemoryPool::instance().trim_cached_memory();
        lfs::core::GlobalArenaManager::instance().get_arena().emergency_cleanup();
        cudaDeviceSynchronize();
        LOG_DEBUG("GPU memory released");

        initialized_ = false;
        is_running_ = false;
        training_complete_ = false;
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

        // Handle save request - save a real checkpoint (not just PLY)
        if (save_requested_.exchange(false)) {
            LOG_INFO("Saving checkpoint at iteration {}...", iter);
            auto result = save_checkpoint(iter);
            if (result) {
                auto checkpoint_path = params_.dataset.output_path / "checkpoints" /
                                       std::format("checkpoint_{}.resume", iter);
                LOG_INFO("Checkpoint saved to {}", lfs::core::path_to_utf8(checkpoint_path));
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

        // Lazy allocate pinned + GPU buffer
        if (bg_rgb_pinned_ == nullptr)
            cudaHostAlloc(&bg_rgb_pinned_, 3 * sizeof(float), cudaHostAllocDefault);
        if (bg_mix_buffer_.is_empty())
            bg_mix_buffer_ = lfs::core::Tensor::empty({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        // Sine-based RGB with prime periods
        const float pr = TWO_PI * static_cast<float>(iter % BG_PERIOD_R) / BG_PERIOD_R;
        const float pg = TWO_PI * static_cast<float>(iter % BG_PERIOD_G) / BG_PERIOD_G;
        const float pb = TWO_PI * static_cast<float>(iter % BG_PERIOD_B) / BG_PERIOD_B;

        bg_rgb_pinned_[0] = std::clamp(0.5f * (1.0f + std::sin(pr)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS);
        bg_rgb_pinned_[1] = std::clamp(0.5f * (1.0f + std::sin(pg + PHASE_OFFSET_G)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS);
        bg_rgb_pinned_[2] = std::clamp(0.5f * (1.0f + std::sin(pb + PHASE_OFFSET_B)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS);

        // Async copy from persistent pinned buffer
        cudaMemcpyAsync(bg_mix_buffer_.ptr<float>(), bg_rgb_pinned_, 3 * sizeof(float),
                        cudaMemcpyHostToDevice, bg_mix_buffer_.stream());
        return bg_mix_buffer_;
    }

    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        lfs::core::Camera* cam,
        lfs::core::Tensor gt_image,
        RenderMode render_mode,
        std::stop_token stop_token) {
        try {
            // GUT mode enables Gaussian Unscented Transform for lens distortion handling
            if (params_.optimization.gut) {
                if (cam->camera_model_type() == core::CameraModelType::ORTHO) {
                    return std::unexpected("Training on cameras with ortho model is not supported yet.");
                }
            } else {
                if (cam->radial_distortion().numel() != 0 ||
                    cam->tangential_distortion().numel() != 0) {
                    return std::unexpected("Distorted images detected.  You can use --gut option to train on cameras with distortion.");
                }
                if (cam->camera_model_type() != core::CameraModelType::PINHOLE) {
                    return std::unexpected("You must use --gut option to train on cameras with non-pinhole model.");
                }
            }

            current_iteration_ = iter;

            // Check control requests at the beginning
            handle_control_requests(iter, stop_token);

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

            // Accumulate loss across tiles (keep on GPU until final sync)
            lfs::core::Tensor loss_tensor_gpu;
            RenderOutput r_output; // Last tile's output (for densification info)

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
                        1.0f, false, GsplatRenderMode::RGB, true);

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
                        params_.optimization.mip_filter);

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
                }

                r_output = output; // Save last tile for densification
                nvtxRangePop();

                // Apply bilateral grid if enabled (before loss computation)
                lfs::core::Tensor corrected_image = output.image;
                if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                    nvtxRangePush("bilateral_grid_forward");
                    corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                    nvtxRangePop();
                }

                // Compute photometric loss and gradients for this tile
                nvtxRangePush("compute_photometric_loss");
                lfs::core::Tensor tile_loss;
                lfs::core::Tensor tile_grad;
                lfs::core::Tensor tile_grad_alpha; // Gradient for alpha (from mask penalty)

                const bool use_mask = params_.optimization.mask_mode != lfs::core::param::MaskMode::None && cam->has_mask();
                if (use_mask) {
                    // Use pipelined mask if available, otherwise load from camera (fallback for validation, etc.)
                    lfs::core::Tensor mask;
                    if (pipelined_mask_.is_valid() && pipelined_mask_.numel() > 0) {
                        mask = pipelined_mask_;
                    } else {
                        // Fallback: load mask from camera (cached after first load)
                        mask = cam->load_and_get_mask(
                            params_.dataset.resize_factor,
                            params_.dataset.max_width,
                            params_.optimization.invert_masks,
                            params_.optimization.mask_threshold);
                    }

                    // Extract mask tile if tiling
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

                // Accumulate tile loss (stay on GPU)
                if (tile_idx == 0) {
                    loss_tensor_gpu = tile_loss;
                } else {
                    loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                }
                nvtxRangePop();

                // Backward through bilateral grid (accumulates gradients, no Adam yet)
                lfs::core::Tensor raster_grad = tile_grad;
                if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                    nvtxRangePush("bilateral_grid_backward");
                    raster_grad = bilateral_grid_->backward(output.image, tile_grad, cam->uid());
                    nvtxRangePop();
                }

                nvtxRangePush("rasterize_backward");
                if (gsplat_ctx) {
                    // GUT mode: use gsplat backward (needs grad_alpha too)
                    auto grad_alpha = tile_grad_alpha.is_valid()
                                          ? tile_grad_alpha
                                          : lfs::core::Tensor::zeros_like(output.alpha);
                    gsplat_rasterize_backward(*gsplat_ctx, raster_grad, grad_alpha,
                                              strategy_->get_model(), strategy_->get_optimizer());
                } else {
                    // Standard mode: use fast rasterizer backward with optional alpha gradient
                    fast_rasterize_backward(*fast_ctx, raster_grad, strategy_->get_model(),
                                            strategy_->get_optimizer(), tile_grad_alpha);
                }
                nvtxRangePop();

                nvtxRangePop(); // End tile
            }

            // Average the loss across tiles for correct reporting
            // (Gradients are already correct from accumulation, this is only for loss display)
            if (num_tiles > 1) {
                loss_tensor_gpu = loss_tensor_gpu / static_cast<float>(num_tiles);
            }

            // Regularization losses are computed ONCE on full model (after all tiles)
            // They accumulate gradients on top of the per-tile gradients

            // Scale regularization loss - accumulate on GPU (AFTER rasterizer backward)
            if (params_.optimization.scale_reg > 0.0f) {
                nvtxRangePush("compute_scale_reg_loss");
                auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                if (!scale_loss_result) {
                    return std::unexpected(scale_loss_result.error());
                }
                loss_tensor_gpu = loss_tensor_gpu + *scale_loss_result;
                nvtxRangePop();
            }

            // Opacity regularization loss - accumulate on GPU (AFTER rasterizer backward)
            if (params_.optimization.opacity_reg > 0.0f) {
                nvtxRangePush("compute_opacity_reg_loss");
                auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                if (!opacity_loss_result) {
                    return std::unexpected(opacity_loss_result.error());
                }
                loss_tensor_gpu = loss_tensor_gpu + *opacity_loss_result;
                nvtxRangePop();
            }

            // Bilateral grid: TV loss + optimizer step
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

                    // Skip post_backward during sparsification phase
                    const bool in_sparsification = params_.optimization.enable_sparsity &&
                                                   iter > (params_.optimization.iterations - params_.optimization.sparsify_steps);
                    if (!in_sparsification) {
                        strategy_->post_backward(iter, r_output);
                    }
                    strategy_->step(iter);
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

            // Configure mask loading if masks are enabled
            PipelinedMaskConfig mask_pipeline_config;
            if (params_.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                mask_pipeline_config.load_masks = true;
                mask_pipeline_config.invert_masks = params_.optimization.invert_masks;
                mask_pipeline_config.mask_threshold = params_.optimization.mask_threshold;
                LOG_INFO("Mask loading enabled in pipeline (invert={}, threshold={})",
                         mask_pipeline_config.invert_masks, mask_pipeline_config.mask_threshold);
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

                auto example_opt = train_dataloader->next();
                if (!example_opt) {
                    LOG_ERROR("DataLoader returned nullopt unexpectedly");
                    break;
                }
                auto& example = *example_opt;
                lfs::core::Camera* cam = example.data.camera;
                lfs::core::Tensor gt_image = std::move(example.data.image);

                // GPU-side sync: make default stream wait for image loading stream
                // This avoids CPU blocking - the GPU handles synchronization
                if (cudaStream_t img_stream = gt_image.stream(); img_stream != nullptr) {
                    cudaEventRecord(img_sync_event_, img_stream);
                    cudaStreamWaitEvent(nullptr, img_sync_event_, 0); // default stream waits
                }

                // Store pipelined mask for use in train_step
                pipelined_mask_ = example.mask.has_value() ? std::move(*example.mask) : lfs::core::Tensor();

                // GPU-side sync for mask stream if different from image stream
                if (pipelined_mask_.is_valid()) {
                    if (cudaStream_t mask_stream = pipelined_mask_.stream();
                        mask_stream != nullptr && mask_stream != gt_image.stream()) {
                        cudaEventRecord(img_sync_event_, mask_stream);
                        cudaStreamWaitEvent(nullptr, img_sync_event_, 0);
                    }
                }

                auto step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                if (!step_result) {
                    // Check if this is an OOM_RETRY signal
                    if (step_result.error() == "OOM_RETRY") {
                        // Aggressive memory cleanup before retry
                        LOG_INFO("Performing aggressive memory cleanup before retry...");

                        // 0. CRITICAL: Synchronize and clear any pending CUDA errors
                        cudaError_t sync_err = cudaDeviceSynchronize();
                        if (sync_err != cudaSuccess) {
                            LOG_WARN("cudaDeviceSynchronize before cleanup returned: {}", cudaGetErrorString(sync_err));
                            // Clear the error so we can continue
                            cudaGetLastError();
                        }

                        // 1. Emergency cleanup of arena (resets offsets, clears inactive frames)
                        lfs::core::GlobalArenaManager::instance().get_arena().emergency_cleanup();

                        // 2. Trim cached memory pool
                        lfs::core::CudaMemoryPool::instance().trim_cached_memory();

                        // 3. Synchronize again after cleanup
                        cudaDeviceSynchronize();

                        // 4. Clear any error state from the OOM
                        cudaGetLastError();

                        // 5. Log memory status
                        size_t free_mem = 0, total_mem = 0;
                        cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
                        if (err == cudaSuccess) {
                            LOG_INFO("CUDA memory after aggressive cleanup: {:.2f} GB free / {:.2f} GB total",
                                     free_mem / (1024.0 * 1024.0 * 1024.0),
                                     total_mem / (1024.0 * 1024.0 * 1024.0));
                        } else {
                            LOG_WARN("cudaMemGetInfo failed: {}", cudaGetErrorString(err));
                            cudaGetLastError(); // Clear this error too
                        }

                        // Retry the same step with upgraded tile mode
                        LOG_INFO("Retrying iteration {} with upgraded tile mode", iter);
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

            LOG_INFO("Training completed successfully");
            return {};
        } catch (const std::exception& e) {
            is_running_ = false;
            cache_loader.clear_cpu_cache();
            lfs::core::image_io::wait_for_pending_saves();

            return std::unexpected(std::format("Training failed: {}", e.what()));
        }
    }

    void Trainer::save_ply(const std::filesystem::path& save_path, int iter_num, bool join_threads) {
        // Save PLY format - join_threads controls sync vs async
        lfs::core::save_ply(strategy_->get_model(), save_path, iter_num, join_threads);

        // Save checkpoint alongside PLY for training resumption
        auto ckpt_result = lfs::training::save_checkpoint(
            save_path, iter_num, *strategy_, params_, bilateral_grid_.get());
        if (!ckpt_result) {
            LOG_WARN("Failed to save checkpoint: {}", ckpt_result.error());
        }

        LOG_DEBUG("PLY save initiated: {} (sync={})", lfs::core::path_to_utf8(save_path), join_threads);
    }

    std::expected<void, std::string> Trainer::save_checkpoint(int iteration) {
        if (!strategy_) {
            return std::unexpected("Cannot save checkpoint: no strategy initialized");
        }

        return lfs::training::save_checkpoint(
            params_.dataset.output_path, iteration, *strategy_, params_,
            bilateral_grid_.get());
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

        auto result = lfs::training::load_checkpoint(
            checkpoint_path, *strategy_, params_, bilateral_grid_.get());
        if (!result) {
            return result;
        }
        current_iteration_ = *result;

        LOG_INFO("Restored training state from checkpoint at iteration {}", *result);
        return result;
    }

} // namespace lfs::training
