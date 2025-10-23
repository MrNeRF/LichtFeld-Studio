/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "trainer.hpp"
#include "loader/filesystem_utils.hpp"
#include "components/bilateral_grid.hpp"
#include "components/poseopt.hpp"
#include "components/sparsity_optimizer.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "kernels/fused_ssim.cuh"
#include "kernels/regularization.cuh"
#include "loader/cache_image_loader.hpp"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/rasterizer.hpp"

#include <ATen/cuda/CUDAEvent.h>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <expected>
#include <memory>

namespace gs::training {

    void Trainer::cleanup() {
        LOG_DEBUG("Cleaning up trainer for re-initialization");

        // Stop any ongoing operations
        stop_requested_ = true;

        // Wait for callback to finish if busy
        if (callback_busy_.load()) {
            callback_stream_.synchronize();
            callback_busy_.store(false);
        }

        // Reset all components
        progress_.reset();
        bilateral_grid_.reset();
        bilateral_grid_optimizer_.reset();
        bilateral_grid_scheduler_.reset();
        poseopt_module_.reset();
        poseopt_optimizer_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();

        // Clear datasets (will be recreated)
        train_dataset_.reset();
        val_dataset_.reset();

        // Clear camera cache
        m_cam_id_to_cam.clear();

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
            bilateral_grid_ = std::make_unique<BilateralGrid>(
                train_dataset_size_,
                params_.optimization.bilateral_grid_X,
                params_.optimization.bilateral_grid_Y,
                params_.optimization.bilateral_grid_W);

            bilateral_grid_optimizer_ = std::make_unique<torch::optim::Adam>(
                std::vector<torch::Tensor>{bilateral_grid_->parameters()},
                torch::optim::AdamOptions(params_.optimization.bilateral_grid_lr)
                    .eps(1e-15));

            // Create scheduler with warmup
            const double gamma = std::pow(0.01, 1.0 / params_.optimization.iterations);
            bilateral_grid_scheduler_ = std::make_unique<WarmupExponentialLR>(
                *bilateral_grid_optimizer_,
                gamma,
                1000, // warmup steps
                0.01, // start at 1% of initial LR
                -1    // all param groups
            );

            LOG_DEBUG("Bilateral grid initialized with size {}x{}x{} and warmup scheduler",
                      params_.optimization.bilateral_grid_X,
                      params_.optimization.bilateral_grid_Y,
                      params_.optimization.bilateral_grid_W);
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize bilateral grid: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_photometric_loss(
        const RenderOutput& render_output,
        const torch::Tensor& gt_image,
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {
        try {
            // Ensure images have same dimensions
            torch::Tensor rendered = render_output.image;
            torch::Tensor gt = gt_image;

            // Ensure both tensors are 4D (batch, height, width, channels)
            rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
            gt = gt.dim() == 3 ? gt.unsqueeze(0) : gt;

            TORCH_CHECK(rendered.sizes() == gt.sizes(),
                        "ERROR: size mismatch – rendered ", rendered.sizes(),
                        " vs. ground truth ", gt.sizes());

            // Base loss: L1 + SSIM
            auto l1_loss = torch::l1_loss(rendered, gt);
            auto ssim_loss = 1.f - fused_ssim(rendered, gt, "valid", /*train=*/true);
            torch::Tensor loss = (1.f - opt_params.lambda_dssim) * l1_loss +
                                 opt_params.lambda_dssim * ssim_loss;
            return loss;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing photometric loss: {}", e.what()));
        }
    }

    std::expected<float, std::string> Trainer::compute_scale_reg_loss(
        SplatData& splatData,
        const param::OptimizationParameters& opt_params) {
        try {
            if (opt_params.scale_reg > 0.0f) {
                // Efficient fused CUDA kernel for exp regularization with chain rule
                // Forward:  scaling = exp(_scaling)
                // Loss:     L = weight * mean(scaling)
                // Gradient: ∂L/∂_scaling = (weight / N) * exp(_scaling)
                float loss = regularization::compute_exp_l1_regularization_with_grad_cuda(
                    splatData.scaling_raw(),
                    opt_params.scale_reg);
                return loss;
            }
            return 0.0f;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing scale regularization loss: {}", e.what()));
        }
    }

    std::expected<float, std::string> Trainer::compute_opacity_reg_loss(
        SplatData& splatData,
        const param::OptimizationParameters& opt_params) {
        try {
            if (opt_params.opacity_reg > 0.0f) {
                // Use efficient fused CUDA kernel that computes:
                // 1. sigmoid(opacity_raw)
                // 2. Accumulates gradient with chain rule: ∂L/∂opacity_raw = (weight/N) * σ(x) * (1 - σ(x))
                // 3. Returns loss: weight * mean(sigmoid(opacity_raw))
                float loss_value = regularization::compute_sigmoid_l1_regularization_with_grad_cuda(
                    splatData.opacity_raw(),
                    opt_params.opacity_reg);
                return loss_value;
            }
            return 0.0f;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing opacity regularization loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_bilateral_grid_tv_loss(
        const std::unique_ptr<BilateralGrid>& bilateral_grid,
        const param::OptimizationParameters& opt_params) {
        try {
            if (opt_params.use_bilateral_grid) {
                return opt_params.tv_loss_weight * bilateral_grid->tv_loss();
            }
            return torch::zeros({1}, torch::kFloat32).requires_grad_();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing bilateral grid TV loss: {}", e.what()));
        }
    }

    std::expected<torch::Tensor, std::string> Trainer::compute_sparsity_loss(
        int iter,
        const SplatData& splatData) {
        try {
            if (sparsity_optimizer_ && sparsity_optimizer_->should_apply_loss(iter)) {
                // Initialize on first use (lazy initialization)
                if (!sparsity_optimizer_->is_initialized()) {
                    auto init_result = sparsity_optimizer_->initialize(splatData.opacity_raw());
                    if (!init_result) {
                        return std::unexpected(init_result.error());
                    }
                    LOG_INFO("Sparsity optimizer initialized at iteration {}", iter);
                }

                auto loss_result = sparsity_optimizer_->compute_loss(splatData.opacity_raw());
                if (!loss_result) {
                    return std::unexpected(loss_result.error());
                }
                return *loss_result;
            }
            return torch::zeros({1}, torch::kFloat32).to(torch::kCUDA).requires_grad_();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error computing sparsity loss: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::handle_sparsity_update(
        int iter,
        SplatData& splatData) {
        try {
            if (sparsity_optimizer_ && sparsity_optimizer_->should_update(iter)) {
                LOG_TRACE("Updating sparsity state at iteration {}", iter);
                auto result = sparsity_optimizer_->update_state(splatData.opacity_raw());
                if (!result) {
                    return std::unexpected(result.error());
                }
            }
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error updating sparsity state: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::apply_sparsity_pruning(
        int iter,
        SplatData& splatData) {
        try {
            if (sparsity_optimizer_ && sparsity_optimizer_->should_prune(iter)) {
                LOG_INFO("Applying sparsity-based pruning at iteration {}", iter);

                auto mask_result = sparsity_optimizer_->get_prune_mask(splatData.opacity_raw());
                if (!mask_result) {
                    return std::unexpected(mask_result.error());
                }

                auto prune_mask = *mask_result;
                int n_prune = prune_mask.sum().item<int>();
                int n_before = splatData.size();

                // Use strategy's remove functionality
                strategy_->remove_gaussians(prune_mask);

                int n_after = splatData.size();
                std::println("Sparsity pruning complete: {} -> {} Gaussians (removed {})",
                             n_before, n_after, n_prune);

                // Clear sparsity optimizer after pruning
                sparsity_optimizer_.reset();
                LOG_DEBUG("Sparsity optimizer cleared after pruning");
            }
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Error applying sparsity pruning: {}", e.what()));
        }
    }

    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy,
                     std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits)
        : base_dataset_(std::move(dataset)),
          strategy_(std::move(strategy)),
          provided_splits_(std::move(provided_splits)) {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }
        LOG_DEBUG("Trainer constructed with {} cameras", base_dataset_->get_cameras().size());
    }

    void Trainer::load_cameras_info() {
        m_cam_id_to_cam.clear();
        // Setup camera cache
        for (const auto& cam : base_dataset_->get_cameras()) {
            m_cam_id_to_cam[cam->uid()] = cam;
        }
    }

    std::expected<void, std::string> Trainer::initialize(const param::TrainingParameters& params) {
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

            // Handle dataset split based on evaluation flag
            if (params.optimization.enable_eval) {
                // Create train/val split
                train_dataset_ = std::make_shared<CameraDataset>(
                    base_dataset_->get_cameras(), params.dataset, CameraDataset::Split::TRAIN,
                    provided_splits_ ? std::make_optional(std::get<0>(*provided_splits_)) : std::nullopt);
                val_dataset_ = std::make_shared<CameraDataset>(
                    base_dataset_->get_cameras(), params.dataset, CameraDataset::Split::VAL,
                    provided_splits_ ? std::make_optional(std::get<1>(*provided_splits_)) : std::nullopt);

                LOG_INFO("Created train/val split: {} train, {} val images",
                         train_dataset_->size().value(),
                         val_dataset_->size().value());
            } else {
                // Use all images for training
                train_dataset_ = base_dataset_;
                val_dataset_ = nullptr;

                LOG_INFO("Using all {} images for training (no evaluation)",
                         train_dataset_->size().value());
            }

            // change resize factor (change may comes from gui)
            if (train_dataset_) {
                train_dataset_->set_resize_factor(params.dataset.resize_factor);
                train_dataset_->set_max_width(params.dataset.max_width);
            }
            if (val_dataset_) {
                val_dataset_->set_resize_factor(params.dataset.resize_factor);
                val_dataset_->set_max_width(params.dataset.max_width);
            }

            train_dataset_size_ = train_dataset_->size().value();

            m_cam_id_to_cam.clear();
            // Setup camera cache
            for (const auto& cam : base_dataset_->get_cameras()) {
                m_cam_id_to_cam[cam->uid()] = cam;
            }
            LOG_DEBUG("Camera cache initialized with {} cameras", m_cam_id_to_cam.size());

            // Re-initialize strategy with new parameters
            strategy_->initialize(params.optimization);
            LOG_DEBUG("Strategy initialized");

            // Initialize bilateral grid if enabled
            if (auto result = initialize_bilateral_grid(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize sparsity optimizer if enabled
            if (params.optimization.enable_sparsity) {
                // Calculate when sparsity should start
                int base_iterations = params.optimization.iterations;
                int sparsity_start = base_iterations; // Start after base training
                int total_iterations = base_iterations + params.optimization.sparsify_steps;

                // Extend the total training iterations
                params_.optimization.iterations = total_iterations;

                ADMMSparsityOptimizer::Config sparsity_config{
                    .sparsify_steps = params.optimization.sparsify_steps,
                    .init_rho = params.optimization.init_rho,
                    .prune_ratio = params.optimization.prune_ratio,
                    .update_every = 50,
                    .start_iteration = sparsity_start // Start after base training completes
                };

                sparsity_optimizer_ = SparsityOptimizerFactory::create("admm", sparsity_config);

                if (sparsity_optimizer_) {
                    // Don't initialize yet - will initialize when we reach start_iteration
                    LOG_INFO("=== Sparsity Optimization Configuration ===");
                    LOG_INFO("Base training iterations: {}", base_iterations);
                    LOG_INFO("Sparsification starts at: iteration {}", sparsity_start);
                    LOG_INFO("Sparsification duration: {} iterations", params.optimization.sparsify_steps);
                    LOG_INFO("Total training iterations: {}", total_iterations);
                    LOG_INFO("Pruning ratio: {}%", params.optimization.prune_ratio * 100);
                    LOG_INFO("ADMM penalty (rho): {}", params.optimization.init_rho);
                }
            }

            background_ = torch::tensor({0.f, 0.f, 0.f},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

            if (params.optimization.pose_optimization != "none") {
                if (params.optimization.enable_eval) {
                    return std::unexpected("Evaluating with pose optimization is not supported yet. "
                                           "Please disable pose optimization or evaluation.");
                }
                if (params.optimization.gut) {
                    return std::unexpected("The 3DGUT rasterizer doesn't have camera gradients yet. "
                                           "Please disable pose optimization or disable gut.");
                }
                if (params.optimization.pose_optimization == "direct") {
                    poseopt_module_ = std::make_unique<DirectPoseOptimizationModule>(train_dataset_->get_cameras().size());
                    LOG_DEBUG("Direct pose optimization module created");
                } else if (params.optimization.pose_optimization == "mlp") {
                    poseopt_module_ = std::make_unique<MLPPoseOptimizationModule>(train_dataset_->get_cameras().size());
                    LOG_DEBUG("MLP pose optimization module created");
                } else {
                    return std::unexpected("Invalid pose optimization type: " + params.optimization.pose_optimization);
                }
                poseopt_optimizer_ = std::make_unique<torch::optim::Adam>(
                    std::vector<torch::Tensor>{poseopt_module_->parameters()},
                    torch::optim::AdamOptions(1e-5));
            } else {
                poseopt_module_ = std::make_unique<PoseOptimizationModule>();
            }

            // Create progress bar based on headless flag
            if (params.optimization.headless) {
                progress_ = std::make_unique<TrainingProgress>(
                    params_.optimization.iterations, // This now includes sparsity steps if enabled
                    /*update_frequency=*/100);
                LOG_DEBUG("Progress bar initialized for {} total iterations", params_.optimization.iterations);
            }

            // Initialize the evaluator - it handles all metrics internally
            evaluator_ = std::make_unique<MetricsEvaluator>(params_);
            LOG_DEBUG("Metrics evaluator initialized");

            // Print configuration
            LOG_INFO("Render mode: {}", params.optimization.render_mode);
            LOG_INFO("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
            LOG_INFO("Strategy: {}", params.optimization.strategy);

            initialized_ = true;
            LOG_INFO("Trainer initialization complete");
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize trainer: {}", e.what()));
        }
    }

    Trainer::~Trainer() {
        // Ensure training is stopped
        stop_requested_ = true;

        // Wait for callback to finish if busy
        if (callback_busy_.load()) {
            callback_stream_.synchronize();
        }
        LOG_DEBUG("Trainer destroyed");
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

        // Handle save request
        if (save_requested_.exchange(false)) {
            LOG_INFO("Saving checkpoint at iteration {}...", iter);
            auto checkpoint_path = params_.dataset.output_path / "checkpoints";
            save_ply(checkpoint_path, iter, /*join=*/true);

            LOG_INFO("Checkpoint saved to {}", checkpoint_path.string());

            // Emit checkpoint saved event
            events::state::CheckpointSaved{
                .iteration = iter,
                .path = checkpoint_path}
                .emit();
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

    torch::Tensor sine_background_for_step(
        int step, int periodR = 37, int periodG = 41, int periodB = 43, bool grayscale_only = false, float jitter_amp = 0.03f) {
        const float eps = 1e-4f;
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        const float two_pi = M_PI * 2.0f;

        // Phase 0..2PI
        const float tR = (periodR > 0) ? float(step % periodR) / float(periodR) : 0.0f;
        const float phaseR = two_pi * tR;

        const float tG = (periodG > 0) ? float(step % periodG) / float(periodG) : 0.0f;
        const float phaseG = two_pi * tG;

        const float tB = (periodB > 0) ? float(step % periodB) / float(periodB) : 0.0f;
        const float phaseB = two_pi * tB;

        torch::Tensor bg;
        if (grayscale_only) {
            // Grayscale: g in [0,1]
            float g = 0.5f * (1.0f + std::sin(phaseG));
            bg = torch::tensor({g, g, g}, opts);
        } else {
            // Phase-shifted RGB: covers the color wheel over the cycle
            float r = 0.5f * (1.0f + std::sin(phaseR + 0.0f * two_pi / 3.0f));
            float g = 0.5f * (1.0f + std::sin(phaseG + 1.0f * two_pi / 3.0f));
            float b = 0.5f * (1.0f + std::sin(phaseB + 2.0f * two_pi / 3.0f));
            bg = torch::tensor({r, g, b}, opts);
        }

        // Small jitter to prevent exact periodic lock-in
        if (jitter_amp > 0.0f) {
            auto jitter = (torch::rand({3}, opts) - 0.5f) * (2.0f * jitter_amp);
            bg = (bg + jitter).clamp(eps, 1.0f - eps);
        } else {
            bg = bg.clamp(eps, 1.0f - eps);
        }
        return bg;
    }

    // Helper to ensure buf matches base (defined, dtype, device, shape)
    static inline void ensure_like(torch::Tensor& buf, const torch::Tensor& base) {
        bool is_undefined = !buf.defined();
        bool dtype_mismatch = (buf.dtype() != base.dtype());

        bool need = (is_undefined || dtype_mismatch);
        if (!need) {
            bool device_mismatch = (buf.device() != base.device());
            bool shape_mismatch = (buf.sizes().vec() != base.sizes().vec());
            need = (device_mismatch || shape_mismatch);
        }

        if (need)
            buf = torch::empty_like(base);
    }

    torch::Tensor& Trainer::background_for_step(int iter) {
        torch::NoGradGuard no_grad;
        const auto& opt = params_.optimization;

        // Fast path: modulation disabled: return base background_
        if (!opt.bg_modulation) {
            return background_;
        }

        const float w_mix = inv_weight_piecewise(iter, opt.iterations);
        if (w_mix <= 0.0f) {
            return background_;
        }

        // Generate per-iteration sine background
        auto sine_bg = sine_background_for_step(iter);

        // Ensure reusable buffer exists
        ensure_like(bg_mix_buffer_, background_);

        bg_mix_buffer_.copy_(background_); // d2d copy of 3 floats
        bg_mix_buffer_.mul_(1.0f - w_mix);
        bg_mix_buffer_.add_(sine_bg, w_mix);

        return bg_mix_buffer_; // const ref to mixed background
    }

    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        Camera* cam,
        torch::Tensor gt_image,
        RenderMode render_mode,
        std::stop_token stop_token) {
        try {
            if (params_.optimization.gut) {
                if (cam->camera_model_type() == gsplat::CameraModelType::ORTHO) {
                    return std::unexpected("Training on cameras with ortho model is not supported yet.");
                }
            } else {
                if (cam->radial_distortion().numel() != 0 ||
                    cam->tangential_distortion().numel() != 0) {
                    return std::unexpected("Distorted images detected.  You can use --gut option to train on cameras with distortion.");
                }
                if (cam->camera_model_type() != gsplat::CameraModelType::PINHOLE) {
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

            // Add phase transition logging for sparsity
            if (params_.optimization.enable_sparsity) {
                // Calculate base iterations (original iterations before extension)
                int base_iterations = params_.optimization.iterations - params_.optimization.sparsify_steps;

                // Log phase transition
                if (iter == base_iterations + 1) {
                    LOG_INFO("=== Entering Sparsification Phase ===");
                    LOG_INFO("Base training complete at iteration {}", base_iterations);
                    LOG_INFO("Starting ADMM sparsification for {} iterations",
                             params_.optimization.sparsify_steps);
                    LOG_INFO("Current model size: {} Gaussians", strategy_->get_model().size());
                    LOG_INFO("Target pruning: {}% of Gaussians", params_.optimization.prune_ratio * 100);
                }

                // Log when approaching pruning
                if (iter == params_.optimization.iterations - 100) {
                    LOG_INFO("Approaching final pruning in 100 iterations (at iteration {})",
                             params_.optimization.iterations);
                }

                // Log when pruning will occur
                if (iter == params_.optimization.iterations - 1) {
                    LOG_INFO("Final pruning will occur next iteration");
                }
            }

            auto adjusted_cam_pos = poseopt_module_->forward(cam->world_view_transform(), torch::tensor({cam->uid()}));
            auto adjusted_cam = Camera(*cam, adjusted_cam_pos);

            torch::Tensor& bg = background_for_step(iter);

            RenderOutput r_output;
            // Use the render mode from parameters
            if (!params_.optimization.gut) {
                r_output = fast_rasterize(adjusted_cam, strategy_->get_model(), bg);
            } else {
                r_output = rasterize(adjusted_cam, strategy_->get_model(), bg, 1.0f, false, false, render_mode,
                                     nullptr);
            }

            // Apply bilateral grid if enabled
            if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                r_output.image = bilateral_grid_->apply(r_output.image, cam->uid());
            }

            // Compute losses
            auto loss_result = compute_photometric_loss(r_output,
                                                        gt_image,
                                                        strategy_->get_model(),
                                                        params_.optimization);
            if (!loss_result) {
                return std::unexpected(loss_result.error());
            }

            torch::Tensor loss = *loss_result;
            loss.backward();
            float loss_value = loss.item<float>();

            // Scale regularization loss
            auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), params_.optimization);
            if (!scale_loss_result) {
                return std::unexpected(scale_loss_result.error());
            }
            loss_value += *scale_loss_result;

            // Opacity regularization loss
            auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), params_.optimization);
            if (!opacity_loss_result) {
                return std::unexpected(opacity_loss_result.error());
            }
            loss_value += *opacity_loss_result;

            // Bilateral grid TV loss
            auto tv_loss_result = compute_bilateral_grid_tv_loss(bilateral_grid_, params_.optimization);
            if (!tv_loss_result) {
                return std::unexpected(tv_loss_result.error());
            }
            loss = *tv_loss_result;
            loss.backward();
            loss_value += loss.item<float>();

            // Add sparsity loss
            auto sparsity_loss_result = compute_sparsity_loss(iter, strategy_->get_model());
            if (!sparsity_loss_result) {
                return std::unexpected(sparsity_loss_result.error());
            }
            loss = *sparsity_loss_result;
            loss.backward();
            loss_value += loss.item<float>();

            // Store the loss value immediately
            current_loss_ = loss_value;

            // Update progress synchronously if needed
            if (progress_) {
                progress_->update(iter, loss_value,
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Emit training progress event (throttled to reduce GUI updates)
            if (iter % 10 == 0 || iter == 1) {
                // Only update every 10 iterations
                events::state::TrainingProgress{
                    .iteration = iter,
                    .loss = loss_value,
                    .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                    .is_refining = strategy_->is_refining(iter)}
                    .emit();
            }
            {
                torch::NoGradGuard no_grad;

                DeferredEvents deferred;
                {
                    std::unique_lock<std::shared_mutex> lock(render_mutex_);

                    // Execute strategy post-backward and step
                    // Only call post_backward during base training (not during sparsification)
                    if (params_.optimization.enable_sparsity) {
                        int base_iterations = params_.optimization.iterations - params_.optimization.sparsify_steps;
                        if (iter <= base_iterations) {
                            strategy_->post_backward(iter, r_output);
                        }
                        // During sparsification phase, skip post_backward entirely
                    } else {
                        // No sparsity, always call post_backward
                        strategy_->post_backward(iter, r_output);
                    }

                    strategy_->step(iter);

                    if (params_.optimization.use_bilateral_grid) {
                        bilateral_grid_optimizer_->step();
                        bilateral_grid_optimizer_->zero_grad(true);
                        bilateral_grid_scheduler_->step();
                    }
                    if (params_.optimization.pose_optimization != "none") {
                        poseopt_optimizer_->step();
                        poseopt_optimizer_->zero_grad(true);
                    }

                    // Queue event for emission after lock release
                    deferred.add(events::state::ModelUpdated{
                        .iteration = iter,
                        .num_gaussians = static_cast<size_t>(strategy_->get_model().size())});
                } // Lock released here

                // Events automatically emitted here when deferred destructs

                // Handle sparsity updates
                if (auto result = handle_sparsity_update(iter, strategy_->get_model()); !result) {
                    LOG_ERROR("Sparsity update failed: {}", result.error());
                }

                // Apply sparsity pruning if needed
                if (auto result = apply_sparsity_pruning(iter, strategy_->get_model()); !result) {
                    LOG_ERROR("Sparsity pruning failed: {}", result.error());
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

                // Save model at specified steps
                if (!params_.optimization.skip_intermediate_saving) {
                    for (size_t save_step : params_.optimization.save_steps) {
                        if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                            const bool join_threads = (iter == params_.optimization.save_steps.back());
                            auto save_path = params_.dataset.output_path;
                            save_ply(save_path, iter, /*join=*/join_threads);
                            // Emit checkpoint saved event
                            events::state::CheckpointSaved{
                                .iteration = iter,
                                .path = save_path}
                                .emit();
                        }
                    }
                }

                if (!params_.dataset.timelapse_images.empty() && iter % params_.dataset.timelapse_every == 0) {
                    for (const auto& img_name : params_.dataset.timelapse_images) {
                        auto train_cam = train_dataset_->get_camera_by_filename(img_name);
                        auto val_cam = val_dataset_ ? val_dataset_->get_camera_by_filename(img_name) : std::nullopt;
                        if (train_cam.has_value() || val_cam.has_value()) {
                            Camera* cam_to_use = train_cam.has_value() ? train_cam.value() : val_cam.value();

                            // Image size isn't correct until the image has been loaded once
                            // If we use the camera before it's loaded, it will render images at the non-scaled size
                            if ((cam_to_use->camera_height() == cam_to_use->image_height() && params_.dataset.resize_factor != 1) ||
                                cam_to_use->image_height() > params_.dataset.max_width ||
                                cam_to_use->image_width() > params_.dataset.max_width) {
                                cam_to_use->load_image_size(params_.dataset.resize_factor, params_.dataset.max_width);
                            }

                            RenderOutput rendered_timelapse_output;
                            if (params_.optimization.gut) {
                                rendered_timelapse_output = rasterize(*cam_to_use, strategy_->get_model(), bg, 1.0f, false,
                                                                     false, RenderMode::RGB, nullptr);
                            } else {
                                rendered_timelapse_output = fast_rasterize(*cam_to_use, strategy_->get_model(), background_);
                            }

                            // Get folder name to save in by stripping file extension
                            std::string folder_name = loader::strip_extension(img_name);

                            auto output_path = params_.dataset.output_path / "timelapse" / folder_name;
                            std::filesystem::create_directories(output_path);

                            image_io::save_image_async(output_path / std::format("{:06d}.jpg", iter),
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

        // Event-based ready signaling
        if (!params_.optimization.headless) {
            // Subscribe to start signal (no need to store handle)
            events::internal::TrainingReadyToStart::when([this](const auto&) {
                ready_to_start_ = true;
            });

            // Signal we're ready
            events::internal::TrainerReady{}.emit();

            // Wait for start signal
            LOG_DEBUG("Waiting for start signal from GUI...");
            while (!ready_to_start_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }

        is_running_ = true; // Now we can start
        LOG_INFO("Starting training loop with {} workers", params_.optimization.num_workers);
        // initializing image loader
        auto& cache_loader = gs::loader::CacheLoader::getInstance(params_.dataset.loading_params.use_cpu_memory, params_.dataset.loading_params.use_fs_cache);
        cache_loader.reset_cache();
        // in case we call getInstance multiple times and cache parameters/dataset were changed by user
        cache_loader.update_cache_params(params_.dataset.loading_params.use_cpu_memory,
                                         params_.dataset.loading_params.use_fs_cache, train_dataset_size_);

        try {
            int iter = 1;
            const int num_workers = params_.optimization.num_workers;
            const RenderMode render_mode = stringToRenderMode(params_.optimization.render_mode);

            if (progress_) {
                progress_->update(iter, current_loss_.load(),
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Use infinite dataloader to avoid epoch restarts
            auto train_dataloader = create_infinite_dataloader_from_dataset(train_dataset_, num_workers);
            auto loader = train_dataloader->begin();

            LOG_DEBUG("Starting training iterations");
            // Single loop without epochs
            while (iter <= params_.optimization.iterations) {
                if (stop_token.stop_requested() || stop_requested_.load()) {
                    break;
                }

                // Wait for previous callback if still running
                if (callback_busy_.load()) {
                    callback_stream_.synchronize();
                }

                auto& batch = *loader;
                auto camera_with_image = batch[0].data;
                Camera* cam = camera_with_image.camera;
                torch::Tensor gt_image = std::move(camera_with_image.image).to(torch::kCUDA, /*non_blocking=*/true);

                auto step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                if (!step_result) {
                    return std::unexpected(step_result.error());
                }

                if (*step_result == StepResult::Stop) {
                    break;
                }

                // Launch callback for async progress update (except first iteration)
                if (iter > 1 && callback_) {
                    callback_busy_ = true;
                    auto err = cudaLaunchHostFunc(
                        callback_stream_.stream(),
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
                ++loader;
            }

            // Ensure callback is finished before final save
            if (callback_busy_.load()) {
                callback_stream_.synchronize();
            }

            // Final save if not already saved by stop request
            if (!stop_requested_.load() && !stop_token.stop_requested()) {
                auto final_path = params_.dataset.output_path;
                save_ply(final_path, params_.optimization.iterations, /*join=*/true);
                // Emit final checkpoint saved event
                events::state::CheckpointSaved{
                    static_cast<int>(params_.optimization.iterations),
                    final_path}
                    .emit();
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

            LOG_INFO("Training completed successfully");
            return {};
        } catch (const std::exception& e) {
            is_running_ = false;
            cache_loader.clear_cpu_cache();

            return std::unexpected(std::format("Training failed: {}", e.what()));
        }
    }

    std::shared_ptr<const Camera> Trainer::getCamById(int camId) const {
        const auto it = m_cam_id_to_cam.find(camId);
        if (it == m_cam_id_to_cam.end()) {
            LOG_ERROR("getCamById - could not find cam with cam id {}", camId);
            return nullptr;
        }
        return it->second;
    }

    std::vector<std::shared_ptr<const Camera>> Trainer::getCamList() const {
        std::vector<std::shared_ptr<const Camera>> cams;
        cams.reserve(m_cam_id_to_cam.size());
        for (auto& [key, value] : m_cam_id_to_cam) {
            cams.push_back(value);
        }

        return cams;
    }

    void Trainer::save_ply(const std::filesystem::path& save_path, int iter_num, bool join_threads) {
        // Save PLY format - join_threads controls sync vs async
        strategy_->get_model().save_ply(save_path, iter_num, join_threads);

        // Save SOG format if requested - ALWAYS synchronous
        std::filesystem::path sog_path;
        if (params_.optimization.save_sog) {
            sog_path = strategy_->get_model().save_sog(save_path, iter_num,
                                                       params_.optimization.sog_iterations,
                                                       true); // Always synchronous
        }

        // Update project with PLY info
        if (lf_project_) {
            const std::string ply_name = "splat_" + std::to_string(iter_num);
            const std::filesystem::path ply_path = save_path / (ply_name + ".ply");
            lf_project_->addPly(gs::management::PlyData(false, ply_path, iter_num, ply_name));
            if (params_.optimization.save_sog) {
                std::string ply_name_sog = sog_path.stem().string();
                lf_project_->addPly(gs::management::PlyData(false, sog_path, iter_num, ply_name_sog));
            }
        }

        LOG_DEBUG("PLY save initiated: {} (sync={}), SOG always sync", save_path.string(), join_threads);
    }
} // namespace gs::training