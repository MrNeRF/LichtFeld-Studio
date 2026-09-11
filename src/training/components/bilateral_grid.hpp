/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/tensor.hpp"
#include "lfs/kernels/bilateral_grid.cuh"
#include <cmath>
#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

namespace lfs::training {

    enum class BilateralGridParameterization {
        Affine = 0,
        ExposureChroma = 1,
    };

    [[nodiscard]] constexpr int bilateral_grid_channel_count(
        const BilateralGridParameterization parameterization) noexcept {
        return parameterization == BilateralGridParameterization::ExposureChroma ? 9 : 12;
    }

    [[nodiscard]] constexpr const char* bilateral_grid_parameterization_name(
        const BilateralGridParameterization parameterization) noexcept {
        return parameterization == BilateralGridParameterization::ExposureChroma
                   ? "ExposureChroma"
                   : "Affine";
    }

    struct BilateralGridConfig {
        double lr = 2e-3;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-15;
        int warmup_steps = 1000;
        double warmup_start_factor = 0.01;
        double final_lr_factor = 0.01;
    };

    /// Bilateral grid for per-image appearance modeling with fused Adam optimizer
    class BilateralGrid {
    public:
        using Config = BilateralGridConfig;

        BilateralGrid(int num_images, int grid_W, int grid_H, int grid_L,
                      int total_iterations, Config config = {},
                      BilateralGridParameterization parameterization =
                          BilateralGridParameterization::Affine);

        /// Forward pass: apply color correction
        lfs::core::Tensor apply(const lfs::core::Tensor& rgb, int image_idx);

        /// Backward pass: accumulate gradients (call optimizer_step after all backward calls)
        lfs::core::Tensor backward(const lfs::core::Tensor& rgb,
                                   const lfs::core::Tensor& grad_output,
                                   int image_idx);

        /// Compute TV loss for regularization (returns GPU tensor for async accumulation)
        lfs::core::Tensor tv_loss_gpu();
        lfs::core::Tensor tv_loss_gpu(int image_idx);

        /// Accumulate TV loss gradients into the current image slice (+=).
        void tv_backward(float tv_weight);
        void tv_backward(float tv_weight, int image_idx);

        /// Adam + projection + scheduler for the image trained this iteration.
        void step_image(int image_idx, float tv_weight);

        /// Apply Adam with all accumulated gradients
        void optimizer_step();
        void optimizer_step(int image_idx);

        /// Clear gradients for next iteration
        void zero_grad();

        /// Update learning rate schedule
        void scheduler_step();

        /// Subtract the mean of each channel so the mean equals the identity.
        /// per_image==false: dataset mean over (N, L, H, W) via shared_offset.
        /// per_image==true: mean over (L, H, W) written in place.
        void project_mean(bool per_image);
        void project_image(int image_idx);

        [[nodiscard]] float max_abs_channel_mean_deviation() const;
        void log_eval_diagnostics() const;

        // Accessors
        int grid_width() const { return grid_width_; }
        int grid_height() const { return grid_height_; }
        int grid_guidance() const { return grid_guidance_; }
        int num_images() const { return num_images_; }
        int channels() const { return channels_; }
        BilateralGridParameterization parameterization() const { return parameterization_; }
        double get_lr() const { return current_lr_; }
        int64_t get_step() const { return step_; }
        const Config& get_config() const { return config_; }
        lfs::core::Tensor& grids() { return grids_; }
        const lfs::core::Tensor& grids() const { return grids_; }
        lfs::core::Tensor& grad_slice() { return slice_grad_; }
        const lfs::core::Tensor& grad_slice() const { return slice_grad_; }
        const lfs::core::Tensor& shared_offset() const { return shared_offset_; }
        const lfs::core::Tensor& channel_sum() const { return channel_sum_; }

        // Serialization
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);
        void adopt_checkpoint_state(BilateralGrid& loaded);

    private:
        void compute_bias_corrections(float& bc1_rcp, float& bc2_sqrt_rcp) const {
            const double bc1 = 1.0 - std::pow(config_.beta1, step_ + 1);
            const double bc2 = 1.0 - std::pow(config_.beta2, step_ + 1);
            bc1_rcp = static_cast<float>(1.0 / bc1);
            bc2_sqrt_rcp = static_cast<float>(1.0 / std::sqrt(bc2));
        }

        void rebuild_identity_mean();
        void rebuild_projection_state();
        [[nodiscard]] size_t slice_elements() const;
        [[nodiscard]] float* slice_ptr(lfs::core::Tensor& tensor, int image_idx);
        [[nodiscard]] const float* slice_ptr(const lfs::core::Tensor& tensor, int image_idx) const;
        lfs::core::Tensor channel_mean_of_image(int image_idx) const;

        // Grid parameters [N, C, L, H, W]
        lfs::core::Tensor grids_;
        lfs::core::Tensor exp_avg_;
        lfs::core::Tensor exp_avg_sq_;
        lfs::core::Tensor slice_grad_; // [C, L, H, W]
        lfs::core::Tensor tv_temp_buffer_;
        lfs::core::Tensor tv_loss_scalar_; // persistent [1], reused by tv_loss_gpu
        lfs::core::Tensor identity_mean_;  // [C]
        lfs::core::Tensor shared_offset_;  // [C]
        lfs::core::Tensor channel_sum_;    // [C]

        Config config_;
        int64_t step_ = 0;
        std::vector<int64_t> last_step_;
        double current_lr_;
        double initial_lr_;
        int total_iterations_;
        int num_images_;
        int grid_width_;
        int grid_height_;
        int grid_guidance_;
        int channels_ = 12;
        BilateralGridParameterization parameterization_ = BilateralGridParameterization::Affine;
    };

} // namespace lfs::training
