/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/tensor.hpp"
#include "lfs/kernels/bilateral_grid.cuh"
#include <cmath>
#include <istream>
#include <ostream>

namespace lfs::training {

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
                      int total_iterations, Config config = {});

        /// Forward pass: apply color correction
        lfs::core::Tensor apply(const lfs::core::Tensor& rgb, int image_idx);

        /// Backward pass: accumulate gradients (call optimizer_step after all backward calls)
        lfs::core::Tensor backward(const lfs::core::Tensor& rgb,
                                   const lfs::core::Tensor& grad_output,
                                   int image_idx);

        /// Compute TV loss for regularization (returns GPU tensor for async accumulation)
        lfs::core::Tensor tv_loss_gpu();

        /// Accumulate TV loss gradients
        void tv_backward(float tv_weight);

        /// Apply Adam with all accumulated gradients
        void optimizer_step();

        /// Clear gradients for next iteration
        void zero_grad();

        /// Update learning rate schedule
        void scheduler_step();

        // Accessors
        int grid_width() const { return grid_width_; }
        int grid_height() const { return grid_height_; }
        int grid_guidance() const { return grid_guidance_; }
        int num_images() const { return num_images_; }
        double get_lr() const { return current_lr_; }
        int64_t get_step() const { return step_; }
        const lfs::core::Tensor& grids() const { return grids_; }

        // Serialization
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);
        void adopt_checkpoint_state(BilateralGrid& loaded) noexcept;

    private:
        void compute_bias_corrections(float& bc1_rcp, float& bc2_sqrt_rcp) const {
            const double bc1 = 1.0 - std::pow(config_.beta1, step_ + 1);
            const double bc2 = 1.0 - std::pow(config_.beta2, step_ + 1);
            bc1_rcp = static_cast<float>(1.0 / bc1);
            bc2_sqrt_rcp = static_cast<float>(1.0 / std::sqrt(bc2));
        }

        // Grid parameters [N, 12, L, H, W]
        lfs::core::Tensor grids_;
        lfs::core::Tensor exp_avg_;
        lfs::core::Tensor exp_avg_sq_;
        lfs::core::Tensor accumulated_grads_;
        lfs::core::Tensor grad_buffer_;
        lfs::core::Tensor tv_temp_buffer_;

        Config config_;
        int64_t step_ = 0;
        double current_lr_;
        double initial_lr_;
        int total_iterations_;
        int num_images_;
        int grid_width_;
        int grid_height_;
        int grid_guidance_;
    };

} // namespace lfs::training
