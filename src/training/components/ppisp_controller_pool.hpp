/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace lfs::training {

    struct PPISPControllerPoolConfig {
        double lr = 1e-3;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;
        int warmup_steps = 100;
        double warmup_start_factor = 0.1;
        double final_lr_factor = 0.01;
    };

    /// Pool of PPISP controllers with shared resources for memory efficiency.
    /// Only per-camera FC weights are unique; CNN, buffers, gradients, and Adam state are shared.
    /// Memory: ~0.9 MB per camera + ~50 MB shared (vs ~3.8 MB per camera without sharing)
    class PPISPControllerPool {
    public:
        using Config = PPISPControllerPoolConfig;

        PPISPControllerPool(int num_cameras, int total_iterations, Config config = {});
        ~PPISPControllerPool() = default;

        PPISPControllerPool(const PPISPControllerPool&) = delete;
        PPISPControllerPool& operator=(const PPISPControllerPool&) = delete;
        PPISPControllerPool(PPISPControllerPool&&) = default;
        PPISPControllerPool& operator=(PPISPControllerPool&&) = default;

        /// Allocate shared buffers for the given max image size. Must be called before predict().
        void allocate_buffers(size_t max_h, size_t max_w);

        /// Forward pass for a specific camera.
        [[nodiscard]] lfs::core::Tensor predict(int camera_idx, const lfs::core::Tensor& rendered_rgb,
                                                float exposure_prior = 1.0f);

        /// Backward pass for a specific camera. Must call predict() first.
        void backward(int camera_idx, const lfs::core::Tensor& grad_output);

        /// Update weights for a specific camera using stored gradients.
        void optimizer_step(int camera_idx);

        /// Zero gradients (shared, so just one call needed).
        void zero_grad();

        /// Update learning rate scheduler for a specific camera.
        void scheduler_step(int camera_idx);

        /// Get current learning rate.
        [[nodiscard]] double get_learning_rate() const { return current_lr_; }

        /// Get number of cameras.
        [[nodiscard]] int num_cameras() const { return num_cameras_; }

        /// Serialize all controller states.
        void serialize(std::ostream& os) const;

        /// Deserialize all controller states.
        void deserialize(std::istream& is);

        /// Serialize only weights for inference (no optimizer state).
        void serialize_inference(std::ostream& os) const;

        /// Deserialize inference weights.
        void deserialize_inference(std::istream& is);

        /// Copy inference weights from another controller pool using explicit source camera indices
        /// for each target camera. Optimizer state is reset.
        [[nodiscard]] std::string copy_inference_weights_from(
            const PPISPControllerPool& source,
            const std::vector<int>& source_camera_indices);

    private:
        void adam_update(lfs::core::Tensor& param, lfs::core::Tensor& exp_avg,
                         lfs::core::Tensor& exp_avg_sq, const lfs::core::Tensor& grad);
        void compute_bias_corrections(float& bc1_rcp, float& bc2_sqrt_rcp) const;

        int num_cameras_;
        int total_iterations_;
        Config config_;
        int64_t step_ = 0;
        double current_lr_;
        double initial_lr_;

        // Shared CNN weights (frozen, same for all cameras)
        lfs::core::Tensor conv1_w_, conv1_b_;
        lfs::core::Tensor conv2_w_, conv2_b_;
        lfs::core::Tensor conv3_w_, conv3_b_;

        // Per-camera FC weights [num_cameras]
        std::vector<lfs::core::Tensor> fc1_w_, fc1_b_;
        std::vector<lfs::core::Tensor> fc2_w_, fc2_b_;
        std::vector<lfs::core::Tensor> fc3_w_, fc3_b_;
        std::vector<lfs::core::Tensor> fc4_w_, fc4_b_;

        // Shared forward buffers
        size_t buf_h_ = 0, buf_w_ = 0;
        lfs::core::Tensor buf_conv1_, buf_pool_, buf_conv2_, buf_conv3_, buf_pool2_;
        lfs::core::Tensor buf_fc1_, buf_fc2_, buf_fc3_, buf_output_;
        lfs::core::Tensor fc_input_buffer_;
        lfs::core::Tensor cached_flat_;

        // Shared backward buffers
        lfs::core::Tensor grad_fc3_out_, grad_fc2_out_, grad_fc1_out_;

        // Shared gradients (only one camera trains at a time)
        lfs::core::Tensor fc1_w_grad_, fc1_b_grad_;
        lfs::core::Tensor fc2_w_grad_, fc2_b_grad_;
        lfs::core::Tensor fc3_w_grad_, fc3_b_grad_;
        lfs::core::Tensor fc4_w_grad_, fc4_b_grad_;

        // Shared Adam state (only one camera trains at a time)
        lfs::core::Tensor fc1_w_m_, fc1_w_v_, fc1_b_m_, fc1_b_v_;
        lfs::core::Tensor fc2_w_m_, fc2_w_v_, fc2_b_m_, fc2_b_v_;
        lfs::core::Tensor fc3_w_m_, fc3_w_v_, fc3_b_m_, fc3_b_v_;
        lfs::core::Tensor fc4_w_m_, fc4_w_v_, fc4_b_m_, fc4_b_v_;

        // Track which camera was last used (for assertions)
        int last_predict_camera_ = -1;
    };

} // namespace lfs::training
