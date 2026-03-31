/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <expected>
#include <memory>
#include <string>

namespace lfs::training {

    /**
     * @brief Context for manual sparsity loss forward/backward (LibTorch-free)
     */
    struct SparsityLossContext {
        const float* opacities_ptr;   // Pointer to opacity values (raw, before sigmoid)
        const float* opa_sigmoid_ptr; // Pointer to sigmoid(opacities)
        const float* z_ptr;           // Pointer to ADMM auxiliary variable
        const float* u_ptr;           // Pointer to ADMM dual variable
        size_t n;                     // Number of elements
        float rho;                    // ADMM penalty parameter
    };

    /**
     * @brief Interface for sparsity optimization methods (LibTorch-free)
     *
     * Provides a clean abstraction for different sparsity-inducing techniques
     * that can be applied during Gaussian Splatting training.
     */
    class ISparsityOptimizer {
    public:
        virtual ~ISparsityOptimizer() = default;

        /**
         * @brief Initialize the optimizer with initial opacities
         * @param opacities Initial opacity values from the model [N, 1]
         * @return Error string if initialization fails
         */
        virtual std::expected<void, std::string> initialize(const lfs::core::Tensor& opacities) = 0;

        /**
         * @brief MANUAL FORWARD: Compute sparsity loss without autograd
         * @param opacities Current opacity values from the model [N, 1]
         * @return (loss_tensor_gpu, context) or error string - NO CPU SYNC
         */
        virtual std::expected<std::pair<lfs::core::Tensor, SparsityLossContext>, std::string>
        compute_loss_forward(const lfs::core::Tensor& opacities) = 0;

        /**
         * @brief MANUAL BACKWARD: Compute gradients manually
         * @param ctx Context from forward pass
         * @param grad_loss Gradient of total loss w.r.t. sparsity loss (usually 1.0)
         * @param grad_opacities [N, 1] - Output gradient buffer to write to
         * @return Error string if backward fails
         * @note Gradients are written directly to grad_opacities (accumulated)
         */
        virtual std::expected<void, std::string>
        compute_loss_backward(const SparsityLossContext& ctx,
                              float grad_loss,
                              lfs::core::Tensor& grad_opacities) = 0;

        /**
         * @brief Update internal state (called periodically during training)
         * @param opacities Current opacity values from the model [N, 1]
         * @return Error string if update fails
         */
        virtual std::expected<void, std::string> update_state(const lfs::core::Tensor& opacities) = 0;

        /**
         * @brief Get mask indicating which Gaussians to prune
         * @param opacities Current opacity values from the model [N, 1]
         * @return Boolean mask tensor [N] or error string
         */
        virtual std::expected<lfs::core::Tensor, std::string>
        get_prune_mask(const lfs::core::Tensor& opacities) = 0;

        /**
         * @brief Check if we should update state at this iteration
         */
        virtual bool should_update(int iter) const = 0;

        /**
         * @brief Check if we should apply loss at this iteration
         */
        virtual bool should_apply_loss(int iter) const = 0;

        /**
         * @brief Check if we should prune at this iteration
         */
        virtual bool should_prune(int iter) const = 0;

        /**
         * @brief Get the number of Gaussians that will be pruned
         */
        virtual int get_num_to_prune(const lfs::core::Tensor& opacities) = 0;

        /**
         * @brief Check if the optimizer has been initialized
         */
        virtual bool is_initialized() const = 0;

        /**
         * @brief Reset cached state after Gaussian topology changes
         */
        virtual void reset() = 0;
    };

    /**
     * @brief ADMM-based sparsity optimizer (LibTorch-free)
     *
     * Implements Alternating Direction Method of Multipliers (ADMM) for
     * inducing sparsity in Gaussian opacity values during training.
     */
    class ADMMSparsityOptimizer : public ISparsityOptimizer {
    public:
        struct Config {
            int sparsify_steps = 15000;  // Total steps for sparsification
            float init_rho = 0.0005f;    // ADMM penalty parameter
            float prune_ratio = 0.6f;    // Final pruning ratio
            int update_every = 50;       // Update ADMM state every N iterations
            int start_iteration = 30000; // When to start sparsification (after base training)
        };

        explicit ADMMSparsityOptimizer(const Config& config);

        std::expected<void, std::string> initialize(const lfs::core::Tensor& opacities) override;

        std::expected<std::pair<lfs::core::Tensor, SparsityLossContext>, std::string>
        compute_loss_forward(const lfs::core::Tensor& opacities) override;

        std::expected<void, std::string>
        compute_loss_backward(const SparsityLossContext& ctx,
                              float grad_loss,
                              lfs::core::Tensor& grad_opacities) override;

        std::expected<void, std::string> update_state(const lfs::core::Tensor& opacities) override;

        std::expected<lfs::core::Tensor, std::string>
        get_prune_mask(const lfs::core::Tensor& opacities) override;

        bool should_update(int iter) const override {
            int relative_iter = iter - config_.start_iteration;
            return config_.sparsify_steps > 0 &&
                   iter > config_.start_iteration &&
                   relative_iter < config_.sparsify_steps &&
                   relative_iter % config_.update_every == 0;
        }

        bool should_apply_loss(int iter) const override {
            return config_.sparsify_steps > 0 &&
                   iter > config_.start_iteration &&
                   iter <= (config_.start_iteration + config_.sparsify_steps);
        }

        bool should_prune(int iter) const override {
            return config_.sparsify_steps > 0 &&
                   iter == (config_.start_iteration + config_.sparsify_steps);
        }

        int get_num_to_prune(const lfs::core::Tensor& opacities) override;

        bool is_initialized() const override { return initialized_; }

        void reset() override;

    private:
        std::expected<void, std::string> ensure_state_matches(const lfs::core::Tensor& opacities, const char* phase);

        /**
         * @brief Apply soft thresholding to enforce sparsity
         * @param z Input tensor [N, 1]
         * @return Thresholded tensor [N, 1]
         */
        lfs::core::Tensor prune_z(const lfs::core::Tensor& z);

        Config config_;
        lfs::core::Tensor u_;           // Dual variable (Lagrange multiplier) [N, 1]
        lfs::core::Tensor z_;           // Auxiliary variable for sparsity [N, 1]
        lfs::core::Tensor opa_sigmoid_; // Cached sigmoid(opacities) [N, 1]

        bool initialized_ = false;
    };

    /**
     * @brief Factory for creating sparsity optimizers (LibTorch-free)
     */
    class SparsityOptimizerFactory {
    public:
        static std::unique_ptr<ISparsityOptimizer> create(
            const std::string& method,
            const ADMMSparsityOptimizer::Config& config);
    };

} // namespace lfs::training
