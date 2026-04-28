/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include <array>
#include <string>
#include <unordered_map>

/**
 * LibTorch-free Adam Optimizer for Gaussian Splatting
 *
 * Owns all training state: gradients, exp_avg (momentum), exp_avg_sq (second moment).
 * SplatData stores only model parameters; gradients are managed here.
 *
 * Usage:
 *   AdamOptimizer optimizer(splat_data, config);
 *   optimizer.allocate_gradients(max_capacity);  // Pre-allocate for MCMC
 *   // ... backward pass writes to optimizer.get_grad(ParamType::*)
 *   optimizer.step(iter);
 *   optimizer.zero_grad(iter);
 *
 * Uses capacity-based growth (like std::vector) to minimize GPU allocations.
 * Set initial_capacity to max Gaussians to avoid reallocations during MCMC.
 */

namespace lfs::training {

    struct AdamConfig {
        float lr = 1e-3f;
        double beta1 = 0.9; // Must be double to match legacy precision
        double beta2 = 0.999;
        double eps = 1e-15; // Tuned for 3DGS (PyTorch default is 1e-8)

        std::unordered_map<std::string, double> param_lrs; // Per-parameter LRs (double for precision)

        float growth_factor = 1.5f;  // Capacity growth factor
        size_t initial_capacity = 0; // Pre-allocation size (0 = auto)
    };

    struct AdamParamState {
        lfs::core::Tensor grad;       // Gradient (transient)
        lfs::core::Tensor exp_avg;    // First moment (m)
        lfs::core::Tensor exp_avg_sq; // Second moment (v)
        int64_t step_count = 0;
        size_t capacity = 0; // Allocated capacity
        size_t size = 0;     // Used size
    };

    enum class ParamType {
        Means,
        Sh0,
        ShN,
        Scaling,
        Rotation,
        Opacity
    };

    struct FastGSFusedAdamParam {
        float* param = nullptr;
        float* exp_avg = nullptr;
        float* exp_avg_sq = nullptr;
        int n_elements = 0;
        int n_attributes = 0;
        float step_size = 0.0f;
        float bias_correction2_sqrt_rcp = 1.0f;
        bool enabled = false;
    };

    struct FastGSFusedAdamState {
        bool enabled = false;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-15f;
        float scale_reg_weight = 0.0f;
        float opacity_reg_weight = 0.0f;
        const float* sparsity_opa_sigmoid = nullptr;
        const float* sparsity_z = nullptr;
        const float* sparsity_u = nullptr;
        int sparsity_n = 0;
        float sparsity_rho = 0.0f;
        float sparsity_grad_loss = 0.0f;
        FastGSFusedAdamParam means;
        FastGSFusedAdamParam sh0;
        FastGSFusedAdamParam shN;
        FastGSFusedAdamParam scaling;
        FastGSFusedAdamParam rotation;
        FastGSFusedAdamParam opacity;
    };

    class AdamOptimizer {
    public:
        explicit AdamOptimizer(lfs::core::SplatData& splat_data, const AdamConfig& config);

        void step(int iteration);
        FastGSFusedAdamState prepare_fastgs_fused_adam(int iteration);
        void commit_fastgs_fused_adam(int iteration);

        // Gradient management
        void allocate_gradients();
        void allocate_gradients(size_t capacity);
        void zero_grad(int iteration);
        bool has_gradients() const;
        lfs::core::Tensor& get_grad(ParamType type);

        // Learning rate
        void set_lr(float lr) { config_.lr = lr; }
        float get_lr() const { return config_.lr; }
        void set_param_lr(ParamType type, double lr) { config_.param_lrs[param_name(type)] = lr; }
        bool has_param_lr(ParamType type) const { return config_.param_lrs.contains(param_name(type)); }

        double get_param_lr(ParamType type) const {
            const auto name = param_name(type);
            const auto it = config_.param_lrs.find(name);
            return (it != config_.param_lrs.end()) ? it->second : static_cast<double>(config_.lr);
        }

        static constexpr std::array<ParamType, 6> all_param_types() {
            return {ParamType::Means, ParamType::Sh0, ParamType::ShN,
                    ParamType::Scaling, ParamType::Rotation, ParamType::Opacity};
        }

        // MCMC operations (atomically update params + optimizer state)
        void add_new_params(ParamType type, const lfs::core::Tensor& new_values, bool validate = false);
        void add_new_params_gather(ParamType type, const lfs::core::Tensor& indices);
        void relocate_params_at_indices(ParamType type, const std::vector<int64_t>& indices);
        void relocate_params_at_indices_gpu(ParamType type, const int64_t* indices_device, size_t n_indices);

        // Low-level state manipulation
        void reset_state_at_indices(ParamType type, const std::vector<int64_t>& indices);
        void extend_state_for_new_params(ParamType type, size_t n_new);
        void extend_state_by_gather(ParamType type, const lfs::core::Tensor& indices);

        // State access
        const AdamParamState* get_state(ParamType type) const;
        AdamParamState* get_state_mutable(ParamType type);
        int64_t get_step_count(ParamType type) const;
        void set_state(ParamType type, const AdamParamState& state);
        const AdamConfig& get_config() const { return config_; }

        // Serialization
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);
        void reserve_capacity(size_t capacity);

        // Control notifications for external mutations
        void reset_state(ParamType type);
        void invalidate_state(ParamType type);

    private:
        AdamConfig config_;
        lfs::core::SplatData& splat_data_;
        std::unordered_map<std::string, AdamParamState> states_;
        int64_t fused_step_iteration_ = -1;
        bool last_step_zeroed_gradients_ = false;

        lfs::core::Tensor& get_param(ParamType type);
        std::string param_name(ParamType type) const;
        void init_state(ParamType type, bool allocate_grad = false);
        void ensure_grad(ParamType type);
        void step_param(ParamType type, int iteration);
        size_t compute_new_capacity(size_t current_capacity, size_t required_size) const;
    };

} // namespace lfs::training
