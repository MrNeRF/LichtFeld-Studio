/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cuda_runtime_api.h>
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
        lfs::core::Tensor grad; // Gradient (transient, fp32)
        // Legacy codec: separate uint8 m / sqrt(v) + per-primitive fp32 scales.
        // Joint codec: exp_avg holds packed (u,log_s) bytes;
        //   joint_bounds holds float4 per 256-splat block; exp_avg_sq/scales empty.
        lfs::core::Tensor exp_avg;
        lfs::core::Tensor exp_avg_sq;
        lfs::core::Tensor exp_avg_scale;
        lfs::core::Tensor exp_avg_sq_scale;
        lfs::core::Tensor joint_bounds; // [n_bounds, 4] fp32; joint codec only
        int joint_bits = 0;             // 0=legacy, 8=SH, 16=non-SH
        int64_t step_count = 0;
        size_t capacity = 0; // Allocated capacity (moment rows / float cells)
        size_t size = 0;     // Used size
        [[nodiscard]] bool is_joint() const noexcept { return joint_bits != 0; }
    };

    /// grow-only joint (u,log_s) bounds table. Reuses capacity via append_zeros
    /// when possible; only driver-allocates when nb exceeds current capacity.
    /// When @p zero_all is true (compact path), existing storage is zeroed in place.
    void ensure_joint_bounds_capacity(lfs::core::Tensor& joint_bounds,
                                      size_t n_prims,
                                      size_t capacity_prims,
                                      lfs::core::Device device,
                                      bool zero_all = false);

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
        uint8_t* joint_packed = nullptr;
        float* joint_bounds = nullptr;
        int joint_bits = 0;
        // SH value quant (shN only)
        float* sh_value_bounds = nullptr;
        int sh_value_bits = 0;
        int sh_value_n_cells = 0;
        int n_primitives = 0;
        const bool* frozen_mask = nullptr;
        int frozen_mask_size = 0;
        float frozen_lr_scale = 0.0f;
        const bool* crop_damping_mask = nullptr;
        int crop_damping_mask_size = 0;
        float cropbox_lr_scale = 1.0f;
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
        FastGSFusedAdamState prepare_fastgs_fused_adam(int iteration, cudaStream_t execution_stream = nullptr);
        void commit_fastgs_fused_adam(int iteration);
        void set_frozen_mask(lfs::core::Tensor mask);
        void set_frozen_lr_scale(float scale);
        void set_crop_damping_mask(lfs::core::Tensor mask);
        void set_cropbox_lr_scale(float scale);
        [[nodiscard]] const lfs::core::Tensor& crop_damping_mask() const noexcept {
            return crop_damping_mask_;
        }
        [[nodiscard]] float cropbox_lr_scale() const noexcept { return cropbox_lr_scale_; }

        // Gradient management
        void allocate_gradients();
        void allocate_gradients(size_t capacity);
        void zero_grad(int iteration);
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

        // preflight exportable capacity for a densify grow of
        /// `n_new` rows BEFORE any free_mask / param mutation. Returns false when
        /// capacity-ensure fails so callers can abort with zero torn state.
        [[nodiscard]] bool preflight_grow_capacity(size_t n_new);
        void relocate_params_at_indices_gpu(ParamType type, const int64_t* indices_device, size_t n_indices);

        // Low-level state manipulation
        void reset_state_at_indices(ParamType type, const std::vector<int64_t>& indices);
        void extend_state_for_new_params(ParamType type, size_t n_new);
        void extend_state_by_gather(ParamType type, const lfs::core::Tensor& indices);

        // State access
        const AdamParamState* get_state(ParamType type) const;
        AdamParamState* get_state_mutable(ParamType type);
        int64_t get_step_count(ParamType type) const;
        const AdamConfig& get_config() const { return config_; }

        // Serialization
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);
        void adopt_checkpoint_state(AdamOptimizer& loaded) noexcept;
        void reserve_capacity(size_t capacity);

        // Control notifications for external mutations
        void reset_state(ParamType type);

        /// Telemetry: times the capacity=0 / no-headroom slow grow path has fired
        /// (process-wide). Loud LOG_WARN is emitted at each site; tests assert this counter.
        [[nodiscard]] static uint64_t slow_path_grow_count() noexcept;
        static void reset_slow_path_grow_count() noexcept;

    private:
        static void note_slow_path_grow(const char* site, const std::string& name);
        AdamConfig config_;
        lfs::core::SplatData& splat_data_;
        std::unordered_map<std::string, AdamParamState> states_;
        lfs::core::Tensor frozen_mask_;
        float frozen_lr_scale_ = 0.0f;
        lfs::core::Tensor crop_damping_mask_;
        float cropbox_lr_scale_ = 1.0f;
        int64_t fused_step_iteration_ = -1;
        bool last_step_zeroed_gradients_ = false;

        lfs::core::Tensor& get_param(ParamType type);
        std::string param_name(ParamType type) const;
        void init_state(ParamType type, bool allocate_grad = false);
        void ensure_grad(ParamType type);
        void step_param(ParamType type, int iteration);
        size_t compute_new_capacity(size_t current_capacity, size_t required_size) const;

        // Quantized-moment helpers. Moments are uint8 (m signed @ zero-point 128, v as
        // quantised sqrt(v)) with per-primitive fp32 scales. shN moments are swizzled, so the
        // byte buffer length is the swizzled float count while the scale count is per-primitive.
        // A zero scale dequantises to a zero moment regardless of the stored byte, so fresh /
        // reset rows only need their scale zeroed.
        void alloc_quantized_state(ParamType type, AdamParamState& state, const lfs::core::Tensor& param,
                                   size_t moment_capacity, size_t prim_capacity);
        size_t scale_row_count(ParamType type) const;
        const bool* frozen_mask_ptr() const;
        int frozen_mask_size() const;
        const bool* crop_damping_mask_ptr() const;
        int crop_damping_mask_size() const;

        // Translate a primitive-row delta into the actual tensor growth along dim 0.
        // shN (1D swizzled): delta = swizzled_float_count(N + n_new) - swizzled_float_count(N).
        // Others: delta = n_new.
        size_t compute_state_growth(ParamType type, size_t n_new) const;
    };

} // namespace lfs::training
