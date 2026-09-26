/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data.hpp"
#include "core/tensor_upload.hpp"
#include "lfs/training/ops/adam.hpp"
#include "lfs/training/ops/raster.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <string>
#include <unordered_map>

/**
 * LibTorch-free Adam Optimizer for Gaussian Splatting
 *
 * Owns all training state: gradients and joint packed Adam moments.
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
        // Joint codec: exp_avg holds packed (u,log_s) bytes;
        //   joint_bounds holds float4 per 256-splat block.
        lfs::core::Tensor exp_avg;
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

    [[nodiscard]] inline bool fastgs_adam_enabled(const lfs::gpu_ops::BackwardAdam& adam) noexcept {
        for (const auto& group : adam.groups) {
            if (group.enabled) {
                return true;
            }
        }
        return false;
    }

    class AdamOptimizer {
    public:
        explicit AdamOptimizer(lfs::core::SplatData& splat_data, const AdamConfig& config);

        void step(int iteration);
        // Groups and optimizer scalars. Loss and sparsity tensors stay unbound (absent_);
        // the caller overlays those when it builds the backward hand-off.
        lfs::gpu_ops::BackwardAdam prepare_fastgs_fused_adam(int iteration, cudaStream_t execution_stream = nullptr);
        void commit_fastgs_fused_adam(int iteration);
        void set_frozen_mask(lfs::core::Tensor mask);
        [[nodiscard]] const lfs::core::Tensor& frozen_mask() const { return frozen_mask_; }
        void set_frozen_lr_scale(float scale);
        void set_crop_damping_mask(lfs::core::Tensor mask);
        void set_cropbox_lr_scale(float scale);
        void set_per_splat_mean_step(bool enabled,
                                     float median_extent,
                                     float r_min,
                                     float r_max);
        // Retain the allocation for both explicit and fused Adam; CPU inputs are uploaded.
        void set_mean_step_far_mask(lfs::core::Tensor mask);
        void set_screen_share_cap(const float* max_share, int n, float limit, float penalty);
        void refresh_screen_share_buffer();
        // Collection is independent of the hinge: refinement can still consume
        // projected sizes while the strategy defers scale shrinkage.
        void set_collect_projected_screen_share(bool enabled) noexcept { collect_projected_screen_share_ = enabled; }
        [[nodiscard]] bool collect_projected_screen_share() const noexcept { return collect_projected_screen_share_; }
        [[nodiscard]] bool per_splat_mean_step() const noexcept { return per_splat_mean_step_; }
        [[nodiscard]] const bool* mean_step_far_mask() const noexcept {
            return mean_step_far_mask_;
        }
        [[nodiscard]] int mean_step_far_mask_n() const noexcept { return mean_step_far_mask_n_; }
        [[nodiscard]] const lfs::core::Tensor& mean_step_far_mask_storage() const noexcept {
            return mean_step_far_mask_storage_;
        }
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
        // indices are int64 device rows.
        void relocate_params_at_indices_gpu(ParamType type, const lfs::core::Tensor& indices);

        // Low-level state manipulation
        void reset_state_at_indices(ParamType type, const std::vector<int64_t>& indices);
        // Keeps CUDA-resident indices on device; accepts int32 or int64 tensors.
        void reset_state_at_indices(ParamType type, const lfs::core::Tensor& indices);
        void extend_state_for_new_params(ParamType type, size_t n_new);

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
        const lfs::gpu_ops::AdamOps* ops_ = nullptr;
        // Binds absent optional op arguments.
        lfs::core::Tensor absent_;
        std::unordered_map<std::string, AdamParamState> states_;
        lfs::core::TensorUpload reset_indices_upload_;
        lfs::core::TensorUpload extend_indices_upload_;
        lfs::core::TensorUpload add_indices_upload_;
        lfs::core::Tensor frozen_mask_;
        float frozen_lr_scale_ = 0.0f;
        lfs::core::Tensor crop_damping_mask_;
        float cropbox_lr_scale_ = 1.0f;
        bool per_splat_mean_step_ = false;
        float mean_step_median_extent_ = 0.0f;
        float mean_step_r_min_ = 1.0f;
        float mean_step_r_max_ = 300.0f;
        lfs::core::Tensor mean_step_far_mask_storage_;
        const bool* mean_step_far_mask_ = nullptr;
        int mean_step_far_mask_n_ = 0;
        const float* screen_share_max_ = nullptr;
        int screen_share_n_ = 0;
        float screen_share_limit_ = 0.0f;
        float screen_share_penalty_ = 0.0f;
        bool collect_projected_screen_share_ = false;
        int64_t fused_step_iteration_ = -1;
        bool last_step_zeroed_gradients_ = false;

        lfs::core::Tensor& get_param(ParamType type);
        std::string param_name(ParamType type) const;
        void init_state(ParamType type, bool allocate_grad = false);
        void ensure_grad(ParamType type);
        void step_shN(int iteration);
        [[nodiscard]] const lfs::gpu_ops::AdamOps& adam_ops() const;
        [[nodiscard]] lfs::gpu_ops::AdamHyper adam_hyper() const;
        [[nodiscard]] lfs::gpu_ops::AdamModifiers adam_modifiers() const;
        void validate_mean_step_far_mask();
        size_t compute_new_capacity(size_t current_capacity, size_t required_size) const;

        // Quantized-moment helpers. Moments are uint8 (m signed @ zero-point 128, v as
        // quantised sqrt(v)) with per-primitive fp32 scales. shN moments are swizzled, so the
        // byte buffer length is the swizzled float count while the scale count is per-primitive.
        // A zero scale dequantises to a zero moment regardless of the stored byte, so fresh /
        // reset rows only need their scale zeroed.
        void alloc_quantized_state(ParamType type, AdamParamState& state, const lfs::core::Tensor& param,
                                   size_t moment_capacity, size_t prim_capacity);
        size_t scale_row_count(ParamType type) const;

        // Translate a primitive-row delta into the actual tensor growth along dim 0.
        // shN (1D swizzled): delta = swizzled_float_count(N + n_new) - swizzled_float_count(N).
        // Others: delta = n_new.
        size_t compute_state_growth(ParamType type, size_t n_new) const;
    };

} // namespace lfs::training
