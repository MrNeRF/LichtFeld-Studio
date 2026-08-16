/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace lfs::training {

    // Initialize Gaussians (move to GPU, pre-allocate capacity, etc.)
    void initialize_gaussians(lfs::core::SplatData& splat_data, int max_cap = 0);

    // Create optimizer for splat data
    std::unique_ptr<AdamOptimizer> create_optimizer(
        lfs::core::SplatData& splat_data,
        const lfs::core::param::OptimizationParameters& params);

    // Create exponential LR scheduler
    std::unique_ptr<ExponentialLR> create_scheduler(
        const lfs::core::param::OptimizationParameters& params,
        AdamOptimizer& optimizer);

    // Build the row mask for splats loaded via --add-splat ... --freeze.
    // Invalid/empty means no frozen rows for the requested size.
    // GPU mask is cached and rebuilt only when frozen ranges or N change
    // (no host vector + H2D on the steady inject_noise path).
    lfs::core::Tensor make_frozen_mask(
        const lfs::core::SplatData& splat_data,
        size_t n,
        lfs::core::Device device);

    /// Test/telemetry: times make_frozen_mask rebuilt the GPU cache.
    [[nodiscard]] std::uint64_t frozen_mask_rebuild_count() noexcept;
    void reset_frozen_mask_rebuild_count() noexcept;

    lfs::core::Tensor make_trainable_mask(
        const lfs::core::SplatData& splat_data,
        size_t n,
        lfs::core::Device device);

    lfs::core::Tensor exclude_frozen_from_mask(
        const lfs::core::SplatData& splat_data,
        const lfs::core::Tensor& mask);

    lfs::core::Tensor zero_frozen_scores(
        const lfs::core::SplatData& splat_data,
        const lfs::core::Tensor& scores);

    void zero_frozen_scores_inplace(
        const lfs::core::SplatData& splat_data,
        lfs::core::Tensor& scores);

    // Scale crop-rejected row signals without mutating accumulated statistics.
    // Invalid crop mask and scale 1 return the original tensor without device work.
    lfs::core::Tensor apply_crop_damping_to_scores(
        const AdamOptimizer& optimizer,
        const lfs::core::Tensor& scores);

    size_t frozen_row_count(const lfs::core::SplatData& splat_data, size_t n);

    // Refresh the topology-derived mask without changing the configured LR scale.
    void apply_frozen_ranges_to_optimizer(
        const lfs::core::SplatData& splat_data,
        AdamOptimizer& optimizer);

    // Configure the LR scale and refresh the topology-derived mask.
    void apply_frozen_ranges_to_optimizer(
        const lfs::core::SplatData& splat_data,
        AdamOptimizer& optimizer,
        float freeze_lr_scale);

    void remap_frozen_ranges_after_compaction(
        lfs::core::SplatData& splat_data,
        const lfs::core::Tensor& kept_old_indices,
        size_t old_size);

    // Returns the fused MCMC-style dead mask:
    // opacity <= min_opacity OR ||rotation||^2 < 1e-8.
    lfs::core::Tensor compute_dead_mask_from_opacity_and_rotation(
        const lfs::core::Tensor& opacities,
        const lfs::core::Tensor& rotations,
        float min_opacity);

    // Returns a mask for rows whose quaternion magnitude is near zero:
    // ||rotation||^2 < 1e-8.
    lfs::core::Tensor compute_near_zero_rotation_mask(
        const lfs::core::Tensor& rotations);

    /**
     * Grow-only N-row densify scratch for masks and weights.
     * Sized once to max_cap so refine never re-driver-allocs as live N climbs.
     */
    struct DensifyNScratch {
        lfs::core::Tensor f32_a;  // weights / scores
        lfs::core::Tensor bool_a; // masks
        lfs::core::Tensor i64_a;  // indices (K, grows with K high-water)
        lfs::core::Tensor i64_b;
        size_t n_capacity = 0;
        size_t k_capacity = 0;
        size_t n_required = 0;
        size_t k_required = 0;

        void ensure_n(size_t n, lfs::core::Device device);
        void ensure_k(size_t k, lfs::core::Device device);

        /// Resident capacity bytes (f32×1 + bool×1 + i64×2 at high-water).
        [[nodiscard]] std::size_t resident_bytes() const noexcept {
            std::size_t bytes = 0;
            if (n_capacity > 0) {
                bytes += n_capacity * (sizeof(float) + sizeof(bool));
            }
            if (k_capacity > 0) {
                bytes += k_capacity * 2 * sizeof(std::int64_t);
            }
            return bytes;
        }

        /// Peak logical rows requested from the grow-only backing buffers.
        [[nodiscard]] std::size_t required_bytes() const noexcept {
            return n_required * (sizeof(float) + sizeof(bool)) +
                   k_required * 2 * sizeof(std::int64_t);
        }

        // Drop all storage. Not process-static (lives on the strategy), but
        // zeros_direct f32/bool + pool-backed i64 free paths are teardown-safe
        // Call when the owning strategy is reset early.
        void release() noexcept {
            f32_a = {};
            bool_a = {};
            i64_a = {};
            i64_b = {};
            n_capacity = 0;
            k_capacity = 0;
            n_required = 0;
            k_required = 0;
        }

        [[nodiscard]] lfs::core::Tensor f32_a_view(size_t n) const;
        [[nodiscard]] lfs::core::Tensor bool_a_view(size_t n) const;
        [[nodiscard]] lfs::core::Tensor i64_a_view(size_t k) const;
        [[nodiscard]] lfs::core::Tensor i64_b_view(size_t k) const;
    };

    /**
     * Reusable densification child-buffer workspace.
     * Grow-only high-water for LAS second-child attribute buffers (K rows).
     * Avoids per-refine empty() allocs for means/rot/scale/sh0/shN/opacity.
     */
    struct DensifyChildWorkspace {
        lfs::core::Tensor means;     // [cap, 3]
        lfs::core::Tensor rotations; // [cap, 4]
        lfs::core::Tensor scales;    // [cap, 3]
        lfs::core::Tensor sh0;       // [cap, 1, 3] (MRNF) or use sh0_flat
        lfs::core::Tensor sh0_flat;  // [cap, 3] (IGS+)
        lfs::core::Tensor shN;       // [cap, sh_rest, 3]
        lfs::core::Tensor opacities; // [cap]
        size_t capacity = 0;
        size_t required_bytes_high_water = 0;
        size_t sh_rest = 0;
        bool sh0_as_flat = false;

        /// Ensure workspace holds at least K rows (×1.2 growth). Returns views of size K.
        void ensure(size_t K, size_t sh_rest_in, bool use_shN, bool sh0_flat_layout,
                    lfs::core::Device device);

        [[nodiscard]] std::size_t required_bytes() const noexcept;
        [[nodiscard]] std::size_t resident_bytes() const noexcept;

        [[nodiscard]] lfs::core::Tensor means_view(size_t K) const;
        [[nodiscard]] lfs::core::Tensor rotations_view(size_t K) const;
        [[nodiscard]] lfs::core::Tensor scales_view(size_t K) const;
        [[nodiscard]] lfs::core::Tensor sh0_view(size_t K) const;
        [[nodiscard]] lfs::core::Tensor shN_view(size_t K) const;
        [[nodiscard]] lfs::core::Tensor opacities_view(size_t K) const;
    };

    /**
     * grow densification_info / 1D score buffers without full realloc
     * when reserved capacity allows. densification_info is [2,N]: when n grows we
     * must reallocate (row1 offset = N); when shape already matches, reuse + zero.
     * For 1D scores: append_zeros into reserved capacity when possible.
     */
    void ensure_densification_info_shape_inplace(
        lfs::core::Tensor& densification_info,
        size_t n,
        lfs::core::Device device,
        size_t reserve_cols = 0);

    void ensure_score_buffer_inplace(
        lfs::core::Tensor& scores,
        size_t n,
        lfs::core::Device device,
        size_t reserve_capacity = 0);

    /// Collect leftover per-primitive Adam scale pointers from the removed
    /// legacy moment codec. Always returns 0 under joint-only Adam state.
    int collect_adam_scale_ptrs(
        AdamOptimizer& optimizer,
        float* out_ptrs[12]);

    /// Zero fp32 Adam grad rows at indices (and ShN via swizzled zero when
    /// layout_rest > 0). Scales/moments are left alone — pair with fused scale
    /// zero. Required because strategy::step runs Adam after densify.
    void zero_adam_grads_at_indices(
        AdamOptimizer& optimizer,
        const lfs::core::Tensor& indices,
        uint32_t shN_layout_rest = 0);

} // namespace lfs::training
