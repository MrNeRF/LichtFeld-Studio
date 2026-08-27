/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>

namespace lfs::training {
    struct GumbelTopKScratch;
    struct PositiveMedianScratch;
} // namespace lfs::training

namespace lfs::training::mrnf_strategy {

    struct MRNFBounds {
        float center[3];
        float extent[3];
        float median_size;
        float max_extent;
    };

    /**
     * Per-iteration exploration noise for low-opacity splats.
     *
     * weight = (1 - sigmoid(raw_opac))^150 * visible * lr_mean * noise_weight
     * means += clamp(N(0,1) * weight, -median_scale, +median_scale)
     *
     * @param means [N, 3] — positions (modified in-place)
     * @param raw_opacities [N] — raw opacity values
     * @param vis_count [N] — visibility counts (> 0 means visible)
     * @param frozen_mask [N] — optional mask of rows that must not be modified
     * @param frozen_mask_size — number of entries in frozen_mask
     * @param lr_mean — current mean learning rate
     * @param noise_weight — exploration noise multiplier
     * @param median_scale — clamp range for noise
     * @param N — number of splats
     * @param seed — RNG seed
     * @param stream — CUDA stream
     */
    void launch_mrnf_noise_injection(
        float* means,
        const float* raw_opacities,
        const float* vis_count,
        const bool* frozen_mask,
        size_t frozen_mask_size,
        float lr_mean,
        float noise_weight,
        float median_scale,
        size_t N,
        uint64_t seed,
        void* stream = nullptr);

    /**
     * Time-dependent opacity and scale decay.
     *
     * opac = sigmoid(raw_opac) - opacity_decay * (1 - train_t)
     * scale = exp(log_scale) * (1 - scale_decay * (1 - train_t))
     *
     * Far-mask rows multiply both decay rates by far_decay_scale before applying.
     *
     * @param raw_opacities [N] — raw opacities (modified in-place)
     * @param log_scales [N, 3] — log scales (modified in-place)
     * @param frozen_mask [N] — optional mask of rows that must not be modified
     * @param frozen_mask_size — number of entries in frozen_mask
     * @param far_mask [N] — optional mask of far-field rows
     * @param far_mask_size — number of entries in far_mask
     * @param opacity_decay — opacity decay rate
     * @param scale_decay — scale decay rate
     * @param far_decay_scale — multiplier applied to opacity/scale decay for far-mask rows
     * @param train_t — current training progress [0, 1]
     * @param N — number of splats
     * @param stream — CUDA stream
     */
    void launch_mrnf_decay(
        float* raw_opacities,
        float* log_scales,
        const bool* frozen_mask,
        size_t frozen_mask_size,
        const bool* far_mask,
        size_t far_mask_size,
        float opacity_decay,
        float scale_decay,
        float far_decay_scale,
        float train_t,
        size_t N,
        void* stream = nullptr);

    /**
     * Compute percentile-based bounding box on GPU.
     *
     * Finds the p-th and (1-p)-th percentiles along each axis using partial sort,
     * then computes center, extent, median_size, and max_extent.
     *
     * @param means [N, 3] — splat positions
     * @param N — number of splats
     * @param percentile — fraction for bounds (0.8 = central 80%, i.e. p10/p90)
     * @param bounds — output bounding box
     * @param stream — CUDA stream
     */
    void launch_percentile_bounds(
        const float* means,
        size_t N,
        float percentile,
        MRNFBounds* bounds,
        void* stream = nullptr);

    // Median of geomean(exp(scaling_raw)); out_valid is false when no usable extent remains.
    void launch_median_geomean_extent(
        const float* scaling_raw,
        size_t N,
        float* out_median,
        bool* out_valid,
        void* stream = nullptr);

    /**
     * Gumbel-top-k sampling — weighted sampling without replacement.
     *
     * key[i] = -log(-log(U)) + log(weight[i])
     * selected = top_k(key, K)
     *
     * @param weights [N] — sampling weights (non-negative)
     * @param N — total number of elements
     * @param K — number of samples to draw
     * @param seed — RNG seed
     * @param output_indices [K] — output: selected indices
     * @param stream — CUDA stream
     */
    void launch_gumbel_topk(
        const float* weights,
        size_t N,
        size_t K,
        uint64_t seed,
        int64_t* output_indices,
        void* stream = nullptr,
        bool compact_sparse = true,
        lfs::training::GumbelTopKScratch* scratch = nullptr,
        size_t known_nnz = 0);

    void launch_elementwise_add_inplace(
        float* a,
        const float* b,
        size_t N,
        void* stream = nullptr);

    // Baked per-splat exploration starvation weights.
    inline constexpr float kStarvEps = 0.0026f;
    inline constexpr float kStarvGamma = 1.72f;
    inline constexpr float kExploreStarvDose = 2.38f;

    /**
     * fold densification_info into vis_count (add row0) and refine_weight_max
     * (max of row1), then zero n_rows rows. When ratio_max is non-null, also
     * keep the per-window max of vis >= 0.05 ? err/pow(vis, ratio_pow) : 0
     * (plain err/vis when ratio_pow == 0).
     */
    void launch_fold_densification_and_zero(
        float* vis_count,
        float* refine_weight_max,
        float* densification_info,
        size_t N,
        void* stream = nullptr,
        size_t n_rows = 2,
        float* ratio_max = nullptr,
        float ratio_pow = 0.0f);

    void launch_project_visible_centers(
        const float* means,
        const float* w2c,
        float fx,
        float fy,
        float cx,
        float cy,
        int width,
        int height,
        float near_plane,
        float* means2d,
        float* radii,
        size_t N,
        void* stream = nullptr);

    void launch_gather_center_error(
        const float* means2d,
        const float* radii,
        const float* error,
        int width,
        int height,
        float* scores,
        size_t N,
        void* stream = nullptr);

    void launch_far_field_mask(
        const float* means,
        float centroid_x,
        float centroid_y,
        float centroid_z,
        float far_radius,
        bool* far_out,
        size_t N,
        void* stream = nullptr);

    void launch_mean_abs_error_hw(
        const float* pred,
        const float* target,
        int channels,
        int height,
        int width,
        float* out_hw,
        void* stream = nullptr);

    void launch_seed_weights_from_error_alpha(
        const float* error_hw,
        const float* alpha,
        float* out_weights,
        size_t hw,
        void* stream = nullptr);

    void launch_gather_seed_payloads(
        const int64_t* pixel_indices,
        size_t K,
        size_t hw,
        const float* target,
        int channels,
        const float* alpha,
        const float* depth,
        float* out_rgb,
        float* out_alpha,
        float* out_depth,
        void* stream = nullptr);

    // Median of all `n` values via one CUB radix-sort (sorted[n/2]).
    // Does not compact to positives. out_median is host-side.
    void launch_sorted_median(
        const float* values,
        size_t n,
        float* out_median,
        lfs::training::PositiveMedianScratch* scratch,
        void* stream = nullptr);

    // weights[i] *= 0 if vis[i]==0, else (kStarvEps + starved^kStarvGamma)
    // where starved = clamp(1 - vis[i]/max(median, tiny), 0, 1).
    void launch_apply_explore_starvation_weights(
        float* weights,
        const float* vis_count,
        size_t n,
        float median_vis,
        void* stream = nullptr);

} // namespace lfs::training::mrnf_strategy
