/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace lfs::training::mrnf_strategy {

    struct MRNFBounds {
        float center[3];
        float extent[3];
        float median_size;
        float max_extent;
    };

    // Edge guidance of row i is 1 + weight * score_i / median, where median is
    // the scores' positive median (0 when none is positive: factor 1) and NaN
    // scores count as 0. Null scores disable it.
    struct EdgeGuidance {
        const float* scores = nullptr;
        float median = 0.0f;
        float weight = 0.0f;
    };

    // sigmoid(raw_opacities) * (visibility > 0), masked and edge-guided.
    void launch_replace_parent_weights(
        const float* raw_opacities,
        const float* visibility,
        const bool* active_mask,
        const bool* trainable_mask,
        EdgeGuidance edge_guidance,
        float* output,
        size_t N,
        void* stream = nullptr);

    // weights *= edge guidance factor.
    void launch_apply_edge_guidance(float* weights, EdgeGuidance edge_guidance, size_t N,
                                    void* stream = nullptr);

    // The `count` eligible rows with the lowest value (exchange victims and retirees): ascending value,
    // NaN last, ties by row; ineligible rows come after every eligible one.
    void launch_lowest_eligible_rows(
        const float* values,
        const bool* eligible,
        size_t N,
        size_t count,
        int64_t* rows,
        void* stream = nullptr);

    void launch_fill_index_mask(
        const int64_t* indices,
        size_t count,
        bool* mask,
        void* stream = nullptr);

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
     * @param raw_opacities [N] — raw opacities (modified in-place)
     * @param log_scales [N, 3] — log scales (modified in-place)
     * @param frozen_mask [N] — optional mask of rows that must not be modified
     * @param frozen_mask_size — number of entries in frozen_mask
     * @param opacity_decay — opacity decay rate
     * @param scale_decay — scale decay rate
     * @param train_t — current training progress [0, 1]
     * @param N — number of splats
     * @param stream — CUDA stream
     * @param reach [N] — optional per-row factor; a row with r < 1 takes opacity_decay x r and scale_decay x r
     *        (r <= 0 leaves it as it is), r >= 1 decays as without it
     */
    void launch_mrnf_decay(
        float* raw_opacities,
        float* log_scales,
        const bool* frozen_mask,
        size_t frozen_mask_size,
        float opacity_decay,
        float scale_decay,
        float train_t,
        size_t N,
        void* stream = nullptr,
        const float* rendered_count = nullptr,
        const float* reach = nullptr);

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

    std::pair<size_t, size_t> percentile_bounds_ranks(size_t n, float percentile);

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
     * Rows with non-positive or NaN weight are picked only when K exceeds
     * the positive rows. Output is in descending key order.
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
        void* stream = nullptr);

    // The K-th largest of values[i] over the rows where include[i] is set
    // (all rows when include is null), without sorting. Needs K <= N and,
    // with include, K <= the included rows.
    [[nodiscard]] float launch_kth_largest(
        const float* values,
        const bool* include,
        size_t N,
        size_t K,
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

    // Ray-depth rows (lfs/training/ray_depth_step.cuh) of split children: anchor and baseline spread from the
    // parent, fresh moments for both, as the parent's other optimizer state is reset too.
    void launch_ray_depth_inherit(
        float* state,
        const int64_t* parents,
        const int64_t* children,
        size_t K,
        void* stream = nullptr);

    // Growth at the cap (mrnf_growth_kernels.cu).
    // Folds one view of a [4, N] densification_info into the window. Rows 0 and 1 hold this view's
    // coverage and error, row 2 the window coverage and row 3 the count of views with error. Adds
    // s^0.2 of this view's error s to refine_weight, moves the coverage to row 2 and clears rows 0
    // and 1. With coverage_views ([N]), also counts the views with coverage > 0.
    void launch_fold_densification_error_view_sum_and_zero(
        float* refine_weight,
        float* densification_info,
        size_t N,
        float* coverage_views = nullptr,
        void* stream = nullptr);

    // refine_weight = refine_weight / max(view count, 1), with the count in row 3.
    void launch_refine_error_view_mean(
        float* refine_weight,
        const float* densification_info,
        size_t N,
        void* stream = nullptr);

    // Need-guided growth: turns the refine window's error into NEED in place,
    //   need = min(error / error_scale, error_max)^power; non-positive and NaN errors give 0.
    void launch_fold_need(
        float* error_inout,
        float error_scale,
        float error_max,
        float power,
        size_t N,
        void* stream = nullptr);

    // Retire quota and keep covered (mrnf_retire_kernels.cu).
    // Peak blend weight, one launch per training step after the backward: renders += 1 where step_peak > 0,
    // peak = max(peak x peak_decay, step_peak), and step_peak is cleared for the next step.
    void launch_blend_peak_update(
        float* step_peak,
        float* peak,
        float* renders,
        float peak_decay,
        size_t N,
        void* stream = nullptr);

    // Keep covered. A row's reach: for a row rendered at least grace times, clamp(peak / max(sigmoid(raw
    // opacity), 1e-6), 0, 1), the share of its opacity that reached a photo in its best recent view; 1 for the
    // others. factor: the reach for a kept row (reach < keep_reach and largest log-scale >= log_size_floor), 1 for
    // every other row.
    void launch_mrnf_covered_factor(
        const float* peak,
        const float* renders,
        const float* raw_opacities,
        const float* log_scales,
        float grace,
        float keep_reach,
        float log_size_floor,
        size_t N,
        float* factor,
        void* stream = nullptr);

    // Retire rank key, ascending: peak - 1 for a dead row (peak < dead_peak), so dead rows come first, by peak;
    // the reach of every other row, as in launch_mrnf_covered_factor.
    void launch_mrnf_retire_key(const float* peak, const float* raw_opacities, float dead_peak, size_t N, float* key,
                                void* stream = nullptr);
} // namespace lfs::training::mrnf_strategy
