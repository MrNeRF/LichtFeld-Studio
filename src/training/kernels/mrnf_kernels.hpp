/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>

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

    // Need-guided growth: turns the refine window's error into NEED in place,
    //   need = clamp(error / error_scale, 0, error_max) *
    //          clamp(footprint / footprint_scale, sparsity_min, sparsity_max).
    // footprint is a splat's largest on-screen share over the window; an area share is
    // square-rooted first so it measures a length. A null footprint or footprint_scale <= 0
    // leaves the sparsity factor at 1. Non-positive and NaN errors give 0.
    struct NeedScales {
        float error_scale = 1.0f;
        float error_max = 20.0f;
        float footprint_scale = 0.0f;
        float sparsity_min = 0.25f;
        float sparsity_max = 16.0f;
        bool footprint_is_area = false;
    };
    void launch_fold_need(
        float* error_inout,
        const float* footprint,
        NeedScales scales,
        size_t N,
        void* stream = nullptr);

    // need = need^power in place on the active rows (free_mask true marks a free row; null: all
    // rows active). Non-positive and NaN needs give 0; free rows are left as they are.
    void launch_need_power(
        float* need_inout,
        const bool* free_mask,
        float power,
        size_t N,
        void* stream = nullptr);

    // weights *= edge guidance factor.
    void launch_apply_edge_guidance(float* weights, EdgeGuidance edge_guidance, size_t N,
                                    void* stream = nullptr);

    // The `count` rows the growth exchange gives up first. A row's rank is the
    // later of its positions in ascending `primary` and `secondary` importance
    // (NaN last, ties by row; `secondary` may be null); rows come out by rank,
    // ties by row, and ineligible rows rank after all eligible ones.
    void launch_rank_exchange_rows(
        const float* primary,
        const float* secondary,
        const bool* eligible,
        size_t N,
        size_t count,
        int64_t* rows,
        void* stream = nullptr);

    // The same for any number of keys (each non-null): a row's rank is the latest of its positions in
    // ascending order of every key, i.e. the largest of its percentiles among the eligible rows.
    void launch_rank_rows_latest(
        const float* const* keys,
        size_t n_keys,
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
        const float* opacity_hold = nullptr);

    // Exploration walk for need-guided growth, one launch per training step after the backward.
    // For every row the backward marked as visible in this view (rendered > 0):
    //   mean += k * R diag(exp(log_scale)) z,  z ~ N(0, 1)^3,  R from the row's quaternion,
    //   k = min(amplitude * E(o) * (1 - g)^150, 1),  E(o) = sqrt(2 ln(255 o)) (0 at o <= 1/255),
    //   g = o, or min(o, blend_peak) when blend_peak is given.
    // The launch also clears rendered, multiplies blend_peak by peak_decay (a running maximum the
    // backward raises with atomic max) and counts renders off scout_renders (floored at 0).
    // amplitude <= 0 moves nothing. Frozen rows do not move.
    struct ExploreWalk {
        const float* rotations = nullptr;  // [N, 4] raw quaternions (w, x, y, z)
        const float* log_scales = nullptr; // [N, 3]
        float* rendered = nullptr;         // [N] visible-view count of this step, cleared here
        float* blend_peak = nullptr;       // [N] optional running peak blend weight
        float* scout_renders = nullptr;    // [N] optional renders a scout still needs
        float amplitude = 0.0f;
        float peak_decay = 1.0f;
    };
    void launch_explore_walk(
        float* means,
        const float* raw_opacities,
        const bool* frozen_mask,
        size_t frozen_mask_size,
        const ExploreWalk& walk,
        size_t N,
        uint64_t seed,
        void* stream = nullptr);

    // Deterministic scout draw: child j of the chunk (first child at first_rows[j], second at
    // second_rows[j]) becomes a scout when hash(seed, 2 (offset + j) + c) < share; a scout gets
    // raw opacity raw_scout and scout_renders = renders (when scout_renders is given).
    void launch_explore_scouts(
        float* raw_opacities,
        float* scout_renders,
        const int64_t* first_rows,
        const int64_t* second_rows,
        size_t count,
        uint64_t seed,
        uint64_t offset,
        float share,
        float raw_scout,
        float renders,
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

    // Row 0 is MRNF's persistent visibility count and must not be cleared.
    // Fold row 1 into the persistent error maximum and clear only the row-1
    // per-iteration scratch.
    // With coverage_views (and coverage_seen, both [N]), also counts this view: row 0 sums the
    // window coverage, so a view saw the splat when row 0 grew past coverage_seen, which then
    // takes row 0's value.
    void launch_fold_densification_error_and_zero(
        float* refine_weight_max,
        float* densification_info,
        size_t N,
        float* coverage_views = nullptr,
        float* coverage_seen = nullptr,
        void* stream = nullptr);

    // Compressed refine error: each view's per-splat error sum s is taken to this power before
    // it enters the window, so a large footprint no longer outweighs a high error per pixel.
    inline constexpr float REFINE_ERROR_COMPRESS_VIEW_POWER = 0.2f;

    // View-mean variant over a [4, N] densification_info. Rows 0 and 1 hold this view's
    // coverage and error; row 2 is the window coverage (row 0's role in the max fold) and
    // row 3 counts the views where the splat was visible (coverage > 0). Adds the error to
    // the window sum in refine_weight, moves the coverage to row 2, then clears rows 0 and 1.
    // With compress, the error added is s^REFINE_ERROR_COMPRESS_VIEW_POWER and row 3 counts
    // the views where that is > 0 instead. With coverage_views ([N]), it also counts the views
    // with coverage > 0 there, in either mode.
    void launch_fold_densification_error_view_sum_and_zero(
        float* refine_weight,
        float* densification_info,
        size_t N,
        bool compress = false,
        float* coverage_views = nullptr,
        void* stream = nullptr);

    // refine_weight = refine_weight / max(view count, 1), with the count in row 3.
    void launch_refine_error_view_mean(
        float* refine_weight,
        const float* densification_info,
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

    // Ray-depth rows (lfs/training/ray_depth_step.cuh) of split children: anchor and baseline spread from the
    // parent, fresh moments for both, as the parent's other optimizer state is reset too.
    void launch_ray_depth_inherit(
        float* state,
        const int64_t* parents,
        const int64_t* children,
        size_t K,
        void* stream = nullptr);

} // namespace lfs::training::mrnf_strategy
