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

    /**
     * Per-iteration exploration noise for low-opacity splats.
     *
     * weight = (1 - sigmoid(raw_opac))^150 * visible * lr_mean * noise_weight
     * means += clamp(N(0,1) * weight, -median_scale, +median_scale)
     *
     * @param means [N, 3] — positions (modified in-place)
     * @param raw_opacities [N] — raw opacity values
     * @param vis_count [N] — visibility counts (> 0 means visible)
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
     * @param opacity_decay — opacity decay rate
     * @param scale_decay — scale decay rate
     * @param train_t — current training progress [0, 1]
     * @param N — number of splats
     * @param stream — CUDA stream
     */
    void launch_mrnf_decay(
        float* raw_opacities,
        float* log_scales,
        float opacity_decay,
        float scale_decay,
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
        void* stream = nullptr);

    void launch_elementwise_add_inplace(
        float* a,
        const float* b,
        size_t N,
        void* stream = nullptr);

} // namespace lfs::training::mrnf_strategy
